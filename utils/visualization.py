import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')  # use a non-interactive backend for matplotlib
from matplotlib.lines import Line2D
import imageio
import torch
import io
import threading
import shutil
from queue import Queue
import numpy as np
from typing import Any, Iterator

class Visualizer:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.tmp_states_dir = os.path.join(self.output_dir, 'tmp_states')
        self.tmp_gradients_dir = os.path.join(self.output_dir, 'tmp_gradients')
        os.makedirs(self.tmp_states_dir, exist_ok=True)
        os.makedirs(self.tmp_gradients_dir, exist_ok=True)

        self.save_queue = Queue()
        self._shutdown_flag = False
        self._start_background_saver()

    def _start_background_saver(self):
        """Start a background thread to save images to disk."""
        def background_saver():
            while not self._shutdown_flag:
                try:
                    item = self.save_queue.get(timeout=1.0)
                    if item is None:    # shutdown signal
                        break
                    image_data, image_path = item
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    self.save_queue.task_done()
                except Exception:  # timeout
                    continue
        
        self.saver_thread = threading.Thread(target=background_saver, daemon=True)
        self.saver_thread.start()

    def _shutdown_background_saver(self):
        """Shutdown the background saver thread."""
        # wait for all tasks to finish
        self.save_queue.join()

        # set shutdown flag and notify the saver thread
        self._shutdown_flag = True
        self.save_queue.put(None)

        # wait for the saver thread to finish
        if hasattr(self, 'saver_thread') and self.saver_thread.is_alive():
            self.saver_thread.join(timeout=5.0)

    def _plot_hist(self, layer_name: str, layer_idx: int, x: torch.Tensor, 
                   name: str, epoch: int, iteration: int):
        with torch.no_grad():
            x_flat = x.reshape(-1)
            min_val, max_val = x_flat.min().item(), x_flat.max().item()
            mean_val, std_val = x_flat.mean().item(), x_flat.std().item()
            counts, bins = torch.histogram(x_flat.cpu(), bins=100, density=False)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # plot histogram
        ax.bar(bin_centers, counts, width=bins[1] - bins[0], 
                    alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel(f'min: {min_val:.6f}, max: {max_val:.6f}, mean: {mean_val:.6f}, std: {std_val:.6f}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Epoch{epoch}, Iter{iteration}, Layer{layer_idx}:{layer_name}-{name}')
        
        # save the figure to I/O buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_data = buf.getvalue()
        buf.close()

        # save the figure to a temporary image file through background saver
        data_dir = os.path.join(self.tmp_states_dir, f'layer{layer_idx:04d}_{layer_name}_{name}')
        file_name = f'epoch{epoch:04d}_iter{iteration:07d}.png'
        image_path = os.path.join(data_dir, file_name)
        self.save_queue.put((image_data, image_path))

        plt.close(fig)
    
    def visualize_model_states(self, model: torch.nn.Module, 
                               epoch: int, iteration: int) -> None:
        """
        Visualize the current model state by plotting histograms of weights and activations.
        Args:
            model (torch.nn.Module): The model whose states are to be visualized.
            epoch (int): The current epoch number.
            iteration (int): The current iteration number.
        """
        with torch.no_grad():
            layer_idx = 0
            for layer in model.layers:
                layer_idx += 1
                if hasattr(layer, 'visualize_cache'):
                    for (name, val) in layer.visualize_cache.items():
                        self._plot_hist(layer._get_name(), layer_idx, val, name, epoch, iteration)
                    layer.visualize_cache.clear()

    def visualize_grad_flow(self, named_parameters, epoch: int, iteration: int):
        '''
        Plots the gradients flowing through different layers in the net during training.
        Highlights W_EI gradients and separates layers visually.
        '''
        param_rename_map = {
            "alpha": r"$g_I$",
            "gain": r"$g_E$",
            "conv_ee": r"$W_{EE}$",
            "conv_ie": r"$W_{IE}$",
            "weight_ei": r"$W_{EI}$",
        }
        ave_grads = []
        layers = []
        colors = []
        
        highlight_color = '#d62728'
        normal_color = '#1f77b4'
        bar_alpha = 0.8

        for n, p in named_parameters:
            if p.grad is None:
                continue
            
            # 过滤掉 bias，只看 weight 以保持图表整洁
            if p.requires_grad and ("bias" not in n):
                short_name = n.replace("module.", "")
                for key, val in param_rename_map.items():
                    if key in short_name:
                        short_name = param_rename_map[key]
                        break
                if 'layers' in short_name:
                    break
                layers.append(short_name)
                
                ave_grad = p.grad.abs().mean().item()
                ave_grads.append(ave_grad)
                
                if "weight_ei" in n.lower(): 
                    colors.append(highlight_color)
                else:
                    colors.append(normal_color)

        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(figsize=(14, 7))

        x_pos = np.arange(len(ave_grads))
        bars = ax.bar(x_pos, ave_grads, alpha=bar_alpha, width=0.6, color=colors, align='center')

        prev_layer_idx = 1
        for i, name in enumerate(layers):
            if 'layer' in name:
                break
            if prev_layer_idx != -1 and name == r"$W_{EE}$":
                ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                prev_layer_idx += 1

        ax.set_ylabel("Average Gradient Magnitude", fontsize=16, labelpad=10)
        ax.set_xlabel("Parameters", fontsize=16, labelpad=10)
        ax.set_title(f"Gradient Magnitude at the Beginning of Training", fontsize=18, pad=15)
        
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layers, ha="center", fontsize=14)
        
        ax.set_ylim(bottom=0)
        
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)

        legend_elements = [
            Line2D([0], [0], color=normal_color, lw=6, label='$W_{EE}, W_{IE}$, $g_I$, $g_E$'),
            Line2D([0], [0], color=highlight_color, lw=6, label='$W_{EI}$')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=True)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', dpi=300)
        buf.seek(0)
        image_data = buf.getvalue()
        buf.close()

        data_dir = os.path.join(self.tmp_gradients_dir, f'gradients')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_name = f'epoch{epoch:04d}_iter{iteration:07d}.pdf'
        image_path = os.path.join(data_dir, file_name)
        
        with open(image_path, 'wb') as f:
            f.write(image_data)

        plt.close(fig)

    def png2mp4(self):
        """Convert cached PNG images to MP4 videos."""
        print("Converting cached images to MP4 videos...")

        # make sure all images are saved
        self._shutdown_background_saver()

        # create video directory
        video_path = os.path.join(self.output_dir, 'videos')
        os.makedirs(video_path, exist_ok=True)

        def _create_video_from_images(image_dir):
            # load images from disk and create videos
            if not os.path.exists(image_dir):
                print("No image directory found, no videos to generate.")
                return
            
            try:
                sub_dirs = [d for d in os.listdir(image_dir) 
                            if os.path.isdir(os.path.join(image_dir, d))]
                
                if not sub_dirs:
                    print("No layer directories found, no videos to generate.")
                    return
                
                for sub_dir in sorted(sub_dirs):
                    try:
                        layer_path = os.path.join(image_dir, sub_dir)
                        images = [f for f in os.listdir(layer_path) if f.endswith('.png')]
                        if not images:
                            print(f"No images found in {layer_path}, skipping.")
                            continue
                        images.sort()  # sort images by name
                        video_name = f"{sub_dir}.mp4"
                        video_file = os.path.join(video_path, video_name)
                        with imageio.get_writer(video_file, mode='I', fps=30) as video:
                            for image in images:
                                try:
                                    image_path = os.path.join(layer_path, image)
                                    image_array = imageio.imread(image_path)
                                    video.append_data(image_array)  # type: ignore
                                except Exception as e:
                                    print(f"Error processing image {image}: {e}")
                                    continue
                    except Exception as e:
                        print(f"Error processing layer directory {sub_dir}: {e}")
                        continue
            except Exception as e:
                print(f"Error during video generation: {e}")
            finally:
                shutil.rmtree(image_dir)

        # create videos from states and gradients
        _create_video_from_images(self.tmp_states_dir)
        _create_video_from_images(self.tmp_gradients_dir)
        print("Video generation completed.")