"""Visualization utilities for model internals."""

import io
import os
import shutil
import threading
from queue import Queue

import imageio
import matplotlib

matplotlib.use('agg')  # use a non-interactive backend for matplotlib

import matplotlib.pyplot as plt
import torch

class Visualizer:
    """Save layer-wise histograms of cached tensors for visualization.

    Example:
        model.set_visualize(True)
        output = model(input)
        visualizer.visualize_model_states(model, epoch, global_steps + 1)
        model.set_visualize(False)
    """
    def __init__(self, output_dir: str) -> None:
        """Initialize the visualizer.

        Args:
            output_dir: Root directory to store outputs.
        """
        self.output_dir = output_dir
        self.tmp_states_dir = os.path.join(self.output_dir, 'tmp-states')
        os.makedirs(self.tmp_states_dir, exist_ok=True)

        self.save_queue = Queue()
        self._shutdown_flag = False
        self._start_background_saver()

    def _start_background_saver(self):
        """Start a background thread to save images to disk."""
        def background_saver():
            """Consume queued images and write them to disk."""
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
        """Plot a histogram for a cached tensor and queue it for saving.

        Args:
            layer_name: Name of the layer.
            layer_idx: Layer index in the model.
            x: Tensor to visualize.
            name: Cache key name.
            epoch: Current epoch number.
            iteration: Global iteration number.
        """
        with torch.no_grad():
            x_flat = x.reshape(-1)
            min_val, max_val = x_flat.min().item(), x_flat.max().item()
            mean_val, std_val = x_flat.mean().item(), x_flat.std().item()
            counts, bins = torch.histogram(x_flat.cpu(), bins=100, density=False)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

        fig, ax = plt.subplots(figsize=(12, 8))

        # plot histogram
        ax.bar(
            bin_centers,
            counts,
            width=bins[1] - bins[0],
            alpha=0.7,
            color='blue',
            edgecolor='black',
        )
        ax.set_xlabel(
            f"min: {min_val:.6f}, max: {max_val:.6f}, "
            f"mean: {mean_val:.6f}, std: {std_val:.6f}"
        )
        ax.set_ylabel('Frequency')
        ax.set_title(
            f"Epoch{epoch}, Iter{iteration}, Layer{layer_idx}:{layer_name}-{name}"
        )

        # save the figure to I/O buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_data = buf.getvalue()
        buf.close()

        # save the figure to a temporary image file through background saver
        data_dir = os.path.join(
            self.tmp_states_dir,
            f'layer{layer_idx:04d}_{layer_name}_{name}',
        )
        file_name = f'epoch{epoch:04d}_iter{iteration:07d}.png'
        image_path = os.path.join(data_dir, file_name)
        self.save_queue.put((image_data, image_path))

        plt.close(fig)

    def visualize_model_states(self, model: torch.nn.Module,
                               epoch: int, iteration: int) -> None:
        """Generate histogram plots for cached layer states.

        Args:
            model: Model containing visualization caches.
            epoch: Current epoch number.
            iteration: Global iteration number.
        """
        with torch.no_grad():
            layer_idx = 0
            for layer in model.layers:
                layer_idx += 1
                if hasattr(layer, 'visualize_cache'):
                    for (name, val) in layer.visualize_cache.items():
                        self._plot_hist(
                            layer.__class__.__name__,
                            layer_idx,
                            val,
                            name,
                            epoch,
                            iteration,
                        )
                    layer.visualize_cache.clear()

    def png2mp4(self):
        """Convert cached PNGs into MP4 videos and cleanup."""
        print("Converting cached images to MP4 videos...")

        # make sure all images are saved
        self._shutdown_background_saver()

        # create video directory
        video_path = os.path.join(self.output_dir, 'videos')
        os.makedirs(video_path, exist_ok=True)

        def _create_video_from_images(image_dir: str) -> None:
            """Create MP4 videos from PNGs in a directory."""
            # load images from disk and create videos
            if not os.path.exists(image_dir):
                print("No image directory found, no videos to generate.")
                return

            try:
                sub_dirs = [
                    d for d in os.listdir(image_dir)
                    if os.path.isdir(os.path.join(image_dir, d))
                ]

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
                                    video.append_data(image_array)
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
        print("Video generation completed.")
