# DeepEISNN

Code for paper [Training Deep Normalization-Free Spiking Neural Networks with
Lateral Inhibition](https://openreview.net/pdf?id=U8preGvn5G).

![overview](assets/overview.png)

## Table of Contents

## 🚀 Quick Start

### Environment setup

All experiments are conducted with

- NVIDIA GeForce RTX 4090 GPU
- Python 3.12.10
- CUDA 12.6
- PyTorch 2.4.1

Create the conda environment with the following command:

```bash
conda create -f environment.yml
```

### Train from scratch

We provide all training configurations in `configs/EI-SNN`.

Simply run the following command to reproduce the results on CIFAR-10 with ResNet-18 in the paper.

```bash
python train.py --scratch configs/EI-SNN/CIFAR10-ResNet18.yaml
```

Please modify `data_path` and `output_dir` in the configuration file before training.

Other configuration files are also named in the format of `[Dataset]-[Arch].yaml`. See `configs/EI-SNN` for more details.

### Resume from checkpoints

To resume training from checkpoints, run the following command:

```bash
python train.py --resume [Checkpoint-Path]
```

## 🎓 Learn More about E-I Interaction

Below is detailed information about this repo, which may help further research on E-I dynamics based on our work.

### Code Structure

### Usage

### Visualization

We provide visualization demonstrating evolving distributions of 
- all trainable parameters
- inputs/outputs of all layers

during training.

Simply add `--log 2` in the shell command to enable visualization. For example:

```bash
python train.py --scratch configs/EI-SNN/[Dataset]-[Arch].yaml --log 2
```

👀 See how bimodality emerges during training here!

<!-- TODO -->

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@inproceedings{
liu2026training,
title={Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition},
author={Peiyu Liu and Jianhao Ding and Zhaofei Yu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=U8preGvn5G}
}
```

## Contact

If you have any questions, please feel free to contact me via [email](mailto:512120lpy@stu.pku.edu.cn) or raise an issue.