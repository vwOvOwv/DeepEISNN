# DeepEISNN

Code for paper [Training Deep Normalization-Free Spiking Neural Networks with
Lateral Inhibition](https://openreview.net/pdf?id=U8preGvn5G).

![overview](assets/overview.png)

## Table of Contents

- [🚀 Quick Start](#-quick-start)
  - [Environment setup](#environment-setup)
  - [Train from scratch](#train-from-scratch)
  - [Resume from checkpoints](#resume-from-checkpoints)
- [🎓 Learn More about E-I Interaction](#-learn-more-about-e-i-interaction)
  - [Code Structure](#code-structure)
  - [Usage](#usage)
  - [Visualization](#visualization)
- [Citation](#citation)
- [Contact](#contact)

## 🚀 Quick Start

### Environment setup

All experiments are conducted with

- NVIDIA GeForce RTX 4090 GPU
- Python 3.12.10
- CUDA 12.6
- PyTorch 2.4.1

Create the conda environment with the following command:

```bash
conda env create -f environment.yml
conda activate deepeisnn
```

### Train from scratch

We provide all training configurations in `configs/EI-SNN`.

Simply run the following command to reproduce the results on CIFAR-10 with
ResNet-18 in the paper.

```bash
python train.py --scratch configs/EI-SNN/CIFAR10-ResNet18.yaml
```

Please modify `data_path` and `output_dir` in the configuration file before
training.

Other configuration files are also named in the format of
`[Dataset]-[Arch].yaml`. See `configs/EI-SNN` and `configs/BN-SNN` for more
examples.

### Resume from checkpoints

To resume training from checkpoints, run the following command:

```bash
python train.py --resume [Checkpoint-Path]
```

## 🎓 Learn More about E-I Interaction

Below is detailed information about this repo, which may help further research on E-I dynamics based on our work.

### Code Structure

```text
DeepEISNN/
├── configs/            # YAML configs for EI-SNN and BN-SNN
│   ├── EI-SNN/         # EI-SNN experiment configs
│   └── BN-SNN/         # BN-SNN experiment configs
├── datasets/           # dataset loaders and preprocessing
├── models/             # MLP/VGG/ResNet architectures
├── modules/            # layers, normalizations, and optimizers
├── utils/              # metrics, visualization, etc.
└── train.py            # training script
```

To keep consistent with standard `Conv/Linear-Norm-Activation` structure, we split 
the E-I layer into `EiConv/EiLinear-EiNorm-Activation`:

- `modules/linear.py`: `SpikingEiLinear` implements E-I linear layers with separate E-to-E and E-to-I pathways, the dynamic initialization, and non-negative clamping.
- `modules/conv2d.py`: `SpikingEiConv2d` is the convolutional counterpart to `SpikingEiLinear`.
- `modules/norm1d.py` and `modules/norm2d.py`: `SpikingEiNorm1d/2d` integrate E-I currents via E-I balance and gain control.

We also implement an SGD optimizer for E-I models:

- `modules/optimizers.py`: `EiSGD` is an SGD variant with optional non-negative clamping for E-I constraints.

### Usage

Key arguments in `train.py`:

- `--scratch <config.yaml>`: train from scratch
- `--resume <ckpt.pth>`: resume training
- `--log {0,1,2}`: logging level
- `--notes <str>`: append notes to the run name
- `--seed <int>`: random seed (default 2025)

To build E-I models, you can either use the provided E-I configs or use the E-I modules in your own model definition. The E-I modules follow the `Conv/Linear-Norm-Activation` convention, but require extra arguments and pass E-I currents between modules:

- `Conv2d` -> `SpikingEiConv2d(in_channels, out_channels, ei_ratio, device, rng, ...)`
- `Linear` -> `SpikingEiLinear(in_features, out_features, ei_ratio, device, rng, ...)`
- `BatchNorm2d` -> `SpikingEiNorm2d(num_features, prev_in_features, ei_ratio, device)`
- `BatchNorm1d` -> `SpikingEiNorm1d(num_features, prev_in_features, ei_ratio, device, output_layer)` (set `output_layer=True` for the final norm)

Here, `prev_in_features` is the fan-in ($d$ in the paper) of the `SpikingEiConv2d/Linear` layer. 

When constructing model, keep the E-I module sequence as `EiConv/EiLinear-EiNorm-Activation`. Note that `EiNorm` does not collect statistics to perform explicit normalization during training.

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

If you have any questions, please feel free to contact me via
[email](mailto:512120lpy@stu.pku.edu.cn) or raise an issue.
