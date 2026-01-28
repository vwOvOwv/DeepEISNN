"""Load and preprocess DVS datasets."""

import os

import torch
from torchvision import transforms
import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets import split_to_train_test_set, RandomTemporalDelete

from utils.dvs_datasets import PackagingClass, function_nda


def get_dvs128gesture(data_path: str, T: int):
    """Load DVS128Gesture dataset.

    Inputs are resized to 48x48.

    Args:
        data_path: Path to the dataset.
        T: Total time steps.

    Returns:
        Tuple of (train_set, val_set).
    """
    print("Loading DVS128Gesture")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0),
                                     interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Resize(size=(48, 48)),
        transforms.RandomHorizontalFlip(),
        RandomTemporalDelete(T_remain=T, batch_first=False),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(size=(48, 48)),
    ])

    train_set = DVS128Gesture(data_path, data_type='frame', frames_number=T,
                              split_by='number', train=True)
    val_set = DVS128Gesture(data_path, data_type='frame', frames_number=T,
                            split_by='number', train=False)
    train_set, val_set = PackagingClass(train_set, transform_train), \
                        PackagingClass(val_set, transform_val)
    return train_set, val_set


def get_cifar10dvs(data_path: str, T: int):
    """Load CIFAR10-DVS dataset.

    Inputs are resized to 48x48.

    Args:
        data_path: Path to the dataset.
        T: Total time steps.

    Returns:
        Tuple of (train_set, val_set).
    """
    print("Loading CIFAR10-DVS")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    def transform_train(data: torch.Tensor) -> torch.Tensor:
        data = transforms.RandomResizedCrop(128, scale=(0.7, 1.0),
                                        interpolation=transforms.InterpolationMode.NEAREST)(data)
        resize = transforms.Resize(size=(48, 48))
        data = resize(data).float()
        flip = np.random.random() > 0.5
        if flip:
            data = torch.flip(data, dims=(3,))
        data = function_nda(data)
        return data.float()

    def transform_val(data: torch.Tensor) -> torch.Tensor:
        resize = transforms.Resize(size=(48, 48))
        data = resize(data).float()
        return data.float()

    dataset = CIFAR10DVS(data_path, data_type='frame', frames_number=T, split_by='number')
    train_set, val_set = split_to_train_test_set(train_ratio=0.9,
                                                 origin_dataset=dataset,
                                                 num_classes=10)
    train_set, val_set = PackagingClass(train_set, transform_train), \
        PackagingClass(val_set, transform_val)
    return train_set, val_set
