"""Load and preprocess normal datasets."""

import os
from typing import Callable, Optional

import torchvision.datasets
from torchvision import transforms
from torchtoolbox.transform import Cutout
from torch.utils.data import Dataset
from PIL import Image


def get_mnist(data_path: str):
    """Load MNIST dataset.

    Args:
        data_path: Path to the dataset.

    Returns:
        Tuple of (train_set, val_set).
    """
    print("Loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(data_path, train=True,
                                           transform=transform_train, download=True)
    val_set = torchvision.datasets.MNIST(data_path, train=False,
                                         transform=transform_val, download=True)

    return train_set, val_set


def get_cifar10(data_path: str):
    """Load CIFAR10 dataset.

    Args:
        data_path: Path to the dataset.

    Returns:
        Tuple of (train_set, val_set).
    """
    print("Loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_set = torchvision.datasets.CIFAR10(data_path, train=True,
                                             transform=transform_train, download=True)
    val_set = torchvision.datasets.CIFAR10(data_path, train=False,
                                            transform=transform_val, download=True)

    return train_set, val_set


def get_cifar100(data_path: str):
    """Load CIFAR100 dataset.

    Args:
        data_path: Path to the dataset.

    Returns:
        Tuple of (train_set, val_set).
    """
    print("Loading CIFAR100")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_set = torchvision.datasets.CIFAR100(data_path, train=True,
                                              transform=transform_train, download=True)
    val_set = torchvision.datasets.CIFAR100(data_path, train=False,
                                            transform=transform_val, download=True)

    return train_set, val_set


class TinyImageNetValDataset(Dataset):
    """TinyImageNet-200 validation dataset.

    Constructed from the `val` folder and `val_annotations.txt`.
    """
    def __init__(
        self,
        val_dir: str,
        class_to_idx: dict[str, int],
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        """Initialize the validation dataset.

        Args:
            val_dir: Path to the validation directory.
            class_to_idx: Mapping from class names to indices.
            transform: Optional transforms applied to images.
        """
        self.val_dir = val_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.images_dir = os.path.join(val_dir, 'images')
        self.annotations_file = os.path.join(val_dir, 'val_annotations.txt')

        self.image_paths = []
        self.labels = []

        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                image_filename = parts[0]
                class_id = parts[1]

                image_path = os.path.join(self.images_dir, image_filename)
                self.image_paths.append(image_path)

                self.labels.append(self.class_to_idx[class_id])

    def __len__(self):
        """Return the number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, label).
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_tinyimagenet(data_path: str):
    """Load TinyImageNet-200 dataset.

    Args:
        data_path: Path to the dataset.

    Returns:
        Tuple of (train_set, val_set).
    """
    print("Loading TinyImageNet-200")
    os.makedirs(data_path, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        Cutout(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)

    class_to_idx = train_set.class_to_idx
    val_set = TinyImageNetValDataset(
        val_dir=val_dir,
        class_to_idx=class_to_idx,
        transform=transform_val
    )

    return train_set, val_set
