"""Helper datasets and augmentations for event-based data."""

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PackagingClass(Dataset):
    """Dataset wrapper that applies transforms to float tensors."""

    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Wrap a dataset to apply transforms after converting to float tensor.

        Args:
            dataset: Underlying dataset instance.
            transform: Optional transform callable.
        """
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index: int):
        """Get a transformed sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of (data, label).
        """
        data, label = self.dataset[index]
        data = torch.FloatTensor(data)
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        """Return the number of samples."""
        return len(self.dataset)


class Cutout:
    """Apply a random square mask to the input tensor."""

    def __init__(self, length: int) -> None:
        """Initialize the cutout transform.

        Args:
            length: Side length of the cutout square.
        """
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout to the input tensor.

        Args:
            img: Input tensor with shape (C, H, W) or (T, C, H, W).

        Returns:
            Tensor with cutout applied.
        """
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def function_nda(data: torch.Tensor, M: int = 1, N: int = 2) -> torch.Tensor:
    """Apply a sequence of random augmentations for DVS data.

    Args:
        data: Input tensor.
        M: Number of random augmentations to apply.
        N: Strength multiplier for augmentations.

    Returns:
        Augmented tensor.
    """
    c = 15 * N
    rotate_tf = transforms.RandomRotation(degrees=c)
    e = 8 * N
    cutout_tf = Cutout(length=e)

    def roll(data: torch.Tensor, N: int = 1) -> torch.Tensor:
        """Randomly roll the tensor spatially."""
        a = N * 2 + 1
        off1 = np.random.randint(-a, a + 1)
        off2 = np.random.randint(-a, a + 1)
        return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

    def rotate(data: torch.Tensor, N: int) -> torch.Tensor:
        """Randomly rotate the tensor."""
        return rotate_tf(data)

    def cutout(data: torch.Tensor, N: int) -> torch.Tensor:
        """Apply cutout augmentation."""
        return cutout_tf(data)

    transforms_list = [roll, rotate, cutout]
    sampled_ops = np.random.choice(transforms_list, M)
    for op in sampled_ops:
        data = op(data, N)
    return data
