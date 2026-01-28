"""Evaluation utilities for classification metrics."""

from typing import Sequence

import torch

class AverageMeter:
    """Track and update running averages of scalar values.

    Note:
        Adapted from the PyTorch ImageNet example.
    """

    def __init__(self):
        """Initialize and reset meters."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter with a new value.

        Args:
            val: Value to add.
            n: Number of samples represented by the value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Sequence[int] = (1,),
):
    """Compute top-k accuracy for specified values of k.

    Args:
        output: Model outputs with shape (B, C).
        target: Ground-truth labels with shape (B,).
        topk: Iterable of k values.

    Returns:
        List of accuracies for each k in `topk`.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
