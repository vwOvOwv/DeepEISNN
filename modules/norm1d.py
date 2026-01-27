import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .activation import LIF
from typing import Any, Literal, Union

__all__ = [
    'SpikingBatchNorm1d', 'SpikingEiNorm1d'
]


class SpikingBatchNorm1d(nn.Module):
    """
    1D batch normalization layer for SNN models.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        """
        Initialize `SpikingStandardBatchNorm1d`.
        Args:
            T (int): Number of time steps.
            num_features (int): Number of features in the input.
            eps (float): A value added to the denominator for numerical stability.
            momentum (float): Momentum for the running mean and variance.
            affine (bool): If True, this module has learnable affine parameters.
            track_running_stats (bool): If True, this module tracks the running mean and variance.
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = True

    def _set_visualize_cache(self, *args) -> None:
        """
        Set visualization cache of `SpikingStandardBatchNorm1d`.
        Args:
            input (torch.Tensor): Input tensor.
            output (torch.Tensor): Output tensor.
        """
        with torch.no_grad():
            input, output = args
            self.visualize_cache['param1:gamma'] = self.bn.weight.detach()
            self.visualize_cache['param2:beta'] = self.bn.bias.detach()
            self.visualize_cache['data1:input'] = input.detach()
            self.visualize_cache['data2:output'] = output.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `SpikingStandardBatchNorm1d`.
        Args:
            input (torch.Tensor): Input tensor. Shape `(T * batch_size, num_features)`.
        Returns:
            torch.Tensor: Output tensor. Shape `(T * batch_size, num_features)`.
        """
        output = self.bn.forward(input)
        if self._need_visualize:
            self._set_visualize_cache(input, output)
        return output
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag


class SpikingEiNorm1d(nn.Module):
    """
    1D batch normalization layer for DSNN models.
    """
    def __init__(self, ei_ratio: int, num_features: int, prev_in_features: int, 
                 device: torch.device, output_layer: bool = False) -> None:
        """
        Initialize `SpikingEiNorm1d`.
        Args:
            T (int): Number of time steps.
            num_features (int): Number of features in the input.
            prev_in_features (int): Number of features input to the previous linear layer.
            ei_ratio (int): Ratio of excitatory to inhibitory neurons.
            output_layer (bool): If True, this is the output layer.
        """
        super().__init__()
        self.n_e = num_features
        self.n_i = num_features // ei_ratio
        self.output_layer = output_layer
        self.prev_in_features = prev_in_features
        self.device = device

        self.weight_ei = nn.Parameter(torch.ones(self.n_e, self.n_i, device=self.device) / self.n_i, requires_grad=True)
        self.alpha = nn.Parameter(torch.empty(1, self.n_i, device=self.device), requires_grad=True)
        self.gain = nn.Parameter(torch.ones(1, self.n_e, device=self.device), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, self.n_e, device=self.device), requires_grad=True)
        
        self.weight_ei.register_hook(lambda grad: grad / self.prev_in_features)
        # self.weight_ei.register_hook(lambda grad: grad / np.sqrt(self.prev_in_features))

        self.visualize_cache = {}
        self._need_visualize = False
        self._inited = False
    
    def _dynamic_init(self, batch_stats: dict[str, Any]) -> None:
        """
        Dynamic initialization of `SpikingEiNorm1d`.
        Args:
            batch_stats (dict[str, Any]): Batch statistics for initialization.
        """
        with torch.no_grad():
            E_x = batch_stats['E_x']
            Var_x = batch_stats['Var_x']
            E_x_square = batch_stats['E_x_square']
            self.alpha.data = torch.ones(1, self.n_i, device=self.device) / \
                np.sqrt(self.prev_in_features) * np.sqrt(E_x_square + Var_x) / E_x

    def _clamp_parameters(self) -> None:
        """
        Clamp the parameters of `SpikingEiNorm1d` to ensure they are non-negative.
        """
        with torch.no_grad():
            self.weight_ei.data.clamp_(min=0)
            self.alpha.data.clamp_(min=0)
            self.gain.data.clamp_(min=0)

    def _set_visualize_cache(self, *args) -> None:
        """
        Set visualization cache of `SpikingEiNorm1d`.
        Args:
            I_ei (torch.Tensor): Input inhibitory current.
            I_balanced (torch.Tensor): Balanced input current.
            I_shunting (torch.Tensor): Shunting inhibitory current.
            I_int (torch.Tensor): Integrated current.
        """
        with torch.no_grad():
            I_ei, I_balanced, I_shunting, I_int = args
            self.visualize_cache['param1: weight_ei'] = self.weight_ei.detach()
            self.visualize_cache['param2: alpha'] = self.alpha.detach()
            self.visualize_cache['param3: gain'] = self.gain.detach()
            self.visualize_cache['param4: bias'] = self.bias.detach()
            self.visualize_cache['data1: I_ei'] = I_ei.detach()
            self.visualize_cache['data2: I_balanced'] = I_balanced.detach()
            self.visualize_cache['data3: I_shunting'] = I_shunting.detach()
            self.visualize_cache['data4: I_int'] = I_int.detach()

    def replace_zero_with_second_min(self, input: torch.Tensor):
        """
        Replace zero values in the tensor with the second minimum value.
        This method is used to avoid division by zero or log(0) issues.
        Args:
            input (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Tensor with zero values replaced by the second minimum value.
        """
        has_zero = (input == 0.).any()
    
        if has_zero:
            mask = (input == 0.)
            tmp = input.clone()
            tmp[mask] = float('inf')

            batch_size = tmp.size(0)
            tmp_flat = tmp.reshape(batch_size, -1)
            sample_wise_second_min_flat, _ = torch.min(tmp_flat, dim=1, keepdim=True)
            second_min = sample_wise_second_min_flat.view(batch_size, *([1] * (input.dim() - 1)))
            # output = input + second_min * mask.float()
            forward_output = torch.where(mask, second_min, input)
            output = forward_output.detach() + (input - input.detach()) # STE
            return output
        else:
            return input

    def forward(self, input: tuple[torch.Tensor, torch.Tensor, Union[dict, None]]):
        """
        Forward pass of `SpikingEiNorm1d`.
        Args:
            input (tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - I_ee (torch.Tensor): Input excitatory current. Shape `(T * batch_size, n_e)`.
                - I_ie (torch.Tensor): Input inhibitory current. Shape `(T * batch_size, n_i)`.
        Returns:
            torch.Tensor: Output tensor. Shape `(T * batch_size, n_e)`.
        """
        self._clamp_parameters()
        I_ee, I_ie, batch_stats = input
        if self._inited == False and batch_stats is not None:
            self._dynamic_init(batch_stats)
            self._inited = True

        I_ei = torch.matmul(self.weight_ei, I_ie.T).T   # shape: (B, n_e)
        
        I_balanced = I_ee - I_ei    # shape: (B, n_e)
        
        I_shunting = torch.matmul(self.weight_ei, (self.alpha * I_ie).T).T    # shape: (B, n_e)
        I_shunting_adjusted = self.replace_zero_with_second_min(I_shunting)  # avoid division by zero

        I_int = self.gain * I_balanced / I_shunting_adjusted + self.bias

        if self._need_visualize:
            self._set_visualize_cache(I_ei, I_balanced, I_shunting_adjusted, I_int)
        
        if self.output_layer:
            return I_balanced
        return I_int
    
    def set_visualize(self, flag: bool):
        self._need_visualize = flag