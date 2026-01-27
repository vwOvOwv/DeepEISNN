import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Any, Union, Literal
import torch.nn.functional as F

__all__ = [
    'SpikingLinear', 'SpikingEiLinear'
]


class SpikingLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Linear layer for SNN.
        Args:
            T (int): Number of time steps. Used for visualization.
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, adds a learnable bias to the output. Default is True.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = True

    def _set_visualize_cache(self, *args) -> None:
        """
        Set visualization cache for the layer.
        Args:
            input (torch.Tensor): Input tensor.
            output (torch.Tensor): Output tensor.
            weight (torch.Tensor): Weight tensor.
        """
        with torch.no_grad():
            print("called")
            input, output = args
            self.visualize_cache['param1:weight'] = self.linear.weight.detach()
            bias = self.linear.bias.detach() if self.linear.bias is not None else None
            if bias is not None:
                self.visualize_cache['param2:bias'] = bias
            
            self.visualize_cache['data1:input'] = input.detach()
            self.visualize_cache['data2:output'] = output.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SpikingStandardLinear.
        Args:
            input (torch.Tensor): Input tensor. Shape `(T * batch_size, in_features)`.
        Returns:
            torch.Tensor: Output tensor. Shape `(T * batch_size, out_features)`.
        """
        output = self.linear.forward(input)
        if self._need_visualize:
            self._set_visualize_cache(input, output)
        return output

    def set_visualize(self, flag: bool):
        self._need_visualize = flag


class SpikingEiLinear(nn.Module):
    """
    Linear layer for DSNN models. Inherits from `SpikingEiModule`.
    """
    def __init__(self, ei_ratio: int, in_features: int, 
                 out_features: int, device: torch.device, rng: np.random.Generator,
                 output_layer: bool = False, ) -> None:
        """
        Initialize `SpikingEiLinear`.
        Args:
            T (int): Number of time steps. Used for visualization.
            ei_ratio (int): Ratio of excitatory to inhibitory neurons.
            in_features (int): Number of input features. Here, `in_features` equals to the number of excitatory neurons in the previous layer.
            out_features (int): Number of output features.
            output_layer (bool): If True, this is the output layer.
        """
        super().__init__()
        self.n_e = out_features
        self.n_i = self.n_e // ei_ratio
        self.in_features = in_features
        self.device = device
        self.rng = rng
        self.output_layer = output_layer
        self.weight_ee = nn.Parameter(torch.empty(self.n_e, self.in_features, device=self.device), requires_grad=True)
        self.weight_ie = nn.Parameter(torch.empty(self.n_i, self.in_features, device=self.device), requires_grad=True)

        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = False

    def _clamp_parameters(self) -> None:
        """
        Clamp `weight_ee` and `weight_ie` to ensure they are non-negative.
        """
        with torch.no_grad():
            self.weight_ee.clamp_(min=0)
            self.weight_ie.clamp_(min=0)

    def _dynamic_init(self, batch_stats: dict[str, float]) -> None:
        """
        Dynamically initialize `SpikingEiLinear` based on batch statistics.
        Args:
            batch_stats (Optional[dict[str, float]]): Statistics from the current batch.
                -   Var_x: Variance of the input features.
                -   E_x_square: Second raw moment of the input features.
        """
        with torch.no_grad():
            Var_x = batch_stats['Var_x']
            E_x_square = batch_stats['E_x_square']
            self.exp_scale = np.sqrt(Var_x / (self.in_features * (E_x_square + Var_x)))

            weight_ee_np = self.rng.exponential(scale=self.exp_scale, 
                                                size=(self.n_e, self.in_features))
            self.weight_ee.data = torch.from_numpy(weight_ee_np).float().to(self.device)

            weight_ie_np = self.rng.exponential(scale=self.exp_scale, 
                                                size=(self.n_i, self.in_features))
            self.weight_ie.data = torch.from_numpy(weight_ie_np).float().to(self.device)

    def _get_batch_stats(self, x:torch.Tensor):
        """
        Get batch statistics for the input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        """
        with torch.no_grad():
            batch_stats = {}
            batch_stats['E_x'] = x.mean().item()
            batch_stats['Var_x'] = x.var(dim=0).mean().item()
            batch_stats['E_x_square'] = (x ** 2).mean().item()
            return batch_stats

    def _set_visualize_cache(self, *args) -> None:
        """
        Set visualization cache of `SpikingEiLinear`.
        Args:
            input (torch.Tensor): Input tensor.
            I_ee (torch.Tensor): Excitatory input.
            I_ie (torch.Tensor): Inhibitory input.
        """
        with torch.no_grad():
            input, I_ee, I_ie = args
            self.visualize_cache['param1:weight_ee'] = self.weight_ee.detach()
            self.visualize_cache['param2:weight_ie'] = self.weight_ie.detach()
            I_ie_np = I_ie.detach() if not self.output_layer else None
            self.visualize_cache['data1:input'] = input.detach()
            self.visualize_cache['data2:I_ee'] = I_ee.detach()
            if I_ie is not None:
                self.visualize_cache['data3:I_ie'] = I_ie_np

    def forward(self, x:torch.Tensor) \
        -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, Union[dict, None]]]:
        """
        Forward pass of `SpikingEiLinear`.
        Args:
            x (torch.Tensor): Input tensor. Shape `(T * batch_size, in_features)`.
        Returns:
            I_ee (torch.Tensor): Excitatory input. Shape `(T * batch_size, n_e)`.
            I_ie (torch.Tensor): Inhibitory input, if not output layer. Shape `(T * batch_size, n_i)`.
        """
        self._clamp_parameters()
        batch_stats = None
        if self._inited == False:
            batch_stats = self._get_batch_stats(x)
            self._dynamic_init(batch_stats)
            self._inited = True

        I_ee = torch.matmul(self.weight_ee, x.T).T # (*, n_e)
        I_ie = torch.matmul(self.weight_ie, x.T).T # (*, n_i)
        if self._need_visualize:
            self._set_visualize_cache(x, I_ee, I_ie)
        if self.output_layer:
            return I_ee
        return I_ee, I_ie, batch_stats
    
    def set_visualize(self, flag: bool):
        self._need_visualize = flag