import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Any, Union, Literal

__all__ = [
    'SpikingConv2d', 'SpikingEiConv2d'
]


class SpikingConv2d(nn.Module):
    """
    2D convolution layer for SNN models.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, tuple[int, int]], 
                 stride: Union[int, tuple[int, int]] = 1, 
                 padding: Union[int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1, 
                 groups: int = 1, bias: bool = False, 
                 padding_mode: str = 'zeros') -> None:
        """
        Initialize `SpikingStandardConv2d`.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple): Stride of the convolution. Default is 1.
            padding (int or tuple): Zero-padding added to both sides of the input. Default is 0.
            dilation (int or tuple): Spacing between kernel elements. Default is 1.
            groups (int): Number of blocked connections from input channels to output channels. Default is 1.
            bias (bool): If True, adds a learnable bias to the output. Default is False.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, dilation, groups, bias, padding_mode)
        
        self.visualize_cache = {}
        self._need_visualize = False
        self._inited = True

    def _set_visualize_cache(self, *args: Any) -> None:
        """
        Set visualization cache of `SpikingStandardConv2d`.
        Args:
            input (torch.Tensor): Input tensor.
            output (torch.Tensor): Output tensor.
            weight (torch.Tensor): Weight tensor.
        """
        with torch.no_grad():
            input, output = args
            self.visualize_cache['param1:weight'] = self.conv.weight.detach()
            self.visualize_cache['data1:input'] = input.detach()
            self.visualize_cache['data2:output'] = output.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `SpikingStandardConv2d`.
        Args:
            input (torch.Tensor): Input tensor. Shape `(T * batch_size, in_channels, H_in, W_in)`.
        Returns:
            torch.Tensor: Output tensor. Shape `(T * batch_size, out_channels, H_out, W_out)`.
        """
        output = self.conv.forward(input)
        if self._need_visualize:
            self._set_visualize_cache(input, output)
        return output
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag
    

class SpikingEiConv2d(nn.Module):
    """
    2D convolution layer for DSNN models.
    """
    def __init__(self, ei_ratio: int, in_channels: int, out_channels: int,
                 rng: np.random.Generator,
                 kernel_size: Union[int, tuple[int, int]], 
                 stride: Union[int, tuple[int, int]] = 1, 
                 padding: Union[int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1, 
                 groups: int = 1, bias: bool = False, 
                 padding_mode: str = 'zeros', 
                 device: torch.device = torch.device('cpu')) -> None:
        """
        Args:
            T (int): Number of time steps.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels. Here, it is the number of excitatory neurons.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple): Stride of the convolution. Default is 1.
            padding (int or tuple): Size of padding added to both sides of the input. Default is 0.
            dilation (int or tuple): Spacing between kernel elements. Default is 1.
            groups (int): Number of blocked connections from input channels to output channels. Default is 1.
            bias (bool): If True, adds a learnable bias to the output. Default is False.
            padding_mode (str): Padding mode (e.g., zero padding).
            device (torch.device): Where the model lives in. Used in dynamic initialization.
        """
        super().__init__()
        self.n_e = out_channels
        self.n_i = self.n_e // ei_ratio
        self.rng = rng
        self.device = device

        self.conv_ee = nn.Conv2d(in_channels, self.n_e, kernel_size, stride, 
                                 padding, dilation, groups, bias, padding_mode,
                                 device=self.device)
        self.conv_ie = nn.Conv2d(in_channels, self.n_i, kernel_size, stride, 
                                 padding, dilation, groups, bias, padding_mode,
                                 device=self.device)
        self.in_features = int(np.prod(self.conv_ee.weight.shape[1:]))

        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = False
    
    def _dynamic_init(self, batch_stats: dict[str, Any]) -> None:
        """
        Dynamically initialize EiConv2d.
        Args:
            batch_stats (dict[str, Any]): Batch statistics for dynamic initialization.
        """
        with torch.no_grad():
            Var_x = batch_stats['Var_x']
            E_x_square = batch_stats['E_x_square']
            self.exp_scale = np.sqrt(Var_x / (self.in_features * (E_x_square + Var_x)))

            weight_ee_np = self.rng.exponential(scale=self.exp_scale, size=(self.conv_ee.weight.shape))
            self.conv_ee.weight.data = torch.from_numpy(weight_ee_np).float().to(self.device)

            weight_ie_np = self.rng.exponential(scale=self.exp_scale, size=(self.conv_ie.weight.shape))
            self.conv_ie.weight.data = torch.from_numpy(weight_ie_np).float().to(self.device)

    def _get_batch_stats(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Get batch statistics from the input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            batch_stats (dict[str, Any]): Dictionary containing
                - 'E_x': Mean of the input features.
                - 'Var_x': Variance of the input features.
                - 'E_x_square': Second raw moment of the input features.
        """
        with torch.no_grad():
            batch_stats = {}
            batch_stats['E_x'] = x.mean().item()
            batch_stats['Var_x'] = x.var(dim=0).mean().item()
            batch_stats['E_x_square'] = (x ** 2).mean().item()
            return batch_stats
    
    def _set_visualize_cache(self, *args) -> None:
        """
        Set visualization cache of `SpikingEiConv2d`.
        Args:
            input (torch.Tensor): Input tensor.
            I_ee (torch.Tensor): Excitatory input.
            I_ie (torch.Tensor): Inhibitory input.
        """
        with torch.no_grad():
            input, I_ee, I_ie = args
            self.visualize_cache['param1:weight_ee'] = self.conv_ee.weight.detach()
            self.visualize_cache['param2:weight_ie'] = self.conv_ie.weight.detach()
            self.visualize_cache['data1:input'] = input.detach()
            self.visualize_cache['data2:I_ee'] = I_ee.detach()
            self.visualize_cache['data3:I_ie'] = I_ie.detach()

    def _clamp_parameters(self) -> None:
        """
        Clamp the parameters of `SpikingEiConv2d` to ensure they are non-negative.
        """
        with torch.no_grad():
            self.conv_ee.weight.data.clamp_(min=0)
            self.conv_ie.weight.data.clamp_(min=0)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Union[dict, None]]:
        """
        Forward pass of `SpikingEiConv2d`.
        Args:
            input (torch.Tensor): Input tensor. Shape `(T * batch_size, in_channels, H_in, W_in)`.
        Returns:
            I_ee (torch.Tensor): Excitatory input. Shape `(T * batch_size, n_e, H_out, W_out)`.
            I_ie (torch.Tensor): Inhibitory input. Shape `(T * batch_size, n_i, H_out, W_out)`.
        """
        self._clamp_parameters()

        batch_stats = None
        if self._inited == False:
            batch_stats = self._get_batch_stats(input)
            self._dynamic_init(batch_stats)
            self._inited = True

        I_ee = self.conv_ee.forward(input)
        I_ie = self.conv_ie.forward(input)
        if self._need_visualize:
            self._set_visualize_cache(input, I_ee, I_ie)
        return I_ee, I_ie, batch_stats
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag