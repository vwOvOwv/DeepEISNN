import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Any, Union, Literal

__all__ = [
    'SpikingConv2d', 'SpikingEiConv2d'
]


class SpikingConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, tuple[int, int]], 
                 stride: Union[int, tuple[int, int]] = 1, 
                 padding: Union[int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1, 
                 groups: int = 1, bias: bool = False, 
                 padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, dilation, groups, bias, padding_mode)
        
        self.visualize_cache = {}
        self._need_visualize = False
        self._inited = True

    def _set_visualize_cache(self, *args: Any) -> None:
        with torch.no_grad():
            input, output = args
            self.visualize_cache['param1:weight'] = self.conv.weight.detach()
            self.visualize_cache['data1:input'] = input.detach()
            self.visualize_cache['data2:output'] = output.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv.forward(input)
        if self._need_visualize:
            self._set_visualize_cache(input, output)
        return output
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag
    

class SpikingEiConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ei_ratio: int, 
                 device: torch.device, rng: np.random.Generator,
                 kernel_size: Union[int, tuple[int, int]], 
                 stride: Union[int, tuple[int, int]] = 1, 
                 padding: Union[int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1, 
                 groups: int = 1, bias: bool = False, 
                 padding_mode: str = 'zeros') -> None:
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
        with torch.no_grad():
            Var_x = batch_stats['Var_x']
            E_x_square = batch_stats['E_x_square']
            self.exp_scale = np.sqrt(Var_x / (self.in_features * (E_x_square + Var_x)))

            weight_ee_np = self.rng.exponential(scale=self.exp_scale, size=(self.conv_ee.weight.shape))
            self.conv_ee.weight.data = torch.from_numpy(weight_ee_np).float().to(self.device)

            weight_ie_np = self.rng.exponential(scale=self.exp_scale, size=(self.conv_ie.weight.shape))
            self.conv_ie.weight.data = torch.from_numpy(weight_ie_np).float().to(self.device)

    def _get_batch_stats(self, x: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            batch_stats = {}
            batch_stats['E_x'] = x.mean().item()
            batch_stats['Var_x'] = x.var(dim=0).mean().item()
            batch_stats['E_x_square'] = (x ** 2).mean().item()
            return batch_stats
    
    def _set_visualize_cache(self, *args) -> None:
        with torch.no_grad():
            input, I_ee, I_ie = args
            self.visualize_cache['param1:weight_ee'] = self.conv_ee.weight.detach()
            self.visualize_cache['param2:weight_ie'] = self.conv_ie.weight.detach()
            self.visualize_cache['data1:input'] = input.detach()
            self.visualize_cache['data2:I_ee'] = I_ee.detach()
            self.visualize_cache['data3:I_ie'] = I_ie.detach()

    def _clamp_parameters(self) -> None:
        with torch.no_grad():
            self.conv_ee.weight.data.clamp_(min=0)
            self.conv_ie.weight.data.clamp_(min=0)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Union[dict, None]]:
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