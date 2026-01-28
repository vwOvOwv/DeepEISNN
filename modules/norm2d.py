import torch
import torch.nn as nn
import numpy as np
from typing import Any, Literal, Union

__all__ = [
    'SpikingBatchNorm2d', 'SpikingEiNorm2d'
]


class SpikingBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        
        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = True

    def _set_visualize_cache(self, *args) -> None:
        with torch.no_grad():
            input, output = args
            self.visualize_cache['param1: gamma'] = self.bn.weight.detach()
            self.visualize_cache['param2: beta'] = self.bn.bias.detach()
            self.visualize_cache['data1: input'] = input.detach()
            self.visualize_cache['data2: output'] = output.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.bn.forward(input)
        with torch.no_grad():
            if self._need_visualize:
                self._set_visualize_cache(input, output)
        return output
    
    def set_visualize(self, flag: bool):
        self._need_visualize = flag
    

class SpikingEiNorm2d(nn.Module):
    def __init__(self, num_features: int, prev_in_features: int, 
                 ei_ratio: int, device: torch.device):
        super().__init__()
        self.n_e = num_features
        self.n_i = self.n_e // ei_ratio
        self.prev_in_features = prev_in_features
        self.device = device

        self.weight_ei = nn.Parameter(torch.ones(self.n_e, self.n_i, device=self.device) / self.n_i, requires_grad=True)
        self.alpha = nn.Parameter(torch.empty(self.n_i, 1, 1, device=self.device), requires_grad=True)
        self.gain = nn.Parameter(torch.ones(self.n_e, 1, 1, device=self.device), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.n_e, 1, 1, device=self.device), requires_grad=True)

        self.weight_ei.register_hook(lambda grad: grad / self.prev_in_features)

        self.visualize_cache = {}
        self._need_visualize = False
        self._inited = False
    
    def _dynamic_init(self, batch_stats: dict[str, Any]) -> None:
        E_x = batch_stats['E_x']
        Var_x = batch_stats['Var_x']
        E_x_square = batch_stats['E_x_square']

        self.alpha.data = torch.ones(self.n_i, 1, 1, device=self.device) / np.sqrt(self.prev_in_features) * np.sqrt(E_x_square + Var_x) / E_x

    def _set_visualize_cache(self, *args: Any) -> None:
        I_ei, I_balanced, I_shunting, I_int = args
        self.visualize_cache['param1:weight_ei'] = self.weight_ei.detach()
        self.visualize_cache['param2:alpha'] = self.alpha.detach()
        self.visualize_cache['param3:gain'] = self.gain.detach()
        self.visualize_cache['param4:bias'] = self.bias.detach()
        self.visualize_cache['data1:I_ei'] = I_ei.detach()
        self.visualize_cache['data2:I_balanced'] = I_balanced.detach()
        self.visualize_cache['data3:I_shunting'] = I_shunting.detach()
        self.visualize_cache['data4:I_int'] = I_int.detach()

    def _replace_zero_with_second_min(self, input: torch.Tensor):
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

    def _clamp_parameters(self) -> None:
        with torch.no_grad():
            self.weight_ei.data.clamp_(min=0)
            self.alpha.data.clamp_(min=0)
            self.gain.data.clamp_(min=0)

    def forward(self, input: tuple[torch.Tensor, torch.Tensor, Union[dict, None]]):
        self._clamp_parameters()
        I_ee, I_ie, batch_stats = input
        if not self._inited and batch_stats is not None:
            self._dynamic_init(batch_stats)
            self._inited = True

        I_ei = torch.matmul(self.weight_ei, I_ie.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)

        I_balanced = I_ee - I_ei

        I_shunting = torch.matmul(self.weight_ei, (self.alpha * I_ie).permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        I_shunting_adjusted = self._replace_zero_with_second_min(I_shunting)

        I_int = self.gain * I_balanced / I_shunting_adjusted + self.bias
        if self._need_visualize:
            self._set_visualize_cache(I_ei, I_balanced, I_shunting_adjusted, I_int)
        
        return I_int
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag