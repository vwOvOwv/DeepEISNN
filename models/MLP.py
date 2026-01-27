import torch.nn as nn
import torch
import numpy as np
from modules.linear import SpikingLinear, SpikingEiLinear
from modules.norm1d import SpikingBatchNorm1d, SpikingEiNorm1d
from modules.activation import LIF
from utils.dim import AddTemporalDim, MergeTemporalDim, SplitTemporalDim
from typing import Any

__all__ = [
    'SpikingMLP', 'SpikingEiMLP'
]

cfg = {
    2: [500],
    4: [500, 500, 300],
    6: [500, 500, 300, 300, 300],
    8: [500, 500, 300, 300, 300, 100, 100],
    12: [500, 500, 500, 500, 300, 300, 300, 300, 100, 100, 100],
    16: [500, 500, 500, 500, 500, 300, 300, 300, 300, 300, 100, 100, 100, 100, 100],
}
 

class SpikingMLP(nn.Module):
    def __init__(self, T: int, num_layers: int, n_inputs: int, n_outputs: int, 
                 neuron_config: dict[str, Any], BN: bool):
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.neuron_config = neuron_config
        self.BN = BN
        self.layers = self._build_model(cfg[self.num_layers])

        self._need_visualize = False

    def _build_model(self, cfg: list[int]):
        layers = []
        in_features = self.n_inputs
        layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        for out_features in cfg:
            layers.append(SpikingLinear(in_features, out_features))
            if self.BN:
                layers.append(SpikingBatchNorm1d(out_features))
            layers.append(SplitTemporalDim(self.T))
            layers.append(LIF(**self.neuron_config))
            layers.append(MergeTemporalDim(self.T))
            in_features = out_features
        layers.append(SpikingLinear(cfg[-1], self.n_outputs))
        layers.append(SplitTemporalDim(self.T))
        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor):
        input = input.view(input.size(0), -1)
        if self._need_visualize:
            self._set_layer_visualize(True)
        output = self.layers(input)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(0)
    
    def set_visualize(self, flag: bool):
        self._need_visualize = flag

    def get_visualize(self):
        return self._need_visualize

    def _set_layer_visualize(self, visualize: bool):
        for layer in self.layers:
            if hasattr(layer, 'set_visualize'):
                layer.set_visualize(visualize)


class SpikingEiMLP(nn.Module):
    def __init__(self, T: int, num_layers: int, n_inputs: int, n_outputs: int,
                 neuron_config: dict[str, Any], ei_ratio: int, 
                 device: torch.device, rng: np.random.Generator) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = n_outputs
        self.n_inputs = n_inputs
        self.ei_ratio = ei_ratio
        self.T = T
        self.neuron_config = neuron_config
        self.device = device
        self.rng = rng
        self.layers = self._build_model(cfg[self.num_layers])

        self._need_visualize = False

    def _build_model(self, cfg: list[int]) -> nn.Sequential:
        layers = []
        in_features = self.n_inputs
        layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        for out_features in cfg:
            layers.append(SpikingEiLinear(self.ei_ratio, in_features, 
                                          out_features, self.device, self.rng))
            layers.append(SpikingEiNorm1d(self.ei_ratio, out_features, 
                                          in_features, self.device))
            layers.append(SplitTemporalDim(self.T))
            layers.append(LIF(**self.neuron_config))
            layers.append(MergeTemporalDim(self.T))
            in_features = out_features
        layers.append(SpikingEiLinear(self.ei_ratio, cfg[-1], 
                                      self.num_classes, self.device, self.rng))
        layers.append(SpikingEiNorm1d(self.ei_ratio, self.num_classes, 
                                      cfg[-1], self.device, True))
        layers.append(SplitTemporalDim(self.T))
        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.view(input.size(0), -1)
        if self._need_visualize:
            self._set_layer_visualize(True)
        output = self.layers(input)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(0)
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag

    def get_visualize(self) -> bool:
        return self._need_visualize

    def _set_layer_visualize(self, flag: bool) -> None:
        for layer in self.layers:
            if hasattr(layer, 'set_visualize'):
                layer.set_visualize(flag)