import torch
import torch.nn as nn
import numpy as np
from modules.conv2d import SpikingConv2d, SpikingEiConv2d
from modules.linear import SpikingLinear, SpikingEiLinear
from modules.norm1d import SpikingBatchNorm1d, SpikingEiNorm1d
from modules.norm2d import SpikingBatchNorm2d, SpikingEiNorm2d
from modules.activation import LIF
from utils.dim import AddTemporalDim, MergeTemporalDim, SplitTemporalDim
from typing import Any

__all__ = [
    'SpikingVGG', 'SpikingEiVGG'
]

layer_config = {
    8:  [64, 'P', 128, 'P', 256, 'P', 512, 'P', 512, 'P'],
    11: [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    16: [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    19: [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
}

conv_config = {
    'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1,
    'groups': 1, 'padding_mode': 'zeros', 'bias': False
}

class SpikingVGG(nn.Module):
    def __init__(self, T: int, num_layers: int, in_channels: int, n_outputs: int,
                neuron_config: dict[str, Any], light_classifier: bool,
                dropout: float, seq_input: bool, BN: bool):
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.neuron_config = neuron_config
        self.BN = BN
        self.light_classifier = light_classifier
        self.dropout = dropout
        self.seq_input = seq_input
        self.conv_config = conv_config

        self.layers = self._build_model(layer_config[self.num_layers])
        self._need_visualize = False
        
    def _build_model(self, cfg):
        layers = []
        if not self.seq_input:
            layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        layers += self._build_extractor(cfg)
        layers += self._build_classifier(cfg[-2])
        return nn.Sequential(*layers)

    def _build_extractor(self, cfg):
        layers = []
        in_channels = self.in_channels
        for i in range(len(cfg)):
            if cfg[i] == 'P':
                layers.append(nn.AvgPool2d(2, 2, 0))
            else:
                layers.append(SpikingConv2d(in_channels, cfg[i], **self.conv_config))
                if self.BN:
                    layers.append(SpikingBatchNorm2d(cfg[i]))
                layers.append(SplitTemporalDim(self.T))
                layers.append(LIF(**self.neuron_config))
                layers.append(MergeTemporalDim(self.T))
                in_channels = cfg[i]

        layers.append(nn.Flatten())
        return layers
    
    def _build_classifier(self, in_channels):
        """
        Construct feature classifier.
        """
        layers = []
        if self.light_classifier:
            layers.append(SpikingLinear(in_channels * 1 * 1, self.n_outputs))
            layers.append(SplitTemporalDim(self.T))
            return layers
        
        layers.append(SpikingLinear(in_channels * 1 * 1, 4096))
        if self.BN:
            layers.append(SpikingBatchNorm1d(4096))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))

        layers.append(SpikingLinear(4096, 4096))
        if self.BN:
            layers.append(SpikingBatchNorm1d(4096))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))
        layers.append(SpikingLinear(4096, self.n_outputs))
        layers.append(SplitTemporalDim(self.T))
        return layers
    
    def forward(self, input: torch.Tensor):
        if self._need_visualize:
            self._set_layer_visualize(True)
        if self.seq_input:
            input = input.transpose(0, 1)
        output = self.layers(input)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(dim=0)
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag

    def get_visualize(self) -> bool:
        return self._need_visualize

    def _set_layer_visualize(self, flag: bool) -> None:
        for layer in self.layers:
            if hasattr(layer, 'set_visualize'):
                layer.set_visualize(flag)

class SpikingEiVGG(nn.Module):
    def __init__(self, T: int, num_layers: int, in_channels: int, n_outputs: int,
                 neuron_config: dict[str, Any], light_classifier: bool,
                 dropout: float, seq_input: bool, ei_ratio: int,
                 device: torch.device, rng: np.random.Generator):
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.neuron_config = neuron_config
        self.light_classifier = light_classifier
        self.dropout = dropout
        self.seq_input = seq_input
        self.ei_ratio = ei_ratio
        self.conv_config = conv_config

        self.device = device
        self.rng = rng

        self.layers = self._build_model(layer_config[self.num_layers])
        self._need_visualize = False

    def _build_model(self, cfg):
        layers = []
        if not self.seq_input:
            layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        layers += self._build_extractor(cfg)
        layers += self._build_classifier(cfg[-2])
        return nn.Sequential(*layers)
    
    def _build_extractor(self, cfg):
        layers = []
        in_channels = self.in_channels
        for i in range(len(cfg)):
            if cfg[i] == 'P':
                layers.append(nn.AvgPool2d(2, 2, 0))
            else:
                layers.append(SpikingEiConv2d(in_channels, cfg[i], self.ei_ratio, 
                                              self.device, self.rng, **self.conv_config))
                layers.append(SpikingEiNorm2d(cfg[i], in_channels * self.conv_config['kernel_size']**2, 
                                              self.ei_ratio, device=self.device))
                layers.append(SplitTemporalDim(self.T))
                layers.append(LIF(**self.neuron_config))
                layers.append(nn.Dropout(self.dropout))
                layers.append(MergeTemporalDim(self.T))
                in_channels = cfg[i]

        layers.append(nn.Flatten())
        return layers
    
    def _build_classifier(self, in_channels):
        layers = []
        if self.light_classifier:
            layers.append(SpikingEiLinear(in_channels * 1 * 1, self.n_outputs, self.ei_ratio, self.device, self.rng))
            layers.append(SpikingEiNorm1d(self.n_outputs, in_channels * 1 * 1, self.ei_ratio, self.device, output_layer=True))
            layers.append(SplitTemporalDim(self.T))
            return layers
        
        layers.append(SpikingEiLinear(in_channels * 1 * 1, 4096, self.ei_ratio, self.device, self.rng))
        layers.append(SpikingEiNorm1d(4096, in_channels * 4 * 4, self.ei_ratio, self.device))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))

        layers.append(SpikingEiLinear(4096, 4096, self.ei_ratio, self.device, self.rng))
        layers.append(SpikingEiNorm1d(4096, 4096, self.ei_ratio, self.device))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))
        layers.append(SpikingEiLinear(4096, self.n_outputs, self.ei_ratio, self.device, self.rng))
        layers.append(SpikingEiNorm1d(self.n_outputs, 4096, self.ei_ratio, self.device, output_layer=True))
        layers.append(SplitTemporalDim(self.T))
        return layers
    
    def forward(self, input: torch.Tensor):
        if self._need_visualize:
            self._set_layer_visualize(True)
        if self.seq_input:
            input = input.transpose(0, 1)
        output = self.layers(input)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(dim=0)
    
    def set_visualize(self, flag: bool) -> None:
        self._need_visualize = flag

    def get_visualize(self) -> bool:
        return self._need_visualize

    def _set_layer_visualize(self, flag: bool) -> None:
        for layer in self.layers:
            if hasattr(layer, 'set_visualize'):
                layer.set_visualize(flag)