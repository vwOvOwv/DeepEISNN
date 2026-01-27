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

cfg = {
    8:  [64, 'P', 128, 'P', 256, 'P', 512, 'P', 512, 'P'],
    11: [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    16: [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    19: [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
}


class SpikingVGG(nn.Module):
    def __init__(self, num_layers: int, num_classes: int, in_channels: int,
                 T: int, neuron_config: dict[str, Any], dropout: float, 
                 conv_config: dict[str, Any], BN: bool, light_classifier: bool, 
                 has_temporal_dim: bool) -> None:
        """
        Args:
            num_layers (int): Number of layers of the model.
            num_classes (int): Number of classes to classify.
            in_channels (int): Number of input channels.
            T (int): Number of time steps.
            neuron_config (dict[str, Any]): Configuration for spiking neurons.
            dropout (float): Dropout rate.
            conv_config (dict[str, Any]): Configuration for convolution layers.
            light_classifier (bool): Whether to use a light classifier.
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.T = T
        self.neuron_config = neuron_config
        self.conv_config = conv_config
        self.light_classifier = light_classifier
        self.has_temporal_dim = has_temporal_dim
        self.dropout = dropout
        self.BN = BN
        self.layers = self._build_model(cfg[self.num_layers])

        self._need_visualize = False
        
    def _build_model(self, cfg):
        layers = []
        if not self.has_temporal_dim:
            layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        layers += self._make_extractor(cfg)
        layers += self._make_classifier(cfg[-2])
        return nn.Sequential(*layers)

    def _make_extractor(self, cfg):
        """
        Construct feature extractor.
        """
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
    
    def _make_classifier(self, in_channels):
        """
        Construct feature classifier.
        """
        layers = []
        if self.light_classifier:
            layers.append(SpikingLinear(in_channels * 1 * 1, self.num_classes))
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
        layers.append(SpikingLinear(4096, self.num_classes))
        layers.append(SplitTemporalDim(self.T))
        return layers
    
    def forward(self, input: torch.Tensor):
        if self._need_visualize:
            self._set_layer_visualize(True)
        if self.has_temporal_dim:
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
    def __init__(self, num_layers: int, num_classes: int, in_channels: int,
                 ei_ratio: int, T: int, neuron_config: dict[str, Any],
                 dropout: float, conv_config: dict[str, Any], light_classifier: bool, 
                 has_temporal_dim: bool,
                 device: torch.device, rng: np.random.Generator) -> None:
        """
        Spiking VGG with E-I mechanisms.
            -   Excitatory neuron: LIF
            -   Inhibitory neuron: None (i.e., ReLU)
        Args:
            num_layers (int): Number of layers of the model.
            num_classes (int): Number of classes to classify.
            in_channels (int): Number of input channels.
            ei_ratio (int): Ratio of excitatory neurons to inhibitory neurons.
            T (int): Number of time steps for spiking behavior.
            neuron_config (dict[str, Any]): Configuration for excitatory neurons.
            dropout (float): Dropout rate.
            conv_config (dict[str, Any]): Configuration for convolutional layers.
            light_classifier (bool): Whether to use a light classifier.
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.ei_ratio = ei_ratio
        self.T = T
        self.neuron_config = neuron_config
        self.conv_config = conv_config
        self.dropout = dropout
        self.light_classifier = light_classifier
        self.has_temporal_dim = has_temporal_dim
        self.device = device
        self.rng = rng
        self.layers = self._build_model(cfg[self.num_layers])

        self._need_visualize = False


    def _build_model(self, cfg):
        layers = []
        if not self.has_temporal_dim:
            layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        layers += self._make_extractor(cfg)
        layers += self._make_classifier(cfg[-2])
        return nn.Sequential(*layers)
    
    def _make_extractor(self, cfg):
        layers = []
        in_channels = self.in_channels
        for i in range(len(cfg)):
            if cfg[i] == 'P':
                layers.append(nn.AvgPool2d(2, 2, 0))
            else:
                layers.append(SpikingEiConv2d(self.ei_ratio, in_channels, 
                                              cfg[i], self.rng, **self.conv_config, device=self.device))
                layers.append(SpikingEiNorm2d(self.ei_ratio, cfg[i], 
                                              in_channels * self.conv_config['kernel_size']**2, 
                                              device=self.device))
                layers.append(SplitTemporalDim(self.T))
                layers.append(LIF(**self.neuron_config))
                layers.append(nn.Dropout(self.dropout))
                layers.append(MergeTemporalDim(self.T))
                in_channels = cfg[i]

        layers.append(nn.Flatten())
        return layers
    
    def _make_classifier(self, in_channels):
        layers = []
        if self.light_classifier:
            layers.append(SpikingEiLinear(self.ei_ratio, in_channels * 1 * 1, 
                                         self.num_classes, self.device, self.rng))
            layers.append(SpikingEiNorm1d(self.ei_ratio, self.num_classes,
                                            in_channels * 1 * 1, self.device, output_layer=True))
            layers.append(SplitTemporalDim(self.T))
            return layers
        
        layers.append(SpikingEiLinear(self.ei_ratio, in_channels * 1 * 1, 
                                      4096, self.device, self.rng))
        layers.append(SpikingEiNorm1d(self.ei_ratio, 4096, 
                                      in_channels * 4 * 4, self.device))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))

        layers.append(SpikingEiLinear(self.ei_ratio, 4096, 
                                    4096, self.device, self.rng))
        layers.append(SpikingEiNorm1d(self.ei_ratio, 4096, 4096, 
                                          self.device))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))
        layers.append(SpikingEiLinear(self.ei_ratio, 4096, 
                                      self.num_classes, self.device, self.rng))
        layers.append(SpikingEiNorm1d(self.ei_ratio, self.num_classes, 
                                      4096, self.device, output_layer=True))
        layers.append(SplitTemporalDim(self.T))
        return layers
    
    def forward(self, input: torch.Tensor):
        if self._need_visualize:
            self._set_layer_visualize(True)
        if self.has_temporal_dim:
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