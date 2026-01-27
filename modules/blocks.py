import torch
import torch.nn as nn
import numpy as np
from modules.conv2d import SpikingConv2d, SpikingEiConv2d
from modules.norm2d import SpikingBatchNorm2d, SpikingEiNorm2d
from modules.activation import LIF
from utils.dim import MergeTemporalDim, SplitTemporalDim
from typing import Any

class SpikingStandardBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, neuron_config: dict, T, stride=1, downsample=None):
        super(SpikingStandardBasicBlock, self).__init__()
        
        self.conv1 = SpikingConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = SpikingBatchNorm2d(out_channels)
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)
        
        self.conv2 = SpikingConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = SpikingBatchNorm2d(out_channels)
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out
    

class SpikingStandardBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, neuron_config: dict, T, stride=1, downsample=None):
        super(SpikingStandardBottleneck, self).__init__()
        
        self.conv1 = SpikingConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = SpikingBatchNorm2d(out_channels)
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)
        
        self.conv2 = SpikingConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = SpikingBatchNorm2d(out_channels)
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)
        
        self.conv3 = SpikingConv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = SpikingBatchNorm2d(out_channels * self.expansion)
        self.split3 = SplitTemporalDim(T)
        self.lif3 = LIF(**neuron_config)
        self.merge3 = MergeTemporalDim(T)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.split3(out)
        out = self.lif3(out)
        out = self.merge3(out)
        out += identity

        return out
    

class SpikingEiBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, ei_ratio: int, neuron_config: dict, T, device: torch.device, rng: np.random.Generator, stride=1, downsample=None):
        super(SpikingEiBasicBlock, self).__init__()
        
        self.conv1 = SpikingEiConv2d(ei_ratio, in_channels, out_channels, rng, kernel_size=3, stride=stride, padding=1, bias=False, device=device)
        self.norm1 = SpikingEiNorm2d(ei_ratio, out_channels, in_channels * 3 * 3, device=device)
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)
        
        self.conv2 = SpikingEiConv2d(ei_ratio, out_channels, out_channels, rng, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.norm2 = SpikingEiNorm2d(ei_ratio, out_channels, out_channels * 3 * 3, device=device)
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    

class SpikingEiBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, ei_ratio: int, neuron_config: dict, T, device: torch.device, rng: np.random.Generator, stride=1, downsample=None):
        super(SpikingEiBottleneck, self).__init__()
        
        self.conv1 = SpikingEiConv2d(ei_ratio, in_channels, out_channels, rng, kernel_size=1, stride=1, bias=False, device=device)
        self.norm1 = SpikingEiNorm2d(ei_ratio, out_channels, in_channels * 1 * 1, device=device)
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)
        
        self.conv2 = SpikingEiConv2d(ei_ratio, out_channels, out_channels, rng, kernel_size=3, stride=stride, padding=1, bias=False, device=device)
        self.norm2 = SpikingEiNorm2d(ei_ratio, out_channels, out_channels * 3 * 3, device=device)
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)
        
        self.conv3 = SpikingEiConv2d(ei_ratio, out_channels, out_channels * self.expansion, rng, kernel_size=1, stride=1, bias=False, device=device)
        self.norm3 = SpikingEiNorm2d(ei_ratio, out_channels * self.expansion, out_channels * self.expansion * 1 * 1, device=device)
        self.split3 = SplitTemporalDim(T)
        self.lif3 = LIF(**neuron_config)
        self.merge3 = MergeTemporalDim(T)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.split3(out)
        out = self.lif3(out)
        out = self.merge3(out)
        out += identity

        return out
    
