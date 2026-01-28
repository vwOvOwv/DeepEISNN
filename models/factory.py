import torch
import numpy as np
from typing import Optional, Union, Literal, Any
from .MLP import SpikingMLP, SpikingEiMLP
from .VGG import SpikingVGG, SpikingEiVGG
from .ResNet import SpikingResNet, SpikingEiResNet

in_channels_dict = {
    'MNIST': 1,
    'CIFAR10': 3,
    'CIFAR100': 3,
    'CIFAR10DVS': 2,
    'DVSGesture': 2,
    'TinyImageNet200': 3
}

n_inputs_dict = {
    'MNIST': 28 * 28,
    'CIFAR10': 32 * 32 * 3,
    'CIFAR100': 32 * 32 * 3,
    'TinyImageNet200': 64 * 64 * 3,
}

num_classes_dict = {
    'MNIST': 10,
    'CIFAR10': 10,
    'CIFAR100': 100,
    'CIFAR10DVS': 10,
    'DVSGesture': 11,
    'TinyImageNet200': 200,
}

def build_model(config: dict, device: torch.device, global_rng: np.random.Generator):
    model_type = config['type']
    T = config['T']
    arch = config['arch']
    num_layers = config['num_layers']
    dataset = config['dataset']
    neuron_config = config['neuron']

    in_channels = in_channels_dict[dataset]
    n_inputs = n_inputs_dict[dataset]
    num_classes = num_classes_dict[dataset]
    seq_input = 'DVS' in dataset

    if model_type == 'BN-SNN':
        if arch == 'MLP':
            if seq_input:
                raise ValueError("MLP does not support sequential input currently.")
            return SpikingMLP(T, num_layers, n_inputs, num_classes, neuron_config, BN=True)
        elif arch == 'VGG':
            dropout = config['dropout']
            light_classifier = config['light_classifier']
            return SpikingVGG(T, num_layers, in_channels, num_classes, neuron_config,
                              light_classifier, dropout, seq_input, BN=True)
        elif arch == 'ResNet':
            zero_init_residual = config['zero_init_residual']
            # ResNet has BN by default
            return SpikingResNet(T, num_layers, in_channels, num_classes,
                                 neuron_config, zero_init_residual, seq_input)
        else:
            raise NotImplementedError(f"Model {model_type}:{arch}\
                                      {num_layers} is not implemented.")
    elif model_type == 'EI-SNN':
        ei_ratio = config['ei_ratio']
        if arch == 'MLP':
            return SpikingEiMLP(T, num_layers, n_inputs, num_classes, 
                                neuron_config, ei_ratio, device, global_rng)
        elif arch == 'VGG':
            dropout = config['dropout']
            light_classifier = config['light_classifier']
            return SpikingEiVGG(T, num_layers, in_channels, num_classes, 
                                neuron_config, light_classifier, dropout, seq_input,
                                ei_ratio, device, global_rng)
        elif arch == 'ResNet':
            zero_init_residual = config['zero_init_residual']
            return SpikingEiResNet(T, num_layers, in_channels, num_classes,
                                   neuron_config, seq_input, ei_ratio, device,
                                   global_rng)
        else:
            raise NotImplementedError(f"Model {model_type}:{arch}\
                                      {num_layers} is not implemented.")