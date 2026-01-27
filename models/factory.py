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
        if neuron_config is None:
            raise ValueError("Neuron is not specified.")
        if arch == 'MLP':
            if seq_input:
                raise ValueError("MLP does not support sequential input currently.")
            return SpikingMLP(T, num_layers, n_inputs, num_classes, neuron_config, BN=True)
        elif arch == 'VGG':
            dropout = config['dropout']
            light_classifier = config['light_classifier']
            return SpikingVGG(T, num_layers, in_channels, num_classes, neuron_config, 
                              light_classifier, dropout, BN=True, seq_input)
        elif arch == 'ResNet':
            pass
        else:
            return None
    elif model_type == 'EI-SNN':
        if neuron_config is None:
            raise ValueError("Neuron is not specified.")
        if ei_ratio is None:
            raise ValueError("`ei_ratio` is not specified.")
        if arch == 'MLP':
            return SpikingEiMLP(T, num_layers, n_inputs, num_classes, 
                                neuron_config, ei_ratio, device, global_rng)
        elif arch == 'VGG':
            # return SpikingVGG(num_layers, num_classes, in_channels, T, 
            #             neuron_config, dropout, conv_config, BN, light_classifier, 
            #             has_temporal_dim)
            pass
        elif arch == 'ResNet':
            pass
        else:
            return None

def build_mlp(config: dict, device: torch.device, rng: np.random.Generator):
    model_type = config['type']
    T = config['T']
    num_layers = config['num_layers']
    dataset = config['dataset']
    n_inputs = n_inputs_dict[dataset]
    num_classes = num_classes_dict[dataset]

    ei_ratio = config.get('ei_ratio', None)
    neuron_config = config.get('neuron', None)
    
    if model_type == 'BN-SNN':
        if neuron_config is None:
            raise ValueError("Neuron is not specified.")
        if ei_ratio is not None:
            raise ValueError("`ei_ratio` should not be specified for BN-SNN.")
        return SpikingMLP(T, num_layers, n_inputs, num_classes, neuron_config, 
                          BN=True)
    
    elif model_type == 'EI-SNN':
        if neuron_config is None:
            raise ValueError("Neuron is not specified.")
        if ei_ratio is None:
            raise ValueError("`ei_ratio` is not specified.")
        return SpikingEiMLP(T, num_layers, n_inputs, num_classes, neuron_config, 
                            ei_ratio, device, rng)
    else:
        return None


def build_vgg(config: dict, device: torch.device, rng: np.random.Generator):
    model_type = config['model_type']
    num_layers = config['num_layers']
    dataset = config['dataset']
    light_classifier = config['light_classifier']
    dropout = config.get('dropout', 0.0)

    in_channels = in_channels_dict[dataset]
    num_classes = num_classes_dict[dataset]
    has_temporal_dim = True if 'DVS' in dataset else False
    conv_config = {
        'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1,
        'groups': 1, 'padding_mode': 'zeros', 'bias': False
    }

    ei_ratio = config.get('ei_ratio', None)
    T = config.get('T', None)
    neuron_config = config.get('neuron_config', None)

    if model_type == 'SNN':
        if T is None or neuron_config is None:
            raise ValueError("`T` and `neuron_config` must be specified for SNN " \
            "models.")
        return SpikingVGG(num_layers, num_classes, in_channels, T, 
                        neuron_config, dropout, conv_config, BN, light_classifier, 
                        has_temporal_dim)
    
    elif model_type == 'EISNN':
        if T is None or neuron_config is None or ei_ratio is None:
            raise ValueError("`T`, `neuron_config`, and `ei_ratio` must be " \
            "specified for EISNN models.")
        return SpikingEiVGG(num_layers, num_classes, in_channels, ei_ratio, T, 
                            neuron_config, dropout, conv_config, light_classifier, has_temporal_dim,
                            device, rng)
    else:
        return None


def build_resnet(parameters: dict, device: torch.device, rng: np.random.Generator):
    model_type = parameters['model_type']
    num_layers = parameters['num_layers']
    dataset = parameters['dataset']

    in_channels = in_channels_dict[dataset]
    num_classes = num_classes_dict[dataset]
    has_temporal_dim = True if 'DVS' in dataset else False

    ei_ratio = parameters.get('ei_ratio', None)
    T = parameters.get('T', None)
    neuron_config = parameters.get('neuron_config', None)
    zero_init_residual = parameters.get('zero_init_residual', False)

    if model_type == 'SNN':
        if T is None or neuron_config is None:
            raise ValueError("`T` and `neuron_config` must be specified for SNN " \
            "models.")
        return SpikingResNet(num_layers, neuron_config, T, in_channels, 
                                     num_classes, zero_init_residual, has_temporal_dim)
    elif model_type == 'EISNN':
        if T is None or neuron_config is None or ei_ratio is None:
            raise ValueError("`T`, `neuron_config`, and `ei_ratio` must be " \
            "specified for DSNN models.")
        return SpikingEiResNet(ei_ratio, num_layers, neuron_config, T, in_channels, 
                               num_classes, has_temporal_dim, rng, device)
    else:
        return None