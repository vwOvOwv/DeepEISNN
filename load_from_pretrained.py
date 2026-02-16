"""Load pretrained model from huggingface and evaluate"""

import argparse
import json

import numpy as np
if not hasattr(np, 'int'):
    np.int = np.int32  # Fix AttributeError: module 'numpy' has no attribute 'int'
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from spikingjelly.activation_based.functional import reset_net

from datasets import dvs, normal
from models.factory import build_model
from train import validate

repo_id = "vwOvOwv/DeepEISNN"

def get_model_and_config():
    parser = argparse.ArgumentParser(description="Load Pretrained DeepEISNN Model")
    parser.add_argument('--model', type=str, default="CIFAR10-VGG8",
                        help="Pretrained model name, in the format [Dataset]-[Arch]")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Local path to the dataset (override default config)")
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        parser.print_help()
        raise SystemExit(0) from exc
    
    print(f"Loading {args.model} from vwOvOwv/DeepEISNN")

    config_path = hf_hub_download(
        repo_id=repo_id, 
        filename="config.json",
        subfolder=args.model
    )
    weight_path = hf_hub_download(
        repo_id=repo_id, 
        filename="model.safetensors", 
        subfolder=args.model
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    config['data_path'] = args.data_path

    model = build_model(config, torch.device("cpu"), None)
    load_model(model, weight_path, strict=True)

    for module in model.modules():
        if hasattr(module, '_inited'):
            module._inited = True

    return model, config

def main():
    model, config = get_model_and_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    dataset_func = {'MNIST': normal.get_mnist,
                    'CIFAR10': normal.get_cifar10,
                    'CIFAR100': normal.get_cifar100,
                    'CIFAR10DVS': dvs.get_cifar10dvs,
                    'DVSGesture': dvs.get_dvs128gesture,
                    'TinyImageNet200': normal.get_tinyimagenet}
    if 'DVS' in config['dataset']:
        _, val_set = dataset_func[config['dataset']](config['data_path'],
                                                             config['T'])
    else:
        _, val_set = dataset_func[config['dataset']](config['data_path'])

    val_loader = DataLoader(val_set,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config.get('num_workers', 8),
                            pin_memory=True)
    
    dummy_input, _ = next(iter(val_loader))
    summary(model, input_data=dummy_input.to(device), device=device.type)
    print(f"Model type: {config['type']}")
    print(f"Architechture: {config['arch']}{config['num_layers']}")
    reset_net(model)
    
    criterion = nn.CrossEntropyLoss()
    validate(val_loader, model, criterion, device)

if __name__ == '__main__':
    main()