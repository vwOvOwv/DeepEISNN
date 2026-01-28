"""Training script."""


import argparse
import os
import random
import shutil
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary
import tqdm
import yaml
import wandb
from spikingjelly.activation_based.functional import reset_net

from datasets import dvs, normal
from models.factory import build_model
from modules.optimizers import EiSGD
from utils.evaluation import AverageMeter, evaluate
from utils.visualization import Visualizer

def get_args_and_config():
    """Parse CLI arguments and load configuration.

    Returns:
        Tuple of (args, config, output_dir). `output_dir` is None when logging
        is disabled.

    Raises:
        ValueError: If scratch/resume arguments are invalid.
        FileNotFoundError: If a referenced config or checkpoint file is missing.
        KeyError: If a resume checkpoint is missing configuration.
    """
    parser = argparse.ArgumentParser(description="DeepEISNN Training")
    parser.add_argument('--scratch', type=str, default=None,
                        help="Train from scratch, need path to YAML config file")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from checkpoint, need path to checkpoint")
    parser.add_argument('--log', type=int, default=1, choices=[0, 1, 2],
                        help='Log mode: 0=off, 1=loss/acc/ckpt, 2=loss/acc/ckpt/vis')
    parser.add_argument('--notes', type=str, default=None,
                        help="Experiment notes for logging")
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed')

    try:
        args = parser.parse_args()
    except SystemExit as exc:
        parser.print_help()
        raise SystemExit(0) from exc

    # verify either --scratch or --resume is provided
    if args.scratch is None and args.resume is None:
        raise ValueError("Either --scratch or --resume must be specified.")
    if args.scratch is not None and args.resume is not None:
        raise ValueError("Only one of --scratch or --resume should be specified.")

    # load raw config
    if args.scratch is not None:
        if not os.path.isfile(args.scratch):
            raise FileNotFoundError(f"Config file {args.scratch} not found.")
        with open(args.scratch, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Training from scratch with config file: {args.scratch}")
    else:   # args.resume is not None
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint file {args.resume} not found.")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'config' not in checkpoint:
            raise KeyError(f"Config file not found in checkpoint {args.resume}.")
        config = checkpoint['config']
        print(f"Resuming training from checkpoint: {args.resume}")

    # add time stamp to output dir if logging is needed
    if args.log > 0:
        exp_id = (
            f"{time.strftime('%Y%m%d-%H%M%S')}"
            f"-{config['type']}"
            f"-{config['dataset']}-{config['arch']}{config['num_layers']}"
            f"-T{config['T']}-ep{config['epochs']}-bs{config['batch_size']}"
            f"-lr{config['learning_rate']}"
            f"-{config['optimizer']}-{config['scheduler']}"
            f"-mo{config['momentum']}-wd{config['weight_decay']}"
            f"-seed{args.seed}"
        )
        if args.notes is not None:
            notes = args.notes.strip().replace(' ', '-').replace('_', '-')
            exp_id += f"-{notes}"
        output_dir = os.path.join(config['output_dir'], exp_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # backup config file
        with open(os.path.join(output_dir, 'config-backup.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        return args, config, output_dir
    return args, config, None

def set_random_state(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed value.

    Returns:
        NumPy random generator seeded with the provided value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Global random generator for all modules.
    return np.random.default_rng(seed=seed)

def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict[str, Any],
    iter_per_epoch: int,
) -> tuple[list[Optimizer], list[LRScheduler], list[LRScheduler]]:
    """Construct optimizers and learning-rate schedulers.

    Args:
        model: Model to optimize.
        config: Training configuration dictionary.
        iter_per_epoch: Number of iterations per epoch.

    Returns:
        Tuple of (optimizer_list, warmup_scheduler_list, scheduler_list).

    Raises:
        NotImplementedError: If optimizer or scheduler is unsupported.
    """
    # optimizer
    optimizer_list = []
    if config['optimizer'] == 'SGD':
        if config['type'] == 'EI-SNN':
            non_negative_params = []
            standard_params = []

            # bias terms (of excitatory neurons) are not constrained
            for name, param in model.named_parameters():
                if 'bias' in name:
                    standard_params.append(param)
                else:
                    non_negative_params.append(param)

            optimizer_non_negative = EiSGD(
                non_negative_params,
                config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
                clamped=True,
            )
            optimizer_standard = EiSGD(
                standard_params,
                config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
                clamped=False,
            )
            optimizer_list.append(optimizer_non_negative)
            optimizer_list.append(optimizer_standard)
        else:   # BN-SNN, no need to split params
            optimizer = EiSGD(
                model.parameters(),
                config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
                clamped=False,
            )
            optimizer_list.append(optimizer)
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} is not implemented.")

    # scheduler
    total_steps = config['epochs'] * iter_per_epoch
    warmup_steps = config.get('warmup_epochs', 0) * iter_per_epoch
    warmup_scheduler_list = []
    scheduler_list = []
    if config.get('warmup_epochs', 0) > 0:
        init_lr = config.get('init_learning_rate', 1e-7)
        start_factor = init_lr / config['learning_rate']
        for optimizer in optimizer_list:
            warmup_scheduler_list.append(
                LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    total_iters=warmup_steps,
                )
            )
    scheduler = config['scheduler']
    if scheduler == 'CosALR':
        for optimizer in optimizer_list:
            scheduler_list.append(
                CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps - warmup_steps,
                )
            )
    else:
        raise NotImplementedError(f"Scheduler {scheduler} is not implemented.")

    return optimizer_list, warmup_scheduler_list, scheduler_list

def train_one_epoch(
    model: nn.Module,
    epoch: int,
    config: dict[str, Any],
    train_loader: DataLoader,
    optimizer_list: Sequence[Optimizer],
    scheduler_list: Sequence[LRScheduler],
    warmup_scheduler_list: Sequence[LRScheduler],
    visualizer: Optional[Visualizer],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train the model for a single epoch.

    Args:
        model: Model to train.
        epoch: Current epoch index (1-based).
        config: Training configuration dictionary.
        train_loader: Training data loader.
        optimizer_list: List of optimizers.
        scheduler_list: List of LR schedulers (post-warmup).
        warmup_scheduler_list: List of warmup schedulers.
        visualizer: Optional visualizer instance.
        criterion: Loss function.
        device: Device for training.

    Returns:
        Tuple of (loss_avg, top1_avg, top5_avg).
    """
    model.train()
    loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    tqdm_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")

    iter_per_epoch = len(train_loader)
    warmup_steps = config.get('warmup_epochs', 0) * iter_per_epoch

    if visualizer is not None:  # only visualize first batch of each epoch
        model.set_visualize(True)

    for idx, (inputs, labels) in enumerate(tqdm_bar):
        global_steps = (epoch - 1) * iter_per_epoch + idx
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)

        for optimizer in optimizer_list:
            optimizer.zero_grad()

        batch_loss.backward()

        if visualizer is not None and model.get_visualize():
            visualizer.visualize_model_states(model, epoch, global_steps + 1)
            model.set_visualize(False)

        for optimizer in optimizer_list:
            optimizer.step()

        reset_net(model)

        loss.update(batch_loss.item(), labels.numel())
        acc1, acc5 = evaluate(outputs.detach(), labels, topk=(1, 5))
        top1.update(acc1.item(), labels.numel())
        top5.update(acc5.item(), labels.numel())

        if global_steps < warmup_steps and config.get('warmup_epochs', 0) > 0:
            for scheduler in warmup_scheduler_list:
                scheduler.step()
        else:
            for scheduler in scheduler_list:
                scheduler.step()

        current_lr = optimizer_list[0].param_groups[0]['lr']
        tqdm_bar.set_postfix(lr=current_lr, loss=loss.avg, acc1=top1.avg, acc5=top5.avg)

    return loss.avg, top1.avg, top5.avg

def validate(
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the model on the validation set.

    Args:
        val_loader: Validation data loader.
        model: Model to evaluate.
        criterion: Loss function.
        device: Device for evaluation.

    Returns:
        Tuple of (loss_avg, top1_avg, top5_avg).
    """
    model.eval()

    with torch.no_grad():
        loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for inputs, labels in tqdm.tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            reset_net(model)

            batch_loss = criterion(outputs, labels)

            loss.update(batch_loss.item(), labels.numel())
            acc1, acc5 = evaluate(outputs.detach(), labels, topk=(1, 5))
            top1.update(acc1.item(), labels.numel())
            top5.update(acc5.item(), labels.numel())

        print(f"Val Loss {loss.avg:.3f}")
        print(f"Val Acc@1 {top1.avg:.3f}")
        print(f"Val Acc@5 {top5.avg:.3f}")

    return loss.avg, top1.avg, top5.avg


if __name__ == '__main__':
    # parse args and config
    args, config, output_dir = get_args_and_config()

    # set random seeds
    global_rng = set_random_state(args.seed)

    # set wandb logger and local visualizer
    logger = wandb.init(project='DeepEISNN',
                        name=os.path.basename(output_dir),
                        config=config,
                        dir=output_dir) if args.log > 0 else None

    visualizer = None
    if args.log > 1:
        if config['arch'] in ['VGG']:
            visualizer = Visualizer(output_dir)
        else:
            print(f"Visualization for {config['arch']} is not supported currently.")

    # build data loader
    dataset_func = {'MNIST': normal.get_mnist,
                    'CIFAR10': normal.get_cifar10,
                    'CIFAR100': normal.get_cifar100,
                    'CIFAR10DVS': dvs.get_cifar10dvs,
                    'DVSGesture': dvs.get_dvs128gesture,
                    'TinyImageNet200': normal.get_tinyimagenet}
    if 'DVS' in config['dataset']:
        train_set, val_set = dataset_func[config['dataset']](config['data_path'],
                                                             config['T'])
    else:
        train_set, val_set = dataset_func[config['dataset']](config['data_path'])

    train_loader = DataLoader(train_set,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config.get('num_workers', 8),
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config.get('num_workers', 8),
                            pin_memory=True)

    # check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # build model
    model = build_model(config, device, global_rng)
    model.to(device)

    # summarize model
    dummy_input, _ = next(iter(train_loader))
    summary(model, input_data=dummy_input.to(device), device=device.type)
    print(f"Model type: {config['type']}")
    print(f"Architechture: {config['arch']}{config['num_layers']}")

    reset_net(model)

    # set optimizer and scheduler
    iter_per_epoch = len(train_loader)
    optimizer_list, warmup_scheduler_list, scheduler_list = \
        build_optimizer_and_scheduler(model, config, iter_per_epoch)

    # set objective function
    criterion = config['loss']
    if criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        raise NotImplementedError(f"Loss {criterion} is not implemented.")

    # resume from checkpoint
    start_epoch = 1
    best_val_acc1 = 0.
    if args.resume is not None:
        print(f"===> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = checkpoint['next_epoch']
        best_val_acc1 = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        for idx, optimizer in enumerate(optimizer_list):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'][idx])
        for idx, scheduler in enumerate(warmup_scheduler_list):
            scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'][idx])
        for idx, scheduler in enumerate(scheduler_list):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'][idx])
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        cuda_states = [t.cpu() for t in checkpoint['cuda_rng_state']]
        torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['py_rng_state'])
        print(f"===> loaded checkpoint (epoch {checkpoint['next_epoch']})")

    # train
    print(f"\nStart training from epoch {start_epoch}")
    if args.log > 0:
        ckpt_save_path = os.path.join(output_dir, 'ckpts')
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path, exist_ok=True)
    for epoch in range(start_epoch, config['epochs'] + 1):
        loss, acc1, acc5 = train_one_epoch(
            model,
            epoch,
            config,
            train_loader,
            optimizer_list,
            scheduler_list,
            warmup_scheduler_list,
            visualizer,
            criterion,
            device,
        )

        # evaluate the model
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, device)
        is_best = val_acc1 > best_val_acc1
        best_val_acc1 = max(best_val_acc1, val_acc1)

        # logging and save checkpoints
        if args.log > 0:
            log_dict = {
                'train/loss': loss,
                'train/acc1': acc1,
                'train/acc5': acc5,
                'val/loss': val_loss,
                'val/acc1': val_acc1,
                'val/acc5': val_acc5,
                'train/lr': optimizer_list[0].param_groups[0]['lr'],
                'epoch': epoch
            }
            logger.log(log_dict, step=epoch)

            latest_path = os.path.join(output_dir, 'ckpts', 'ckpt-latest.pth')
            torch.save(
                {
                    'next_epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': [
                        optimizer.state_dict() for optimizer in optimizer_list
                    ],
                    'warmup_scheduler_state_dict': [
                        scheduler.state_dict() for scheduler in warmup_scheduler_list
                    ],
                    'scheduler_state_dict': [
                        scheduler.state_dict() for scheduler in scheduler_list
                    ],
                    'best_val_acc': best_val_acc1,
                    'config': config,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'np_rng_state': np.random.get_state(),
                    'py_rng_state': random.getstate(),
                },
                latest_path,
            )
            if is_best:
                shutil.copyfile(
                    latest_path,
                    os.path.join(output_dir, 'ckpts', 'ckpt-best.pth'),
                )

    print("Train finished")
    print(f"Best validation accuracy: {best_val_acc1:.3f}")
    print(f"Log directory: {output_dir if args.log > 0 else 'None'}")
    print(f"Args: {args}")

    if visualizer is not None:
        visualizer.png2mp4()
