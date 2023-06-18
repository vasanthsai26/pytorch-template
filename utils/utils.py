import torch
import random
import os
import warnings
import matplotlib
import numpy as np
import constants
import torch.optim as optim
import torch.nn as nn

from datetime import datetime


warnings.filterwarnings('ignore')
matplotlib.use('Agg')

def get_device() -> str:
    """
    Returns the default device available
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(TORCH_SEED: int) -> None:
    """
    Sets the manual SEED  
    """
    random.seed(TORCH_SEED)  
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denormalize(images, 
            means=constants.IMAGENET_MEAN, 
            stds=constants.IMAGENET_STD,
            device=get_device()):
    """
    denormalize the image with imagenet stats
    """
    means = torch.tensor(means).reshape(1, 3, 1, 1).to(device)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1).to(device)
    return (images * stds + means)

def generate_experiment_name(args):
    """
    Generate an experiment name based on the provided arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The generated experiment name.
    """
    project_name = args.project_name
    model = args.model
    optimizer_type = args.optimizer
    scheduler_type = args.scheduler

    experiment_name_parts = [
        project_name,
        model,
        optimizer_type,
        scheduler_type
    ]
    return "_".join(experiment_name_parts)

def generate_run_id(args):
    """
    Generate a run ID based on the provided arguments with out special characters.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The run ID.
    """
    learning_rate = f"{args.learning_rate}".replace(".", "") 
    weight_decay = f"{args.weight_decay}".replace(".", "") 
    time = datetime.now().strftime("H%HM%M")

    run_id_parts = [
        'run',
        time,
        f'lr{learning_rate}',
        f'wd{weight_decay}'
    ]
    return "".join(run_id_parts)

import torch.optim as optim

def create_optimizer(args, model_parameters):
    """
    Create an optimizer based on the specified arguments.

    Args:
        args (argparse.Namespace): Arguments containing optimizer-related configuration.
        model_parameters (iterable): Iterable of model parameters.

    Returns:
        torch.optim.Optimizer: Created optimizer.

    Raises:
        ValueError: If an invalid optimizer type is specified.
    """
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model_parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model_parameters,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-08
        )
    else:
        raise ValueError(f"Invalid optimizer type: {args.optimizer}")

    return optimizer


def create_loss_function(args):
    """
    Create a loss function based on the specified arguments.

    Args:
        args (argparse.Namespace): Arguments containing loss function-related configuration.

    Returns:
        torch.nn.Module: Created loss function.

    Raises:
        ValueError: If an invalid loss function type is specified.
    """
    if args.loss_function == 'cross_entropy':
        loss_function = nn.CrossEntropyLoss()
    elif args.loss_function == 'mse':
        loss_function = nn.MSELoss()
    elif args.loss_function == 'l1':
        loss_function = nn.L1Loss()
    elif args.loss_function == 'smooth_l1':
        loss_function = nn.SmoothL1Loss()
    elif args.loss_function == 'bce':
        loss_function = nn.BCELoss()
    else:
        raise ValueError(f"Invalid loss function type: {args.loss_function}")

    return loss_function

def create_scheduler(args):
    pass

def load_checkpoint(checkpoint_path,model,optimizer):
    """Loads a checkpoint into the model.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.Optim): The optimiser to load thr checkpoint into.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        None.
    """
    checkpoint = torch.load(checkpoint_path)

    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    # Load the state dict into the model.
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    checkpoint_values = {
        'epoch':checkpoint["epoch"],
        'epoch_metrics': checkpoint["epoch_metrics"]
    }
    return model,optimizer,checkpoint_values 


















