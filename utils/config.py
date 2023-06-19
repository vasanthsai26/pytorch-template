import argparse
import json
import os
import sys
import torch

from datetime import datetime
from munch import Munch
from helper import seed_everything,generate_experiment_name,generate_run_id


def load_config():
    """
    Load the configuration from a JSON file or command line arguments.

    If a JSON file is provided as a command line argument, it is loaded and returned as the configuration.
    Otherwise, the configuration is parsed from command line arguments.

    Returns:
        Munch: The loaded configuration.
    """
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1], 'r') as f:
            print(f"Loading configuration from {sys.argv[1]} ...")
            cfg = json.load(f)
            cfg = Munch(cfg)
    else:
        print("Loading configuration from command line arguments ...")
        cfg = parse_args()
        cfg = Munch(cfg.__dict__)
    return cfg

def parse_args():
    """
    Parse command line arguments for the PyTorch Image Classification project.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')

    # Data configs
    parser.add_argument('--experiment_name', type=str, default='_', help='Name of the experiment')
    parser.add_argument('--project_name', type=str, default='project', help='Name of the project')
    parser.add_argument('--exp_date', type=str, help='Folder name and date for this experiment.')
    parser.add_argument('--exp_dir', type=str, default='experiments')
    parser.add_argument('--run_id', type=str, default='run001')
    parser.add_argument('--checkpt_path', type=str, help='chekpoint model path to resume the experiment.')

    parser.add_argument('--models_dir', type=str, default='models', help='Name of the models directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Name of the results directory')
    parser.add_argument('--history_dir', type=str, default='history', help='Name of the history directory')
    
    parser.add_argument('--data_dir', type=str, default='data', help='Name of the data directory')
    parser.add_argument('--meta_dir', type=str, default='data/meta', help='Name of the data directory')
    parser.add_argument('--train_dir', type=str, default='data/train', help='Name of the train data directory')
    parser.add_argument('--test_dir', type=str, default='data/test', help='Name of the test data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 32)')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training split (default: 0.8)')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='Number of workers for data loading (default: os.cpu_count())')
    parser.add_argument('--means', nargs='+', type=float, action='append', default=[[0.485, 0.456, 0.406]],
                        help='List of means for image normalization (default=[0.485, 0.456, 0.406])')
    parser.add_argument('--stds', nargs='+', type=float, action='append', default=[[0.229, 0.224, 0.225]],
                        help='List of stds for image normalization (default:[0.229, 0.224, 0.225])')

    # Model settings
    parser.add_argument('--model', type=str, default='model', help='Type of model architecture (default: model)')
    parser.add_argument('--resume', action='store_true', default=False, help='Path to a checkpoint file to resume training from')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pre-trained model from torchvision')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')

    # Training settings
    parser.add_argument('--epochs', type=int, default=10, help='Number of total epochs to train (default: 10)')
    parser.add_argument('--start_iter', type=int, default=0, help='Training start point')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--early_stop_threshold', type=float, default=0.001,
                        help='Minimum improvement threshold required for early stopping')
    parser.add_argument('--minimum_threshold', type=float, default=0.5,
                        help='Minimum metric threshold required for saving the models')
    parser.add_argument('--checkpoint_period', type=int, default=1, help='Interval for saving checkpoints during training')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping threshold')
    parser.add_argument('--metric', type=str, default='val_f1',
                        choices=["val_loss", "val_acc", "val_precision", "val_recall", "val_f1"],
                        help="Evaluation metric to use")

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'SGD'], 
                        help='Optimizer for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,help='weight decay (default: 1e-5)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')

    # Scheduler
    parser.add_argument('--scheduler_type', type=str, 
                        choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CyclicLR'], 
                        default='StepLR', help='Type of LR scheduler')
    parser.add_argument('--level', default='epoch', choices=['epoch', 'batch'],
                            help='Scheduling level: epoch or batch')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, 
                        help='Gamma value for StepLR, MultiStepLR, and ExponentialLR schedulers')
    parser.add_argument('--milestones', type=int, nargs='+', default=[10, 20, 30], 
                        help='Milestone values for MultiStepLR scheduler')
    parser.add_argument('--mode', type=str, choices=['min', 'max'], default='min', 
                        help='Mode for ReduceLROnPlateau scheduler')
    parser.add_argument('--factor', type=float, default=0.1, 
                        help='Factor value for ReduceLROnPlateau scheduler')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience value for ReduceLROnPlateau scheduler')
    parser.add_argument('--T_max', type=int, default=10, 
                        help='T_max value for CosineAnnealingLR scheduler')
    parser.add_argument('--eta_min', type=float, default=0, 
                        help='Eta_min value for CosineAnnealingLR scheduler')
    parser.add_argument('--base_lr', type=float, default=0.001, 
                        help='Base learning rate for CyclicLR scheduler')
    parser.add_argument('--max_lr', type=float, default=0.01, 
                        help='Max learning rate for CyclicLR scheduler')

    # Loss Function
    parser.add_argument('--loss_function', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'mse', 'l1', 'smooth_l1', 'bce'], 
                        help='Loss function for training')

    # Other settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                                help='device to use for training (default: cuda if available, else cpu)')
    parser.add_argument('--seed', type=int, default=42,help='random seed (default: 42)')

    return parser.parse_args()

def setup_and_validate_config(args):
    """
    Setup the configuration based on the provided arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        argparse.Namespace: The updated configuration.
    """
    seed_everything(args.seed)

    if args.resume:
        if not os.path.isfile(args.checkpt_path) or not args.checkpt_path.endswith(".pth"):
            raise ValueError(f"Invalid path: '{args.checkpt_path}' does not exist.")  
    else:
        args.experiment_name = generate_experiment_name(args)
        args.run_id = generate_run_id(args)
        args.exp_date = datetime.now().strftime("%Y%m%d")
        args.exp_dir = os.path.join(os.getcwd(),args.experiment_name,args.exp_date)
        os.makedirs(args.exp_dir, exist_ok=True)
        for folder in ['models', 'results', 'history']:
            args[f'{folder}_dir'] = os.path.join(args.exp_dir, folder)
            os.makedirs(args[f'{folder}_dir'], exist_ok=True)
    
    print(f"Model Training from {args.exp_dir}")
    return args

def save_config(args):
    """
    Save the configuration to a JSON file.

    Args:
        args (argparse.Namespace): The configuration to save.

    Returns:
        None
    """
    args.config_dir = os.path.join(args.exp_dir, f'config_{args.start_iter}.json')

    print(args.config_dir)
    with open(args.config_dir, 'w+') as config_file:
        json.dump(args, config_file, indent=4)

    print(f"Configuration file '{args.config_dir}' has been created.")
