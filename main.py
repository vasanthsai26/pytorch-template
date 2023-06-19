from utils.config import load_config,setup_and_validate_config,save_config
from utils import helper
from data_setup import get_data,create_dataloaders
from engine import ClassificationTrainer
from model_builder import VehicleImageClassifier
import wandb

def main(args):
    """
    Main function for training a vehicle image classifier.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Obtain train, validation, and test data
    train_df, val_df, test_df = get_data(args)

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_df, val_df, test_df, args)

    # Create the model
    model = VehicleImageClassifier()
    optimizer = helper.create_optimizer(args, model.parameters())
    criterion = helper.create_loss_function(args)
    scheduler = helper.create_lr_scheduler(args, optimizer)

    # Load checkpoint if specified
    if args.resume:
        model, optimizer, checkpoint_values = helper.load_checkpoint(
            args.checkpt_path, model, optimizer)

    wandb.login()

    trainer = ClassificationTrainer(model, criterion, optimizer, scheduler, args)
    history_df = trainer.fit(train_dataloader, val_dataloader)

    wandb.finish()


if __name__ == '__main__':
    cfg = load_config()
    cfg = setup_and_validate_config(cfg)
    save_config(cfg)
    main(cfg)