import os
import opendatasets as od
import wandb
import pandas as pd
from tqdm.notebook import tqdm
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

## torch imports
import torch
import torch.nn as nn

## sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

class ClassificationTrainer():
    def __init__(self,
                 model: torch.nn.Module, 
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 args
                 ):
        self.args = args
        self.model= model.to(self.args.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader):
        
        history_df = self.get_history_df()
        max_metric = 0 if pd.isna(history_df[self.args.metric].max()) else history_df[self.args.metric].max()
        early_stopping_counter = 0

        self.initialize_wandb()

        for epoch in tqdm(range(self.args.start_iter, self.args.epochs)):
            train_metrics = self.train_epoch(dataloader=train_dataloader,
                                             criterion=self.criterion,
                                             optimizer= self.optimizer,
                                             scheduler=self.scheduler)
            val_metrics   = self.test_epoch(dataloader=test_dataloader,
                                          criterion=self.criterion)
            epoch_metrics,history_df = self.log_metrics(epoch,
                                                        train_metrics,
                                                        val_metrics,
                                                        history_df)
            
            max_metric,early_stopping_counter = self.update_early_stopping(epoch,
                                                                           epoch_metrics,
                                                                           max_metric,
                                                                           early_stopping_counter)

            self.take_checkpoint(epoch, epoch_metrics, early_stopping_counter)
  
            if early_stopping_counter > self.args.early_stop_patience:
                print(f"[INFO] Early stopping: no improvement for \
                {self.args.early_stop_patience} epochs.")
                return history_df
        return history_df 
    
    def update_early_stopping(self, epoch, epoch_metrics, max_metric, early_stopping_counter):
        """
        Update the early stopping mechanism based on the epoch metrics.

        Args:
            epoch (int): The current epoch number.
            epoch_metrics (dict): Dictionary containing metrics for the current epoch.
            max_metric (float): The maximum value of the chosen metric observed so far.
            early_stopping_counter (int): Counter for early stopping mechanism.

        Returns:
            Tuple[float, int]: Updated maximum metric value and early stopping counter.
        """
        if epoch_metrics[self.args.metric] <= max_metric:
            early_stopping_counter += 1
            return max_metric, early_stopping_counter

        print(f"[INFO] {self.args.metric} improved")
        self.print_metrics(epoch, epoch_metrics)
        max_metric = epoch_metrics[self.args.metric]
        early_stopping_counter = 0

        if epoch_metrics[self.args.metric] >= self.args.minimum_threshold:
            self.save_model(
                model_name=f'{self.args.run_id}_best_model_{self.args.metric}_{epoch_metrics[self.args.metric]:.4f}.pth',
                epoch=epoch,
                epoch_metrics=epoch_metrics
            )
        return max_metric, early_stopping_counter

    def take_checkpoint(self, epoch, epoch_metrics, early_stopping_counter):
        """
        Save a checkpoint of the model and its associated metadata.

        Args:
            epoch (int): The current epoch number.
            epoch_metrics (dict): Dictionary containing metrics for the current epoch.
            early_stopping_counter (int): Counter for early stopping mechanism.

        Returns:
            None
        """
        if (epoch + 1) % self.args.checkpoint_period == 0 and early_stopping_counter != 0:
            print("[INFO] Checkpoint reached")
            self.print_metrics(epoch, epoch_metrics)
            self.save_model(
                model_name=f'{self.args.run_id}_chkpointat_{epoch+1}.pth',
                epoch=epoch,
                epoch_metrics=epoch_metrics
            )

    def initialize_wandb(self):
        """
        Initializes and configures the Weights & Biases (wandb) run for logging experiment data.

        Args:
            project_name (str): The name of the project for the run.
            run_name (str): The name of the run.
            resume (bool, optional): Specifies whether to resume the run if it exists. Defaults to False.
        """
        if self.args.resume:
            wandb.init(project=self.args.project_name, 
                name=self.args.experiment_name,
                id=self.args.run_id, 
                resume="allow",
                config=self.args)
        else:
            wandb.init(
                project=self.args.project_name, 
                name=self.args.experiment_name,
                id=self.args.run_id, 
                config=self.args
                )

    def log_metrics(self, epoch, train_metrics, val_metrics, history_df):
        """
        Logs the metrics for a given epoch, updates the history dataframe,
        and saves it to a CSV file at specified checkpoint periods.
        
        Args:
            epoch (int): The current epoch number.
            train_metrics (dict): Dictionary containing training metrics.
            val_metrics (dict): Dictionary containing validation metrics.
            history_df (pandas.DataFrame): The history dataframe to update.
            
        Returns:
            tuple: A tuple containing the epoch metrics and updated history dataframe.
        """
        epoch_metrics = OrderedDict()
        epoch_metrics.update(train_metrics)
        epoch_metrics.update(val_metrics)

        history_df = history_df.append(epoch_metrics, ignore_index=True)
        
        # Save history_df to CSV file at specified checkpoint periods
        if epoch % self.args.checkpoint_period == 0:
            file_path = os.path.join(self.args.history_dir, f"{self.args.run_id}.csv")
            history_df.to_csv(file_path, index=False)

        wandb.log(epoch_metrics, step=epoch)
        
        return epoch_metrics, history_df

    def print_metrics(self, epoch, epoch_metrics):
        """
        Print the metrics for the given epoch.

        Args:
            epoch (int): The current epoch number.
            epoch_metrics (dict): Dictionary containing metrics for the current epoch.

        Returns:
            None
        """
        train_loss = epoch_metrics['train_loss']
        train_acc = epoch_metrics['train_acc']
        train_f1_score = epoch_metrics['train_f1']
        val_loss = epoch_metrics['val_loss']
        val_acc = epoch_metrics['val_acc']
        val_f1_score = epoch_metrics['val_f1']

        print(f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | train_f1_score: {train_f1_score:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1_score: {val_f1_score:.4f}")
        
    def get_history_df(self)->pd.DataFrame:
        """
        Create an empty history dataframe containing values for train and validation metrics.

        Returns:
            A DataFrame with train and validation metrics
        """
        file_path = os.path.join(self.args.history_dir,f"{self.args.run_id}.csv")
        if os.path.isfile(file_path):
            history_df = pd.read_csv(file_path)
        else:
            history_df = pd.DataFrame(
                columns=["train_loss","train_acc","train_precision",
                        "train_recall","train_f1",
                        "val_loss","val_acc","val_precision",
                        "val_recall","val_f1"
                        ])
        return history_df
    
    def reset_epoch_metrics(self):
        """
        Resets the epoch metrics to 0.

        Returns:
            A dictionary of epoch metrics, with the following keys:
                * epoch_loss: The epoch loss.
                * epoch_accuracy: The epoch accuracy.
                * epoch_precision: The epoch precision.
                * epoch_recall: The epoch recall.
        """
        metrics = OrderedDict()
        metrics.update({
            "epoch_loss": 0,
            "epoch_accuracy": 0,
            "epoch_precision": 0,
            "epoch_recall": 0,
            "epoch_f1_score":0,
        })
        return metrics
    
    def evaluate_batch(self,y_true, y_pred):
        """
        Evaluate the accuracy, precision, and recall of a batch of predicted labels relative to ground truth labels.

        Args:
            y_true (torch.Tensor): A tensor of ground truth labels of shape (batch_size,)
            y_pred (torch.Tensor): A tensor of predicted labels of shape (batch_size,)

        Returns:
            dict: A dict containing the accuracy (float), precision (float), recall (float), and f1 score of the predicted labels.
        """
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        batch_metrics = {
            "batch_accuracy"  : accuracy_score(y_true, y_pred),
            "batch_precision" : precision_score(y_true, y_pred, average='macro'),
            "batch_recall"    : recall_score(y_true, y_pred, average='macro'),
            "batch_f1_score"  : f1_score(y_true, y_pred, average='macro')
        }
        return batch_metrics

    def update_metrics(self,batch_metrics,epoch_metrics) -> None:
        """
        Updates the epoch metrics with the batch metrics.

        Args:
            batch_metrics: The batch metrics.
            epoch_metrics: The epoch metrics.
        """
        epoch_metrics["epoch_accuracy"]  += batch_metrics["batch_accuracy"]
        epoch_metrics["epoch_precision"] += batch_metrics["batch_precision"]
        epoch_metrics["epoch_recall"]    += batch_metrics["batch_recall"]
        epoch_metrics["epoch_f1_score"]  += batch_metrics["batch_f1_score"]
    
    def get_avg_metrics(self,epoch_metrics, dataloader_len,train=True):
        """
        Get the average metrics for an epoch.

        Args:
            epoch_metrics: The epoch metrics.
            dataloader_len: The length of the dataloader.

        Returns:
            A dictionary of average metrics.
        """

        avg_metrics = OrderedDict()
        avg_metrics.update({
            f"{'train' if train else 'val'}_loss": round(epoch_metrics["epoch_loss"] / dataloader_len, 4),
            f"{'train' if train else 'val'}_acc": round(epoch_metrics["epoch_accuracy"] / dataloader_len, 4),
            f"{'train' if train else 'val'}_precision": round(epoch_metrics["epoch_precision"] / dataloader_len, 4),
            f"{'train' if train else 'val'}_recall": round(epoch_metrics["epoch_recall"] / dataloader_len, 4),
            f"{'train' if train else 'val'}_f1": round(epoch_metrics["epoch_f1_score"] / dataloader_len, 4),
        })

        return avg_metrics
    
    def train_epoch(self,dataloader: torch.utils.data.DataLoader, 
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler : torch.optim.lr_scheduler._LRScheduler) :
        """
        Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Returns:
            dict: A dictionary containing the train epoch metrics
        """
        # Put model in train mode
        self.model.train()
        # wandb.watch(self.model, criterion, log="all", log_freq=10)

        # Setup train metrics values
        epoch_metrics = self.reset_epoch_metrics()

        for batch_idx, (data, target) in enumerate(dataloader):
            # Send data to target device
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            y_pred = self.model(data)

            # Calculate  and accumulate loss
            loss = criterion(y_pred, target)
            epoch_metrics["epoch_loss"] += loss.item()

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            if self.args.grad_clip:
                nn.utils.clip_grad_value_(self.model.parameters(), 
                                          self.args.grad_clip)
            # Optimizer step
            optimizer.step()

            # Scheduler step
            scheduler.step()

            # Calculate and accumulate accuracy metric across all batches
            with torch.no_grad():
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                batch_metrics = self.evaluate_batch(target,y_pred_class)
                self.update_metrics(batch_metrics,epoch_metrics)

        # calculate avg metrics per batch
        avg_epoch_metrics = self.get_avg_metrics(epoch_metrics,len(dataloader))

        return avg_epoch_metrics
    
    def test_epoch(self,dataloader: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module):
        """
        Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on the dataloader.

        Returns:
            dict: A dictionary containing the test epoch metrics
        """
        # Put model in eval mode
        self.model.eval()

        # Setup test loss and test metric values
        epoch_metrics = self.reset_epoch_metrics()

        # Turn on inference mode
        with torch.inference_mode():

            for batch_idx, (data, target) in enumerate(dataloader):
                # Send data to target device
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                test_pred_logits = self.model(data)

                # Calculate and accumulate loss
                loss = criterion(test_pred_logits, target)
                epoch_metrics["epoch_loss"] += loss.item()

                #Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                batch_metrics = self.evaluate_batch(target,test_pred_labels)
                self.update_metrics(batch_metrics,epoch_metrics)

            # calculate avg metrics per batch
            avg_epoch_metrics = self.get_avg_metrics(epoch_metrics,len(dataloader),train=False)

        return avg_epoch_metrics
   
    def save_model(self, model_name, epoch, epoch_metrics):
        """
        Save a PyTorch model, its associated metadata, and optimizer state to a target directory.

        Args:
            model_name (str): Name of the model file to be saved.
            epoch (int): The current epoch number.
            epoch_metrics (float): The value of the validation metrics at the current epoch.

        Returns:
            None
        """

        model_save_path = os.path.join(self.args.models_dir, model_name)
        print(f"[INFO] Saving model to: {model_save_path}")

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch_metrics': epoch_metrics
            },
            f=model_save_path
        )

    
    

    