import collections
import json
import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from utils import log_utils as LogUtils
import numpy as np


class BehaviorCloning():
    """
    Behavior Cloning implementation using supervised learning.

    Args:
        model (torch.nn.Module): Neural network model to learn the policy.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion: Loss function to optimize.
    """
    def __init__(self, model, optimizer, criterion):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_loop(
            self,
            train_loader,
            test_loader,
            model_out_dir=None,
            num_epochs=10,
            num_epochs_logging=1,
            num_epochs_save=1,
            num_train_steps=1,
            num_val_steps=1,
            logger=None,
    ):
        """
        Training loop for multiple epochs with evaluation and logging.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for evaluation data.
            num_epochs (int): Number of training epochs.
            num_epochs_logging (int): Frequency of logging and evaluation in epochs.
            logger (Logger): Logger for training metrics.
        """
        #initial evaluation
        epoch = 0
        best_eval_loss = 100000  # for validation
        best_eval_flag = False
        writer = SummaryWriter(log_dir=os.path.join(model_out_dir, 'logs'))

        eval_loss = self.evaluate(test_loader,num_val_steps)
        if logger is not None:
            logger.info(f"Epoch: {epoch}, Evaluation Loss: {eval_loss:.4f}")

        for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training (For {num_epochs} Epochs)", unit="epoch", leave=False):#tqdm:显示进度条
            tr_loss=self.train(train_loader, num_train_steps,logger=logger)
            print("training epoch_{}".format(epoch+1))
            # visualize training curve
            writer.add_scalar('Loss/train', tr_loss, epoch)
            #eval:
            if epoch % num_epochs_logging == 0:
                eval_loss = self.evaluate(test_loader,num_val_steps)
                # visualize training curve
                writer.add_scalar('Loss/val', eval_loss, epoch)
                if logger is not None:
                    logger.info(f"Epoch: {epoch}, Evaluation Loss: {eval_loss:.4f}")
                best_eval_flag = eval_loss < best_eval_loss
                #save model
                if best_eval_flag:
                    ckpt_name='epoch_{}_best_validation_loss:'.format(epoch)+str(eval_loss)+'.pth'
                    torch.save(self.model.state_dict(), os.path.join(model_out_dir, ckpt_name))
                    continue
                if epoch%num_epochs_save==0:
                    ckpt_name='epoch_{}_validation_loss:'.format(epoch)+str(eval_loss)+'.pth'
                    torch.save(self.model.state_dict(), os.path.join(model_out_dir, ckpt_name))
        writer.close()



    def train(self, train_loader,num_train_steps, logger=None):
        """
        Trains the policy model using supervised learning on state-action pairs for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            logger (Logger): Logger for training metrics.
            log_wandb (bool): Whether to log training metrics to Weights & Biases.
        """
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training (Single Epoch)", unit="batch", leave=False):
            batch["actions"]=batch["actions"].to(self.device)
            batch_loss = self.train_on_batch(batch)
            epoch_loss += batch_loss

        avg_loss = epoch_loss / len(train_loader)

        if logger is not None:
            logger.info(f"Training Loss: {avg_loss:.4f}")
        return avg_loss


    def evaluate(self, eval_loader,num_eval_steps):
        """
        Evaluates the policy model on the test dataset and optionally in an environment.

        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.
            eval_env: Environment for additional evaluation metrics.
            obs_keys: List of observation keys to extract from the environment observation.
            num_episodes (int): Number of episodes to evaluate in the environment.

        Returns:
            tuple: Average evaluation loss and a dictionary of evaluation metrics.
        """
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluation", unit="batch", leave=False):
                batch["actions"] = batch["actions"].to(self.device)
                batch_loss = self.eval_on_batch(batch)
                epoch_loss += batch_loss

        avg_loss = epoch_loss / len(eval_loader)
        return avg_loss

    def train_on_batch(self, batch) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(batch["obs"])
        loss = self.criterion(predictions, batch["actions"])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_on_batch(self, batch) -> float:
        self.model.eval()
        predictions = self.model(batch["obs"])
        loss = self.criterion(predictions, batch["actions"])
        return loss.item()

    def call_policy(self, obs):
        return self.model(obs).detach().numpy()

