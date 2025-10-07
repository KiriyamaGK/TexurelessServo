import torch
import os
from torch.utils.tensorboard import SummaryWriter
from utils import log_utils as LogUtils
# from outdated_codes.CenterNet2 import CenterNet_ResNet18

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

        eval_loss_dict = self.evaluate(test_loader, num_val_steps)
        eval_loss = eval_loss_dict['loss']
        if logger is not None:
            logger.info(f"Epoch: {epoch}, Evaluation Loss: {eval_loss:.4f}")

        for epoch in range(1, num_epochs + 1):
            print("====================epoch_{}=====================".format(epoch))
            train_loss_dict=self.train(train_loader, num_train_steps,logger=logger)
            # visualize training curve
            for itm_tr in train_loss_dict:
                writer.add_scalar('Loss/train/{}'.format(itm_tr), train_loss_dict[itm_tr], epoch)
            #eval:
            if epoch % num_epochs_logging == 0:
                eval_loss_dict = self.evaluate(test_loader,num_val_steps)
                eval_loss=eval_loss_dict['loss']
                # visualize training curve
                for itm_val in eval_loss_dict:
                    writer.add_scalar('Loss/val/{}'.format(itm_val), eval_loss_dict[itm_val], epoch)
                if logger is not None:
                    logger.info(f"Epoch: {epoch}, Evaluation Loss: {eval_loss:.4f}")
                best_eval_flag = eval_loss < best_eval_loss
                #save model
                if best_eval_flag:
                    best_eval_loss=eval_loss
                    ckpt_name='epoch_{}_best_validation_loss_'.format(epoch)+str(eval_loss)+'.pth'
                    torch.save(self.model.state_dict(), os.path.join(model_out_dir, ckpt_name))
                    continue
            if epoch%num_epochs_save==0:
                ckpt_name='epoch_{}_validation_loss_'.format(epoch)+str(eval_loss)+'.pth'
                torch.save(self.model.state_dict(), os.path.join(model_out_dir, ckpt_name))
        writer.close()



    def train(self, train_loader,num_train_steps, logger=None, is_ewc_epoch = False,ewc_batch_penalty_func = None):
        """
        Trains the policy model using supervised learning on state-action pairs for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            logger (Logger): Logger for training metrics.
            log_wandb (bool): Whether to log training metrics to Weights & Biases.
        """
        data_loader_iter = iter(train_loader)
        num_train_steps=min(num_train_steps,len(train_loader))
        print("num_train_steps={}".format(num_train_steps))
        idx=0
        epoch_loss_dict={}
        # for _ in tqdm(range(num_train_steps)):
        for _ in LogUtils.custom_tqdm(range(num_train_steps),desc="Training......"):
            try:
                batch = next(data_loader_iter)  # 从迭代器data_loader_iter中获取下一个数据批次
            except StopIteration:
                # reset for next dataset pass
                data_loader_iter = iter(train_loader)
                batch = next(data_loader_iter)

            for k, _ in batch.items():
                if k !="obs":
                    batch[k] = batch[k].to(self.device)

            batch_loss_dict = self.train_on_batch(batch,is_ewc_epoch = is_ewc_epoch,ewc_batch_penalty_func = ewc_batch_penalty_func)
            if idx==0:
                for k,v in batch_loss_dict.items():
                    epoch_loss_dict[k]=v
                    # print("===========key:",v)
            else:
                for k,v in batch_loss_dict.items():
                    epoch_loss_dict[k]+=v
                    # print("===========key:", v)
            idx+=1
        for k, v in epoch_loss_dict.items():
            epoch_loss_dict[k] /= num_train_steps
        avg_loss = epoch_loss_dict["loss"]
        if logger is not None:
            logger.info(f"Training Loss: {avg_loss:.4f}")
        return epoch_loss_dict


    def evaluate(self, eval_loader,num_eval_steps):
        """
        Evaluates the policy model on the test dataset and optionally in an environment.

        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.
            eval_env: Environment for additional evaluation metrics.
            obs_keys: List of observation keys to extract from the environment observation.
            num_episodes (int): Number of episodes to evaluate in the environment.

        Returns:
            dict: Average evaluation loss and a dictionary of evaluation metrics.
        """
        num_eval_steps=min(num_eval_steps,len(eval_loader))
        with torch.no_grad():
            data_loader_iter = iter(eval_loader)
            idx = 0
            epoch_loss_dict = {}
            # for _ in tqdm(range(num_eval_steps)):
            for _ in LogUtils.custom_tqdm(range(num_eval_steps),desc="Evaluating......"):
                try:
                    batch = next(data_loader_iter)  # 从迭代器data_loader_iter中获取下一个数据批次
                except StopIteration:
                    # reset for next dataset pass
                    data_loader_iter = iter(eval_loader)
                    batch = next(data_loader_iter)

                for k, _ in batch.items():
                    if k != "obs":
                        batch[k] = batch[k].to(self.device)

                batch_loss_dict = self.eval_on_batch(batch)
                if idx == 0:
                    for k, v in batch_loss_dict.items():
                        epoch_loss_dict[k] = v
                else:
                    for k, v in batch_loss_dict.items():
                        epoch_loss_dict[k] += v
                idx += 1
            for k, v in epoch_loss_dict.items():
                epoch_loss_dict[k] /= num_eval_steps
            return epoch_loss_dict

    def train_on_batch(self, batch, is_ewc_epoch = False, ewc_batch_penalty_func = None) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        # if isinstance(self.model, CenterNet_ResNet18):
        #     pred,label= self.model(batch["obs"])
        #     loss_1 = self.criterion(pred,label)
        #     # loss_2 = self.criterion(gau2, x4)
        #     # loss = loss_1 + loss_2
        #     loss_dict = {'loss': loss_1}
        # else:
        predictions = self.model(batch["obs"])
        loss_dict = self.criterion(predictions, {k:batch[k] for k in batch if k != "obs"})
        if is_ewc_epoch:
            loss_dict["loss_ewc"] = ewc_batch_penalty_func(self.model)
            loss_dict["loss"] += ewc_batch_penalty_func(self.model)
        loss=loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()  #torch.tensor->float
        return loss_dict

    def eval_on_batch(self, batch) -> dict:
        self.model.eval()
        # if isinstance(self.model, CenterNet_ResNet18):
        #     pred, label = self.model(batch["obs"])
        #     loss_1 = self.criterion(pred, label)
        #     # loss_2 = self.criterion(gau2, x4)
        #     # loss = loss_1 + loss_2
        #     loss_dict = {'loss': loss_1}
        # else:
        predictions = self.model(batch["obs"])
        loss_dict = self.criterion(predictions, {k:batch[k] for k in batch if k != "obs"})
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
        return loss_dict

    def call_policy(self, obs):
        return self.model(obs).detach().numpy()

