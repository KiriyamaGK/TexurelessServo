import torch.nn as nn
import torch.optim as optim
import torch
import json
import os
from utils.paths import PROJECT_ROOT_DIR

class EWC:
    def __init__(self, model, dataloader, weight, device='cuda'):
        """
        EWC 实现类

        Args:
            model: 神经网络模型
            dataloader: 旧任务的数据加载器
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.fisher = {}
        self.params = {}
        self.weight = weight

        # 计算费雪信息矩阵
        config_dir = os.path.join(PROJECT_ROOT_DIR, 'configs/train_mlp.json')
        with open(config_dir, "r") as f:
            self._config = json.load(f)
        self.optimizer = self._setup_optimizer(self._config["algorithm"]["optimizer"], model)
        self.criterion = self._setup_criterion(self._config["algorithm"]["loss"], self._config["dataset"]["seq_length"],
                                          self._config["dataset"]["output_dim"])
        self.compute_fisher(dataloader)

    @staticmethod
    def _setup_optimizer(optimizer_config: dict, model: torch.nn.Module):
        """
        Set up the optimizer.
        """
        from networks.helpers import get_optimizer_cls
        optimizer = get_optimizer_cls(optimizer_config["name"])
        return optimizer(model.parameters(), **optimizer_config["params"])

    @staticmethod
    def _setup_criterion(criterion_config_name: dict,seq_length: int,output_dim: int):
        """
        Set up the criterion.
        """

        def composed_loss_fn(x, x_hat):
            # x = x.reshape(x.size(0), -1)
            # x_hat = x_hat.reshape(x_hat.size(0), -1)
            from networks.helpers import get_loss_fn
            loss_fn = get_loss_fn(criterion_config_name["name"],criterion_config_name["weight"],seq_length,output_dim)
            loss_dict = loss_fn(x, x_hat)
            return loss_dict

        return composed_loss_fn
    def compute_fisher(self, dataloader, num_steps=1000):
        """
        计算费雪信息矩阵

        Args:
            dataloader: 数据加载器
            num_samples: 用于估计的样本数量
        """
        self.model.eval()

        # 首先保存当前模型参数（旧任务训练完的参数）
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
                self.fisher[name] = torch.zeros_like(param.data)

        # 收集梯度平方的期望值
        samples_count = 0
        data_loader_iter = iter(dataloader)
        num_steps = min(num_steps, len(dataloader))

        for _ in range(num_steps):
            try:
                batch = next(data_loader_iter)  # 从迭代器data_loader_iter中获取下一个数据批次
            except StopIteration:
                # reset for next dataset pass
                data_loader_iter = iter(dataloader)
                batch = next(data_loader_iter)

            for k, _ in batch.items():
                if k != "obs":
                    batch[k] = batch[k].to(self.device)

            batch_loss_dict = self.compute_grad_on_batch(batch)

            # 累加梯度平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2 / num_steps

        print(f"Computed Fisher information using {num_steps} steps.")

    def compute_grad_on_batch(self, batch, is_ewc_episode = False, ewc_batch_penalty = 0.00) -> dict:
        self.optimizer.zero_grad()
        predictions = self.model(batch["obs"])
        loss_dict = self.criterion(predictions, {k:batch[k] for k in batch if k != "obs"})
        if is_ewc_episode:
            loss_dict["loss_ewc"] = ewc_batch_penalty
            loss_dict["loss"] += ewc_batch_penalty
        loss=loss_dict['loss']
        loss.backward()
        # self.optimizer.step()   #NN gradient should NOT be updated when obtaining fisher matrix
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()  #torch.tensor->float
        return loss_dict

    def penalty(self, model):
        """
        计算EWC惩罚项

        Args:
            model: 当前模型

        Returns:
            penalty: EWC惩罚值
        """
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                # 惩罚项: 0.5 * fisher * (current_param - old_param)^2
                loss += torch.sum(
                    self.fisher[name] * (param - self.params[name]) ** 2
                )
        return 0.5 * self.weight * loss
