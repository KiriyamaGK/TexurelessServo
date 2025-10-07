import torch.nn as nn
import torch.optim as optim
import torch

class EWC:
    def __init__(self, model, dataloader, device='cuda'):
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

        # 计算费雪信息矩阵
        self.compute_fisher(dataloader)

    def compute_fisher(self, dataloader, num_samples=1000):
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

        for batch_idx, (data, target) in enumerate(dataloader):
            if samples_count >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            output = self.model(data)

            # 计算损失（使用对数似然，与费雪信息定义一致）
            # 这里假设是分类任务
            loss = nn.functional.cross_entropy(output, target)

            # 反向传播计算梯度
            self.model.zero_grad()
            loss.backward()

            # 累加梯度平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2 / len(dataloader.dataset)

            samples_count += data.size(0)

        print(f"Computed Fisher information using {samples_count} samples")

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
        return 0.5 * loss
