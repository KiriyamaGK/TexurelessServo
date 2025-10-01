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


def train_with_ewc(model, train_loader, ewc_object, importance=1000,
                   epochs=10, lr=0.001, device='cuda'):
    """
    使用EWC训练新任务

    Args:
        model: 神经网络模型
        train_loader: 新任务的数据加载器
        ewc_object: EWC对象
        importance: EWC重要性系数λ
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        ewc_loss = 0
        task_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # 新任务的损失
            output = model(data)
            loss_task = nn.functional.cross_entropy(output, target)

            # EWC惩罚项
            loss_ewc = ewc_object.penalty(model)

            # 总损失 = 新任务损失 + λ * EWC惩罚
            loss = loss_task + importance * loss_ewc

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            task_loss += loss_task.item()
            ewc_loss += loss_ewc.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                      f'\tTask Loss: {loss_task.item():.6f}'
                      f'\tEWC Loss: {loss_ewc.item():.6f}')

        avg_loss = total_loss / len(train_loader)
        avg_task_loss = task_loss / len(train_loader)
        avg_ewc_loss = ewc_loss / len(train_loader)

        print(f'Epoch {epoch} Average Loss: {avg_loss:.4f} '
              f'(Task: {avg_task_loss:.4f}, EWC: {avg_ewc_loss:.4f})')