import torch
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self,weight,seq_length,output_dim):
        super().__init__()
        self.weight_tr = weight["weight_tr"]
        self.weight_rot = weight["weight_rot"]
        self.seq_length = seq_length
        self.output_dim = output_dim
    def forward(self, inputs, targets):
        # 自定义损失计算逻辑
        inputs=inputs.view(-1,self.seq_length,self.output_dim)
        targets=targets.view(-1,self.seq_length,self.output_dim)
        loss_tr = torch.mean((inputs[:,:,0:2] - targets[:,:,0:2]) ** 2)*self.weight_tr  # mse,平方和/(b*t*n_dim)
        loss_rot = torch.mean((inputs[:, :, 2:] - targets[:, :, 2:]) ** 2)*self.weight_rot
        loss = loss_tr+ loss_rot
        loss_dict = {
        "loss": loss,
        "loss_tr": loss_tr,
        "loss_rot": loss_rot}
        return loss_dict



