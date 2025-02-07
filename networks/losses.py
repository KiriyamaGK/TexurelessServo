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

class ModifiedFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,preds, targets):
      ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
          Arguments:
          preds (B x c x h x w)
          gt_regr (B x c x h x w)
      '''
      assert preds.size() == targets.size()
      pos_inds = targets.eq(1).float()
      neg_inds = targets.lt(1).float()

      neg_weights = torch.pow(1 - targets, 4)

      loss = 0
      for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
          loss = loss - neg_loss
        else:
          loss = loss - (pos_loss + neg_loss) / num_pos
      return loss / len(preds)

class MSE_and_ModifiedFocalLoss(nn.Module):
    def __init__(self,weight,seq_length,output_dim):
        super().__init__()
        self.weight_tr = weight["weight_tr"]
        self.weight_rot = weight["weight_rot"]
        self.weight_htmp = weight["weight_heatmap"]
        self.seq_length = seq_length
        self.output_dim = output_dim
    def forward(self,preds_dict,act_gt):
      ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
          Arguments:
          preds (B x c x h x w)
          gt_regr (B x c x h x w)
      '''
      #================for heatmap=============
      preds=preds_dict["pred_heatmap"]
      targets=preds_dict["kpt_heatmap_gt"]
      assert preds.size() == targets.size()
      pos_inds = targets.eq(1).float()
      neg_inds = targets.lt(1).float()

      neg_weights = torch.pow(1 - targets, 4)

      loss_htmp = 0
      for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
          loss_htmp = loss_htmp - neg_loss
        else:
          loss_htmp = loss_htmp - (pos_loss + neg_loss) / num_pos
      loss_htmp=loss_htmp / len(preds)*self.weight_htmp

      #===========for action=================
      inputs=preds_dict["pred_act"]
      targets=act_gt
      inputs = inputs.view(-1, self.seq_length, self.output_dim)
      targets = targets.view(-1, self.seq_length, self.output_dim)
      loss_tr = torch.mean((inputs[:, :, 0:2] - targets[:, :, 0:2]) ** 2) * self.weight_tr  # mse,平方和/(b*t*n_dim)
      loss_rot = torch.mean((inputs[:, :, 2:] - targets[:, :, 2:]) ** 2) * self.weight_rot
      loss = loss_tr + loss_rot+loss_htmp
      loss_dict = {
          "loss": loss,
          "loss_tr": loss_tr,
          "loss_rot": loss_rot,
          "loss_heatmap": loss_htmp}
      return loss_dict




