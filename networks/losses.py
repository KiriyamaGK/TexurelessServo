import torch
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self,weight,seq_length,output_dim):
        super().__init__()
        self.weight_tr = weight["weight_tr"] #m
        self.weight_rot = weight["weight_rot"] #deg
        self.weight_pos = weight["weight_pose_estm"] if "weight_pose_estm" in weight else None #mm,deg
        self.seq_length = seq_length
        self.output_dim = output_dim
        assert self.output_dim in [3, 6]
        self.division=2 if self.output_dim == 3 else 3
        # print("action_dim", self.output_dim)

    def forward(self, pred_dict, label_dict):
        # 自定义损失计算逻辑
        inputs=pred_dict["output_tensor"]
        targets=label_dict["actions"]

        inputs=inputs.view(-1,self.seq_length,self.output_dim)
        targets=targets.view(-1,self.seq_length,self.output_dim)
        loss_tr = torch.mean((inputs[:,:,0:self.division] - targets[:,:,0:self.division]) ** 2)*self.weight_tr  # mse,平方和/(b*t*n_dim)
        loss_rot = torch.mean((inputs[:, :, self.division:] - targets[:, :, self.division:]) ** 2)*self.weight_rot
        loss = loss_tr+ loss_rot
        loss_dict = {
        "loss": loss,
        "loss_tr": loss_tr,
        "loss_rot": loss_rot}

        if pred_dict["pred_delta_pos"] is not None:
            label_dict["delta_pos_curgoal"]=label_dict["delta_pos_curgoal"].view(-1,self.seq_length,6)
            pred_dict["pred_delta_pos"]=pred_dict["pred_delta_pos"].view(-1,self.seq_length,6)

            loss_pos=torch.mean((label_dict["delta_pos_curgoal"] - pred_dict["pred_delta_pos"]) ** 2)*self.weight_pos
            loss_dict["loss_pos"] = loss_pos
            loss_dict["loss"]+=loss_pos
        # print(list(loss_dict.keys()))
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

class TCL_MSE(nn.Module):
    def __init__(self,weight,seq_length,output_dim):
        super().__init__()
        self.weight_tr = weight["weight_tr"]#m
        self.weight_rot = weight["weight_rot"] #deg
        self.weight_pos = weight["weight_pose_estm"] if "weight_pose_estm" in weight else None #mm deg

        self.weight_tcl_img= weight["weight_tcl_img"]
        self.weight_tcl_act= weight["weight_tcl_act"]
        self.weight_tcl_pos = weight["weight_tcl_pos"] if "weight_tcl_pos" in weight else None

        self.seq_length = seq_length
        self.output_dim = output_dim
        assert self.output_dim in [3,6]
        self.division = 2 if self.output_dim == 3 else 3

        # print("action_dim",self.output_dim)

    def forward(self, pred_dict, label_dict):
        # 自定义损失计算逻辑
        pred_act = pred_dict["output_tensor"]
        pred_act_aug=pred_dict["output_tensor_aug"]
        img_feat=pred_dict["x_img_feat"]
        img_goal_feat=pred_dict["x_img_goal_feat"]
        img_aug_feat = pred_dict["x_img_aug_feat"]
        img_goal_aug_feat = pred_dict["x_img_goal_aug_feat"]
        target_act = label_dict["actions"]

        #act
        pred_act=pred_act.view(-1,self.seq_length,self.output_dim)
        target_act=target_act.view(-1,self.seq_length,self.output_dim)
        loss_tr = torch.mean((pred_act[:,:,0:self.division] - target_act[:,:,0:self.division]) ** 2)*self.weight_tr                   # mse,平方和/(b*t*n_dim)
        loss_rot = torch.mean((pred_act[:, :, self.division:] - target_act[:, :, self.division:]) ** 2)*self.weight_rot

        #tcl
        pred_act_aug = pred_act_aug.view(-1, self.seq_length, self.output_dim)
        loss_tcl_act=torch.mean((pred_act - pred_act_aug) ** 2)*self.weight_tcl_act
        loss_tcl_img=torch.mean((img_feat - img_aug_feat) ** 2)*self.weight_tcl_img+torch.mean((img_goal_feat - img_goal_aug_feat) ** 2)*self.weight_tcl_img

        loss = loss_tr+ loss_rot+loss_tcl_act+loss_tcl_img

        loss_dict = {
        "loss": loss,
        "loss_tr": loss_tr,
        "loss_rot": loss_rot,
        "loss_tcl_act": loss_tcl_act,
        "loss_tcl_img": loss_tcl_img}

        if pred_dict["pred_delta_pos"] is not None:
            label_dict["delta_pos_curgoal"] = label_dict["delta_pos_curgoal"].view(-1, self.seq_length, 6)
            pred_dict["pred_delta_pos"] = pred_dict["pred_delta_pos"].view(-1, self.seq_length, 6)
            if label_dict["delta_pos_curgoal"] is None or pred_dict["pred_delta_pos"] is None:
                raise ValueError("Predictions or targets cannot be None.")
            loss_pos = torch.mean(
                (label_dict["delta_pos_curgoal"] - pred_dict["pred_delta_pos"]) ** 2) * self.weight_pos
            loss_dict["loss_pos"] = loss_pos
            loss_dict["loss"] += loss_pos

            if pred_dict["pred_delta_pos_aug"] is not None:
                pred_dict["pred_delta_pos_aug"] = pred_dict["pred_delta_pos_aug"].view(-1, self.seq_length, 6)
                loss_tcl_pos = torch.mean(
                    (pred_dict["pred_delta_pos_aug"]- pred_dict["pred_delta_pos"]) ** 2) * self.weight_tcl_pos
                loss_dict["loss_tcl_pos"] = loss_tcl_pos
                loss_dict["loss"] += loss_tcl_pos
        # print(list(loss_dict.keys()))
        return loss_dict


