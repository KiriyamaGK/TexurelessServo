import collections
import random
import cv2
import torch
import math
import numpy as np
import torch.nn as nn
from networks.Network import NetworkBase
from networks import rgbd_resnet
from networks.helpers import get_activation_fn
from networks.SpatialSoftmax import SpatialSoftmax
from torchvision import models
from torch.nn import functional as F
import torch.distributions as D
import functools
import operator
# from ultralytics import YOLO
import warnings
from utils.input_process import add_gaussian_spot_to_image


class MLP(NetworkBase):
    def __init__(self, input_low_dim, output_dim,obs_keys,batch_size,seq_length,training,low_dim_hidden_sizes=None,hidden_sizes=None, activation="relu", output_activation=None,use_gmm=False,
                 encoder=None):
        super().__init__(input_low_dim, output_dim)

        #initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.freeze_encoder=encoder['freeze']
        self.encoder_name = encoder['name']
        self.use_siamese = encoder['siamese']
        self.using_pos_estm=encoder['using_pose_estimation'] if 'using_pose_estimation' in encoder else False
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.using_depth = encoder['using_depth'] if "using_depth" in encoder else False

        self.ss_num_kp = encoder['params']['SpatialSoftmax']['num_kp']
        self.ss_in_c = encoder['params']['SpatialSoftmax']['in_c'] if not self.using_depth else encoder['params']['SpatialSoftmax']['in_c']*4
        self.ss_in_h = encoder['params']['SpatialSoftmax']['in_h']
        self.ss_in_w = encoder['params']['SpatialSoftmax']['in_w']

        self.input_low_dim = input_low_dim
        self.output_dim = output_dim
        self.img_size = encoder['params']['img_size']
        self.crop_size = encoder['params']['crop_size']

        self.obs_keys = obs_keys
        self.low_dim_keys = []
        for obs_key in obs_keys:
            if "image" not in obs_key and "img" not in obs_key:
                self.low_dim_keys.append(obs_key)

        self.low_dim_hidden_sizes = low_dim_hidden_sizes if self.input_low_dim!=0 else [0]

        self.hidden_sizes = hidden_sizes
        self.use_GMM = use_gmm
        self.is_training = training

        self.num_cameras=encoder['num_cameras'] if "num_cameras" in encoder else 1
        if self.is_training:
            self.use_data_augmentation = encoder['data_augmentation']
            self.use_tcl_loss = encoder['task_consistency_loss']
            self.create_mixed_light_dataset = encoder['mixed_light_dataset'] if "mixed_light_dataset" in encoder else False
        else:
            self.use_data_augmentation = False
            self.use_tcl_loss = False
            self.create_mixed_light_dataset = False

        if self.use_tcl_loss:
            assert self.use_data_augmentation


        self.activation = get_activation_fn(activation)
        self.output_activation = get_activation_fn(output_activation) if output_activation is not None else None

        # img
        assert self.encoder_name in ['ResNet18', 'YOLO_v11']
        self.grid_source = self.build_grid(self.crop_size, self.crop_size)

        self.img_encs = nn.ModuleList()
        self.img_enc_goals = nn.ModuleList()
        self.spatial_softmaxs = nn.ModuleList()
        self.spatial_softmax_goals = nn.ModuleList()
        self.ee_lns = nn.ModuleList()
        self.ee_ln_goals = nn.ModuleList()

        if self.encoder_name == 'ResNet18':
            for _ in range(self.num_cameras):
                resnet18 = models.resnet18(pretrained=True) if not self.using_depth else rgbd_resnet.FusionEnhancedNet()
                resnet18_goal = models.resnet18(pretrained=True) if not self.using_depth else rgbd_resnet.FusionEnhancedNet()
                self.img_encs.append(torch.nn.Sequential(*(list(resnet18.children())[:-2])) if not self.using_depth else resnet18_goal)
                self.img_enc_goals.append(torch.nn.Sequential(*(list(resnet18_goal.children())[:-2])) if not self.using_depth else resnet18_goal)
                self.spatial_softmaxs.append(SpatialSoftmax(self.ss_in_c,self.ss_in_h,self.ss_in_w,self.ss_num_kp))
                self.spatial_softmax_goals.append(SpatialSoftmax(self.ss_in_c, self.ss_in_h, self.ss_in_w, self.ss_num_kp))
                self.ee_lns.append(nn.Linear(self.ss_num_kp * 2, (self.hidden_sizes[0]-self.low_dim_hidden_sizes[-1])//(2*self.num_cameras)))
                self.ee_ln_goals.append(nn.Linear(self.ss_num_kp * 2, (self.hidden_sizes[0]-self.low_dim_hidden_sizes[-1])//(2*self.num_cameras)))

        if self.freeze_encoder:
            for img_enc in self.img_encs:
                for param in img_enc.parameters():
                    param.requires_grad = False
            for img_enc in self.img_enc_goals:
                for param in img_enc.parameters():
                    param.requires_grad = False

        #for low_dim
        if self.input_low_dim!=0:
            self.mlp_pos = self._build_mlp([self.input_low_dim]+self.low_dim_hidden_sizes)

        #policy
        self.buffer=[]
        if self.use_GMM:
            self.gmm_modes = 5
            self.mlp_decoder_mean = self._build_mlp(self.hidden_sizes+[output_dim * self.gmm_modes])
            self.mlp_decoder_scale = self._build_mlp(self.hidden_sizes+[output_dim * self.gmm_modes])
            self.mlp_decoder_logits = self._build_mlp(self.hidden_sizes+[self.gmm_modes])
            self.min_std=0.0001
            self.activations = {
                "softplus": F.softplus,
                "exp": torch.exp,
            }
            self.std_activation = "softplus"
            self.low_noise_eval = False

        else:
            self.policy_mlp=self._build_mlp(self.hidden_sizes+[output_dim])

        if self.using_pos_estm:
            self.pos_estm_layer = self._build_mlp(self.hidden_sizes+[6])
            self.pos_estm_bottleneck=self._create_fcn(6,self.hidden_sizes[0])

        self.print_training_settings()

    def gmm_policy_mlp(self,x,b,seq):
        x_means = self.mlp_decoder_mean(x)
        x_scales = self.mlp_decoder_scale(x)
        x_logits = self.mlp_decoder_logits(x)

        x_means = x_means.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
        x_scales = x_scales.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
        x_logits = x_logits.view(b, seq, self.gmm_modes).contiguous()
        return x_means, x_scales, x_logits

    def forward(self, x):
        pos_pred= None
        pos_aug_pred= None

        # determine b,seq
        b,seq = self.determine_batch_and_seq_len(x['robot0_eye_in_hand_image'].shape)

        #change dimension
        for itms in x.keys():
            if "image" in itms or "img" in itms:
                x[itms]=x[itms].view(-1,3,self.img_size,self.img_size) if "depth" not in itms else x[itms].view(-1,1,self.img_size,self.img_size)

        #low_dim
        if self.input_low_dim!=0:
            x_low_dim = torch.tensor([])
            assert len(self.low_dim_keys) == 1
            for k in x:
                if k in self.low_dim_keys:
                    x_low_dim = torch.cat((x_low_dim, x[k].view(b * seq, -1)), dim=-1).contiguous()  # [b*seq,total_len]
            x_low_dim = x_low_dim / 360
            x_low_dim = x_low_dim.to(self.device)
            x_low_dim = self.mlp_pos(x_low_dim)  # [b*seq,ldhs]
        else:
            x_low_dim = None

        #imgs
        x=self.merging_depth(x)
        x = self.create_util_img_tensors(x)
        x=self.preprocess_imgs(x)
        x=self.img_branch(x,b=b,seq=seq) # sequentially:  id + "goal" + "aug" + "feat"

        # policy
        plc,plc_aug,x=self.determine_policy_inputs(x,x_low_dim)

        plc=plc.view(b,seq,-1).contiguous()
        if self.using_pos_estm:
            plc=self.pos_estm_layer(plc)
            pos_pred=plc.clone()
            plc=self.pos_estm_bottleneck(plc)

        if self.use_tcl_loss:
            plc_aug = plc_aug.view(b,seq,-1).contiguous()
            if self.using_pos_estm:
                plc_aug = self.pos_estm_layer(plc_aug)
                pos_aug_pred = plc_aug.clone()
                plc_aug = self.pos_estm_bottleneck(plc_aug)

        if not self.use_GMM:
            output=self.policy_mlp(plc)
            if self.output_activation is not None:
                output = self.output_activation(output)

            if self.use_tcl_loss:
                output_aug=self.policy_mlp(plc_aug)
                if self.output_activation is not None:
                    output_aug = self.output_activation(output_aug)

        else:
            x_means,x_scales,x_logits=self.gmm_policy_mlp(x=plc,b=b,seq=seq)
            dists = self.create_mixed_distribution(x_means, x_scales, x_logits,seq)
            output= dists.mean

            if self.use_tcl_loss:
                x_means_aug,x_scales_aug,x_logits_aug=self.gmm_policy_mlp(x=plc_aug,b=b,seq=seq)
                dists_aug = self.create_mixed_distribution(x_means_aug, x_scales_aug, x_logits_aug,seq)
                output_aug = dists_aug.mean

        if not self.use_tcl_loss:
            rtn_dict={"output_tensor": output, "pred_delta_pos": pos_pred}
        else:
            rtn_dict={"output_tensor": output, "output_tensor_aug": output_aug,"x_img_feat": x["x_0_feat"],
                    "x_img_goal_feat": x["x_0_goal_feat"], "x_img_aug_feat": x["x_0_aug_feat"],
                    "x_img_goal_aug_feat": x["x_0_goal_aug_feat"],"pred_delta_pos":pos_pred,"pred_delta_pos_aug":pos_aug_pred}

            if self.num_cameras == 2:
                rtn_dict["x_img_feat"] = torch.cat((rtn_dict["x_img_feat"],x["x_1_feat"]),dim=-1)
                rtn_dict["x_img_aug_feat"] = torch.cat((rtn_dict["x_img_aug_feat"],x["x_1_aug_feat"]),dim=-1)
                rtn_dict["x_img_goal_feat"] =torch.cat((rtn_dict["x_img_goal_feat"], x["x_1_goal_feat"]),dim=-1)
                rtn_dict["x_img_goal_aug_feat"] = torch.cat((rtn_dict["x_img_goal_aug_feat"],x["x_1_goal_aug_feat"]),dim=-1)
        return rtn_dict


