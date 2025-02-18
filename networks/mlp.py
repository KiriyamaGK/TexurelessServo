import collections
import random

import torch
import math
import numpy as np
import torch.nn as nn
from networks.Network import NetworkBase
from networks.helpers import get_activation_fn
from networks.SpatialSoftmax import SpatialSoftmax
from torchvision import models
from torch.nn import functional as F
import torch.distributions as D
import functools
import operator
from ultralytics import YOLO

from utils.input_process import add_gaussian_spot_to_image


class MLP(NetworkBase):
    def __init__(self, input_low_dim, output_dim,obs_keys,batch_size,seq_length,training,low_dim_hidden_sizes=None,hidden_sizes=None, activation="relu", output_activation=None,use_gmm=False,
                 encoder=None):
        super().__init__(input_low_dim, output_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.freeze_encoder=encoder['freeze']
        #img
        self.encoder_name = encoder['name']
        assert self.encoder_name in ['ResNet18', 'YOLO_v11']
        if self.encoder_name == 'YOLO_v11':
            self.pretrained_path = encoder['pretrained_path']
            self.img_enc_num_layers = encoder['params']['num_layers']
            yolo_mdl = YOLO(self.pretrained_path)
            self.img_enc = nn.Sequential(*list(yolo_mdl.model.model)[:self.img_enc_num_layers])
        elif self.encoder_name == 'ResNet18':
            self.img_enc = torch.nn.Sequential(*(list(models.resnet18().children())[:-2]))

        if self.freeze_encoder:
            for param in self.img_enc.parameters():
                param.requires_grad = False
        self.use_siamese = encoder['siamese']
        self.batch_size=batch_size
        self.seq_length=seq_length

        self.ss_num_kp=encoder['params']['SpatialSoftmax']['num_kp']
        self.ss_in_c=encoder['params']['SpatialSoftmax']['in_c']
        self.ss_in_h=encoder['params']['SpatialSoftmax']['in_h']
        self.ss_in_w=encoder['params']['SpatialSoftmax']['in_w']

        self.input_low_dim = input_low_dim
        self.output_dim = output_dim
        self.img_size=encoder['params']['img_size']
        self.crop_size=encoder['params']['crop_size']

        self.obs_keys = obs_keys
        self.low_dim_keys = []
        for obs_key in obs_keys:
            if "image" not in obs_key and "img" not in obs_key:
                self.low_dim_keys.append(obs_key)

        self.low_dim_hidden_sizes = low_dim_hidden_sizes
        self.hidden_sizes = hidden_sizes
        self.use_GMM = use_gmm
        self.is_training = training

        if self.is_training:
            self.use_data_augmentation=encoder['data_augmentation']
            self.use_tcl_loss=encoder['task_consistency_loss']
        else:
            self.use_data_augmentation=False
            self.use_tcl_loss=False
        if self.use_tcl_loss:
            assert self.use_data_augmentation

        self.activation = get_activation_fn(activation)
        self.output_activation = get_activation_fn(output_activation) if output_activation is not None else None

        self.spatial_softmax=SpatialSoftmax(self.ss_in_c,self.ss_in_h,self.ss_in_w,self.ss_num_kp)
        self.ee_ln = nn.Linear(self.ss_num_kp * 2, (self.hidden_sizes[0]-self.low_dim_hidden_sizes[-1])//2)
        self.grid_source = self.build_grid(self.crop_size, self.crop_size)

        if not self.use_siamese:
            if self.encoder_name == 'ResNet18':
                self.img_enc_goal = torch.nn.Sequential(*(list(models.resnet18().children())[:-2]))
            else:
                self.img_enc_goal = nn.Sequential(*list(yolo_mdl.model.model)[:self.img_enc_num_layers])
            self.spatial_softmax_goal = SpatialSoftmax(self.ss_in_c, self.ss_in_h, self.ss_in_w, self.ss_num_kp)
            self.ee_ln_goal = nn.Linear(self.ss_num_kp * 2, (self.hidden_sizes[0] - self.low_dim_hidden_sizes[-1]) // 2)
            if self.freeze_encoder:
                for param in self.img_enc_goal.parameters():
                    param.requires_grad = False

        #for low_dim
        self.mlp_pos = nn.Sequential(
            nn.Linear(self.input_low_dim, self.low_dim_hidden_sizes[0]),
            self.activation(),
            nn.Linear(self.low_dim_hidden_sizes[0], self.low_dim_hidden_sizes[1]),
        )
        self.buffer=[]

        if self.use_GMM:
            self.gmm_modes = 5
            self.mlp_decoder_mean = nn.Sequential(
            nn.Linear(self.hidden_sizes[0], hidden_sizes[1]),
            self.activation(),
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            self.activation(),
            nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3]),
            self.activation(),
            nn.Linear(self.hidden_sizes[3], self.output_dim* self.gmm_modes),
        )
            self.mlp_decoder_scale = nn.Sequential(
            nn.Linear(self.hidden_sizes[0], hidden_sizes[1]),
            self.activation(),
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            self.activation(),
            nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3]),
            self.activation(),
            nn.Linear(self.hidden_sizes[3], self.output_dim* self.gmm_modes),
        )
            self.mlp_decoder_logits = nn.Sequential(
            nn.Linear(self.hidden_sizes[0], hidden_sizes[1]),
            self.activation(),
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            self.activation(),
            nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3]),
            self.activation(),
            nn.Linear(self.hidden_sizes[3], self.gmm_modes),
        )
            self.min_std=0.0001
            self.activations = {
                "softplus": F.softplus,
                "exp": torch.exp,
            }
            self.std_activation = "softplus"
            self.low_noise_eval = False
        else:
            self.policy_mlp=nn.Sequential(
            nn.Linear(self.hidden_sizes[0], hidden_sizes[1]),
            self.activation(),
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            self.activation(),
            nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3]),
            self.activation(),
            nn.Linear(self.hidden_sizes[3], self.output_dim),
        )

    def random_crop_grid(self, x, grid):
        delta = x.size(2) - grid.size(1)
        grid = grid.repeat(x.size(0), 1, 1, 1).cuda()
        # Add random shifts by x
        grid[:, :, :, 0] = grid[:, :, :, 0] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(
            -1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
        # Add random shifts by y
        grid[:, :, :, 1] = grid[:, :, :, 1] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(
            -1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
        return grid

    def build_grid(self, source_size, target_size):
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-k, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full.cuda()

    def forward(self, x):
        x_img = x['robot0_eye_in_hand_image'].to(self.device)
        x_img_goal = x['robot0_eye_in_hand_image_goal'].to(self.device)

        # assert x_img.shape[-3:] == (3, self.img_size, self.img_size) #元组而不是列表
        if not x_img.shape[-3:] == (3, self.img_size, self.img_size):
            print(x_img.shape)
        if len(x_img.shape) == 3:
            b, seq = 1, 1
            x_img = x_img.view(1, 3, self.img_size, self.img_size)  # [b,c,h,w]
            x_img_goal = x_img_goal.view(1, 3, self.img_size, self.img_size)
        elif len(x_img.shape) == 4:
            assert self.batch_size >= x_img.shape[0]
            b, seq = x_img.shape[0], 1
        elif len(x_img.shape) == 5:
            assert self.batch_size >= x_img.shape[0] and self.seq_length == x_img.shape[1]
            b, seq = x_img.shape[0], x_img.shape[1]
        else:
            raise RuntimeError('x_img.shape should between 3 and 5')

        x_low_dim = torch.tensor([])
        assert len(self.low_dim_keys) == 1
        for k in x:
            if k in self.low_dim_keys:
                x_low_dim = torch.cat((x_low_dim, x[k].view(b * seq, -1)), dim=-1).contiguous()  # [b*seq,total_len]
        x_low_dim = x_low_dim / 360
        x_low_dim = x_low_dim.to(self.device)

        x_img = x_img.view(b * seq, 3, self.img_size, self.img_size)
        if self.crop_size < self.img_size:
            x_img_grid_shifted = self.random_crop_grid(x_img, self.grid_source)
            x_img = F.grid_sample(x_img, x_img_grid_shifted, align_corners=True)

        if self.use_tcl_loss:
            x_img_aug = x['robot0_eye_in_hand_image_light'].to(self.device)
            x_img_aug = x_img_aug.view(b * seq, 3, self.img_size, self.img_size)
            if self.crop_size < self.img_size:
                x_img_aug_grid_shifted = self.random_crop_grid(x_img_aug, self.grid_source)
                x_img_aug = F.grid_sample(x_img_aug, x_img_aug_grid_shifted, align_corners=True)
            for idx in range(x_img.shape[0]):
                x0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                y0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                x_img_aug[idx] = add_gaussian_spot_to_image(x_img_aug[idx], size=50, sigma=10, position=(x0, y0),
                                                            to_device=True)
        else:
            if self.use_data_augmentation:
                for idx in range(x_img.shape[0]):
                    x0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    y0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    x_img[idx] = add_gaussian_spot_to_image(x_img[idx], size=50, sigma=10, position=(x0, y0),
                                                            to_device=True)

        x_img = self.img_enc(x_img)
        if self.use_tcl_loss:
            x_img_feat = x_img.clone()
        x_img = self.spatial_softmax(x_img)
        x_img = self.ee_ln(x_img)
        x_img = x_img.view(b * seq, -1).contiguous()

        if self.use_tcl_loss:
            x_img_aug = self.img_enc(x_img_aug)
            if self.use_tcl_loss:
                x_img_aug_feat = x_img_aug.clone()
            x_img_aug = self.spatial_softmax(x_img_aug)
            x_img_aug = self.ee_ln(x_img_aug)
            x_img_aug = x_img_aug.view(b * seq, -1).contiguous()

        x_img_goal = x_img_goal.view(b * seq, 3, self.img_size, self.img_size)
        if self.crop_size < self.img_size:
            x_img_goal_grid_shifted = self.random_crop_grid(x_img_goal, self.grid_source)
            x_img_goal = F.grid_sample(x_img_goal, x_img_goal_grid_shifted, align_corners=True)
        if self.use_tcl_loss:
            x_img_goal_aug = x['robot0_eye_in_hand_image_light_goal'].to(self.device)
            x_img_goal_aug = x_img_goal_aug.view(b * seq, 3, self.img_size, self.img_size)
            if self.crop_size < self.img_size:
                x_img_goal_aug_grid_shifted = self.random_crop_grid(x_img_goal_aug, self.grid_source)
                x_img_goal_aug = F.grid_sample(x_img_goal_aug, x_img_goal_aug_grid_shifted, align_corners=True)

            for idx in range(x_img_goal.shape[0]):
                x0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                y0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                x_img_goal_aug[idx] = add_gaussian_spot_to_image(x_img_goal_aug[idx], size=50, sigma=10,
                                                                 position=(x0, y0), to_device=True)

        else:
            if self.use_data_augmentation:
                for idx in range(x_img.shape[0]):
                    x0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    y0 = random.randint(0, min(self.img_size, self.crop_size) - 1)
                    x_img_goal[idx] = add_gaussian_spot_to_image(x_img_goal[idx], size=50, sigma=10, position=(x0, y0),
                                                                 to_device=True)

        if not self.use_siamese:
            x_img_goal = self.img_enc_goal(x_img_goal)
            if self.use_tcl_loss:
                x_img_goal_feat = x_img_goal.clone()
            x_img_goal = self.spatial_softmax_goal(x_img_goal)
            x_img_goal = self.ee_ln_goal(x_img_goal)
            x_img_goal = x_img_goal.view(b * seq, -1).contiguous()
            if self.use_tcl_loss:
                x_img_goal_aug = self.img_enc_goal(x_img_goal_aug)
                if self.use_tcl_loss:
                    x_img_goal_aug_feat = x_img_goal_aug.clone()
                x_img_goal_aug = self.spatial_softmax_goal(x_img_goal_aug)
                x_img_goal_aug = self.ee_ln_goal(x_img_goal_aug)
                x_img_goal_aug = x_img_goal_aug.view(b * seq, -1).contiguous()
        else:
            x_img_goal = self.img_enc(x_img_goal)
            if self.use_tcl_loss:
                x_img_goal_feat = x_img_goal.clone()
            x_img_goal = self.spatial_softmax(x_img_goal)
            x_img_goal = self.ee_ln(x_img_goal)
            x_img_goal = x_img_goal.view(b * seq, -1).contiguous()
            if self.use_tcl_loss:
                x_img_goal_aug = self.img_enc(x_img_goal_aug)
                if self.use_tcl_loss:
                    x_img_goal_aug_feat = x_img_goal_aug.clone()
                x_img_goal_aug = self.spatial_softmax(x_img_goal_aug)
                x_img_goal_aug = self.ee_ln(x_img_goal_aug)
                x_img_goal_aug = x_img_goal_aug.view(b * seq, -1).contiguous()

        x_low_dim=self.mlp_pos(x_low_dim) #[b*seq,ldhs]
        x_policy=torch.cat((x_img, x_img_goal,x_low_dim), dim=-1).contiguous()  #[b*seq,hs]
        x_policy=x_policy.view(b,seq,-1).contiguous()
        if self.use_tcl_loss:
            x_policy_aug = torch.cat((x_img_aug, x_img_goal_aug, x_low_dim), dim=-1).contiguous()  # [b*seq,hs]
            x_policy_aug = x_policy_aug.view(b, seq, -1).contiguous()

        #policy
        if not self.use_GMM:
            output=self.policy_mlp(x_policy)
            if self.output_activation is not None:
                output = self.output_activation(output)
            if self.use_tcl_loss:
                output_aug=self.policy_mlp(x_policy_aug)
                if self.output_activation is not None:
                    output_aug = self.output_activation(output_aug)
        else:
            x_means = self.mlp_decoder_mean(x_policy)
            x_scales = self.mlp_decoder_scale(x_policy)
            x_logits = self.mlp_decoder_logits(x_policy)

            x_means = x_means.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
            x_scales = x_scales.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
            x_logits = x_logits.view(b, seq, self.gmm_modes).contiguous()

            if self.low_noise_eval and seq == 1:
                # low-noise for all Gaussian dists
                x_scales = torch.ones_like(x_means) * 1e-4
            else:
                # post-process the scale accordingly
                x_scales = self.activations[self.std_activation](x_scales) + self.min_std

            component_distribution = D.Normal(loc=x_means, scale=x_scales)
            component_distribution = D.Independent(component_distribution, 1)  # shift action dim to event shape

            # unnormalized logits to categorical distribution for mixing the modes
            mixture_distribution = D.Categorical(logits=x_logits)

            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )
            output= dists.mean

            if self.use_tcl_loss:
                x_means_aug = self.mlp_decoder_mean(x_policy_aug)
                x_scales_aug = self.mlp_decoder_scale(x_policy_aug)
                x_logits_aug = self.mlp_decoder_logits(x_policy_aug)

                x_means_aug = x_means_aug.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
                x_scales_aug = x_scales_aug.view(b, seq, self.gmm_modes, self.output_dim).contiguous()
                x_logits_aug = x_logits_aug.view(b, seq, self.gmm_modes).contiguous()

                if self.low_noise_eval and seq == 1:
                    # low-noise for all Gaussian dists
                    x_scales_aug = torch.ones_like(x_means_aug) * 1e-4
                else:
                    # post-process the scale accordingly
                    x_scales_aug = self.activations[self.std_activation](x_scales_aug) + self.min_std

                component_distribution_aug = D.Normal(loc=x_means_aug, scale=x_scales_aug)
                component_distribution_aug = D.Independent(component_distribution_aug,
                                                           1)  # shift action dim to event shape

                # unnormalized logits to categorical distribution for mixing the modes
                mixture_distribution_aug = D.Categorical(logits=x_logits_aug)

                dists_aug = D.MixtureSameFamily(
                    mixture_distribution=mixture_distribution_aug,
                    component_distribution=component_distribution_aug,
                )
                output_aug = dists_aug.mean

        if not self.use_tcl_loss:
            return output
        else:
            return {"output_tensor": output, "output_tensor_aug": output_aug, "x_img_feat": x_img_feat,
                    "x_img_goal_feat": x_img_goal_feat, "x_img_aug_feat": x_img_aug_feat,
                    "x_img_goal_aug_feat": x_img_goal_aug_feat}



