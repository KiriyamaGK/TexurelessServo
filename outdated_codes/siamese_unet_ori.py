# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F

class SiamUnet(nn.Module):

    def __init__(self, in_channels=3, out_channels_s=1, init_features=16):
        super(SiamUnet, self).__init__()
        #in、out:(256,256)
        features = init_features
        
        # UNet layers
        self.encoder1 = SiamUnet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = SiamUnet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = SiamUnet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = SiamUnet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = SiamUnet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = SiamUnet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = SiamUnet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = SiamUnet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = SiamUnet._block(features * 2, features, name="dec1")

        self.conv_s = nn.Conv2d(in_channels=features, out_channels=out_channels_s, kernel_size=1)

        self.grid_source = self.build_grid(256, 256)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def build_grid(self, source_size, target_size):
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-k, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full.cuda()
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

    def forward(self, x):
        # UNet on x1
        x1=x['robot0_eye_in_hand_image'].to(self.device)
        x1=x1.view(-1,3,280,280)
        x1_grid_shifted= self.random_crop_grid(x1, self.grid_source)
        x1=F.grid_sample(x1, x1_grid_shifted, align_corners=True)
        x2=x['robot0_eye_in_hand_image_goal'].to(self.device)
        x2=x2.view(-1,3,280,280)
        x2_grid_shifted = self.random_crop_grid(x2, self.grid_source)
        x2=F.grid_sample(x2, x2_grid_shifted, align_corners=True)

        x3 = x['gaussian_img'].to(self.device)
        x3 = x3.view(-1, 1, 280, 280)
        x3_grid_shifted = self.random_crop_grid(x3, self.grid_source)
        x3 = F.grid_sample(x3, x3_grid_shifted, align_corners=True)
        x4 = x['gaussian_img_goal'].to(self.device)
        x4 = x4.view(-1, 1, 280, 280)
        x4_grid_shifted = self.random_crop_grid(x4, self.grid_source)
        x4 = F.grid_sample(x4, x4_grid_shifted, align_corners=True)

        enc1_1 = self.encoder1(x1)
        enc2_1 = self.encoder2(self.pool1(enc1_1))
        enc3_1 = self.encoder3(self.pool2(enc2_1))
        enc4_1 = self.encoder4(self.pool3(enc3_1))

        bottleneck_1 = self.bottleneck(self.pool4(enc4_1))

        dec4_1 = self.upconv4(bottleneck_1)
        dec4_1 = torch.cat((dec4_1, enc4_1), dim=1)
        dec4_1 = self.decoder4(dec4_1)
        dec3_1 = self.upconv3(dec4_1)
        dec3_1 = torch.cat((dec3_1, enc3_1), dim=1)
        dec3_1 = self.decoder3(dec3_1)
        dec2_1 = self.upconv2(dec3_1)
        dec2_1 = torch.cat((dec2_1, enc2_1), dim=1)
        dec2_1 = self.decoder2(dec2_1)
        dec1_1 = self.upconv1(dec2_1)
        dec1_1 = torch.cat((dec1_1, enc1_1), dim=1)
        dec1_1 = self.decoder1(dec1_1)
        
        # UNet on x2
        enc1_2 = self.encoder1(x2)
        enc2_2 = self.encoder2(self.pool1(enc1_2))
        enc3_2 = self.encoder3(self.pool2(enc2_2))
        enc4_2 = self.encoder4(self.pool3(enc3_2))

        bottleneck_2 = self.bottleneck(self.pool4(enc4_2))

        dec4_2 = self.upconv4(bottleneck_2)
        dec4_2 = torch.cat((dec4_2, enc4_2), dim=1)
        dec4_2 = self.decoder4(dec4_2)
        dec3_2 = self.upconv3(dec4_2)
        dec3_2 = torch.cat((dec3_2, enc3_2), dim=1)
        dec3_2 = self.decoder3(dec3_2)
        dec2_2 = self.upconv2(dec3_2)
        dec2_2 = torch.cat((dec2_2, enc2_2), dim=1)
        dec2_2 = self.decoder2(dec2_2)
        dec1_2 = self.upconv1(dec2_2)
        dec1_2 = torch.cat((dec1_2, enc1_2), dim=1)
        dec1_2 = self.decoder1(dec1_2)

        out_1=self.conv_s(dec1_1)
        out_2=self.conv_s(dec1_2)

        return out_1, out_2,x3,x4

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )