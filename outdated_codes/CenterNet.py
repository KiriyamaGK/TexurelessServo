import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CenterNet_ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(CenterNet_ResNet18, self).__init__()
        # 加载预训练的 ResNet18
        self.backbone = resnet18(pretrained=True)

        # 修改 ResNet18 的第一层，支持更高分辨率的输入
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 添加上采样层，将特征图分辨率提高到输入的 1/4
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 输出头：预测中心点热图、偏移量和物体尺寸
        self.head_heatmap = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.grid_source = self.build_grid(280, 256)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        x1 = x['robot0_eye_in_hand_image'].to(self.device)
        x1 = x1.view(-1, 3, 280, 280)
        x1_grid_shifted = self.random_crop_grid(x1, self.grid_source)
        x1 = F.grid_sample(x1, x1_grid_shifted, align_corners=True)

        x2 = x['gaussian_img_ct'].to(self.device)
        x2 = x2.view(-1, 1, 280, 280)
        x2_grid_shifted = self.random_crop_grid(x2, self.grid_source)
        x2 = F.grid_sample(x2, x2_grid_shifted, align_corners=True)
        x2_max = x2.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x2_mask = x2_max == 0
        x2_max  = x2_max + x2_mask.float()
        x2 = x2 / x2_max
        new_x2_max = x2.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        assert torch.all((new_x2_max == 0) | (new_x2_max == 1))
        # xx=x2[0][0].detach().cpu().numpy()

        x3 = x['gaussian_img_kpt'].to(self.device)
        x3 = x3.view(-1, 1, 280, 280)
        x3_grid_shifted = self.random_crop_grid(x3, self.grid_source)
        x3 = F.grid_sample(x3, x3_grid_shifted, align_corners=True)
        x3_max = x3.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x3_mask = x3_max == 0
        x3_max = x3_max + x3_mask.float()
        x3 = x3 / x3_max
        new_x3_max = x3.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        assert torch.all((new_x3_max == 0) | (new_x3_max == 1))

        x1 = self.backbone.conv1(x1)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x1 = self.backbone.layer1(x1)
        x1 = self.backbone.layer2(x1)
        x1 = self.backbone.layer3(x1)
        x1 = self.backbone.layer4(x1)

        x1 = self.deconv_layers(x1)

        heatmap = self.head_heatmap(x1)
        heatmap = F.interpolate(heatmap, size=(256, 256), mode='bilinear', align_corners=True)
        heatmap = heatmap.view(-1, 2, 256, 256)

        combined=torch.concatenate((x2, x3), dim=1)
        combined = F.interpolate(combined, size=(256, 256), mode='bilinear', align_corners=True)
        return heatmap , combined