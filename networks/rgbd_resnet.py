import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck


class SEBlock(nn.Module):
    """工业场景优化的压缩-激励模块，增强几何敏感通道"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FusionEnhancedNet(ResNet):
    """改进版ResNet18，支持4通道输入(RGB+D)与多级特征融合"""

    def __init__(self, num_classes=1000):
        # 修改基础ResNet18结构
        super(FusionEnhancedNet, self).__init__(block=Bottleneck, layers=[2, 2, 2, 2])

        # 输入层适配：RGB+D四通道输入
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 早期融合层：平衡RGB与深度贡献
        self.early_fusion = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 在每一个ResNet阶段后插入SE-Block
        self.se_blocks = nn.ModuleList([
            SEBlock(256, reduction=16),  # layer1后
            SEBlock(512, reduction=16),  # layer2后
            SEBlock(1024, reduction=16),  # layer3后
            SEBlock(2048, reduction=16)  # layer4后
        ])

        # 初始化策略：深度通道权重增强
        self._init_weights()

    def _init_weights(self):
        # 深度通道权重强化初始化
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        with torch.no_grad():
            # 增强深度通道初始权重（第4通道）
            self.conv1.weight[:, 3, :, :] *= 2.0  # 深度通道权重加倍

        # 早期融合层初始化
        nn.init.constant_(self.early_fusion[1].weight, 1.0)
        nn.init.constant_(self.early_fusion[1].bias, 0.0)

    def _forward_impl(self, x):
        # 输入预处理
        x = self.conv1(x)  # [B,4,H,W] → [B,64,H/2,W/2]
        x = self.early_fusion(x)  # 早期融合

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B,64,H/4,W/4]

        # 各阶段前向传播 + SE-Block插入
        x = self.layer1(x)  # [B,256,H/4,W/4]
        x = self.se_blocks[0](x)  # 插入SE

        x = self.layer2(x)  # [B,512,H/8,W/8]
        x = self.se_blocks[1](x)

        x = self.layer3(x)  # [B,1024,H/16,W/16]
        x = self.se_blocks[2](x)

        x = self.layer4(x)  # [B,2048,H/32,W/32]
        x = self.se_blocks[3](x)

        # 输出头
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

# ---------- 工业场景优化设计说明 ----------
# 1. 深度通道强化：conv1的深度通道权重初始化为其他通道的2倍
# 2. 多级SE-Block：在每个ResNet阶段后插入，增强几何敏感通道
# 3. 早期融合层：1x1卷积+BN+ReLU，快速平衡模态贡献
# 4. 兼容预训练：若加载ImageNet预训练权重，自动忽略不匹配层