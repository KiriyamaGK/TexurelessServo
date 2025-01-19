import torch
import numpy as np
from torch.nn import functional as F

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = torch.nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
                                      torch.from_numpy(np.linspace(-1, 1, in_h)).float())

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert(x.shape[1] == self._in_c)
        assert(x.shape[2] == self._in_h)
        assert(x.shape[3] == self._in_w)


        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.view(-1, self._in_h * self._in_w)
        attention = F.softmax(h, dim=-1)

        keypoint_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True).view(-1, self._num_kp)
        keypoint_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True).view(-1, self._num_kp)

        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints