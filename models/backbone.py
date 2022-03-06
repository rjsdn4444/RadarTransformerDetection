from typing import Type, Any, Callable, Union, List, Optional
import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionEmbeddingSine(nn.Module):
    def __init__(self, hidden_dim, max_length, normalize, scale):
        super(PositionEmbeddingSine, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        batch_size, antenna_combination, sampling_size = x.shape
        sequence = torch.ones((batch_size, antenna_combination)).to(x.device)
        seq_embed = sequence.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            seq_embed = seq_embed / (seq_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=x.device)
        dim_t = self.max_length ** (2 * (dim_t // 2) / self.hidden_dim)

        position = seq_embed[:, :, None] / dim_t
        position = torch.stack((position[:, :, 0::2].sin(), position[:, :, 1::2].cos()), dim=3).flatten(2)
        return position


class CNN_layer(nn.Module):
    def __init__(self):
        super(CNN_layer, self).__init__()
        self.out_channel = 256
        self.conv0 = nn.Conv2d(1, 16, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, self.out_channel, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm5 = nn.BatchNorm2d(self.out_channel)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch_norm0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        # x = self.conv5(x)
        # x = self.batch_norm5(x)
        # x = self.relu5(x)
        return x
