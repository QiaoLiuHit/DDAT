import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from ltr.models.backbone import GRL


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            GRL.GradientReversal(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to('cuda')

    def forward(self, x):
        #print('输入x维度={}'.format(x.shape))
        x = x.view(-1, 256)
        #print('进入对抗维度x={}'.format(x.shape))
        x = self.discriminator(x)
        return x

# class Discriminator2(torch.nn.Module):
#     def __init__(self):
#         super(Discriminator2, self).__init__()
#         self.discriminator = nn.Sequential(
#             GRL.GradientReversal(),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         ).to('cuda')
#
#     def forward(self, x):
#         x = x.view(-1, 256)
#         x = self.discriminator(x)
#         return x

