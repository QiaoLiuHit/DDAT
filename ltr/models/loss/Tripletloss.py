import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
import numpy as np
import time
import datetime
import copy
from torch.autograd import Function

import warnings
warnings.filterwarnings("ignore")

class TripletLoss(nn.Module):
    def __init__(self, margin=100.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.avg_hd = AveragedHausdorffLoss()

    def forward(self, source_feat, target_feat, activation_rates):
        """
        Triplet Loss function

        """
        distance_positive = self.avg_hd.forward(source_feat, target_feat, activation_rates)
        distance_negative = self.avg_hd.forward(source_feat, target_feat, 1.0 - activation_rates)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2, activation_rates):
        """
        Average Hausdorff Loss computation

        """
        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.' \
            % (set2.size()[1], set2.size()[1])

        d2_matrix = torch.cdist(set1, set2) * activation_rates

        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])
        res = term_1 + term_2

        return res
