import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
    """

    def __init__(self, num_classes, drop_rate):

        super(FCNet, self).__init__()
        self.num_layer = 2
        self.fc1 = nn.Linear(2097152, 100)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)
        self.drop_rate = drop_rate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.input_imgs = x.detach().cpu().numpy()
        x_flat = x.view(x.size(0), -1)
        h = self.fc1(self.ReLU(x_flat))
        return self.sigmoid(self.fc2(h))

    def get_current_visuals(self):
        return self.input_imgs
