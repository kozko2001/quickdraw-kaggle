# -*- coding: utf-8 -*-
from __future__ import division
from torch.autograd import Variable
import torch
from torchvision.models import resnet
from torch import nn

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()

        conf = int(config["model"]["conf"])
        num_classes = int(config["num_classes"])

        if conf == 18:
            self.model = resnet.resnet18(num_classes = num_classes)
        elif conf == 50:
            self.model = resnet.resnet50(num_classes = num_classes)
        elif conf == 34:
            self.model = resnet.resnet34(num_classes = num_classes)
        elif conf == 101:
            self.model = resnet.resnet101(num_classes = num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    config = {
        "model": {
            "conf": "18"
            },
        "num_classes": 340
        }
    net = ResNet(config).cuda()
    i = Variable(torch.randn(6, 3, 224, 224)).cuda()
    print(i.shape)
    y = net(i)
    print(y)
    print(y.shape)
