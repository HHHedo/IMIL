#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from .backbones.ResNet import ResNet10, ResNet18, ResNet34, ResNet50

class AttentionClsNet(nn.Module):
    '''
    Basic classification network. With single classification head and single backbone.

    '''
    def __init__(self, backbone, num_classes=2, attention_m=128, attention_o=1):
        super(AttentionClsNet, self).__init__()
        # self.backbone = backbone
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.L = backbone.feat_dim
        self.D = attention_m
        self.K = attention_o

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Linear(self.L, num_classes)

        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # if isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0.0, std=0.01)
            #     m.bias.data.zero_()

    def forward(self, x):
        # x = self.backbone(x)
        # x = self.avgpool(x)
        # x = x.view(x.shape[0], -1) #N*512
        A = self.attention(x)  # Nx1
        A = torch.transpose(A, 1, 0)  # 1xN
        A = F.softmax(A, dim=1)  # softmax over N
        x = torch.mm(A, x)  # KxL
        x = self.classifier(x)
        return x


class GAttentionClsNet(nn.Module):
    '''
    Basic classification network. With single classification head and single backbone.

    '''
    def __init__(self, backbone, num_classes=2, attention_m=128, attention_o=1):
        super(GAttentionClsNet, self).__init__()
        # self.backbone = backbone
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.L = backbone.feat_dim
        self.D = attention_m
        self.K = attention_o

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L, num_classes)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):


        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        x = torch.mm(A, x)  # KxL
        x = self.classifier(x)
        return x

if __name__=='__main__':
    backbone = Dense_small()
    net = ClsNet(backbone, 2).cuda()
    print(net)
    with torch.no_grad():
        x = torch.randn([32,3,224,224]).cuda()
        print(net(x))
        print('ok')