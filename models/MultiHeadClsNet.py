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

class MultiHeadClsNet(nn.Module):
    '''
    Classification network with a shared backbone (feature extractor) 
    and multiple classification head.
    
    '''
    def __init__(self, backbone, num_classes=2, num_heads=2):
        super(MultiHeadClsNet, self).__init__()
        if isinstance(num_classes, int):
            self.num_classes = [num_classes] * num_heads
        self.num_heads = num_heads
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.ModuleList([nn.Linear(self.backbone.feat_dim, num_classes[i]) for i in range(num_heads)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        cls_results = []
        for idx in range(self.num_heads):
            cls_results.append(self.classifier[idx](x.view(x.shape[0],-1)))

        return cls_results