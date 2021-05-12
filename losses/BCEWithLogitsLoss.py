#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
import torch
import torch.nn.functional as F

class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """
    This loss formulation is exactly the same as torch.nn.BCEWithLogitsLoss.
    The forward() is modified,  weight can be passed.
    This class can be used to implement Rectified Cross Entropy Loss.
    """
    __constants__ = ['weight', 'pos_weight', 'reduction']
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        # self.register_buffer('weight', weight)
        # self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target, weight=None, pos_weight=None):
        if weight is None:
            pass
        else:
            self.weight = weight
        # print(self.pos_weight)
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)

if __name__=="__main__":
    pass
            