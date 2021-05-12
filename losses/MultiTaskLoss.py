#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BCEWithLogitsLoss import BCEWithLogitsLoss
from .HierarchicalLoss import HierarchicalLoss

class MultiTaskLoss(nn.Module):
    """
    MultiTask Loss for our MIL learning. A multi task loss
    is actually made up with inner loss and inter loss, where
    inner loss is for each branch and inter loss needs interaction
    between branchs.

    Args:
        inner_loss_cfg: (list of str) 
        inter_loss_cfg: (str)
        inner_weights: (list of float): weights for each inner loss
        inter_weights: (float): weight for inter loss.

    
    Notes:
        1. You might have multiple inter loss. Well, it is easy to 
            extend this code.
        
        2. All loss type is passed as string. Why? Because then you just
        need to modify the build function, which could be easily backward compatiable.

        3. `inter_loss` could be None. 
    """
    def __init__(self, inner_loss_cfg=["CE", "CE"], inter_loss_cfg="Hierarchical",
                inner_weights=[1.0, 1.0], inter_weights=10.0):
        self.inner_weights = inner_weights
        self.inter_weights = inter_weights
        self.inner_loss_cfg = inner_loss_cfg
        self.inter_loss_cfg = inter_loss_cfg
        self.inner_loss = self.buildInnerLoss(inner_loss_cfg)
        self.inter_loss = self.buildInterLoss(inter_loss_cfg)

    def buildInnerLoss(self, args):
        """
        You can modify this to add different loss.
        """
        losses = []
        for arg in args:
            if "CE" in arg:
                losses.append(BCEWithLogitsLoss())
            else:
                losses.append(BCEWithLogitsLoss())
        return losses

    def buildInterLoss(self, args):
        """
        You can modify this to add different loss.
        """
        if "Hierarchical" in args:
            return HierarchicalLoss()
        else:
            return None
        
    def forward(self, preds, labels, mmbanks=None):
        """
        Args:
            preds: (list of tensor)
            labels: (list of tensor | tensor)
            mmbanks: (list of container)
        """
        loss = 0
        for idx, pred in enumerate(preds):
            if "RCE" in self.inner_loss_cfg[idx]:
                weight = self.mmbanks[idx]
            else:
                loss += self.inner_weights[idx] * self.inner_loss[idx](pred, labels[idx])
        
        loss += self.inter_weights * self.inter_loss(preds, labels, mmbanks)