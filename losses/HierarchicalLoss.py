#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss for our MIL work.
    """
    def __init__(self):
        super(HierarchicalLoss, self).__init__()
        self.soft_max = nn.Softmax(dim=1)
        
    def forward(self, bag_index, inner_index, preds, labels, pos_ratios=None, mmbanks=None):
        """
        (*) preds[0] is hard task, preds[1] is easy task.
        
        Notes:
            1. Design rule:
                * negative preds in easy task should not be
                positive in hard task.
            
            2. Default assume that preds are not in sigmoid or softmax format

            3. (*) in-place operation is taken on preds. We will convert
                preds into probability of positive prediction first.
        Args:
            bag_index: (list of int)
            inner_index: (list of int)
            preds: (list of tensor or tensor[2,N,C])
            labels: (list of tensor | tensor)
            pos_ratios: (list of int)
            mmbanks: (list of container)
        """
        #make it probability of pos prediction
        for idx in range(len(preds)):
            if preds[idx].dim() == 1:
                preds[idx] = preds[idx].sigmoid()
            elif preds[idx].dim() == 2:
                if preds[idx].shape[-1] <= 1:
                    preds[idx] = preds[idx].sigmoid().view(-1)
                elif preds[idx].shape[-1] == 2: ##only task pos prediction prob
                    preds[idx] = self.soft_max(preds[idx])[:, 1]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        #could be modified
        pred_easy = preds[1]
        pred_hard = preds[0]
        differ_idx = (pred_easy < 0.5) & (pred_hard >  0.5)
        if not sum(differ_idx):
            return torch.tensor(0.0).cuda()
        else:
            loss = self.loss_on_diff(pred_easy[differ_idx], pred_hard[differ_idx])
            return loss

    @staticmethod
    def loss_on_diff(pred_easy, pred_hard):
        """
        Currently the design is:
        1. Use BCE loss (hard pooling pred_hard > 0.5) (since pred easy
            as target, no gradient flows to easy branch)
        """
        return F.binary_cross_entropy(pred_hard, (pred_easy>0.5).float())
        
        


        

        

        