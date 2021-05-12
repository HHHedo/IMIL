#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os,sys
sys.path.append('../')
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

class MultiTaskTrainer(object):
    """
    A multiple task trainer. Used in our MIL project to do
    gene classification | nodule classification two tasks.

    Notes:
        1. Currently I just re-define the train() and the interface of train loader(*).
         How to better design it? (Maybe I'll make all these into dictionary, somehow like mmdetection)

        2. memory bank is designed as list now. criterion() should be wrapped up.

        3. criterion should be re-designed.
    """
    def __init__(self,
                 net,
                 optimizer,
                 lrsch,
                 inner_loss,
                 inter_loss,
                 train_loader,
                 memory_bank,
                 save_interval=10,
                 logger=None,
                 configs=None,
                 device=torch.device("cuda"),
                 inner_weights=[1.0, 1.0],
                 inter_weights=0.0):
        self.net = net
        self.optimizer = optimizer
        self.lrsch = lrsch
        self.device = device
        self.inner_loss = inner_loss
        self.inter_loss = inter_loss
        self.train_loader = train_loader
        self.memory_bank = memory_bank
        self.save_interval = save_interval
        self.logger = logger
        self.configs = configs
        self.device = device
        self.inner_weights = inner_weights
        self.inter_weights = inter_weights
    
    def train(self):
        self.net.train()
        self.logger.update_step()
        show_losses = {"loss_total":[], "loss_inner":[], "loss_inter":[]}
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios) in enumerate(tqdm(self.train_loader, ascii=True, ncols=60)):
            self.logger.update_iter()
            instance_preds = self.net(imgs.to(self.device))
            instance_labels = instance_labels.to(self.device)
            ## update memory bank
            for idx, mmbank in enumerate(self.memory_bank):
                mmbank.update(bag_index, inner_index, instance_preds[idx].sigmoid())
            ##instancee-level prediction loss
            loss_total, loss_inner, loss_inter = self.optimize(bag_index, inner_index, instance_preds, instance_labels, nodule_ratios)
            show_losses["loss_total"].append(loss_total)
            show_losses["loss_inner"].append(loss_inner)
            show_losses["loss_inter"].append(loss_inter)
        
        
        for k, v in show_losses.items():
            avg_loss = sum(v) / (len(v))
            self.logger.log_scalar(k, avg_loss, print=True)
        self.logger.clear_inner_iter()
        if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
            self.lrsch.step(avg_loss)
        else:
            self.lrsch.step()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.net, self.optimizer, self.lrsch, self.inter_loss)
        
        for idx, mmbank in enumerate(self.memory_bank):
            self.logger.save_result("train_mmbank_{}".format(idx), mmbank.state_dict())
    
    def optimize(self, bag_index, inner_index, instance_preds, instance_labels, nodule_ratios):
        """
        I wrap up the optimization process in the training into this function.
        Currently I use self.configs to confirm whether to use RCE loss
        """
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0).to(self.device)
        for idx, pred in enumerate(instance_preds):
            if self.configs is None: #both use CE loss
                inner_loss = self.inner_loss[idx](pred, instance_labels[:, idx].view(-1,1))
            elif self.configs > 0: #single branch use RCE loss
                inner_loss = self.inner_loss[idx](pred, instance_labels[:, idx].view(-1,1),
                                self.memory_bank[idx].get_weight(bag_index, inner_index, nodule_ratios).view(-1,1).to(self.device))
            else:
                inner_loss = self.inner_loss[idx](pred, instance_labels[:, idx].view(-1,1))
            loss += inner_loss * self.inner_weights[idx]


        inter_loss = self.inter_loss(bag_index, inner_index, instance_preds, 
                                        instance_labels, nodule_ratios, self.memory_bank)

        loss_total = loss + inter_loss * self.inter_weights
        loss_total.backward()
        self.optimizer.step()
        return loss_total.item(), loss.item(), inter_loss.item()


            
            