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
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from utils import utility

class MultiTaskTester(object):
    """
    Multiple Task Tester for our MIL work.
    Notes:
        1. (important) instance_pred is logits without sigmoid!!!
        2. memory_bank is now a list.
    
    """
    def __init__(self, net, test_data, memory_bank, logger=None, device=torch.device("cuda")):
        self.net = net
        self.test_data = test_data
        self.test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        self.memory_bank = memory_bank
        self.logger = logger
        self.device = device

    def inference(self):
        self.net.eval()
        with torch.no_grad():
            for batch_idx, (img, instance_labels, bag_idx, innner_idx, _) in enumerate(self.test_loader):
                instance_preds = self.net(img.to(self.device))
                for idx, mmbank in enumerate(self.memory_bank):
                    mmbank.update(bag_idx, innner_idx, instance_preds[idx].sigmoid())
        
        for idx, mmbank in enumerate(self.memory_bank):
            self.logger.save_result("test_mmbank_{}".format(idx), mmbank.state_dict())
    
    def evaluate(self):
        for idx, mmbank in enumerate(self.memory_bank):
            # read labels
            bag_labels = [x[idx] for x in self.test_data.bag_labels]
            # bag max evaluate
            bag_max_pred = mmbank.max_pool()
            self.cls_report(bag_max_pred, bag_labels, "t{}/bag_max".format(idx))
            # bag mean evaluate
            bag_mean_pred = mmbank.avg_pool(self.test_data.bag_lengths)
            self.cls_report(bag_mean_pred, bag_labels, "t{}/bag_avg".format(idx))

        #TODO: evalaute nodule_ratio list

    def cls_report(self, y_pred, y_true, prefix=""):
        """
        A combination of sklearn.metrics function and our logger (use tensorboard)

        """
        ##make hard prediction
        y_pred_hard = [(x > 0.5) for x in y_pred]
        cls_report = classification_report(y_true, y_pred_hard, output_dict=True)
        auc_score = roc_auc_score(y_true, y_pred)

        self.logger.log_scalar(prefix+'/'+'Accuracy', cls_report['accuracy'], print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1.0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1.0']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)