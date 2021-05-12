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
from data.DigestSegBag import DigestSegIns
from sklearn.metrics import confusion_matrix
class BaseTester(object):
    """
    Basic Tester for our MIL work.
    Core methods include:
    1. inference():
        Read batch of instances (not necessary from a bag) and do instance-level prediction;
        Save the predicted result into memory bank momentarily.
    2. evaluate():
        Do evaluation based on memory bank. Currently includes:
            a. bag-level classification report (on bag_max/ bag_mean method)
            b. instance-level auc.
            c. (*) nodule ratio mae distance?
    Notes:
        1. (important) instance_pred is logits without sigmoid!!!
    
    """
    def __init__(self, backbone, clsnet, test_loader, test_data, loader_list,
                 memory_bank, logger=None, old_backbone=None, bs=64, device=torch.device("cuda")):
        self.backbone = backbone
        self.clsnet = clsnet
        self.test_data = test_data
        self.test_loader = test_loader
        self.loader_list = loader_list
        self.memory_bank = memory_bank
        self.logger = logger
        if old_backbone is not None:
            self.old_backbone = old_backbone.eval()
        self.bs = bs
        self.device = device

    def inference(self, bs):

        self.backbone.eval()
        self.clsnet.eval()
        with torch.no_grad():
            for batch_idx, (img, instance_label, bag_idx, inner_idx, _, _) in enumerate(tqdm(self.test_loader, ascii=True, ncols=60)):
                instance_preds = self.clsnet(self.backbone(img.to(self.device)), bag_idx, inner_idx, self.old_backbone, img.to(self.device))
                self.memory_bank.update(bag_idx, inner_idx, instance_preds.sigmoid())
            self.logger.save_result(
            "test_mmbank", self.memory_bank.state_dict())

    def inference_twostage(self):
        self.backbone.eval()
        self.clsnet.eval()
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for (bag_labels, bag_dataloader) in tqdm(self.loader_list, ascii=True, ncols=60):
                bag_labels = bag_labels.to(self.device)
                instance_emb = []
                for img_in_bag in bag_dataloader:
                    emb_in_bag = self.backbone(img_in_bag.to(self.device))
                    instance_emb.append(emb_in_bag)
                bag_emb = torch.cat(instance_emb)
                bag_preds = self.clsnet(bag_emb)
                preds_list.append(bag_preds.cpu())
                labels_list.append(bag_labels.cpu())
        self.cls_report(preds_list, labels_list)

    def test_nonparametric_pool(self, pool):
        self.backbone.eval()
        self.clsnet.eval()
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for (bag_labels, bag_dataloader) in tqdm(self.loader_list, ascii=True, ncols=60):
                bag_labels = bag_labels.to(self.device)
                instance_emb = []
                for img_in_bag in bag_dataloader:
                    emb_in_bag = self.backbone(img_in_bag.to(self.device))
                    instance_emb.append(emb_in_bag)
                bag_emb = torch.cat(instance_emb)
                if pool == 'max':
                    bag_emb = bag_emb.max(dim=0)[0]
                else:
                    bag_emb = bag_emb.mean(dim=0)
                bag_preds = self.clsnet(bag_emb.unsqueeze(0))
                preds_list.append(bag_preds.cpu())
                labels_list.append(bag_labels.cpu())
        self.cls_report(preds_list, labels_list)

    def test_RNN(self, batch_size):
        # batch_size =2
        self.backbone.eval()
        self.clsnet['cls'].eval()
        self.clsnet['RNN'].eval()
        bag_label_list = []
        bag_pred_list = []
        with torch.no_grad():
            for i in range(0, len(self.loader_list), batch_size):
                label_and_dataset_list = self.loader_list[i: min((i + batch_size), len(self.loader_list))]
            #     # bag_label_list = []
                bag_embed_list = []
                for bag_idx, (bag_labels, bag_dataloader) in enumerate(tqdm(label_and_dataset_list, ascii=True, ncols=60)):
                # label
                    bag_label_list.append(bag_labels.cpu().item())
                    # preds
                    isntance_preds = []
                    instance_emb = []
                    for img_in_bag in bag_dataloader:
                        emb_in_bag = self.backbone(img_in_bag.to(self.device))
                        pred_in_bag = self.clsnet['cls'](emb_in_bag)
                        instance_emb.append(emb_in_bag)
                        isntance_preds.append(pred_in_bag)
                    instance_emb = torch.cat(instance_emb)
                    instance_preds = torch.cat(isntance_preds)
                    select_num = min(10, instance_emb.shape[0])
                    _, selected_idx = torch.topk(instance_preds, select_num, dim=0)
                    bag_embed_list.append(instance_emb[selected_idx])
                bag_embed_tesnor = torch.cat(bag_embed_list, dim=1)
                state = self.clsnet['RNN'].init_hidden(bag_embed_tesnor.shape[1]).cuda()
                for s in range(bag_embed_tesnor.shape[0]):
                    input = bag_embed_tesnor[s]
                    bag_pred, state = self.clsnet['RNN'](input, state)
                bag_pred_list.append(bag_pred.cpu())
        # import pdb
        # pdb.set_trace()
        bag_pred_list = torch.cat(bag_pred_list).squeeze(1).numpy().tolist()
        self.cls_report(bag_pred_list, bag_label_list)

    def test_instance(self):
        self.backbone.eval()
        self.clsnet.eval()
        preds_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, (img, instance_labels) in enumerate(tqdm(self.test_loader, ascii=True, ncols=60)):
                instance_preds = self.clsnet(self.backbone(img.to(self.device)))
                preds_list.append(instance_preds.cpu())
                label_list.append(instance_labels)
                # if batch_idx >10:
                #     break
        preds = np.concatenate(preds_list)
        labels = np.concatenate(label_list)
        self.cls_report(preds, labels, "ins")


    def evaluate(self):
        # read labels
        bag_labels = self.test_data.bag_labels
        # bag max evaluate
        bag_max_pred = self.memory_bank.max_pool()
        self.cls_report(bag_max_pred, bag_labels, "bag_max")
        # bag mean evaluate
        bag_mean_pred = self.memory_bank.avg_pool(self.test_data.bag_lengths)
        self.cls_report(bag_mean_pred, bag_labels, "bag_avg")

        # evaluate instance AUC
        if hasattr(self.test_data, "instance_real_labels"):
            instance_real_labels = self.test_data.instance_real_labels
            self.cls_report(self.memory_bank.to_list(self.test_data.bag_lengths), instance_real_labels, "ins")

    def cls_report(self, y_pred, y_true, prefix=""):
        """
        A combination of sklearn.metrics function and our logger (use tensorboard)

        """
        ##make hard prediction
        y_pred_hard = [(x > 0.5) for x in y_pred]
        cls_report = classification_report(y_true, y_pred_hard, output_dict=True)
        print(cls_report)
        auc_score = roc_auc_score(y_true, y_pred)

        self.logger.log_scalar(prefix+'/'+'Accuracy', cls_report['accuracy'], print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1.0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1.0']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)
        # self.logger.log_scalar(prefix + '/' + 'AUC', auc_score, print=True)
        print(confusion_matrix(y_true, y_pred_hard))

    def test_nct(self):
        self.backbone.eval()
        self.clsnet.eval()
        preds_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, (img, instance_labels) in enumerate(tqdm(self.test_loader, ascii=True, ncols=60)):
                instance_preds = self.clsnet(self.backbone(img.to(self.device)))
                preds_list.append(torch.argmax(instance_preds, dim=1).cpu().detach())
                label_list.append(instance_labels.detach())
                # if batch_idx >10:
                #     break
        preds = np.concatenate(preds_list)
        labels = np.concatenate(label_list)
        # print(preds, labels)
        cls_report = classification_report(labels, preds, output_dict=True)
        print(cls_report)
        self.logger.log_scalar('ins' + '/' + 'Accuracy', cls_report['accuracy'], print=True)
        print(confusion_matrix(labels, preds))

        

        

        
        
                
                
                
