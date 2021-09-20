#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2021 HHHedo
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
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, average_precision_score
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
                 memory_bank, logger=None, old_backbone=None, clsnet_causal=None, bs=64, device=torch.device("cuda")):
        self.backbone = backbone
        self.clsnet = clsnet
        self.test_data = test_data
        self.test_loader = test_loader
        self.loader_list = loader_list
        self.memory_bank = memory_bank
        self.logger = logger
        if old_backbone is not None:
            self.old_backbone = old_backbone.eval()
        self.clsnet_causal = clsnet_causal
        self.bs = bs
        self.device = device

    def inference(self, bs):

        self.backbone.eval()
        self.clsnet.eval()
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for batch_idx, (img, instance_label, bag_idx, inner_idx, _, _) in enumerate(tqdm(self.test_loader, ascii=True, ncols=60)):
                # instance_preds,_= self.clsnet(self.backbone(img.to(self.device)), bag_idx, inner_idx, self.old_backbone, img.to(self.device))
                instance_preds = self.clsnet(self.backbone(img.to(self.device)))
                # self.memory_bank.update(bag_idx, inner_idx, softmax(instance_preds)[:,1])
                self.memory_bank.update(bag_idx, inner_idx, instance_preds.sigmoid())
            self.logger.save_result(
            "test_mmbank", self.memory_bank.state_dict())


    def evaluate(self):
        # read labels
        bag_labels = torch.stack(self.test_data.bag_labels)
        # bag max evaluate
        bag_max_pred = self.memory_bank.max_pool(self.test_data.bag_lengths)
        # self.cls_report(bag_max_pred[:,0], bag_labels[:,0], "bag_max")

        # bag mean evaluate
        bag_mean_pred = self.memory_bank.avg_pool(self.test_data.bag_lengths)
        # self.cls_report(bag_mean_pred[:,0], bag_labels[:,0], "bag_avg")
        # bag voting evaluate
        bag_voting_pred = self.memory_bank.voting_pool(self.test_data.bag_lengths)
        # self.cls_report(bag_mean_pred[:,0], bag_labels[:,0], "bag_voting")

        AP_list = torch.zeros([3,bag_max_pred.shape[1]])
        for i in range(bag_max_pred.shape[1]):
            AP_max = average_precision_score(bag_labels[:, i], bag_max_pred[:, i])
            AP_mean = average_precision_score(bag_labels[:, i], bag_mean_pred[:, i])
            AP_voting = average_precision_score(bag_labels[:, i], bag_voting_pred[:, i])
            AP_list[0,i], AP_list[1,i], AP_list[2,i] = AP_max, AP_mean, AP_voting
        # print('mAP for max, mean, voting',AP_list.mean(1))
        self.logger.log_scalar('max' + '/' + 'AP', AP_list.mean(1)[0], print=True)
        self.logger.log_string(AP_list[0])
        self.logger.log_scalar('mean' + '/' + 'AP', AP_list.mean(1)[1], print=True)
        self.logger.log_string(AP_list[1])
        self.logger.log_scalar('voting' + '/' + 'AP', AP_list.mean(1)[2], print=True)
        self.logger.log_string(AP_list[2])
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
        AP = average_precision_score(y_true, y_pred)
        # print(AP)
        self.logger.log_scalar(prefix+'/'+'Accuracy', cls_report['accuracy'], print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1.0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1.0']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)
        self.logger.log_scalar(prefix + '/' + 'AP', AP, print=True)
        # self.logger.log_scalar(prefix + '/' + 'AUC', auc_score, print=True)
        print(confusion_matrix(y_true, y_pred_hard))


    # def causalconcat_full(self, bs):
    #     self.backbone.eval()
    #     self.clsnet.eval()
    #     with torch.no_grad():
    #         for batch_idx, (img, instance_label, bag_idx, inner_idx, _, _) in enumerate(tqdm(self.test_loader, ascii=True, ncols=60)):
    #             # instance_preds = self.clsnet(self.backbone(img.to(self.device)), bag_idx, inner_idx, self.old_backbone, img.to(self.device))
    #             # original_feature = self.old_backbone(img.to(self.device))
    #             causal_feature = self.backbone(img.to(self.device))
    #             feature = causal_feature
    #             # feature = torch.cat((original_feature, causal_feature), 1)
    #             instance_preds = self.clsnet(feature)
    #             self.memory_bank.update(bag_idx, inner_idx, instance_preds.sigmoid())
    #         self.logger.save_result(
    #         "test_mmbank", self.memory_bank.state_dict())
    #
    # def inference_twostage(self):
    #     self.backbone.eval()
    #     self.clsnet.eval()
    #     preds_list = []
    #     labels_list = []
    #     with torch.no_grad():
    #         for (bag_labels, bag_dataloader) in tqdm(self.loader_list, ascii=True, ncols=60):
    #             bag_labels = bag_labels.to(self.device)
    #             instance_emb = []
    #             for img_in_bag in bag_dataloader:
    #                 emb_in_bag = self.backbone(img_in_bag.to(self.device))
    #                 instance_emb.append(emb_in_bag)
    #             bag_emb = torch.cat(instance_emb)
    #             bag_preds = self.clsnet(bag_emb)
    #             preds_list.append(bag_preds.cpu())
    #             labels_list.append(bag_labels.cpu())
    #     self.cls_report(preds_list, labels_list)
    #
    # def test_nonparametric_pool(self, pool):
    #     self.backbone.eval()
    #     self.clsnet.eval()
    #     preds_list = []
    #     labels_list = []
    #     with torch.no_grad():
    #         for (bag_labels, bag_dataloader) in tqdm(self.loader_list, ascii=True, ncols=60):
    #             bag_labels = bag_labels.to(self.device)
    #             instance_emb = []
    #             for img_in_bag in bag_dataloader:
    #                 emb_in_bag = self.backbone(img_in_bag.to(self.device))
    #                 instance_emb.append(emb_in_bag)
    #             bag_emb = torch.cat(instance_emb)
    #             if pool == 'max':
    #                 bag_emb = bag_emb.max(dim=0)[0]
    #             else:
    #                 bag_emb = bag_emb.mean(dim=0)
    #             bag_preds = self.clsnet(bag_emb.unsqueeze(0))
    #             preds_list.append(bag_preds.cpu())
    #             labels_list.append(bag_labels.cpu())
    #     self.cls_report(preds_list, labels_list)
    #
    # def test_RNN(self, batch_size):
    #     # batch_size =2
    #     self.backbone.eval()
    #     self.clsnet['cls'].eval()
    #     self.clsnet['RNN'].eval()
    #     bag_label_list = []
    #     bag_pred_list = []
    #     with torch.no_grad():
    #         for i in range(0, len(self.loader_list), batch_size):
    #             label_and_dataset_list = self.loader_list[i: min((i + batch_size), len(self.loader_list))]
    #         #     # bag_label_list = []
    #             bag_embed_list = []
    #             for bag_idx, (bag_labels, bag_dataloader) in enumerate(tqdm(label_and_dataset_list, ascii=True, ncols=60)):
    #             # label
    #                 bag_label_list.append(bag_labels.cpu().item())
    #                 # preds
    #                 isntance_preds = []
    #                 instance_emb = []
    #                 for img_in_bag in bag_dataloader:
    #                     emb_in_bag = self.backbone(img_in_bag.to(self.device))
    #                     pred_in_bag = self.clsnet['cls'](emb_in_bag)
    #                     instance_emb.append(emb_in_bag)
    #                     isntance_preds.append(pred_in_bag)
    #                 instance_emb = torch.cat(instance_emb)
    #                 instance_preds = torch.cat(isntance_preds)
    #                 select_num = min(10, instance_emb.shape[0])
    #                 _, selected_idx = torch.topk(instance_preds, select_num, dim=0)
    #                 bag_embed_list.append(instance_emb[selected_idx])
    #             bag_embed_tesnor = torch.cat(bag_embed_list, dim=1)
    #             state = self.clsnet['RNN'].init_hidden(bag_embed_tesnor.shape[1]).cuda()
    #             for s in range(bag_embed_tesnor.shape[0]):
    #                 input = bag_embed_tesnor[s]
    #                 bag_pred, state = self.clsnet['RNN'](input, state)
    #             bag_pred_list.append(bag_pred.cpu())
    #     # import pdb
    #     # pdb.set_trace()
    #     bag_pred_list = torch.cat(bag_pred_list).squeeze(1).numpy().tolist()
    #     self.cls_report(bag_pred_list, bag_label_list)
    #
    # def test_instance(self):
    #     self.backbone.eval()
    #     self.clsnet.eval()
    #     preds_list = []
    #     label_list = []
    #     with torch.no_grad():
    #         for batch_idx, (img, instance_labels) in enumerate(tqdm(self.test_loader, ascii=True, ncols=60)):
    #             instance_preds = self.clsnet(self.backbone(img.to(self.device)))
    #             preds_list.append(instance_preds.cpu())
    #             label_list.append(instance_labels)
    #             # if batch_idx >10:
    #             #     break
    #     preds = np.concatenate(preds_list)
    #     labels = np.concatenate(label_list)
    #     self.cls_report(preds, labels, "ins")
    #
    # def test_nct(self):
    #     self.backbone.eval()
    #     self.clsnet.eval()
    #     preds_list = []
    #     label_list = []
    #     with torch.no_grad():
    #         for batch_idx, (img, instance_labels) in enumerate(tqdm(self.test_loader, ascii=True, ncols=60)):
    #             instance_preds = self.clsnet(self.backbone(img.to(self.device)))
    #             preds_list.append(torch.argmax(instance_preds, dim=1).cpu().detach())
    #             label_list.append(instance_labels.detach())
    #             # if batch_idx >10:
    #             #     break
    #     preds = np.concatenate(preds_list)
    #     labels = np.concatenate(label_list)
    #     # print(preds, labels)
    #     cls_report = classification_report(labels, preds, output_dict=True)
    #     print(cls_report)
    #     self.logger.log_scalar('ins' + '/' + 'Accuracy', cls_report['accuracy'], print=True)
    #     print(confusion_matrix(labels, preds))



    def eval_(self, gs, trainset):
        # self.backbone = self.logger.load_backbone_fromold(self.backbone, global_step=gs)
        # self.clsnet = self.logger.load_clsnet_fromold(self.clsnet, global_step=gs)
        self.backbone = self.logger.load_backbone(self.backbone, global_step=gs)
        self.clsnet = self.logger.load_clsnet(self.clsnet, global_step=gs)
        self.backbone.eval()
        self.clsnet.eval()
        import pickle
        # with open('/remote-home/ltc/HisMIL/testset.pickle', 'rb') as f:
        #     self.testset = pickle.load(f)
        #     val_loader = DataLoader(self.testset, 256, shuffle=False, num_workers=4)
        val_loader = DataLoader(trainset, 256, shuffle=True, num_workers=8)
        preds_list = []
        bag_index_list = []
        label_list = []
        idx_x_list = []
        idx_y_list = []
        path_list = []
        idx_list = []
        with torch.no_grad():
            for batch_idx, (imgs, instance_labels,bag_idx, inner_idx, _,most_conf_idx) in enumerate(
                    tqdm(val_loader, ascii=True, ncols=60)):
                instance_preds = torch.sigmoid(self.clsnet(self.backbone(imgs.to(self.device))))
                instance_labels = instance_labels.to(self.device)
                preds_list.append(instance_preds.cpu().detach())
                # bag_index_list.append(bag_index.cpu().detach())
                label_list.append(instance_labels.cpu().detach())
                idx_list.append(most_conf_idx.cpu().detach())
                # idx_x_list.append(inner_index[1])
                # idx_y_list.append(inner_index[2])
                # path_list.extend(list(inner_index[3]))
                self.memory_bank.update(bag_idx, inner_idx, instance_preds)

                # if batch_idx >10:
                #     break
        preds_tensor = torch.cat(preds_list)

        labels_tensor = torch.cat(label_list)
        index_tensor = torch.cat(idx_list)

        selected_preds = preds_tensor[index_tensor==1]
        selected_labels = labels_tensor[index_tensor==1]
        bag_max_pred = self.memory_bank.max_pool(trainset.bag_lengths)
        # self.cls_report(bag_max_pred[:,0], bag_labels[:,0], "bag_max")

        # bag mean evaluate
        bag_mean_pred = self.memory_bank.avg_pool(trainset.bag_lengths)
        # self.cls_report(bag_mean_pred[:,0], bag_labels[:,0], "bag_avg")
        # bag voting evaluate
        bag_voting_pred = self.memory_bank.voting_pool(trainset.bag_lengths)
        bag_labels = torch.stack(trainset.bag_labels)
        AP_list = torch.zeros([4, bag_max_pred.shape[1]])
        for i in range(bag_max_pred.shape[1]):
            AP_max = average_precision_score(bag_labels[:, i], bag_max_pred[:, i])
            AP_mean = average_precision_score(bag_labels[:, i], bag_mean_pred[:, i])
            AP_voting = average_precision_score(bag_labels[:, i], bag_voting_pred[:, i])
            AP_selected = average_precision_score(selected_labels[:, i], selected_preds[:, i])
            AP_list[0, i], AP_list[1, i], AP_list[2, i] , AP_list[3, i] = AP_max, AP_mean, AP_voting, AP_selected
        print(AP_list.mean(1))

        

        

        
        
                
                
                
