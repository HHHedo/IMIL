#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2021 HHHedo
'''
from __future__ import absolute_import
import os, sys
sys.path.append('../')
from tqdm import tqdm
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve , average_precision_score, accuracy_score
import matplotlib.pyplot as plt
import random
from utils.utility import AverageMeter, accuracy, ProgressMeter
from sklearn.metrics import confusion_matrix,roc_curve
import datetime

class BaseTrainer(object):
    """
    Task: single branch multipe instance classification
    Dictionary update (with or without momentum)

    Args:
        memory_bank: (object: Iterable) a memory bank which store the prediction result.
            I recommend using structured memory bank. See `TensorMemoryBank` for details.
                        Example (dict):
                            {1: {0: 0.5, 1: 0.7, 2: 0.9},
                             2: {0: 0.3, 1: 0.2}
                             }
                        Example (list): 
                            [[0.5, 0.7, 0.9], [0.4, 0.2, 0.3]]
                        Example (torch.Tensor)
        
        configs: (*) extra config for training. Currently used to select whether to use
                 RCE loss. (updated 2020.2.4)
    Notes:
        1. The network outputs are instance-level prediction, 
           each instance with a bag idx and inner idx (idx inside the bag)
        2. The instnace-level prediction will be dumped into the whole prediction dictionary.
    """
    def __init__(self,
                 backbone,
                 clsnet,
                 optimizer,
                 # lrsch,
                 criterion,
                 train_loader,
                 trainset,
                 loader_list,
                 valset,
                 val_loader,
                 memory_bank,
                 save_interval=10,
                 logger=None,
                 configs=None, #self RCE or not
                 old_backbone=None,
                 clsnet_causal=None,
                 device=torch.device("cuda"),
                 **kwargs):
        self.backbone = backbone
        self.clsnet = clsnet
        self.optimizer = optimizer
        # self.lrsch = lrsch
        self.device = device
        self.criterion = criterion
        self.train_loader = train_loader
        self.trainset = trainset
        self.loader_list = loader_list
        self.valset = valset
        self.val_loader = val_loader
        self.memory_bank = memory_bank
        self.save_interval = save_interval
        self.logger = logger
        self.old_backbone = old_backbone
        self.clsnet_causal = clsnet_causal
        self.configs = configs
        self.device = device

    # - SimpleMIL, RCE
    def train(self, epoch):
        self.backbone.train()
        self.clsnet.train()
        self.logger.update_step()
        show_loss = []
        # # int info
        preds_list = []
        bag_index_list = []
        label_list = []
        ACC = AverageMeter('Acc', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, ACC],
            prefix="Epoch: [{}]".format(epoch))
        # softmax = nn.Softmax(dim=1)
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            # instance_preds = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index, None, None)
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = instance_labels.to(self.device)
            real_ins_labels = real_ins_labels.to(self.device)

            ##instancee-level prediction loss
            if self.configs == 'DigestSeg':
                loss = self.criterion(instance_preds, instance_labels)
            elif self.configs == 'DigestSegRCE':
                # pos_weight is not used
                weight, _ = self.memory_bank.get_weight(bag_index, inner_index, nodule_ratios,
                                                                 preds=instance_preds.sigmoid(),
                                                                 cur_epoch=epoch,
                                                                 )
                loss = self.criterion(instance_preds, instance_labels,
                                                          weight=weight,
                                                          )
            acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1) == instance_labels).sum().float() / len(
                real_ins_labels)
            # acc = (torch.argmax(instance_preds, dim=1)== instance_labels).sum().float() / len(
            #     instance_labels)
            losses.update(loss.item(), imgs.size(0))

            # digest
            ACC.update(acc, imgs.size(0))

            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            # self.memory_bank.update(bag_index, inner_index, softmax(instance_preds)[:,1], epoch)
            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())

            # digest
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            # preds_list.append(softmax(instance_preds)[:,1].cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            # if batch_idx >8:
            #     break

        #print info
        progress.display(batch_idx)
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        # self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        
        ##after epoch memory bank operation
        self.memory_bank.update_rank()
        self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    # - EM for patch-based (PB) CNN
    def train_EM(self, epoch, configs):
        self.backbone.train()
        self.clsnet.train()
        self.logger.update_step()
        show_loss = []
        # int info
        preds_list = []
        bag_index_list = []
        label_list = []
        debug = False
        for batch_idx, (imgs, instance_labels, 
        bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            # instance_preds = self.clsnet(self.backbone(imgs.to(self.device)),bag_index, inner_index, None, None)
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = instance_labels.to(self.device)
            loss = self.criterion(instance_preds, instance_labels)
            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            # if batch_idx>10:
            #     # debug = True
            #     break

        # print info
        if not debug:
            preds_tensor = torch.cat(preds_list)
            bag_index_tensor = torch.cat(bag_index_list)
            labels_tensor = torch.cat(label_list)
            # self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
            avg_loss = sum(show_loss) / (len(show_loss))
            self.logger.log_scalar("loss", avg_loss, print=True)
            self.logger.clear_inner_iter()

        ##after epoch memory bank operation
        if (epoch+1) % 2 == 0:
            self.EM_eval(epoch)
            self.memory_bank.Gaussian_smooth()
            self.memory_bank.update_rank()
            select_pos_idx = self.memory_bank.select_samples()
            self.trainset.generate_new_data(select_pos_idx)
            # self.trainset = EMDigestSeg(os.path.join(configs.data_root, "train"),
            #                             configs.train_transform, None, None,
            #                             pos_select_idx=select_pos_idx,
            #                             database=configs.database)
            # self.train_loader = DataLoader(self.trainset, configs.batch_size, shuffle=True, num_workers=configs.workers)
        self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)
        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    def EM_eval(self, epoch):
        self.backbone.eval()
        self.clsnet.eval()
        with torch.no_grad():
            for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(self.val_loader, ascii=True, ncols=60)):
                # instance_preds = self.clsnet(self.backbone(imgs.to(self.device)),bag_index, inner_index, None, None)
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
                self.memory_bank.update_tmp(bag_index, inner_index, instance_preds.sigmoid(), epoch) #bag-level
                # if batch_idx > 1:
                #     break

    # - Oricla
    def train_fullsupervision(self, epoch, configs):
        self.backbone.train()
        self.clsnet.train()
        self.logger.update_step()
        show_loss = []
        # int info
        preds_list = []
        bag_index_list = []
        label_list = []
        ACC = AverageMeter('Acc', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, ACC],
            prefix="Epoch: [{}]".format(epoch))
        for batch_idx, (imgs, _, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = real_ins_labels.to(self.device)
            loss = self.criterion(instance_preds, instance_labels.view(-1, 1))
            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            # print(instance_preds, softmax(instance_preds), softmax(instance_preds)[1])
            # self.memory_bank.update(bag_index, inner_index, softmax(instance_preds)[:,1], epoch)
            loss.backward()
            self.optimizer.step()
            acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1) == instance_labels).sum().float() / len(
                instance_labels)
            # acc = (torch.argmax(instance_preds, dim=1)== instance_labels).sum().float() / len(
                # instance_labels)
            losses.update(loss.item(), imgs.size(0))
            ACC.update(acc, imgs.size(0))
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            # preds_list.append(softmax(instance_preds)[:,1].cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            
                
            # if  batch_idx > 1:
            #     break

        # print info
        print('\n')
        progress.display(batch_idx)
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        # self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()

        ##after epoch memory bank operation
        # self.memory_bank.Gaussian_smooth()
        self.memory_bank.update_rank()
        self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    def train_semi(self, epoch, configs):
        self.backbone.train()
        self.clsnet.train()
        self.logger.update_step()
        show_loss = []
        # int info
        preds_list = []
        bag_index_list = []
        label_list = []
        ACC = AverageMeter('Acc', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, ACC],
            prefix="Epoch: [{}]".format(epoch))
        softmax = nn.Softmax(dim=1)
        for batch_idx, (imgs, semi_labels, bag_index, inner_index, nodule_ratios, real_ins_labels, semi_index) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
                # img, semi_labels, bag_idx, inner_idx, nodule_ratio, real_label, semi_index
            self.logger.update_iter()
            self.optimizer.zero_grad()
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = semi_labels.to(self.device)
            semi_index = semi_index.to(self.device).unsqueeze(1)
            loss = self.criterion(instance_preds, instance_labels.view(-1,1), weight = semi_index) 

            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            # self.memory_bank.update(bag_index, inner_index, softmax(instance_preds)[:,1], epoch)
            loss.backward()
            self.optimizer.step()
            acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1) == instance_labels).sum().float() / len(
                instance_labels)
            # acc = (torch.argmax(instance_preds, dim=1)== instance_labels).sum().float() / len(
            #     instance_labels)
            losses.update(loss.item(), imgs.size(0))
            ACC.update(acc, imgs.size(0))
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            # preds_list.append(softmax(instance_preds)[:,1].cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            
                
            # if  batch_idx > 8:
            #     break

        # print info
        print('\n')
        progress.display(batch_idx)
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        # self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()

        ##after epoch memory bank operation
        # self.memory_bank.Gaussian_smooth()
        # self.memory_bank.update_rank()
        self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    # - IMIL
    def train_EMCA(self, epoch, configs):
        self.backbone.train()
        self.clsnet.train()
        self.logger.update_step()
        show_loss = []
        # int info
        preds_list = []
        bag_index_list = []
        label_list = []
        debug = False
        # debug = True
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = instance_labels.to(self.device)

            ## Digestpath
            loss = self.criterion(instance_preds, instance_labels)

            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())

        ## Digestpath
        ## print info
        if not debug:
            avg_loss = sum(show_loss) / (len(show_loss))
            self.logger.log_scalar("loss", avg_loss, print=True)
            self.logger.clear_inner_iter()
        ## Digestpath

        ##after epoch memory bank operation

        new_labels = None
        weight = None
        epoch_thres = 2
        if epoch >= epoch_thres:
            if configs.config == 'DigestSegEMCA':
                select_pos_idx = self.EMCA_eval(epoch, configs)
            elif configs.config == 'DigestSegEMCAV2':
                select_pos_idx, new_labels, weight = self.EMCA_evalv2(epoch, configs, epoch_thres)
                # select_pos_idx = self.EMCA_evalv2(epoch, configs)
            elif configs.config == 'DigestSegEMnocahalf':
                select_pos_idx = self.EMCA_noca_half(epoch, configs)
            elif configs.config == 'DigestSegEMnocamean':
                select_pos_idx = self.EMCA_noca_mean(epoch, configs)
                # select_pos_idx = self.EMCA_noca(epoch, configs)
            elif configs.config == 'DigestSegGT':
                select_pos_idx = self.EMCA_globalT(epoch, configs)
            elif configs.config == 'DigestSegGM':
                select_pos_idx = self.EMCA_globalM(epoch, configs)
            self.trainset.generate_new_data(select_pos_idx, new_labels, weight)
            # self.trainset.generate_new_data(select_pos_idx)
            # self.trainset = EMDigestSeg(os.path.join(configs.data_root, "train"),
            #                             configs.train_transform, None, None,
            #                             pos_select_idx=select_pos_idx,
            #                             database=configs.database)
            # self.train_loader = DataLoader(self.trainset, configs.batch_size, shuffle=True, num_workers=configs.workers)

            self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    def EMCA_eval(self, epoch, configs):
        self.backbone.eval()
        self.clsnet.eval()
        preds_list = []
        bag_idx_list = []
        inner_idx_list = []
        with torch.no_grad():
            for batch_idx, (
            imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(self.val_loader, ascii=True, ncols=60)):
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
                self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)  # bag-level
                # global
                preds_list.append(instance_preds.sigmoid()[nodule_ratios > 1e-6])
                bag_idx_list.append(bag_index[nodule_ratios > 1e-6])
                inner_idx_list.append(inner_index[nodule_ratios > 1e-6])
                # if batch_idx > 10:
                #     break
            # self.memory_bank.Gaussian_smooth()
            # self.memory_bank.dictionary = self.memory_bank.tmp_dict
            mean_preds = self.memory_bank.get_mean()
            preds_tensor = torch.cat(preds_list)
            bag_idx_tensor = torch.cat(bag_idx_list)
            inner_idx_tensor = torch.cat(inner_idx_list)
            preds_tensor = preds_tensor / mean_preds[bag_idx_tensor]
            ignore_num = int(configs.ignore_ratio * len(preds_tensor))
            k = min(int((epoch / configs.stop_epoch) * ignore_num), ignore_num)
            selcted_idx = torch.ones_like(self.memory_bank.dictionary).cuda()
            print(k)
            if k != 0:
                _, selected_idx = torch.topk(preds_tensor, k, dim=0, largest=False)
                bag_idx_tensor = bag_idx_tensor[selected_idx]
                inner_idx_tensor = inner_idx_tensor[selected_idx]

                # print(weight.sum())
                selcted_idx[bag_idx_tensor, inner_idx_tensor] = 0
            # print(weight.sum())
            return selcted_idx

    def EMCA_evalv2(self, epoch, configs, epoch_thres=2):
        self.backbone.eval()
        self.clsnet.eval()
        # configs.ignore_goon = True
        # epoch_thres=0

        with torch.no_grad():
            for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(self.val_loader, ascii=True, ncols=60)):
                # 1. forward and update memory bank
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
                self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch) #bag-level
            # 2.get mean preds of each bag
            k = ((epoch - epoch_thres) * configs.ignore_step)
            pos_capreds_new, selected_idx, new_labels, weight =\
                self.memory_bank.select_calibrated(k, configs.ignore_thres, configs.ignore_step,
                                                   torch.stack(self.valset.bag_labels))
            self.logger.log_string('Ignore num:')
            self.logger.log_string(self.memory_bank.ignore_num)
            self.logger.log_string('labels num')
            self.logger.log_string(selected_idx.sum())
            self.logger.log_string('weight sum')
            self.logger.log_string(weight.sum())
            return selected_idx, new_labels, weight


    def get_ROC(self, dictionary, dataset):
        bag_accumulated_length = np.cumsum(np.array(dataset.bag_lengths))
        bag_accumulated_length = np.insert(bag_accumulated_length, 0, 0)
        score_list = []
        y_list = []
        for idx, bag in enumerate(dictionary):
            if self.memory_bank.bag_pos_ratio_tensor[idx]>0:
                score_list.extend(list(bag[:dataset.bag_lengths[idx]].cpu()))
                y_list.extend(dataset.instance_real_labels[bag_accumulated_length[idx]:bag_accumulated_length[idx+1]])
        AUC = roc_auc_score(y_list, score_list)
        fpr, tpr, thresholds = roc_curve(y_list, score_list, pos_label=1)
        return  AUC, fpr, tpr
  

    def train_TOPK(self, epoch, configs):
        self.backbone.train()
        self.clsnet.train()
        self.logger.update_step()
        show_loss_ce = []
        show_loss_center = []
        # int info
        preds_list = []
        bag_index_list = []
        label_list = []
        debug = False
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            feat = self.backbone(imgs.to(self.device))
            instance_preds = self.clsnet(feat)
            instance_labels = instance_labels.to(self.device)
            bag_index = bag_index.to(self.device)
            loss_ce  = self.criterion(instance_preds, instance_labels)
            loss = loss_ce
            # print(loss_center)
            loss.backward()
            self.optimizer.step()
            show_loss_ce.append(loss_ce.item())
            # show_loss_center.append(loss_center.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())


        ##after epoch memory bank operation
        select_pos_idx = self.TOPK_eval(epoch, configs)
        self.trainset.generate_new_data(select_pos_idx)
        self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)
        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    def TOPK_eval(self, epoch, configs):
        self.backbone.eval()
        self.clsnet.eval()
        with torch.no_grad():
            for batch_idx, (
            imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(self.val_loader, ascii=True, ncols=60)):
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
                self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
                # if batch_idx >1:
                #     break
        selected_idx = self.memory_bank.select_topk(torch.stack(self.valset.bag_labels))
        return selected_idx


    def cal_preds_in_training(self, preds_tensor, bag_index_tensor, labels_tensor, epoch):
        bag_pos_ratio = torch.stack(self.trainset.bag_pos_ratios).squeeze(-1)
        y_pred_hard = [(x > 0.5) for x in preds_tensor]
        labels_tensor = labels_tensor.numpy()
        cls_report = classification_report(labels_tensor, y_pred_hard, output_dict=False)
        auc_score = roc_auc_score(labels_tensor, preds_tensor.numpy())
        print(cls_report)
        print('AUC:', auc_score)
        print(confusion_matrix(labels_tensor, np.array(y_pred_hard)))
        self.logger.log_string("{}:{}\n".format('Time', datetime.datetime.now()))
        # print(bag_index_tensor.max() + 1)
        # e mean predictions of bags
        bag_pred_ratio = torch.stack(
            [preds_tensor[bag_index_tensor == i].mean() for i in range(bag_index_tensor.max() + 1)])
        # the mean predictions of positive instance of bags
        bag_pos_pred_mean = torch.stack(
            [preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 1].mean()
             for i in range(bag_index_tensor.max() + 1)])
        # the mean predictions of negative instance of bags
        bag_neg_pred_mean = torch.stack(
            [preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 0].mean()
             for i in range(bag_index_tensor.max() + 1)])
        # the predictions of positive instance of bags
        bag_pos_pred = [
            preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 1]
            for i in range(bag_index_tensor.max() + 1)]
        # the predictions of negative instance of bags
        bag_neg_pred = [
            preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 0]
            for i in range(bag_index_tensor.max() + 1)]
        # expand the mean predictions of bags as bag_pos_pred
        bag_pred_ratio_pos = torch.cat(
            [preds_tensor[bag_index_tensor == i].mean().expand_as(bag_pos_pred[i]) for i in
             range(bag_index_tensor.max() + 1)])
        # expand the mean predictions of bags as bag_neg_pred
        bag_pred_ratio_neg = torch.cat(
            [preds_tensor[bag_index_tensor == i].mean().expand_as(bag_neg_pred[i]) for i in
             range(bag_index_tensor.max() + 1)])
        bag_pos_pred = torch.cat(bag_pos_pred)
        bag_neg_pred = torch.cat(bag_neg_pred)


        x = bag_pred_ratio.cpu().numpy()  # mean preds
        y = bag_pos_pred_mean.cpu().numpy()  # mean preds of pos
        z = bag_neg_pred_mean.cpu().numpy()  # mean preds of neg
        w = bag_pos_ratio.cpu().numpy()  # pos ratio
        # print('mean preds:{}, mean preds of pos:{}, mean preds of neg:{}'.format(x.shape, y.shape ,z.shape))
        # print(bag_pos_ratio)
        # only choose pos bag
        # import pdb
        # pdb.set_trace()
        x = x[bag_pos_ratio != 0]
        y = y[bag_pos_ratio != 0]
        z = z[bag_pos_ratio != 0]
        w = w[bag_pos_ratio != 0]
        # print((z))
        # import pdb
        # pdb.set_trace()
        # # To see every instance, no result
        # y1 = bag_pred_ratio_pos.cpu().numpy()
        # x1 = bag_pos_pred.cpu().numpy()
        # y2 = bag_pred_ratio_neg.cpu().numpy()
        # x2 = bag_neg_pred.cpu().numpy()
        # x1 = x1[y1!=0]
        # y1 = y1[y1!=0]
        # x2 = x2[y2!=0]
        # y2 = y2[y2!=0]

        plt.figure()
        # plt.scatter(y1, x1)
        # plt.scatter(y2, x2, color="green")
        # pos ratio VS preds
        # plt.scatter(w, y)
        # plt.scatter(w, z, color="green")
        # mean preds VS (pos/neg) preds
        plt.scatter(x[x!=y], y[x!=y], label='++')
        plt.scatter(x[x==y], y[x==y], label='all+', color="red")
        plt.scatter(x, z, label='+-', color="green")
        plt.legend()
        # plt.scatter(w,x)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        if not os.path.exists(os.path.join(self.logger.logdir, 'preds_figs')):
            os.makedirs(os.path.join(self.logger.logdir, 'preds_figs'))
        plt.savefig(os.path.join(self.logger.logdir, 'preds_figs', "{}.png".format(epoch)))
        # plt.show()
        plt.close()



