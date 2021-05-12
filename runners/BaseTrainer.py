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
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import pandas as pd
# import seaborn as sns
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve 
from data.DigestSegBag import DigestSegIns
import matplotlib.pyplot as plt
import random
from data.EMDigestSeg import EMDigestSeg
from utils.utility import AverageMeter, accuracy, ProgressMeter
from sklearn.metrics import confusion_matrix


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
                 lrsch,
                 criterion,
                 train_loader,
                 trainset,
                 loader_list,
                 val_loader,
                 memory_bank,
                 save_interval=10,
                 logger=None,
                 configs=None, #self RCE or not
                 device=torch.device("cuda"),
                 **kwargs):
        self.backbone = backbone
        self.clsnet = clsnet
        self.optimizer = optimizer
        self.lrsch = lrsch
        self.device = device
        self.criterion = criterion
        self.train_loader = train_loader
        self.trainset = trainset
        self.loader_list = loader_list
        self.val_loader = val_loader
        # self.train_loader = DataLoader(self.trainset, batch_sizef, shuffle=True, num_workers=num_workers)
        self.memory_bank = memory_bank
        self.save_interval = save_interval
        self.logger = logger
        self.configs = configs
        self.device = device

    # - SimpleMIL, RCE, Ca
    def train(self, epoch):
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
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index, None, None)
            instance_labels = instance_labels.to(self.device)
            real_ins_labels = real_ins_labels.to(self.device)

            ##instancee-level prediction loss
            if self.configs == 'DigestSeg':
                loss = self.criterion(instance_preds, instance_labels.view(-1, 1))

            elif self.configs == 'DigestSegRCE':
                # pos_weight is not used
                weight, _ = self.memory_bank.get_weight(bag_index, inner_index, nodule_ratios,
                                                                 preds=instance_preds.sigmoid(),
                                                                 cur_epoch=epoch,
                                                                 )
                loss = self.criterion(instance_preds, instance_labels.view(-1, 1),
                                                          weight=weight,
                                                          )
            elif self.configs == 'DigestSegCa':
                # Ca
                weight, _ = self.memory_bank.get_weight(bag_index, inner_index, nodule_ratios,
                                                                 preds=instance_preds.sigmoid(),
                                                                 cur_epoch=epoch,
                                                                 )

                loss = F.binary_cross_entropy_with_logits(instance_preds, instance_labels.view(-1, 1),
                                                          weight=weight,
                                                          reduction='sum'
                                                          )/weight.sum()

            acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1)==instance_labels).sum().float()/len(real_ins_labels)
            losses.update(loss.item(), imgs.size(0))
            ACC.update(acc, imgs.size(0))
            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())

            if batch_idx  % 1000 == 0:
                progress.display(batch_idx)
                print(torch.ge(instance_preds.sigmoid(),0.5).float().squeeze(1).sum())
            # if batch_idx >8:
            #     print(bag_index_list)
            #     int(label_list)
            #     break

        #print info
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        if self.lrsch is not None:
            if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
                self.lrsch.step(avg_loss)
                print('lr changs wrongly')
            else:
                self.lrsch.step()
                print('lr changs wrongly')
        
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
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)),bag_index, inner_index, None, None)
            instance_labels = instance_labels.to(self.device)

            # weight, pos_weight = self.memory_bank.get_weight(bag_index,
            #                                                  inner_index,
            #                                                  nodule_ratios,
            #                                                  preds=instance_preds.sigmoid(),
            #                                                  cur_epoch=epoch,
            #                                                  )
            loss = self.criterion(instance_preds, instance_labels.view(-1, 1))
            # loss = F.binary_cross_entropy_with_logits(instance_preds, instance_labels.view(-1, 1))


            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            # if batch_idx>1:
            #     debug = True
            #     break

        # print info
        if not debug:
            preds_tensor = torch.cat(preds_list)
            bag_index_tensor = torch.cat(bag_index_list)
            labels_tensor = torch.cat(label_list)
            self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
            avg_loss = sum(show_loss) / (len(show_loss))
            self.logger.log_scalar("loss", avg_loss, print=True)
            self.logger.clear_inner_iter()
            if self.lrsch is not None:
                if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lrsch.step(avg_loss)
                    print('lr changs wrongly')
                else:
                    self.lrsch.step()
                    print('lr changs wrongly')

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
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)),bag_index, inner_index, None, None)
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
            # instance_preds, cluster_preds ,cluster_labels = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index)
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index, None, None)
            instance_labels = real_ins_labels.to(self.device)
            # cluster_labels = cluster_labels.to(self.device)
            # loss = self.criterion(instance_preds, instance_labels.view(-1, 1)) + configs.CE(cluster_preds ,cluster_labels)
            loss_self = self.criterion(instance_preds, instance_labels.view(-1, 1))
            # loss_conf = configs.CE(cluster_preds ,cluster_labels)
            loss = loss_self 
            # print('loss:',loss.item())
            # print('loss_self: {}, loss_conf: {}'.format(loss_self, loss_conf))
            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            loss.backward()
            self.optimizer.step()
            acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1) == instance_labels).sum().float() / len(
                instance_labels)
            losses.update(loss.item(), imgs.size(0))
            ACC.update(acc, imgs.size(0))
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            
                
            # if  batch_idx > 10:
            #     break

        # print info
        print('\n')
        progress.display(batch_idx)
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        if self.lrsch is not None:
            if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
                self.lrsch.step(avg_loss)
                print('lr changs wrongly')
            else:
                self.lrsch.step()
                print('lr changs wrongly')

        ##after epoch memory bank operation
        # self.memory_bank.Gaussian_smooth()
        self.memory_bank.update_rank()
        self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    # - EM based Ca
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
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = instance_labels.to(self.device)

            loss = self.criterion(instance_preds, instance_labels.view(-1, 1))

            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            # print(instance_labels.sum(), real_ins_labels.sum())

            # if batch_idx > 8:
            # #     # import pdb
            # #     # pdb.set_trace()
            # #     # print(bag_index_list)
            # #     # print(label_list)
            #     break

        # print info
        if not debug:
            preds_tensor = torch.cat(preds_list)
            bag_index_tensor = torch.cat(bag_index_list)
            labels_tensor = torch.cat(label_list)
            self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
            print('save imgaes')
            avg_loss = sum(show_loss) / (len(show_loss))
            self.logger.log_scalar("loss", avg_loss, print=True)
            self.logger.clear_inner_iter()
            if self.lrsch is not None:
                if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lrsch.step(avg_loss)
                    print('lr changs wrongly')
                else:
                    self.lrsch.step()
                    print('lr changs wrongly')

        ##after epoch memory bank operation
        # if (epoch+1) % 2 == 0:
        # self.EMCA_eval(epoch, configs)
        # self.memory_bank.Gaussian_smooth()
        # self.memory_bank.update_rank()
        if configs.config == 'DigestSegEMCA':
            select_pos_idx = self.EMCA_eval(epoch, configs)
        elif configs.config == 'DigestSegEMCAV2':
            select_pos_idx = self.EMCA_evalv2(epoch, configs)
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
    #
    #     # - EM based Ca
    # - EM based Ca

    def EMCA_evalv2(self, epoch, configs, epoch_thres=1):
        self.backbone.eval()
        self.clsnet.eval()
        # configs.ignore_goon = True
        # epoch_thres=0
        with torch.no_grad():
            for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(self.val_loader, ascii=True, ncols=60)):
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
                self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch) #bag-level
                # if batch_idx > 10:
                #     break
            mean_preds = self.memory_bank.get_mean()
            # pos bag & not -1
            calibrated_preds = self.memory_bank.dictionary/mean_preds
            pos_calibrated_preds = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor>0] #postive_bag
            pos_calibrated_preds_valid = pos_calibrated_preds[pos_calibrated_preds>0] #postive_instance

            # ignore_num = int(configs.ignore_ratio*self.memory_bank.bag_lens[self.memory_bank.bag_pos_ratio_tensor>0].sum())
            # k = min(int((epoch / configs.stop_epoch) * ignore_num), ignore_num)
            pos_ins_num = self.memory_bank.bag_lens[self.memory_bank.bag_pos_ratio_tensor>0].sum()
            if configs.ignore_goon and epoch > epoch_thres:
                k = int((epoch - epoch_thres)*configs.ignore_step*pos_ins_num)
            else:
                k = self.memory_bank.ignore_num
            selected_idx = torch.ones_like(self.memory_bank.dictionary).cuda()
            selected_idx[self.memory_bank.dictionary == -1] = 0
            self.logger.log_string('{:.3}%/{} Ignored samples.'.format(k/pos_ins_num.float()*100, k))
            if k != 0:
                k_preds, _ = torch.topk(pos_calibrated_preds_valid, k, dim=0, largest=False)
                pos_selected = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor>0] > k_preds[-1]
                neg_selected = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor==0] >0
                selected_idx = torch.cat((pos_selected, neg_selected)).float() # contain both positive and negative
                if configs.ignore_goon:
                    new_selected_idx = ((calibrated_preds <= k_preds[-1]) & \
                                        (calibrated_preds > k_preds[-int(configs.ignore_step*pos_ins_num)])).float()
                    pos_selected_idx = new_selected_idx[self.memory_bank.bag_pos_ratio_tensor > 0]
                    pos_capreds_new = (pos_calibrated_preds[pos_selected_idx==1]).mean()
                    self.logger.log_string('Mean calibrated preds is {}, from ({}  to {}] with {:.3} and {:.3}.'
                                           .format(pos_capreds_new, -int(configs.ignore_step*pos_ins_num), -1,
                                                   k_preds[-int(configs.ignore_step*pos_ins_num)], k_preds[-1]))

                    top_selected_idx = ((calibrated_preds > k_preds[0]) & \
                                        (calibrated_preds <= k_preds[int(configs.ignore_step * pos_ins_num)-1])).float()
                    top_selected_idx = top_selected_idx[self.memory_bank.bag_pos_ratio_tensor > 0]
                    top_capreds_new = (pos_calibrated_preds[top_selected_idx == 1]).mean()
                    print('Top Mean calibrated preds is {}, from ({}  to {}] with {:.3} and {:.3}.'
                          .format(top_capreds_new, 0, int(configs.ignore_step * pos_ins_num)-1,
                                  k_preds[0], k_preds[int(configs.ignore_step * pos_ins_num)-1] ))

                    if pos_capreds_new > configs.ignore_thres and (epoch-epoch_thres)==1:
                        epoch_thres +=1
                    elif pos_capreds_new < configs.ignore_thres:
                        self.memory_bank.ignore_num = k
                    else:
                        configs.ignore_goon = False
            return selected_idx
            # return torch.ones_like(self.memory_bank.dictionary).cuda()

        # - EM based Ca

    # def EMCA_evalv3(self, epoch, configs):
    #     self.backbone.eval()
    #     self.clsnet.eval()
    #     # preds_list = []
    #     # bag_idx_list = []
    #     # inner_idx_list = []
    #     with torch.no_grad():
    #         for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
    #                 tqdm(self.val_loader, ascii=True, ncols=60)):
    #             instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
    #             self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch) #bag-level
    #             # global
    #             # preds_list.append(instance_preds.sigmoid()[nodule_ratios > 1e-6])
    #             # bag_idx_list.append(bag_index[nodule_ratios > 1e-6])
    #             # inner_idx_list.append(inner_index[nodule_ratios > 1e-6])
    #             # if batch_idx > 10:
    #             #     break
    #         # self.memory_bank.Gaussian_smooth()
    #         # self.memory_bank.dictionary = self.memory_bank.tmp_dict
    #         mean_preds = self.memory_bank.get_mean()
    #         # pos bag & not -1
    #         calibrated_preds = self.memory_bank.dictionary/mean_preds
    #         pos_calibrated_preds = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor>0]
    #         pos_calibrated_preds_valid = pos_calibrated_preds[pos_calibrated_preds>0]
    #
    #         ignore_num = int(configs.ignore_ratio*self.memory_bank.bag_lens[self.memory_bank.bag_pos_ratio_tensor>0].sum())
    #         k = min(int((epoch / configs.stop_epoch) * ignore_num), ignore_num)
    #         # last_k = min(int(((epoch -1) / configs.stop_epoch) * ignore_num), ignore_num)
    #         selected_idx = torch.ones_like(self.memory_bank.dictionary).cuda()
    #         print(k)
    #         if k != 0:
    #             k_preds, _ = torch.topk(pos_calibrated_preds_valid, k, dim=0, largest=False)
    #             # last_k_preds, _ = torch.topk(pos_calibrated_preds_valid, last_k, dim=0, largest=False)
    #             selected_idx = (calibrated_preds > k_preds[-1]).float()
    #             last_selected_idx = (calibrated_preds > k_preds[-1]) & \
    #                                 (calibrated_preds < k_preds[-1 - int(ignore_num/configs.stop_epoch)]).float()
    #
    #             pos_selected_idx = last_selected_idx[self.memory_bank.bag_pos_ratio_tensor>0].long()
    #             pos_ca_preds = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor>0]
    #             pos_preds_diff = (pos_ca_preds[pos_selected_idx]).mean()
    #
    #             print(self.memory_bank.last_diff-pos_preds_diff)
    #             self.memory_bank.last_diff = pos_preds_diff
    #         return selected_idx
    #
    #     # - EM based Ca

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
            loss_ce = self.criterion['CE'](instance_preds, instance_labels.view(-1, 1))
            loss_center = self.criterion['Center'](bag_index, feat)
            loss = loss_ce + 0.01*loss_center
            # print(loss_center)
            loss.backward()
            self.optimizer.step()
            show_loss_ce.append(loss_ce.item())
            show_loss_center.append(loss_center.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            # if batch_idx>1:
            #     debug = True
            #     break

        # print info
        if not debug:
            preds_tensor = torch.cat(preds_list)
            bag_index_tensor = torch.cat(bag_index_list)
            labels_tensor = torch.cat(label_list)
            self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
            avg_ce_loss = sum(show_loss_ce) / (len(show_loss_ce))
            avg_center_loss = sum(show_loss_center)/ (len(show_loss_center))
            self.logger.log_scalar("loss", avg_ce_loss, print=True)
            self.logger.log_scalar("loss", avg_center_loss, print=True)
            self.logger.clear_inner_iter()


        ##after epoch memory bank operation
        # if (epoch+1) % 2 == 0:
        # self.EMCA_eval(epoch, configs)
        # self.memory_bank.Gaussian_smooth()
        # self.memory_bank.update_rank()
        select_pos_idx = self.TOPK_eval(epoch, configs)
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
        selected_idx = self.memory_bank.select_topk()
        return selected_idx


               

#########################################
    # - AMIL
    def train_AMIL(self):
        self.backbone.eval()
        self.clsnet.train()
        self.logger.update_step()
        show_loss = []

        # for batch_idx, (img_dirs, bag_labels) in enumerate(tqdm(self.train_loader, ascii=True, ncols=60)):
        #     self.logger.update_iter()
        random.shuffle(self.loader_list)
        for (bag_labels, bag_dataloader) in tqdm(self.loader_list, ascii=True, ncols=60):
            bag_labels = bag_labels.to(self.device)

            instance_emb = []
            for img_in_bag in bag_dataloader:
                with torch.no_grad():
                    emb_in_bag = self.backbone(img_in_bag.to(self.device))
                    instance_emb.append(emb_in_bag)
            instance_emb = torch.cat(instance_emb)
            instance_preds = self.clsnet(instance_emb)
            loss = self.criterion(instance_preds, bag_labels.view(-1, 1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())

        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    def train_nonparametric_pool(self, pool, batch_size):
        # batch_size=2
        self.backbone.eval()
        self.clsnet.train()
        self.logger.update_step()
        show_loss = []

        random.shuffle(self.loader_list)
        # Get a batch of [bag_label, bag_dataloader]
        for i in range(0, len(self.loader_list), batch_size):
            label_and_dataset_list = self.loader_list[i: min((i + batch_size), len(self.loader_list))]
            bag_label_list = []
            bag_embed_list = []
            for (bag_labels, bag_dataloader) in tqdm(label_and_dataset_list, ascii=True, ncols=60):
                bag_labels = bag_labels.to(self.device)
                bag_label_list.append(bag_labels)
                insEmbinOneBag = []
                for img_in_bag in bag_dataloader:
                    with torch.no_grad():
                        emb_in_bag = self.backbone(img_in_bag.to(self.device))
                        insEmbinOneBag.append(emb_in_bag)
                insEmbinOneBag = torch.cat(insEmbinOneBag)
                if pool == 'max':
                    bag_emb = insEmbinOneBag.max(dim=0)[0]
                else:
                    bag_emb = insEmbinOneBag.mean(dim=0)
                bag_embed_list.append(bag_emb)
            bag_labels = torch.stack(bag_label_list)
            bag_embds = torch.stack(bag_embed_list)
            bag_preds = self.clsnet(bag_embds)
            loss = self.criterion(bag_preds, bag_labels.view(-1, 1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())
            # break

        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    # - RNN aggregation
    def train_RNN(self, epoch, batch_size):
        # batch_size = 3
        self.backbone.eval()
        self.clsnet['cls'].eval()
        self.clsnet['RNN'].train()
        self.logger.update_step()
        show_loss = []

        random.shuffle(self.loader_list)
        # Get a batch of [bag_label, bag_dataloader]
        for i in range(0, len(self.loader_list), batch_size):
            label_and_dataset_list = self.loader_list[i: min((i + batch_size), len(self.loader_list))]
            bag_label_list = []
            bag_embed_list = []
            # Getting one dataloader
            for bag_idx, (bag_labels, bag_dataloader) in enumerate(tqdm(label_and_dataset_list, ascii=True, ncols=60)):
                # label
                bag_label_list.append(bag_labels.to(self.device))
                # preds
                isntance_preds = []
                instance_emb = []
                for img_in_bag in bag_dataloader:
                    # Getting imgs
                    with torch.no_grad():
                        emb_in_bag = self.backbone(img_in_bag.to(self.device))
                        pred_in_bag = self.clsnet['cls'](emb_in_bag)
                        instance_emb.append(emb_in_bag)
                        isntance_preds.append(pred_in_bag)
                instance_emb = torch.cat(instance_emb)
                instance_preds = torch.cat(isntance_preds)
                select_num = min(10, instance_emb.shape[0])
                _, selected_idx = torch.topk(instance_preds, select_num, dim=0)
                bag_embed_list.append(instance_emb[selected_idx])
            # import pdb
            # pdb.set_trace()
            # print(bag_label_list)
            # Stacking the batch of bags
            bag_label_tensor = torch.cat(bag_label_list)
            bag_embed_tesnor = torch.cat(bag_embed_list, dim=1) # k*bs*512
            state = self.clsnet['RNN'].init_hidden(bag_embed_tesnor.shape[1]).cuda()
            for s in range(bag_embed_tesnor.shape[0]):
                input = bag_embed_tesnor[s]
                bag_pred, state = self.clsnet['RNN'](input, state)
            loss = self.criterion(bag_pred, bag_label_tensor.view(-1, 1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())

        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save_rnn(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())

    # def train_ratio(self, epoch, trans, database):
    #     self.backbone.train()
    #     self.clsnet.train()
    #     self.logger.update_step()
    #     show_loss = []
    #
    #     # for batch_idx, (img_dirs, bag_labels) in enumerate(tqdm(self.train_loader, ascii=True, ncols=60)):
    #     #     self.logger.update_iter()
    #     pos_bag_list = self.loader_list[0]
    #     neg_bag_list = self.loader_list[1]
    #     random.shuffle(pos_bag_list)
    #     random.shuffle(neg_bag_list)
    #     #TODO: 1) the ratio of pos and neg; 2) One sample for pos bag; 3) Many samples for pos&neg to mixip
    #     #TODO: ow to sample in dataloader?
    #
    #     for ((pos_bag_labels, pos_bag_dataloader), (neg_bag_labels,neg_dataloader)) in \
    #             tqdm(zip(pos_bag_list, neg_bag_list), ascii=True, ncols=60):
    #         instance_emb = []
    #         for i in range(2):
    #             for j, img_in_bag in enumerate(pos_bag_dataloader):
    #                 if j > 2:
    #                     break
    #
    #                 # emb_in_bag = self.backbone(img_in_bag.to(self.device))
    #                 # instance_emb.append(emb_in_bag)
    #         instance_emb = torch.cat(instance_emb)
    #         instance_preds = self.clsnet(instance_emb)
    #         loss = self.criterion(instance_preds, bag_labels.view(-1, 1))
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         show_loss.append(loss.item())
    #
    #     avg_loss = sum(show_loss) / (len(show_loss))
    #     self.logger.log_scalar("loss", avg_loss, print=True)
    #     self.logger.clear_inner_iter()
    #     ##saving
    #     if self.logger.global_step % self.save_interval == 0:
    #         self.logger.save(self.backbone, self.clsnet, self.optimizer)
    #
    #     self.logger.save_result("train_mmbank", self.memory_bank.state_dict())/

    # save figures during training
    def cal_preds_in_training(self, preds_tensor, bag_index_tensor, labels_tensor, epoch):
        bag_pos_ratio = torch.stack(self.trainset.bag_pos_ratios).squeeze(-1)
        y_pred_hard = [(x > 0.5) for x in preds_tensor]
        labels_tensor = labels_tensor.numpy()
        cls_report = classification_report(labels_tensor, y_pred_hard, output_dict=False)
        auc_score = roc_auc_score(labels_tensor, preds_tensor.numpy())
        print(cls_report)
        print('AUC:', auc_score)
        print(confusion_matrix(labels_tensor, np.array(y_pred_hard)))
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
        plt.scatter(x, y, label='blue')
        plt.scatter(x, z, label='green', color="green")
        plt.legend()
        # plt.scatter(w,x)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        if not os.path.exists(os.path.join(self.logger.logdir, 'preds_figs')):
            os.makedirs(os.path.join(self.logger.logdir, 'preds_figs'))
        plt.savefig(os.path.join(self.logger.logdir, 'preds_figs', "{}.png".format(epoch)))
        # plt.show()
        plt.close()

    # showing the motivation
    def eval(self, gs, trainset):
        # self.backbone = self.logger.load_backbone_fromold(self.backbone, global_step=gs)
        # self.clsnet = self.logger.load_clsnet_fromold(self.clsnet, global_step=gs)
        self.backbone = self.logger.load_backbone(self.backbone, global_step=gs)
        self.clsnet = self.logger.load_clsnet(self.clsnet, global_step=gs)
        self.backbone.eval()
        self.clsnet.eval()
        val_loader = DataLoader(trainset, 256, shuffle=True, num_workers=0)
        preds_list = []
        bag_index_list = []
        label_list = []
        idx_x_list = []
        idx_y_list = []
        path_list = []
        with torch.no_grad():
            for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(val_loader, ascii=True, ncols=60)):
                instance_preds = torch.sigmoid(self.clsnet(self.backbone(imgs.to(self.device))))
                # instance_labels = instance_labels.to(self.device)
                preds_list.append(instance_preds)
                bag_index_list.append(bag_index)
                label_list.append(real_ins_labels)
                idx_x_list.append(inner_index[1])
                idx_y_list.append(inner_index[2])
                path_list.extend(list(inner_index[3]))

                # if batch_idx >10:
                #     break
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        bag_pos_ratio = torch.stack(trainset.bag_pos_ratios)

        rank_tensor = torch.topk(preds_tensor.squeeze(-1), len(preds_tensor))[1]
        #  f0,40 f1,35, f2 30%,f3:45%, f4 40
        ratio = 1 - 0.3
        binary_mask_tensor = (rank_tensor > ratio*len(rank_tensor)).float().cpu()
        bag_labels_in_list = [labels_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                              torch.stack(trainset.bag_pos_ratios)[i] != 0]
        bag_selection_in_list = [binary_mask_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                                 torch.stack(trainset.bag_pos_ratios)[i] != 0]
        path_in_list = [np.array(path_list)[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                                 torch.stack(trainset.bag_pos_ratios)[i] != 0]
        selection_acc_in_list = [
            (bag_selection_in_list[i] == bag_labels_in_list[i]).sum().float() / len(bag_selection_in_list[i]) for i in
            range(len(bag_selection_in_list))]

        # e mean predictions of bags
        bag_pred_ratio = torch.stack([preds_tensor[bag_index_tensor == i].mean() for i in range(bag_index_tensor.max()+1)])
        # the mean predictions of positive instance of bags
        bag_pos_pred_mean = torch.stack(
            [preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 1].mean()
             for i in range(bag_index_tensor.max() + 1)])
        # the mean predictions of negative instance of bags
        bag_neg_pred_mean = torch.stack(
            [preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 0].mean()
             for i in range(bag_index_tensor.max() + 1)])
        # the predictions of all positive instances of each bag
        bag_pos_pred = [preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 1]
             for i in range(bag_index_tensor.max() + 1)]
        # the predictions of all negative instances of each bag
        bag_neg_pred = [preds_tensor[bag_index_tensor == i].squeeze(1)[labels_tensor[bag_index_tensor == i] == 0]
             for i in range(bag_index_tensor.max() + 1)]
        # expand the mean predictions of bags as bag_pos_pred
        bag_pred_ratio_pos = torch.cat(
            [preds_tensor[bag_index_tensor == i].mean().expand_as(bag_pos_pred[i]) for i in range(bag_index_tensor.max() + 1)])
        # expand the mean predictions of bags as bag_neg_pred
        bag_pred_ratio_neg = torch.cat(
            [preds_tensor[bag_index_tensor == i].mean().expand_as(bag_neg_pred[i]) for i in
             range(bag_index_tensor.max() + 1)])
        bag_pos_pred = torch.cat(bag_pos_pred)
        bag_neg_pred = torch.cat(bag_neg_pred)

        # bag_preds_in_list = [preds_tensor[bag_index_tensor == i].squeeze(1) for i in range(bag_index_tensor.max() + 1) if torch.stack(trainset.bag_pos_ratios)[i]!=0]


        x = bag_pred_ratio.cpu().numpy() #mean preds
        y = bag_pos_pred_mean.cpu().numpy() #mean preds of pos
        z = bag_neg_pred_mean.cpu().numpy() #mean preds of neg
        w = bag_pos_ratio.cpu().numpy() # pos ratio
        #only choose pos bag
        x = x[bag_pos_ratio!=0]
        y = y[bag_pos_ratio!=0]
        z = z[bag_pos_ratio!=0]
        w = w[bag_pos_ratio!=0]
        n = np.zeros_like(x)
        p = np.ones_like(x)
        num = np.arange(len(p))

        bag_preds = np.concatenate((x, x))
        partial_instance_preds = np.concatenate((y, z)) # pos ,neg
        bag_pos_ratio = np.concatenate((w, w))
        pos_or_neg = np.concatenate((p, n))
        bag_num = np.concatenate((num, num))
        calibrated_partial_instance_preds = partial_instance_preds/bag_preds


        data = {'Bag scores': bag_preds,
                'Subbag scores': partial_instance_preds,
                'Ratio': bag_pos_ratio,
                'Label': pos_or_neg,
                'Bag index': bag_num}
        df = pd.DataFrame(data)
        # markers = {"0.0": "s", "1.0": "X"}
        sns.scatterplot(data=df, x="Bag scores", y="Subbag scores", hue='Bag index',
                        style='Label', size=np.ones_like(bag_preds), sizes=(100, 100),
                        )
        plt.savefig(os.path.join(self.logger.logdir,  "{}.png".format('ori')))
        plt.show()

        data = {'Bag scores': bag_preds,
                'Subbag scores': calibrated_partial_instance_preds,
                'Ratio': bag_pos_ratio,
                'Label': pos_or_neg,
                'Bag index': bag_num}
        df = pd.DataFrame(data)
        # markers = {"0.0": "s", "1.0": "X"}
        sns.scatterplot(data=df, x="Bag scores", y="Subbag scores", hue='Bag index',
                        style='Label', size=np.ones_like(bag_preds), sizes=(100, 100),
                        )
        plt.savefig(os.path.join(self.logger.logdir, "{}.png".format('ca')))
        plt.show()

        # bar bag preds by ratio
        plt.bar(x, w, width=0.005)
        plt.show()

        # if not os.path.exists(os.path.join(self.logger.logdir, 'preds_figs')):
        #     os.makedirs(os.path.join(self.logger.logdir, 'preds_figs'))
        # plt.savefig(os.path.join(self.logger.logdir, 'preds_figs', "{}.png".format(epoch)))
        # # plt.show()
        # plt.close()
        print('done')

    def eval_(self, gs, trainset):
        # self.backbone = self.logger.load_backbone_fromold(self.backbone, global_step=gs)
        # self.clsnet = self.logger.load_clsnet_fromold(self.clsnet, global_step=gs)
        self.backbone = self.logger.load_backbone(self.backbone, global_step=gs)
        self.clsnet = self.logger.load_clsnet(self.clsnet, global_step=gs)
        self.backbone.eval()
        self.clsnet.eval()
        val_loader = DataLoader(trainset, 256, shuffle=True, num_workers=4)
        preds_list = []
        bag_index_list = []
        label_list = []
        idx_x_list = []
        idx_y_list = []
        path_list = []
        with torch.no_grad():
            for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(val_loader, ascii=True, ncols=60)):
                instance_preds = torch.sigmoid(self.clsnet(self.backbone(imgs.to(self.device))))
                # instance_labels = instance_labels.to(self.device)
                preds_list.append(instance_preds)
                bag_index_list.append(bag_index)
                label_list.append(real_ins_labels)
                idx_x_list.append(inner_index[1])
                idx_y_list.append(inner_index[2])
                path_list.extend(list(inner_index[3]))

                # if batch_idx >10:
                #     break
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        bag_pos_ratio = torch.stack(trainset.bag_pos_ratios)
        rank_tensor = torch.topk(preds_tensor.squeeze(-1), len(preds_tensor))[1]
        idx_x_tensor = torch.cat(idx_x_list)
        idx_y_tensor = torch.cat(idx_y_list)
        #  f0,40 f1,35, f2 30%,f3:45%, f4 40
        ratio = 1 - 0.4
        binary_mask_tensor = (rank_tensor > ratio * len(rank_tensor)).float().cpu()
        bag_selection_in_list = [binary_mask_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                                 torch.stack(trainset.bag_pos_ratios)[i] != 0]
        bag_labels_in_list = [labels_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                              torch.stack(trainset.bag_pos_ratios)[i] != 0]

        path_in_list = [np.array(path_list)[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                        torch.stack(trainset.bag_pos_ratios)[i] != 0]
        selection_acc_in_list = [
            (bag_selection_in_list[i] == bag_labels_in_list[i]).sum().float() / len(bag_selection_in_list[i]) for i in
            range(len(bag_selection_in_list))]
        idx_xs = [idx_x_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                  torch.stack(trainset.bag_pos_ratios)[i] != 0]
        idx_ys = [idx_y_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
                  torch.stack(trainset.bag_pos_ratios)[i] != 0]
        values, indices = torch.topk(torch.stack(selection_acc_in_list), len(selection_acc_in_list))
        for i in range(len(indices)):
            chosen_idx = indices[i].item()
            bag_len = len(bag_labels_in_list[chosen_idx])
            #     print('baglen',len(bag_labels_in_list[chosen_idx]))
            one_path = path_in_list[chosen_idx][0].split('/')[-2]
            #     print(one_path, path_in_list[chosen_idx][0])
            #     break
            chosen_x = idx_xs[chosen_idx][bag_selection_in_list[chosen_idx] == 1]
            chosen_y = idx_ys[chosen_idx][bag_selection_in_list[chosen_idx] == 1]
            real_x = idx_xs[chosen_idx][bag_labels_in_list[chosen_idx] == 1]
            real_y = idx_ys[chosen_idx][bag_labels_in_list[chosen_idx] == 1]
            # chosen_x ,chosen_y, real_x, real_y
            tmp = np.zeros((idx_xs[chosen_idx].max() + 2, idx_ys[chosen_idx].max() + 2))
            tmp[chosen_x, chosen_y] = 1
            from PIL import Image
            import matplotlib.pyplot as plt
            A = Image.fromarray(np.uint8(tmp) * 255)
            tmp = np.zeros((idx_xs[chosen_idx].max() + 2, idx_ys[chosen_idx].max() + 2))
            tmp[real_x, real_y] = 1
            B = Image.fromarray(np.uint8(tmp) * 255)
            tmp = np.zeros((idx_xs[chosen_idx].max() + 2, idx_ys[chosen_idx].max() + 2))
            tmp[idx_xs[chosen_idx], idx_ys[chosen_idx]] = 1
            C = Image.fromarray(np.uint8(tmp) * 255)
            img_list = [A, B]
            plt.figure()
            import os
            epoch=gs
            if not os.path.exists(os.path.join(self.logger.logdir, 'pred_figs', str(epoch))):
                os.makedirs(os.path.join(self.logger.logdir, 'pred_figs', str(epoch)))
            for j in range(1, 3):
                plt.subplot(1, 2, j)
                plt.imshow(img_list[j - 1])
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(self.logger.logdir, 'pred_figs', str(epoch), "{}.png".format(
                    str(epoch) + '_' + str(i) + '_' + str(bag_len) + '_' + str(chosen_idx) + one_path)))
            print('{}'.format((str(epoch) + '_' + str(i) + '_' + str(bag_len) + '_' + str(chosen_idx) + one_path)))
            plt.show()  # real vs learned, name:epoch+rank+bag_len+bag_num


    def train_full_nct(self, epoch, configs):
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
        for batch_idx, (imgs, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = real_ins_labels.to(self.device)
            # print(instance_preds.shape, instance_labels.shape)
            loss = self.criterion(instance_preds, instance_labels)

            ## update memory bank
            # self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            loss.backward()
            self.optimizer.step()
            # print(torch.argmax(instance_preds, dim=1), instance_labels)
            acc = (torch.argmax(instance_preds, dim=1) == instance_labels).sum().float() / len(
                instance_labels)
            losses.update(loss.item(), imgs.size(0))
            ACC.update(acc, imgs.size(0))
            show_loss.append(loss.item())
            preds_list.append(
                
            )
            # bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            if batch_idx % 1000 == 0:
                progress.display(batch_idx)
            # if  batch_idx > 10:
            #     break

        # print info
        preds_tensor = torch.cat(preds_list)
        labels_tensor = torch.cat(label_list)
        cls_report = classification_report(preds_tensor.numpy(), labels_tensor.numpy(), output_dict=True)
        # auc_score = roc_auc_score(labels_tensor, preds_tensor.numpy())
        print(cls_report)
        # print('AUC:', auc_score)
        print(confusion_matrix(preds_tensor, labels_tensor.numpy()))
        avg_loss = sum(show_loss) / (len(show_loss))
        self.logger.log_scalar("loss", avg_loss, print=True)
        self.logger.clear_inner_iter()
        if self.lrsch is not None:
            if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
                self.lrsch.step(avg_loss)
                print('lr changs wrongly')
            else:
                self.lrsch.step()
                print('lr changs wrongly')

        ##after epoch memory bank operation
        # self.memory_bank.Gaussian_smooth()
        # self.memory_bank.update_rank()
        # self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        # self.logger.save_result("train_mmbank", self.memory_bank.state_dict())