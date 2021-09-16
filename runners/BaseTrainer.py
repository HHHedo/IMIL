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
import seaborn as sns
import torch.nn.functional as F
# import utils.utility as utility
# from utils.logger import Logger
# import argparse
# from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve , average_precision_score, accuracy_score
import matplotlib.pyplot as plt
import random
# from data.EMDigestSeg import EMDigestSeg
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
                 lrsch,
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
        self.lrsch = lrsch
        self.device = device
        self.criterion = criterion
        self.train_loader = train_loader
        self.trainset = trainset
        self.loader_list = loader_list
        self.valset = valset
        self.val_loader = val_loader
        # self.train_loader = DataLoader(self.trainset, batch_sizef, shuffle=True, num_workers=num_workers)
        self.memory_bank = memory_bank
        self.save_interval = save_interval
        self.logger = logger
        self.old_backbone = old_backbone
        self.clsnet_causal = clsnet_causal
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
        softmax = nn.Softmax(dim=1)
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
                # loss = self.criterion(instance_preds, instance_labels.long())
                # digest
                #  loss = self.criterion(instance_preds, instance_labels.view(-1, 1))
                # pascal
                loss = self.criterion(instance_preds, instance_labels)


            elif self.configs == 'DigestSegRCE':
                # pos_weight is not used
                weight, _ = self.memory_bank.get_weight(bag_index, inner_index, nodule_ratios,
                                                                 preds=instance_preds.sigmoid(),
                                                                 cur_epoch=epoch,
                                                                 )
                # print(weight.shape,instance_preds.shape, instance_labels.view(-1, 1).shape, )
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

            #digest
            # acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1)==instance_labels).sum().float()/len(real_ins_labels)

            # acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1) == instance_labels).sum().float() / len(
            #     real_ins_labels)
            # acc = (torch.argmax(instance_preds, dim=1)== instance_labels).sum().float() / len(
            #     instance_labels)
            losses.update(loss.item(), imgs.size(0))

            # digest
            # ACC.update(acc, imgs.size(0))

            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            # self.memory_bank.update(bag_index, inner_index, softmax(instance_preds)[:,1], epoch)
            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())

            # digest
            # preds_list.append(instance_preds.sigmoid().cpu().detach())
            # # preds_list.append(softmax(instance_preds)[:,1].cpu().detach())
            # bag_index_list.append(bag_index.cpu().detach())
            # label_list.append(real_ins_labels.cpu().detach())

            # if batch_idx  % 1000 == 0:
            #     progress.display(batch_idx)
            #     # print(torch.ge(instance_preds.sigmoid(),0.5).float().squeeze(1).sum())
            #     print(torch.ge(softmax(instance_preds)[:,1],0.5).float().squeeze(1).sum())
            # if batch_idx >8:
            #     print(bag_index_list)
            #     int(label_list)
            #     break

        #print info
        progress.display(batch_idx)
        # preds_tensor = torch.cat(preds_list)
        # bag_index_tensor = torch.cat(bag_index_list)
        # labels_tensor = torch.cat(label_list)
        # self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        #.unsqueeze(1)
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
        # digest
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
                # instance_preds = self.clsnet(self.backbone(imgs.to(self.device)),bag_index, inner_index, None, None)
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
                self.memory_bank.update_tmp(bag_index, inner_index, instance_preds.sigmoid(), epoch) #bag-level
                # if batch_idx > 1:
                #     break

    # - Oricla
    def train_fullsupervision(self, epoch, configs):
        self.backbone.train()
        if self.old_backbone is not None:
            self.old_backbone.eval()
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
        for batch_idx, (imgs, _, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            # print(imgs.shape, real_ins_labels.shape)
            # print(bag_index, inner_index, bag_index.type(),bag_index.shape, inner_index.shape)
            # if batch_idx > 1 :
            #     break
            self.logger.update_iter()
            self.optimizer.zero_grad()
            # instance_preds, cluster_preds ,cluster_labels = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index, self.old_backbone, imgs.to(self.device))
            # instance_preds, _ = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index, self.old_backbone, imgs.to(self.device))
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            # instance_preds_old = self.clsnet_causal(self.old_backbone(imgs.to(self.device)))
            # preds = instance_preds + 0.1*instance_preds_old
            instance_labels = real_ins_labels.to(self.device)
            # print('ins_preds',instance_preds.shape,'ins_labels',instance_labels.view(-1, 1).shape)
            # cluster_labels = cluster_labels.to(self.device)
            # loss_self = self.criterion(preds, instance_labels.long()) 
            # loss_conf = self.criterion(cluster_preds ,cluster_labels.long())
            loss_self = self.criterion(instance_preds, instance_labels.view(-1, 1))
            # loss_self = self.criterion(instance_preds, instance_labels.long())
            # loss_conf = configs.CE(cluster_preds ,cluster_labels)
            # loss = loss_self +0*loss_conf
            loss = loss_self
            # print('loss:',loss.item())
            # print('loss_self: {}, loss_conf: {}'.format(loss_self, loss_conf))
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
            
                
            # if  batch_idx > 100:
            #     break

        # print info
        print('\n')
        progress.display(batch_idx)
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        # print(preds_tensor, bag_index_tensor, labels_tensor)
        # self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
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

    def train_semi(self, epoch, configs):
        self.backbone.train()
        if self.old_backbone is not None:
            self.old_backbone.eval()
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
            # instance_preds, cluster_preds ,cluster_labels = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index, self.old_backbone, imgs.to(self.device))
            # instance_preds, _ = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index, self.old_backbone, imgs.to(self.device))
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            # instance_preds_old = self.clsnet_causal(self.old_backbone(imgs.to(self.device)))
            # preds = instance_preds + 0.1*instance_preds_old
            instance_labels = semi_labels.to(self.device)
            semi_index = semi_index.to(self.device).unsqueeze(1)
            loss = self.criterion(instance_preds, instance_labels.view(-1,1), weight = semi_index) 
           
            


            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            # print(instance_preds, softmax(instance_preds), softmax(instance_preds)[1])
            # self.memory_bank.update(bag_index, inner_index, softmax(instance_preds)[:,1], epoch)
            loss.backward()
            self.optimizer.step()
            acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1) == instance_labels).sum().float() / len(
            #     instance_labels)
            # acc = (torch.argmax(instance_preds, dim=1)== instance_labels).sum().float() / len(
                instance_labels)
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
        # print(preds_tensor, bag_index_tensor, labels_tensor)
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
        # self.memory_bank.update_rank()
        self.memory_bank.update_epoch()
        ##saving
        if self.logger.global_step % self.save_interval == 0:
            self.logger.save(self.backbone, self.clsnet, self.optimizer)

        self.logger.save_result("train_mmbank", self.memory_bank.state_dict())
    # - Oricla
    def causalconcat_full(self, epoch, configs):
        self.backbone.train()
        self.clsnet.train()
        self.clsnet_causal.train()
        self.logger.update_step()
        show_loss = []
        # int info
        preds_list = []
        bag_index_list = []
        label_list = []
        ACC = AverageMeter('Acc', ':6.2f')
        losses_ins = AverageMeter('Loss', ':.4e')
        losses_bag = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses_ins, losses_bag, ACC],
            prefix="Epoch: [{}]".format(epoch))
        for batch_idx, (imgs, bag_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            # with torch.no_grad():
            #     # original_feature = self.old_backbone(imgs.to(self.device))
            #     causal_feature = self.backbone(imgs.to(self.device))
            #     # feature = torch.cat((original_feature, causal_feature), 1)
            #     feature = causal_feature
            # instance_preds, cluster_preds ,cluster_labels = self.clsnet(self.backbone(imgs.to(self.device)), bag_index, inner_index)
            feature = self.backbone(imgs.to(self.device))
            instance_preds = self.clsnet(feature)
            bag_preds,_ = self.clsnet_causal(feature, bag_index, inner_index, None, imgs)
            # bag_preds = self.clsnet_causal(feature)
            instance_labels = real_ins_labels.to(self.device)
            bag_labels = bag_labels.to(self.device)
            # cluster_labels = cluster_labels.to(self.device)
            # loss = self.criterion(instance_preds, instance_labels.view(-1, 1)) + configs.CE(cluster_preds ,cluster_labels)
            loss_ins = self.criterion(instance_preds, instance_labels.view(-1, 1))
            loss_bag = configs.bce(bag_preds, bag_labels.view(-1, 1))
            # loss_conf = configs.CE(cluster_preds ,cluster_labels)
            loss = loss_ins + loss_bag
            # print('loss:',loss.item())
            # print('loss_self: {}, loss_conf: {}'.format(loss_self, loss_conf))
            ## update memory bank
            self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch)
            loss.backward()
            self.optimizer.step()
            acc = (torch.ge(instance_preds.sigmoid(), 0.5).float().squeeze(1) == instance_labels).sum().float() / len(
                instance_labels)
            losses_ins.update(loss_ins.item(), imgs.size(0))
            losses_bag.update(loss_bag.item(), imgs.size(0))
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

    def train_bagdis(self, epoch, configs):
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
            instance_labels = bag_index.to(self.device)
            # print(torch.argmax(instance_preds, dim=1))
            # print(instance_labels)
            loss_self = self.criterion(instance_preds, instance_labels)
            loss = loss_self 
            loss.backward()
            self.optimizer.step()
            acc = (torch.argmax(instance_preds, dim=1) == instance_labels).sum().float() / len(
                instance_labels)
            losses.update(loss.item(), imgs.size(0))
            ACC.update(acc, imgs.size(0))
            show_loss.append(loss.item())
            preds_list.append(torch.argmax(instance_preds, dim=1).cpu().detach())
            # bag_index_list.append(bag_index.cpu().detach())
            label_list.append(instance_labels.cpu().detach())
                
            # if  batch_idx > 10:
            #     break

        # print info
        print('\n')
        progress.display(batch_idx)
        # print info
        preds_tensor = torch.cat(preds_list)
        labels_tensor = torch.cat(label_list)
        cls_report = classification_report(labels_tensor.numpy(),preds_tensor.numpy(), output_dict=True)
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
        # debug = True
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(self.train_loader, ascii=True, ncols=60,)):
            self.logger.update_iter()
            self.optimizer.zero_grad()
            instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
            instance_labels = instance_labels.to(self.device)

            ## Digestpath
            # loss = self.criterion(instance_preds, instance_labels.view(-1, 1))
            ## Digestpath
            loss = self.criterion(instance_preds, instance_labels, nodule_ratios.cuda())

            loss.backward()
            self.optimizer.step()
            show_loss.append(loss.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            # print(instance_labels.sum(), real_ins_labels.sum())
            #
            # if batch_idx > 1:
            # #     # import pdb
            # #     # pdb.set_trace()
            # #     # print(bag_index_list)
            # #     # print(label_list)
            #     break

        ## Digestpath
        ## print info
        if not debug:
        #     preds_tensor = torch.cat(preds_list)
        #     bag_index_tensor = torch.cat(bag_index_list)
        #     labels_tensor = torch.cat(label_list)
        #     self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        #     print('save imgaes')
            avg_loss = sum(show_loss) / (len(show_loss))
            self.logger.log_scalar("loss", avg_loss, print=True)
            self.logger.clear_inner_iter()
        #     if self.lrsch is not None:
        #         if isinstance(self.lrsch, optim.lr_scheduler.ReduceLROnPlateau):
        #             self.lrsch.step(avg_loss)
        #             print('lr changs wrongly')
        #         else:
        #             self.lrsch.step()
        #             print('lr changs wrongly')
        ## Digestpath

        ##after epoch memory bank operation
        # if (epoch+1) % 2 == 0:
        # self.EMCA_eval(epoch, configs)
        # self.memory_bank.Gaussian_smooth()
        # self.memory_bank.update_rank()
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
                # if batch_idx > 100:
                #     break
            # 2.get mean preds of each bag
            # k = self.memory_bank.ignore_num
            # if epoch >= epoch_thres:
            k = ((epoch - epoch_thres) * configs.ignore_step)
            # for i in range(len(self.memory_bank.ignore_goon)):
            #     if (self.memory_bank.ignore_goon[i] and epoch >= epoch_thres:
            #         # mask of the rest
            #         k[i] = (epoch - epoch_thres) * configs.ignore_step
            #     else:
            #         k[i] = self.memory_bank.ignore_num[i]  #defalut k=0, using all
            pos_capreds_new, selected_idx, new_labels, weight = self.memory_bank.select_calibrated(k, configs.ignore_thres, configs.ignore_step, torch.stack(self.valset.bag_labels))
            self.logger.log_string('Ignore num:')
            self.logger.log_string(self.memory_bank.ignore_num)
            self.logger.log_string('labels num')
            self.logger.log_string(selected_idx.sum())
            self.logger.log_string('weight sum')
            self.logger.log_string(weight.sum())

            # if pos_capreds_new > configs.ignore_thres and (epoch - epoch_thres) == 1:
            #     epoch_thres += 1
            # elif pos_capreds_new < configs.ignore_thres:
            #     self.memory_bank.ignore_num = k
            # else:
            #     configs.ignore_goon = False

            return selected_idx, new_labels, weight
            # 2.get mean preds of each bag
            mean_preds = self.memory_bank.get_mean().unsqueeze(-1)
            # 3.Calibration & select pos bag & select instance(not -1)
            calibrated_preds = self.memory_bank.dictionary/mean_preds
        #     calibrated_preds_minus = self.memory_bank.dictionary/mean_preds + self.memory_bank.dictionary
        #     # original preds
        #     # y = self.valset.instance_real_labels
        #     bag_accumulated_length = np.cumsum(np.array(self.valset.bag_lengths))
        #     bag_accumulated_length = np.insert(bag_accumulated_length, 0, 0)
        #     return_list = []
        #     y_list = []
        #     for idx, bag in enumerate(self.memory_bank.dictionary):
        #         if self.memory_bank.bag_pos_ratio_tensor[idx]>0:
        #             return_list.extend(list(bag[:self.valset.bag_lengths[idx]].cpu()))
        #             y_list.extend(self.valset.instance_real_labels[bag_accumulated_length[idx]:bag_accumulated_length[idx+1]])
        #     scores = return_list
        #     y = y_list
        #     # scores = self.memory_bank.to_list(self.valset.bag_lengths)
        #     AUC_ori = roc_auc_score(y, scores)
        #     fpr_o, tpr_o, thresholds_o = roc_curve(y, scores, pos_label=1)
        #     # calibrated
        #     return_list = []
        #     y_list = []
        #     for idx, bag in enumerate(calibrated_preds):
        #         if self.memory_bank.bag_pos_ratio_tensor[idx]>0:
        #             return_list.extend(list(bag[:self.valset.bag_lengths[idx]].cpu()))
        #             y_list.extend(self.valset.instance_real_labels[bag_accumulated_length[idx]:bag_accumulated_length[idx+1]])
        #     # y = self.valset.instance_real_labels
        #     scores = return_list
        #     y = y_list
        #     AUC_ca = roc_auc_score(y, scores)
        #     fpr_c, tpr_c, thresholds_c = roc_curve(y, scores, pos_label=1)
        #     # calibrated_minus
        #     return_list = []
        #     y_list = []
        #     for idx, bag in enumerate(calibrated_preds_minus):
        #         if self.memory_bank.bag_pos_ratio_tensor[idx]>0:
        #             return_list.extend(list(bag[:self.valset.bag_lengths[idx]].cpu()))
        #             y_list.extend(self.valset.instance_real_labels[bag_accumulated_length[idx]:bag_accumulated_length[idx+1]])
        #     # y = self.valset.instance_real_labels
        #     scores = return_list
        #     y = y_list
        #     AUC_ca_m = roc_auc_score(y, scores)
        #     fpr_m, tpr_m, thresholds_c = roc_curve(y, scores, pos_label=1)
        #
        #
        #     print('AUC before/after Calibration divide/minus: {}/{}/{}'.format(AUC_ori,AUC_ca, AUC_ca_m))
        #     # draw ROC
        #     plt.title('ROC curve')
        #     plt.plot(fpr_o, tpr_o, label='Before calibration')
        #     plt.plot(fpr_c, tpr_c, label='After calibration divide')
        #     plt.plot(fpr_m, tpr_m, label='After calibration minus')
        #     plt.xlabel('False positive rate')
        #     plt.ylabel('True positive rate')
        #     plt.xlim([-0.05,1.05])
        #     plt.ylim([-0.05,1.05])
        #     plt.legend(loc="lower right")
        #     plt.savefig(os.path.join(self.logger.logdir, 'preds_figs', 'ROC_figs{}.png'.format(epoch)))
        # # plt.show()
        #     plt.close()
            pos_calibrated_preds = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor>0] #postive_bag
            pos_calibrated_preds_valid = pos_calibrated_preds[pos_calibrated_preds>0] #postive_instance, ignore -1

            # ignore_num = int(configs.ignore_ratio*self.memory_bank.bag_lens[self.memory_bank.bag_pos_ratio_tensor>0].sum())
            # k = min(int((epoch / configs.stop_epoch) * ignore_num), ignore_num)
            # 4. Calculate the number of instances from positive bags, using the scheme to select positive instances
            pos_ins_num = self.memory_bank.bag_lens[self.memory_bank.bag_pos_ratio_tensor>0].sum()
            if configs.ignore_goon and epoch > epoch_thres:
                k = int((epoch - epoch_thres)*configs.ignore_step*pos_ins_num)
            else:
                k = self.memory_bank.ignore_num
            selected_idx = torch.ones_like(self.memory_bank.dictionary).cuda()
            selected_idx[self.memory_bank.dictionary == -1] = 0 # chosen all as default
            self.logger.log_string('{}%/{} Ignored samples.'.format(k/pos_ins_num.float()*100, k))
            if k != 0:
                # Choose the top-k for positive and all negative instances
                k_preds, _ = torch.topk(pos_calibrated_preds_valid, k, dim=0, largest=False)
                pos_selected = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor>0] > k_preds[-1]
                neg_selected = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor==0] >0
                selected_idx = torch.cat((pos_selected, neg_selected)).float() 
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

    def EMCA_noca_half(self, epoch, configs, epoch_thres=1):
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
            ###CA
            # mean_preds = self.memory_bank.get_mean()
            # # pos bag & not -1
            # calibrated_preds = self.memory_bank.dictionary/mean_preds
            ###NO CA
            calibrated_preds = self.memory_bank.dictionary
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

                    if pos_capreds_new > 0.5 and (epoch-epoch_thres)==1:
                        epoch_thres +=1
                    elif pos_capreds_new < 0.5:
                        self.memory_bank.ignore_num = k
                    else:
                        configs.ignore_goon = False
            return selected_idx
            # return torch.ones_like(self.memory_bank.dictionary).cuda()

        # - EM based Ca
    
    def EMCA_noca_mean(self, epoch, configs, epoch_thres=1):
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
            ###CA
            mean_preds = self.memory_bank.get_mean().mean()
            # # pos bag & not -1
            # calibrated_preds = self.memory_bank.dictionary/mean_preds
            ###NO CA
            calibrated_preds = self.memory_bank.dictionary
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
            self.logger.log_string('Mean is {}'.format(mean_preds))
            # if k != 0:
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
                if k != 0:
                        if pos_capreds_new > mean_preds and (epoch-epoch_thres)==1:
                            epoch_thres +=1
                        elif pos_capreds_new < mean_preds:
                            self.memory_bank.ignore_num = k
                        else:
                            configs.ignore_goon = False
            return selected_idx
            # return torch.ones_like(self.memory_bank.dictionary).cuda()

        # - EM based Ca

    def EMCA_noca(self, epoch, configs, epoch_thres=1):
        self.backbone.eval()
        self.clsnet.eval()
        counter_epoch = [10,9,8,11,10]
        chosen_stop_e = int(configs.data_root.split('/')[-1])
        stop_e = counter_epoch[chosen_stop_e]
        # configs.ignore_goon = True
        # epoch_thres=0
        with torch.no_grad():
            for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                    tqdm(self.val_loader, ascii=True, ncols=60)):
                instance_preds = self.clsnet(self.backbone(imgs.to(self.device)))
                self.memory_bank.update(bag_index, inner_index, instance_preds.sigmoid(), epoch) #bag-level
                # if batch_idx > 10:
                #     break
            
            calibrated_preds = self.memory_bank.dictionary
            pos_calibrated_preds = calibrated_preds[self.memory_bank.bag_pos_ratio_tensor>0] #postive_bag
            pos_calibrated_preds_valid = pos_calibrated_preds[pos_calibrated_preds>0] #postive_instance
            pos_ins_num = self.memory_bank.bag_lens[self.memory_bank.bag_pos_ratio_tensor>0].sum()

            # mean_preds = self.memory_bank.get_mean()
            # calibrated_preds = self.memory_bank.dictionary/mean_preds

            if configs.ignore_goon and epoch > epoch_thres:
                k = int((epoch - epoch_thres)*configs.ignore_step*pos_ins_num)
            else:
                k = self.memory_bank.ignore_num
            selected_idx = torch.ones_like(self.memory_bank.dictionary).cuda()
            selected_idx[self.memory_bank.dictionary == -1] = 0
            self.logger.log_string('{:.3}%/{} Ignored samples.'.format(k/pos_ins_num.float()*100, k))
            # self.logger.log_string('Mean is {}'.format(mean_preds))
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
                # if k != 0:
                #         if pos_capreds_new > mean_preds and (epoch-epoch_thres)==1:
                #             epoch_thres +=1
                
                if epoch < stop_e:
                    self.memory_bank.ignore_num = k
                else:
                    configs.ignore_goon = False
            return selected_idx
            # return torch.ones_like(self.memory_bank.dictionary).cuda()

        # - EM based Ca
    
    def EMCA_globalT(self, epoch, configs, epoch_thres=1):
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
            pos_selected = self.memory_bank.dictionary[self.memory_bank.bag_pos_ratio_tensor>0] > 0.5
            neg_selected = self.memory_bank.dictionary[self.memory_bank.bag_pos_ratio_tensor==0] > 0
            selected_idx = torch.cat((pos_selected, neg_selected)).float() # contain both positive and negative
            return selected_idx

    def EMCA_globalM(self, epoch, configs, epoch_thres=1):
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
            mean_preds = self.memory_bank.get_mean().mean()
            pos_selected = self.memory_bank.dictionary[self.memory_bank.bag_pos_ratio_tensor>0] > mean_preds
            neg_selected = self.memory_bank.dictionary[self.memory_bank.bag_pos_ratio_tensor==0] > 0
            selected_idx = torch.cat((pos_selected, neg_selected)).float() # contain both positive and negative
            return selected_idx
  

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
            ##digest
            # loss_ce = self.criterion['CE'](instance_preds, instance_labels.view(-1, 1))
            ##digest
            loss_ce = loss = self.criterion(instance_preds, instance_labels)
            # loss_center = self.criterion['Center'](bag_index, feat)
            # loss = loss_ce + 0.01*loss_center
            loss = loss_ce
            # print(loss_center)
            loss.backward()
            self.optimizer.step()
            show_loss_ce.append(loss_ce.item())
            # show_loss_center.append(loss_center.item())
            preds_list.append(instance_preds.sigmoid().cpu().detach())
            bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            # if batch_idx>1:
            #     debug = True
            #     break
        # ##digest
        # # print info
        # if not debug:
        #     preds_tensor = torch.cat(preds_list)
        #     bag_index_tensor = torch.cat(bag_index_list)
        #     labels_tensor = torch.cat(label_list)
        #     self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, epoch)
        #     avg_ce_loss = sum(show_loss_ce) / (len(show_loss_ce))
        #     # avg_center_loss = sum(show_loss_center)/ (len(show_loss_center))
        #     self.logger.log_scalar("loss", avg_ce_loss, print=True)
        #     # self.logger.log_scalar("loss", avg_center_loss, print=True)
        #     self.logger.clear_inner_iter()
        # ##digest


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
        selected_idx = self.memory_bank.select_topk(torch.stack(self.valset.bag_labels))
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

    # showing the motivation
    def eval(self, gs, trainset):
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
                # idx_x_list.append(inner_index[1])
                # idx_y_list.append(inner_index[2])
                # path_list.extend(list(inner_index[3]))

                # if batch_idx >5:
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
        # path_in_list = [np.array(path_list)[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
        #                          torch.stack(trainset.bag_pos_ratios)[i] != 0]
        # selection_acc_in_list = [
        #     (bag_selection_in_list[i] == bag_labels_in_list[i]).sum().float() / len(bag_selection_in_list[i]) for i in
        #     range(len(bag_selection_in_list))]

        acc_pos_list = []
        acc_neg_list = []
        acc_all_list = []
        for i in range(bag_index_tensor.max() + 1):
            if bag_pos_ratio[i] != 0:
                bag_preds = preds_tensor[bag_index_tensor == i].cpu().numpy()
                bag_labels = labels_tensor[bag_index_tensor == i].cpu().numpy()
                if (bag_labels == 0).sum() != 0:
                    bag_pos_preds = bag_preds[bag_labels == 1]
                    bag_neg_preds = bag_preds[bag_labels == 0]
                    acc_pos_list.append(bag_pos_preds.mean())
                    acc_neg_list.append(bag_neg_preds.mean())
                    acc_all_list.append(bag_preds.mean())
        x = np.array(acc_all_list)
        y = np.array(acc_pos_list)
        z = np.array(acc_neg_list)
        # sort by x
        idx = np.argsort(x)
        x, y, z = x[idx], y[idx], z[idx]
        # sampling 100
        tmp_idx = np.zeros_like(x)
        tmp_idx[:100] = 1
        np.random.shuffle(tmp_idx)
        x, y, z = x[tmp_idx == 1], y[tmp_idx == 1], z[tmp_idx == 1]
        bag_idx = np.arange(len(x))
        # label
        mean_pos_one = ['Positive'] * len(x)
        mean_neg_one = ['Negative'] * len(x)
        bag_preds = np.concatenate((y, z))
        bag_idx = np.concatenate((bag_idx, bag_idx))
        pos_or_neg = np.concatenate((mean_pos_one, mean_neg_one))
        calibrated_partial_instance_preds = bag_preds / np.concatenate((x, x))
        data = {'Average scores': bag_preds,
                'Bag index': bag_idx,
                'Reweighted average scores': calibrated_partial_instance_preds,
                'Instance label': pos_or_neg, }
        df = pd.DataFrame(data)
        # markers = {"0.0": "s", "1.0": "X"}
        # plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x="Bag index", y="Average scores", hue='Bag index',
                        style='Instance label', s=100
                        )
        ax = plt.gca()
        # ax.xaxis.set_ticks_position('top')
        # ax.xaxis.set_label_position('top')
        # # 把x轴的主刻度设置为1的倍数
        ax.spines['top'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')
        plt.tight_layout()
        # plt.savefig('AbAug.png', dpi=600)
        plt.xlabel('Bag index sorted by average scores of all instances', fontsize=15)
        # plt.xlabel('', fontsize=15)
        plt.ylabel('Average scores', fontsize=15)
        plt.xticks(fontsize=12)
        # plt.xticks([])
        plt.yticks(fontsize=12)
        plt.savefig(os.path.join(self.logger.logdir, "{}.png".format('ori')), dpi=600)
        # plt.legend(loc=2)
        plt.show()
        data = {'Average scores': bag_preds,
                'Bag index': bag_idx,
                'Reweighted average scores': calibrated_partial_instance_preds,
                'Instance label': pos_or_neg, }
        df = pd.DataFrame(data)
        # markers = {"0.0": "s", "1.0": "X"}
        # plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x="Bag index", y="Reweighted average scores", hue='Bag index',
                        style='Instance label', s=100, legend=False,
                        )
        ax = plt.gca()
        # ax.xaxis.set_ticks_position('top')
        # ax.xaxis.set_label_position('top')
        # # 把x轴的主刻度设置为1的倍数
        ax.spines['top'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')
        plt.tight_layout()
        # plt.savefig('AbAug.png', dpi=600)
        plt.xlabel('Bag index sorted by average scores of all instances', fontsize=15)
        # plt.xlabel('', fontsize=15)
        plt.ylabel('Reweighted average scores', fontsize=15)
        plt.xticks(fontsize=12)
        # plt.xticks([])
        plt.yticks(fontsize=12)
        plt.savefig(os.path.join(self.logger.logdir, "{}.png".format('ca')), dpi=600)
        # plt.legend(loc=2)
        plt.show()


        # recall and specifi
        tmp_label_list = [labels_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1)]
        tmp_preds_list = [preds_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1)]
        sensitivity_list = []
        specificity_list = []
        for i in range(len(tmp_preds_list)):
            if (bag_pos_ratio[i] != 0) & (bag_pos_ratio[i] != 1):
                cls_report = classification_report(np.array(tmp_label_list[i].cpu().squeeze(1)),
                                                   (np.array(tmp_preds_list[i].cpu().squeeze(1)) > 0.5).astype(int),
                                                   output_dict=True)

                sensitivity_list.append(cls_report['1.0']['recall'])
                specificity_list.append(cls_report['0.0']['recall'])
        idx = np.argsort(np.array(sensitivity_list))
        sorted_sen = np.array(sensitivity_list)[idx]
        sorted_spc = np.array(specificity_list)[idx]
        plt.plot(sorted_sen)
        plt.plot(sorted_spc)
        plt.show()

        # data = {'Bag scores': bag_preds,
        #         'Subbag scores': calibrated_partial_instance_preds,
        #         'Ratio': bag_pos_ratio,
        #         'Label': pos_or_neg,
        #         'Bag index': bag_num}
        # df = pd.DataFrame(data)
        # # markers = {"0.0": "s", "1.0": "X"}
        # sns.scatterplot(data=df, x="Bag scores", y="Subbag scores", hue='Bag index',
        #                 style='Label', size=np.ones_like(bag_preds), sizes=(100, 100),
        #                 )
        # plt.savefig(os.path.join(self.logger.logdir, "{}.png".format('ca')))
        # plt.show()
        #
        # # bar bag preds by ratio
        # plt.bar(x, w, width=0.005)
        # plt.show()

        # if not os.path.exists(os.path.join(self.logger.logdir, 'preds_figs')):
        #     os.makedirs(os.path.join(self.logger.logdir, 'preds_figs'))
        # plt.savefig(os.path.join(self.logger.logdir, 'preds_figs', "{}.png".format(epoch)))
        # # plt.show()
        # plt.close()
        # print('done')

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
            AP_list[0, i], AP_list[1, i], AP_list[2, i] , AP_list[4, i] = AP_max, AP_mean, AP_voting, AP_selected
        print(AP_list.mean(1))
        # AP_list = torch.zeros([2, preds_tensor.shape[1]])
        # for i in range(preds_tensor.shape[1]):
        #     AP_1 = average_precision_score(labels_tensor[:, i], preds_tensor[:, i])
        #
        #     AP_list[0, i], AP_list[1, i] = AP_1, AP_2
        # print(AP_list.mean(1)[0], AP_list.mean(1)[1])
        # print(preds_tensor,  bag_index_tensor, labels_tensor )
        # print(preds_tensor.shape,  bag_index_tensor.shape, labels_tensor.shape )
        # y_pred_hard = [(x > 0.5) for x in preds_tensor]
        # labels_tensor = labels_tensor.numpy()
        # cls_report = classification_report(labels_tensor, y_pred_hard, digits=4, output_dict=False)
        # auc_score = roc_auc_score(labels_tensor, preds_tensor.numpy())
        # print(cls_report)
        # print('AUC:', auc_score)
        # print(confusion_matrix(labels_tensor, np.array(y_pred_hard)))
        # bag_pos_ratio = torch.stack(trainset.bag_pos_ratios)
        # rank_tensor = torch.topk(preds_tensor.squeeze(-1), len(preds_tensor))[1]
        # idx_x_tensor = torch.cat(idx_x_list)
        # idx_y_tensor = torch.cat(idx_y_list)
        # self.cal_preds_in_training(preds_tensor, bag_index_tensor, labels_tensor, gs)
        #  f0,40 f1,35, f2 30%,f3:45%, f4 40
        # ratio = 1 - 0.4
        # binary_mask_tensor = (rank_tensor > ratio * len(rank_tensor)).float().cpu()
        # bag_selection_in_list = [binary_mask_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
        #                          torch.stack(trainset.bag_pos_ratios)[i] != 0]
        # bag_labels_in_list = [labels_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
        #                       torch.stack(trainset.bag_pos_ratios)[i] != 0]

        # path_in_list = [np.array(path_list)[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
        #                 torch.stack(trainset.bag_pos_ratios)[i] != 0]
        # selection_acc_in_list = [
        #     (bag_selection_in_list[i] == bag_labels_in_list[i]).sum().float() / len(bag_selection_in_list[i]) for i in
        #     range(len(bag_selection_in_list))]
        # idx_xs = [idx_x_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
        #           torch.stack(trainset.bag_pos_ratios)[i] != 0]
        # idx_ys = [idx_y_tensor[bag_index_tensor == i] for i in range(bag_index_tensor.max() + 1) if
        #           torch.stack(trainset.bag_pos_ratios)[i] != 0]
        # values, indices = torch.topk(torch.stack(selection_acc_in_list), len(selection_acc_in_list))
        # for i in range(len(indices)):
        #     chosen_idx = indices[i].item()
        #     bag_len = len(bag_labels_in_list[chosen_idx])
        #     #     print('baglen',len(bag_labels_in_list[chosen_idx]))
        #     one_path = path_in_list[chosen_idx][0].split('/')[-2]
        #     #     print(one_path, path_in_list[chosen_idx][0])
        #     #     break
        #     chosen_x = idx_xs[chosen_idx][bag_selection_in_list[chosen_idx] == 1]
        #     chosen_y = idx_ys[chosen_idx][bag_selection_in_list[chosen_idx] == 1]
        #     real_x = idx_xs[chosen_idx][bag_labels_in_list[chosen_idx] == 1]
        #     real_y = idx_ys[chosen_idx][bag_labels_in_list[chosen_idx] == 1]
        #     # chosen_x ,chosen_y, real_x, real_y
        #     tmp = np.zeros((idx_xs[chosen_idx].max() + 2, idx_ys[chosen_idx].max() + 2))
        #     tmp[chosen_x, chosen_y] = 1
        #     from PIL import Image
        #     import matplotlib.pyplot as plt
        #     A = Image.fromarray(np.uint8(tmp) * 255)
        #     tmp = np.zeros((idx_xs[chosen_idx].max() + 2, idx_ys[chosen_idx].max() + 2))
        #     tmp[real_x, real_y] = 1
        #     B = Image.fromarray(np.uint8(tmp) * 255)
        #     tmp = np.zeros((idx_xs[chosen_idx].max() + 2, idx_ys[chosen_idx].max() + 2))
        #     tmp[idx_xs[chosen_idx], idx_ys[chosen_idx]] = 1
        #     C = Image.fromarray(np.uint8(tmp) * 255)
        #     img_list = [A, B]
        #     plt.figure()
        #     import os
        #     epoch=gs
        #     if not os.path.exists(os.path.join(self.logger.logdir, 'pred_figs', str(epoch))):
        #         os.makedirs(os.path.join(self.logger.logdir, 'pred_figs', str(epoch)))
        #     for j in range(1, 3):
        #         plt.subplot(1, 2, j)
        #         plt.imshow(img_list[j - 1])
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.savefig(os.path.join(self.logger.logdir, 'pred_figs', str(epoch), "{}.png".format(
        #             str(epoch) + '_' + str(i) + '_' + str(bag_len) + '_' + str(chosen_idx) + one_path)))
        #     print('{}'.format((str(epoch) + '_' + str(i) + '_' + str(bag_len) + '_' + str(chosen_idx) + one_path)))
        #     plt.show()  # real vs learned, name:epoch+rank+bag_len+bag_num


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
            preds_list.append(torch.argmax(instance_preds, dim=1).cpu().detach())
            # bag_index_list.append(bag_index.cpu().detach())
            label_list.append(real_ins_labels.cpu().detach())
            if batch_idx % 1000 == 0:
                progress.display(batch_idx)
            # if  batch_idx > 10:
            #     break

        # print info
        preds_tensor = torch.cat(preds_list)
        labels_tensor = torch.cat(label_list)
        cls_report = classification_report(labels_tensor.numpy(), preds_tensor.numpy(), output_dict=True)
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


    def eval_catplot(self, gs, trainset):
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
                # idx_x_list.append(inner_index[1])
                # idx_y_list.append(inner_index[2])
                # path_list.extend(list(inner_index[3]))

                # if batch_idx >5:
                #     break
        preds_tensor = torch.cat(preds_list)
        bag_index_tensor = torch.cat(bag_index_list)
        labels_tensor = torch.cat(label_list)
        bag_pos_ratio = torch.stack(trainset.bag_pos_ratios)

        # acc_pos_list = []
        # acc_neg_list = []
        # acc_all_list = []
        # for i in range(bag_index_tensor.max() + 1):
        #     if bag_pos_ratio[i]!=0:
        #         bag_preds = (preds_tensor[bag_index_tensor == i]>0.5).cpu().numpy()
        #         bag_labels = labels_tensor[bag_index_tensor == i].cpu().numpy()
        #         if (bag_labels == 0).sum() != 0:
        #             bag_pos_preds = bag_preds[bag_labels == 1]
        #             bag_pos_labels = np.ones_like(bag_pos_preds)
        #             bag_neg_preds = bag_preds[bag_labels == 0]
        #             bag_neg_labels = np.zeros_like(bag_neg_preds)
        #             acc_pos = accuracy_score(bag_pos_labels, bag_pos_preds)
        #             acc_neg = accuracy_score(bag_neg_labels, bag_neg_preds)
        #             acc_all = accuracy_score(bag_labels, bag_preds)
        #             acc_pos_list.append(acc_pos)
        #             acc_neg_list.append(acc_neg)
        #             acc_all_list.append(acc_all)
        # pos_name = ['positive instance']*len(acc_pos_list)
        # neg_name = ['negative instance']*len(acc_neg_list)
        # all_name = ['bag level']*len(acc_all_list)
        # ACC = np.concatenate((acc_pos_list, acc_neg_list, acc_all_list))
        # names = np.concatenate((pos_name, neg_name, all_name))
        # data = {'whichAcc': names,
        #         'AccVaule': ACC}
        # df = pd.DataFrame(data)
        # sns.catplot(
        #     data=df, kind="bar",
        #     x="whichAcc", y="AccVaule",
        #     ci=95, height=6
        # )
        # plt.tight_layout()
        # ax = plt.gca()
        # ax.spines['right'].set_color('black')
        # # , alpha=.6
        # # plt.despine(left=True)
        # plt.ylabel("accuracy")
        # plt.xlabel("")
        # plt.title(gs)
        # plt.tight_layout()
        # plt.show()
        acc_pos_list = []
        acc_neg_list = []
        acc_all_list = []
        for i in range(bag_index_tensor.max() + 1):
            if bag_pos_ratio[i] != 0:
                bag_preds = (preds_tensor[bag_index_tensor == i] > 0.5).cpu().numpy()
                bag_labels = labels_tensor[bag_index_tensor == i].cpu().numpy()
                if (bag_labels == 0).sum() != 0:
                    bag_pos_preds = bag_preds[bag_labels == 1]
                    bag_neg_preds = bag_preds[bag_labels == 0]
                    acc_pos_list.append(bag_pos_preds.mean())
                    acc_neg_list.append(bag_neg_preds.mean())
                    acc_all_list.append(bag_preds.mean())
        pos_name = ['positive\ninstance'] * len(acc_pos_list)
        neg_name = ['negative\ninstance'] * len(acc_neg_list)
        all_name = ['bag\nlevel'] * len(acc_all_list)
        ACC = np.concatenate((acc_pos_list, acc_neg_list, acc_all_list))
        names = np.concatenate((pos_name, neg_name, all_name))
        data = {'whichAcc': names,
                'AccVaule': ACC}
        df = pd.DataFrame(data)
        plt.rc('font', family="Times New Roman")
        # sns.set(font='Times New Roman')
        sns.catplot(
            data=df, kind="bar",
            x="whichAcc", y="AccVaule",
            ci=95, height=6, aspect=.9
        )
        plt.tight_layout()
        ax = plt.gca()
        # ax.spines['right'].set_color('black')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')
        plt.ylabel("Score", fontsize=22)
        plt.xlabel("")
        plt.title("")
        plt.xticks(rotation=0, fontsize=20)
        # plt.xticks([])
        plt.yticks(fontsize=20)
        plt.ylim([0.5, 0.95])
        my_y_ticks = np.arange(0.5, .95, 0.1)  # 显示范围为-5至5，每0.5显示一刻度
        plt.yticks(my_y_ticks)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logger.logdir, "score.png"), dpi=600)
        plt.show()

