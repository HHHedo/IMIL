#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
utils functions
'''
from __future__ import absolute_import
import torch
import os, sys
import math
sys.path.append('..')
import time
import numpy as np
from functools import reduce
from data.DigestSegBag import DigestSegIns
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter
import random

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('epoch:{}, lr:{}'.format(epoch, lr))
def count_params(model):
    if issubclass(model.__class__, torch.nn.Module):
        return sum(reduce(lambda x,y: x*y, p.size()) for p in model.parameters() if p.requires_grad)
    else:
        return reduce(lambda x,y: x*y, model.size())

def bag_sample_with_replacement(all_labels, bag_num, ins_num, r, mean_bag_length, var_bag_length, target_numbers):
    """
    split from AttentionDeepMIL/dataloader.py _create_bags()
    This function generate bag by sampling an instance list with replacement.

    Notes:
        1. Sampling rule:
            <a> the bag length is normal distribution.
            <b> the total number of bag is fixed.
            <c> with replacement: one instance could be sampled more than once.
        2. Bag generation rule:
            instance label is from [0~N]
            target_number is a list or single number from [0,N]
            pos bag is a bag that contains numbers in target_number
            
    Args:
        all_labels: (list/torch.tensor) list of the label of instances
        bag_num: (int) how many bags to sample from instance list.
        ins_num: (int) max instance number to sample (could be 
            different from `len(all_imgs)` but should be smaller than it.)
        r: (np.random.RandomState) random number generator used.
        mean_bag_length: (int)
        var_bag_length: (int)
        target_number: (int or list) see Bag generation rule.
    """
    bag_idx_list = []
    bag_label_list = []
    ## wrap single number into list
    if not isinstance(target_numbers, list):
        target_numbers = [target_numbers]

    for i in range(bag_num):
        bag_length = np.int(r.normal(mean_bag_length, var_bag_length, 1))
        if bag_length < 1:
            bag_length = 1
            indices = torch.LongTensor(r.randint(0, ins_num, bag_length))
        labels_in_bag = torch.zeros_like(all_labels[indices])
        for target_number in target_numbers:
            labels_in_bag += (labels_in_bag == target_number) 
        labels_in_bag = (labels_in_bag > 0)
        bag_idx_list.append(indices)
        bag_label_list.append(labels_in_bag)
    return bag_idx_list, bag_label_list

def pos_instance_mask(target_numbers, instance_labels):
    """
    Transfer original instance label into index mask.

    Args:
        target_numbers: (list of int)
        instance_labels: (torch.tensor(M, ))
    Returns:
        pos_index: (torch.tensor(M, dtype=bool))
    """
    pos_index = torch.zeros_like(instance_labels).bool()
    for i in target_numbers:
        pos_index += torch.eq(instance_labels, i)
    return pos_index

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    # for i in labels:
    #     if i == 2:
    #         print(i)
    return y[labels]            # [N,D]


def modifiedRCEWeight(bag_index, inner_index, ranks, nodule_ratios, dictionary=None, preds=None):
    """Modifed RCE weighting function by ltc.
    The core motivation is:
        1. nodule_ratio might be a noisy value, therefore we average it
            with model prediction.
        2. not only consider relative rank, but also consider the absolute 
            prediction value.
    """
    weights = torch.ones_like(ranks).float()
    threshold = torch.stack((1 - nodule_ratios, ((dictionary<0.5) & (dictionary>=0)).float().mean(dim=1)[bag_index])).max(dim=0)[0]
    
    mask = (ranks < threshold) & (preds[:, 1].cpu().detach() <0.5) & pos_mask

    weights[nodule_ratios<1e-6] = 1
    return weights

def init_all_dl(dataloader, batch_size, shuffle, trans, database=None):
    dataloader_list = []
    for batch_idx, (img_dirs, bag_labels) in enumerate(dataloader):
        bag_labels = bag_labels
        bag_dataset = DigestSegIns(img_dirs, trans, database)
        # num_workers = min(8, math.ceil())
        bag_dataloader = DataLoader(bag_dataset, batch_size, shuffle=shuffle, num_workers=8)
        dataloader_list.append((bag_labels, bag_dataloader))
    return dataloader_list

def init_pn_dl(dataloader, batch_size, shuffle, trans, database):
    dataloader_list_pos = []
    dataloader_list_neg = []
    for batch_idx, (img_dirs, bag_labels) in enumerate(dataloader):
        bag_dataset = DigestSegIns(img_dirs, trans, database)
        bag_dataloader = DataLoader(bag_dataset, batch_size, shuffle=shuffle, num_workers=8)
        if bag_labels > 0:
            dataloader_list_pos.append((bag_labels, bag_dataloader))
        else:
            dataloader_list_neg.append((bag_labels, bag_dataloader))
    return dataloader_list_pos, dataloader_list_neg

# import torch


# from torch.autograd import Variable
# import numpy as np
# import cv2


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=1):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.0571, 0.1248, 0.0571],
                  [0.1248, 0.2725, 0.1248],
                  [0.0571, 0.1248, 0.0571]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=1, groups=self.channels)
        return x.squeeze(0)

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
