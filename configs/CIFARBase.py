from __future__ import absolute_import
import os, sys
sys.path.append("../")
from utils.logger import Logger
from data.CIFARMIL import CIFARMIL
from runners.BaseTester import BaseTester
from runners.BaseTrainer import BaseTrainer
from runners.MultiTaskTester import MultiTaskTester
from runners.MultiTaskTrainer import MultiTaskTrainer
from models.BaseClsNet import BaseClsNet
from models.MultiHeadClsNet import MultiHeadClsNet
from models.backbones.ResNet import ResNet18, ResNet34, ResNet50
from utils.MemoryBank import TensorMemoryBank
from losses.BCEWithLogitsLoss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Config(object):
    ## logging/loading configs
    logger = Logger("/remote-home/gyf/project/HisMIL/experiments/CIFAR/debug/test_2020_02_10/")
    resume = -1
    tasks_num = 2
    ## dataset configs
    trainset = CIFARMIL(train=True)
    testset = CIFARMIL(train=True)
    ## training configs: build trainer | build memory bank
    batch_size = 16
    epochs = 25
    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    backbone = ResNet18(False)
    net = MultiHeadClsNet(backbone, 1, 2).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    lrsch = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.1)
    criterion = BCEWithLogitsLoss()

    batch_size = 16
    epochs = 25
    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    backbone = ResNet18(False)
    net = MultiHeadClsNet(backbone, 1, 2).cuda()
    ###to change
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    lrsch = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.1)
    criterion = BCEWithLogitsLoss()

    net, optimizer, lrsch, criterion = logger.load(net, optimizer, lrsch, criterion, resume)
    train_mmbanks = []
    for _ in range(tasks_num):
        train_mmbanks.append(TensorMemoryBank(trainset.bag_num, trainset.max_ins_num, trainset.bag_lengths, 0.75))
    ###wrap up a trainer
    trainer = MultiTaskTrainer(net, optimizer, lrsch, criterion, train_loader, train_mmbanks, 1, logger)

    ## testing configs: build tester | build memory bank
    test_mmbanks = []
    for _ in range(tasks_num):
        test_mmbanks.append(TensorMemoryBank(testset.bag_num, testset.max_ins_num, testset.bag_lengths, 0.75))
    tester = MultiTaskTester(net, testset, test_mmbanks, logger)
    def __init__(self):
        pass


    def update(self, args):
        """
        parse the arguments and update the config.
        could be modified.
        """
        pass