from __future__ import absolute_import
import os, sys
sys.path.append("../")
from utils.logger import Logger
from data.BaseHis import BaseHis
from runners.MultiTaskTester import MultiTaskTester
from runners.MultiTaskTrainer import MultiTaskTrainer
from models.MultiHeadClsNet import MultiHeadClsNet
from models.backbones.ResNet import ResNet18, ResNet34, ResNet50
from utils.MemoryBank import TensorMemoryBank
from losses.BCEWithLogitsLoss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from losses.HierarchicalLoss import HierarchicalLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Config(object):
    ## logging/loading configs
    logger = Logger("/remote-home/gyf/project/HisMIL/experiments/HISPhase3_MTL/BACE+H10.0+HVFlip+CJ0.25/2020_03_11_1")
    resume = -1
    ## dataset configs
    root = "/remote-home/gyf/DATA/HISMIL/Phase3/ratio_0.7/"
    train_root = os.path.join(root, "train")
    test_root  = os.path.join(root, "test")

    train_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.25,0.25,0.25,0.25),
                transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor()
    ])
    
    tasks_num = 2
    ##task 0: gene; task 1: nodule
    label_dict = [{"gene":1, "nodule":0, "normal":0}, {"gene":1, "nodule":1, "normal":0}]

    trainset = BaseHis(train_root, train_transform, None, label_dict)
    testset = BaseHis(test_root, test_transform, None, label_dict)

    ## training configs: build trainer | build memory bank
    batch_size = 16
    epochs = 150
    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    backbone = ResNet50(False)
    net = MultiHeadClsNet(backbone, 1, 2).cuda()
    ###to change
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    lrsch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    inner_loss = [BCEWithLogitsLoss(), BCEWithLogitsLoss()]
    inner_weights = [1.0, 1.0]
    inter_loss = HierarchicalLoss()
    inter_weights = 10.0
    # net, optimizer, lrsch, criterion = logger.load(net, optimizer, lrsch, criterion, resume)
    train_mmbanks = []
    for _ in range(tasks_num):
        train_mmbanks.append(TensorMemoryBank(trainset.bag_num, trainset.max_ins_num, trainset.bag_lengths, 0.75))
    ###wrap up a trainer
    trainer = MultiTaskTrainer(net, optimizer, lrsch, inner_loss, inter_loss, 
                     train_loader, train_mmbanks, 1, logger, inner_weights=inner_weights, 
                     inter_weights=inter_weights, configs=1)

    ## testing configs: build tester | build memory bank
    test_mmbanks = []
    for _ in range(tasks_num):
        test_mmbanks.append(TensorMemoryBank(testset.bag_num, testset.max_ins_num, testset.bag_lengths, 0.0))
    tester = MultiTaskTester(net, testset, test_mmbanks, logger)
    
    def __init__(self):
        pass


    def update(self, args):
        pass

    @property
    def __dict__(self):
        dic = {}
        for key, value in Config.__dict__.items():
            if key.startswith("__") and key.endswith("__"):
                pass
            else:
                dic[key] = value
        
        return dic