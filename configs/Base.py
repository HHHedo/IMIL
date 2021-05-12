from __future__ import absolute_import
import os, sys
sys.path.append("../")
from utils.logger import Logger
from data.BaseHis import BaseHis
from runners.BaseTester import BaseTester
from runners.BaseTrainer import BaseTrainer
from models.BaseClsNet import BaseClsNet
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
    log_dir = "./experiments/test"
    logger = Logger(log_dir)
    resume = -1
    ## dataset configs
    root = "/remote-home/gyf/DATA/HISMIL/Phase2/ratio_0.7_nonor/"
    train_root = os.path.join(root, "train")
    test_root  = os.path.join(root, "test")

    train_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor()
    ])
    
    label_dict = {"gene":1, "nodule":1, "normal":0}

    trainset = BaseHis(train_root, train_transform, None, label_dict)
    testset = BaseHis(test_root, test_transform, None, label_dict)

    ## training configs: build trainer | build memory bank
    batch_size = 16
    epochs = 25
    learning_rate = 0.0001
    weight_decay = 1e-5
    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    backbone = ResNet18(False)
    net = BaseClsNet(backbone, 1).cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lrsch = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.1)
    criterion = BCEWithLogitsLoss()

    net, optimizer, lrsch, criterion = logger.load(net, optimizer, lrsch, criterion, resume)
    train_mmbank = TensorMemoryBank(trainset.bag_num, trainset.max_ins_num, trainset.bag_lengths, 0.75)
    train_mmbank.load(os.path.join(logger.logdir, "train_mmbank"), resume)
    ###wrap up a trainer
    trainer = BaseTrainer(net, optimizer, lrsch, criterion, train_loader, train_mmbank, 1, logger)

    ## testing configs: build tester | build memory bank
    test_mmbank = TensorMemoryBank(testset.bag_num, testset.max_ins_num, testset.bag_lengths, 0.75)
    test_mmbank.load(os.path.join(logger.logdir, "test_mmbank"), resume)

    tester = BaseTester(net, testset, test_mmbank, logger)
    
    def __init__(self):
        pass


    def update(self, args):
        """
        parse the arguments and update the config.
        could be modified.
        """
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
        


        







    