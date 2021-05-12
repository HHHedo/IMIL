from __future__ import absolute_import
import os, sys
sys.path.append("../")
from utils.logger import Logger
from data.DigestSeg import DigestSeg
from runners.BaseTester import BaseTester
from runners.BaseTrainer import BaseTrainer
from models.BaseClsNet import BaseClsNet
from models.backbones.ResNet import ResNet18, ResNet34, ResNet50
from utils.MemoryBank import RCETensorMemoryBank
from losses.BCEWithLogitsLoss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Config(object):
    ## device config
    device = torch.device("cuda")
    ## logging/loading configs
    log_dir = "./experiments/DigestSeg/MICCAI21/Baseline/2020_12_2/"
    resume = -1
    ## dataset configs
    data_root = "/home/tclin/Phase2/ratio_0.7_wsi/"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((448, 448), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    batch_size = 64
    ## training configs
    backbone = "res18"
    epochs = 30
    learning_rate = 1e-3
    lrsch = None
    weight_decay = 0
    rce = None
    save_interval = 5

    ##memory bank config
    mmt = 0.9
    
    def __init__(self):
        pass

    def update(self, args):
        """
        parse the arguments and update the config.
        Notes:
            1. value should not be None
            2. Non-checking set, allow new key.
        """
        for key, value in vars(args).items():
            if value is not None:
                setattr(Config, key, value)
        
        self.build_all()

    @property
    def __dict__(self):
        dic = {}
        for key, value in Config.__dict__.items():
            if key.startswith("__") and key.endswith("__"):
                pass
            else:
                dic[key] = value
        
        return dic

    @classmethod
    def build_all(self):
        """
        build objects based on attributes.
        """
        ##task configuration
        self.label_dict = None
        ##build logger
        self.logger = Logger(self.log_dir)
        ##build dataset
        train_root = os.path.join(self.data_root, "train")
        test_root  = os.path.join(self.data_root, "test")
        self.trainset = DigestSeg(train_root, self.train_transform, None, None)
        self.min_ratios = self.trainset.min_ratios
        self.mean_ratios = self.trainset.mean_ratios

        # self.pos_weight = torch.tensor([self.pos_weight, 1])
        self.testset = DigestSeg(test_root, self.test_transform, None, None)
        self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=8)
        ##build model
        self.backbone = self.build_backbone(self.backbone).to(self.device)
        self.clsnet = BaseClsNet(self.backbone, 1).to(self.device)
        ##build optimizer
        self.optimizer = optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.clsnet.parameters()}
        ], lr=self.learning_rate, weight_decay=self.weight_decay)

        ##build loss function
        self.pos_weight = 1/((self.mean_ratios - self.min_ratios)/2 + (1 - self.mean_ratios))
        self.criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        ##load and build trainer
        self.backbone, self.clsnet, self.optimizer = self.logger.load(self.backbone,
                                                                      self.clsnet,
                                                                      self.optimizer,
                                                                      self.resume)
        self.train_mmbank = RCETensorMemoryBank(self.trainset.bag_num, 
                                             self.trainset.max_ins_num,
                                             self.trainset.bag_lengths,
                                             self.mmt, self.mean_ratios, self.min_ratios)
        self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
        self.trainer = BaseTrainer(self.backbone, self.clsnet, self.optimizer, self.lrsch, self.criterion,
                                   self.train_loader, self.train_mmbank, self.save_interval,
                                   self.logger, self.rce)
        ##build tester
        self.test_mmbank = RCETensorMemoryBank(self.testset.bag_num, 
                                            self.testset.max_ins_num, 
                                            self.testset.bag_lengths, 
                                            0.0)
        self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)
        self.tester = BaseTester(self.backbone, self.clsnet, self.testset, self.test_mmbank, self.logger)

    @classmethod
    def parse_task(self, task_str):
        return {"pos": 1, "neg": 0}
        
    @classmethod
    def build_backbone(self, backbone_type):
        if backbone_type.startswith("res"):
            if backbone_type.endswith("18"):
                return ResNet18(False)
            elif backbone_type.endswith("34"):
                return ResNet34(False)
            elif backbone_type.endswith("50"):
                return ResNet50(False)
        else:
            return ResNet50(False)
        