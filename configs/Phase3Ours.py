from __future__ import absolute_import
import os, sys
sys.path.append("../")
from utils.logger import Logger
from data.BaseHis import BaseHis
from runners.BaseTester import BaseTester
from runners.BaseTrainer import BaseTrainer
from models.BaseClsNet import BaseClsNet
from models.backbones.ResNet import ResNet18, ResNet34, ResNet50
from utils.MemoryBank import MRCETensorMemoryBank
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
    log_dir = "/remote-home/gyf/project/HisMIL/experiments/HISPhase3_Gene/CE+HVFlip+CJ0.25/2020_03_13_0/"
    resume = -1
    ## dataset configs
    data_root = "/remote-home/gyf/DATA/HISMIL/Phase3/ratio_0.7/"

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
    task = "gene"
    batch_size = 16
    ## training configs
    backbone = "res50"
    epochs = 150
    lrsch = "ReduceOnLoss"
    learning_rate = 1e-4
    weight_decay = 1e-5
    rce = None
    save_interval = 1

    ##memory bank config
    mmt = 0.75
    
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
        self.label_dict = self.parse_task(self.task)
        ##build logger
        self.logger = Logger(self.log_dir)
        ##build dataset
        train_root = os.path.join(self.data_root, "train")
        test_root  = os.path.join(self.data_root, "test")
        self.trainset = BaseHis(train_root, self.train_transform, None, self.label_dict)
        self.testset = BaseHis(test_root, self.test_transform, None, self.label_dict)
        self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True)
        ##build model
        self.backbone = self.build_backbone(self.backbone)
        self.net = BaseClsNet(self.backbone, 1).to(self.device)
        ##build optimizer
        self.optimizer = optim.Adam(self.net.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        if "ReduceOnLoss" in self.lrsch:
            self.lrsch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        else:
            self.lrsch = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                    milestones=[5, 10, 15, 20], 
                                                    gamma=0.1)
        ##build loss function
        self.criterion = BCEWithLogitsLoss()
        ##load and build trainer
        self.net, self.optimizer, self.lrsch, self.criterion = self.logger.load(self.net, 
                                                                                self.optimizer,
                                                                                self.lrsch,
                                                                                self.criterion, 
                                                                                self.resume)
        self.train_mmbank = MRCETensorMemoryBank(self.trainset.bag_num, 
                                             self.trainset.max_ins_num,
                                             self.trainset.bag_lengths,
                                             self.mmt)
        self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
        self.trainer = BaseTrainer(self.net, self.optimizer, self.lrsch, self.criterion, 
                                   self.train_loader, self.train_mmbank, self.save_interval,
                                   self.logger, self.rce)
        ##build tester
        self.test_mmbank = MRCETensorMemoryBank(self.testset.bag_num, 
                                            self.testset.max_ins_num, 
                                            self.testset.bag_lengths, 
                                            0.0)
        self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)
        self.tester = BaseTester(self.net, self.testset, self.test_mmbank, self.logger)

    @classmethod
    def parse_task(self, task_str):
        if "gene" in task_str:
            return {"gene": 1, "nodule": 0, "normal": 0}
        elif "nodule" in task_str:
            return {"gene": 1, "nodule": 1, "normal": 0}
        
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
        