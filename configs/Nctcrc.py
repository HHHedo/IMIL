from __future__ import absolute_import
import os, sys

sys.path.append("../")
from utils.logger import Logger
from data.Camelyon import Camelyon
from data.DigestSeg import DigestSeg
from data.DigestSegBag import DigestSegBag, DigestSegBagRatio
from runners.BaseTester import BaseTester
from runners.BaseTrainer import BaseTrainer
from models.BaseClsNet import BaseClsNet
from models.AttentionClsNet import AttentionClsNet, GAttentionClsNet
from models.backbones.ResNet import ResNet18, ResNet34, ResNet50
from models.RNN import rnn_single as RNN
from utils.MemoryBank import SPCETensorMemoryBank, CaliTensorMemoryBank, RCETensorMemoryBank, PBTensorMemoryBank
from losses.BCEWithLogitsLoss import BCEWithLogitsLoss
import torch.nn as nn
from losses.CenterLoss import CenterLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data.EMDigestSeg import EMDigestSeg
import torchvision.transforms as transforms
import torch.optim as optim
import torch
from utils.utility import init_all_dl, init_pn_dl
# torch.multiprocessing.set_sharing_strategy('file_system')

# from scripts import redis_script
import random
import pickle
from utils.utility import RandomApply


class Config(object):
    ## device config
    device = torch.device("cuda")

    ## logging/loading configs
    log_dir = ""
    resume = -1
    save_interval = 1

    ## dataset configs
    data_root = ""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #############
    train_transform  = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
        transforms.ToTensor(),
        normalize
    ])
    test_transform  = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    batch_size = 64

    ## training configs
    backbone = "res18"
    epochs = 30
    lr = 1e-3
    lrsch = None
    weight_decay = 0
    # PB
    noisy = False
    ignore_step = 0.05
    ignore_goon = True
    ignore_thres = 0.95
    ##memory bank config
    mmt = 0.90

    ## calibrated loss
    ignore_ratio = 0.2
    stop_epoch = 20
    # for aggnet like RNN and Twostage, the bag_len should be at least 10
    bag_len_thres = 9
    ssl = False

    def __init__(self, args):
        print('init')
        self.update(args)

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
        # database
        if self.database:
            self.database = redis_script.Redis(host='localhost', port=6379)

        if self.config == 'DigestSegAMIL':
            self.batch_size = 1

        ##task configuration
        self.label_dict = None
        print('build_logger')
        self.build_logger(self)
        print('build_data')
        self.build_data(self)
        print('build_model')
        self.build_model(self)
        print('build_criterion')
        self.build_criterion(self)
        print('build_optimizer')
        self.build_optimizer(self)
        print('load_model_and_optimizer')
        self.load_model_and_optimizer(self)
        print('build_memoryBank')
        self.build_memoryBank(self)
        print('build_runner')
        self.build_runner(self)

    def build_logger(self):
        self.logger = Logger(self.log_dir)

    def build_data(self):
        ## 1. build dataset & dataloader
        # import sys
        # import torch
        # from torch.utils.data import dataloader
        # from torch.multiprocessing import reductions
        # from multiprocessing.reduction import ForkingPickler

        # default_collate_func = dataloader.default_collate

        # def default_collate_override(batch):
        #     dataloader._use_shared_memory = False
        #     return default_collate_func(batch)

        # setattr(dataloader, 'default_collate', default_collate_override)

        # for t in torch._storage_classes:
        #     if sys.version_info[0] == 2:
        #         if t in ForkingPickler.dispatch:
        #             del ForkingPickler.dispatch[t]
        #     else:
        #         if t in ForkingPickler._extra_reducers:
        #             del ForkingPickler._extra_reducers[t]


        train_root = os.path.join(self.data_root, "NCT-CRC-HE-100K")
        self.trainset = ImageFolder(root=train_root, transform=self.train_transform)
        self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=self.workers)
        test_root = os.path.join(self.data_root, "CRC-VAL-HE-7K")
        self.testset = ImageFolder(root=test_root, transform=self.test_transform)
        self.test_loader = DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=self.workers)
        self.valset = ImageFolder(root=train_root, transform=self.test_transform)
        self.val_loader = DataLoader(self.valset, self.batch_size, shuffle=False, num_workers=self.workers)
        self.train_loader_list = []
        self.test_loader_list = []

    def build_model(self):
        ## 2. build model
        self.backbone = self.build_backbone(self.backbone).to(self.device)
        self.clsnet = BaseClsNet(self.backbone, 9).to(self.device)

    def build_criterion(self):
        self.criterion = nn.CrossEntropyLoss()


    def build_optimizer(self):
        self.optimizer = optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.clsnet.parameters()}
        ], lr=self.lr, weight_decay=self.weight_decay)

    def load_model_and_optimizer(self):
        ## 5. load and build trainer
        # load all BB+CLS+OPIM
        self.backbone, self.clsnet, self.optimizer = self.logger.load(self.backbone,
                                                                      self.clsnet,
                                                                      self.optimizer,
                                                                      self.resume)

    def build_memoryBank(self):
        self.train_mmbank= None
        self.test_mmbank =None
    def build_runner(self):
        # 7. Buil trainer and tester
        self.trainer = BaseTrainer(self.backbone, self.clsnet, self.optimizer, self.lrsch, self.criterion,
                                   self.train_loader, self.trainset, self.train_loader_list, self.val_loader,
                                   self.train_mmbank, self.save_interval,
                                   self.logger, self.config)
        self.tester = BaseTester(self.backbone, self.clsnet, self.test_loader, self.testset, self.test_loader_list,
                                 self.test_mmbank, self.logger)

    @classmethod
    def parse_task(self, task_str):
        return {"pos": 1, "neg": 0}

    @classmethod
    def build_backbone(self, backbone_type):
        if backbone_type.startswith("res"):
            if backbone_type.endswith("18"):
                return ResNet18(self.pretrained)
            elif backbone_type.endswith("34"):
                return ResNet34(self.pretrained)
            elif backbone_type.endswith("50"):
                return ResNet50(self.pretrained)
        else:
            return ResNet50(self.pretrained)

