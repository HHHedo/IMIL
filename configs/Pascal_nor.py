from __future__ import absolute_import
import os, sys

sys.path.append("../")
from utils.logger import Logger
from data.Pascal import VOCDataset
from runners.BaseTester import BaseTester
from runners.BaseTrainer import BaseTrainer
from models.BaseClsNet import BaseClsNet
from models.backbones.ResNet import ResNet18, ResNet34, ResNet50
from utils.MemoryBank import SPCETensorMemoryBank, PBTensorMemoryBank
from losses.BCEWithLogitsLoss import BCEWithLogitsLoss
from losses.multilabel import Multilabel_categorical_crossentropy as MCC
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

import redis
from utils.utility import GaussianBlur

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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    test_transform = transforms.Compose([
        transforms.Resize((128)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        normalize
    ])

    batch_size = 64  # 64 for training, 256 for figure generation

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
    semi_ratio = None

    def __init__(self, args):
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
            self.database = redis.Redis(host='localhost', port=6379)

        if self.config == 'DigestSegAMIL':
            self.batch_size = 1

        ##task configuration
        self.label_dict = None

        self.build_logger(self)

        self.build_data(self)

        self.build_model(self)

        self.build_criterion(self)

        self.build_optimizer(self)

        self.load_model_and_optimizer(self)

        self.build_memoryBank(self)

        self.build_runner(self)

    def build_logger(self):
        self.logger = Logger(self.log_dir)

    def build_data(self):
        root = '/remote-home/share/DATA/VOCdevkit/VOC2007/JPEGImages'
        self.trainset = VOCDataset(root, self.target_cls, 'trainval', self.train_transform)
        self.testset = VOCDataset(root, self.target_cls, 'test', self.test_transform)
        self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=self.workers)
        self.test_loader = DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=self.workers)
        self.train_loader_list = []
        self.test_loader_list = []
        # only for eval alone
        self.valset = VOCDataset(root, self.target_cls, 'trainval', self.test_transform)
        self.val_loader = DataLoader(self.valset, self.batch_size, shuffle=False, num_workers=self.workers)

    def build_model(self):
        ## 2. build model
        self.old_backbone = self.build_backbone(self.backbone).to(self.device)
        self.old_clsnet = BaseClsNet(self.old_backbone, 2).to(self.device)
        # backbone
        self.backbone = self.build_backbone(self.backbone).to(self.device)
        self.clsnet = BaseClsNet(self.backbone, 20).to(self.device)

    def build_criterion(self):
        # self.pos_weight = (len(self.trainset.instance_labels) /
        #                    (torch.stack(self.trainset.instance_labels).sum())
        #                    ) ** 0.5
        # self.criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        # print(self.pos_weight)
        self.criterion = BCEWithLogitsLoss()
        # self.criterion = MCC()

    def build_optimizer(self):
            self.optimizer = optim.Adam([
                {'params': self.backbone.parameters()},
                {'params': self.clsnet.parameters()},
            ], lr=self.lr, weight_decay=self.weight_decay)

    def load_model_and_optimizer(self):
        ## 5. load and build trainer
        # load all BB+CLS+OPIM
        self.backbone, self.clsnet, self.optimizer = self.logger.load(self.backbone,
                                                                      self.clsnet,
                                                                      self.optimizer,
                                                                      self.resume)
        # only load BB and fixed
        if self.load > 0:
            self.old_backbone = self.logger.load_backbone(self.old_backbone, self.load, self.load_path)
            self.old_clsnet = self.logger.load_clsnet(self.old_clsnet, self.load, self.load_path)

    def build_memoryBank(self):
        # 6. Build & load Memory bank
        # two-stage MIL don't need mb, thus anyone is OK.
        if self.config == 'DigestSeg':
            self.train_mmbank = SPCETensorMemoryBank(self.trainset.bag_num,
                                                     self.trainset.max_ins_num,
                                                     self.trainset.bag_lengths,
                                                     self.trainset.cls_num,
                                                     self.mmt)
            self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
            self.test_mmbank = SPCETensorMemoryBank(self.testset.bag_num,
                                                    self.testset.max_ins_num,
                                                    self.testset.bag_lengths,
                                                    self.trainset.cls_num,
                                                    0.0)
            if self.config == 'DigestSeg':  # AMIL no loading
                self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)

        elif self.config == self.config == 'DigestSegTOPK' \
                or self.config == 'DigestSegEMCAV2':

            self.train_mmbank = PBTensorMemoryBank(self.trainset.bag_num,
                                                   self.trainset.max_ins_num,
                                                   self.trainset.bag_lengths,
                                                   self.trainset.cls_num,
                                                   self.mmt,
                                                   self.trainset.instance_in_which_bag,
                                                   self.trainset.instance_in_where,
                                                   None,
                                                   None,
                                                   self.trainset.bag_pos_ratios,
                                                   2,
                                                   )
            self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
            self.test_mmbank = PBTensorMemoryBank(self.testset.bag_num,
                                                  self.testset.max_ins_num,
                                                  self.testset.bag_lengths,
                                                  self.trainset.cls_num,
                                                  0.0,
                                                  self.testset.instance_in_which_bag,
                                                  self.testset.instance_in_where,
                                                  None,
                                                  None,
                                                  self.testset.bag_pos_ratios,
                                                  2
                                                  )
            self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)

    def build_runner(self):
        # 7. Buil trainer and tester
        self.trainer = BaseTrainer(self.backbone, self.clsnet, self.optimizer, self.lrsch, self.criterion,
                                   self.train_loader, self.trainset, self.train_loader_list, self.valset,
                                   self.val_loader,
                                   self.train_mmbank, self.save_interval,
                                   self.logger, self.config, self.old_backbone, self.old_clsnet)
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

