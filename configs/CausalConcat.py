from __future__ import absolute_import
import os, sys

sys.path.append("../")
from utils.logger import Logger
from data.Camelyon import Camelyon
from data.DigestSeg import DigestSeg
from data.DigestSegBag import DigestSegBag, DigestSegBagRatio
from runners.BaseTester import BaseTester
from runners.BaseTrainer import BaseTrainer
from models.BaseClsNet import BaseClsNet, CausalPredictor, CausalConClsNet
from models.AttentionClsNet import AttentionClsNet, GAttentionClsNet
from models.backbones.ResNet import ResNet18, ResNet34, ResNet50
from models.RNN import rnn_single as RNN
from utils.MemoryBank import SPCETensorMemoryBank, CaliTensorMemoryBank, RCETensorMemoryBank, PBTensorMemoryBank
from losses.BCEWithLogitsLoss import BCEWithLogitsLoss
from losses.CenterLoss import CenterLoss
from torch.utils.data import DataLoader
from data.EMDigestSeg import EMDigestSeg
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch
from utils.utility import init_all_dl, init_pn_dl
import redis
import random
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
    train_transform_C = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
        transforms.ToTensor(),
        normalize
    ])
    test_transform_C = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    batch_size = 64 # 64 for training, 256 for figure generation

    ## training configs
    backbone = "res18"
    epochs = 30
    backbone_lr = 1e-3
    cls_lr = 1e-3
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
        # print('1')
        self.build_data(self)
        # print('2')
        self.build_model(self)
        # print('3')
        self.build_criterion(self)
        # print('4')

        self.build_optimizer(self)
        # print('5')
        self.load_model_and_optimizer(self)
        # print('6')
        self.build_memoryBank(self)
        # print('7')
        self.build_runner(self)

    def build_logger(self):
        self.logger = Logger(self.log_dir)

    def build_data(self):
        ## 1. build dataset & dataloader
        train_root = os.path.join(self.data_root, "train")
        test_root = os.path.join(self.data_root, "test")
        if self.config == 'DigestSegAMIL' or self.config == 'DigestSegMaxPool' or self.config == 'DigestSegMeanPool':
            self.trainset = DigestSegBag(train_root, None, None, None)
            self.testset = DigestSegBag(test_root, None, None, None)
            self.train_loader = DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=0)
            self.test_loader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=0)
            self.train_loader_list = init_all_dl(self.train_loader, self.batch_size, shuffle=True,
                                                 trans=self.train_transform, database=self.database)
            self.test_loader_list = init_all_dl(self.test_loader, self.batch_size, shuffle=False,
                                                trans=self.test_transform, database=self.database)
        elif self.config == 'DigestSegRNN' or self.config == 'DigestSegTwostage':  # fixed_len to k
            self.trainset = DigestSegBag(train_root, None, None, None, bag_len_thres=self.bag_len_thres)
            self.testset = DigestSegBag(test_root, None, None, None, bag_len_thres=self.bag_len_thres)
            self.train_loader = DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=0)
            self.test_loader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=0)
            self.train_loader_list = init_all_dl(self.train_loader, self.batch_size, shuffle=True,
                                                 trans=self.train_transform, database=self.database)
            self.test_loader_list = init_all_dl(self.test_loader, self.batch_size, shuffle=False,
                                                trans=self.test_transform, database=self.database)
        elif self.config == 'DigestSegRatio':
            self.trainset = DigestSegBagRatio(train_root, None, None, None)
            self.testset = DigestSegBagRatio(test_root, None, None, None)
            self.train_loader = DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=0)
            self.test_loader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=0)
            self.train_loader_list = []
            self.test_loader_list = []
            # self.train_loader_list = init_pn_dl(self.train_loader, self.batch_size, shuffle=True,
            #                                      trans=self.train_transform, database=self.database)
            # self.test_loader_list = init_all_dl(self.test_loader, self.batch_size, shuffle=False,
            #                                     trans=self.test_transform, database=self.database)
        # if self.config == 'DigestSegPB':
        #     self.trainset_all = DigestSeg(train_root, self.train_transform, None, None, database=self.database)
        #     self.trainset = EMDigestSeg(train_root, self.train_transform, None, None,
        #                                    pos_select_idx=torch.ones([self.trainset_all.bag_num,
        #                                                                     self.trainset_all.max_ins_num]),
        #                                    database=self.database)
        #     self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=self.workers)
        #     self.testset = DigestSeg(test_root, self.test_transform, None, None, database=self.database)
        #     self.test_loader = DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=self.workers)
        #     self.train_loader_list = []
        #     self.test_loader_list = []
        elif self.config == 'DigestSegCASSL':
            self.trainset = DigestSeg(train_root, self.train_transform, None, None, database=self.database)
            self.min_ratios = self.trainset.min_ratios
            self.mean_ratios = self.trainset.mean_ratios
            self.testset = DigestSeg(test_root, self.test_transform, None, None, database=self.database)
            self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=self.workers)
            self.test_loader = DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=self.workers)
            self.train_loader_list = []
            self.test_loader_list = []
        elif self.config == 'DigestSegCamelyon':  # instance dataloader
            self.trainset = Camelyon(train_root, self.train_transform_C, None, None, database=self.database, train=True)
            self.min_ratios = self.trainset.min_ratios
            self.mean_ratios = self.trainset.mean_ratios
            test_root = os.path.join(self.data_root, "validation")
            self.testset = Camelyon(test_root, self.test_transform_C, None, None, database=self.database, train=False)
            self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=self.workers)
            self.test_loader = DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=self.workers)
            self.train_loader_list = []
            self.test_loader_list = []
        else:  # instance dataloader
            self.trainset = DigestSeg(train_root, self.train_transform, None, None, database=self.database)
            self.min_ratios = self.trainset.min_ratios
            self.mean_ratios = self.trainset.mean_ratios
            self.testset = DigestSeg(test_root, self.test_transform, None, None, database=self.database)
            self.train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=self.workers)
            self.test_loader = DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=self.workers)
            self.train_loader_list = []
            self.test_loader_list = []
        # only for eval alone
        self.valset = DigestSeg(train_root, self.test_transform, None, None, database=self.database)
        self.val_loader = DataLoader(self.valset, self.batch_size, shuffle=False, num_workers=self.workers)

    def build_model(self):
        ## 2. build model
        # self.old_backbone = self.build_backbone(self.backbone).to(self.device)
        self.old_backbone = None
        self.backbone = self.build_backbone(self.backbone).to(self.device)
        self.clsnet_base = BaseClsNet(self.backbone, 1).to(self.device)
        self.dic = torch.tensor(np.load(os.path.join(self.load_path, 'conf.npy')), dtype=torch.float)
        self.clsnet_causal = CausalPredictor(self.backbone, self.dic, 1).to(self.device)

    def build_criterion(self):
        # same loss should be instance twice?
        self.criterion = BCEWithLogitsLoss()
        self.bce = BCEWithLogitsLoss()

    def build_optimizer(self):
        ## 3. build optimizer
        # question: Not put backbone's parameters in optimizer & with torch.no_grad():backbone(input) 
        #                   = (param.requires_grad=False ?)
        # question2: backbone.eval()? (Try both maybe)
        self.optimizer = optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.clsnet_base.parameters(), 'lr': self.cls_lr},
            {'params': self.clsnet_causal.parameters(), 'lr': self.cls_lr}
        ], lr=self.backbone_lr, weight_decay=self.weight_decay)

    def load_model_and_optimizer(self):
        ## 5. load and build trainer
        self.backbone, self.clsnet, self.optimizer = self.logger.load(self.backbone,
                                                                      self.clsnet_base,
                                                                      self.optimizer,
                                                                      self.resume)
        self.clsnet_causal = self.logger.load_backbone(self.clsnet_causal,self.resume)
        # self.backbone = self.logger.load_backbone(self.backbone, self.load, self.load_path_causal)
        # self.old_backbone = self.logger.load_backbone(self.old_backbone, self.load, self.load_path)
        # self.clsnet = CausalPredictor(self.old_backbone, self.dic, 1).to(self.device)
        # return 0

    def build_memoryBank(self):
        # 6. Build & load Memory bank
        # two-stage MIL don't need mb, thus anyone is OK.
        if self.config == 'DigestSeg' or self.config == 'DigestSegAMIL':
            self.train_mmbank = SPCETensorMemoryBank(self.trainset.bag_num,
                                                     self.trainset.max_ins_num,
                                                     self.trainset.bag_lengths,
                                                     self.mmt)
            self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
            self.test_mmbank = SPCETensorMemoryBank(self.testset.bag_num,
                                                    self.testset.max_ins_num,
                                                    self.testset.bag_lengths,
                                                    0.0)
            if self.config == 'DigestSeg':  # AMIL no loading
                self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)

        elif self.config == 'DigestSegRCE':
            if self.noisy:
                self.noisy1 = random.uniform(-1, 1)
                self.noisy2 = random.uniform(0, 1)
                print('mean {}, min {}'.format(self.mean_ratios, self.min_ratios))
                self.mean_ratios = (self.mean_ratios + self.noisy1).clamp(max=1.0, min=1e-6)
                self.min_ratios = (self.min_ratios + self.noisy2).clamp(max=self.mean_ratios, min=0)
                print('mean {}, min {}'.format(self.mean_ratios, self.min_ratios))
            self.train_mmbank = RCETensorMemoryBank(self.trainset.bag_num,
                                                    self.trainset.max_ins_num,
                                                    self.trainset.bag_lengths,
                                                    self.mmt, self.mean_ratios, self.min_ratios)
            self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
            self.test_mmbank = RCETensorMemoryBank(self.testset.bag_num,
                                                   self.testset.max_ins_num,
                                                   self.testset.bag_lengths,
                                                   0.0)
            self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)

        elif self.config == 'DigestSegPB' or self.config == 'DigestSegEMCA' or self.config == 'DigestSegTOPK' \
                or self.config == 'DigestSegEMCAV2':
            if self.noisy:
                # noisy = torch.randn(self.trainset.bag_num)
                pos_num = (np.array(self.trainset.bag_pos_ratios) > 0).sum()
                noisy = [torch.randn([1]) for _ in range(pos_num)]
                noisy = noisy + [torch.tensor([0])] * (len(self.trainset.bag_pos_ratios) - pos_num)
                # noisy = random.uniform(-1, 1)
                self.trainset.bag_pos_ratios = [
                    ((self.trainset.bag_pos_ratios)[i] + noisy[i]).clamp(max=1.0, min=0).squeeze(-1)
                    for i in range(self.trainset.bag_num)]
                # self.trainset.bag_pos_ratios = (self.trainset.bag_pos_ratios + noisy).clamp(max=1.0, min=0)

            self.train_mmbank = PBTensorMemoryBank(self.trainset.bag_num,
                                                   self.trainset.max_ins_num,
                                                   self.trainset.bag_lengths,
                                                   self.mmt,
                                                   self.trainset.instance_in_which_bag,
                                                   self.trainset.instance_in_where,
                                                   self.trainset.instance_c_x,
                                                   self.trainset.instance_c_y,
                                                   self.trainset.bag_pos_ratios,
                                                   2,
                                                   )
            self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
            self.test_mmbank = PBTensorMemoryBank(self.testset.bag_num,
                                                  self.testset.max_ins_num,
                                                  self.testset.bag_lengths,
                                                  0.0,
                                                  self.testset.instance_in_which_bag,
                                                  self.testset.instance_in_where,
                                                  self.testset.instance_c_x,
                                                  self.testset.instance_c_y,
                                                  self.testset.bag_pos_ratios,
                                                  2
                                                  )
            self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)
        else:  # CA
            self.train_mmbank = CaliTensorMemoryBank(self.trainset.bag_num,
                                                     self.trainset.max_ins_num,
                                                     self.trainset.bag_lengths,
                                                     self.mmt,
                                                     self.ignore_ratio,
                                                     self.stop_epoch)
            self.train_mmbank.load(os.path.join(self.logger.logdir, "train_mmbank"), self.resume)
            self.test_mmbank = CaliTensorMemoryBank(self.testset.bag_num,
                                                    self.testset.max_ins_num,
                                                    self.testset.bag_lengths,
                                                    0.0)
            self.test_mmbank.load(os.path.join(self.logger.logdir, "test_mmbank"), self.resume)

    def build_runner(self):
        # 7. Buil trainer and tester
        self.trainer = BaseTrainer(self.backbone, self.clsnet_base, self.optimizer, self.lrsch, self.criterion,
                                   self.train_loader, self.trainset, self.train_loader_list, self.val_loader,
                                   self.train_mmbank, self.save_interval,
                                   self.logger, self.config, self.old_backbone, self.clsnet_causal)
        self.tester = BaseTester(self.backbone, self.clsnet_base, self.test_loader, self.testset, self.test_loader_list,
                                 self.test_mmbank, self.logger, self.old_backbone, self.clsnet_causal)

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

