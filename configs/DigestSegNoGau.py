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
torch.multiprocessing.set_sharing_strategy('file_system')

import redis
import random
from utils.utility import RandomApply
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
        transforms.RandomResizedCrop(448, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop((448, 448)),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.RandomVerticalFlip(0.5),
    #     # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    test_transform = transforms.Compose([
        transforms.Resize((512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        normalize
    ])
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop((448, 448), scale=(0.2, 1.0)),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomVerticalFlip(0.5),
    #     transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # test_transform = transforms.Compose([
    #     transforms.Resize((448, 448)),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    ########
    # ssl_transform = transforms.Compose([
    #     RandomApply(
    #         transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    #         p=0.3
    #     ),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.RandomHorizontalFlip(),
    #     RandomApply(
    #         transforms.GaussianBlur((3, 3), (1.0, 2.0)),
    #         p=0.2
    #     ),
    #     transforms.RandomResizedCrop((448, 448), scale=(0.2, 1.0)),
    #     transforms.ToTensor(),
    #     normalize
    #     ])
    #############
    # train_transform_C = transforms.Compose([
    #     transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomVerticalFlip(0.5),
    #     transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # test_transform_C = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    batch_size = 64 # 64 for training, 256 for figure generation

    ## training configs
    backbone = "res18"
    epochs = 30
    lr = 1e-3
    lrsch = None
    weight_decay = 0
    #PB
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
        self.old_backbone = self.build_backbone(self.backbone).to(self.device)
        self.old_clsnet = BaseClsNet(self.old_backbone, 2).to(self.device)
        # backbone
        self.backbone = self.build_backbone(self.backbone).to(self.device)
        # head
        if self.config == 'DigestSegAMIL':
            self.clsnet = AttentionClsNet(self.backbone, 1, 128, 1).to(self.device)
        elif self.config == 'DigestSegRNN':
            self.clsnet = {'cls': BaseClsNet(self.backbone, 1).to(self.device),
                           'RNN': RNN(ndims=512).to(self.device)}
        # elif self.config == 'DigestSegAMIL':
        #     self.clsnet = AttentionClsNet(self.backbone, 1, 128, 1).to(self.device)
        else:  # instance/max/mean pooling
            self.clsnet = BaseClsNet(self.backbone, 1).to(self.device)

    def build_criterion(self):
        ## 4. build loss function
        if self.config == 'DigestSegFull':
            print("-" * 60)
            # self.pos_weight = (len(self.trainset.instance_real_labels) /
            #                    (torch.stack(self.trainset.instance_real_labels).sum())
            #                    ) ** 0.5
            # self.criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
            self.criterion = BCEWithLogitsLoss()
        elif self.config == 'DigestSegTOPK':
            self.criterion = {'CE': BCEWithLogitsLoss(),
                              'Center': CenterLoss(self.trainset.bag_num, 512)
                              }
        else:
            
            # print(len(self.trainset.instance_real_labels), (torch.stack(self.trainset.instance_real_labels).sum()))
            # self.pos_weight = (len(self.trainset.instance_real_labels) /
            #                    (torch.stack(self.trainset.instance_real_labels).sum())
            #                    ) ** 0.5
            # print(self.pos_weight)
            # self.pos_weight = (torch.stack(self.trainset.instance_labels)==0).sum().float()/(torch.stack(self.trainset.instance_labels)==1).sum().float()
            # print('neg num:{}, pos num:{}'.format((torch.stack(self.trainset.instance_labels)==0).sum(),
            #                                         (torch.stack(self.trainset.instance_labels)==1).sum()))
            # print(self.pos_weight)
            # self.criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
            self.criterion = BCEWithLogitsLoss()
        # if self.config == 'DigestSegRCE':
        #     self.pos_weight = 1 / ((self.mean_ratios - self.min_ratios) / 2 + self.min_ratios)
        #     self.criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        # elif self.config == 'DigestSegPB':
        #     self.pos_weight = 1 / self.mean_ratios
        #     self.criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        # else:
        #     self.criterion = BCEWithLogitsLoss()

    def build_optimizer(self):
        ## 3. build optimizer
        if self.config == 'DigestSegAMIL' or self.config == 'DigestSegMaxPool' or self.config == 'DigestSegMeanPool':
            self.optimizer = optim.Adam(self.clsnet.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        elif self.config == 'DigestSegRNN':
            self.optimizer = optim.Adam(self.clsnet['RNN'].parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        elif self.config == 'DigestSegTOPK':
            self.optimizer = optim.Adam([
                {'params': self.backbone.parameters()},
                {'params': self.clsnet.parameters()},
                {'params': self.criterion['Center'].parameters()}
            ], lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam([
                {'params': self.backbone.parameters()},
                {'params': self.clsnet.parameters()},
                # {'params': self.old_backbone.parameters(), 'lr': self.lr*0.05},
                # {'params': self.old_clsnet.parameters(), 'lr': self.lr*0.05}
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
            # if self.config == 'DigestSegAMIL':
            #     self.backbone = self.logger.load_BB_and_freeze(self.backbone, self.load, self.load_path)
            # # only load BB and CLS, then fixed
            # elif self.config == 'DigestSegRNN':
            #     self.backbone = self.logger.load_BB_and_freeze(self.backbone, self.load, self.load_path)
            #     self.clsnet['cls'] = self.logger.load_clsnet_and_freeze(self.clsnet['cls'], self.load, self.load_path)
            # elif  self.config == 'DigestSegMaxPool' or self.config == 'DigestSegMeanPool':
            #     self.backbone = self.logger.load_BB_and_freeze(self.backbone, self.load, self.load_path)
                # self.clsnet = self.logger.load_clsnet_and_freeze(self.clsnet, self.load, self.load_path, notFreeze=True)


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
                or self.config == 'DigestSegEMCAV2'\
                or self.config == 'DigestSegEMnocahalf' or self.config == 'DigestSegEMnocamean'\
                or self.config == 'DigestSegGT' or self.config == 'DigestSegGM':
            if self.noisy:
                # noisy = torch.randn(self.trainset.bag_num)
                pos_num = (np.array(self.trainset.bag_pos_ratios)>0).sum()
                noisy = [torch.randn([1]) for _ in range(pos_num)]
                noisy = noisy+[torch.tensor([0])]*(len(self.trainset.bag_pos_ratios)-pos_num)
                # noisy = random.uniform(-1, 1)
                self.trainset.bag_pos_ratios = [((self.trainset.bag_pos_ratios)[i] + noisy[i]).clamp(max=1.0, min=0).squeeze(-1)
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
        self.trainer = BaseTrainer(self.backbone, self.clsnet, self.optimizer, self.lrsch, self.criterion,
                                   self.train_loader, self.trainset, self.train_loader_list, self.valset, self.val_loader,
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
        
