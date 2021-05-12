from __future__ import absolute_import
import os
from tqdm import tqdm
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
from torchvision.models import resnet18
from data.Camelyon import Camelyon
import argparse
from importlib import import_module
from utils.utility import adjust_learning_rate
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from torch.utils.data import DataLoader
import numpy as np


##epoch #############################ratio
### log_dir and data_root
def parse_args():
    parser = argparse.ArgumentParser(description='317 MIL Framework')
    parser.add_argument("--task", type=str, default="Digest",
                        help="Which task to perform",
                        choices=["Digest", "Camelyon"])
    parser.add_argument('--config', type=str, default='DigestSegFull',
                        help="Config file to use.",
                        choices=["DigestSegEMCAV2", "DigestSeg", 'DigestSegPB',
                                 'DigestSegTOPK', 'DigestSegFull', 'DigestSegRCE'])
    parser.add_argument("--log_dir", type=str,
                        default="/home/ltc/1T/HisMIL/experiments/Full/2020_12_26/f4",
                        help="The experiment log directory")
    parser.add_argument("--data_root", "-d", type=str,
                        default="/home/ltc/Phase2/5_folder/4",
                        help="root directory of data used")
    parser.add_argument("--resume", type=int, default=-1, help="Resume epoch")
    parser.add_argument("--bs", type=int, default=256, help="Resume epoch")
    parser.add_argument("--backbone", type=str, default='res18',
                        help="which backbone to use.")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Whether to use weighted BCE loss")

    parser.add_argument('-lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="How many epochs to train")
    parser.add_argument('--cos', action='store_true', default=True,
                        help='use cosine lr schedule')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    # choose loss
    # parser.add_argument("--rce", action="store_true", default=False, help="Whether to use weighted BCE loss")
    # Calibration loss
    parser.add_argument("--stop_epoch", type=int, default=-1, help="stop")
    parser.add_argument("--ignore_thres", type=float, default=0.95, help="ignore")
    # one-stage or two stage training
    parser.add_argument("--database", action="store_true", default=False, help="Using database")
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument("--load", type=int, default=50, help="the epoch to be loaded")
    parser.add_argument("--load_path", type=str,
                        default="/home/tclin/1T/Experiment/move/Baseline/f3",
                        help="The load directory")
    parser.add_argument("--save_path", type=str,
                        default="/home/ltc/1T/HisMIL/experiments/Full/2020_12_26/f4",
                        help="The load directory")
    # EM
    parser.add_argument("--mmt", type=float, default=0.9, help="mmt")
    parser.add_argument("--noisy", action="store_true", default=False, help="Noisy bag pos ratio")

    # ssl
    parser.add_argument("--ssl", action="store_true", default=False, help="Self supervised learning")
    parser.add_argument("--camelyon", action="store_true", default=False, help="Training on camelyon")
    parser.add_argument("--pickle", action="store_true", default=False, help="Using pickle")
    args = parser.parse_args()
    return args


##epoch #############################ratio
### log_dir and data_root

def eval_(self, gs, trainset):
    # self.backbone = self.logger.load_backbone_fromold(self.backbone, global_step=gs)
    # self.clsnet = self.logger.load_clsnet_fromold(self.clsnet, global_step=gs)
    if gs != -1:
        print('Loading')
        self.backbone = self.logger.load_backbone(self.backbone, global_step=gs)
    #     self.clsnet = self.logger.load_clsnet(self.clsnet, global_step=gs)
    self.backbone.eval()
    #     self.clsnet.eval()
    val_loader = DataLoader(trainset, args.bs, shuffle=True, num_workers=4)
    #     confoundSet = torch.randn([trainset.__len__(), 514])
    bag_idx_list = []
    inner_idx_list = []
    feature_list = []
    with torch.no_grad():
        for batch_idx, (imgs, instance_labels, bag_index, inner_index, nodule_ratios, real_ins_labels) in enumerate(
                tqdm(val_loader, ascii=True, ncols=60)):
            instance_preds = self.backbone(imgs.to(self.device))
            feature_list.append(instance_preds)
            bag_idx_list.append(bag_index)
            inner_idx_list.append(inner_index)

    return feature_list, bag_idx_list, inner_idx_list


#############################ratio
### log_dir ## data_root ##epoch
args = parse_args()
if args.task == 'Digest':
    configs = getattr(import_module('configs.' + 'DigestSeg'), 'Config')(args)
elif args.task == 'Camelyon':
    configs = getattr(import_module('configs.' + 'Camelyon'), 'Config')(args)
else:
    raise NotImplementedError
# configs.logger.init_backup(configs)
trainer = configs.trainer
tester = configs.tester
trainer.eval = eval_
epoch = configs.load
feature_list, bag_idx_list, inner_idx_list = trainer.eval(trainer, epoch, configs.valset)
features = torch.cat(feature_list)
bag_indexs = torch.cat(bag_idx_list).unsqueeze(-1).float().cuda()
inner_indexs = torch.cat(inner_idx_list).unsqueeze(-1).float().cuda()
confounderSet = torch.cat((features, bag_indexs, inner_indexs),dim=1)

import numpy as np
confounder = np.array(confounderSet.cpu())
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
np.save(args.log_dir+'/conf.npy', confounder)