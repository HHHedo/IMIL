#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
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
def parse_args():
    parser = argparse.ArgumentParser(description='317 MIL Framework')
    parser.add_argument("--task", type=str, default="Digest",
                        help="Which task to perform",
                        choices=["Digest", "Camelyon"])
    parser.add_argument('--config', type=str, default='DigestSegPB',
                        help="Config file to use.",
                        choices=["DigestSegEMCAV2", "DigestSeg", 'DigestSegPB',
                                 'DigestSegTOPK', 'DigestSegFull','DigestSegRCE'])
    parser.add_argument("--log_dir", type=str,
                        default="./experiments/test",
                        help="The experiment log directory")
    parser.add_argument("--data_root", "-d", type=str,
                        default="/home/ltc/1T/Phase2/5_folder/2",
                        help="root directory of data used")
    parser.add_argument("--resume", type=int, default=-1, help="Resume epoch")

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
    #choose loss
    # parser.add_argument("--rce", action="store_true", default=False, help="Whether to use weighted BCE loss")
    #Calibration loss
    parser.add_argument("--stop_epoch", type=int, default=-1, help="stop")
    parser.add_argument("--ignore_thres", type=float, default=0.95, help="ignore")
    #one-stage or two stage training
    parser.add_argument("--database", action="store_true", default=False, help="Using database")
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument("--load", type=int, default=-1, help="the epoch to be loaded")
    parser.add_argument("--load_path", type=str,
                        default="/home/tclin/1T/Experiment/move/Baseline/f3",
                        help="The load directory")
    #EM
    parser.add_argument("--mmt", type=float, default=0.9, help="mmt")
    parser.add_argument("--noisy", action="store_true", default=False, help="Noisy bag pos ratio")

    # ssl
    parser.add_argument("--ssl", action="store_true", default=False, help="Self supervised learning")
    parser.add_argument("--camelyon", action="store_true", default=False, help="Training on camelyon")
    parser.add_argument("--pickle", action="store_true", default=False, help="Using pickle")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # res = resnet18()
    args = parse_args()
    if args.task == 'Camelyon':
        configs = getattr(import_module('configs.' + 'camelyontest'), 'Config')(args)
    else:
        raise NotImplementedError
    configs.logger.init_backup(configs)
    # trainer = configs.trainer
    tester = configs.tester

    tester.test_instance()



