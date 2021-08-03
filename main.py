#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
from torchvision.models import resnet18
from data.Camelyon import Camelyon
import argparse
from importlib import import_module
from utils.utility import adjust_learning_rate
import ssl
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ssl._create_default_https_context = ssl._create_unverified_context
def parse_args():
    parser = argparse.ArgumentParser(description='317 MIL Framework')
    parser.add_argument("--task", type=str, default="DigestSeg",
                        help="CCCCCConfig file",
                        choices=["DigestSeg", "Camelyon", "Nctcrc", "CausalDigest", "CausalConcat", 'BagDis'])
    parser.add_argument('--config', type=str, default='DigestSegEMCAV2',
                        help=" (Sub) Config file, MIL method, since some config using the same dataset written in the same config",)
                        # choices=["DigestSegEMCAV2", "DigestSeg", 'DigestSegFull',"DigestSegPB",
                        #          'DigestSegTOPK', 'DigestSegFull', 'DigestSegRCE', 'NctcrcFull', 
                        #          'DigestSegEMnocahalf', 'DigestSegEMnocamean',
                        #          'DigestSegGT', 'DigestSegGM', 'DigestSegCausalConFull', 'BagDis'])
    parser.add_argument("--log_dir", type=str,
                        default="./experiments/EMCA_thres_0.95_stable/test/",
                        help="The experiment log directory")
    parser.add_argument("--data_root", "-d", type=str,
                        default="/home/ltc/Documents/ltc/DATA/HISMIL/0",
                        help="root directory of data used, e.g. /remote-home/share/DATA/HISMIL/5_folder/1")
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
    parser.add_argument("--ignore_step", type=float, default=0.05, help="ignore")
    #one-stage or two stage training
    parser.add_argument("--database", action="store_true", default=False, help="Using database")
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument("--load", type=int, default=-1, help="the epoch to be loaded")
    parser.add_argument("--load_path", type=str,
                        default="/remote-home/ltc/HisMIL/experiments/Full/2020_12_26/f1",
                        help="The load directory")
    parser.add_argument("--load_path_causal", type=str,
                        default="/remote-home/ltc/HisMIL/experiments/Full/2020_12_26/f1",
                        help="The load directory")
    #EM
    parser.add_argument("--mmt", type=float, default=0, help="mmt")
    parser.add_argument("--noisy", action="store_true", default=False, help="Noisy bag pos ratio")

    # ssl
    parser.add_argument("--ssl", action="store_true", default=False, help="Self supervised learning")
    parser.add_argument("--camelyon", action="store_true", default=False, help="Training on camelyon")
    parser.add_argument("--pickle", action="store_true", default=False, help="Using pickle")
    #semi
    parser.add_argument("--semi_ratio", type=float, default=None, help="semi")
    # args = parser.parse_args(['-lr', '1e-3'])
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # res = resnet18()
    args = parse_args()
    if args.task == 'DigestSeg':
        configs = getattr(import_module('configs.'+'DigestSeg'), 'Config')(args)
    elif args.task == 'Camelyon':
        configs = getattr(import_module('configs.' + 'Camelyon'), 'Config')(args)
    elif args.task == 'Nctcrc':
        print('HI.............')
        configs = getattr(import_module('configs.' + 'Nctcrc'), 'Config')(args)
    elif args.task == 'CausalDigest':
        configs = getattr(import_module('configs.'+'CausalDigest'), 'Config')(args)
    elif args.task == 'CausalConcat':
        configs = getattr(import_module('configs.'+'CausalConcat'), 'Config')(args)
    elif args.task == 'BagDis':
        configs = getattr(import_module('configs.'+'BagDis'), 'Config')(args)
    else:
        raise NotImplementedError
    configs.logger.init_backup(configs, args.task)
    trainer = configs.trainer
    tester = configs.tester
    # for epoch in range(configs.logger.global_step, configs.epochs):
    #     adjust_learning_rate(configs.optimizer, epoch, args)
    #     trainer.train(epoch)
    #     tester.inference()
    #     tester.evaluate()
    for i in range(1, 11):
        trainer.eval(i, configs.testset)
    # trainer.eval_(50, configs.testset)
    # for epoch in range(configs.logger.global_step, configs.epochs):
    #     adjust_learning_rate(configs.optimizer, epoch, args)
    #     if configs.config == 'DigestSegFull':
    #         print('Here full')
    #         trainer.train_fullsupervision(epoch, configs)
    #         # if (epoch + 1) % 10 == 0:
    #         tester.inference(configs.batch_size)
    #         tester.evaluate()
    #     elif configs.config == 'DigestSegCausalConFull':
    #         trainer.causalconcat_full(epoch, configs)
    #         tester.causalconcat_full(configs.batch_size)
    #         tester.evaluate()
    #     elif configs.config == 'NctcrcFull':
    #         trainer.train_full_nct(epoch, configs)
    #         if (epoch + 1) % 10 == 0:

    #             tester.test_nct()
    #     elif configs.config == 'DigestSemi':
    #         trainer.train_semi(epoch, configs)
    #         tester.inference(configs.batch_size)
    #         tester.evaluate()

    #     elif configs.config == 'DigestSegPB':
    #         trainer.train_EM(epoch, configs)
    #         # if (epoch + 1) % 10 == 0:
    #         tester.inference(configs.batch_size)
    #         tester.evaluate()
    #     elif configs.config == 'DigestSeg' or\
    #             configs.config == 'DigestSegRCE' or\
    #             configs.config == 'DigestSegCa':
    #         trainer.train(epoch)
    #         # if (epoch+1) % 10 == 0:
    #         tester.inference(configs.batch_size)
    #         tester.evaluate()
    #     elif configs.config == 'DigestSegEMCA' or \
    #             configs.config == 'DigestSegEMCAV2' or \
    #             configs.config == 'DigestSegEMnocahalf' or\
    #             configs.config ==   'DigestSegEMnocamean' or\
    #             configs.config ==   'DigestSegGT' or\
    #                 configs.config ==   'DigestSegGM'      :
    #         trainer.train_EMCA(epoch, configs)
    #         # if (epoch + 1) % 10 == 0:
    #         tester.inference(configs.batch_size)
    #         tester.evaluate()
    #     elif configs.config == 'DigestSegTOPK':
    #         trainer.train_TOPK(epoch, configs)
    #         # if (epoch + 1) % 10 == 0:
    #         tester.inference(configs.batch_size)
    #         tester.evaluate()

    #     elif configs.config == 'DigestSegAMIL':
    #         trainer.train_AMIL()
    #         tester.inference_twostage()
    #     elif configs.config == 'DigestSegRNN':
    #         trainer.train_RNN(epoch, configs.batch_size)
    #         tester.test_RNN(configs.batch_size)
    #     elif configs.config == 'DigestSegMaxPool':
    #         trainer.train_nonparametric_pool('max', configs.batch_size)
    #         tester.test_nonparametric_pool('max')
    #     elif configs.config == 'DigestSegMeanPool':
    #         trainer.train_nonparametric_pool('mean', configs.batch_size)
    #         tester.test_nonparametric_pool('mean')

    #     elif configs.config == 'DigestSegRatio':
    #         trainer.train_ratio(epoch, configs.train_transform, configs.database)
    #         tester.inference_ratio(configs.test_transform, configs.database)
    #     elif configs.config == 'DigestSegCASSL':
    #         trainer.train(epoch)
    #         tester.inference(configs.batch_size)
    #         tester.evaluate()
    #     elif configs.config == 'BagDis':
    #         print('Here full')
    #         trainer.train_bagdis(epoch, configs)
    #     else:
    #         raise NotImplementedError