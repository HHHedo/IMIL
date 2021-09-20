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
                        help="Config file",
                        choices=["DigestSeg", "Camelyon", "Pascal"]
                        )
    parser.add_argument('--config', type=str, default='DigestSeg',
                        help=" (Sub) Config file, MIL method, since some config using the same dataset written in the same config",
                        choices=["DigestSegEMCAV2", "DigestSeg", 'DigestSegFull',"DigestSegPB",
                                 'DigestSegTOPK', 'DigestSegFull', 'DigestSegRCE', 'DigestSemi'])
    parser.add_argument("--log_dir", type=str,
                        default="./experiments/Debug",
                        help="The experiment log directory")
    parser.add_argument("--data_root", "-d", type=str,
                        default="./data/DigestPath",
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
    #Calibration loss
    parser.add_argument("--stop_epoch", type=int, default=-1, help="stop")
    parser.add_argument("--ignore_thres", type=float, default=0.95, help="ignore")
    parser.add_argument("--ignore_step", type=float, default=0.05, help="ignore")
    #one-stage or two stage training
    parser.add_argument("--database", action="store_true", default=False, help="Using database")
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument("--load", type=int, default=-1, help="the epoch to be loaded")
    parser.add_argument("--load_path", type=str,
                        default="./",
                        help="The load directory")
    # parser.add_argument("--load_path_causal", type=str,
    #                     default="/remote-home/ltc/HisMIL/experiments/Full/2020_12_26/f1",
    #                     help="The load directory")
    #EM
    parser.add_argument("--mmt", type=float, default=0, help="mmt")
    parser.add_argument("--noisy", action="store_true", default=False, help="Noisy bag pos ratio")

    # ssl
    # parser.add_argument("--ssl", action="store_true", default=False, help="Self supervised learning")
    # parser.add_argument("--camelyon", action="store_true", default=False, help="Training on camelyon")
    # parser.add_argument("--pickle", action="store_true", default=False, help="Using pickle")
    #semi
    parser.add_argument("--semi_ratio", type=float, default=None, help="semi")
    # args = parser.parse_args(['-lr', '1e-3'])
    parser.add_argument("--target_cls", type=str,
                        default="dog",
                        help="The target object for Pascal voc")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # res = resnet18()
    args = parse_args()
    if args.task == 'DigestSeg':
        configs = getattr(import_module('configs.'+'DigestSeg'), 'Config')(args)
    # elif args.task == 'DigestSegNoCrop':
    #     configs = getattr(import_module('configs.' + 'DigestSegNoCrop'), 'Config')(args)
    # elif args.task == 'DigestSegNoFlip':
    #     configs = getattr(import_module('configs.' + 'DigestSegNoFlip'), 'Config')(args)
    # elif args.task == 'DigestSegNoGau':
    #     configs = getattr(import_module('configs.' + 'DigestSegNoGau'), 'Config')(args)
    # elif args.task == 'DigestSegNoGray':
    #     configs = getattr(import_module('configs.' + 'DigestSegNoGray'), 'Config')(args)
    # elif args.task == 'DigestSegNojitter':
    #     configs = getattr(import_module('configs.' + 'DigestSegNojitter'), 'Config')(args)
    # elif args.task == 'DigestSegNoall':
    #     configs = getattr(import_module('configs.' + 'DigestSegNoall'), 'Config')(args)
    elif args.task == 'DigestSegRot':
        configs = getattr(import_module('configs.' + 'DigestSegRot'), 'Config')(args)
    elif args.task == 'Camelyon':
        print('HI.............')
        configs = getattr(import_module('configs.' + 'Camelyon'), 'Config')(args)
    elif args.task == 'CamelyonNorAug':
        print('HI.............')
        configs = getattr(import_module('configs.' + 'CamelyonNorAug'), 'Config')(args)
    # elif args.task == 'Nctcrc':
    #
    #     configs = getattr(import_module('configs.' + 'Nctcrc'), 'Config')(args)
    # elif args.task == 'CausalDigest':
    #     configs = getattr(import_module('configs.'+'CausalDigest'), 'Config')(args)
    # elif args.task == 'CausalConcat':
    #     configs = getattr(import_module('configs.'+'CausalConcat'), 'Config')(args)
    # elif args.task == 'BagDis':
    #     configs = getattr(import_module('configs.'+'BagDis'), 'Config')(args)

    elif args.task == 'Pascal':
        configs = getattr(import_module('configs.' + 'Pascal'), 'Config')(args)
    elif args.task == 'Pascal_nor':
        configs = getattr(import_module('configs.' + 'Pascal_nor'), 'Config')(args)
    else:
        raise NotImplementedError
    configs.logger.init_backup(configs, args.task)
    trainer = configs.trainer
    tester = configs.tester

    for epoch in range(configs.logger.global_step, configs.epochs):
        adjust_learning_rate(configs.optimizer, epoch, args)
        if configs.config == 'DigestSegFull':
            print('Here full')
            trainer.train_fullsupervision(epoch, configs)
            if (epoch + 1) % 10 == 0:
                tester.inference(configs.batch_size)
                tester.evaluate()
        elif configs.config == 'DigestSegCausalConFull':
            trainer.causalconcat_full(epoch, configs)
            tester.causalconcat_full(configs.batch_size)
            tester.evaluate()
        # elif configs.config == 'NctcrcFull':
        #     trainer.train_full_nct(epoch, configs)
        #     if (epoch + 1) % 10 == 0:
        #
        #         tester.test_nct()
        elif configs.config == 'DigestSemi':
            trainer.train_semi(epoch, configs)
            if (epoch + 1) % 10 == 0:
                tester.inference(configs.batch_size)
                tester.evaluate()
        elif configs.config == 'DigestSegPB':
            trainer.train_EM(epoch, configs)
            if (epoch + 1) % 10 == 0:
                tester.inference(configs.batch_size)
                tester.evaluate()
        elif configs.config == 'DigestSeg' or\
                configs.config == 'DigestSegRCE' or\
                configs.config == 'DigestSegCa':
            trainer.train(epoch)
            if (epoch+1) % 10 == 0:
                tester.inference(configs.batch_size)
                tester.evaluate()
        elif configs.config == 'DigestSegEMCA' or \
                configs.config == 'DigestSegEMCAV2' or \
                configs.config == 'DigestSegEMnocahalf' or\
                configs.config ==   'DigestSegEMnocamean' or\
                configs.config ==   'DigestSegGT' or\
                    configs.config ==   'DigestSegGM'      :
            trainer.train_EMCA(epoch, configs)
            if (epoch + 1) % 10 == 0:
                tester.inference(configs.batch_size)
                tester.evaluate()
        elif configs.config == 'DigestSegTOPK':
            trainer.train_TOPK(epoch, configs)
            if (epoch + 1) % 10 == 0:
                tester.inference(configs.batch_size)
                tester.evaluate()

        # elif configs.config == 'DigestSegAMIL':
        #     trainer.train_AMIL()
        #     tester.inference_twostage()
        # elif configs.config == 'DigestSegRNN':
        #     trainer.train_RNN(epoch, configs.batch_size)
        #     tester.test_RNN(configs.batch_size)
        # elif configs.config == 'DigestSegMaxPool':
        #     trainer.train_nonparametric_pool('max', configs.batch_size)
        #     tester.test_nonparametric_pool('max')
        # elif configs.config == 'DigestSegMeanPool':
        #     trainer.train_nonparametric_pool('mean', configs.batch_size)
        #     tester.test_nonparametric_pool('mean')
        #
        # elif configs.config == 'DigestSegRatio':
        #     trainer.train_ratio(epoch, configs.train_transform, configs.database)
        #     tester.inference_ratio(configs.test_transform, configs.database)
        # elif configs.config == 'DigestSegCASSL':
        #     trainer.train(epoch)
        #     tester.inference(configs.batch_size)
        #     tester.evaluate()
        # elif configs.config == 'BagDis':
        #     print('Here full')
        #     trainer.train_bagdis(epoch, configs)
        else:
            raise NotImplementedError

    # for epoch in range(configs.logger.global_step, configs.epochs):
    #     adjust_learning_rate(configs.optimizer, epoch, args)
    #     trainer.train(epoch)
    # tester.inference()
    # tester.evaluate()
    # for i in range(1, 11):
    #fig 2 local
    # trainer.eval(8, configs.valset)
    # trainer.eval_catplot(2, configs.testset)
    # tester.eval_(50, configs.testset)
    # configs = getattr(import_module('configs.' + 'camelyontest'), 'Config')(args)
    # tester = configs.tester
    # tester.test_instance()