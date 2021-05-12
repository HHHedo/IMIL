from __future__ import absolute_import
import os, sys
sys.path.append("../")
from utils.logger import Logger
from utils.MemoryBank import TensorMemoryBank
from data.BaseHis import BaseHis
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from utils import utility
"""
2020.2.21 by gyfastas:
    This script is for evaluate training metrics.
"""
class Evaluator(object):
    def __init__(self, data, log_dir, mmbank_dir="train_mmbank",ts_dir="temp_tsboard"):
        self.test_data = data
        self.log_dir = log_dir
        self.logger = Logger(self.log_dir, ts_dir=ts_dir)
        self.memorybank = TensorMemoryBank(self.test_data.bag_num, 
                                           self.test_data.max_ins_num, 
                                           self.test_data.bag_lengths, 0.75)
        self.mmbank_dir = mmbank_dir
    
    def run(self):
        load_dir = os.path.join(self.log_dir, self.mmbank_dir)
        for idx in range(len(os.listdir(load_dir))):
            self.memorybank.load(load_dir, idx+1)
            self.evaluate(self.memorybank)

    def evaluate(self, memory_bank):
        # read labels
        bag_labels = self.test_data.bag_labels
        # bag max evaluate
        bag_max_pred = memory_bank.max_pool()
        self.cls_report(bag_max_pred, bag_labels, "bag_max")
        # bag mean evaluate
        bag_mean_pred = memory_bank.avg_pool(self.test_data.bag_lengths)
        self.cls_report(bag_mean_pred, bag_labels, "bag_avg")

    def cls_report(self, y_pred, y_true, prefix=""):
        """
        A combination of sklearn.metrics function and our logger (use tensorboard)

        """
        ##make hard prediction
        y_pred_hard = [(x > 0.5) for x in y_pred]
        cls_report = classification_report(y_true, y_pred_hard, output_dict=True)
        auc_score = roc_auc_score(y_true, y_pred)

        self.logger.log_scalar(prefix+'/'+'Accuracy', cls_report['accuracy'], print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1.0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1.0']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)


if __name__=="__main__":
    data_root = "/remote-home/gyf/DATA/HISMIL/Phase2/ratio_0.7_nonor/"
    train_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor()
    ])
    label_dict = {"gene":1, "nodule":1, "normal":0}
    train_root = os.path.join(data_root, "train")
    trainset = BaseHis(train_root, train_transform, None, label_dict)
    log_dir = "/remote-home/gyf/project/HisMIL/experiments/HISPhase2/nonor/ratio0.7/CE/lr0.001/"

    evaluator = Evaluator(trainset, log_dir)
    evaluator.run()