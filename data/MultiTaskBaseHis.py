from __future__ import absolute_import
import os
import sys
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from data.BaseHis import BaseHis
sys.path.append("../")

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class MultiTaskBaseHis(BaseHis):
    """
    MultiTask version of BaseHis dataset. I'm trying to explore a better way to wrap up a 
    dataset into multitask version. This is a first try. If one day I find a better way to
    generally wrap up datasets into multi-task style, I'll write a new class, maybe called
    `MultiTaskDataset`.

    Well, you may also call this a `MultiLabelBaseHis` since only different in label.

    (Updated 2020.2.6) I think this is ugly... not fully free for wrapping datasets.

    Notes:
        1. Children number (single_datas) is decided by cls_label_dict! This should be a list!!
        
        2. (*) Whether to inherit from BaseHis?
        
        3. !!!! Remember that the order of `cls_label_dict` matters, currently just remind to check
        the order of `cls_label_dict`. I'll figure out how to wrap datas with different bag index and
        inner index later.

        4. Protocols for __getitem__(), __len__() and properties:
            a. I only wrap up `bag_labels` and `instnace_labels` into list since they are different.
            b. For other properties, use the same one as the first single-dataset.
            c. __len__() also use the first single-dataset.
            d. __getitem__() the only different item is label, which is wrapped up into a list.

    Args:
        cls_label_dict: (list)
    """
    def __init__(self, root, ins_transform=None, label_transform=None, cls_label_dict=[None]*2):
        ##(*) num_tasks would be modified later.
        self.num_tasks = len(cls_label_dict)
        self.root = root
        self.ins_transform = ins_transform
        self.label_transform = label_transform
        self.cls_label_dict = cls_label_dict
        self.wrap_args()
        ##(*) to be modified to controlled by a function later.
        self.single_datas = [BaseHis(self.root[idx], self.ins_transform[idx], 
                            self.label_transform[idx], cls_label_dict[idx]) for idx in range(self.num_tasks)]
        
        ##wrap up properties
        self.bag_names = self.single_datas[0].bag_names
        self.bag_paths = self.single_datas[0].bag_paths
        self.bag_lengths = self.single_datas[0].bag_lengths
        ##instance_infos: [bag index, inner index, nodule ratios]
        self.instance_infos = self.single_datas[0].instance_infos
        self.instance_paths = self.single_datas[0].instance_paths
        self.bag_labels = [self.single_datas[idx].bag_labels for idx in range(self.num_tasks)]
        self.instance_labels = [self.single_datas[idx].instance_labels for idx in range(self.num_tasks)]
    
    def wrap_args(self):
        """
        Wrap up arguments into list.
        """
        if not isinstance(self.ins_transform, list):
            self.ins_transform = [self.ins_transform] * self.num_tasks
        
        if not isinstance(self.label_transform, list):
            self.label_transform = [self.label_transform] * self.num_tasks
        
        if not isinstance(self.root, list):
            self.root = [self.root] * self.num_tasks
    
    def __getitem__(self, idx):
        """
        I just use the `__getitem__` from BaseHis (well, ugly)
        """
        img, _, bag_idx, inner_idx, nodule_ratio = self.single_datas[0][idx]
        label = [self.instance_labels[idx_task][idx] for idx_task in range(self.num_tasks)]
        return img, label, bag_idx, inner_idx, nodule_ratio
        
