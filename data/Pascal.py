from PIL import Image
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

def is_img(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

pascal_labels= ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                'aeroplane', 'bottle', 'bicycle', 'boat', 'bus', 'car', 'chair', 'motorbike',
                'train', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
# pascal_labels= ['person']
class VOCDataset(Dataset):
    def __init__(self, root, target_cls, fold, pre_transform):
        self.root = root
        self.target_cls = target_cls
        self.fold = fold
        self.pre_transform = pre_transform
        self.bag_list = []  # a list of dict, each dict contains basic info of a bag
        self.bag_label = [] # list of {'img_name':xx,'label':xx}

        self.bag_labels = [] # list of bag labels
        self.bag_lengths = [] # list of bag lengths
        self.instance_labels = []
        self.instance_paths = []
        self.instance_in_which_bag = []
        self.instance_in_where = []
        self.bag_pos_ratios = []
        self.instance_weight = []

        self.label_dict = {}
        self.most_conf_idx = []
        self.scan_label()
        self.scan_img()


        self.tmp_instance_paths = self.instance_paths
        self.tmp_instance_in_which_bag = self.instance_in_which_bag
        self.tmp_instance_in_where = self.instance_in_where
        self.tmp_instance_labels = self.instance_labels

        # self.most_conf_ins = []
        # self.most_conf_label = []


    def scan_label(self):
        label_path = os.path.join(self.root.split('/JPEGImages')[0], 'ImageSets/Main')
        for target_cls in pascal_labels:
            label_txt = label_path + '/{}_{}.txt'.format(target_cls, self.fold)
            f = open(label_txt, 'r')
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                img_name = line.split(' ')[0]
                label = line.split(' ')[1]
                if label == '':
                    label = 1.
                else:
                    label = 0.
                if img_name in self.label_dict.keys():
                    self.label_dict[img_name].append(label)
                else:
                    self.label_dict[img_name] = [label]
                # label_dict = {
                #     'img_name': img_name,
                #     'label': label
                # }
                # self.bag_label.append(label_dict)

    def scan_img(self):
        bag_idx = 0
        for bag in self.label_dict.keys():
            img_name = bag
            label = self.label_dict[img_name]
            # clean for test , noisy for training
            if self.fold == 'test':
                img_path = os.path.join(self.root, 'rpn', img_name)
                # img_path = os.path.join(self.root, img_name)
            elif self.fold == 'trainval':
                # img_path = os.path.join(self.root, 'rpn25+25', img_name)
                # print('25+25')
                img_path = os.path.join(self.root, 'rpn', img_name)
                # print('25')
                # img_path = os.path.join(self.root, img_name)
            patch_list = os.listdir(img_path)
            inner_idx = 0
            conf_list = []

            for patch in patch_list:
                patch_path = os.path.join(img_path, patch)
                # patch_name = patch.split('.')[0]
                conf_list.append(patch.split('_')[-1][:-4])
                # path_in_bag.append(patch_path)
                patch_dict = {
                    'img_path': patch_path,
                    'label': label,
                    'bag_idx': bag_idx,
                    'inner_idx': inner_idx,
                    'coordinate': 0
                }
                self.bag_list.append(patch_dict)
                self.instance_labels.append(label)
                self.instance_paths.append(patch_path)
                self.instance_in_which_bag.append(bag_idx)
                self.instance_in_where.append(inner_idx)
                self.instance_weight.append([1]*len(label))
                inner_idx += 1

            max_conf_idx = np.argmax(np.array(conf_list))
            for i in range(len(conf_list)):
                if i == max_conf_idx:
                    self.most_conf_idx.append(torch.tensor(1.0))
                else:
                    self.most_conf_idx.append(torch.tensor(0.0))
            # if label == 1:
            self.bag_pos_ratios.append(torch.tensor(1.0))
            # else:
            #     self.bag_pos_ratios.append(torch.tensor(0.0))
            self.bag_labels.append(torch.tensor(label))
            self.bag_lengths.append(inner_idx)
            bag_idx += 1

    def generate_new_data(self, selected_idx, new_label=None, weight= None):
        idx_matrix = torch.nonzero(selected_idx).cpu().numpy()
        bag_accumulated_length = np.cumsum(np.array(self.bag_lengths))
        bag_accumulated_length = np.insert(bag_accumulated_length, 0, 0)
        # idx_matrix: [N,2]
        ins_idx_vec = bag_accumulated_length[idx_matrix[:, 0]] + idx_matrix[:, 1]
        # ins_idx_vec: [N]
        self.tmp_instance_paths = np.array(self.instance_paths)[ins_idx_vec]
        self.tmp_instance_in_which_bag = np.array(self.instance_in_which_bag)[ins_idx_vec]
        self.tmp_instance_in_where = np.array(self.instance_in_where)[ins_idx_vec]
        if new_label is None:
            self.tmp_instance_labels = np.array(self.instance_labels)[ins_idx_vec]
        else:
            self.tmp_instance_labels = new_label.view(-1, new_label.shape[2]).cpu().numpy()[ins_idx_vec]
        if weight is not None:
            self.instance_weight = weight.view(-1, weight.shape[2]).cpu().numpy()[ins_idx_vec]



    def __getitem__(self, idx):
        # bag = self.bag_list[idx]
        # img_path = bag['img_path']
        # label = bag['label']
        # bag_idx = bag['bag_idx']
        # inner_idx = bag['inner_idx']
        img_path = self.tmp_instance_paths[idx]
        bag_idx, inner_idx = self.tmp_instance_in_which_bag[idx], self.tmp_instance_in_where[idx]
        label = torch.tensor(self.tmp_instance_labels[idx])
        weight = torch.tensor(self.instance_weight[idx])
        img = Image.open(img_path)
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        return img, label, bag_idx, inner_idx, weight, self.most_conf_idx[idx]

    def __len__(self):
        return len(self.tmp_instance_paths)

    @property
    def bag_num(self):
        return len(self.bag_labels)

    @property
    def cls_num(self):
        return len(pascal_labels)

    @property
    def max_ins_num(self):
        return max(self.bag_lengths)

