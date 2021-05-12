from __future__ import absolute_import
import os
import sys
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
sys.path.append("../")
from tqdm import tqdm

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class Camelyon(Dataset):
    """
    This dataset is the real dataset for our MIL work.(train and test use the same design
    and the only difference is the data root).

    Notices:
        1. Folder organization:
            root/
                tumor/
                    class1/
                        bag1/
                            p1.png
                            p2.png
                            ...
                        bag2/
                            ...
                        ...
                    class2/
                        ...
                non_tmuor
                validation/

            Tumor and non_tmuor have the same positive bags with positive instances and negative instances respectively.




        3. changing `cls_label_dict` might change the order of the bag!!! Be careful!
            If you find `bag_lengths` and the bag lens stored in memory bank are not the same,
            check if you changed it.

        4. (updated 2020.2.6) multi-label enabled!! The bag label and instance label is wrapped up

        5. (updated 2020.3.2) you can give an threshold for a instance to be assigned as positive,
              the threshold means the lower bound of percentage of positive mask in the instance image
              required to assign it as a positive instance.
              The default setting is 0.0. That is, an instance image would be assigned as positive
              if at least 1 pixel in the image is lesion.

    Args:
        root (str): the root directory of the data
        ins_transform (torchvision.transforms): transforms on instance
        label_transform (torchvision.transforms): transforms on label
        cls_label_dict (dict): key-value pair of class name and its encoded label number.
                        (list of dict): you can also pass a list of dict, which enable multi-label.
        pos_ins_threshold (float): with at least how many percents of positive instances in a bag
            to assign the bag as positive.
        use_clear_pos_ratios: (bool) if `True`, use pos instance ratios in bag instead of nodule ratios (which
        is a noisy term.)
    """

    def __init__(self, root, ins_transform=None, label_transform=None, cls_label_dict=None, database=None):
        self.root = root
        self.class_path_list = self.init_cls_path()
        self.ins_transform = ins_transform
        self.label_transform = label_transform

        # Bag Infos
        # self.bag_names = []
        # self.bag_paths = []
        self.bag_info = {}
        #   {pos_bag_name:{
        #                     path :[pos_path, neg_path], * list
        #                     nodule_ratio:0.7, /
        #                     label:1,
        #                     bag_len:90, *---inner_idx
        #                     bag_idx:0},
        #     n\\\\\\\\\\\\\\\
        #         {path :[neg_path],
        #          ...}
        # self.bag_labels = [] # self.bag_info['label']
        # self.bag_lengths = [] # self.bag_info['bag_len']
        # self.bag_pos_ratios = []
        # self.real_labels_in_bags = []  # list of list [[0,0,1,1], [1,0,1,]]
        # self.pos_masks_ratios_in_bags = []  # list of list
        # instance info

        ##instance_infos: [bag index, inner index, nodule ratios]
        self.instance_infos = []
        # self.instance_labels = []  # self.instance_infos['label']
        self.instance_real_labels = []  # self.instance_infos['real_label']
        # self.instance_pos_masks_ratios = []  # list
        # self.instance_paths = []
        # instance coor
        self.instance_c_x = []
        self.instance_c_y = []
        self.instance_in_which_bag = []
        self.instance_in_where = []
        self.cls_label_dict = cls_label_dict
        self.ignore_bags = np.load(os.path.join(root, 'select_bag.npy'))
        self.ignore_negs = np.load(os.path.join(root, 'select_neg.npy'))
        self._scan()

        self.database = database


    def _scan(self):
        bag_idx = 0
        for class_path in self.class_path_list:
            class_folder = class_path.rsplit("/", 1)[-1] # [tumor, nontumor, neg]
            if os.path.isdir(class_path):
                ### bag
                for bag_name in tqdm(os.listdir(class_path)):
                    if bag_name not in self.ignore_bags and bag_name not in self.ignore_negs:
                        continue
                    bag_path = os.path.join(class_path, bag_name)
                # if os.path.isdir(os.path.join(class_path, bag_dir)):
                    if len(os.listdir(os.path.join(class_path, bag_name))):##must contain at least 1 instance
                        if bag_name in self.bag_info: # tumor and non tumor
                            self.bag_info[bag_name]['path'].append(bag_path)
                            inner_idx = self.bag_info[bag_name]['bag_len']
                            flag = 0
                            tmp_bag_idx = self.bag_info[bag_name]['bag_idx']
                            # the same bag, thus bag_idx +=0
                        else:
                            self.bag_info[bag_name] = {}
                            self.bag_info[bag_name]['bag_idx'] = bag_idx
                            self.bag_info[bag_name]['path'] = [bag_path]
                            self.bag_info[bag_name]['label'] = torch.tensor(
                                self.assign_bag_label(class_folder)).float()
                            self.bag_info[bag_name]['ins_label_list'] = []
                            self.bag_info[bag_name]['ins_real_label_list'] = []
                            inner_idx = 0
                            flag = 1
                            tmp_bag_idx = bag_idx

# TODO Inner_idx
#                             real_label_list = []
#                             label_list = []
                        # pos_masks_in_bag = []
                        #instance
                        for instance_file in os.listdir(bag_path):
                            instance_path = os.path.join(bag_path, instance_file)
                            if is_image_file(instance_path):
                                self.instance_c_x.append(int(instance_file.split('_')[-3]))
                                self.instance_c_y.append(int(instance_file.split('_')[-2]))
                                self.instance_in_which_bag.append(tmp_bag_idx)
                                self.instance_in_where.append(inner_idx)
                                self.instance_real_labels.append(torch.tensor(self.get_instance_real_label(class_folder)).float())
                                # self.instance_infos.append([bag_idx, inner_idx])
                                # self.instance_paths.append(instance_path)
                                # Giving real instance

                                c_x = int(instance_file.split('_')[-3])
                                c_y = int(instance_file.split('_')[-2])
                                label = self.bag_info[bag_name]['label']
                                real_label = torch.tensor(self.get_instance_real_label(class_folder)).float()
                                self.instance_infos.append({'bag_name': bag_name,
                                                            'path': instance_path,

                                                            'bag_idx': tmp_bag_idx,
                                                            'inner_idx': inner_idx,

                                                            'c_x': c_x,
                                                            'c_y': c_y,
                                                            'label': label,
                                                            'real_label': real_label})
                                self.bag_info[bag_name]['ins_label_list'].append(label)
                                self.bag_info[bag_name]['ins_real_label_list'].append(real_label)
                                # self.instance_pos_masks_ratios.append(
                                #     self.get_instance_pos_masks_ratios(instance_file))
                                # real_label_list.append(self.get_instance_real_label(class_folder))
                                # label_list.append(self.bag_info[bag_name]['label'])
                                # pos_masks_in_bag.append(self.get_instance_pos_masks_ratios(instance_file))
                                inner_idx += 1

                        # for instance_file in os.listdir(bag_path):
                        #     if is_image_file(os.path.join(bag_path, instance_file)):
                        #         self.instance_labels.append(self.bag_info[bag_name]['label'])
                        # self.bag_labels.append(self.bag_info[bag_name]['label'])
                        # self.real_labels_in_bags.append(real_label_list)
                        # self.pos_masks_ratios_in_bags.append(pos_masks_in_bag)
                        self.bag_info[bag_name]['bag_len'] = inner_idx
                        bag_idx = bag_idx + flag
                        # self.bag_lengths.append(inner_idx)
        self.confirm_info()

        #TODO:
        # nodule_ratio = self.get_nodule_ratio(os.path.join(class_path, bag_dir))
        # bag_pos_ratios


    def init_cls_path(self):
            pos_path = os.path.join(self.root, "tumor_256_20X")
            false_pos_path = os.path.join(self.root, "nontumor_256_20X")
            neg_path = os.path.join(self.root, "neg_256_20X")
            return [pos_path, false_pos_path, neg_path]

    def assign_bag_label(self, class_folder):
        if class_folder == "tumor_256_20X" or class_folder == "nontumor_256_20X":
            return 1.0
        elif class_folder == "neg_256_20X":
            return 0.0
        else:
            raise Exception("The class folder is incorrect!")

    def get_instance_real_label(self, class_folder):
        if class_folder == "tumor_256_20X":
            return 1.0
        elif class_folder == "neg_256_20X" or class_folder == "nontumor_256_20X":
            return 0.0
        else:
            raise Exception("The class folder is incorrect!")
    def confirm_info(self):
        for bag_name in self.bag_info.keys():
            # 1.get_nodule_ratio
            label_sum = sum(self.bag_info[bag_name]['ins_label_list'])
            real_label_sum = sum(self.bag_info[bag_name]['ins_real_label_list'])
            if real_label_sum ==0:
                self.bag_info[bag_name]['ratio'] = torch.tensor(0.0)
            else:
                self.bag_info[bag_name]['ratio'] = real_label_sum/label_sum
            # 2. mean&min ratios
            # import pdb
            # pdb.set_trace()
        self.bag_pos_ratios = [i['ratio'] for i in self.bag_info.values()]
        # print(self.bag_pos_ratios)
        self.mean_ratios = torch.stack(self.bag_pos_ratios)[torch.stack(self.bag_pos_ratios) > 0].mean()
        self.min_ratios = torch.stack(self.bag_pos_ratios)[torch.stack(self.bag_pos_ratios) > 0].min()
        # 3. bag_lenths
        self.bag_lengths = [i['bag_len'] for i in self.bag_info.values()]
        self.bag_labels = [i['label'] for i in self.bag_info.values()]


        ######ins





    def get_instance_pos_masks_ratios(self, filename):
        last_name = filename.rsplit("_", 1)[-1]
        if last_name.startswith("("):
            label = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[0].split("(")[-1].strip())
            return torch.tensor(label)
        else:
            return torch.tensor(0.0)



    def __getitem__(self, idx):
        """
        Return:
            img: (?) an instance
            label: (int) bag label
            bag_idx: (int) the bag index
            inner_idx: (int) inner index of current instance
            nodule_ratio: (float) the nodule ratio of current instance.
        """
        patch = self.instance_infos[idx]
        img_dir, bag_name, bag_idx, inner_idx = patch['path'], patch['bag_name'], patch['bag_idx'], patch['inner_idx']
        # img_dir = self.instance_paths[idx]
        # bag_idx, inner_idx = self.instance_infos[idx][0], self.instance_infos[idx][1]
        nodule_ratio = self.bag_info[bag_name]['ratio']
        if not self.database:
            img = Image.open(img_dir).convert('RGB')
        else:
            key = img_dir.split('/')[-1]
            img = pickle.loads(self.database.get(key))
        label = patch['label']
        real_label = patch['real_label']
        if callable(self.ins_transform):
            img = self.ins_transform(img)
        if callable(self.label_transform):
            label = self.label_transform
        return img, label, bag_idx, inner_idx, nodule_ratio, real_label

    def __len__(self):
        return len(self.instance_infos)

    @property
    def bag_num(self):
        return len(self.bag_info)

    @property
    def max_ins_num(self):
        bag_lengths = [bag['bag_len'] for bag in self.bag_info.values()]
        return max(bag_lengths)

    def __str__(self):
        # print("bag_idx-name-class-instance_nums-ratio:\n")
        # for idx, bag_name in enumerate(self.bag_names):
        #     print("{}, {}, {}, {}, {}\n".format(idx, bag_name, self.bag_labels[idx],
        #                             self.bag_lengths[idx], self.bag_pos_ratios[idx]))
        # total_lengths = [0, 0]
        # for idx, bag_label in enumerate(self.bag_labels):
        #     if bag_label > 0:
        #         total_lengths[1] += self.bag_lengths[idx]
        #     else:
        #         total_lengths[0] += self.bag_lengths[idx]
        # print("pos bag nums: {}\n".format(sum(self.bag_labels)))
        # print("neg bag nums: {}\n".format(len(self.bag_labels) - sum(self.bag_labels)))
        # print("patch nums for 0: {}\n".format(total_lengths[0]))
        # print("patch nums for 1: {}\n".format(total_lengths[1]))
        pos_ins = 0
        neg_ins = 0
        pos_bag = 0
        neg_bag = 0
        for bag in self.bag_info.values():
            if bag['label'] > 0:
                pos_bag += 1
                pos_ins += bag['bag_len']
            else:
                neg_bag += 1
                neg_ins += bag['bag_len']

        print("pos bag nums: {}\n".format(pos_bag))
        print("neg bag nums: {}\n".format(neg_bag))
        print("patch nums for 0: {}\n".format(neg_ins))
        print("patch nums for 1: {}\n".format(pos_ins))

        return "print done.\n"


if __name__ == "__main__":
    # root = "/remote-home/my/DATA/HISMIL/Digest/Phase2/ratio_0.7_wsi/train/"
    # dataset = DigestSeg(root)
    # print(dataset)
    # total_lengths = [0, 0]
    # for idx, bag_label in enumerate(dataset.bag_labels):
    #     if bag_label > 0:
    #         total_lengths[1] += dataset.bag_lengths[idx]
    #     else:
    #         total_lengths[0] += dataset.bag_lengths[idx]
    #
    # print("patch nums for 0: {}\n".format(total_lengths[0]))
    # print("patch nums for 1: {}\n".format(total_lengths[1]))
    #
    # print("Minimum nodule ratios: {}\n".format(dataset.min_ratios))
    # print("Avg nodule ratios: {}\n".format(dataset.mean_ratios))

    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #############
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
    import pickle
    train_root = "/remote-home/source/DATA/CAMELYON16/DATA/train"
    test_root = "/remote-home/source/DATA/CAMELYON16/DATA/validation"
    trainset = Camelyon(train_root, train_transform_C, None, None, None)
    with open('/remote-home/my/HisMIL/trainset.pickle','wb') as f:
        pickle.dump(trainset, f)
    del trainset
    testset = Camelyon(test_root, test_transform_C, None, None, None)
    with open('/remote-home/my/HisMIL/testset.pickle','wb') as f:
        pickle.dump(testset, f)
    del testset
    valset =  Camelyon(train_root, test_transform_C, None, None, None)
    with open('/remote-home/my/HisMIL/valset.pickle','wb') as f:
        pickle.dump(valset, f)

    # print('done')



