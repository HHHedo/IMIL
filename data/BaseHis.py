from __future__ import absolute_import
import os
import sys
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
sys.path.append("../")

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

class BaseHis(Dataset):
    """
    This dataset is the baseline real dataset of our work (train and test use the same design
    and the only difference is the data root).

    Notices:
        1. Folder organization:
            root/
                nodule_ratios.txt
                train/
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
                test/
        2. nodule_ratios.txt should be provided (TODO: make it optional)
        3. changing `cls_label_dict` might change the order of the bag!!! Be careful!
            If you find `bag_lengths` and the bag lens stored in memory bank are not the same,
            check if you changed it.
        
        4. (updated 2020.2.6) multi-label enabled!! The bag label and instance label is wrapped up
    
    Args:
        root (str): the root directory of the data
        ins_transform (torchvision.transforms): transforms on instance
        label_transform (torchvision.transforms): transforms on label
        cls_label_dict (dict): key-value pair of class name and its encoded label number.
                        (list of dict): you can also pass a list of dict, which enable multi-label.
    """
    def __init__(self, root, ins_transform=None, label_transform=None, cls_label_dict=None):
        self.root = root
        self.class_path_list = self.init_cls_path(cls_label_dict)
        self.ins_transform = ins_transform
        self.label_transform = label_transform
        self.bag_names = []
        self.bag_paths = []
        self.bag_labels = []
        self.bag_lengths = []
        self.bag_pos_ratios = []
        self.instance_labels = []
        ##instance_infos: [bag index, inner index, nodule ratios]
        self.instance_infos = []
        self.instance_paths = []
        self.cls_label_dict = cls_label_dict
        self._scan()

    def _scan(self):
        bag_idx = 0
        for class_path in self.class_path_list:
            class_folder = class_path.rsplit("/", 1)[-1]
            if os.path.isdir(class_path):
                for bag_dir in os.listdir(class_path):
                    if os.path.isdir(os.path.join(class_path, bag_dir)):
                        ##must contain at least 1 instance
                        if len(os.listdir(os.path.join(class_path, bag_dir))):
                            self.bag_names.append(bag_dir)
                            self.bag_paths.append(os.path.join(class_path, bag_dir))
                            ## there is a type conversion
                            label = torch.tensor(self.assign_bag_label(class_folder)).float()
                            nodule_ratio = self.get_nodule_ratio(bag_dir)
                            self.bag_labels.append(label)
                            self.bag_pos_ratios.append(nodule_ratio)
                            inner_idx = 0
                            for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                                if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                                    self.instance_infos.append([bag_idx, inner_idx, nodule_ratio])
                                    self.instance_paths.append(os.path.join(
                                        class_path, bag_dir, instance_file))
                                    self.instance_labels.append(label)
                                    inner_idx += 1
                            self.bag_lengths.append(inner_idx)
                            bag_idx += 1

    def assign_bag_label(self, class_folder):
        """
        Get the bag lebel from self.cls_label_dict if given.
        If not, we use the default setting (easy to understand).
        """
        ##single-label
        if isinstance(self.cls_label_dict, dict):
            return self.cls_label_dict[class_folder]
        ##multi-label
        elif isinstance(self.cls_label_dict, list):
            return [x[class_folder] for x in self.cls_label_dict]
        else:
            if class_folder == "normal":
                return 0
            elif class_folder == "nodule":
                return 1
            elif class_folder == "gene":
                return 2
            else:
                raise Exception("The class folder is incorrect!")
    
    def get_nodule_ratio(self, bag_name):
        """
        Get the nodule ratio in an index file named nodule_ratios.txt
        each line is: bag_name, ratio 
        Example:
        XK002, 0.5
        1, 0.0

        Notes:
            I directly use the selected_genelabel.txt, the format is:
                bag name, label, ratio
            where label is {0, 1} => {nodule, gene}
        
        """
        file_dir = os.path.join(self.root, "nodule_ratios.txt")
        with open(file_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                name = line.split(",")[0]
                ratio = line.split(",")[-1]
                if name==bag_name:
                    return torch.tensor(float(ratio))
        ## if the bag name is not found, no nodule ratio, return 0.0
        return torch.tensor(0.0)

    def init_cls_path(self, cls_label_dict):
        """
        Class paths are sub-folders in the root. Folder name is
        the class name.
        If multi-label enabled, use the order of first class-label pair.
        """
        if isinstance(cls_label_dict, dict):
            return_list = []
            for key, value in cls_label_dict.items():
                return_list.append(os.path.join(self.root, key))
            return return_list
        elif isinstance(cls_label_dict, list):
            return_list = []
            for key, value in cls_label_dict[0].items():
                return_list.append(os.path.join(self.root, key))
            return return_list
        else:
            normal_path = os.path.join(self.root, "normal")
            nodule_path = os.path.join(self.root, "nodule")
            gene_path = os.path.join(self.root, "gene")
            return [gene_path, nodule_path, normal_path]

    def __getitem__(self, idx):
        """
        Return:
            img: (?) an instance
            label: (int) bag label
            bag_idx: (int) the bag index
            inner_idx: (int) inner index of current instance
            nodule_ratio: (float) the nodule ratio of current instance.
        """
        img_dir = self.instance_paths[idx]
        bag_idx, inner_idx = self.instance_infos[idx][0], self.instance_infos[idx][1]
        nodule_ratio = self.instance_infos[idx][2]
        img = Image.open(img_dir).convert('RGB')
        label = self.instance_labels[idx]
        
        if callable(self.ins_transform):
            img = self.ins_transform(img)
        
        if callable(self.label_transform):
            label = self.label_transform

        return img, label, bag_idx, inner_idx, nodule_ratio
    c


    def __len__(self):
        return len(self.instance_paths)
    
    @property
    def bag_num(self):
        return len(self.bag_names)

    @property
    def max_ins_num(self):
        return max(self.bag_lengths)

    def __str__(self):
        print("bag_idx-name-class-instance_nums-ratio:\n")
        for idx, bag_name in enumerate(self.bag_names):
            print("{}, {}, {}, {}, {}\n".format(idx, bag_name, self.bag_labels[idx], 
                                    self.bag_lengths[idx], self.bag_pos_ratios[idx]))


        return "print done!"


if __name__=="__main__":
    root = "/remote-home/gyf/DATA/HISMIL/Phase3/ratio_0.7/test/"
    dataset = BaseHis(root, cls_label_dict={"gene":1, "nodule":0, "normal":0})
    print(dataset)
    total_lengths = [0, 0]
    for idx, bag_label in enumerate(dataset.bag_labels):
        if bag_label > 0:
            total_lengths[1] += dataset.bag_lengths[idx]
        else:
            total_lengths[0] += dataset.bag_lengths[idx]
    
    print("patch nums for 0: {}\n".format(total_lengths[0]))
    print("patch nums for 1: {}\n".format(total_lengths[1]))


