from __future__ import absolute_import
import os
import sys
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pickle

sys.path.append("../")


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class EMDigestSeg(Dataset):
    """
    This dataset is the real dataset for our MIL work.(train and test use the same design
    and the only difference is the data root).

    Notices:
        1. Folder organization:
            root/
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

        2. (updated 2020.3.2) for each instance image file in pos bag, the last name before ".png" is
        (positive mask ratio in patch, positive mask ratio in bag). Since such information is
        provided, the instance-level AUC is available for testing.
            Example:
                xxxx_yyyy_ddd_(0.11, 0.32).png


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

    def __init__(self, root, ins_transform=None, label_transform=None, cls_label_dict=None, pos_ins_threshold=0,
                 use_clear_pos_ratios=True, pos_select_idx=None, database=None):
        self.root = root
        self.class_path_list = self.init_cls_path(cls_label_dict)
        self.ins_transform = ins_transform
        self.label_transform = label_transform
        self.pos_ins_threshold = pos_ins_threshold
        self.use_clear_pos_ratios = use_clear_pos_ratios
        # Bag Infos
        self.bag_names = []
        self.bag_paths = []
        self.bag_labels = []
        self.bag_lengths = []
        self.bag_pos_ratios = []
        self.real_labels_in_bags = []  # list of list
        self.pos_masks_ratios_in_bags = []  # list of list
        # instance info
        self.instance_labels = []
        ##instance_infos: [bag index, inner index, nodule ratios]
        self.instance_infos = []
        self.instance_real_labels = []  # list
        self.instance_pos_masks_ratios = []  # list
        self.instance_paths = []
        # instance coor
        self.instance_c_x = []
        self.instance_c_y = []
        self.instance_in_which_bag = []
        self.instance_in_where = []
        self.cls_label_dict = cls_label_dict
        # EM selection
        self.pos_select_idx = pos_select_idx
        self._scan()
        self.mean_ratios = torch.stack(self.bag_pos_ratios)[torch.stack(self.bag_pos_ratios) > 0].mean()
        self.min_ratios = torch.stack(self.bag_pos_ratios)[torch.stack(self.bag_pos_ratios) > 0].min()
        self.database = database


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
                            nodule_ratio = self.get_nodule_ratio(os.path.join(class_path, bag_dir))
                            inner_idx = 0
                            real_label_list = []
                            pos_masks_in_bag = []
                            cnt = 0
                            for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                                if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                                    if self.pos_select_idx[bag_idx][inner_idx] == 1 or label == 0:
                                        cnt += 1
                                        self.instance_c_x.append(int(instance_file.split('_')[-3]))
                                        self.instance_c_y.append(int(instance_file.split('_')[-2]))
                                        self.instance_in_which_bag.append(bag_idx)
                                        self.instance_in_where.append(inner_idx)
                                        self.instance_infos.append([bag_idx, inner_idx, nodule_ratio])
                                        self.instance_paths.append(os.path.join(
                                            class_path, bag_dir, instance_file))
                                        # Giving real instance label depending on threshold
                                        self.instance_real_labels.append(self.get_instance_real_label(instance_file))
                                        self.instance_pos_masks_ratios.append(
                                            self.get_instance_pos_masks_ratios(instance_file))
                                        real_label_list.append(self.get_instance_real_label(instance_file))
                                        pos_masks_in_bag.append(self.get_instance_pos_masks_ratios(instance_file))
                                    inner_idx += 1

                            if cnt >0:
                                inner_idx_1 = 0
                                # print(real_label_list, os.listdir(os.path.join(class_path, bag_dir)))
                                # calibrated bag label depending on real instance label
                                label = self.calibrate_bag_labels(real_label_list)
                                for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                                    if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                                        if self.pos_select_idx[bag_idx][inner_idx_1] == 1 or label == 0:
                                            self.instance_labels.append(label)
                                        inner_idx_1 += 1
                                self.bag_labels.append(label)
                                self.bag_pos_ratios.append(nodule_ratio)
                                self.real_labels_in_bags.append(real_label_list)
                                self.pos_masks_ratios_in_bags.append(pos_masks_in_bag)
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
            if class_folder == "pos":
                return 1.0
            elif class_folder == "neg":
                return 0.0
            else:
                raise Exception("The class folder is incorrect!")

    def get_nodule_ratio(self, bag_dir):
        """
        Get the nodule ratio information for a bag from instance file name
        The rule is:
            1. The file name split by the last "_" would rather be
                in "(a, b).png" format or ".png", where b is the
                nodule ratio information we want.

            2. It is assumed that all instances in bag folder share
                the same nodule ratios. Therefore, we read the
                first ".png" file and check if (b) exists.

            3. (updated 2020.3.3) if `self.use_clear_pos_ratios` is `True`,
                return the ratio calculated as positive instances ratios in a bag.

        """
        if not self.use_clear_pos_ratios:
            sample_fname = os.listdir(bag_dir)[0]
            last_name = sample_fname.rsplit("_", 1)[-1]
            if last_name.startswith("("):
                ratio = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[-1].strip().rsplit(")", 1)[0])
                return torch.tensor(ratio)
            else:
                return torch.tensor(0.0)
        else:
            file_list = os.listdir(bag_dir)
            real_labels = []
            for fname in file_list:
                if is_image_file(os.path.join(bag_dir, fname)):
                    real_label = self.get_instance_real_label(fname)
                    real_labels.append(real_label)
            real_labels = torch.tensor(real_labels)
            return real_labels.mean()

    def get_nodule_ratio_and_coordinate(self, bag_dir):
        """
        Get the nodule ratio information for a bag from instance file name
        The rule is:
            1. The file name split by the last "_" would rather be
                in "(a, b).png" format or ".png", where b is the
                nodule ratio information we want.

            2. It is assumed that all instances in bag folder share
                the same nodule ratios. Therefore, we read the
                first ".png" file and check if (b) exists.

            3. (updated 2020.3.3) if `self.use_clear_pos_ratios` is `True`,
                return the ratio calculated as positive instances ratios in a bag.

        """
        if not self.use_clear_pos_ratios:
            sample_fname = os.listdir(bag_dir)[0]
            last_name = sample_fname.rsplit("_", 1)[-1]
            if last_name.startswith("("):
                ratio = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[-1].strip().rsplit(")", 1)[0])

                return torch.tensor(ratio)
            else:
                return torch.tensor(0.0)
        else:
            file_list = os.listdir(bag_dir)
            real_labels = []
            coordinates = []
            for fname in file_list:
                if is_image_file(os.path.join(bag_dir, fname)):
                    real_label = self.get_instance_real_label(fname)
                    real_labels.append(real_label)
                    coordinate_x, coordinate_y = int(fname.split('_')[-3]), int(fname.split('_')[-2])
                    coordinates.append(torch.tensor([coordinate_x, coordinate_y]))
            real_labels = torch.tensor(real_labels)
            coordinates = torch.stack(coordinates)
            return real_labels.mean(), coordinates

    def calibrate_bag_labels(self, real_label_list):

        return (sum(real_label_list) > 0).float()

    def get_instance_real_label(self, filename):
        """
        get instance real label (1 or 0) for an instance (useful for test)
        """
        last_name = filename.rsplit("_", 1)[-1]
        if last_name.startswith("("):
            label = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[0].split("(")[-1].strip())
            label = float(label > self.pos_ins_threshold)
            return torch.tensor(label)
        else:
            return torch.tensor(0.0)

    def get_instance_pos_masks_ratios(self, filename):
        last_name = filename.rsplit("_", 1)[-1]
        if last_name.startswith("("):
            label = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[0].split("(")[-1].strip())
            return torch.tensor(label)
        else:
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
            pos_path = os.path.join(self.root, "pos")
            neg_path = os.path.join(self.root, "neg")
            return [pos_path, neg_path]

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
        if not self.database:
            img = Image.open(img_dir).convert('RGB')
        else:
            key = img_dir.split('/')[-1]
            img = pickle.loads(self.database.get(key))
        label = self.instance_labels[idx]
        real_label = self.instance_real_labels[idx]
        if callable(self.ins_transform):
            img = self.ins_transform(img)

        if callable(self.label_transform):
            label = self.label_transform

        return img, label, bag_idx, inner_idx, nodule_ratio, real_label

    def __len__(self):
        return len(self.instance_paths)

    @property
    def bag_num(self):
        return len(self.bag_names)

    @property
    def max_ins_num(self):
        return max(self.bag_lengths)

    def __str__(self):
        # print("bag_idx-name-class-instance_nums-ratio:\n")
        # for idx, bag_name in enumerate(self.bag_names):
        #     print("{}, {}, {}, {}, {}\n".format(idx, bag_name, self.bag_labels[idx],
        #                             self.bag_lengths[idx], self.bag_pos_ratios[idx]))
        total_lengths = [0, 0]
        for idx, bag_label in enumerate(self.bag_labels):
            if bag_label > 0:
                total_lengths[1] += self.bag_lengths[idx]
            else:
                total_lengths[0] += self.bag_lengths[idx]

        print("pos bag nums: {}\n".format(sum(self.bag_labels)))
        print("neg bag nums: {}\n".format(len(self.bag_labels) - sum(self.bag_labels)))
        print("patch nums for 0: {}\n".format(total_lengths[0]))
        print("patch nums for 1: {}\n".format(total_lengths[1]))

        return "print done.\n"




