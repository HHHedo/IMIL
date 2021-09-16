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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

class DigestSeg(Dataset):
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
        use_clear_pos_ratios=True, database=None, ssl=None, ssl_transform=None, semi_ratio=None ,alpha=2, vis=None):
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
        self.real_labels_in_bags = [] # list of list
        self.pos_masks_ratios_in_bags = [] # list of list
        # instance info
        self.instance_labels = []
        ##instance_infos: [bag index, inner index, nodule ratios]
        self.instance_infos = []
        self.instance_real_labels = [] # list
        self.instance_pos_masks_ratios = [] # list
        self.instance_paths = []
        # instance coor
        self.instance_c_x = []
        self.instance_c_y = []
        self.instance_in_which_bag = []
        self.instance_in_where = []
        self.cls_label_dict = cls_label_dict
        self._scan()
        self.mean_ratios = torch.stack(self.bag_pos_ratios)[torch.stack(self.bag_pos_ratios)>0].mean()
        self.min_ratios  = torch.stack(self.bag_pos_ratios)[torch.stack(self.bag_pos_ratios)>0].min()
        self.database = database
        self.ssl = ssl
        self.ssl_transform = ssl_transform

        self.tmp_instance_paths = self.instance_paths
        self.tmp_instance_in_which_bag = self.instance_in_which_bag
        self.tmp_instance_in_where = self.instance_in_where
        self.tmp_instance_labels = self.instance_labels
        self.tmp_instance_real_labels = self.instance_real_labels
        self.semi_ratio = semi_ratio
        if self.semi_ratio:
            self.semi_labels, self.semi_index = self.get_semi_label(alpha)
        self.vis=vis

    def get_semi_label(self, alpha):
        num = len(self.tmp_instance_labels)
        masks = np.ones(num)
        masks[:int(self.semi_ratio*num)] = 0 # ratio real(mask=1), (1-ratio) fake(propogation, mask=0)
        print(num,self.semi_ratio, int(self.semi_ratio*num))
        np.random.shuffle(masks)
        print(masks)
        semi_labels = masks*np.array(self.tmp_instance_labels) +(1-masks)*np.array(self.tmp_instance_real_labels)
        semi_labels = semi_labels.tolist()
        masks = (alpha - (alpha-1)*masks).tolist()
        return semi_labels, masks

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
                            # label = torch.tensor(self.assign_bag_label(class_folder)).float()
                            nodule_ratio = self.get_nodule_ratio(os.path.join(class_path, bag_dir))
                            inner_idx = 0
                            real_label_list = []
                            pos_masks_in_bag = []
                            for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                                if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                                    self.instance_c_x.append(int(instance_file.split('_')[-3]))
                                    self.instance_c_y.append(int(instance_file.split('_')[-2]))
                                    self.instance_in_which_bag.append(bag_idx)
                                    self.instance_in_where.append(inner_idx)
                                    self.instance_infos.append([bag_idx, inner_idx, nodule_ratio])
                                    self.instance_paths.append(os.path.join(
                                        class_path, bag_dir, instance_file))
                                    #Giving real instance label depending on threshold
                                    self.instance_real_labels.append(self.get_instance_real_label(instance_file))
                                    self.instance_pos_masks_ratios.append(self.get_instance_pos_masks_ratios(instance_file))
                                    real_label_list.append(self.get_instance_real_label(instance_file))
                                    pos_masks_in_bag.append(self.get_instance_pos_masks_ratios(instance_file))
                                    inner_idx += 1
                            #calibrated bag label depending on real instance label
                            if torch.tensor(self.assign_bag_label(class_folder)).float() != self.calibrate_bag_labels(real_label_list):
                                print(bag_dir)
                            label = self.calibrate_bag_labels(real_label_list)
                            for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                                if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                                    self.instance_labels.append(label)
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
                ratio = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[-1].strip().rsplit(")",1)[0])
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
                    ratio = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[-1].strip().rsplit(")",1)[0])

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
        return (sum(real_label_list)>0).float()


    def get_instance_real_label(self, filename):
        """
        get instance real label (1 or 0) for an instance (useful for test)
        """
        last_name = filename.rsplit("_", 1)[-1]
        if last_name.startswith("("):
            label = float(last_name.rsplit(".", 1)[0].rsplit(",", 1)[0].split("(")[-1].strip())
            label = float(label > self.pos_ins_threshold)
            return torch.tensor([label])
        else:
            return torch.tensor([0.0])

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

    def generate_new_data(self, selected_idx):
        idx_matrix = torch.nonzero(selected_idx).cpu().numpy()
        bag_accumulated_length = np.cumsum(np.array(self.bag_lengths))
        bag_accumulated_length = np.insert(bag_accumulated_length, 0, 0)
        # idx_matrix: [N,2]
        ins_idx_vec = bag_accumulated_length[idx_matrix[:, 0]] + idx_matrix[:, 1]
        # ins_idx_vec: [N]
        self.tmp_instance_paths = np.array(self.instance_paths)[ins_idx_vec]
        self.tmp_instance_in_which_bag = np.array(self.instance_in_which_bag)[ins_idx_vec]
        self.tmp_instance_in_where = np.array(self.instance_in_where)[ins_idx_vec]
        self.tmp_instance_labels = np.array(self.instance_labels)[ins_idx_vec]
        self.tmp_instance_real_labels = np.array(self.instance_real_labels)[ins_idx_vec]
        print('The Topk-K ACC of Pos is {}'.format(self.tmp_instance_real_labels.sum() / self.tmp_instance_labels.sum()))

    def __getitem__(self, idx):
        """
        Return:
            img: (?) an instance
            label: (int) bag label
            bag_idx: (int) the bag index
            inner_idx: (int) inner index of current instance
            nodule_ratio: (float) the nodule ratio of current instance.
        """
        # img_dir = self.instance_paths[idx]
        # bag_idx, inner_idx = self.instance_infos[idx][0], self.instance_infos[idx][1]
        # nodule_ratio = self.instance_infos[idx][2]
        # if not self.database:
        #     img = Image.open(img_dir).convert('RGB')
        # else:
        #     key = img_dir.split('/')[-1]
        #     img = pickle.loads(self.database.get(key))
        # label = self.instance_labels[idx]
        # real_label = self.instance_real_labels[idx]
        # if callable(self.ins_transform):
        #     img = self.ins_transform(img)
        #     if self.ssl:
        #         img2 = self.ssl_transform(img)
        #
        # if callable(self.label_transform):
        #     label = self.label_transform
        # if self.ssl:
        #     return (img, img2), label, bag_idx, inner_idx, nodule_ratio, real_label
        # return img, label, bag_idx, inner_idx, nodule_ratio, real_label
        img_dir = self.tmp_instance_paths[idx]
        bag_idx, inner_idx = self.tmp_instance_in_which_bag[idx], self.tmp_instance_in_where[idx]
        # print(bag_idx, inner_idx)
        #TODO:inner_idx = self.tmp_instance_in_where[idx]
        # nodule_ratio = self.bag_pos_ratios[self.bag_idx_list.index(bag_idx)]
        nodule_ratio = self.instance_infos[idx][2]
        if not self.database:
            try:
                img = Image.open(img_dir).convert('RGB')
            except:
                retry_times = 0
                while retry_times <= 10:
                    try:
                        retry_times += 1
                        print('OSError occured. Retry [{}/10] ...'.format(retry_times))
                        img = Image.open(img_dir).convert('RGB')
                        break
                    except:
                        pass
                if retry_times > 10:
                    raise
        else:
            key = img_dir.split('/')[-1]
            img = pickle.loads(self.database.get(key))
        # label = patch['label']
        # real_label = patch['real_label']
        label = self.tmp_instance_labels[idx]
        real_label = self.tmp_instance_real_labels[idx]
        if callable(self.ins_transform):
            img = self.ins_transform(img)
        if callable(self.label_transform):
            label = self.label_transform
        if self.semi_ratio:
            semi_label = self.semi_labels[idx]
            semi_index = self.semi_index[idx]
            return img, semi_label, bag_idx, inner_idx, nodule_ratio, real_label, semi_index
        if self.vis:
            return img, label, bag_idx, [self.instance_c_x[idx], self.instance_c_y[idx],self.instance_paths[idx]], nodule_ratio, real_label
        return img, label, bag_idx, inner_idx, nodule_ratio, real_label
    
    def __len__(self):
        return len(self.tmp_instance_paths)
    
    @property
    def bag_num(self):
        return len(self.bag_names)

    @property
    def cls_num(self):
        return torch.tensor([1]).int()

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


if __name__=="__main__":
    root = "/remote-home/my/DATA/HISMIL/Digest/Phase2/ratio_0.7_wsi/train/"
    dataset = DigestSeg(root)
    print(dataset)
    total_lengths = [0, 0]
    for idx, bag_label in enumerate(dataset.bag_labels):
        if bag_label > 0:
            total_lengths[1] += dataset.bag_lengths[idx]
        else:
            total_lengths[0] += dataset.bag_lengths[idx]
    
    print("patch nums for 0: {}\n".format(total_lengths[0]))
    print("patch nums for 1: {}\n".format(total_lengths[1]))

    print("Minimum nodule ratios: {}\n".format(dataset.min_ratios))
    print("Avg nodule ratios: {}\n".format(dataset.mean_ratios))


