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


class DigestSegBag(Dataset):
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
                 use_clear_pos_ratios=True, bag_len_thres=0):
        self.root = root
        self.class_path_list = self.init_cls_path(cls_label_dict) #[pos_path, neg_path]
        self.ins_transform = ins_transform
        self.label_transform = label_transform
        self.pos_ins_threshold = pos_ins_threshold #the threshold of positive mask for a instance to be positive
        self.use_clear_pos_ratios = use_clear_pos_ratios
        self.bag_names = []
        self.bag_paths = []
        self.bag_labels = []
        self.bag_lengths = []
        self.bag_pos_ratios = []
        self.real_labels_in_bags = []
        self.pos_masks_ratios_in_bags = []
        self.instance_labels = []
        ##instance_infos: [bag index, inner index, nodule ratios]
        self.instance_infos = []
        self.instance_real_labels = []
        self.instance_pos_masks_ratios = []
        self.instance_paths = []
        self.cls_label_dict = cls_label_dict


        # bag cls
        self.bag_len_thres = bag_len_thres
        self.instances_in_bag_list = []
        self.bag_label_uncalibrated = []
        self._scan()
        self.mean_ratios = torch.tensor([x[2] for x in self.instance_infos if x[2] > 0]).mean()
        self.min_ratios = torch.tensor([x[2] for x in self.instance_infos if x[2] > 0]).min()

    def _scan(self):
        bag_idx = 0
        for class_path in self.class_path_list:
            class_folder = class_path.rsplit("/", 1)[-1]
            if os.path.isdir(class_path):
                for bag_dir in os.listdir(class_path):
                    if os.path.isdir(os.path.join(class_path, bag_dir)):
                        ##must contain at least 1 instance
                        if len(os.listdir(os.path.join(class_path, bag_dir))) > self.bag_len_thres:
                            self.bag_names.append(bag_dir)
                            self.bag_paths.append(os.path.join(class_path, bag_dir))
                            ## there is a type conversion
                            label = torch.tensor(self.assign_bag_label(class_folder)).float()
                            self.bag_label_uncalibrated.append(label)
                            nodule_ratio = self.get_nodule_ratio(os.path.join(class_path, bag_dir))
                            inner_idx = 0
                            real_label_list = []
                            pos_masks_in_bag = []
                            ins_in_a_bag = []
                            for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                                if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                                    self.instance_infos.append([bag_idx, inner_idx, nodule_ratio])
                                    self.instance_paths.append(os.path.join(
                                        class_path, bag_dir, instance_file))
                                    self.instance_real_labels.append(self.get_instance_real_label(instance_file))
                                    self.instance_pos_masks_ratios.append(
                                        self.get_instance_pos_masks_ratios(instance_file))
                                    real_label_list.append(self.get_instance_real_label(instance_file))
                                    pos_masks_in_bag.append(self.get_instance_pos_masks_ratios(instance_file))
                                    inner_idx += 1
                                    ins_in_a_bag.append(os.path.join(
                                        class_path, bag_dir, instance_file))
                            self.instances_in_bag_list.append(ins_in_a_bag)
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
        img_dirs = self.instances_in_bag_list[idx]
        bag_label = self.bag_label_uncalibrated[idx]
        # img_list = []
        # for img_dir in img_dirs:
        #     img = Image.open(img_dir).convert('RGB')
        #     if callable(self.ins_transform):
        #         img = self.ins_transform(img)
        #     img_list.append(img)
        # img_list = torch.stack(img_list)
        # if callable(self.label_transform):
        #     bag_label = self.label_transform(bag_label)

        return img_dirs, bag_label

    def __len__(self):
        return len(self.instances_in_bag_list)

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

class DigestSegBagRatio(DigestSegBag):
    def __init__(self, root, ins_transform=None, label_transform=None, cls_label_dict=None, pos_ins_threshold=0,
                 use_clear_pos_ratios=True, mix_bag_len=64, train=True
                 ):
        super(DigestSegBagRatio, self).__init__(root, ins_transform, label_transform, cls_label_dict,
                                                pos_ins_threshold, use_clear_pos_ratios)
        self.pos_bags = [self.instances_in_bag_list[k] for k,  i in enumerate(self.bag_pos_ratios) if i>0]
        self.pos_bag_ratios = [i for k, i in enumerate(self.bag_pos_ratios) if i>0]
        self.neg_bags = [self.instances_in_bag_list[k] for k,  i in enumerate(self.bag_pos_ratios) if i==0]
        self.neg_bag_ratios = [i for k, i in enumerate(self.bag_pos_ratios) if i==0]
        self.mix_bag_len = mix_bag_len
        self.train=train

    def mix_bags(self, pos_bag, neg_bag, pos_ins_num, neg_ins_num):
        pos_selected_idx = torch.randint(len(pos_bag), (pos_ins_num,))
        neg_selected_idx = torch.randint(len(neg_bag), (neg_ins_num,))
        img_dirs = [pos_bag[i] for i in pos_selected_idx] + [neg_bag[i] for i in neg_selected_idx]
        return img_dirs

    def load_imgs(self, img_dirs):
        img_list = []
        for img_dir in img_dirs:
            img = Image.open(img_dir).convert('RGB')
            print(img_dir)
            if callable(self.ins_transform):
                img = self.ins_transform(img)
            img_list.append(img)
        img_list = torch.stack(img_list)
        return img_list

    def __getitem__(self, idx):
        pos_bag_imgs = self.pos_bags[idx]
        neg_bag_imgs = self.neg_bags[torch.randint(len(self.neg_bags), (1,))]
        alpha = torch.rand((1,))
        pos_ins_num = int(alpha*self.mix_bag_len)
        neg_ins_num = self.mix_bag_len - pos_ins_num
        mix_img_dirs = self.mix_bags(pos_bag_imgs, neg_bag_imgs, pos_ins_num, neg_ins_num)
        pos_img_dirs = self.mix_bags(pos_bag_imgs, neg_bag_imgs, self.mix_bag_len, 0)
        if self.train:
            mix_imgs = self.load_imgs(mix_img_dirs)
            pos_imgs = self.load_imgs(pos_img_dirs)
        else:
            mix_imgs = None
            pos_imgs = self.load_imgs(pos_img_dirs)
        return mix_imgs, pos_imgs, alpha, self.pos_bag_ratios[idx]

        # if callable(self.label_transform):
        #     bag_label = self.label_transform(bag_label)





        # img_dirs = self.instances_in_bag_list[idx]
        # bag_ratios = self.bag_pos_ratios[idx]
        # return img_dirs, bag_ratios

    def __len__(self):
        return len(self.pos_bags)

class DigestSegIns(Dataset):
    def __init__(self, root, ins_transform=None, database=None):
        self.root = root
        self.ins_transform = ins_transform
        self.database = database

    def __getitem__(self, idx):
        img_dir = self.root[idx][0]
        if not self.database:
            try:
                img = Image.open(img_dir).convert('RGB')
            except:
                print(img_dir)
        else:
            key = img_dir.split('/')[-1]
            img = pickle.loads(self.database.get(key))
        # img = Image.open(self.root[idx][0]).convert('RGB')
        if callable(self.ins_transform):
            img = self.ins_transform(img)

        return img

    def __len__(self):
        return len(self.root)
if __name__ == "__main__":
    from tqdm import tqdm
    from utils.utility import init_all_dl
    import torchvision.transforms as transforms
    train_root = "/home/tclin/Phase2/5_folder/0/train"
    # dataset = DigestSegBag(root)
    trainset = DigestSegBag(train_root, None, None, None)
    # testset = DigestSegBag(test_root, None, None, None)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    # self.test_loader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=0)
    train_loader_list = init_all_dl(train_loader, 64, shuffle=True,
                                         trans=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]))
    # self.test_loader_list = init_all_dl(self.test_loader, self.batch_size, shuffle=False,
    #                                     trans=self.test_transform, database=self.database)
    for (bag_labels, bag_dataloader) in tqdm(train_loader_list, ascii=True, ncols=60):

        for img_in_bag in bag_dataloader:
            idx=2
    # for idx, bag_label in tqdm(enumerate(dataset.bag_labels)):
    #         idx =idx
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


