import os
from PIL import Image
import numpy as np
import torch.utils.data as data

CAMELYON16_DIR = None
DIR_LIST = ['/lustre/home/acct-seexy/seexy/main/datasets/CAMELYON16/']

for path in DIR_LIST:
    if os.path.exists(path):
        CAMELYON16_DIR = path
        break

assert CAMELYON16_DIR is not None

all_dir = np.load('/lustre/home/acct-seexy/seexy/main/ltc/HisMIL/test_dataset.npy', allow_pickle = True)
all_label = np.load('/lustre/home/acct-seexy/seexy/main/ltc/HisMIL/test_label.npy', allow_pickle = True)

for i, path in enumerate(all_dir):
    all_dir[i]=os.path.join('/lustre/home/acct-seexy/seexy/main/datasets/CAMELYON16/validation', path.split('/')[-3], path.split('/')[-2], path.split('/')[-1])

class CAMELYON16(data.Dataset):
    def __init__(self, image_transforms=None):
        super().__init__()
        self.transform = image_transforms
        self.instance = self._make_dataset()

    def _make_dataset(self):
        instances = []
        for i in range(len(all_dir)):
            item = all_dir[i], float(all_label[i])
            instances.append(item)
            # print(len(instances))
        return instances

    def __getitem__(self, index):
        path, target = self.instance[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        data = [sample, target]
        return tuple(data)

    def __len__(self):
        return len(self.instance)