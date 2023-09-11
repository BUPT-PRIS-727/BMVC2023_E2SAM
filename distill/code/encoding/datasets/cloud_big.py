import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch._C import dtype
from torchvision.transforms import RandomCrop

class CloudBigSet(data.Dataset):
    NUM_CLASS = 10
    def __init__(self, root, split, mode=None, transform=None, fft=False,
                 target_transform=None, base_size=256, crop_size=256):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.fft=fft
        self.base_size = base_size
        self.crop_size = crop_size
        self.norm = torch.FloatTensor([211, 211, 211, 211, 208, 211, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]).view(1, 1, -1)

        self.file_list = glob(os.path.join(self.root, '*.npz'))

        if len(self.file_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def make_pred(self, x):
        return x

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        pair_dir = self.file_list[index]
        pair = np.load(pair_dir)['x'].astype(np.uint8)
        img = pair[..., :16]

        # Data Augmentationaw
        img = rot_90(img=img)
        img = horizontal_flip(img=img)
        img = vertical_flip(img=img)
        # img = dropout(img=img)
        # Normalize
        img = torch.from_numpy(np.array(img, dtype=np.float32)).float()
        img = img / self.norm
        if self.transform is not None:
            img = self.transform(img.permute(2, 0, 1))
        return img


def dropout(img, d=0.1, p=0.5):
    if random.random() > p:
        h, w, c = img.shape
        num_drop = int(h*d)
        y = np.random.randint(0, h, num_drop)
        x = np.random.randint(0, w, num_drop)
        for y_, x_ in zip(y, x):
            drop_c = np.random.randint(0, c, 1)
            img[y_, x_, drop_c] = 0
    return img

def horizontal_flip(img, p=0.5):
    if random.random() > p: 
        img = img[:, ::-1, :]
    return img.copy()

def vertical_flip(img, p=0.5):
    if random.random() > p: 
        img = img[::-1, :, :]
    return img.copy()

def rot_90(img, p=0.5):
    if random.random() > p: 
        img = np.rot90(img, k=-1)
    return img.copy()


