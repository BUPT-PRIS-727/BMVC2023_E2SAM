import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch._C import dtype
import torchvision.transforms as transform

class CloudSet_1024(data.Dataset):
    NUM_CLASS = 10
    def __init__(self, root, split, mode=None, transform=transform.Compose([
        # transform.ToTensor(),
        transform.Normalize(
                    [0.48969566, 0.46122111, 0.40845247, 0.42575709, 0.34868406, 0.29353659,
                     0.34558332, 0.14569656, 0.17792995, 0.20482719, 0.28424213, 0.21427191,
                     0.29169121, 0.28796465, 0.27809529, 0.23621918], # mean
                    [0.12632009, 0.14350215, 0.18924507, 0.22366015, 0.19707532, 0.18446651,
                     0.04896662, 0.02361904, 0.02939885, 0.03480839, 0.06311043, 0.04050178,
                     0.06538713, 0.06614832, 0.06286721, 0.04817895] # std
                    )]),
                 target_transform=None, base_size=1024, crop_size=1024):
        self.metainfo={'classes':[0,1,2,3,4,5,6,7,8,9]}
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.norm = torch.FloatTensor([211, 211, 211, 211, 208, 211, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]).view(1, 1, -1)
        # .npy list
        if self.split == 'train':
            with open("train_data.txt","r") as f:
                self.file_list=f.readlines()
        elif self.split == 'val':
            with open("test_data.txt","r") as f:
                self.file_list=f.readlines()
        else:
            raise RuntimeError("Error mode! Please use 'train' or 'val'.")
        self.file_list=[filename.strip("\n") for filename in self.file_list]
        if len(self.file_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def make_pred(self, x):
        return x

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        pair_dir = os.path.join(self.root,self.file_list[index])
        if self.mode == 'test':
            pair = np.load(pair_dir)
            img = pair[..., :16]
            mask = pair[..., -1]
            mask[mask == 255] = 10
            img = torch.from_numpy(np.array(img, dtype=np.float32)).float()
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
            # img = img / self.norm
            if self.transform is not None:
                img/=self.norm
                img = self.transform(img.permute(2, 0, 1))
            return img, mask
        pair = np.load(pair_dir)
        img = pair[..., :16]
        mask = pair[..., -1]
        mask[mask == 255] = 10
        # Normalize
        img = torch.from_numpy(np.array(img, dtype=np.float32)).float()
        # img = img / self.norm
        if self.transform is not None:
            img/=self.norm
            img = self.transform(img.permute(2, 0, 1))
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        mask=self._mask_transform(mask)
        if("train" in self.split):
            output=transform.RandomCrop(512)(torch.concat((img,mask.unsqueeze(0)),dim=0))
            img=output[:6,:,:]
            mask=output[6,:,:]
        else:
            # img=img.contiguous().reshape(1,6,2,512,2,512).permute(0,1,2,4,3,5).reshape(4,6,512,512)
            # mask=mask.contiguous().reshape(1,1,2,512,2,512).permute(0,1,2,4,3,5).reshape(4,1,512,512)
            split_img=[]
            split_mask=[]
            for i in range(2):
                for j in range(2):
                    split_img.append(img[:,i*512:(i+1)*512,j*512:(j+1)*512].unsqueeze(0))
                    split_mask.append(mask[i*512:(i+1)*512,j*512:(j+1)*512].unsqueeze(0))
            img=torch.concat(tuple(split_img),dim=0)
            mask=torch.concat(tuple(split_mask),dim=0)
        return img,mask 

    def _mask_transform(self, mask):
        return torch.from_numpy(mask).long()

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

def horizontal_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return img.copy(), mask.copy()

def vertical_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[::-1, :, :]
        mask = mask[::-1, :]
    return img.copy(), mask.copy()

def rot_90(img, mask, p=0.5):
    if random.random() > p: 
        img = np.rot90(img, k=-1)
        mask = np.rot90(mask, k=-1)
    return img.copy(), mask.copy()

