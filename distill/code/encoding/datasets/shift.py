from numpy.random import shuffle
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision import transforms
import cv2
import numpy as np
import os
from PIL import Image



def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    times=int(np.random.random()*4)
    for i in range(times):
        image=np.rot90(image)
        #print("rotimg", image.shape)
        # print("typtofmask",type(mask))

        mask=np.rot90(mask)
        #print("rotmask", mask.shape)
    return image, mask

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def default_loader(id, root):
    img_path=os.path.join(root,'imgRGB',id)
    # print("aaaaa",img_path)
    depth_path=os.path.join(root,'depth',id.replace("img","depth").replace('jpg','png'))

    img = cv2.imread(img_path,-1)/255.0
    depth=np.array(cv2.imread(depth_path,-1),dtype=np.float32)


    depth=(depth[:,:,0]*256*256+depth[:,:,1]*256+depth[:,:,2]) / (256*256*256 -1) *1000
    depth[depth>8000]=0
    depth=np.array(np.log(depth+1),dtype=np.float16)
    

    return img, depth
    
class Shift(data.Dataset):
    NUM_CLASS = 2  # NUM_CLASS = 2
    def __init__(self, root,split,txt_path):
        self.metainfo={'classes':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]}
        exts={'img':'jpg','semseg':'png','depth':'png'}
        file_paths=[]
        if(not os.path.exists(txt_path)):
            for _root,dirs,files in os.walk(os.path.join(root,"imgRGB")):
                for file in files:
                    file_paths.append(os.path.join(_root,file).replace(root + "imgRGB/","")+"\n")
            shuffle(file_paths)
            with open(txt_path,'w') as f:
                f.writelines(file_paths)
        else:
            with open(txt_path, 'r') as fp:
                for line in fp:
                    # remove linebreak from a current name
                    # linebreak is the last character of each line
                    x = line[:-1]

                    # add current item to the list
                    file_paths.append(x)
        self.ids=file_paths
        # print(self.ids)
        self.loader = default_loader
        self.root = root
        self.tran = transforms.ToTensor()
        self.resize_img = transforms.Resize(size = (1024,1024))
    def __getitem__(self, index):
        id = self.ids[index]

        img,depth = self.loader(id, self.root)
        input=np.concatenate((img,depth[:,:,np.newaxis]),axis=2)
        input = torch.Tensor(input).permute(2,0,1)
        input=self.resize_img(input)

        return input
    def __len__(self):
        return len(self.ids)

