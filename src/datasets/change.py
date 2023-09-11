import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision import transforms
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transform

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if len(image.shape)!=4:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))

    return image, mask

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

        mask=np.rot90(mask)
    return image, mask

def heb(image,id):
    date=int(id.split('_')[-1])
    # if (date>=800 and date <= 930) and date==2330 :
    if (date>=800 and date <= 930) and date==2330:
        pass
    else:
        return image
    channels=[0,1,2,3,4,5]
    for i in channels:
        tmp=image[:,:,i]
        heb_image=cv2.equalizeHist(tmp)
        image[:,:,i]=heb_image
    return image

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def default_loader(id, root, split):
    
    img1 = cv2.imread(os.path.join(root+'/A/', '{}').format(id))
    img2 = cv2.imread(os.path.join(root+'/B/', '{}').format(id))
    img  = cv2.merge((img1, img2))
    mask = cv2.imread(os.path.join(root+'/label/', '{}').format(id), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (1024, 1024))

    img = np.transpose(img,[2,0,1])

    img = np.array(img, np.float32)/255.0

    mask = np.array(mask, np.float32)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    # print('default_loader mask: ', mask.shape)
    return img, mask
    
class ChangeSet(data.Dataset):
    NUM_CLASS = 2  # NUM_CLASS = 2

    def __init__(self, root,split):
        self.metainfo={'classes':[0,1]}
        self.split=split
        root=os.path.join(root,split)

        # import pdb; pdb.set_trace()
        # 2.Get data list (P.S.Data and Labels are in the same folder)
        # imagelist = filter(lambda x: x.find('png') == -1, os.listdir(root+'/A/'))
        imagelist = os.listdir(root+'/A/')

        self.ids = list(imagelist)
        # print(self.ids)

        self.loader = default_loader
        self.root = root
        self.tran = transforms.ToTensor()
    
    def __getitem__(self, index):
        id = self.ids[index]

        img, mask = self.loader(id, self.root,self.split)
        img = torch.Tensor(img)
        mask = torch.FloatTensor(mask)
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
        return img, mask
    def __len__(self):
        return len(self.ids)

