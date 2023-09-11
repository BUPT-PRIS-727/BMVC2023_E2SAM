import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
class focal_loss(nn.Module):
    def __init__(self):
        super(focal_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
    def focal_loss(self, loss):
        gamma = 2
        alpha = 0.25

        pt = 1 - torch.exp(-1 * loss)
        ptgamma = pt ** gamma
        result = ptgamma * pt *alpha
        return result
    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        c = self.focal_loss(a)
        # d = self.mse_loss(y_true,y_pred)
        return c

class bce_loss(nn.Module):
    def __init__(self):
        super(bce_loss, self).__init__()
        # self.bce_loss = nn.BCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred.float(), y_true.float())
        return a

class ce_loss(nn.Module):
    def __init__(self):
        super(ce_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    def __call__(self, y_pred, y_true):
        # y_true = torch.unsqueeze(y_true,1)
        # print('y_pred: ', y_pred.shape)
        # print('y_true: ', y_true.shape)
        a = self.ce_loss(y_pred, y_true.long())
        return a
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)

def MMD(source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source.flatten(1), target.flatten(1),
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    return (XX + YY - XY -YX)