import argparse
import logging
import os
import random
import time

import cv2
import numpy as np
from clearml import Task
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transform
from tensorboardX import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model
from encoding.nn import BootstrappedCELoss, OhemCELoss, SegmentationLosses, dice_ce_loss
from encoding.utils import (AverageMeter, LR_Scheduler,
                            intersectionAndUnionGPU, save_checkpoint)
from option import Options
from torch.autograd import Variable as V

import torch._dynamo


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = Options().parse()
    args.train_gpu = list(range(torch.cuda.device_count()))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    main_worker(args)

def main_worker(argss):
    global args
    args = argss
    ngpus_per_node=torch.cuda.device_count()
    args.ngpus_per_node=ngpus_per_node

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    args.rank=rank
    device_id = rank % torch.cuda.device_count()

    cloud_weight = torch.FloatTensor([0.9172, 1.0003, 0.9828, 1.0141, 1.1030,
                                      1.0033, 1.0212, 1.0909, 0.9946, 1.1225])

    criterion_seg = dice_ce_loss(ignore_index=10, weight=cloud_weight)
    criterion_aux = nn.CrossEntropyLoss(ignore_index=10, weight=cloud_weight)


    # for s2unet
    model = get_segmentation_model(args.model, dataset = args.dataset, 
                                   norm_layer = nn.LayerNorm,
                                   criterion_seg=criterion_seg) ##aux
    model.load_state_dict(torch.load(args.pretrained)['state_dict'])
    model=model.to(device_id)
    

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.vis_root) ##tensorboard --logdir vis_root
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(model.nclass))
        logger.info(model)



    scaler = GradScaler()
    model = torch.nn.DataParallel(model.to(device_id))


    best_mIoU = 0.0



    # model = torch.compile(model=model)

    input_transform = transform.Compose([
        # transform.ToTensor(),
        transform.Normalize(
                    [0.48969566, 0.46122111, 0.40845247, 0.42575709, 0.34868406, 0.29353659,
                     0.34558332, 0.14569656, 0.17792995, 0.20482719, 0.28424213, 0.21427191,
                     0.29169121, 0.28796465, 0.27809529, 0.23621918], # mean
                    [0.12632009, 0.14350215, 0.18924507, 0.22366015, 0.19707532, 0.18446651,
                     0.04896662, 0.02361904, 0.02939885, 0.03480839, 0.06311043, 0.04050178,
                     0.06538713, 0.06614832, 0.06286721, 0.04817895] # std
                    )])
    # dataset
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}
    
    val_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='val', mode ='val',
                                           **data_kwargs)

    if ngpus_per_node>1:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=args.workers, 
                                             pin_memory=True, sampler=val_sampler)

    currentSteps=[0]
    loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion_seg, 10)
    if main_process():
        writer.add_scalar('loss_val', loss_val, 0)
        writer.add_scalar('mIoU_val', mIoU_val, 0)
        writer.add_scalar('mAcc_val', mAcc_val, 0)
        writer.add_scalar('allAcc_val', allAcc_val, 0)
    if main_process():
        filename = args.model_savefolder + 'last_model_1006.pth'

def validate(val_loader, model, criterion, nclass):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = V(target.cuda(non_blocking=True),False)
        with torch.no_grad():
            output = model(input)
        # if args.zoom_factor != 8:
        #     output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, nclass, 10)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 100 == 0) and main_process():
            logger.info('Evaluation: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) '
                        'Accuracy {accuracy:.2f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy * 100))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f}.'.format(loss_meter.avg, mIoU*100, mAcc*100, allAcc*100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class[i]*100, accuracy_class[i]*100))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        logger.info('\n')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    args = Options().parse()
    if(not os.path.exists(args.model_savefolder)):
        import pathlib
    
    task = Task.init(project_name='cloud', task_name='patch fft syncbatchNorm test')  # noqa: F841
    
    main()
