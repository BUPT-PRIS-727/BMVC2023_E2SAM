import argparse
import logging
import os
import random
import time
# import wandb
import cv2
import numpy as np
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
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable as V

from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model
from encoding.nn import BootstrappedCELoss, OhemCELoss, SegmentationLosses, dice_ce_loss,focal_loss,ce_loss,bce_loss
from encoding.utils import (AverageMeter, LR_Scheduler,
                            intersectionAndUnionGPU, save_checkpoint)
from option_distill_change import Options
from clearml import Logger, Task
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

# os.environ["NCCL_DEBUG"] = "INFO"
# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)

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
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)

    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # criterion_seg =nn.KLDivLoss(reduction ='mean')
    criterion_seg =nn.MSELoss()

    model = get_segmentation_model(args.model,criterion_seg=criterion_seg) ##aux
    # model = torch.compile(model=model)


    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.vis_root) ##tensorboard --logdir vis_root
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(model.nclass))
        logger.info(model)
    share_encoder_params = list(map(id, model.share_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in share_encoder_params, model.parameters())
    params = [
    {'params': base_params, 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        scaler = GradScaler(enabled=True)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        scaler = GradScaler()
        model = torch.nn.DataParallel(model.cuda())

    # resuming checkpoint
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
        args.start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        if not args.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    # clear start epoch if fine-tuning
    if args.ft:
        args.start_epoch = 0

       
    # dataset
    train_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='train')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                               shuffle=(train_sampler is None), num_workers=args.workers, 
                                               pin_memory=True, sampler=train_sampler, drop_last=True)

    print(len(train_loader))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 2268, eta_min = 1e-6)
    # scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), warmup_epochs=10)
    currentSteps=[0]
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if main_process():
            logger.info('>>>>>>>>>>>>>>>> Start One Epoch Training >>>>>>>>>>>>>>>>')
        # import pdb;pdb.set_trace()
        try:
            loss_train= train(train_loader, model, scaler, optimizer, epoch, scheduler, train_data.NUM_CLASS,currentSteps,wandbHandle=None)
        except:
            if main_process():
                print("start save")
                filename = args.model_savefolder + 'model_'+str(epoch)+'.pth'
                torch.save({'epoch': epoch_log, 
                            'state_dict': model.module.state_dict(), 
                            'optimizer': optimizer.state_dict()}, 
                            filename)
                print("save finished")

        if (epoch) % 20 == 0  and main_process():
            filename = args.model_savefolder + 'model_'+str(epoch)+'.pth'
            torch.save({'epoch': epoch_log, 
                        'state_dict': model.module.state_dict(), 
                        'optimizer': optimizer.state_dict()}, 
                        filename)
        
        # wandb.finish()

def train(train_loader, model, scaler, optimizer, epoch, scheduler, nclass,currentSteps,wandbHandle=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    channel_meter=[AverageMeter() for i in range(2)]
    f_loss_meter = AverageMeter()
    rec_loss_meter = AverageMeter()


    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    if main_process():
        vis=epoch
    else:
        vis=None
    for i, (input,target) in enumerate(train_loader):
        # import pdb;pdb.set_trace()
        currentSteps[0]+=1
        data_time.update(time.time() - end)
        channels = [i for i in range(0, 2)]
        selected_channels = random.sample(channels, 1)
        for c in selected_channels:
            channel_meter[c].update(0,input.size(0))
        # scheduler(optimizer, i, epoch)
        input = input.cuda(non_blocking=True)
        target =torch.unsqueeze(target, dim=1) 
        optimizer.zero_grad()
        # with autocast():
        with autocast():
            main_loss = model(input,target, selected_channels[0],epoch=vis) ##jpu
            vis=None
                # Task.current_task().get_logger().report_image("image", "input", iteration=currentSteps[0], image=input.cpu().numpy().astype(np.uint8)[0,[2,3,13]].transpose([1,2,0])[:,:,[2,1,0]])
                # Task.current_task().get_logger().report_image("image", "target", iteration=currentSteps[0], image=target.cpu().numpy().astype(np.uint8)[0]*255)
                # Task.current_task().get_logger().report_image("image", "output", iteration=currentSteps[0], image=output.cpu().numpy().astype(np.uint8)[0]*255)

            loss = main_loss['f_loss']*0.1+main_loss['rec_loss'] #+ 0.25 * 0.4 * jpu_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # scheduler.step()  # batch内更新lr
        scaler.update()

        # loss.backward()
        # optimizer.step()

        n = input.size(0)

        if args.multiprocessing_distributed:
            loss = loss.detach() * n  # not considering ignore pixels ## loss


        f_loss_meter.update(main_loss['f_loss'].item(), n) ##
        rec_loss_meter.update(main_loss['rec_loss'].item(), n) ##
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        lr = optimizer.param_groups[0]['lr']
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % 5 == 0 and main_process():
            logger.info('Epoch:[{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'LR:{lr:.6f} '
                        'Remain:{remain_time} '
                        # 'MainLoss:{main_loss_meter.val:.3f}({main_loss_meter.avg:.3f}) '
                        # 'AuxLoss:{aux_loss_meter.val:.3f}({aux_loss_meter.avg:.3f}) '
                        # 'JPU Loss:{jpu_loss_meter.val:.3f}({jpu_loss_meter.avg:.3f}) ' ##
                        'F_Loss:{f_loss_meter.val:.3f}({f_loss_meter.avg:.3f}) '
                        'Rec_Loss:{rec_loss_meter.val:.3f}({rec_loss_meter.avg:.3f}) '.format(epoch+1, args.epochs,
                                                         i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                         lr=lr,
                                                         remain_time=remain_time,

                                                         f_loss_meter=f_loss_meter,
                                                         rec_loss_meter=rec_loss_meter,
                                                         ))
        # if i % 200 == 0:
        #     filename = args.model_savefolder + 'model_'+str(epoch)+'.pth'
        #     torch.save({'iter': i, 
        #                 'state_dict': model.module.state_dict(), 
        #                 'optimizer': optimizer.state_dict()}, 
        #                 filename)
        if main_process():
            writer.add_scalar('f_loss_train_batch', f_loss_meter.val, currentSteps[0]) ##
            writer.add_scalar('rec_loss_train_batch', rec_loss_meter.val, currentSteps[0]) ##
            for iiii in range(2):
                Logger.current_logger().report_scalar(
                    "channels count", "channel_{}".format(iiii), iteration=currentSteps[0], value=channel_meter[iiii].count
                )
            writer.add_scalar('Lr', lr, currentSteps[0])
        # scheduler.step()
    scheduler.step()  # epoch内不更新lr，epoch之间更新lr

    if main_process():
        logger.info('>>>>>>>>>>>>>>>> One Training Epoch Done >>>>>>>>>>>>>>>>')
        logger.info('Train epoch [{}/{}]: F_Loss {:.3f} '.format(epoch+1, args.epochs,f_loss_meter.avg))
        logger.info('Train epoch [{}/{}]: R_Loss {:.3f} '.format(epoch+1, args.epochs,rec_loss_meter.avg))
        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        logger.info('\n')
    return f_loss_meter.avg

def validate(val_loader, model, criterion, nclass,epoch):
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
        target = target.cuda(non_blocking=True)
        # target = (target.cuda(non_blocking=True),False)
        with torch.no_grad():
            output = model(input)
        # if args.zoom_factor != 8:
        #     output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output,target)

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
        if(i==50):
            Task.current_task().get_logger().report_image("val_all", "input", iteration=epoch, image=input.cpu().numpy().astype(np.uint8)[0,[2,3,13]].transpose([1,2,0])[:,:,[2,1,0]])
            Task.current_task().get_logger().report_image("val_all", "target", iteration=epoch, image=target.cpu().numpy().astype(np.uint8)[0]*255)
            Task.current_task().get_logger().report_image("val_all", "output", iteration=epoch, image=output.cpu().numpy().astype(np.uint8)[0]*255)

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

