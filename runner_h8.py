from mmengine.runner import Runner
from src.build.segment_build import get_dlinkvit_16_base
from src.build.h8_dataset_build import get_dataset
from src.evaluators.iou_metric import IoUMetric
from torch.optim import AdamW
from clearml import Task
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from mmengine.utils.version_utils import digit_version
from mmengine.model import MMFullyShardedDataParallel
if __name__=="__main__":
    pretrained=""
    model=get_dlinkvit_16_base(pretrained)
    encoder_params = list(map(id, model.vit_backbone.student_encoder.student_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in encoder_params, model.parameters())
    params = [
    {'params': base_params, 'lr': 1e-4},
    {'params': model.vit_backbone.student_encoder.student_encoder.parameters(), 'lr': 5e-5},
    ]
    train_dataloader,val_dataloader=get_dataset()
    param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=20)
    runner=Runner(
        model=model,
        work_dir="./save",
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type=AdamW,betas=(0.9, 0.999), eps=1e-8,lr=1e-4,weight_decay=0.05)),
        train_cfg=dict(by_epoch=True,max_epochs=20,val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=IoUMetric),
        param_scheduler=param_scheduler,
        # load_from="/data3/mmengine/epoch_16.pth",
        cfg=dict(compile=True)
    )
    runner.train()