from mmengine.runner import Runner
from src.build.segment_build import get_dlinkvit_4_base
from src.build.shift_dataset_build import get_dataset

from src.evaluators.iou_metric import IoUMetric
from torch.optim import AdamW
from clearml import Task
import torch
if __name__=="__main__":

    model=get_dlinkvit_4_base()
    
    encoder_params = list(map(id, model.vit_backbone.student_encoder.student_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in encoder_params, model.parameters())
    params = [
    {'params': base_params, 'lr': 1e-4},
    {'params': model.vit_backbone.student_encoder.student_encoder.parameters(), 'lr': 5e-5},
    ]
    optimizer = AdamW(params, lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0.05)
    train_dataloader,val_dataloader=get_dataset()
    param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=100)
    runner=Runner(
        model=model,
        work_dir="./save",
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=optimizer),
        train_cfg=dict(by_epoch=True,max_epochs=100,val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=IoUMetric),
        param_scheduler=param_scheduler,
        cfg=dict(compile=False)
    )
    runner.train()