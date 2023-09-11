# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from clearml import Task
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from typing import Optional, Tuple, Type
from functools import partial

from .common import LayerNorm2d, MLPBlock
from .segment_encoder import ImageEncoderViT
def get_sa_change(checkpoint=None,criterion_seg=None):
    return SA(criterion_seg=criterion_seg)

class teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        self.teacher_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2,5,8,11],
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        out = self.teacher_encoder(x)
        return out


class student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        self.student_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2,5,8,11],
            window_size=14,
            out_chans=prompt_embed_dim,
            in_chans=6
        )
        # self.student_conv= nn.Sequential(
        #     nn.Conv2d(
        #         768,
        #         256,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(256),
        # )
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        out = self.student_encoder(x)
        # out=self.student_conv(out.permute(0, 3, 1, 2))

        return out

class SA(nn.Module):
    def __init__(self,criterion_seg=None) -> None:
        super().__init__()
        self.nclass=23
        self.criterion=criterion_seg
        self.student_encoder=student()
        self.teacher_encoder=teacher()
        self.sa_decoder=SA_decoder()
        self.teacher_conv= nn.Sequential(
            nn.Conv2d(
                768,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

        self.share_encoder=nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

        self.student_split=nn.ModuleList()
        for _ in range(2):
            self.student_split.append(nn.Sequential(
            nn.Linear(768,256),
            nn.LayerNorm(256)
        ))
        self.student_conv= nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )
        self.criterion2=nn.CrossEntropyLoss()
        self.student_combine=nn.Sequential(
            nn.Linear(256*3,256),
            nn.LayerNorm(256)
        )
        self.load_weights(pretrained='./pretrain/sam_vit_h_4b8939.pth')
    def forward(self,x: torch.Tensor,y: torch.Tensor,channel,epoch=None) -> torch.Tensor:
        B,C,H,W=x.shape
        student_feat=self.student_encoder(x)
        teacher_x=x
        with torch.no_grad():
            teacher_feat=self.teacher_encoder(teacher_x[:,[channel*3,channel*3+1,channel*3+2],:,:])
            teacher_feat=self.teacher_conv(teacher_feat.permute(0, 3, 1, 2))
        
        student_feat_combine=self.student_split[channel](student_feat)
        student_feat_combine=self.student_conv(student_feat_combine.permute(0, 3, 1, 2))

        if self.criterion is not None:
            rec=self.sa_decoder(student_feat)
            if epoch is not None and epoch%2==0:
                Task.current_task().get_logger().report_image("image", "image/A", iteration=epoch, image=(x.detach().cpu().numpy()*255).astype(np.uint8)[0,:3].transpose([1,2,0])[:,:,[2,1,0]])
                Task.current_task().get_logger().report_image("image", "image/B", iteration=epoch, image=(x.detach().cpu().numpy()*255).astype(np.uint8)[0,3:].transpose([1,2,0])[:,:,[2,1,0]])

                Task.current_task().get_logger().report_image("image", "rec", iteration=epoch, image=(rec.max(1)[1].detach().cpu().numpy()*255).astype(np.uint8)[0,:,:])
            
            loss1= self.criterion(student_feat_combine,teacher_feat)
            loss2=self.criterion2(rec.view(B,2,-1),y.view(B,-1).long())
            return {'f_loss':loss1,'rec_loss':loss2}
        return student_feat,teacher_feat
    def load_weights(self,pretrained=None):
        if pretrained is not None:
            state_dict = {}
            with open(pretrained, "rb") as f:
                pretrained_model = torch.load(f)
                
                for key, value in pretrained_model.items():
                    if key.startswith("image_encoder.") and 'neck' not in key:
                        state_dict[key.replace("image_encoder","teacher_encoder")]=value
                self.teacher_encoder.load_state_dict(state_dict)
                state_dict = {}
                state_dict['teacher_conv.0.weight']=pretrained_model['image_encoder.neck.0.weight']
                state_dict['teacher_conv.1.weight']=pretrained_model['image_encoder.neck.1.weight']
                state_dict['teacher_conv.1.bias']=pretrained_model['image_encoder.neck.1.bias']
                state_dict['share_encoder.0.weight']=pretrained_model['image_encoder.neck.2.weight']
                state_dict['share_encoder.1.weight']=pretrained_model['image_encoder.neck.3.weight']
                state_dict['share_encoder.1.bias']=pretrained_model['image_encoder.neck.3.bias']
                self.load_state_dict(state_dict,strict=False)
class SA_decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(
                768,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )
        filters = [64, 128, 256, 256]
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, 1, 1)
        

        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 2, 3, padding=1)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x=self.neck(x.permute(0, 3, 1, 2))
        d4 = self.decoder4(x) 
        d3 = self.decoder3(d4) 
        d2 = self.decoder2(d3) 
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out
           
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x    
nonlinearity = partial(F.relu, inplace=True)