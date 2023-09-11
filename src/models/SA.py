# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from typing import Optional, Tuple, Type
from functools import partial

from .common import LayerNorm2d, MLPBlock
from .distill_encoder import ImageEncoderViT
def get_sa(checkpoint=None,criterion_seg=None):
    return SA(criterion_seg=criterion_seg)




class student(nn.Module):
    def __init__(self,in_chans=6) -> None:
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
            in_chans=in_chans
        )
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

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        out = self.student_encoder(x)
        out = self.neck(out.permute(0, 3, 1, 2))

        return out

class SA(nn.Module):
    def __init__(self,criterion_seg=None,in_chans=6) -> None:
        super().__init__()
        self.nclass=23
        self.criterion=criterion_seg
        self.student_encoder=student(in_chans=in_chans)

        # self.load_weights(pretrained=)
    def forward(self,x) -> torch.Tensor:
        B,C,H,W=x.shape
        student_feat=self.student_encoder(x)
        return student_feat # [B,256,64,64]
    def load_weights(self,pretrained=None):
        if pretrained is not None:
            state_dict = {}
            with open(pretrained, "rb") as f:
                pretrained_model = torch.load(f,map_location=torch.device('cpu'))['state_dict']
                
                for key, value in pretrained_model.items():
                    if key.startswith("student_encoder.student_encoder.") and key in self.state_dict().keys():
                        state_dict[key]=value
                    else:
                        print('Droppig: ', key)
                self.load_state_dict(state_dict,strict=False)
                # import pdb;pdb.set_trace()
                state_dict = {}
                pretrained_model2 = torch.load("./pretrained/sam_vit_b_01ec64.pth",map_location=torch.device('cpu'))
                for key, value in pretrained_model2.items():
                    if "neck" in key:
                        state_dict[key.replace('image_encoder','student_encoder')]=value    
                self.load_state_dict(state_dict,strict=False)

