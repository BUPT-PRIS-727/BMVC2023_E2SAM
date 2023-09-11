from ..losses.loss_h8 import ce_loss, dice_ce_loss
from .segment_encoder import ImageEncoderViT
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from clearml import Logger
import numpy as np
from .SA import SA
from .common import LayerNorm2d, MLPBlock

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )





def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

build_sam = build_sam_vit_b

sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder=SA(in_chans=4)
    if checkpoint is not None:
        state_dict = {}
        with open(checkpoint, "rb") as f:
            pretrained_model = torch.load(f)
            
            for key, value in pretrained_model.items():
                if key.startswith("image_encoder."):
                    state_dict[key.replace("image_encoder.","")]=value
                else:
                    print('Droppig: ', key)
        image_encoder.load_state_dict(state_dict)
    return image_encoder

class DlinkVit_4_base(BaseModel):
    def __init__(self,checkpoint=None):
        super().__init__()
        
        self.vit_backbone=build_sam_vit_b(checkpoint=checkpoint)

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
        self.finalconv3 = nn.Conv2d(32, 23, 3, padding=1)
        self.softmax=nn.Softmax(dim=1)
        self.loss=nn.CrossEntropyLoss()

        # self.loss=dice_ce_loss(ignore_index=10, weight=torch.FloatTensor([0.9172, 1.0003, 0.9828, 1.0141, 1.1030,
        #                               1.0033, 1.0212, 1.0909, 0.9946, 1.1225]))
    def forward(self, x, y,mode=None):
        B,C,H,W=x.shape
        e4 = self.vit_backbone(x)
        d4 = self.decoder4(e4) 
        d3 = self.decoder3(d4) 
        d2 = self.decoder2(d3) 
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        if(mode=='loss'):
            res={'loss':self.loss(out.view(B,23,-1), y.view(B,-1).long())}
            return res
        out=out.max(1)[1].detach()
        return out,y

nonlinearity = partial(F.relu, inplace=True)
class Dblock(nn.Module):
    def __init__(self,channel,pretrained=None):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        if pretrained and pretrained.endswith('.pth'):
            self.load_pytorch_pretrained_dblock(pretrained)

    def load_pytorch_pretrained_dblock(self, pretrained_backbone_path):
        print('***In Dblock, Loading MAE pretrained backbone from', pretrained_backbone_path)
        pretrained_model = torch.load(pretrained_backbone_path, map_location='cuda')['model']
        state_dict = {}

        for key, value in pretrained_model.items():
            if key.startswith('dblock'):
                state_dict[key] = value

        self.load_state_dict(state_dict, strict=False)
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out # + dilate5_out
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
