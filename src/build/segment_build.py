# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import yaml

from functools import partial

from ..models.dlink_vit_16_base import DlinkVit_16_base
from ..models.dlink_vit_4_base import DlinkVit_4_base
from ..models.distill_SA_change import get_sa_change_512
from ..models.SA_h8 import get_sa_h8_512

def get_dlinkvit_16_base(checkpoint=None):
    return DlinkVit_16_base(checkpoint)

def get_dlinkvit_4_base(checkpoint=None):
    return DlinkVit_4_base(checkpoint)


def get_sa_change_512_base():
    return get_sa_change_512()


def get_sa_h8_512_base():
    return get_sa_h8_512()
