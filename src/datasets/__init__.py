from .h8 import H8Set
from .change import ChangeSet
from .cloud_1024 import CloudSet_1024
datasets = {
    'h8': H8Set,
    'change': ChangeSet,
    'cloud_1024':CloudSet_1024
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
