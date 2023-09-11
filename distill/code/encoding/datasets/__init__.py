from .cloud import CloudSet
from .h8 import H8Set
from .cloud_big import CloudBigSet
from .shift import Shift
from .change import ChangeSet
datasets = {
    'cloud': CloudSet,
    'h8': H8Set,
    'cloud_big': CloudBigSet,
    'shift':Shift,
    'change':ChangeSet
}

def get_segmentation_dataset(name,**kwargs):
    return datasets[name.lower()](**kwargs)
