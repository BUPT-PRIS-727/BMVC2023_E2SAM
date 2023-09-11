#from .encnet import *
from .common import *
from .SA import *
from .SA_change import *
from .SA_h8 import *
def get_segmentation_model(name, **kwargs):
    models = {
        'sa': get_sa,
        'change':get_sa_change,
        'h8':get_h8_512
    }
    print(kwargs)
    return models[name.lower()](**kwargs)
