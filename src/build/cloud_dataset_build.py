from torch.utils.data import DataLoader
from ..datasets.cloud import CloudSet
from ..datasets.cloud_1024 import CloudSet_1024

def get_dataset():
    train_dataset=CloudSet("datapath","train",mode="train")
    val_dataset=CloudSet("datapath","val",mode='test')

    train_dataloader=DataLoader(batch_size=2,
                                shuffle=True,
                                dataset=train_dataset)

    val_dataloader=DataLoader(batch_size=4,
                            shuffle=False,
                            dataset=val_dataset)
    return (train_dataloader,val_dataloader)


def get_dataset_16():
    train_dataset=CloudSet_1024("/datapath","train",mode="train")
    val_dataset=CloudSet_1024("/datapath","val",mode='test')

    train_dataloader=DataLoader(batch_size=2,
                                shuffle=True,
                                dataset=train_dataset)

    val_dataloader=DataLoader(batch_size=4,
                            shuffle=False,
                            dataset=val_dataset)
    return (train_dataloader,val_dataloader)