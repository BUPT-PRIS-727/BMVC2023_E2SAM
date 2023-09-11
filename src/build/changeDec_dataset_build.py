from torch.utils.data import DataLoader
from ..datasets.change import ChangeSet

def get_dataset():
    train_dataset=ChangeSet("datapath","train")
    val_dataset=ChangeSet("datapath","test")

    train_dataloader=DataLoader(batch_size=2,
                                shuffle=True,
                                dataset=train_dataset)

    val_dataloader=DataLoader(batch_size=4,
                            shuffle=False,
                            dataset=val_dataset)
    return (train_dataloader,val_dataloader)