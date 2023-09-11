from torch.utils.data import DataLoader
from ..datasets.shift import Shift

def get_dataset():
    train_dataset=Shift("train path","train","shift_train.txt")
    val_dataset=Shift("val path","val","shift_val.txt")

    train_dataloader=DataLoader(batch_size=2,
                                shuffle=True,
                                dataset=train_dataset)

    val_dataloader=DataLoader(batch_size=4,
                            shuffle=False,
                            dataset=val_dataset)
    return (train_dataloader,val_dataloader)