from torch.utils.data import DataLoader
from ..datasets.h8 import H8Set
 
def get_dataset():
    train_dataset=H8Set("datapath","train_all")
    val_dataset=H8Set("datapath","val_2021")

    train_dataloader=DataLoader(batch_size=4,
                                shuffle=True,
                                num_workers=4,
                                dataset=train_dataset)

    val_dataloader=DataLoader(batch_size=8,
                            shuffle=False,
                            num_workers=2,
                            dataset=val_dataset)
    return (train_dataloader,val_dataloader)