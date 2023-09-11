import os
from src.datasets.fog import FogSet
from torch.utils.data import DataLoader
from src.build.segment_build import get_sa_h8_512
import torch
import tqdm
from src.evaluators.iou_metric import IoUMetric
from torch.optim import AdamW
from mmengine.runner import Runner
from torchvision import transforms
from PIL import Image
import numpy as np

test_dataset=FogSet("","test")
test_dataloader=DataLoader(batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                dataset=test_dataset)



def inference(model, save_path):
    model.eval()
    model=model.cuda()
    with torch.no_grad():
        for inputs in test_dataloader:
            x,y,id=inputs
            x=x.cuda()
            y=y.cuda()
            img_name = id[0]+"_output.png"
            outputs = model(x,y,mode='val')
            outputs = outputs[0]
            outputs = outputs.squeeze(0)
            img = outputs.cpu().detach().numpy()*255
            print("shape", img.shape)

            img = Image.fromarray(np.uint8(img))
            savepath = os.path.join(save_path, img_name)
            img.save(savepath)
            print("image:"+img_name)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=get_sa_h8_512()
    model.load_state_dict(torch.load("", map_location='cpu')['state_dict'])
    model.to(device)
    save_path = './save'
    inference(model, save_path)


