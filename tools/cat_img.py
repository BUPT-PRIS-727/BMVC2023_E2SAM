from random import shuffle
import numpy as np
import os
from tqdm import tqdm
path=""
save=""
filelist=os.listdir(path)
shuffle(filelist)
file_num=len(filelist)//16
for i in tqdm(range(file_num)):
    tmp=np.zeros((1024,1024,17),dtype=np.uint8)
    for j,filename in enumerate(filelist[i*16:(i+1)*16]):
        filepath=os.path.join(path,filename)
        npy=np.load(filepath)
        tmp[(j//4)*256:((j//4)+1)*256,(j%4)*256:((j%4)+1)*256,:]=npy
    np.save(os.path.join(save,filelist[i*16]),tmp)

