# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:03:52 2018

@author: yzzhao2
"""

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class GrayImageNetTrain(Dataset):
    def __init__(self, baseroot, root, stringlist, scalarlist):             # the inputs should be lists
        self.base = baseroot
        self.imgname = root
        self.stringlist = stringlist
        self.scalarlist = scalarlist
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __getitem__(self, index):
        # image processing
        imgname = self.imgname[index]                                       # name of one image
        imgpath = self.base + imgname                                       # path of one image
        img = Image.open(imgpath)                                           # read one image
        img = img.convert('L').resize((224, 224), Image.ANTIALIAS)          # pre-processing
        #img = img.convert('L').resize((256, 256), Image.ANTIALIAS)         # pre-processing
        img = self.transform(img)                                           # normalized Gray Tensor
        
        stringname = imgname[:9]                                            # category by str: like n01440764
        for index, value in enumerate(self.stringlist):
            if stringname == value:
                target = self.scalarlist[index]                             # target: 1~1000
                target = int(target) - 1                                    # target: 0~999
                target = np.array(target, dtype = np.int64)
                target = torch.from_numpy(target)
        return img, target

    def __len__(self):
        return len(self.imgname)
