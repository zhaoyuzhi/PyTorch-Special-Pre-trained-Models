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
from skimage import color

class LabImageNetTrain(Dataset):
    def __init__(self, baseroot, root, stringlist, scalarlist):             # the inputs should be lists
        self.base = baseroot
        self.imgname = root
        self.stringlist = stringlist
        self.scalarlist = scalarlist
        self.transform = transforms.ToTensor()
    
    def get_lab(self, img):
        # pre-processing, let all the images are in RGB color space
        img = img.resize((224, 224), Image.ANTIALIAS).convert('RGB')        # PIL Image RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        img = np.array(img)                                                 # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        # convert RGB to Lab, finally get Tensor
        lab = color.rgb2lab(img).astype(np.float32)                         # skimage Lab: L [0, 100], a [-128, 127], b [-128, 127], order [H, W, C]
        lab = self.transform(lab)                                           # Tensor Lab: L [0, 100], a [-128, 127], b [-128, 127], order [C, H, W]
        # normaization
        lab[[0], ...] = lab[[0], ...] / 50 - 1.0                            # L, normalized to [-1, 1]
        lab[[1, 2], ...] = lab[[1, 2], ...] / 110.0                         # a and b, normalized to [-1, 1], approximately
        return lab
    
    def __getitem__(self, index):
        # image processing
        imgname = self.imgname[index]                                       # name of one image
        imgpath = self.base + imgname                                       # path of one image
        img = Image.open(imgpath)                                           # read one image
        img = self.get_lab(img)                                             # normalized Lab Tensor
        
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
