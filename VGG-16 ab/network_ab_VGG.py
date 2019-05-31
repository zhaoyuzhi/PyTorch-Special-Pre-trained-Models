# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:04:52 2018

@author: yzzhao2
"""

import torch
import torch.nn as nn
from spectralnorm import SpectralNorm

#-----------------------------------------------
#           VGG16 for ab color space
#-----------------------------------------------
class abVGG16(nn.Module):
    def __init__(self, in_dim = 2, num_classes = 1000):
        super(abVGG16, self).__init__()
        # feature extraction part
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_dim, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool1 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool2 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool3 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool4 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool5 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):                                   # shape: [B, 3, 224, 224]
        conv1 = self.conv1(x)                               # shape: [B, 64, 224, 224]
        pool1 = self.pool1(conv1)                           # shape: [B, 64, 112, 112]
        conv2 = self.conv2(pool1)                           # shape: [B, 128, 112, 112]
        pool2 = self.pool2(conv2)                           # shape: [B, 128, 56, 56]
        conv3 = self.conv3(pool2)                           # shape: [B, 256, 56, 56]
        pool3 = self.pool3(conv3)                           # shape: [B, 256, 28, 28]
        conv4 = self.conv4(pool3)                           # shape: [B, 512, 28, 28]
        pool4 = self.pool4(conv4)                           # shape: [B, 512, 14, 14]
        conv5 = self.conv5(pool4)                           # shape: [B, 512, 14, 14]
        pool5 = self.pool5(conv5)                           # shape: [B, 512, 7, 7]
        pool5 = pool5.view(x.size(0), -1)                   # shape: [B, 512 * 7 * 7]
        x = self.classifier(pool5)                          # shape: [B, 1000]
        return x

class abVGG16_BN(nn.Module):
    def __init__(self, in_dim = 2, num_classes = 1000):
        super(abVGG16_BN, self).__init__()
        # feature extraction part
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_dim, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool5 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = False),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):                                   # shape: [B, 3, 224, 224]
        conv1 = self.conv1(x)                               # shape: [B, 64, 224, 224]
        pool1 = self.pool1(conv1)                           # shape: [B, 64, 112, 112]
        conv2 = self.conv2(pool1)                           # shape: [B, 128, 112, 112]
        pool2 = self.pool2(conv2)                           # shape: [B, 128, 56, 56]
        conv3 = self.conv3(pool2)                           # shape: [B, 256, 56, 56]
        pool3 = self.pool3(conv3)                           # shape: [B, 256, 28, 28]
        conv4 = self.conv4(pool3)                           # shape: [B, 512, 28, 28]
        pool4 = self.pool4(conv4)                           # shape: [B, 512, 14, 14]
        conv5 = self.conv5(pool4)                           # shape: [B, 512, 14, 14]
        pool5 = self.pool5(conv5)                           # shape: [B, 512, 7, 7]
        pool5 = pool5.view(x.size(0), -1)                   # shape: [B, 512 * 7 * 7]
        x = self.classifier(pool5)                          # shape: [B, 1000]
        return x

# Fully convolutional layers in feature extraction part compared to abVGG16_BN
# This is for generator only, so BN cannot be attached to the input and output layers of feature extraction part
# Each output of block (conv*) is "convolutional layer + LeakyReLU" that avoids feature sparse
# We replace the adaptive average pooling layer with a convolutional layer with stride = 2, to ensure the size of feature maps fit classifier
class abVGG16_FC_BN(nn.Module):
    def __init__(self, in_dim = 2, num_classes = 1000):
        super(abVGG16_FC_BN, self).__init__()
        # feature extraction part
        # conv1 output size 224 * 224
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = in_dim, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv2 output size 112 * 112
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv3 output size 56 * 56
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv4 output size 28 * 28
        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv5 output size 14 * 14
        self.conv5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # conv6 output size 7 * 7
        self.conv6 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)),
            nn.LeakyReLU(0.2, inplace = True)
        )
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):                                   # shape: [B, 3, 224, 224]
        conv1 = self.conv1(x)                               # shape: [B, 64, 224, 224]
        conv2 = self.conv2(conv1)                           # shape: [B, 128, 112, 112]
        conv3 = self.conv3(conv2)                           # shape: [B, 256, 56, 56]
        conv4 = self.conv4(conv3)                           # shape: [B, 512, 28, 28]
        conv5 = self.conv5(conv4)                           # shape: [B, 512, 14, 14]
        conv6 = self.conv6(conv5)                           # shape: [B, 512, 7, 7]
        conv6 = conv6.view(x.size(0), -1)                   # shape: [B, 512 * 7 * 7]
        x = self.classifier(conv6)                          # shape: [B, 1000]
        return x
