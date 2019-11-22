# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:04:34 2018

@author: ZHAO Yuzhi
"""

import argparse
import os

import trainer

def main(opt):
    trainer.Trainer(opt)

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    # Training setting refers to VGG-Net paper in ICLR 2015
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 120, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'size of the batches')
    parser.add_argument('--baseroot', type = str, default = "/media/ztt/6864FEA364FE72E4/zhaoyuzhi/ILSVRC2012_train_256/",
        help = 'the training folder, while adding "\\" at the end of this string')
    parser.add_argument('--lr', type = float, default = 0.01, help='SGD: learning rate')
    parser.add_argument('--momentum', type=float, default = 0.9, help='SGD: momentum')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'SGD: weight-decay, L2 normalization')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 30, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.1, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--img_height', type = int, default = 256, help = 'size of image height')
    parser.add_argument('--img_width', type = int, default = 256, help = 'size of image width')
    parser.add_argument('--checkpoint_interval', type = int, default = 5, help = 'interval between model checkpoints')
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-trained model exists ot not')
    parser.add_argument('--load_type', type = str, default = "ResNet18_Gray",
        help = 'load_type: load the model corresponding to network type; there are 3 types: GrayVGG16 | GrayVGG16_BN | GrayVGG16_FC_BN')
    parser.add_argument('--load_epoch', type = str, default = "10", help = 'load id: load the model corresponding to certain epoch')
    parser.add_argument('--load_batchsize', type = str, default = "64", help = 'load id: load the model corresponding to certain batchsize')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0, 1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'if input data structure is unchanged, set it as True')
    opt = parser.parse_args()
    print(opt)
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Enter main function
    main(opt)
