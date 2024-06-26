# PyTorch-Special-Pre-trained-Models

Special pre-trained VGG-16 network on CIE Lab and Grayscale images converted from ImageNet training set

## 1 Model Validation Accuracy (on ImageNet Validation 50k)

Compared to the official model provided by PyTorch, the classification ability of our model is only slightly weaker. Basically, these models are targeted for regression task, so we think the small improvement is unnecessary.

The **Fully Convolutional** models do not include MaxPooling layer & AdaptiveAvgPooling layer, which is replaced by a convolutional layer with stride = 2. Note that, the total amount of convolutional layer is unchanged. Each convolutional layer is spectral normalized (you may find the source code in this [project](https://github.com/zhaoyuzhi/PyTorch-Useful-Codes)), which is very useful for the training of WGAN.

### 1.1  Fully Convolutional Gray VGG-16

- epoch 30 | top 1: 32.50% | top 5: 59.90%

- epoch 60 | top 1: 56.28% | top 5: 80.66%

- epoch 90 | top 1: 57.96% | top 5: 81.80%

- epoch 120 | top 1: 57.46% | top 5: 81.31%

### 1.2  CIE Lab VGG-16

- epoch 30 | top 1: 37.41% | top 5: 65.55%

- epoch 60 | top 1: 60.29% | top 5: 83.83%

- epoch 90 | top 1: 60.38% | top 5: 83.11%

- epoch 120 | top 1: 60.87% | top 5: 83.17%

### 1.3  Fully Convolutional CIE Lab VGG-16

- epoch 30 | top 1: 35.70% | top 5: 63.22%

- epoch 60 | top 1: 59.21% | top 5: 83.16%

- epoch 90 | top 1: 65.15% | top 5: 86.72%

- epoch 120 | top 1: 65.13% | top 5: 86.80%

- We further tested validation accuracy when the learning was further decreasing to 1e-6; however, there is no explicit improvement. For example, when epoch = 125, the top 1 accuracy = 65.16% and top 5 accuracy = 86.81%.

### 1.4  Fully Convolutional RGB ResNet-50 IN

- For epoch 5, the top 1 accuracy is 55.77% and top 5 accuracy is 79.64%. For epoch 15, the top 1 accuracy is 57.16% and top 5 accuracy is 80.79%.

## 2 Download Link

### 2.1  Fully Convolutional Gray VGG-16

Now the `epoch 120 model of Fully Convolutional Gray VGG-16` is available: [Link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/Eo5EUfHprwJMrUTAXnFy0_AB45n_EHQC3RNs-mcZgnFcVA?e=npBwqL).

### 2.2  CIE Lab VGG-16 BN

Now the `epoch 120 model of CIE Lab VGG-16 BN` is available: [Link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EtNJ_q3FIBNLjlH730hvH9oB5alYbw4LjOYS-54FzmVOZg?e=qKAW0I)

### 2.3  Fully Convolutional CIE Lab VGG-16

Now the `epoch 120 model of Fully Convolutional CIE Lab VGG-16` is available: [Link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EpFho59uMWZJm67k9I_1hoYB7veVRjUooywOBOKrj_-2hw?e=b8Tc9E)

### 2.4  Gray VGG-16 with Spectral Norm

Now the `epoch 120 model of Gray VGG-16 with Spectral Norm` is available: [Link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EiZ_t8UZBnJCtJ-FGAqtg18BHfH99BILJDoWKBxwdL_OQg?e=E7jIlT)

### 2.5  ResNet-50-RGB IN

Now the `epoch 60 model of ResNet-50-RGB IN` is available: [Link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EkLRIq3lay1ImMd58PcFya4BDRZ4VgMXmYe92UNmK2UpYw?e=g4acZT)

### 2.6  ResNet-50-Gray BN

Now the `epoch 60 model of ResNet-50-Gray BN` is available: [Link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/Ei_r7TdTv2dAjCTi2h55r_YBjqMuQOW-uAWiJXwQ0IMfFQ?e=B24HQ0)

### 2.7  ResNet-50-Gray IN

Now the `epoch 120 model of ResNet-50-Gray BN` is available: [Link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/EtXtamdjGEZFrD3tFuHOX-YBi3-Is3yXRF2NZt5EDZkvZA?e=0mRyvj)

Other models pending...

## 3 Convert

Normally, we save the whole model as a `.pth` file. If you want the weights only, please run `convert.py`

## 4 Acknowledgement

If you use `Fully Convolutional Gray VGG-16` or `Gray VGG-16 with Spectral Norm`, please consider cite:
```bash
@article{zhao2020scgan,
  title={SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network},
  author={Zhao, Yuzhi and Po, Lai-Man and Cheung, Kwok-Wai and Yu, Wing-Yin and Abbas Ur Rehman, Yasar},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2020},
  publisher={IEEE}
}
```

If you use `Fully Convolutional ResNet-50-Gray BN`, please consider cite:
```bash
@article{zhao2021vcgan,
  title={VCGAN: Video Colorization with Hybrid Generative Adversarial Network},
  author={Zhao, Yuzhi and Po, Lai-Man and Yu, Wing-Yin and Rehman, Yasar Abbas Ur and Liu, Mengyang and Zhang, Yujia and Ou, Weifeng},
  journal={arXiv preprint arXiv:2104.12357},
  year={2021}
}
```
