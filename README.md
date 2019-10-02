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

Now the `epoch 120 model of Fully Convolutional Gray VGG-16` is available: [Link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_ad_cityu_edu_hk/Eewuv4ALcYlMrSGKET_92I8BeScScW_NTFWWgPNj3bSBzQ?e=hy8dki)

### 2.2  CIE Lab VGG-16

Now the `epoch 120 model of CIE Lab VGG-16` is available: [Link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_ad_cityu_edu_hk/EXlHehh_tj1Jsg4RIMMXlw0BE4BG5kjR9hx4-uiCj6tAVg?e=7UveYh)

### 2.3  Fully Convolutional CIE Lab VGG-16

Now the `epoch 120 model of Fully Convolutional CIE Lab VGG-16` is available: [Link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_ad_cityu_edu_hk/EfzwFPcpxJJLupdH6lesDowBxkPEWyyw1PEsLI6DEDbJew?e=8ITWBT)

Other models pending...

## 3 Convert

Normally, we save the whole model as a `.pth` file. If you want the weights only, please run `convert.py`

## 4 Acknowledgement

If you think this work is helpful for your research, please consider cite:

