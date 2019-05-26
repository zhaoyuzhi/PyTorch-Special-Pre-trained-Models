# PyTorch-Special-Pre-trained-Models

Special pre-trained VGG-16 network on CIE Lab and Grayscale images

## 1 Model Validation Accuracy (on ImageNet Validation 50k)

Compared to the official model provided by PyTorch, the classification ability of our model is only slightly weaker. Basically, these models are targeted for regression task, so we think the small improvement is unnecessary.

The **Fully Convolutional** models do not include MaxPooling layer & AdaptiveAvgPooling layer, which is replaced by a convolutional layer with stride = 2. Note that, the total amount of convolutional layer is unchanged. Each convolutional layer is spectral normalized (you may find the source code in this [project](https://github.com/zhaoyuzhi/PyTorch-Useful-Codes)), which is very useful for the training of WGAN.

### 1.1  Fully Convolutional Gray VGG-16

- epoch 30 | top 1:  | top 5: 

- epoch 60 | top 1:  | top 5: 

- epoch 90 | top 1:  | top 5: 

- epoch 120 | top 1:  | top 5: 

### 1.1  CIE Lab VGG-16

- epoch 30 | top 1:  | top 5: 

- epoch 60 | top 1:  | top 5: 

- epoch 90 | top 1:  | top 5: 

- epoch 120 | top 1:  | top 5: 

### 1.3  Fully Convolutional CIE Lab VGG-16

- epoch 30 | top 1: 35.70% | top 5: 63.22%

- epoch 60 | top 1: 59.21% | top 5: 83.16%

- epoch 90 | top 1: 65.15% | top 5: 86.72%

- epoch 120 | top 1: 65.13% | top 5: 86.80%

- We further tested validation accuracy when the learning was further decreasing to 1e-6; however, there is no explicit improvement. For example, when epoch = 125, the top 1 accuracy = 65.16% and top 5 accuracy = 86.81%.

## 2 Download Link

### 2.1  Fully Convolutional Gray VGG-16

### 2.1  CIE Lab VGG-16

### 2.3  Fully Convolutional CIE Lab VGG-16

Now the `epoch 120 model` is available: [Link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_ad_cityu_edu_hk/EfzwFPcpxJJLupdH6lesDowBxkPEWyyw1PEsLI6DEDbJew?e=8ITWBT)

Other models pending...

## 3 Acknowledgement

If you think this work is helpful for your research, please consider cite:
