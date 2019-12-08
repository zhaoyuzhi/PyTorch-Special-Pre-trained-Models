# Pre-trained VGG-16 Network on Grayscale Images

## 1 Network Structure

### 1.1 Original VGG-16 Structure without BN

### 1.2 Original VGG-16 Structure with BN

### 1.3 Revised VGG-16 Structure with stride = 2, BN, and SpectralNorm

## 2 Training Method

- Epochs: 120

- Batch size: 32

- Learning rate: 0.01

- Momentum: 0.9

- Wight decay (L2 Regularization): 5e-4

- Learning rate decrease: Multiply by 0.1 every 30 epochs

- GPU: single GPU (NVIDIA Quadro P4000 / GeForce GTX 1080 Ti)

- Num of workers: 8

- Input range: [-1, 1]

- Output: 1000 dimension one-hot vector
