# Pre-trained ResNet-50 Network on Grayscale Images

## 1 Network Structure

### 1.1 Original ResNet-50 Structure without IN

### 1.2 Original ResNet-50 Structure with IN

### 1.3 Revised ResNet-50 Structure with stride = 2, IN, and SpectralNorm

## 2 Training Method

- Epochs: 120

- Batch size: 64

- Learning rate: 0.01

- Momentum: 0.9

- Wight decay (L2 Regularization): 0

- Learning rate decrease: Multiply by 0.1 every 30 epochs

- GPU: single GPU (NVIDIA Quadro P4000 / GeForce GTX 1080 Ti)

- Num of workers: 8

- Input range: [-1, 1]

- Output: 1000 dimension one-hot vector
