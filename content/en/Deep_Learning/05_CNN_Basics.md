# 05. CNN Basics (Convolutional Neural Networks)

## Learning Objectives

- Understand the principles of convolution operations
- Learn pooling, padding, and stride concepts
- Implement CNNs with PyTorch
- Classify MNIST/CIFAR-10 datasets

---

## 1. Convolution Operation

### Concept

Detects local patterns in images (edges, textures).

```
Input Image     Filter(Kernel)     Output
[1 2 3 4]       [1 0]              [?]
[5 6 7 8]  *    [0 1]   =
[9 0 1 2]
```

### Formula

```
Output[i,j] = Σ Σ Input[i+m, j+n] × Filter[m, n]
```

### Dimension Calculation

```
Output size = (Input - Kernel + 2×Padding) / Stride + 1

Example: Input 32×32, Kernel 3×3, Padding 1, Stride 1
         = (32 - 3 + 2) / 1 + 1 = 32
```

---

## 2. Key Concepts

### Padding

```
Add zeros to input borders to maintain output size

padding='same': Output = Input size
padding='valid': No padding (Output < Input)
```

### Stride

```
Filter movement interval

stride=1: Move one pixel at a time (default)
stride=2: Move two pixels at a time → Output size halved
```

### Pooling

```
Reduce spatial size, increase invariance

Max Pooling: Maximum value in region
Avg Pooling: Average value in region
```

---

## 3. CNN Architecture

### Basic Structure

```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

### LeNet-5 (1998)

```
Input (32×32×1)
  ↓
Conv1 (5×5, 6 channels) → 28×28×6
  ↓
MaxPool (2×2) → 14×14×6
  ↓
Conv2 (5×5, 16 channels) → 10×10×16
  ↓
MaxPool (2×2) → 5×5×16
  ↓
Flatten → 400
  ↓
FC → 120 → 84 → 10
```

---

## 4. PyTorch Conv2d

### Basic Usage

```python
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv = nn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=64,    # 64 filters
    kernel_size=3,      # 3×3 kernel
    stride=1,
    padding=1           # same padding
)

# Input: (batch, channels, height, width)
x = torch.randn(1, 3, 32, 32)
out = conv(x)  # (1, 64, 32, 32)
```

### MaxPool2d

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 32×32 → 16×16
```

---

## 5. MNIST CNN Implementation

### Model Definition

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # FC block
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))  # (batch, 32, 28, 28)
        x = self.pool1(x)          # (batch, 32, 14, 14)

        x = F.relu(self.conv2(x))  # (batch, 64, 14, 14)
        x = self.pool2(x)          # (batch, 64, 7, 7)

        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Training Code

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Model, loss, optimizer
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 6. Feature Map Visualization

```python
def visualize_feature_maps(model, image):
    """Visualize feature maps from the first Conv layer"""
    model.eval()
    with torch.no_grad():
        # First Conv output
        x = model.conv1(image)
        x = F.relu(x)

    # Display in grid
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < x.shape[1]:
            ax.imshow(x[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('feature_maps.png')
```

---

## 7. Understanding Convolution with NumPy (Reference)

```python
def conv2d_numpy(image, kernel):
    """2D convolution implementation with NumPy (educational)"""
    h, w = image.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # Extract region
            region = image[i:i+kh, j:j+kw]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)

    return output

# Sobel edge detection example
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

edges = conv2d_numpy(image, sobel_x)
```

> **Note**: In actual CNNs, use PyTorch's optimized implementation.

---

## 8. Batch Normalization and Dropout

### Usage in CNNs

```python
class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BN for Conv
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)  # 2D Dropout

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.bn_fc = nn.BatchNorm1d(128)  # BN for FC
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

---

## 9. CIFAR-10 Classification

### Data

- 32×32 RGB images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Model

```python
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32→16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16→8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x
```

---

## 10. Summary

### Core Concepts

1. **Convolution**: Local pattern extraction, parameter sharing
2. **Pooling**: Spatial reduction, increased invariance
3. **Channels**: Learn diverse features
4. **Hierarchical Learning**: Low-level → High-level features

### CNN vs MLP

| Item | MLP | CNN |
|------|-----|-----|
| Connectivity | Fully connected | Local connections |
| Parameters | Many | Few (shared) |
| Spatial Information | Ignored | Preserved |
| Images | Inefficient | Efficient |

### Next Steps

In [06_CNN_Advanced.md](./06_CNN_Advanced.md), we'll learn famous architectures like ResNet and VGG.
