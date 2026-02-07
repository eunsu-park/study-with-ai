# 07. Transfer Learning

## Learning Objectives

- Understand the concept and benefits of transfer learning
- Utilize pretrained models
- Learn fine-tuning strategies
- Practical image classification project

---

## 1. What is Transfer Learning?

### Concept

```
Model trained on ImageNet
        ↓
    Low-level features (edges, textures) → Reuse
        ↓
    High-level features → Adapt to new data
        ↓
    New classification task
```

### Benefits

- High performance with limited data
- Faster training
- Better generalization

---

## 2. Transfer Learning Strategies

### Strategy 1: Feature Extraction

```python
# Freeze pretrained model weights
for param in model.parameters():
    param.requires_grad = False

# Replace only the last layer
model.fc = nn.Linear(2048, num_classes)
```

- Use pretrained features as-is
- Train only the final classification layer
- Suitable when data is limited

### Strategy 2: Fine-tuning

```python
# Train all or some layers
for param in model.parameters():
    param.requires_grad = True

# Use low learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

- Start from pretrained weights
- Fine-tune the entire network
- Suitable when sufficient data available

### Strategy 3: Gradual Unfreezing

```python
# Step 1: Last layer only
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
train_for_epochs(5)

# Step 2: Last block too
model.layer4.requires_grad_(True)
train_for_epochs(5)

# Step 3: Entire network
model.requires_grad_(True)
train_for_epochs(10)
```

---

## 3. PyTorch Implementation

### Basic Transfer Learning

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets

# 1. Load pretrained model
model = models.resnet50(weights='IMAGENET1K_V2')

# 2. Use as feature extractor (freeze weights)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace last layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
```

### Data Preprocessing

```python
# Use ImageNet normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

---

## 4. Training Strategies

### Discriminative Learning Rates

```python
# Different learning rates for each layer
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 5e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 5e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

### Learning Rate Scheduling

```python
# Warmup + Cosine Decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1  # 10% warmup
)
```

---

## 5. Various Pretrained Models

### torchvision Models

```python
# Classification
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vit = models.vit_b_16(weights='IMAGENET1K_V1')

# Object detection
fasterrcnn = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Segmentation
deeplabv3 = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
```

### timm Library

```python
import timm

# Check available models
print(timm.list_models('*efficientnet*'))

# Load model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
```

---

## 6. Practical Project: Flower Classification

### Data Preparation

```python
# Flowers102 dataset
from torchvision.datasets import Flowers102

train_data = Flowers102(
    root='data',
    split='train',
    transform=train_transform,
    download=True
)

test_data = Flowers102(
    root='data',
    split='test',
    transform=val_transform
)
```

### Model and Training

```python
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Replace last layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Training
model = FlowerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

---

## 7. Considerations

### Strategy by Data Size

| Data Size | Strategy | Description |
|-----------|---------|-------------|
| Very Small (<1000) | Feature Extraction | Train only last layer |
| Small (1000-10000) | Gradual Unfreezing | Unfreeze from later layers |
| Medium (10000+) | Full Fine-tuning | Train all with low LR |

### Domain Similarity

```
Similar to ImageNet (animals, objects):
    → Can use shallow layers as-is

Different from ImageNet (medical, satellite):
    → Need to fine-tune deeper layers
```

### Common Mistakes

1. Missing ImageNet normalization
2. Learning rate too high
3. Forgetting to switch train/eval mode
4. Including frozen weights in optimizer

---

## 8. Performance Improvement Tips

### Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    normalize
])
```

### Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Mixup / CutMix

```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam
```

---

## Summary

### Core Concepts

1. **Feature Extraction**: Reuse pretrained features
2. **Fine-tuning**: Adjust entire network with low LR
3. **Gradual Unfreezing**: Sequential training from later layers

### Checklist

- [ ] Use ImageNet normalization
- [ ] Choose appropriate learning rate (1e-4 ~ 1e-5)
- [ ] Switch model.train() / model.eval()
- [ ] Apply data augmentation
- [ ] Set up early stopping

---

## Next Steps

In [08_RNN_Basics.md](./08_RNN_Basics.md), we'll learn recurrent neural networks.
