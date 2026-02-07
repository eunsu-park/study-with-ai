# Model Implementations Learning Guide

## Overview

This folder contains materials for learning deep learning models through **from-scratch implementation**. The goal is to deeply understand the internal workings of models, beyond simply using libraries.

### Learning Philosophy
> "What I cannot create, I do not understand." - Richard Feynman

### 4-Level Implementation Approach
| Level | Name | Description | Purpose |
|-------|------|-------------|---------|
| L1 | **NumPy Scratch** | Direct implementation with matrix operations | Understanding mathematical principles |
| L2 | **PyTorch Low-level** | Using only basic ops | Understanding frameworks |
| L3 | **Paper Implementation** | Reproducing papers | Research capabilities |
| L4 | **Code Analysis** | Analyzing production code | Practical application |

---

## Learning Sequence (12 Models)

### Tier 1: Fundamentals (Week 1-2)
| # | Model | Key Concepts | L1 | L2 | L3 | L4 |
|---|------|--------------|----|----|----|----|
| 01 | [Linear/Logistic](01_Linear_Logistic/) | Gradient Descent, Loss | ✅ | ✅ | ✅ | - |
| 02 | [MLP](02_MLP/) | Backpropagation, Activations | ✅ | ✅ | ✅ | - |
| 03 | [CNN (LeNet)](03_CNN_LeNet/) | Convolution, Pooling | ✅ | ✅ | ✅ | - |

### Tier 2: Classic Deep Learning (Week 3-5)
| # | Model | Key Concepts | L1 | L2 | L3 | L4 |
|---|------|--------------|----|----|----|----|
| 04 | [VGG](04_VGG/) | Deep Stacking, Features | - | ✅ | ✅ | - |
| 05 | [ResNet](05_ResNet/) | Skip Connections, Residual | - | ✅ | ✅ | ✅ |
| 06 | [LSTM/GRU](06_LSTM_GRU/) | Gating, BPTT | ✅ | ✅ | ✅ | - |

### Tier 3: Transformer-Based (Week 6-8)
| # | Model | Key Concepts | L1 | L2 | L3 | L4 |
|---|------|--------------|----|----|----|----|
| 07 | [Transformer](07_Transformer/) | Self-Attention, PE | - | ✅ | ✅ | - |
| 08 | [BERT](08_BERT/) | Masked LM, Bidirectional | - | ✅ | ✅ | ✅ |
| 09 | [GPT](09_GPT/) | Autoregressive, Causal | - | ✅ | ✅ | ✅ |
| 10 | [ViT](10_ViT/) | Patch Embedding, CLS | - | ✅ | ✅ | ✅ |

### Tier 4: Generative/Multimodal (Week 9-10)
| # | Model | Key Concepts | L1 | L2 | L3 | L4 |
|---|------|--------------|----|----|----|----|
| 11 | [VAE](11_VAE/) | ELBO, Reparameterization | ✅ | ✅ | ✅ | - |
| 12 | [CLIP](12_CLIP/) | Contrastive, Zero-shot | - | ✅ | ✅ | ✅ |

---

## Learning Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                Model Dependencies and Learning Order             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  01_Linear ─┬─► 02_MLP ─┬─► 03_CNN ──► 04_VGG ──► 05_ResNet    │
│             │           │                                       │
│             │           └─► 06_LSTM                             │
│             │                                                   │
│             └─► 07_Transformer ─┬─► 08_BERT                     │
│                                 │                               │
│                                 ├─► 09_GPT                      │
│                                 │                               │
│                                 └─► 10_ViT ──► 12_CLIP          │
│                                                                 │
│  02_MLP ──────────────────────────► 11_VAE                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Arrow: prerequisite knowledge relationship
```

---

## Folder Structure

```
Model_Implementations/
├── 00_Overview.md              # This file
├── docs/
│   ├── 01_Implementation_Philosophy.md    # Value of implementation learning
│   ├── 02_NumPy_vs_PyTorch_Comparison.md  # Framework comparison
│   ├── 03_Reading_HuggingFace_Code.md     # HF code reading guide
│   └── 04_Reading_timm_Code.md            # timm code reading guide
│
├── 01_Linear_Logistic/
│   ├── README.md               # Model overview and mathematics
│   ├── theory.md               # Detailed theory
│   ├── numpy/
│   │   ├── linear_numpy.py     # NumPy implementation
│   │   ├── logistic_numpy.py
│   │   └── test_numpy.py       # Unit tests
│   ├── pytorch_lowlevel/
│   │   └── linear_lowlevel.py  # PyTorch basic ops
│   ├── paper/
│   │   └── linear_paper.py     # Clean implementation
│   └── exercises/
│       └── 01_regularization.md
│
├── 02_MLP/
│   ├── ... (same structure)
│
└── utils/
    ├── data_loaders.py         # Common data loading
    ├── visualization.py        # Visualization utilities
    └── training_utils.py       # Training helpers
```

---

## Level Descriptions

### Level 1: NumPy From-Scratch

**Goal**: Implement with pure matrix operations without frameworks

```python
# Example: Linear Regression
import numpy as np

class LinearRegression:
    def __init__(self, input_dim):
        # Xavier initialization
        self.W = np.random.randn(input_dim, 1) * np.sqrt(2/input_dim)
        self.b = np.zeros(1)

    def forward(self, X):
        """y = XW + b"""
        self.X = X  # Cache for backward
        return X @ self.W + self.b

    def backward(self, y, y_pred, lr=0.01):
        """Manual gradient computation"""
        m = y.shape[0]
        error = y_pred - y

        # dL/dW = X^T @ error / m
        dW = self.X.T @ error / m
        db = np.mean(error)

        # Weight update
        self.W -= lr * dW
        self.b -= lr * db
```

**Applied to models**: 01-03, 06, 11

### Level 2: PyTorch Low-Level

**Goal**: Use only PyTorch basic operations, minimize nn.Module

```python
# Example: MLP (without nn.Linear)
import torch
import torch.nn.functional as F

class MLPLowLevel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Manual parameter management
        self.W1 = torch.randn(input_dim, hidden_dim, requires_grad=True) * 0.01
        self.b1 = torch.zeros(hidden_dim, requires_grad=True)
        self.W2 = torch.randn(hidden_dim, output_dim, requires_grad=True) * 0.01
        self.b2 = torch.zeros(output_dim, requires_grad=True)

    def forward(self, x):
        # Use F.relu, not nn.ReLU
        h = F.relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

# Manual SGD
def sgd_step(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()
```

**Applied to models**: 01-12 (all)

### Level 3: Paper Implementation

**Goal**: Accurately reproduce paper architectures

```python
# Example: ResNet BasicBlock
"""
Paper: "Deep Residual Learning for Image Recognition"
       He et al., 2015
"""
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic residual block from paper Section 3.1

    Structure: conv-bn-relu-conv-bn + shortcut
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # First 3x3 conv (downsampling with stride)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second 3x3 conv (always stride=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut: adjust dimensions if needed with 1x1 conv
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # F(x) = conv-bn-relu-conv-bn
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # H(x) = F(x) + x
        out += self.shortcut(x)
        return F.relu(out)
```

**Applied to models**: 01-12 (all)

### Level 4: Code Analysis

**Goal**: Read, understand, and modify production code

```markdown
# Example: HuggingFace BERT Analysis

## File Structure
transformers/models/bert/
├── configuration_bert.py    # BertConfig
├── modeling_bert.py         # Main model
└── tokenization_bert.py     # Tokenizer

## Core Class Analysis

### BertModel (modeling_bert.py:800)
```python
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
```

### Modification Tasks
1. Add custom pooling
2. Visualize attention patterns
3. Implement new task heads
```

**Applied to models**: 05, 08-10, 12

---

## Practice Environment

### Required Packages
```bash
pip install numpy torch torchvision matplotlib
pip install transformers timm  # For L4
```

### Recommended Hardware
| Level | Minimum | Recommended |
|-------|---------|-------------|
| L1 NumPy | CPU only | CPU |
| L2-L3 (small models) | CPU/GPU 4GB | GPU 8GB |
| L3 (large models) | GPU 8GB | GPU 16GB |
| L4 | GPU 8GB | GPU 16GB |

### Datasets
- **MNIST**: MLP, CNN testing
- **CIFAR-10**: VGG, ResNet
- **IMDB**: LSTM, BERT
- **ImageNet (subset)**: ViT, CLIP

---

## Learning Tips

### 1. Progressive Approach
```
NumPy (principles) → PyTorch Low (framework) → Paper (completeness) → Analysis (production)
```

### 2. Importance of Testing
- Unit test each layer/module
- Verify shapes with small inputs
- Compare outputs with existing implementations

### 3. Debugging Strategy
```python
# Print shapes
print(f"Input: {x.shape}")
print(f"After conv1: {self.conv1(x).shape}")

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}")
```

### 4. Use Visualization
- Loss curves
- Attention patterns
- Feature maps
- Gradient flow

---

## Prerequisites

### Required
- Python programming
- Linear algebra (matrix operations, eigenvalues)
- Calculus (partial derivatives, chain rule)
- Basic probability/statistics

### Recommended
- Complete Deep_Learning folder lessons 01-10
- Basic NumPy, PyTorch usage

---

## References

### Books
- "Deep Learning" (Goodfellow et al.) - Theory
- "Dive into Deep Learning" - Implementation-focused
- "Neural Networks from Scratch" - NumPy implementation

### Online Resources
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [minGPT](https://github.com/karpathy/minGPT)
- [labml.ai Annotated Papers](https://nn.labml.ai/)

### Papers (Must-Read)
| Model | Paper |
|-------|-------|
| ResNet | He et al., 2015 |
| Transformer | Vaswani et al., 2017 |
| BERT | Devlin et al., 2018 |
| GPT | Radford et al., 2018 |
| ViT | Dosovitskiy et al., 2020 |
| CLIP | Radford et al., 2021 |

---

## Next Steps

After completing this folder:
- **Foundation_Models**: Understanding latest large models
- **Deep_Learning examples**: Practical applications
- **MLOps**: Model deployment
