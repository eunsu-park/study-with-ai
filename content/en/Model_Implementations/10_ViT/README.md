# 10. Vision Transformer (ViT)

## Overview

Vision Transformer (ViT) applies the Transformer architecture to image classification. It divides images into patches and treats each patch like a token. "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

---

## Mathematical Background

### 1. Image Patchification

```
Input image: x ∈ R^(H × W × C)
Patch size: P × P

Patch sequence:
x_p ∈ R^(N × P² × C)  where N = (H × W) / P²

Example:
- Image: 224 × 224 × 3
- Patch: 16 × 16
- N = (224 × 224) / (16 × 16) = 196 patches
- Each patch: 16 × 16 × 3 = 768 dimensions
```

### 2. Patch Embedding

```
Linear Projection:
z_0 = [x_class; x_p¹E; x_p²E; ...; x_pⁿE] + E_pos

Where:
- x_class: learnable [CLS] token
- E ∈ R^(P²C × D): patch embedding matrix
- E_pos ∈ R^((N+1) × D): position embedding

z_0 ∈ R^((N+1) × D): initial embedding sequence
```

### 3. Transformer Encoder

```
Encoder block (L layers):

z'_l = MSA(LN(z_{l-1})) + z_{l-1}
z_l = MLP(LN(z'_l)) + z'_l

Final output:
y = LN(z_L⁰)  # use only [CLS] token

Where z_L⁰ is the [CLS] token at layer L
```

---

## ViT Architecture Variants

```
ViT-Base (B/16):
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- MLP size: 3072
- Patch size: 16
- Parameters: 86M

ViT-Large (L/16):
- Hidden size: 1024
- Layers: 24
- Attention heads: 16
- MLP size: 4096
- Patch size: 16
- Parameters: 307M

ViT-Huge (H/14):
- Hidden size: 1280
- Layers: 32
- Attention heads: 16
- MLP size: 5120
- Patch size: 14
- Parameters: 632M
```

---

## File Structure

```
10_ViT/
├── README.md
├── pytorch_lowlevel/
│   └── vit_lowlevel.py         # Direct ViT implementation
├── paper/
│   └── vit_paper.py            # Paper reproduction
└── exercises/
    ├── 01_patch_embedding.md   # Patch embedding visualization
    └── 02_attention_maps.md    # Attention visualization
```

---

## Core Concepts

### 1. CNN vs ViT

```
CNN:
- Local receptive field
- Inductive bias: locality, translation equivariance
- Favorable for small datasets

ViT:
- Global receptive field (global from start)
- Minimal inductive bias
- Favorable for large-scale datasets (JFT-300M)
- Small data: needs pre-training
```

### 2. Position Embedding

```
1D Learnable (ViT default):
- N+1 learnable vectors
- Learn order information

2D Positional (variant):
- Separate embedding for (row, col)
- Reflects image structure

Sinusoidal:
- Fixed trigonometric functions
- Extrapolation capability
```

### 3. [CLS] Token vs Global Average Pooling

```
[CLS] Token:
- Added at first position
- Aggregates entire image representation
- BERT style

Global Average Pooling:
- Average all patches
- CNN style
- Similar performance
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Use F.linear, F.layer_norm
- Don't use nn.TransformerEncoder
- Direct patchification implementation

### Level 3: Paper Implementation (paper/)
- Exact paper specifications
- JFT/ImageNet pre-training
- Fine-tuning code

### Level 4: Code Analysis (separate)
- Analyze timm library
- Analyze HuggingFace ViT

---

## Learning Checklist

- [ ] Understand patch embedding formula
- [ ] Role of position embedding
- [ ] Role of [CLS] token
- [ ] Pros/cons compared to CNN
- [ ] Visualize attention maps
- [ ] Fine-tuning strategy

---

## References

- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- [timm ViT](https://github.com/rwightman/pytorch-image-models)
- [../Deep_Learning/19_Vision_Transformer.md](../Deep_Learning/19_Vision_Transformer.md)
