# DINOv2 & Self-Supervised Vision

## Learning Objectives
- Understand the Self-distillation mechanism of DINO/DINOv2
- Grasp the Teacher-Student learning paradigm
- Learn how to utilize Dense Visual Features
- Apply DINOv2 as a Vision Foundation Model

---

## 1. Review: Self-Supervised Learning in Vision

### 1.1 Why Self-Supervised?

```
┌─────────────────────────────────────────────────────────────────┐
│              Self-Supervised Learning in Vision                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Limitations of Supervised Learning:                            │
│  • ImageNet: 1.4M images, 1000 classes                          │
│  • High labeling cost                                           │
│  • Class labels = limited information                           │
│                                                                 │
│  Self-Supervised Learning:                                      │
│  • Learn without labels (using pretext tasks)                   │
│  • Can utilize billions of images                               │
│  • Learn richer representations                                 │
│                                                                 │
│  Main Methods:                                                  │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Contrastive   │ SimCLR, MoCo  │ Learn similar/     │         │
│  │               │               │ dissimilar pairs   │         │
│  │ Distillation  │ DINO, BYOL    │ Teacher-Student    │         │
│  │ Masked        │ MAE, BEiT     │ Mask and restore   │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Prerequisite Review

> **Prerequisites**: [Deep_Learning/21_Self_Supervised_Learning.md](../Deep_Learning/21_Self_Supervised_Learning.md)
> - SimCLR: Contrastive Learning basics
> - MoCo: Momentum Contrast
> - BYOL: Bootstrap Your Own Latent
> - MAE: Masked Autoencoders

---

## 2. DINO (2021)

### 2.1 Core Idea

**DINO** (Self-**Di**stillation with **No** labels) applies Knowledge Distillation in a Self-supervised manner.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DINO Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                        Input Image                              │
│                            │                                    │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│     ┌─────────────────┐         ┌─────────────────┐            │
│     │  Global Crops   │         │  Local Crops    │            │
│     │   (224×224)     │         │   (96×96)       │            │
│     │    × 2          │         │    × 6+         │            │
│     └────────┬────────┘         └────────┬────────┘            │
│              │                           │                      │
│              ▼                           ▼                      │
│     ┌─────────────────┐         ┌─────────────────┐            │
│     │ Teacher Network │         │ Student Network │            │
│     │   (EMA update)  │         │   (Gradient)    │            │
│     │   [stop-grad]   │         │                 │            │
│     └────────┬────────┘         └────────┬────────┘            │
│              │                           │                      │
│              ▼                           ▼                      │
│     ┌─────────────────┐         ┌─────────────────┐            │
│     │  Teacher Head   │         │  Student Head   │            │
│     │  (Projection)   │         │  (Projection)   │            │
│     └────────┬────────┘         └────────┬────────┘            │
│              │                           │                      │
│              ▼                           ▼                      │
│          P_teacher                   P_student                  │
│              │                           │                      │
│              └───────────┬───────────────┘                      │
│                          ▼                                      │
│                  Cross-Entropy Loss                             │
│                  H(P_t, P_s) = -Σ P_t log(P_s)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):
    """
    DINO Projection Head

    Structure: Linear → GELU → Linear → L2 Norm
    Output: K dimensions (e.g., 65536)
    """
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # L2 normalization
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    """
    DINO Loss: Cross-entropy between teacher and student

    Features:
    - Teacher: Centering + Sharpening (temperature τ_t < τ_s)
    - Student: regular softmax
    - Center: moving average of all teacher outputs (prevents collapse)
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: (batch, n_crops, out_dim)
            teacher_output: (batch, n_global_crops, out_dim)
        """
        # Teacher: centering + sharpening
        teacher_out = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach()  # stop gradient

        # Student: softmax with higher temperature
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)

        # Cross-entropy loss
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()

        # Update center (EMA)
        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

### 2.3 Multi-crop Strategy

```python
"""
Multi-crop Strategy:

Global crops (2):
- Size: 224×224 (50-100% of original)
- Input to both Teacher and Student
- Learn full image context

Local crops (multiple, usually 6-8):
- Size: 96×96 (5-50% of original)
- Input to Student only
- Learn local patterns

Purpose:
- Learn "Local-to-Global" correspondence
- Learn what part of the whole image a small region represents
- Naturally acquire semantic segmentation capabilities
"""

from torchvision import transforms

class DINODataAugmentation:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                 n_local_crops=8):
        # Global crops (224×224)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Local crops (96×96)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.n_local_crops = n_local_crops

    def __call__(self, image):
        crops = []
        # 2 global crops
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        # n local crops
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(image))
        return crops
```

### 2.4 Teacher-Student Update

```python
class DINOTrainer:
    """
    DINO Training Loop

    Key:
    - Student: updated via gradient
    - Teacher: EMA (Exponential Moving Average) of Student
    """
    def __init__(self, student, teacher, optimizer, loss_fn, momentum=0.996):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.momentum = momentum

        # Teacher initialized from Student
        self.teacher.load_state_dict(self.student.state_dict())
        # Teacher doesn't compute gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """EMA update: θ_t = m * θ_t + (1-m) * θ_s"""
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(self.momentum).add_((1 - self.momentum) * param_s.data)

    def train_step(self, images):
        """
        images: list of crops [global1, global2, local1, ..., localN]
        """
        # Only global crops to Teacher
        teacher_output = self.teacher(torch.cat(images[:2]))

        # All crops to Student
        student_output = self.student(torch.cat(images))

        # Compute loss (each student crop vs each teacher crop)
        loss = self.loss_fn(student_output, teacher_output)

        # Student update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Teacher EMA update
        self.update_teacher()

        return loss.item()
```

---

## 3. DINOv2 (2023)

### 3.1 DINOv2 Improvements

```
┌─────────────────────────────────────────────────────────────────┐
│                 DINO vs DINOv2 Comparison                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Aspect             │ DINO (2021)       │ DINOv2 (2023)         │
│  ───────────────────│───────────────────│─────────────────────  │
│  Data               │ ImageNet (1.3M)   │ LVD-142M (142M)       │
│  Data Curation      │ None              │ Auto curation pipeline│
│  Model Sizes        │ ViT-S/B           │ ViT-S/B/L/g           │
│  Training Objective │ DINO only         │ DINO + iBOT (masked)  │
│  Regularization     │ Basic             │ KoLeo + enhanced reg  │
│  Resolution         │ 224               │ 518 (high resolution) │
│  Performance (k-NN) │ ~74% (IN-1K)      │ ~86% (IN-1K)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 LVD-142M Dataset

```python
"""
LVD-142M (Learning with large Visual Datasets)

Auto curation pipeline:
1. Collect images from web (billions)
2. Duplicate removal (copy detection)
3. Quality filtering
4. ImageNet similarity-based sampling
5. Final 142M images

Key Technologies:
- Self-supervised copy detection
- Embedding-based clustering
- Retrieval-based data selection

Why Important:
- Data quality is key to model performance
- Scaling requires data curation
- Automated pipeline enables scale
"""
```

### 3.3 iBOT Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                 DINOv2 = DINO + iBOT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DINO Loss (image level):                                       │
│  • Global/Local crop consistency                                │
│  • CLS token based                                              │
│                                                                 │
│  iBOT Loss (patch level):                                       │
│  • Masked patches prediction                                    │
│  • Similar to MAE but uses Teacher                              │
│                                                                 │
│                    Input Image                                  │
│                         │                                       │
│          ┌─────────────┴─────────────┐                          │
│          ▼                           ▼                          │
│     ┌─────────┐                ┌─────────┐                      │
│     │ Teacher │                │ Student │                      │
│     │ (full)  │                │ (masked)│ ← Some patches masked│
│     └────┬────┘                └────┬────┘                      │
│          │                          │                           │
│     ┌────┴────┐                ┌────┴────┐                      │
│     │CLS│Patch│                │CLS│Patch│                      │
│     └─┬───┬───┘                └─┬───┬───┘                      │
│       │   │                      │   │                          │
│       │   └──────────────────────│───┤                          │
│       │          iBOT Loss       │   │                          │
│       │     (masked patches)     │   │                          │
│       │                          │   │                          │
│       └──────────────────────────┘   │                          │
│              DINO Loss               │                          │
│           (CLS tokens)               │                          │
│                                                                 │
│  Total Loss = L_DINO + λ × L_iBOT                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Model Architecture

```python
"""
DINOv2 Model Specifications

Model      │ Layers │ Hidden │ Heads │ Params │ Patch
──────────│────────│────────│───────│────────│───────
ViT-S/14  │ 12     │ 384    │ 6     │ 21M    │ 14×14
ViT-B/14  │ 12     │ 768    │ 12    │ 86M    │ 14×14
ViT-L/14  │ 24     │ 1024   │ 16    │ 300M   │ 14×14
ViT-g/14  │ 40     │ 1536   │ 24    │ 1.1B   │ 14×14

Features:
- Patch size 14 (original ViT uses 16)
- Higher resolution support
- Register tokens (solves attention artifacts)
"""
```

---

## 4. Using DINOv2

### 4.1 Loading with HuggingFace

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

# Load model
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess and inference
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Output structure
print(f"Last hidden state: {outputs.last_hidden_state.shape}")
# (1, 257, 768) = (batch, 1 CLS + 256 patches, hidden_dim)

# CLS token (full image representation)
cls_token = outputs.last_hidden_state[:, 0]
print(f"CLS token: {cls_token.shape}")  # (1, 768)

# Patch tokens (local representations)
patch_tokens = outputs.last_hidden_state[:, 1:]
print(f"Patch tokens: {patch_tokens.shape}")  # (1, 256, 768)
```

### 4.2 Feature Extraction and Usage

```python
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from sklearn.neighbors import NearestNeighbors

class DINOv2FeatureExtractor:
    """Image feature extractor using DINOv2"""

    def __init__(self, model_name="facebook/dinov2-base"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, images, return_patches=False):
        """
        Extract features from images

        Args:
            images: PIL Image or list
            return_patches: also return patch-level features

        Returns:
            cls_features: (n_images, hidden_dim)
            patch_features: (n_images, n_patches, hidden_dim) - optional
        """
        if not isinstance(images, list):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)

        cls_features = outputs.last_hidden_state[:, 0]

        if return_patches:
            patch_features = outputs.last_hidden_state[:, 1:]
            return cls_features, patch_features

        return cls_features

    def compute_similarity(self, image1, image2):
        """Similarity between two images (cosine)"""
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)
        similarity = F.cosine_similarity(feat1, feat2)
        return similarity.item()

# Usage example
extractor = DINOv2FeatureExtractor()

# Image search
def build_image_index(images):
    """Build image index"""
    features = []
    for img in images:
        feat = extractor.extract_features(img)
        features.append(feat.numpy())
    features = np.vstack(features)

    # k-NN index
    index = NearestNeighbors(n_neighbors=5, metric='cosine')
    index.fit(features)
    return index, features

def search_similar(query_image, index, features, k=5):
    """Search for similar images"""
    query_feat = extractor.extract_features(query_image).numpy()
    distances, indices = index.kneighbors(query_feat, n_neighbors=k)
    return indices[0], distances[0]
```

### 4.3 Dense Prediction (Semantic Segmentation)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_attention_maps(model, processor, image):
    """Visualize DINOv2 attention maps"""

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Last layer attention
    attentions = outputs.attentions[-1]  # (1, n_heads, n_tokens, n_tokens)

    # Attention from CLS token to each patch
    cls_attn = attentions[0, :, 0, 1:]  # (n_heads, n_patches)

    # Average
    cls_attn_mean = cls_attn.mean(dim=0)  # (n_patches,)

    # Reshape to 2D
    n_patches = int(np.sqrt(cls_attn_mean.shape[0]))
    attn_map = cls_attn_mean.reshape(n_patches, n_patches)

    return attn_map.numpy()

def visualize_patch_pca(model, processor, image, n_components=3):
    """PCA visualization of patch features (check semantic regions)"""

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Patch tokens
    patch_tokens = outputs.last_hidden_state[0, 1:].numpy()  # (n_patches, hidden)

    # PCA
    pca = PCA(n_components=n_components)
    patch_pca = pca.fit_transform(patch_tokens)

    # Normalize to [0, 1] for visualization
    patch_pca = (patch_pca - patch_pca.min()) / (patch_pca.max() - patch_pca.min())

    # Reshape
    n_patches = int(np.sqrt(patch_tokens.shape[0]))
    pca_image = patch_pca.reshape(n_patches, n_patches, n_components)

    return pca_image

# Visualization
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(image)
# axes[0].set_title('Original')
# axes[1].imshow(visualize_attention_maps(model, processor, image), cmap='hot')
# axes[1].set_title('Attention Map')
# axes[2].imshow(visualize_patch_pca(model, processor, image))
# axes[2].set_title('PCA of Patches')
```

---

## 5. DINOv2 Applications

### 5.1 Zero-shot Semantic Segmentation

```python
"""
Segmentation using DINOv2 patch features

Method:
1. Extract DINOv2 patch features from image
2. Extract features from region of interest in reference image
3. Find matching regions using cosine similarity

Advantages:
- Segmentation without training
- Can handle new object classes
"""

def segment_with_reference(model, processor, target_image, reference_image, reference_mask):
    """
    Segment target image using mask from reference image

    Args:
        target_image: image to segment
        reference_image: reference image
        reference_mask: binary mask of region of interest
    """
    # Extract features
    with torch.no_grad():
        target_inputs = processor(images=target_image, return_tensors="pt")
        target_outputs = model(**target_inputs)
        target_patches = target_outputs.last_hidden_state[0, 1:]  # (n_patches, hidden)

        ref_inputs = processor(images=reference_image, return_tensors="pt")
        ref_outputs = model(**ref_inputs)
        ref_patches = ref_outputs.last_hidden_state[0, 1:]  # (n_patches, hidden)

    # Average features from region of interest in reference mask
    n_patches = int(np.sqrt(ref_patches.shape[0]))
    mask_resized = F.interpolate(
        reference_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(n_patches, n_patches),
        mode='nearest'
    ).squeeze().bool()

    foreground_features = ref_patches[mask_resized.flatten()].mean(dim=0)

    # Compute similarity between each patch in target and foreground
    similarities = F.cosine_similarity(
        target_patches,
        foreground_features.unsqueeze(0),
        dim=1
    )

    # Reshape to 2D
    similarity_map = similarities.reshape(n_patches, n_patches)

    return similarity_map.numpy()
```

### 5.2 Depth Estimation

```python
"""
Depth Estimation with DINOv2 + Linear Probe

Method:
1. Extract patch features with DINOv2
2. Predict depth with simple Linear layer
3. Good performance even with small data

Reason:
- DINOv2 already learns 3D structure information
- Depth cues are encoded in patch features
"""

class DepthEstimator(nn.Module):
    def __init__(self, dinov2_model, hidden_dim=768):
        super().__init__()
        self.backbone = dinov2_model
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x).last_hidden_state[:, 1:]  # patch tokens

        depth = self.head(features)  # (batch, n_patches, 1)

        # Reshape to image
        batch, n_patches, _ = depth.shape
        h = w = int(np.sqrt(n_patches))
        depth = depth.reshape(batch, h, w)

        return depth
```

---

## Summary

### DINO/DINOv2 Key Concepts
| Concept | Description |
|------|------|
| **Self-distillation** | Teacher-Student structure, learn without labels |
| **Multi-crop** | Learn various scales with Global + Local crops |
| **Centering** | Center Teacher output to prevent collapse |
| **EMA Teacher** | Provide stable targets with momentum |
| **iBOT** | Added masked patch prediction (DINOv2) |

### Applications
- **Image Retrieval**: Search similar images with CLS token
- **Semantic Segmentation**: Zero-shot segmentation with patch features
- **Depth Estimation**: Predict depth with linear probe
- **Fine-tuning**: Train on downstream tasks

### Next Steps
- [13_Segment_Anything.md](13_Segment_Anything.md): SAM's promptable segmentation
- [14_Unified_Vision_Models.md](14_Unified_Vision_Models.md): Unified Vision Foundation Models

---

## References

### Papers
- Caron et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers" (DINO)
- Oquab et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision"
- Zhou et al. (2021). "iBOT: Image BERT Pre-Training with Online Tokenizer"

### Code
- [DINO GitHub](https://github.com/facebookresearch/dino)
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [HuggingFace DINOv2](https://huggingface.co/facebook/dinov2-base)
