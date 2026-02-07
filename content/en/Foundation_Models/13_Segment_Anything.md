# Segment Anything Model (SAM)

## Learning Objectives
- Understand SAM's "Promptable Segmentation" paradigm
- Grasp the Image Encoder, Prompt Encoder, Mask Decoder structure
- Understand SAM's training data and methodology
- Learn practical SAM usage

---

## 1. SAM Overview

### 1.1 Foundation Model for Segmentation

**SAM** (Segment Anything Model) is a Vision Foundation Model released by Meta AI in 2023 that can segment **any object in any image**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM's Innovation                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Traditional Segmentation:                                      │
│  • Only specific classes (people, cars, etc.)                   │
│  • Only objects in training data                                │
│  • Model per class or fixed number of classes                   │
│                                                                 │
│  SAM:                                                           │
│  • Can segment any object                                       │
│  • Specify desired object with prompts                          │
│  • Zero-shot: handles new objects immediately                   │
│                                                                 │
│  Prompt Types:                                                  │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Point   │ Click location (foreground/background)   │         │
│  │ Box     │ Bounding box                             │         │
│  │ Mask    │ Rough mask (for refinement)              │         │
│  │ Text    │ Text description (SAM 2, Grounding SAM)  │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 SA-1B Dataset

```
┌─────────────────────────────────────────────────────────────────┐
│                    SA-1B Dataset                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scale:                                                         │
│  • 11M images                                                   │
│  • 1.1B (1.1 billion) masks                                     │
│  • Average ~100 masks per image                                 │
│                                                                 │
│  Collection Method (Data Engine):                               │
│                                                                 │
│  Phase 1: Assisted Manual (4.3M masks)                          │
│  ───────────────────────────────────                            │
│  • Professional annotators label with SAM assistance            │
│  • SAM proposes → humans correct                                │
│                                                                 │
│  Phase 2: Semi-Automatic (5.9M masks)                           │
│  ───────────────────────────────────                            │
│  • SAM auto-generates confident masks                           │
│  • Humans only label the rest                                   │
│                                                                 │
│  Phase 3: Fully Automatic (1.1B masks)                          │
│  ───────────────────────────────────                            │
│  • Auto-generate with 32×32 grid points                         │
│  • Filter to select final masks                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. SAM Architecture

### 2.1 Overall Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM Architecture                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         Input Image                             │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Image Encoder                          │    │
│  │           (MAE pre-trained ViT-H/16)                    │    │
│  │                                                          │    │
│  │  • 1024×1024 input → 64×64 feature map                  │    │
│  │  • 632M parameters                                       │    │
│  │  • Run only once (expensive)                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│                     Image Embeddings                            │
│                       (64×64×256)                               │
│                              │                                  │
│              ┌───────────────┴───────────────┐                  │
│              │                               │                  │
│              ▼                               ▼                  │
│  ┌───────────────────┐           ┌───────────────────┐         │
│  │  Prompt Encoder   │           │  Prompt Encoder   │         │
│  │  (Points/Boxes)   │           │  (Dense: Mask)    │         │
│  │                   │           │                   │         │
│  │  Sparse Embed     │           │  Conv downscale   │         │
│  │  (N×256)          │           │  (256×64×64)      │         │
│  └─────────┬─────────┘           └─────────┬─────────┘         │
│            │                               │                    │
│            └───────────────┬───────────────┘                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Mask Decoder                           │    │
│  │           (Lightweight Transformer)                      │    │
│  │                                                          │    │
│  │  • 2-layer Transformer decoder                          │    │
│  │  • Cross-attention: prompt ↔ image                      │    │
│  │  • Self-attention: prompt tokens                        │    │
│  │  • 4M parameters (very lightweight)                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                    │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│         3 Mask Outputs             IoU Scores                   │
│     (256×256, upscaled)          (confidence)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Image Encoder

```python
"""
SAM Image Encoder: MAE pre-trained ViT-H

Features:
- ViT-H/16: 632M parameters
- Input: 1024×1024 (high resolution)
- Output: 64×64×256 feature map
- Positional Embedding: Windowed + Global attention

Why MAE pre-training?
- Masking-based learning suits dense prediction
- Self-supervised learning utilizes large-scale data
- Effective for patch-level representation learning
"""

import torch
import torch.nn as nn

class SAMImageEncoder(nn.Module):
    """
    SAM's Image Encoder (simplified version)

    Actually uses ViT-H, but this is
    simplified for understanding the structure
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 1280,  # ViT-H
        depth: int = 32,
        num_heads: int = 16,
        out_chans: int = 256,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // patch_size) ** 2, embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1),
            nn.LayerNorm(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.LayerNorm(out_chans),
        )

    def forward(self, x):
        # x: (B, 3, 1024, 1024)
        x = self.patch_embed(x)  # (B, embed_dim, 64, 64)
        x = x.flatten(2).transpose(1, 2)  # (B, 4096, embed_dim)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        # Reshape back to 2D
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.neck(x)  # (B, 256, 64, 64)
        return x
```

### 2.3 Prompt Encoder

```python
class SAMPromptEncoder(nn.Module):
    """
    SAM Prompt Encoder

    Prompt types:
    1. Points: (x, y) + label (foreground/background)
    2. Boxes: (x1, y1, x2, y2)
    3. Masks: previous mask (for refinement)
    """
    def __init__(self, embed_dim: int = 256, image_size: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size

        # Point embeddings
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim),  # foreground
            nn.Embedding(1, embed_dim),  # background
        ])

        # Positional encoding for points
        self.pe_layer = PositionalEncoding(embed_dim, image_size)

        # Box corner embeddings
        self.box_embeddings = nn.Embedding(2, embed_dim)  # top-left, bottom-right

        # Mask encoder (for dense prompts)
        self.mask_downscaler = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        # No-mask embedding
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(self, points=None, boxes=None, masks=None):
        """
        Args:
            points: (B, N, 2) coordinates + (B, N) labels
            boxes: (B, 4) bounding box
            masks: (B, 1, H, W) previous mask

        Returns:
            sparse_embeddings: (B, N_prompts, embed_dim)
            dense_embeddings: (B, embed_dim, H, W)
        """
        sparse_embeddings = []

        # Point prompts
        if points is not None:
            coords, labels = points
            point_embed = self.pe_layer(coords)  # positional encoding

            for i in range(coords.shape[1]):
                label = labels[:, i]
                type_embed = self.point_embeddings[label](label)
                sparse_embeddings.append(point_embed[:, i] + type_embed)

        # Box prompts
        if boxes is not None:
            # Box = 2 corner points
            corners = boxes.reshape(-1, 2, 2)  # (B, 2, 2)
            corner_embed = self.pe_layer(corners)
            corner_embed += self.box_embeddings.weight
            sparse_embeddings.extend([corner_embed[:, 0], corner_embed[:, 1]])

        sparse_embeddings = torch.stack(sparse_embeddings, dim=1) if sparse_embeddings else None

        # Dense prompt (mask)
        if masks is not None:
            dense_embeddings = self.mask_downscaler(masks)
        else:
            # No mask: learnable embedding
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(-1, -1, 64, 64)

        return sparse_embeddings, dense_embeddings
```

### 2.4 Mask Decoder

```python
class SAMMaskDecoder(nn.Module):
    """
    SAM Mask Decoder

    Structure:
    - 2-layer Transformer decoder
    - Cross-attention: tokens ↔ image
    - Self-attention: tokens
    - 3 mask outputs (multi-scale)
    - IoU prediction head
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_mask_tokens: int = 4,  # 3 masks + 1 IoU
    ):
        super().__init__()

        # Mask tokens (learnable)
        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)

        # Transformer layers
        self.transformer = TwoWayTransformer(
            depth=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        # Output heads
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_mask_tokens - 1),  # 3 IoU scores
        )

        self.mask_prediction_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, num_mask_tokens - 1, kernel_size=1),
        )

    def forward(self, image_embeddings, sparse_embeddings, dense_embeddings):
        """
        Args:
            image_embeddings: (B, 256, 64, 64)
            sparse_embeddings: (B, N_prompts, 256)
            dense_embeddings: (B, 256, 64, 64)

        Returns:
            masks: (B, 3, 256, 256)
            iou_predictions: (B, 3)
        """
        # Combine sparse and mask tokens
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(
            sparse_embeddings.shape[0], -1, -1
        )
        tokens = torch.cat([mask_tokens, sparse_embeddings], dim=1)

        # Add dense embeddings to image
        image_pe = dense_embeddings
        src = image_embeddings + dense_embeddings

        # Transformer decoder
        # Cross-attention between tokens and image
        tokens, src = self.transformer(tokens, src, image_pe)

        # Extract mask tokens
        mask_tokens_out = tokens[:, :self.mask_tokens.num_embeddings - 1]

        # IoU prediction
        iou_predictions = self.iou_prediction_head(mask_tokens_out[:, 0])

        # Mask prediction
        # Upscale and predict
        src = src.reshape(-1, 256, 64, 64)
        masks = self.mask_prediction_head(src)  # (B, 3, 256, 256)

        return masks, iou_predictions


class TwoWayTransformer(nn.Module):
    """
    Two-way Transformer for SAM

    Features:
    - Token → Image cross-attention
    - Image → Token cross-attention
    - Token self-attention
    """
    def __init__(self, depth, embed_dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

    def forward(self, tokens, image, image_pe):
        for layer in self.layers:
            tokens, image = layer(tokens, image, image_pe)
        return tokens, image
```

---

## 3. Using SAM

### 3.1 Basic Usage

```python
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# Set image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# Segment with point prompt
input_point = np.array([[500, 375]])  # click location
input_label = np.array([1])  # 1: foreground, 0: background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # output 3 masks
)

# Select mask with highest score
best_mask = masks[np.argmax(scores)]
```

### 3.2 Various Prompts

```python
# 1. Multiple points
input_points = np.array([[500, 375], [600, 400], [450, 350]])
input_labels = np.array([1, 1, 0])  # 2 foreground, 1 background

masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,  # single mask
)

# 2. Box prompt
input_box = np.array([100, 100, 500, 400])  # x1, y1, x2, y2

masks, scores, _ = predictor.predict(
    box=input_box,
    multimask_output=False,
)

# 3. Point + Box combined
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)

# 4. Iterative refinement (using previous mask)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=logits[np.argmax(scores)][None, :, :],  # previous logits
    multimask_output=False,
)
```

### 3.3 Automatic Mask Generation

```python
from segment_anything import SamAutomaticMaskGenerator

# Automatic mask generator
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,           # 32×32 grid
    pred_iou_thresh=0.88,         # IoU threshold
    stability_score_thresh=0.95,  # stability threshold
    min_mask_region_area=100,     # minimum mask size
)

# Generate all masks in image
masks = mask_generator.generate(image)

# Result: list of dicts
# {
#     'segmentation': binary mask,
#     'area': mask area,
#     'bbox': bounding box,
#     'predicted_iou': IoU score,
#     'stability_score': stability score,
#     'crop_box': crop used for generation,
# }

print(f"Found {len(masks)} masks")

# Visualization
import matplotlib.pyplot as plt

def show_masks(image, masks):
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    for mask in masks:
        m = mask['segmentation']
        color = np.random.random(3)
        colored_mask = np.zeros((*m.shape, 4))
        colored_mask[m] = [*color, 0.5]
        plt.imshow(colored_mask)
    plt.axis('off')
    plt.show()

show_masks(image, masks)
```

### 3.4 Using HuggingFace Transformers

```python
from transformers import SamModel, SamProcessor
import torch
from PIL import Image

# Load model
model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Load image
image = Image.open("image.jpg")

# Point prompt
input_points = [[[500, 375]]]  # batch of points

inputs = processor(image, input_points=input_points, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)

scores = outputs.iou_scores
```

---

## 4. SAM 2 (2024)

### 4.1 SAM 2 Improvements

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM vs SAM 2                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SAM (2023):                                                    │
│  • Images only                                                  │
│  • Independent frame processing                                 │
│  • Video: needs prompt for each frame                           │
│                                                                 │
│  SAM 2 (2024):                                                  │
│  • Unified images + video                                       │
│  • Temporal consistency with memory attention                   │
│  • One prompt → track through entire video                      │
│                                                                 │
│  New Components:                                                │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Memory Encoder   │ Encode past frame info          │         │
│  │ Memory Bank      │ Store past masks and features   │         │
│  │ Memory Attention │ Current frame ↔ past info attn  │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 SAM 2 Video Usage

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    "sam2_hiera_large.pt",
    device="cuda"
)

# Load video frames
video_path = "video.mp4"

with predictor.init_state(video_path) as state:
    # Prompt on first frame
    _, _, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=[[500, 375]],
        labels=[1],
    )

    # Auto-propagate to remaining frames
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # masks: segmentation result for each frame
        print(f"Frame {frame_idx}: {len(object_ids)} objects")
```

---

## 5. SAM Applications

### 5.1 Grounding SAM (Text → Segment)

```python
"""
Grounding SAM = Grounding DINO + SAM

1. Grounding DINO: text → bounding box
2. SAM: bounding box → segmentation

Result: Segmentation from text prompts
"""

from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry

# Detect boxes with Grounding DINO
grounding_dino = load_model("groundingdino_swinb.pth")
boxes, logits, phrases = predict(
    grounding_dino,
    image,
    text_prompt="a cat",
    box_threshold=0.3,
    text_threshold=0.25,
)

# Segment with SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)

masks = []
for box in boxes:
    mask, _, _ = predictor.predict(box=box.numpy(), multimask_output=False)
    masks.append(mask)
```

### 5.2 Interactive Annotation Tool

```python
"""
SAM-based Interactive Labeling Tool

1. Load image
2. User clicks points/boxes
3. SAM generates masks in real-time
4. User refines (positive/negative points)
5. Save final mask
"""

import cv2
import numpy as np
from segment_anything import SamPredictor

class SAMAnnotator:
    def __init__(self, sam_checkpoint):
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        self.points = []
        self.labels = []

    def set_image(self, image):
        self.image = image.copy()
        self.predictor.set_image(image)
        self.points = []
        self.labels = []

    def add_point(self, x, y, is_foreground=True):
        self.points.append([x, y])
        self.labels.append(1 if is_foreground else 0)
        return self.predict()

    def predict(self):
        if not self.points:
            return None

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(self.points),
            point_labels=np.array(self.labels),
            multimask_output=False,
        )
        return masks[0]

    def reset(self):
        self.points = []
        self.labels = []

# Usage example (with OpenCV mouse callback)
# annotator = SAMAnnotator("sam_vit_h.pth")
# annotator.set_image(image)
# mask = annotator.add_point(500, 375, is_foreground=True)
```

### 5.3 Medical Imaging

```python
"""
Medical Image Segmentation

SAM's strengths:
- Zero-shot segmentation of new organs/lesions
- Precise masks from expert point clicks

MedSAM: SAM fine-tuned on medical images
"""

# MedSAM usage example
from medsam import MedSAMPredictor

predictor = MedSAMPredictor("medsam_checkpoint.pth")

# Load CT/MRI image
medical_image = load_medical_image("ct_scan.nii")

# Slice-by-slice segmentation
for slice_idx in range(medical_image.shape[0]):
    slice_img = medical_image[slice_idx]
    predictor.set_image(slice_img)

    # Expert clicks lesion location
    mask, _, _ = predictor.predict(
        point_coords=np.array([[tumor_x, tumor_y]]),
        point_labels=np.array([1]),
    )
```

---

## Summary

### SAM Key Components
| Component | Role | Features |
|---------|------|------|
| **Image Encoder** | Image feature extraction | MAE ViT-H, 632M params |
| **Prompt Encoder** | Prompt encoding | Point/Box/Mask support |
| **Mask Decoder** | Mask generation | 2-layer Transformer, 4M params |

### Prompt Types
- **Point**: Click location (foreground/background)
- **Box**: Bounding box
- **Mask**: Previous mask (refinement)
- **Text**: Supported via Grounding SAM

### Applications
| Use Case | Method |
|------|------|
| Interactive Annotation | Fast labeling with clicks |
| Automatic Segmentation | Grid points for all objects |
| Video Tracking | Object tracking with SAM 2 |
| Medical Imaging | Specialized with MedSAM |

### Next Steps
- [14_Unified_Vision_Models.md](14_Unified_Vision_Models.md): Unified Vision Models
- [16_Vision_Language_Deep.md](16_Vision_Language_Deep.md): Multimodal (LLaVA)

---

## References

### Papers
- Kirillov et al. (2023). "Segment Anything"
- Ravi et al. (2024). "SAM 2: Segment Anything in Images and Videos"
- Liu et al. (2023). "Grounding DINO"
- Ma et al. (2023). "Segment Anything in Medical Images" (MedSAM)

### Code
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM 2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [Grounding SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [HuggingFace SAM](https://huggingface.co/facebook/sam-vit-huge)
