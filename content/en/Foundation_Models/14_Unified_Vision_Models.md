# 14. Unified Vision Models

## Overview

Unified Vision Models represent a paradigm that processes various vision tasks (classification, detection, segmentation, etc.) with a **single model**. Instead of task-specific models, the goal is to build general-purpose vision models.

---

## 1. Paradigm Shift

### 1.1 Traditional Approach vs Unified Approach

```
┌──────────────────────────────────────────────────────────────────┐
│                    Vision Model Paradigms                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Traditional (Task-Specific):                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ ResNet       │  │ Faster R-CNN │  │ DeepLab      │           │
│  │ (classif.)   │  │ (detection)  │  │ (segment.)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  Unified (Task-Agnostic):                                        │
│  ┌───────────────────────────────────────────────┐              │
│  │              Unified Vision Model              │              │
│  │  "Classify this" → Classification result       │              │
│  │  "Find objects" → Bounding boxes               │              │
│  │  "Segment this" → Masks                        │              │
│  └───────────────────────────────────────────────┘              │
│                                                                  │
│  Advantages: Knowledge sharing, Easy maintenance, Zero-shot     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Major Model Comparison

| Model | Developer | Features | Supported Tasks |
|-------|-----------|----------|-----------------|
| **Florence** | Microsoft | Large-scale Image-Text | Classification, Detection, Captioning, VQA |
| **PaLI** | Google | Multilingual VLM | Captioning, VQA, OCR |
| **Unified-IO** | Allen AI | All modalities | Image, Audio, Text |
| **OFA** | Alibaba | Seq2Seq unified | Various vision-language |
| **GPT-4V** | OpenAI | Commercial multimodal | General vision understanding |

---

## 2. Florence: Foundation Model for Vision

### 2.1 Architecture

```
Florence Architecture:

Image Encoder: CoSwin Transformer (Hierarchical)
Text Encoder: UniCL (Unified Contrastive Learning)

Training:
1. Image-Text Contrastive (CLIP style)
2. Image-Text Matching
3. Masked Language Modeling

Features:
- Trained on 900M Image-Text pairs
- Various granularity (image → region → pixel)
- Dynamic Head for task adaptation
```

### 2.2 Implementation Example

```python
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class FlorenceStyleModel(nn.Module):
    """
    Florence-style Unified Vision Model (Simplified)

    Core: CLIP backbone + Task-specific Heads
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        num_classes: int = 1000,
        num_detection_classes: int = 80
    ):
        super().__init__()

        # CLIP backbone (Image + Text encoder)
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        hidden_size = self.clip.config.vision_config.hidden_size

        # Task Heads
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.detection_head = DetectionHead(hidden_size, num_detection_classes)
        self.segmentation_head = SegmentationHead(hidden_size)
        self.caption_head = CaptionHead(hidden_size, self.clip.config.text_config)

    def forward(
        self,
        images: torch.Tensor,
        task: str = "classification",
        text_prompts: list = None
    ):
        """
        Args:
            images: (B, 3, H, W)
            task: "classification", "detection", "segmentation", "caption"
            text_prompts: Text prompts (for zero-shot)
        """
        # Image features
        vision_outputs = self.clip.vision_model(images)
        image_features = vision_outputs.last_hidden_state  # (B, num_patches+1, hidden)
        pooled_features = vision_outputs.pooler_output  # (B, hidden)

        if task == "classification":
            if text_prompts:
                # Zero-shot classification (CLIP style)
                return self._zero_shot_classify(pooled_features, text_prompts)
            else:
                return self.classification_head(pooled_features)

        elif task == "detection":
            return self.detection_head(image_features)

        elif task == "segmentation":
            return self.segmentation_head(image_features)

        elif task == "caption":
            return self.caption_head(pooled_features)

    def _zero_shot_classify(
        self,
        image_features: torch.Tensor,
        text_prompts: list
    ) -> torch.Tensor:
        """Zero-shot classification with text prompts"""
        # Text encoding
        text_inputs = self.processor(
            text=text_prompts,
            return_tensors="pt",
            padding=True
        ).to(image_features.device)

        text_features = self.clip.get_text_features(**text_inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Similarity
        similarity = image_features @ text_features.T
        return similarity


class DetectionHead(nn.Module):
    """Object Detection Head (DETR style)"""

    def __init__(self, hidden_size: int, num_classes: int, num_queries: int = 100):
        super().__init__()
        self.num_queries = num_queries

        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, 8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Prediction heads
        self.class_head = nn.Linear(hidden_size, num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # (cx, cy, w, h)
        )

    def forward(self, image_features: torch.Tensor):
        B = image_features.size(0)

        # Query embedding
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Decoder
        hs = self.decoder(queries, image_features)

        # Predictions
        class_logits = self.class_head(hs)
        bbox_pred = self.bbox_head(hs).sigmoid()

        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred
        }


class SegmentationHead(nn.Module):
    """Semantic Segmentation Head"""

    def __init__(self, hidden_size: int, num_classes: int = 150):
        super().__init__()

        # FPN-style decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, image_features: torch.Tensor):
        # Reshape patches to spatial
        B, N, C = image_features.shape
        H = W = int((N - 1) ** 0.5)  # -1 for CLS token
        features = image_features[:, 1:, :].transpose(1, 2).view(B, C, H, W)

        return self.decoder(features)


class CaptionHead(nn.Module):
    """Image Captioning Head"""

    def __init__(self, hidden_size: int, text_config):
        super().__init__()
        self.vocab_size = text_config.vocab_size

        # Cross-attention decoder
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, 8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.lm_head = nn.Linear(hidden_size, self.vocab_size)

    def forward(
        self,
        image_features: torch.Tensor,
        target_ids: torch.Tensor = None
    ):
        # Autoregressive during generation
        # Teacher forcing during training
        pass  # Implementation omitted
```

---

## 3. PaLI (Pathways Language and Image model)

### 3.1 Architecture

```
PaLI Structure:

┌────────────────────────────────────────────────────────┐
│                      PaLI                              │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Image Encoder: ViT-e (4B params, trained on 22B imgs) │
│       ↓                                                │
│  Visual Tokens: [IMG1] [IMG2] ... [IMGn]              │
│       ↓                                                │
│  Text Encoder-Decoder: mT5 (multilingual)             │
│       ↓                                                │
│  Output: Text (multilingual support)                   │
│                                                        │
│  Input format:                                         │
│  "<image> Describe this image" → "A cat is..."        │
│  "<image> What is in the image?" → "A cat..."         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 3.2 Task Unification

```python
class PaLITaskFormats:
    """PaLI task-specific input formats"""

    TASK_FORMATS = {
        # Classification
        "classification": "What is in this image?",
        "fine_grained": "What species of bird is this?",

        # Captioning
        "caption_en": "Generate a caption for this image.",
        "caption_ko": "Write a description for this image.",

        # VQA
        "vqa": "Question: {question} Answer:",

        # OCR
        "ocr": "What text is in this image?",

        # Detection (expressed as text)
        "detection": "Detect all objects in this image.",
        # Output: "cat [100, 200, 300, 400]; dog [50, 60, 150, 200]"

        # Referring segmentation
        "referring": "Segment the {object}.",
    }

    @staticmethod
    def format_input(task: str, **kwargs) -> str:
        template = PaLITaskFormats.TASK_FORMATS.get(task, "")
        return template.format(**kwargs)


# Usage example
def process_with_pali(model, image, task, **kwargs):
    """PaLI-style processing"""

    # Task-specific prompt
    prompt = PaLITaskFormats.format_input(task, **kwargs)

    # Visual tokens + Text tokens
    inputs = model.prepare_inputs(image, prompt)

    # Generate
    outputs = model.generate(**inputs)

    # Parse output based on task
    if task == "detection":
        return parse_detection_output(outputs)
    elif task == "caption_en":
        return outputs
    else:
        return outputs
```

---

## 4. Unified-IO

### 4.1 True Unification: All Modalities

```
Unified-IO: Process all I/O with a single model

Input/Output formats:
- Image → VQ-VAE tokens
- Text → Subword tokens
- Bounding box → Coordinate tokens (discretized)
- Mask → VQ-VAE tokens
- Audio → Spectrogram VQ-VAE

Convert everything to token sequences → Seq2Seq Transformer
```

### 4.2 Implementation Concept

```python
class UnifiedIOTokenizer:
    """Unified-IO style tokenization"""

    def __init__(self, vocab_size: int = 50000, image_vocab_size: int = 16384):
        self.vocab_size = vocab_size
        self.image_vocab_size = image_vocab_size

        # Special tokens
        self.SPECIAL_TOKENS = {
            '<image>': vocab_size,
            '</image>': vocab_size + 1,
            '<box>': vocab_size + 2,
            '</box>': vocab_size + 3,
            '<mask>': vocab_size + 4,
            '</mask>': vocab_size + 5,
            '<audio>': vocab_size + 6,
            '</audio>': vocab_size + 7,
        }

        # Coordinate discretization bins
        self.num_bins = 1000

    def tokenize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Tokenize image with VQ-VAE"""
        # Extract discrete codes with VQ-VAE encoder
        # codes shape: (H', W')
        codes = self.vqvae.encode(image)

        # Flatten + offset
        tokens = codes.flatten() + self.vocab_size + len(self.SPECIAL_TOKENS)

        return tokens

    def tokenize_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Bounding box to discrete tokens

        bbox: (x1, y1, x2, y2) normalized [0, 1]
        """
        # Discretize each coordinate to bins
        bins = (bbox * self.num_bins).long()

        # Special tokens + bins
        tokens = torch.tensor([
            self.SPECIAL_TOKENS['<box>'],
            bins[0], bins[1], bins[2], bins[3],
            self.SPECIAL_TOKENS['</box>']
        ])

        return tokens

    def decode_bbox(self, tokens: torch.Tensor) -> torch.Tensor:
        """Restore bounding box from tokens"""
        # Find <box> token position
        # Extract 4 number tokens
        # Denormalize
        pass


class UnifiedIOModel(nn.Module):
    """Unified-IO style model"""

    def __init__(self, config):
        super().__init__()

        # Unified Embedding
        self.embeddings = nn.ModuleDict({
            'text': nn.Embedding(config.text_vocab_size, config.hidden_size),
            'image': nn.Embedding(config.image_vocab_size, config.hidden_size),
            'coord': nn.Embedding(config.num_bins, config.hidden_size),
        })

        # Encoder-Decoder Transformer
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

        # Unified LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.total_vocab_size)

    def forward(self, input_tokens, output_tokens=None):
        """
        Seq2Seq forward

        input_tokens: Mixed modality tokens
        output_tokens: Target output tokens
        """
        # Embedding by token type
        embeddings = self._get_embeddings(input_tokens)

        # Encoder
        encoder_output = self.encoder(embeddings)

        # Decoder
        if output_tokens is not None:
            decoder_input = self._get_embeddings(output_tokens)
            decoder_output = self.decoder(decoder_input, encoder_output)
            logits = self.lm_head(decoder_output)
            return logits

        return encoder_output

    def _get_embeddings(self, tokens):
        """Select appropriate embedding based on token type"""
        # Distinguish text/image/coord based on token range
        pass


# Various task examples
def unified_io_examples():
    """Unified-IO task examples"""

    examples = {
        # Image Captioning
        "caption": {
            "input": "<image> {image_tokens} </image> Describe this image.",
            "output": "A cat sitting on a windowsill."
        },

        # Object Detection
        "detection": {
            "input": "<image> {image_tokens} </image> Detect all objects.",
            "output": "cat <box> 100 200 300 400 </box> dog <box> 50 60 150 200 </box>"
        },

        # Segmentation
        "segmentation": {
            "input": "<image> {image_tokens} </image> Segment the cat.",
            "output": "<mask> {mask_tokens} </mask>"
        },

        # Image Generation (reverse)
        "generation": {
            "input": "Generate an image of a sunset over mountains.",
            "output": "<image> {image_tokens} </image>"
        },

        # VQA
        "vqa": {
            "input": "<image> {image_tokens} </image> How many cats are there?",
            "output": "2"
        }
    }

    return examples
```

---

## 5. Practical Usage

### 5.1 Using Florence-2 (HuggingFace)

```python
from transformers import AutoProcessor, AutoModelForCausalLM

def use_florence2():
    """Florence-2 practical usage"""

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )

    from PIL import Image
    import requests

    url = "https://example.com/image.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Various tasks
    tasks = {
        "<CAPTION>": "Short caption",
        "<DETAILED_CAPTION>": "Detailed caption",
        "<MORE_DETAILED_CAPTION>": "Very detailed caption",
        "<OD>": "Object detection",
        "<DENSE_REGION_CAPTION>": "Region-wise caption",
        "<REGION_PROPOSAL>": "Region proposal",
        "<CAPTION_TO_PHRASE_GROUNDING>": "Text→Region grounding",
        "<REFERRING_EXPRESSION_SEGMENTATION>": "Referring expression segmentation",
        "<OCR>": "OCR",
        "<OCR_WITH_REGION>": "Region-wise OCR",
    }

    for task_prompt, description in tasks.items():
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)

        print(f"\n{description} ({task_prompt}):")
        print(parsed)


# Run
use_florence2()
```

### 5.2 Custom Task Training

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

def finetune_unified_vision():
    """Fine-tuning unified vision model"""

    # Prepare multitask dataset
    def create_multitask_dataset():
        """Multiple tasks into one dataset"""
        samples = []

        # Classification samples
        for img_path, label in classification_data:
            samples.append({
                'image': img_path,
                'task': '<CLASSIFICATION>',
                'input_text': '<CLASSIFICATION>',
                'output_text': label
            })

        # Caption samples
        for img_path, caption in caption_data:
            samples.append({
                'image': img_path,
                'task': '<CAPTION>',
                'input_text': '<CAPTION>',
                'output_text': caption
            })

        # VQA samples
        for img_path, question, answer in vqa_data:
            samples.append({
                'image': img_path,
                'task': '<VQA>',
                'input_text': f'<VQA> {question}',
                'output_text': answer
            })

        return Dataset.from_list(samples)

    dataset = create_multitask_dataset()

    # Training
    training_args = TrainingArguments(
        output_dir="./unified-vision-finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=1e-5,
        # Task sampling strategy
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
```

---

## 6. Future Directions

### 6.1 World Models

```
Next Step: World Models

Vision models + Physical understanding + Action prediction

Examples:
- Understanding physics laws from images
- "Where will the ball go if thrown?"
- Next frame prediction in video
- Robot manipulation planning
```

### 6.2 Limitations and Trade-offs of Unification

```
Advantages:
✓ Knowledge sharing across tasks
✓ Single model maintenance
✓ Zero-shot transfer
✓ Easy adaptation to new tasks

Disadvantages:
✗ May not achieve best performance on individual tasks
✗ Training complexity
✗ Task interference
✗ Large model size

Trade-offs:
- Versatility vs Specialization
- Convenience vs Optimal performance
```

---

## References

### Papers
- Yuan et al. (2021). "Florence: A New Foundation Model for Computer Vision"
- Chen et al. (2022). "PaLI: A Jointly-Scaled Multilingual Language-Image Model"
- Lu et al. (2022). "Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks"

### Models
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- [PaLI](https://github.com/google-research/pali)

### Related Lessons
- [12_DINOv2_Self_Supervised.md](12_DINOv2_Self_Supervised.md)
- [13_Segment_Anything.md](13_Segment_Anything.md)
