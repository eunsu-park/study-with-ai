# 16. Advanced Vision-Language

## Overview

Vision-Language Models (VLMs) are models that understand both images and text together. This lesson covers state-of-the-art VLM architectures like LLaVA and Qwen-VL, as well as Visual Instruction Tuning techniques.

---

## 1. VLM Paradigm

### 1.1 Evolution

```
┌──────────────────────────────────────────────────────────────────┐
│                    VLM Evolution                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2021: CLIP                                                      │
│  - Image-Text contrastive learning                              │
│  - Zero-shot classification                                     │
│                                                                  │
│  2022: Flamingo                                                  │
│  - Inject visual tokens into LLM                                │
│  - Few-shot vision-language learning                            │
│                                                                  │
│  2023: LLaVA                                                     │
│  - Visual Instruction Tuning                                    │
│  - Open-source GPT-4V alternative                               │
│                                                                  │
│  2024: LLaVA-NeXT, Qwen-VL, Phi-3-Vision                        │
│  - High resolution, multi-image, video                          │
│  - Commercial-grade performance                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Architecture Comparison

| Model | Vision Encoder | LLM | Connection Method |
|------|---------------|-----|----------|
| **LLaVA** | CLIP ViT-L | Vicuna/LLaMA | Linear Projection |
| **Qwen-VL** | ViT-G | Qwen | Cross-Attention |
| **InternVL** | InternViT | InternLM | MLP |
| **Phi-3-Vision** | CLIP ViT | Phi-3 | Linear |
| **GPT-4V** | Unknown | GPT-4 | Unknown |

---

## 2. LLaVA (Large Language and Vision Assistant)

### 2.1 Architecture

```
LLaVA Structure:

Image → CLIP ViT-L/14 → Visual Features (576 tokens)
                ↓
         Linear Projection
                ↓
         Visual Tokens
                ↓
[System] [Visual Tokens] [User Query] → LLaMA/Vicuna → Response

Training Stages:
1. Pre-training: Image-Text alignment (CC3M)
2. Fine-tuning: Visual Instruction Tuning (158K)
```

### 2.2 Implementation

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

class LLaVAModel(nn.Module):
    """LLaVA-style Vision-Language Model"""

    def __init__(
        self,
        vision_encoder: str = "openai/clip-vit-large-patch14",
        llm: str = "lmsys/vicuna-7b-v1.5",
        freeze_vision: bool = True,
        freeze_llm: bool = False
    ):
        super().__init__()

        # Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder)
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        # Language Model
        self.llm = LlamaForCausalLM.from_pretrained(llm)
        self.llm_hidden_size = self.llm.config.hidden_size

        # Vision-Language Projection
        self.vision_projection = nn.Linear(
            self.vision_hidden_size,
            self.llm_hidden_size
        )

        # Freeze encoders
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Image encoding

        Args:
            images: (B, C, H, W)

        Returns:
            visual_tokens: (B, num_patches, llm_hidden_size)
        """
        # CLIP encoding
        vision_outputs = self.vision_encoder(images)
        image_features = vision_outputs.last_hidden_state  # (B, 257, 1024)

        # Exclude [CLS] token
        image_features = image_features[:, 1:, :]  # (B, 256, 1024)

        # Project to LLM space
        visual_tokens = self.vision_projection(image_features)

        return visual_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor = None,
        image_positions: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        """
        Forward pass

        Args:
            input_ids: (B, seq_len) text tokens
            attention_mask: (B, seq_len)
            images: (B, C, H, W) images
            image_positions: positions where image tokens should be inserted
            labels: (B, seq_len) for training
        """
        B, seq_len = input_ids.shape

        # Text embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)

        # Image embeddings
        if images is not None:
            visual_tokens = self.encode_images(images)  # (B, num_patches, hidden)

            # Interleave visual tokens with text
            # Simplified: add images before text
            combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            # Adjust attention mask
            visual_mask = torch.ones(B, visual_tokens.shape[1], device=attention_mask.device)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_embeds = text_embeds
            combined_mask = attention_mask

        # LLM forward
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels,
            return_dict=True
        )

        return outputs


class VisualInstructionDataset:
    """Visual Instruction Tuning Dataset"""

    INSTRUCTION_TEMPLATES = [
        "Describe this image in detail.",
        "What can you see in this image?",
        "Explain what is happening in this picture.",
        "<question>",  # VQA
    ]

    def __init__(self, data_path: str):
        """
        Data format:
        {
            "image": "path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image."},
                {"from": "gpt", "value": "This image shows..."}
            ]
        }
        """
        import json
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        from PIL import Image
        image = Image.open(item['image']).convert('RGB')

        # Construct conversation
        conversations = item['conversations']
        human_input = conversations[0]['value']
        assistant_output = conversations[1]['value']

        return {
            'image': image,
            'human': human_input,
            'assistant': assistant_output
        }
```

### 2.3 LLaVA-NeXT Improvements

```python
class LLaVANeXTConfig:
    """
    LLaVA-NeXT Improvements

    1. High-resolution support (AnyRes)
    2. Better Vision Encoder (SigLIP)
    3. Larger LLM (Llama 3, Qwen 2)
    """

    # AnyRes: handle various resolutions
    SUPPORTED_RESOLUTIONS = [
        (336, 336),
        (672, 336),
        (336, 672),
        (672, 672),
        (1008, 336),
        (336, 1008),
    ]

    @staticmethod
    def select_best_resolution(image_size: tuple, resolutions: list):
        """Select best resolution for image"""
        img_h, img_w = image_size
        img_ratio = img_w / img_h

        best_res = None
        best_ratio_diff = float('inf')

        for res in resolutions:
            res_ratio = res[1] / res[0]
            ratio_diff = abs(img_ratio - res_ratio)

            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_res = res

        return best_res


def anyres_processing(image, base_resolution=336):
    """
    AnyRes image processing

    Split high-resolution image into base resolution tiles
    + downscaled full image
    """
    from PIL import Image
    import torch

    # 1. Resize full image (global context)
    global_image = image.resize((base_resolution, base_resolution))

    # 2. Split into tiles (local details)
    W, H = image.size
    num_tiles_w = (W + base_resolution - 1) // base_resolution
    num_tiles_h = (H + base_resolution - 1) // base_resolution

    tiles = []
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            left = j * base_resolution
            top = i * base_resolution
            right = min(left + base_resolution, W)
            bottom = min(top + base_resolution, H)

            tile = image.crop((left, top, right, bottom))
            # Padding
            padded_tile = Image.new('RGB', (base_resolution, base_resolution))
            padded_tile.paste(tile, (0, 0))
            tiles.append(padded_tile)

    # [global_image] + [tile1, tile2, ...]
    all_images = [global_image] + tiles

    return all_images
```

---

## 3. Qwen-VL

### 3.1 Architecture

```
Qwen-VL Features:

1. Vision Encoder: ViT-bigG (1.9B params)
2. High resolution: 448×448 (variable)
3. Grounding support: bounding box output
4. OCR strength: excellent text recognition

Input format:
<img>image_path</img> User question
<ref>object name</ref><box>(x1,y1),(x2,y2)</box>
```

### 3.2 Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def use_qwen_vl():
    """Using Qwen-VL"""

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True
    )

    # Basic VQA
    query = tokenizer.from_list_format([
        {'image': 'path/to/image.jpg'},
        {'text': 'What is in this image?'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)

    # Grounding (find object locations)
    query = tokenizer.from_list_format([
        {'image': 'path/to/image.jpg'},
        {'text': 'Find all the cats in this image and output their bounding boxes.'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    # Output: <ref>cat</ref><box>(100,200),(300,400)</box>

    # Multiple images
    query = tokenizer.from_list_format([
        {'image': 'image1.jpg'},
        {'image': 'image2.jpg'},
        {'text': 'What is the difference between these two images?'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)

    return response
```

---

## 4. Visual Instruction Tuning

### 4.1 Data Generation

```python
class VisualInstructionGenerator:
    """Visual Instruction data generator"""

    def __init__(self, teacher_model="gpt-4-vision-preview"):
        from openai import OpenAI
        self.client = OpenAI()
        self.teacher_model = teacher_model

    def generate_conversation(
        self,
        image_path: str,
        task_type: str = "detailed_description"
    ):
        """Generate training data with GPT-4V"""
        import base64

        # Encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        task_prompts = {
            "detailed_description": "Describe this image in detail.",
            "reasoning": "What conclusions can you draw from this image? Explain your reasoning.",
            "conversation": "Generate a multi-turn conversation about this image.",
            "creative": "Write a creative story inspired by this image."
        }

        prompt = task_prompts.get(task_type, task_prompts["detailed_description"])

        response = self.client.chat.completions.create(
            model=self.teacher_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=1024
        )

        return {
            "image": image_path,
            "task": task_type,
            "question": prompt,
            "answer": response.choices[0].message.content
        }

    def generate_dataset(
        self,
        image_paths: list,
        output_path: str,
        tasks: list = None
    ):
        """Generate large-scale dataset"""
        import json
        from tqdm import tqdm

        if tasks is None:
            tasks = ["detailed_description", "reasoning", "conversation"]

        dataset = []

        for image_path in tqdm(image_paths):
            for task in tasks:
                try:
                    data = self.generate_conversation(image_path, task)
                    dataset.append(data)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        return dataset
```

### 4.2 Training Strategy

```python
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

def finetune_vlm():
    """VLM Fine-tuning"""

    # Load model
    model = LLaVAModel(
        freeze_vision=True,  # Freeze vision encoder
        freeze_llm=False     # Fine-tune LLM
    )

    # Apply LoRA (efficient training)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
    )

    model.llm = get_peft_model(model.llm, lora_config)

    # Training setup
    training_args = TrainingArguments(
        output_dir="./llava-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=500,
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vlm_data_collator,
    )

    trainer.train()


def vlm_data_collator(features):
    """VLM data collator"""
    batch = {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'images': torch.stack([f['image'] for f in features]),
        'labels': torch.stack([f['labels'] for f in features]),
    }
    return batch
```

---

## 5. Evaluation Benchmarks

### 5.1 Major Benchmarks

```
VLM Evaluation Benchmarks:

1. VQA-v2: General Visual QA
2. GQA: Structural reasoning QA
3. TextVQA: Understanding text in images
4. POPE: Hallucination evaluation
5. MME: 14 subtask suite
6. MMBench: 20 capability evaluation
7. SEED-Bench: 19K multiple choice problems
```

### 5.2 Evaluation Code

```python
def evaluate_vlm(model, dataset_name: str = "vqav2"):
    """VLM evaluation"""

    if dataset_name == "vqav2":
        return evaluate_vqa_v2(model)
    elif dataset_name == "textvqa":
        return evaluate_textvqa(model)
    elif dataset_name == "pope":
        return evaluate_pope(model)


def evaluate_pope(model):
    """
    POPE: Polling-based Object Probing Evaluation

    Hallucination evaluation: "Is there a [object] in the image?"
    """
    from datasets import load_dataset

    dataset = load_dataset("lmms-lab/POPE")

    correct = 0
    total = 0

    for item in dataset['test']:
        image = item['image']
        question = item['question']  # "Is there a dog in the image?"
        answer = item['answer']      # "yes" or "no"

        # Model prediction
        prediction = model.generate(image, question)
        pred_answer = "yes" if "yes" in prediction.lower() else "no"

        if pred_answer == answer:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"POPE Accuracy: {accuracy:.4f}")

    return accuracy
```

---

## 6. Practical Applications

### 6.1 Document Understanding

```python
def document_understanding():
    """Document understanding application"""

    model = load_qwen_vl()  # OCR strength

    # PDF page analysis
    def analyze_document_page(image_path: str, questions: list):
        results = []

        for question in questions:
            query = f"<img>{image_path}</img>{question}"
            answer = model.generate(query)
            results.append({
                'question': question,
                'answer': answer
            })

        return results

    # Example questions
    questions = [
        "What is the title of this document?",
        "Summarize the main points.",
        "Extract all dates mentioned.",
        "What tables are present? Describe their contents.",
    ]

    results = analyze_document_page("document_page.png", questions)


def chart_understanding():
    """Chart/graph understanding"""

    prompts = [
        "What type of chart is this?",
        "What is the trend shown in this chart?",
        "What are the maximum and minimum values?",
        "Describe the relationship between X and Y.",
    ]

    # Chart analysis with VLM
    for prompt in prompts:
        response = model.generate(chart_image, prompt)
        print(f"Q: {prompt}")
        print(f"A: {response}\n")
```

---

## References

### Papers
- Liu et al. (2023). "Visual Instruction Tuning" (LLaVA)
- Liu et al. (2024). "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge"
- Bai et al. (2023). "Qwen-VL: A Versatile Vision-Language Model"

### Models
- [LLaVA](https://llava-vl.github.io/)
- [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [InternVL](https://github.com/OpenGVLab/InternVL)

### Related Lessons
- [../Deep_Learning/20_CLIP_Multimodal.md](../Deep_Learning/20_CLIP_Multimodal.md)
- [12_DINOv2_Self_Supervised.md](12_DINOv2_Self_Supervised.md)
