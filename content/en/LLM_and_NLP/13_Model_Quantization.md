# 13. Model Quantization

## Learning Objectives

- Understand the concept and necessity of quantization
- INT8/INT4 quantization techniques
- Practice with GPTQ, AWQ, bitsandbytes
- Efficient fine-tuning with QLoRA

---

## 1. Quantization Overview

### Why is Quantization Needed?

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Memory Requirements                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Model Size   │  FP32      │  FP16      │  INT8    │  INT4   │
│  ─────────────┼────────────┼────────────┼──────────┼─────────│
│  7B params    │  28GB      │  14GB      │  7GB     │  3.5GB  │
│  13B params   │  52GB      │  26GB      │  13GB    │  6.5GB  │
│  70B params   │  280GB     │  140GB     │  70GB    │  35GB   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Quantization Types

| Type | Description | Advantages | Disadvantages |
|------|-------------|------------|---------------|
| Post-Training Quantization (PTQ) | Quantize after training | Fast, simple | Possible accuracy loss |
| Quantization-Aware Training (QAT) | Simulate quantization during training | High accuracy | Increased training time |
| Dynamic Quantization | Runtime quantization | Flexible | Inference overhead |
| Static Quantization | Calibration-based | Fast inference | Calibration required |

### Bit Precision Comparison

```python
# FP32 (32-bit floating point)
# Sign 1bit + Exponent 8bit + Mantissa 23bit
# Range: ±3.4 × 10^38, Precision: ~7 digits

# FP16 (16-bit floating point)
# Sign 1bit + Exponent 5bit + Mantissa 10bit
# Range: ±65,504, Precision: ~3 digits

# BF16 (Brain Float 16)
# Sign 1bit + Exponent 8bit + Mantissa 7bit
# Same range as FP32, lower precision

# INT8 (8-bit integer)
# Range: -128 ~ 127 or 0 ~ 255

# INT4 (4-bit integer)
# Range: -8 ~ 7 or 0 ~ 15
```

---

## 2. Quantization Mathematics

### Uniform Quantization

```python
import numpy as np

def quantize_symmetric(tensor, bits=8):
    """Symmetric quantization"""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    # Calculate scale
    abs_max = np.abs(tensor).max()
    scale = abs_max / qmax

    # Quantize
    quantized = np.round(tensor / scale).astype(np.int8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale

def dequantize(quantized, scale):
    """Dequantization"""
    return quantized.astype(np.float32) * scale


# Test
original = np.array([0.5, -1.2, 0.3, 2.1, -0.8], dtype=np.float32)
quantized, scale = quantize_symmetric(original, bits=8)
recovered = dequantize(quantized, scale)

print(f"Original: {original}")
print(f"Quantized: {quantized}")
print(f"Recovered: {recovered}")
print(f"Error: {np.abs(original - recovered).mean():.6f}")
```

### Asymmetric Quantization

```python
def quantize_asymmetric(tensor, bits=8):
    """Asymmetric quantization (zero is exactly represented)"""
    qmin = 0
    qmax = 2 ** bits - 1

    # Scale and zero point
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = round(-min_val / scale)

    # Quantize
    quantized = np.round(tensor / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale, zero_point

def dequantize_asymmetric(quantized, scale, zero_point):
    """Asymmetric dequantization"""
    return (quantized.astype(np.float32) - zero_point) * scale
```

### Group Quantization

```python
def group_quantize(tensor, group_size=128, bits=4):
    """Group-wise quantization - improved accuracy"""
    # Split tensor into groups
    flat = tensor.flatten()
    pad_size = (group_size - len(flat) % group_size) % group_size
    flat = np.pad(flat, (0, pad_size))

    groups = flat.reshape(-1, group_size)

    quantized_groups = []
    scales = []

    for group in groups:
        q, s = quantize_symmetric(group, bits)
        quantized_groups.append(q)
        scales.append(s)

    return np.array(quantized_groups), np.array(scales)
```

---

## 3. bitsandbytes Library

### Installation

```bash
pip install bitsandbytes
pip install accelerate
```

### 8-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load in 8-bit
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Check memory
print(f"8bit model memory: {model_8bit.get_memory_footprint() / 1e9:.2f} GB")

# Inference
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
outputs = model_8bit.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 4-bit Quantization (NF4)

```python
from transformers import BitsAndBytesConfig

# 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Normal Float 4 (optimized data type)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation data type
    bnb_4bit_use_double_quant=True      # Double quantization (quantize scales too)
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"4bit model memory: {model_4bit.get_memory_footprint() / 1e9:.2f} GB")
```

### NF4 vs FP4

```python
# NF4 (Normal Float 4)
# - Optimal quantization assuming normal distribution
# - Optimized for LLM weights

# FP4 (Floating Point 4)
# - General 4-bit floating point
# - General purpose

bnb_config_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Use FP4
)
```

---

## 4. GPTQ (GPU-optimized Post-Training Quantization)

### Concept

```
GPTQ quantization process:
    1. Prepare small calibration dataset
    2. Layer-wise sequential quantization
    3. Identify important weights using Hessian matrix
    4. Minimize reconstruction error

Advantages:
    - High compression ratio (3-4bit)
    - Fast inference speed
    - GPU optimized
```

### Performing GPTQ Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Calibration data
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    # ... more data
]

# GPTQ configuration
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,                    # Group size
    desc_act=True,                     # Activation order descending
    dataset=calibration_data,
    tokenizer=tokenizer
)

# Quantize and save
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto"
)

model.save_pretrained("./llama-2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-2-7b-gptq-4bit")
```

### Using AutoGPTQ

```python
# pip install auto-gptq
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantization configuration
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config
)

# Calibration data
examples = [tokenizer(text, return_tensors="pt") for text in calibration_data]

# Quantize
model.quantize(examples, batch_size=1)

# Save
model.save_quantized("./llama-2-7b-gptq")
```

### Using Pre-quantized Models

```python
# Download GPTQ models from TheBloke, etc.
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

# Inference
inputs = tokenizer("What is AI?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 5. AWQ (Activation-aware Weight Quantization)

### Concept

```
AWQ features:
    - Calculate weight importance based on activations
    - Maintain high precision for important weights
    - Faster quantization than GPTQ
    - Similar or better quality
```

### AWQ Quantization

```python
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "./llama-2-7b-awq"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# AWQ quantization configuration
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # GEMM or GEMV
}

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### AWQ Model Inference

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load AWQ model
model = AutoAWQForCausalLM.from_quantized(
    "./llama-2-7b-awq",
    fuse_layers=True  # Speed up with layer fusion
)
tokenizer = AutoTokenizer.from_pretrained("./llama-2-7b-awq")

# Inference
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 6. QLoRA (Quantized LoRA)

### Concept

```
QLoRA = 4bit quantization + LoRA

    Base model (4bit quantized, frozen)
         │
         ▼
    ┌─────────────┐
    │  LoRA A     │  (FP16, trainable)
    │  (r × d)    │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  LoRA B     │  (FP16, trainable)
    │  (d × r)    │
    └─────────────┘
         │
         ▼
    Final output = quantized weights + LoRA correction
```

### QLoRA Fine-tuning

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # Scaling factor
    target_modules=[               # Modules to apply
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable: ~0.1%, total ~400MB

# Dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['context']}

### Response:
{example['response']}"""

# Training configuration
training_args = TrainingArguments(
    output_dir="./qlora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit"  # Memory-efficient optimizer
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_prompt,
    max_seq_length=512,
    args=training_args,
)

# Train
trainer.train()

# Save
model.save_pretrained("./qlora_adapter")
```

### Merging QLoRA Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Base model (load in FP16)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Merge LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qlora_adapter")
model = model.merge_and_unload()  # Merge adapter to base model

# Save merged model
model.save_pretrained("./llama-2-7b-finetuned")
```

---

## 7. Quantization Performance Comparison

### Benchmark

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_model(model, tokenizer, prompt, num_runs=5):
    """Model inference benchmark"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)

    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    return {
        "avg_time": sum(times) / len(times),
        "memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "output": tokenizer.decode(outputs[0])
    }

# Compare results
models = {
    "FP16": model_fp16,
    "INT8": model_8bit,
    "INT4 (NF4)": model_4bit,
    "GPTQ-4bit": model_gptq,
    "AWQ-4bit": model_awq,
}

prompt = "Explain the theory of relativity:"

for name, model in models.items():
    result = benchmark_model(model, tokenizer, prompt)
    print(f"{name}:")
    print(f"  Time: {result['avg_time']:.2f}s")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
```

### Accuracy Evaluation

```python
from datasets import load_dataset
import evaluate

# Evaluation dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def compute_perplexity(model, tokenizer, texts, max_length=1024):
    """Calculate perplexity"""
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# Compare
for name, model in models.items():
    ppl = compute_perplexity(model, tokenizer, dataset["text"][:100])
    print(f"{name} Perplexity: {ppl:.2f}")
```

---

## 8. Practical Guide

### Choosing Quantization Method

```
┌─────────────────────────────────────────────────────────────┐
│                Quantization Method Selection Guide           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Purpose                 │  Recommended Method               │
│  ────────────────────────┼────────────────────────────────────│
│  Fast prototyping        │  bitsandbytes (load_in_8bit)      │
│  Memory-constrained env  │  bitsandbytes (load_in_4bit)      │
│  Production deployment   │  GPTQ or AWQ                      │
│  Fine-tuning needed      │  QLoRA                            │
│  Maximum speed           │  AWQ + fuse_layers                │
│  Maximum quality         │  GPTQ (desc_act=True)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Troubleshooting

```python
# 1. CUDA Out of Memory
# - Reduce batch size
# - Enable gradient_checkpointing
# - Use lower bit quantization

model.gradient_checkpointing_enable()

# 2. Quality degradation after quantization
# - Reduce group_size (64 or 32)
# - Increase calibration data
# - Try GPTQ instead of AWQ

# 3. Slow inference
# - Enable fuse_layers=True
# - Use exllama backend (GPTQ)
# - Utilize batch processing

from auto_gptq import exllama_set_max_input_length
exllama_set_max_input_length(model, 4096)
```

---

## Summary

### Quantization Comparison Table

| Method | Bits | Speed | Quality | Ease of Use |
|--------|------|-------|---------|-------------|
| FP16 | 16 | Baseline | Baseline | Easy |
| INT8 (bitsandbytes) | 8 | Fast | High | Easy |
| INT4 (NF4) | 4 | Fast | Good | Easy |
| GPTQ | 4/3/2 | Very Fast | Good | Medium |
| AWQ | 4 | Very Fast | Good | Medium |
| QLoRA | 4 | - | Training | Medium |

### Core Code

```python
# bitsandbytes 4-bit
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config)

# QLoRA
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# GPTQ
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
model.quantize(examples)

# AWQ
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
```

---

## Next Steps

In [14_RLHF_Alignment.md](./14_RLHF_Alignment.md), we'll learn about LLM alignment techniques (RLHF, DPO).
