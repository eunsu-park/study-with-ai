# 13. 모델 양자화 (Model Quantization)

## 학습 목표

- 양자화 개념과 필요성 이해
- INT8/INT4 양자화 기법
- GPTQ, AWQ, bitsandbytes 실습
- QLoRA를 통한 효율적인 파인튜닝

---

## 1. 양자화 개요

### 왜 양자화가 필요한가?

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM 메모리 요구량                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  모델 크기    │  FP32      │  FP16      │  INT8    │  INT4   │
│  ─────────────┼────────────┼────────────┼──────────┼─────────│
│  7B 파라미터  │  28GB      │  14GB      │  7GB     │  3.5GB  │
│  13B 파라미터 │  52GB      │  26GB      │  13GB    │  6.5GB  │
│  70B 파라미터 │  280GB     │  140GB     │  70GB    │  35GB   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 양자화 유형

| 유형 | 설명 | 장점 | 단점 |
|------|------|------|------|
| Post-Training Quantization (PTQ) | 학습 후 양자화 | 빠름, 간편 | 정확도 손실 가능 |
| Quantization-Aware Training (QAT) | 학습 중 양자화 시뮬레이션 | 높은 정확도 | 학습 시간 증가 |
| Dynamic Quantization | 런타임 양자화 | 유연함 | 추론 오버헤드 |
| Static Quantization | 캘리브레이션 기반 | 빠른 추론 | 캘리브레이션 필요 |

### 비트 정밀도 비교

```python
# FP32 (32-bit floating point)
# 부호 1bit + 지수 8bit + 가수 23bit
# 범위: ±3.4 × 10^38, 정밀도: ~7자리

# FP16 (16-bit floating point)
# 부호 1bit + 지수 5bit + 가수 10bit
# 범위: ±65,504, 정밀도: ~3자리

# BF16 (Brain Float 16)
# 부호 1bit + 지수 8bit + 가수 7bit
# FP32와 같은 범위, 낮은 정밀도

# INT8 (8-bit integer)
# 범위: -128 ~ 127 또는 0 ~ 255

# INT4 (4-bit integer)
# 범위: -8 ~ 7 또는 0 ~ 15
```

---

## 2. 양자화 수학

### 균일 양자화 (Uniform Quantization)

```python
import numpy as np

def quantize_symmetric(tensor, bits=8):
    """대칭 양자화"""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    # 스케일 계산
    abs_max = np.abs(tensor).max()
    scale = abs_max / qmax

    # 양자화
    quantized = np.round(tensor / scale).astype(np.int8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale

def dequantize(quantized, scale):
    """역양자화"""
    return quantized.astype(np.float32) * scale


# 테스트
original = np.array([0.5, -1.2, 0.3, 2.1, -0.8], dtype=np.float32)
quantized, scale = quantize_symmetric(original, bits=8)
recovered = dequantize(quantized, scale)

print(f"원본: {original}")
print(f"양자화: {quantized}")
print(f"복원: {recovered}")
print(f"오차: {np.abs(original - recovered).mean():.6f}")
```

### 비대칭 양자화

```python
def quantize_asymmetric(tensor, bits=8):
    """비대칭 양자화 (0이 정확히 표현됨)"""
    qmin = 0
    qmax = 2 ** bits - 1

    # 스케일과 제로포인트
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = round(-min_val / scale)

    # 양자화
    quantized = np.round(tensor / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale, zero_point

def dequantize_asymmetric(quantized, scale, zero_point):
    """비대칭 역양자화"""
    return (quantized.astype(np.float32) - zero_point) * scale
```

### 블록별 양자화 (Group Quantization)

```python
def group_quantize(tensor, group_size=128, bits=4):
    """그룹별 양자화 - 정확도 향상"""
    # 텐서를 그룹으로 나눔
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

## 3. bitsandbytes 라이브러리

### 설치

```bash
pip install bitsandbytes
pip install accelerate
```

### 8비트 양자화

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 8비트 로드
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 메모리 확인
print(f"8bit 모델 메모리: {model_8bit.get_memory_footprint() / 1e9:.2f} GB")

# 추론
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
outputs = model_8bit.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 4비트 양자화 (NF4)

```python
from transformers import BitsAndBytesConfig

# 4비트 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Normal Float 4 (최적화된 데이터 타입)
    bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 시 데이터 타입
    bnb_4bit_use_double_quant=True      # 이중 양자화 (스케일도 양자화)
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"4bit 모델 메모리: {model_4bit.get_memory_footprint() / 1e9:.2f} GB")
```

### NF4 vs FP4

```python
# NF4 (Normal Float 4)
# - 정규분포를 가정한 최적 양자화
# - LLM 가중치에 최적화

# FP4 (Floating Point 4)
# - 일반적인 4비트 부동소수점
# - 범용적

bnb_config_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # FP4 사용
)
```

---

## 4. GPTQ (GPU-optimized Post-Training Quantization)

### 개념

```
GPTQ 양자화 과정:
    1. 작은 캘리브레이션 데이터셋 준비
    2. 레이어별 순차 양자화
    3. Hessian 행렬로 중요 가중치 판별
    4. 재구성 오차 최소화

장점:
    - 높은 압축률 (3-4bit)
    - 빠른 추론 속도
    - GPU 최적화
```

### GPTQ 양자화 수행

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# 캘리브레이션 데이터
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    # ... 더 많은 데이터
]

# GPTQ 설정
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,                    # 그룹 크기
    desc_act=True,                     # Activation order descending
    dataset=calibration_data,
    tokenizer=tokenizer
)

# 양자화 및 저장
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto"
)

model.save_pretrained("./llama-2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-2-7b-gptq-4bit")
```

### AutoGPTQ 사용

```python
# pip install auto-gptq
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 양자화 설정
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# 모델 로드
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config
)

# 캘리브레이션 데이터
examples = [tokenizer(text, return_tensors="pt") for text in calibration_data]

# 양자화
model.quantize(examples, batch_size=1)

# 저장
model.save_quantized("./llama-2-7b-gptq")
```

### 사전 양자화 모델 사용

```python
# TheBloke 등에서 GPTQ 모델 다운로드
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

# 추론
inputs = tokenizer("What is AI?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 5. AWQ (Activation-aware Weight Quantization)

### 개념

```
AWQ 특징:
    - 활성화 기반 가중치 중요도 계산
    - 중요한 가중치는 높은 정밀도 유지
    - GPTQ보다 빠른 양자화
    - 비슷하거나 더 나은 품질
```

### AWQ 양자화

```python
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 모델 로드
model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "./llama-2-7b-awq"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# AWQ 양자화 설정
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # GEMM or GEMV
}

# 양자화
model.quantize(tokenizer, quant_config=quant_config)

# 저장
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### AWQ 모델 추론

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# AWQ 모델 로드
model = AutoAWQForCausalLM.from_quantized(
    "./llama-2-7b-awq",
    fuse_layers=True  # 레이어 퓨전으로 속도 향상
)
tokenizer = AutoTokenizer.from_pretrained("./llama-2-7b-awq")

# 추론
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 6. QLoRA (Quantized LoRA)

### 개념

```
QLoRA = 4bit 양자화 + LoRA

    기본 모델 (4bit 양자화, 고정)
         │
         ▼
    ┌─────────────┐
    │  LoRA A     │  (FP16, 학습)
    │  (r × d)    │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  LoRA B     │  (FP16, 학습)
    │  (d × r)    │
    └─────────────┘
         │
         ▼
    최종 출력 = 양자화된 가중치 + LoRA 보정
```

### QLoRA 파인튜닝

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

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# k-bit 학습 준비
model = prepare_model_for_kbit_training(model)

# LoRA 설정
lora_config = LoraConfig(
    r=16,                          # LoRA 랭크
    lora_alpha=32,                 # 스케일링 팩터
    target_modules=[               # 적용할 모듈
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 학습 가능: ~0.1%, 전체의 ~400MB

# 데이터셋
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['context']}

### Response:
{example['response']}"""

# 학습 설정
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
    optim="paged_adamw_8bit"  # 메모리 효율적인 옵티마이저
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_prompt,
    max_seq_length=512,
    args=training_args,
)

# 학습
trainer.train()

# 저장
model.save_pretrained("./qlora_adapter")
```

### QLoRA 모델 병합

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 기본 모델 (FP16으로 로드)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 어댑터 병합
model = PeftModel.from_pretrained(base_model, "./qlora_adapter")
model = model.merge_and_unload()  # 어댑터를 기본 모델에 병합

# 병합된 모델 저장
model.save_pretrained("./llama-2-7b-finetuned")
```

---

## 7. 양자화 성능 비교

### 벤치마크

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_model(model, tokenizer, prompt, num_runs=5):
    """모델 추론 벤치마크"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 워밍업
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)

    # 벤치마크
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

# 결과 비교
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
    print(f"  시간: {result['avg_time']:.2f}s")
    print(f"  메모리: {result['memory_gb']:.2f} GB")
```

### 정확도 평가

```python
from datasets import load_dataset
import evaluate

# 평가 데이터셋
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def compute_perplexity(model, tokenizer, texts, max_length=1024):
    """퍼플렉시티 계산"""
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

# 비교
for name, model in models.items():
    ppl = compute_perplexity(model, tokenizer, dataset["text"][:100])
    print(f"{name} Perplexity: {ppl:.2f}")
```

---

## 8. 실전 가이드

### 양자화 방법 선택

```
┌─────────────────────────────────────────────────────────────┐
│                양자화 방법 선택 가이드                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  목적                    │  추천 방법                         │
│  ────────────────────────┼────────────────────────────────────│
│  빠른 프로토타이핑        │  bitsandbytes (load_in_8bit)      │
│  메모리 제한 환경         │  bitsandbytes (load_in_4bit)      │
│  프로덕션 배포            │  GPTQ 또는 AWQ                    │
│  파인튜닝 필요            │  QLoRA                            │
│  최대 속도                │  AWQ + fuse_layers                │
│  최대 품질                │  GPTQ (desc_act=True)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 문제 해결

```python
# 1. CUDA Out of Memory
# - 배치 크기 줄이기
# - gradient_checkpointing 활성화
# - 더 낮은 비트 양자화 사용

model.gradient_checkpointing_enable()

# 2. 양자화 후 품질 저하
# - group_size 줄이기 (64 또는 32)
# - 캘리브레이션 데이터 늘리기
# - AWQ 대신 GPTQ 시도

# 3. 추론 속도 느림
# - fuse_layers=True 활성화
# - exllama 백엔드 사용 (GPTQ)
# - 배치 처리 활용

from auto_gptq import exllama_set_max_input_length
exllama_set_max_input_length(model, 4096)
```

---

## 정리

### 양자화 비교표

| 방법 | 비트 | 속도 | 품질 | 사용 난이도 |
|------|------|------|------|-------------|
| FP16 | 16 | 기준 | 기준 | 쉬움 |
| INT8 (bitsandbytes) | 8 | 빠름 | 높음 | 쉬움 |
| INT4 (NF4) | 4 | 빠름 | 좋음 | 쉬움 |
| GPTQ | 4/3/2 | 매우 빠름 | 좋음 | 보통 |
| AWQ | 4 | 매우 빠름 | 좋음 | 보통 |
| QLoRA | 4 | - | 학습용 | 보통 |

### 핵심 코드

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

## 다음 단계

[14_RLHF_Alignment.md](./14_RLHF_Alignment.md)에서 LLM 정렬 기법(RLHF, DPO)을 학습합니다.
