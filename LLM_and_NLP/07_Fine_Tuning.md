# 07. 파인튜닝

## 학습 목표

- 파인튜닝 전략 이해
- 다양한 태스크 파인튜닝
- 효율적인 파인튜닝 기법 (LoRA, QLoRA)
- 실전 파인튜닝 파이프라인

---

## 1. 파인튜닝 개요

### 전이학습 패러다임

```
사전학습 (Pre-training)
    │  대규모 텍스트로 일반적인 언어 이해 학습
    ▼
파인튜닝 (Fine-tuning)
    │  특정 태스크 데이터로 모델 조정
    ▼
태스크 수행
```

### 파인튜닝 전략

| 전략 | 설명 | 사용 시점 |
|------|------|----------|
| Full Fine-tuning | 전체 파라미터 업데이트 | 충분한 데이터, 컴퓨팅 |
| Feature Extraction | 분류기만 학습 | 적은 데이터 |
| LoRA | 저랭크 어댑터 | 효율적인 학습 |
| Prompt Tuning | 프롬프트만 학습 | 매우 적은 데이터 |

---

## 2. 텍스트 분류 파인튜닝

### 기본 파이프라인

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# 데이터 로드
dataset = load_dataset("imdb")

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

tokenized = dataset.map(tokenize, batched=True)

# 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 학습 설정
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
)

trainer.train()
```

### 다중 레이블 분류

```python
from transformers import AutoModelForSequenceClassification
import torch

# 다중 레이블용 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)

# 손실 함수 자동으로 BCEWithLogitsLoss 사용

# 레이블 형식: [1, 0, 1, 0, 1] (다중 레이블)
```

---

## 3. 토큰 분류 (NER) 파인튜닝

### NER 데이터 형식

```python
from datasets import load_dataset

# CoNLL-2003 NER 데이터셋
dataset = load_dataset("conll2003")

# 샘플
print(dataset['train'][0])
# {'tokens': ['EU', 'rejects', 'German', 'call', ...],
#  'ner_tags': [3, 0, 7, 0, ...]}

# 레이블
label_names = dataset['train'].features['ner_tags'].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### 토큰 정렬

```python
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True  # 이미 토큰화된 입력
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 특수 토큰
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # 첫 토큰
            else:
                label_ids.append(-100)  # 서브워드 무시
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized['labels'] = labels
    return tokenized
```

### NER 파인튜닝

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_names)
)

# seqeval 메트릭
import evaluate
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # 실제 레이블만 추출
    true_predictions = []
    true_labels = []

    for pred, label in zip(predictions, labels):
        true_preds = []
        true_labs = []
        for p, l in zip(pred, label):
            if l != -100:
                true_preds.append(label_names[p])
                true_labs.append(label_names[l])
        true_predictions.append(true_preds)
        true_labels.append(true_labs)

    return seqeval.compute(predictions=true_predictions, references=true_labels)
```

---

## 4. 질의응답 (QA) 파인튜닝

### SQuAD 데이터

```python
dataset = load_dataset("squad")

print(dataset['train'][0])
# {'id': '...', 'title': 'University_of_Notre_Dame',
#  'context': 'Architecturally, the school has...',
#  'question': 'To whom did the Virgin Mary appear in 1858?',
#  'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}
```

### QA 전처리

```python
def prepare_train_features(examples):
    tokenized = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 토큰 위치 찾기
            token_start = 0
            token_end = 0
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    token_start = idx
                if start < end_char <= end:
                    token_end = idx
                    break

            tokenized["start_positions"].append(token_start)
            tokenized["end_positions"].append(token_end)

    return tokenized
```

### QA 모델

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# 출력: start_logits, end_logits
```

---

## 5. 효율적인 파인튜닝 (PEFT)

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정
lora_config = LoraConfig(
    r=8,                      # 랭크
    lora_alpha=32,            # 스케일링
    target_modules=["query", "value"],  # 적용 모듈
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# 모델에 LoRA 적용
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
```

### Prompt Tuning

```python
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the sentiment: "
)

model = get_peft_model(model, config)
```

---

## 6. 대화형 모델 파인튜닝

### Instruction Tuning 데이터 형식

```python
# Alpaca 형식
{
    "instruction": "Summarize the following text.",
    "input": "Long article text here...",
    "output": "Summary of the article."
}

# ChatML 형식
"""
<|system|>
You are a helpful assistant.
<|user|>
What is the capital of France?
<|assistant|>
The capital of France is Paris.
"""
```

### SFT (Supervised Fine-Tuning)

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
    ),
)

trainer.train()
```

### DPO (Direct Preference Optimization)

```python
from trl import DPOTrainer

# 선호도 데이터
# {'prompt': '...', 'chosen': '...', 'rejected': '...'}

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 기준 모델
    train_dataset=dataset,
    beta=0.1,
    args=TrainingArguments(...),
)

trainer.train()
```

---

## 7. 학습 최적화

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```

### Mixed Precision

```python
args = TrainingArguments(
    ...,
    fp16=True,  # 또는 bf16=True
)
```

### Gradient Accumulation

```python
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 실효 배치 = 4 * 8 = 32
)
```

### DeepSpeed

```python
args = TrainingArguments(
    ...,
    deepspeed="ds_config.json"
)

# ds_config.json
{
    "fp16": {"enabled": true},
    "zero_optimization": {"stage": 2}
}
```

---

## 8. 전체 파인튜닝 예제

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import evaluate

# 1. 데이터
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 2. 모델 + LoRA
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 학습 설정
args = TrainingArguments(
    output_dir="./lora_imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

# 4. 메트릭
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. 학습
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. 저장
model.save_pretrained("./lora_imdb_final")
```

---

## 정리

### 파인튜닝 선택 가이드

| 상황 | 추천 방법 |
|------|----------|
| 충분한 데이터 + GPU | Full Fine-tuning |
| 제한된 GPU 메모리 | LoRA / QLoRA |
| 매우 적은 데이터 | Prompt Tuning |
| LLM 정렬 | SFT + DPO/RLHF |

### 핵심 코드

```python
# LoRA
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, target_modules=["query", "value"])
model = get_peft_model(model, lora_config)

# Trainer
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
```

---

## 다음 단계

[08_Prompt_Engineering.md](./08_Prompt_Engineering.md)에서 효과적인 프롬프트 작성 기법을 학습합니다.
