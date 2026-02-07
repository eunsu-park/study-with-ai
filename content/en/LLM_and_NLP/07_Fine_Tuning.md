# 07. Fine-Tuning

## Learning Objectives

- Understanding fine-tuning strategies
- Fine-tuning for various tasks
- Efficient fine-tuning techniques (LoRA, QLoRA)
- Practical fine-tuning pipelines

---

## 1. Fine-Tuning Overview

### Transfer Learning Paradigm

```
Pre-training
    │  Learn general language understanding from large-scale text
    ▼
Fine-tuning
    │  Adapt model to specific task data
    ▼
Task Performance
```

### Fine-Tuning Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Full Fine-tuning | Update all parameters | Sufficient data, compute |
| Feature Extraction | Train classifier only | Limited data |
| LoRA | Low-rank adapters | Efficient training |
| Prompt Tuning | Train prompts only | Very limited data |

---

## 2. Text Classification Fine-Tuning

### Basic Pipeline

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# Load data
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

tokenized = dataset.map(tokenize, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training configuration
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

### Multi-Label Classification

```python
from transformers import AutoModelForSequenceClassification
import torch

# Model for multi-label
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)

# Automatically uses BCEWithLogitsLoss

# Label format: [1, 0, 1, 0, 1] (multi-label)
```

---

## 3. Token Classification (NER) Fine-Tuning

### NER Data Format

```python
from datasets import load_dataset

# CoNLL-2003 NER dataset
dataset = load_dataset("conll2003")

# Sample
print(dataset['train'][0])
# {'tokens': ['EU', 'rejects', 'German', 'call', ...],
#  'ner_tags': [3, 0, 7, 0, ...]}

# Labels
label_names = dataset['train'].features['ner_tags'].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### Token Alignment

```python
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True  # Already tokenized input
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # First token
            else:
                label_ids.append(-100)  # Ignore subwords
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized['labels'] = labels
    return tokenized
```

### NER Fine-Tuning

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_names)
)

# seqeval metric
import evaluate
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # Extract only actual labels
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

## 4. Question Answering (QA) Fine-Tuning

### SQuAD Data

```python
dataset = load_dataset("squad")

print(dataset['train'][0])
# {'id': '...', 'title': 'University_of_Notre_Dame',
#  'context': 'Architecturally, the school has...',
#  'question': 'To whom did the Virgin Mary appear in 1858?',
#  'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}
```

### QA Preprocessing

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

            # Find token positions
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

### QA Model

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Output: start_logits, end_logits
```

---

## 5. Efficient Fine-Tuning (PEFT)

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    r=8,                      # Rank
    lora_alpha=32,            # Scaling
    target_modules=["query", "value"],  # Target modules
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# Apply LoRA to model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
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

## 6. Conversational Model Fine-Tuning

### Instruction Tuning Data Format

```python
# Alpaca format
{
    "instruction": "Summarize the following text.",
    "input": "Long article text here...",
    "output": "Summary of the article."
}

# ChatML format
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

# Preference data
# {'prompt': '...', 'chosen': '...', 'rejected': '...'}

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # Reference model
    train_dataset=dataset,
    beta=0.1,
    args=TrainingArguments(...),
)

trainer.train()
```

---

## 7. Training Optimization

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```

### Mixed Precision

```python
args = TrainingArguments(
    ...,
    fp16=True,  # or bf16=True
)
```

### Gradient Accumulation

```python
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 4 * 8 = 32
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

## 8. Complete Fine-Tuning Example

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

# 1. Data
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 2. Model + LoRA
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

# 3. Training configuration
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

# 4. Metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. Training
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. Save
model.save_pretrained("./lora_imdb_final")
```

---

## Summary

### Fine-Tuning Selection Guide

| Situation | Recommended Method |
|-----------|-------------------|
| Sufficient data + GPU | Full Fine-tuning |
| Limited GPU memory | LoRA / QLoRA |
| Very limited data | Prompt Tuning |
| LLM alignment | SFT + DPO/RLHF |

### Key Code

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

## Next Steps

Learn effective prompt engineering techniques in [08_Prompt_Engineering.md](./08_Prompt_Engineering.md).
