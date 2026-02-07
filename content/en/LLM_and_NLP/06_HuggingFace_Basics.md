# 06. HuggingFace Basics

## Learning Objectives

- Understanding the Transformers library
- Using Pipeline API
- Loading tokenizers and models
- Performing various tasks

---

## 1. HuggingFace Ecosystem

### Main Components

```
HuggingFace
├── Transformers   # Model library
├── Datasets       # Datasets
├── Tokenizers     # Tokenizers
├── Hub            # Model/data repository
├── Accelerate     # Distributed training
└── Evaluate       # Evaluation metrics
```

### Installation

```bash
pip install transformers datasets tokenizers accelerate evaluate
```

---

## 2. Pipeline API

### Simplest Usage

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch processing
results = classifier([
    "I love this movie!",
    "This is terrible."
])
```

### Supported Tasks

| Task | Pipeline Name | Description |
|------|--------------|-------------|
| Sentiment Analysis | sentiment-analysis | Positive/negative classification |
| Text Classification | text-classification | General classification |
| NER | ner | Named entity recognition |
| QA | question-answering | Question answering |
| Summarization | summarization | Text summarization |
| Translation | translation | Language translation |
| Text Generation | text-generation | Sentence generation |
| Fill-Mask | fill-mask | Mask prediction |
| Zero-shot | zero-shot-classification | Classification without labels |

### Various Pipeline Examples

```python
# Question answering
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France."
)
# {'answer': 'Paris', 'score': 0.99, 'start': 0, 'end': 5}

# Summarization
summarizer = pipeline("summarization")
text = "Very long article text here..."
summary = summarizer(text, max_length=50, min_length=10)

# Translation
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# NER
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is John and I work at Google in New York")
# [{'entity_group': 'PER', 'word': 'John', ...},
#  {'entity_group': 'ORG', 'word': 'Google', ...},
#  {'entity_group': 'LOC', 'word': 'New York', ...}]

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "I want to go to the beach",
    candidate_labels=["travel", "cooking", "technology"]
)
# {'labels': ['travel', 'cooking', 'technology'], 'scores': [0.95, 0.03, 0.02]}
```

### Specifying Models

```python
# Korean model
classifier = pipeline(
    "sentiment-analysis",
    model="beomi/kcbert-base"
)

# Multilingual model
qa = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2"
)
```

---

## 3. Tokenizers

### AutoTokenizer

```python
from transformers import AutoTokenizer

# Automatically load appropriate tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encoding
text = "Hello, how are you?"
encoded = tokenizer(text)
print(encoded)
# {'input_ids': [101, 7592, ...], 'attention_mask': [1, 1, ...], ...}

# Return as tensors
encoded = tokenizer(text, return_tensors='pt')
```

### Key Parameters

```python
encoded = tokenizer(
    text,
    padding=True,              # Add padding
    truncation=True,           # Truncate to max length
    max_length=128,            # Maximum length
    return_tensors='pt',       # PyTorch tensors
    return_attention_mask=True,
    return_token_type_ids=True
)
```

### Batch Encoding

```python
texts = ["Hello world", "How are you?", "I'm fine"]

# Dynamic padding
encoded = tokenizer(
    texts,
    padding=True,     # Pad to longest sequence
    truncation=True,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)  # (3, max_len)
```

### Decoding

```python
# Decoding
decoded = tokenizer.decode(encoded['input_ids'][0])
print(decoded)  # "[CLS] hello world [SEP]"

# Remove special tokens
decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print(decoded)  # "hello world"
```

### Token Inspection

```python
# Token list
tokens = tokenizer.tokenize("Hello, how are you?")
print(tokens)  # ['hello', ',', 'how', 'are', 'you', '?']

# Tokens → IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

# IDs → Tokens
tokens = tokenizer.convert_ids_to_tokens(ids)
```

---

## 4. Model Loading

### AutoModel

```python
from transformers import AutoModel, AutoModelForSequenceClassification

# Base model (output: hidden states)
model = AutoModel.from_pretrained("bert-base-uncased")

# Classification model (output: logits)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

### Task-specific AutoModels

```python
from transformers import (
    AutoModelForSequenceClassification,  # Sequence classification
    AutoModelForTokenClassification,      # Token classification (NER)
    AutoModelForQuestionAnswering,        # QA
    AutoModelForCausalLM,                 # GPT-style generation
    AutoModelForSeq2SeqLM,                # Encoder-decoder (translation, summarization)
    AutoModelForMaskedLM                  # BERT-style MLM
)
```

### Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Encoding
inputs = tokenizer("I love this movie!", return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Prediction
predictions = torch.softmax(logits, dim=-1)
predicted_class = predictions.argmax().item()
print(f"Class: {predicted_class}, Confidence: {predictions[0][predicted_class]:.4f}")
```

---

## 5. Datasets Library

### Loading Datasets

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# Specify split
train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test[:1000]")  # First 1000

# Check sample
print(train_data[0])
# {'text': '...', 'label': 1}
```

### Data Preprocessing

```python
def preprocess(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

# Apply map
tokenized_dataset = dataset.map(preprocess, batched=True)

# Remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# Set PyTorch format
tokenized_dataset.set_format('torch')
```

### Creating DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    tokenized_dataset['train'],
    batch_size=16,
    shuffle=True
)

for batch in train_loader:
    print(batch['input_ids'].shape)  # (16, 256)
    break
```

---

## 6. Trainer API

### Basic Training

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Data
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
)

# Training
trainer.train()

# Evaluation
results = trainer.evaluate()
print(results)
```

### Custom Metrics

```python
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics
)
```

---

## 7. Model Saving/Loading

### Local Save

```python
# Save
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

### Upload to Hub

```python
# Login
from huggingface_hub import login
login(token="your_token")

# Upload
model.push_to_hub("my-username/my-model")
tokenizer.push_to_hub("my-username/my-model")

# Or with Trainer
trainer.push_to_hub("my-model")
```

---

## 8. Practical Example: Sentiment Classification

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# 1. Load data
dataset = load_dataset("imdb")

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 4. Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. Training configuration
args = TrainingArguments(
    output_dir="./imdb_classifier",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Mixed Precision
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
)

# 7. Training
trainer.train()

# 8. Inference
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return "Positive" if probs[0][1] > 0.5 else "Negative", probs[0][1].item()

print(predict("This movie was amazing!"))
# ('Positive', 0.9876)
```

---

## Summary

### Key Classes

| Class | Purpose |
|-------|---------|
| pipeline | Quick inference |
| AutoTokenizer | Automatic tokenizer loading |
| AutoModel* | Automatic model loading |
| Trainer | Training loop automation |
| TrainingArguments | Training configuration |

### Key Code

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Quick inference
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# Custom inference
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs)
```

---

## Next Steps

Learn fine-tuning techniques for various tasks in [07_Fine_Tuning.md](./07_Fine_Tuning.md).
