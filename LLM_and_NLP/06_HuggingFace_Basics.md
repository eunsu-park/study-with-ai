# 06. HuggingFace 기초

## 학습 목표

- Transformers 라이브러리 이해
- Pipeline API 사용
- 토크나이저와 모델 로드
- 다양한 태스크 수행

---

## 1. HuggingFace 생태계

### 주요 구성요소

```
HuggingFace
├── Transformers   # 모델 라이브러리
├── Datasets       # 데이터셋
├── Tokenizers     # 토크나이저
├── Hub            # 모델/데이터 저장소
├── Accelerate     # 분산 학습
└── Evaluate       # 평가 메트릭
```

### 설치

```bash
pip install transformers datasets tokenizers accelerate evaluate
```

---

## 2. Pipeline API

### 가장 간단한 사용법

```python
from transformers import pipeline

# 감성 분석
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 배치 처리
results = classifier([
    "I love this movie!",
    "This is terrible."
])
```

### 지원 태스크

| 태스크 | Pipeline 이름 | 설명 |
|--------|--------------|------|
| 감성 분석 | sentiment-analysis | 긍정/부정 분류 |
| 텍스트 분류 | text-classification | 일반 분류 |
| NER | ner | 개체명 인식 |
| QA | question-answering | 질의응답 |
| 요약 | summarization | 텍스트 요약 |
| 번역 | translation | 언어 번역 |
| 텍스트 생성 | text-generation | 문장 생성 |
| Fill-Mask | fill-mask | 마스크 예측 |
| Zero-shot | zero-shot-classification | 레이블 없는 분류 |

### 다양한 Pipeline 예제

```python
# 질의응답
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France."
)
# {'answer': 'Paris', 'score': 0.99, 'start': 0, 'end': 5}

# 요약
summarizer = pipeline("summarization")
text = "Very long article text here..."
summary = summarizer(text, max_length=50, min_length=10)

# 번역
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]

# 텍스트 생성
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# NER
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is John and I work at Google in New York")
# [{'entity_group': 'PER', 'word': 'John', ...},
#  {'entity_group': 'ORG', 'word': 'Google', ...},
#  {'entity_group': 'LOC', 'word': 'New York', ...}]

# Zero-shot 분류
classifier = pipeline("zero-shot-classification")
result = classifier(
    "I want to go to the beach",
    candidate_labels=["travel", "cooking", "technology"]
)
# {'labels': ['travel', 'cooking', 'technology'], 'scores': [0.95, 0.03, 0.02]}
```

### 특정 모델 지정

```python
# 한국어 모델
classifier = pipeline(
    "sentiment-analysis",
    model="beomi/kcbert-base"
)

# 다국어 모델
qa = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2"
)
```

---

## 3. 토크나이저

### AutoTokenizer

```python
from transformers import AutoTokenizer

# 자동으로 적합한 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 인코딩
text = "Hello, how are you?"
encoded = tokenizer(text)
print(encoded)
# {'input_ids': [101, 7592, ...], 'attention_mask': [1, 1, ...], ...}

# 텐서로 반환
encoded = tokenizer(text, return_tensors='pt')
```

### 주요 파라미터

```python
encoded = tokenizer(
    text,
    padding=True,              # 패딩 추가
    truncation=True,           # 최대 길이 자르기
    max_length=128,            # 최대 길이
    return_tensors='pt',       # PyTorch 텐서
    return_attention_mask=True,
    return_token_type_ids=True
)
```

### 배치 인코딩

```python
texts = ["Hello world", "How are you?", "I'm fine"]

# 동적 패딩
encoded = tokenizer(
    texts,
    padding=True,     # 가장 긴 시퀀스에 맞춤
    truncation=True,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)  # (3, max_len)
```

### 디코딩

```python
# 디코딩
decoded = tokenizer.decode(encoded['input_ids'][0])
print(decoded)  # "[CLS] hello world [SEP]"

# 특수 토큰 제거
decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print(decoded)  # "hello world"
```

### 토큰 확인

```python
# 토큰 목록
tokens = tokenizer.tokenize("Hello, how are you?")
print(tokens)  # ['hello', ',', 'how', 'are', 'you', '?']

# 토큰 → ID
ids = tokenizer.convert_tokens_to_ids(tokens)

# ID → 토큰
tokens = tokenizer.convert_ids_to_tokens(ids)
```

---

## 4. 모델 로드

### AutoModel

```python
from transformers import AutoModel, AutoModelForSequenceClassification

# 기본 모델 (출력: 은닉 상태)
model = AutoModel.from_pretrained("bert-base-uncased")

# 분류 모델 (출력: 로짓)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

### 태스크별 AutoModel

```python
from transformers import (
    AutoModelForSequenceClassification,  # 문장 분류
    AutoModelForTokenClassification,      # 토큰 분류 (NER)
    AutoModelForQuestionAnswering,        # QA
    AutoModelForCausalLM,                 # GPT 스타일 생성
    AutoModelForSeq2SeqLM,                # 인코더-디코더 (번역, 요약)
    AutoModelForMaskedLM                  # BERT 스타일 MLM
)
```

### 추론

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 인코딩
inputs = tokenizer("I love this movie!", return_tensors="pt")

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 예측
predictions = torch.softmax(logits, dim=-1)
predicted_class = predictions.argmax().item()
print(f"Class: {predicted_class}, Confidence: {predictions[0][predicted_class]:.4f}")
```

---

## 5. Datasets 라이브러리

### 데이터셋 로드

```python
from datasets import load_dataset

# HuggingFace Hub에서 로드
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# 분할 지정
train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test[:1000]")  # 처음 1000개

# 샘플 확인
print(train_data[0])
# {'text': '...', 'label': 1}
```

### 데이터 전처리

```python
def preprocess(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

# map 적용
tokenized_dataset = dataset.map(preprocess, batched=True)

# 불필요한 컬럼 제거
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# PyTorch 포맷 설정
tokenized_dataset.set_format('torch')
```

### DataLoader 생성

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

### 기본 학습

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 데이터
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 학습 설정
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

# 학습
trainer.train()

# 평가
results = trainer.evaluate()
print(results)
```

### 커스텀 메트릭

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

## 7. 모델 저장/로드

### 로컬 저장

```python
# 저장
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 로드
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

### Hub에 업로드

```python
# 로그인
from huggingface_hub import login
login(token="your_token")

# 업로드
model.push_to_hub("my-username/my-model")
tokenizer.push_to_hub("my-username/my-model")

# 또는 Trainer로
trainer.push_to_hub("my-model")
```

---

## 8. 실전 예제: 감성 분류

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

# 1. 데이터 로드
dataset = load_dataset("imdb")

# 2. 토크나이저
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. 모델
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 4. 메트릭
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. 학습 설정
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

# 7. 학습
trainer.train()

# 8. 추론
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

## 정리

### 핵심 클래스

| 클래스 | 용도 |
|--------|------|
| pipeline | 빠른 추론 |
| AutoTokenizer | 토크나이저 자동 로드 |
| AutoModel* | 모델 자동 로드 |
| Trainer | 학습 루프 자동화 |
| TrainingArguments | 학습 설정 |

### 핵심 코드

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 빠른 추론
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# 커스텀 추론
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs)
```

---

## 다음 단계

[07_Fine_Tuning.md](./07_Fine_Tuning.md)에서 다양한 태스크에 대한 파인튜닝 기법을 학습합니다.
