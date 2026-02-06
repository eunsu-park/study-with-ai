# 04. BERT 이해

## 학습 목표

- BERT 아키텍처 이해
- 사전학습 목표 (MLM, NSP)
- 입력 표현
- 다양한 BERT 변형

---

## 1. BERT 개요

### Bidirectional Encoder Representations from Transformers

```
BERT = Transformer 인코더 스택

특징:
- 양방향 문맥 이해
- 사전학습 + 파인튜닝 패러다임
- 다양한 NLP 태스크에 범용 적용
```

### 모델 크기

| 모델 | 레이어 | d_model | 헤드 | 파라미터 |
|------|-------|---------|------|---------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

---

## 2. 입력 표현

### 세 가지 임베딩의 합

```
입력: [CLS] I love NLP [SEP] It is fun [SEP]

Token Embedding:    [E_CLS, E_I, E_love, E_NLP, E_SEP, E_It, E_is, E_fun, E_SEP]
Segment Embedding:  [E_A,   E_A, E_A,    E_A,   E_A,   E_B,  E_B,  E_B,   E_B  ]
Position Embedding: [E_0,   E_1, E_2,    E_3,   E_4,   E_5,  E_6,  E_7,   E_8  ]
                    ─────────────────────────────────────────────────────────────
                    = 최종 입력 임베딩 (합)
```

### 특수 토큰

| 토큰 | 역할 |
|------|------|
| [CLS] | 분류 태스크용 집계 토큰 |
| [SEP] | 문장 구분자 |
| [PAD] | 패딩 |
| [MASK] | MLM에서 마스킹된 토큰 |
| [UNK] | 미등록 단어 |

### 입력 구현

```python
import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=768, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)

        # 위치 인덱스
        position_ids = torch.arange(seq_len, device=input_ids.device)

        # 임베딩 합
        embeddings = (
            self.token_embedding(input_ids) +
            self.position_embedding(position_ids) +
            self.segment_embedding(segment_ids)
        )

        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
```

---

## 3. 사전학습 목표

### Masked Language Model (MLM)

```
15%의 토큰을 선택:
- 80%: [MASK]로 교체
- 10%: 랜덤 토큰으로 교체
- 10%: 그대로 유지

예시:
입력: "The cat sat on the mat"
     → "The [MASK] sat on the mat"
목표: [MASK] → "cat" 예측
```

```python
import random

def create_mlm_data(tokens, vocab, mask_prob=0.15):
    """MLM 학습 데이터 생성"""
    labels = [-100] * len(tokens)  # -100은 손실 계산에서 무시

    for i, token in enumerate(tokens):
        if random.random() < mask_prob:
            labels[i] = vocab[token]  # 원래 토큰 ID

            rand = random.random()
            if rand < 0.8:
                tokens[i] = '[MASK]'
            elif rand < 0.9:
                tokens[i] = random.choice(list(vocab.keys()))
            # else: 그대로 유지

    return tokens, labels
```

### Next Sentence Prediction (NSP)

```
입력: [CLS] 문장A [SEP] 문장B [SEP]
목표: 문장B가 문장A의 실제 다음 문장인지 이진 분류

예시:
긍정 (IsNext):
    A: "The man went to the store"
    B: "He bought a gallon of milk"

부정 (NotNext):
    A: "The man went to the store"
    B: "Penguins are flightless birds"
```

```python
class BERTPreTrainingHeads(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # MLM 헤드
        self.mlm = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        # NSP 헤드
        self.nsp = nn.Linear(d_model, 2)

    def forward(self, sequence_output, cls_output):
        mlm_scores = self.mlm(sequence_output)  # (batch, seq, vocab)
        nsp_scores = self.nsp(cls_output)       # (batch, 2)
        return mlm_scores, nsp_scores
```

---

## 4. BERT 전체 구조

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()

        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        # 임베딩
        x = self.embedding(input_ids, segment_ids)

        # 패딩 마스크 변환
        if attention_mask is not None:
            # (batch, seq) → (batch, seq) with True for padding
            attention_mask = (attention_mask == 0)

        # 인코더
        output = self.encoder(x, src_key_padding_mask=attention_mask)

        return output  # (batch, seq, d_model)


class BERTForPreTraining(nn.Module):
    def __init__(self, vocab_size, d_model=768, **kwargs):
        super().__init__()
        self.bert = BERT(vocab_size, d_model, **kwargs)
        self.heads = BERTPreTrainingHeads(d_model, vocab_size)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        sequence_output = self.bert(input_ids, segment_ids, attention_mask)
        cls_output = sequence_output[:, 0]  # [CLS] 토큰

        mlm_scores, nsp_scores = self.heads(sequence_output, cls_output)
        return mlm_scores, nsp_scores
```

---

## 5. 파인튜닝 패턴

### 문장 분류 (Single Sentence)

```python
class BERTForSequenceClassification(nn.Module):
    def __init__(self, bert, num_classes, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.embedding.token_embedding.embedding_dim,
                                    num_classes)

    def forward(self, input_ids, segment_ids, attention_mask):
        output = self.bert(input_ids, segment_ids, attention_mask)
        cls_output = output[:, 0]  # [CLS]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)
```

### 토큰 분류 (NER)

```python
class BERTForTokenClassification(nn.Module):
    def __init__(self, bert, num_labels, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.embedding.token_embedding.embedding_dim,
                                    num_labels)

    def forward(self, input_ids, segment_ids, attention_mask):
        output = self.bert(input_ids, segment_ids, attention_mask)
        output = self.dropout(output)
        return self.classifier(output)  # (batch, seq, num_labels)
```

### 질의응답 (QA)

```python
class BERTForQuestionAnswering(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        hidden_size = bert.embedding.token_embedding.embedding_dim
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start, end

    def forward(self, input_ids, segment_ids, attention_mask):
        output = self.bert(input_ids, segment_ids, attention_mask)
        logits = self.qa_outputs(output)  # (batch, seq, 2)

        start_logits = logits[:, :, 0]  # (batch, seq)
        end_logits = logits[:, :, 1]

        return start_logits, end_logits
```

---

## 6. BERT 변형 모델

### RoBERTa

```
변경점:
- NSP 제거 (MLM만 사용)
- 동적 마스킹 (매 에포크 다른 마스킹)
- 더 큰 배치, 더 긴 학습
- Byte-Level BPE 토크나이저

결과: BERT보다 성능 향상
```

### ALBERT

```
변경점:
- 임베딩 분해 (V×E, E×H → V×E, E<<H)
- 레이어 파라미터 공유
- NSP → SOP (Sentence Order Prediction)

결과: 파라미터 대폭 감소, 유사 성능
```

### DistilBERT

```
변경점:
- 지식 증류 (Teacher: BERT → Student: 작은 모델)
- 6 레이어 (BERT의 절반)

결과: 40% 작음, 60% 빠름, 97% 성능 유지
```

### Comparison

| 모델 | 레이어 | 파라미터 | 속도 | 특징 |
|------|-------|---------|------|------|
| BERT-base | 12 | 110M | 1x | 기준 |
| RoBERTa | 12 | 125M | 1x | 최적화된 학습 |
| ALBERT-base | 12 | 12M | 1x | 파라미터 공유 |
| DistilBERT | 6 | 66M | 2x | 지식 증류 |

---

## 7. HuggingFace BERT 사용

### 기본 사용

```python
from transformers import BertTokenizer, BertModel

# 토크나이저와 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 인코딩
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# 순전파
outputs = model(**inputs)

# 출력
last_hidden_state = outputs.last_hidden_state  # (1, seq, 768)
pooler_output = outputs.pooler_output          # (1, 768) - [CLS] 변환
```

### 분류 모델

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

inputs = tokenizer("I love this movie!", return_tensors='pt')
outputs = model(**inputs)
logits = outputs.logits  # (1, 2)
```

### Attention 시각화

```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

inputs = tokenizer("The cat sat on the mat", return_tensors='pt')
outputs = model(**inputs)

# Attention weights: (num_layers, batch, heads, seq, seq)
attentions = outputs.attentions

# 첫 번째 레이어, 첫 번째 헤드
attn = attentions[0][0, 0].detach().numpy()
```

---

## 8. BERT 입력 포맷

### Single Sentence

```
[CLS] sentence [SEP]
segment_ids: [0, 0, 0, ..., 0]
```

### Sentence Pair

```
[CLS] sentence A [SEP] sentence B [SEP]
segment_ids: [0, 0, ..., 0, 1, 1, ..., 1]
```

### HuggingFace에서 Pair 처리

```python
# 두 문장 입력
text_a = "How old are you?"
text_b = "I am 25 years old."

inputs = tokenizer(
    text_a, text_b,
    padding='max_length',
    max_length=32,
    truncation=True,
    return_tensors='pt'
)

print(inputs['token_type_ids'])  # segment_ids
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...]
```

---

## 정리

### 핵심 개념

1. **양방향 인코더**: 전체 문맥을 양방향으로 이해
2. **MLM**: 마스킹된 토큰 예측으로 문맥 학습
3. **NSP**: 문장 관계 이해 (RoBERTa에서 제거)
4. **[CLS] 토큰**: 문장 수준 표현
5. **Segment Embedding**: 문장 구분

### 핵심 코드

```python
# HuggingFace BERT
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 인코딩
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]
```

---

## 다음 단계

[05_GPT_Understanding.md](./05_GPT_Understanding.md)에서 GPT 모델과 자기회귀 언어 모델을 학습합니다.
