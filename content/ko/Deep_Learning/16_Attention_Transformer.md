# 10. Attention과 Transformer

## 학습 목표

- Attention 메커니즘의 원리
- Self-Attention 이해
- Transformer 아키텍처
- PyTorch 구현

---

## 1. Attention의 필요성

### Seq2Seq의 한계

```
인코더: "나는 학교에 간다" → 고정 크기 벡터
                              ↓
디코더: 고정 벡터 → "I go to school"

문제: 긴 문장의 정보가 압축되어 손실
```

### Attention의 해결

```
디코더가 각 출력 단어를 생성할 때
인코더의 모든 단어를 "주목"할 수 있음

"I" 생성 시 → "나는"에 높은 attention
"school" 생성 시 → "학교"에 높은 attention
```

---

## 2. Attention 메커니즘

### 수식

```python
# Query, Key, Value
Q = 현재 디코더 상태
K = 인코더 모든 상태
V = 인코더 모든 상태 (보통 K와 동일)

# Attention Score
score = Q @ K.T  # (query_len, key_len)

# Attention Weight (softmax)
weight = softmax(score / sqrt(d_k))  # 스케일링

# Context
context = weight @ V  # 가중 합
```

### Scaled Dot-Product Attention

```python
def attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

---

## 3. Self-Attention

### 개념

```
같은 시퀀스 내에서 각 단어가 다른 모든 단어에 attention

"The cat sat on the mat because it was tired"
"it"이 "cat"에 높은 attention → 대명사 해석
```

### 수식

```python
# 입력 X에서 Q, K, V 생성
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Self-Attention
output = attention(Q, K, V)
```

---

## 4. Multi-Head Attention

### 아이디어

```
여러 개의 attention head가 서로 다른 관계 학습

Head 1: 문법적 관계
Head 2: 의미적 관계
Head 3: 위치 관계
...
```

### 수식

```python
def multi_head_attention(Q, K, V, num_heads):
    d_model = Q.size(-1)
    d_k = d_model // num_heads

    # 헤드 분할
    Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
    K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)

    # 각 헤드에서 attention
    attn_output, _ = attention(Q, K, V)

    # 헤드 결합
    output = attn_output.transpose(1, 2).contiguous().view(batch, seq, d_model)
    return output
```

---

## 5. Transformer 아키텍처

### 구조

```
입력 → Embedding → Positional Encoding
                      ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│           ↓                         │
│  Add & LayerNorm                    │
│           ↓                         │
│  Feed Forward Network               │
│           ↓                         │
│  Add & LayerNorm                    │
└─────────────────────────────────────┘
            × N layers
                ↓
             출력
```

### 핵심 컴포넌트

1. **Multi-Head Attention**
2. **Position-wise Feed Forward**
3. **Residual Connection**
4. **Layer Normalization**
5. **Positional Encoding**

---

## 6. Positional Encoding

### 필요성

```
Attention은 순서 정보가 없음
→ 위치 정보를 명시적으로 추가
```

### Sinusoidal Encoding

```python
def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE
```

---

## 7. PyTorch Transformer

### 기본 사용

```python
import torch.nn as nn

# Transformer 인코더
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# 순전파
x = torch.randn(10, 32, 512)  # (seq, batch, d_model)
output = encoder(x)
```

### 분류 모델

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 평균 풀링
        return self.fc(x)
```

---

## 8. Vision Transformer (ViT)

### 아이디어

```
이미지를 패치로 분할 → 시퀀스로 처리

이미지 (224×224) → 16×16 패치 196개 → Transformer
```

### 구조

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model, nhead, num_layers):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 패치 추출 및 임베딩
        patches = extract_patches(x)
        x = self.patch_embed(patches)

        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 위치 임베딩
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x.transpose(0, 1))

        # 분류 (CLS 토큰 사용)
        return self.fc(x[0])
```

---

## 9. Attention vs RNN 비교

| 항목 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 병렬화 | 어려움 | 용이 |
| 장거리 의존성 | 어려움 | 용이 |
| 학습 속도 | 느림 | 빠름 |
| 메모리 | O(n) | O(n²) |
| 위치 정보 | 암시적 | 명시적 |

---

## 10. 실전 활용

### NLP

- BERT: 양방향 인코더
- GPT: 디코더 기반 생성
- T5: 인코더-디코더

### Vision

- ViT: 이미지 분류
- DETR: 객체 검출
- Swin Transformer: 계층적 구조

---

## 정리

### 핵심 개념

1. **Attention**: Query-Key-Value로 관련성 계산
2. **Self-Attention**: 시퀀스 내 모든 위치 참조
3. **Multi-Head**: 다양한 관계 동시 학습
4. **Positional Encoding**: 순서 정보 추가

### 핵심 코드

```python
# Scaled Dot-Product Attention
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V

# PyTorch Transformer
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
)
```

---

## 다음 단계

[23_Training_Optimization.md](./23_Training_Optimization.md)에서 고급 학습 기법을 배웁니다.
