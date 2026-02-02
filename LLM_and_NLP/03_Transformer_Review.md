# 03. Transformer 복습

## 학습 목표

- NLP 관점에서 Transformer 이해
- Encoder와 Decoder 구조
- 언어 모델링 관점의 Attention
- BERT/GPT 기반 구조 이해

---

## 1. Transformer 개요

### 구조 요약

```
인코더 (BERT 스타일):
    입력 → [Embedding + Positional] → [Self-Attention + FFN] × N → 출력

디코더 (GPT 스타일):
    입력 → [Embedding + Positional] → [Masked Self-Attention + FFN] × N → 출력

인코더-디코더 (T5 스타일):
    입력 → 인코더 → [Cross-Attention] → 디코더 → 출력
```

### NLP에서의 역할

| 모델 | 구조 | 용도 |
|------|------|------|
| BERT | 인코더 only | 분류, QA, NER |
| GPT | 디코더 only | 텍스트 생성 |
| T5, BART | 인코더-디코더 | 번역, 요약 |

---

## 2. Self-Attention (NLP 관점)

### 문장 내 관계 학습

```
"The cat sat on the mat because it was tired"

"it" → Attention → "cat" (높은 가중치)
                → "mat" (낮은 가중치)

모델이 대명사 "it"이 "cat"을 지칭함을 학습
```

### Query, Key, Value

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V 계산
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Multi-head 분할
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq, d_k)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # 헤드 결합
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attention_weights
```

---

## 3. Causal Masking (GPT 스타일)

### 자기회귀 언어 모델

```
"I love NLP" 학습:
    입력: [I]         → 예측: love
    입력: [I, love]   → 예측: NLP
    입력: [I, love, NLP] → 예측: <eos>

미래 토큰을 보면 안 됨 → Causal Mask 필요
```

### Causal Mask 구현

```python
def create_causal_mask(seq_len):
    """하삼각 마스크 생성 (미래 토큰 차단)"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 = 참조 가능, 0 = 마스킹

# 예시 (seq_len=4)
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=512):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        # 미리 계산된 마스크 등록
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask)

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.mask[:seq_len, :seq_len]
        return self.attention(x, mask)
```

---

## 4. Encoder vs Decoder

### 인코더 (양방향)

```python
class TransformerEncoderBlock(nn.Module):
    """BERT 스타일 인코더 블록"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # Self-Attention (양방향)
        attn_out, _ = self.self_attn(x, padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

### 디코더 (단방향)

```python
class TransformerDecoderBlock(nn.Module):
    """GPT 스타일 디코더 블록"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Masked Self-Attention (단방향)
        attn_out, _ = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

---

## 5. Positional Encoding

### Sinusoidal (원본 Transformer)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Learnable (BERT, GPT)

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
```

---

## 6. Complete Transformer Model

### GPT-스타일 언어 모델

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 토큰 + 위치 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Decoder 블록
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (선택)
        self.head.weight = self.token_embedding.weight

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape

        # 임베딩
        tok_emb = self.token_embedding(x)
        pos = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        # Transformer 블록
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab_size)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """자기회귀 텍스트 생성"""
        for _ in range(max_new_tokens):
            # 마지막 위치의 logits
            logits = self(idx)[:, -1, :]  # (batch, vocab)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
```

### BERT-스타일 인코더

```python
class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # 문장 구분

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # 임베딩 결합
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        seg_emb = self.segment_embedding(segment_ids)

        x = tok_emb + pos_emb + seg_emb

        # Transformer 블록
        for block in self.blocks:
            x = block(x, attention_mask)

        return self.ln_f(x)
```

---

## 7. 학습 목표별 비교

### Masked Language Modeling (BERT)

```
입력: "The [MASK] sat on the mat"
예측: [MASK] → "cat"

15% 토큰을 마스킹하여 예측
양방향 문맥 활용
```

### Causal Language Modeling (GPT)

```
입력: "The cat sat on"
예측: "the" "cat" "sat" "on" "the" "mat"

다음 토큰 예측
단방향 (왼쪽→오른쪽)
```

### Seq2Seq (T5, BART)

```
입력: "translate English to French: Hello"
출력: "Bonjour"

인코더: 입력 이해
디코더: 출력 생성
```

---

## 8. PyTorch 내장 Transformer

```python
import torch.nn as nn

# 인코더
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# 디코더
decoder_layer = nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

# 사용
x = torch.randn(32, 100, 512)  # (batch, seq, d_model)
encoded = encoder(x)
decoded = decoder(x, encoded)
```

---

## 정리

### 모델 비교

| 항목 | BERT (인코더) | GPT (디코더) | T5 (Enc-Dec) |
|------|--------------|-------------|--------------|
| Attention | 양방향 | 단방향 (Causal) | 양방향 + 단방향 |
| 학습 | MLM + NSP | 다음 토큰 예측 | Denoising |
| 출력 | 문맥 벡터 | 생성 | 생성 |
| 용도 | 분류, QA | 생성, 대화 | 번역, 요약 |

### 핵심 코드

```python
# Causal Mask
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, -1e9)

# Multi-Head Attention 분할
Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)

# Scaled Dot-Product
scores = Q @ K.T / sqrt(d_k)
attn = softmax(scores) @ V
```

---

## 다음 단계

[04_BERT_Understanding.md](./04_BERT_Understanding.md)에서 BERT의 구조와 학습 방법을 상세히 학습합니다.
