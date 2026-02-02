# 05. GPT 이해

## 학습 목표

- GPT 아키텍처 이해
- 자기회귀 언어 모델링
- 텍스트 생성 기법
- GPT 시리즈 발전

---

## 1. GPT 개요

### Generative Pre-trained Transformer

```
GPT = Transformer 디코더 스택

특징:
- 단방향 (왼쪽→오른쪽)
- 자기회귀 생성
- 다음 토큰 예측으로 학습
```

### BERT vs GPT

| 항목 | BERT | GPT |
|------|------|-----|
| 구조 | 인코더 | 디코더 |
| 방향 | 양방향 | 단방향 |
| 학습 | MLM | 다음 토큰 예측 |
| 용도 | 이해 (분류, QA) | 생성 (대화, 작문) |

---

## 2. 자기회귀 언어 모델링

### 학습 목표

```
P(x) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ...

문장: "I love NLP"
P("I") × P("love"|"I") × P("NLP"|"I love") × P("<eos>"|"I love NLP")

손실: -log P(다음 토큰 | 이전 토큰들)
```

### Causal Language Modeling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_lm_loss(logits, targets):
    """
    logits: (batch, seq, vocab_size)
    targets: (batch, seq) - 다음 토큰

    입력: [BOS, I, love, NLP]
    타겟: [I, love, NLP, EOS]
    """
    batch_size, seq_len, vocab_size = logits.shape

    # (batch*seq, vocab) vs (batch*seq,)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-100  # 패딩 무시
    )
    return loss
```

---

## 3. GPT 아키텍처

### 구조

```
입력 토큰
    ↓
Token Embedding + Position Embedding
    ↓
┌─────────────────────────────────┐
│  Masked Multi-Head Attention    │
│           ↓                     │
│  Add & LayerNorm                │
│           ↓                     │
│  Feed Forward                   │
│           ↓                     │
│  Add & LayerNorm                │
└─────────────────────────────────┘
            × N layers
    ↓
LayerNorm
    ↓
Linear (vocab_size)
    ↓
Softmax → 다음 토큰 확률
```

### 구현

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-LayerNorm (GPT-2 스타일)
        ln_x = self.ln1(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        ln_x = self.ln2(x)
        x = x + self.ffn(ln_x)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        # Causal mask 등록
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_len

        # 임베딩
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Causal mask
        mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer 블록
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab)

        return logits
```

---

## 4. 텍스트 생성

### Greedy Decoding

```python
def generate_greedy(model, input_ids, max_new_tokens):
    """항상 가장 확률 높은 토큰 선택"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Temperature Sampling

```python
def generate_with_temperature(model, input_ids, max_new_tokens, temperature=1.0):
    """Temperature로 분포 조절"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

# temperature < 1: 더 결정적 (높은 확률 토큰 선호)
# temperature > 1: 더 무작위 (다양성 증가)
```

### Top-k Sampling

```python
def generate_top_k(model, input_ids, max_new_tokens, k=50, temperature=1.0):
    """상위 k개 토큰에서만 샘플링"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature

            # Top-k 필터링
            top_k_logits, top_k_indices = logits.topk(k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            # 샘플링
            idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Top-p (Nucleus) Sampling

```python
def generate_top_p(model, input_ids, max_new_tokens, p=0.9, temperature=1.0):
    """누적 확률 p까지의 토큰에서 샘플링"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # 확률 내림차순 정렬
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            # p 이후 토큰 마스킹
            mask = cumsum - sorted_probs > p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # 샘플링
            idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

---

## 5. GPT 시리즈

### GPT-1 (2018)

```
- 12 레이어, 768 차원, 117M 파라미터
- BooksCorpus로 학습
- 파인튜닝 패러다임 도입
```

### GPT-2 (2019)

```
- 최대 48 레이어, 1.5B 파라미터
- WebText (40GB) 학습
- Zero-shot 능력 발견
- "Too dangerous to release"

크기 변형:
- Small: 117M (GPT-1과 동일)
- Medium: 345M
- Large: 762M
- XL: 1.5B
```

### GPT-3 (2020)

```
- 96 레이어, 175B 파라미터
- Few-shot / In-context Learning
- API로만 제공

주요 발견:
- 프롬프트만으로 다양한 태스크 수행
- 스케일링 법칙: 모델 크기 ↑ = 성능 ↑
```

### GPT-4 (2023)

```
- 멀티모달 (텍스트 + 이미지)
- 더 긴 컨텍스트 (8K, 32K, 128K)
- 향상된 추론 능력
- RLHF로 정렬
```

---

## 6. HuggingFace GPT-2

### 기본 사용

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 텍스트 생성
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 생성
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### 생성 파라미터

```python
output = model.generate(
    input_ids,
    max_length=100,           # 최대 길이
    min_length=10,            # 최소 길이
    do_sample=True,           # 샘플링 사용
    temperature=0.8,          # 온도
    top_k=50,                 # Top-k
    top_p=0.95,               # Top-p
    num_return_sequences=3,   # 생성 개수
    no_repeat_ngram_size=2,   # n-gram 반복 방지
    repetition_penalty=1.2,   # 반복 페널티
    pad_token_id=tokenizer.eos_token_id
)
```

### 조건부 생성

```python
# 프롬프트 기반 생성
prompt = """
Q: What is the capital of France?
A:"""

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=False  # Greedy
)
print(tokenizer.decode(output[0]))
```

---

## 7. In-Context Learning

### Zero-shot

```
프롬프트만으로 태스크 수행:

"Translate English to French:
Hello, how are you? →"
```

### Few-shot

```
예제를 프롬프트에 포함:

"Translate English to French:
Hello → Bonjour
Thank you → Merci
Good morning → Bonjour
How are you? →"
```

### Chain-of-Thought (CoT)

```
단계별 추론 유도:

"Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
How many balls does he have now?
A: Let's think step by step.
Roger started with 5 balls.
2 cans of 3 balls each = 6 balls.
5 + 6 = 11 balls.
The answer is 11."
```

---

## 8. KV Cache

### 효율적인 생성

```python
class GPTWithKVCache(nn.Module):
    def forward(self, input_ids, past_key_values=None):
        """
        past_key_values: 이전 토큰의 K, V 캐시
        새 토큰에 대해서만 계산 후 캐시 업데이트
        """
        if past_key_values is None:
            # 전체 시퀀스 계산
            ...
        else:
            # 마지막 토큰만 계산
            ...

        return logits, new_past_key_values

# 생성 시
past = None
for _ in range(max_new_tokens):
    logits, past = model(new_token, past_key_values=past)
    # O(n) 대신 O(1) 복잡도
```

### HuggingFace KV Cache

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True  # KV Cache 활성화 (기본값)
)
```

---

## 정리

### 생성 전략 비교

| 방법 | 장점 | 단점 | 용도 |
|------|------|------|------|
| Greedy | 빠름, 일관성 | 반복, 지루함 | 번역, QA |
| Temperature | 다양성 조절 | 튜닝 필요 | 일반 생성 |
| Top-k | 안정적 | 고정 k | 일반 생성 |
| Top-p | 적응적 | 약간 느림 | 창작, 대화 |

### 핵심 코드

```python
# HuggingFace GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 생성
output = model.generate(input_ids, max_length=50, do_sample=True,
                        temperature=0.7, top_p=0.9)
```

---

## 다음 단계

[06_HuggingFace_Basics.md](./06_HuggingFace_Basics.md)에서 HuggingFace Transformers 라이브러리를 학습합니다.
