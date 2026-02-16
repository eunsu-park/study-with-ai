# 09. LSTM과 GRU

## 학습 목표

- LSTM과 GRU의 구조 이해
- 게이트 메커니즘
- 장기 의존성 학습
- PyTorch 구현

---

## 1. LSTM (Long Short-Term Memory)

### 문제: RNN의 기울기 소실

```
h100 ← W × W × ... × W × h1
            ↑
    기울기가 0에 수렴
```

### 해결: 셀 상태 (Cell State)

```
LSTM = 셀 상태 (장기 기억) + 은닉 상태 (단기 기억)
```

### LSTM 구조

```
       ┌──────────────────────────────────────┐
       │            셀 상태 (C)                 │
       │     ×─────(+)─────────────────────►   │
       │     ↑      ↑                          │
       │    forget  input                      │
       │    gate    gate                       │
       │     ↑      ↑                          │
h(t-1)─┴──►[σ]   [σ][tanh]    [σ]──►×──────►h(t)
           f(t)   i(t) g(t)   o(t)     ↑
                              output gate
```

### 게이트 수식

```python
# Forget Gate: 이전 기억 중 얼마나 잊을지
f(t) = σ(W_f × [h(t-1), x(t)] + b_f)

# Input Gate: 새 정보 중 얼마나 저장할지
i(t) = σ(W_i × [h(t-1), x(t)] + b_i)

# Cell Candidate: 새로운 후보 정보
g(t) = tanh(W_g × [h(t-1), x(t)] + b_g)

# Cell State Update
C(t) = f(t) × C(t-1) + i(t) × g(t)

# Output Gate: 셀 상태 중 얼마나 출력할지
o(t) = σ(W_o × [h(t-1), x(t)] + b_o)

# Hidden State
h(t) = o(t) × tanh(C(t))
```

---

## 2. GRU (Gated Recurrent Unit)

### LSTM의 단순화 버전

```
GRU = Reset Gate + Update Gate
(셀 상태와 은닉 상태 통합)
```

### GRU 구조

```
       Update Gate (z)
       ┌────────────────────────────┐
       │                            │
h(t-1)─┴──►[σ]───z(t)──────×──(+)──►h(t)
              │           ↑    ↑
              │      ┌────┘    │
              │      │   ×─────┘
              │      │   ↑
              ├──►[σ]   [tanh]
              │   r(t)    │
              │    │      │
              └────×──────┘
                Reset Gate (r)
```

### 게이트 수식

```python
# Update Gate: 이전 상태 vs 새 상태 비율
z(t) = σ(W_z × [h(t-1), x(t)] + b_z)

# Reset Gate: 이전 상태를 얼마나 잊을지
r(t) = σ(W_r × [h(t-1), x(t)] + b_r)

# Candidate Hidden
h̃(t) = tanh(W × [r(t) × h(t-1), x(t)] + b)

# Hidden State Update
h(t) = (1 - z(t)) × h(t-1) + z(t) × h̃(t)
```

---

## 3. PyTorch LSTM/GRU

### LSTM

```python
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.1,
    bidirectional=False
)

# 순전파
# output: 모든 시간의 은닉 상태
# (h_n, c_n): 마지막 (은닉, 셀) 상태
output, (h_n, c_n) = lstm(x)
```

### GRU

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# 순전파 (셀 상태 없음)
output, h_n = gru(x)
```

---

## 4. LSTM 분류기

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        # 양방향이므로 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq) - 토큰 인덱스
        embedded = self.embedding(x)

        # LSTM
        output, (h_n, c_n) = self.lstm(embedded)

        # 양방향 마지막 은닉 상태 결합
        # h_n: (num_layers*2, batch, hidden)
        forward_last = h_n[-2]  # 정방향 마지막 층
        backward_last = h_n[-1]  # 역방향 마지막 층
        combined = torch.cat([forward_last, backward_last], dim=1)

        return self.fc(combined)
```

---

## 5. 시퀀스 생성 (언어 모델)

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_token, max_len, temperature=1.0):
        self.eval()
        tokens = [start_token]
        hidden = None

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([[tokens[-1]]])
                logits, hidden = self(x, hidden)

                # Temperature sampling
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

        return tokens
```

---

## 6. LSTM vs GRU 비교

| 항목 | LSTM | GRU |
|------|------|-----|
| 게이트 수 | 3개 (f, i, o) | 2개 (r, z) |
| 상태 | 셀 + 은닉 | 은닉만 |
| 파라미터 | 더 많음 | 더 적음 |
| 학습 속도 | 느림 | 빠름 |
| 성능 | 복잡한 패턴 | 비슷하거나 약간 낮음 |

### 선택 가이드

- **LSTM**: 긴 시퀀스, 복잡한 의존성
- **GRU**: 빠른 학습, 제한된 자원

---

## 7. 실전 팁

### 초기화

```python
# 은닉 상태 초기화
def init_hidden(batch_size, hidden_size, num_layers, bidirectional):
    num_directions = 2 if bidirectional else 1
    h = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    return (h.to(device), c.to(device))
```

### Dropout 패턴

```python
class LSTMWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # 마지막 은닉 상태에 dropout
        dropped = self.dropout(h_n[-1])
        return self.fc(dropped)
```

---

## 정리

### 핵심 개념

1. **LSTM**: 셀 상태로 장기 기억 유지, 3개 게이트
2. **GRU**: LSTM 단순화, 2개 게이트
3. **게이트**: 정보 흐름 제어 (시그모이드 × 값)

### 핵심 코드

```python
# LSTM
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
output, (h_n, c_n) = lstm(x)

# GRU
gru = nn.GRU(input_size, hidden_size, batch_first=True)
output, h_n = gru(x)
```

---

## 다음 단계

[16_Attention_Transformer.md](./16_Attention_Transformer.md)에서 Seq2Seq와 Attention을 학습합니다.
