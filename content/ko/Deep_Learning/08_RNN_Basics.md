# 08. RNN 기초 (Recurrent Neural Networks)

## 학습 목표

- 순환 신경망의 개념과 구조
- 시퀀스 데이터 처리
- PyTorch nn.RNN 사용법
- 기울기 소실 문제 이해

---

## 1. RNN이란?

### 순차 데이터의 특성

```
시계열: [1, 2, 3, 4, 5, ...]  - 이전 값이 다음 값에 영향
텍스트: "나는 학교에 간다"     - 이전 단어가 다음 단어에 영향
```

### MLP의 한계

- 고정된 입력 크기
- 순서 정보 무시
- 가변 길이 시퀀스 처리 불가

### RNN의 해결

```
h(t) = tanh(W_xh × x(t) + W_hh × h(t-1) + b)

h(t): 현재 은닉 상태
x(t): 현재 입력
h(t-1): 이전 은닉 상태
```

---

## 2. RNN 구조

### 시간 펼침 (Unrolling)

```
    x1      x2      x3      x4
    ↓       ↓       ↓       ↓
  ┌───┐   ┌───┐   ┌───┐   ┌───┐
  │ h │──►│ h │──►│ h │──►│ h │──► 출력
  └───┘   └───┘   └───┘   └───┘
    h0      h1      h2      h3
```

### 파라미터 공유

- 모든 시간 단계에서 동일한 W_xh, W_hh 사용
- 가변 길이 시퀀스 처리 가능

---

## 3. PyTorch RNN

### 기본 사용법

```python
import torch
import torch.nn as nn

# RNN 생성
rnn = nn.RNN(
    input_size=10,    # 입력 차원
    hidden_size=20,   # 은닉 상태 차원
    num_layers=2,     # RNN 층 수
    batch_first=True  # 입력: (batch, seq, feature)
)

# 입력 형태: (batch_size, seq_len, input_size)
x = torch.randn(32, 15, 10)  # 배치 32, 시퀀스 15, 특성 10

# 순전파
# output: 모든 시간의 은닉 상태 (batch, seq, hidden)
# h_n: 마지막 은닉 상태 (layers, batch, hidden)
output, h_n = rnn(x)

print(f"output: {output.shape}")  # (32, 15, 20)
print(f"h_n: {h_n.shape}")        # (2, 32, 20)
```

### 양방향 RNN

```python
rnn_bi = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=1,
    batch_first=True,
    bidirectional=True  # 양방향
)

output, h_n = rnn_bi(x)
print(f"output: {output.shape}")  # (32, 15, 40) - 정방향+역방향
print(f"h_n: {h_n.shape}")        # (2, 32, 20) - 방향별 마지막 상태
```

---

## 4. RNN 분류기 구현

### 시퀀스 분류 모델

```python
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        output, h_n = self.rnn(x)

        # 마지막 시간의 은닉 상태 사용
        # h_n[-1]: 마지막 층의 은닉 상태
        out = self.fc(h_n[-1])
        return out
```

### Many-to-Many 구조

```python
class RNNSeq2Seq(nn.Module):
    """시퀀스 → 시퀀스"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        # 모든 시간 단계에 FC 적용
        out = self.fc(output)  # (batch, seq, output_size)
        return out
```

---

## 5. 기울기 소실 문제

### 문제

```
긴 시퀀스에서:
h100 ← W_hh × W_hh × ... × W_hh × h1
                    ↑
            100번 곱셈 → 기울기 폭발 또는 소실
```

### 원인

- |W_hh| > 1: 기울기 폭발
- |W_hh| < 1: 기울기 소실

### 해결책

1. **LSTM/GRU 사용** (다음 레슨)
2. **Gradient Clipping**

```python
# 기울기 클리핑
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 6. 시계열 예측 예제

### 사인파 예측

```python
import numpy as np

# 데이터 생성
def generate_sin_data(seq_len=50, n_samples=1000):
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2*np.pi)
        seq = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[-1])
    return np.array(X), np.array(y)

X, y = generate_sin_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 모델
class SinPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1]).squeeze()
```

---

## 7. 텍스트 분류 예제

### 문자 수준 RNN

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq) - 문자 인덱스
        embedded = self.embedding(x)  # (batch, seq, embed)
        output, h_n = self.rnn(embedded)
        out = self.fc(h_n[-1])
        return out

# 예시
vocab_size = 27  # a-z + 공백
model = CharRNN(vocab_size, embed_size=32, hidden_size=64, num_classes=5)
```

---

## 8. 주의사항

### 입력 형태

```python
# batch_first=True  → (batch, seq, feature)
# batch_first=False → (seq, batch, feature)  # 기본값
```

### 가변 길이 시퀀스

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 패딩된 시퀀스와 실제 길이
padded_seqs = ...  # (batch, max_len, features)
lengths = ...      # 각 시퀀스의 실제 길이

# 패킹 (패딩 무시)
packed = pack_padded_sequence(padded_seqs, lengths,
                               batch_first=True, enforce_sorted=False)
output, h_n = rnn(packed)

# 언패킹
output_padded, _ = pad_packed_sequence(output, batch_first=True)
```

---

## 9. RNN 변형 비교

| 모델 | 장점 | 단점 |
|------|------|------|
| Simple RNN | 단순, 빠름 | 긴 시퀀스 학습 어려움 |
| LSTM | 장기 의존성 학습 | 복잡, 느림 |
| GRU | LSTM과 유사, 더 단순 | - |

---

## 정리

### 핵심 개념

1. **순환 구조**: 이전 상태가 다음 계산에 영향
2. **파라미터 공유**: 시간 독립적 가중치
3. **기울기 문제**: 긴 시퀀스에서 학습 어려움

### 핵심 코드

```python
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
output, h_n = rnn(x)  # output: 전체, h_n: 마지막
```

---

## 다음 단계

[09_LSTM_GRU.md](./09_LSTM_GRU.md)에서 LSTM과 GRU를 학습합니다.
