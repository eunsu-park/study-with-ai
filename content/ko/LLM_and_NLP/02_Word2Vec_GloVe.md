# 02. Word2Vec과 GloVe

## 학습 목표

- 분산 표현의 개념
- Word2Vec (Skip-gram, CBOW)
- GloVe 임베딩
- 사전학습 임베딩 활용

---

## 1. 단어 임베딩 개요

### One-Hot vs 분산 표현

```
One-Hot (희소 표현):
    "king"  → [1, 0, 0, 0, ...]  (V차원)
    "queen" → [0, 1, 0, 0, ...]

문제: 의미적 유사성 표현 불가
      cosine_similarity(king, queen) = 0

분산 표현 (Dense):
    "king"  → [0.2, -0.5, 0.8, ...]  (d차원, d << V)
    "queen" → [0.3, -0.4, 0.7, ...]

장점: 의미적 유사성 반영
      cosine_similarity(king, queen) ≈ 0.9
```

### 분산 가설

> "같은 맥락에서 등장하는 단어는 비슷한 의미를 갖는다"
> (You shall know a word by the company it keeps)

```
"The cat sat on the ___"  → mat, floor, couch
"The dog lay on the ___"  → mat, floor, couch

cat ≈ dog (유사한 맥락)
```

---

## 2. Word2Vec

### Skip-gram

주변 단어를 예측하여 중심 단어 표현 학습

```
입력: center word → 예측: context words

문장: "The quick brown fox jumps"
중심 단어: "brown" (window=2)
예측 대상: ["quick", "fox"] 또는 ["The", "quick", "fox", "jumps"]

모델:
    "brown" → 임베딩 → Softmax → P(context | center)
```

### CBOW (Continuous Bag of Words)

주변 단어로 중심 단어 예측

```
입력: context words → 예측: center word

문장: "The quick brown fox jumps"
주변 단어: ["quick", "fox"]
예측 대상: "brown"

모델:
    ["quick", "fox"] → 평균 임베딩 → Softmax → P(center | context)
```

### Word2Vec 구조

```python
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 입력 임베딩 (중심 단어)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 출력 임베딩 (주변 단어)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        # center: (batch,)
        # context: (batch,)
        center_emb = self.center_embeddings(center)   # (batch, embed)
        context_emb = self.context_embeddings(context)  # (batch, embed)

        # 내적으로 유사도 계산
        score = (center_emb * context_emb).sum(dim=1)  # (batch,)
        return score

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, context, center):
        # context: (batch, window*2)
        # center: (batch,)
        context_emb = self.context_embeddings(context)  # (batch, window*2, embed)
        context_mean = context_emb.mean(dim=1)  # (batch, embed)

        center_emb = self.center_embeddings(center)  # (batch, embed)

        score = (context_mean * center_emb).sum(dim=1)
        return score
```

### Negative Sampling

전체 어휘에 대한 Softmax는 계산 비용이 큼

```python
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, neg_context):
        # center: (batch,)
        # context: (batch,) - 실제 주변 단어
        # neg_context: (batch, k) - 랜덤 샘플링된 단어

        center_emb = self.center_embeddings(center)  # (batch, embed)

        # Positive: 실제 주변 단어와의 유사도
        pos_emb = self.context_embeddings(context)
        pos_score = (center_emb * pos_emb).sum(dim=1)  # (batch,)

        # Negative: 랜덤 단어와의 유사도
        neg_emb = self.context_embeddings(neg_context)  # (batch, k, embed)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # (batch, k)

        return pos_score, neg_score

# 손실 함수
def negative_sampling_loss(pos_score, neg_score):
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
    neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)
    return (pos_loss + neg_loss).mean()
```

---

## 3. GloVe

### 개념

전역 동시 출현 통계 활용

```
동시 출현 행렬 X:
    X[i,j] = 단어 i와 j가 함께 등장한 횟수

목표:
    w_i · w_j + b_i + b_j ≈ log(X[i,j])
```

### GloVe 손실 함수

```python
def glove_loss(w_i, w_j, b_i, b_j, X_ij, x_max=100, alpha=0.75):
    """
    w_i, w_j: 단어 임베딩
    b_i, b_j: 편향
    X_ij: 동시 출현 횟수
    """
    # 가중치 함수 (빈도가 너무 높은 단어 완화)
    weight = torch.clamp(X_ij / x_max, max=1.0) ** alpha

    # 예측과 실제의 차이
    prediction = (w_i * w_j).sum(dim=1) + b_i + b_j
    target = torch.log(X_ij + 1e-10)

    loss = weight * (prediction - target) ** 2
    return loss.mean()
```

### GloVe 구현

```python
class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 두 임베딩 행렬
        self.w_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.c_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.w_bias = nn.Embedding(vocab_size, 1)
        self.c_bias = nn.Embedding(vocab_size, 1)

    def forward(self, i, j, cooccur):
        w_i = self.w_embeddings(i)
        w_j = self.c_embeddings(j)
        b_i = self.w_bias(i).squeeze()
        b_j = self.c_bias(j).squeeze()

        return glove_loss(w_i, w_j, b_i, b_j, cooccur)

    def get_embedding(self, word_idx):
        # 최종 임베딩: 두 임베딩의 평균
        return (self.w_embeddings.weight[word_idx] +
                self.c_embeddings.weight[word_idx]) / 2
```

---

## 4. 사전학습 임베딩 사용

### Gensim Word2Vec

```python
from gensim.models import Word2Vec

# 학습
sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 유사 단어
similar = model.wv.most_similar("NLP", topn=5)

# 벡터 가져오기
vector = model.wv["NLP"]

# 저장/로드
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
```

### 사전학습 GloVe

```python
import numpy as np

def load_glove(path, embed_dim=100):
    """GloVe 텍스트 파일 로드"""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 사용
glove = load_glove('glove.6B.100d.txt')
vector = glove.get('king', np.zeros(100))
```

### PyTorch 임베딩 레이어에 적용

```python
import torch
import torch.nn as nn

def create_embedding_layer(vocab, glove, embed_dim=100, freeze=True):
    """사전학습 임베딩으로 Embedding 레이어 초기화"""
    vocab_size = len(vocab)
    embedding_matrix = torch.zeros(vocab_size, embed_dim)

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            embedding_matrix[idx] = torch.from_numpy(glove[word])
            found += 1
        else:
            # 랜덤 초기화
            embedding_matrix[idx] = torch.randn(embed_dim) * 0.1

    print(f"사전학습 임베딩 적용: {found}/{vocab_size}")

    embedding = nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=freeze,  # True면 학습하지 않음
        padding_idx=vocab.word2idx.get('<pad>', 0)
    )
    return embedding

# 모델에 적용
class TextClassifier(nn.Module):
    def __init__(self, vocab, glove, num_classes):
        super().__init__()
        self.embedding = create_embedding_layer(vocab, glove, freeze=False)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, 100)
        pooled = embedded.mean(dim=1)  # 평균 풀링
        return self.fc(pooled)
```

---

## 5. 임베딩 연산

### 유사도 계산

```python
import torch
import torch.nn.functional as F

def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))

# 가장 유사한 단어 찾기
def most_similar(word, embeddings, vocab, topk=5):
    word_vec = embeddings[vocab[word]]
    similarities = F.cosine_similarity(word_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 1)

    results = []
    for val, idx in zip(values[1:], indices[1:]):  # 자기 자신 제외
        results.append((vocab.idx2word[idx.item()], val.item()))
    return results
```

### 단어 연산

```python
def word_analogy(a, b, c, embeddings, vocab, topk=5):
    """
    a : b = c : ?
    예: king : queen = man : woman

    vector(?) = vector(b) - vector(a) + vector(c)
    """
    vec_a = embeddings[vocab[a]]
    vec_b = embeddings[vocab[b]]
    vec_c = embeddings[vocab[c]]

    # 유추 벡터
    target_vec = vec_b - vec_a + vec_c

    # 가장 유사한 단어 찾기
    similarities = F.cosine_similarity(target_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 3)

    # a, b, c 제외
    exclude = {vocab[a], vocab[b], vocab[c]}
    results = []
    for val, idx in zip(values, indices):
        if idx.item() not in exclude:
            results.append((vocab.idx2word[idx.item()], val.item()))
        if len(results) == topk:
            break
    return results

# 예시
# word_analogy("king", "queen", "man", embeddings, vocab)
# → [("woman", 0.85), ...]
```

---

## 6. 시각화

### t-SNE 시각화

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words, vocab):
    # 선택한 단어의 임베딩
    indices = [vocab[w] for w in words]
    vectors = embeddings[indices].numpy()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    reduced = tsne.fit_transform(vectors)

    # 시각화
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))

    plt.title('Word Embeddings (t-SNE)')
    plt.savefig('embeddings_tsne.png')
    plt.close()

# 사용
words = ['king', 'queen', 'man', 'woman', 'dog', 'cat', 'apple', 'orange']
visualize_embeddings(embeddings, words, vocab)
```

---

## 7. Word2Vec vs GloVe 비교

| 항목 | Word2Vec | GloVe |
|------|----------|-------|
| 방식 | 예측 기반 | 통계 기반 |
| 학습 | 윈도우 내 단어 | 전역 동시 출현 |
| 메모리 | 적음 | 동시 출현 행렬 필요 |
| 학습 속도 | Negative Sampling으로 빠름 | 행렬 전처리 후 빠름 |
| 성능 | 유사 | 유사 |

---

## 정리

### 핵심 개념

1. **분산 표현**: 단어를 밀집 벡터로 표현
2. **Skip-gram**: 중심 → 주변 예측
3. **CBOW**: 주변 → 중심 예측
4. **GloVe**: 동시 출현 통계 활용
5. **단어 연산**: king - queen + man ≈ woman

### 핵심 코드

```python
# Gensim Word2Vec
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)

# 사전학습 임베딩 적용
embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)

# 유사도
similarity = F.cosine_similarity(vec1, vec2)
```

---

## 다음 단계

[03_Transformer_Review.md](./03_Transformer_Review.md)에서 Transformer 아키텍처를 NLP 관점에서 복습합니다.
