# 02. Word2Vec and GloVe

## Learning Objectives

- Understanding distributed representations
- Word2Vec (Skip-gram, CBOW)
- GloVe embeddings
- Using pre-trained embeddings

---

## 1. Word Embedding Overview

### One-Hot vs Distributed Representation

```
One-Hot (Sparse Representation):
    "king"  → [1, 0, 0, 0, ...]  (V-dimensional)
    "queen" → [0, 1, 0, 0, ...]

Problem: Cannot express semantic similarity
         cosine_similarity(king, queen) = 0

Distributed Representation (Dense):
    "king"  → [0.2, -0.5, 0.8, ...]  (d-dimensional, d << V)
    "queen" → [0.3, -0.4, 0.7, ...]

Advantage: Reflects semantic similarity
           cosine_similarity(king, queen) ≈ 0.9
```

### Distributional Hypothesis

> "Words that appear in similar contexts have similar meanings"
> (You shall know a word by the company it keeps)

```
"The cat sat on the ___"  → mat, floor, couch
"The dog lay on the ___"  → mat, floor, couch

cat ≈ dog (similar context)
```

---

## 2. Word2Vec

### Skip-gram

Learn center word representation by predicting surrounding words

```
Input: center word → Predict: context words

Sentence: "The quick brown fox jumps"
Center word: "brown" (window=2)
Target predictions: ["quick", "fox"] or ["The", "quick", "fox", "jumps"]

Model:
    "brown" → Embedding → Softmax → P(context | center)
```

### CBOW (Continuous Bag of Words)

Predict center word from surrounding words

```
Input: context words → Predict: center word

Sentence: "The quick brown fox jumps"
Context words: ["quick", "fox"]
Target prediction: "brown"

Model:
    ["quick", "fox"] → Average Embedding → Softmax → P(center | context)
```

### Word2Vec Architecture

```python
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Input embedding (center word)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Output embedding (context word)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        # center: (batch,)
        # context: (batch,)
        center_emb = self.center_embeddings(center)   # (batch, embed)
        context_emb = self.context_embeddings(context)  # (batch, embed)

        # Calculate similarity via dot product
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

Softmax over entire vocabulary is computationally expensive

```python
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, neg_context):
        # center: (batch,)
        # context: (batch,) - actual context words
        # neg_context: (batch, k) - randomly sampled words

        center_emb = self.center_embeddings(center)  # (batch, embed)

        # Positive: similarity with actual context words
        pos_emb = self.context_embeddings(context)
        pos_score = (center_emb * pos_emb).sum(dim=1)  # (batch,)

        # Negative: similarity with random words
        neg_emb = self.context_embeddings(neg_context)  # (batch, k, embed)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # (batch, k)

        return pos_score, neg_score

# Loss function
def negative_sampling_loss(pos_score, neg_score):
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)
    neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)
    return (pos_loss + neg_loss).mean()
```

---

## 3. GloVe

### Concept

Utilize global co-occurrence statistics

```
Co-occurrence matrix X:
    X[i,j] = number of times word i and j appear together

Objective:
    w_i · w_j + b_i + b_j ≈ log(X[i,j])
```

### GloVe Loss Function

```python
def glove_loss(w_i, w_j, b_i, b_j, X_ij, x_max=100, alpha=0.75):
    """
    w_i, w_j: word embeddings
    b_i, b_j: biases
    X_ij: co-occurrence count
    """
    # Weighting function (dampen very frequent words)
    weight = torch.clamp(X_ij / x_max, max=1.0) ** alpha

    # Difference between prediction and actual
    prediction = (w_i * w_j).sum(dim=1) + b_i + b_j
    target = torch.log(X_ij + 1e-10)

    loss = weight * (prediction - target) ** 2
    return loss.mean()
```

### GloVe Implementation

```python
class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Two embedding matrices
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
        # Final embedding: average of two embeddings
        return (self.w_embeddings.weight[word_idx] +
                self.c_embeddings.weight[word_idx]) / 2
```

---

## 4. Using Pre-trained Embeddings

### Gensim Word2Vec

```python
from gensim.models import Word2Vec

# Training
sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Similar words
similar = model.wv.most_similar("NLP", topn=5)

# Get vector
vector = model.wv["NLP"]

# Save/Load
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
```

### Pre-trained GloVe

```python
import numpy as np

def load_glove(path, embed_dim=100):
    """Load GloVe text file"""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Usage
glove = load_glove('glove.6B.100d.txt')
vector = glove.get('king', np.zeros(100))
```

### Apply to PyTorch Embedding Layer

```python
import torch
import torch.nn as nn

def create_embedding_layer(vocab, glove, embed_dim=100, freeze=True):
    """Initialize Embedding layer with pre-trained embeddings"""
    vocab_size = len(vocab)
    embedding_matrix = torch.zeros(vocab_size, embed_dim)

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            embedding_matrix[idx] = torch.from_numpy(glove[word])
            found += 1
        else:
            # Random initialization
            embedding_matrix[idx] = torch.randn(embed_dim) * 0.1

    print(f"Pre-trained embeddings applied: {found}/{vocab_size}")

    embedding = nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=freeze,  # If True, don't train
        padding_idx=vocab.word2idx.get('<pad>', 0)
    )
    return embedding

# Apply to model
class TextClassifier(nn.Module):
    def __init__(self, vocab, glove, num_classes):
        super().__init__()
        self.embedding = create_embedding_layer(vocab, glove, freeze=False)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, 100)
        pooled = embedded.mean(dim=1)  # Average pooling
        return self.fc(pooled)
```

---

## 5. Embedding Operations

### Similarity Calculation

```python
import torch
import torch.nn.functional as F

def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))

# Find most similar words
def most_similar(word, embeddings, vocab, topk=5):
    word_vec = embeddings[vocab[word]]
    similarities = F.cosine_similarity(word_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 1)

    results = []
    for val, idx in zip(values[1:], indices[1:]):  # Exclude self
        results.append((vocab.idx2word[idx.item()], val.item()))
    return results
```

### Word Arithmetic

```python
def word_analogy(a, b, c, embeddings, vocab, topk=5):
    """
    a : b = c : ?
    Example: king : queen = man : woman

    vector(?) = vector(b) - vector(a) + vector(c)
    """
    vec_a = embeddings[vocab[a]]
    vec_b = embeddings[vocab[b]]
    vec_c = embeddings[vocab[c]]

    # Analogy vector
    target_vec = vec_b - vec_a + vec_c

    # Find most similar words
    similarities = F.cosine_similarity(target_vec.unsqueeze(0), embeddings)
    values, indices = similarities.topk(topk + 3)

    # Exclude a, b, c
    exclude = {vocab[a], vocab[b], vocab[c]}
    results = []
    for val, idx in zip(values, indices):
        if idx.item() not in exclude:
            results.append((vocab.idx2word[idx.item()], val.item()))
        if len(results) == topk:
            break
    return results

# Example
# word_analogy("king", "queen", "man", embeddings, vocab)
# → [("woman", 0.85), ...]
```

---

## 6. Visualization

### t-SNE Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words, vocab):
    # Embeddings of selected words
    indices = [vocab[w] for w in words]
    vectors = embeddings[indices].numpy()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    reduced = tsne.fit_transform(vectors)

    # Visualize
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))

    plt.title('Word Embeddings (t-SNE)')
    plt.savefig('embeddings_tsne.png')
    plt.close()

# Usage
words = ['king', 'queen', 'man', 'woman', 'dog', 'cat', 'apple', 'orange']
visualize_embeddings(embeddings, words, vocab)
```

---

## 7. Word2Vec vs GloVe Comparison

| Item | Word2Vec | GloVe |
|------|----------|-------|
| Approach | Prediction-based | Statistics-based |
| Training | Words within window | Global co-occurrence |
| Memory | Low | Requires co-occurrence matrix |
| Training Speed | Fast with Negative Sampling | Fast after matrix preprocessing |
| Performance | Similar | Similar |

---

## Summary

### Key Concepts

1. **Distributed Representation**: Represent words as dense vectors
2. **Skip-gram**: Predict context from center word
3. **CBOW**: Predict center word from context
4. **GloVe**: Utilize co-occurrence statistics
5. **Word Arithmetic**: king - queen + man ≈ woman

### Key Code

```python
# Gensim Word2Vec
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)

# Apply pre-trained embeddings
embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)

# Similarity
similarity = F.cosine_similarity(vec1, vec2)
```

---

## Next Steps

Review Transformer architecture from an NLP perspective in [03_Transformer_Review.md](./03_Transformer_Review.md).
