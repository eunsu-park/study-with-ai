# 14. Practical Text Classification Project

## Learning Objectives

- Text preprocessing and tokenization
- Using embedding layers
- LSTM/Transformer-based classifiers
- Sentiment analysis project

---

## 1. Text Preprocessing

### Tokenization

```python
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
text = "This is a sample sentence!"
tokens = tokenizer(text)
# ['this', 'is', 'a', 'sample', 'sentence', '!']
```

### Building Vocabulary

```python
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    specials=['<unk>', '<pad>'],
    min_freq=5
)
vocab.set_default_index(vocab['<unk>'])
```

### Text → Tensor

```python
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

def collate_fn(batch):
    texts, labels = zip(*batch)
    # 토큰화 및 패딩
    encoded = [torch.tensor(text_pipeline(t)) for t in texts]
    padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True)
    labels = torch.tensor(labels)
    return padded, labels
```

---

## 2. Embedding Layer

### Basic Embedding

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq, embed)
        # 평균 풀링
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)
```

### Pre-trained Embeddings (GloVe)

```python
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=100)

# 임베딩 행렬 생성
embedding_matrix = torch.zeros(len(vocab), 100)
for i, word in enumerate(vocab.get_itos()):
    if word in glove.stoi:
        embedding_matrix[i] = glove[word]

# 모델에 적용
model.embedding.weight = nn.Parameter(embedding_matrix)
model.embedding.weight.requires_grad = False  # 고정 또는 미세조정
```

---

## 3. LSTM Classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        hidden_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq)
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)

        # 양방향: 마지막 정방향 + 마지막 역방향
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(hidden)
```

---

## 4. Transformer Classifier

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, mask=None):
        # 패딩 마스크
        padding_mask = (x == 0)

        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)

        output = self.transformer(embedded, src_key_padding_mask=padding_mask)

        # [CLS] 토큰 또는 평균 풀링
        pooled = output.mean(dim=1)
        return self.fc(pooled)
```

---

## 5. Sentiment Analysis Dataset

### IMDb

```python
from torchtext.datasets import IMDB

train_data, test_data = IMDB(split=('train', 'test'))

# 라벨: 'pos' → 1, 'neg' → 0
def label_pipeline(label):
    return 1 if label == 'pos' else 0
```

### Data Loader

```python
def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        processed = torch.tensor(text_pipeline(text), dtype=torch.long)
        texts.append(processed)

    labels = torch.tensor(labels)
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    # 최대 길이 제한
    if texts.size(1) > 256:
        texts = texts[:, :256]

    return texts, labels

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          collate_fn=collate_batch)
```

---

## 6. Training Pipeline

```python
def train_text_classifier():
    # 모델
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 학습
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output, labels)
            loss.backward()

            # 기울기 클리핑 (RNN에 중요)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Acc={train_acc:.2f}%")
```

---

## 7. Inference

```python
def predict_sentiment(model, text, vocab, tokenizer):
    model.eval()
    tokens = [vocab[t] for t in tokenizer(text.lower())]
    tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()

    sentiment = 'Positive' if pred == 1 else 'Negative'
    confidence = prob[0, pred].item()

    return sentiment, confidence

# 사용
text = "This movie was absolutely amazing! I loved every minute of it."
sentiment, conf = predict_sentiment(model, text, vocab, tokenizer)
print(f"{sentiment} ({conf*100:.1f}%)")
```

---

## 8. Using Hugging Face

### BERT Classifier

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# 토크나이저와 모델
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# 데이터 전처리
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length',
                     truncation=True, max_length=256)

# 학습
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

## Summary

### Text Classification Checklist

- [ ] Tokenization and vocabulary building
- [ ] Padding handling
- [ ] Embeddings (trained or pre-trained)
- [ ] Model selection (LSTM/Transformer)
- [ ] Gradient clipping
- [ ] Evaluation and inference

### Model Selection Guide

| Model | Advantages | Disadvantages |
|-------|-----------|---------------|
| LSTM | Simple implementation, fast training | Difficult with long sequences |
| Transformer | Parallelization, long sequences | High memory requirements |
| BERT (transfer learning) | Best performance | Slow, heavy |

### Expected Accuracy (IMDb)

| Model | Accuracy |
|-------|----------|
| LSTM | 85-88% |
| Transformer | 87-90% |
| BERT (fine-tuned) | 93-95% |

---

## Conclusion

This completes the Deep Learning learning course!

### Learning Summary

1. **Basics (01-04)**: Tensors, neural networks, backpropagation, training techniques
2. **CNN (05-07)**: Convolution, ResNet, transfer learning
3. **Sequences (08-10)**: RNN, LSTM, Transformer
4. **Practical (11-14)**: Optimization, deployment, projects

### Recommended Next Steps

- Study large language models in the LLM_and_NLP folder
- Apply to real projects
- Participate in Kaggle competitions
