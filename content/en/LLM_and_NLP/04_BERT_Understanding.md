# 04. BERT Understanding

## Learning Objectives

- Understanding BERT architecture
- Pre-training objectives (MLM, NSP)
- Input representations
- Various BERT variants

---

## 1. BERT Overview

### Bidirectional Encoder Representations from Transformers

```
BERT = Stack of Transformer encoders

Features:
- Bidirectional context understanding
- Pre-training + Fine-tuning paradigm
- General-purpose application to various NLP tasks
```

### Model Sizes

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|------------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

---

## 2. Input Representation

### Sum of Three Embeddings

```
Input: [CLS] I love NLP [SEP] It is fun [SEP]

Token Embedding:    [E_CLS, E_I, E_love, E_NLP, E_SEP, E_It, E_is, E_fun, E_SEP]
Segment Embedding:  [E_A,   E_A, E_A,    E_A,   E_A,   E_B,  E_B,  E_B,   E_B  ]
Position Embedding: [E_0,   E_1, E_2,    E_3,   E_4,   E_5,  E_6,  E_7,   E_8  ]
                    ─────────────────────────────────────────────────────────────
                    = Final input embedding (sum)
```

### Special Tokens

| Token | Role |
|-------|------|
| [CLS] | Aggregation token for classification tasks |
| [SEP] | Sentence separator |
| [PAD] | Padding |
| [MASK] | Masked token in MLM |
| [UNK] | Unknown word |

### Input Implementation

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

        # Position indices
        position_ids = torch.arange(seq_len, device=input_ids.device)

        # Sum embeddings
        embeddings = (
            self.token_embedding(input_ids) +
            self.position_embedding(position_ids) +
            self.segment_embedding(segment_ids)
        )

        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
```

---

## 3. Pre-training Objectives

### Masked Language Model (MLM)

```
Select 15% of tokens:
- 80%: Replace with [MASK]
- 10%: Replace with random token
- 10%: Keep unchanged

Example:
Input: "The cat sat on the mat"
     → "The [MASK] sat on the mat"
Target: Predict [MASK] → "cat"
```

```python
import random

def create_mlm_data(tokens, vocab, mask_prob=0.15):
    """Generate MLM training data"""
    labels = [-100] * len(tokens)  # -100 is ignored in loss calculation

    for i, token in enumerate(tokens):
        if random.random() < mask_prob:
            labels[i] = vocab[token]  # Original token ID

            rand = random.random()
            if rand < 0.8:
                tokens[i] = '[MASK]'
            elif rand < 0.9:
                tokens[i] = random.choice(list(vocab.keys()))
            # else: keep unchanged

    return tokens, labels
```

### Next Sentence Prediction (NSP)

```
Input: [CLS] sentence A [SEP] sentence B [SEP]
Target: Binary classification - is sentence B the actual next sentence after A?

Example:
Positive (IsNext):
    A: "The man went to the store"
    B: "He bought a gallon of milk"

Negative (NotNext):
    A: "The man went to the store"
    B: "Penguins are flightless birds"
```

```python
class BERTPreTrainingHeads(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # MLM head
        self.mlm = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        # NSP head
        self.nsp = nn.Linear(d_model, 2)

    def forward(self, sequence_output, cls_output):
        mlm_scores = self.mlm(sequence_output)  # (batch, seq, vocab)
        nsp_scores = self.nsp(cls_output)       # (batch, 2)
        return mlm_scores, nsp_scores
```

---

## 4. Complete BERT Architecture

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
        # Embedding
        x = self.embedding(input_ids, segment_ids)

        # Convert padding mask
        if attention_mask is not None:
            # (batch, seq) → (batch, seq) with True for padding
            attention_mask = (attention_mask == 0)

        # Encoder
        output = self.encoder(x, src_key_padding_mask=attention_mask)

        return output  # (batch, seq, d_model)


class BERTForPreTraining(nn.Module):
    def __init__(self, vocab_size, d_model=768, **kwargs):
        super().__init__()
        self.bert = BERT(vocab_size, d_model, **kwargs)
        self.heads = BERTPreTrainingHeads(d_model, vocab_size)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        sequence_output = self.bert(input_ids, segment_ids, attention_mask)
        cls_output = sequence_output[:, 0]  # [CLS] token

        mlm_scores, nsp_scores = self.heads(sequence_output, cls_output)
        return mlm_scores, nsp_scores
```

---

## 5. Fine-tuning Patterns

### Sentence Classification (Single Sentence)

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

### Token Classification (NER)

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

### Question Answering (QA)

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

## 6. BERT Variants

### RoBERTa

```
Changes:
- Remove NSP (use MLM only)
- Dynamic masking (different masking per epoch)
- Larger batches, longer training
- Byte-Level BPE tokenizer

Result: Improved performance over BERT
```

### ALBERT

```
Changes:
- Embedding factorization (V×E, E×H → V×E, E<<H)
- Cross-layer parameter sharing
- NSP → SOP (Sentence Order Prediction)

Result: Significantly reduced parameters, similar performance
```

### DistilBERT

```
Changes:
- Knowledge distillation (Teacher: BERT → Student: smaller model)
- 6 layers (half of BERT)

Result: 40% smaller, 60% faster, retains 97% performance
```

### Comparison

| Model | Layers | Parameters | Speed | Features |
|-------|--------|------------|-------|----------|
| BERT-base | 12 | 110M | 1x | Baseline |
| RoBERTa | 12 | 125M | 1x | Optimized training |
| ALBERT-base | 12 | 12M | 1x | Parameter sharing |
| DistilBERT | 6 | 66M | 2x | Knowledge distillation |

---

## 7. Using HuggingFace BERT

### Basic Usage

```python
from transformers import BertTokenizer, BertModel

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encoding
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# Forward pass
outputs = model(**inputs)

# Outputs
last_hidden_state = outputs.last_hidden_state  # (1, seq, 768)
pooler_output = outputs.pooler_output          # (1, 768) - [CLS] transformation
```

### Classification Model

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

### Attention Visualization

```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

inputs = tokenizer("The cat sat on the mat", return_tensors='pt')
outputs = model(**inputs)

# Attention weights: (num_layers, batch, heads, seq, seq)
attentions = outputs.attentions

# First layer, first head
attn = attentions[0][0, 0].detach().numpy()
```

---

## 8. BERT Input Formats

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

### Pair Processing in HuggingFace

```python
# Two sentence input
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

## Summary

### Key Concepts

1. **Bidirectional Encoder**: Understand full context bidirectionally
2. **MLM**: Learn context by predicting masked tokens
3. **NSP**: Understand sentence relationships (removed in RoBERTa)
4. **[CLS] Token**: Sentence-level representation
5. **Segment Embedding**: Sentence differentiation

### Key Code

```python
# HuggingFace BERT
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encoding
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]
```

---

## Next Steps

Learn about GPT model and autoregressive language modeling in [05_GPT_Understanding.md](./05_GPT_Understanding.md).
