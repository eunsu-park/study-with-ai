# 01. NLP Basics

## Learning Objectives

- Text preprocessing techniques
- Understanding tokenization methods
- Vocabulary building and encoding
- Text normalization

---

## 1. Text Preprocessing

### Preprocessing Pipeline

```
Raw Text
    ↓
Normalization (lowercase, remove special characters)
    ↓
Tokenization (word/subword splitting)
    ↓
Stopword Removal (optional)
    ↓
Vocabulary Building
    ↓
Encoding (text → numbers)
```

### Basic Preprocessing

```python
import re

def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

text = "Hello, World! This is NLP   processing."
print(preprocess(text))
# "hello world this is nlp processing"
```

---

## 2. Tokenization

### Word Tokenization

```python
# Space-based
text = "I love natural language processing"
tokens = text.split()
# ['I', 'love', 'natural', 'language', 'processing']

# NLTK
import nltk
from nltk.tokenize import word_tokenize
tokens = word_tokenize("I don't like it.")
# ['I', 'do', "n't", 'like', 'it', '.']
```

### Subword Tokenization

Subwords break words into smaller units

```
"unhappiness" → ["un", "##happiness"] (WordPiece)
"unhappiness" → ["un", "happi", "ness"] (BPE)
```

**Advantages**:
- Handle out-of-vocabulary (OOV) words
- Reduce vocabulary size
- Preserve morphological information

### BPE (Byte Pair Encoding)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Tokenize
output = tokenizer.encode("Hello, world!")
print(output.tokens)
```

### WordPiece (BERT)

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "I love natural language processing"
tokens = tokenizer.tokenize(text)
# ['i', 'love', 'natural', 'language', 'processing']

# Encode
encoded = tokenizer.encode(text)
# [101, 1045, 2293, 3019, 2653, 6364, 102]

# Decode
decoded = tokenizer.decode(encoded)
# "[CLS] i love natural language processing [SEP]"
```

### SentencePiece (GPT, T5)

```python
import sentencepiece as spm

# Train
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='spm',
    vocab_size=8000,
    model_type='bpe'
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('spm.model')

tokens = sp.encode_as_pieces("Hello, world!")
# ['▁Hello', ',', '▁world', '!']

ids = sp.encode_as_ids("Hello, world!")
# [1234, 567, 890, 12]
```

---

## 3. Vocabulary Building

### Basic Vocabulary Dictionary

```python
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts, tokenizer):
        # Count word frequencies
        for text in texts:
            tokens = tokenizer(text)
            self.word_freq.update(tokens)

        # Filter by frequency and add
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text, tokenizer):
        tokens = tokenizer(text)
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]

    def decode(self, indices):
        return [self.idx2word.get(i, '<unk>') for i in indices]

    def __len__(self):
        return len(self.word2idx)

# Usage
vocab = Vocabulary(min_freq=2)
vocab.build(texts, str.split)
encoded = vocab.encode("hello world", str.split)
```

### torchtext Vocabulary

```python
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(texts, tokenizer),
    specials=['<pad>', '<unk>'],
    min_freq=2
)
vocab.set_default_index(vocab['<unk>'])

# Usage
indices = vocab(tokenizer("hello world"))
```

---

## 4. Padding and Batch Processing

### Sequence Padding

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts, labels = zip(*batch)

    # Tokenize and encode
    encoded = [torch.tensor(vocab.encode(t, tokenizer)) for t in texts]

    # Pad (to longest sequence)
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)

    # Limit maximum length
    if padded.size(1) > max_len:
        padded = padded[:, :max_len]

    labels = torch.tensor(labels)
    return padded, labels

# Apply to DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

### Attention Mask

```python
def create_attention_mask(input_ids, pad_token_id=0):
    """1 for non-padding positions, 0 for padding"""
    return (input_ids != pad_token_id).long()

# Example
input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
attention_mask = create_attention_mask(input_ids)
# tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
```

---

## 5. Text Normalization

### Various Normalization Techniques

```python
import unicodedata

def normalize_text(text):
    # Unicode normalization (NFD → NFC)
    text = unicodedata.normalize('NFC', text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Normalize numbers (optional)
    text = re.sub(r'\d+', '<NUM>', text)

    # Reduce repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # "sooooo" → "soo"

    return text.strip()
```

### Stopword Removal

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stop_words]

tokens = ['this', 'is', 'a', 'test', 'sentence']
filtered = remove_stopwords(tokens)
# ['test', 'sentence']
```

### Lemmatization

```python
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words = ['running', 'runs', 'ran', 'better', 'cats']
lemmas = [lemmatizer.lemmatize(w) for w in words]
# ['running', 'run', 'ran', 'better', 'cat']
```

---

## 6. HuggingFace Tokenizers

### Basic Usage

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Encode
text = "Hello, how are you?"
encoded = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=32,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)      # torch.Size([1, 32])
print(encoded['attention_mask'].shape) # torch.Size([1, 32])
```

### Batch Encoding

```python
texts = ["Hello world", "NLP is fun", "I love Python"]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=16,
    return_tensors='pt'
)

print(encoded['input_ids'].shape)  # torch.Size([3, 16])
```

### Special Tokens

```python
# BERT special tokens
print(tokenizer.special_tokens_map)
# {'unk_token': '[UNK]', 'sep_token': '[SEP]',
#  'pad_token': '[PAD]', 'cls_token': '[CLS]',
#  'mask_token': '[MASK]'}

# Token IDs
print(tokenizer.cls_token_id)  # 101
print(tokenizer.sep_token_id)  # 102
print(tokenizer.pad_token_id)  # 0
```

---

## 7. Practice: Text Classification Preprocessing

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

# Usage
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TextClassificationDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    input_ids = batch['input_ids']       # (32, 128)
    attention_mask = batch['attention_mask']  # (32, 128)
    labels = batch['label']              # (32,)
    break
```

---

## Summary

### Tokenization Methods Comparison

| Method | Advantages | Disadvantages | Used in Models |
|--------|-----------|---------------|----------------|
| Word-level | Intuitive | OOV problem | Traditional NLP |
| BPE | Solves OOV | Requires training | GPT |
| WordPiece | Solves OOV | Requires training | BERT |
| SentencePiece | Language-agnostic | Requires training | T5, GPT |

### Key Code

```python
# HuggingFace tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Vocabulary building
vocab = build_vocab_from_iterator(yield_tokens(texts), specials=['<pad>', '<unk>'])

# Padding
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
```

---

## Next Steps

Learn about word embeddings in [02_Word2Vec_GloVe.md](./02_Word2Vec_GloVe.md).
