# 01. NLP 기초

## 학습 목표

- 텍스트 전처리 기법
- 토큰화 방법 이해
- 어휘 구축과 인코딩
- 텍스트 정규화

---

## 1. 텍스트 전처리

### 전처리 파이프라인

```
원본 텍스트
    ↓
정규화 (소문자, 특수문자 제거)
    ↓
토큰화 (단어/서브워드 분리)
    ↓
불용어 제거 (선택)
    ↓
어휘 구축
    ↓
인코딩 (텍스트 → 숫자)
```

### 기본 전처리

```python
import re

def preprocess(text):
    # 소문자 변환
    text = text.lower()

    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)

    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text).strip()

    return text

text = "Hello, World! This is NLP   processing."
print(preprocess(text))
# "hello world this is nlp processing"
```

---

## 2. 토큰화 (Tokenization)

### 단어 토큰화

```python
# 공백 기반
text = "I love natural language processing"
tokens = text.split()
# ['I', 'love', 'natural', 'language', 'processing']

# NLTK
import nltk
from nltk.tokenize import word_tokenize
tokens = word_tokenize("I don't like it.")
# ['I', 'do', "n't", 'like', 'it', '.']
```

### 서브워드 토큰화

서브워드는 단어를 더 작은 단위로 분리

```
"unhappiness" → ["un", "##happiness"] (WordPiece)
"unhappiness" → ["un", "happi", "ness"] (BPE)
```

**장점**:
- 미등록 단어(OOV) 처리 가능
- 어휘 크기 축소
- 형태소 정보 보존

### BPE (Byte Pair Encoding)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# BPE 토크나이저 생성
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 학습
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# 토큰화
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

# 인코딩
encoded = tokenizer.encode(text)
# [101, 1045, 2293, 3019, 2653, 6364, 102]

# 디코딩
decoded = tokenizer.decode(encoded)
# "[CLS] i love natural language processing [SEP]"
```

### SentencePiece (GPT, T5)

```python
import sentencepiece as spm

# 학습
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='spm',
    vocab_size=8000,
    model_type='bpe'
)

# 로드 및 사용
sp = spm.SentencePieceProcessor()
sp.load('spm.model')

tokens = sp.encode_as_pieces("Hello, world!")
# ['▁Hello', ',', '▁world', '!']

ids = sp.encode_as_ids("Hello, world!")
# [1234, 567, 890, 12]
```

---

## 3. 어휘 구축 (Vocabulary)

### 기본 어휘 사전

```python
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts, tokenizer):
        # 단어 빈도 계산
        for text in texts:
            tokens = tokenizer(text)
            self.word_freq.update(tokens)

        # 빈도 기준 필터링 후 추가
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

# 사용
vocab = Vocabulary(min_freq=2)
vocab.build(texts, str.split)
encoded = vocab.encode("hello world", str.split)
```

### torchtext 어휘

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

# 사용
indices = vocab(tokenizer("hello world"))
```

---

## 4. 패딩과 배치 처리

### 시퀀스 패딩

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts, labels = zip(*batch)

    # 토큰화 및 인코딩
    encoded = [torch.tensor(vocab.encode(t, tokenizer)) for t in texts]

    # 패딩 (가장 긴 시퀀스에 맞춤)
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)

    # 최대 길이 제한
    if padded.size(1) > max_len:
        padded = padded[:, :max_len]

    labels = torch.tensor(labels)
    return padded, labels

# DataLoader에 적용
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

### Attention Mask

```python
def create_attention_mask(input_ids, pad_token_id=0):
    """패딩이 아닌 위치는 1, 패딩은 0"""
    return (input_ids != pad_token_id).long()

# 예시
input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
attention_mask = create_attention_mask(input_ids)
# tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
```

---

## 5. 텍스트 정규화

### 다양한 정규화 기법

```python
import unicodedata

def normalize_text(text):
    # Unicode 정규화 (NFD → NFC)
    text = unicodedata.normalize('NFC', text)

    # 소문자 변환
    text = text.lower()

    # URL 제거
    text = re.sub(r'http\S+', '', text)

    # 이메일 제거
    text = re.sub(r'\S+@\S+', '', text)

    # 숫자 정규화 (선택)
    text = re.sub(r'\d+', '<NUM>', text)

    # 반복 문자 축소
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # "sooooo" → "soo"

    return text.strip()
```

### 불용어 제거

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

### 표제어 추출 (Lemmatization)

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

## 6. HuggingFace 토크나이저

### 기본 사용

```python
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 인코딩
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

### 배치 인코딩

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

### 특수 토큰

```python
# BERT 특수 토큰
print(tokenizer.special_tokens_map)
# {'unk_token': '[UNK]', 'sep_token': '[SEP]',
#  'pad_token': '[PAD]', 'cls_token': '[CLS]',
#  'mask_token': '[MASK]'}

# 토큰 ID
print(tokenizer.cls_token_id)  # 101
print(tokenizer.sep_token_id)  # 102
print(tokenizer.pad_token_id)  # 0
```

---

## 7. 실습: 텍스트 분류 전처리

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

# 사용
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

## 정리

### 토큰화 방법 비교

| 방법 | 장점 | 단점 | 사용 모델 |
|------|------|------|----------|
| 단어 단위 | 직관적 | OOV 문제 | 전통 NLP |
| BPE | OOV 해결 | 학습 필요 | GPT |
| WordPiece | OOV 해결 | 학습 필요 | BERT |
| SentencePiece | 언어 무관 | 학습 필요 | T5, GPT |

### 핵심 코드

```python
# HuggingFace 토크나이저
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# 어휘 구축
vocab = build_vocab_from_iterator(yield_tokens(texts), specials=['<pad>', '<unk>'])

# 패딩
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
```

---

## 다음 단계

[02_Word2Vec_GloVe.md](./02_Word2Vec_GloVe.md)에서 단어 임베딩을 학습합니다.
