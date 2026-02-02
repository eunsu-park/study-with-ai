# 09. RAG 기초

## 학습 목표

- RAG (Retrieval-Augmented Generation) 이해
- 문서 임베딩과 검색
- 청킹 전략
- RAG 파이프라인 구현

---

## 1. RAG 개요

### 왜 RAG인가?

```
LLM의 한계:
- 학습 데이터 이후 정보 모름 (지식 컷오프)
- 환각 (잘못된 정보 생성)
- 특정 도메인 지식 부족

RAG 해결책:
- 외부 지식 검색 후 답변 생성
- 최신 정보 반영 가능
- 출처 제공으로 신뢰성 향상
```

### RAG 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                     RAG Pipeline                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│   질문 ──▶ 임베딩 ──▶ 벡터 검색 ──▶ 관련 문서      │
│                           │                          │
│                           ▼                          │
│               질문 + 문서 ──▶ LLM ──▶ 답변          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 2. 문서 전처리

### 청킹 (Chunking)

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """텍스트를 오버랩이 있는 청크로 분할"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

# 사용
text = "Very long document text here..."
chunks = chunk_text(text, chunk_size=500, overlap=100)
```

### 문장 기반 청킹

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def chunk_by_sentences(text, max_sentences=5, overlap_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences - overlap_sentences):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks
```

### 시맨틱 청킹

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

chunks = splitter.split_text(text)
```

---

## 3. 임베딩 생성

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 임베딩 생성
texts = ["Hello world", "How are you?"]
embeddings = model.encode(texts)

print(embeddings.shape)  # (2, 384)
```

### HuggingFace 임베딩

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()
```

### OpenAI 임베딩

```python
from openai import OpenAI

client = OpenAI()

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [r.embedding for r in response.data]
```

---

## 4. 벡터 검색

### 코사인 유사도

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search(query_embedding, document_embeddings, top_k=5):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# 사용
query_emb = model.encode(["What is machine learning?"])[0]
doc_embs = model.encode(documents)

indices, scores = search(query_emb, doc_embs, top_k=3)
```

### FAISS 사용

```python
import faiss
import numpy as np

# 인덱스 생성
dimension = 384  # 임베딩 차원
index = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도용 정규화 필요)

# 정규화 후 추가
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 검색
query_emb = model.encode(["query"])[0].astype('float32').reshape(1, -1)
faiss.normalize_L2(query_emb)

distances, indices = index.search(query_emb, k=5)
```

---

## 5. 간단한 RAG 구현

```python
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

class SimpleRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embed_model = SentenceTransformer(embedding_model)
        self.client = OpenAI()
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        """문서 추가 및 임베딩"""
        self.documents.extend(documents)
        self.embeddings = self.embed_model.encode(self.documents)

    def search(self, query, top_k=3):
        """관련 문서 검색"""
        query_emb = self.embed_model.encode([query])[0]

        # 코사인 유사도
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate(self, query, top_k=3):
        """RAG 답변 생성"""
        # 검색
        relevant_docs = self.search(query, top_k)
        context = "\n\n".join(relevant_docs)

        # 프롬프트 구성
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

        # LLM 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

# 사용
rag = SimpleRAG()
rag.add_documents([
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
    "RAG combines retrieval with generation."
])

answer = rag.generate("What is RAG?")
print(answer)
```

---

## 6. 고급 RAG 기법

### Hybrid Search

```python
from rank_bm25 import BM25Okapi

class HybridRAG:
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents

        # BM25 (키워드 검색)
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # 임베딩 (시맨틱 검색)
        self.embeddings = model.encode(documents)

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        # BM25 점수
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = bm25_scores / bm25_scores.max()  # 정규화

        # 임베딩 점수
        query_emb = model.encode([query])[0]
        embed_scores = cosine_similarity([query_emb], self.embeddings)[0]

        # 결합
        combined = alpha * embed_scores + (1 - alpha) * bm25_scores

        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### Query Expansion

```python
def expand_query(query, llm_client):
    """쿼리 확장으로 검색 성능 향상"""
    prompt = f"""Generate 3 alternative versions of this search query:
    Original: {query}

    Alternatives:
    1."""

    response = llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    expanded = [query] + parse_alternatives(response.choices[0].message.content)
    return expanded
```

### Reranking

```python
from sentence_transformers import CrossEncoder

class RAGWithReranker:
    def __init__(self):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search_and_rerank(self, query, candidates, top_k=3):
        # 1단계: 초기 검색 (후보 많이)
        initial_results = self.search(query, top_k=20)

        # 2단계: 리랭킹
        pairs = [[query, doc] for doc in initial_results]
        scores = self.reranker.predict(pairs)

        # 상위 k개 선택
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [initial_results[i] for i in top_indices]
```

### Multi-Query RAG

```python
def multi_query_rag(question, rag, num_queries=3):
    """여러 관점의 쿼리로 검색"""
    # 다양한 쿼리 생성
    prompt = f"""Generate {num_queries} different search queries for:
    Question: {question}

    Queries:"""

    queries = generate_queries(prompt)

    # 각 쿼리로 검색
    all_docs = set()
    for q in queries:
        docs = rag.search(q, top_k=3)
        all_docs.update(docs)

    return list(all_docs)
```

---

## 7. 청킹 전략 비교

| 전략 | 장점 | 단점 | 사용 시점 |
|------|------|------|----------|
| 고정 크기 | 구현 간단 | 문맥 단절 | 일반적인 텍스트 |
| 문장 기반 | 의미 단위 | 길이 불균일 | 구조화된 텍스트 |
| 시맨틱 | 의미 보존 | 계산 비용 | 고품질 필요 |
| 계층적 | 다단계 검색 | 복잡함 | 긴 문서 |

---

## 8. 평가 메트릭

### 검색 평가

```python
def calculate_recall_at_k(retrieved, relevant, k):
    """Recall@K 계산"""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)

def calculate_mrr(retrieved, relevant):
    """Mean Reciprocal Rank"""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0
```

### 생성 평가

```python
# RAGAS 라이브러리 사용
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## 정리

### RAG 체크리스트

```
□ 적절한 청킹 크기 선택
□ 임베딩 모델 선택 (도메인 고려)
□ 검색 top-k 튜닝
□ 프롬프트 최적화
□ 평가 메트릭 설정
```

### 핵심 코드

```python
# 임베딩
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# 검색
query_emb = model.encode([query])[0]
similarities = cosine_similarity([query_emb], embeddings)

# 생성
context = "\n".join(relevant_docs)
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
```

---

## 다음 단계

[10_LangChain_Basics.md](./10_LangChain_Basics.md)에서 LangChain 프레임워크를 학습합니다.
