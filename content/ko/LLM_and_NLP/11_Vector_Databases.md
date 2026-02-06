# 11. 벡터 데이터베이스

## 학습 목표

- 벡터 데이터베이스 개념
- Chroma, FAISS, Pinecone 사용
- 인덱싱과 검색 최적화
- 실전 활용 패턴

---

## 1. 벡터 데이터베이스 개요

### 왜 벡터 DB인가?

```
전통 DB:
    SELECT * FROM docs WHERE text LIKE '%machine learning%'
    → 키워드 매칭만 가능

벡터 DB:
    query_vector = embed("What is AI?")
    SELECT * FROM docs ORDER BY similarity(vector, query_vector)
    → 의미적 유사성 검색
```

### 주요 벡터 DB

| 이름 | 타입 | 특징 |
|------|------|------|
| Chroma | 로컬/임베디드 | 간단, 개발용 |
| FAISS | 라이브러리 | 빠름, 대규모 |
| Pinecone | 클라우드 | 관리형, 확장성 |
| Weaviate | 오픈소스 | 하이브리드 검색 |
| Qdrant | 오픈소스 | 필터링 강점 |
| Milvus | 오픈소스 | 대규모, 분산 |

---

## 2. Chroma

### 설치 및 기본 사용

```python
pip install chromadb
```

```python
import chromadb
from chromadb.utils import embedding_functions

# 클라이언트 생성
client = chromadb.Client()  # 메모리
# client = chromadb.PersistentClient(path="./chroma_db")  # 영구 저장

# 임베딩 함수
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 컬렉션 생성
collection = client.create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)
```

### 문서 추가

```python
# 문서 추가
collection.add(
    documents=["Document 1 text", "Document 2 text", "Document 3 text"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "a"}],
    ids=["doc1", "doc2", "doc3"]
)

# 임베딩 직접 제공
collection.add(
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"]
)
```

### 검색

```python
# 쿼리 검색
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)

print(results['documents'])  # 문서 내용
print(results['distances'])  # 거리
print(results['metadatas'])  # 메타데이터

# 메타데이터 필터링
results = collection.query(
    query_texts=["query"],
    n_results=5,
    where={"source": "a"}  # source가 "a"인 것만
)

# 복합 필터
results = collection.query(
    query_texts=["query"],
    where={"$and": [{"source": "a"}, {"year": {"$gt": 2020}}]}
)
```

### LangChain 연동

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 생성
vectorstore = Chroma.from_texts(
    texts=["text1", "text2", "text3"],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 검색
docs = vectorstore.similarity_search("query", k=3)

# Retriever로 사용
retriever = vectorstore.as_retriever()
```

---

## 3. FAISS

### 설치 및 기본 사용

```python
pip install faiss-cpu  # CPU 버전
# pip install faiss-gpu  # GPU 버전
```

```python
import faiss
import numpy as np

# 인덱스 생성
dimension = 384
index = faiss.IndexFlatL2(dimension)  # L2 거리

# 벡터 추가
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

print(f"Total vectors: {index.ntotal}")
```

### 검색

```python
# 검색
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)

print(f"Indices: {indices}")
print(f"Distances: {distances}")
```

### 인덱스 타입

```python
# Flat (정확, 느림)
index = faiss.IndexFlatL2(dimension)

# IVF (근사, 빠름)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(vectors)  # 학습 필요
index.add(vectors)
index.nprobe = 10  # 검색할 클러스터 수

# HNSW (매우 빠름)
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
index.add(vectors)

# PQ (메모리 효율)
index = faiss.IndexPQ(dimension, 8, 8)  # M=8, nbits=8
index.train(vectors)
index.add(vectors)
```

### 저장/로드

```python
# 저장
faiss.write_index(index, "index.faiss")

# 로드
index = faiss.read_index("index.faiss")
```

### LangChain 연동

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 생성
vectorstore = FAISS.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings
)

# 저장/로드
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## 4. Pinecone

### 설치 및 설정

```python
pip install pinecone-client
```

```python
from pinecone import Pinecone, ServerlessSpec

# 클라이언트 생성
pc = Pinecone(api_key="your-api-key")

# 인덱스 생성
pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# 인덱스 연결
index = pc.Index("my-index")
```

### 문서 추가

```python
# Upsert (추가/업데이트)
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"source": "a"}},
        {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"source": "b"}},
    ]
)

# 배치 upsert
from itertools import islice

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = list(islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(islice(it, batch_size))

for batch in chunks(vectors, batch_size=100):
    index.upsert(vectors=batch)
```

### 검색

```python
# 쿼리
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True
)

for match in results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")

# 메타데이터 필터링
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"source": {"$eq": "a"}}
)
```

### LangChain 연동

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings,
    index_name="my-index"
)

# 검색
docs = vectorstore.similarity_search("query", k=3)
```

---

## 5. 인덱싱 전략

### 인덱스 타입 비교

| 타입 | 정확도 | 속도 | 메모리 | 사용 시점 |
|------|--------|------|--------|----------|
| Flat | 100% | 느림 | 높음 | 소규모 (<100K) |
| IVF | 95%+ | 빠름 | 중간 | 중규모 |
| HNSW | 98%+ | 매우 빠름 | 높음 | 대규모, 실시간 |
| PQ | 90%+ | 빠름 | 낮음 | 메모리 제한 |

### 하이브리드 인덱스

```python
# IVF + PQ
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist=100,   # 클러스터 수
    m=8,         # PQ 세그먼트 수
    nbits=8      # PQ 비트 수
)
index.train(vectors)
index.add(vectors)
```

---

## 6. 메타데이터 활용

### 필터링 패턴

```python
# Chroma 필터 문법
results = collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"category": "tech"},
            {"year": {"$gte": 2023}},
            {"author": {"$in": ["Alice", "Bob"]}}
        ]
    }
)

# 지원 연산자
# $eq, $ne: 같음, 다름
# $gt, $gte, $lt, $lte: 비교
# $in, $nin: 포함, 미포함
# $and, $or: 논리 연산
```

### 메타데이터 업데이트

```python
# Chroma
collection.update(
    ids=["doc1"],
    metadatas=[{"source": "updated"}]
)

# Pinecone
index.update(
    id="vec1",
    set_metadata={"source": "updated"}
)
```

---

## 7. 실전 패턴

### 문서 관리 클래스

```python
class VectorStore:
    def __init__(self, persist_dir="./db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn
        )

    def add_documents(self, texts, metadatas=None, ids=None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def search(self, query, k=5, where=None):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )
        return results

    def delete(self, ids):
        self.collection.delete(ids=ids)
```

### 증분 업데이트

```python
import hashlib

def get_doc_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def upsert_documents(texts, collection):
    """중복 방지 업서트"""
    ids = [get_doc_id(t) for t in texts]

    # 기존 문서 확인
    existing = collection.get(ids=ids)
    existing_ids = set(existing['ids'])

    # 새 문서만 추가
    new_texts = []
    new_ids = []
    for text, doc_id in zip(texts, ids):
        if doc_id not in existing_ids:
            new_texts.append(text)
            new_ids.append(doc_id)

    if new_texts:
        collection.add(documents=new_texts, ids=new_ids)

    return len(new_texts)
```

### 배치 처리

```python
def batch_add(collection, texts, batch_size=100):
    """대량 문서 배치 추가"""
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        ids = [str(uuid.uuid4()) for _ in batch]
        collection.add(documents=batch, ids=ids)
        print(f"Added {min(i + batch_size, total)}/{total}")
```

---

## 8. 성능 최적화

### 임베딩 캐싱

```python
import pickle
import os

class CachedEmbeddings:
    def __init__(self, model, cache_dir="./embed_cache"):
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def embed(self, text):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        embedding = self.model.encode(text)

        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding
```

### 인덱스 최적화

```python
# FAISS 검색 파라미터 튜닝
index.nprobe = 20  # 더 많은 클러스터 검색 (정확도 ↑, 속도 ↓)

# 병렬 검색
faiss.omp_set_num_threads(4)  # 스레드 수 설정
```

---

## 정리

### 선택 가이드

| 상황 | 추천 |
|------|------|
| 개발/프로토타입 | Chroma |
| 대규모 로컬 | FAISS |
| 프로덕션 관리형 | Pinecone |
| 오픈소스 셀프호스트 | Qdrant, Milvus |

### 핵심 코드

```python
# Chroma
collection = client.create_collection("name")
collection.add(documents=texts, ids=ids)
results = collection.query(query_texts=["query"], n_results=5)

# FAISS
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
distances, indices = index.search(query, k=5)

# LangChain
vectorstore = Chroma.from_texts(texts, embeddings)
docs = vectorstore.similarity_search("query", k=3)
```

---

## 다음 단계

[12_Practical_Chatbot.md](./12_Practical_Chatbot.md)에서 대화형 AI 시스템을 구축합니다.
