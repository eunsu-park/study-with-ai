# 11. Vector Databases

## Learning Objectives

- Vector database concepts
- Using Chroma, FAISS, Pinecone
- Indexing and search optimization
- Practical usage patterns

---

## 1. Vector Database Overview

### Why Vector DB?

```
Traditional DB:
    SELECT * FROM docs WHERE text LIKE '%machine learning%'
    → Keyword matching only

Vector DB:
    query_vector = embed("What is AI?")
    SELECT * FROM docs ORDER BY similarity(vector, query_vector)
    → Semantic similarity search
```

### Major Vector DBs

| Name | Type | Features |
|------|------|----------|
| Chroma | Local/Embedded | Simple, for development |
| FAISS | Library | Fast, large-scale |
| Pinecone | Cloud | Managed, scalable |
| Weaviate | Open source | Hybrid search |
| Qdrant | Open source | Strong filtering |
| Milvus | Open source | Large-scale, distributed |

---

## 2. Chroma

### Installation and Basic Usage

```python
pip install chromadb
```

```python
import chromadb
from chromadb.utils import embedding_functions

# Create client
client = chromadb.Client()  # In-memory
# client = chromadb.PersistentClient(path="./chroma_db")  # Persistent

# Embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)
```

### Adding Documents

```python
# Add documents
collection.add(
    documents=["Document 1 text", "Document 2 text", "Document 3 text"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "a"}],
    ids=["doc1", "doc2", "doc3"]
)

# Provide embeddings directly
collection.add(
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"]
)
```

### Search

```python
# Query search
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)

print(results['documents'])  # Document content
print(results['distances'])  # Distances
print(results['metadatas'])  # Metadata

# Metadata filtering
results = collection.query(
    query_texts=["query"],
    n_results=5,
    where={"source": "a"}  # Only source "a"
)

# Complex filters
results = collection.query(
    query_texts=["query"],
    where={"$and": [{"source": "a"}, {"year": {"$gt": 2020}}]}
)
```

### LangChain Integration

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create
vectorstore = Chroma.from_texts(
    texts=["text1", "text2", "text3"],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
docs = vectorstore.similarity_search("query", k=3)

# Use as Retriever
retriever = vectorstore.as_retriever()
```

---

## 3. FAISS

### Installation and Basic Usage

```python
pip install faiss-cpu  # CPU version
# pip install faiss-gpu  # GPU version
```

```python
import faiss
import numpy as np

# Create index
dimension = 384
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add vectors
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

print(f"Total vectors: {index.ntotal}")
```

### Search

```python
# Search
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)

print(f"Indices: {indices}")
print(f"Distances: {distances}")
```

### Index Types

```python
# Flat (accurate, slow)
index = faiss.IndexFlatL2(dimension)

# IVF (approximate, fast)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(vectors)  # Training required
index.add(vectors)
index.nprobe = 10  # Number of clusters to search

# HNSW (very fast)
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
index.add(vectors)

# PQ (memory efficient)
index = faiss.IndexPQ(dimension, 8, 8)  # M=8, nbits=8
index.train(vectors)
index.add(vectors)
```

### Save/Load

```python
# Save
faiss.write_index(index, "index.faiss")

# Load
index = faiss.read_index("index.faiss")
```

### LangChain Integration

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create
vectorstore = FAISS.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings
)

# Save/Load
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## 4. Pinecone

### Installation and Setup

```python
pip install pinecone-client
```

```python
from pinecone import Pinecone, ServerlessSpec

# Create client
pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Connect to index
index = pc.Index("my-index")
```

### Adding Documents

```python
# Upsert (add/update)
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"source": "a"}},
        {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"source": "b"}},
    ]
)

# Batch upsert
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

### Search

```python
# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True
)

for match in results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")

# Metadata filtering
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"source": {"$eq": "a"}}
)
```

### LangChain Integration

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings,
    index_name="my-index"
)

# Search
docs = vectorstore.similarity_search("query", k=3)
```

---

## 5. Indexing Strategies

### Index Type Comparison

| Type | Accuracy | Speed | Memory | When to Use |
|------|----------|-------|--------|-------------|
| Flat | 100% | Slow | High | Small-scale (<100K) |
| IVF | 95%+ | Fast | Medium | Medium-scale |
| HNSW | 98%+ | Very fast | High | Large-scale, real-time |
| PQ | 90%+ | Fast | Low | Memory limited |

### Hybrid Index

```python
# IVF + PQ
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist=100,   # Number of clusters
    m=8,         # PQ segments
    nbits=8      # PQ bits
)
index.train(vectors)
index.add(vectors)
```

---

## 6. Using Metadata

### Filtering Patterns

```python
# Chroma filter syntax
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

# Supported operators
# $eq, $ne: equal, not equal
# $gt, $gte, $lt, $lte: comparison
# $in, $nin: in, not in
# $and, $or: logical operations
```

### Metadata Updates

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

## 7. Practical Patterns

### Document Management Class

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

### Incremental Updates

```python
import hashlib

def get_doc_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def upsert_documents(texts, collection):
    """Upsert with deduplication"""
    ids = [get_doc_id(t) for t in texts]

    # Check existing documents
    existing = collection.get(ids=ids)
    existing_ids = set(existing['ids'])

    # Add only new documents
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

### Batch Processing

```python
def batch_add(collection, texts, batch_size=100):
    """Add large number of documents in batches"""
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        ids = [str(uuid.uuid4()) for _ in batch]
        collection.add(documents=batch, ids=ids)
        print(f"Added {min(i + batch_size, total)}/{total}")
```

---

## 8. Performance Optimization

### Embedding Caching

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

### Index Optimization

```python
# FAISS search parameter tuning
index.nprobe = 20  # Search more clusters (accuracy ↑, speed ↓)

# Parallel search
faiss.omp_set_num_threads(4)  # Set number of threads
```

---

## Summary

### Selection Guide

| Situation | Recommendation |
|-----------|----------------|
| Development/Prototype | Chroma |
| Large-scale local | FAISS |
| Production managed | Pinecone |
| Open source self-hosted | Qdrant, Milvus |

### Key Code

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

## Next Steps

Build a conversational AI system in [12_Practical_Chatbot.md](./12_Practical_Chatbot.md).
