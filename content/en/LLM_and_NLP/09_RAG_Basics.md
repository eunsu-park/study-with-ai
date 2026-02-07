# 09. RAG Basics

## Learning Objectives

- Understanding RAG (Retrieval-Augmented Generation)
- Document embedding and retrieval
- Chunking strategies
- Implementing RAG pipelines

---

## 1. RAG Overview

### Why RAG?

```
LLM Limitations:
- No knowledge of information after training (knowledge cutoff)
- Hallucination (generating incorrect information)
- Lack of specific domain knowledge

RAG Solution:
- Generate answers after retrieving external knowledge
- Can reflect latest information
- Improved trustworthiness by providing sources
```

### RAG Architecture

```
┌─────────────────────────────────────────────────────┐
│                     RAG Pipeline                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│   Question ──▶ Embedding ──▶ Vector Search ──▶ Relevant Docs │
│                           │                          │
│                           ▼                          │
│               Question + Docs ──▶ LLM ──▶ Answer    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 2. Document Preprocessing

### Chunking

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

# Usage
text = "Very long document text here..."
chunks = chunk_text(text, chunk_size=500, overlap=100)
```

### Sentence-based Chunking

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

### Semantic Chunking

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

## 3. Embedding Generation

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Hello world", "How are you?"]
embeddings = model.encode(texts)

print(embeddings.shape)  # (2, 384)
```

### HuggingFace Embeddings

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

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [r.embedding for r in response.data]
```

---

## 4. Vector Search

### Cosine Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search(query_embedding, document_embeddings, top_k=5):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# Usage
query_emb = model.encode(["What is machine learning?"])[0]
doc_embs = model.encode(documents)

indices, scores = search(query_emb, doc_embs, top_k=3)
```

### Using FAISS

```python
import faiss
import numpy as np

# Create index
dimension = 384  # Embedding dimension
index = faiss.IndexFlatIP(dimension)  # Inner Product (requires normalization for cosine similarity)

# Add after normalization
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Search
query_emb = model.encode(["query"])[0].astype('float32').reshape(1, -1)
faiss.normalize_L2(query_emb)

distances, indices = index.search(query_emb, k=5)
```

---

## 5. Simple RAG Implementation

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
        """Add documents and generate embeddings"""
        self.documents.extend(documents)
        self.embeddings = self.embed_model.encode(self.documents)

    def search(self, query, top_k=3):
        """Search relevant documents"""
        query_emb = self.embed_model.encode([query])[0]

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate(self, query, top_k=3):
        """Generate RAG answer"""
        # Search
        relevant_docs = self.search(query, top_k)
        context = "\n\n".join(relevant_docs)

        # Construct prompt
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

        # Call LLM
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

# Usage
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

## 6. Advanced RAG Techniques

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

        # BM25 (keyword search)
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Embeddings (semantic search)
        self.embeddings = model.encode(documents)

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = bm25_scores / bm25_scores.max()  # Normalize

        # Embedding scores
        query_emb = model.encode([query])[0]
        embed_scores = cosine_similarity([query_emb], self.embeddings)[0]

        # Combine
        combined = alpha * embed_scores + (1 - alpha) * bm25_scores

        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### Query Expansion

```python
def expand_query(query, llm_client):
    """Improve search performance with query expansion"""
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
        # Stage 1: Initial search (many candidates)
        initial_results = self.search(query, top_k=20)

        # Stage 2: Reranking
        pairs = [[query, doc] for doc in initial_results]
        scores = self.reranker.predict(pairs)

        # Select top k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [initial_results[i] for i in top_indices]
```

### Multi-Query RAG

```python
def multi_query_rag(question, rag, num_queries=3):
    """Search with queries from multiple perspectives"""
    # Generate diverse queries
    prompt = f"""Generate {num_queries} different search queries for:
    Question: {question}

    Queries:"""

    queries = generate_queries(prompt)

    # Search with each query
    all_docs = set()
    for q in queries:
        docs = rag.search(q, top_k=3)
        all_docs.update(docs)

    return list(all_docs)
```

---

## 7. Chunking Strategy Comparison

| Strategy | Advantages | Disadvantages | When to Use |
|----------|-----------|---------------|-------------|
| Fixed size | Simple implementation | Context breaks | General text |
| Sentence-based | Semantic units | Uneven lengths | Structured text |
| Semantic | Preserves meaning | Computation cost | Quality required |
| Hierarchical | Multi-level search | Complex | Long documents |

---

## 8. Evaluation Metrics

### Retrieval Evaluation

```python
def calculate_recall_at_k(retrieved, relevant, k):
    """Calculate Recall@K"""
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

### Generation Evaluation

```python
# Using RAGAS library
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## Summary

### RAG Checklist

```
□ Choose appropriate chunk size
□ Select embedding model (consider domain)
□ Tune retrieval top-k
□ Optimize prompts
□ Set evaluation metrics
```

### Key Code

```python
# Embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Search
query_emb = model.encode([query])[0]
similarities = cosine_similarity([query_emb], embeddings)

# Generation
context = "\n".join(relevant_docs)
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
```

---

## Next Steps

Learn about the LangChain framework in [10_LangChain_Basics.md](./10_LangChain_Basics.md).
