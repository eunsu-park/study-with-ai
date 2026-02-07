# 23. Advanced RAG

## Overview

This lesson covers more sophisticated retrieval and generation strategies beyond basic RAG. We explore Agentic RAG, Multi-hop Reasoning, HyDE, RAPTOR, and other cutting-edge techniques.

---

## 1. RAG Limitations and Advanced Techniques

### 1.1 Limitations of Basic RAG

```
Basic RAG Problems:
┌─────────────────────────────────────────────────────────┐
│  1. Single Retrieval Limitation                         │
│     - Single search insufficient for complex questions  │
│     - Multi-step reasoning required                     │
│                                                         │
│  2. Retrieval-Question Mismatch                         │
│     - Style difference between questions and documents  │
│     - Limitations of embedding similarity               │
│                                                         │
│  3. Context Length Constraint                           │
│     - Difficult to handle many relevant documents       │
│     - Possible omission of important information        │
│                                                         │
│  4. Freshness/Accuracy                                  │
│     - Outdated information                              │
│     - Difficult to verify reliability                   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Advanced RAG Technique Classification

```
Advanced RAG Techniques:
┌─────────────────────────────────────────────────────────┐
│  Pre-Retrieval                                          │
│  ├── Query Transformation (HyDE, Query Expansion)       │
│  └── Query Routing                                      │
│                                                         │
│  Retrieval                                              │
│  ├── Hybrid Search (Dense + Sparse)                     │
│  ├── Multi-step Retrieval                               │
│  └── Hierarchical Retrieval (RAPTOR)                    │
│                                                         │
│  Post-Retrieval                                         │
│  ├── Reranking                                          │
│  ├── Context Compression                                │
│  └── Self-Reflection                                    │
│                                                         │
│  Generation                                             │
│  ├── Chain-of-Thought RAG                               │
│  └── Agentic RAG                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Query Transformation

### 2.1 HyDE (Hypothetical Document Embeddings)

```
HyDE Idea:
┌─────────────────────────────────────────────────────────┐
│  Query: "What is the capital of France?"                │
│                                                         │
│  Traditional: Search directly with query embedding      │
│        (question ↔ document style difference)           │
│                                                         │
│  HyDE: Generate hypothetical document with LLM then     │
│        search                                           │
│        Query → "Paris is the capital of France..."      │
│        → Search with this hypothetical document's       │
│          embedding                                      │
│        (document ↔ document style match)                │
└─────────────────────────────────────────────────────────┘
```

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

class HyDERetriever:
    """HyDE Retriever"""

    def __init__(self, llm, embeddings, vectorstore):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document"""
        prompt = f"""Write a short passage that would answer the following question.
The passage should be factual and informative.

Question: {query}

Passage:"""

        response = self.llm.invoke(prompt)
        return response

    def retrieve(self, query: str, k: int = 5) -> list:
        """HyDE retrieval"""
        # 1. Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query)

        # 2. Embed hypothetical document
        doc_embedding = self.embeddings.embed_query(hypothetical_doc)

        # 3. Search for similar documents
        results = self.vectorstore.similarity_search_by_vector(
            doc_embedding, k=k
        )

        return results


# LangChain built-in HyDE
def setup_hyde_chain():
    base_embeddings = OpenAIEmbeddings()
    llm = OpenAI(temperature=0)

    embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm, base_embeddings, "web_search"
    )

    return embeddings
```

### 2.2 Query Expansion

```python
class QueryExpander:
    """Query expansion"""

    def __init__(self, llm):
        self.llm = llm

    def expand_query(self, query: str, num_variations: int = 3) -> list:
        """Expand query into multiple variations"""
        prompt = f"""Generate {num_variations} different versions of the following question.
Each version should ask the same thing but use different words or perspectives.

Original question: {query}

Variations:
1."""

        response = self.llm.invoke(prompt)

        # Parse
        variations = [query]  # Include original
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # "1. question" format
                variation = line.split(".", 1)[-1].strip()
                variations.append(variation)

        return variations[:num_variations + 1]

    def retrieve_with_expansion(
        self,
        query: str,
        retriever,
        k: int = 5
    ) -> list:
        """Retrieve with expanded queries"""
        variations = self.expand_query(query)

        all_docs = []
        seen = set()

        for variation in variations:
            docs = retriever.get_relevant_documents(variation)
            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    seen.add(doc_id)
                    all_docs.append(doc)

        # Return top k (sorted by RRF or other method)
        return all_docs[:k]
```

---

## 3. Agentic RAG

### 3.1 Concept

```
Agentic RAG:
┌─────────────────────────────────────────────────────────┐
│  LLM Agent dynamically uses retrieval tools             │
│                                                         │
│  Agent Loop:                                            │
│  1. Analyze question                                    │
│  2. Determine needed information                        │
│  3. Call retrieval tools (optional, repeatable)         │
│  4. Evaluate results                                    │
│  5. Need more search? → Repeat                          │
│  6. Generate final answer                               │
│                                                         │
│  vs Basic RAG:                                          │
│  Query → Retrieve → Generate (fixed pipeline)           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Implementation

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

class AgenticRAG:
    """Agentic RAG System"""

    def __init__(self, llm, vectorstore, web_search=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.web_search = web_search

        self.tools = self._setup_tools()
        self.agent = self._create_agent()

    def _setup_tools(self) -> list:
        """Setup tools"""
        tools = [
            Tool(
                name="search_knowledge_base",
                func=self._search_kb,
                description="Search the internal knowledge base for relevant information. Use this for company-specific or domain-specific questions."
            ),
            Tool(
                name="search_web",
                func=self._search_web,
                description="Search the web for current information. Use this for recent events or general knowledge."
            ),
            Tool(
                name="lookup_specific",
                func=self._lookup_specific,
                description="Look up specific facts or definitions. Use this when you need precise information."
            )
        ]
        return tools

    def _search_kb(self, query: str) -> str:
        """Search knowledge base"""
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    def _search_web(self, query: str) -> str:
        """Web search (requires external API)"""
        if self.web_search:
            return self.web_search.run(query)
        return "Web search not available."

    def _lookup_specific(self, query: str) -> str:
        """Specific information lookup"""
        docs = self.vectorstore.similarity_search(query, k=1)
        if docs:
            return docs[0].page_content
        return "No specific information found."

    def _create_agent(self):
        """Create ReAct Agent"""
        prompt = PromptTemplate.from_template("""Answer the following question using the available tools.
Think step by step about what information you need.

Question: {input}

You have access to these tools:
{tools}

Use the following format:
Thought: What do I need to find out?
Action: tool_name
Action Input: the input to the tool
Observation: the result of the tool
... (repeat as needed)
Thought: I now have enough information
Final Answer: the final answer

Begin!

{agent_scratchpad}""")

        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def query(self, question: str) -> str:
        """Process question"""
        result = self.agent.invoke({"input": question})
        return result["output"]


# Usage example
def agentic_rag_example():
    from langchain.llms import OpenAI
    from langchain.vectorstores import Chroma

    llm = OpenAI(temperature=0)
    vectorstore = Chroma(...)  # Setup required

    rag = AgenticRAG(llm, vectorstore)

    # Complex question
    answer = rag.query(
        "Compare our company's revenue growth in 2023 with the industry average"
    )
    print(answer)
```

---

## 4. Multi-hop Reasoning

### 4.1 Concept

```
Multi-hop Reasoning:
┌─────────────────────────────────────────────────────────┐
│  Question: "What is the population of Biden's birthplace?" │
│                                                         │
│  Hop 1: "What is Biden's birthplace?" → "Scranton, PA" │
│  Hop 2: "What is Scranton's population?" → "76,328"    │
│                                                         │
│  Final Answer: "76,328"                                 │
│                                                         │
│  Single search cannot directly find the answer          │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
class MultiHopRAG:
    """Multi-hop Reasoning RAG"""

    def __init__(self, llm, retriever, max_hops: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.max_hops = max_hops

    def decompose_question(self, question: str) -> list:
        """Decompose question into sub-questions"""
        prompt = f"""Break down the following complex question into simpler sub-questions.
Each sub-question should be answerable independently.

Question: {question}

Sub-questions (one per line):"""

        response = self.llm.invoke(prompt)
        sub_questions = [q.strip() for q in response.split("\n") if q.strip()]
        return sub_questions

    def answer_with_hops(self, question: str) -> dict:
        """Answer with multi-step reasoning"""
        reasoning_chain = []
        context = ""

        for hop in range(self.max_hops):
            # Determine next query based on current context
            if hop == 0:
                current_query = question
            else:
                current_query = self._generate_follow_up(
                    question, context, reasoning_chain
                )

            if current_query is None:
                break

            # Retrieve
            docs = self.retriever.get_relevant_documents(current_query)
            new_context = "\n".join([doc.page_content for doc in docs])

            # Generate intermediate answer
            intermediate_answer = self._generate_intermediate_answer(
                current_query, new_context
            )

            reasoning_chain.append({
                "hop": hop + 1,
                "query": current_query,
                "answer": intermediate_answer
            })

            context += f"\n{intermediate_answer}"

            # Check if enough information
            if self._has_enough_info(question, context):
                break

        # Final answer
        final_answer = self._generate_final_answer(question, reasoning_chain)

        return {
            "question": question,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer
        }

    def _generate_follow_up(self, original_q, context, chain) -> str:
        """Generate follow-up question"""
        chain_text = "\n".join([
            f"Q: {step['query']}\nA: {step['answer']}"
            for step in chain
        ])

        prompt = f"""Based on the original question and what we've learned so far,
what additional information do we need?

Original question: {original_q}

What we've found:
{chain_text}

If we have enough information to answer, respond with "DONE".
Otherwise, provide the next question to search for:"""

        response = self.llm.invoke(prompt)

        if "DONE" in response.upper():
            return None
        return response.strip()

    def _generate_intermediate_answer(self, query, context) -> str:
        """Generate intermediate answer"""
        prompt = f"""Based on the following context, answer the question briefly.

Context: {context}

Question: {query}

Answer:"""

        return self.llm.invoke(prompt)

    def _has_enough_info(self, question, context) -> bool:
        """Check if there's enough information"""
        prompt = f"""Can you answer the following question based on this information?

Question: {question}
Information: {context}

Answer YES or NO:"""

        response = self.llm.invoke(prompt)
        return "YES" in response.upper()

    def _generate_final_answer(self, question, chain) -> str:
        """Generate final answer"""
        chain_text = "\n".join([
            f"Step {step['hop']}: {step['query']} → {step['answer']}"
            for step in chain
        ])

        prompt = f"""Based on the reasoning chain below, provide a final answer.

Question: {question}

Reasoning:
{chain_text}

Final Answer:"""

        return self.llm.invoke(prompt)
```

---

## 5. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

### 5.1 Concept

```
RAPTOR Structure:
┌─────────────────────────────────────────────────────────┐
│  Level 3 (Highest-level summary)                        │
│  ┌──────────────────────────────────┐                   │
│  │     Abstract Summary              │                  │
│  └──────────────────────────────────┘                   │
│              ↑                                          │
│  Level 2 (Cluster summaries)                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Summary1 │    │ Summary2 │    │ Summary3 │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│      ↑   ↑          ↑   ↑          ↑   ↑              │
│  Level 1 (Chunk clustering)                             │
│  [C1][C2][C3]    [C4][C5][C6]    [C7][C8][C9]          │
│      ↑   ↑   ↑      ↑   ↑   ↑      ↑   ↑   ↑          │
│  Level 0 (Original chunks)                              │
│  [Chunk1][Chunk2]...[ChunkN]                           │
└─────────────────────────────────────────────────────────┘

Retrieval: Search across multiple levels simultaneously to get information at various abstraction levels
```

### 5.2 Implementation

```python
from sklearn.cluster import KMeans
import numpy as np

class RAPTOR:
    """RAPTOR hierarchical retrieval"""

    def __init__(self, llm, embeddings, num_levels: int = 3):
        self.llm = llm
        self.embeddings = embeddings
        self.num_levels = num_levels
        self.tree = {}

    def build_tree(self, documents: list, cluster_size: int = 5):
        """Build RAPTOR tree"""
        # Level 0: Original chunks
        self.tree[0] = documents
        current_docs = documents

        for level in range(1, self.num_levels):
            # Compute embeddings
            texts = [doc.page_content for doc in current_docs]
            embeddings = self.embeddings.embed_documents(texts)

            # Clustering
            n_clusters = max(len(current_docs) // cluster_size, 1)
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(embeddings)

            # Summarize each cluster
            summaries = []
            for cluster_id in range(n_clusters):
                cluster_docs = [
                    doc for doc, c in zip(current_docs, clusters)
                    if c == cluster_id
                ]
                summary = self._summarize_cluster(cluster_docs)
                summaries.append(summary)

            self.tree[level] = summaries
            current_docs = summaries

    def _summarize_cluster(self, docs: list) -> str:
        """Summarize cluster"""
        combined_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Summarize the following texts into a concise summary that captures the key information.

Texts:
{combined_text}

Summary:"""

        summary = self.llm.invoke(prompt)

        # Wrap as Document object
        from langchain.schema import Document
        return Document(page_content=summary)

    def retrieve(self, query: str, k_per_level: int = 2) -> list:
        """Hierarchical retrieval"""
        all_results = []

        for level, docs in self.tree.items():
            # Search at each level
            texts = [doc.page_content for doc in docs]
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self.embeddings.embed_documents(texts)

            # Cosine similarity
            similarities = np.dot(doc_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-k_per_level:]

            for idx in top_indices:
                all_results.append({
                    "level": level,
                    "document": docs[idx],
                    "score": similarities[idx]
                })

        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results
```

---

## 6. ColBERT (Contextualized Late Interaction)

### 6.1 Concept

```
ColBERT vs Dense Retrieval:
┌─────────────────────────────────────────────────────────┐
│  Dense Retrieval (bi-encoder):                          │
│  Query → [CLS] embedding                                │
│  Doc   → [CLS] embedding                                │
│  Score = dot(query_emb, doc_emb)                        │
│  Problem: Hard to represent complex meaning in single   │
│           vector                                        │
│                                                         │
│  ColBERT (late interaction):                            │
│  Query → [q1, q2, ..., qn] (per-token embeddings)       │
│  Doc   → [d1, d2, ..., dm] (per-token embeddings)       │
│  Score = Σᵢ maxⱼ sim(qᵢ, dⱼ)                           │
│  Advantage: More precise search through token-level     │
│             matching                                    │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Usage

```python
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

class ColBERTRetriever:
    """ColBERT retriever"""

    def __init__(self, index_name: str = "my_index"):
        self.index_name = index_name
        self.config = ColBERTConfig(
            nbits=2,
            doc_maxlen=300,
            query_maxlen=32
        )

    def build_index(self, documents: list, collection_path: str):
        """Build index"""
        # Save documents to file
        with open(collection_path, 'w') as f:
            for doc in documents:
                f.write(doc + "\n")

        with Run().context(RunConfig(nranks=1)):
            indexer = Indexer(
                checkpoint="colbert-ir/colbertv2.0",
                config=self.config
            )
            indexer.index(
                name=self.index_name,
                collection=collection_path
            )

    def search(self, query: str, k: int = 10) -> list:
        """Search"""
        with Run().context(RunConfig(nranks=1)):
            searcher = Searcher(index=self.index_name)
            results = searcher.search(query, k=k)

        return results


# RAGatouille (easier ColBERT wrapper)
def colbert_with_ragatouille():
    from ragatouille import RAGPretrainedModel

    rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # Indexing
    rag.index(
        collection=[
            "Document 1 content...",
            "Document 2 content..."
        ],
        index_name="my_index"
    )

    # Search
    results = rag.search("my query", k=5)
    return results
```

---

## 7. Self-RAG (Self-Reflective RAG)

```python
class SelfRAG:
    """Self-RAG: Self-reflective RAG"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def query(self, question: str) -> dict:
        """Self-RAG query"""
        # 1. Assess retrieval need
        needs_retrieval = self._assess_retrieval_need(question)

        if not needs_retrieval:
            # Answer directly without retrieval
            answer = self._generate_without_retrieval(question)
            return {"answer": answer, "retrieval_used": False}

        # 2. Retrieve
        docs = self.retriever.get_relevant_documents(question)

        # 3. Evaluate relevance (per document)
        relevant_docs = []
        for doc in docs:
            if self._is_relevant(question, doc):
                relevant_docs.append(doc)

        # 4. Generate answer
        answer = self._generate_with_context(question, relevant_docs)

        # 5. Evaluate answer quality
        is_supported = self._check_support(answer, relevant_docs)
        is_useful = self._check_usefulness(question, answer)

        # 6. Retry if needed
        if not is_supported or not is_useful:
            answer = self._refine_answer(question, relevant_docs, answer)

        return {
            "answer": answer,
            "retrieval_used": True,
            "relevant_docs": relevant_docs,
            "is_supported": is_supported,
            "is_useful": is_useful
        }

    def _assess_retrieval_need(self, question: str) -> bool:
        """Assess retrieval need"""
        prompt = f"""Determine if external knowledge is needed to answer this question.

Question: {question}

Answer YES if retrieval is needed, NO if you can answer from general knowledge:"""

        response = self.llm.invoke(prompt)
        return "YES" in response.upper()

    def _is_relevant(self, question: str, doc) -> bool:
        """Evaluate document relevance"""
        prompt = f"""Is this document relevant to the question?

Question: {question}
Document: {doc.page_content[:500]}

Answer RELEVANT or IRRELEVANT:"""

        response = self.llm.invoke(prompt)
        return "RELEVANT" in response.upper()

    def _check_support(self, answer: str, docs: list) -> bool:
        """Check if answer is supported by documents"""
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""Is this answer supported by the given context?

Context: {context}
Answer: {answer}

Respond SUPPORTED or NOT_SUPPORTED:"""

        response = self.llm.invoke(prompt)
        return "SUPPORTED" in response.upper()

    def _check_usefulness(self, question: str, answer: str) -> bool:
        """Check answer usefulness"""
        prompt = f"""Does this answer actually address the question?

Question: {question}
Answer: {answer}

Respond USEFUL or NOT_USEFUL:"""

        response = self.llm.invoke(prompt)
        return "USEFUL" in response.upper()
```

---

## Key Summary

### Advanced RAG Techniques
```
1. HyDE: Improve retrieval quality with hypothetical documents
2. Query Expansion: Search with diverse queries
3. Agentic RAG: Dynamic retrieval by LLM Agent
4. Multi-hop: Multi-step reasoning
5. RAPTOR: Hierarchical summary tree
6. ColBERT: Token-level late interaction
7. Self-RAG: Self-reflection and verification
```

### Selection Guide
```
Simple QA → Basic RAG
Complex questions → Multi-hop + Agentic
Long documents → RAPTOR
Precise search → ColBERT
Quality critical → Self-RAG
```

---

## References

1. Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)
2. Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
3. Khattab et al. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
4. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
