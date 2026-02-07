# LLM & NLP Learning Guide

## Introduction

This folder contains materials for learning Natural Language Processing (NLP) and Large Language Models (LLM). It is structured step-by-step from basic NLP to modern LLM applications.

**Target Audience**: Learners who have completed the Deep_Learning folder (understanding of Transformer and Attention is required)

---

## Learning Roadmap

```
[NLP Basics]              [Pre-trained Models]         [LLM Applications]
    │                          │                          │
    ▼                          ▼                          ▼
Tokenization/Embedding ─▶ BERT Understanding ─▶ Prompt Engineering
    │                          │                          │
    ▼                          ▼                          ▼
Word2Vec/GloVe ─────────▶ GPT Understanding ──▶ RAG Systems
    │                          │                          │
    ▼                          ▼                          ▼
Transformer Review ─────▶ HuggingFace ────────▶ LangChain
                               │                          │
                               ▼                          ▼
                          Fine-Tuning ────────▶ Practical Chatbot
```

---

## File List

### NLP Basics

| File | Difficulty | Key Topics |
|------|------------|------------|
| [01_NLP_Basics.md](./01_NLP_Basics.md) | ⭐⭐ | Tokenization, normalization, vocabulary building |
| [02_Word2Vec_GloVe.md](./02_Word2Vec_GloVe.md) | ⭐⭐ | Word embeddings, Skip-gram, CBOW |
| [03_Transformer_Review.md](./03_Transformer_Review.md) | ⭐⭐⭐ | Attention, Encoder-Decoder |

### Pre-trained Models

| File | Difficulty | Key Topics |
|------|------------|------------|
| [04_BERT_Understanding.md](./04_BERT_Understanding.md) | ⭐⭐⭐ | MLM, NSP, bidirectional encoder |
| [05_GPT_Understanding.md](./05_GPT_Understanding.md) | ⭐⭐⭐ | Autoregressive model, text generation |
| [06_HuggingFace_Basics.md](./06_HuggingFace_Basics.md) | ⭐⭐ | Transformers library, Pipeline |
| [07_Fine_Tuning.md](./07_Fine_Tuning.md) | ⭐⭐⭐ | Classification, QA, summarization fine-tuning |

### LLM Applications

| File | Difficulty | Key Topics |
|------|------------|------------|
| [08_Prompt_Engineering.md](./08_Prompt_Engineering.md) | ⭐⭐ | Prompt design, Few-shot, CoT |
| [09_RAG_Basics.md](./09_RAG_Basics.md) | ⭐⭐⭐ | Retrieval-Augmented Generation, chunking strategies |
| [10_LangChain_Basics.md](./10_LangChain_Basics.md) | ⭐⭐⭐ | Chains, agents, memory |
| [11_Vector_Databases.md](./11_Vector_Databases.md) | ⭐⭐⭐ | Chroma, Pinecone, FAISS |
| [12_Practical_Chatbot.md](./12_Practical_Chatbot.md) | ⭐⭐⭐⭐ | Building conversational AI systems |

### Advanced LLM

| File | Difficulty | Key Topics |
|------|------------|------------|
| [13_Model_Quantization.md](./13_Model_Quantization.md) | ⭐⭐⭐ | INT8/INT4, GPTQ, AWQ, bitsandbytes, QLoRA |
| [14_RLHF_Alignment.md](./14_RLHF_Alignment.md) | ⭐⭐⭐⭐ | PPO, Reward Model, DPO, Constitutional AI |
| [15_LLM_Agents.md](./15_LLM_Agents.md) | ⭐⭐⭐⭐ | ReAct, Tool Use, AutoGPT, LangChain Agent |
| [16_Evaluation_Metrics.md](./16_Evaluation_Metrics.md) | ⭐⭐⭐ | BLEU, ROUGE, BERTScore, Human Eval, Benchmarks |

---

## Key Concepts Preview

### NLP Pipeline

```python
# Basic NLP Pipeline
Text → Tokenization → Embedding → Model → Output

# HuggingFace Pipeline
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
```

### BERT vs GPT

| Item | BERT | GPT |
|------|------|-----|
| Direction | Bidirectional (encoder) | Unidirectional (decoder) |
| Training | MLM + NSP | Next token prediction |
| Use Cases | Classification, QA, NER | Generation, dialogue |
| Features | Context understanding | Text generation |

### RAG System

```
Question → Retrieval (Vector DB) → Relevant Docs → LLM + Docs → Answer
```

---

## Prerequisites

- Deep_Learning folder (required)
  - Attention mechanism
  - Transformer architecture
  - Text classification basics
- Advanced Python
- PyTorch basics

---

## Environment Setup

### Required Packages

```bash
# PyTorch
pip install torch torchvision torchaudio

# HuggingFace
pip install transformers datasets tokenizers accelerate

# LangChain
pip install langchain langchain-community langchain-openai

# Vector Databases
pip install chromadb faiss-cpu sentence-transformers

# Others
pip install openai tiktoken numpy pandas
```

### API Key Setup

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# HuggingFace (for model downloads)
export HUGGINGFACE_TOKEN="your-token"
```

---

## Recommended Learning Order

1. **NLP Basics (3 days)**: 01 → 02 → 03
   - Solidify understanding of tokenization and embedding concepts
2. **Pre-trained Models (5 days)**: 04 → 05 → 06 → 07
   - Focus on HuggingFace hands-on practice
3. **LLM Applications (7 days)**: 08 → 09 → 10 → 11 → 12
   - Project-based learning
4. **Advanced LLM (5 days)**: 13 → 14 → 15 → 16
   - Quantization, RLHF, Agents, evaluation metrics

---

## Related Materials

- [Deep_Learning/](../Deep_Learning/00_Overview.md) - Prerequisite (Transformer)
- [Python/](../Python/00_Overview.md) - Advanced Python
- [Data_Analysis/](../Data_Analysis/00_Overview.md) - Data processing

---

## Reference Links

- [HuggingFace Documentation](https://huggingface.co/docs)
- [LangChain Documentation](https://python.langchain.com/docs)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
