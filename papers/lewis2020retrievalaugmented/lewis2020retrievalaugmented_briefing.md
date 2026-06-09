---
title: "Pre-Reading Briefing: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
paper_id: "33_lewis_2020"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., Kiela, D. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems (NeurIPS) 33*, 2020. arXiv:2005.11401.
**Author(s)**: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
**Year**: 2020 (NeurIPS)

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 **RAG (Retrieval-Augmented Generation)**, 즉 사전학습된 **parametric memory** (BART seq2seq의 가중치 안에 저장된 지식)와 **non-parametric memory** (Wikipedia 21M 패시지의 dense vector index)를 결합한 일반 목적 fine-tuning 레시피를 제안합니다. 핵심 아이디어는 (i) DPR 기반 retriever $p_\eta(z|x)$가 입력 $x$로부터 top-$K$ 패시지를 가져오고, (ii) BART 기반 generator $p_\theta(y|x,z)$가 이를 conditioning context로 사용해 출력을 생성하며, (iii) 검색된 문서를 잠재 변수로 보고 **marginalization** $p(y|x) = \sum_z p_\eta(z|x) p_\theta(y|x,z)$로 end-to-end 학습하는 것입니다. 결과적으로 Natural Questions/TriviaQA/WebQuestions/CuratedTrec에서 SOTA를 달성하고, 추출형(extractive) 모델이 도달하지 못하는 abstractive 영역까지 확장됩니다.

**English**
This paper introduces **RAG (Retrieval-Augmented Generation)**, a general-purpose fine-tuning recipe that combines **parametric memory** (knowledge stored in a pretrained BART seq2seq) with **non-parametric memory** (a dense vector index of 21M Wikipedia passages). The recipe has three pieces: (i) a DPR-based retriever $p_\eta(z|x)$ that fetches top-$K$ passages from the query, (ii) a BART generator $p_\theta(y|x,z)$ that conditions on those passages to produce the output, and (iii) end-to-end training that treats the retrieved document as a latent variable and marginalizes it: $p(y|x) = \sum_z p_\eta(z|x) p_\theta(y|x,z)$. RAG sets new SOTA on Natural Questions, TriviaQA, WebQuestions, and CuratedTrec, and extends naturally to generative tasks (abstractive QA, Jeopardy generation, FEVER fact verification) where extractive systems cannot operate.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2018–2020년은 **대형 사전학습 LM이 모든 것을 외운다**는 패러다임이 절정에 달한 시기였습니다. BERT(2018), GPT-2(2019), T5(2020), GPT-3(2020)는 가중치 안에 사실 지식을 암묵적으로 저장하고, 별도의 외부 데이터베이스 없이도 closed-book QA에서 놀라운 성능을 보였습니다. 그러나 이 접근법에는 명확한 한계가 있었습니다: (1) **hallucination** — 모델이 그럴듯하지만 틀린 사실을 만들어냄, (2) **opacity** — 답이 어디서 왔는지 추적 불가, (3) **staleness** — 세계가 변할 때 지식을 갱신하려면 재학습이 필요. 같은 시기에 **REALM** (Guu et al., 2020)과 **ORQA** (Lee et al., 2019)는 differentiable retriever를 masked LM과 결합했지만 추출형 QA에 국한되어 있었습니다. RAG는 이를 **모든 seq2seq 작업**으로 일반화한 최초의 시도입니다.

**English**
2018–2020 was the high-water mark of the "**LMs as implicit knowledge bases**" paradigm. BERT (2018), GPT-2 (2019), T5 (2020), and GPT-3 (2020) stored factual knowledge implicitly in their weights and achieved surprising results on closed-book QA without any external database. But this approach had clear limits: (1) **hallucination** — models produce plausible but wrong facts; (2) **opacity** — no way to trace where an answer came from; (3) **staleness** — updating knowledge requires retraining. Concurrent work like **REALM** (Guu et al., 2020) and **ORQA** (Lee et al., 2019) combined a differentiable retriever with masked LMs but was confined to extractive QA. RAG was the first to generalize this hybrid memory pattern to **arbitrary seq2seq tasks**.

### 타임라인 / Timeline

```
2014  Memory Networks (Weston) ─── attention over discrete memory cells
2017  Transformer (Vaswani) ─── attention as universal mixing primitive
2018  BERT (Devlin) ─── pretrained bidirectional encoder, foundation for DPR
2019  ORQA (Lee) ─── latent retrieval for extractive open-domain QA
2019  BART (Lewis) ─── pretrained denoising seq2seq
2020  REALM (Guu) ─── retriever + masked LM, pretrained jointly
2020  DPR (Karpukhin) ─── BERT bi-encoder for dense passage retrieval
2020  T5 (Raffel) ─── closed-book QA via parametric memory only
2020  RAG (Lewis, this paper) ─── DPR + BART, hybrid memory for generation
2020  GPT-3 (Brown) ─── 175B parametric-only LM, "in-context learning"
2022  Atlas, RETRO ─── scaled retrieval-augmented LMs
2023  GPT-4 + retrieval plugins ─── retrieval ubiquitous in production LLMs
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Seq2seq with attention**: encoder가 입력을 인코딩하고 decoder가 cross-attention으로 출력을 생성하는 구조 (Bahdanau 2014, Vaswani 2017).
- **BART**: denoising autoencoder로 사전학습된 encoder-decoder transformer. RAG의 generator로 사용됩니다 (400M 파라미터의 BART-large).
- **BERT**: bidirectional transformer encoder. DPR retriever의 query/document encoder로 사용됩니다 (BERT-base).
- **Latent variable models & marginalization**: $p(y|x) = \sum_z p(z|x) p(y|x,z)$. EM, VAE 등에서 익숙한 형태이지만 RAG는 top-$K$로 truncate해 미분 가능한 합으로 처리합니다.
- **Maximum Inner Product Search (MIPS)**: query 벡터와 가장 큰 내적을 갖는 문서 벡터를 빠르게 찾는 검색 문제. FAISS의 HNSW 같은 sub-linear 알고리즘이 필요합니다.
- **Open-Domain QA**: Wikipedia 등 대규모 코퍼스에서 답을 찾아야 하는 QA 설정 (Natural Questions, TriviaQA).
- **Negative log-likelihood training & teacher forcing**: seq2seq 학습의 표준 절차.
- **Beam search**: seq2seq 추론에서 사용하는 휴리스틱 검색.

**English**
- **Seq2seq with attention**: encoder–decoder with cross-attention (Bahdanau 2014, Vaswani 2017).
- **BART**: a denoising-autoencoder-pretrained encoder-decoder transformer; used as RAG's generator (BART-large, 400M params).
- **BERT**: bidirectional transformer encoder; used as both query and document encoder in DPR (BERT-base).
- **Latent-variable models & marginalization**: $p(y|x) = \sum_z p(z|x) p(y|x,z)$. RAG truncates this sum to top-$K$, making it a differentiable approximation.
- **Maximum Inner Product Search (MIPS)**: finding the document vector with the largest inner product against a query vector. Requires sub-linear methods like FAISS / HNSW.
- **Open-Domain QA**: answering questions over a large corpus like Wikipedia (Natural Questions, TriviaQA).
- **Teacher-forced NLL training**: the standard seq2seq training objective.
- **Beam search**: standard heuristic decoding for seq2seq.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Parametric memory | LM 가중치 안에 저장된 암묵적 지식 / Implicit knowledge stored in LM weights (here, BART). |
| Non-parametric memory | 외부 코퍼스의 명시적 지식, 인덱스를 통해 접근 / Explicit knowledge in an external corpus accessed via an index (here, Wikipedia dense index). |
| RAG-Sequence | 한 출력 sequence 전체를 같은 검색 문서 하나에 conditioning / Uses one retrieved doc to condition the entire output sequence. |
| RAG-Token | 토큰별로 다른 검색 문서를 쓸 수 있도록 marginalize / Marginalizes per token, allowing different docs for different tokens. |
| DPR (Dense Passage Retrieval) | BERT 기반 dual-encoder retriever, $p(z\|x) \propto \exp(d(z)^\top q(x))$ / BERT-based bi-encoder retriever. |
| MIPS / FAISS | Maximum Inner Product Search; FAISS는 GPU 가속 라이브러리 / Sub-linear nearest-vector search; FAISS is the GPU-accelerated library used. |
| Marginalization | 잠재 변수 $z$를 합산해 제거 / Summing out latent doc $z$ to get $p(y\|x)$. |
| Top-$K$ truncation | 합산을 가장 가능성 높은 $K$개 문서로 제한 / Restricting the marginal sum to the top-$K$ retrieved documents. |
| Thorough vs Fast Decoding | RAG-Sequence의 두 가지 추론 절차 / Two RAG-Sequence inference modes (exact vs approximate). |
| Index hot-swapping | 비파라메트릭 메모리를 교체해 모델 지식을 갱신 / Swap the document index to update model knowledge without retraining. |
| Closed-book QA | 외부 메모리 없이 LM 가중치만으로 답하는 QA / Answering QA purely from LM parameters with no retrieval. |
| FEVER | 사실 검증 데이터셋 (supports/refutes/not enough info) / Fact-verification benchmark over Wikipedia claims. |

---

## 5. 수식 미리보기 / Equations Preview

### Marginal likelihood (RAG-Sequence)

$$p_{\text{RAG-Seq}}(y|x) \approx \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y|x, z) = \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) \prod_{i=1}^{N} p_\theta(y_i | x, z, y_{1:i-1})$$

**한국어**: 검색된 단일 문서 $z$가 출력 sequence 전체를 조건화. 토큰 확률은 sequence-level로 곱한 뒤 문서별로 marginalize.
**English**: A single retrieved document conditions the entire output sequence; token probabilities are multiplied first, then marginalized over documents.

### Marginal likelihood (RAG-Token)

$$p_{\text{RAG-Tok}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y_i | x, z, y_{1:i-1})$$

**한국어**: 각 토큰마다 다른 문서를 쓸 수 있도록, 합산을 곱셈 안쪽에 둠. 여러 문서의 정보를 결합해 답할 때 유리.
**English**: Marginalization is moved inside the token product, so each token can draw from a different document — useful when multiple sources must be combined.

### Retriever score (DPR)

$$p_\eta(z|x) \propto \exp\!\big(\mathbf{d}(z)^\top \mathbf{q}(x)\big), \quad \mathbf{d}(z) = \text{BERT}_d(z),\ \mathbf{q}(x) = \text{BERT}_q(x)$$

**한국어**: BERT-base 기반 dual encoder. 학습 시 query encoder만 fine-tune되고 document index는 frozen.
**English**: BERT-base bi-encoder. During fine-tuning only the query encoder is updated; the document encoder and index stay frozen.

### Training objective

$$\mathcal{L}(\theta, \eta) = \sum_j -\log p(y_j | x_j) = \sum_j -\log \sum_{z \in \text{top-}k} p_\eta(z|x_j) p_\theta(y_j | x_j, z)$$

**한국어**: marginal NLL을 Adam으로 최소화. retrieval supervision은 필요 없음.
**English**: Minimize marginal NLL with Adam; no retrieval supervision is needed — the retriever learns purely from end-task signal.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
- §1 Introduction과 Figure 1을 먼저 정독해 RAG의 데이터 흐름(query → retriever → top-K docs → generator → marginalize)을 머릿속에 그린다.
- §2.1의 RAG-Sequence vs RAG-Token 비교는 핵심 — 두 식이 어떻게 다르고 언제 한쪽이 유리한지 직관적으로 이해할 것.
- §2.2 DPR과 §2.3 BART는 다른 논문에서 차용한 컴포넌트이므로 빠르게 읽고, §2.4 training의 "왜 document encoder를 freeze 했는가"를 주목.
- §3–§4의 실험은 표 1 (ODQA), 표 2 (생성/분류), 표 6 (ablation: BM25 vs DPR, frozen vs learned)을 비교하며 읽으면 RAG의 강점이 어디서 오는지 분명해짐.
- §4.5의 index hot-swapping은 RAG의 가장 큰 실용적 매력 — 재학습 없이 지식 갱신.

**English**
- Read §1 and Figure 1 first to lock in the data flow: query → retriever → top-K docs → generator → marginalize.
- §2.1's RAG-Sequence vs RAG-Token contrast is the conceptual heart — understand how the two equations differ and when each wins.
- §2.2 (DPR) and §2.3 (BART) are imported components; skim them. Pause at §2.4 to note **why the document encoder is frozen** (avoiding REALM-style index re-encoding cost).
- §3–§4: read Tables 1 (ODQA), 2 (generation/classification), and 6 (ablations: BM25 vs DPR, frozen vs learned retriever) together — they reveal where RAG's gains come from.
- §4.5's "index hot-swapping" is RAG's most practical selling point: knowledge updates without retraining.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
RAG는 2023–2026년 LLM 생태계에서 **사실상 표준 인프라**가 되었습니다. ChatGPT의 검색 플러그인, Perplexity, Bing Chat, Claude의 long-context retrieval, 그리고 거의 모든 enterprise LLM 솔루션이 이 논문의 패턴을 따릅니다: vector store + LLM + marginalization. 또한 이 논문은 (1) **hallucination을 줄이는 가장 효과적인 방법은 외부 grounding**이라는 합의를 만들었고, (2) **vector database 산업** (Pinecone, Weaviate, Chroma, Milvus)을 직접적으로 촉발했으며, (3) **agentic AI**와 **tool use** 패러다임의 출발점이 되었습니다 (검색을 tool로 보는 시각). 후속 연구로는 Atlas (Izacard 2022, scaled RAG), RETRO (Borgeaud 2022, retrieval at every layer), Self-RAG (Asai 2023, retrieval를 self-reflective하게 결정), Long RAG, GraphRAG 등이 있으며, 실용적으로는 **retrieval quality가 LLM 성능의 가장 큰 lever**라는 인식이 자리잡았습니다.

**English**
RAG became **de-facto infrastructure** in the 2023–2026 LLM ecosystem. ChatGPT's retrieval plugins, Perplexity, Bing Chat, Claude's long-context retrieval, and essentially every enterprise LLM stack follow this paper's pattern: vector store + LLM + marginalization. Specifically, the paper (1) cemented the consensus that **external grounding is the most effective hallucination-reduction tool**, (2) directly catalyzed the vector-database industry (Pinecone, Weaviate, Chroma, Milvus), and (3) seeded the "**retrieval as a tool**" view that underpins agentic AI and tool-use paradigms. Follow-ups include Atlas (Izacard 2022, scaled RAG), RETRO (Borgeaud 2022, retrieval at every layer), Self-RAG (Asai 2023, self-reflective retrieval), Long RAG, and GraphRAG. Practically, **retrieval quality is now widely seen as the single biggest lever on LLM end-task quality**.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
