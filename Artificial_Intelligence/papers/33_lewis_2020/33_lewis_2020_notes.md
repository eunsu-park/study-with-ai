---
title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
authors: [Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela]
year: 2020
journal: "Advances in Neural Information Processing Systems (NeurIPS) 33"
doi: "arXiv:2005.11401"
topic: Artificial_Intelligence
tags: [retrieval-augmented-generation, RAG, dense-passage-retrieval, DPR, BART, open-domain-QA, hybrid-memory, FAISS, hallucination, knowledge-intensive-NLP]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 33. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks / 지식 집약 NLP 작업을 위한 검색 증강 생성

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 **RAG (Retrieval-Augmented Generation)**를 제안합니다. 이는 사전학습된 seq2seq 모델의 **parametric memory** (BART-large 400M의 가중치)와 Wikipedia 21M 패시지로 구성된 **non-parametric memory** (dense vector index)를 결합한 일반 목적 fine-tuning 레시피입니다. 핵심 메커니즘은 세 단계로 정리됩니다. (1) **Retriever** $p_\eta(z|x)$ — DPR (Karpukhin et al., 2020)에서 차용한 BERT-base bi-encoder가 query $x$에 대해 top-$K$ Wikipedia 패시지를 FAISS MIPS로 검색합니다. (2) **Generator** $p_\theta(y_i | x, z, y_{1:i-1})$ — BART-large가 query와 검색된 패시지를 concat한 입력으로부터 토큰을 자기회귀적으로 생성합니다. (3) **Marginalization** — 검색된 문서를 잠재 변수로 보고 $p(y|x) = \sum_z p_\eta(z|x) p_\theta(y|x,z)$로 합산합니다. 이 marginalization을 어디서 수행하느냐에 따라 두 변종이 나뉩니다: **RAG-Sequence**는 sequence 전체에 같은 문서를 가정하고 sequence 확률을 곱한 뒤 합산하며, **RAG-Token**은 토큰별로 합산해 여러 문서의 정보를 조합할 수 있게 합니다. 학습은 retrieval supervision 없이 marginal NLL만 Adam으로 최소화하며, query encoder와 BART만 fine-tune하고 document encoder와 index는 frozen 상태를 유지합니다.

**English**
This paper introduces **RAG (Retrieval-Augmented Generation)** — a general-purpose fine-tuning recipe that fuses a pretrained seq2seq model's **parametric memory** (BART-large, 400M params) with a **non-parametric memory** consisting of a dense vector index over 21M Wikipedia passages. The mechanism has three pieces. (1) **Retriever** $p_\eta(z|x)$ — a BERT-base bi-encoder borrowed from DPR (Karpukhin et al., 2020) returns top-$K$ Wikipedia passages for query $x$ via FAISS MIPS. (2) **Generator** $p_\theta(y_i | x, z, y_{1:i-1})$ — BART-large autoregressively generates tokens from the concatenation of the query and the retrieved passage. (3) **Marginalization** — the retrieved document is treated as a latent variable and summed out: $p(y|x) = \sum_z p_\eta(z|x) p_\theta(y|x,z)$. Where this marginalization is performed splits the two variants: **RAG-Sequence** assumes one document conditions the whole output sequence (multiply token probabilities, then marginalize), while **RAG-Token** marginalizes per token, letting different documents contribute to different tokens. Training minimizes marginal NLL with Adam — no retrieval supervision is required — and updates only the query encoder and BART, keeping the document encoder and index frozen. RAG sets new SOTA on Natural Questions (44.5 EM), TriviaQA (56.8/68.0), WebQuestions (45.2), and CuratedTrec (52.2), beats BART by 2.6 BLEU on MS-MARCO abstractive QA, dominates BART by 42.7% vs 7.1% on human factuality judgments for Jeopardy generation, and lands within 4.3% of pipeline SOTA on FEVER without ever using retrieval supervision.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 사전학습된 LM이 사실 지식을 가중치 안에 저장한다는 사실에서 출발하지만 (REALM, ORQA의 발견), 이 **parametric-only** 패러다임의 세 가지 한계를 지적합니다.

1. **지식 확장/갱신 불가**: 새로운 사실을 추가하려면 재학습이 필요.
2. **결정 추적 불가 (no provenance)**: 왜 그 답이 나왔는지 설명할 방법이 없음.
3. **Hallucination**: 그럴듯하지만 사실과 다른 출력을 만들어냄.

저자들은 **hybrid memory** 아키텍처가 이 문제들을 해결한다고 주장합니다. REALM과 ORQA가 같은 방향이지만 (i) **추출형(extractive) ODQA**에만 적용되었고 (ii) generation에는 활용되지 못했습니다. RAG는 이를 **모든 seq2seq 작업**으로 확장하는 것이 목표입니다. 그림 1은 데이터 흐름을 명확히 보여줍니다: 입력 $x$ → query encoder → MIPS로 top-K 문서 $z_i$ 검색 → generator가 marginalize → 출력 $y$. 이 모든 컴포넌트는 사전학습되어 있으므로 추가 retrieval supervision 없이 즉시 fine-tune 가능합니다.

**English**
The authors start from the well-known fact that pretrained LMs store factual knowledge in their parameters (REALM, ORQA), but flag three limits of the **parametric-only** paradigm.

1. **Knowledge cannot be expanded or revised** without retraining.
2. **No provenance**: there is no way to explain *why* a given answer was produced.
3. **Hallucination**: outputs are fluent but often factually wrong.

The fix proposed is **hybrid memory**. REALM and ORQA pointed the same direction but (i) were limited to **extractive open-domain QA** and (ii) were never applied to generation. RAG extends the hybrid pattern to **arbitrary seq2seq tasks**. Figure 1 makes the dataflow concrete: input $x$ → query encoder → MIPS retrieves top-K docs $z_i$ → generator marginalizes → output $y$. Every component is pretrained, so no retrieval supervision is needed — fine-tuning starts from a strong initialization.

### Part II: Methods (§2) / 방법

#### 2.1 Two RAG Models / 두 RAG 모델

**한국어**
RAG는 입력 $x$로부터 텍스트 문서 $z$를 검색하고, 이를 추가 컨텍스트로 사용해 출력 $y$를 생성합니다. 두 컴포넌트:

- **Retriever** $p_\eta(z|x)$ — top-$K$ truncated distribution over Wikipedia passages.
- **Generator** $p_\theta(y_i | x, z, y_{1:i-1})$ — 이전 토큰 $y_{1:i-1}$, 입력 $x$, 검색된 패시지 $z$로 다음 토큰 $y_i$를 생성.

학습 시 검색된 문서를 **잠재 변수**로 취급해 marginalize합니다. 어디서 합산하느냐에 따라 두 모델이 정의됩니다.

**RAG-Sequence**는 한 sequence 전체에 동일한 $z$를 가정합니다:

$$p_{\text{RAG-Seq}}(y|x) \approx \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y|x, z) = \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) \prod_{i=1}^{N} p_\theta(y_i | x, z, y_{1:i-1})$$

**RAG-Token**은 매 토큰마다 다른 $z$가 가능하도록 marginalize합니다:

$$p_{\text{RAG-Tok}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y_i | x, z, y_{1:i-1})$$

흥미로운 사실: 분류 작업(타겟 길이=1)에서는 두 모델이 동등합니다.

**English**
RAG retrieves text documents $z$ from input $x$ and uses them as additional context to generate $y$. Two components:

- **Retriever** $p_\eta(z|x)$ — a top-$K$ truncated distribution over Wikipedia passages.
- **Generator** $p_\theta(y_i | x, z, y_{1:i-1})$ — produces the next token from the prefix, input, and retrieved passage.

The retrieved document is treated as a **latent variable** and marginalized at training time. Where this marginalization is placed defines the two variants.

**RAG-Sequence** assumes a single $z$ conditions the whole sequence:

$$p_{\text{RAG-Seq}}(y|x) \approx \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) \prod_{i=1}^{N} p_\theta(y_i | x, z, y_{1:i-1})$$

**RAG-Token** allows a different $z$ per token by moving the sum inside the product:

$$p_{\text{RAG-Tok}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y_i | x, z, y_{1:i-1})$$

Note: for classification tasks (target length 1), the two models are equivalent.

#### 2.2 Retriever: DPR / 리트리버

**한국어**
Retrieval은 **DPR**의 bi-encoder 구조를 따릅니다:

$$p_\eta(z|x) \propto \exp\!\big(\mathbf{d}(z)^\top \mathbf{q}(x)\big), \quad \mathbf{d}(z) = \text{BERT}_d(z),\ \mathbf{q}(x) = \text{BERT}_q(x)$$

두 BERT-base 인코더의 [CLS] 출력을 dense vector로 사용합니다. Top-$K$를 계산하려면 21M 패시지에 대한 **MIPS**가 필요한데, FAISS의 HNSW 근사로 sub-linear time에 처리합니다. 초기화는 TriviaQA와 Natural Questions에서 학습된 DPR 가중치를 사용합니다.

**English**
Retrieval follows the **DPR** bi-encoder design:

$$p_\eta(z|x) \propto \exp\!\big(\mathbf{d}(z)^\top \mathbf{q}(x)\big)$$

Both encoders are BERT-base; the [CLS] output is used as the dense vector. Top-$K$ over 21M passages requires **MIPS**, handled by FAISS's HNSW in sub-linear time. The retriever is initialized from a DPR checkpoint pretrained on TriviaQA and Natural Questions.

#### 2.3 Generator: BART / 생성기

**한국어**
Generator는 **BART-large** (400M params, denoising-pretrained encoder-decoder transformer)입니다. 입력 $x$와 검색된 패시지 $z$를 단순히 concat해 입력으로 사용합니다. BART는 다양한 노이즈 함수로 사전학습되어 있어 비슷한 크기의 T5보다 강한 generation 성능을 보입니다. 저자들은 BART의 파라미터 $\theta$를 **parametric memory**라 부릅니다.

**English**
The generator is **BART-large** (400M params, a denoising-pretrained encoder-decoder transformer). The input $x$ and retrieved passage $z$ are simply concatenated. BART, pretrained with diverse noising functions, beats similarly-sized T5 on generation. The authors call BART's parameters $\theta$ the **parametric memory**.

#### 2.4 Training / 학습

**한국어**
학습 데이터는 입력/출력 쌍 $(x_j, y_j)$의 fine-tuning corpus입니다. **어떤 문서를 검색해야 하는지에 대한 직접 supervision은 없습니다** — 모델은 end-task signal로부터 retrieval을 학습합니다. 목적함수는 marginal NLL:

$$\mathcal{L}(\theta, \eta) = \sum_j -\log p(y_j | x_j) = \sum_j -\log \sum_{z \in \text{top-}k(p(\cdot|x_j))} p_\eta(z|x_j)\, p_\theta(y_j|x_j, z)$$

Adam으로 최적화. **document encoder $\text{BERT}_d$와 index를 frozen 상태로 유지**하는 것이 RAG의 실용적 핵심: REALM처럼 학습 중 인덱스를 주기적으로 다시 임베딩하지 않으므로 비용이 극적으로 절감됩니다. Query encoder $\text{BERT}_q$와 BART generator만 업데이트하면 충분히 강한 성능이 나옵니다.

**English**
The training corpus consists of input/output pairs $(x_j, y_j)$. There is **no direct supervision on which document to retrieve** — retrieval is learned from end-task signal alone. The objective is marginal NLL:

$$\mathcal{L}(\theta, \eta) = \sum_j -\log \sum_{z \in \text{top-}k(p(\cdot|x_j))} p_\eta(z|x_j)\, p_\theta(y_j|x_j, z)$$

Optimized with Adam. **Critically, the document encoder $\text{BERT}_d$ and index are frozen** — unlike REALM, RAG does not re-embed the index during training, saving enormous cost. Updating only the query encoder $\text{BERT}_q$ and BART generator already yields strong performance.

#### 2.5 Decoding / 디코딩

**한국어**
**RAG-Token**: 토큰별 marginal $p'_\theta(y_i | x, y_{1:i-1}) = \sum_z p_\eta(z|x) p_\theta(y_i | x, z, y_{1:i-1})$를 standard transition probability로 보고 표준 beam search를 적용합니다.

**RAG-Sequence**: sequence likelihood가 token-level로 분해되지 않으므로 표준 beam search를 직접 사용할 수 없습니다. 대신 각 문서 $z$에 대해 별도로 beam search를 돌려 후보 sequence 집합 $Y$를 모은 뒤, 각 hypothesis $y$의 확률을 추정합니다.
- **Thorough Decoding**: $y$가 어떤 문서 $z$의 beam에 등장하지 않으면 추가 forward pass를 돌려 $p_\theta(y|x,z)$를 계산.
- **Fast Decoding**: 미등장 시 $p_\theta(y|x,z) \approx 0$로 근사. 출력이 길 때 효율적.

**English**
**RAG-Token**: the per-token marginal $p'_\theta(y_i | x, y_{1:i-1}) = \sum_z p_\eta(z|x) p_\theta(y_i | x, z, y_{1:i-1})$ acts as a standard transition probability, so any beam decoder works.

**RAG-Sequence**: the sequence likelihood does not factorize over tokens, so beam search must be run separately per document $z$. The candidate set $Y$ is the union; probabilities are estimated by either:
- **Thorough Decoding**: extra forward passes for any missing $(y,z)$ pair.
- **Fast Decoding**: approximate missing $p_\theta(y|x,z) \approx 0$. Cheaper for long outputs.

### Part III: Experiments (§3) / 실험

**한국어**
모든 실험에서 비파라메트릭 메모리는 **2018년 12월 Wikipedia dump**입니다. 각 article을 disjoint한 100-word chunk로 split해 21M passage를 얻고, FAISS HNSW로 단일 인덱스를 만듭니다. 학습 시 $K \in \{5, 10\}$, 테스트 시 dev에서 튜닝합니다.

저자들은 네 가지 작업을 평가합니다.

1. **Open-Domain QA (§3.1)**: NQ, TriviaQA, WebQuestions, CuratedTrec. EM 점수. CT/WQ는 데이터가 작아 NQ RAG로 초기화.
2. **Abstractive QA (§3.2)**: MS-MARCO NLG v2.1. 골드 패시지를 사용하지 않고 Wikipedia만 사용해 generation 평가.
3. **Jeopardy Question Generation (§3.3)**: 답 entity로부터 Jeopardy 형식 질문 생성. Q-BLEU-1, 인간 평가 (factuality, specificity).
4. **Fact Verification (§3.4)**: FEVER 3-way (supports/refutes/NEI), 2-way. **Retrieval supervision 없이** 학습.

**English**
The non-parametric memory is fixed throughout: a **December 2018 Wikipedia dump** split into disjoint 100-word chunks (21M passages total), indexed with FAISS HNSW. Training uses $K \in \{5, 10\}$; test $K$ is tuned on dev.

Four tasks are evaluated.

1. **Open-Domain QA (§3.1)**: NQ, TriviaQA, WebQuestions, CuratedTrec. Exact-match (EM). CT/WQ are small, so initialized from the NQ RAG.
2. **Abstractive QA (§3.2)**: MS-MARCO NLG v2.1, **without gold passages** — RAG must rely on Wikipedia.
3. **Jeopardy Question Generation (§3.3)**: generate Jeopardy-style questions from answer entities. Q-BLEU-1 plus human judgments (factuality, specificity).
4. **Fact Verification (§3.4)**: FEVER 3-way and 2-way classification, **without retrieval supervision**.

### Part IV: Results (§4) / 결과

#### 4.1 Open-Domain QA / 개방 도메인 QA

| Model | NQ | TQA | WQ | CT |
|---|---|---|---|---|
| T5-11B (closed-book) | 34.5 | -/50.1 | 37.4 | - |
| T5-11B+SSM | 36.6 | -/60.5 | 44.7 | - |
| REALM | 40.4 | -/- | 40.7 | 46.8 |
| DPR (extractive) | 41.5 | 57.9/- | 41.1 | 50.6 |
| **RAG-Token** | 44.1 | 55.2/66.1 | **45.5** | 50.0 |
| **RAG-Sequence** | **44.5** | **56.8/68.0** | 45.2 | **52.2** |

**한국어**
RAG는 4개 ODQA 데이터셋 모두에서 SOTA입니다. 흥미로운 발견:
- RAG는 **closed-book** (T5)의 생성 유연성과 **open-book** (DPR)의 검색 정확도를 모두 갖춤.
- REALM의 expensive **salient span masking** 사전학습 없이 더 나은 성능.
- **검색된 문서에 정답이 없어도 generation이 가능**: NQ에서 11.8% 정답 케이스가 retrieval에 답을 포함하지 않음 (extractive 모델은 0%).
- DPR QA system은 BERT cross-encoder re-ranker + extractive reader가 필요한 반면 RAG는 둘 다 불필요.

**English**
RAG is SOTA on all four ODQA datasets. Notable findings:
- RAG combines **closed-book** generative flexibility (T5) with **open-book** retrieval accuracy (DPR).
- It beats REALM **without** expensive salient-span-masking pretraining.
- It can **answer correctly even when retrieval misses the answer**: 11.8% of NQ correct answers don't appear in any retrieved doc (an extractive system would score 0% on those).
- The DPR QA system uses a BERT cross-encoder re-ranker plus an extractive reader; RAG needs neither.

#### 4.2 Abstractive QA (MS-MARCO) / 추상적 QA

| Model | R-L | B-1 |
|---|---|---|
| SotA (gold passages) | 49.8* | 49.9* |
| BART | 38.2 | 41.6 |
| **RAG-Token** | 40.1 | 41.5 |
| **RAG-Sequence** | **40.8** | **44.2** |

**한국어**
RAG-Sequence는 **gold passage 없이도** BART 대비 R-L +2.6, B-1 +2.6점 향상. SotA에 근접하지만 SotA는 gold passages를 사용하므로 직접 비교는 어렵습니다. 저자들은 RAG가 **hallucination이 적고 사실적이다**라고 정성적으로 보고합니다.

**English**
RAG-Sequence improves over BART by 2.6 R-L and 2.6 B-1 points **without gold passages**, approaching the gold-passage-using SotA. Qualitatively, RAG hallucinates less and produces more factually correct text.

#### 4.3 Jeopardy Question Generation / Jeopardy 생성

| Metric | BART | RAG-Token | RAG-Sequence |
|---|---|---|---|
| B-1 | 15.1 | **17.3** | 14.7 |
| Q-BLEU-1 | 19.7 | **22.2** | 21.4 |
| Human factuality | 7.1% better | **42.7% better** | — |
| Human specificity | 16.8% better | **37.4% better** | — |

**한국어**
RAG-Token은 RAG-Sequence보다 우수합니다 — Jeopardy 질문은 **여러 사실의 결합**이 필요하기 때문입니다. 그림 2의 정성적 분석: "Hemingway"를 입력으로 RAG-Token이 "A Farewell to Arms"와 "The Sun Also Rises"를 생성할 때, 각 책 제목 첫 토큰에서 해당 문서의 posterior가 높았다가 이후에는 평탄해집니다. 이는 **non-parametric memory가 책 제목을 트리거하고 parametric memory가 완성을 수행한다**는 두 메모리의 협력 동학을 보여줍니다. 인간 평가 (452 페어, 표 4): RAG가 더 사실적인 비율 42.7% vs BART 7.1%, 더 구체적인 비율 37.4% vs 16.8%.

**English**
RAG-Token beats RAG-Sequence here because Jeopardy questions often **combine multiple facts**. Figure 2: feeding "Hemingway", RAG-Token's per-document posterior spikes for the doc about *A Farewell to Arms* on the first token of that title, then flattens — showing the **non-parametric memory triggers titles, the parametric memory completes them**, a clean illustration of the two memories cooperating. Human study (452 pairs, Table 4): RAG more factual 42.7% vs BART 7.1%; more specific 37.4% vs 16.8%.

#### 4.4 Fact Verification (FEVER) / 사실 검증

| Model | FVR-3 | FVR-2 |
|---|---|---|
| SotA | 76.8 | 92.2* |
| RAG | 72.5 | 89.5 |

**한국어**
RAG는 **retrieval supervision 없이** 3-way에서 SotA 대비 -4.3%, 2-way에서 -2.7%에 도달. SotA는 도메인 특화 파이프라인 + 골드 evidence supervision을 사용. RAG가 검색한 top-1 문서가 골드 article과 일치하는 비율이 71%, top-10 안에 골드가 있는 비율이 90%로 매우 강한 retrieval 품질을 보입니다.

**English**
RAG lands within 4.3% (3-way) and 2.7% (2-way) of SotA **without retrieval supervision**. SotA uses domain-specific pipelines and gold-evidence supervision. RAG's top-1 retrieval matches the gold article 71% of the time; the gold article appears in top-10 90% of the time.

#### 4.5 Additional Results / 추가 결과

**한국어**
- **Generation Diversity (Table 5)**: RAG-Sequence 83.5% distinct tri-grams vs BART 70.7% (MS-MARCO). 별도의 diversity decoding 없이 다양성 확보.
- **Retrieval Ablations (Table 6)**: BM25로 교체하면 NQ에서 큰 폭 하락 (44.0 → 31.8), retriever를 freeze하면 ODQA 성능 저하. 단 FEVER에선 BM25가 가장 좋음 — **entity 중심** claim에 word-overlap이 효과적.
- **Index Hot-Swapping (§4.5)**: 2016년 Wikipedia 인덱스로 2016년 세계 지도자 70% 정답, 2018년 지도자 4%. 2018년 인덱스로 2018년 지도자 68%, 2016년 지도자 12%. **인덱스 교체만으로 지식 갱신** 가능.
- **More retrieved docs (Figure 3)**: RAG-Sequence는 $K$가 클수록 monotone improvement, RAG-Token은 $K{=}10$에서 peak. MS-MARCO Bleu-1은 RAG-Token이 더 가파르게 상승.

**English**
- **Generation Diversity (Table 5)**: RAG-Sequence is 83.5% distinct tri-grams vs BART 70.7% on MS-MARCO — diverse output without diversity-decoding tricks.
- **Retrieval Ablations (Table 6)**: replacing DPR with BM25 hurts NQ badly (44.0 → 31.8); freezing the retriever hurts ODQA. Exception: BM25 wins on FEVER, where claims are **entity-centric** and word-overlap is well-suited.
- **Index Hot-Swapping (§4.5)**: with a 2016 index, RAG answers 70% of 2016-leader queries correctly; only 4% of 2018-leader queries. Switching to the 2018 index reverses this. Knowledge is updated **purely by swapping the index** — no retraining.
- **More retrieved docs (Figure 3)**: RAG-Sequence improves monotonically with $K$ on NQ; RAG-Token peaks around $K{=}10$. RAG-Token gains more steeply on MS-MARCO Bleu-1.

### Part V: Related Work, Discussion, Broader Impact (§5–§6) / 관련 연구, 논의, 광범위한 영향

**한국어**
저자들은 네 가지 분류로 관련 연구를 정리합니다.
- **Single-Task Retrieval**: 검색이 ODQA, fact checking, dialogue, translation, LM 등에 단일 작업별로 적용된 사례. RAG는 이를 **단일 통합 아키텍처**로 묶음.
- **General-Purpose NLP Architectures**: BERT, GPT-2, BART, T5와 같이 사전학습된 단일 모델로 다양한 작업을 처리하는 흐름. RAG는 여기에 **검색 모듈**을 추가.
- **Learned Retrieval**: search, RL, latent variable 등 다양한 retrieval 학습 방법.
- **Memory-based Architectures & Retrieve-and-Edit**: memory networks (Sukhbaatar), KNN-LM (Khandelwal), Hashimoto's retrieve-and-edit. RAG는 raw text를 메모리로 사용하므로 **인간이 읽고 쓸 수 있다**는 특성을 가짐.

논의 섹션은 향후 연구 방향으로 (1) retriever와 generator를 처음부터 jointly pretrain, (2) BART 같은 denoising objective와 retrieval을 결합하는 사전학습을 제시합니다.

**Broader Impact**: RAG는 hallucination 감소와 interpretability를 제공하지만, Wikipedia의 편향이 그대로 전파될 수 있고, GPT-2와 유사한 misuse 우려 (페이크뉴스, 스팸)가 있다고 인정합니다.

**English**
Related work is organized into four buckets.
- **Single-Task Retrieval**: retrieval used per-task for ODQA, fact-checking, dialogue, translation, LM. RAG unifies these into one architecture.
- **General-Purpose NLP Architectures**: BERT, GPT-2, BART, T5 — single pretrained models for many tasks. RAG adds a **retrieval module** to this lineage.
- **Learned Retrieval**: methods using search, RL, or latent-variable approaches.
- **Memory-based Architectures & Retrieve-and-Edit**: memory networks, KNN-LM, Hashimoto's retrieve-and-edit. RAG's memory is **raw text**, so it is human-readable and human-writable.

The Discussion proposes future directions: (1) jointly pretraining retriever and generator from scratch, (2) combining a BART-style denoising objective with retrieval during pretraining.

**Broader Impact**: RAG reduces hallucination and gains interpretability, but Wikipedia bias propagates, and GPT-2-style misuse concerns (fake news, spam) remain.

---

## 3. Key Takeaways / 핵심 시사점

1. **Hybrid memory > pure parametric memory** — RAG는 BART의 generative 유연성과 DPR의 검색 정확도를 결합해 closed-book과 extractive open-book의 한계를 동시에 극복합니다 / RAG marries BART's generative flexibility with DPR's retrieval precision, escaping the limits of closed-book LMs and extractive open-book systems simultaneously.

2. **Marginalization은 retrieval supervision을 없앤다** — 잠재 문서를 합산함으로써 어떤 문서를 검색해야 하는지에 대한 ground-truth label이 필요 없어집니다. End-task gradient만으로 retriever가 학습됩니다 / Marginalizing over latent documents removes the need for retrieval ground-truth — the retriever learns purely from end-task gradients flowing through the marginal NLL.

3. **RAG-Sequence vs RAG-Token은 실용적 trade-off** — 단일 사실 답변(ODQA)에는 Sequence가, 다중 사실 결합(Jeopardy)에는 Token이 적합. 분류처럼 출력 길이=1이면 둘이 동등 / RAG-Sequence excels at single-fact answers (ODQA), RAG-Token at multi-fact composition (Jeopardy). For length-1 outputs (classification), they are mathematically equivalent.

4. **Frozen document index가 RAG의 실용성을 만듦** — REALM처럼 학습 중 인덱스를 다시 임베딩하지 않으므로 학습 비용이 극적으로 작고 인덱스 hot-swapping이 가능 / Keeping the document encoder frozen avoids REALM's costly periodic re-embedding, making training cheap and enabling test-time index swapping.

5. **검색 누락에도 답할 수 있는 generative grounding** — NQ에서 정답이 검색 결과에 없는 경우의 11.8%를 RAG가 맞춤. parametric memory가 빈틈을 메움 / RAG correctly answers 11.8% of NQ questions where the answer isn't in any retrieved document — parametric memory fills the gap that extractive systems cannot.

6. **Index hot-swapping은 지식 갱신 비용을 0에 가깝게 만듦** — 2016 → 2018 인덱스 교체만으로 지식 시점이 바뀜. parametric LM이 풀지 못한 staleness 문제를 우아하게 해결 / Swapping a 2016 index for a 2018 one updates the model's world without retraining — an elegant fix to the staleness problem that pure parametric LMs cannot solve.

7. **DPR ≫ BM25 (대부분 작업), 단 entity-overlap 작업은 예외** — Open-domain QA, MS-MARCO에서 dense retrieval이 우세, 그러나 FEVER 같이 word-overlap이 강한 entity-centric 작업에서는 BM25가 경쟁력 / Dense DPR dominates BM25 on open-domain QA and MS-MARCO, but BM25 wins on FEVER where claims are entity-centric and word-overlap is well-suited.

8. **RAG는 LLM 시대 RAG-as-infrastructure의 청사진** — 2023–2026 ChatGPT plugins, Perplexity, Bing Chat, Claude long-context, Pinecone/Weaviate 산업 모두가 이 논문의 패턴(vector store + LM + marginalize)을 따름 / RAG is the blueprint for retrieval-as-infrastructure in the LLM era — ChatGPT plugins, Perplexity, Bing Chat, Claude long-context, and the Pinecone/Weaviate industry all follow its vector-store + LM + marginalize template.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Generative model with latent retrieval / 잠재 검색을 갖는 생성 모델

The RAG joint distribution factorizes as:

$$p(y, z | x) = p_\eta(z|x)\, p_\theta(y|x, z)$$

Marginalizing the latent $z$ gives the observed-data likelihood, which is approximated by truncating to top-$K$:

$$p(y|x) = \sum_z p_\eta(z|x)\, p_\theta(y|x, z) \;\approx\; \sum_{z \in \text{top-}k(p_\eta(\cdot|x))} p_\eta(z|x)\, p_\theta(y|x, z)$$

### 4.2 RAG-Sequence (sequence-level marginal) / RAG-Sequence

Sum is **outside** the per-token product:

$$p_{\text{RAG-Seq}}(y|x) \approx \sum_{z \in \text{top-}k} p_\eta(z|x) \prod_{i=1}^{N} p_\theta(y_i | x, z, y_{1:i-1})$$

**Interpretation / 해석**: pick a single document, evaluate the full sequence likelihood under it, then mix. Encourages the retriever to find a document that supports the **whole answer**. / 한 문서에 대해 전체 sequence 확률을 평가하고 문서별로 혼합. retriever가 **답 전체**를 뒷받침하는 문서를 선택하도록 유도.

### 4.3 RAG-Token (token-level marginal) / RAG-Token

Sum is **inside** the per-token product:

$$p_{\text{RAG-Tok}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k} p_\eta(z|x)\, p_\theta(y_i | x, z, y_{1:i-1})$$

**Interpretation / 해석**: at each token, mix predictions across documents. Allows different tokens to be supported by different documents — useful when the answer aggregates multiple facts. / 매 토큰마다 문서별 예측을 혼합. 토큰별로 다른 문서가 책임질 수 있어 **여러 사실의 결합**에 유리.

### 4.4 Retriever score / 리트리버 점수

DPR bi-encoder with [CLS] dense vectors:

$$p_\eta(z|x) = \frac{\exp\!\big(\mathbf{d}(z)^\top \mathbf{q}(x)\big)}{\sum_{z' \in \mathcal{Z}} \exp\!\big(\mathbf{d}(z')^\top \mathbf{q}(x)\big)}, \qquad \mathbf{d}(z) = \text{BERT}_d(z),\ \mathbf{q}(x) = \text{BERT}_q(x)$$

In practice the denominator sum is replaced by the FAISS-MIPS top-$K$ partition function, and $\mathbf{d}(z)$ is **frozen** during fine-tuning.

### 4.5 Training objective / 학습 목적

Negative marginal log-likelihood with Adam:

$$\mathcal{L}(\theta, \eta) = \sum_j -\log p(y_j | x_j) = \sum_j -\log \sum_{z \in \text{top-}k(p_\eta(\cdot|x_j))} p_\eta(z|x_j)\, p_\theta(y_j | x_j, z)$$

**Gradient flow / 그래디언트 흐름**: gradients reach $\eta$ (query encoder) only through $p_\eta(z|x)$ inside the marginal — there is no per-document supervision label.

### 4.6 RAG-Token decoding transition / RAG-Token 디코딩 전이

For beam search RAG-Token uses the per-step transition

$$p'_\theta(y_i | x, y_{1:i-1}) = \sum_{z \in \text{top-}k(p_\eta(\cdot|x))} p_\eta(z|x)\, p_\theta(y_i | x, z, y_{1:i-1})$$

which is plugged into a standard autoregressive beam decoder.

### 4.7 RAG-Sequence Thorough vs Fast decoding / RAG-Sequence 디코딩 변종

For each candidate $y \in Y$ produced by per-document beam search:

- **Thorough**: $p(y|x) = \sum_z p_\eta(z|x) p_\theta(y|x,z)$ — extra forward pass for missing $(y,z)$.
- **Fast**: $p_\theta(y|x,z) \approx 0$ if $y$ wasn't in $z$'s beam.

### 4.8 Worked numerical example / 수치 예제

**Setup / 설정**: query $x$ = "Define middle ear", $K{=}3$ documents with retriever scores $p_\eta(z|x) = (0.5, 0.3, 0.2)$. Answer candidate $y$ = "the tympanic cavity" of length $N{=}3$ tokens.

Suppose generator probabilities $p_\theta(y|x,z)$ for the **whole sequence** under each doc:

| z | $p_\eta(z\|x)$ | $p_\theta(y\|x,z)$ |
|---|---|---|
| z1 | 0.5 | 0.40 |
| z2 | 0.3 | 0.10 |
| z3 | 0.2 | 0.01 |

**RAG-Sequence**: $p(y|x) = 0.5 \cdot 0.40 + 0.3 \cdot 0.10 + 0.2 \cdot 0.01 = 0.232$.

For **RAG-Token** with per-token probabilities $p_\theta(y_i | x, z, y_{<i})$ at token $i{=}2$ (say "tympanic"):

| z | $p_\eta(z\|x)$ | $p_\theta(y_2\|...)$ |
|---|---|---|
| z1 | 0.5 | 0.7 |
| z2 | 0.3 | 0.2 |
| z3 | 0.2 | 0.05 |

Token-2 marginal: $0.5\cdot 0.7 + 0.3\cdot 0.2 + 0.2\cdot 0.05 = 0.42$. Repeat for each token and multiply.

This trace shows how RAG-Sequence and RAG-Token differ even on identical retrieval / generation probabilities: RAG-Token can "switch documents" between tokens, RAG-Sequence cannot. / 동일한 검색/생성 확률에서도 RAG-Token은 토큰 사이에서 문서를 바꿀 수 있는 반면 RAG-Sequence는 한 문서에 고정됨을 보여줍니다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1990s ─ Memory-based reasoning, IR + QA pipelines, TF-IDF / BM25
   │
2014 ─ Memory Networks (Weston) ─ attention over discrete memory cells
   │
2015 ─ End-to-End Memory Networks (Sukhbaatar)
   │
2017 ─ Transformer (Vaswani) ─ attention as universal mixer
   │
2018 ─ BERT (Devlin) ─ pretrained encoder, foundation for DPR
   │
2019 ─ ORQA (Lee, Chang, Toutanova) ─ latent retrieval for extractive ODQA
   │
2019 ─ BART (Lewis) ─ denoising-pretrained seq2seq
   │
2020 ─ T5 (Raffel) ─ pure parametric closed-book QA
   │
2020 ─ REALM (Guu) ─ retriever + masked LM, jointly pretrained
   │
2020 ─ DPR (Karpukhin) ─ BERT bi-encoder dense passage retrieval
   │
2020 ─ ★ RAG (Lewis et al., this paper) ★ ─ DPR + BART, hybrid memory for generation
   │
2020 ─ GPT-3 (Brown) ─ 175B parametric LM, in-context learning
   │
2022 ─ Atlas (Izacard) ─ scaled RAG, few-shot
   │
2022 ─ RETRO (Borgeaud, DeepMind) ─ retrieval at every transformer layer
   │
2023 ─ Self-RAG (Asai) ─ self-reflective retrieval
   │
2023 ─ ChatGPT plugins, Perplexity, Bing Chat ─ RAG as production infrastructure
   │
2024 ─ Long-context LLMs vs RAG debate; GraphRAG (Microsoft)
   │
2025–26─ Multi-modal RAG, agentic RAG, RAG + tool use as standard pattern
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Vaswani et al. 2017 — *Attention Is All You Need* | Transformer는 BART와 BERT의 backbone | 매우 높음 / Very high |
| Devlin et al. 2018 — *BERT* | DPR retriever의 query/document encoder는 BERT-base | 매우 높음 / Very high |
| Lewis et al. 2019 — *BART* | RAG의 generator는 BART-large; 같은 1저자(Mike Lewis)는 본 논문의 공저자 | 매우 높음 / Very high |
| Karpukhin et al. 2020 — *DPR* | RAG의 retriever 컴포넌트 자체. Retriever 초기화는 NQ/TriviaQA로 사전학습된 DPR | 매우 높음 / Very high |
| Guu et al. 2020 — *REALM* | 직접적 비교 baseline. REALM은 retriever를 masked LM과 함께 사전학습; RAG는 fine-tuning만 수행. 둘 다 latent retrieval | 매우 높음 / Very high |
| Lee et al. 2019 — *ORQA* | RAG의 가장 가까운 선행 연구; latent retrieval을 ODQA에 적용. RAG는 이를 generation으로 확장 | 높음 / High |
| Raffel et al. 2020 — *T5* | Closed-book QA baseline. RAG는 T5-11B(11B params)을 400M BART + 검색으로 능가 | 높음 / High |
| Khandelwal et al. 2020 — *KNN-LM* | 또 다른 retrieval-augmented LM. Token-level kNN으로 LM 분포를 보강 | 중간 / Medium |
| Borgeaud et al. 2022 — *RETRO* | RAG의 후속; 모든 transformer layer에서 검색을 수행하는 더 깊은 통합 | 높음 / High |
| Izacard et al. 2022 — *Atlas* | Few-shot에 강한 scaled RAG; FiD encoder로 멀티 문서 통합 | 높음 / High |
| Asai et al. 2023 — *Self-RAG* | RAG 위에 self-reflection token을 추가해 언제 검색할지 학습 | 중간 / Medium |
| Brown et al. 2020 — *GPT-3* | Parametric-only 패러다임의 정점; RAG는 대안 패러다임의 청사진 | 중간 / Medium |
| Sukhbaatar et al. 2015 — *End-to-End Memory Networks* | RAG의 정신적 선조; 미분 가능한 메모리 attention을 처음 정립 | 중간 / Medium |
| Johnson, Douze, Jégou 2017 — *FAISS* | RAG의 인덱싱 인프라; HNSW 근사로 21M 패시지 sub-linear search | 중간 / Medium |

---

## 7. References / 참고문헌

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., Kiela, D. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 33*, 2020. arXiv:2005.11401.
- Karpukhin, V. et al. "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*, 2020. arXiv:2004.04906.
- Lewis, M. et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL*, 2020.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.-W. "REALM: Retrieval-Augmented Language Model Pre-Training." *ICML*, 2020. arXiv:2002.08909.
- Lee, K., Chang, M.-W., Toutanova, K. "Latent Retrieval for Weakly Supervised Open Domain Question Answering." *ACL*, 2019.
- Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
- Vaswani, A. et al. "Attention Is All You Need." *NeurIPS*, 2017.
- Raffel, C. et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*, 2020. (T5)
- Johnson, J., Douze, M., Jégou, H. "Billion-Scale Similarity Search with GPUs." *arXiv:1702.08734*, 2017. (FAISS)
- Kwiatkowski, T. et al. "Natural Questions: A Benchmark for Question Answering Research." *TACL*, 2019.
- Joshi, M. et al. "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension." *ACL*, 2017.
- Bajaj, P. et al. "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset." *arXiv:1611.09268*, 2016.
- Thorne, J. et al. "FEVER: a Large-scale Dataset for Fact Extraction and VERification." *NAACL*, 2018.
- Khandelwal, U. et al. "Generalization through Memorization: Nearest Neighbor Language Models." *ICLR*, 2020. (KNN-LM)
- Borgeaud, S. et al. "Improving Language Models by Retrieving from Trillions of Tokens." *ICML*, 2022. (RETRO)
- Izacard, G. et al. "Atlas: Few-shot Learning with Retrieval Augmented Language Models." *JMLR*, 2023.
- Asai, A. et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR*, 2024.
