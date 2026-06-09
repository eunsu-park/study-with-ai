---
title: "Pre-Reading Briefing: Language Models are Unsupervised Multitask Learners (GPT-2)"
paper_id: "29_radford_2019"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Language Models are Unsupervised Multitask Learners (GPT-2): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI Technical Report.
**Author(s)**: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
**Year**: 2019

---

## 1. 핵심 기여 / Core Contribution

**한국어**
GPT-2는 **충분히 크고 다양한 코퍼스(WebText, 40GB)에서 학습된 단일 언어 모델이, 그 어떤 task-specific fine-tuning도 없이 zero-shot으로 다양한 NLP 태스크를 수행할 수 있음**을 보였습니다. 핵심 명제는 두 가지입니다. 첫째, 언어 모델링이라는 단일 비지도 목적 함수가 충분히 일반적이라면, 자연어로 표현 가능한 모든 태스크(요약, 번역, QA, 독해 등)는 그 conditional probability $p(\text{output} \mid \text{input}, \text{task})$의 추정으로 환원됩니다. 둘째, 모델 용량(117M → 345M → 762M → **1542M** 파라미터)이 커질수록 zero-shot 성능이 **log-linear**하게 향상되며, 이는 단순한 scaling이 task-specific 학습 없이도 generalist 시스템을 만들 수 있음을 시사합니다. 결과적으로 8개 LM 벤치마크 중 7개에서 SOTA를 달성했고(zero-shot), CoQA 55 F1, LAMBADA 8.6 perplexity, WikiText-2 18.34 등을 기록했습니다.

**English**
GPT-2 demonstrates that **a single language model trained on a sufficiently large and diverse corpus (WebText, 40GB) can perform a wide range of NLP tasks in a zero-shot setting** — without any task-specific fine-tuning, parameter updates, or architecture modifications. Two propositions anchor the paper. First, if language modeling is general enough, every NLP task expressible in natural language reduces to estimating $p(\text{output} \mid \text{input}, \text{task})$. Second, capacity matters: as parameters scale from 117M to 1.5B, zero-shot performance improves **log-linearly**, suggesting raw scale plus diverse pre-training can yield generalists. The largest GPT-2 model achieves state-of-the-art on 7 of 8 LM benchmarks zero-shot, including 55 F1 on CoQA, 8.6 perplexity on LAMBADA, and 18.34 perplexity on WikiText-2.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2018년의 NLP는 **pre-train + fine-tune** 패러다임이 빠르게 굳어지던 시기였습니다. ELMo(Peters 2018), GPT-1(Radford 2018), BERT(Devlin 2018)가 모두 같은 전략을 따랐습니다 — 큰 corpus에서 비지도로 사전 학습한 뒤, 각 task에 대해 라벨된 데이터로 fine-tune. 이는 task-specific 모델보다 훨씬 강력했지만, 여전히 **각 task마다 라벨 데이터가 필요**하다는 본질적 한계가 있었습니다. McCann et al. (2018)의 decaNLP, Bowman et al.의 GLUE 등이 multi-task 학습을 시도했지만, 모델은 여전히 "narrow expert" 범주를 못 벗어났습니다. GPT-2는 이 패러다임을 한 단계 더 밀어 — fine-tuning 자체를 제거하고, 모델 용량과 데이터의 다양성만으로 zero-shot generalist를 만들겠다고 선언합니다.

**English**
By 2018, the **pre-train + fine-tune** paradigm had crystallized in NLP. ELMo (Peters 2018), GPT-1 (Radford 2018), and BERT (Devlin 2018) all followed the same recipe: unsupervised pre-training on a large corpus, then supervised fine-tuning on each downstream task. This dramatically outperformed task-specific models, but still required **labeled data for every task**. Multi-task efforts like McCann's decaNLP and the GLUE benchmark probed multitask training, but models remained "narrow experts." GPT-2 pushes the paradigm one step further — eliminating fine-tuning altogether, and arguing that scale + data diversity alone can produce a zero-shot generalist.

### 타임라인 / Timeline

```
2013 ─ word2vec (Mikolov)
2014 ─ Seq2Seq, GloVe
2015 ─ Attention (Bahdanau)
2017 ─ Transformer (Vaswani)
2018 ─ GPT-1 (Radford), BERT (Devlin), ELMo (Peters), decaNLP (McCann)
2019 ─ ★ GPT-2 (this paper) — zero-shot scaling
2020 ─ GPT-3 (in-context learning)
2022 ─ ChatGPT
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Transformer architecture (Vaswani 2017)**: multi-head self-attention, feed-forward, layer norm, residual connection
- **GPT-1 (Radford 2018)**: decoder-only transformer, causal masking, pre-train + fine-tune
- **Language modeling**: autoregressive 분포 분해 $p(x) = \prod_i p(x_i \mid x_{<i})$, perplexity 정의
- **Byte Pair Encoding (BPE) (Sennrich 2015)**: subword tokenization, vocabulary size trade-off
- **Zero-shot/few-shot learning**: task-specific 학습 없이 일반화하는 개념
- **Perplexity, BLEU, ROUGE, F1**: 평가 지표의 정의
- **Multitask learning (Caruana 1997)**: 단일 모델로 여러 task 동시 학습

**English**
- **Transformer architecture** (Vaswani 2017) — multi-head attention, residual connections, layer norm
- **GPT-1** (Radford 2018) — decoder-only transformer with causal masking; pre-train + fine-tune setup
- **Language modeling fundamentals** — autoregressive factorization $p(x) = \prod_i p(x_i \mid x_{<i})$, perplexity
- **Byte Pair Encoding (BPE)** (Sennrich 2015) — subword tokenization
- **Zero/few-shot learning** — solving tasks without task-specific supervision
- **Evaluation metrics** — perplexity, BLEU, ROUGE, F1, exact match
- **Multitask learning** (Caruana 1997)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **WebText** | Reddit에서 ≥3 karma를 받은 outbound link만 수집해 만든 40GB, 8M-document 데이터셋. Wikipedia 제외. / 40GB, 8M-document corpus scraped from Reddit-recommended links, Wikipedia removed |
| **Zero-shot transfer** | 어떤 task-specific 학습 데이터도 사용하지 않고 task를 수행 / Performing a task with zero task-specific training examples |
| **Byte-level BPE** | UTF-8 byte sequence 위에서 동작하는 BPE; vocabulary 50,257; 모든 Unicode 표현 가능 / BPE operating on UTF-8 bytes (vocab 50,257), can model any Unicode string |
| **Autoregressive LM** | $p(x) = \prod_i p(x_i \mid x_{<i})$ 형태의 left-to-right 모델 / Left-to-right factorized model |
| **Conditional language modeling** | $p(\text{output} \mid \text{input}, \text{task})$로 task를 표현 / Tasks framed as conditional generation given input + task description |
| **GPT-2 (1.5B)** | 48-layer, 1600-dim, 1542M-parameter Transformer decoder / 48-layer, 1600-dim, 1.5B-param decoder |
| **Layer norm relocation** | LN을 각 sub-block 입력으로 이동(pre-LN); 마지막 self-attention 뒤에 추가 LN / Pre-LN: LN moved to the input of each sub-block, plus extra final LN |
| **Residual scaling** | residual layer 가중치를 $1/\sqrt{N}$로 초기화 ($N$ = residual layer 수) / Residual weights initialized scaled by $1/\sqrt{N}$ |
| **Perplexity (PPL)** | $\exp(-\frac{1}{T}\sum_t \log p(x_t \mid x_{<t}))$; 낮을수록 좋음 / $\exp$ of average negative log-likelihood per token |
| **LAMBADA** | 50+ 토큰 문맥에서 마지막 단어 예측 — long-range dependency 측정 / Predict final word with ≥50 tokens of context |
| **CoQA** | Conversational QA — 7 도메인의 대화형 독해 / Conversational reading comprehension over 7 domains |
| **TL;DR** | 요약을 유도하는 zero-shot 프롬프트 / "Too long; didn't read" prompt to induce summarization |

---

## 5. 수식 미리보기 / Equations Preview

**Eq. 1 — Autoregressive factorization / 자기회귀 분해**
$$p(x) = \prod_{i=1}^{n} p(s_n \mid s_1, \dots, s_{n-1})$$
모든 시퀀스 확률을 조건부 곱으로 분해. tractable sampling/estimation의 기반. / Decomposes joint probability into a product of conditionals — enables tractable sampling and estimation.

**Eq. 2 — Conditional task formulation / 조건부 태스크 정식화**
$$p(\text{output} \mid \text{input}, \text{task})$$
모든 NLP task를 자연어 prompt + 입력에 대한 조건부 분포로 표현. / Every NLP task expressed as a conditional distribution over outputs given inputs and a natural-language task descriptor.

**Eq. 3 — Transformer self-attention recap / 트랜스포머 self-attention 복습**
$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$
여기서 $M$은 causal mask ($M_{ij} = -\infty$ for $j > i$). 각 토큰은 자신과 이전 토큰만 참조. / $M$ is the causal mask preventing future-token attention.

**Eq. 4 — Perplexity / 퍼플렉시티**
$$\mathrm{PPL}(x) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(x_t \mid x_{<t})\right)$$
모델의 surprise를 측정. 1에 가까울수록 완벽한 예측. / Measures the model's average surprise per token; lower is better.

**Eq. 5 — Residual init scaling / 잔차 초기화 스케일링**
$$W_{\text{residual}} \sim \mathcal{N}(0, \sigma^2 / N)$$
$N$개의 residual layer가 있을 때, residual path 누적이 폭주하지 않도록 가중치를 $1/\sqrt{N}$로 스케일. / Scales residual-path weights by $1/\sqrt{N}$ to prevent activation blowup as depth grows.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
- **§1 Introduction**: "single-task training on single-domain data → narrow experts" 비판을 주의 깊게 읽기. 다음 패러다임이 무엇이어야 하는지의 motivation.
- **§2 Approach**: 핵심 철학. $p(\text{output} \mid \text{input}, \text{task})$ 식과 multitask = next-token-prediction 환원 논증을 이해할 것.
- **§2.1 Training Dataset (WebText)**: Reddit ≥3 karma 휴리스틱이 어떻게 quality filter가 되는지. 8M docs, 40GB.
- **§2.2 Input Representation (BPE)**: byte-level BPE의 이점(Unicode 전체 커버) vs. 비효율(merge across categories) → 카테고리 경계에서 merge 차단하는 트릭.
- **§2.3 Model**: GPT-1 대비 변경 사항 — pre-LN, 추가 final LN, residual scaling $1/\sqrt{N}$, 어휘 50,257, 컨텍스트 1024, batch 512.
- **§3 Experiments**: 8개 LM 벤치마크 → CBT → LAMBADA → Winograd → CoQA → 요약 → 번역 → QA. Figure 1의 log-linear scaling을 주목.
- **§4 Generalization vs Memorization**: 8-gram bloom filter로 train/test overlap 측정. 작지만 일관된 영향.
- **§5–8 Discussion/Conclusion**: zero-shot은 baseline일 뿐, fine-tuning 했을 때 어떨지 — GPT-3로 가는 길의 실마리.

**English**
- **§1 Introduction**: read the "narrow experts" critique closely — it motivates the entire paradigm shift.
- **§2 Approach**: the philosophical heart — understand the $p(\text{output} \mid \text{input}, \text{task})$ framing.
- **§2.1 Training Dataset (WebText)**: how Reddit karma serves as a quality heuristic; 8M docs, 40GB.
- **§2.2 Input Representation (BPE)**: byte-level BPE — pros (full Unicode) vs. cons (cross-category merges). Note the category-boundary trick.
- **§2.3 Model**: changes from GPT-1 — pre-LN, extra final LN, $1/\sqrt{N}$ residual scaling, vocab 50,257, ctx 1024, batch 512.
- **§3 Experiments**: 8 LM benchmarks → CBT → LAMBADA → Winograd → CoQA → summarization → translation → QA. Watch Figure 1's log-linear curves.
- **§4 Generalization vs Memorization**: 8-gram bloom filter analysis; small but consistent overlap.
- **§5-8**: discussion — zero-shot is a floor, not a ceiling; this seeds GPT-3's in-context learning.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
GPT-2는 현대 LLM 시대의 **개념적 출발점**입니다. 세 가지 핵심 영향:
1. **Scaling hypothesis의 첫 강력한 증거**: log-linear improvement 곡선은 GPT-3, GPT-4, Chinchilla scaling laws의 직접적 선조입니다.
2. **Prompting의 탄생**: "TL;DR:", "translate to french =", "answer:" 같은 자연어 prompt는 후일 in-context learning, instruction tuning, ChatGPT의 근간이 됩니다.
3. **사회적 임팩트의 시작**: OpenAI는 처음에 1.5B 모델 release를 보류했는데(misuse 우려), 이는 AI safety 정책 논쟁의 출발점이 됐습니다.

오늘날의 ChatGPT, Claude, Gemini는 모두 GPT-2가 제시한 두 원칙 — "scale wins" + "tasks-as-text" — 위에 세워져 있습니다.

**English**
GPT-2 is the **conceptual launch point** of the modern LLM era. Three key influences:
1. **First strong evidence for the scaling hypothesis** — the log-linear curves directly prefigure GPT-3, Chinchilla scaling laws, and GPT-4.
2. **Birth of prompting** — natural-language cues like "TL;DR:", "translate to french =", and "answer:" foreshadow in-context learning, instruction tuning, and ChatGPT.
3. **Start of AI-safety policy debates** — OpenAI initially withheld the 1.5B model over misuse concerns, opening a still-active conversation about staged release.

ChatGPT, Claude, and Gemini all rest on GPT-2's twin principles: **scale wins** and **tasks are text**.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
