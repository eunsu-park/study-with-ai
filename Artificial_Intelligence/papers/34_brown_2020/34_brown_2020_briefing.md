---
title: "Pre-Reading Briefing: Language Models are Few-Shot Learners (GPT-3)"
paper_id: "34_brown_2020"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Language Models are Few-Shot Learners (GPT-3): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., et al. "Language Models are Few-Shot Learners". *NeurIPS 2020*. arXiv:2005.14165
**Author(s)**: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, ... Ilya Sutskever, Dario Amodei (OpenAI, 31 authors)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 **1,750억 (175B) 파라미터** autoregressive transformer 언어 모델 **GPT-3**를 학습시키고, **gradient update 없이** 프롬프트 안에 자연어 지시(instruction)와 0–100개의 예시(demonstration)만 넣어주면 새 작업을 수행할 수 있다는 사실을 보였습니다. 이 능력을 **in-context learning** 또는 **few-shot learning**이라 부릅니다. 모델은 fine-tuning 없이도 LAMBADA 86.4%, TriviaQA 71.2%, PTB perplexity 20.5 등 여러 벤치마크에서 SOTA fine-tuned 모델과 경쟁하거나 능가합니다. 핵심 메시지는 "**모델 크기를 키우면 in-context learning 능력 자체가 매끄럽게 향상된다**"는 것입니다.

**English**
This paper trains **GPT-3**, a 175-billion-parameter autoregressive transformer language model, and shows that simply prompting it with a natural-language instruction and 0-100 demonstrations — **with no gradient updates at all** — is enough to perform many new tasks. The authors call this capability **in-context learning** (zero-/one-/few-shot). Without any fine-tuning, GPT-3 reaches LAMBADA 86.4%, TriviaQA 71.2%, PTB perplexity 20.5, and matches or beats specialized fine-tuned baselines on many benchmarks. The headline finding is that **scaling alone monotonically improves few-shot learning ability**, often at a steeper slope than zero-shot improves with scale.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2018년 BERT (Devlin et al.)와 GPT-1 (Radford et al.) 이후 NLP는 **"pre-train + task-specific fine-tune"** 패러다임에 정착했습니다. GPT-2(2019, 1.5B)는 zero-shot transfer가 가능함을 보였지만 SOTA에는 한참 못 미쳤고, Kaplan et al. (2020)의 **scaling laws**는 손실이 모델 크기·데이터·컴퓨트의 멱법칙(power law)으로 매끄럽게 줄어든다는 사실을 발견했습니다. GPT-3는 이 두 흐름의 자연스러운 결론입니다 — 만약 scaling laws가 옳다면, 모델을 100배 키우면 어디까지 가능한가?

**English**
Since BERT (2018) and GPT-1, NLP had settled into a "pre-train then fine-tune" paradigm. GPT-2 (2019, 1.5B parameters) demonstrated zero-shot transfer was possible, but lagged SOTA badly. Kaplan et al. (2020) established **neural scaling laws** showing loss falls as a smooth power law in model size, data, and compute. GPT-3 is the natural follow-through: if scaling laws hold, what happens when you scale 100× past GPT-2?

### 타임라인 / Timeline

```
2017 ─ Transformer (Vaswani et al.) — Attention Is All You Need
2018 ─ GPT-1 (Radford) 117M; BERT (Devlin) 340M — pre-train + fine-tune
2019 ─ GPT-2 (Radford) 1.5B — zero-shot LM, "language models are unsupervised multitask learners"
2019 ─ T5 (Raffel) 11B — text-to-text fine-tune
2020 ─ Scaling Laws (Kaplan et al.) — power-law L(N,D,C)
2020 ─ ★ GPT-3 (Brown et al.) 175B — in-context learning emerges with scale ★
2022 ─ Chinchilla (Hoffmann), InstructGPT, Emergent Abilities
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Transformer architecture**: multi-head self-attention, residual + LayerNorm, feed-forward $d_{ff}=4d_{model}$ (Vaswani 2017).
- **Autoregressive language modeling**: $p(x) = \prod_t p(x_t | x_{<t})$, cross-entropy training, BPE tokenization.
- **Sparse attention**: Sparse Transformer (Child et al., 2019) — locally banded + strided patterns to reduce $O(n^2)$ complexity.
- **Scaling laws** (Kaplan 2020, paper #39): $L(N) \propto N^{-\alpha_N}$.
- **Fine-tuning vs. transfer learning**: BERT-style task heads vs. text-to-text reframing.
- **Meta-learning**: inner-loop / outer-loop framing; here the inner loop is the forward pass itself.

**English**
- **Transformer architecture**: multi-head self-attention, residual + LayerNorm, feed-forward $d_{ff}=4d_{model}$ (Vaswani 2017).
- **Autoregressive LM**: cross-entropy training $p(x) = \prod_t p(x_t | x_{<t})$, BPE tokenization.
- **Sparse attention**: Sparse Transformer (Child et al., 2019) — banded + strided sparse patterns.
- **Scaling laws** (Kaplan 2020): $L(N) \propto N^{-\alpha_N}$.
- **Fine-tuning vs. transfer**: BERT task heads vs. T5 text-to-text reframing.
- **Meta-learning**: inner/outer-loop view; the inner loop here is a single forward pass conditioned on the prompt.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **In-context learning (ICL)** | 가중치 업데이트 없이 프롬프트만으로 작업을 학습 / Learning a task purely from context in the forward pass — no gradient updates |
| **Zero-shot / One-shot / Few-shot** | $K=0,1,K$ 개의 demonstration. 모두 weight update 없음 / Number of in-context demonstrations $K$; never any gradient updates |
| **Demonstration** | 프롬프트에 포함된 입력→출력 예시 / Input→output example shown in the prompt as conditioning |
| **Context window** $n_{ctx}$ | GPT-3에서는 2048 token / Maximum tokens the model attends over (2048 for GPT-3) |
| **Autoregressive LM** | 좌→우 단방향, $\prod_t p(x_t \mid x_{<t})$ / Left-to-right factorization $\prod_t p(x_t \mid x_{<t})$ |
| **Sparse attention** | Sparse Transformer 패턴: 일부 layer는 dense, 일부는 locally banded sparse / Alternating dense and locally banded sparse attention layers |
| **BPE** | Byte-Pair Encoding tokenizer; 단어 ≈ 0.7 token / ~0.7 words per token; subword units used by GPT-2/3 |
| **PetaFLOP/s-day** | 컴퓨트 단위, $10^{15} \times 86400$ FLOPs / Compute unit: $10^{15} \times 86400$ FLOPs |
| **Closed-book QA** | 외부 검색 없이 모델 파라미터만으로 답하는 QA / Answering questions purely from model weights without retrieval |
| **Data contamination** | 사전학습 데이터에 평가 셋 일부가 들어간 현상 / Test-set leakage into the pre-training corpus |
| **Meta-learning** | "learning to learn"; 여기서 outer loop = pretraining, inner loop = forward pass / Outer loop = pretraining; inner loop = the forward pass conditioned on the prompt |

---

## 5. 수식 미리보기 / Equations Preview

**1. Autoregressive LM objective / 자기회귀 언어모델 목적함수**

$$\mathcal{L}(\theta) = - \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

**한국어**: 토큰 $x_t$의 log-likelihood 합. GPT-3는 300B token에 대해 이 한 가지 손실만 최적화합니다.
**English**: Sum of log-likelihoods. GPT-3 optimizes only this single loss over 300B tokens.

**2. In-context few-shot probability / In-context few-shot 확률**

$$p_\theta\!\left(y_\text{test} \,\Big|\, T,\; (x_1,y_1), \ldots, (x_K,y_K),\; x_\text{test}\right)$$

**한국어**: task description $T$, $K$개의 (input, output) demonstration, test input $x_\text{test}$를 모두 단일 프롬프트로 연결한 후, 다음 토큰 분포에서 $y_\text{test}$를 추출. **$\theta$는 변하지 않는다** (no gradient).
**English**: Task description $T$, $K$ demonstrations, and the test input are concatenated as a single prompt; the answer is sampled from the next-token distribution. **$\theta$ is never updated** during inference.

**3. Compute scaling / 컴퓨트 스케일링** (Kaplan 2020 power law continues)

$$L(C) \approx 2.57 \cdot C^{-0.048}$$

**한국어**: validation loss $L$이 컴퓨트 $C$ (PetaFLOP/s-days)에 멱법칙으로 감소. GPT-3는 GPT-2 이후 두 자리수 더 이 trend가 유지됨을 확인.
**English**: Validation loss falls as a power law in compute. GPT-3 extends Kaplan's curve by two more orders of magnitude with only minor deviations.

**4. Multi-task perspective / 다중 작업 관점**

$$p(\text{task}, x, y) = p(\text{task}) \cdot p(x \mid \text{task}) \cdot p(y \mid x, \text{task})$$

**한국어**: 사전학습 코퍼스에는 무수한 latent task가 섞여 있다. ICL은 prompt를 통해 task를 식별/조건화하는 것으로 해석됨.
**English**: The pretraining corpus implicitly contains many latent tasks. ICL can be viewed as the prompt selecting/conditioning on a task.

**5. Few-shot scoring (multiple choice) / 객관식 점수**

$$\text{score}(c) = \frac{P(c \mid \text{context})}{P(c \mid \text{answer\_context})}$$

**한국어**: 일부 작업(ARC, OpenBookQA, RACE)에서 unconditional 확률로 normalize하여 길이 편향 보정.
**English**: For some multiple-choice tasks (ARC, OpenBookQA, RACE), normalize by unconditional completion probability to debias.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
- **§1 Introduction (3–6쪽)**: in-context learning 컨셉 (Figure 1.1, 1.2, 1.3) 꼼꼼히 — 논문의 핵심 메시지가 모두 여기 있음.
- **§2.1–2.2 Architecture & Data (8–9쪽)**: Table 2.1 (8개 모델 사이즈), Table 2.2 (데이터셋 5종 가중치) — 이 두 표만 외워도 절반.
- **§3 Results (10–28쪽)**: 매우 길지만, 각 subsection의 첫 단락과 표만 봐도 충분. 특히 §3.1 LAMBADA, §3.2 TriviaQA, §3.7 SuperGLUE, §3.9 Synthetic.
- **§4 Contamination (29–33쪽)**: 인터넷 스케일 데이터의 train/test overlap 문제. 빠르게 훑기.
- **§5 Limitations (33–34쪽)**: 반드시 읽기. bidirectional 부재, sample efficiency, 해석 가능성.
- **§6 Broader Impacts (34–39쪽)**: 부분적으로. bias 분석 표만 봐도 OK.
- **부록 G (task phrasings)**: 실제 prompt 형식이 궁금하면 참고.

**English**
- **§1 Introduction (pp 3–6)**: Read carefully — Figures 1.1/1.2/1.3 and the in-context-learning framing carry the entire thesis.
- **§2.1–2.2 (pp 8–9)**: Master Tables 2.1 (8 model sizes) and 2.2 (5 dataset weights).
- **§3 Results (pp 10–28)**: Long; first paragraph + main table of each subsection is enough. Highlights: §3.1 LAMBADA, §3.2 TriviaQA, §3.7 SuperGLUE, §3.9 Synthetic.
- **§4 Contamination (pp 29–33)**: Skim — train/test overlap analysis.
- **§5 Limitations (pp 33–34)**: Must-read. Bidirectionality, sample efficiency, interpretability.
- **§6 Broader Impacts (pp 34–39)**: Skim; bias tables are the takeaway.
- **Appendix G**: Useful for actual prompt formats per task.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
GPT-3는 LLM 시대의 **출발 신호**입니다. 그 영향을 정리하면:

1. **In-context learning이 표준 인터페이스가 됨** — 이후 모든 LLM(Chinchilla, PaLM, LLaMA, GPT-4, Claude)은 동일한 prompting 패러다임을 채택.
2. **"Foundation Model" 개념의 실증** — Bommasani et al. (2021)이 이 용어를 정립한 직접적 근거.
3. **Prompt engineering, chain-of-thought, RLHF (InstructGPT 2022)** — 모두 GPT-3 base를 기반으로 발전.
4. **Scaling laws의 확인 + 한계** — Chinchilla(2022)는 GPT-3가 데이터 부족 상태(under-trained)였음을 보였고, GPT-3보다 작은 모델이 더 많은 데이터로 더 좋은 성능을 낼 수 있음을 입증.
5. **API as deployment** — 모델 가중치를 공개하지 않고 API로만 제공하는 비즈니스 모델의 시작 (OpenAI API, ChatGPT 2022).
6. **AI safety의 본격화** — §6 Broader Impacts 절은 LLM의 misinformation/bias 문제를 학계 주류로 끌어올림.

**English**
GPT-3 is the **starting gun** for the LLM era. Its lasting impacts:

1. **In-context learning became the standard interface** — every subsequent LLM (Chinchilla, PaLM, LLaMA, GPT-4, Claude) adopts the same prompting paradigm.
2. **Empirical anchor for "foundation models"** — Bommasani et al. (2021) coined the term largely in response to GPT-3.
3. **Prompt engineering, CoT, RLHF (InstructGPT 2022)** — all developed on top of the GPT-3 base.
4. **Confirmed and complicated scaling laws** — Chinchilla (2022) later showed GPT-3 was under-trained; smaller models with more data outperform it.
5. **API-only deployment** — pioneered the closed-weight, paid-API business model (OpenAI API → ChatGPT, 2022).
6. **AI safety mainstreamed** — §6 Broader Impacts pushed misinformation/bias risks into the core ML conversation.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
