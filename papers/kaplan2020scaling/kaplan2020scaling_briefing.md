---
title: "Pre-Reading Briefing: Scaling Laws for Neural Language Models"
paper_id: "39_kaplan_2020"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Scaling Laws for Neural Language Models: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
**Author(s)**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 **Transformer 언어 모델의 손실(cross-entropy loss)이 모델 크기 $N$, 데이터셋 크기 $D$, 학습에 사용된 컴퓨트 $C$의 멱법칙(power law) 함수**임을 7자리 이상의 규모에 걸쳐 경험적으로 입증합니다. 핵심 발견: (1) 세 가지 요인 중 하나에 의해 병목이 되지 않을 때 손실은 단순한 멱법칙을 따른다 — $L(N) \propto N^{-0.076}$, $L(D) \propto D^{-0.095}$, $L(C_{\min}) \propto C_{\min}^{-0.050}$. (2) 모델의 모양(깊이/너비/헤드 수)은 거의 영향이 없다. (3) **고정된 컴퓨트 예산에서 최적 전략은 매우 큰 모델을 적당량의 데이터에 훈련시키며 수렴 훨씬 전에 멈추는 것**이다 — 컴퓨트 10배 증가 시 모델 크기는 5배 늘려야 하지만 데이터는 1.4배만 늘리면 충분. 이 결과는 GPT-3(2020) 등 대형 모델 등장의 직접적 근거가 되었습니다.

**English**
This paper empirically demonstrates that **Transformer language model loss (cross-entropy) is a power-law function of model size $N$, dataset size $D$, and training compute $C$**, with trends spanning more than seven orders of magnitude. Key findings: (1) When not bottlenecked by the other two factors, loss follows simple power laws — $L(N) \propto N^{-0.076}$, $L(D) \propto D^{-0.095}$, $L(C_{\min}) \propto C_{\min}^{-0.050}$. (2) Model shape (depth, width, attention heads) has minimal effect. (3) **Optimal strategy under a fixed compute budget is to train very large models on a modest amount of data and stop well before convergence** — for every 10× increase in compute, model size should grow 5× but data only ~1.4×. The results directly motivated GPT-3 (2020) and the era of large language models.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2017년 Transformer(Vaswani et al.) 등장과 2018년 GPT-1, BERT의 성공으로 NLP 분야는 빠르게 사전학습-미세조정 패러다임으로 이동했습니다. 2019년 GPT-2(15억 매개변수)는 큰 모델일수록 더 잘한다는 직관적 증거를 제시했지만, **얼마나 더 좋아지고**, **추가 자원을 어디에 투입해야 가장 효율적인가**라는 양적인 질문에는 답하지 못했습니다. 동시에 OpenAI의 McCandlish et al. (2018) "An Empirical Model of Large-Batch Training"은 critical batch size 개념을 도입했습니다. 본 논문은 이 흐름을 종합하여 언어 모델 성능의 정량적 법칙을 제시합니다.

**English**
After the 2017 Transformer (Vaswani et al.) and the 2018 success of GPT-1 and BERT, NLP rapidly moved to the pretrain-finetune paradigm. GPT-2 (1.5B parameters, 2019) gave anecdotal evidence that bigger is better but did not answer the quantitative questions: *how much* better, and *where* should additional resources go? Concurrently, McCandlish et al. (2018) introduced the critical batch size as an empirical theory of large-batch training. This paper synthesizes that line of work into precise quantitative laws.

### 타임라인 / Timeline

```
2017 ─ Vaswani et al.: Transformer (paper #25)
2018 ─ Devlin et al.: BERT; Radford et al.: GPT-1
2018 ─ McCandlish et al.: Empirical Model of Large-Batch Training
2019 ─ Radford et al.: GPT-2 (1.5B params)
2020 ─ ★ Kaplan et al.: Scaling Laws (this paper)
2020 ─ Brown et al.: GPT-3 (175B params, paper #34) — direct application
2022 ─ Hoffmann et al.: Chinchilla — challenges Kaplan's allocation, says data was undersized
2023+─ LLaMA, GPT-4 — Chinchilla-optimal scaling
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Transformer 아키텍처**: $n_{\text{layer}}, d_{\text{model}}, d_{\text{ff}}, n_{\text{heads}}$ 매개변수 (paper #25 참조)
- **Cross-entropy loss와 자기회귀 언어 모델링**: $L = -\sum_t \log P(x_t | x_{<t})$
- **멱법칙(Power law)**: $y = a x^{-\alpha}$ 형태, log-log plot에서 직선
- **부동소수점 연산(FLOPs) 회계**: forward $\approx 2N$ FLOPs/token, with backward $\approx 6N$
- **Adam optimizer**, learning rate schedule (warmup + cosine decay)
- **Critical batch size** 개념 (McCandlish et al., 2018)
- **Early stopping**과 overfitting 진단

**English**
- **Transformer architecture**: hyperparameters $n_{\text{layer}}, d_{\text{model}}, d_{\text{ff}}, n_{\text{heads}}$ (see paper #25)
- **Cross-entropy loss and autoregressive language modeling**: $L = -\sum_t \log P(x_t | x_{<t})$
- **Power laws**: form $y = a x^{-\alpha}$ — straight lines on log-log plots
- **FLOP accounting**: forward $\approx 2N$ FLOPs/token, with backward $\approx 6N$
- **Adam optimizer** and learning rate schedules (warmup + cosine decay)
- **Critical batch size** concept (McCandlish et al., 2018)
- **Early stopping** and overfitting diagnostics

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Power law / 멱법칙** | $y = a x^{-\alpha}$ 형태의 관계. log-log plot에서 직선. $\alpha$는 지수(exponent). / Relation $y = a x^{-\alpha}$, linear on log-log axes; $\alpha$ is the exponent. |
| **Non-embedding parameters $N$** | 임베딩과 위치 인코딩 매개변수를 제외한 모델 매개변수 수. $N \approx 12 n_{\text{layer}} d_{\text{model}}^2$. / Model parameter count excluding embedding/positional matrices. |
| **Compute $C$ (PF-days)** | 학습에 사용된 비임베딩 계산량. $C \approx 6NBS$ (forward+backward). 1 PF-day = $8.64 \times 10^{19}$ FLOPs. / Non-embedding training compute; $C \approx 6NBS$; 1 PF-day = $8.64 \times 10^{19}$ FLOPs. |
| **Dataset size $D$ (tokens)** | BPE 토큰 단위 학습 데이터 크기. WebText2는 $2.29 \times 10^{10}$ tokens. / Training data in BPE tokens; WebText2 has $2.29 \times 10^{10}$. |
| **Critical batch size $B_{\text{crit}}$** | 시간/컴퓨트 효율의 절충점이 되는 배치 크기. $B_{\text{crit}} = B_*/L^{1/\alpha_B}$, $B_* \approx 2 \times 10^8$ tokens, $\alpha_B \approx 0.21$. / Batch size at which time-vs-compute tradeoff is optimal. |
| **$C_{\min}$** | 주어진 손실 도달에 필요한 최소 비임베딩 컴퓨트 (배치 크기 $\ll B_{\text{crit}}$ 가정). / Minimum compute to reach a given loss assuming $B \ll B_{\text{crit}}$. |
| **$S_{\min}$** | 주어진 손실 도달에 필요한 최소 학습 step 수 (배치 크기 $\gg B_{\text{crit}}$ 가정). / Minimum optimization steps to reach a given loss assuming $B \gg B_{\text{crit}}$. |
| **Compute-efficient frontier / 컴퓨트 효율적 경계** | 주어진 컴퓨트 예산에서 최저 손실을 달성하는 $(N, D, S, B)$ 조합. / The $(N, D, S, B)$ combination achieving lowest loss for a given compute budget. |
| **Sample efficiency / 표본 효율성** | 같은 손실 도달에 필요한 데이터/스텝 수. 큰 모델일수록 높음. / Data or steps needed to reach a target loss; higher for larger models. |
| **WebText2** | OpenAI가 GPT-2/3에 사용한 Reddit 링크 기반 웹 텍스트 데이터셋. 96GB, 23B tokens. / OpenAI's Reddit-derived web corpus, 96GB, 23B tokens. |
| **Aspect ratio / 종횡비** | $d_{\text{model}}/n_{\text{layer}}$. 40배 변화에도 손실은 거의 같음. / The ratio $d_{\text{model}}/n_{\text{layer}}$; loss varies only slightly across 40× range. |
| **Overfitting penalty / 과적합 페널티** | $\delta L = L(N,D)/L(N,\infty) - 1$. $N^{0.74}/D$에 의존. / $\delta L = L(N,D)/L(N,\infty) - 1$, depending on $N^{0.74}/D$. |

---

## 5. 수식 미리보기 / Equations Preview

**한국어**

**(1) 모델 크기에 대한 멱법칙 / Power law in $N$** (충분한 데이터, 수렴까지 학습 시):
$$L(N) = (N_c/N)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$
모델을 8배 키우면 손실이 $2^{-3 \alpha_N} \approx 0.85$배가 됩니다 (15% 감소).

**(2) 데이터 크기에 대한 멱법칙 / Power law in $D$** (충분히 큰 모델, early stopping 사용 시):
$$L(D) = (D_c/D)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}$$

**(3) 컴퓨트에 대한 멱법칙 / Power law in $C_{\min}$** (최적 모델 크기, 충분한 데이터 사용 시):
$$L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \quad C_c^{\min} \approx 3.1 \times 10^8 \text{ PF-days}$$

**(4) 결합식 (Joint $L(N,D)$) / Joint formula**:
$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$
$N \to \infty$일 때 $L(D)$로, $D \to \infty$일 때 $L(N)$으로 환원됩니다.

**(5) 최적 자원 배분 / Optimal allocation**:
$$N \propto C_{\min}^{0.73}, \quad B \propto C_{\min}^{0.24}, \quad S \propto C_{\min}^{0.03}, \quad D = B \cdot S$$
컴퓨트 10배 증가 시 모델 5배 (≈$10^{0.73}$), 배치 1.7배, step 1.07배 증가. 데이터는 매우 천천히.

**English**

**(1) Power law in $N$** (with sufficient data, trained to convergence):
$$L(N) = (N_c/N)^{\alpha_N}, \quad \alpha_N \approx 0.076$$
Doubling model size three times (8×) reduces loss by factor $2^{-3 \alpha_N} \approx 0.85$ (15% drop).

**(2) Power law in $D$** (with sufficiently large model, with early stopping):
$$L(D) = (D_c/D)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

**(3) Power law in $C_{\min}$** (optimal model, sufficient data):
$$L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050$$

**(4) Joint $L(N,D)$**:
$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$
Reduces to $L(D)$ as $N \to \infty$ and to $L(N)$ as $D \to \infty$.

**(5) Optimal allocation**:
$$N \propto C_{\min}^{0.73}, \quad B \propto C_{\min}^{0.24}, \quad S \propto C_{\min}^{0.03}, \quad D = B \cdot S$$
For each 10× compute increase, model grows 5×, batch 1.7×, steps barely (1.07×); data grows slowly.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
1. **§1 (Introduction)** — Figure 1과 1.2 Summary of Scaling Laws에 모든 핵심이 압축되어 있습니다. 식 (1.1)~(1.3) 외워두세요.
2. **§2 (Methods)** — Table 1의 매개변수/FLOP 회계를 정확히 이해하세요. $N \approx 12 n_{\text{layer}} d_{\text{model}}^2$, $C \approx 6NBS$.
3. **§3 (Empirical Results)** — Figure 5/6은 **shape는 중요하지 않다**는 핵심 메시지. Figure 7은 LSTM과의 비교.
4. **§4 (Infinite Data Limit)** — 식 (1.5) $L(N,D)$의 유도와 검증. Figure 9 오른쪽에서 universal curve 관찰.
5. **§5 (Training Time)** — Critical batch size $B_{\text{crit}}$, $S_{\min}$, $C_{\min}$ 도입. 식 (5.1)~(5.5)는 처음에는 어려우니 한 번 더 읽으세요.
6. **§6 (Optimal Allocation)** — 가장 영향력 있는 결론. **모델 크기를 빠르게, 데이터를 천천히** 증가시켜라.
7. **§6.3 (Contradictions and a Conjecture)** — 매우 중요. $L(C_{\min})$과 $L(D)$의 모순이 $C^* \sim 10^4$ PF-days에서 발생한다고 예측 → 실제로 GPT-3 이후 Chinchilla(2022)가 이 문제를 다시 다룹니다.

**English**
1. **§1 (Introduction)** — Figure 1 and Section 1.2 contain all the essence. Memorize Eqs. (1.1)–(1.3).
2. **§2 (Methods)** — Make sure you understand Table 1's parameter/FLOP accounting: $N \approx 12 n_{\text{layer}} d_{\text{model}}^2$, $C \approx 6NBS$.
3. **§3 (Empirical Results)** — Figures 5/6 deliver the key message: **shape doesn't matter much**. Figure 7 contrasts with LSTMs.
4. **§4 (Infinite Data Limit)** — Derivation/validation of Eq. (1.5) for $L(N,D)$. Figure 9 right shows the universal curve.
5. **§5 (Training Time)** — Introduces critical batch size $B_{\text{crit}}$, $S_{\min}$, $C_{\min}$. Eqs. (5.1)–(5.5) deserve a second read.
6. **§6 (Optimal Allocation)** — The most consequential conclusion. **Grow model size fast, data slowly.**
7. **§6.3 (Contradictions and a Conjecture)** — Crucial. The predicted contradiction between $L(C_{\min})$ and $L(D)$ at $C^* \sim 10^4$ PF-days is exactly what Chinchilla (2022) later revisits.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
이 논문은 현대 LLM 시대의 **이론적-경제적 출발점**입니다. 직접적 영향:

1. **GPT-3 (2020)의 즉각적 정당화**: Brown et al.은 본 논문의 처방대로 175B 매개변수의 거대 모델을 훈련, 적은 epoch만 학습. Few-shot 능력의 출현은 멱법칙의 부드러운 외삽 너머 새로운 현상.
2. **컴퓨트 예산 책정**: 산업계는 "X PF-days 예산에서 Y 매개변수 모델"이라는 정량적 의사결정에 이 법칙을 사용.
3. **Chinchilla 도전 (Hoffmann et al., 2022)**: 후속 연구는 Kaplan et al.의 학습률 스케줄이 작은 모델에 불리하게 설정되어 있어 **모델 크기:데이터 = 1:1로 함께 키워야** 한다고 주장. Chinchilla 70B는 GPT-3 175B보다 데이터를 4배 많이 학습받아 더 우수한 성능. 이로써 "Chinchilla-optimal" 비율이 표준이 됨.
4. **확장 가능한 과학(scalable science)의 패러다임**: 이후 vision, RL, 멀티모달 등 모든 영역에서 scaling law 측정이 의무가 됨 (e.g., Henighan et al. 2020, Hoffmann 2022, Hernandez 2021).
5. **AGI 안전성 논의**: 손실의 매끄러운 멱법칙은 일부 능력의 갑작스러운 출현(emergence)을 가리고, 이는 alignment 연구의 동기 중 하나.

**English**
This paper is the **theoretical-economic starting point of the modern LLM era**. Direct impact:

1. **Immediate justification of GPT-3 (2020)**: Brown et al. trained a 175B-parameter giant on relatively few epochs, exactly as Kaplan et al. prescribed. Few-shot in-context learning emerged as a phenomenon beyond smooth extrapolation.
2. **Compute budgeting**: Industry uses these laws to make quantitative decisions like "given X PF-days, train a Y-parameter model".
3. **Chinchilla challenge (Hoffmann et al., 2022)**: Subsequent work argued that Kaplan's learning rate schedule was unfair to smaller models, and that **model size and data should grow at roughly equal rates** (1:1, not 5:1). Chinchilla 70B trained on 4× more data than GPT-3 175B and outperformed it. "Chinchilla-optimal" ratios became the new standard.
4. **Paradigm of scalable science**: Measuring scaling laws is now mandatory in vision, RL, multimodal (e.g., Henighan et al. 2020, Hoffmann 2022, Hernandez 2021).
5. **AI safety discussions**: Smooth power laws can hide sudden emergence of capabilities, motivating alignment research.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
