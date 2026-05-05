---
title: "Pre-Reading Briefing: Training Language Models to Follow Instructions with Human Feedback (InstructGPT)"
paper_id: "38_ouyang_2022"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Training Language Models to Follow Instructions with Human Feedback (InstructGPT): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*. arXiv:2203.02155.
**Author(s)**: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (OpenAI)
**Year**: 2022

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 거대 언어 모델(LM)의 **next-token prediction objective**가 사용자가 실제로 원하는 행동(지시 따르기, 진실성, 무해성)과 misaligned 되어 있다는 문제의식에서 출발합니다. 저자들은 **3단계 RLHF (Reinforcement Learning from Human Feedback) 파이프라인**을 통해 GPT-3를 fine-tune 합니다: (1) 40명 contractor가 작성한 데모로 **Supervised Fine-Tuning (SFT)**, (2) 모델 출력의 인간 선호 비교 데이터로 **Reward Model (RM)** 학습, (3) **PPO**로 RM을 보상으로 정책을 최적화하되 SFT 모델 대비 **per-token KL penalty**를 부여해 reward hacking을 방지. 결과적으로 1.3B InstructGPT가 **175B GPT-3보다 라벨러 선호 85% vs ~50%**라는 극적인 정렬(alignment) 성과를 보였고, TruthfulQA 진실성 ~2배, hallucination 21% (vs GPT-3 41%), 독성 25% 감소를 달성하면서 NLP 벤치마크 회귀(alignment tax)는 PPO-ptx로 상쇄되었습니다.

**English**
This paper diagnoses a fundamental misalignment in large language models: the **next-token prediction objective** used to pretrain models like GPT-3 is not aligned with what users actually want (helpfulness, honesty, harmlessness). The authors propose a **three-step RLHF (Reinforcement Learning from Human Feedback) pipeline** to align GPT-3: (1) **Supervised Fine-Tuning (SFT)** on demonstrations written by 40 trained contractors; (2) train a **Reward Model (RM)** on pairwise human preference comparisons of model outputs; (3) optimize the policy with **PPO** against the RM while applying a **per-token KL penalty** against the SFT model to prevent reward hacking. The 1.3B InstructGPT model is preferred by labelers over the **175B GPT-3 (~85% vs. ~50%)** despite being **100× smaller**, with substantial improvements in TruthfulQA truthfulness, halved hallucination (21% vs 41%), and 25% reduction in toxicity under "respectful" prompts; performance regressions on public NLP benchmarks ("alignment tax") are mitigated by mixing pretraining gradients (PPO-ptx). This is the canonical recipe behind ChatGPT.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2020년 GPT-3가 175B 파라미터로 in-context learning을 보여주면서 NLP의 패러다임을 바꿨지만, 곧 한계가 드러났습니다: 모델이 **거짓말을 하고**, **편향/독성**을 보이고, **사용자 의도와 무관한 출력**을 자주 생성했습니다. 이는 "language modeling objective"(다음 토큰 예측)가 "follow user instructions helpfully and safely"라는 실제 목표와 다르기 때문입니다 (Bender et al. 2021의 "Stochastic Parrots", Bommasani et al. 2021의 "Foundation Models" 비판). 동시에 Christiano et al. (2017)의 RLHF가 Atari/MuJoCo에서, Ziegler et al. (2019), Stiennon et al. (2020)이 텍스트 요약에서 RLHF가 효과적임을 입증했습니다. InstructGPT는 이 라인을 **광범위한 자연어 지시 따르기**로 확장합니다.

**English**
In 2020, GPT-3 (175B parameters) revolutionized NLP through in-context learning, but its limitations became apparent: models **fabricated facts**, generated **biased/toxic content**, and frequently **failed to follow user intent**. The root cause: the language-modeling objective (predict next token) is misaligned with the deployment objective ("follow user instructions helpfully and safely")—a critique articulated by Bender et al. (2021) "Stochastic Parrots" and Bommasani et al. (2021) "Foundation Models". Meanwhile, Christiano et al. (2017) demonstrated RLHF on Atari/MuJoCo, and Ziegler et al. (2019) and Stiennon et al. (2020) showed RLHF beats supervised fine-tuning for text summarization. InstructGPT extends this line to **broad-distribution instruction following** on real OpenAI API customer prompts—a setting orders of magnitude more diverse than prior work.

### 타임라인 / Timeline

```
2017 ── Christiano et al.: Deep RL from Human Preferences (Atari/MuJoCo)
        │
2017 ── Schulman et al.: PPO (paper #24)
        │
2019 ── Ziegler et al.: Fine-tuning LMs from human preferences (stylistic)
        │
2020 ── Brown et al.: GPT-3 (paper #34, in-context learning)
        │
2020 ── Stiennon et al.: Learning to summarize with human feedback
        │       ★ 직접 선조 / Direct predecessor
        │
2021 ── Askell et al.: A general language assistant as a laboratory for alignment
        │
2021 ── Bender et al.: Stochastic Parrots (alignment critique)
        │
2022 ── ★★★ Ouyang et al.: InstructGPT (this paper) ★★★
        │       ─ SFT → RM → PPO + KL on broad API distribution
        │
2022 ── Wei et al.: FLAN, Sanh et al.: T0 (instruction tuning, supervised only)
        │
2022 ── ChatGPT public release (Nov, RLHF productionized)
        │
2022 ── Bai et al.: Constitutional AI (paper #40, RLAIF)
        │
2023 ── GPT-4, Llama-2-Chat (RLHF mainstream)
        │
2023+── DPO (RM-free), RLAIF, RLHF on multimodal models
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **GPT-3 (paper #34)**: pretraining objective, 175B 모델 크기, in-context learning
- **PPO (paper #24)**: clipped surrogate objective, KL penalty, advantage estimation
- **Bradley-Terry 모델**: 쌍별 선호 비교의 확률적 모델 — $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$
- **KL divergence**: $D_{KL}(\pi_{RL} \| \pi_{SFT})$의 의미와 reverse KL의 mode-seeking 성질
- **Likert scale**: 1-7점 척도로 인간이 텍스트 품질을 평가
- **Cross-entropy loss**: SFT의 maximum likelihood estimation
- **Softmax / log-sigmoid**: RM loss 계산

**English**
- **GPT-3 (paper #34)**: pretraining objective, 175B model scale, in-context learning
- **PPO (paper #24)**: clipped surrogate objective, KL penalty, advantage estimation, value function
- **Bradley-Terry model**: probabilistic model of pairwise comparisons — $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$
- **KL divergence**: $D_{KL}(\pi_{RL} \| \pi_{SFT})$ and the mode-seeking behavior of reverse KL
- **Likert scale**: 1-7 rating used by labelers to score quality
- **Cross-entropy loss**: SFT's maximum likelihood objective
- **Softmax / log-sigmoid**: components of the RM loss

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **RLHF** | Reinforcement Learning from Human Feedback. 사람의 선호를 보상 신호로 사용해 모델을 학습하는 패러다임. The paradigm of using human preferences as a reward signal for fine-tuning. |
| **SFT (Supervised Fine-Tuning)** | 사람이 작성한 데모로 GPT-3를 cross-entropy loss로 fine-tune. Cross-entropy fine-tuning of GPT-3 on labeler-written demonstrations. |
| **Reward Model (RM)** | (prompt, completion) → scalar reward. SFT에서 unembedding 제거 후 6B 크기로 학습. (prompt, completion) → scalar; trained by removing the SFT unembedding layer (6B). |
| **PPO** | Proximal Policy Optimization. clipped surrogate objective로 정책 업데이트의 KL을 제한하는 policy-gradient 알고리즘. Policy gradient with a clipped surrogate that bounds per-update KL. |
| **KL penalty (β)** | $\beta \log(\pi_{RL}/\pi_{SFT})$ per-token. reward hacking 방지 및 SFT 분포 근처 유지. Per-token penalty preventing reward hacking and keeping the policy close to the SFT prior. |
| **PPO-ptx** | PPO + pretraining gradient mix ($\gamma$ coefficient). alignment tax 완화. PPO objective augmented with a pretraining log-likelihood term to mitigate the alignment tax. |
| **Alignment tax** | 정렬 fine-tuning이 공개 NLP 벤치마크에서 야기하는 성능 저하. Performance regression on public NLP benchmarks induced by alignment fine-tuning. |
| **Reward hacking** | 정책이 RM의 결함을 익스플로잇해 실제 인간 선호 없이 높은 RM 점수를 얻는 현상. Policy exploits RM imperfections to attain high RM score without truly satisfying humans. |
| **HHH** | Helpful, Honest, Harmless — 정렬 평가의 3대 기준. Three criteria for alignment evaluation. |
| **Bradley-Terry** | 쌍별 선호의 확률 모델. 보상 차이의 sigmoid가 선호 확률. Probabilistic model where preference probability is sigmoid of reward differences. |
| **TruthfulQA** | 모델의 진실성을 평가하는 벤치마크 (Lin et al. 2021). Benchmark for measuring truthfulness of LMs. |
| **Win rate** | A의 출력이 baseline B 대비 라벨러에게 선호되는 비율. Fraction of times A's outputs are preferred over baseline B by labelers. |

---

## 5. 수식 미리보기 / Equations Preview

### Equation (1): Reward Model Loss (Bradley-Terry pairwise) / 보상 모델 손실

$$\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D}\!\left[\log \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

**한국어** $x$는 프롬프트, $y_w$는 선호된 응답, $y_l$은 비선호 응답. RM이 둘의 보상 차이를 sigmoid에 통과시켜 $y_w$가 선호될 로그-우도를 최대화. $K=4 \sim 9$개 응답을 한 번에 랭킹해 $\binom{K}{2}$개 비교 쌍을 한 batch로 처리.

**English** $x$ = prompt, $y_w$ = preferred completion, $y_l$ = dispreferred. The RM maximizes the log-likelihood (under Bradley-Terry) that $y_w$ ranks above $y_l$. $K \in [4,9]$ completions are ranked at once, producing $\binom{K}{2}$ pairs processed as one batch element.

### Equation (2): PPO Objective with KL penalty (PPO-ptx) / PPO 목적함수

$$\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{RL}}}\!\left[r_\theta(x, y) - \beta \log\!\frac{\pi_\phi^{RL}(y \mid x)}{\pi^{SFT}(y \mid x)}\right] + \gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}\!\left[\log \pi_\phi^{RL}(x)\right]$$

**한국어** 첫째 항: RM 보상에서 SFT 대비 KL penalty를 뺀 형태로 reward hacking 방지. 둘째 항: pretraining log-likelihood로 alignment tax 완화. PPO에서는 $\gamma=0$, PPO-ptx에서는 $\gamma > 0$.

**English** First term: RM reward minus per-token KL from SFT—this discourages drifting too far from the SFT prior. Second term: pretraining log-likelihood ("ptx") to counter the alignment tax. PPO sets $\gamma=0$; PPO-ptx uses $\gamma > 0$.

### Implicit: SFT Loss / SFT 손실

$$\mathcal{L}_{SFT}(\phi) = -\mathbb{E}_{(x, y) \sim D_{\text{demo}}}\!\left[\sum_t \log \pi_\phi(y_t \mid x, y_{<t})\right]$$

**한국어** 표준 next-token prediction. 16 epoch, cosine LR decay, residual dropout 0.2.

**English** Standard next-token prediction; trained 16 epochs with cosine LR decay and 0.2 residual dropout.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
- **§1 Introduction**: 핵심 결과 (1.3B preferred over 175B, TruthfulQA, hallucination, alignment tax)를 빠르게 흡수하세요. 이 문단들은 abstract의 확장입니다.
- **§3 Methods**: 가장 중요한 섹션. (1) 3단계 파이프라인 그림(Figure 2), (2) 식 (1)/(2), (3) Table 6의 데이터셋 크기를 정확히 이해하세요.
- **§3.4 Human data collection**: alignment의 사회기술적 측면 — labeler 선택, screening test, inter-annotator agreement 73%.
- **§4.1 API distribution results**: Figure 1의 win-rate 그래프가 핵심 메시지입니다.
- **§4.2 Public NLP datasets**: alignment tax와 PPO-ptx의 trade-off.
- **§5.2 Who are we aligning to?**: 정렬의 정치/윤리적 맥락 — 누구의 선호인가?
- **Appendix C**: 학습 하이퍼파라미터 (lr=9.65e-6 PPO, KL coef β=0.02, ptx γ=27.8 등).

**English**
- **§1 Introduction**: absorb the headline results quickly (1.3B preferred over 175B, TruthfulQA, hallucination, alignment tax). These paragraphs expand the abstract.
- **§3 Methods**: most important section. Master (1) the three-step pipeline (Figure 2), (2) equations (1) and (2), and (3) the dataset sizes in Table 6.
- **§3.4 Human data collection**: the sociotechnical side of alignment—labeler selection, screening test, 73% inter-annotator agreement.
- **§4.1 API distribution**: Figure 1's win-rate plot is the headline.
- **§4.2 Public NLP datasets**: the alignment tax and the PPO-ptx fix.
- **§5.2 Who are we aligning to?**: politics/ethics of alignment—whose preferences?
- **Appendix C**: training hyperparameters (lr=9.65e-6 for PPO, KL coef β=0.02, ptx γ=27.8 etc.).

---

## 7. 현대적 의의 / Modern Significance

**한국어**
이 논문은 ChatGPT(2022년 11월)의 직접적 전신이며, 이후 GPT-4, Claude, Llama-2-Chat, Gemini 등 모든 instruction-tuned LLM의 표준 레시피가 되었습니다. RLHF는 단순한 기술이 아니라 **AI alignment**라는 분야를 학문적·산업적으로 확립한 인공지능 역사에서 손꼽히는 전환점입니다.

후속 연구들은 다음 방향으로 확장합니다:
- **Constitutional AI / RLAIF (Bai et al. 2022, paper #40)**: 인간 라벨러를 AI 모델로 대체.
- **DPO (Rafailov et al. 2023)**: RM 없이 직접 선호 데이터로 정책 학습.
- **PPO-free 방법들 (Reinforce, RPO, GRPO)**: 학습 안정성 개선.
- **Reasoning RLHF (o1, R1)**: Chain-of-Thought 추론에 RL 적용.
- **Multimodal RLHF**: vision-language 모델 정렬.

**한계와 비판**: (1) 40명 라벨러는 인구학적으로 좁음 — 누구의 가치인가? (2) "Helpful"이 항상 "Safe"와 충돌; InstructGPT는 helpful 우선이라 악의적 요청에 더 toxic하게 응답. (3) Reward hacking과 sycophancy(아첨) 문제. (4) Alignment tax 완전 해소 못함.

**English**
This paper is the **direct precursor to ChatGPT** (released November 2022) and established the canonical recipe for every modern instruction-tuned LLM (GPT-4, Claude, Llama-2-Chat, Gemini). RLHF is not just a technique—it founded **AI alignment** as both an academic and industrial field, marking one of the most important inflection points in AI history.

Follow-up directions include:
- **Constitutional AI / RLAIF (Bai et al. 2022, paper #40)**: replace human labelers with AI feedback.
- **DPO (Rafailov et al. 2023)**: skip the RM, train policy directly on preference data via a clever closed-form solution.
- **PPO-free methods (Reinforce, RPO, GRPO)**: improve training stability.
- **Reasoning RLHF (o1, R1)**: apply RL to long-form chain-of-thought reasoning.
- **Multimodal RLHF**: align vision-language models.

**Limitations and critiques**: (1) 40 labelers is demographically narrow—whose values? (2) "Helpful" can conflict with "Safe"; InstructGPT prioritizes helpfulness, so when prompted to be biased it's *more* toxic than GPT-3. (3) Reward hacking and sycophancy. (4) Alignment tax not fully eliminated. (5) Mode collapse — RLHF often reduces output diversity.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
