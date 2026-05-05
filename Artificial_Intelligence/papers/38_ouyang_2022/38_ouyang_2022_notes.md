---
title: "Training Language Models to Follow Instructions with Human Feedback"
authors: [Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe]
year: 2022
journal: "Advances in Neural Information Processing Systems (NeurIPS)"
doi: "arXiv:2203.02155"
topic: Artificial_Intelligence
tags: [RLHF, instruction-tuning, alignment, PPO, reward-modeling, GPT-3, InstructGPT, language-models, human-feedback]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 38. Training Language Models to Follow Instructions with Human Feedback / 인간 피드백을 이용한 언어 모델 지시 따르기 학습

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 **거대 언어 모델의 정렬(alignment)** 문제를 정의하고 해결한 인공지능 역사의 분수령입니다. 핵심 진단: GPT-3 같은 모델의 pretraining objective(다음 토큰 예측)는 사용자가 원하는 것 — "instruction을 helpfully, honestly, harmlessly 따르기"(HHH 기준; Askell et al. 2021) — 과 **misaligned** 되어 있습니다. 그 결과 모델은 거짓 정보를 생성하고, 독성/편향된 출력을 만들고, 사용자 의도를 무시합니다. 저자들은 **3단계 RLHF 파이프라인**으로 이 문제를 해결합니다:

1. **Step 1 — SFT**: 40명의 contractor가 OpenAI API로부터 수집된 13K개 prompt에 대해 desired behavior 데모를 작성하고, 이를 cross-entropy loss로 GPT-3에 fine-tune.
2. **Step 2 — RM**: 각 prompt에 대해 모델이 4–9개 응답을 생성, contractor가 랭킹. 이 33K 비교 쌍 데이터셋에 Bradley-Terry 형태의 pairwise loss로 6B reward model 학습 (SFT 모델에서 unembedding 제거).
3. **Step 3 — PPO**: SFT 정책을 RM 보상으로 PPO로 fine-tune하되, **per-token KL penalty** $\beta \log(\pi_{RL}/\pi_{SFT})$를 추가해 reward hacking 방지. 추가로 pretraining log-likelihood mix($\gamma$ 항)를 더한 **PPO-ptx** 변형이 alignment tax를 완화.

핵심 결과는 충격적입니다: **1.3B InstructGPT가 175B GPT-3보다 라벨러에게 85±3% 선호** (100배 더 큰 모델 대비 같은 데이터 분포에서 압도). 또한 TruthfulQA에서 truthfulness ~2배, closed-domain hallucination 21% (vs GPT-3 41%), respectful prompt 시 toxicity 25% 감소, 그리고 RLHF 학습 비용은 GPT-3 pretraining의 ~1.6%(60 vs 3,640 PF·days). 본 논문은 ChatGPT(2022.11), GPT-4, Claude, Llama-2-Chat 등 모든 현대 instruction-tuned LLM의 직접적 전신입니다.

**English**
This paper is a watershed moment in AI history: it defines and solves the **alignment problem** for large language models. The diagnosis: the pretraining objective (next-token prediction) used to train GPT-3 is **misaligned** with what users actually want—models that follow instructions **helpfully, honestly, and harmlessly** (the HHH criteria of Askell et al. 2021). The consequence: models hallucinate, produce toxic and biased outputs, and ignore user intent. The authors solve this with a **three-step RLHF pipeline**:

1. **Step 1 — SFT**: 40 hired contractors write demonstrations of desired behavior for 13K prompts (mostly drawn from the OpenAI API Playground); GPT-3 is fine-tuned on these via cross-entropy loss.
2. **Step 2 — RM**: For each prompt, the SFT model generates 4–9 candidate responses; contractors rank them. A 33K-comparison dataset trains a 6B reward model (the SFT model with the unembedding layer replaced by a scalar head) using a Bradley-Terry pairwise loss.
3. **Step 3 — PPO**: The SFT policy is further fine-tuned with PPO against the RM reward, with a **per-token KL penalty** $\beta \log(\pi_{RL}/\pi_{SFT})$ that prevents reward hacking. A variant **PPO-ptx** mixes pretraining log-likelihood ($\gamma$ term) to mitigate the alignment tax on public NLP benchmarks.

The headline result is striking: **the 1.3B InstructGPT model is preferred to the 175B GPT-3 by labelers 85±3% of the time**—even though GPT-3 has 100× more parameters. Truthfulness on TruthfulQA roughly doubles, closed-domain hallucination drops to 21% (vs GPT-3's 41%), toxicity falls 25% under "respectful" prompts, and the entire RLHF pipeline costs only ~1.6% of GPT-3 pretraining (60 vs 3,640 petaflop·days). This paper is the direct precursor to ChatGPT (released November 2022), GPT-4, Claude, and Llama-2-Chat, and it founded the modern field of LLM alignment.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 GPT-3 등 거대 LM이 **misalignment**를 보인다는 점에서 출발합니다 (Bender et al. 2021; Bommasani et al. 2021). 이는 단순히 모델이 더 커진다고 해결되지 않는 문제입니다. 저자들은 alignment를 Askell et al. (2021)을 따라 **HHH (Helpful, Honest, Harmless)** 기준으로 정의합니다.

3단계 방법론이 Figure 2에 요약됩니다:
- **Step 1 (SFT)**: 라벨러 데모 → GPT-3 fine-tune.
- **Step 2 (RM)**: 라벨러가 모델 응답을 랭킹 → reward model 학습.
- **Step 3 (PPO)**: RM을 보상 함수로 RL fine-tune.

핵심 결과 7가지가 §1에서 미리 발표됩니다:
1. **Labelers significantly prefer InstructGPT outputs**: 175B InstructGPT vs 175B GPT-3 = **85±3%**, vs few-shot GPT-3 = **71±4%**. **1.3B InstructGPT > 175B GPT-3** (Figure 1).
2. **Truthfulness on TruthfulQA ~2배**: hallucination rate 21% vs 41%.
3. **Toxicity 25% 감소** (respectful prompt 시).
4. **Alignment tax** (NLP 벤치마크 회귀)는 PPO-ptx로 완화.
5. **Held-out labelers**도 InstructGPT를 선호 → over-fitting 아님.
6. **Public NLP datasets** (FLAN, T0)는 API 분포 반영 못함; InstructGPT가 78–79% 선호.
7. **Generalization**: 코드, 비영어 instruction에도 일반화.
8. **Simple mistakes 잔존**: false premise, over-hedging.

**English**
The paper opens with a clear articulation of the alignment problem. The pretraining objective—predict the next token on a webpage—is fundamentally different from the deployment objective—follow user instructions helpfully and safely. This mismatch produces fabrication, toxicity, bias, and instruction non-compliance, even at 175B parameters. Following Askell et al. (2021), the authors adopt the **HHH framework** (Helpful, Honest, Harmless) as their alignment target.

The three-step pipeline is summarized in Figure 2: SFT on demonstrations → RM on rankings → PPO on RM with KL constraint. The introduction previews seven main findings: labelers strongly prefer InstructGPT (1.3B InstructGPT > 175B GPT-3 in head-to-head), TruthfulQA roughly doubles, hallucination halves, toxicity drops, the alignment tax is mitigated by PPO-ptx, results generalize to held-out labelers, public NLP datasets do not capture the API use distribution, the model generalizes to non-English and code instructions, and InstructGPT still makes simple mistakes (false premises, over-hedging).

### Part II: Methods (§3) / 방법론

#### 3.1 High-level methodology / 전반적 방법론

**한국어**
방법론은 Ziegler et al. (2019), Stiennon et al. (2020) (text summarization)을 따릅니다. 출발점:
- Pretrained GPT-3 (Brown et al. 2020) 1.3B / 6B / 175B,
- API에서 수집된 prompt 분포 + 라벨러 작성 prompt,
- 40명 trained contractor 팀.

**English**
The methodology directly builds on Ziegler et al. (2019) and Stiennon et al. (2020), who applied RLHF to stylistic continuation and text summarization. The starting points are: (a) pretrained GPT-3 in three sizes (1.3B/6B/175B), (b) a prompt distribution drawn from the OpenAI API plus labeler-written prompts, (c) a team of 40 hired contractors.

#### 3.2 Dataset / 데이터셋

**한국어**
**프롬프트 출처**: 주로 OpenAI API Playground에 제출된 프롬프트(InstructGPT 초기 버전 사용 시). 보호자: prefix 중복 제거, 사용자당 200개 제한, train/val/test split을 user ID 기준으로 분리, PII 필터링.

**라벨러 작성 prompt 3종** (초기 부트스트랩):
- **Plain**: 임의 task 자유 작성.
- **Few-shot**: instruction + 다중 query/response 예시.
- **User-based**: API 대기자 명단의 use case에 맞춰 작성.

**3개 데이터셋 (Table 6)**:
- **SFT dataset**: 13K training prompts (API + labeler).
- **RM dataset**: 33K training prompts.
- **PPO dataset**: 31K training prompts (API only, no labels).

**Use case 분포 (Table 1)**: Generation 45.6%, Open QA 12.4%, Brainstorming 11.2%, Chat 8.4%, Rewrite 6.6%, Summarization 4.2%, Classification 3.5%, Other 3.5%, Closed QA 2.6%, Extract 1.9% — **96% English**.

**English**
Prompts come primarily from the OpenAI API Playground (using earlier InstructGPT versions). Safeguards: deduplicate by long shared prefix, cap at 200 prompts per user, split train/val/test by user ID, strip PII.

To bootstrap, labelers wrote three prompt types: **Plain** (free-form tasks), **Few-shot** (instruction + multi-turn examples), **User-based** (modeled on API waitlist use cases).

The pipeline produces three datasets (Table 6): **SFT** (13K prompts, API + labeler), **RM** (33K prompts, used for ranking), **PPO** (31K prompts, API only, unlabeled). The use-case distribution skews heavily generative: Generation 45.6%, Open QA 12.4%, Brainstorming 11.2%, etc. The corpus is over 96% English.

#### 3.4 Human data collection / 인간 데이터 수집

**한국어**
**라벨러 선발**: Upwork와 ScaleAI를 통해 약 40명을 선별. Screening test로 (1) 민감한 prompt 식별 능력, (2) 라벨링 task에서 연구자 합의율을 평가. 인구통계는 주로 미국·동남아 거주 영어 사용자.

**Inter-annotator agreement**: training labelers 간 **72.6 ± 1.5%**, held-out labelers 간 **77.3 ± 1.3%** (Stiennon et al. 2020의 researcher-researcher agreement 73 ± 4%와 유사).

**메타데이터 (Table 3)**: overall quality (1-7 Likert) + binary 11종 (instruction following, hallucination, toxicity 카테고리, bias, harmful advice 등).

**Generalization 실험**: 별도의 held-out labeler 그룹을 고용 (training data 미생성). 이들도 InstructGPT를 비슷한 비율로 선호 → 모델이 단지 training labelers의 idiosyncratic 선호에 over-fit한 것이 아님을 입증.

**English**
**Labeler selection**: ~40 contractors hired through Upwork and ScaleAI, filtered by a screening test measuring (1) sensitivity to harmful prompts and (2) agreement with researchers on detailed labeling tasks. Demographics: mostly English-speaking, US/Southeast Asia.

**Inter-annotator agreement**: training labelers agree **72.6 ± 1.5%** of the time; held-out labelers agree **77.3 ± 1.3%** with each other—comparable to the 73 ± 4% researcher-researcher agreement in Stiennon et al. (2020).

**Metadata (Table 3)**: overall quality on a 1-7 Likert scale, plus 11 binary attributes (failure to follow instruction, hallucination, sexual/violent content, denigration, harmful advice, expresses opinion/moral judgment, etc.).

**Generalization test**: a separately hired held-out labeler group (no training data overlap) ranks InstructGPT comparably to training labelers, showing the alignment is not an artifact of idiosyncratic training-labeler preferences.

#### 3.5 Models (the three steps) / 모델 (3단계)

**한국어**

**Step 1: Supervised Fine-Tuning (SFT)**
- Loss: standard next-token cross-entropy on labeler demonstrations.
- 16 epochs, cosine LR decay, residual dropout 0.2.
- 모델은 1 epoch 후 validation loss에서 overfit하지만, 더 학습할수록 RM score와 인간 선호가 개선됨 → **validation loss와 alignment 성능의 분리**라는 흥미로운 현상.

**SFT loss**:
$$\mathcal{L}_{SFT}(\phi) = -\mathbb{E}_{(x, y) \sim D_{\text{demo}}}\!\left[\sum_{t=1}^{|y|} \log \pi_\phi(y_t \mid x, y_{<t})\right]$$

**Step 2: Reward Modeling (RM)**
- 시작점: SFT 모델에서 final unembedding layer 제거, 스칼라 출력 head 추가.
- **6B 모델만 사용** (175B RM은 학습 불안정, value function으로 부적합).
- Labelers가 한 prompt 당 $K \in [4, 9]$개 응답을 동시에 랭킹 → $\binom{K}{2}$개 비교 쌍.
- **단일 batch element**로 처리: 각 응답에 대해 단 1번의 forward pass로 $\binom{K}{2}$ 쌍 모두 계산 → 효율성 + over-fitting 방지.

**RM loss (Bradley-Terry pairwise)**:
$$\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D}\!\left[\log \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

여기서 $r_\theta(x, y)$는 (prompt, completion) → 스칼라 보상. $\sigma$는 sigmoid.

**Reward normalization**: RM loss는 reward에 대한 shift-invariant이므로, fine-tuning 후 bias를 조정해 demonstration 평균 reward = 0이 되도록.

**Step 3: Reinforcement Learning (PPO / PPO-ptx)**
- 환경: **bandit environment** — 임의의 prompt 받음 → 응답 생성 → RM이 reward 부여 → episode 종료.
- Value function: RM에서 초기화.
- Per-token KL penalty: SFT 모델 대비 발산 방지.
- PPO-ptx: pretraining gradient mix.

**PPO-ptx objective**:
$$\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{RL}}}\!\left[r_\theta(x, y) - \beta \log\!\frac{\pi_\phi^{RL}(y \mid x)}{\pi^{SFT}(y \mid x)}\right] + \gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}\!\left[\log \pi_\phi^{RL}(x)\right]$$

PPO에서 $\gamma=0$, PPO-ptx에서 $\gamma>0$ (Appendix C에 따르면 $\gamma=27.8$, $\beta=0.02$).

**English**

**Step 1: Supervised Fine-Tuning (SFT)**
- Standard next-token cross-entropy on labeler demonstrations.
- 16 epochs, cosine LR decay, residual dropout 0.2.
- A subtle finding: SFT models overfit on validation loss after 1 epoch, yet RM score and human preferences keep improving with more epochs—a striking dissociation between validation loss and alignment performance.

**SFT loss**:
$$\mathcal{L}_{SFT}(\phi) = -\mathbb{E}_{(x, y) \sim D_{\text{demo}}}\!\left[\sum_{t} \log \pi_\phi(y_t \mid x, y_{<t})\right]$$

**Step 2: Reward Modeling**
- Initialized from the SFT model with the final unembedding layer replaced by a scalar reward head.
- **Only the 6B RM is used** (175B RM training was unstable and unsuitable as the PPO value function).
- Labelers rank $K \in [4, 9]$ responses per prompt, producing $\binom{K}{2}$ comparison pairs.
- All $\binom{K}{2}$ pairs from one prompt are processed as a **single batch element**—a single forward pass per completion. This both (a) cuts compute by a factor of $K-1$ and (b) prevents the overfitting that occurs when each pair is treated as an independent datapoint.

**RM loss (Bradley-Terry pairwise)**:
$$\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D}\!\left[\log \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

where $r_\theta(x, y)$ outputs a scalar reward and $\sigma$ is the sigmoid. This is exactly the maximum-likelihood estimator under the Bradley-Terry preference model.

**Reward normalization**: because the RM loss is shift-invariant in reward, the bias is calibrated post-hoc so that demonstration responses have mean reward 0.

**Step 3: Reinforcement Learning (PPO / PPO-ptx)**
- Environment: a **bandit**—each episode presents a random prompt, the policy emits a response, the RM scores it, and the episode ends.
- The PPO value function is initialized from the RM (a useful warm-start since both share the SFT backbone).
- A per-token KL penalty against the SFT model prevents reward hacking.
- PPO-ptx adds a pretraining log-likelihood term to mitigate the alignment tax.

**PPO-ptx objective**:
$$\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{RL}}}\!\left[r_\theta(x, y) - \beta \log\!\frac{\pi_\phi^{RL}(y \mid x)}{\pi^{SFT}(y \mid x)}\right] + \gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}\!\left[\log \pi_\phi^{RL}(x)\right]$$

Plain PPO uses $\gamma = 0$; PPO-ptx uses $\gamma > 0$ (Appendix C reports $\gamma=27.8$, $\beta=0.02$).

### Part III: Results (§4) / 결과

#### 4.1 API distribution / API 분포

**한국어**
**Figure 1 (the headline)**: 175B SFT를 baseline으로 win-rate를 측정. 결과:
- GPT-3 (no prompt): ~10–20% win-rate.
- GPT-3 (few-shot prompted): ~40–50%.
- SFT 175B: ~50% (baseline).
- PPO / PPO-ptx (1.3B, 6B, 175B 모두): SFT를 능가.
- **PPO 1.3B > GPT-3 175B** — 100배 작은 모델이 더 선호됨.

**직접 비교**: 175B InstructGPT vs 175B GPT-3 = **85±3%** preferred. vs few-shot GPT-3 = **71±4%**.

**Figure 4 (메타데이터)**: PPO 모델은 GPT-3 대비:
- Attempts correct instruction: 더 자주.
- Follows explicit constraints: 더 잘.
- Hallucinations: 절반 (21% vs 41%).
- Customer-appropriate language: 더 자주.

**Held-out labelers** (Figure 3): training labelers와 비슷한 선호 패턴 → InstructGPT는 overfit이 아님. RM은 5-fold CV에서 held-out group 예측 정확도 69.6%, train group 72.4% — 일반화 양호.

**FLAN/T0 비교** (Figure 5): instruction-tuning 데이터셋들도 GPT-3보다 좋지만, SFT보다 못함. InstructGPT vs FLAN = **78±4%**, vs T0 = **79±4%**. 이유: (1) public NLP datasets은 classification/QA에 편중 (API의 18%), API는 generation/brainstorming(57%)이 주류. (2) 입력 다양성 부족.

**English**
**Figure 1 (headline)**: win rate against the 175B SFT baseline, broken out by model size.
- GPT-3 (raw): ~10–20% win rate.
- GPT-3 (few-shot prompted): ~40–50%.
- 175B SFT: ~50% (baseline by definition).
- PPO / PPO-ptx at 1.3B, 6B, 175B: all beat SFT.
- **The 1.3B PPO model beats the 175B GPT-3** — a 100× parameter gap closed by alignment.

**Direct comparison**: 175B InstructGPT vs 175B GPT-3 = **85±3%** preferred; vs few-shot GPT-3 = **71±4%**.

**Figure 4 (metadata axes)**: relative to GPT-3, PPO models more often attempt the correct instruction, better follow explicit constraints (e.g., word/length limits), hallucinate less (21% vs 41% on closed-domain tasks), and use language more appropriate for a customer assistant.

**Held-out labelers** (Figure 3): roughly the same preference pattern as training labelers, ruling out the "model only learned the training labelers' idiosyncratic taste" hypothesis. RM 5-fold CV: 69.6% on held-out groups, 72.4% on training groups—decent generalization.

**FLAN/T0 baselines** (Figure 5): instruction-tuning corpora outperform raw GPT-3 but underperform SFT. InstructGPT wins **78±4%** vs FLAN, **79±4%** vs T0. Two reasons: (1) public NLP datasets are heavy on classification/QA (18% of API use cases) but light on generation/brainstorming (57% of API). (2) Input diversity is limited.

#### 4.2 Public NLP datasets / 공개 NLP 벤치마크

**한국어**
**Truthfulness (TruthfulQA, Figure 6)**: PPO 모델이 truthful + informative 답변을 GPT-3의 ~2배 빈도로 생성. 1.3B PPO-ptx만 GPT-3 동급 모델보다 약간 낮음. "Instruction+QA" 프롬프트 사용 시 PPO 모델은 "I have no comment"로 epistemic humility 표현.

**Toxicity (RealToxicityPrompts, Figure 7)**:
- "Respectful" prompt 시 InstructGPT가 GPT-3보다 ~25% 덜 toxic.
- "No prompt" 시 차이 없음.
- **악의적 prompt 시 InstructGPT가 더 toxic** — helpful 우선이 backfire.

**Bias (Winogender, CrowS-Pairs)**: GPT-3 대비 개선 없음. PPO-ptx는 respectful instruction 시 entropy 감소 (편향 증가).

**Alignment Tax**:
- PPO 단독은 SQuAD, DROP, HellaSwag, WMT 15 fr-en에서 회귀.
- **PPO-ptx**가 회귀 대부분 완화 (HellaSwag는 GPT-3 능가).
- DROP, SQuADv2, translation은 일부 잔존.
- Pretraining mix가 KL coefficient 증가보다 효과적 (Figure 33 vs 34).

**English**
**Truthfulness (TruthfulQA, Figure 6)**: PPO models produce truthful+informative answers about **twice as often** as GPT-3. The exception is the 1.3B PPO-ptx, which slightly underperforms its same-size GPT-3 counterpart. With an "Instruction+QA" prompt that licenses "I have no comment," PPO models exhibit appropriate epistemic humility; raw GPT-3 does not.

**Toxicity (RealToxicityPrompts, Figure 7)**:
- Under a "respectful" prompt, InstructGPT produces ~25% fewer toxic outputs than GPT-3.
- Under "no prompt," parity.
- Under explicitly toxic prompts, **InstructGPT is *more* toxic than GPT-3**—a direct consequence of prioritizing helpfulness over harmlessness during training.

**Bias (Winogender, CrowS-Pairs)**: no improvement over GPT-3. PPO-ptx, when "instructed to act respectfully," shows lower entropy (i.e., higher bias)—a striking failure mode.

**Alignment Tax**:
- Plain PPO regresses on SQuAD, DROP, HellaSwag, WMT 15 fr-en translation.
- **PPO-ptx mitigates most regressions** (and even surpasses GPT-3 on HellaSwag).
- DROP, SQuADv2, translation regressions are reduced but not eliminated.
- Mixing in pretraining gradients is more effective than simply increasing the KL coefficient (Figures 33 vs 34); raising $\beta$ alone causes large reward losses without recovering benchmark performance.

#### 4.3 Qualitative results / 정성적 결과

**한국어**
**일반화** (Figure 8):
- **비영어 instruction**: 프랑스어로 "고대 그리스로 시간 여행하는 개구리 이야기" 요청 시 GPT-3는 비슷한 prompt들을 나열, InstructGPT는 실제 이야기 작성. 단, 종종 영어로 응답.
- **코드 QA/요약**: InstructGPT가 더 안정적으로 코드 질문에 답변.

**Failure modes** (Figure 9):
- **False premise** ("Why is it important to eat socks after meditating?"): InstructGPT는 가짜 전제를 받아들이고 그럴듯한 이론을 만들어냄.
- **Over-hedging**: 단순한 질문에 "여러 가지 답이 있을 수 있다"며 우물쭈물.
- **Multiple constraints**: "1930년대 프랑스 배경 영화 10개" 같은 다중 제약 처리 미흡.

**English**
**Generalization** (Figure 8):
- **Non-English**: prompted in French to write a story about a frog time-traveling to ancient Greece, GPT-3 just lists similar prompts, while InstructGPT writes an actual story (but sometimes responds in English).
- **Code Q&A and summarization**: InstructGPT reliably answers questions about code where GPT-3 needs careful prompting.

**Failure modes** (Figure 9):
- **False premise**: asked "Why is it important to eat socks after meditating?", InstructGPT accepts the false premise and invents plausible-sounding theories.
- **Over-hedging**: simple questions get "there is no clear answer..." responses—a likely artifact of labeler instructions to reward epistemic humility.
- **Multiple constraints**: "list 10 movies set in 1930s France" and similar multi-constraint prompts confuse the model.

### Part IV: Discussion (§5) / 논의

**한국어**

**§5.1 Implications for alignment research**:
1. **정렬 비용은 pretraining 대비 작다**: 175B SFT = 4.9 PF·days, 175B PPO-ptx = 60 PF·days vs GPT-3 pretraining = 3,640 PF·days. → "100배 큰 모델 만들기보다 정렬에 투자하는 것이 비용 효과적".
2. **일반화의 증거**: 비영어, 코드처럼 적게 학습한 도메인에도 instruction-following이 전이됨.
3. **Alignment tax 완화**: PPO-ptx로 대부분 해소.
4. **Alignment 연구의 grounding**: 추상적·이론적이던 alignment를 실제 deployed system에 적용.

**§5.2 Who are we aligning to?**: 우리는 (a) training labelers의 선호, (b) 연구자 작성 instruction, (c) OpenAI 고객의 use case, (d) waitlist 통과한 사람들에게 정렬됩니다. 누구의 가치를 정렬할 것인가는 sociotechnical 문제이며 단일 답이 없습니다.

**§5.3 Limitations**: (a) 40명 contractor의 인구학적 좁음, (b) 비교당 1명 라벨러만 (cost), (c) 모델이 여전히 toxic/biased, (d) helpful 우선이 harm 유발 가능.

**§5.4 Open questions**: adversarial data, pretraining filtering, expert iteration, behavior cloning, DPO 같은 RM-free 방법, edit 기반 feedback, principle-based alignment.

**English**

**§5.1 Implications for alignment research**:
1. **Cost of alignment is modest**: 175B SFT = 4.9 PF·days, 175B PPO-ptx = 60 PF·days, vs 3,640 PF·days for GPT-3 pretraining. Investing in alignment is more cost-effective than scaling another 100×.
2. **Generalization evidence**: instruction-following transfers to under-supervised domains (non-English, code).
3. **Alignment tax mitigation**: PPO-ptx handles most of it—a "low-tax" alignment technique.
4. **Grounding alignment research**: shifts alignment from abstract theory to deployed systems.

**§5.2 Who are we aligning to?**: a frank acknowledgment that we are aligning to (a) training-labeler preferences, (b) researcher-written instructions, (c) OpenAI customer use cases, (d) the waitlist-filtered sample of API users. There is no single right answer to whose values to align to.

**§5.3 Limitations**: (a) 40 contractors are demographically narrow; (b) most comparisons have only one labeler (cost), so disagreements are not surfaced; (c) models still produce toxic/biased outputs; (d) prioritizing helpfulness can amplify harms when users request them.

**§5.4 Open questions**: adversarial data collection (Dinan et al. 2019b), pretraining-data filtering (Ngo et al. 2021), expert iteration (Anthony et al. 2017; Silver et al. 2017), behavior cloning, RM-free methods (foreshadowing DPO), edit-based feedback, principle-based alignment (Gabriel 2020).

---

## 3. Key Takeaways / 핵심 시사점

1. **Alignment is a learning problem, not a scaling problem** — **정렬은 스케일링 문제가 아닌 학습 문제**.
   **English** Making models bigger does not make them follow instructions better. 1.3B InstructGPT > 175B GPT-3 demonstrates this concretely. The bottleneck is the *objective*, not the parameter count.
   **한국어** 모델을 더 크게 만든다고 instruction을 더 잘 따르지 않습니다. 1.3B InstructGPT > 175B GPT-3가 이를 정확히 보여줍니다. 병목은 *목적함수*이지 파라미터 수가 아닙니다.

2. **The three-step recipe (SFT → RM → PPO+KL) is now canonical** — **3단계 RLHF 레시피가 표준이 됨**.
   **English** Every modern instruction-tuned LLM (ChatGPT, GPT-4, Claude, Llama-2-Chat, Gemini) uses this recipe or a close variant. SFT teaches format; RM captures preferences; PPO+KL optimizes against preferences while staying close to the SFT prior.
   **한국어** 모든 현대 instruction-tuned LLM이 이 레시피 또는 변형을 사용합니다. SFT는 포맷 학습, RM은 선호 포착, PPO+KL은 SFT prior 근처에서 선호 최대화.

3. **The KL penalty is the single most important regularizer in RLHF** — **KL penalty는 RLHF의 핵심 정규화자**.
   **English** Without $\beta \log(\pi_{RL}/\pi_{SFT})$, the policy quickly exploits RM imperfections (reward hacking) and collapses to degenerate outputs. The KL term anchors the policy to the SFT manifold of fluent text.
   **한국어** $\beta \log(\pi_{RL}/\pi_{SFT})$ 없이는 정책이 RM의 결함을 익스플로잇해(reward hacking) degenerate한 출력으로 붕괴합니다. KL 항이 정책을 SFT의 fluent text manifold에 anchor 시킵니다.

4. **Alignment tax is real but mitigable via PPO-ptx** — **Alignment tax는 실재하지만 PPO-ptx로 완화 가능**.
   **English** Plain PPO regresses on SQuAD/DROP/HellaSwag/translation. Mixing pretraining gradients (PPO-ptx with $\gamma$ term) recovers most of this without sacrificing labeler preference. Increasing KL alone does not work.
   **한국어** PPO 단독은 SQuAD/DROP/HellaSwag/번역에서 회귀합니다. Pretraining gradient mix($\gamma$ 항)가 라벨러 선호를 희생하지 않고 대부분 회복시킵니다. KL coefficient만 키우는 것은 효과 없음.

5. **Public NLP benchmarks are not what users actually want** — **공개 NLP 벤치마크는 사용자 니즈를 반영하지 못함**.
   **English** API users want generation, brainstorming, and chat (~57% of prompts). Public datasets focus on classification/QA (~18%). FLAN and T0 instruction-tune well for benchmarks but lose to SFT on the API distribution.
   **한국어** API 사용자는 generation, brainstorming, chat을 원합니다 (~57%). 공개 데이터셋은 classification/QA에 편중 (~18%). FLAN과 T0는 벤치마크에는 좋으나 API 분포에서는 SFT보다 못함.

6. **"Helpful" can conflict with "Harmless"** — **"Helpful"과 "Harmless"는 충돌할 수 있음**.
   **English** When users explicitly request toxic outputs, InstructGPT produces *more* toxic outputs than GPT-3. The model has learned to follow instructions—including bad ones. This is a fundamental tension in RLHF that motivates Constitutional AI and refusal training.
   **한국어** 사용자가 명시적으로 toxic 출력을 요청하면, InstructGPT는 GPT-3보다 *더* toxic합니다. 모델은 instruction 따르기를 학습했고, 나쁜 instruction도 포함됩니다. 이는 Constitutional AI와 refusal training을 정당화하는 본질적 긴장.

7. **Alignment cost is ~1.6% of pretraining cost** — **정렬 비용은 pretraining의 ~1.6%**.
   **English** 60 PF·days for 175B PPO-ptx vs 3,640 PF·days for GPT-3 pretraining. Aligning existing models is dramatically cheaper than training larger ones, fundamentally shifting the economics of capability vs alignment investment.
   **한국어** 175B PPO-ptx 60 PF·days vs GPT-3 pretraining 3,640 PF·days. 기존 모델 정렬이 더 큰 모델 학습보다 훨씬 저렴 — capability vs alignment 투자 경제학을 근본적으로 바꿈.

8. **Generalization to held-out instructions and labelers is non-trivial** — **held-out instruction과 labeler에 대한 일반화는 자명하지 않음**.
   **English** InstructGPT generalizes to non-English and code (rare in fine-tuning data) and to held-out labelers (similar preferences). This suggests the model learned a more abstract notion of "follow instructions" rather than memorizing patterns.
   **한국어** InstructGPT는 fine-tuning 데이터에 드문 비영어와 코드, 그리고 held-out labelers에게도 일반화. 모델이 패턴 암기가 아닌 "instruction 따르기"의 추상적 개념을 학습했음을 시사.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Step 1 — SFT Loss / SFT 손실

표준 next-token cross-entropy:

$$\mathcal{L}_{SFT}(\phi) = -\mathbb{E}_{(x, y) \sim D_{\text{demo}}}\!\left[\sum_{t=1}^{|y|} \log \pi_\phi(y_t \mid x, y_{<t})\right]$$

- $\phi$: 정책(policy) 파라미터 (GPT-3 backbone). Policy parameters.
- $D_{\text{demo}}$: 13K labeler demonstrations. 13K labeler-written demonstrations.
- $\pi_\phi(y_t \mid x, y_{<t})$: 토큰 $y_t$의 conditional 확률. Conditional probability of token $y_t$.

### 4.2 Step 2 — Bradley-Terry Reward Model Loss / 보상 모델 손실

$K$개 응답 비교에서 $\binom{K}{2}$ 쌍에 대한 손실:

$$\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D_{\text{RM}}}\!\left[\log \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

**유도 / Derivation**: Bradley-Terry 모델 가정 — 응답 $y$의 latent quality를 $r(y)$라 하면, $y_w$가 $y_l$보다 선호될 확률:

$$P(y_w \succ y_l \mid x) = \frac{\exp(r(y_w))}{\exp(r(y_w)) + \exp(r(y_l))} = \sigma(r(y_w) - r(y_l))$$

이를 cross-entropy로 학습 → 위 손실. 모든 $K$개 응답을 한 번의 forward pass로 처리해 $\binom{K}{2}$ 쌍 손실을 계산하므로 효율성과 over-fitting 방지를 동시 달성.

**Variables / 변수**:
- $\theta$: RM 파라미터 (6B). RM parameters (6B).
- $y_w, y_l$: preferred / dispreferred completions.
- $\sigma(z) = 1/(1+e^{-z})$: sigmoid.

### 4.3 Step 3 — PPO Objective with KL Penalty / PPO 목적함수

**PPO**:
$$\text{objective}_{\text{PPO}}(\phi) = \mathbb{E}_{(x,y) \sim \pi_\phi^{RL}}\!\left[r_\theta(x, y) - \beta \log\!\frac{\pi_\phi^{RL}(y \mid x)}{\pi^{SFT}(y \mid x)}\right]$$

**PPO-ptx (with pretraining mix)**:
$$\text{objective}_{\text{PPO-ptx}}(\phi) = \text{objective}_{\text{PPO}}(\phi) + \gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}\!\left[\log \pi_\phi^{RL}(x)\right]$$

**Term-by-term / 항별 해석**:
- $r_\theta(x, y)$: RM의 스칼라 보상 (정렬 신호). RM scalar reward — the alignment signal.
- $-\beta \log(\pi_\phi^{RL}/\pi^{SFT})$: per-token KL penalty. SFT 분포에서 멀어질수록 비용 증가 → reward hacking 방지. Penalizes drifting from the SFT prior; prevents reward hacking and degenerate solutions.
- $\gamma \log \pi_\phi^{RL}(x)$ (PPO-ptx만): pretraining log-likelihood. 공개 NLP 벤치마크 회귀 완화. Pretraining log-likelihood; mitigates the alignment tax.

**Hyperparameters (Appendix C)**: $\beta \approx 0.02$, $\gamma \approx 27.8$ (PPO-ptx), $\text{lr} = 9.65 \times 10^{-6}$, batch 512, 256K episodes.

### 4.4 KL Penalty Equivalence / KL penalty 등가성

per-token KL penalty는 다음과 등가 (under expectation):

$$\beta\, \mathbb{E}_{y \sim \pi_\phi^{RL}(\cdot \mid x)}\!\left[\log\!\frac{\pi_\phi^{RL}(y \mid x)}{\pi^{SFT}(y \mid x)}\right] = \beta\, D_{KL}\!\left(\pi_\phi^{RL}(\cdot \mid x) \,\|\, \pi^{SFT}(\cdot \mid x)\right)$$

**한국어** Reverse KL 형태이므로 mode-seeking — 정책이 SFT의 high-density 영역으로 수렴하는 경향이 있어 mode collapse(다양성 감소) 부작용도 있음.

**English** This is a reverse-KL term, hence mode-seeking—the policy tends to concentrate on high-density regions of the SFT prior, which is a known cause of post-RLHF mode collapse / diversity reduction.

### 4.5 Worked Example: Prompt → SFT → RM → PPO Update / 워크드 예제

**Setting / 설정**: 프롬프트 $x$ = "Explain photosynthesis in two sentences."

**(a) SFT response**: $y_1$ = "Photosynthesis is the process by which plants convert sunlight into chemical energy. They use chlorophyll to absorb light, water from the soil, and CO₂ from the air to produce glucose and oxygen."

**(b) Alternative SFT samples for ranking** (라벨러가 4개 응답 받음):
- $y_1$ (위, 정확하고 간결): label as $y_w$ (rank 1).
- $y_2$ = "Photosynthesis is when plants eat sunlight." → simple but inaccurate (rank 3).
- $y_3$ = "Photosynthesis is a complex biochemical process. It involves many enzymatic reactions in chloroplasts including the Calvin cycle and electron transport chain." → too technical for "two sentences" constraint (rank 2).
- $y_4$ = "I cannot answer that question." → unhelpful (rank 4 = $y_l$ candidate).

**(c) RM training signal**: 6 비교 쌍 ($\binom{4}{2}=6$):
- $y_1 \succ y_2$, $y_1 \succ y_3$, $y_1 \succ y_4$, $y_3 \succ y_2$, $y_3 \succ y_4$, $y_2 \succ y_4$.

각 쌍에 대해 $-\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$ 손실로 RM 업데이트. 학습 후 RM은 다음과 같은 스칼라 반환 (예시 값):
- $r_\theta(x, y_1) \approx +1.8$,
- $r_\theta(x, y_2) \approx -0.5$,
- $r_\theta(x, y_3) \approx +0.6$,
- $r_\theta(x, y_4) \approx -1.2$.

**(d) PPO update**: 정책 $\pi_\phi^{RL}$이 $x$에 대해 $\hat{y}$ 샘플링. 가령 $\hat{y} = $ "Photosynthesis lets plants make food from sunlight, water, and air. The byproduct is oxygen, which we breathe."
- RM 평가: $r_\theta(x, \hat{y}) = +1.5$.
- KL term: $\beta \cdot \log(\pi_\phi^{RL}(\hat{y}|x) / \pi^{SFT}(\hat{y}|x)) \approx 0.02 \cdot 0.3 = 0.006$ (정책이 SFT와 매우 가까움).
- 총 reward: $1.5 - 0.006 \approx 1.494$.
- Pretraining mix: 별도 batch에서 $\gamma \log \pi_\phi^{RL}(x_{\text{pretrain}})$를 더해 pretraining 분포 유지.

**Reward hacking 시나리오**: KL penalty 없을 때 정책이 학습 후 출력:
- $\hat{y}_{\text{hack}} = $ "Photosynthesis Photosynthesis Photosynthesis... [반복 256토큰]"
- RM이 잘못 학습되어 단어 "photosynthesis" 빈도에 양의 가중을 둘 경우 $r_\theta = +5.0$ 같은 높은 점수.
- 그러나 KL은 폭발: $\log(\pi_\phi^{RL}/\pi^{SFT}) \to \infty$ (SFT는 절대 이런 출력 안 함).
- $\beta = 0.02$가 KL을 충분히 높게 페널티하면 정책이 이 솔루션 회피.

**English version of the example**: prompt $x$ = "Explain photosynthesis in two sentences." The SFT model emits $y_1$ (concise, accurate), $y_2$ (oversimplified), $y_3$ (overly technical, violates length constraint), $y_4$ ("I cannot answer"). The labeler ranks $y_1 > y_3 > y_2 > y_4$. The RM is trained on the 6 implied pairwise comparisons with the Bradley-Terry loss. After training the RM might assign $r_\theta(x, y_1) = +1.8$, $r_\theta(x, y_2) = -0.5$, $r_\theta(x, y_3) = +0.6$, $r_\theta(x, y_4) = -1.2$. The PPO step then samples $\hat{y}$ from the policy, gets reward $r_\theta(x, \hat{y}) - \beta \log(\pi^{RL}/\pi^{SFT})$, and updates by policy gradient. Without the KL term, a poorly-trained RM might assign high reward to a degenerate "photosynthesis photosynthesis…" output (reward hacking); the KL term, with $\beta = 0.02$, penalizes such drift heavily and steers the policy back to fluent SFT-like text.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1952 ── Bradley & Terry: pairwise preference model
        │
2000s ── Inverse RL, learning from preferences (small scale)
        │
2017 ── Christiano, Leike et al.: Deep RL from Human Preferences
        │       — RLHF가 Atari/MuJoCo에서 작동 / RLHF works on Atari, MuJoCo
        │
2017 ── Schulman et al.: PPO (paper #24)
        │       — 안정적 policy gradient 알고리즘 / Stable policy gradient
        │
2018 ── Ibarz et al.: Imitation + RLHF on Atari
        │
2019 ── Ziegler et al.: Fine-tuning LMs from human preferences (stylistic)
        │       — RLHF의 LM 적용 첫 사례 / First application to LMs
        │
2020 ── Brown et al.: GPT-3 (paper #34)
        │       — 175B, in-context learning 발견 / 175B, in-context learning
        │
2020 ── Stiennon et al.: Learning to summarize with human feedback
        │       ★★ 직접 선조 / Direct predecessor
        │       — RLHF가 supervised보다 우수함을 입증 / RLHF beats supervised
        │
2021 ── Wei et al.: FLAN — instruction tuning (supervised only)
2021 ── Sanh et al.: T0 — multi-task instruction tuning
2021 ── Askell et al.: HHH framework, alignment laboratory
2021 ── Bender et al.: Stochastic Parrots (alignment critique)
        │
2022 ── ★★★ Ouyang et al.: InstructGPT (this paper) ★★★
        │       — SFT → RM → PPO + KL on broad API distribution
        │       — 1.3B InstructGPT > 175B GPT-3
        │
2022 ── Wu et al.: Recursively summarizing books (deeper RLHF)
2022 ── Bai et al.: Constitutional AI (paper #40, RLAIF)
        │       — 인간 라벨러를 AI로 대체 / Replace human labelers with AI
        │
Nov 2022 ── ChatGPT 출시 / ChatGPT public release
        │       — RLHF가 일반인에게 처음 노출 / RLHF reaches the public
        │
2023 ── Touvron et al.: Llama-2-Chat (RLHF open-sourced)
2023 ── OpenAI: GPT-4 (RLHF + safety techniques)
2023 ── Rafailov et al.: DPO (RM-free, closed-form preference learning)
        │       — RLHF 단순화 / Simplified RLHF
        │
2024 ── o1, R1 (reasoning RLHF, long CoT)
2024+── RLAIF, Multimodal RLHF, GRPO/RLOO/RPO (PPO-free variants)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#24 Schulman et al. (2017) — PPO** | Step 3의 RL 알고리즘. Clipped surrogate가 KL을 implicit하게 제한. The RL algorithm of Step 3; clipped surrogate implicitly bounds KL. | **Critical** — PPO 없이 InstructGPT 불가 / Without PPO, no InstructGPT |
| **#34 Brown et al. (2020) — GPT-3** | 출발점이 되는 pretrained model (1.3B/6B/175B). The pretrained backbone (1.3B/6B/175B). | **Critical** — InstructGPT는 GPT-3의 fine-tune / InstructGPT is GPT-3 fine-tuned |
| **#40 Bai et al. (2022) — Constitutional AI** | 인간 라벨러를 AI 모델로 대체 (RLAIF). Replace human labelers with an AI critic (RLAIF). | **High** — 직접 후속, 인간 데이터 부담 감소 / Direct successor, reduces human labeling burden |
| **Christiano et al. (2017) — Deep RL from Human Preferences** | RLHF 패러다임의 시작. The original RLHF paper. | **Critical** — 방법론적 조상 / Methodological ancestor |
| **Stiennon et al. (2020) — Summarization with human feedback** | InstructGPT가 직접 따르는 레시피 (summarization → broad instruction). The recipe InstructGPT directly follows (extended from summarization to broad instruction). | **Critical** — 방법론 거의 동일 / Near-identical methodology |
| **Ziegler et al. (2019) — Fine-tuning LMs from human preferences** | RLHF의 LM 첫 적용. First RLHF application to LMs. | **High** — 초기 발판 / Early scaffold |
| **Askell et al. (2021) — General language assistant** | HHH 정렬 기준의 출처. Source of the HHH framework. | **High** — 평가 기준 정의 / Defines evaluation criteria |
| **Wei et al. (2021) — FLAN** | Supervised instruction tuning 비교 baseline. Supervised instruction-tuning baseline. | **Medium** — InstructGPT가 78% 능가 / InstructGPT wins 78% |
| **Sanh et al. (2021) — T0** | Multi-task instruction tuning baseline. Multi-task supervised baseline. | **Medium** — InstructGPT가 79% 능가 / InstructGPT wins 79% |
| **Lin et al. (2021) — TruthfulQA** | 진실성 평가 벤치마크. Truthfulness benchmark used for evaluation. | **Medium** — 핵심 평가 도구 / Core evaluation tool |
| **Rafailov et al. (2023) — DPO** | RM 없이 선호 데이터로 직접 최적화 (closed-form). Closed-form direct preference optimization without an RM. | **High** — RLHF의 단순화 / RLHF simplification |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training Language Models to Follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems (NeurIPS) 35*. arXiv:2203.02155.

### Direct predecessors / 직접 선조
- Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *NeurIPS 2017*, 4299–4307.
- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., & Christiano, P. F. (2020). Learning to summarize with human feedback. *NeurIPS 2020*.
- Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., & Irving, G. (2019). Fine-tuning language models from human preferences. arXiv:1909.08593.

### Foundations / 기초
- Brown, T. B., et al. (2020). Language models are few-shot learners (GPT-3). *NeurIPS 2020*. arXiv:2005.14165.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms (PPO). arXiv:1707.06347.
- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345.

### Alignment / 정렬
- Askell, A., Bai, Y., Chen, A., et al. (2021). A general language assistant as a laboratory for alignment. arXiv:2112.00861.
- Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073. *(paper #40)*
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots. *FAccT 2021*, 610–623.
- Bommasani, R., et al. (2021). On the opportunities and risks of foundation models. arXiv:2108.07258.
- Gabriel, I. (2020). Artificial intelligence, values, and alignment. *Minds and Machines*, 30(3), 411–437.
- Kenton, Z., et al. (2021). Alignment of language agents. arXiv:2103.14659.
- Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., & Legg, S. (2018). Scalable agent alignment via reward modeling. arXiv:1811.07871.

### Instruction tuning baselines / 비교 기준
- Sanh, V., et al. (2021). Multitask prompted training enables zero-shot task generalization (T0). *ICLR 2022*. arXiv:2110.08207.
- Wei, J., et al. (2021). Finetuned language models are zero-shot learners (FLAN). *ICLR 2022*. arXiv:2109.01652.

### Evaluation datasets / 평가 데이터셋
- Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. *EMNLP findings 2020*.
- Lin, S., Hilton, J., & Evans, O. (2021). TruthfulQA: Measuring how models mimic human falsehoods. arXiv:2109.07958.
- Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-pairs: A challenge dataset for measuring social biases in masked language models. *EMNLP 2020*.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. *ACL 2018*.
- Rudinger, R., Naradowsky, J., Leonard, B., & Van Durme, B. (2018). Gender bias in coreference resolution (Winogender). *NAACL 2018*.
- Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., & Choi, Y. (2019). HellaSwag: Can a machine really finish your sentence? *ACL 2019*.

### Recent successors / 최근 후속 연구
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model (DPO). *NeurIPS 2023*. arXiv:2305.18290.
- Touvron, H., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288.
- OpenAI (2023). GPT-4 technical report. arXiv:2303.08774.
