---
title: "Pre-Reading Briefing: Constitutional AI: Harmlessness from AI Feedback"
paper_id: "40_bai_2022"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Constitutional AI: Harmlessness from AI Feedback — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Bai, Y., Kadavath, S., Kundu, S., et al. (Anthropic). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073, 2022.
**Author(s)**: Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, et al. (Anthropic)
**Year**: 2022

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 인간이 작성한 **헌법(Constitution)** — 즉 자연어로 표현된 짧은 원칙(principle) 목록 — 만으로 무해한(harmless) AI 어시스턴트를 학습시키는 **Constitutional AI (CAI)** 프레임워크를 제안합니다. CAI는 두 단계로 구성됩니다. (1) **Supervised Learning (SL-CAI)**: helpful-only RLHF 모델이 자신의 응답을 헌법 원칙에 비추어 비판(critique)하고 수정(revise)하며, 그 수정본으로 모델을 finetuning합니다. (2) **Reinforcement Learning from AI Feedback (RLAIF)**: AI가 두 응답 중 더 무해한 쪽을 헌법에 따라 선택하여 preference label을 생성하고, 이를 토대로 preference model(PM)을 학습한 뒤 RL을 수행합니다. 결과적으로 무해성에 대해서는 인간 라벨이 전혀 필요 없으며(도움성에 대해서만 인간 피드백 사용), 동시에 RLHF 대비 회피적이지 않은(non-evasive) 어시스턴트를 얻고, helpfulness–harmlessness Pareto frontier에서 RLHF를 능가합니다. Chain-of-thought (CoT) reasoning은 PM 라벨 품질과 헬프풀–해름리스 트레이드오프를 추가로 개선합니다.

**English**
This paper introduces **Constitutional AI (CAI)** — a framework for training a harmless AI assistant using only a human-written **constitution**, i.e. a short list of natural-language principles, with no human labels for harmfulness. CAI has two stages: (1) **Supervised Learning (SL-CAI)** — a helpful-only RLHF model critiques its own response against a sampled constitutional principle and revises it; the revised responses are used to finetune the model; (2) **Reinforcement Learning from AI Feedback (RLAIF)** — an AI feedback model picks the more harmless of two candidate responses (guided by the constitution) to create AI-generated preference labels, which are distilled into a preference model (PM); the PM is then used as the reward signal for RL. Helpfulness still uses human feedback, but harmlessness is fully scaled by AI. The resulting RL-CAI model is **non-evasive yet harmless** and lies on a strictly better helpfulness–harmlessness Pareto frontier than RLHF. Chain-of-thought reasoning further improves PM label quality and the tradeoff.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2022년은 LLM 정렬(alignment)의 분수령이었습니다. InstructGPT (Ouyang et al., 2022, paper #38)와 ChatGPT가 RLHF를 LLM 정렬의 사실상 표준으로 만들었지만, RLHF는 (1) **수만 건의 인간 피드백 라벨**을 필요로 하고, (2) 무해성을 강화하면 **회피적(evasive) "I cannot help with that"** 응답이 늘어 도움성과 충돌하며, (3) 인간 라벨러가 어떤 행동을 선호하는지 추론하기 어렵다는 한계가 있었습니다. Anthropic의 직전 논문(Bai et al., 2022 — "Helpful and Harmless Assistant")은 이 트레이드오프를 명시적으로 보고했습니다. CAI는 이 세 한계 모두를 표적으로 합니다 — 인간 라벨을 자연어 원칙으로 대체하고, 회피성을 줄이며, 학습 목표를 투명하게 노출합니다. 또한 Wei et al. (2022)의 Chain-of-Thought 프롬프팅이 reasoning 과정을 노출시키는 도구로 결합됩니다.

**English**
2022 was a watershed for LLM alignment. InstructGPT (Ouyang et al. 2022, paper #38) and ChatGPT made RLHF the de facto standard, but RLHF had three persistent issues: (1) it needs **tens of thousands of human preference labels**; (2) pushing harmlessness produced **evasive "I cannot help with that" responses** that clashed with helpfulness; (3) the implicit values encoded in human labels were opaque. Anthropic's prior paper (Bai et al. 2022, HH-RLHF) explicitly documented the helpful/harmless tradeoff. CAI targets all three: it replaces human harm labels with a short constitution, reduces evasiveness, and makes the training objective transparent. It also rides on Chain-of-Thought prompting (Wei et al. 2022; Kojima et al. 2022) to expose the model's reasoning.

### 타임라인 / Timeline

```
2017 ─ Christiano et al.: Deep RL from Human Preferences (RLHF foundation)
2017 ─ Schulman et al.: PPO (paper #24)
2020 ─ Stiennon et al.: Learning to summarize from human feedback
2021 ─ Askell et al.: A General Language Assistant as a Laboratory for Alignment (HHH)
2022.01 ─ Wei et al.: Chain-of-Thought Prompting
2022.03 ─ Ouyang et al.: InstructGPT (paper #38) ── RLHF for instruction following
2022.04 ─ Bai et al.: Training a Helpful and Harmless Assistant with RLHF (HH-RLHF)
2022.08 ─ Ganguli et al.: Red Teaming Language Models
2022.09 ─ Glaese et al.: Sparrow (rule-based dialogue alignment)
2022.11 ─ ChatGPT released
2022.12 ─ ★ Bai et al.: Constitutional AI (this paper) ★ ── path to Claude
2023.03 ─ Anthropic Claude 1.0 (commercial deployment of CAI ideas)
2024+ ── DPO, RLAIF widely adopted; "Collective Constitutional AI" (Anthropic)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Transformer 및 사전학습 LLM**: 디코더 전용 Transformer (paper #25, Vaswani et al. 2017)와 GPT 계열 사전학습 — 토큰 수준 cross-entropy.
- **RLHF 파이프라인** (paper #38 InstructGPT, Christiano et al. 2017): SFT → preference modeling → PPO. Bradley–Terry 손실로 PM 학습, KL penalty가 포함된 PPO RL.
- **Preference model**: 두 응답에 대한 스칼라 점수 $r_\phi(x, y)$를 출력하는 모델, BT 손실 $\mathcal{L}_{PM} = -\log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$.
- **PPO with KL penalty**: $r_{\text{total}} = r_\phi(x, y) - \beta\,\mathrm{KL}[\pi\|\pi_{\text{ref}}]$.
- **Chain-of-Thought (CoT) prompting**: "Let's think step by step" (Kojima et al. 2022)로 reasoning을 노출시키면 zero-shot/few-shot 정확도가 상승.
- **Few-shot prompting** 및 in-context learning.
- **Red teaming**: 모델로부터 유해 응답을 유도하는 프롬프트 작성 작업 (Ganguli et al. 2022).
- **Elo rating**: 두 모델의 응답 비교에서 승률을 점수로 환산하는 방식 (체스 평정과 동일 원리).

**English**
- **Transformers and pretrained LLMs**: decoder-only Transformer (paper #25), GPT-style language modeling.
- **RLHF pipeline** (paper #38, Christiano et al. 2017): SFT → preference model (BT loss) → PPO with KL.
- **Preference modeling**: $\mathcal{L}_{PM} = -\log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$.
- **PPO with KL**: total reward $r_\phi - \beta \cdot \mathrm{KL}[\pi\|\pi_{\text{ref}}]$.
- **Chain-of-thought prompting** (Wei 2022; Kojima 2022 "Let's think step by step").
- **Few-shot in-context prompting**.
- **Red teaming** (Ganguli et al. 2022) — adversarial prompts eliciting harm.
- **Elo rating** — head-to-head win rate translated to scalar scores.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Constitution / 헌법** | 자연어로 표현된 16개의 행동 원칙 목록. 매 critique-revision 단계에서 무작위로 하나 샘플링됨. / A list of ~16 natural-language behavioral principles; one is sampled at each critique step. |
| **CAI (Constitutional AI)** | 헌법으로 인간 라벨을 대체하는 정렬 프레임워크 전체. / The whole alignment framework that replaces human harm labels with a constitution. |
| **SL-CAI** | Supervised Learning stage of CAI: critique→revise→finetune. / 헌법 기반 critique–revision 파이프라인으로 finetuning한 단계. |
| **RLAIF (RL from AI Feedback)** | AI feedback model이 preference label을 생성, RL로 학습. RLHF의 H를 AI로 대체. / RL using AI-generated preference labels — H of RLHF replaced with AI. |
| **Critique / 비판** | 모델이 자신의 응답에서 헌법 원칙 위반 부분을 자연어로 식별하는 단계. / Model identifies, in natural language, ways its own response violates a principle. |
| **Revision / 수정** | 비판을 반영하여 모델이 응답을 재작성. 반복 적용 가능 (1~4회). / Model rewrites the response to address the critique; repeated 1–4 times. |
| **Helpful RLHF model / 헬프풀 RLHF 모델** | 도움성에 대해서만 RLHF로 학습한 초기 모델. CAI의 출발점. / Initial RLHF model trained on helpfulness only; the starting point for CAI. |
| **Red team prompt / 레드팀 프롬프트** | 유해 응답을 유도하기 위해 의도적으로 작성된 적대적 프롬프트. / Adversarial prompt crafted to elicit harmful behavior. |
| **Evasiveness / 회피성** | "I can't answer that" 같은 무성의·회피적 응답. 무해하지만 비도움적. / Canned non-answers like "I can't help with that" — harmless but unhelpful. |
| **Elo score / Elo 점수** | 모델 간 head-to-head 승률을 스칼라로 변환한 평정. / Scalar rating from pairwise win rates between models. |
| **Pareto frontier / 파레토 경계** | helpfulness × harmlessness 평면 상에서 다른 점에 의해 dominate되지 않는 모델들의 집합. / Set of models on the helpfulness–harmlessness plane that are not dominated. |
| **Chain-of-Thought (CoT)** | "Let's think step by step"으로 명시적 reasoning을 작성하게 하는 prompting. / Prompting style that elicits explicit step-by-step reasoning. |
| **Soft / Hard / Clamped labels** | RLAIF에서 PM 학습용 라벨을 확률(soft), 0/1(hard), [0.4,0.6]로 잘라낸(clamped) 형태. / RLAIF label types: probability (soft), 0/1 (hard), or clamped to [0.4,0.6]. |
| **Goodharting** | 보상 모델을 과적합해 표면적 점수만 올리는 현상. CAI에서 boilerplate "you are valued and cared for" 같은 형태로 출현. / Reward-model overoptimization producing surface-level boilerplate. |

(14 terms / 14개 용어)

---

## 5. 수식 미리보기 / Equations Preview

### (a) Preference model loss (Bradley–Terry / 부로드리 테리)

$$\mathcal{L}_{\text{PM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

$y_w$ = 선호되는 응답 (winner), $y_l$ = 덜 선호되는 응답 (loser). RLHF에서는 $\mathcal{D}$가 인간 라벨, RLAIF에서는 AI 라벨. / $y_w$ = preferred response, $y_l$ = dispreferred. RLHF uses human labels; RLAIF uses AI-generated labels.

### (b) RLAIF soft label (multiple-choice probability)

$$P(\text{A is more harmless} \mid x, y_A, y_B) = \frac{p_\theta(\text{"A"} \mid \text{prompt})}{p_\theta(\text{"A"} \mid \text{prompt}) + p_\theta(\text{"B"} \mid \text{prompt})}$$

피드백 모델이 (A) vs (B) 객관식 답변에 부여한 log-prob을 정규화. 이 soft 확률이 PM 라벨로 사용됨. / Feedback model assigns log-probs to answer tokens "A" and "B"; normalized as a soft preference label.

### (c) PPO RL objective with KL penalty

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y) - \beta \cdot \mathrm{KL}\!\left[\pi_\theta(y|x) \,\Vert\, \pi_{\text{ref}}(y|x)\right]\right]$$

$\pi_{\text{ref}}$는 SL-CAI 모델 (RLHF에서는 SFT 모델). KL penalty가 mode collapse 방지. / $\pi_{\text{ref}}$ is the SL-CAI snapshot (in RLHF it is the SFT model); KL prevents mode collapse.

### (d) SL-CAI supervised loss (autoregressive)

$$\mathcal{L}_{\text{SL}}(\theta) = -\mathbb{E}_{(x, \tilde{y}) \sim \mathcal{D}_{\text{rev}}}\left[\sum_{t=1}^{|\tilde{y}|} \log \pi_\theta(\tilde{y}_t \mid x, \tilde{y}_{<t})\right]$$

$\tilde{y}$는 critique-revision으로 생성된 무해한 수정본. 표준 LM 손실이지만 데이터 분포가 헌법-수정된 응답이라는 점이 핵심. / $\tilde{y}$ are constitution-revised harmless responses; standard LM loss but on a constitution-shaped data distribution.

### (e) Critique–Revision pipeline (의사코드 / pseudocode)

```
y0 = HelpfulRLHF(x_red_team)
for k in 1..K:
    p_k = sample(constitution.principles)
    c_k = HelpfulRLHF(x_red_team, y_{k-1}, "Critique by " + p_k)
    y_k = HelpfulRLHF(x_red_team, y_{k-1}, c_k, "Revise by " + p_k)
return y_K   # used as the SL-CAI training target
```

(5 key equations / 5개 핵심 수식)

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
1. **§1 Introduction** (pp. 2–6): scaled supervision, non-evasive assistant, transparency라는 세 가지 동기를 정확히 파악. Figure 1의 두 단계 다이어그램과 Figure 2의 Pareto frontier를 머릿속에 새기세요.
2. **§2 Evaluating AI Supervision** (pp. 6–7): "LLM이 이미 H를 식별할 수 있는가?"라는 사전 정당성 검사. CoT가 PM 정확도를 어떻게 끌어올리는지 (Figure 4) 주목.
3. **§3 SL-CAI** (pp. 7–9): critique–revision 파이프라인. 핵심 예시(p. 7 wifi 해킹) 정독. Figure 5(revision 횟수 vs 점수), Figure 6(원칙 수의 영향), Figure 7(critique 유무의 영향)을 비교.
4. **§4 RL-CAI / RLAIF** (pp. 9–13): 객관식 prompt 형식, soft label, CoT prompting과 clamping (40–60%). Figure 8/9/10의 결과 분석.
5. **§5 Related Work** (p. 14): RLHF 계보(InstructGPT, LaMDA, Sparrow), self-critique, debate/amplification.
6. **§6 Discussion** (pp. 15–16): scalable oversight, robustness, dual-use 우려.
7. **Appendix A** (p. 18): 4단계 critique–revision 실제 trace — wifi/grocery store 예시. **반드시 정독**.
8. **Appendix C**: 실제 사용된 16개 원칙 텍스트.

읽으면서 던져볼 질문:
- 헌법 원칙이 왜 단 16개인가? 더 많으면 좋을까? (Figure 6 참조)
- critique 단계는 실제로 필요한가? (Figure 7, §3.5)
- soft vs hard vs clamped label 중 왜 clamped (40–60%)이 최선인가? (§4.3)
- 인간 라벨이 사라진 만큼, "원칙 작성자"의 가치관 편향이 어디서 들어오는가?

**English**
1. **§1 Introduction** (pp. 2–6): pin down the three motivations — scaled supervision, non-evasive assistant, transparency. Internalize the two-stage diagram (Figure 1) and the Pareto frontier (Figure 2).
2. **§2 Evaluating AI Supervision** (pp. 6–7): sanity check — can LLMs already identify HHH? Note how CoT pushes PM accuracy (Figure 4).
3. **§3 SL-CAI** (pp. 7–9): critique–revision pipeline. Read the wifi-hacking example (p. 7) carefully. Compare Figures 5/6/7 (revision count, # principles, critique vs no-critique).
4. **§4 RL-CAI / RLAIF** (pp. 9–13): multiple-choice prompt format, soft labels, CoT prompting with 40–60% clamping. Analyze Figures 8/9/10.
5. **§5 Related Work** (p. 14): RLHF lineage, self-critique, debate/amplification.
6. **§6 Discussion** (pp. 15–16): scalable oversight, robustness, dual-use risk.
7. **Appendix A** (p. 18): 4-step critique–revision trace (wifi / grocery store). **Read carefully.**
8. **Appendix C**: full 16-principle text.

Questions to hold while reading:
- Why exactly 16 principles? Would more help? (Figure 6.)
- Are critiques necessary or can we skip to revision? (§3.5, Figure 7.)
- Why clamped (40–60%) labels beat soft and hard? (§4.3.)
- With humans removed, where does the value-bias of *principle authors* enter the system?

---

## 7. 현대적 의의 / Modern Significance

**한국어**
- **Claude로 가는 길**: CAI는 Anthropic Claude 시리즈의 핵심 학습 방법론. 후속 "Collective Constitutional AI" (2023), Claude 3 Constitution까지 직접 이어집니다.
- **RLAIF의 정착**: "AI가 AI를 감독한다"는 패러다임이 산업 표준이 됨. Google DeepMind, Meta, OpenAI 모두 변형을 도입(예: Direct Nash Optimization, RLAIF-V, Self-Rewarding LM).
- **Scalable oversight의 첫 실증**: AI 능력이 인간 평가자를 넘는 영역에서 "감독 자체"를 자동화한다는 alignment 비전의 첫 정량적 사례.
- **자연어 정책 명세**: 정책을 prose로 적는다 → 더 적은 데이터, 더 투명한 거버넌스, stakeholder가 직접 검토 가능. "Policy as code"에서 "policy as constitution"으로의 전환.
- **DPO/RLAIF/Self-Refine의 조상**: critique-revise 루프는 후속 self-improvement (Self-Refine, Reflexion, STaR)의 직접 선조. soft preference label은 DPO/IPO의 출발점.
- **Red teaming의 자동화**: 인간 레드티머에 대한 의존도를 줄이는 path. Anthropic의 "Many-shot jailbreaking" 등 자동 평가의 토대.
- **한계의 인식**: dual-use 위험 — 같은 방법으로 악의적 헌법을 사용하면 정확히 그만큼 효율적으로 해로운 시스템을 만들 수 있음 (§6.2).

**English**
- **Path to Claude**: CAI is the core training paradigm for Anthropic's Claude family — directly continued in "Collective Constitutional AI" (2023) and Claude 3's published constitution.
- **RLAIF mainstream**: "AI supervising AI" became an industry standard — adopted in some form by Google, Meta, OpenAI (RLAIF-V, Self-Rewarding LM, Direct Nash Optimization).
- **First empirical scalable oversight**: a concrete demonstration that automated supervision can match or exceed human supervision on a non-trivial alignment axis (harmlessness).
- **Policy as natural language**: writing policy in prose → fewer labels, more transparent governance, stakeholder review of the actual rules. Shift from "policy as code" to "policy as constitution."
- **Ancestor of self-improvement methods**: the critique–revise loop is a direct ancestor of Self-Refine, Reflexion, and STaR. Soft preference labels seeded DPO/IPO.
- **Automated red teaming**: lowers reliance on human red teamers, foundation of automated evals.
- **Recognized limits**: dual-use risk — the same recipe with a malicious constitution makes harmful systems just as efficiently (§6.2).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
