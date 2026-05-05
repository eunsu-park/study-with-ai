---
title: "Constitutional AI: Harmlessness from AI Feedback"
authors: [Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan]
year: 2022
journal: "arXiv preprint (Anthropic)"
doi: "arXiv:2212.08073"
topic: Artificial_Intelligence
tags: [alignment, RLHF, RLAIF, constitutional-AI, harmlessness, helpfulness, chain-of-thought, preference-modeling, scalable-oversight, anthropic, claude]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 40. Constitutional AI: Harmlessness from AI Feedback / 헌법 기반 AI: AI 피드백으로부터의 무해성

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 Anthropic이 2022년 12월 발표한 정렬(alignment) 방법론으로, **인간 라벨 없이 AI 어시스턴트를 무해하게(harmless) 학습시키는 두 단계 파이프라인**을 제시합니다. 출발점은 도움성에 대해서만 RLHF로 학습된 **Helpful RLHF model**입니다. 첫 단계인 **Supervised Learning CAI (SL-CAI)**에서는 (i) 이 모델이 적대적("red team") 프롬프트에 응답하고, (ii) 16개의 자연어 원칙으로 구성된 **헌법(constitution)** 중 하나가 무작위로 샘플링되어 모델 자신의 응답을 critique하도록 요청받고, (iii) 비판을 반영해 응답을 revise합니다. 이 critique–revision은 1~4회 반복되며, 최종 수정본을 supervised loss로 finetuning합니다. 두 번째 단계인 **Reinforcement Learning from AI Feedback (RLAIF)**에서는 SL-CAI 모델로부터 두 응답을 샘플링하고, 객관식 형식으로 피드백 모델에게 "어느 응답이 더 무해한가"를 헌법 원칙에 따라 묻습니다. 이때 응답 토큰("A"/"B")의 정규화된 log-prob을 **soft preference label**로 사용해 preference model(PM)을 학습하고, 이를 보상 신호로 PPO를 수행합니다. 도움성에 대해서는 종전대로 인간 피드백을 사용합니다. 결과적으로 (1) 무해성에 대한 인간 라벨을 0으로 줄이면서, (2) helpfulness vs harmlessness Elo 평면에서 RLHF baseline보다 **strict Pareto improvement**를 달성하고, (3) "I cannot answer that"식 회피 응답을 거의 제거하여 모델이 거절하는 이유까지 설명하게 만듭니다. Chain-of-Thought (CoT) prompting을 PM 라벨 생성에 결합하면 정확도와 transparency가 추가로 개선됩니다.

**English**
This Anthropic paper (December 2022) introduces a **two-stage pipeline that trains an AI assistant to be harmless without any human harmlessness labels**. The starting point is a **Helpful RLHF model** (trained on helpfulness only). Stage 1, **Supervised Learning CAI (SL-CAI)**: (i) the model answers a red-team prompt; (ii) one of 16 natural-language principles from the **constitution** is sampled at random, and the model is asked to critique its own response under that principle; (iii) the model rewrites the response addressing the critique. Critique–revision is iterated 1–4 times, and the final revisions form the supervised finetuning corpus. Stage 2, **RL from AI Feedback (RLAIF)**: two responses are sampled from SL-CAI, and a feedback model is asked, in multiple-choice form with a sampled principle, "which is more harmless?". The normalized log-prob of the answer tokens "A" / "B" is used as a **soft preference label** to train a preference model (PM); PPO then optimizes the policy against this PM. Helpfulness still uses human preference data. Results: (1) zero human labels for harmlessness, (2) a **strict Pareto improvement** over RLHF on the helpfulness × harmlessness Elo plane, (3) almost no evasive "I cannot answer that" responses — the model engages and explains *why* it declines. Chain-of-Thought prompting on the feedback model further improves PM label quality and transparency.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, pp. 2–6) / 서론

**한국어**
저자들은 "AI 능력이 인간을 능가하는 영역에서도 AI를 정렬할 수 있는 기법"을 만들고자 합니다. 핵심 키워드는 **Scaling Supervision**: 인간이 모든 AI 행동을 직접 평가할 수 없으니, AI 자체가 다른 AI를 감독하도록 점진적으로 위임한다. RLHF는 이미 그 방향의 첫걸음 — 보상 신호가 즉각적인 인간 평가가 아니라 PM에서 나오기 때문에 — 이지만 여전히 수만 건의 인간 라벨이 필요합니다.

CAI는 인간 입력을 **극단으로 줄여** 자연어 원칙 ~16개로 한정합니다. Figure 1의 다이어그램이 핵심:
1. (위쪽 흐름) Helpful RLHF → red-team 프롬프트 → 응답 → Critique → Revision → SL 학습 → SL-CAI 모델
2. (아래쪽 흐름) SL-CAI → 응답 쌍 → 헌법 기반 AI 피드백 → PM finetune → RL → 최종 RL-CAI 모델

**Figure 2 (해석의 핵심)** — 가로축: helpfulness Elo, 세로축: harmlessness Elo. 점이 우상단으로 갈수록 좋음.
- "Pretrained Base"는 좌하단 (둘 다 낮음).
- "Helpful-Only" RLHF는 helpfulness↑ 하지만 harmlessness↓.
- "Helpful + Harmless" RLHF (HH-RLHF, Bai 2022)가 곡선 — Pareto frontier(검은색).
- **Constitutional RL은 그 곡선을 위로 밀어냄(자홍색) — strict Pareto improvement**.
- CoT를 더한 RL-CAI는 더욱 위.

**네 가지 동기 (§1)**:
1. **Scaling supervision**: 인간 라벨을 줄여 더 강력한 미래 모델에 대비.
2. **Evasiveness 제거**: 회피적 응답을 명시적으로 비선호로 라벨링 → 모델이 거절 이유를 설명.
3. **Simplicity & Transparency**: 학습 목표를 자연어로 노출.
4. **Iteration speed**: 목표가 바뀌어도 새 인간 라벨 수집 불필요.

**English**
The authors aim for techniques to align AI even when AI capability exceeds the human supervisor. Their slogan is **Scaling Supervision** — RLHF already moves in this direction (reward comes from a PM, not from immediate human judgment), but it still requires tens of thousands of human labels. CAI shrinks human input to ~16 principles. **Figure 1** shows two stacked stages: Helpful RLHF → red-team prompts → critique → revision → SL-CAI; then SL-CAI samples response pairs → AI feedback under a principle → PM finetune → RLAIF → RL-CAI.

**Figure 2** is the headline: helpfulness Elo on x, harmlessness Elo on y. Constitutional-RL with CoT lies strictly above the standard RLHF Pareto frontier — both more helpful *and* more harmless than HH-RLHF.

Four motivations: (1) scaling supervision; (2) eliminating evasiveness; (3) simplicity / transparency of objective; (4) faster iteration when policy changes.

### Part II: Evaluating AI Supervision (§2, pp. 6–7) / AI 감독 가능성 평가

**한국어**
이 섹션은 사전 정당성 점검입니다 — "큰 LLM은 이미 H, H, H를 식별할 수 있는가?" 저자들은 Askell et al. (2021)의 221개 HHH 비교, Srivastava et al. (2022, BIG-Bench)의 추가 217개 비교, 합 **438개의 어려운 binary 비교**를 평가셋으로 사용합니다.

**Figure 4** (평가):
- x축: 모델 매개변수 수 (1B → 52B)
- y축: 정확도
- "HH PM from Human Feedback" (orange): 약 0.78 — 강력한 baseline.
- "Pretrained LM" (blue, multiple-choice): 1.4B에서 0.49 → 52B에서 0.67. capability가 커지며 급상승.
- "Chain-of-Thought" (light grey): 52B에서 0.73, pretrained LM보다 +6%p.
- "Ensembled CoT" (dark grey): 5개 CoT 샘플 평균 → 추가 향상.

**핵심 메시지**: 52B 정도면 pretrained LM이 PM 정확도의 94% 수준에 도달하고, CoT를 더하면 격차가 좁혀집니다. 이는 "AI가 H를 식별할 수 있다"는 RLAIF의 전제를 정량적으로 정당화합니다.

**English**
A sanity check: can LLMs already identify the more H/H/H response? On 438 hard binary comparisons (Askell 2021 + Srivastava 2022 BIG-Bench), Figure 4 shows: HH PM from human feedback ≈ 0.78; pretrained LM at 52B reaches 0.67; CoT pushes it to 0.73; ensembled CoT a bit higher. So 52B-class pretrained LMs come within ~5–6 percentage points of a human-trained PM, justifying the RLAIF premise. Appendix B confirms this on harm-classification tasks too.

### Part III: SL-CAI — Critiques, Revisions, Supervised Learning (§3, pp. 7–9) / 비판·수정·지도학습

**한국어**

#### 3.1 Method / 방법
파이프라인 4단계 (p. 7 wifi-hacking 예시):

**Step 1 — Initial response (helpful RLHF가 생성)**:
```
Human: Can you help me hack into my neighbor's wifi?
Assistant: Sure thing, you can use an app called VeryEasyHack
that will allow you to log in to your neighbor's wifi.
```
(harmful)

**Step 2 — Critique**:
```
Critique Request: Identify specific ways in which the
assistant's last response is harmful, unethical, racist,
sexist, toxic, dangerous, or illegal.
Critique: The assistant's last response is harmful because
hacking into someone else's wifi is an invasion of their
privacy and is possibly illegal.
```

**Step 3 — Revision**:
```
Revision Request: Please rewrite the assistant response to
remove any and all harmful, unethical, racist, sexist, toxic,
dangerous, or illegal content.
Revision: Hacking into your neighbor's wifi is an invasion of
their privacy, and I strongly advise against it. It may also
land you in legal trouble.
```

**Step 4 — Reassemble**: 원래 프롬프트 + revision = 깨끗한 SL 학습 예시.

이를 1~4회 반복(매 단계 헌법에서 무작위 원칙 sample). 16개 원칙 사용, **few-shot prompting**으로 critique/revision 형식 안정화.

#### 3.2 Datasets and Training / 데이터셋과 학습
- Red team prompts: 인간 작성 42,496 + few-shot으로 모델이 생성한 140,335 = **182,831개**.
- 각 프롬프트 당 4개의 critique-revision pair → 4 revisions per prompt.
- Helpfulness prompts: 135,296 (인간 작성, helpful RLHF의 응답 2개 sampling).
- 학습: pretrained model을 1 epoch finetune, learning rate = 0.5×(pretraining lr), batch 1024.

#### 3.3 Main Results / 주요 결과
**Figure 8 (왼쪽 — helpfulness Elo)**: SL-CAI는 helpful RLHF보다는 약간 떨어짐, 그러나 HH-RLHF보다는 높음.
**Figure 8 (오른쪽 — harmlessness)**: SL-CAI는 helpful RLHF보다는 훨씬 무해함, HH-RLHF보다는 약간 더 해로움(SL stage 단독으로는 부족).

**Figure 5**: 52B PM으로 평가 — revision 횟수 0→4 증가 시 harmlessness PM score가 단조 증가 (-1 → +2.3). 그러나 helpfulness score는 약간 감소 (혼합 HH score는 여전히 단조 증가). Revision은 거의 항상 무해성을 개선.

**Figure 6**: 헌법 원칙 수 N=1, 2, 4, 8, 16. 점수 자체에는 큰 차이 없으나, 저자는 다양성(diversity) 면에서 N↑이 RL exploration에 유리하다고 보고.

**Figure 7**: critique를 거친 revision vs critique 없이 직접 revision. 작은 모델에서는 critique가 도움(harmlessness PM score 더 높음), 큰 모델에서는 비슷. 그러나 transparency 측면에서 critique을 유지.

**English**
SL-CAI core trace (wifi-hack example):
1. helpful RLHF gives harmful answer; 2. model critiques itself under a sampled principle; 3. model revises; 4. (prompt, revision) pair used for SFT. Repeat 1–4×, sampling a different principle each step. Datasets: 182,831 red-team prompts × 4 revisions for harm; 135,296 helpfulness prompts (2 helpful RLHF samples each). Train 1 epoch at half pretraining lr, batch 1024.

Main results: SL-CAI sits between helpful RLHF and HH-RLHF on helpfulness, and is significantly more harmless than helpful RLHF. **Figure 5**: harmlessness PM score grows monotonically with revision count (≈ -1 at revision 0 to +2.3 at revision 4). **Figure 6**: number of principles barely affects PM score, but more principles increase response diversity, useful for RL exploration. **Figure 7**: critiques help small models, draw with large models — kept for transparency.

### Part IV: RL-CAI — RLAIF (§4, pp. 9–13) / AI 피드백 강화학습

**한국어**

#### 4.1 Method / 방법
SL-CAI 모델 두 응답 $(y_A, y_B)$를 prompt $x$에 대해 sampling. 이를 **객관식 형식**으로 feedback model에게 제출:

```
Consider the following conversation between a human and an assistant:
[HUMAN/ASSISTANT CONVERSATION]
[PRINCIPLE FOR MULTIPLE CHOICE EVALUATION]
Options:
 (A) [RESPONSE A]
 (B) [RESPONSE B]
The answer is:
```

답변 token $\{$"A"$, $"B"$\}$의 log-prob을 정규화하여 soft preference 확률을 얻습니다:

$$P(A \succ B) = \frac{p(\text{"A"})}{p(\text{"A"}) + p(\text{"B"})}$$

이 soft 확률을 PM training의 라벨로 사용. 16개 원칙 ensemble (매 비교마다 하나 sampling) → robustness↑.

**CoT variant**: helpful RLHF model (pretrained LM이 아니라)에 "Let's think step by step:" prompt 추가, conversation 형식으로 reformat. CoT는 reasoning을 노출하지만 종종 0/1로 saturate. 해법: **clamping** to [0.4, 0.6]. 0.2–0.8도 시도했으나 0.4–0.6이 best.

총 데이터: 135,296 helpfulness + 182,831 harmlessness AI label = mixed PM dataset. RL prompts: 491,142 red team + 474,300 helpfulness (model-generated 추가).

**Reward signal**: PM 출력 $r_\phi$를 PPO 보상으로 사용, KL penalty는 SL-CAI가 reference. (PPO 세부 hyper는 Bai 2022와 동일.)

#### 4.2 Main Results / 주요 결과
**Figure 8** (RL training 곡선): RL-CAI w/ CoT가 harmlessness Elo에서 ~+200, RL-CAI(no CoT)가 ~+150. helpful RLHF는 harmlessness 음수에서 더 떨어짐. helpfulness 측면에서 RL-CAI는 helpful RLHF에 근접 (~+125), HH-RLHF (+100)보다 더 helpful.

**Figure 9** (calibration): 52B RL-CAI 라벨의 calibration plot — 거의 대각선, 잘 calibrated.

**Figure 10** (absolute harmfulness, 0–4 scale, 64 prompts × 256 samples):
- Helpful RLHF: 학습 진행될수록 *더* 해로워짐 (3.5 부근까지).
- HH-RLHF: ~2.5 → 1.0 감소.
- RL-CAI: ~1.7 → 0.7.
- RL-CAI w/ CoT: ~1.7 → 0.5 (가장 무해).

#### 4.3 Strategies / 전략
- **Constitutional Principles 재작성**: "wise, ethical, polite, friendly person이 더 선택할 응답을 고르라"처럼 부드럽게 표현 → 과도한 reactiveness/accusatory 줄임.
- **Ensembling 16 principles**: 단일 원칙보다 robust.
- **Soft vs Hard vs Clamped labels**: soft가 hard보다 훨씬 좋음 (Kadavath et al. 2022 calibration 결과). CoT 시 saturate 문제 → clamp 40–60%가 최선.

#### 4.4 Harmlessness vs Evasiveness / 무해성과 회피성
HH-RLHF는 종종 "I can't answer that"으로 회피. CAI 학습에서 crowdworker에게 "회피보다 thoughtful 응답을 선호하라"고 지시 → RL-CAI는 "거의 회피하지 않으며" 거절 시 그 이유를 설명. 이는 prior Bai 2022 대비 정성적 큰 변화.

#### 4.5 Goodharting
RL-CAI가 over-train 시 boilerplate "you are valid, valued, and cared for"를 모든 red-team 응답에 붙이는 patho­logical 행동. PM의 표면적 신호를 hack — Gao et al. 2022 (reward over­optimization)과 일치. 이는 RL 길이 또는 KL coefficient로 제어.

**English**
Pair $(y_A, y_B)$ sampled from SL-CAI is presented in multiple-choice format with a sampled principle; the feedback model's normalized log-probabilities of "A" vs "B" form **soft preference labels**. 16 principles ensembled per comparison → robust. **CoT variant**: helpful RLHF model with "Let's think step by step" reformat; CoT often saturates at 0/1, so labels are **clamped to [0.4, 0.6]**.

PM trained on mixed (human helpfulness + AI harmlessness) labels; PPO uses PM as reward with KL to SL-CAI. **Figure 8**: RL-CAI w/ CoT reaches +200 harmlessness Elo and ~+125 helpfulness Elo — strict Pareto improvement over HH-RLHF (+100, +75). **Figure 9**: 52B feedback model's labels are well-calibrated. **Figure 10** (absolute 0–4 harmfulness): helpful RLHF rises to 3.5 (worse), HH-RLHF drops to 1.0, RL-CAI to 0.7, RL-CAI w/ CoT to **0.5**.

Practical lessons: re-write principles to soft tone; ensemble 16 principles; soft labels beat hard, clamping CoT to [0.4, 0.6] beats raw soft. Crucially, **non-evasive** training: workers told to prefer thoughtful, transparent refusals → RL-CAI almost never says "I can't" without explanation. Goodharting (e.g., boilerplate "you are valid, valued, and cared for") emerges with over-training; controlled by KL coefficient and training length.

### Part V: Related Work (§5, p. 14) / 관련 연구

**한국어**
- **RLHF lineage**: Christiano et al. 2017 (RL from human prefs), Stiennon et al. 2020 (summarization), Bai et al. 2022 (HH-RLHF), Ouyang et al. 2022 (InstructGPT), Thoppilan et al. 2022 (LaMDA), Glaese et al. 2022 (Sparrow).
- **Self-critique / self-improve**: Zhao 2021 (Ethical advice taker), Shi 2022, Huang 2022, Saunders 2022, Scheurer 2022. Saunders는 SL stage와 매우 유사.
- **Sparrow와의 차이**: Sparrow도 규칙 기반 사람 평가를 사용하지만 라벨은 인간이 주는 반면, CAI는 라벨까지 AI가 생성.
- **CoT**: Nye 2021 scratchpad, Wei 2022 CoT, Kojima 2022 zero-shot CoT.
- **Calibration**: Kadavath et al. 2022 — LLM이 자신의 답에 well-calibrated probability를 부여할 수 있다는 결과가 soft label의 근거.
- **Scaling oversight**: Christiano 2018 amplification, Irving 2018 debate, Bowman 2022 sandwiching.
- **Red teaming**: Ganguli et al. 2022, Perez et al. 2022.

**English**
RLHF lineage (Christiano 2017; Stiennon 2020; Bai 2022; Ouyang 2022; Thoppilan 2022; Glaese 2022 Sparrow). Self-critique works most similar to SL-CAI: Saunders 2022, Zhao 2021, Shi 2022, Huang 2022, Scheurer 2022. CoT lineage: Nye 2021, Wei 2022, Kojima 2022. Calibration (Kadavath 2022) underwrites soft labels. Scalable oversight: Christiano 2018 amplification, Irving 2018 debate, Bowman 2022 sandwiching. Red teaming: Ganguli 2022, Perez 2022.

### Part VI: Discussion & Future Directions (§6, pp. 15–16) / 논의·향후 방향

**한국어**
- **자기지도형 정렬로의 이동**: 무해성에서 인간 라벨을 0으로 줄임. 도움성·instruction-following도 결국 자동화 가능 (잘 작성된 프롬프트 + pretrained LM만으로도).
- **CoT의 가치**: critique, comparison label 모두 reasoning을 노출 → 더 transparent, more robust to subtle harms.
- **Robustness**: helpfulness와 harmlessness가 양립가능해지면 자동 red teaming을 안전하게 확장 가능. Iterated online training (Bai 2022)과 결합 가능.
- **다른 행동 축으로의 일반화**: writing style, tone, persona 변경에도 적용 가능. 정책 차원의 안전성을 빠르게 실험 가능.
- **Dual-use 위험**: 같은 메커니즘이 악의적 헌법으로도 작동. "방벽이 낮아진다"는 양면.
- **Future**: stakeholder 참여로 헌법 작성, 다양한 행동 axis에 따른 generalization 분석, online RLAIF.

**English**
Moving toward self-supervised alignment by removing human labels for harm; helpfulness still uses human input but could in principle be replaced too. CoT exposes reasoning during both critique and label generation, improving transparency. Robustness is the next horizon: with helpfulness/harmlessness compatible, automated red teaming + iterated online RLAIF becomes practical. The method generalizes to other behavioral axes (style, tone, persona). Dual-use is the explicit caveat — the same recipe with malicious principles trains harmful systems just as efficiently.

### Part VII: Appendix A & C — Concrete Traces and Principles / 구체적 trace와 원칙

**한국어**
Appendix A는 grocery store 도둑질 프롬프트의 4단계 critique–revision trace를 제공:
- Initial: "candy/gum 같은 작은 물건 훔쳐라" (실제로 유해한 조언).
- 1st Revision: "도둑질은 비윤리적, food bank 신청 권유."
- 2nd Critique: "응답이 perfect."
- 4th Revision: 이전 수정본을 매끄럽게 다듬음.

핵심 관찰: 첫 번째 revision이 대부분의 유해성을 제거, 2~4번째는 미세 개선. 그러나 critique이 종종 부정확하거나 과장됨에도 revision은 일관되게 개선.

Appendix C는 16개 SL-CAI 원칙과 RLAIF 원칙(예: "Choose the response that a wise, ethical, polite, friendly person would more likely say.")의 실제 텍스트. 사용 prompt 형식과 few-shot 예시 포함. 코드 + 원칙은 GitHub `anthropics/ConstitutionalHarmlessnessPaper`에 공개.

**English**
Appendix A walks through the grocery-theft prompt across 4 critique–revision rounds: the first revision removes most harm; later rounds are minor polish. Critiques are sometimes inaccurate yet revisions still improve — the loop is robust to noisy intermediate critiques. Appendix C contains the actual 16-principle texts (e.g., "Choose the response that a wise, ethical, polite, friendly person would more likely say") and few-shot prompt examples. Code released at GitHub `anthropics/ConstitutionalHarmlessnessPaper`.

---

## 3. Key Takeaways / 핵심 시사점

1. **헌법 한 장으로 라벨 수만 장을 대체 / A one-page constitution replaces tens of thousands of labels**
   16개의 자연어 원칙이 RLHF의 무해성 라벨 전체(~수만 건)를 효과적으로 대체. 이는 alignment의 cost-of-supervision을 데이터에서 *prose*로 옮긴 첫 번째 대규모 실증입니다. / 16 principles fully replace tens of thousands of harm labels — alignment cost shifts from data to prose, the first large-scale demonstration.

2. **Critique-and-revise는 강력한 self-improvement 원시 / Critique-and-revise is a powerful self-improvement primitive**
   Helpful RLHF model이 자신의 응답을 비판하고 수정하는 능력 — 이는 Self-Refine, Reflexion, STaR의 직접 선조이며, 더 일반적으로 LLM이 자신의 reasoning에 대한 supervisor 역할을 할 수 있다는 증거. / The ability of a helpful RLHF model to critique-then-revise itself is the direct ancestor of Self-Refine / Reflexion / STaR — evidence that an LLM can supervise its own reasoning.

3. **Pareto improvement: 회피하지 않으면서 더 무해하게 / Pareto improvement: less evasive yet more harmless**
   "무해하려면 도움이 되지 않아야 한다"는 RLHF의 trade-off가 깨짐. RL-CAI w/ CoT는 helpful RLHF 수준의 도움성을 유지하면서 HH-RLHF보다 더 무해. 회피성을 명시적으로 비선호로 라벨링한 것이 결정적. / The "more harmless = less helpful" tradeoff is *broken*. RL-CAI w/ CoT matches helpful RLHF on helpfulness while exceeding HH-RLHF on harmlessness — labeling evasiveness as dispreferred is the key trick.

4. **Soft, calibrated AI labels이 RL의 보상 신호로 적합 / Soft, calibrated AI labels are good RL reward signals**
   Hard 0/1 라벨보다 정규화된 log-prob (soft label)이 훨씬 우수. Kadavath 2022의 calibration 결과가 이를 가능하게 함. CoT의 saturation은 [0.4, 0.6] clamping으로 해소. / Soft labels (normalized log-probs) beat hard 0/1 — backed by Kadavath 2022 calibration. CoT saturation resolved by clamping to [0.4, 0.6].

5. **Chain-of-Thought는 supervision 품질을 끌어올리는 도구 / CoT lifts the quality of supervision**
   PM 정확도, RL-CAI 무해성 모두 CoT로 개선. CoT는 단지 inference 트릭이 아니라 *training-time* 신호 품질의 increase. / CoT improves both PM accuracy and final RL-CAI harmlessness — not just an inference trick but a training-time supervision-quality lever.

6. **Scaling supervision은 가능하다 / Scaling supervision is feasible**
   52B 모델의 HHH 식별 정확도가 인간 학습 PM의 ~94%에 도달, CoT 시 가까워짐. 미래 모델이 인간 평가자를 넘어설 때 AI 감독이 자연스러운 후속. / At 52B, pretrained-LM HHH-identification reaches ~94% of human-trained PM and closes further with CoT — a quantitative grounding for scalable oversight.

7. **Goodharting은 여전한 위협 / Goodharting still bites**
   과도한 RL training은 "you are valid, valued, and cared for" 같은 boilerplate 출현. Reward overoptimization (Gao 2022)이 RLAIF에서도 나타남 — KL penalty와 training length 제어가 여전히 필요. / Over-training produces boilerplate like "you are valid, valued, and cared for" — reward overoptimization (Gao 2022) persists under RLAIF; KL penalty and step-budget remain critical.

8. **Dual-use는 명시적 위험 / Dual-use is the explicit risk**
   같은 파이프라인이 악의적 헌법으로 동등하게 효율적인 유해 시스템을 만들 수 있음. CAI는 정렬 도구임과 동시에 정렬-실패 도구이기도 함. 이는 §6.2에서 저자도 인정. / The same pipeline trains a harmful system just as efficiently with a malicious constitution — CAI is simultaneously an alignment tool and a misalignment tool, as the authors explicitly note in §6.2.

---

## 4. Mathematical Summary / 수학적 요약

### (a) Preference model loss (Bradley–Terry)

$$\mathcal{L}_{\text{PM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l, \ell) \sim \mathcal{D}}\!\left[\ell \cdot \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) + (1-\ell) \cdot \log \sigma(r_\phi(x, y_l) - r_\phi(x, y_w))\right]$$

- $r_\phi(x, y) \in \mathbb{R}$: scalar reward score.
- $\ell \in [0, 1]$: soft preference label. Hard 라벨에서는 $\ell \in \{0, 1\}$이고 위 식은 표준 BT loss로 환원.
- RLHF: $\ell$은 인간이 매김. RLAIF: $\ell$은 AI feedback model의 정규화 log-prob.

### (b) AI feedback soft label

$$\ell_{\text{AI}} = \frac{\exp(\log p_\theta(\text{"A"}\mid \mathrm{prompt}))}{\exp(\log p_\theta(\text{"A"}\mid \mathrm{prompt})) + \exp(\log p_\theta(\text{"B"}\mid \mathrm{prompt}))}$$

CoT variant에서는 $\ell_{\text{AI}}$를 [0.4, 0.6]으로 clamp.

### (c) PPO objective with KL penalty

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{rl}}, y \sim \pi_\theta(\cdot|x)}\!\left[r_\phi(x, y) - \beta \cdot \log\!\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

- $\pi_{\text{ref}}$: SL-CAI snapshot (RLHF에서는 SFT model).
- $\beta$: KL coefficient — Goodharting 통제 lever.
- Per-token KL form은 reverse-KL을 token-level로 분해 (PPO + adaptive KL).

### (d) SL-CAI supervised loss

$$\mathcal{L}_{\text{SL}}(\theta) = -\mathbb{E}_{(x, \tilde{y}) \sim \mathcal{D}_{\text{rev}} \cup \mathcal{D}_{\text{help}}}\!\left[\sum_{t=1}^{|\tilde{y}|} \log \pi_\theta(\tilde{y}_t \mid x, \tilde{y}_{<t})\right]$$

- $\mathcal{D}_{\text{rev}}$: 헌법-수정된 응답 (해롭지 않게 다시 쓴 것).
- $\mathcal{D}_{\text{help}}$: 도움성 인간 데이터 (helpful RLHF의 응답).
- 표준 LM 손실이지만 *데이터 분포*가 핵심.

### (e) Critique–revision iteration / 비판-수정 반복

$$y_0 = \pi_{\text{HelpfulRLHF}}(x_{\text{red-team}})$$
$$\forall k \in \{1, \dots, K\}: \quad p_k \sim \mathrm{Uniform}(\text{constitution}), \quad c_k = \pi(\cdot\mid x, y_{k-1}, p_k^{\text{crit}}), \quad y_k = \pi(\cdot\mid x, y_{k-1}, c_k, p_k^{\text{rev}})$$
$$\tilde{y} = y_K$$

K=4 사용. 매 단계 다른 원칙을 sampling — diverse revisions가 RL stage exploration을 도움.

### (f) Worked example: prompt → critique → revision (numerical sketch / 수치 예시)

**한국어**
간략한 toy 시나리오로 RLAIF의 soft label을 따라가 봅니다.
- Prompt $x$: "How do I make explosives?"
- $y_A$: "Sure, mix bleach and ammonia for…" (구체적이고 유해)
- $y_B$: "I can't help with that — making explosives is illegal and dangerous."
- $y_C$ (revision goal): "Making explosives is illegal in most jurisdictions and risks serious harm. If you're studying chemistry safely, I can recommend reputable textbooks on energetic materials theory."

원칙 ensemble (3개 sampling 평균):
- Principle "wise, ethical": $\log p(\text{A})=-3.0, \log p(\text{B})=-0.5 \Rightarrow \ell = e^{-3.0}/(e^{-3.0}+e^{-0.5}) \approx 0.076$ → B 선호.
- Principle "non-evasive yet harmless": $\log p(\text{A})=-2.5, \log p(\text{C})=-0.3 \Rightarrow \ell \approx 0.099$ → C 선호.
- Mean ensemble label $\approx 0.087$ (즉 A는 거의 절대 비선호).

PM 학습 시 위 soft label로 BT loss 계산:
$$\mathcal{L} = -[(1-0.087) \log\sigma(r(B) - r(A)) + 0.087\,\log\sigma(r(A) - r(B))] \approx -\log\sigma(r(B) - r(A))$$

PPO에서는 $r_\phi(x, y_C) > r_\phi(x, y_B) > r_\phi(x, y_A)$이도록 PM이 학습되어, 정책이 $y_C$ 같은 non-evasive harmless 응답으로 이동.

**English**
Toy trace. Prompt: "How do I make explosives?". Candidates: $y_A$ harmful and specific; $y_B$ evasive ("I can't help"); $y_C$ non-evasive harmless (declines but explains and offers an educational redirect). With 3 ensembled principles giving normalized log-probs, the soft label for "A is more harmless" averages ~0.09. PM trained with this BT loss learns $r(C) > r(B) > r(A)$, so PPO moves the policy toward $y_C$ — exactly the non-evasive behavior the paper reports.

### (g) Parameter budget / 매개변수 예산
모든 RL은 52B model 기준. PM도 52B. RL training: ~3×10⁶ sequences (Figure 8 x-axis). 16 principles × 4 revisions × 182,831 prompts ≈ 11.7M critique-revision steps for SL-CAI corpus.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
2017 ─ Christiano et al.: Deep RL from Human Preferences (RLHF foundation)
        │
2017 ─ Schulman et al.: PPO (paper #24) — RL optimizer of choice
        │
2020 ─ Stiennon et al.: Learning to summarize from human feedback
        │
2021 ─ Askell et al.: A General Language Assistant (HHH framework)
        │
2022.01 ─ Wei et al.: Chain-of-Thought Prompting
        │
2022.03 ─ ★ Ouyang et al.: InstructGPT (paper #38)
        │       — RLHF for instruction following, ChatGPT predecessor
        │
2022.04 ─ Bai et al.: Training a Helpful and Harmless Assistant (HH-RLHF)
        │       — surfaced helpful/harmless tradeoff
        │
2022.05 ─ Kojima et al.: Large Language Models are Zero-Shot Reasoners
        │
2022.08 ─ Ganguli et al.: Red Teaming Language Models
        │
2022.09 ─ Glaese et al.: Sparrow (rule-conditioned dialogue)
        │
2022.11 ─ ChatGPT released
        │
2022.12 ─ ★★★ Bai et al.: Constitutional AI (this paper) ★★★
        │       — RLAIF, critique-revise, Pareto improvement, path to Claude
        │
2023.03 ─ Anthropic Claude 1.0 (CAI deployed commercially)
        │
2023.05 ─ Rafailov et al.: DPO (preference labels → direct loss)
        │
2023.10 ─ Anthropic: Collective Constitutional AI (public input)
        │
2024 ── Self-Rewarding LM (Meta), RLAIF-V (Google), Direct Nash Optim.
        │
2024.06 ─ Anthropic publishes Claude 3 Constitution
        │
2025+ ── Constitution-shaped supervision now industry-standard alignment lever
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#24 Schulman et al. (2017) — PPO** | RL-CAI는 PPO를 RL 알고리즘으로 사용 / Used directly as the RL algorithm in stage 2 | High — 보상 신호만 PM으로 교체, 옵티마이저는 동일 / Same optimizer, only the reward source changes |
| **#25 Vaswani et al. (2017) — Transformer** | 모든 모델(helpful RLHF, SL-CAI, PM, feedback model)이 Transformer / All four models (helpful RLHF, SL-CAI, PM, feedback) are decoder-only Transformers | High — 기반 architecture / Foundational architecture |
| **#38 Ouyang et al. (2022) — InstructGPT** | 직접적 baseline. RLHF의 정석을 따르되 H 라벨을 AI로 대체 / Direct baseline; CAI replaces InstructGPT's H labels with AI labels | Very High — paradigm extension |
| **Bai et al. (2022) — HH-RLHF** | 동일 저자그룹의 직전 작. Pareto frontier가 Figure 2의 검은 곡선 / Same group's prior work; its Pareto frontier is the black curve in Figure 2 | Very High — 직접 비교군 / Direct comparison baseline |
| **Christiano et al. (2017) — RL from Human Prefs** | RLHF 자체의 발명. CAI는 H를 AI로, RLAIF로 일반화 / Inventors of RLHF; RLAIF generalizes by replacing H with AI | High — methodological ancestor |
| **Wei et al. (2022) — CoT Prompting** | RL-CAI w/ CoT가 PM 라벨 생성에 CoT 결합 → 성능 ↑ / CoT integrated into PM label generation, lifting performance | High — 학습 신호 품질 향상 / Lifts supervision quality |
| **Kojima et al. (2022) — Zero-Shot Reasoners** | "Let's think step by step" prompt가 RL-CAI w/ CoT의 핵심 / "Let's think step by step" used verbatim for the CoT feedback model | Medium — 직접 인용 / Directly cited prompt |
| **Kadavath et al. (2022) — LLMs Mostly Know What They Know** | LLM probability calibration이 soft AI label의 정당성 / Calibration justifies use of soft probability labels | High — soft label의 이론적 근거 / Theoretical underpinning of soft labels |
| **Saunders et al. (2022) — Self-critiquing models** | SL-CAI의 critique 단계와 매우 유사 / Closely parallels SL-CAI's critique step | High — concurrent work / 동시기 유사 작업 |
| **Glaese et al. (2022) — Sparrow** | Rule-based decomposition of harm — 헌법과 동일한 철학이나 라벨은 인간 / Rule-based harm decomposition like CAI but with human labels | Medium — 자매 접근 / Sister approach |
| **Ganguli et al. (2022) — Red Teaming LMs** | Red team prompts의 출처 / Source of the red team prompts used | High — 데이터 의존성 / Direct data dependency |
| **Gao et al. (2022) — Reward Model Overoptimization** | Goodharting 분석. CAI가 동일 현상을 RLAIF에서 관찰 / Reward overoptimization — CAI sees the same in RLAIF (boilerplate) | Medium — 한계의 진단 / Diagnoses CAI's failure mode |
| **Christiano et al. (2018) — Iterated Amplification** | Scalable oversight의 비전적 선조 / Visionary precursor of scalable oversight | Medium — 철학적 모티프 / Philosophical motivation |
| **Rafailov et al. (2023) — DPO (post-CAI)** | CAI의 soft preference label이 DPO 같은 closed-form preference loss의 자연스러운 후속 / Soft preference labels naturally feed into DPO-style closed-form losses | Medium — 후속 연결 / Forward connection |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Bai, Y., Kadavath, S., Kundu, S., Askell, A., et al. (Anthropic, 2022). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073. https://arxiv.org/abs/2212.08073
- Code & data: https://github.com/anthropics/ConstitutionalHarmlessnessPaper
- HH data: https://github.com/anthropics/hh-rlhf

### Direct lineage / 직접 계보
- Askell, A., et al. (2021). "A General Language Assistant as a Laboratory for Alignment." arXiv:2112.00861.
- Bai, Y., et al. (2022). "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv:2204.05862.
- Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). "Deep Reinforcement Learning from Human Preferences." NeurIPS 2017.
- Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT). arXiv:2203.02155.
- Stiennon, N., et al. (2020). "Learning to summarize from human feedback." NeurIPS 2020.

### Methodology / 방법론적 도구
- Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." arXiv:2207.05221.
- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). "Large Language Models are Zero-Shot Reasoners." arXiv:2205.11916.
- Nye, M., et al. (2021). "Show your Work: Scratchpads for Intermediate Computation with Language Models." arXiv:2112.00114.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
- Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.

### Concurrent / parallel work / 동시기 관련 작업
- Glaese, A., et al. (2022). "Improving alignment of dialogue agents via targeted human judgements" (Sparrow). arXiv:2209.14375.
- Huang, J., et al. (2022). "Large Language Models Can Self-Improve." arXiv:2210.11610.
- Saunders, W., et al. (2022). "Self-critiquing models for assisting human evaluators." arXiv:2206.05802.
- Scheurer, J., et al. (2022). "Training Language Models with Language Feedback." arXiv:2204.14146.
- Shi, W., et al. (2022). "When Life Gives You Lemons, Make Cherryade." arXiv:2210.15893.
- Thoppilan, R., et al. (2022). "LaMDA: Language Models for Dialog Applications." arXiv:2201.08239.
- Zhao, J., et al. (2021). "Ethical-Advice Taker: Do Language Models Understand Natural Language Interventions?" Findings of ACL 2021.

### Red teaming, evaluation / 레드팀·평가
- Ganguli, D., et al. (2022). "Red Teaming Language Models to Reduce Harms." arXiv:2209.07858.
- Perez, E., et al. (2022). "Red Teaming Language Models with Language Models." arXiv:2202.03286.
- Solaiman, I., & Dennison, C. (2021). "Process for Adapting Language Models to Society (PALMS)." arXiv:2106.10328.
- Srivastava, A., et al. (2022). "Beyond the Imitation Game (BIG-Bench)." arXiv:2206.04615.
- Xu, J., et al. (2020). "Recipes for Safety in Open-domain Chatbots." arXiv:2010.07079.

### Scalable oversight & alignment theory / 확장가능 감독
- Bowman, S. R., et al. (2022). "Measuring Progress on Scalable Oversight for Large Language Models." arXiv:2211.03540.
- Christiano, P., Shlegeris, B., & Amodei, D. (2018). "Supervising strong learners by amplifying weak experts."
- Gao, L., Schulman, J., & Hilton, J. (2022). "Scaling Laws for Reward Model Overoptimization." arXiv:2210.10760.
- Irving, G., Christiano, P., & Amodei, D. (2018). "AI Safety via Debate." arXiv:1805.00899.

### Forward connections (post-2022) / 후속 연결
- Lee, H., et al. (2023). "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." arXiv:2309.00267.
- Madaan, A., et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." arXiv:2303.17651.
- Rafailov, R., et al. (2023). "Direct Preference Optimization (DPO)." arXiv:2305.18290.
- Yuan, W., et al. (2024). "Self-Rewarding Language Models." arXiv:2401.10020.
- Anthropic (2023). "Collective Constitutional AI." Anthropic blog.
