---
title: "Pre-Reading Briefing: Cold Diffusion - Inverting Arbitrary Image Transforms without Noise"
paper_id: "27_bansal_2023"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: A. Bansal, E. Borgnia, H.-M. Chu, J. S. Li, H. Kazemi, F. Huang, M. Goldblum, J. Geiping, T. Goldstein, *NeurIPS* 36 (2023), arXiv:2208.09392
**Author(s)**: Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 확산 모델 (diffusion model) 의 정설을 정면으로 흔든다. 표준 diffusion model은 가우시안 노이즈 추가→제거의 alternation으로 동작하며 이론적 정당화 (score matching, Langevin dynamics, ELBO)는 모두 *Gaussian noise의 무작위성*에 의존한다. 저자들은 **노이즈 없이도, 어떤 결정적 (deterministic) 영상 변환으로도 diffusion-style 생성·복원이 동작함**을 실증한다. Forward를 $x_t = D(x_0, t)$로 일반화 ($D$는 blur, masking, downsampling, snowification, animorphosis 등 *결정적*), restoration network $R_\theta$를 단순 $\ell_1$ loss로 학습. 핵심 기술적 기여는 **개선된 sampler (Algorithm 2)** $x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1)$ 인데, first-order Taylor 분석으로 *$R$이 imperfect해도 결정적 degradation을 정확히 invert*함이 증명된다.

### English
This paper challenges a foundational assumption of diffusion models: **noise is not necessary for diffusion**. Standard diffusion uses Gaussian noise as the forward corruption, and the entire theoretical scaffolding (score matching, Langevin dynamics, variational ELBO) rests on Gaussian-noise properties. The authors demonstrate empirically that *generative diffusion behaviour persists when forward noise is replaced by any deterministic image transform* — blur, masking, downsampling, "snowification" (ImageNet-C operator), or "animorphosis" (animal-image overlay). They define generalised diffusion via $x_t = D(x_0, t)$, train $R_\theta(x_t, t) \approx x_0$ with $\ell_1$ loss, and propose an **improved sampler (Algorithm 2)** $x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1)$ which they prove (first-order Taylor) inverts smooth $D$ exactly even when $R$ is imperfect. The take-away: Gaussian noise is one option among many — a generalised theory of diffusion is needed.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 2015-2022년 동안 diffusion model의 모든 주요 진보 (Sohl-Dickstein 2015, NCSN 2019, DDPM 2020, DDIM 2020, score-based SDE 2021, DDRM 2022, DPS 2022)는 *Gaussian noise를 forward corruption으로* 가정한다. 이론적 정당화 — score matching, Langevin reversal, ELBO — 가 모두 가우시안 통계에 의존하기 때문이다. 그러나 2022년 후반-2023년 사이에 동시에 여러 그룹 (Rissanen+ 2022 Inverse Heat Dissipation, Hoogeboom-Salimans 2022 Blurring Diffusion)이 *blur diffusion* 이 작동함을 보고한다. Cold Diffusion은 이를 가장 일반적·실험적으로 광범위한 형태로 제시: **5개의 매우 다른 deterministic degradation에서 generative behaviour 입증**, 그리고 안정성을 보장하는 새로운 sampler (Algorithm 2) 제시. 이 논문 이후 diffusion 이해는 "Gaussian noise = diffusion의 정의"에서 "Gaussian noise = diffusion의 *한 인스턴스*"로 패러다임 전환.

**English**: Through 2015-2022 every major diffusion advance (Sohl-Dickstein 2015, NCSN, DDPM, DDIM, score-based SDEs, DDRM, DPS) assumed *Gaussian noise as the forward corruption*, because the theoretical scaffolding requires Gaussian statistics. In late 2022-2023 several groups concurrently reported that *blur diffusion* works (Rissanen 2022 Inverse Heat Dissipation, Hoogeboom-Salimans 2022). Cold Diffusion is the most general statement: generative behaviour on five very different deterministic degradations plus a new stability-guaranteed sampler. This paper marks a *paradigm divergence*: pre-2022, Gaussian noise *was* diffusion; post-2022, Gaussian noise is *one instance* of diffusion.

### 타임라인 / Timeline

```
2015      Sohl-Dickstein+ — non-equilibrium thermodynamics generative model
2019      Song-Ermon — NCSN (Gaussian-noise score matching)
2020      Ho+ — DDPM (Gaussian-noise variational inference)
2021      Song+ — Score-based SDEs (continuous-time Gaussian diffusion)
2022      Karras+ — EDM (Gaussian-noise design space)
2022      Rissanen+ — Inverse Heat Dissipation (concurrent: blur diffusion)
2022      Hoogeboom-Salimans — Blurring Diffusion (concurrent)
2023 ★★   BANSAL+ — COLD DIFFUSION (THIS PAPER)
2023      Daras+ — Soft Diffusion (interpolation between hot and cold)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **DDPM forward/reverse process**: standard diffusion의 Gaussian-noise forward chain.
- **Score matching ↔ DAE 동치 (Vincent 2011)**: 표준 이론적 정당화.
- **Langevin dynamics**: random walk 기반 sampling.
- **Image degradation operators**: Gaussian blur 시퀀스 ($\bar G_t * x_0$), pixel masking, bicubic downsampling.
- **ImageNet-C "snow" corruption**: structured 비-가우시안 손상 모델.
- **First-order Taylor expansion**: smooth function의 1차 근사 분석.
- **U-Net / CNN regression with $\ell_1$ loss**: 단순 supervised denoiser 학습.
- **FID metric**: generation quality 평가.

**English**:
- **DDPM forward/reverse process**: the standard Gaussian-noise diffusion chain.
- **Score matching ↔ DAE equivalence (Vincent 2011)**: standard theoretical justification.
- **Langevin dynamics**: random-walk-based sampling.
- **Image degradation operators**: iterated Gaussian blur ($\bar G_t * x_0$), pixel masking, bicubic downsampling.
- **ImageNet-C "snow" corruption**: a structured non-Gaussian degradation.
- **First-order Taylor expansion**: linearised analysis of smooth maps.
- **U-Net / CNN regression with $\ell_1$ loss**: simple supervised regressor training.
- **FID metric**: standard generation-quality measure.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Cold diffusion | Gaussian noise 대신 결정적 변환으로 forward chain 정의 / Diffusion with deterministic forward corruption instead of Gaussian noise. |
| Hot diffusion | 표준 Gaussian-noise diffusion (DDPM 등) / Standard Gaussian-noise diffusion for contrast. |
| Degradation operator $D(x, t)$ | 시간 $t$ 인덱스의 결정적 영상 변환. boundary $D(x, 0) = x$ / Deterministic transform of an image at level $t$, with $D(x,0) = x$. |
| Restoration network $R_\theta$ | $R_\theta(x_t, t) \approx x_0$ 추정 네트워크. $\ell_1$로 학습 / Network estimating $x_0$ from $x_t$, trained with $\ell_1$ loss. |
| Naive sampler (Algorithm 1) | $x_{s-1} = D(R(x_s, s), s-1)$. hot diffusion에서는 OK, cold에서는 fail / Standard DDPM-style update; works for noisy diffusion but accumulates error in cold setting. |
| Improved sampler (Algorithm 2) | $x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1)$. error cancellation / Update rule using error cancellation; the key technical contribution. |
| First-order stability theorem | smooth $D$에 대해 Algorithm 2가 $R$ 정확도와 무관하게 정확한 chain을 따름 / For smooth $D$, Algorithm 2 produces exact $D(x_0, s-1)$ regardless of $R$'s accuracy (§3.3). |
| Snowification | ImageNet-C dataset의 snow corruption operator. structured 비-가우시안 / ImageNet-C "snow" overlay used as a structured non-Gaussian forward. |
| Animorphosis | forward $D(x, t) = (1-\alpha_t) x + \alpha_t z$, $z$ = random animal image (AFHQ) / Forward replaces face with a random animal; reverse generates faces. The most provocative experiment. |
| Band-pass interpretation | blur diffusion에서 Algorithm 2가 frequency-add cascade로 해석 / For blur diffusion, Algorithm 2 progressively re-introduces frequencies in band-pass order. |
| Random init dependence | 결정적 forward에서 unconditional generation을 위한 외부 randomness / Generative variability must be injected externally because $D$ is deterministic. |

---

## 5. 수식 미리보기 / Equations Preview

**핵심 1: 일반화 forward / Generalised forward (Eq. 1 setup)**

$$
x_t = D(x_0, t),\qquad D(x_0, 0) = x_0,\qquad t \in [0, T]
$$

**한국어**: forward chain이 임의의 결정적 변환. Gaussian-noise는 $D(x_0, t) = x_0 + \sqrt{t}\,\epsilon$ 의 한 특수 경우.

**English**: The forward chain is any deterministic transform. Gaussian-noise diffusion is the special case $D(x_0, t) = x_0 + \sqrt{t}\,\epsilon$.

**핵심 2: 학습 손실 / Training loss (Eq. 1)**

$$
\min_\theta \mathbb E_{x \sim \mathcal X, t}\,\bigl\|R_\theta(D(x, t), t) - x\bigr\|_1
$$

**한국어**: 단순 $\ell_1$ regression. score matching이나 ELBO 같은 복잡한 손실 불필요.

**English**: Simple $\ell_1$ regression — no score matching, no ELBO. The architecture is identical to a supervised regressor.

**핵심 3: 개선된 sampler / Improved sampler (Algorithm 2, key)**

$$
x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1),\qquad \hat x_0 = R(x_s, s)
$$

**한국어**: $x_s$의 추정값 $D(\hat x_0, s)$를 빼고, 한 step 더 깨끗한 $D(\hat x_0, s-1)$을 더함. $R$의 오차가 두 항에 동일하게 나타나 *상쇄*.

**English**: Subtract the current-step estimate $D(\hat x_0, s)$ and add the next-step estimate $D(\hat x_0, s-1)$. Errors in $R$ appear in both terms and *cancel*.

**핵심 4: First-order Taylor stability / 1차 테일러 안정성 (§3.3 결과)**

$$
D(x, s) = x + s\cdot e + \mathcal O(s^2)\;\;\Longrightarrow\;\; x_{s-1} = D(x_0, s-1) + \mathcal O(s^2)
$$

**한국어**: smooth $D$에 대해 Algorithm 2가 *$R$의 정확도와 무관하게* exact forward chain을 따름. 귀납으로 정확 reconstruction.

**English**: For smooth $D$, Algorithm 2 produces $D(x_0, s-1)$ to first order *independently of $R$'s accuracy*. By induction, the chain is followed exactly.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
- **§2 (Background)**: 표준 diffusion의 *Gaussian-noise 의존성*이 어디에 들어가는지에 주목 (score matching, Langevin reversal, ELBO).
- **§3.2-3.3**: 핵심 기술. Algorithm 1과 Algorithm 2의 차이를 직접 시뮬레이션해 볼 것 (paper-and-pencil 1D toy). first-order Taylor 증명을 검증.
- **§4 (5 degradations)**: blur, masking, downsampling, snow, animorphosis 각각이 *왜* 작동하는지에 주목. animorphosis가 가장 도발적 — forward "noise"가 데이터와 구조적으로 멀어도 generative behaviour 유지.
- **§5 (Generation FID)**: Cold diffusion FID가 hot보다 *나쁘다*는 점에 정직하라. 이 논문의 주장은 *대체*가 아닌 *대안*.
- **§6 (Discussion)**: 이론의 미완성 — 현재 score matching/Langevin/ELBO이 모두 Gaussian 가정. 이 논문 후 어떤 일반화가 필요한가에 대한 논의.
- **Common stumbling blocks**: (1) Algorithm 1 vs 2의 차이가 왜 cold에서만 중요한지 (hot diffusion의 noise injection이 자동 보정), (2) animorphosis에서 "noise"가 random animal인 것의 의미, (3) FID gap이 의미하는 한계.

**English**:
- **§2 Background**: notice exactly where Gaussian-noise dependence enters the standard theory (score matching, Langevin reversal, ELBO).
- **§3.2-3.3**: the technical core. Simulate Algorithm 1 vs 2 by hand on a 1-D toy and verify the first-order Taylor proof.
- **§4 (5 degradations)**: focus on *why* each works. Animorphosis is the most provocative — generative behaviour persists when "noise" is far from the data distribution.
- **§5 (Generation FID)**: be honest that cold-diffusion FID is *worse* than DDPM. The claim is "alternative paradigm", not "replacement".
- **§6 (Discussion)**: theory gaps — score matching/Langevin/ELBO all assume Gaussian. What generalisation is needed?
- **Stumbling blocks**: (1) why Algorithm 1 vs 2 only matters in the cold regime (hot diffusion's injected noise self-corrects), (2) the meaning of "noise = random animal" in animorphosis, (3) what the FID gap implies for practical use.

---

## 7. 현대적 의의 / Modern Significance

**한국어**: 이 논문은 *diffusion model 이해의 패러다임 분기*다. 이후 후속 연구(Daras+ 2023 Soft Diffusion 등)가 hot과 cold의 *interpolation*을 탐색하며, EDM/DDPM이 spanning하는 design space를 훨씬 넓게 펼쳤다. 실용적 측면에서는 노이즈 모델이 명확하지 않은 산업/현미경/방송 영상에서 cold-style diffusion이 더 자연스러운 prior가 될 수 있다. 또한 Diffusion 모델의 *flow-based generalisation* 과 *projection-based interpretation* 으로 가는 길을 열어, ICLR/NeurIPS 2024 이후의 operator-theoretic 일반화 논문들의 출발점이 되었다. 본 reading list에서는 paper #25-28의 Gaussian-noise framework와 직접 대비되며, 노이즈가 아닌 *결정적 손상*이 지배적인 (motion blur, downsampling, mask 등) 실 응용 시나리오에서의 design rationale을 제공한다.

**English**: This paper marks a *paradigm divergence* in diffusion-model understanding. Successors (Daras 2023 Soft Diffusion etc.) explored interpolation between hot and cold, vastly expanding the design space spanned by EDM/DDPM. Practically, where the noise model is poorly characterised (industrial, microscopy, broadcast), cold-style diffusion may be a more natural prior. The paper also opened the door to *flow-based generalisations* and *projection-based interpretations* of diffusion, becoming the seed for ICLR/NeurIPS 2024+ operator-theoretic generalisations. Within this reading list it directly contrasts the Gaussian-noise framework of papers #25-28, providing design rationale for applications where deterministic corruptions (motion blur, downsampling, masks) dominate.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
