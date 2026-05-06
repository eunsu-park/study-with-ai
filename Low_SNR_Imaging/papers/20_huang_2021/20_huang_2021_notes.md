---
title: "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
authors: Tao Huang, Songjiang Li, Xu Jia, Huchuan Lu, Jianzhuang Liu
year: 2021
journal: "IEEE/CVF CVPR 2021, pp. 14781–14790"
doi: "10.1109/CVPR46437.2021.01454"
topic: Low-SNR Imaging / Self-Supervised Deep Denoising
tags: [neighbor2neighbor, sub-sampling, noise2noise, regularizer, single-image, real-world-denoising, sidd, huang-2021]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 20. Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images / 이웃 픽셀 기반 단일 영상 자기지도 잡음 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
Neighbor2Neighbor (Nb2Nb)는 **단일 잡음 영상**에서 *random neighbour sub-sampling*으로 **pseudo-noisy pair**를 생성하고, **Noise2Noise 손실 + regulariser**로 학습하는 framework다. N2V의 blind-spot, N2S의 J-invariance, S2S의 dropout이 모두 *네트워크 구조 또는 stochastic 추론*에 의존했던 데 반해, Nb2Nb는 *입력 데이터 단계에서* 두 가지를 분리한다 — N2N의 단순한 학습 절차에 single-image regime을 접목한다는 의미.

핵심 기여:

(i) **Random neighbour sub-sampler $G = (g_1, g_2)$**: 잡음 영상 $\mathbf y$를 $2\times 2$ 셀로 분할 → 각 셀에서 *서로 인접한 두 픽셀*을 무작위로 골라 한 픽셀은 $g_1(\mathbf y)$, 다른 픽셀은 $g_2(\mathbf y)$로 보낸다. 결과: 같은 장면에 대한 두 sub-image $(g_1(\mathbf y), g_2(\mathbf y))$ — 픽셀 단위 *조건부 독립 잡음 + 거의 동일 신호*.

(ii) **Noise2Noise 손실의 직접 적용 + regularisation term** (Eq. 7):
$$
\mathcal L = \underbrace{\|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y)\|^2_2}_{\mathcal L_{\text{rec}}} + \gamma \cdot \underbrace{\|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y) - (g_1(f_\theta(\mathbf y)) - g_2(f_\theta(\mathbf y)))\|^2_2}_{\mathcal L_{\text{reg}}}
$$
$\mathcal L_{\text{reg}}$는 sub-sampled 신호 사이의 *non-zero gap* $\varepsilon = \mathbb E_{\mathbf y|\mathbf x}[g_2(\mathbf y)] - \mathbb E_{\mathbf y|\mathbf x}[g_1(\mathbf y)]$을 보정 — over-smoothing 방지.

(iii) **이론 (Theorem 1)**: $\mathbf y, \mathbf z$가 $\mathbf x$에 conditionally independent이고 $\mathbb E[\mathbf y|\mathbf x] = \mathbf x$, $\mathbb E[\mathbf z|\mathbf x] = \mathbf x + \boldsymbol\varepsilon$일 때
$$
\mathbb E_{\mathbf x, \mathbf y, \mathbf z}\|f_\theta(\mathbf y) - \mathbf z\|^2 = \mathbb E_{\mathbf x, \mathbf y, \mathbf z}\|f_\theta(\mathbf y) - \mathbf x\|^2 - \sigma^2_{\mathbf z} + 2\,\mathbb E_{\mathbf x, \mathbf y}\langle f_\theta(\mathbf y) - \mathbf x, \boldsymbol\varepsilon\rangle
$$
즉 N2N loss = supervised loss + 잡음 분산 + $\boldsymbol\varepsilon$이 만드는 *bias term*. $\boldsymbol\varepsilon \to 0$이면 N2N과 동등.

(iv) **추론 단계의 단순함**: 학습된 $f_\theta$를 *full noisy image*에 직접 적용 (Fig. 1b) — DIP/S2S의 ensemble inference 불필요.

(v) **State-of-the-art 성과 (2021)**: SIDD Benchmark (real-world raw-RGB) **50.76 / 0.991** (RRG architecture), DBSN(50.13) / Laine19-pme(48.46) / N2V(48.01) 추월. supervised N2C baseline (50.60)도 능가하는 매우 강력한 결과.

### English
Neighbor2Neighbor (Nb2Nb) generates a *pseudo-noisy pair* from a single noisy image via random neighbour sub-sampling, then applies the Noise2Noise loss with an additional regulariser to compensate for the small but non-zero signal gap between the two sub-images. Three core ingredients:

(i) **Random neighbour sub-sampler** $G = (g_1, g_2)$: partition image into $2 \times 2$ cells, randomly pick two adjacent pixels per cell to form sub-images.

(ii) **Regularised loss** (Eq. 7): reconstruction term $\mathcal L_{\text{rec}} = \|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y)\|^2$ plus regulariser $\mathcal L_{\text{reg}}$ penalising over-smoothing from non-zero $\varepsilon$.

(iii) **No noise model assumption + no architectural restriction** — any denoising network architecture works (UNet, DnCNN, RRG); no blind-spot constraint, no Bernoulli dropout. Inference is single forward pass.

Theorem 1 quantifies how training on noisy-noisy pairs with conditional bias $\varepsilon$ deviates from the supervised loss — small gap = small error.

Achieves SOTA on synthetic Gaussian/Poisson and real-world SIDD Benchmark (50.76/0.991 with RRG backbone), surpassing all prior single-image self-supervised denoisers as of 2021.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & Motivation / 도입과 동기

#### 한국어
- **문제**: deep denoising은 paired clean/noisy data 필요. 실제 영상에서는 비싸거나 불가능 (motion, medical).
- **기존 self-sup**: (a) multiple noisy observations needed (N2N), (b) blind-spot architecture restricts network design (N2V, N2S, Laine), (c) dependence on noise model (Noisier2Noise, R2R).
- **Nb2Nb의 도전**: single noisy image, no network constraint, no noise model.
- **핵심 통찰**: 자연 영상에서 *neighbouring pixels* signal 거의 동일, 잡음 (typically pixel-independent) 통계적 독립 → "pseudo-N2N pair"의 자연적 후보.
- **차이점 vs N2V/N2S**: N2V/N2S는 *output-level masking*으로 J-invariance 강제. Nb2Nb는 *input-level sub-sampling*으로 두 noisy view 생성 → 표준 supervised pipeline 그대로 사용.

#### English
- Goal: train a deep denoiser using *only single noisy images*, without (a) multiple noisy observations, (b) blind-spot constraints, or (c) noise-model assumptions.
- Key insight: spatially adjacent pixels share nearly identical clean signal but have conditionally independent noise — naturally usable as a Noise2Noise pair.
- Decoupling principle: shifts the self-supervision burden from network architecture (N2V/N2S/Laine) to data construction (sub-sampling), keeping training pipeline identical to ordinary supervised denoising.

---

### Part II: §2 Related Work / 관련 연구

#### 한국어
- **Supervised on pairs**: DnCNN, FFDNet, RIDNet, MIRNet — needs clean GT.
- **Multi-noisy**: Noise2Noise (#16), Cryo-CARE (#15) — needs paired noisy observations.
- **Blind-spot single-image**: N2V (#17), N2S (#18), Probabilistic-N2V (Krull 2019), Laine'19 — restrict architecture.
- **Synthesised pairs**: Noisier2Noise (Moran 2020), R2R (Pang 2021) — assume known noise model.
- **DIP** (Ulyanov 2018), **S2S** (#19) — train per-image, much slower.

#### English
- Multi-noisy: N2N, Cryo-CARE.
- Blind-spot: N2V, N2S, P-N2V, Laine'19, DBSN.
- Noise-injected pairs: Noisier2Noise, R2R.
- Per-image training: DIP, Self2Self.
- Nb2Nb's niche: single noisy image, free architecture, no noise model.

---

### Part III: §3 Theoretical Motivation / 이론적 동기

#### 한국어 — §3.1 Noise2Noise revisit
N2N 손실 (Eq. 1):
$$
\arg\min_\theta \mathbb E_{\mathbf x, \mathbf y, \mathbf z}\|f_\theta(\mathbf y) - \mathbf z\|^2_2
$$
$\mathbf y, \mathbf z$ independent given $\mathbf x$, $\mathbb E[\mathbf y|\mathbf x] = \mathbb E[\mathbf z|\mathbf x] = \mathbf x$. Lehtinen 보임: 이 손실의 minimiser = supervised loss minimiser.

#### English — §3.1
Recap of N2N loss; equivalence to supervised loss when both noisy targets have zero conditional bias.

#### 한국어 — §3.2 Paired images with similar (but not identical) ground truths

**Theorem 1**: $\mathbf y, \mathbf z$ noisy pair, $\mathbb E[\mathbf y|\mathbf x] = \mathbf x$, $\mathbb E[\mathbf z|\mathbf x] = \mathbf x + \boldsymbol\varepsilon$. $\boldsymbol\sigma_{\mathbf z}^2$ variance of $\mathbf z$. Then:
$$
\mathbb E_{\mathbf x, \mathbf y, \mathbf z}\|f_\theta(\mathbf y) - \mathbf z\|^2_2 = \mathbb E_{\mathbf x, \mathbf y, \mathbf z}\|f_\theta(\mathbf y) - \mathbf z\|^2_2 - \sigma^2_{\mathbf z} + 2\,\mathbb E_{\mathbf x, \mathbf y}\langle f_\theta(\mathbf y) - \mathbf x, \boldsymbol\varepsilon\rangle \quad (\text{Eq. 2})
$$
(원논문 표기): $\mathbb E\|f - \mathbf z\|^2 = \mathbb E\|f - \mathbf x\|^2 - \sigma^2_{\mathbf z} + 2\mathbb E\langle f - \mathbf x, \boldsymbol\varepsilon\rangle$. 첫 두 항은 N2N과 동일, 세 번째 항이 *bias-induced over-smoothing*. $\boldsymbol\varepsilon \to 0$이면 사라짐.

#### English — §3.2 Theorem 1
When ground-truth values of the two noisy realisations differ by $\boldsymbol\varepsilon$, the standard N2N loss equals the supervised loss minus noise variance plus a cross term $2\langle f-\mathbf x, \boldsymbol\varepsilon\rangle$ — the second term causes systematic over-smoothing toward $\mathbf x + \boldsymbol\varepsilon/2$.

#### 한국어 — §3.3 Extension to single noisy image
Sub-sampler $G = (g_1, g_2)$. $g_1(\mathbf y), g_2(\mathbf y)$는 신호가 거의 같고 잡음이 conditionally independent → $\boldsymbol\varepsilon = \mathbb E_{\mathbf y|\mathbf x}[g_2(\mathbf y)] - \mathbb E_{\mathbf y|\mathbf x}[g_1(\mathbf y)] \neq 0$.

직접 $\|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y)\|^2$을 minimise하면 over-smoothing.

**Constrained optimisation** (Eq. 4-5): 이상적 denoiser $f^*_\theta$는 $f^*_\theta(\mathbf y) = \mathbf x$, $f^*_\theta(g_\ell(\mathbf y)) = g_\ell(\mathbf x)$이므로
$$
g_1(\mathbf y) - g_2(\mathbf y) - (g_1(f^*_\theta(\mathbf y)) - g_2(f^*_\theta(\mathbf y))) = 0
$$
즉 *입력 측 sub-sample 차이*와 *출력 측 sub-sample 차이*가 같아야 한다. 이를 regulariser로 사용:
$$
\mathcal L_{\text{reg}} = \|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y) - (g_1(f_\theta(\mathbf y)) - g_2(f_\theta(\mathbf y)))\|^2
$$
$\gamma$로 가중. $g_1(f_\theta(\mathbf y))$, $g_2(f_\theta(\mathbf y))$에는 *gradient 차단* (no_grad) → 학습 안정성.

**Total loss** (Eq. 7):
$$
\mathcal L = \mathcal L_{\text{rec}} + \gamma \mathcal L_{\text{reg}}
$$
#### English — §3.3
Direct N2N loss on sub-sampled pair over-smooths because $\boldsymbol\varepsilon \neq 0$. Constrained-optimisation reformulation introduces a regulariser that penalises mismatch between (input-side sub-sample difference) and (output-side sub-sample difference), forcing the network to preserve high-frequency content. Gradients on $f_\theta(\mathbf y)$ are stopped to stabilise training.

---

### Part IV: §4 Method / 방법

#### 한국어 — §4.1 Generation of training pairs

**Random neighbour sub-sampler** (Fig. 2):
1. $\mathbf y$ (size $W \times H$)를 $\lfloor W/k\rfloor \times \lfloor H/k\rfloor$ 셀로 분할 (cell size $k \times k$, 논문에서 $k=2$).
2. 각 $2\times 2$ 셀에서 *두 인접 위치*를 무작위 선택 → $g_1, g_2$ 위치.
3. 결과: 두 sub-image $g_1(\mathbf y), g_2(\mathbf y)$ 각각 size $\lfloor W/2\rfloor \times \lfloor H/2\rfloor$.

(Fig. 2): 빨간 픽셀 = $g_1$, 파란 픽셀 = $g_2$. 한 셀에 두 인접 위치(가로, 세로, 또는 대각선)를 random 선택. 매 iteration마다 새 random sub-sampler.

#### English — §4.1
Algorithm: divide image into 2×2 cells; in each cell, randomly select two neighbouring (orthogonally adjacent or diagonal) positions as members of $g_1$ and $g_2$. Generates two half-resolution sub-images per random sample. Rationale: paired pixels share nearly identical clean signal (1-pixel apart) but independent noise.

#### 한국어 — §4.2 Self-supervised training

Algorithm 1:
```
while not converged:
    sample noisy image y from training set
    generate random sub-sampler G = (g_1, g_2)
    compute L_rec = || f_θ(g_1(y)) - g_2(y) ||^2
    compute f_θ(y) WITHOUT gradient
    apply same G to f_θ(y) → g_1(f_θ(y)), g_2(f_θ(y))
    compute L_reg = || f_θ(g_1(y)) - g_2(y) - (g_1(f_θ(y)) - g_2(f_θ(y))) ||^2
    minimise L_rec + γ L_reg
```

#### English — §4.2
Algorithm 1: random sub-sampler each step → $\mathcal L_{\text{rec}}$ on sub-sampled pair → forward without gradient on full $\mathbf y$ → apply same sub-sampler to outputs → compute $\mathcal L_{\text{reg}}$ → backprop combined loss. Inference is a single forward pass on full $\mathbf y$.

#### 한국어 — §5.1 Implementation details
- Architecture: modified UNet (Krull 2019) with three 1×1 conv at end; or RRG (Restormer-style).
- Adam, lr 3e-4 (synthetic) / 1e-4 (real-world), 100 epochs, half-decay every 20 epochs.
- Batch 4. $\gamma = 2$ (synthetic), $\gamma = 1$ (real-world).
- Synthetic: 50k ImageNet validation patches, Gaussian σ=25 / σ∈[5,50] / Poisson λ=30 / λ∈[5,50].
- Real-world: SIDD Medium (raw-RGB, smartphone).
- Test: Kodak (24), BSD300, Set14, SIDD Validation/Benchmark.

#### English — §5.1
UNet variant or RRG backbone. Adam, lr 3e-4/1e-4, 100 epochs. $\gamma = 2$ for synthetic, 1 for real-world. Trained on 50k ImageNet patches with synthetic Gaussian/Poisson noise; SIDD Medium for real-world.

---

### Part V: §5 Experiments / 실험

#### 한국어 — §5.2 Synthetic Gaussian / Poisson (Table 1)

| Noise | Method        | Kodak | BSD300 | Set14 |
|-------|---------------|-------|--------|-------|
| Gauss σ=25 | Baseline N2C  | 32.43/0.884 | 31.05/0.879 | 31.40/0.869 |
|       | Baseline N2N  | 32.41/0.884 | 31.04/0.878 | 31.37/0.868 |
|       | CBM3D         | 31.87/0.868 | 30.48/0.861 | 30.88/0.854 |
|       | DIP           | 27.20/0.720 | 26.38/0.708 | 27.16/0.758 |
|       | Self2Self     | 31.28/0.864 | 29.86/0.849 | 30.08/0.839 |
|       | N2V           | 30.32/0.821 | 29.34/0.824 | 28.84/0.802 |
|       | Laine19-pme   | 32.40/0.883 | 30.99/0.877 | 31.36/0.866 |
|       | DBSN          | 31.64/0.856 | 29.80/0.839 | 30.63/0.846 |
|       | **Ours**      | **32.08/0.879** | **30.79/0.873** | **31.09/0.864** |

Self-supervised single-image 중 Laine19-pme(가장 강한 baseline)와 동등 또는 우월. CBM3D 일관 추월.

#### English — §5.2 Synthetic
Across 4 noise types and 3 datasets, Nb2Nb beats CBM3D, DIP, Self2Self, N2V, DBSN; matches Laine19-pme (which uses probabilistic post-processing requiring exact noise prior); approaches supervised N2C/N2N within 0.3-0.5 dB.

#### 한국어 — §5.3 Real-world SIDD (Table 2)

| Method                  | SIDD Validation | SIDD Benchmark |
|-------------------------|-----------------|----------------|
| Baseline N2C            | 51.19/0.991     | 50.60/0.991    |
| Baseline N2N            | 51.21/0.991     | 50.62/0.991    |
| BM3D                    | 48.92/0.986     | 48.60/0.986    |
| N2V                     | 48.55/0.984     | 48.01/0.983    |
| Laine19-pme(Gauss)      | 42.87/0.939     | 42.17/0.935    |
| Laine19-pme(Poisson)    | 48.98/0.985     | 48.46/0.984    |
| DBSN                    | 50.13/0.987     | 49.56/0.987    |
| **Ours (UNet)**         | **51.06/0.991** | **50.47/0.990**|
| **Ours (RRGs)**         | **51.39/0.991** | **50.76/0.991**|

Nb2Nb (RRGs)는 supervised N2C(50.60)도 추월. **단일 영상 self-sup가 supervised baseline 능가하는 첫 사례 중 하나**. Laine19는 noise model을 잘못 잡으면(Gauss로 잡으면 42 dB) 폭락하나 Nb2Nb는 noise-free.

#### English — §5.3 Real-world
On SIDD Benchmark (raw-RGB smartphone noise), Nb2Nb (RRGs) reaches 50.76/0.991, beating supervised N2C (50.60), all blind-spot single-image methods, and BM3D. Demonstrates that explicit noise modelling (Laine'19 with Gaussian post-processing) is *fragile* — wrong noise prior crashes performance to 42 dB. Nb2Nb's noise-model freedom is a major practical advantage.

#### 한국어 — §5.4 Ablation

**γ sensitivity (Table 3, Kodak)**:
| γ           | Gauss σ=25 | Gauss σ∈[5,50] | Poisson λ=30 | Poisson λ∈[5,50] |
|-------------|-----------|----------------|--------------|------------------|
| 0           | 31.77/0.874 | 31.67/0.866 | 31.21/0.866 | 30.67/0.853 |
| **2**       | **32.08/0.879** | **32.10/0.870** | **31.44/0.870** | **30.86/0.855** |
| 8           | 32.02/0.878 | 31.99/0.865 | 31.38/0.870 | 30.74/0.850 |
| 20          | 31.95/0.874 | 31.87/0.861 | 31.21/0.864 | 30.58/0.846 |

$\gamma = 0$ (regulariser 없음): smoothness 과도. $\gamma$ 너무 크면 noisy. Sweet spot $\gamma = 2$.

**Sampling strategy (Table 4)**: random sub-sampler vs fixed-location sub-sampler — random이 +0.1 dB 우위. Randomness가 sample-efficiency에 도움.

#### English — §5.4 Ablation
Regulariser weight $\gamma$ shows clear U-curve: $\gamma=0$ over-smooths, $\gamma=20$ under-denoises, $\gamma=2$ optimal. Random vs fixed-location sub-sampler: random +0.1 dB.

---

## 3. Key Takeaways / 핵심 시사점

1. **Sub-sampling = data-side self-supervision** — N2V/N2S/S2S/Laine은 모두 *네트워크 구조*에 self-supervision constraint(blind-spot, J-invariance, dropout) 부과. Nb2Nb는 *입력 데이터 단계*에서 두 noisy view를 만들고 표준 N2N 학습 적용 → architecture-agnostic. / Nb2Nb shifts self-supervision from network architecture to data construction. Any standard denoising network (UNet, DnCNN, RRG, Restormer) works without modification.

2. **자연 영상의 spatial correlation이 핵심** — 이웃 픽셀의 신호 거의 동일 + 잡음 독립 → "pseudo-N2N pair" 자연 형성. 이 방법은 **noise model을 가정하지 않음** (Gaussian, Poisson, real raw 모두 작동). / Exploits the ubiquitous spatial correlation of natural-image signals: neighbouring pixels share signal but have independent noise. Works for Gaussian, Poisson, and real-world camera noise without modification.

3. **Theorem 1: signal gap $\boldsymbol\varepsilon$이 over-smoothing의 원인** — 이웃 픽셀이 *완전히 같지 않으므로* ($\boldsymbol\varepsilon \neq 0$) 단순 N2N 손실은 $\langle f - \mathbf x, \boldsymbol\varepsilon\rangle$이라는 cross term이 남는다. Regulariser $\mathcal L_{\text{reg}}$가 이를 상쇄. / The non-zero signal gap between sub-sampled pixels ($\boldsymbol\varepsilon \neq 0$) introduces a bias term in the N2N loss; the regulariser specifically cancels this bias, preventing over-smoothing of sharp edges.

4. **Regulariser의 형태가 우아하다** — $\mathcal L_{\text{reg}}$는 *입력 sub-sample 차이*와 *출력 sub-sample 차이*가 같아야 한다는 자연스러운 consistency constraint. 즉 denoiser는 sub-sampling operation과 *commutative*해야 한다. / The regulariser enforces that the denoiser commutes with the sub-sampling operation — a natural geometric consistency constraint that preserves high-frequency edge content.

5. **Real-world SIDD에서 supervised를 능가** — Single noisy image로 학습한 self-sup가 paired clean/noisy로 학습한 N2C(50.60)를 능가(50.76). 이유: SIDD의 supervised pair가 motion·다른 ISP로 인해 *완벽한 GT가 아님* — Nb2Nb는 이러한 misalignment를 회피. / On real-world SIDD, single-image self-supervised Nb2Nb outperforms supervised N2C — likely because SIDD's "clean" reference is itself imperfect (motion, ISP differences), and Nb2Nb sidesteps this misalignment.

6. **Inference is single forward pass** — DIP은 per-image training이 느리고, S2S는 $N=50$ ensemble. Nb2Nb는 학습이 amortised(데이터셋 단위), 추론은 single pass → **practical deployability** 양호. / Unlike DIP and Self2Self, Nb2Nb amortises training across a dataset and requires only a single forward pass at inference — practically deployable at scale.

7. **$\gamma$ hyperparameter의 명확한 sweet spot** — $\gamma = 0$ 과한 평탄, $\gamma$ 너무 크면 noisy 잔존. Synthetic $\gamma=2$, real-world $\gamma=1$로 robust selection. / The regulariser weight $\gamma$ has an interpretable role (smoothness ↔ noisiness tradeoff) and shows a clear optimum around 1-2 across datasets — not finicky.

8. **Random sub-sampling > fixed-location sub-sampling** — 매 iteration마다 random partition이 *training distribution diversity*를 늘려 +0.1-0.3 dB 우위 (Table 4). Stochastic sampler의 implicit regularisation. / Stochastic sub-samplers outperform fixed-location ones by ~0.1-0.3 dB — randomness in pair construction acts as implicit data augmentation across training iterations.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Random neighbour sub-sampler / 무작위 이웃 sub-sampler
For image $\mathbf y \in \mathbb R^{W \times H}$, partition into $\lfloor W/k\rfloor \times \lfloor H/k\rfloor$ cells ($k=2$). In each cell, choose two neighbouring positions; denote them $g_1, g_2$. Output:
$$
g_1(\mathbf y), g_2(\mathbf y) \in \mathbb R^{\lfloor W/2\rfloor \times \lfloor H/2\rfloor}
$$
### 4.2 Theorem 1 / 정리 1 (Eq. 2)
Let $\mathbf y, \mathbf z$ be two noisy realisations, $\mathbb E[\mathbf y|\mathbf x] = \mathbf x$, $\mathbb E[\mathbf z|\mathbf x] = \mathbf x + \boldsymbol\varepsilon$, $\mathbf z$ variance $\sigma^2_{\mathbf z}$. Then:
$$
\boxed{\mathbb E\|f_\theta(\mathbf y) - \mathbf z\|^2 = \mathbb E\|f_\theta(\mathbf y) - \mathbf x\|^2 - \sigma^2_{\mathbf z} + 2\,\mathbb E\langle f_\theta(\mathbf y) - \mathbf x, \boldsymbol\varepsilon\rangle}
$$
- 1st RHS: supervised loss (target).
- 2nd RHS: noise variance, constant.
- 3rd RHS: bias-induced cross term — vanishes when $\boldsymbol\varepsilon \to 0$.

### 4.3 Total training loss / 학습 손실 (Eq. 7)
$$
\boxed{\mathcal L = \underbrace{\|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y)\|^2_2}_{\mathcal L_{\text{rec}}} + \gamma \underbrace{\|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y) - (g_1(f_\theta(\mathbf y)) - g_2(f_\theta(\mathbf y)))\|^2_2}_{\mathcal L_{\text{reg}}}}
$$
**Term-by-term**:
- $\mathcal L_{\text{rec}}$: standard N2N loss on sub-sampled pair.
- $\mathcal L_{\text{reg}}$: consistency between input-side sub-sample diff and output-side sub-sample diff. With *gradient stopped* on $g_1(f_\theta(\mathbf y))$, $g_2(f_\theta(\mathbf y))$.
- $\gamma$: trade-off (smoothness vs noise). $\gamma = 2$ (synthetic), $\gamma = 1$ (real).

### 4.4 Optimal denoiser constraint / 최적 denoiser 제약 (Eq. 5)
For ideal $f^*_\theta(\mathbf y) = \mathbf x$:
$$
\mathbb E_{\mathbf y|\mathbf x}\bigl[f^*_\theta(g_1(\mathbf y)) - g_2(\mathbf y) - g_1(f^*_\theta(\mathbf y)) + g_2(f^*_\theta(\mathbf y))\bigr] = 0
$$
i.e., $f^*$ commutes (in expectation) with the sub-sampling operation.

### 4.5 Worked numerical example / 수치 예시
SIDD raw-RGB benchmark: noisy PSNR ~30 dB, clean ~50 dB. Take 2×2 cell of clean image $\mathbf x$ with values $[100, 102, 100, 103]$ (mild gradient). Random sub-sampler picks $(100, 102)$ for $(g_1, g_2)$:
- True signal gap: $g_2(\mathbf x) - g_1(\mathbf x) = 2$ — non-zero.
- Add Gaussian noise $\sigma = 5$: $g_1(\mathbf y) \approx 103, g_2(\mathbf y) \approx 99$ (noise $\pm 5$).
- Empirical $\boldsymbol\varepsilon$ (over many cells): mean $\approx 2$, std $\approx \sqrt 2 \cdot \sigma$.

Without regulariser: $\arg\min_\theta \mathbb E\|f - g_2\|^2 \to f \approx \mathbf x + \boldsymbol\varepsilon/2$ — biased toward $g_2$ by half the gap. **Visible over-smoothing in textured regions**.

With regulariser at $\gamma=2$: $\mathcal L_{\text{reg}}$ penalises this average shift; recovers PSNR from 31.67 (γ=0) to 32.10 (γ=2) on Kodak Gauss σ∈[5,50] (Table 3).

$\Delta\text{PSNR} = 0.43$ dB attributable purely to bias removal.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
2005  Buades NLM (#4) — exploit spatial self-similarity (patch-level)
                  ↓
2018  Lehtinen Noise2Noise (#16) — clean GT unnecessary, noisy pairs
                  ↓
2019  Krull Noise2Void (#17) — single-image, blind-spot CNN
                  ↓
2019  Batson Noise2Self (#18) — J-invariance theory
                  ↓
2019  Laine et al. — high-quality blind-spot with probabilistic post-processing
                  ↓
2020  Quan Self2Self (#19) — Bernoulli + dropout ensemble, beats BM3D
                  ↓
2020  Moran Noisier2Noise / Pang R2R — synthesised pairs with known noise model
                  ↓
2021  Huang Neighbor2Neighbor (★) — neighbour sub-sampling + reg, NOISE-MODEL-FREE
                  ↓
2022  Wang Blind2Unblind (#22) — visible blind spots, current SOTA
                  ↓
2023+ Self-supervised diffusion (Ambient #29) — generative version
```

**위치 / Position**: Nb2Nb는 2018-2021의 self-supervised denoising race에서 **state-of-the-art**를 차지. N2N의 단순함 + N2V/N2S의 단일 영상 조건 + S2S의 BM3D 능가 + 추가로 *noise-model-free*. SIDD에서 supervised 능가하며 self-supervised denoising의 *practical 성숙*을 입증.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문                     | Connection / 연결                                                                                          |
|----------------------------------|----------------------------------------------------------------------------------------------------------|
| #16 Lehtinen 2018 (Noise2Noise)  | Loss form 동일; Nb2Nb는 N2N pair를 *single image의 sub-sampling*으로 합성. Theorem 1은 Lehtinen 정리의 generalisation ($\boldsymbol\varepsilon \neq 0$ case). |
| #17 Krull 2019 (Noise2Void)      | 같은 single-image regime, 다른 메커니즘. N2V는 architecture-side blind spot, Nb2Nb는 data-side sub-sampling. |
| #18 Batson 2019 (Noise2Self)     | J-invariance의 다른 instantiation으로 볼 수 있음 — $g_1$과 $g_2$가 disjoint pixel sets이면 partition $\mathcal J = \{g_1, g_2\}$. |
| #19 Quan 2020 (Self2Self)        | S2S는 Bernoulli random masking + dropout ensemble; Nb2Nb는 spatially-structured 2-pixel-cell sub-sampling. Inference 측면에서 Nb2Nb가 훨씬 빠름 (single pass vs 50 passes). |
| #4  Buades 2005 (NLM)            | NLM의 spatial-similarity prior를 deep version으로 일반화 — patch가 아닌 *adjacent pixel*을 직접 활용.        |
| Laine et al. 2019                | Probabilistic post-processing 방식; noise model 가정 필요. Nb2Nb는 noise-model-free → real-world에서 robust. |
| Probabilistic-N2V (Krull 2019)   | noise model을 학습; Nb2Nb는 회피.                                                                         |
| #22 Wang 2022 (Blind2Unblind)    | Nb2Nb 직후의 SOTA 후보; visible blind spot으로 information loss 회피. Nb2Nb는 sub-sampling으로 같은 문제 우회. |

---

## 7. References / 참고문헌

- **Primary**: Huang, T., Li, S., Jia, X., Lu, H., Liu, J. "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images". CVPR 2021, pp. 14781–14790. DOI: 10.1109/CVPR46437.2021.01454.
- **Code**: https://github.com/TaoHuang2018/Neighbor2Neighbor
- **Related**:
  - Lehtinen, J. et al. "Noise2Noise: Learning image restoration without clean data". ICML 2018.
  - Krull, A., Buchholz, T.-O., Jug, F. "Noise2Void". CVPR 2019.
  - Batson, J., Royer, L. "Noise2Self: Blind denoising by self-supervision". ICML 2019.
  - Laine, S., Karras, T., Lehtinen, J., Aila, T. "High-quality self-supervised deep image denoising". NeurIPS 2019.
  - Quan, Y. et al. "Self2Self with Dropout". CVPR 2020.
  - Krull, A. et al. "Probabilistic Noise2Void". arXiv:1906.00651, 2019.
  - Wu, X. et al. "DBSN: Unsupervised learning for image denoising based on dilated blind-spot network". 2020.
  - Abdelhamed, A., Lin, S., Brown, M.S. "A high-quality denoising dataset for smartphone cameras (SIDD)". CVPR 2018.
  - Dabov, K. et al. "Image denoising by sparse 3-D transform-domain collaborative filtering" (BM3D). IEEE TIP 16(8), 2007.
  - Ronneberger, O., Fischer, P., Brox, T. "U-Net". MICCAI 2015.
