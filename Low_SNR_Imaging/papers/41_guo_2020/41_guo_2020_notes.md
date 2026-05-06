---
title: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE)"
authors: Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, Runmin Cong
year: 2020
journal: "IEEE/CVF CVPR 2020, pp. 1780-1789"
doi: "10.1109/CVPR42600.2020.00185"
topic: Low_SNR_Imaging
tags: [deep-learning, low-light, zero-reference, curve-estimation, non-reference-loss, self-supervised]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 41. Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE) / 저조도 영상 향상을 위한 영-참조 심층 곡선 추정

---

## 1. Core Contribution / 핵심 기여

Zero-DCE는 **paired·unpaired 데이터 모두 없이** 저조도 영상 향상을 학습하는 최초의 방법이다. 핵심은 enhancement를 **영상별 비선형 곡선 추정 문제**로 재구성하는 것이다. 가벼운 7-layer CNN인 **DCE-Net**이 픽셀별·고차 light-enhancement curve의 매개변수 맵 $\mathcal{A}_n \in \mathbb{R}^{H\times W \times 3}$ ($n=1, \dots, 8$)을 예측하고, 단순한 $LE_n(\mathbf{x}) = LE_{n-1}(\mathbf{x}) + \mathcal{A}_n(\mathbf{x}) LE_{n-1}(\mathbf{x})(1-LE_{n-1}(\mathbf{x}))$를 8회 반복 적용하여 영상을 점진적으로 밝힌다. 학습은 **네 가지 비참조 손실** — spatial consistency, exposure control, color constancy, illumination smoothness — 만으로 이루어지며, 정답 영상을 단 한 장도 사용하지 않는다.

Zero-DCE is the first method to train a low-light enhancement network **without any reference images** (neither paired nor unpaired). It recasts enhancement as **image-specific nonlinear curve estimation**: a lightweight 7-layer CNN (DCE-Net) predicts pixel-wise, high-order curve parameter maps $\mathcal{A}_n \in \mathbb{R}^{H\times W \times 3}$ for $n = 1, \dots, 8$ iterations of the simple update $LE_n(\mathbf{x}) = LE_{n-1}(\mathbf{x}) + \mathcal{A}_n(\mathbf{x}) LE_{n-1}(\mathbf{x})(1-LE_{n-1}(\mathbf{x}))$. Training is driven entirely by **four non-reference losses** — spatial consistency, exposure control, color constancy, and illumination smoothness — so no reference image is ever needed.

DCE-Net의 크기는 단 79,416 매개변수, 5.21G FLOPs (256×256×3 입력)에 불과하며 GPU에서 ∼500 FPS로 동작해 모바일·임베디드 배포에 적합하다. 다섯 개의 표준 데이터셋 (NPE, LIME, MEF, DICM, VV) 과 SICE Part2 paired set, DARK FACE 얼굴 검출 benchmark에서 SRIE, LIME, RetinexNet, EnlightenGAN, Wang et al. 같은 최신 방법들을 시각적·정량적·다운스트림 task에서 모두 능가하거나 동등한 성능을 보였다.

DCE-Net has only 79,416 parameters and 5.21 G FLOPs (for 256×256×3 input), running at ~500 FPS on GPU and easily deployable on mobile / embedded devices. On five standard low-light datasets (NPE, LIME, MEF, DICM, VV), the SICE Part2 paired set, and the DARK FACE detection benchmark, Zero-DCE matches or beats SRIE, LIME, RetinexNet, EnlightenGAN, and Wang et al. visually, quantitatively (PSNR / SSIM / MAE / Perceptual Index / User Study), and on downstream face detection.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Contributions / 도입 (Section 1, p. 1)

- 저조도 사진은 노출 부족, 비균일 조명 (back-light) 등으로 미적·정보 손실. / Low-light photos suffer aesthetic and information loss.
- 기존 deep-learning 모델들은 paired (LLNet, RetinexNet) 또는 unpaired (EnlightenGAN) 데이터에 의존 — 모두 **데이터 수집 비용** 이 큼.
- Zero-DCE의 세 가지 기여:
  1. paired/unpaired 둘 다 불필요한 first low-light enhancement network. 일반화가 잘 됨.
  2. **image-specific curve** 설계 — 픽셀별 단조 매핑으로 wide dynamic range 커버.
  3. reference 없이도 향상 품질 학습이 가능함을 **non-reference task-specific losses** 로 보임.

### Part II: Related Work / 관련 연구 (Section 2, pp. 1-2)

**Conventional methods**: HE (global/local), Retinex-based (LIME, SRIE, Fu et al., Guo et al.). 모두 hand-crafted 가정에 의존, generalisation 한계.

**Data-driven CNN-based**: LLNet (synthetic pairs), MIT-Adobe FiveK retouched, LOL paired (RetinexNet), Wang et al. (paired with retouching experts) — 모두 paired data 필요.

**GAN-based**: EnlightenGAN (unpaired), discriminator + adversarial loss — 학습 불안정, careful curation 필요.

Zero-DCE의 차별점: **purely data-driven (CNN)** + **zero reference** + **lightweight**.

### Part III: Methodology / 방법론 (Section 3, pp. 2-4)

**3.1 Light-Enhancement Curve (LE-curve)** — 곡선의 세 가지 설계 요건:
1. 출력이 $[0,1]$ 안에 있어야 함 (overflow 방지).
2. 단조 증가 (인접 픽셀 contrast 보존).
3. 단순 + 미분 가능 (역전파 가능).

이를 만족하는 가장 단순한 형태는 quadratic curve:

$$
LE(I(\mathbf{x}); \alpha) = I(\mathbf{x}) + \alpha \, I(\mathbf{x}) \big(1 - I(\mathbf{x})\big), \quad \alpha \in [-1, 1]
$$

각 픽셀은 $[0,1]$로 정규화된 후 모든 연산이 픽셀 단위로 적용된다. **세 RGB 채널 각각**에 별도 곡선 적용 (단일 illumination 채널이 아닌 것 — 색 보존을 위해).

**Higher-order curve** — 단일 곡선은 강한 어둠을 펼치기 부족하므로 반복 적용:

$$
LE_n(\mathbf{x}) = LE_{n-1}(\mathbf{x}) + \alpha_n LE_{n-1}(\mathbf{x})\big(1 - LE_{n-1}(\mathbf{x})\big)
$$

논문은 $n = 8$로 설정.

**Pixel-wise curve** — global $\alpha$ 대신 각 픽셀이 자체 $\alpha$를 가지도록 **parameter map** $\mathcal{A}_n(\mathbf{x})$를 도입:

$$
LE_n(\mathbf{x}) = LE_{n-1}(\mathbf{x}) + \mathcal{A}_n(\mathbf{x}) \, LE_{n-1}(\mathbf{x})\big(1 - LE_{n-1}(\mathbf{x})\big)
$$

작은 local region 내 픽셀들은 비슷한 곡선 매개변수를 가진다고 가정 → smoothness 손실로 강제.

**3.2 DCE-Net Architecture** (Section 3.2):

- Plain CNN, **7 conv layers**, 32 channels, 3×3 kernels, stride 1.
- ReLU activation (down-sampling/batch-norm 없음 — 인접 관계 보존).
- **Symmetric concatenation skip connection**: layer pairs (1↔7, 2↔6, 3↔5).
- 마지막 layer Tanh activation → 24 parameter maps (8 iterations × 3 RGB).
- 79,416 trainable params, 5.21 G FLOPs for 256×256×3 입력.

**3.3 Non-Reference Loss Functions** (Section 3.3) — 핵심:

1. **Spatial consistency loss** — enhanced 영상이 입력의 인접 영역 contrast를 보존.
   $$ L_{spa} = \frac{1}{K}\sum_{i=1}^{K} \sum_{j \in \Omega(i)} \big(|Y_i - Y_j| - |I_i - I_j|\big)^2 $$
   $K$ = 4×4 local regions, $\Omega(i)$ = top/down/left/right neighbours.

2. **Exposure control loss** — under/over-exposure 방지, 평균 강도가 $E = 0.6$이 되도록.
   $$ L_{exp} = \frac{1}{M}\sum_{k=1}^{M} |Y_k - E| $$
   $M$ = 16×16 non-overlapping regions.

3. **Color constancy loss** — Gray-World 가정 (Buchsbaum 1980): R, G, B 채널 평균이 같아야 함.
   $$ L_{col} = \sum_{(p,q) \in \varepsilon} (J^p - J^q)^2, \quad \varepsilon = \{(R,G), (R,B), (G,B)\} $$
   $J^c$ = 채널 $c$의 enhanced 영상 평균.

4. **Illumination smoothness loss** — 곡선 매개변수 맵의 부드러움 (TV).
   $$ L_{tv\mathcal{A}} = \frac{1}{N}\sum_{n=1}^{N} \sum_{c \in \xi} (|\nabla_x \mathcal{A}_n^c| + |\nabla_y \mathcal{A}_n^c|)^2 $$
   $N = 8$ iterations, $\xi = \{R, G, B\}$.

**Total loss**:

$$
L_{total} = L_{spa} + L_{exp} + W_{col} L_{col} + W_{tv\mathcal{A}} L_{tv\mathcal{A}}, \quad W_{col} = 0.5, \; W_{tv\mathcal{A}} = 20
$$

### Part IV: Experiments / 실험 (Section 4, pp. 4-7)

**Implementation**: PyTorch, NVIDIA 2080Ti, batch=8, Adam $1e-4$, weights init $\mathcal{N}(0, 0.02^2)$.

**Training data**: SICE Part1 multi-exposure dataset에서 3,022장 임의 분할 → 2,422 train, 600 validation. 512×512 resize. 정상 노출과 과노출 모두 사용 (over-/under-exposed 모두 다룰 수 있도록).

**4.1 Ablation Studies (Fig. 4-6, pp. 5-6)**:

- 네 손실 중 하나라도 제거하면 가시적 결손:
  - $L_{spa}$ 제거 → contrast 감소.
  - $L_{exp}$ 제거 → 어두운 영역 회복 실패.
  - $L_{col}$ 제거 → 강한 color cast.
  - $L_{tv\mathcal{A}}$ 제거 → 인접 영역 간 인공물 (artefacts).
- Layer/feature ablation: 3-32-8 (작은 모델)도 만족스러운 결과 → "zero-reference" 자체가 강력함을 시사.
- Iteration $n$: $n=1$이면 곡률 부족, $n=8$이 시각적 균형.
- Training data ablation: low-light만 사용 (Zero-DCE_Low) → over-enhance, low+over-exposed 다양 사용이 안정적.

**4.2 Benchmark Evaluations (Fig. 7-8, Tables 1-3, pp. 6-8)**:

- 비교 대상: SRIE [8], LIME [9], Li et al. [19], RetinexNet [32], Wang et al. [28], EnlightenGAN [12].
- 데이터셋: NPE (84), LIME (10), MEF (17), DICM (64), VV (24) — 모두 **참조 없는** 저조도 영상셋.
- **User Study (US, 1-5 scale, higher=better)**:
  | Method | Average US ↑ |
  |---|---|
  | SRIE | 3.32 |
  | LIME | 3.59 |
  | Li et al. | 3.37 |
  | RetinexNet | 3.30 |
  | Wang et al. | 3.43 |
  | EnlightenGAN | 3.50 |
  | **Zero-DCE** | **3.81** |
- **Perceptual Index (PI, lower=better)**: Zero-DCE 평균 2.84 (RetinexNet 3.18, EnlightenGAN 3.13).
- **Quantitative on SICE Part2 (Table 2)**:
  | Method | PSNR↑ | SSIM↑ | MAE↓ |
  |---|---|---|---|
  | SRIE | 14.41 | 0.54 | 127.08 |
  | LIME | 16.17 | 0.57 | 108.12 |
  | RetinexNet | 15.99 | 0.53 | 104.81 |
  | EnlightenGAN | 16.21 | 0.59 | 102.78 |
  | **Zero-DCE** | **16.57** | 0.59 | **98.78** |

  ⇒ Zero-DCE가 **paired·unpaired data 없이도** SOTA 동급 또는 그 이상 성능.

- **Runtime (Table 3)**: Zero-DCE 0.0025 s/image (PyTorch GPU) — EnlightenGAN 0.0078, RetinexNet 0.12, LIME 0.49.

**4.2.3 Face Detection in the Dark (Fig. 8, p. 8)**:

DARK FACE dataset에 DSFD baseline detector 적용:
- Raw 영상 AP: 0.231.
- LIME 영상: 0.293, RetinexNet: 0.304, **Zero-DCE: 0.303**.
- 시각적으로 어두운 얼굴 영역에서 선명한 face detection.

### Part V: Conclusion / 결론 (Section 5, p. 8)

- **Contributions 재확인**: zero-reference learning, image-specific curve, lightweight (79K params, 5.21G FLOPs).
- **Future**: semantic information 통합, 잡음 명시 모델링.

---

## 3. Key Takeaways / 핵심 시사점

1. **Reference 없는 학습이 가능하다** / **Reference-free training is possible** — 영상 향상을 잘 정의된 손실 함수의 합으로 표현하면 paired·unpaired 데이터 없이 deep network를 학습할 수 있다. / By framing enhancement as a sum of well-defined losses, deep networks can be trained without paired or unpaired data.

2. **Curve estimation reformulation이 핵심** / **Curve estimation is the key reformulation** — 직접 픽셀 매핑을 출력하지 않고 quadratic curve의 픽셀별 매개변수만 추정 → 단조성, 범위 보존, 미분 가능성이 자동 보장. / Predicting curve parameters (not pixel mappings) automatically guarantees monotonicity, range preservation, differentiability.

3. **반복 적용이 dynamic range를 확장** / **Iterating extends dynamic range** — 단일 곡선은 약한 enhancement만 가능; 8회 반복으로 다양한 조명 조건 대응. / A single curve is too weak; 8 iterations cover diverse lighting.

4. **네 손실의 상호 보완성** / **Four losses are complementary** — 각각이 (a) 구조 보존, (b) 노출 수준, (c) 색 항상성, (d) 부드러움 매개변수 맵을 강제. 어느 하나도 없어선 안 된다 (Fig. 4 ablation). / Each loss covers an independent quality axis (structure, exposure, colour, smoothness); none is dispensable.

5. **Lightweight와 빠른 inference** / **Lightweight and fast inference** — 79K params, 500 FPS는 모바일·임베디드 카메라(스마트폰 야간 모드)에 직접 적용 가능. / 79K params and 500 FPS make it phone-/embedded-deployable.

6. **다운스트림 task에 직접 도움** / **Helps downstream tasks directly** — DARK FACE 얼굴 검출 AP 31% 상대 향상 (0.231 → 0.303) — 단순 시각적 enhancement를 넘어선 활용성. / 31% relative AP improvement on DARK FACE shows enhancement helps downstream models.

7. **Gray-World 가정의 한계** / **Gray-World assumption has limits** — color constancy loss는 mixed-illumination 또는 단색 우세 장면에서 색 왜곡을 만들 수 있음. 천체나 의료 영상에서는 재설계 필요. / The gray-world assumption can distort scenes dominated by one colour; needs redesign for astronomy / medicine.

8. **Self-supervised LLE의 시발점** / **Founding work for self-supervised LLE** — Zero-DCE++, RUAS, SCI 같은 후속 모델들이 모두 zero-reference framework를 확장. / Zero-DCE++, RUAS, SCI all build on this zero-reference framework.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Quadratic LE-curve / 2차 LE-curve

$$
LE(I(\mathbf{x}); \alpha) = I(\mathbf{x}) + \alpha \, I(\mathbf{x}) \big(1 - I(\mathbf{x})\big)
$$

Properties:
- $\alpha = 0$: identity.
- $\alpha > 0$: 어두운 픽셀 ($I < 0.5$) 더 밝아짐 (밝히기).
- $\alpha < 0$: 어두워지기.
- $\alpha \in [-1, 1]$: 출력이 $[0, 1]$ 안에 있음을 보장.
- 연속·미분 가능 ($d LE/dI = 1 + \alpha(1 - 2I)$).

### 4.2 Higher-order pixel-wise curve / 고차 픽셀별 곡선

$$
LE_n(\mathbf{x}) = LE_{n-1}(\mathbf{x}) + \mathcal{A}_n(\mathbf{x}) \, LE_{n-1}(\mathbf{x})\big(1 - LE_{n-1}(\mathbf{x})\big), \quad n = 1, \dots, 8
$$

with $\mathcal{A}_n \in [-1, 1]^{H \times W \times 3}$. Final output = $LE_8(\mathbf{x})$.

### 4.3 Spatial consistency loss / 공간 일관성

$$
L_{spa} = \frac{1}{K}\sum_{i=1}^{K} \sum_{j \in \Omega(i)} \big(|Y_i - Y_j| - |I_i - I_j|\big)^2
$$

- $Y_i$, $I_i$: enhanced와 입력 영상의 region $i$ 평균.
- $\Omega(i)$: 4-connected neighbours.
- 4×4 region 사용 (실험에서 stable).

### 4.4 Exposure control loss / 노출 제어

$$
L_{exp} = \frac{1}{M}\sum_{k=1}^{M} |Y_k - E|, \quad E = 0.6
$$

- $M$: 16×16 non-overlapping regions의 수.
- $E$ ∈ [0.4, 0.7]에서 큰 차이 없음.
- gray level (RGB 평균)에 적용.

### 4.5 Color constancy loss / 색 항상성

$$
L_{col} = \sum_{(p,q) \in \varepsilon} (J^p - J^q)^2, \quad \varepsilon = \{(R,G), (R,B), (G,B)\}
$$

- $J^c$: 채널 $c$의 전체 영상 평균.
- Gray-World 가정 (Buchsbaum 1980).

### 4.6 Illumination smoothness loss / 조명 부드러움

$$
L_{tv\mathcal{A}} = \frac{1}{N}\sum_{n=1}^{N} \sum_{c \in \xi} \big(|\nabla_x \mathcal{A}_n^c| + |\nabla_y \mathcal{A}_n^c|\big)^2
$$

- $\nabla_x, \nabla_y$: horizontal/vertical 차분.
- 인접 픽셀이 비슷한 곡선 매개변수를 갖도록.

### 4.7 Total loss / 총 손실

$$
L_{total} = L_{spa} + L_{exp} + W_{col} L_{col} + W_{tv\mathcal{A}} L_{tv\mathcal{A}}
$$

with $W_{col} = 0.5$, $W_{tv\mathcal{A}} = 20$.

### 4.8 Worked example / 풀이 예제

가정 / Assume: 픽셀 $I = 0.1$ (어두움), $\alpha = 0.8$.

Iteration 1: $LE_1 = 0.1 + 0.8 \times 0.1 \times 0.9 = 0.172$.
Iteration 2: $LE_2 = 0.172 + 0.8 \times 0.172 \times 0.828 = 0.286$.
Iteration 3: $LE_3 = 0.286 + 0.8 \times 0.286 \times 0.714 = 0.449$.
Iteration 4: $LE_4 = 0.449 + 0.8 \times 0.449 \times 0.551 = 0.647$.
...

→ 8 iteration 후 어두운 픽셀이 well-exposed 수준으로 수렴.

For the inverse direction: $\alpha < 0$ darkens bright pixels symmetrically.

### 4.9 Algorithm pseudocode / 의사코드

```
Forward:
   I  ← input low-light image (H, W, 3) in [0,1]
   maps ← DCE-Net(I)  # tensor (H, W, 24); 8 iters × 3 channels
   Y ← I
   for n in 1..8:
       A_n ← maps[:, :, 3(n-1):3n]
       Y ← Y + A_n * Y * (1 - Y)
   return Y

Loss:
   L_spa ← spatial consistency between I and Y (4x4 patches)
   L_exp ← |mean(Y over 16x16 patches) - 0.6|
   L_col ← sum of (mean_R - mean_G)^2, (mean_R - mean_B)^2, (mean_G - mean_B)^2
   L_tvA ← TV(A_1, ..., A_8)
   L_total ← L_spa + L_exp + 0.5 * L_col + 20 * L_tvA

Adam(L_total) for ~50 epochs.
```

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
2011 ── LIME (Guo+)               Retinex illumination estimation
2017 ── LLNet (★ paper #40)       first deep-learning LLE, synthetic pairs
2017 ── MSR-net                   multi-scale Retinex CNN
2018 ── Retinex-Net + LOL         paired Retinex CNN
2018 ── MBLLEN                    multi-branch CNN
2019 ── EnlightenGAN              unpaired GAN (CycleGAN-style)
   ★ 2020 ── Zero-DCE (this paper) zero-reference, image-specific curve
2021 ── Zero-DCE++                10K params, real-time
2022 ── SCI (Self-Calibrated Illumination)
2022 ── URetinex-Net              unfolded Retinex
2023 ── PairLIE, NeRCo            zero-reference + neural rep.
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Lore+ (2017) — LLNet (★ paper #40) | LLNet은 paired synthetic data 필요; Zero-DCE는 그 의존을 제거 / LLNet still needs (synthetic) pairs; Zero-DCE removes that | Very High |
| Wei+ (2018) — Retinex-Net | paired LOL dataset; Zero-DCE는 그 의존을 zero로 / paired dependence Zero-DCE removes | Very High |
| Jiang+ (2019) — EnlightenGAN | unpaired GAN; Zero-DCE는 더 간단·빠름 / unpaired GAN, Zero-DCE simpler & faster | High |
| Li+ (2021) — Zero-DCE++ | 본 논문의 직접 후속 (10K params) / direct follow-up (10K params) | Very High |
| Liu+ (2022) — SCI | self-calibrated illumination, zero-reference 계보 / continues zero-reference lineage | High |
| Buchsbaum (1980) — Gray-World | $L_{col}$의 이론적 근거 / theoretical basis for $L_{col}$ | Medium |
| Wang+ (2014) — TV regularisation | $L_{tv\mathcal{A}}$의 영감 / inspiration for $L_{tv\mathcal{A}}$ | Medium |
| Land (1977) — Retinex theory | enhancement 분야의 이론적 토대 / theoretical foundation of LLE | Medium |

---

## 7. References / 참고문헌

### Additional notes / 추가 노트

**왜 quadratic curve인가?** $LE(I; \alpha) = I + \alpha I (1 - I)$는 (a) $\alpha \in [-1, 1]$이면 $I \in [0,1]$ → $LE \in [0,1]$ 자동 보장, (b) $LE'(I) = 1 + \alpha(1 - 2I) \geq 0$이면 단조성 유지 ($\alpha \in [-1, 1]$에서 항상 성립), (c) 곱셈과 덧셈만으로 미분 가능. 더 복잡한 sigmoid나 tanh를 쓸 수도 있지만 학습 안정성은 quadratic이 최고. 8회 반복으로 비선형성을 충분히 확장 가능.

**Why quadratic curve?** $LE(I; \alpha) = I + \alpha I (1 - I)$ guarantees (a) $LE \in [0,1]$ for $\alpha \in [-1, 1]$, $I \in [0, 1]$; (b) monotonicity since $LE'(I) = 1 + \alpha(1 - 2I) \geq 0$ for $\alpha \in [-1, 1]$; (c) differentiability with only multiplications and additions. More complex sigmoids or tanh could be used but quadratic is most stable to train, and 8 iterations provide enough nonlinearity.

**왜 픽셀별 매개변수?** Global $\alpha$는 전체 영상을 같은 강도로 밝히기 때문에 이미 밝은 영역이 over-exposed될 수 있다. 픽셀별 $\mathcal{A}_n(\mathbf{x})$는 어두운 픽셀은 크게, 밝은 픽셀은 작게 enhancement → uneven illumination 해결.

**Why pixel-wise parameters?** A global $\alpha$ enhances every pixel equally, over-exposing already-bright regions. The pixel-wise $\mathcal{A}_n(\mathbf{x})$ enhances dark pixels more and bright pixels less, naturally addressing uneven illumination.

**$L_{spa}$의 직관**: 입력의 인접 영역 contrast $|I_i - I_j|$가 출력 $|Y_i - Y_j|$에서 그대로 유지되도록 강제 → "**구조 (edge, texture)는 변하지 말고, 평균만 밝아져라**". 이는 over-smoothing이나 detail 손실을 방지.

**Intuition for $L_{spa}$**: forces the contrast between neighbour regions in the output to match the input ("don't change structure, just brighten"). This prevents over-smoothing and detail loss.

**$L_{exp}$ target $E = 0.6$의 선택**: well-exposed grey level은 보통 $0.5 \sim 0.7$. 0.6은 시각적으로 "잘 노출된" 평균 강도. $E \in [0.4, 0.7]$에서 큰 차이 없이 수렴 (논문 Section 3.3에서 확인).

**Why $E = 0.6$?** The well-exposed grey level is typically in $[0.5, 0.7]$; 0.6 is an aesthetically "well-exposed" average. The paper reports robustness for $E \in [0.4, 0.7]$.

**$L_{col}$ Gray-World 가정의 한계**: 단색 우세 장면 (석양, 단색 벽)에서 색 왜곡 가능. 천체 영상이나 의료 영상에는 이 가정이 부적절 → 대체 색 손실 필요. 그래도 자연 사진에는 견고.

**$L_{col}$ Gray-World limitation**: scenes dominated by a single colour (sunset, monochrome wall) can suffer colour distortion. The assumption is unsuitable for astronomical / medical imaging — domain-specific colour losses are needed there. For everyday photos it remains robust.

**$L_{tv\mathcal{A}}$ smoothness의 역할**: 인접 픽셀이 비슷한 enhancement를 받도록 유도 → 인공적인 patchy artefact 방지. weight $W_{tv\mathcal{A}} = 20$으로 다른 손실보다 매우 큼 — TV가 자연스럽게 작은 값이라서 큰 가중치가 필요.

**$L_{tv\mathcal{A}}$ smoothness role**: encourages neighbour pixels to receive similar enhancement, suppressing patchy artefacts. The weight $W_{tv\mathcal{A}} = 20$ is large because TV magnitudes are naturally small.

### Loss-function summary / 손실 요약

| Loss | Symbol | Region size | Purpose |
|---|---|---|---|
| Spatial consistency | $L_{spa}$ | 4×4 | preserve neighbour-region contrast |
| Exposure control | $L_{exp}$ | 16×16 | drive mean intensity to 0.6 |
| Color constancy | $L_{col}$ | global | gray-world (R=G=B) |
| Illumination smoothness | $L_{tv\mathcal{A}}$ | 1-pixel grad | smooth curve maps |

### References / 참고문헌

- Guo, C., et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement", *CVPR* 2020, pp. 1780-1789. DOI: 10.1109/CVPR42600.2020.00185
- Lore, K. G., Akintayo, A., Sarkar, S., "LLNet", *Pattern Recognition* 61, 650-662 (2017).
- Wei, C., Wang, W., Yang, W., Liu, J., "Deep Retinex Decomposition for Low-Light Enhancement", *BMVC* (2018).
- Jiang, Y., et al., "EnlightenGAN: Deep Light Enhancement without Paired Supervision", *IEEE TIP* (2021).
- Guo, X., Li, Y., Ling, H., "LIME: Low-Light Image Enhancement via Illumination Map Estimation", *IEEE TIP* (2016).
- Fu, X., et al., "A Weighted Variational Model for Simultaneous Reflectance and Illumination Estimation", *CVPR* (2016).
- Li, M., et al., "Structure-revealing Low-light Image Enhancement via Robust Retinex Model", *IEEE TIP* (2018).
- Buchsbaum, G., "A Spatial Processor Model for Object Colour Perception", *J. Franklin Inst.* (1980).
- Wang, S., et al., "Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images", *IEEE TIP* (2013).
- Cai, J., Gu, S., Zhang, L., "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images" (SICE), *IEEE TIP* (2018).
- Li, C., et al., "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation" (Zero-DCE++), *IEEE TPAMI* (2021).
- Liu, R., et al., "Toward Fast, Flexible, and Robust Low-Light Image Enhancement" (SCI), *CVPR* (2022).
