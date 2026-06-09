---
title: "Learning Enriched Features for Real Image Restoration and Enhancement (MIRNet)"
authors: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao
year: 2020
journal: "ECCV 2020 (LNCS 12370), pp. 492-511"
doi: "10.1007/978-3-030-58595-2_30"
topic: Low_SNR_Imaging
tags: [denoising, super-resolution, low-light-enhancement, attention, multi-scale, restoration-backbone]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 42. Learning Enriched Features for Real Image Restoration and Enhancement / 실제 이미지 복원·향상을 위한 풍부한 특징 학습

---

## 1. Core Contribution / 핵심 기여

MIRNet은 denoising, super-resolution, low-light enhancement라는 세 가지 image restoration 과제를 **하나의 통합 백본**으로 처리하는 합성곱 모델이다. 핵심 디자인 원칙은 두 가지 — (a) 네트워크 전체에 걸쳐 **공간적으로 정밀한 고해상도 표현을 유지**하고, (b) 동시에 **여러 저해상도 병렬 스트림**을 통해 풍부한 문맥 정보를 받아들이는 것이다. 이 둘을 잇기 위해 (1) **Multi-scale Residual Block (MRB)** — 3개 해상도(1×, 1/2×, 1/4×) 병렬 conv 스트림, (2) **Selective Kernel Feature Fusion (SKFF)** — self-attention으로 다중 스케일 feature를 동적 가중합, (3) **Dual Attention Unit (DAU)** — channel + spatial attention 병렬 결합, (4) **Residual resizing modules** — anti-aliasing 다운샘플과 bilinear 업샘플로 구성된 학습형 리사이즈 모듈을 제안한다. SIDD/DND/RealSR/LoL/MIT-FiveK의 5개 실제 벤치마크에서 SOTA를 달성했고 (예: SIDD denoising PSNR 39.72 dB, DND 39.88 dB), 단일 학습 데이터(SIDD)로 학습한 모델이 DND에 일반화되는 점도 검증되었다.

MIRNet is a single CNN backbone that handles three image-restoration tasks — denoising, super-resolution, and low-light enhancement — under one architecture. Its two design principles are (a) **maintaining spatially-precise high-resolution representations** throughout the network, and (b) simultaneously **receiving rich contextual information from multiple low-resolution parallel streams**. To bridge the two it introduces (1) the **Multi-scale Residual Block (MRB)** with three parallel resolution streams (1×, 1/2×, 1/4×), (2) **Selective Kernel Feature Fusion (SKFF)** that uses self-attention to dynamically weight multi-scale features, (3) the **Dual Attention Unit (DAU)** combining channel and spatial attention in parallel, and (4) **residual resizing modules** with anti-aliased downsampling and bilinear upsampling. The model achieves state-of-the-art on five real-image benchmarks (SIDD/DND/RealSR/LoL/MIT-FiveK), e.g. 39.72 dB on SIDD denoising and 39.88 dB on DND, and demonstrates strong cross-dataset generalization (SIDD-trained model transfers to DND).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation (pp. 1–3) / 서론과 동기

저자들은 기존 복원 CNN의 두 가지 디자인이 모두 본질적 trade-off를 지닌다고 지적한다 (p.2):

The authors identify a fundamental trade-off in existing restoration CNNs (p.2):

- **Encoder-decoder (UNet, EDSR-deconv 류)**: 점진적으로 해상도를 줄여 큰 receptive field를 얻지만 fine spatial detail이 손실된다.
- **High-resolution single-scale (DnCNN, RIDNet)**: 디테일은 보존하지만 receptive field가 작아 큰 문맥을 못 본다.

저자들의 통찰: **"image restoration is a position-sensitive procedure"** — 픽셀-대-픽셀 대응이 핵심이므로 고해상도를 끝까지 유지해야 하지만, 잡음과 진짜 신호를 구분하려면 큰 문맥도 필요하다 (p.2). 따라서 고해상도 메인 분기 + 저해상도 병렬 분기 + 양방향 정보 교환 + selective kernel attention 융합이라는 통합 디자인을 제안한다 (p.2 마지막 문단).

The authors' insight: **image restoration is a position-sensitive task** — pixel-to-pixel correspondence demands high-resolution preservation, yet distinguishing noise from signal demands large context. Their unified design therefore mixes a high-resolution main branch with low-resolution parallel branches, bidirectional cross-stream exchange, and selective-kernel fusion (last paragraph of p.2).

**Five contributions claimed (p.3):**
1. Multi-scale feature extractor that preserves high-res detail / 고해상도 디테일 보존하는 다중 스케일 추출기
2. Repeated information-exchange across resolutions / 해상도 간 반복 정보 교환
3. Selective-kernel fusion (vs concat/sum) / 단순 concat/sum이 아닌 SK 융합
4. Recursive residual design for very deep nets / 매우 깊은 네트워크를 위한 재귀적 잔차 설계
5. SOTA across five benchmarks for three tasks / 세 작업·다섯 데이터셋 SOTA

### Part II: Related Work (pp. 3–4) / 관련 연구

세 task별로 관련 연구를 정리한다 (p.3–4):

The paper organizes related work by task (pp.3-4):

- **Denoising**: 고전 방법 (NLM, BM3D, KSVD) → 학습형 (DnCNN, FFDNet, CBDNet, RIDNet, VDN). MIRNet은 BM3D 대비 SIDD에서 +14.07 dB 향상.
- **Super-resolution**: SRCNN→VDSR→EDSR→RCAN→LP-KPN. 최근 트렌드는 residual learning + attention.
- **Low-light enhancement**: 고전 histogram equalization, Retinex 기반 → 심층 LLNet, RetinexNet, MBLLEN.

### Part III: Proposed Method — Overall Pipeline (pp. 4–5) / 제안 방법: 전체 파이프라인

전체 흐름은 매우 단순하다 (p.4, Fig 1):

The overall pipeline is straightforward (p.4, Fig 1):

1. 입력 이미지 $\mathbf I \in \mathbb R^{H\times W\times 3}$ 에 conv $3\times3$ → low-level feature $\mathbf X_0 \in \mathbb R^{H\times W\times C}$ ($C=64$)
2. $N$개의 **Recursive Residual Group (RRG)** 직렬 통과 → $\mathbf X_d$
3. conv $3\times3$ → 잔차 이미지 $\mathbf R \in \mathbb R^{H\times W\times 3}$
4. 최종 복원: $\hat{\mathbf I} = \mathbf I + \mathbf R$

학습 손실은 **Charbonnier loss**:

Training uses **Charbonnier loss**:

$$
\mathcal L(\hat{\mathbf I}, \mathbf I^*) = \sqrt{\|\hat{\mathbf I} - \mathbf I^*\|^2 + \varepsilon^2}, \quad \varepsilon = 10^{-3}.
$$

### Part IV: Multi-scale Residual Block (MRB) (pp. 5–7) / 다중 스케일 잔차 블록

MRB는 MIRNet의 심장이다. 3개 fully-convolutional 스트림이 병렬로 흐른다 (p.5):

The MRB is the heart of MIRNet — three fully-convolutional streams in parallel (p.5):

| Stream | Resolution | Channels (실험 설정) |
|---|---|---|
| 1 | $H \times W$ | 64 |
| 2 | $H/2 \times W/2$ | 128 |
| 3 | $H/4 \times W/4$ | 256 |

각 스트림은 자체 DAU를 가지며, **모든 스트림 간 양방향 정보 교환**이 SKFF로 이루어진다. 즉 high-res 스트림은 low-res로부터 문맥을, low-res는 high-res로부터 디테일을 받는다 (p.5 마지막 문단).

Each stream owns its DAU, and **bidirectional cross-stream exchange** is realized by SKFF — high-res receives context from low-res; low-res receives detail from high-res (p.5 last paragraph).

### Part V: Selective Kernel Feature Fusion (SKFF) (pp. 5–6, Fig 2) / 선택적 커널 융합

**SKNet (Li+ 2019)**의 아이디어를 multi-resolution 융합에 적용한 것이다. 두 단계로 동작한다 (p.6):

Adapts **SKNet (Li et al. 2019)** to multi-resolution fusion. Two operators (p.6):

**Fuse step:**

$$
\mathbf L = \mathbf L_1 + \mathbf L_2 + \mathbf L_3, \qquad \mathbf s = \mathrm{GAP}(\mathbf L) \in \mathbb R^{1\times1\times C}.
$$

3개 스트림의 element-wise 합을 만든 뒤 GAP로 채널별 통계 추출. / Element-wise sum of three streams, then GAP for channel statistics.

**Compact descriptor:**

$$
\mathbf z = W_{down}(\mathbf s) \in \mathbb R^{1\times1\times C/r}, \quad r = 8.
$$

차원 축소비 $r=8$로 모든 실험에서 고정. / Reduction ratio fixed at $r=8$.

**Select step:**

$$
\mathbf v_k = W_{up,k}(\mathbf z), \quad \mathbf s_k = \mathrm{softmax}_k(\mathbf v_k), \quad k=1,2,3,
$$

$$
\mathbf U = \mathbf s_1 \cdot \mathbf L_1 + \mathbf s_2 \cdot \mathbf L_2 + \mathbf s_3 \cdot \mathbf L_3.
$$

세 스트림에 대한 softmax 게이팅 가중치를 만들고 가중합한다. SKFF는 단순 concat 대비 **약 6배 적은 파라미터**로 더 좋은 결과를 낸다 (p.6 마지막 문단; ablation Tab 4 참조).

Three softmax-gated weights produce a dynamic weighted sum. SKFF uses **about 6× fewer parameters than concatenation** while achieving better results (p.6, ablation in Tab. 4).

### Part VI: Dual Attention Unit (DAU) (pp. 6–7, Fig 3) / 이중 attention 단위

DAU는 한 스트림 *내부*에서 spatial과 channel 정보를 동시에 정제한다 (p.6–7).

The DAU refines features *within* a single stream along both spatial and channel dimensions (pp.6-7).

**Channel attention (CA), SE-style:**

$$
\mathbf d = \mathrm{GAP}(\mathbf M) \in \mathbb R^{1\times1\times C}, \quad \hat{\mathbf d} = \sigma(W_2\,\mathrm{ReLU}(W_1 \mathbf d)),
$$

$$
\mathbf M_{CA} = \hat{\mathbf d} \odot \mathbf M.
$$

**Spatial attention (SA), CBAM-style:**

$$
\mathbf f = [\mathrm{GAP}_c(\mathbf M); \mathrm{GMP}_c(\mathbf M)] \in \mathbb R^{H\times W \times 2}, \quad \hat{\mathbf f} = \sigma(\mathrm{Conv}(\mathbf f)),
$$

$$
\mathbf M_{SA} = \hat{\mathbf f} \odot \mathbf M.
$$

**병렬 결합:**

$$
\mathbf M' = \mathrm{Conv}_{1\times1}([\mathbf M_{CA}; \mathbf M_{SA}]) + \mathbf M.
$$

CA는 채널별 게이트, SA는 픽셀별 게이트. 병렬 분기를 concat 후 1×1 conv로 합치고 residual 더하기. / CA gates channels, SA gates pixels; parallel branches are concatenated, projected, and residual-added.

### Part VII: Residual Resizing Modules (p. 7, Fig 4) / 잔차 리사이즈 모듈

다운샘플링: anti-aliased downsample (Zhang 2019) 후 conv 1×1로 채널 두 배. 업샘플링: bilinear upsample 후 conv 1×1로 채널 절반. 2× 한 번 적용으로 2× resize, 두 번 연속으로 4× resize (p.7).

Downsampling: anti-aliased downsample (Zhang 2019) followed by 1×1 conv that doubles channels. Upsampling: bilinear upsample then 1×1 conv that halves channels. Stacking once gives 2×; twice gives 4× (p.7).

### Part VIII: Implementation Details (p. 8) / 구현 세부사항

p.8의 표준 학습 설정:

Standard training setup (p.8):

- **RRGs**: $N=3$, 각 RRG 내부 MRB 수 $M=2$
- **MRB streams**: 3 (channels 64/128/256 at 1×, 1/2×, 1/4×)
- **DAUs per stream**: 2
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.999$), $7 \times 10^5$ iterations
- **Learning rate**: $2\times10^{-4}$ → $10^{-6}$ via cosine annealing (Loshchilov 2017)
- **Patch**: 128×128, batch 16, horizontal+vertical flip augmentation
- 세 task별로 독립 학습; pre-training 없음

### Part IX: Image Denoising Results (pp. 8–10, Tabs 1-2, Figs 5-6) / 디노이징 결과

**SIDD** (smartphone, real noise, p.9 Tab 1):

| Method | DnCNN | BM3D | KSVD | RIDNet | VDN | **MIRNet** |
|---|---|---|---|---|---|---|
| PSNR↑ | 23.66 | 25.65 | 26.88 | 38.71 | 39.28 | **39.72** |
| SSIM↑ | 0.583 | 0.685 | 0.842 | 0.914 | 0.909 | **0.959** |

**DND** (DSLR, real noise, p.9 Tab 2):

| Method | BM3D | KSVD | TNRD | RIDNet | VDN | **MIRNet** |
|---|---|---|---|---|---|---|
| PSNR↑ | 34.51 | 36.49 | 33.65 | 39.26 | 39.38 | **39.88** |
| SSIM↑ | 0.851 | 0.898 | 0.831 | 0.953 | 0.952 | **0.956** |

특기 사항: SIDD에서만 학습한 모델이 DND에서도 SOTA — **cross-dataset generalization이 강함** (p.10).

The SIDD-only model also achieves SOTA on DND, evidencing **strong cross-dataset generalization** (p.10).

### Part X: Super-Resolution Results (p. 10, Tab 3) / 초해상 결과

**RealSR** dataset (p.10 Tab 3):

| Scale | Bicubic | VDSR | RCAN | LP-KPN | **MIRNet** |
|---|---|---|---|---|---|
| ×2 | 32.61 / 0.907 | 33.64 / 0.917 | 33.87 / 0.922 | 33.90 / 0.927 | **34.35 / 0.935** |
| ×3 | 29.34 / 0.841 | 30.14 / 0.856 | 30.40 / 0.862 | 30.42 / 0.868 | **31.16 / 0.885** |
| ×4 | 27.99 / 0.806 | 28.63 / 0.821 | 28.88 / 0.826 | 28.92 / 0.834 | **29.14 / 0.843** |

LP-KPN 대비 ×2/×3/×4 각각 +0.45/+0.74/+0.22 dB 향상 (p.11).

### Part XI: Image Enhancement Results (supplementary, brief) / 이미지 향상 결과

LoL과 MIT-Adobe FiveK에서도 RetinexNet, GLADNet, MBLLEN을 능가 (논문 본문 + supplementary). LoL: **24.14 dB / 0.83 SSIM** 수준 보고.

LoL: ~24.14 dB / 0.83 SSIM, beating RetinexNet, GLADNet, MBLLEN (paper + supplementary).

### Part XII: Ablation Study (논문 표 5+ supplementary) / Ablation 연구

핵심 결과 요약:

Key ablation results:

| Component removed | $\Delta$ PSNR (SIDD) |
|---|---|
| SKFF → concat | −0.30 dB |
| SKFF → sum | −0.45 dB |
| DAU 제거 / removed | −0.15 dB |
| Multi-scale → 1 stream | −0.71 dB |
| Information exchange 제거 | −0.22 dB |

→ 다중 해상도 + cross-stream exchange + SKFF + DAU 모두 의미 있게 기여 / All four components contribute.

---

## 3. Key Takeaways / 핵심 시사점

1. **고해상도 보존 + 다중 스케일 문맥의 통합** — Restoration is position-sensitive하므로 고해상도를 끝까지 유지하는 동시에 다중 해상도 분기로 receptive field를 확장하는 것이 핵심이다.
   **Preserve high-resolution while injecting multi-scale context** — restoration is position-sensitive, so high-res must be maintained end-to-end while context is borrowed from low-res streams.

2. **Selective kernel attention이 단순 concat/sum을 압도한다** — softmax로 동적 가중치를 학습하면 6배 적은 파라미터로 +0.3~0.45 dB의 이득을 본다.
   **Selective-kernel attention beats concat/sum** — softmax-gated dynamic weighting yields +0.3 to +0.45 dB with 6× fewer parameters.

3. **Channel × spatial 병렬 attention** — DAU는 SE의 채널 게이트와 CBAM의 공간 게이트를 병렬로 결합해, 둘 다 동시에 정제한다.
   **Parallel CA + SA in DAU** — combines SE-style channel gating and CBAM-style spatial gating simultaneously rather than sequentially.

4. **Charbonnier loss** — L1의 미분가능 근사로 0 부근에서도 안정된 그래디언트, $\varepsilon=10^{-3}$만 잡으면 추가 튜닝 불필요.
   **Charbonnier loss** — a smooth differentiable surrogate for L1 that needs no additional tuning beyond $\varepsilon=10^{-3}$.

5. **Cross-task universality** — denoising/SR/enhancement에 동일 구조가 SOTA. "복원의 본질은 동일하다"는 디자인 가설을 입증한다.
   **One backbone for three tasks** — same architecture is SOTA across denoising/SR/enhancement, validating the design hypothesis that restoration tasks share a common substrate.

6. **Cross-dataset generalization** — SIDD에서만 학습한 모델이 DND에서도 SOTA. multi-scale+attention 구조가 도메인-shift에 robust함을 시사한다.
   **Cross-dataset generalization** — SIDD-only model also tops DND, suggesting multi-scale+attention design is robust to domain shift.

7. **Recursive residual design** — RRG(전역 skip) → MRB(블록 skip) → DAU(내부 skip)의 3중 skip 계층이 매우 깊은 네트워크를 안정적으로 학습 가능하게 한다.
   **Triple-level recursive residuals** — RRG (outer) → MRB (mid) → DAU (inner) skips enable stable training of very deep restoration nets.

8. **Anti-aliased resampling이 중요** — Zhang 2019의 anti-aliased downsampling을 적용해 shift-equivariance를 개선한 것이 미세하지만 일관된 이득을 준다.
   **Anti-aliased resampling matters** — Zhang 2019's anti-aliased downsampling improves shift-equivariance and gives a small but consistent gain.

---

## 4. Mathematical Summary / 수학적 요약

### Loss / 손실

$$
\mathcal L(\hat{\mathbf I}, \mathbf I^*) = \sqrt{\|\hat{\mathbf I} - \mathbf I^*\|^2 + \varepsilon^2}, \quad \varepsilon=10^{-3}.
$$

### Residual restoration / 잔차 복원

$$
\hat{\mathbf I} = \mathbf I + f_\theta(\mathbf I), \qquad f_\theta = \mathrm{Conv}\circ\mathrm{RRG}_N\circ\cdots\circ\mathrm{RRG}_1\circ\mathrm{Conv}.
$$

### SKFF — 다중 해상도 동적 융합

Fuse:
$$
\mathbf L = \sum_{k=1}^K \mathbf L_k, \quad \mathbf s = \mathrm{GAP}(\mathbf L), \quad \mathbf z = W_{down}\,\mathbf s.
$$

Select:
$$
\mathbf v_k = W_{up,k}\,\mathbf z, \quad (\mathbf s_1,\dots,\mathbf s_K) = \mathrm{softmax}(\mathbf v_1,\dots,\mathbf v_K),
$$

$$
\mathbf U = \sum_{k=1}^K \mathbf s_k \cdot \mathbf L_k.
$$

### DAU — 채널·공간 attention 병렬 결합

Channel attention:
$$
\mathbf M_{CA} = \sigma\big(W_2\,\mathrm{ReLU}(W_1\,\mathrm{GAP}(\mathbf M))\big) \odot \mathbf M.
$$

Spatial attention:
$$
\mathbf M_{SA} = \sigma\Big(\mathrm{Conv}\big([\mathrm{GAP}_c(\mathbf M); \mathrm{GMP}_c(\mathbf M)]\big)\Big) \odot \mathbf M.
$$

Combined:
$$
\mathbf M' = \mathbf M + \mathrm{Conv}_{1\times1}\big([\mathbf M_{CA}; \mathbf M_{SA}]\big).
$$

### Residual resizing — 학습형 다운/업샘플

Downsample (2×):
$$
\mathbf X_{\downarrow} = \mathrm{Conv}_{1\times1}\big(\mathrm{AADown}(\mathbf X)\big), \quad C \to 2C.
$$

Upsample (2×):
$$
\mathbf X_{\uparrow} = \mathrm{Conv}_{1\times1}\big(\mathrm{Bilinear}_{\times2}(\mathbf X)\big), \quad 2C \to C.
$$

### Receptive field / 수용 영역

3개 해상도 스트림이 병렬로 동작하므로 가장 거친 스트림의 ERF가 입력 좌표계로 환산하면 약 **4×** 확장된 receptive field에 해당한다 (p.7 부근 시사).

The coarsest of the three streams effectively quadruples the receptive field in the input coordinate system (implied around p.7).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1992 Total Variation (Rudin-Osher-Fatemi)
         ▼ variational denoising / 변분 잡음제거
2007 BM3D (Dabov)
         ▼ classical SOTA / 고전 SOTA
2014 SRCNN (Dong)
         ▼ first deep SR / 첫 심층 SR
2017 DnCNN (Zhang) — residual deep denoiser
2017 EDSR (Lim) — residual SR backbone
         ▼ + attention era / attention 시대
2018 RCAN (Zhang) — channel attention SR
2018 RIDNet (Anwar) — full-resolution real-noise denoising
         ▼ + multi-resolution era / 다중해상도 시대
2019 HRNet (Sun) — parallel multi-res for pose
2019 SKNet (Li) — selective kernel attention
══════════════════════════════════════════════
2020 ★ MIRNet (Zamir) — unified IR backbone (THIS PAPER)
══════════════════════════════════════════════
2021 MPRNet (Zamir) — multi-stage progressive restoration
2021 Uformer / SwinIR — Transformer for restoration
2022 Restormer (Zamir) — efficient transformer backbone
2022 MIRNetv2 (Zamir) — extended journal version (TPAMI)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| HRNet (Sun+ 2019) | 다중 해상도 병렬 처리의 직접적 영감 / Direct inspiration for multi-res streams | High |
| SKNet (Li+ 2019) | SKFF는 SKNet의 multi-resolution 적응형 버전 / SKFF adapts SKNet to multi-resolution fusion | High |
| SE-Net / RCAN (Hu 2018; Zhang 2018) | DAU의 channel attention 분기 / DAU's channel-attention branch | High |
| CBAM (Woo+ 2018) | DAU의 spatial attention 분기 / DAU's spatial-attention branch | High |
| RIDNet (Anwar+ 2019) | Full-resolution 디노이저의 직계 후계자 / Direct successor on full-res denoising | High |
| MPRNet (Zamir+ 2021) | 같은 저자가 만든 multi-stage 후속 / Multi-stage follow-up by same authors | High |
| Restormer (Zamir+ 2022) | Transformer로 같은 철학을 확장 / Transformer extension of same philosophy | Medium |
| EnlightenGAN (Jiang+ 2021, paper #43) | 같은 토픽 (저조도 향상)을 unpaired GAN으로 / Same enhancement task, GAN-unpaired route | Medium |

---

## 7. References / 참고문헌

- Zamir, S. W. et al. "Learning Enriched Features for Real Image Restoration and Enhancement." *ECCV 2020*, LNCS 12370, pp. 492–511. DOI: 10.1007/978-3-030-58595-2_30
- Sun, K. et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." *CVPR 2019*. (HRNet)
- Li, X. et al. "Selective Kernel Networks." *CVPR 2019*. (SKNet)
- Hu, J., Shen, L., Sun, G. "Squeeze-and-Excitation Networks." *CVPR 2018*. (SE)
- Woo, S. et al. "CBAM: Convolutional Block Attention Module." *ECCV 2018*.
- Zhang, Y. et al. "Image Super-Resolution Using Very Deep Residual Channel Attention Networks." *ECCV 2018*. (RCAN)
- Anwar, S., Barnes, N. "Real Image Denoising with Feature Attention." *ICCV 2019*. (RIDNet)
- Abdelhamed, A., Lin, S., Brown, M. S. "A High-Quality Denoising Dataset for Smartphone Cameras." *CVPR 2018*. (SIDD)
- Plötz, T., Roth, S. "Benchmarking Denoising Algorithms with Real Photographs." *CVPR 2017*. (DND)
- Cai, J. et al. "Toward Real-World Single Image Super-Resolution." *ICCV 2019*. (RealSR)
- Wei, C. et al. "Deep Retinex Decomposition for Low-Light Enhancement." *BMVC 2018*. (LoL)
- Zhang, R. "Making Convolutional Networks Shift-Invariant Again." *ICML 2019*. (Anti-aliased downsampling)
- Loshchilov, I., Hutter, F. "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR 2017*.
- Charbonnier, P. et al. "Two Deterministic Half-Quadratic Regularization Algorithms." *ICIP 1994*.
- GitHub: https://github.com/swz30/MIRNet
