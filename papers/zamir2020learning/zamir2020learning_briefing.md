---
title: "Pre-Reading Briefing: Learning Enriched Features for Real Image Restoration and Enhancement (MIRNet)"
paper_id: "42_zamir_2020"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# MIRNet: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H., & Shao, L. (2020). Learning Enriched Features for Real Image Restoration and Enhancement. *ECCV 2020*, LNCS 12370, pp. 492–511. DOI: 10.1007/978-3-030-58595-2_30
**Author(s)**: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

이 논문은 노이즈 제거(denoising), 초해상(super-resolution), 저조도 향상(image enhancement) 등 다양한 이미지 복원 작업을 단일 백본으로 처리하는 **MIRNet**을 제안한다. 핵심 아이디어는 (1) 네트워크 전체에 걸쳐 **고해상도 표현을 유지**하면서 (2) **다중 해상도 병렬 스트림**을 통해 풍부한 문맥(context)을 받아들이고, (3) 두 흐름을 **선택적 커널(Selective Kernel) attention**으로 융합하는 것이다. 결과적으로 5개 실제 이미지 벤치마크에서 SOTA를 달성한다.

This paper proposes **MIRNet**, a unified backbone that handles diverse image-restoration tasks — denoising, super-resolution, and low-light enhancement — with a single architecture. The key idea is to (i) **preserve high-resolution representations** throughout the network, (ii) inject rich **context via parallel multi-resolution streams**, and (iii) fuse the two via **Selective Kernel Feature Fusion (SKFF)** based on self-attention. The model achieves SOTA on five real-image benchmarks.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2020년 시점, 이미지 복원 CNN은 두 갈래로 나뉘어 있었다: U-Net 류의 **encoder-decoder**는 저해상도 표현으로 내려가 큰 receptive field를 얻는 대신 공간 디테일을 잃었고, EDSR/RIDNet 같은 **단일 해상도(full-resolution)** 모델은 디테일은 보존했으나 큰 문맥을 놓쳤다. 한편 HRNet(Sun+ 2019)이 자세 추정에서 다중 해상도 병렬 처리의 위력을 보여주면서, 이 디자인 패턴을 low-level vision에 가져오는 것이 자연스러운 다음 단계였다.

By 2020, restoration CNNs split into two camps: **encoder-decoders** (UNet, DnCNN-derived) gained large receptive fields by downsampling but lost fine spatial detail; **single-resolution** networks (EDSR, RIDNet) preserved detail but had limited context. Meanwhile HRNet (Sun et al., 2019) had shown that parallel multi-resolution streams excel at pose estimation. Bringing that design to low-level vision was the obvious next move.

### 타임라인 / Timeline

```
2009 BM3D (Dabov)      classical denoising baseline / 고전 베이스라인
2017 DnCNN (Zhang)     first major deep denoiser   / 첫 본격 심층 디노이저
2017 SRResNet/EDSR     residual SR backbone         / 잔차 기반 SR
2018 RCAN (Zhang)      channel attention for SR     / 채널 attention SR
2018 RIDNet (Anwar)    full-resolution real denoise / 전해상도 실제 잡음
2019 HRNet (Sun)       parallel multi-res for pose  / 자세용 다중해상도
2019 SKNet (Li)        Selective Kernel attention   / 선택적 커널 attention
→ 2020 MIRNet          unified IR backbone (ECCV)   / 통합 복원 백본
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Residual learning** (He+ 2016): skip connection / 잔차 연결
- **Encoder-decoder & U-Net** (Ronneberger+ 2015): 다운/업샘플링 구조
- **Attention mechanisms**: channel attention (SE/RCAN), spatial attention (CBAM)
- **Selective Kernel Networks** (Li+ 2019): 다중 분기를 self-attention으로 동적 융합
- **Charbonnier loss**: $\sqrt{x^2 + \varepsilon^2}$, L1보다 미분 가능하고 outlier에 robust
- **PSNR / SSIM**: 복원 품질 정량 지표
- **Bilinear upsampling, anti-aliasing downsampling** (Zhang 2019)

Familiarity with residual networks, encoder-decoder design, channel/spatial attention (SE, CBAM), Selective Kernel Networks (SKNet), and standard restoration losses (L1, Charbonnier) and metrics (PSNR/SSIM).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| MRB (Multi-scale Residual Block) | 3개 병렬 해상도 스트림(1×, 1/2×, 1/4×)을 가진 핵심 블록 / Core block with 3 parallel resolution streams |
| RRG (Recursive Residual Group) | 여러 MRB를 묶은 그룹, 외곽 skip 포함 / Group of MRBs with outer skip |
| SKFF (Selective Kernel Feature Fusion) | 다중 스케일 feature를 self-attention으로 합산 / Sums multi-scale features via self-attention |
| DAU (Dual Attention Unit) | Channel attention + Spatial attention 병렬 결합 / CA in parallel with SA |
| Channel attention (CA) | GAP → 1×1 conv → sigmoid → 채널별 가중 / Channel-wise gating via GAP |
| Spatial attention (SA) | GAP+GMP across channels → conv → sigmoid → 픽셀별 가중 / Pixel-wise gating |
| Charbonnier loss | $\sqrt{\|\hat I - I^*\|^2 + \varepsilon^2}$, robust L1 근사 / robust L1 surrogate |
| Anti-aliasing downsampling | Blur kernel + stride로 aliasing 억제 (Zhang 2019) / Blurred-stride to suppress aliasing |
| Residual resizing module | 학습 가능한 다운/업샘플 모듈 / Learnable down/up-sample block |
| Real noise (DND/SIDD) | 합성이 아닌 실제 카메라 잡음 데이터셋 / Real-camera noise benchmark |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Charbonnier loss / 샤르보니에 손실:**

$$
\mathcal{L}(\hat I, I^*) = \sqrt{\|\hat I - I^*\|^2 + \varepsilon^2}, \qquad \varepsilon = 10^{-3}.
$$

L1 norm의 미분가능 근사로, 0 부근에서도 매끄러운 그래디언트를 준다. / A smooth, differentiable surrogate for L1 that remains stable near zero.

**(2) SKFF fuse / 융합 단계:**

$$
\mathbf L = \mathbf L_1 + \mathbf L_2 + \mathbf L_3, \qquad \mathbf s = \mathrm{GAP}(\mathbf L) \in \mathbb R^{1\times1\times C}.
$$

세 해상도 스트림을 element-wise 합산 후 GAP로 채널 통계를 추출한다. / Sums three resolution streams and reduces to per-channel statistics via global pooling.

**(3) SKFF select / 선택 단계:**

$$
\mathbf U = \mathbf s_1 \cdot \mathbf L_1 + \mathbf s_2 \cdot \mathbf L_2 + \mathbf s_3 \cdot \mathbf L_3, \qquad \sum_k \mathbf s_k = 1 \text{ (softmax)}.
$$

Softmax 게이팅으로 각 스트림에 대한 동적 가중치를 만든다. / Softmax gating produces dynamic per-stream weights.

**(4) Restoration as residual / 잔차 복원:**

$$
\hat I = I + \mathbf R, \qquad \mathbf R = f_\theta(I).
$$

네트워크는 잡음/오염분만 학습한다. / The network learns only the residual corruption.

**(5) Channel attention (SE-style):**

$$
\hat{\mathbf d} = \sigma(W_2\,\mathrm{ReLU}(W_1\,\mathrm{GAP}(\mathbf M))), \qquad \mathbf M' = \hat{\mathbf d} \odot \mathbf M.
$$

Squeeze-and-excitation 스타일 채널 게이팅. / Squeeze-and-excitation channel gating.

---

## 6. 읽기 가이드 / Reading Guide

1. **Sec 1–2 (서론·관련 연구, p.1–4)**: 두 가지 기존 디자인의 trade-off를 명확히 짚는다 — 빠르게 훑어도 좋다.
2. **Sec 3 + Fig 1–4 (제안 방법, p.4–7)**: 가장 중요한 부분. MRB, SKFF, DAU, residual resize를 그림과 함께 천천히 읽는다.
3. **Sec 4 (실험, p.8–11)**: 표 1–3과 ablation 결과를 확인하라. 특히 `concat vs sum vs SKFF`, `DAU 유무`, `# of MRB` ablation은 디자인 의사결정을 정당화한다.
4. **Supplementary**: 저조도(LoL, MIT-Adobe FiveK) 결과와 추가 visual 비교가 있다.

Read sections 1-2 quickly for context; spend most time on Section 3 (Figs 1-4) which defines MRB, SKFF, DAU, and residual resizing. Section 4 has the ablations that justify the design choices.

---

## 7. 현대적 의의 / Modern Significance

MIRNet은 이후 **MIRNetv2** (TPAMI 2022)와 **MPRNet, Restormer, Uformer** 등 multi-stage / Transformer 기반 복원 모델의 직접적 사촌이 된다. "고해상도를 유지하면서 다중 스케일 문맥을 attention으로 융합한다"는 디자인 철학은 현대 복원 백본의 표준 레시피가 되었고, MIRNet은 그 결정적 통합 사례로 인용된다.

MIRNet became the direct ancestor of **MIRNetv2** and a sibling to **MPRNet, Restormer, Uformer** — all sharing the philosophy of *preserve high-res spatial detail while injecting multi-scale context via attention-based fusion*. That recipe is now the de-facto template for image-restoration backbones, and MIRNet is the canonical citation that unified it.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
