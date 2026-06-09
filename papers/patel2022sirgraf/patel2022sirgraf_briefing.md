---
title: "Pre-Reading Briefing: SiRGraF — Simple Radial Gradient Filter for Coronagraph Images"
paper_id: "39_patel_2022"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# SiRGraF: A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images / 사전 읽기 브리핑

**Paper**: Patel, R., Majumdar, S., Pant, V., Banerjee, D., "A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images", *Solar Physics* 297, 27 (2022). DOI: 10.1007/s11207-022-01957-y
**Author(s)**: Ritesh Patel, Satabdwa Majumdar, Vaibhav Pant, Dipankar Banerjee
**Year**: 2022

---

## 1. 핵심 기여 / Core Contribution

본 논문은 백색광 코로나그래프(LASCO/C2, STEREO/COR-1A, COR-2A, MLSO/KCor) 영상의 가파른 동경 강도 기울기(radial intensity gradient)를 빠르게 제거하기 위한 **SiRGraF (Simple Radial Gradient Filter)** 알고리즘을 제안한다. 핵심 절차는 (i) 하루 분량 Level-1 영상의 픽셀별 최소값으로 **minimum background** $I_m$을 만들고, (ii) 그것의 방위각 평균으로 **radial 1D profile**을 얻은 뒤 회전시켜 **uniform background** $I_u$를 생성하고, (iii) 단순히 $I' = (I - I_m)/I_u$를 적용하는 것이다. NRGF/MGN 대비 코드가 간단하고 처리 속도가 빨라서 수백∼수천 장의 코로나그래프 이미지를 일괄(batch) 처리하기에 적합하다.

This paper introduces **SiRGraF (Simple Radial Gradient Filter)**, a fast algorithm that removes the steep radial intensity gradient in white-light coronagraph images (LASCO/C2, STEREO/COR-1A, COR-2A, MLSO/KCor). The pipeline consists of (i) building a **minimum background** $I_m$ from one day of Level-1 images by taking the per-pixel minimum (excluding zeros), (ii) azimuthally averaging $I_m$ to obtain a 1-D radial profile, then rotating it back into a **uniform background** $I_u$, and (iii) applying $I' = (I - I_m)/I_u$. Compared to NRGF or MGN it is simpler, faster, and well suited to batch processing hundreds of coronagraph images for transient (CME) studies.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

LASCO (1995-)와 STEREO/SECCHI (2006-)를 거치며 코로나그래프 데이터는 폭발적으로 증가했고, Parker Solar Probe/WISPR, Solar Orbiter/Metis 같은 후속 임무가 수백만 장 단위의 영상을 추가로 만들어내고 있다. K-corona의 동적 구조를 보려면 F-corona와 기기 산란광 같은 거의 정적인 배경을 제거하고 가파른 $r^{-3}$ 정도의 동경 강도 변화를 평탄화해야 한다. NRGF (Morgan, Habbal, Woo 2006), FNRGF (Druckmüllerová+ 2011), RLMF (Qiang+ 2020), MGN (Morgan & Druckmüller 2014) 등 다양한 필터가 제안되었지만, 일괄 처리 효율과 코드 단순성 면에서 부족한 부분이 있었다.

LASCO (1995–) and STEREO/SECCHI (2006–) generated an avalanche of coronagraph data, and follow-on missions such as Parker Solar Probe/WISPR and Solar Orbiter/Metis are adding millions of frames. Studying the dynamic K-corona requires removing the near-static background (F-corona + instrumental scatter) and flattening the steep ($\sim r^{-3}$) radial intensity drop. Several radial-gradient filters exist — NRGF (Morgan, Habbal & Woo 2006), FNRGF (Druckmüllerová et al. 2011), RLMF (Qiang et al. 2020), MGN (Morgan & Druckmüller 2014) — but each is either computationally heavy or not optimized for batch processing of large archives.

### 타임라인 / Timeline

```
1950 ── van de Hulst: K + F corona decomposition theory
1968 ── Newkirk & Harvey: radial density filter for eclipse
2006 ── Morgan, Habbal & Woo: NRGF (subtract mean / divide by std at each height)
2011 ── Druckmüllerová+: FNRGF (Fourier-based local NRGF)
2014 ── Morgan & Druckmüller: Multi-Scale Gaussian Normalization (MGN)
2020 ── Qiang+: RLMF (Radial Local Multi-Scale Filter)
2022 ── Patel+: SiRGraF (this paper) — minimum + uniform background, batch-friendly
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **K-corona / F-corona 분리**: K-corona는 자유전자에 의한 톰슨 산란, F-corona는 황도면 먼지에 의한 산란. F는 $\gtrsim 2.6 R_\odot$에서 거의 시간 불변. / K-corona = Thomson scattering by free electrons; F-corona = scattering by interplanetary dust, almost time-invariant beyond $\sim 2.6 R_\odot$.
- **Radial intensity profile**: $I(r) \propto r^{-n}$ with $n \approx 2.5\!-\!3$ inside $\sim 5 R_\odot$.
- **Coronagraph instruments**: LASCO/C2 (2-6 $R_\odot$), STEREO/COR-1A (1.4-4 $R_\odot$), COR-2A (2-15 $R_\odot$), MLSO/KCor (1.05-3 $R_\odot$).
- **Existing filters**: NRGF (subtract azimuthal mean, divide by azimuthal std at each $r$); MGN (multi-scale Gaussian).
- **Image processing**: pixel-wise min/median, polar↔Cartesian resampling.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| K-corona | 자유전자 톰슨 산란 성분 — 동적 구조 (CME, streamer) 포함 / Free-electron Thomson-scattered component, carries dynamic structures |
| F-corona | 행성간 먼지 산란 — 거의 정적 / Dust-scattered Fraunhofer corona, near-static |
| Radial gradient filter (RGF) | 동경 방향 강도 변화를 정규화하는 모든 알고리즘 클래스 / Class of algorithms that normalise the radial intensity drop |
| NRGF | Normalising Radial Gradient Filter — 각 $r$에서 평균 빼고 표준편차로 나눔 / subtract azimuthal mean, divide by std at each height |
| Minimum background ($I_m$) | 하루 동안 픽셀별 최솟값 (0 초과)로 만든 배경 / Per-pixel minimum (>0) over one day of frames |
| Uniform background ($I_u$) | $I_m$의 방위각 평균으로 회전 대칭하게 만든 배경 / Azimuthally averaged $I_m$ rotated back into 2-D |
| Level-1 image | flat·dark 보정 + 정렬 + 태양 디스크 정북 정렬 완료된 영상 / calibrated, aligned, north-up image |
| Streamer / CME | 길게 늘어진 자기 닫힌 구조 / explosively erupting plasma with magnetic field |
| Vignetting | 광학계의 외곽 강도 감소 — outer-edge ring artefact의 원인 / optical falloff at field edges, source of outer-edge ring artefacts |
| Cadence | 영상 간격 — KCor 15 s, COR-2A 5 min / image sampling interval |

---

## 5. 수식 미리보기 / Equations Preview

**(1) SiRGraF 핵심 / Core SiRGraF**

$$
I' = \frac{I - I_m}{I_u}
$$

$I$ = 원본 Level-1 영상, $I_m$ = 하루 minimum background, $I_u$ = $I_m$에서 만든 uniform (radial) background. 분자에서 정적·준정적 배경(F-corona + 산란광)을 빼고, 분모로 동경 강도 변화를 정규화한다.

$I$ = original Level-1 image; $I_m$ = one-day minimum background; $I_u$ = uniform radial background built from $I_m$. The numerator removes static / quasi-static background, the denominator normalises the radial brightness drop.

**(2) Minimum background / 최소 배경**

$$
I_m(x,y) = \min_{t \in \text{day}, \; I_t(x,y) > 0} I_t(x,y)
$$

**(3) Uniform background / 균일 배경 (방위각 평균)**

$$
I_u(r, \phi) = \langle I_m(r, \phi') \rangle_{\phi'}
$$

방위각 $\phi'$ 전체에 대한 평균 — 결과는 $\phi$ 방향으로 일정한 회전 대칭 배경.

Mean over all azimuthal angles $\phi'$, yielding a rotation-symmetric background.

**(4) NRGF (비교용 / for comparison)**

$$
I'_{NRGF}(r, \phi) = \frac{I(r,\phi) - \langle I(r,\phi')\rangle_{\phi'}}{\sigma_\phi(I(r,\phi'))}
$$

각 height $r$에서 평균을 빼고 표준편차로 나눔.

Subtract azimuthal mean and divide by azimuthal standard deviation at each height $r$.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1 (Introduction)**: K/F-corona 분리 역사, NRGF/MGN/RLMF/CIISCO 등 선행 필터 소개. / History of K/F separation and prior filters.
- **Section 2 (Algorithm)**: 4-단계 절차 (i, ii, iii, iv) — Figure 1이 가장 중요. $I_m$, $I_u$, $I'$의 시각화. / The 4-step recipe; Fig. 1 illustrates each stage.
- **Section 3 (Results)**: LASCO-C2, COR-2A, KCor 적용 결과 + NRGF 직접 비교 + 7-day extended background 실험. / Application to multiple coronagraphs and direct NRGF comparison.
- **Section 4 (Discussion)**: 한계 — 정량 분석에는 부적합 (intensity 정보 잃음), 지상 KCor에서는 streamer 손실. / Limitations: not for quantitative photometry, ground-based KCor loses streamers.

읽으면서 확인할 질문 / Questions to keep in mind:
1. 왜 daily median이 아니라 daily minimum인가? (CME contamination 방지) / Why daily minimum instead of median?
2. $I_u$를 만드는 데 azimuthal mean 대신 median을 쓰면 어떻게 달라질까? / What changes if we use median instead of mean for $I_u$?
3. Extended-period background (7 days)와 1-day의 trade-off는? / What is the trade-off between 1-day and 7-day backgrounds?

---

## 7. 현대적 의의 / Modern Significance

SiRGraF는 PSP/WISPR, Solar Orbiter/Metis, ASO-S/LST, ADITYA-L1/VELC 같은 대용량 코로나그래프 미션의 실시간 또는 준실시간 CME 검출 파이프라인에 적합한 가벼운 전처리 단계로 사용될 수 있다. 머신러닝 기반 CME segmentation (예: CMEs Identification in Inner Solar Corona, CIISCO; deep-learning CACTus 후속 모델)의 입력 정규화 단계로도 가치가 있다 — 단순한 식과 적은 하이퍼파라미터 덕분에 도메인 적응 부담이 작다.

SiRGraF is well-positioned as a lightweight preprocessing stage in real-time or near-real-time CME detection pipelines for high-throughput coronagraph missions (PSP/WISPR, Solar Orbiter/Metis, ASO-S/LST, ADITYA-L1/VELC). Its simplicity (one division, one subtraction, almost no hyperparameters) makes it ideal as an input-normalisation step ahead of ML-based CME segmentation models (e.g., CIISCO, deep-learning successors of CACTus), where domain adaptation cost matters.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
