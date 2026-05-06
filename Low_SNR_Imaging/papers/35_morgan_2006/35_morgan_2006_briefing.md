---
title: "Pre-Reading Briefing: The Depiction of Coronal Structure in White-Light Images (NRGF)"
paper_id: "35_morgan_2006"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Normalising Radial Graded Filter (NRGF): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Huw Morgan, Shadia Rifai Habbal, Richard Woo. "The Depiction of Coronal Structure in White Light Images." *Solar Physics*, 236, 263-272 (2006). DOI: 10.1007/s11207-006-0113-6.
**Author(s)**: Huw Morgan (Univ. of Hawaii), Shadia Rifai Habbal (Univ. of Hawaii), Richard Woo (JPL)
**Year**: 2006

---

## 1. 핵심 기여 / Core Contribution

이 논문은 백색광 코로나 영상의 가장 큰 방해 요소인 **반경(거리에 따른) 강도 기울기**를 제거하는 매우 단순하지만 강력한 필터 — **Normalising Radial Graded Filter (NRGF)** — 를 도입한다. NRGF는 각 반경 $r$에서 모든 위치각 $\phi$에 대한 평균과 표준편차를 계산해, 픽셀값을 $(I-\bar{I}_r)/\sigma_r$ 형태로 정규화한다. 이것만으로도 streamer, plume, CME 미세구조가 1.0 $R_\odot$부터 6 $R_\odot$까지 *동시에* 잘 보이게 된다. 또한 LASCO C2 *총 밝기(total brightness)* 영상에서 사용할 수 있는, 시간에 거의 변하지 않는 **편광되지 않은 배경(unpolarized background)** 빼기 절차를 함께 제안한다.

This paper presents the **Normalising Radial Graded Filter (NRGF)** — a deceptively simple yet powerful filter that removes the dominating radial brightness gradient of the white-light corona by *standardising* each height: at every radius $r$ the pixel intensity is replaced by $(I-\bar{I}_r)/\sigma_r$, where $\bar{I}_r$ and $\sigma_r$ are the azimuthal mean and standard deviation. The output reveals streamers, polar plumes and CME fine structure simultaneously over the full $1.0\!-\!6.0\,R_\odot$ field. The paper also introduces a complementary procedure to subtract a **time-stable unpolarized background** from LASCO C2 total-brightness images, enabling the production of NRGF-quality images at the high cadence of total-brightness rather than polarized-brightness data.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

K-corona 밝기는 limb에서 3 $R_\odot$까지 약 $10^4$배 떨어지므로, 어떤 단일 stretch로도 streamer/plume/CME 구조를 동시에 보여줄 수 없다. 1968년 Newkirk & Harvey의 광학 RGF 같은 *하드웨어 RGF*가 있었지만, 디지털 시대의 LASCO 영상은 소프트웨어 후처리로 같은 효과를 내야 했다. 2000년대 초까지 LASCO Quick Look 소프트웨어는 *월/년 평균을 빼고 long-term 평균으로 나누는* 방식이었으나, streamer를 균일한 덩어리로 보이게 만들고 동/서 비대칭을 도입하는 등의 인공물을 갖고 있었다. 본 논문은 그 자리를 *통계적 정규화* 한 줄로 대체했다.

The K-corona brightness drops by ~$10^4$ between 1 $R_\odot$ and 3 $R_\odot$, so no single linear stretch can show streamers, plumes and CME structure together. Hardware RGFs (Newkirk & Harvey 1968) had solved this in eclipse photography, and various unsharp/edge filters had been used digitally, but the LASCO Quick Look pipeline (subtract a monthly mean, divide by a yearly mean) introduced east-west asymmetries and other artifacts. The NRGF replaces all of that with a single normalisation step.

### 타임라인 / Timeline

```
1950  van de Hulst pB observations          ── basic K-corona
1968  Newkirk & Harvey hardware RGF         ── optical/mechanical filter
1995  SOHO/LASCO operations begin           ── routine WL coronagraphy
1999  Guillermier & Koutchmy multi-exposure ── HDR-style coronal compositing
2003  Stenborg & Cobelli wavelet packets    ── multiscale enhancement
2006  *Morgan, Habbal & Woo NRGF (this)*    ── radial standardisation filter
2014  Morgan & Druckmüller MGN              ── multiscale Gaussian normalisation (NRGF descendant)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **K-corona vs F-corona vs stray light**: 백색광 영상의 세 가지 기여 성분. / Three contributors to white-light corona images.
- **편광 밝기 $pB$ vs 총 밝기 $tB$**: pB는 K-corona만 잡지만 빈도가 낮음, tB는 자주 찍히지만 F-corona/stray light가 섞임. / pB is K-corona-only but infrequent; tB is frequent but contaminated.
- **극좌표 변환 $(x,y)\to(r,\phi)$**: NRGF의 핵심은 동심원 환(annulus)에서의 평균/표준편차. / NRGF needs the image in heliocentric polar coordinates.
- **표준화(z-score)**: $z = (x-\mu)/\sigma$ — 통계 기초. / Basic statistical standardisation.
- **LASCO C2** 시야: 2.0–6.0 $R_\odot$, 외부 occulter 코로나그래프. / LASCO C2 field of view.
- **MLSO MKIII/MKIV** 코로나미터 / ground-based pB instruments (1.1–2.4 $R_\odot$).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| NRGF | Normalising Radial Graded Filter — 본 논문이 도입한 필터 / the filter introduced here |
| Radial gradient | 거리에 따른 밝기의 가파른 감소 / steep brightness drop with heliocentric distance |
| pB (polarized brightness) | K-corona 전용 측정량 / K-corona-only measurement |
| tB (total brightness) | 모든 백색광 / all white-light, includes F-corona + stray light |
| Position angle $\phi$ | 태양 중심 기준 방위각 / azimuthal angle around the Sun centre |
| Heliocentric distance $r$ | 태양 중심 기준 반경 / radial distance from Sun centre |
| F-corona | 황도 먼지의 산란광 / dust-scattered (zodiacal) component |
| Stray light | 기기 내 산란광 / instrument-scattered light |
| Unpolarized background | 시간에 거의 안 변하는 배경 ($\langle tB\rangle - \langle pB\rangle$의 장기 평균) / time-stable background = long-term average of (tB-pB) |
| Linear triangulation | 서로 다른 기기(MLSO, EIT, LASCO C2) 영상을 합성할 때 사용 / used to merge multi-instrument composites |
| Streamer | 적도 부근의 밝은 헬멧형 구조 / equatorial bright helmet structure |
| Solar minimum/maximum corona | 활동 극소기/극대기 코로나 / cycle-phase coronae |

---

## 5. 수식 미리보기 / Equations Preview

(1) NRGF processed image (Eq. 1 of paper):

$$
I'(r,\phi) = \frac{I(r,\phi) - \langle I(r)\rangle_{\phi}}{\sigma(r)_{\phi}}
$$

여기서 / where:

$$
\langle I(r)\rangle_{\phi} = \frac{1}{N_\phi}\sum_{\phi} I(r,\phi),\qquad
\sigma(r)_{\phi} = \sqrt{\frac{1}{N_\phi}\sum_{\phi}\big(I(r,\phi)-\langle I(r)\rangle_\phi\big)^2}
$$

(2) Total-brightness background subtraction:

$$
I_{tB}^{\text{corr}}(r,\phi) = I_{tB}(r,\phi) - B_{unpol}(r,\phi)
$$

(3) Long-term unpolarized background construction:

$$
B_{unpol}(r,\phi) = \big\langle\, I_{tB}(r,\phi;t) - I_{pB}(r,\phi;t)\,\big\rangle_{t \in \text{Carrington rotation}}
$$

세 식이 알고리즘 전체이다. (1)은 NRGF 본체, (2)–(3)은 high-cadence tB 영상을 pB-quality로 만들기 위한 보조 단계다. / These three equations are the entire algorithm: (1) is NRGF itself, (2)-(3) bring high-cadence total-brightness images to pB quality.

---

## 6. 읽기 가이드 / Reading Guide

- **Sec. 2 The NRGF (Fig. 1)**: Eq. 1을 직접 손으로 적어보고, Fig. 1(a)→(b)에서 standardisation이 무엇을 하는지 시각적으로 확인. / Reproduce Eq. 1 by hand; eye-ball Fig. 1(a)→(b) to see what z-scoring does to a $pB$ profile.
- **Sec. 3 Background subtraction**: Fig. 3 left에서 unpolarized background가 1997년과 2000년에 거의 같다는 점이 핵심 — 그래서 *시간 평균*이 가능하다. / The fact that unpolarized background varies little over the cycle (Fig. 3 left) is what makes the long-term average viable.
- **Sec. 4 CME case (2001 Jan 7, Fig. 5)**: NRGF + background subtraction의 조합이 *6시간 동안의 CME 발달*을 동일 stretch로 보여준다. / NRGF+bg-subtract reveals 6 h of CME evolution at uniform contrast.
- **Sec. 5 Conclusions**: streamer가 *균일 덩어리가 아니라 filament 다발*임을 강조 — 코로나 자기장 위상학에 대한 함의. / Streamers are filament bundles, not solid blocks — implication for coronal magnetic-field topology.
- **Reading tip**: 알고리즘 자체는 한 줄이지만, *왜 단순한 z-score가 잘 작동하는가*에 대한 통찰이 글 곳곳에 있음. / The algorithm is one line; the insight is in the discussion.

---

## 7. 현대적 의의 / Modern Significance

NRGF는 **가장 단순하고 가장 널리 쓰이는 코로나 영상 향상 필터**가 되었으며, SunPy 같은 표준 라이브러리에 직접 구현되어 있다. 후속작 MGN(Morgan & Druckmüller 2014)은 이 z-score 아이디어를 *다중 스케일 가우시안*으로 일반화했고, 둘은 지금도 SOHO/LASCO, STEREO/SECCHI, Solar Orbiter/Metis, PSP/WISPR, Aditya-L1/VELC 같은 모든 백색광 코로나그래프 데이터의 표준 전처리에 등장한다. 또한 NRGF의 "각 반경에서 통계적으로 정규화한다"는 발상은 머신러닝 시대에 *batch/instance/layer normalisation* 같은 표준 정규화 연산의 천문학적 선조로 회고되기도 한다.

NRGF has become the **most widely used coronal-image enhancement filter** in heliophysics and is implemented in SunPy. Its descendant MGN (Morgan & Druckmüller 2014) generalises the z-score idea to multi-scale Gaussians, and together they form the standard preprocessing for SOHO/LASCO, STEREO/SECCHI, Solar Orbiter/Metis, PSP/WISPR and Aditya-L1/VELC. The "standardise each annulus" idea is, in hindsight, an astronomical ancestor of the per-feature normalisation operators (batch/instance/layer norm) that became central to deep learning.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
