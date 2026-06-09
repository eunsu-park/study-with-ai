---
title: "SiRGraF: A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images"
authors: Ritesh Patel, Satabdwa Majumdar, Vaibhav Pant, Dipankar Banerjee
year: 2022
journal: "Solar Physics, Vol. 297, 27"
doi: "10.1007/s11207-022-01957-y"
topic: Low_SNR_Imaging
tags: [coronagraph, radial-gradient-filter, batch-processing, CME, K-corona, F-corona, image-enhancement]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 39. SiRGraF — A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images / 코로나그래프 영상 배치 처리를 위한 단순 동경 기울기 필터

---

## 1. Core Contribution / 핵심 기여

SiRGraF는 백색광(white-light) 코로나그래프 영상의 가파른 동경 강도 변화를 **두 번의 산술 연산** ($I' = (I - I_m)/I_u$)만으로 평탄화하는 매우 단순한 필터이다. 핵심 아이디어는 두 가지다. 첫째, 하루 분량(또는 며칠치) 영상의 **픽셀별 최솟값** $I_m$을 만들어 정적인 F-corona, 기기 산란광, 그리고 거의 정적인 K-corona 성분을 동시에 제거한다. 둘째, 이 minimum background를 방위각 평균으로 회전 대칭화한 **uniform background** $I_u$로 결과를 나누어 동경 강도 정규화를 수행한다. 결과적으로 K-corona의 동적 성분(CME, 단주기 streamer) 만 두드러지는 정규화된 영상이 만들어진다.

SiRGraF reduces the steep radial intensity drop of white-light coronagraph images using just **two arithmetic operations** ($I' = (I - I_m)/I_u$). The two ideas are: (i) build a **per-pixel minimum** $I_m$ over a day (or several days) of frames, which removes the static F-corona, instrumental scatter, and the slowly varying K-corona; and (ii) divide the residual by a rotationally symmetric **uniform background** $I_u$ obtained by azimuthally averaging $I_m$. The output highlights only the dynamic K-corona components — CMEs and short-lived streamers.

논문은 LASCO-C2, STEREO/COR-1A, COR-2A, MLSO/KCor의 Level-1 영상에 SiRGraF를 적용하고 NRGF와 정량/정성적으로 비교한다. 결과적으로 SiRGraF는 (a) 처리 속도가 NRGF보다 빠르고, (b) 저-SNR (COR-1A) 영상에서 NRGF보다 동적 구조를 더 잘 드러내며, (c) 방대한 코로나그래프 데이터를 일괄 처리할 때 명확한 이점을 갖는다. 다만 강도 정보를 비선형으로 변형하므로 정량 광도 분석(전자 밀도 추출 등)에는 부적합하다.

The paper applies SiRGraF to LASCO-C2, STEREO/COR-1A, COR-2A and MLSO/KCor Level-1 images and benchmarks against NRGF. SiRGraF is (a) faster than NRGF, (b) better at revealing dynamic features in low-SNR COR-1A frames, and (c) clearly advantageous when processing huge image volumes in batch mode. The downside: because intensities are non-linearly remapped, SiRGraF is not suitable for quantitative photometric work (e.g. electron-density retrieval).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Background / 도입과 배경 (pp. 1-3)

- 백색광 코로나는 K-corona (자유전자 톰슨 산란)와 F-corona (행성간 먼지 산란)로 구성. F는 $\gtrsim 2.6 R_\odot$에서 거의 시간 불변 (Morgan & Habbal 2007).
- CME는 K-corona의 전자 밀도 구조 (Howard 2011); 분석을 위해 두 성분을 분리해야 한다.
- 기존 분리법:
  1. 장기 minimum 영상 (DeForest, Howard, McComas 2014; DeForest+ 2018)
  2. 월별 daily-median 영상의 최솟값 (Morrill+ 2006)
  3. 동경 방향 다항식 fitting (Morgan & Habbal 2010, Morgan, Byrne, Habbal 2012)
  4. LASCO-C2의 측광 보정에 기반한 K-corona 추출 (Morgan 2015)
- 동경 강도 변화 평탄화 (radial gradient filter, RGF) 의 흐름:
  - 1968 Newkirk & Harvey — 일식 사진을 위한 mechanical radial density filter
  - 2006 Morgan, Habbal & Woo — **NRGF** (subtract azimuthal mean, divide by azimuthal std at each height)
  - 2009 Byrne+ — multi-scale filter for CME detection
  - 2011 Druckmüllerová+ — **FNRGF** (Fourier 기반 local NRGF)
  - 2014 Morgan & Druckmüller — **MGN** (multi-scale Gaussian normalisation)
  - 2020 Qiang+ — **RLMF**
  - 그 외 EUV 적용 사례: AIA aia_rfilter, CIISCO (Patel+ 2021)
- 한계: NRGF나 FNRGF가 효과는 좋지만 **수백∼수천 영상의 일괄 처리에서는 느리다**. 이 논문의 동기는 "충분히 좋은데 훨씬 빠른" 알고리즘을 만드는 것.

- The white-light corona = K-corona (electron Thomson scattering) + F-corona (interplanetary dust). F is essentially time-invariant beyond $\sim 2.6 R_\odot$.
- CMEs are electron-density structures in the K-corona — we must separate the two.
- Existing background-removal options range from long-term minimum images to monthly daily-median minima to radial polynomial fits.
- The radial-gradient-filter family includes NRGF (2006), FNRGF (2011), MGN (2014), RLMF (2020). All are effective but many are computationally heavy when run over thousands of frames.
- Motivation: a fast, simple, batch-friendly RGF.

### Part II: The SiRGraF Algorithm / 알고리즘 (Section 2, pp. 4-5)

저자들은 STEREO/COR-1A의 2010-08-01 Level-1 데이터(5분 cadence, 약 24시간치)로 절차를 설명한다.

The authors illustrate the procedure on STEREO/COR-1A Level-1 data of 1 Aug 2010, taken at 5-minute cadence (~24 h).

**Step (i) — Minimum background $I_m$ (Fig. 1a):**

각 픽셀 $(x, y)$에서 하루 동안의 영상들 중 **0보다 큰 최솟값**을 취한다. 이는 1-percentile minimum 대신 사용하는 빠른 근사로, F-corona + 변동이 작은 K-corona + 산란광을 잡아낸다. CME나 transient는 해당 픽셀의 최솟값에 거의 영향을 주지 않으므로 자연스럽게 배경에서 빠진다.

For each pixel $(x, y)$ take the minimum over the day's frames excluding zeros. This is a fast surrogate for the 1-percentile minimum and captures the F-corona + slowly varying K-corona + scatter. CMEs barely affect the per-pixel minimum, so they are naturally excluded from the background.

**Step (ii) — Radial profile from $I_m$ (Fig. 1b):**

$I_m$을 polar coordinates $(r, \phi)$로 변환하고, 각 height $r$에서 모든 방위각 $\phi'$에 대해 강도를 평균하여 1-D radial profile $\bar I_m(r)$을 얻는다. 결과적으로 streamer 같은 방위각 의존 구조는 평균화되어 사라진다.

Convert $I_m$ to polar coordinates and average over azimuth at each radius to get a 1-D profile $\bar I_m(r)$. Streamers and other azimuth-dependent structures wash out.

**Step (iii) — Uniform background $I_u$ (Fig. 1c):**

$\bar I_m(r)$을 모든 방위각에 대해 그대로 회전시켜 다시 2-D 영상으로 만든다. 결과는 회전 대칭 (circularly symmetric) 배경으로, 동경 강도 변화만 가지고 있고 방위각 정보는 없다. 이는 기기 vignetting과 산란광이 사실상 적도에서 극까지 거의 균일하다는 사실 (Morgan & Habbal 2007; Patel+ 2018)에 기반한다.

Rotate $\bar I_m(r)$ across all azimuths to rebuild a 2-D rotation-symmetric image. The instrumental background and scatter are nearly azimuth-independent (Morgan & Habbal 2007), so this is a faithful "what the radial gradient should look like" estimate.

**Step (iv) — Apply the filter (Fig. 1d):**

$$
I' = \frac{I - I_m}{I_u}
$$

분자: 정적/준정적 배경 제거 → K-corona의 동적 성분만 남김. 분모: 동경 강도 변화를 평탄화 (정규화). 결과 픽셀 값은 대략 0-1 범위의 무차원 (normalised intensity).

Numerator: subtracts static / quasi-static background → leaves only dynamic K-corona. Denominator: flattens the radial intensity drop. Output is a dimensionless normalised intensity, roughly in $[0, 1]$.

논문의 중요한 실용적 디테일: **daily-median 대신 daily-minimum**을 쓰는 이유는 daily-median이 K-corona 기여를 더 많이 포함하여 dynamic K-corona 신호까지 깎아내기 때문 (Thompson+ 2010).

A key practical detail: the paper deliberately chooses **daily-minimum** over **daily-median** — the median image already contains substantial K-corona, so subtracting it would also remove dynamic K-corona signal we care about (Thompson et al. 2010).

### Part III: Results / 결과 (Section 3, pp. 6-9)

**3.1 Application to coronagraph images (Fig. 1, Fig. 2):**

- COR-1A (2010-08-01): SiRGraF가 CME 3-part 구조를 FOV 외곽까지 명확히 드러냄 (Fig. 1d, Fig. 2 middle row).
- LASCO-C2 (2001-07-07): 고전적 3-part CME (front, cavity, core) 구조 분명 (Fig. 2 top row).
- COR-2A (2010-08-01): 더 큰 height까지 추적 가능, 단 outer-edge에 vignetting 기인 ring artefact가 보임 (DeForest+ 2018).
- MLSO/KCor (2015-07-02, 지상 기반, 15-s cadence): CME는 잘 잡히지만 atmospheric contamination 때문에 streamer 신호는 손실됨.

**3.2 Comparison with extended-period background (Fig. 3, p. 9):**

7-day minimum (2010-07-26 ∼ 2010-08-01)을 사용해 동일한 COR-1A 영상을 처리하면, **streamer**가 더 두드러진다 (장기 평균이 streamer를 더 잘 빼주기 때문). CME는 1-day 배경에서도 잘 보이지만, **장기 정적 구조**를 보려면 1-week+ 배경이 유리하다.

**3.3 Comparison with NRGF (Fig. 4-5, pp. 10-12):**

- LASCO-C2와 COR-2A의 고-SNR 영상에서는 NRGF와 SiRGraF가 시각적으로 비슷한 brightness/contrast를 제공.
- **COR-1A (저-SNR)에서는 SiRGraF가 분명히 우월** — NRGF는 noise를 더 증폭하고 동적 구조 식별이 어려움.
- 처리 시간: 1024×1024 영상 한 장 기준 SiRGraF가 NRGF보다 빠름. (정확한 수치는 본문에 짧게 언급되어 있으나, NRGF는 polar warp을 매 프레임마다 다시 하는 반면 SiRGraF는 배경을 한 번만 만들고 재사용하기 때문임.)

**Quantitative observations / 정량 관찰:**

| Coronagraph | Cadence | Field of view | SiRGraF result |
|---|---|---|---|
| LASCO-C2 | ~12 min | 2-6 $R_\odot$ | 3-part CME 구조 명확 |
| STEREO/COR-1A | 5 min | 1.4-4 $R_\odot$ | NRGF 대비 우수 |
| STEREO/COR-2A | 5-15 min | 2-15 $R_\odot$ | 큰 height까지 추적 가능 |
| MLSO/KCor | 15 s | 1.05-3 $R_\odot$ | CME OK, streamer 손실 |

### Part IV: Discussion & Limitations / 논의 (Section 4, pp. 12-14)

- **장점**: (i) 단순 — 4단계, 거의 무파라미터. (ii) 빠름 — 배경을 한 번 만들고 그 날 모든 frame에 재사용. (iii) batch-friendly. (iv) 저-SNR에서도 안정적.
- **한계**:
  1. **정량 분석 불가** — 분모가 영상마다 같지 않으면 안 되므로 photometry나 electron-density 추정에 쓸 수 없다.
  2. **지상 코로나그래프 (KCor)에서는 atmospheric contamination 때문에 streamer 손실**.
  3. **Outer-edge ring artefact** (vignetting 기원) 별도 처리 필요.
  4. **장기 정적 구조 (long-lived streamer)**를 보려면 7-day+ 배경이 필요.
- **향후 활용**: 자동화된 CME 검출 파이프라인의 전처리, COR-1A 데이터 long-term study, ML 입력 정규화.

- Strengths: simplicity (4 steps, almost no hyperparameters), speed (build the background once, reuse across all frames of the day), batch-friendly, robust at low SNR.
- Limitations: (1) not for quantitative photometry, (2) ground-based KCor loses streamers due to atmospheric contamination, (3) outer-edge ring artefacts from instrumental vignetting need a separate fix, (4) for long-lived streamers an extended-period (≥7 day) background is preferable.
- Future use: preprocessing for automated CME detection pipelines, long-term COR-1A studies, normalisation for ML inputs.

---

## 3. Key Takeaways / 핵심 시사점

1. **단순함이 곧 속도** / **Simplicity is speed** — 두 번의 픽셀 연산 ($I - I_m$, ÷ $I_u$)만으로 NRGF급 결과를 얻는다. 일괄 처리에 결정적. / Two pixel-wise operations achieve NRGF-level enhancement, decisive for batch mode.

2. **Daily minimum > daily median** — daily-median은 K-corona를 너무 많이 포함하므로 동적 신호를 깎는다. **0 초과 daily minimum**은 정적 배경을 더 깨끗이 분리한다. / Daily minimum (excluding zeros) cleanly separates the static background; the median includes too much K-corona.

3. **회전 대칭 분모는 충분히 좋은 근사** / **A circularly symmetric denominator is a good enough approximation** — 기기 산란과 vignetting이 거의 azimuth-independent라는 관측 사실 (Morgan & Habbal 2007)에 의존한다. / Relies on the observed near-azimuthal-symmetry of instrumental scatter and vignetting.

4. **저-SNR (COR-1A)에서 NRGF보다 우월** / **Outperforms NRGF at low SNR** — NRGF는 std로 나누기 때문에 noisy region에서 잡음을 증폭. SiRGraF는 한 장의 회전 대칭 배경으로 나누므로 더 안정적. / NRGF amplifies noise via division by per-height std; SiRGraF divides by a single smooth background, hence more stable.

5. **정량 분석에는 부적합** / **Not for quantitative photometry** — 비선형 변환이 들어가므로 brightness 단위가 깨진다. 동적 구조 시각화·검출 전용. / The non-linear transformation breaks the brightness units; use it only for dynamic-feature visualisation/detection.

6. **Background 기간 (1-day vs 7-day)이 trade-off** / **Background window is a trade-off** — 1-day는 짧은 transient 강조, 7-day는 long-lived streamer 강조. 응용에 맞춰 선택. / 1-day window highlights short-lived transients; 7-day window highlights long-lived streamers.

7. **지상 코로나그래프에는 한계** / **Ground-based limitation** — KCor 같은 ground-based instrument에서는 atmospheric variability가 daily-minimum을 오염시킨다. extended-period background로 부분적 보완 가능. / Atmospheric variability contaminates the daily minimum; an extended-period background partially mitigates this.

8. **CME 자동 검출의 전처리로 이상적** / **Ideal preprocessing for automated CME detection** — 단순하고 속도가 빨라서 머신러닝 segmentation 전 단계 정규화로 적합. / Its simplicity and speed make it a natural normalisation step ahead of ML-based CME segmentation.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Core SiRGraF formula / 핵심 식

$$
I'(x, y) = \frac{I(x, y) - I_m(x, y)}{I_u(x, y)}
$$

| 기호 / Symbol | 의미 / Meaning |
|---|---|
| $I(x,y)$ | 처리할 Level-1 영상 (단일 frame) / single Level-1 frame |
| $I_m(x,y)$ | 하루 분량 영상의 픽셀별 최솟값 (>0) / per-pixel minimum (>0) over one day |
| $I_u(x,y)$ | $I_m$의 azimuthal 평균을 회전 대칭하게 만든 배경 / azimuthally averaged $I_m$ rebuilt as a rotation-symmetric image |
| $I'(x,y)$ | 정규화된 출력 (∼ $[0,1]$) / normalised output |

### 4.2 Minimum background / 최소 배경

$$
I_m(x, y) = \min_{\substack{t \in \mathcal{T}_{\text{day}} \\ I_t(x,y) > 0}} I_t(x, y)
$$

$\mathcal{T}_{\text{day}}$ = 하루(또는 N-day) 동안 수집된 frame 시각의 집합. 0 픽셀(데이터 누락, 마스크 영역)은 제외.

$\mathcal{T}_{\text{day}}$ = the set of frame times in one day (or $N$ days). Zero pixels (missing data, masked-out regions) are excluded.

### 4.3 Radial profile / 동경 프로파일

극좌표로 변환 후 / In polar coordinates:

$$
\bar I_m(r) = \frac{1}{N_\phi} \sum_{j=1}^{N_\phi} I_m\!\big(r, \phi_j\big)
$$

$N_\phi$ = 방위각 샘플 수.

### 4.4 Uniform background reconstruction / 균일 배경 재구성

$$
I_u(r, \phi) = \bar I_m(r) \quad \forall \phi
$$

Cartesian으로 다시 변환하면 회전 대칭 2-D 배경.

Convert back to Cartesian to get a 2-D rotation-symmetric background.

### 4.5 NRGF (비교용 / for comparison)

$$
I'_{\text{NRGF}}(r, \phi) = \frac{I(r,\phi) - \mu_\phi(r)}{\sigma_\phi(r)},
$$

with $\mu_\phi(r) = \langle I(r,\phi') \rangle_{\phi'}$, $\sigma_\phi(r) = \mathrm{std}_{\phi'}\big(I(r,\phi')\big)$.

차이: NRGF는 **한 장**의 영상에서 mean·std를 azimuthal하게 추정. SiRGraF는 **여러 장**의 minimum stack을 사용 → static background를 정확히 분리.

Difference: NRGF estimates azimuthal mean & std from **a single frame**, whereas SiRGraF stacks **many frames** for the minimum, properly separating the static background.

### 4.6 Worked example / 풀이 예제

가정 / Assume: 한 픽셀 $(x_0, y_0)$ 위치에서 하루 동안 256장의 frame이 들어왔고, 강도 $I_t$가 다음과 같다:

$\{1.0, 1.1, 1.0, 1.0, 1.2, \dots, 1.0, 1.0\}$ (대부분 ~1.0, 가끔 transient로 1.5).

- $I_m(x_0, y_0) = 1.0$ (최솟값).
- $\bar I_m(r)$ at the same radius $r = r_0$: 모든 방위각 평균이 $\approx 1.0$ (assume azimuth-independent), so $I_u = 1.0$ at $(x_0, y_0)$.
- 어느 frame $I_t(x_0, y_0) = 1.5$ (CME front 통과)에 대해:

$$
I'(x_0, y_0) = \frac{1.5 - 1.0}{1.0} = 0.5
$$

→ 정규화된 영상에서 이 픽셀은 0.5 (밝은 dynamic feature). 반면 정적 영역은 $I'\approx 0$.

→ in the normalised image this pixel is 0.5 (a bright dynamic feature) while static regions are $I' \approx 0$.

### 4.7 Algorithm pseudocode / 의사코드

```
Input: list of N Level-1 frames I_1, ..., I_N (size H×W) for one day
Output: normalised frames I'_1, ..., I'_N

1. I_m  ← per-pixel minimum of {I_t : I_t > 0} over t = 1..N
2. (r, phi) ← polar grid centred on Sun
3. for each radius r_k:
       I_bar(r_k) ← mean over phi of I_m(r_k, phi)
4. I_u ← I_bar(r) replicated across all phi, mapped back to Cartesian
5. for each frame t = 1..N:
       I'_t ← (I_t - I_m) / I_u
```

복잡도: 배경 생성 $O(N \cdot H W)$ 한 번 + 프레임당 $O(HW)$ → 일괄 처리에서 매우 효율적.

Complexity: one $O(N \cdot HW)$ pass to build the background, then $O(HW)$ per frame — extremely efficient in batch.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1950 ── van de Hulst         Theoretical K + F corona decomposition
                              ↓
1968 ── Newkirk & Harvey      Mechanical radial density filter for eclipses
                              ↓
1995 ── Brueckner+ (LASCO)    Continuous coronagraph data stream begins
                              ↓
2006 ── Morgan, Habbal & Woo  NRGF — first widely adopted RGF
                              ↓
2007 ── Morgan & Habbal       Quantified K/F radial profiles over solar cycle
                              ↓
2011 ── Druckmüllerová+       FNRGF — Fourier-based local NRGF
                              ↓
2014 ── Morgan & Druckmüller  MGN — Multi-Scale Gaussian Normalization
                              ↓
2018 ── DeForest+ (Karl)      DeForest "polar" minimum background, vignetting work
                              ↓
2020 ── Qiang+                RLMF — Radial Local Multi-Scale Filter
2021 ── Patel+                CIISCO — automated EUV CME detection
   ★ 2022 ── PATEL+ (this)    SiRGraF — minimum + uniform background, batch-friendly
                              ↓
202X ── future                ML-segmentation pipelines using SiRGraF as input normaliser
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Morgan, Habbal & Woo (2006) — NRGF | 직접적 비교 대상; SiRGraF는 NRGF의 batch-속도 약점을 보완 / Direct comparison; SiRGraF improves on NRGF's batch-speed weakness | Very High |
| Morgan & Druckmüller (2014) — MGN | 같은 enhancement 가족; MGN은 multi-scale, SiRGraF는 single-radial / Same enhancement family; MGN is multi-scale, SiRGraF is single-radial | High |
| DeForest, Howard & McComas (2014) | minimum-background subtraction의 선구자 / Pioneered minimum-background subtraction for inbound waves | High |
| Druckmüllerová, Morgan, Habbal (2011) — FNRGF | Local Fourier 기반 RGF; SiRGraF는 더 단순한 대안 / Local Fourier RGF; SiRGraF is the simpler alternative | Medium |
| Patel+ (2021) — CIISCO | 같은 그룹의 EUV CME detection 알고리즘; SiRGraF가 입력 전처리로 적합 / Same group's EUV CME detector; SiRGraF natural front-end | High |
| Byrne+ (2009, 2012) — CORIMP | Multi-scale filter + CME detection; 비교 컨텍스트 / Multi-scale filter + CME detection (context) | Medium |
| Qiang+ (2020) — RLMF | 최신 RGF 변종 / Latest RGF variant for context | Medium |

---

## 7. References / 참고문헌

### Additional algorithmic notes / 추가 알고리즘 노트

**왜 daily minimum이 작동하는가?** 매 frame에서 같은 픽셀 위치의 강도는 (정적 F-corona) + (느린 K-corona 변화) + (transient CME) + (noise)의 합이다. transient는 픽셀당 방문 시간이 짧으므로 시간 평균에는 거의 영향을 주지 않지만 **최솟값**에는 더 민감하지 않다 — transient는 최솟값이 아니라 **최댓값**에 영향. 따라서 최솟값은 정적·준정적 성분만 깨끗하게 잡아낸다.

**Why does the daily minimum work?** The intensity at a fixed pixel in each frame is (static F-corona) + (slowly varying K-corona) + (transient CME) + (noise). Transients are short-lived and affect the **maximum** more than the **minimum**, so the per-pixel minimum cleanly captures only the static / quasi-static component.

**왜 mean이 아니라 minimum?** Daily mean이나 daily median은 K-corona의 long-lived (수 시간) streamer를 포함하기 때문에 SiRGraF로 dynamic K-corona를 보고 싶을 때 신호의 일부를 제거해 버린다. Minimum은 이 문제를 피한다.

**Why minimum, not mean/median?** The daily mean or daily median includes long-lived (hours-scale) K-corona streamers and thus subtracts away part of the dynamic K-corona we want to keep. The minimum avoids this by definition.

**SiRGraF가 NRGF보다 빠른 이유**: NRGF는 매 프레임마다 polar warp + 방위각 통계를 다시 계산해야 한다. SiRGraF는 minimum background와 uniform background를 **한 번만** 만들고, 그 날의 모든 frame에 동일한 배경을 재사용하므로 frame 단위 cost는 단순한 element-wise 산술뿐이다.

**Why SiRGraF is faster than NRGF**: NRGF must redo polar warping and per-radius statistics for every frame. SiRGraF builds the minimum and uniform backgrounds **once** and reuses them across the day's frames, so the per-frame cost reduces to element-wise arithmetic.

**Outer-edge ring artefacts** (COR-2A에서 관측됨): 기기 vignetting의 깊은 minimum 근처에서 photon noise가 증가하면서 minimum background가 ring 패턴을 가진다. 이는 minimum subtraction 후에도 ring으로 남아 정규화 영상에 인공물을 만든다. DeForest+ (2018)의 vignetting 보정 또는 별도 ring suppression이 필요.

**Outer-edge ring artefacts** (visible in COR-2A): photon noise rises near the deep minimum of the instrumental vignetting, so the minimum background itself acquires a ring pattern. After subtraction the ring persists in the normalised image. Either DeForest+ (2018) vignetting correction or a dedicated ring-suppression step is required.

### Comparison summary table / 비교 요약 표

| Aspect | NRGF | FNRGF | MGN | SiRGraF |
|---|---|---|---|---|
| Per-frame cost | Polar warp + stats | Polar warp + Fourier | multi-scale Gaussian | divide by precomputed $I_u$ |
| Background built per | frame | frame | frame | day (reused) |
| Hyperparameters | 0 | Fourier order | scale set $\{\sigma_i\}$ | period (1 day default) |
| Quantitative photometry | × | × | × | × |
| Best for | quick visualisation | low-contrast regions | multi-scale features | batch dynamic-K-corona |

### Practical recipe for application / 실제 적용 레시피

1. **Data ingestion**: Level-1 영상 (flat·dark 보정, 정렬, 정북 정렬 완료) 일치성 확인. 필요시 1024×1024 또는 2048×2048 표준 크기로 정렬. / Verify Level-1 calibration and alignment; standardise to 1024² or 2048².
2. **Frame selection**: 24시간치 (보통 100-300 frame, instrument cadence dependent)를 모은다. 단주기 transient 강조에는 1-day가 좋고, long-lived streamer 강조에는 7-day 이상 추천. / Collect 24 h of frames; use 1-day window for transients, ≥7-day for streamers.
3. **Build $I_m$**: per-pixel minimum (zero exclusion). NumPy `np.where(stack > 0, stack, np.inf).min(axis=0)` 한 줄. / One NumPy line builds $I_m$.
4. **Build $I_u$**: $I_m$의 polar warp → 각 radius bin에서 azimuthal mean → Cartesian으로 다시 변환. / Polar-warp $I_m$, mean over $\phi$, warp back.
5. **Apply**: `(I - I_m) / np.where(I_u > 0, I_u, 1)`로 모든 frame을 한 번에 처리. / Vectorise across frames.
6. **Output normalisation**: 결과 픽셀은 대략 $[0, 1]$이지만 outlier가 있을 수 있어 시각화에는 percentile clipping (예: 0.5%-99.5%) 권장. / Use percentile clipping for display.
7. **CME tracking 등 다운스트림**: SiRGraF 정규화 영상은 ML segmentation의 자연스러운 입력이지만 정량 photometry에는 raw frame 사용. / Use SiRGraF for ML, raw frames for photometry.

### Limitations & caveats summary / 한계 요약

- **No photometry**: division by $I_u$가 비선형 변환을 도입 → 절대 강도 단위 손실. / Non-linear transform breaks intensity units.
- **Ground-based contamination**: KCor 같은 지상 코로나그래프의 atmospheric variability가 $I_m$을 오염. / Atmospheric variability contaminates $I_m$ on ground-based data.
- **Outer-edge ring artefacts**: vignetting 깊은 곳의 photon noise가 $I_m$에 ring을 만든다 → 정규화 후에도 잔존. / Photon-noise rings persist near deep vignetting.
- **Streamer suppression with 1-day window**: streamer가 24시간 동안 거의 정적일 경우 minimum에 포함되어 함께 빠진다. 장기 dynamic streamer는 보존되지만 quasi-static streamer는 안 보임. / Quasi-static streamers vanish with a 1-day window.
- **Need for $\geq 100$ frames**: 너무 적으면 minimum이 noisy. cadence가 낮은 instrument (예: SOHO/UVCS)에는 부적합. / Needs sufficient frames; not for low-cadence instruments.

### References / 참고문헌

- Patel, R., Majumdar, S., Pant, V., Banerjee, D., "A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images", *Solar Physics* 297, 27 (2022). DOI: 10.1007/s11207-022-01957-y
- Morgan, H., Habbal, S. R., Woo, R., "The Depiction of Coronal Structure in White-Light Images", *Solar Physics* 236, 263 (2006).
- Morgan, H., Habbal, S. R., "An empirical 3D model of the large-scale coronal structure based on the distribution of prominences on the solar disc", *ApJ* 710, 1 (2010).
- Morgan, H., Druckmüller, M., "Multi-Scale Gaussian Normalization for Solar Image Processing", *Solar Physics* 289, 2945 (2014).
- DeForest, C. E., Howard, T. A., McComas, D. J., "Inbound Waves in the Solar Corona", *ApJ* 787, 124 (2014).
- DeForest, C. E.+, "Polarimeter-quality observations of the solar corona", *ApJ* 862, 18 (2018).
- Druckmüllerová, H., Morgan, H., Habbal, S. R., "Enhancing Coronal Structures with the Fourier Normalizing-Radial-Graded Filter", *ApJ* 737, 88 (2011).
- Brueckner, G. E., et al., "The Large Angle Spectroscopic Coronagraph (LASCO)", *Solar Physics* 162, 357 (1995).
- Howard, R. A., et al., "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)", *Space Sci. Rev.* 136, 67 (2008).
- Morgan, H., "An Atlas of Coronal Electron Density at 5 R⊙", *ApJ* 800, 53 (2015).
- Patel, R., et al., "CIISCO: Coronal mass ejection Identification In Inner Solar Corona", *Frontiers in Astronomy and Space Sciences* 8, 752748 (2021).
- Qiang, Z., et al., "Radial Local Multi-Scale Filter (RLMF)", *Solar Physics* 295, 152 (2020).
- van de Hulst, H. C., "The electron density of the solar corona", *Bull. Astron. Inst. Neth.* 11, 135 (1950).
