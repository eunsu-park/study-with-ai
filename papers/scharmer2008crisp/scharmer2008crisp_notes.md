---
title: "CRISP spectropolarimetric imaging of penumbral fine structure"
authors: Scharmer, G. B., Narayan, G., Hillberg, T., de la Cruz Rodriguez, J., Löfdahl, M. G., Kiselman, D., Sütterlin, P., van Noort, M., Lagg, A.
year: 2008
journal: "ApJ Letters, 689, L69–L72"
doi: "10.1086/595744"
topic: Solar Observation
tags: [CRISP, SST, Fabry-Perot, spectropolarimetry, penumbra, Evershed-flow, uncombed-model, dark-core, Stokes, Zeeman, MOMFBD]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 22. CRISP Spectropolarimetric Imaging of Penumbral Fine Structure / CRISP 분광편광 영상으로 본 반암부 미세구조

---

## 1. Core Contribution / 핵심 기여

이 4쪽짜리 **ApJ Letter**는 SST(Swedish 1-m Solar Telescope)의 신규 장비 **CRISP(CRisp Imaging SpectroPolarimeter)** 로 얻은 첫 과학 결과를 발표한다. CRISP은 **이중 Fabry–Pérot 에탈론** 기반 tunable narrow-band filter로, Fe I 6301.5/6302.5 Å 선을 ~50 mÅ 간격으로 스캔하며 각 파장에서 Stokes $I, Q, U, V$ 를 동시에 영상화한다. 이 기능을 SST의 AO와 MOMFBD 후처리에 결합하여 **sunspot penumbra**의 공간·분광·편광 구조를 ~0.15″ 해상도로 해상한다. 주요 과학 결과: (1) **penumbral dark core**(Scharmer 2002)는 주변 밝은 측면부에 비해 낮은 Stokes $V$ 진폭과 높은 field inclination(수평에 가까움)을 보여 **자기장이 필라멘트 중심에서 더 수평**임을 입증한다. (2) **Evershed flow**(filament 내부의 외향 수평 유동)의 종착점인 filament tail에서 **강한 하향류(downflow, 수 km/s)** 를 분광으로 직접 검출하여 Evershed 흐름이 filament tail에서 magnetic canopy 아래로 하강한다는 uncombed 페넘브라 모델을 뒷받침한다. (3) 페넘브라 전체에 걸쳐 자기장 방향이 **가로·세로로 교대(interleaving)** 하는 것을 Stokes $Q, U$ 맵으로 확인한다. 논문 자체는 Letter 규모지만, 이후 **SST/CRISP 과학 프로그램과 DKIST/GREGOR imaging spectropolarimetry**의 기준을 세운 기념비적 연구다.

This 4-page **ApJ Letter** presents the first science results from **CRISP (CRisp Imaging SpectroPolarimeter)**, newly installed at the **SST (Swedish 1-m Solar Telescope)**. CRISP is a **dual Fabry–Pérot tunable filter** that scans the Fe I 6301.5/6302.5 Å lines in ~50 mÅ steps while simultaneously imaging Stokes $I, Q, U, V$. Combined with SST AO and MOMFBD post-processing, it resolves **sunspot penumbra** spatially (~0.15″), spectrally, and polarimetrically. Key results: (1) **penumbral dark cores** (Scharmer 2002) exhibit **weaker Stokes $V$** and **more inclined (more horizontal) fields** than the bright lateral sides — confirming that the magnetic field is more horizontal at filament centers; (2) at **filament tails**, strong **downflows of several km/s** are detected, providing direct evidence that the Evershed outflow dives below the magnetic canopy — a core prediction of the **uncombed penumbra model** (Solanki & Montavon 1993); (3) the **interleaving of horizontal and vertical magnetic fields** across the penumbra is visible in Stokes $Q, U$ maps. Beyond its Letter-scale scope, this paper established the benchmark for the entire **SST/CRISP science program** and for imaging spectropolarimetry at **GREGOR and DKIST**.

---

## 2. Reading Notes / 읽기 노트

### Part 0: Sunspot Anatomy — Quick Reference / 흑점 해부도 참조

Penumbra 논의를 이해하기 위한 기본 구조.

- **Umbra (암부)** — 중심의 가장 어두운 부분. $T \sim 4000$ K (주변보다 2000 K 저온), $B \sim 2$–3 kG (거의 수직).
- **Penumbra (반암부)** — umbra 바깥을 둘러싼 방사상 필라멘트 구조. $T \sim 5000$ K, $B \sim 1$–2 kG (기울어진 자기장).
- **Light bridge** — umbra를 가로지르는 밝은 구조. 자기장 약화 영역.
- **Pores** — penumbra 없는 작은 흑점.
- **Moat** — penumbra 바깥의 외부 흐름 영역. Evershed flow의 연장.

Basic sunspot anatomy: **umbra** (darkest center, $T \sim 4000$ K, $B \sim 2$–3 kG, near-vertical), **penumbra** (radial filaments, $T \sim 5000$ K, $B \sim 1$–2 kG, inclined), **light bridge** (bright gap in umbra), **pores** (umbra-only spots), **moat** (outer outflow region extending the Evershed flow).

**반암부 논쟁 / The penumbra puzzle**: 왜 umbra는 순수 자기 대류 억제로 어둡고, penumbra는 필라멘트 구조를 가지며 밝은가? 답은 **경사진 자기장에서의 대류 규칙**이 달라지기 때문. 수평 성분이 생기면 플럭스 관이 부력에 의해 솟아오르고, 냉각되면 가라앉는 **수문(siphon) 순환**이 가능 → Evershed flow.

Why is the umbra dark (convection suppressed) while the penumbra is bright and filamented? Because **convection in inclined magnetic fields obeys different rules**: horizontal field components allow flux tubes to rise by buoyancy and sink when cooled, producing siphon flows — the Evershed circulation.

### Part I: CRISP Instrument (§1–§2) / CRISP 장비

**설계 / Design**
- **이중 Fabry–Pérot** — 낮은 finesse 에탈론(prefilter)과 높은 finesse 에탈론을 조합. 높은 finesse는 좁은 bandpass(~6 pm FWHM @ 630 nm), 낮은 finesse는 FSR(free spectral range)을 넓혀 원치 않는 order를 배제.
- **튜닝** — 각 에탈론을 압전(piezo) 구동으로 미세 간격 조정 → 파장 선택. 전체 스캔은 50 mÅ 스텝이 표준.
- **편광 변조** — 액정 가변 리타더(LCVR) 또는 회전 파장판으로 Stokes 상태 순환 측정 후 이미지 연산으로 $I, Q, U, V$ 복원.

The filter consists of **two Fabry–Pérot etalons in series**: a low-finesse prefilter broadens the FSR to suppress adjacent orders, and a high-finesse etalon sets the ~6 pm FWHM bandpass at 630 nm. Piezo-driven cavity tuning scans wavelengths in ~50 mÅ steps. Polarization modulation is done via LCVRs (liquid-crystal variable retarders) cycling the Stokes states, followed by image arithmetic to recover $I, Q, U, V$.

**관측 / Observations**
- 대상: 활성영역 AR 10933의 sunspot (disk center 근처, $\mu \approx 0.97$).
- 스캔: Fe I 6301.5/6302.5 Å, 약 ±500 mÅ 범위에서 ~16 파장점.
- 시간 해상도: burst 한 사이클당 ~30 s (편광 상태 4개 × 파장 16개 × 프레임 burst).
- 공간 해상도: AO + MOMFBD 후 ~0.15″ (0.1 arcsec pixel 스케일, 회절 한계).

Target: AR 10933 near disk center ($\mu \approx 0.97$). Wavelength scan: Fe I 6301.5 and 6302.5 Å across ±500 mÅ in ~16 points. Cadence: ~30 s per full Stokes cycle. Angular resolution: ~0.15″ after AO + MOMFBD, near the SST diffraction limit.

### Part II: Penumbral Fine Structure — Intensity & Continuum (§3 Figs 1–2)

**Figure 1** — 스팟의 continuum intensity 이미지. Umbra 중앙 암부, 그 주변 penumbra에 수백 개의 방사형 필라멘트. 개별 필라멘트 내부에 **dark core**(Scharmer 2002 발견)가 뚜렷이 보인다. 필라멘트 양 끝(tail)이 외부 moat 영역으로 연결.

Figure 1 shows the continuum intensity: the umbra is dark, the penumbra is filled with hundreds of radial filaments, each displaying a distinct central **dark core** (Scharmer 2002). Filament tails connect to the outer moat region.

**Figure 2** — Line-core Doppler map과 continuum 이미지 overlay. Filament 내부 전체에 **적색 편이(redshift, downflow)** 가 아니라 주로 blueshift(외향 유동 = Evershed flow). 하지만 **필라멘트 끝부분 tail 영역에 강한 redshift 패치**가 존재 → tail에서 downflow 검출.

Figure 2 overlays a line-core Doppler map. Filament interiors show blueshift (outflow = Evershed flow), but **filament tails show strong redshift patches**, i.e. localized downflows.

### Part III: Magnetic Field from Stokes V (§3 Figs 3–4) / Stokes V로 본 자기장

**Stokes V magnetogram**
- Weak-field 근사 또는 center-of-gravity 방법으로 $B_{\rm LOS}$ 추출.
- Umbra: $B_{\rm LOS} \sim 2$–3 kG, penumbra 외곽: $\sim 1$ kG.
- 필라멘트 dark core를 따라 **$|V|$가 주변보다 감소** → 같은 $B$에 대해 $V$가 작으려면 $\theta_B$ 가 커야(수평에 가까움). 즉 **dark core에서 자기장이 더 수평**.

From Stokes $V$ the line-of-sight field $B_{\rm LOS}$ is extracted (weak-field / center-of-gravity). Umbra $\sim$ 2–3 kG, outer penumbra $\sim$ 1 kG. Along filament **dark cores**, $|V|$ is reduced relative to the bright lateral sides — implying **larger inclination angle $\theta_B$** (more horizontal), consistent with horizontal flux tubes at filament centers.

**Field inclination**
- Stokes $Q, U$ 진폭이 horizontal field component에 비례.
- Filament을 가로지르는 cross-cut 프로파일에서 $|Q|, |U|$가 필라멘트 중심에서 최대, 측면부에서 최소 → **uncombed penumbra model 확증**.

Stokes $Q, U$ scale with the horizontal field. Cross-filament profiles show $|Q|, |U|$ peaking at filament centers — direct observational evidence for the **uncombed penumbra model**.

### Part IV: Evershed Flow at Filament Tails (§3) / 필라멘트 끝부분의 Evershed 흐름

핵심 발견:
- Filament 내부는 외향 blueshift. Filament **끝부분 tail**에 좁은 redshift (downflow up to ~5 km/s).
- Downflow 패치의 위치는 $V$ 신호가 약해지는 지점과 일치 — magnetic canopy 아래로 flow가 dive함을 시사.
- 이는 **"Uncombed penumbra + siphon flow"** 모델 (Meyer & Schmidt 1968, Thomas & Montesinos 1993) 의 수문(siphon) 종단점 시나리오와 일치.

Key discovery: filament interiors are blueshifted (outflow), but narrow redshift patches (up to ~5 km/s downflow) appear at filament **tails**. Stokes $V$ weakens at the same locations, suggesting the outflow dives below the magnetic canopy — consistent with the **uncombed penumbra + siphon-flow** scenario (Meyer & Schmidt 1968; Thomas & Montesinos 1993).

### Part IV-bis: Line Formation and Interpretation Caveats / 선 형성과 해석 주의점

**Weak-field limit의 유효성 / Validity of weak-field limit**
- $\Delta\lambda_B \ll \Delta\lambda_D$ (Doppler 폭)일 때 Weak-field 근사가 성립. Fe I 6302.5, $B \sim 1000$ G: $\Delta\lambda_B \sim 46$ mÅ vs $\Delta\lambda_D \sim 30$–40 mÅ → **경계 영역**. Umbra(~3 kG)에서는 saturation.
- 진짜 정밀한 분석은 **Milne–Eddington 역전(inversion)** 또는 full Stokes inversion 코드(SIR, Héliore, Nicole 등) 필요.

The weak-field approximation is valid when $\Delta\lambda_B \ll \Delta\lambda_D$. For Fe I 6302.5 at ~1000 G, $\Delta\lambda_B \sim 46$ mÅ is borderline with thermal Doppler widths of 30–40 mÅ; umbral fields (~3 kG) saturate. Precise analysis requires **Milne–Eddington inversion** or full non-LTE Stokes codes (SIR, Héliore, Nicole).

**높이 민감도 / Height sensitivity**
- Fe I 6301.5/6302.5는 **광구 중층** (τ ~ 1 기준 약 150–300 km 위) 형성.
- Dark core의 자기장 inclination은 높이에 따라 변화 → 선택한 선의 형성 고도가 해석에 영향.

The Fe I 6301.5/6302.5 lines form in the **mid-photosphere** (~150–300 km above $\tau=1$). Inclination measurements are altitude-dependent; different line choices probe different heights.

**다른 효과 / Other effects**
- **Chromospheric canopy** — 상위 Ca II 8542 Å 관측과 결합하면 높이별 구조 추적 가능.
- **3D radiative transfer / scattering** — filament의 얇은 구조에서는 LOS 근사 한계.

### Part V: Discussion (§4) / 토의

- **Uncombed model** (Solanki & Montavon 1993)이 관측적으로 지지됨:
  - Filament 중심 = 수평 flux tube (흐름 있음, 낮은 $|V|$).
  - Filament lateral sides = 수직 배경장 (흐름 없음, 높은 $|V|$).
  - Tail = flux tube이 다시 surface 아래로 잠수.
- MHD 시뮬레이션 (Heinemann 2007, Rempel 2009, Schüssler & Vögler 2006)과 비교 필요 — 본 Letter는 관측 제약을 제공.
- 미래: CRISP의 고카던스 + 다파장(예: Ca II 8542 Å 채층선) 관측으로 filament 동역학 시간 추적.

The **uncombed model** is confirmed observationally: filament centers = horizontal flux tubes (with flow, low $|V|$), lateral sides = vertical background field (static, high $|V|$), tails = where flux tubes dive subsurface. MHD simulations (Heinemann 2007; Rempel 2009; Schüssler & Vögler 2006) must match these constraints. Future work: use CRISP's high cadence and multi-line capability (including the Ca II 8542 Å chromospheric line) to time-resolve filament dynamics.

### Part VI: CRISP vs Other Instruments / 다른 장비와의 비교

| 장비 / Instrument | 망원경 / Telescope | 형식 / Type | 해상도 / Resolution | 특징 / Notes |
|---|---|---|---|---|
| **CRISP** | SST (1 m) | Imaging FP, dual etalon | 0.15″ | High cadence, tunable, full Stokes |
| Hinode/SP | SOT (0.5 m, 우주) | Slit spectrograph | 0.32″ | Seeing-free, slow (slit scan) |
| Hinode/NFI | SOT | Lyot filter | 0.2″ | Narrow-band imaging, limited lines |
| TIP-2 | VTT (0.7 m) | Slit spectropolarimeter | 0.5″ | High SNR, slow |
| IBIS | DST (0.76 m) | Dual FP | 0.2″ | Similar concept, older |
| **GFPI (later)** | GREGOR (1.5 m) | Dual FP | 0.1″ | CRISP 설계 계승 |
| **ViSP / VBI (DKIST)** | DKIST (4 m) | Slit spectrograph / Imager | 0.03″ | 차세대 |

**결론**: CRISP은 2008년 당시 **지상에서 imaging + spectropolarimetry + 고카던스**를 동시에 제공하는 유일한 장비. Hinode/SP는 우주에서 seeing-free이지만 slit 스캔의 시간 지연(~30분 per FOV) 때문에 dynamic structure 추적 불가. CRISP의 ~30 s 풀-Stokes 스캔은 결정적 차별점.

CRISP in 2008 was the only ground instrument combining **imaging + spectropolarimetry + high cadence**. Hinode/SP is seeing-free but suffers from minute-to-hour slit-scan latency, making dynamic tracking impossible. CRISP's ~30 s full-Stokes scan is the differentiator.

### Part VII: Relation to MHD Simulations / MHD 시뮬레이션과의 관계

2009년 Rempel et al. (*Science*)의 전역 sunspot MHD 시뮬레이션이 본 논문의 주요 관측 결과를 재현:
- 필라멘트 dark core 내부의 **수평 자기장** 강화.
- Filament tail의 **하향류** 자연스럽게 발생.
- Uncombed 자기장 구조가 시뮬레이션의 **자기 대류 자가 조직화** 결과로 출현.

따라서 본 Letter는 관측 발견이자 이론 검증 역할. Heinemann et al. (2007)의 이전 2D 시뮬레이션은 물리 가능성을 보였으나 전역 공간 분포까지는 재현 못함. 본 논문이 Rempel 2009 시뮬레이션의 **정량 benchmark**가 되었다.

Rempel et al. (2009, *Science*) performed global MHD simulations of sunspots that reproduced the key observations here: enhanced horizontal field in dark cores, natural tail downflows, and uncombed structure emerging as a self-organized convective state. This Letter is thus both observation and theoretical benchmark. Earlier 2D simulations (Heinemann 2007) showed physical plausibility but not global spatial structure; Rempel's 2009 run matched CRISP quantitatively.

### Part VIII: What This Paper Did NOT Resolve / 본 논문이 해결하지 못한 것

Letter 규모의 한계로 남겨진 의문들:

1. **Filament 수명과 동역학** — 30 s 스캔은 느린 dynamics에는 충분하지만 수 초 규모의 파동/점등 이벤트는 포착 못함.
2. **3D 자기장 구조** — LOS 성분만 직접 측정. Azimuth 모호성(180° degenerate)은 이 논문에서 해결 안 함.
3. **Chromospheric 연결** — Fe I 광구선만 사용, penumbra가 상층에서 어떻게 연결되는지는 Ca II 8542 Å/Hα 관측 필요.
4. **Energy transport** — Evershed flow의 질량·에너지 플럭스 정량화는 후속 연구 (Franz & Schlichenmaier 2013, Tiwari 2013).
5. **Sunspot 일생 / Evolution** — 관측은 스냅샷. Penumbra 형성·붕괴 과정은 후속 시계열 연구의 몫.

Open issues left by this Letter: filament lifetimes and wave dynamics (faster cadence needed); 3D magnetic structure (azimuth ambiguity unresolved); chromospheric connection (needs Ca II 8542 / Hα); quantitative mass/energy flux (follow-up studies); sunspot evolution on hours-to-days scales (time-series extensions).

### Part IX: From Letter to Decade of Science / 편지에서 10년의 과학으로

이 Letter 이후 CRISP과 그 파생 기기로 수행된 대표 연구:
- **de la Cruz Rodriguez et al. 2012** — Ca II 8542 Å 채층 Stokes inversion (chromospheric magnetism).
- **Schlichenmaier et al. 2010–** — penumbra 형성 과정의 time-resolved 관측.
- **Scharmer et al. 2011 (*Science*)** — penumbra 내 대류 셀 직접 관측.
- **Joshi et al. 2017** — CRISP로 magnetic twist 추출하여 flare trigger 연구.
- **Vissers et al. 2015** — Ellerman bomb의 자기장 복원.
- GREGOR/GFPI, DKIST/VBI가 같은 원리로 더 큰 구경·해상도에서 연속 수행.

A decade of science enabled: chromospheric Stokes inversion (de la Cruz Rodriguez 2012), penumbra formation (Schlichenmaier 2010+), convective cells inside penumbrae (Scharmer 2011, *Science*), flare-trigger twist (Joshi 2017), Ellerman bomb magnetism (Vissers 2015). GREGOR/GFPI and DKIST/VBI continue the program at larger apertures.

---

## 3. Key Takeaways / 핵심 시사점

1. **CRISP establishes ground-based imaging spectropolarimetry / CRISP이 지상 imaging 분광편광의 표준을 세움** — 이중 FP 에탈론 + AO + MOMFBD 조합으로 Hinode/SOT 우주 관측에 필적하는 품질을 지상에서 달성. / The dual FP etalon + AO + MOMFBD combination matches Hinode/SOT quality from the ground.

2. **Dark cores are horizontal flux tubes / Dark core는 수평 자기 관** — Stokes $V$ 약화와 $Q, U$ 강화가 동시에 필라멘트 중심에서 관측 → **자기장 inclination이 중심에서 수평에 가까움**이 직접 입증. / Weaker $V$ and stronger $Q, U$ at filament centers directly prove near-horizontal fields — dark cores = horizontal flux tubes.

3. **Filament tails host downflows / Filament tail에 하향류** — Evershed outflow의 운명에 대한 수십 년 논쟁을 해결. Tail에서 수 km/s downflow 분광 검출 → outflow가 magnetic canopy 아래로 잠수. / Spectrally detected few-km/s downflows at filament tails resolve decades of debate: the Evershed outflow dives beneath the canopy.

4. **Uncombed penumbra model vindicated / Uncombed 페넘브라 모델 입증** — Solanki & Montavon(1993)의 수평-수직 자기장 interleaving 모델이 공간 해상 관측으로 확증. MHD 시뮬레이션의 관측 제약 제공. / Solanki & Montavon's interleaved horizontal/vertical field model is confirmed at resolution, constraining MHD simulations.

5. **Two spectral lines double the diagnostic power / 두 Fe I 선이 진단 능력을 두 배로** — 6301.5 (g=1.67)와 6302.5 (g=2.5) Å의 다른 Landé factor로 자기장과 온도·속도 기여를 분리 (line-ratio method). / Different Landé factors of 6301.5 ($g=1.67$) and 6302.5 ($g=2.5$) disentangle magnetic from thermal/kinematic effects via line ratios.

6. **Imaging spectropolarimetry needs all four legs / 네 기술의 결합이 필수** — 고해상 광학 + AO + MOMFBD + tunable FP. 하나라도 빠지면 결과가 해석 불가. 현대 태양 관측의 표준 조합. / AO + MOMFBD + tunable FP + high-resolution optics must work together; missing any leg degrades interpretability.

7. **High cadence opens dynamics / 고카던스가 동역학을 연다** — CRISP의 ~30 s 스캔 주기는 flux tube lifetime(~분)보다 짧아 **시간 분해 magnetogram/Dopplergram**이 가능. Hinode/SP의 slit 스캔보다 공간-시간 동시 커버에서 우월. / At ~30 s per Stokes cycle (shorter than flux-tube lifetimes), CRISP enables time-resolved magneto/Dopplergrams, outperforming slit-scan polarimeters in spatio-temporal coverage.

8. **Benchmark for next-generation instruments / 차세대 장비의 기준점** — CRISP의 설계는 GREGOR의 **GFPI**, DKIST의 **VBI/ViSP** 에 계승. 본 논문이 정의한 과학 가능성이 4 m급 기기 과학 로드맵의 출발점. / CRISP's design is inherited by GREGOR/GFPI and DKIST/VBI/ViSP; this paper's results define the baseline science roadmap for 4-m-class instruments.

9. **Ground can match space with right technology stack / 올바른 기술 스택이면 지상이 우주에 필적** — AO + MOMFBD + CRISP 조합이 Hinode/SOT와 해상도·정밀도에서 비교 가능. 큰 구경과 고카던스에서는 지상이 유리. / The AO + MOMFBD + CRISP stack reaches Hinode/SOT-level resolution and precision; ground wins at large aperture and high cadence.

10. **Simultaneous multi-parameter imaging is a paradigm shift / 다-파라미터 동시 영상이 패러다임 전환** — 과거: intensity only, 또는 magnetogram only, 또는 spectroscopy only. 이제: 한 burst로 $(I, Q, U, V, \lambda)$ 전체 큐브 동시 획득. 분석이 모델-기반에서 데이터-기반으로 전환. / Historically intensity, magnetograms, and spectroscopy were separate; CRISP captures the full $(I, Q, U, V, \lambda)$ cube simultaneously, shifting analysis from model-driven to data-driven.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Zeeman splitting

$$
\boxed{\Delta\lambda_B = 4.67 \times 10^{-13}\, g_{\rm eff}\, \lambda^2\, B\ \text{[Å]}} \quad (\lambda\ \text{in Å},\ B\ \text{in Gauss})
$$

Fe I 6301.5, $g_{\rm eff}=1.67$: $\Delta\lambda_B = 31$ mÅ @ $B=1000$ G.
Fe I 6302.5, $g_{\rm eff}=2.5$: $\Delta\lambda_B = 46$ mÅ @ $B=1000$ G.

### 4.2 Weak-field Stokes profiles

$$
V(\lambda) = -C\, g_{\rm eff}\, \lambda^2\, B\cos\theta_B\, \frac{\partial I_0(\lambda)}{\partial \lambda}
$$

$$
Q(\lambda), U(\lambda) \propto (g_{\rm eff}\, \lambda^2\, B\sin\theta_B)^2\, \frac{\partial^2 I_0(\lambda)}{\partial \lambda^2}
$$

- $V$: linear in $B\cos\theta_B$.
- $Q, U$: quadratic in $B\sin\theta_B$.
- $\theta_B$: angle between $\vec{B}$ and line of sight.

### 4.3 Center-of-gravity magnetogram

$$
B_{\rm LOS} \propto \frac{1}{C\, g_{\rm eff}\, \lambda_0^2}\, \frac{\int V(\lambda)(\lambda - \lambda_0)\, d\lambda}{\int [I_c - I(\lambda)]\, d\lambda}
$$

Weak-field에서 $B_{\rm LOS}$의 직접 추정자.

### 4.4 Fabry–Pérot transmission

$$
T(\lambda) = \frac{1}{1 + F \sin^2(\delta/2)},\quad \delta(\lambda) = \frac{4\pi n t \cos\theta}{\lambda},\quad F = \frac{4R}{(1-R)^2}
$$

- FWHM: $\Delta\lambda = \frac{\lambda^2}{\pi n t}\frac{1-R}{\sqrt{R}}$ = **finesse**.
- CRISP 고-finesse 에탈론: $R \approx 0.93$, $t \sim$ 수 mm → $\Delta\lambda \simeq$ 6 pm @ 630 nm.
- 이중 구성: $T_{\rm total}(\lambda) = T_1(\lambda)\, T_2(\lambda)$.

### 4.5 Doppler shift to velocity

$$
\frac{\Delta\lambda}{\lambda_0} = \frac{v_{\rm LOS}}{c}
$$

- CRISP 50 mÅ sampling @ 630 nm → $\delta v \approx 24$ m/s 분해능 (SNR 제한 전 이론).

### 4.6 Line ratio (Stenflo 1973)

$$
r = \frac{V_{6301}(\lambda_1)}{V_{6302}(\lambda_1)}
$$

- 두 선 모두 같은 $B$ 경험하지만 다른 $g_{\rm eff}$로 다른 split 신호 → $r$은 saturation 여부와 $B$ 세기의 지표.
- CRISP는 두 선을 동시 관측하여 line ratio diagnostics 자동 수행.

### 4.7 Milne–Eddington Stokes model (context)

Milne–Eddington 가정: source function이 광학 두께에 선형, 흡수 계수는 파장 의존.

$$
I(\lambda) = I_0 + \mu \beta_0 \cdot K(\lambda, \vec{p})
$$

$\vec{p} = (B, \theta_B, \phi_B, v_{\rm LOS}, \Delta\lambda_D, \eta_0, a)$ — 7개 파라미터로 전체 Stokes 프로파일을 예측. 관측된 $(I, Q, U, V)(\lambda)$를 $\chi^2$ 최소화로 피팅.

Milne–Eddington assumes a linear source function in optical depth. The Stokes vector depends on 7 parameters (field strength, inclination, azimuth, LOS velocity, Doppler width, line-to-continuum opacity ratio, damping), fit by $\chi^2$ minimization to the observed $(I, Q, U, V)(\lambda)$.

### 4.8 Line ratio diagnostic

$$
R(B, T) = \frac{\Delta\lambda_{B,6302}}{\Delta\lambda_{B,6301}} = \frac{g_{6302}}{g_{6301}} = \frac{2.5}{1.67} = 1.50
$$

이상적 (weak-field) 조건에서 $V_{6302}/V_{6301} = 1.50$. 관측값이 1.50보다 작으면 saturation; 크면 사용된 템플릿 오류. Stenflo 1973의 경험적 진단.

In the ideal weak-field regime, the ratio of Stokes $V$ amplitudes $V_{6302}/V_{6301}$ equals the Landé ratio 1.50. Deviations from 1.50 signal Zeeman saturation (small ratio) or template issues (large ratio) — Stenflo's 1973 empirical diagnostic.

### 4.9 Worked example / 수치 예제

$B = 1500$ G, $\theta_B = 90°$ (horizontal field, dark core 조건), $\lambda = 6302.5$ Å:
- Splitting: $\Delta\lambda_B = 4.67 \times 10^{-13} \times 2.5 \times 6302.5^2 \times 1500 \approx 70$ mÅ.
- $V$: $\cos\theta_B = 0 → V = 0$ (수평장은 $V$를 생성하지 않음).
- $Q, U$: $\sin\theta_B = 1 → (g\lambda^2 B)^2$ 항 크게. 관측자는 **$V$ 감소 + $Q/U$ 증가**로 inclination 판별.

Doppler: $v = 5$ km/s downflow @ $\lambda_0 = 6302.5$ Å:
- $\Delta\lambda = 6302.5 \times 5000 / (3\times 10^8) = 0.105$ Å = 105 mÅ.
- CRISP 50 mÅ 스캔에서 명확히 분해.

### 4.10 Signal-to-noise requirements / SNR 요건

- Penumbra 자기장 ~1 kG에서 $V/I \sim 1$–3 %.
- $10^{-3}$ 수준 polarimetric precision 필요 → 프레임당 $\sim 10^{6}$ photons/pixel 필요.
- Full burst (Stokes × 파장 × frames)은 수백 MB → MOMFBD 후처리가 계산적으로 무거움. Wöger 2008(#21)의 병렬화가 실질적 해결.

~1 kG penumbral fields produce $V/I \sim 1$–3 %, demanding $\sim 10^{-3}$ polarimetric precision. Each pixel needs $\sim 10^6$ photons per frame. A full burst exceeds hundreds of MB; MOMFBD is computationally heavy, and parallelized reconstruction (cf. #21 Wöger 2008) is the practical enabler.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1909   Evershed — penumbral outflow
1952   Leighton — Babcock magnetograph
1968   Meyer & Schmidt — siphon flow concept
1973   Stenflo — Fe I 6301/6302 line-ratio method
1973   Skumanich & Lites — Milne-Eddington Stokes inversion
1993   Solanki & Montavon — uncombed penumbra model
1993   Thomas & Montesinos — siphon-flow penumbra model
1999   Rimmele — real-time solar AO (NSO)
2002   Scharmer et al. — SST dark-core discovery (Nature)
2005   van Noort et al. — MOMFBD
2006   Hinode/SOT launch
2007   CRISP installed at SST
2008   ★ Scharmer et al. — THIS PAPER (first CRISP science)
2009   Rempel — MHD simulations of penumbra
2011   Rimmele & Marino — SAO review
2012   GREGOR first light (GFPI inherits CRISP design)
2020   DKIST first light (VBI/ViSP inherit concepts)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Scharmer et al. (2002) "Dark cores in sunspot penumbral filaments" | Predecessor discovery / 직전 발견 | **Natural continuation** — CRISP now measures magnetic fields of the dark cores this paper discovered. / 본 논문이 자기장까지 측정. |
| Solanki & Montavon (1993) "Uncombed magnetic fields..." | Theoretical model tested / 이론 모델 | **Direct observational test** — this paper confirms the interleaved field structure. / 본 논문이 직접 검증. |
| Meyer & Schmidt (1968); Thomas & Montesinos (1993) siphon flow | Theoretical Evershed mechanism / 이론적 Evershed 메커니즘 | **Mechanism confirmed** — filament-tail downflows match siphon-flow predictions. / Tail downflow가 siphon-flow 예측 부합. |
| Stenflo (1973) Fe I 6301/6302 line-ratio | Diagnostic method / 진단 방법 | **Built into CRISP's design** — simultaneous observation of the two lines. / CRISP는 두 선을 동시 관측. |
| #20 Rimmele & Marino (2011) SAO | AO context / AO 맥락 | **Enabling technology** — CRISP requires SST's AO to reach 0.15″ resolution. / CRISP가 의존하는 AO. |
| #21 Wöger et al. (2008) KISIP | Post-processing competitor / 경쟁 후처리 | **Complementary choice** — this paper uses MOMFBD instead; both serve the same end. / 본 논문은 MOMFBD 선택; 같은 목적. |
| van Noort et al. (2005) MOMFBD | Post-processing used here / 본 논문이 사용한 후처리 | **Enabling technology** — multi-wavelength MOMFBD handles CRISP's Stokes cube. / Stokes cube 복원에 MOMFBD 사용. |
| Rempel (2009, 2012) MHD simulations of penumbra | Theory counterpart / 이론 대응 | **Observational constraints** — CRISP results constrain Rempel's simulations. / 본 관측이 시뮬레이션 제약. |
| Ichimoto et al. (2007) Hinode Evershed studies | Space-based analog / 우주 대응 | **Complementary viewpoint** — Hinode's seeing-free images verify CRISP findings. / Hinode와 상호 검증. |

---

## 7. References / 참고문헌

- Scharmer, G. B., Narayan, G., Hillberg, T. et al., "CRISP spectropolarimetric imaging of penumbral fine structure", *ApJL*, 689, L69 (2008). [DOI: 10.1086/595744]
- Scharmer, G. B., Gudiksen, B. V., Kiselman, D., Löfdahl, M. G., & Rouppe van der Voort, L. H. M., "Dark cores in sunspot penumbral filaments", *Nature*, 420, 151 (2002).
- Solanki, S. K. & Montavon, C. A. P., "Uncombed magnetic fields in penumbrae of sunspots", *A&A*, 275, 283 (1993).
- Thomas, J. H. & Montesinos, B., "Siphon flows in isolated magnetic flux tubes. IV. Penumbral flows", *ApJ*, 407, 398 (1993).
- Evershed, J., "Radial movement in sun-spots", *MNRAS*, 69, 454 (1909).
- Stenflo, J. O., "Magnetic-field structure of the photospheric network", *Solar Physics*, 32, 41 (1973).
- Meyer, F. & Schmidt, H. U., "Magnetodynamical vortex flow as a model for solar spicules", *Zeitschrift für Angewandte Mathematik und Mechanik*, 48, 218 (1968).
- van Noort, M., Rouppe van der Voort, L., & Löfdahl, M. G., "Solar Image Restoration by use of MOMFBD", *Solar Phys.*, 228, 191 (2005).
- Heinemann, T., Nordlund, Å., Scharmer, G. B., & Spruit, H. C., "MHD simulations of penumbra fine structure", *ApJ*, 669, 1390 (2007).
- Rempel, M., Schüssler, M., Cameron, R. H., & Knölker, M., "Penumbral structure and outflows in simulated sunspots", *Science*, 325, 171 (2009).
- Ichimoto, K. et al., "Twisting motions of sunspot penumbral filaments", *Science*, 318, 1597 (2007).
- Rimmele, T. R. & Marino, J., "Solar Adaptive Optics", *Living Rev. Solar Phys.*, 8, 2 (2011).
