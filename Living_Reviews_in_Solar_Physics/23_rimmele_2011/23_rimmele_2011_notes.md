---
title: "Solar Adaptive Optics"
authors: Thomas R. Rimmele, Jose Marino
year: 2011
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2011-2"
topic: Living_Reviews_in_Solar_Physics
tags: [adaptive-optics, wavefront-sensing, shack-hartmann, solar-telescope, atmospheric-turbulence, MCAO, GLAO, DKIST]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 23. Solar Adaptive Optics / 태양 적응 광학

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 리뷰는 **태양 적응 광학(Solar Adaptive Optics, SAO)** 의 원리, 역사, 공학적 구현, 성능 한계, 향후 전망을 총체적으로 정리한 논문이다. SAO는 지상 태양 망원경이 대기 난류로 인한 파면 왜곡을 실시간 보정하여 **회절 한계(diffraction-limited)** 관측을 달성하게 해준 핵심 기술이다. 저자들은 (1) Kolmogorov 난류 이론과 Fried parameter $r_0$, Greenwood frequency $f_G$, seeing time $\tau_0$ 등 AO 설계의 수학적 기반을 정의하고, (2) 야간 천문학 AO와 달리 **확장 광원(granulation)** 을 기준으로 파면을 감지해야 하는 solar AO의 고유 난제를 분석하며, (3) 이 난제를 해결한 **상관형 Shack-Hartmann 파면 감지기(Correlating SHWFS)** 의 발명과 알고리즘을 설명한다. 또한 (4) DST의 AO76 시스템을 구체적 구현 사례로 제시하면서 9종의 wavefront error 구성요소($\sigma^2_{\text{fit}}, \sigma^2_{\text{aliasing}}, \sigma^2_\theta, \sigma^2_{\text{BW}}, \sigma^2_{\text{wfs}}, \sigma^2_{\text{wfs,aniso}}, \sigma^2_{\text{ncp}}, \sigma^2_{T/T}, \sigma^2_{\text{other}}$)를 모두 수식화하고, (5) AO telemetry로부터 PSF를 추정하여 post-facto deconvolution(speckle, phase diversity, MOMFBD)과 결합하는 기법을 제시한다. 마지막으로 (6) 4m급 ATST(현 DKIST)·EST·NLST 등 차세대 망원경을 위한 **extreme AO(>1300 actuators)**, **MCAO(Multi-Conjugate AO)**, **GLAO(Ground-Layer AO)** 의 설계 이슈와 성능 시뮬레이션을 제공한다. 이 논문은 결국 "소규경 태양 망원경의 AO 시대(2000-2010)에서 대구경·광시야 AO 시대(2020s-)로의 전환점"에서 쓰인 **통합 기준 문서**이다.

### English
This review comprehensively summarizes the principles, history, engineering implementation, performance limits, and future directions of **Solar Adaptive Optics (SAO)**. Solar AO is the enabling technology through which ground-based solar telescopes achieve **diffraction-limited** imaging by correcting atmospheric-turbulence-induced wavefront distortion in real time. The authors (1) lay down the mathematical foundation — Kolmogorov turbulence theory, the Fried parameter $r_0$, Greenwood frequency $f_G$, and seeing time $\tau_0$; (2) analyze the challenges that set solar AO apart from night-time AO, namely the need to sense the wavefront on an **extended, low-contrast source (granulation)** rather than a point star; (3) describe the **Correlating Shack-Hartmann Wavefront Sensor (CSHWFS)** that solved this problem via real-time cross-correlation of subaperture images; (4) take the DST AO76 system as a case study and build a full nine-term wavefront error budget ($\sigma^2_{\text{fit}}, \sigma^2_{\text{aliasing}}, \sigma^2_\theta, \sigma^2_{\text{BW}}, \sigma^2_{\text{wfs}}, \sigma^2_{\text{wfs,aniso}}, \sigma^2_{\text{ncp}}, \sigma^2_{T/T}, \sigma^2_{\text{other}}$); (5) develop the method for estimating a long-exposure PSF from AO telemetry and combining it with post-facto deconvolution (speckle, phase diversity, MOMFBD); and (6) present simulations and design issues for next-generation 4 m-class telescopes (ATST/DKIST, EST, NLST) including **extreme AO (>1300 actuators), MCAO, and GLAO**. The paper stands as the **authoritative reference document** written at the inflection point between the small-aperture solar-AO era (2000–2010) and the 4 m MCAO era (2020s–).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

#### 한국어 (pp. 5–10)
**과학적 동기**: 태양 물리의 핵심 질문 — 태양 복사강도 변동(지구 기후 영향), 다이나모 자기장 생성, 자기에너지 저장·방출(플레어·CME) 등 — 은 모두 태양 대기의 **미세 구조**를 분해해야 답할 수 있다. 지배적 스케일은 두 가지: **압력 scale height (~70 km ≈ 0.1")** 와 **광자 평균자유경로**. 따라서 0.1" 이상의 각해상도가 필수인데, 정교한 MHD 시뮬레이션(Cattaneo et al. 2003; Nordlund & Stein 2009)은 수십 km, 즉 수십 milli-arcsec 스케일의 자기 구조까지 예측한다. 그림 1(DST)과 그림 2(SST)는 AO와 post-processing을 결합해 sunspot의 penumbral filament, granulation, dark core를 0.1" 스케일로 분해한 대표 예. 그림 3(ATST용 시뮬레이션)은 보정 모드 수가 증가함에 따라 Strehl이 0.001 → 0.554로 증가하며, 최종 PSF 품질이 **정량적 측정의 유효성**을 결정함을 보인다. 새로운 4m ATST는 AO 없이는 목표를 달성할 수 없고, AO를 통해 대기 한계를 극복한다.

#### English (pp. 5–10)
**Scientific motivation**: The core questions of solar physics — solar-luminosity variations that drive Earth's climate, dynamo generation of magnetic field, storage and release of magnetic energy in flares and CMEs — all demand resolution of the Sun's fine structure. Two dominant scales set the bar: the **photospheric pressure scale height (~70 km ≈ 0.1")** and the photon mean free path. Thus angular resolution better than 0.1" is required; sophisticated MHD simulations (Cattaneo et al. 2003; Nordlund & Stein 2009) predict magnetic structure down to tens of kilometers — tens of milli-arcseconds. Figures 1 (DST) and 2 (SST) show how AO + post-processing resolves penumbral filaments, granulation, and dark cores at 0.1". Figure 3 (ATST simulation) demonstrates that Strehl rises from 0.001 to 0.554 as the number of corrected modes goes from 0 to 2000, and that final PSF quality determines whether **quantitative measurements** are meaningful. The coming 4 m ATST cannot meet its science goals without AO; AO is what lets it beat the atmosphere.

### Part II: AO Basics (§2.1–2.3) / 적응 광학 기초

#### 대기 난류 / Atmospheric turbulence (§2.1, pp. 11–14)

**한국어**: Kolmogorov (1941)가 제안한 난류 이론이 기반. 에너지는 outer scale에서 주입되어 inertial range를 거쳐 inner scale에서 소산된다. 온도 변동 $\Phi_T$, 굴절률 변동 $\Phi_N$의 공간 파워 스펙트럼은 멱법칙 $\kappa^{-5/3}$(1D) 또는 $\kappa^{-11/3}$(3D)을 따른다. **굴절률 구조 함수** $D_n(\rho)=C_n^2\rho^{2/3}$는 두 지점 사이의 굴절률 변동의 분산이며, $C_n^2$은 난류 강도 지표다. **파면 구조 함수** $D_\varphi(\rho,h) = 2.914\,k^2\,\delta h\,C_n^2(h)\,\rho^{5/3}$는 두 퓨필 지점 사이의 위상 차이 분산이고, 전 대기에 적분하면 $D_\varphi(\rho) = 2.914\,k^2(\sec\gamma)\rho^{5/3}\int C_n^2(h)\,dh$가 된다. 이로부터 **Fried parameter** 정의: $r_0 \equiv [0.423\,k^2(\sec\gamma)\int C_n^2(h)dh]^{-3/5}$ — 대기 seeing을 **한 숫자**로 요약하며, $r_0\propto\lambda^{6/5}$로 적외선에서 더 관대하다. 장노출 구조 함수 $D_{\text{long}}(\rho)=6.88(\rho/r_0)^{5/3}$, OTF $\text{OTF}_{\text{atm}}=\exp[-3.44(\rho/r_0)^{5/3}]$. Taylor의 "얼어붙은 난류" 가설 하에 $\tau_0\equiv r_0/v$ — 가시광 태양 AO에서 $r_0\sim10$ cm, $v\sim10$ m/s → $\tau_0\sim10$ ms, 따라서 제어 대역폭 **>100 Hz 필요**.

**English**: Based on Kolmogorov's (1941) turbulence theory. Energy injects at an outer scale, cascades through an inertial range, and dissipates at an inner scale. The power spectra of temperature and refractive-index fluctuations follow power laws $\kappa^{-5/3}$ (1D) or $\kappa^{-11/3}$ (3D). The **refractive-index structure function** $D_n(\rho) = C_n^2\rho^{2/3}$ describes the variance of refractive-index differences between two points, with $C_n^2$ the turbulence strength. The **phase structure function** $D_\varphi(\rho,h) = 2.914\,k^2\,\delta h\,C_n^2(h)\,\rho^{5/3}$ gives the variance of phase differences between two pupil points; integrated over the atmosphere, $D_\varphi(\rho) = 2.914\,k^2(\sec\gamma)\rho^{5/3}\int C_n^2(h)\,dh$. This yields the **Fried parameter** $r_0 \equiv [0.423\,k^2(\sec\gamma)\int C_n^2(h)dh]^{-3/5}$ — a one-number summary of seeing, scaling as $r_0\propto\lambda^{6/5}$ (IR is more forgiving). The long-exposure structure function becomes $D_{\text{long}}(\rho)=6.88(\rho/r_0)^{5/3}$, and the OTF is $\text{OTF}_{\text{atm}}=\exp[-3.44(\rho/r_0)^{5/3}]$. Under Taylor's frozen-turbulence hypothesis, $\tau_0\equiv r_0/v$ — for solar visible observations with $r_0\sim10$ cm and $v\sim10$ m/s, $\tau_0\sim10$ ms, demanding AO control bandwidth **>100 Hz**.

#### AO 시스템 설계 / Design of an AO system (§2.2, pp. 14–15)

**한국어**: 기본 3개 구성요소: (1) **파면 감지기(WFS)** — 전형적으로 pupil plane의 여러 지점에서 파면 기울기를 측정 (Shack-Hartmann WFS), (2) **파면 보정자(DM)** — actuator로 거울 표면을 변형, (3) **재구성자(reconstructor)** — WFS 출력으로 DM actuator 명령 계산, 폐루프 서보 알고리즘. 자유도(Degrees of Freedom) 요구치: $\text{DOF} \approx (D/r_0)^2$. 4 m 망원경, $r_0=10$ cm → DOF ≈ 1600. 높은 Strehl = 많은 DM actuator + 높은 sampling density + 충분한 temporal bandwidth.

**English**: Three basic components: (1) **wavefront sensor (WFS)** — typically measures wavefront gradients at many pupil-plane locations (Shack-Hartmann WFS); (2) **wavefront corrector (DM)** — actuators deform the mirror face; (3) **reconstructor** — processes WFS output into DM actuator commands via a closed-loop servo. Required degrees of freedom: $\text{DOF} \approx (D/r_0)^2$. A 4 m telescope at $r_0=10$ cm needs ≈1600 DOF. High Strehl demands many actuators, high sampling density, and sufficient temporal bandwidth.

#### 태양 AO의 고유 난제 / Solar AO challenges (§2.3, pp. 15–18)

**한국어**: 야간 AO와 가장 큰 차이는 **seeing 악조건 + 확장 광원 + 가시광 관측**의 삼중고.
1. **주간 seeing**: 햇볕으로 지표 가열 → 근지표 난류 강함. Sac Peak DST 중앙값 $r_0(500\text{ nm}) \approx 8.7$ cm (Brandt 1987), ATST site survey는 <5 cm로 재측정 (Hill 2004, 2006). 그림 6은 500초 시계열에서 $r_0$가 5–30 cm 사이에서 크게 변동함을 보인다.
2. **시간 변동성**: 그림 7은 Zernike Z4(astigmatism) PSD의 꺾임점이 ~10 Hz, Z24는 ~20 Hz로, Greenwood frequency가 radial mode number에 비례하여 증가함을 보인다. 200 Hz 이상에서는 노이즈가 우세해진다.
3. **확장 타겟**: 별이 아니라 granulation 위에서 lock. Granulation 대비도는 ~13% (D=100 cm, full aperture) 하지만 subaperture 수준에서는 1–3%까지 떨어짐(그림 22 참조).
4. **LGS 불가능**: 태양 원반이 너무 밝아서 laser spot을 투사해도 안 보임. 예외적으로 corona 관측에는 LGS가 장래에 유용할 수 있음.

**English**: The main differences from night-time AO form a triple whammy — **bad daytime seeing + extended source + visible-wavelength science**.
1. **Daytime seeing**: ground heating drives strong near-ground turbulence. Median $r_0(500\text{ nm}) \approx 8.7$ cm at Sac Peak DST (Brandt 1987), revised to <5 cm by ATST site survey (Hill 2004, 2006). Figure 6 shows $r_0$ fluctuating between 5 and 30 cm over 500 s.
2. **Temporal variability**: Figure 7 shows PSD break points near 10 Hz for Zernike Z4 (astigmatism) and ~20 Hz for Z24 — the Greenwood frequency rises with radial mode number. Above 200 Hz noise dominates.
3. **Extended target**: lock on granulation, not a star. Granulation rms contrast is ~13% at full aperture (D=100 cm) but drops to 1–3% at the subaperture level (Figure 22).
4. **LGS impractical**: the solar disk is too bright for laser-projected spots to be visible. A possible future exception is corona observation.

### Part III: Brief History of Solar AO (§3, pp. 19–24)

**한국어**: 최초 태양 AO 실험은 1979-80년 DST의 Hardy (shearing interferometer + 21-actuator DM). 이후 NSO는 von der Lühe의 focal plane LCD mask WFS(Foucault knife-edge 테스트에서 유래) 시도 — 확장 광원에는 S/N 문제. 1980년대 말 Lockheed는 19-element segmented DM + quad-cell SHWFS (그림 10) — 작은 고대비 타겟(pore)만 가능. 돌파구는 1998년 DST의 **NSO low-order AO** — 상관형 SHWFS(24 subapertures, 97-actuator Xinetics DM) + DSP 기반 실시간 상관. 양호한 seeing에서 Strehl 0.6 달성, granulation에서 최초로 lock 성공. 이후 SST(La Palma, 1999), VTT/KAOS(Tenerife, 2002), DST AO76 고차 시스템이 차례로 가동. 태양 AO의 90% 난제는 "granulation 위에서 파면을 측정할 수 있는가"였고, **상관 추적(correlation tracking)** 이 이를 해결했다.

**English**: The first solar AO experiment was Hardy's 1979–80 shearing-interferometer test at the DST with a 21-actuator DM. NSO then pursued von der Lühe's focal-plane LCD-mask WFS (derived from the Foucault knife-edge test) — but S/N was poor on extended sources. In the late 1980s Lockheed built a 19-element segmented DM + quad-cell SHWFS (Figure 10), usable only on small high-contrast targets (pores). The breakthrough came in 1998 with the **NSO low-order AO** at the DST: a correlating SHWFS (24 subapertures, 97-actuator Xinetics DM) and DSP-based real-time correlation. It reached Strehl ~0.6 in good seeing and was the first system to lock on granulation. SST (La Palma, 1999), VTT/KAOS (Tenerife, 2002), and the high-order DST AO76 followed. Ninety percent of the solar-AO problem was "can we measure the wavefront on granulation?", and **correlation tracking** was the answer.

### Part IV: Correlating Shack-Hartmann WFS (§4, pp. 25–27)

**한국어**: 핵심 알고리즘: 각 subaperture가 렌즈렛 배열로 퓨필을 나누어 20×20 픽셀의 granulation 이미지($I_M$)를 형성. 임의로 선택한 참조 subaperture 이미지($I_R$)와 실시간 **2D cross-correlation** 계산:
$$CC(\vec\Delta_i) = \sum\sum I_M(\vec{x}) \cdot I_R(\vec{x}+\vec\Delta_i) \quad \text{(Eq. 15)}$$
상관 피크 위치 $\vec\Delta_i$가 해당 subaperture의 local tilt에 해당. Subpixel 정밀도는 피크 주변에 **포물선 피팅**. 대안으로 Square Difference Function(SDF), Absolute Difference Squared(ADS)도 사용 — 성능 유사하나 CPU 친화적. FOV는 보통 10"×10" (20×20 pixels)이나, 너무 크면 anisoplanatism으로 파면 방향 평균 효과(§6.1.6). 작으면 granule이 충분하지 않아 상관 피크가 불안정. **Subaperture 최소 크기는 ~8 cm** — 그 이하에서는 회절에 의해 granulation 대비가 소실됨(Berkefeld & Soltau 2010). 잠재적 대안: Phase Diversity WFS — 전 퓨필 WFS이므로 subaperture diffraction 제약 없음 (Paxman et al. 2007).

**English**: The core algorithm: lenslets divide the pupil into subapertures, each forming a 20×20-pixel granulation image ($I_M$). The real-time **2D cross-correlation** with a reference subaperture image $I_R$ is
$$CC(\vec\Delta_i) = \sum\sum I_M(\vec{x}) \cdot I_R(\vec{x}+\vec\Delta_i) \quad \text{(Eq. 15)}.$$
The correlation-peak location $\vec\Delta_i$ is the local wavefront tilt. Sub-pixel precision comes from **parabolic fitting** around the peak. Alternatives — Square Difference Function (SDF) or Absolute Difference Squared (ADS) — offer similar accuracy and are more CPU-friendly. Typical WFS FOV is 10″×10″ (20×20 pixels); larger FOVs average over multiple directions (anisoplanatism, §6.1.6), smaller ones contain too few granules to produce a stable peak. The **minimum subaperture size is ~8 cm** — below this, diffraction smears out granulation contrast (Berkefeld & Soltau 2010). A future alternative: Phase Diversity WFS, a full-aperture WFS free of subaperture diffraction limits (Paxman et al. 2007).

### Part V: AO System Implementation — DST AO76 (§5, pp. 28–32)

**한국어**: DST AO76 시스템은 상용 부품을 병렬 처리로 조합한 사례.
- **WFS**: 76 subapertures × 20×20 픽셀 = granulation 이미지; subaperture 크기 $d=7.5$ cm; 전형적 WFS FOV 10"×10"; 2500 fps; 40개 DSP로 10개 파이프라인에서 상관 계산.
- **DM**: 상용 Xinetics 97-actuator continuous faceplate DM — actuator 간 crosstalk <10%.
- **교정(Calibration)**: prime focus에 자동 aperture wheel — field stops, resolution target, pinhole, 단모드 광섬유에서 레이저 공급. 간섭계로 non-common path 오차를 측정해 DM을 평탄화하는 방식으로 수정.
- **Tip/tilt 분리**: 높은 분산을 갖는 tip/tilt 모드를 고대역으로 제어하려고 별도의 작은 tip/tilt 거울 사용 — 가장 효율적인 오차 제거.
- **처리 단계**: flat/dark → intensity gradient 제거(주변 limb) → 76개 subaperture × 상관 → parabola 피팅으로 subpixel shift → global tilt 분리(tip/tilt 거울로) → reconstruction matrix → PI servo → actuator 전압.

**English**: The DST AO76 system illustrates implementation with off-the-shelf parts and parallel processing.
- **WFS**: 76 subapertures of 20×20 pixels each; $d=7.5$ cm per subaperture; typical WFS FOV 10″×10″; 2500 fps; cross-correlation computed on 40 DSPs across 10 pipelines.
- **DM**: commercial Xinetics 97-actuator continuous-faceplate mirror with <10% actuator cross-talk.
- **Calibration**: an automated aperture wheel at prime focus carries field stops, a resolution target, a pinhole, and a laser feed via single-mode fiber. A laser interferometer measures non-common-path errors and flattens the DM.
- **Split tip/tilt**: a separate small tip/tilt mirror handles the high-variance tip/tilt modes at high bandwidth — the single most effective error reduction.
- **Processing chain**: flat/dark → remove intensity gradient (near the limb) → cross-correlate 76 subapertures → parabolic subpixel fit → split out global tilt (to tip/tilt) → reconstruction matrix → PI servo → actuator voltages.

### Part VI: Wavefront Error Budget (§6, pp. 33–48) — 핵심 섹션

**한국어**: 총 잔류 파면 오차는 독립적 기여도의 RSS(root-sum-square):
$$\sigma^2_{\text{tot}} = \sigma^2_{\text{BW}} + \sigma^2_\theta + \sigma^2_{\text{fit}} + \sigma^2_{\text{aliasing}} + \sigma^2_{\text{wfs}} + \sigma^2_{\text{wfs,aniso}} + \sigma^2_{\text{ncp}} + \sigma^2_{T/T} + \sigma^2_{\text{other}} \quad \text{(Eq. 26)}$$

| 오차 항 / Error term | 식 | 주도 요인 / Driver |
|---|---|---|
| Fitting | $\sigma^2_F = a(d/r_0)^{5/3}$, $a=0.28$ | DM 액추에이터 간격 $d$ 대비 $r_0$ |
| Aliasing | $\sigma^2 = 0.08(d/r_0)^{5/3}$ (≈ 30% of fitting) | WFS 공간 표본화 한계 |
| Angular anisoplanatism | $\sigma^2_\theta = (\theta/\theta_0)^{5/3}$, $\theta_0 = [2.914\,k^2(\sec\gamma)^{8/3}\int C_n^2(h)h^{5/3}dh]^{-3/5}$ | 시야 각 $\theta$와 등화면각 $\theta_0$ |
| Bandwidth | $\sigma^2_{\text{BW}} = (f_G/f_S)^{5/3}$, $f_G \approx 0.427\,v/r_0$ | Greenwood 주파수 $f_G$와 시스템 대역폭 $f_S$ |
| WFS measurement noise | $\sigma^2_x = 5m^2\sigma_b^2 / (4n_r^2\sigma_i^2)$ (Michau 1993) | 배경 노이즈, 이미지 대비 |
| WFS anisoplanatism | (Wöger & Rimmele 2009) | WFS FOV 내 다방향 평균 |
| Non-common path | 0.5–1 nm per meter optical path | 빔 분할 후 광학 경로 차이 |
| Tip/tilt | (별도 예산) | CT sensor 노이즈, BW |

**등화면 각도 핵심 공식** (Eq. 19):
$$\theta_0 = \left[2.914\,k^2(\sec\gamma)^{8/3}\int C_n^2(h)\,h^{5/3}\,dh\right]^{-3/5}$$
주목: 적분 안에 **$h^{5/3}$** 가중치 — 고도가 높은 난류가 훨씬 중요. 제트 스트림(h~10 km)이 있으면 isoplanatic patch가 수 arcsec까지 줄어듦. 지표층 지배 사이트는 patch가 훨씬 큼. $\theta_0 \propto \lambda^{6/5}$.

**Haleakala vs. Mt. Graham 시뮬레이션** (Table 1, Figs. 20–21): 같은 $r_0$=10 cm라도 Haleakala(고층 난류 5% of total)는 $r_0=20$ cm에서 30" FOV에 걸쳐 Strehl>0.5; Mt. Graham(고층 40%)은 10" FOV 밖에서 Strehl<0.2. → **ATST가 Haleakala로 선택된 이유**.

**파면 감지기 노이즈 (Eq. 24)**:
$$\sigma^2_x = \frac{5m^2\sigma^2_b}{4n^2_r\sigma^2_i} \quad \text{(waves}^2\text{)}$$
$\sigma_b$는 배경 노이즈, $n_r^2$는 subaperture 이미지 픽셀 수, $\sigma_i/I_{\text{mean}}$은 rms 이미지 대비. **대비가 S/N의 근본 한계**. 그림 22: well depth 40 ke⁻ 이상에서는 성능 이득이 미미. 실제 측정 tilt noise: sunspot/good seeing 5–8 nm, granulation/good-excellent 15–25 nm.

**PSF 추정** (§6.3, Eqs. 27–39): AO 보정 후 PSF는 **회절 한계 core + seeing-limited halo**. 장노출 OTF는 3개 독립 항의 곱:
$$\text{OTF}_{\text{ao}}(\vec\rho/\lambda) = \text{OTF}_{\phi_{e\parallel}}\cdot\text{OTF}_{\phi_{e\perp}}\cdot\text{OTF}_{\text{tel}} \quad \text{(Eq. 29)}$$
$\text{OTF}_{\phi_{e\parallel}}$는 보정된 KL(Karhunen-Loève) 모드의 잔차(WFS telemetry로부터), $\text{OTF}_{\phi_{e\perp}}$는 보정되지 않은 고차 모드(Kolmogorov 통계 + $r_0$ 추정치), $\text{OTF}_{\text{tel}}$은 회절 한계. Marino (2007) 방법으로 DST에서 Sirius에 lock한 후 실제 별 PSF와 추정 PSF를 비교 — 그림 25 훌륭한 일치.

**Strehl vs r_0** (그림 26): 실측으로 확인: $r_0=20$ cm → $S\approx 0.9$, $r_0=4$ cm → $S\approx 0.3$. 이론 곡선(fitting+aliasing+BW+WFS noise)과 잘 맞으나, 때로 일부 데이터가 non-optimal 분기 (미해명).

**English**: Total residual variance is the RSS of independent contributions (Eq. 26 above). Key points:
- **Fitting** scales as $(d/r_0)^{5/3}$ with coefficient 0.28 for a continuous-facesheet DM.
- **Aliasing** is ~30% of fitting error.
- **Angular anisoplanatism** is crucial: $\theta_0$ involves an $h^{5/3}$ weight, so high-altitude turbulence dominates. A jet stream at 10 km can shrink $\theta_0$ to a few arcseconds (Fig. 19).
- **Haleakala vs. Mt. Graham**: for the same $r_0$, Haleakala (5% power above 6 km) gives Strehl > 0.5 over 30″ FOV at $r_0=20$ cm, whereas Mt. Graham (40%) collapses beyond 10″ — this motivated Haleakala as the ATST site.
- **WFS noise** (Eq. 24) fundamentally tracks $1/(\text{image contrast})^2$; measured tilt noise is 5–8 nm on sunspots, 15–25 nm on granulation in good seeing.
- **PSF estimation** from telemetry (Eqs. 27–39): decompose residual phase into corrected KL modes ($\phi_{e\parallel}$) and uncorrected high-order modes ($\phi_{e\perp}$); three-term OTF product (Eq. 29). Validated against Sirius (Fig. 25) with excellent agreement.
- **Strehl vs $r_0$** (Fig. 26): theoretical and measured Strehl agree well; $S\sim0.9$ at $r_0=20$ cm, $S\sim0.3$ at $r_0=4$ cm.

### Part VII: Post-Facto Processing (§7, pp. 49–54)

**한국어**: AO는 부분 보정일 뿐. Post-facto 기법은 잔여 파면 오차를 제거하고 전 FOV에 걸쳐 균일한 품질을 제공.
- **Speckle interferometry** (Wöger et al. 2008): 짧은 노출로 seeing을 "frozen"시킨 여러 이미지로 Fourier 도메인에서 true image 복원. AO telemetry를 이용하면 field-dependent 보정까지 가능.
- **Phase diversity / phase-diverse speckle** (Löfdahl & Scharmer 1994): focused + defocused 이미지 쌍으로 pupil phase를 함께 추정.
- **MFBD / MOMFBD** (van Noort et al. 2005): Multi-Object Multi-Frame Blind Deconvolution — 여러 파장/편광 데이터의 다중 프레임을 공동 deconvolve. 대표 사례: Schlichenmaier et al. 2010의 4h40m penumbra 형성 시퀀스.
- **Long-exposure deconvolution**: AO telemetry PSF로 장노출 narrow-band 이미지 deconvolution. 그림 31: $r_0=5.4$ cm ($S=0.46$) 와 $r_0=16.5$ cm ($S=0.88$) 이미지가 deconvolution 후 대비가 거의 같아짐. 그림 33: dopplergram의 intensity-velocity crosstalk 현저히 감소.

**English**: AO provides only partial correction; post-processing removes residual wavefront errors and delivers uniform quality across the FOV.
- **Speckle interferometry** (Wöger et al. 2008) — short exposures "freeze" seeing; Fourier-domain recovery of the true image; can fold in AO telemetry for field-dependent correction.
- **Phase diversity / phase-diverse speckle** (Löfdahl & Scharmer 1994) — pairs of focused + defocused images jointly estimate pupil phase.
- **MFBD / MOMFBD** (van Noort et al. 2005) — Multi-Object Multi-Frame Blind Deconvolution jointly deconvolves multi-wavelength / multi-polarization data across frames.
- **Long-exposure deconvolution**: Fig. 31 shows that after deconvolution images taken at $r_0=5.4$ cm ($S=0.46$) and $r_0=16.5$ cm ($S=0.88$) have nearly identical contrast; Fig. 33 shows dopplergram intensity–velocity crosstalk largely removed.

### Part VIII: Operational Solar AO Systems (§8, pp. 55–58)

**한국어**:
| 시스템 | 망원경 | WFS | DM | 비고 |
|---|---|---|---|---|
| SST AO | 97 cm SST, La Palma | 37-sub correlating SH | 37-el bimorph | Penumbral dark core 발견 (Scharmer 2002, 2005) |
| KAOS | 70 cm VTT, Tenerife | 36-sub ($d=10$ cm) | 35-el bimorph | 100 Hz BW, Strehl 0.7 at $r_0=20$ cm |
| DST AO76 | 76 cm DST, Sac Peak | 76-sub ($d=7.5$ cm) | 97-actuator Xinetics | 고차, NST에 복제 |
| McMath-Pierce AO | 1.5 m | 37 sub | 37-actuator membrane | 저비용 IR 전용 |
| 개발 중 | Hida 60 cm, CSUN, THEMIS | 다양 | 다양 | 소규모 업그레이드 |

SST는 큰 구경 + 37 mode 부분 보정 + post-processing으로 최고 해상도 이미지 다수. KAOS는 설계가 SST와 유사(상관형 SHWFS + bimorph DM). McMath-Pierce는 적외선(더 큰 $r_0$) 전용으로 저비용.

**English**: Key operational systems summarized in the table above. SST has produced many of the highest-resolution solar images via large aperture + 37-mode partial correction + post-processing. KAOS parallels SST (correlating SHWFS + bimorph DM). McMath-Pierce targets IR only (larger $r_0$) with lower cost.

### Part IX: Future Developments (§9, pp. 59–75) — 두 번째 핵심 섹션

#### 9.1 Large Aperture AO / 대구경 AO

**한국어**: 1.5 m GREGOR (196 actuators, 156 sub, d=10 cm, 130 Hz) 와 1.6 m NST (357 actuators, 308 sub, d=8.4 cm, 130 Hz)가 2010년대 초 commissioning. ATST (4 m, 2018 first light 목표 → 실제 DKIST 2020)의 High-Order AO (HOAO)는 설계 단계에서 1313 actuators + 1280 subapertures였으나 error budget 압력으로 1700 actuators로 증가. 과학 요구: $r_0(500)>7$ cm에서 Strehl>0.3, $r_0(630)>20$ cm에서 Strehl>0.6. 열 관리가 핵심 공학 이슈 — 4 m 구경 → tip/tilt 거울 표면에 100 W/m² 흡수 → 수십 도 상승 가능 → 능동 냉각 필수.

**English**: GREGOR (1.5 m, 196 actuators) and NST (1.6 m, 357 actuators) entered commissioning around 2010. The 4 m ATST HOAO was originally designed with 1313 actuators + 1280 subapertures but grew to 1700 to accommodate error-budget pressure. Science requirements: $S>0.3$ at $r_0(500)>7$ cm and $S>0.6$ at $r_0(630)>20$ cm. Thermal management is a key engineering issue — the 4 m aperture deposits ~100 W/m² on the tip/tilt mirror, requiring active cooling.

#### 9.1.3 Anisoplanatism — 대구경에서의 심각한 난제

**한국어**: 4 m 망원경 + 1300 actuator AO + Haleakala 프로파일 시뮬레이션(그림 42): vertical pointing $r_0=20$ cm에서도 10" off-axis에서 $S$ < 0.2로 급감. Mt. Graham에서는 훨씬 더 나빠 — vertical $S$ 거의 0. Isoplanatic patch: $r_0=20$ cm, zenith 45° Haleakala → 10"; $r_0=10$ cm → 3". 결론: **MCAO가 필수**.

**English**: Simulations for a 4 m telescope + 1300-actuator AO on Haleakala (Fig. 42) show Strehl collapsing to <0.2 at 10″ off-axis even at vertical pointing with $r_0=20$ cm. On Mt. Graham, even vertical pointing yields Strehl ≈ 0. Isoplanatic patch at $r_0=20$ cm, 45° zenith: ~10″ Haleakala, ~3″ at $r_0=10$ cm. Conclusion: **MCAO is essential**.

#### 9.1.4 Chromatic Anisoplanatism / 색수차성 비등화성

**한국어**: 다른 파장의 빛은 대기에서 다른 경로로 굴절 → 다른 난류 부피 통과. 가시광 태양 관측은 높은 zenith 각에서 ~1" atmospheric dispersion 발생 (430 nm vs 500 nm). 그림 46: WFS가 500 nm에서 감지하고 science가 430 nm로 관측하면 zenith <70°에서 Strehl 심각한 감소. 해결: WFS의 primary 과학 파장 선택, 또는 다파장 WFS.

**English**: Different wavelengths refract along different paths, sampling different turbulence volumes. Visible solar observations at high zenith angles suffer ~1″ atmospheric dispersion between 430 and 500 nm. Figure 46: sensing at 500 nm but imaging at 430 nm severely reduces Strehl at zenith <70°. Fix: pick WFS wavelength near the primary science band, or use a multi-wavelength WFS.

#### 9.2 MCAO / 다중 결합 AO

**한국어**: 원리(그림 47): 여러 고도의 주 난류 층에 **conjugate된 여러 DM**을 사용. 다중 "guide field"(granulation의 여러 영역)를 동시에 샘플링하여 **tomography**로 3D 난류 분포 추정 → 각 DM에 최적 보정 분배 → **넓은 FOV에서 회절 한계**.
- **야간 AO에는 다중 LGS 필요** — 솔라는 granulation이 "자연 guide field" 역할 → **복잡성 감소**.
- 실제 성과 (그림 50): VTT KIS MCAO (Langlois 2004; Berkefeld 2005; von der Lühe 2005) + DST MCAO (Rimmele 2009, 2010a). DST 5-guide "asterism" → 40-45" FOV에서 residual image motion <0.01" rms (vs. 전통적 AO의 <10").
- GREGOR MCAO (그림 51): 3 DMs conjugated to 0 km, 8 km, 25 km; first light 후 설치 예정.
- EST (4 m, 그림 52–53): 5 DMs at 0, 5, 9, 15, 30 km. 시뮬레이션: 60" FOV에서 $r_0=20$ cm, zenith 0°에서 $S\sim0.6$; zenith 60°에서는 $S\sim0.2$로 붕괴. → MCAO는 **near-zenith 관측**에 가장 유리 (하지만 태양 관측은 보통 오전 seeing이 가장 좋음 — **trade-off**).

**English**: Principle (Fig. 47): multiple DMs **conjugated to the dominant turbulence altitudes**, fed by multiple "guide fields" (granulation regions) for tomographic 3D turbulence reconstruction → wide-field diffraction-limited imaging.
- Night-time MCAO needs multiple LGS; solar MCAO gets its "guide stars" free from the omnipresent granulation — **complexity reduced**.
- Results (Fig. 50): VTT KIS MCAO (Langlois 2004; Berkefeld 2005; von der Lühe 2005); DST MCAO (Rimmele 2009, 2010a). The DST five-guide "asterism" achieves residual image motion <0.01″ rms over 40–45″ FOV vs. <10″ for conventional AO.
- GREGOR MCAO (Fig. 51) uses three DMs at 0, 8, 25 km; scheduled after first light.
- EST (4 m, Figs. 52–53) plans five DMs at 0, 5, 9, 15, 30 km. At 60″ FOV and $r_0=20$ cm, $S\sim0.6$ at 0° zenith but collapses to $\sim0.2$ at 60°. MCAO favors near-zenith pointing — **in tension with solar observing, which is usually best in the morning at moderate zenith angles**.

#### 9.3 GLAO / 지표층 AO

**한국어**: 주간에는 난류의 90%가 지표 100-200 m 내에 집중. 상층 $r_0$는 오히려 큼. GLAO는 **지표 층만** 보정해 sub-arcsec(0.25–0.5") 해상도를 넓은 FOV에서 달성 — 회절 한계는 아님. 4 m 망원경 + NIR 1.6 μm에서 특히 유용: 대구경 → 광자 플럭스 유리, $\lambda^{6/5}$로 등화면각 확대. 응용: SOLIS 같은 synoptic 망원경 — 50 cm 구경이 회절 한계로 **전 태양 원반**을 관측하는 시나리오 가능.

**English**: During the day 90% of turbulence is within 100–200 m of the ground, while upper-atmosphere $r_0$ is often large. GLAO corrects **only the ground layer**, delivering sub-arcsecond (0.25–0.5″) resolution over a wide FOV — not diffraction-limited. Particularly attractive on a 4 m + NIR 1.6 μm: large aperture gives more photons, and $\theta_0\propto\lambda^{6/5}$ widens the corrected patch. Application: synoptic telescopes like SOLIS — a 50 cm aperture could in principle image the full disk at the diffraction limit.

### Part X: Summary (§10, p. 76)

**한국어**: Solar AO는 **성공 스토리**다. 상관형 SHWFS + 상용 DM + 빠른 컴퓨터 → granulation 위 실시간 보정 + post-facto processing → 지상에서 우주 수준 관측. 향후: 4 m 망원경을 위한 extreme AO (수천 actuators), MCAO (광시야), GLAO (sub-arcsec 광시야). MCAO는 대구경에서 isoplanatic patch가 작다는 근본 한계를 해결하는 유일한 방법.

**English**: Solar AO is a success story. Correlating SHWFS + commercial DMs + fast computing = real-time granulation-based correction + post-facto processing → ground-based observations rivaling space. Next steps: extreme AO (thousands of actuators) for 4 m telescopes, MCAO (wide field), GLAO (sub-arcsec wide field). MCAO is the only way around the fundamentally small isoplanatic patch on large apertures.

---

## 3. Key Takeaways / 핵심 시사점

1. **Fried parameter $r_0$가 모든 scaling의 중심** — AO 설계의 모든 오차 항이 $r_0$의 거듭제곱 함수로 표현된다. Fitting error $\propto(d/r_0)^{5/3}$, anisoplanatism $\propto(\theta/\theta_0)^{5/3}$, bandwidth $\propto(f_G/f_S)^{5/3}$. / The Fried parameter $r_0$ is the single most important number — every error term in the AO budget scales as a power of $r_0$ (fitting $\propto(d/r_0)^{5/3}$, anisoplanatism $\propto(\theta/\theta_0)^{5/3}$, bandwidth $\propto(f_G/f_S)^{5/3}$).

2. **Correlating SHWFS가 solar AO를 가능하게 했다** — 별 대신 granulation 이미지 간 2D cross-correlation으로 local tilt를 측정하는 아이디어가 핵심 돌파구. 이것이 1998년에 실시간화된 후에야 solar AO가 실용화되었다. / The correlating SHWFS made solar AO practical. Replacing point-source centroiding with cross-correlation of granulation subimages was the single breakthrough; solar AO only became routine after this was implemented in real time in 1998.

3. **주간 seeing은 $r_0$뿐만 아니라 $\tau_0$도 가혹하다** — $r_0\sim10$ cm, $v\sim10$ m/s → $\tau_0\sim10$ ms → 2.5 kHz 샘플링, 100–200 Hz closed-loop bandwidth 필요. 이는 야간 AO의 수십 Hz보다 **수 배 높은 요구**다. / Daytime seeing is brutal in $\tau_0$ as much as in $r_0$: $r_0\sim10$ cm with $v\sim10$ m/s gives $\tau_0\sim10$ ms, forcing 2.5 kHz sampling and 100–200 Hz closed-loop bandwidth — several times what typical night-time AO requires.

4. **Anisoplanatism이 대구경 태양 AO의 근본 한계** — 등화면 각도 $\theta_0\propto[\int C_n^2(h)h^{5/3}dh]^{-3/5}$의 $h^{5/3}$ 가중치 때문에 고층 난류(제트스트림 ~10 km)가 지배적. 4 m ATST에서 $r_0=20$ cm라도 10" 이상 off-axis에서 Strehl 급락 → MCAO 없이는 광시야 회절 한계 불가능. / Anisoplanatism is the fundamental limitation for large-aperture solar AO. The $h^{5/3}$ weighting in $\theta_0\propto[\int C_n^2(h)h^{5/3}dh]^{-3/5}$ makes high-altitude (jet-stream) turbulence dominant. Even at $r_0=20$ cm on the 4 m ATST, Strehl collapses beyond 10″ off-axis — wide-field diffraction-limited imaging is impossible without MCAO.

5. **Error budget이 AO 시스템 설계의 바이블이다** — 9개 항의 RSS로 총 오차 계산 (Eq. 26). "모든 항이 비슷한 크기" 가 최적 설계 원칙 — 한 항을 극단적으로 줄이는 것은 낭비. $S=0.7$을 500 nm에서 달성하려면 $\sigma_{\text{tot}}<50$ nm. / The error budget is the system-design bible. Total variance is the RSS of nine terms (Eq. 26). The optimality principle: all terms should be of similar magnitude — driving one term far below the others is wasteful. Reaching $S=0.7$ at 500 nm demands $\sigma_{\text{tot}}<50$ nm.

6. **AO + post-facto processing이 최강 조합** — AO는 Strehl 0.3–0.6의 부분 보정; speckle/phase-diversity/MOMFBD가 잔차를 복원하고 FOV에 걸쳐 균일화한다. AO telemetry로부터 long-exposure PSF를 추정(Eq. 29)하여 대비를 정량적으로 회복. Sirius PSF 검증(그림 25)이 기법 신뢰성을 입증. / AO + post-facto processing is the winning combination. AO gives partial Strehl 0.3–0.6; speckle, phase diversity, and MOMFBD recover residuals and deliver uniform quality over the FOV. A long-exposure PSF can be estimated from AO telemetry (Eq. 29), validated against Sirius (Fig. 25), and used for quantitative contrast recovery.

7. **Site selection이 AO 성능의 절반을 결정** — 같은 $r_0$에서도 $C_n^2(h)$ 프로파일에 따라 등화면각이 3–4배 차이. Haleakala (5% above 6 km) vs Mt. Graham (40% above 6 km)의 시뮬레이션 차이가 극적. 따라서 "ATST는 왜 Haleakala에 갔는가"에 대한 직접적 정량 근거 제공. / Site selection sets half the AO performance. For the same $r_0$, the $C_n^2(h)$ profile changes the isoplanatic angle by factors of 3–4. Simulations comparing Haleakala (5% power above 6 km) vs Mt. Graham (40%) are dramatic — direct quantitative justification for placing ATST on Haleakala.

8. **MCAO는 태양에서 구현이 오히려 쉬운 측면이 있다** — 야간은 여러 laser guide star를 만들어야 하지만, 태양은 ubiquitous granulation이 자연 guide field. 따라서 multiple correlating SHWFS만 있으면 된다. DST의 40-45" MCAO 시연(2010)은 이를 증명. 하지만 DMs 수 증가, 열 관리, zenith angle 의존성 등이 여전히 난제. / MCAO is actually somewhat easier on the Sun than at night. Night-time MCAO needs multiple LGS; solar MCAO's guide fields come free from ubiquitous granulation — only multiple correlating SHWFS are needed. DST's 40–45″ MCAO demonstration (2010) confirms this. Yet multiple DMs, thermal management, and zenith-angle dependence remain open challenges.

---

## 4. Mathematical Summary / 수학적 요약

### (A) 대기 난류 기초 / Atmospheric turbulence basics

**Refractive-index PSD (Kolmogorov)**:
$$\Phi_N(\kappa) = 0.0365\,C_n^2\,\kappa^{-5/3}\qquad{}^{3D}\Phi_n(\kappa)=0.033\,C_n^2\,\kappa^{-11/3}$$

**Refractive-index structure function**:
$$D_n(\rho) = \langle|n(\vec{r}) - n(\vec{r}+\vec\rho)|^2\rangle = C_n^2\rho^{2/3}$$

**Phase structure function (single layer, integrated)**:
$$D_\varphi(\rho,h) = 2.914\,k^2\,\delta h\,C_n^2(h)\,\rho^{5/3}\quad\Rightarrow\quad D_\varphi(\rho) = 2.914\,k^2(\sec\gamma)\rho^{5/3}\int C_n^2(h)\,dh$$

### (B) Fried parameter / 프리드 파라미터

$$\boxed{\,r_0 \equiv \left[0.423\,k^2(\sec\gamma)\int C_n^2(h)\,dh\right]^{-3/5}\,}$$

- $r_0\propto\lambda^{6/5}$ (IR seeing >> visible seeing)
- DOF $\approx(D/r_0)^2$
- Long-exposure structure function: $D_{\text{long}}(\rho) = 6.88(\rho/r_0)^{5/3}$
- Short-exposure structure function: $D_{\text{short}}(\rho) = 6.88(\rho/r_0)^{5/3}[1-(\rho/D)^{1/3}]$

### (C) 장노출 OTF (대기만) / Long-exposure atmospheric OTF

$$\text{OTF}_{\text{atm}}(\vec\rho/\lambda) = \exp\left[-3.44\left(\frac{\rho}{r_0}\right)^{5/3}\right]$$

OTF_tel은 회절 한계; OTF_{\text{total}} = OTF_atm × OTF_tel.

### (D) 시간 척도 / Timescales

$$\tau_0 \equiv r_0/v\qquad f_G \approx 0.427\,v/r_0\qquad f_G(n) \propto 0.3(n+1)v/D$$

$n$은 Zernike radial mode number. Higher-order modes need higher bandwidth.

### (E) 등화면 각도 / Isoplanatic angle

$$\boxed{\theta_0 = \left[2.914\,k^2(\sec\gamma)^{8/3}\int C_n^2(h)\,h^{5/3}\,dh\right]^{-3/5}}$$

- $\theta_0\propto\lambda^{6/5}$
- $h^{5/3}$ 가중치로 **고층 난류가 훨씬 더 중요**
- Anisoplanatic variance: $\sigma^2_\theta = (\theta/\theta_0)^{5/3}$

### (F) Wavefront Error Budget (종합) / Full error budget

$$\sigma^2_{\text{tot}} = \underbrace{\sigma^2_{\text{BW}}}_{\text{bandwidth}} + \underbrace{\sigma^2_\theta}_{\text{aniso}} + \underbrace{\sigma^2_{\text{fit}}}_{\text{DM density}} + \underbrace{\sigma^2_{\text{aliasing}}}_{\text{WFS sampling}} + \underbrace{\sigma^2_{\text{wfs}}}_{\text{WFS noise}} + \underbrace{\sigma^2_{\text{wfs,aniso}}}_{\text{extended WFS}} + \underbrace{\sigma^2_{\text{ncp}}}_{\text{optics}} + \underbrace{\sigma^2_{T/T}}_{\text{tip-tilt}} + \sigma^2_{\text{other}}$$

$$\sigma^2_{\text{fit}} = 0.28\,(d/r_0)^{5/3}\qquad \sigma^2_{\text{aliasing}}\approx 0.08\,(d/r_0)^{5/3}\qquad \sigma^2_{\text{BW}} = (f_G/f_S)^{5/3}$$

$$\sigma^2_{x,\text{WFS}} = \frac{5m^2\sigma_b^2}{4n_r^2\sigma_i^2}\quad\text{(waves}^2)\qquad \sigma^2_{\text{wavefront}} = \frac{1}{N}\text{trace}(\mathbf{BB}^T)\sigma^2_{\text{wfs}}$$

목표: 500 nm에서 Strehl = 0.7 → $\sigma^2_{\text{tot}} < 50\text{ nm}^2$.

### (G) Correlating SHWFS algorithm / 상관형 SHWFS 알고리즘

$$CC(\vec\Delta_i) = \sum_x\sum_y I_M(\vec{x})\cdot I_R(\vec{x}+\vec\Delta_i)$$

1. 각 subaperture의 20×20 픽셀 이미지 $I_M$ 획득.
2. 참조 이미지 $I_R$ (무작위 선택 subaperture) 과 cross-correlation 계산.
3. 5×5 픽셀 창에서 피크 위치 찾기.
4. 포물선 피팅으로 subpixel 정밀도 (또는 centroid 법).
5. 각 subaperture tilt → reconstructor matrix $\mathbf{B}$ → DM 명령.
6. Global tilt 분리 → 전용 tip/tilt 거울.

### (H) Long-exposure PSF estimation (AO telemetry) / 장노출 PSF 추정

$$\text{OTF}_{\text{ao}}(\vec\rho/\lambda) = \text{OTF}_{\phi_{e\parallel}}(\vec\rho/\lambda)\cdot\text{OTF}_{\phi_{e\perp}}(\vec\rho/\lambda)\cdot\text{OTF}_{\text{tel}}(\vec\rho/\lambda)$$

$$\text{OTF}_{\phi_{e\parallel}} = \exp\left[-\tfrac{1}{2}\bar D_{\phi_{e\parallel}}(\vec\rho)\right]\qquad \text{OTF}_{\phi_{e\perp}} = \exp\left[-\tfrac{1}{2}D_{\phi_{\text{atm}\perp}}(\vec\rho)\right]$$

$$\bar D_{\phi_{e\parallel}}(\vec\rho) = \sum_{i,j=1}^N\langle\epsilon_i\epsilon_j\rangle\,U_{ij}(\vec\rho),\qquad U_{ij}(\vec\rho) = \frac{\int P(\vec{x})P(\vec{x}+\vec\rho)[K_i(\vec{x})-K_i(\vec{x}+\vec\rho)][K_j(\vec{x})-K_j(\vec{x}+\vec\rho)]\,d\vec{x}}{\int P(\vec{x})P(\vec{x}+\vec\rho)\,d\vec{x}}$$

$K_i$는 Karhunen-Loève 모드, $\langle\epsilon_i\epsilon_j\rangle$는 WFS telemetry에서 얻는 잔차 KL 계수의 공분산. 보정되지 않은 고차 모드는 Kolmogorov 통계 + $r_0$ 추정치로 계산.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1941 ┃ Kolmogorov — turbulence theory (5/3 power law)
     ┃
1953 ┃ Babcock — proposes adaptive optics concept
     ┃
1966 ┃ Fried — defines r_0 (Fried parameter)
     ┃
1970s┃ Classified US military AO (Starfire, satellite tracking)
     ┃
1977 ┃ Greenwood — defines Greenwood frequency f_G
     ┃
1979 ┃ Hardy — first solar AO experiment at DST (shearing interferometer)
     ┃
1988 ┃ von der Lühe — focal-plane LCD mask WFS; correlation-tracker concept
     ┃
1992 ┃ Acton, Smithson (Lockheed) — segmented DM + SHWFS on solar pores
     ┃
1993 ┃ Beckers — Annual Review of A&A on AO for astronomy (foundational)
     ┃
1995 ┃ NSF investment in DKIST predecessors
     ┃
1998 ┃★NSO low-order AO — first closed-loop AO on granulation (DST)
     ┃
1999 ┃ SST (97 cm, La Palma) AO installation
     ┃
2002 ┃ Scharmer et al. — discover dark cores in penumbral filaments at SST
     ┃  (landmark science enabled by AO)
     ┃
2002 ┃ KAOS installed at VTT Tenerife
     ┃
2004 ┃ DST AO76 — high-order solar AO operational
     ┃
2005 ┃ ATST project science team — imaging requirements (Rimmele)
     ┃
2009 ┃ Rimmele et al. — first DST MCAO demonstration
     ┃
★2011┃ ← THIS REVIEW (Rimmele & Marino, Living Reviews in Solar Physics)
     ┃   Written at the threshold of the 4 m MCAO era
     ┃
2012 ┃ GREGOR (1.5 m, Tenerife) first light
     ┃
2020 ┃ DKIST (formerly ATST, 4 m) first light — Strehl >0.3 at 500 nm
     ┃
2023 ┃ DKIST Visible Broadband Imager MCAO commissioning
     ┃
2026 ┃ Active solar AO research: deep-learning wavefront sensing,
     ┃  predictive control, EST (4 m European) design finalization
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Scharmer et al. 2002 "Dark Cores in Penumbral Filaments"** (Nature) | 본 논문이 solar AO의 과학적 성공 사례로 인용 (Fig. 2) — SST AO + phase diversity로 penumbral dark core 최초 발견 | 이 논문이 "왜 solar AO가 필요한가"에 대한 결정적 증거 제공. Solar AO가 과학적 발견으로 이어지는 전형적 예. |
| **Kolmogorov 1941 "Local structure of turbulence"** | 전 error budget의 이론적 기반. $\kappa^{-5/3}$ 멱법칙, 구조함수 개념 | 모든 AO 이론의 뿌리. 본 논문 §2.1에서 Eqs. 1–4로 인용. |
| **Hardy 1998 "Adaptive Optics for Astronomical Telescopes"** (textbook) | 본 논문의 수식 대부분의 유도 출처. 교과서적 보완. | 본 논문을 읽을 때 옆에 두어야 할 표준 참고서. |
| **Beckers 1993 "Adaptive optics for astronomy" (ARA&A)** | 본 논문 이전 AO 분야의 가장 광범위한 리뷰 | 야간 AO까지 포괄. 본 논문은 solar AO 특화로 차별화. |
| **Marino & Rimmele 2011 (simulations)** | 본 논문 §6.1.3과 §9.1.3의 모든 anisoplanatism 시뮬레이션 데이터 (Figs. 20–21, 42–46)의 출처 | 동일 저자의 보완 논문. 정량적 근거 제공. |
| **Nordlund & Stein 2009 "Solar surface convection"** (LRSP) | 본 논문 Fig. 3의 granulation MHD 시뮬레이션 출처 | AO가 분해해야 할 "진짜" 태양 구조를 시뮬레이션으로 예측 — solar AO 목표 설정의 근거. |
| **van Noort et al. 2005 "MOMFBD"** | §7의 핵심 post-processing 기법 | AO 잔차 제거에 가장 널리 쓰이는 알고리즘. |
| **Noll 1976 "Zernike polynomials and atmospheric turbulence"** | Zernike 모드 분산의 $(D/r_0)^{5/3}$ scaling 출처 (Eq. 39의 토대) | 본 논문 §6.3에서 인용, PSF 추정의 수학적 기반. |

---

## 7. References / 참고문헌

**본 논문 / This paper**:
- Rimmele, T. R. & Marino, J., "Solar Adaptive Optics", *Living Reviews in Solar Physics*, **8**, 2 (2011). [DOI: 10.12942/lrsp-2011-2]

**핵심 참고문헌 / Key references**:
- Kolmogorov, A. N., "The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers", *Dokl. Akad. Nauk SSSR*, 30, 301 (1941).
- Fried, D. L., "Optical Resolution Through a Randomly Inhomogeneous Medium for Very Long and Very Short Exposures", *J. Opt. Soc. Am.*, **56**, 1372 (1966).
- Greenwood, D. P., "Bandwidth specification for adaptive optics systems", *J. Opt. Soc. Am.*, 67, 390 (1977).
- Noll, R. J., "Zernike polynomials and atmospheric turbulence", *J. Opt. Soc. Am.*, **66**, 207 (1976).
- Hardy, J. W., *Adaptive Optics for Astronomical Telescopes*, Oxford University Press (1998).
- Roddier, F. (ed.), *Adaptive Optics in Astronomy*, Cambridge University Press (1999).
- Tyson, R. K., *Principles of Adaptive Optics*, CRC Press (3rd ed., 2011).
- Beckers, J. M., "Adaptive optics for astronomy: Principles, performance, and applications", *Annu. Rev. Astron. Astrophys.*, **31**, 13 (1993).
- von der Lühe, O., "Wavefront error measurement technique using extended, incoherent light sources", *Opt. Eng.*, **27**, 1078 (1988).
- Rimmele, T. R. & Radick, R. R., "Solar adaptive optics at the National Solar Observatory", *Proc. SPIE*, **3353**, 72 (1998).
- Scharmer, G. B. et al., "Dark cores in sunspot penumbral filaments", *Nature*, **420**, 151 (2002).
- Marino, J., "Long Exposure Point Spread Function Estimation from Adaptive Optics Loop Data", *PhD thesis*, NMSU (2007).
- Marino, J. & Rimmele, T. R., "Long exposure point spread function estimation from solar adaptive optics loop data", *Appl. Opt.*, **49**, G95 (2010).
- van Noort, M. et al., "Solar image restoration by use of multi-frame blind de-convolution with multiple objects and phase diversity", *Solar Phys.*, **228**, 191 (2005).
- Löfdahl, M. G. & Scharmer, G. B., "Wavefront sensing and image restoration from focused and defocused solar images", *A&AS*, **107**, 243 (1994).
- Paxman, R. G. et al., "Spatial stabilization of deep-turbulence-induced anisoplanatic blur", *Opt. Express*, **17**, 15886 (2009).
- Rimmele, T. R. et al., "Solar multiconjugate adaptive optics at the Dunn Solar Telescope", *Proc. SPIE*, **7736** (2010).
- Berkefeld, T., Soltau, D., del Moro, D., Löfdahl, M. G., "Wavefront sensing and wavefront reconstruction for the 4m European Solar Telescope EST", *Proc. SPIE*, **7736** (2010).
- Richards, K. et al., "The adaptive optics and wavefront correction system for the Advanced Technology Solar Telescope", *Proc. SPIE*, **7736** (2010).
- Cattaneo, F., Emonet, T. & Weiss, N., "On the Interaction between Convection and Magnetic Fields", *ApJ*, **588**, 1183 (2003).
- Nordlund, Å. & Stein, R. F., "Solar surface convection", *Living Reviews in Solar Physics*, **6**, 2 (2009).
