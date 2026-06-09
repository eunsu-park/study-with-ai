---
title: "Pre-Reading Briefing: A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft"
paper_id: "63_lin_1995"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Lin, R. P., Anderson, K. A., Ashford, S., Carlson, C., Curtis, D., Ergun, R., Larson, D., McFadden, J., McCarthy, M., Parks, G. K., Rème, H., Bosqued, J. M., Coutelier, J., Cotin, F., d'Uston, C., Wenzel, K.-P., Sanderson, T. R., Henrion, J., Ronnet, J. C., and Paschmann, G., "A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft", *Space Science Reviews* **71**, 125–153, 1995. DOI: 10.1007/BF00751328
**Author(s)**: R. P. Lin et al. (UC Berkeley SSL, U. Washington, CESR Toulouse, ESA/ESTEC, MPI Garching)
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA의 Wind 우주선에 탑재된 3차원 플라스마 및 고에너지 입자 관측기(3DP, 3D Plasma and Energetic Particles instrument)의 설계, 측정 원리, 운용 모드를 종합적으로 기술한다. 3DP는 태양풍 코어(~3 eV)부터 저에너지 우주선 영역(~수백 keV ~ 11 MeV)까지의 입자 분포를 한 자전(3 s) 안에 4π sr로 측정하도록 설계되었으며, **(i) 정전 분석기(EESA-L/H, PESA-L/H) — top-hat 대칭 반구형 분석기**, **(ii) 반도체 검출기 망원경(SST) — 자석/포일 이중 콜리메이터로 전자/이온 분리**, **(iii) Fast Particle Correlator(FPC) — 파-입자 상관 측정** 의 3개 검출기 시스템을 결합한다. 이는 ISEE-3가 부족했던 통계, 시간/각 분해능, 동적 범위를 모두 끌어올린 본격적 "suprathermal 입자(eV–수백 keV)" 영역의 정밀 3D 분포 함수 측정 도구이다.

This paper is the comprehensive instrument paper for the 3D Plasma and Energetic Particles (3DP) experiment on the NASA Wind spacecraft. 3DP is designed to measure full three-dimensional distributions of electrons and ions from solar wind core energies (~3 eV) up to low-energy cosmic-ray energies (~hundreds of keV to ~11 MeV), in one spin (3 s), over 4π sr. It combines three detector systems: **(i) top-hat symmetric quadrispherical electrostatic analyzers (EESA-L/H for electrons, PESA-L/H for ions)** spanning ~3 eV–30 keV with two sensitivities to handle the huge dynamic range; **(ii) Solid State Telescope (SST) silicon-detector triplets** with foil-side (electrons 25–400 keV) and magnet-side (ions 20 keV–6 MeV) ends giving electron–ion separation up to ~1 MeV and ~11 MeV; and **(iii) the Fast Particle Correlator (FPC)** for direct wave–particle correlation in Langmuir-wave / type III burst regimes. Together these surpass the ISEE-3 generation in sensitivity, angular resolution, dynamic range, and time resolution, providing the first detailed in-situ exploration of the full suprathermal regime.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 초는 ISTP/GGS(International Solar Terrestrial Physics / Global Geospace Science) 프로그램 시대였다. ISEE-3(1978–1982)이 L1 점에서 처음으로 태양풍 ~2 keV 전자 분포를 정밀 측정하면서, 태양풍 코어/halo 전자 위에 존재하는 "조용한 시기(quiet-time) 2–20 keV 전자", scatter-free하게 전파되는 sub-10 keV impulsive 전자 이벤트, 행성간 충격파에서 가속된 ~2–10 keV 피치각 분포, type III burst를 만드는 전자 빔 등 새로운 suprathermal 현상이 잇따라 발견되었다. 그러나 ISEE-3 검출기는 통계와 분해능이 부족해 정량 분석에 한계가 있었다. Wind 우주선은 1994년 11월 발사되어 처음 1–2년은 달 swingby(double lunar) 궤도로 자기권 상류와 측면을 훑은 뒤 L1으로 이동하는 것이 계획되었고, 3DP는 GGS 임무의 입자 분야 핵심 계측기로 설계되었다.

The early 1990s were the ISTP/GGS era. ISEE-3 (1978–1982) had, for the first time, made high-sensitivity measurements of ~2 keV solar-wind electrons and revealed several new suprathermal phenomena: a quiet-time 2–20 keV electron population apparently of solar origin, impulsive sub-10 keV electron events that propagate scatter-free, structured pitch-angle distributions of ~2–10 keV electrons at interplanetary shocks and large flares, and electron beams responsible for type III radio bursts. ISEE-3 lacked the statistics, time resolution, and angular resolution for quantitative analysis. Wind, launched 1994 Nov, was planned to spend 1–2 years in a double-lunar-swingby orbit (sampling foreshock, magnetotail, and upstream), then move to L1 — and 3DP was the core particles experiment for GGS.

### 타임라인 / Timeline

```
1968  Asbridge+: upstream solar-wind ions discovered (foreshock seed)
1971  Lin & Hudson: flash-phase electrons in solar flares
1973  Feldman+: solar-wind electron heat flux and halo
1978  Sarris+: magnetospheric burst particles in IPM
1978  ISEE-3 launched -- first high-sensitivity ~2 keV electron analyzer
1979  Filbert & Kellogg; Gloeckler+: foreshock electrons & CIR ions
1980  Gosling+; Paschmann+; Potter+: shock-accelerated ions, reflected beams
1981  Lin+: type III electron distribution f(v_||)
1982  Anderson+: pitch-angle structure at IP shocks
1983  Carlson+: top-hat symmetric ESA design (heritage for 3DP)
1985  Lin: quiet-time 2-20 keV electron population
1985  Gough: particle auto-correlator technique (heritage for FPC)
1992  Lin & Kahler: long-range probes via electron PAD
1993  Manuscript received (28 January)
1994  Wind launch (1994 Nov 1)
1995  This paper published in Space Science Reviews
2000+ Wind/3DP becomes the workhorse for type III, halo, strahl,
      ESP shock acceleration, foreshock electron studies
```

---

## 3. 필요한 배경 지식 / Prerequisites

**플라스마/우주물리 기초 / Plasma & space physics basics**
- 분포함수 $f(\vec{v})$, 모멘트(밀도, 속도, 압력 텐서, 열속), 피치각 / Distribution function, moments, pitch angle.
- 태양풍 전자: core (Maxwellian, ~10 eV), halo (suprathermal tail, ~50 eV–수 keV), strahl (자기력선 따라 anti-sunward beam) / Solar-wind electron components.
- Type III 라디오 버스트와 Langmuir wave 메커니즘 (bump-on-tail 불안정성 → $\omega_p$, $2\omega_p$ 방출) / Type III bursts and bump-on-tail.
- 행성간 충격파에서의 입자 가속 (diffusive shock acceleration, drift acceleration), bow shock foreshock.

**검출기 물리 / Detector physics**
- Top-hat 대칭 반구형(quadrispherical) 정전 분석기 원리: 두 동심 반구 사이 전위 $V$로 입자의 $E/q$ 선택, $\Delta E/E \approx 0.2$ / Spherical-section ESA principles, $E/q$ filter.
- 마이크로채널 플레이트(MCP)와 chevron 구성, anode segmentation으로 방위각 측정 / MCPs, chevrons, position-sensitive anodes.
- 반도체 검출기에서의 에너지 손실: 두께 vs. range, anti-coincidence, 전자/이온 분리를 위한 sweeping magnet과 Lexan foil의 역할 / Si detector energy deposition, foil/magnet separation.
- Geometric factor $G$ (cm² sr) 정의: $\text{count rate} = G \cdot E \cdot j(E)$ for differential energy flux.

**수학 도구 / Math tools**
- 4π sr를 spinning spacecraft에서 어떻게 sampling하는가: 한 평면(360°)은 분석기 자체로, 수직 방향(±90°)은 spin으로 / Spin sampling geometry.
- 시간-주파수 신호처리 (FPC의 phase splitter, 90° 위상차 SIN/COS 상관) / Quadrature-phase correlator, auto-correlation histograms.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Suprathermal particles** | 태양풍 열에너지(~10 eV) 위, 저에너지 우주선(MeV) 아래의 ~수십 eV–수백 keV 입자. 이 논문이 정의한 핵심 측정 대상 / Particles between solar-wind thermal (~10 eV) and low-energy cosmic-ray (~MeV) — the regime 3DP is built for. |
| **EESA / PESA** | Electron / Proton Electrostatic Analyzer. -L = low-sensitivity (180° FOV, 큰 입자 플럭스용), -H = high-sensitivity (360° FOV, 작은 플럭스용) / Two-sensitivity analyzer pairs. |
| **SST (Solid State Telescope)** | 반도체 검출기 망원경. 한쪽은 Lexan **foil**로 양성자 차단(전자 측정), 반대쪽은 sweeping **magnet**으로 <400 keV 전자 제거(이온 측정) / Double-ended Si telescope; foil-side electrons, magnet-side ions. |
| **Top-hat ESA / Quadrispherical analyzer** | 두 동심 반구 사이 통로로 입자 진입, 360° 평면 시야와 ~1° 분해능 제공. Carlson et al. 1983 설계 / Carlson 1983 symmetric spherical-section analyzer with disk-shaped 360°/180° FOV. |
| **Geometric factor G** | $\text{count rate} = G \cdot E \cdot j(E)$의 비례 상수. EESA-L: $1.3\times10^{-2}E$, EESA-H/FPC: $0.1E$, PESA-L: $1.6\times10^{-4}E$ cm² sr / Sensitivity-times-aperture in cm² sr. |
| **Pitch Angle Distribution (PAD)** | 자력선 방향에 대한 입자 분포각의 1D 빈ning. magnetometer 자료를 받아 onboard에서 계산 / 1D rebinning of $f(v)$ vs. angle to $\vec{B}$, computed onboard. |
| **Type III radio burst** | 태양 폭발에서 가속된 ~수 keV–수십 keV 전자가 코로나/IPM을 통과하며 만드는 라디오 방출. f가 시간에 따라 빠르게 감소 / Solar electron beams producing $\omega_p, 2\omega_p$ emission. |
| **FPC (Fast Particle Correlator)** | EESA-H 전자 데이터를 WAVE 실험의 전기장 신호와 직접 상관, ~3 ms 시간 분해능으로 Langmuir wave에서의 전자 bunching 측정 / Direct wave-particle correlator at ~3 ms. |
| **Anti-coincidence** | 동일 입자가 여러 검출기를 관통하는 신호를 거부하는 전자 회로. background 억제 핵심 / Veto used to reject penetrating particles. |
| **Foreshock / Upstream ions** | 지구 bow shock에서 반사/가속된 ~keV-수십 keV 이온/전자가 IMF 따라 상류로 흘러나오는 영역 / Bow-shock-reflected populations upstream of Earth. |
| **Strahl** | 태양풍 halo 전자 중 자기장과 평행한 방향으로 강한 anti-sunward beam을 만드는 성분 / Field-aligned anti-sunward halo electrons. |
| **Burst memory / Snapshot** | 트리거 이벤트 시 2 MB 메모리에 고시간 분해능 데이터를 저장 후 천천히 텔레메트리로 재생 / 2 MB high-cadence ring buffer for triggered events. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Electrostatic analyzer energy selection / 정전 분석기의 에너지 선택

두 동심 반구(반지름 $R_1 < R_2$, 전위차 $V$) 사이를 통과하는 입자의 $E/q$:
$$ \frac{E}{q} = k \cdot V, \qquad k \approx \frac{R_1 R_2}{(R_2^2 - R_1^2)} \approx \frac{R_0}{2 \Delta R} $$

PESA-H/EESA-H의 경우 $R_1 = 8.0$ cm, $\Delta R = 0.6$ cm, $\Delta E/E = 0.20$ FWHM. 즉 hemispherical 전압 $V$를 로그 스윕하여 32–64 에너지 채널을 한 spin(3 s)에 1024회 샘플링한다 / The hemisphere voltage selects $E/q$; logarithmic sweep gives energy channels.

### (2) Counting rate 식 / Counting-rate equation

$$ R = G(E) \cdot j(E) \cdot \Delta E = G_0 \cdot E \cdot j(E) \cdot (\Delta E/E) $$

여기서 $j(E)$ [particles cm⁻² s⁻¹ sr⁻¹ keV⁻¹]는 differential directional flux. Table I의 $G_0 \cdot E$ 형태는 ESA의 $\Delta E \propto E$를 반영 / Counts/s = $G_0 E \cdot j \cdot (\Delta E/E)$ — explains the "$E$ in $G$" notation in Table I.

### (3) SST에서 전자/이온 분리 / SST particle separation

Lexan foil 측: foil 두께가 ~400 keV 양성자 range와 같으므로
$$ E_p < E_p^{\text{foil}} \;(\sim 400~\text{keV}) \implies \text{stopped by foil}; \qquad E_e \;\text{essentially unchanged}. $$

Magnet 측: broom magnet으로 $<400$ keV 전자가 휘어 검출기를 비껴 가도록 함 — 이온은 거의 영향 없음. 따라서 한 telescope의 양 끝에서 동일 에너지의 전자/양성자가 깨끗이 분리 / Foil stops ions ≤400 keV (electrons unaffected); broom magnet sweeps electrons <400 keV (ions unaffected). Clean species separation.

### (4) Direct wave-particle correlation / 직접 파-입자 상관

$$ C(v, \Theta) \;=\; \frac{\displaystyle\int E_0 \sin(kx - \omega t + \Theta)\, F(v,t)\, dt}{\langle E^2(t)\rangle_t^{1/2}\, \langle F^2(v,t)\rangle_t^{1/2}} $$

$E_0\sin(kx-\omega t)$는 wave 전기장, $F(v,t)$는 전자 플럭스, $\Theta$는 instrumental phase shift. SIN(0°)와 COS(90°) 두 상관을 결합하면 진폭 $A = \sqrt{C_S^2 + C_C^2}$와 위상 $\phi = -\arctan(C_S/C_C)$를 얻는다 / Quadrature direct correlation gives amplitude and phase of electron bunching at the wave frequency.

### (5) Counting statistics / 계수 통계

전자 분포의 1% 변화를 $3\sigma$로 검출하려면
$$ N \gtrsim \left(\frac{3}{0.01}\right)^2 = 9.0 \times 10^4 \quad \text{counts per energy bin}. $$

Langmuir wave가 ~수백 ms 동안만 지속하므로, 이를 충족하려면 ~1 MHz 카운트 레이트가 필요 → EESA-H의 큰 geometric factor(0.1E cm² sr) 정당화 / Justifies the large EESA-H geometric factor.

---

## 6. 읽기 가이드 / Reading Guide

**1회독 / First pass (~30 min)**: §1–2에서 무엇을 측정하려는지 (8가지 과학 목표) 파악하고, §3에서 instrument suite의 구성도(EESA / PESA / SST / FPC / DPU)를 머리에 그린다. Table I과 Fig. 1, 2(에너지-flux 다이어그램)를 자세히 본다 — 왜 두 sensitivity 분석기가 필요한지가 직관적으로 보인다 / Goal: build the system block diagram in your head.

**2회독 / Second pass (~1 hr)**: §4(ESA)는 Carlson 1983 top-hat 설계의 광학 원리(Fig. 5의 quadrisphere focusing)를 이해하는 데 집중. §5(SST)는 Fig. 9의 단면도와 Fig. 10의 시야 배치(텔레스코프 1–6)를 노트한다. 5×telescope × 2-end → 4π sr coverage가 어떻게 만들어지는지 종이에 그려본다 / Sketch FOV geometry yourself.

**3회독 / Third pass (~30 min)**: §7(operation modes)와 §8(FPC)는 operational document에 가깝다. FPC의 (5) 식 신호처리 체인(antenna → BPF → phase splitter → comparator → counter gate)만 정확히 따라가면 충분 / Focus on the FPC signal-processing chain.

**숨은 보석 / Hidden gem**: §2.1–2.8의 8가지 과학 의문은 1995년 시점에 미해결이던 핵심 질문 모음 — 이후 Wind/3DP가 어떤 답을 냈는지를 현대 논문과 비교하면 매우 교육적 / The 8 science questions of §2 are a research roadmap.

---

## 7. 현대적 의의 / Modern Significance

Wind/3DP는 1994년 발사 후 30여 년간 운용되며 우주 플라스마 in-situ 분야의 표준 reference 데이터셋이 되었다. 이 논문이 정의한 EESA-H 360° top-hat ESA + 24-anode 디자인은 Cluster/PEACE, MMS/FPI(Fast Plasma Investigation), THEMIS/ESA, Parker Solar Probe/SWEAP-SPAN, Solar Orbiter/SWA로 그대로 계승되었다. SST의 foil/magnet 이중 콜리메이터 컨셉은 Parker Solar Probe ISʘIS, Solar Orbiter EPD-STEP 등으로 이어진다. FPC의 직접 파-입자 상관 기법은 MMS의 wave-particle correlation 분석(Fast Plasma Investigation + EDP)에서 디지털 후처리 형태로 재현된다. 무엇보다 §2의 과학 의문들 — quiet-time 2–20 keV 전자의 기원, type III 빔의 Langmuir wave 상호작용, 행성간 충격파 전자 가속, foreshock electron beam의 wave 생성 — 은 이후 Lin, Larson, Krucker, Wang 등의 후속 논문에서 답이 채워졌고, 현재의 PSP/SO 미션은 이 질문들을 태양 가까이에서 재검증하고 있다.

Wind/3DP, operating since 1994, became the standard in-situ reference dataset for suprathermal plasma physics. The 360° top-hat ESA + segmented-anode design defined here is the direct ancestor of Cluster/PEACE, MMS/FPI, THEMIS/ESA, Parker Solar Probe SPAN-A/E, and Solar Orbiter/SWA. The SST foil-vs-magnet double-end concept lives on in PSP/ISʘIS and SO/EPD-STEP. The FPC direct wave-particle correlation idea appears today as digital post-processing in MMS FPI+EDP. Above all, the eight science questions in §2 became the research program that Lin, Larson, Krucker, Wang and colleagues addressed in the following decades; PSP and Solar Orbiter now revisit these questions much closer to the Sun.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
