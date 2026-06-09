---
title: "Pre-Reading Briefing: The Search Coil Magnetometer for THEMIS"
paper_id: "79_roux_2008"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# The Search Coil Magnetometer for THEMIS: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: A. Roux, O. Le Contel, C. Coillot, A. Bouabdellah, B. de la Porte, D. Alison, S. Ruocco, M.C. Vassal, "The Search Coil Magnetometer for THEMIS", Space Sci. Rev. 141, 265–275 (2008). DOI: 10.1007/s11214-008-9455-8
**Author(s)**: A. Roux et al. (CETP / 3D Plus, France)
**Year**: 2008

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA THEMIS 5기 위성에 탑재된 삼축 서치코일 자력계(Search Coil Magnetometer, SCM) 의 설계, 모델링, 보정(calibration) 결과를 보고한다. SCM은 0.1 Hz–4 kHz 의 ULF/ELF 대역 자기장 변동을 측정하며, 서브스톰(substorm) 발생·확장과 관련된 플라즈마 파동(이온 사이클로트론파, 휘슬러 모드, 저주파 하이브리드, 풍선 모드 등)을 원격 탐지하기 위한 핵심 장비이다. 논문은 (i) RLC 등가 회로와 자기 증폭률 $\mu_{app}$ 모델링, (ii) 플럭스 피드백을 통한 주파수 응답 평탄화, (iii) MCM-V 3D 패키징 전치증폭기 설계, (iv) Chambon-la-Forêt 보정 시설에서의 NEMI/전달함수 측정, (v) 5기 비행 모델(FM1–FM5) 사이의 일관성을 다룬다.

This paper documents the design, electrical modeling, and ground calibration of the tri-axial Search Coil Magnetometer (SCM) flown on each of the five THEMIS probes. The SCM measures magnetic-field fluctuations from 0.1 Hz to 4 kHz, covering the ULF/ELF range where waves believed to mediate substorm onset and expansion (whistler-mode, ion-cyclotron, lower-hybrid, and ballooning modes) live. Key contributions include: (i) an RLC equivalent-circuit model with apparent permeability $\mu_{app}$ derived from a high-permeability ferromagnetic core and 51,600-turn winding, (ii) a flux-feedback scheme (secondary winding) that flattens the response and removes the resonance, (iii) MCM-V (Multi-Chip Module Vertical) preamplifiers fitting 200 g and 75 mW per probe, and (iv) calibration data showing a Noise Equivalent Magnetic Induction (NEMI) better than 0.76 pT/√Hz at 10 Hz across all five flight models, satisfying the substorm-physics science requirement.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2000년대 초반 자기권 물리학의 가장 큰 미해결 문제는 "서브스톰 onset이 어디서, 어떻게 시작되는가?"였다. 두 경쟁 모델 — Magnetic Reconnection (MR, ~25 $R_E$에서 시작) 과 Current Disruption (CD, ~8–10 $R_E$ 근지구 플라즈마 시트에서 시작) — 의 시간·공간 순서를 구분하기 위해 NASA THEMIS 미션은 5기 위성을 자오선상에 배열하여 다중 지점 측정을 수행했다. 두 모델 모두 ULF/ELF 파동이 결정적 역할을 하므로(휘슬러 모드는 MR에서 사출, 이온 사이클로트론·풍선 모드는 CD에서 발달), SCM은 onset의 본질을 규명할 핵심 진단 장비였다. THEMIS SCM은 GEOS 1/2, Ulysses, Galileo, Interball, Cluster STAFF, Cassini로 이어지는 CETP의 서치코일 계보(heritage)를 잇는다.

In the early 2000s, the central open question in magnetospheric physics was the location and trigger of substorm onset. The Magnetic Reconnection (MR) model placed the trigger in the mid-tail (~25 $R_E$), while the Current Disruption (CD) model placed it near 8–10 $R_E$ in the inner plasma sheet. To disentangle the timing, the THEMIS mission deployed five spacecraft along a meridian. Because both models invoke ULF/ELF waves — whistlers ejected from reconnection sites, ion-cyclotron and ballooning modes feeding current disruption — a sensitive high-bandwidth magnetic wave instrument was indispensable. THEMIS SCM directly inherits design lineage from CETP search coils flown on GEOS 1/2, Ulysses, Galileo, Interball, Cluster STAFF, and Cassini.

### 타임라인 / Timeline

```
1942 ── Bozorth & Chapin: demagnetizing factors of rods (foundation for μ_app)
1945 ── Osborn: demagnetizing factors of the general ellipsoid
1977 ── GEOS-1 search coil (CETP heritage begins)
1992 ── Lui et al.: Current Disruption observations (Bx≃50 nT at 8 R_E)
1992 ── Bulanov et al.: HF tearing instability for thin current sheets
1994 ── Mandt et al.: whistler-mediated reconnection
1997 ── Cornilleau-Wehrlin et al.: Cluster STAFF design (immediate predecessor)
2001 ── Lui: review of substorm controversies
2007 ── Coillot et al.: improved search-coil design (sensors letter)
2008 ── ★ Roux et al.: THEMIS SCM (this paper)
2008 ── THEMIS launched; first conjunction observations
2015 ── MMS launched with 4-spacecraft search-coil cluster
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Faraday's law of induction / 패러데이 유도법칙**: $\mathcal{E} = -d\Phi/dt$. SCM은 dB/dt 센서이므로 응답이 주파수에 비례.
- **Magnetic permeability and demagnetizing factor / 자기 투자율과 반자기장 계수**: 유한한 코어에서는 모양 의존 $N_z$ 때문에 외부장 $B_{ext}$ 와 내부장 $B_{core}$ 이 다르다.
- **RLC resonance / RLC 공진**: 권선 인덕턴스 $L$, 등가 저항 $R$, 분포 정전용량 $C$ 가 만드는 $\omega_0=1/\sqrt{LC}$ 공진.
- **Negative feedback (operational amplifier) / 부귀환 증폭기**: 2차 권선을 통한 자속 피드백으로 응답 평탄화·위상 안정화.
- **Plasma waves / 플라즈마 파동**: 휘슬러 ($f_{ce}$ 이하), 이온 사이클로트론 ($f_{ci}$ 부근), 저주파 하이브리드 ($f_{LH}$), 풍선 모드(MHD).
- **NEMI / 등가 자기 잡음 밀도**: 입력 환산 자기장 잡음을 pT/√Hz 로 표현. 천체 신호와 직접 비교 가능.
- **Telemetry, FFT, filter banks / 텔레메트리, 필터뱅크**: 위성 대역폭 한계 안에서 파형/스펙트럼/필터 출력 모드를 운용.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Search Coil Magnetometer (SCM) / 서치코일 자력계 | dB/dt 를 감지하는 유도형 자기 센서. AC(파동) 측정에 적합. An induction-type magnetic sensor that detects dB/dt; suited to AC (wave) measurements. |
| Apparent permeability $\mu_{app}$ / 겉보기 투자율 | 코어 내부 평균장 / 외부장 비율. 모양과 $\mu_r$, $N_z$ 에 의존. Ratio of average internal to external field; depends on shape and $\mu_r$, $N_z$. |
| Demagnetizing factor $N_z$ / 반자기장 계수 | 자화된 시료 내부에 형성되는 역방향 장의 비율. 길이/지름 비 $m=L/d$ 가 클수록 작음. Fraction of the magnetization that produces an opposing internal field; smaller for slender rods. |
| NEMI / 등가 자기 잡음 | Noise Equivalent Magnetic Induction, 입력 환산 잡음 자기장 밀도 (pT/√Hz). Input-referred magnetic-noise density. |
| Flux feedback / 자속 피드백 | 2차 권선으로 외부장과 반대 방향 자속을 주입해 공진 제거·응답 평탄화. Secondary winding injects an opposing flux to flatten the response. |
| Transmittance $T(j\omega)$ / 전달함수 | $V/B$ 비로 표현된 센서 응답. Sensor response expressed as the ratio $V/B$. |
| Whistler mode / 휘슬러 모드 | $f_{ci} \ll f \lesssim f_{ce}$ 우원편파 전자기파. Right-hand circularly polarized EM mode between ion and electron gyrofrequencies. |
| Ion cyclotron wave / 이온 사이클로트론파 | $f \sim f_{ci}$ 좌원편파 파동, CD 모델의 핵심 후보. Left-hand mode near the ion gyrofrequency, key in CD models. |
| Lower-hybrid wave / 저주파 하이브리드 | $f_{LH}=\sqrt{f_{ci}f_{ce}}$ 부근 정전 파동. Mostly electrostatic wave near $f_{LH}=\sqrt{f_{ci}f_{ce}}$. |
| Ballooning mode / 풍선 모드 | 곡률·압력 구동 MHD 불안정성, CD 트리거 후보. Curvature/pressure-driven MHD instability candidate trigger of CD. |
| MCM-V / 다중칩 수직 모듈 | Multi-Chip Module Vertical: 얇은 PCB를 적층해 큐브 형태로 만든 고밀도 패키징. Cube-shaped 3D packaging with stacked PCBs. |
| DFB / IDPU | Digital Field Board / Instrument Data Processing Unit. 디지털화·FFT·필터뱅크·텔레메트리 처리 담당. Digitize, FFT, filter, and telemetry the analog SCM signals. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Induced voltage (Faraday's law applied to N turns) / 유도전압**:

$$
e = N S \left( \frac{1}{L}\int_0^L \mu_{app}(l)\, dl \right) B_{ext}\, \omega \;=\; N S\, \langle\mu_{app}\rangle\, B_{ext}\, \omega
$$

→ $e \propto \omega$ 이므로 저주파에서 신호가 작아지고, 큰 $N$, $\mu_{app}$ 가 필수.
→ Output is proportional to $\omega$, so low frequencies need very high $N$ and $\mu_{app}$.

**(2) Apparent permeability for a cylinder / 원기둥 코어의 겉보기 투자율**:

$$
\mu_{app}(m) \;=\; \frac{B_{core}}{B_{ext}} \;=\; \frac{\mu_r}{1+(\mu_r-1)N_z(m)} \;\xrightarrow[\mu_r \to \infty]{}\; \frac{1}{N_z(m)}
$$

→ 매우 높은 $\mu_r$ 코어에서는 형상비 $m=L/d$ 만이 결정 인자.
→ For very high $\mu_r$, only the shape ratio $m=L/d$ matters.

**(3) Transmittance (RLC sensor without feedback) / 전달함수**:

$$
T(j\omega) \;=\; \frac{V}{B} \;=\; \frac{-j\omega N S \mu_{app}}{(1-LC\omega^2) + jRC\omega}
$$

→ 공진 주파수 $\omega_0=1/\sqrt{LC}$ 에서 진폭이 폭주하므로, 피드백으로 평탄화.
→ Diverges at $\omega_0=1/\sqrt{LC}$; flux feedback removes the peak.

**(4) NEMI requirement / NEMI 요구 사양**:

$$
\text{NEMI}(10\,\text{Hz}) < 1\;\text{pT}/\sqrt{\text{Hz}}
$$

→ 플라즈마 시트 파동 진폭 (10–100 pT/√Hz @ 10 Hz, Cluster) 보다 한 자릿수 이상 작아야 함.
→ Must be an order of magnitude below the 10–100 pT/√Hz wave amplitudes seen by Cluster.

**(5) Whistler upper cutoff / 휘슬러 상한 컷오프**:

$$
f_{ce} = \frac{eB}{2\pi m_e} \quad\Rightarrow\quad B=50\,\text{nT} \;\Rightarrow\; f_{ce}\simeq 1.4\,\text{kHz}
$$

→ 8 $R_E$ 부근 표준값으로 4 kHz 상한 대역폭 결정.
→ Sets the 4 kHz upper bandwidth based on plasma-sheet B at ~8 $R_E$.

---

## 6. 읽기 가이드 / Reading Guide

- **Sect. 1 Introduction**: THEMIS 미션 전체 목표(서브스톰 onset 위치) 와 SCM의 역할 정리. Skim quickly.
- **Sect. 2 Measurement Requirements**: 과학 목표 → 주파수 범위(0.1 Hz–4 kHz) 와 NEMI(<1 pT/√Hz @ 10 Hz) 가 어떻게 도출되는지 주의 깊게. 위성 위치별 $f_{ce}$ 추정이 핵심.
- **Sect. 3 Description of the Instrument**: ★ 가장 기술적으로 풍부한 섹션. (i) Eq. 1–3 자기 증폭·RLC 모델, (ii) flux feedback 원리(Fig. 3), (iii) MCM-V 전치증폭기. 식 (2) 의 $\mu_{app} \to 1/N_z$ 극한 이해 필수.
- **Sect. 4 Calibrations and Tests**: Fig. 7(전달함수), Fig. 8(NEMI) 그래프와 Tables 1–5 (FM1–FM5 NEMI) 비교. 5기간 1 dB 이내 일관성이 핵심 결과.
- **Sect. 5 Summary**: 짧지만 모든 수치 사양을 요약하므로 체크리스트로 활용.
- **Table 6**: 텔레메트리 모드(필터뱅크/Fast Survey/Wave Burst/FFT) 를 구분. 이후 데이터 분석 시 다시 참조.

---

## 7. 현대적 의의 / Modern Significance

THEMIS SCM은 **서브스톰 onset 위치 결정** 이라는 미션 핵심 결과(Angelopoulos et al. 2008, Science) 에 직접 기여했고, 이후 MMS(2015) 4기 위성의 4 kHz 대역 SCM 클러스터, Parker Solar Probe FIELDS의 SCM, JUICE의 RPWI 등 모든 후속 자기장 파동 측정 미션의 설계 표준이 되었다. RLC + 플럭스 피드백 모델, MCM-V 패키징, Chambon-la-Forêt 보정 절차는 현대 우주용 자기 센서의 사실상 표준이다. NEMI 0.76 pT/√Hz @ 10 Hz 라는 수치는 ground-based SQUID 자력계와 비슷한 수준이며, 작은 질량(568 g 안테나) 대비 기록적 감도이다.

THEMIS SCM directly contributed to the mission-defining result that substorm onset begins in the mid-tail (Angelopoulos et al. 2008, Science). Its design — flux-feedback RLC search coil, MCM-V preamplifier, and Chambon-la-Forêt calibration protocol — became the de-facto standard for subsequent space-based magnetic-wave sensors, including MMS (2015) with its four-spacecraft SCM cluster, Parker Solar Probe's FIELDS, and JUICE's RPWI. The achieved 0.76 pT/√Hz NEMI at 10 Hz rivals ground-based SQUID sensitivity at a fraction of the mass (568 g per tri-axial antenna).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
