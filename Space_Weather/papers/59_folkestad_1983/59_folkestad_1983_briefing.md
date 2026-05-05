---
title: "Pre-Reading Briefing: EISCAT — An Updated Description of Technical Characteristics and Operational Capabilities"
paper_id: "59_folkestad_1983"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# EISCAT: An Updated Description of Technical Characteristics and Operational Capabilities — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Folkestad, K., Hagfors, T., and Westerlund, S. (1983). EISCAT: An updated description of technical characteristics and operational capabilities. *Radio Science*, 18(6), 867–879. DOI: 10.1029/RS018i006p00867
**Author(s)**: Kristen Folkestad (EISCAT, Norway); Tor Hagfors (EISCAT, Sweden — now NAIC, Cornell); Svante Westerlund (EISCAT, Sweden — now AB RIFA)
**Year**: 1983

---

## 1. 핵심 기여 / Core Contribution

**EN**: This paper is the canonical engineering reference for EISCAT (European Incoherent SCATter), the first tristatic incoherent-scatter radar (ISR) facility built in northern Scandinavia. It documents the system as commissioned: two transmitters at Ramfjordmoen, Norway (UHF at 933.5 MHz, 2 MW peak; VHF at 224 MHz, 5 MW peak), two remote receive-only sites (Kiruna, Sweden and Sodankylä, Finland), the steerable parabolic dish and split-beam parabolic-cylinder antennas, the helium-cooled parametric front-ends, the 16-bit programmable hardware correlator clocked at 5 MHz, and the EROS/TARLAN software stack. The tristatic geometry is the headline scientific advantage: a single scattering volume in the auroral ionosphere is observed simultaneously from three baselines, allowing the full 3-D ion-drift velocity vector (and hence the convection electric field via $\vec{E} = -\vec{v}_i \times \vec{B}$) to be inverted from three line-of-sight Doppler measurements.

**KR**: 본 논문은 북유럽 스칸디나비아에 건설된 최초의 삼정점(tristatic) 비간섭산란(ISR) 레이더 시설 EISCAT의 정식 공학 레퍼런스이다. 노르웨이 Ramfjordmoen에 위치한 두 송신기(UHF 933.5 MHz, 첨두 2 MW; VHF 224 MHz, 첨두 5 MW), 스웨덴 Kiruna와 핀란드 Sodankylä의 원격 수신 전용 사이트, 조향식 파라볼라 안테나와 split-beam 원통형 안테나, 헬륨 냉각 파라메트릭 전치증폭기, 5 MHz 클럭의 16-bit 프로그래머블 하드웨어 상관기(correlator), 그리고 EROS/TARLAN 소프트웨어 스택을 정리한다. 삼정점 기하학이 핵심 과학적 장점인데, 오로라 전리권 내 단일 산란체적을 세 기준선에서 동시에 관측해 세 개의 시선(line-of-sight) 도플러 속도로부터 3차원 이온 표류 속도 $\vec{v}_i$와 그로부터 $\vec{E} = -\vec{v}_i \times \vec{B}$ 관계로 대류 전기장을 복원할 수 있게 한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**EN**: Incoherent scatter as a remote-sensing technique was proposed by Gordon (1958) and demonstrated by Bowles (1958) at 41 MHz. Through the 1960s–70s the technique matured at single-dish facilities: Jicamarca (Peru, 50 MHz), Arecibo (Puerto Rico, 430 MHz), Millstone Hill (Massachusetts, 440 MHz), St. Santin (France, 935 MHz, the first tristatic system but at mid-latitudes), Chatanika and later Sondrestrom (Alaska/Greenland), and EISCAT itself. By the late 1970s, ISR's ability to measure $N_e, T_e, T_i, \vec{v}_i$, ion composition, and collision frequencies was well understood. What remained missing was a high-latitude facility positioned inside the auroral oval that could resolve the full electric-field convection vector — the engine of magnetosphere-ionosphere coupling. EISCAT was conceived as a multinational European answer (Norway, Sweden, Finland, France, UK, West Germany), with design studies running from the early 1970s and acceptance tests completing in 1981–1982 — exactly the timeframe this paper documents.

**KR**: 비간섭산란(IS) 기술은 Gordon(1958)이 이론으로 제안하고 Bowles(1958)가 41 MHz에서 실증했다. 1960~70년대를 거치면서 Jicamarca(50 MHz), Arecibo(430 MHz), Millstone Hill(440 MHz), 프랑스 St. Santin(중위도 최초의 삼정점, 935 MHz), Chatanika·Sondrestrom 등에서 기술이 성숙했다. 1970년대 말에 이르러 ISR이 $N_e, T_e, T_i, \vec{v}_i$, 이온 조성, 충돌 진동수를 모두 측정할 수 있다는 사실은 잘 알려져 있었으나, 자기권-전리권 결합의 엔진인 대류 전기장 벡터를 풀기 위해 오로라 오발 내부에 위치한 고위도 시설이 부재했다. EISCAT은 그 답으로 노르웨이·스웨덴·핀란드·프랑스·영국·서독의 다국적 사업으로 1970년대 초 설계 검토에 들어가 1981~1982년 인수시험을 마쳤으며, 본 논문은 정확히 이 시점의 시스템을 기록한다.

### 타임라인 / Timeline

```
1958 ── Gordon proposes IS / Bowles demonstrates at 41 MHz
1960s ── Jicamarca, Arecibo, Millstone Hill mature single-dish ISRs
1970s ── St. Santin tristatic (mid-lat) / Chatanika at Alaska auroral zone
1974 ── du Castel & Testud: original EISCAT design concept paper
1978 ── Rishbeth: EISCAT review (Esrange Symposium)
1980 ── UHF system installed at Ramfjordmoen
1981 ── First spectra recorded (June); Lehtinen-Turunen pointing calibration
1982 ── Hagfors (Nobel Symp 54): updated overview; rocket campaigns
1983 ── This paper: definitive engineering description
1984 ── VHF expected operational
```

---

## 3. 필요한 배경 지식 / Prerequisites

**EN**:
- **Radar fundamentals**: pulse-Doppler radar, range-time-frequency tradeoff, the radar equation, monostatic vs bistatic vs tristatic geometry, peak-vs-average power and duty cycle.
- **Incoherent-scatter theory**: thermal density fluctuations of the plasma (Salpeter 1960; Hagfors 1961), the Bragg condition $k_{\text{Bragg}} = 2 k_0 \sin(\theta/2)$ (backscatter $\theta = \pi$ ⇒ $k = 2k_0$), the ion-line spectrum with double-humped shape controlled by $T_e/T_i$ at small Debye length $k\lambda_D \ll 1$, and the plasma line at $\omega = \omega_p + \frac{3}{2}k^2 v_{T_e}^2/\omega_p$.
- **Signal processing**: complex baseband I/Q sampling, autocorrelation function (ACF), Wiener–Khinchin theorem ($S(\omega) = \mathcal{F}\{R(\tau)\}$), lag products, range gating, multipulse and Barker-code waveforms, lag-weighting (triangular for matched filter).
- **Antenna theory**: parabolic dish gain $G = 4\pi A_e/\lambda^2$, half-power beamwidth $\theta_{\rm HPBW} \approx 70\lambda/D$, parabolic cylinder with phased-array line feed, beam steering by phase progression $\Delta\phi = (2\pi d/\lambda)\sin\theta$.
- **Ionospheric physics**: $E$- and $F$-region structure, ion composition (NO$^+$, O$_2^+$, O$^+$, He$^+$, H$^+$), magnetospheric convection $\vec{E} = -\vec{v}_i \times \vec{B}$.

**KR**:
- **레이더 기초**: 펄스-도플러 레이더, 거리-시간-주파수 trade-off, 레이더 방정식, monostatic/bistatic/tristatic 기하, 첨두 대 평균 출력과 듀티 사이클.
- **비간섭산란 이론**: 플라스마 열적 밀도 요동(Salpeter 1960; Hagfors 1961), Bragg 조건 $k = 2k_0 \sin(\theta/2)$ (후방산란에서 $k=2k_0$), $T_e/T_i$로 형상이 결정되는 이온선 이중봉 스펙트럼($k\lambda_D \ll 1$ 영역), 플라스마선 $\omega = \omega_p + (3/2)k^2 v_{T_e}^2/\omega_p$.
- **신호처리**: 복소 기저대역 I/Q 표본화, 자기상관함수(ACF), Wiener–Khinchin 정리($S(\omega)=\mathcal{F}\{R(\tau)\}$), 래그곱(lag product), 거리 게이팅, multipulse·Barker 코드 파형, 래그 가중치(matched filter의 삼각 가중).
- **안테나 이론**: 파라볼라 이득 $G = 4\pi A_e/\lambda^2$, 반전력빔폭 $\theta_{\rm HPBW} \approx 70\lambda/D$, 위상 배열 라인피드를 가진 원통형 파라볼라, 위상 진행에 의한 빔 조향 $\Delta\phi = (2\pi d/\lambda)\sin\theta$.
- **전리권 물리**: $E$ 영역·$F$ 영역 구조, 이온 조성(NO$^+$, O$_2^+$, O$^+$, He$^+$, H$^+$), 자기권 대류 $\vec{E} = -\vec{v}_i \times \vec{B}$.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Incoherent Scatter (IS)** | EN: Bragg-scale Thomson scattering from electron density fluctuations whose statistics are dressed by ion dynamics; spectrum encodes $N_e, T_e, T_i, v_i$. KR: 이온 동역학에 의해 변조된 전자 밀도 요동의 Bragg 산란; 스펙트럼이 $N_e, T_e, T_i, v_i$ 정보를 담음. |
| **Tristatic** | EN: One transmitter and three receiving sites observing a common volume → three independent line-of-sight Doppler velocities ⇒ full 3-D $\vec{v}_i$. KR: 송신기 1대 + 수신 사이트 3곳이 공통 산란체적을 관측해 3개의 시선 도플러로 3차원 $\vec{v}_i$ 복원. |
| **UHF / VHF systems** | EN: 933.5 MHz steerable dish (32 m) for ion-line work, 2 MW peak, 12.5% duty; 224 MHz parabolic-cylinder (4×30×40 m) for $D$-region and plasma-line, 5 MW peak. KR: 933.5 MHz 32 m 조향식 안테나(첨두 2 MW, 듀티 12.5%, 이온선 측정용); 224 MHz 4×30×40 m 원통형(첨두 5 MW, $D$ 영역·플라스마선용). |
| **ACF (Autocorrelation Function)** | EN: $R(\tau)=\langle V(t)V^*(t+\tau)\rangle$; primary EISCAT observable, transformed to spectrum via FFT. KR: $R(\tau)=\langle V(t)V^*(t+\tau)\rangle$, EISCAT 주관측량으로 FFT 통해 스펙트럼화. |
| **Multipulse / Barker code** | EN: Pulse-compression schemes giving simultaneous good range and frequency resolution; 13-bit Barker decoder included on hardware. KR: 좋은 거리·주파수 분해능을 동시에 얻는 펄스 압축; 13-bit Barker 복호기가 하드웨어에 내장. |
| **Lag products** | EN: Discrete samples of the ACF computed by the correlator at integer multiples of the sampling interval. KR: 표본 간격의 정수배 지연에서 계산한 ACF의 이산 표본. |
| **Range gating** | EN: Time-windowing of received samples to localise scattering to a height interval $\Delta h = c\tau_p/2$. KR: 수신 표본의 시간 게이팅으로 산란을 $\Delta h = c\tau_p/2$ 고도 구간에 국한. |
| **Common Programme (CP)** | EN: Scheduled experiments (CP 0/1/2/3) reserved monthly to give all associates equal access; SP = special programme for one-off proposals. KR: 정기 실험(CP 0/1/2/3)으로 회원국에 균등 시간 배정; SP는 개별 특별 제안. |
| **EROS / TARLAN** | EN: EISCAT real-time OS (SINTRAN III based) and the radar-controller assembly language; CORLAN is the higher-level correlator-programming language. KR: EISCAT 실시간 운영체제(SINTRAN III 기반)와 레이더 제어 어셈블리어; CORLAN은 상관기 프로그래밍의 고수준 언어. |
| **Plasma line** | EN: Electron-acoustic resonance in the IS spectrum near $\omega_p$; weak (~10$^{-3}$ of ion-line) but diagnostic of $N_e$ and suprathermal electrons. KR: $\omega_p$ 근처 전자 음파 공명; 신호는 약하지만($\sim 10^{-3}$) $N_e$와 초열 전자의 진단자. |
| **CAMAC** | EN: Standardised modular instrumentation interface used between NORD computers and EISCAT's hardware. KR: NORD 컴퓨터와 EISCAT 하드웨어를 잇는 표준 모듈 계측 인터페이스. |
| **Polarizer** | EN: UHF feed component selecting RHCP/LHCP/elliptical/linear polarization for matching Faraday rotation at remote sites. KR: 원격 사이트에서의 패러데이 회전을 보상하기 위한 RHCP/LHCP/타원/직선 편파 선택용 급전부 부품. |

---

## 5. 수식 미리보기 / Equations Preview

**EN/KR**: Five formulas underpin the engineering numbers in this paper.

**(1) Radar equation for incoherent scatter**

$$P_r = \frac{P_t G_t A_e}{(4\pi)^2 R^4} \, \sigma_{\rm IS} N_e V \quad ; \quad \sigma_{\rm IS} \approx \frac{\sigma_T}{1+T_e/T_i}$$

EN: Backscattered power from a volume $V$ at range $R$. Note the volumetric (not point) scattering cross-section $\sigma_{\rm IS} N_e V$ and the suppression by $1+T_e/T_i$ relative to free Thomson $\sigma_T = 6.65\times10^{-29}$ m². KR: 거리 $R$의 체적 $V$에서의 후방산란 전력. 점목표가 아닌 체적 산란 단면적 $\sigma_{\rm IS} N_e V$이며, 자유 톰슨 단면적 $\sigma_T$ 대비 $1+T_e/T_i$ 만큼 억제.

**(2) Wiener–Khinchin / ACF–spectrum pair**

$$S(\omega) = \int_{-\infty}^{\infty} R(\tau)\, e^{-i\omega\tau}\, d\tau, \qquad R(\tau) = \frac{1}{2\pi}\int S(\omega)\, e^{i\omega\tau}\, d\omega$$

EN: EISCAT measures the ACF in hardware; the spectrum used to fit $T_e, T_i$ comes from the FFT. KR: EISCAT은 ACF를 하드웨어로 측정하고, $T_e, T_i$ 적합에 쓰이는 스펙트럼은 FFT로 얻음.

**(3) Tristatic ion-velocity inversion**

$$v_{\rm los}^{(i)} = \hat{k}_i \cdot \vec{v}_i, \quad i=1,2,3 \qquad \Rightarrow \qquad \vec{v}_i = \mathbf{K}^{-1} \begin{pmatrix} v_{\rm los}^{(1)}\\ v_{\rm los}^{(2)}\\ v_{\rm los}^{(3)}\end{pmatrix}$$

with $\mathbf{K}$ the $3\times3$ matrix whose rows are the bistatic $\hat{k}$-vectors $\hat{k}_i = \hat{n}_{\rm tx} + \hat{n}_{{\rm rx},i}$ (normalised). EN: Inverts three line-of-sight Doppler velocities into the full vector. KR: 세 시선 도플러를 3D 벡터로 역변환.

**(4) Convection electric field**

$$\vec{E}_\perp = -\vec{v}_i \times \vec{B}$$

EN: With $\vec{v}_i$ from (3), the perpendicular convection field is obtained directly — EISCAT's headline magnetospheric product. KR: (3)으로 얻은 $\vec{v}_i$에서 수직 대류 전기장을 즉시 도출 — EISCAT의 대표적 자기권 산물.

**(5) Range resolution and matched-filter weighting**

$$\Delta R = \frac{c\tau_p}{2}, \qquad w(\tau) = \tau_p - |\tau| \quad (|\tau|\le\tau_p)$$

EN: A 200 µs pulse gives 30 km range resolution; lag estimates are weighted triangularly when the range gate equals the pulse length. KR: 200 µs 펄스는 30 km 거리 분해능을 주며, 거리 게이트가 펄스 길이와 같을 때 래그 추정값에 삼각 가중치가 곱해짐.

---

## 6. 읽기 가이드 / Reading Guide

**EN**:
1. **Read the abstract and §1 Introduction first**: get oriented on what's new vs. earlier descriptions (du Castel & Testud 1974; Rishbeth 1978; Hagfors 1982).
2. **§2 Design characteristics** is dense — skim Tables 1–4 to internalise the headline numbers (frequencies, powers, antenna sizes), then re-read §2.4 (correlators) carefully — this is the heart of the digital backend.
3. **§3 Preparation and operational philosophy** explains how an experimenter actually books and runs time. The CP/SP split is the operational model that EISCAT has used ever since.
4. **§4 Observational results**: focus on Figure 7 (single-pulse ACF and spectra) and Figure 9 (multipulse 3 km resolution) — these are the smoking-gun proofs that the system works.
5. Watch for engineering numbers worth remembering: peak power 2 MW UHF / 5 MW VHF; duty cycle 12.5%; pulse range 10 µs–10 ms; correlator clock 5 MHz; system noise 40 K (cooled) / 120–150 K (uncooled).

**KR**:
1. **초록과 §1 서론을 먼저 읽기**: 이전 기술(du Castel & Testud 1974; Rishbeth 1978; Hagfors 1982) 대비 새 점을 파악.
2. **§2 설계 특성**은 정보 밀도가 높음 — Table 1~4로 핵심 수치(주파수, 출력, 안테나 크기)를 잡은 뒤 §2.4(상관기)를 정독. 디지털 백엔드의 핵심.
3. **§3 실험 준비와 운용 철학**은 실제 연구자가 시간을 신청·운용하는 방식 설명. CP/SP 이원 체제는 EISCAT의 영구적인 운용 모델.
4. **§4 관측 결과**: Figure 7(단일 펄스 ACF·스펙트럼)과 Figure 9(multipulse 3 km 분해능)에 집중 — 시스템 작동의 결정적 증거.
5. 기억할 공학 수치: 첨두 2 MW UHF / 5 MW VHF; 듀티 12.5%; 펄스 10 µs–10 ms; 상관기 클럭 5 MHz; 시스템 잡음 40 K(냉각) / 120–150 K(비냉각).

---

## 7. 현대적 의의 / Modern Significance

**EN**: EISCAT remains operational 40+ years later, and this paper is the engineering baseline against which all subsequent upgrades are measured: the EISCAT Svalbard Radar (ESR, 1996, 500 MHz at 78°N), Heating facility expansions, the GUISDAP analysis package (Lehtinen & Huuskonen 1996) descended directly from §3.4 here, and the under-construction EISCAT_3D phased-array system (multistatic, MIMO, electronically steered) inherits the tristatic philosophy in fully digital form. The CP/SP scheduling model has been adopted, in spirit, by every subsequent ISR consortium. Methodologically, the lag-weighted ACF approach described in §3.2 is foundational to the maximum-likelihood estimators used in modern AMISR and EISCAT_3D processing. For space weather, EISCAT's tristatic ion-velocity vector remains the gold-standard ground-truth for assimilation into AMIE, SuperDARN cross-polar-cap potential maps, and high-latitude convection models like Weimer/Heelis.

**KR**: EISCAT은 40여 년이 지난 지금도 가동 중이며, 본 논문은 모든 후속 업그레이드의 공학적 기준점이다: EISCAT Svalbard Radar(ESR, 1996, 500 MHz, 78°N), Heating 확장, §3.4의 분석 흐름을 직접 계승한 GUISDAP(Lehtinen & Huuskonen 1996), 그리고 현재 건설 중인 EISCAT_3D(다중정점, MIMO, 디지털 조향) 모두 본 논문의 삼정점 철학을 디지털 형태로 계승한다. CP/SP 운영 모델은 이후 모든 ISR 컨소시엄이 사실상 채택했다. 방법론적으로 §3.2의 래그 가중 ACF 접근은 현대 AMISR·EISCAT_3D 처리에서 쓰이는 최대우도 추정기의 토대다. 우주기상 측면에서 EISCAT의 삼정점 이온 속도 벡터는 AMIE 동화, SuperDARN 극관 횡단 전위 지도, Weimer/Heelis 고위도 대류 모델 검증의 황금 기준이다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
