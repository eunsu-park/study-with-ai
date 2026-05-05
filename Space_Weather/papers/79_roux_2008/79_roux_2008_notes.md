---
title: "The Search Coil Magnetometer for THEMIS"
authors: "A. Roux, O. Le Contel, C. Coillot, A. Bouabdellah, B. de la Porte, D. Alison, S. Ruocco, M.C. Vassal"
year: 2008
journal: "Space Science Reviews"
doi: "10.1007/s11214-008-9455-8"
topic: Space_Weather
tags: [THEMIS, search-coil, magnetometer, substorm, plasma-waves, induction-coil, NEMI, flux-feedback, ULF, ELF]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 79. The Search Coil Magnetometer for THEMIS / THEMIS 서치코일 자력계

---

## 1. Core Contribution / 핵심 기여

이 논문은 NASA THEMIS 미션 5기 위성에 모두 탑재된 **삼축 서치코일 자력계(Search Coil Magnetometer, SCM)** 의 설계, 전기적 모델링, 그리고 지상 보정(calibration) 결과를 보고하는 instrument paper이다. SCM은 0.1 Hz – 4 kHz 의 ULF/ELF 자기장 변동을 측정해 서브스톰(substorm) 폭발과 확장상에 동반되는 플라즈마 파동(휘슬러 모드, 이온 사이클로트론파, 저주파 하이브리드 파, 풍선 모드 등)을 원격 진단한다. 핵심 설계 요소는 (a) 길이 170 mm × 직경 7 mm 의 고투자율 페로마그네틱 코어와 51,600 회 주권선이 만드는 큰 자기 증폭률 $\mu_{app}\simeq 1/N_z$, (b) 보조권선을 통한 자속 피드백(flux feedback)으로 RLC 공진을 제거하고 응답을 평탄화한 점, (c) MCM-V (Multi-Chip Module Vertical) 3D 패키징을 적용한 저잡음·저전력 전치증폭기(200 g, 75 mW), (d) Chambon-la-Forêt 보정 시설에서의 NEMI 측정 결과 5기 비행 모델(FM1–FM5) 모두 NEMI < 0.76 pT/√Hz @ 10 Hz 를 달성하여 1 pT/√Hz 사양을 능가한 것이다.

This instrument paper documents the design, electrical modeling, and ground calibration of the **tri-axial Search Coil Magnetometer (SCM)** flown on each of the five THEMIS probes. The SCM measures magnetic-field fluctuations in the ULF/ELF band (0.1 Hz – 4 kHz), where plasma waves implicated in substorm onset and expansion (whistler-mode, ion-cyclotron, lower-hybrid, ballooning) are expected. Key design choices include (a) a 170 mm × 7 mm high-permeability ferromagnetic core wound with 51,600 turns on the primary, yielding apparent permeability $\mu_{app}\simeq 1/N_z$, (b) flux feedback through a secondary winding that flattens the response and cancels the RLC resonance, (c) a low-noise low-power preamplifier built with MCM-V (Multi-Chip Module Vertical) 3D packaging at 200 g and 75 mW per probe, and (d) calibration at the Chambon-la-Forêt facility showing NEMI < 0.76 pT/√Hz at 10 Hz on every flight model (FM1–FM5), beating the 1 pT/√Hz specification.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Why a Search Coil for THEMIS? / 서론 — 왜 THEMIS에 서치코일인가? (p. 265–266)

THEMIS의 1차 목표는 "**서브스톰이 어디서, 언제 시작되는가**" 그리고 "이 폭발 과정의 본질이 무엇인가" 이다. 자기권 플라즈마는 충돌이 거의 없는 collisionless 환경이므로, 일부 서브스톰 모델이 요구하는 산일(dissipation)은 결국 wave-particle interaction을 통해 일어난다. 따라서 EFI(전기장)와 SCM(자기장)이 함께 같은 ULF/ELF 대역을 커버해 파동을 원격 탐지한다. THEMIS SCM은 GEOS 1/2, Ulysses, Galileo, Interball, Cluster STAFF, Cassini로 이어지는 CETP의 긴 heritage를 잇는다(Cornilleau-Wehrlin et al. 1997, 2003). 각 미션마다 주파수 범위·NEMI·질량 제약이 달라 매번 재설계가 필요하다.

THEMIS's primary thrust is to determine **where and when substorms start** and the nature of the explosive instability that triggers them. Because the magnetospheric plasma is almost collisionless, the dissipation required by some substorm models must occur via wave–particle interaction. Hence EFI (electric) and SCM (magnetic) together cover the same ULF/ELF range to remote-sense the waves. THEMIS SCM has a long lineage dating to CETP search coils on GEOS 1/2, Ulysses, Galileo, Interball, Cluster STAFF, and Cassini. Each mission imposes its own constraints on frequency range, NEMI, and mass, so the sensor is re-engineered every time.

### Part II: Measurement Requirements / 측정 요구사항 (Sect. 2, p. 266–267)

#### 2.1 Science Objectives / 과학 목표

서브스톰 폭발에는 두 큰 모델이 경합한다:

1. **Magnetic Reconnection (MR) 모델** — 자기 재결합이 먼저 발생해 서브스톰을 트리거. 재결합 영역에서 휘슬러 모드 파동이 사출되고, 이 파가 입자를 super-Alfvénic 속도까지 가속한다(Mandt et al. 1994). 또한 얇은 전류 시트는 HF tearing instability에 의해 불안정해진다(Bulanov et al. 1992).
2. **Current Disruption (CD) 모델** — 근지구 ($\sim 8\, R_E$) 플라즈마 시트의 cross-tail current가 감소하면서 instability가 발달. CD 모델에서는 cross-tail current가 (a) HF cross-field instability(Lui et al. 1992) 또는 (b) LF ballooning mode (Roux et al. 1991)에 의해 disrupt된다. Ballooning이 이온 사이클로트론·저주파 하이브리드 파와 결합한다.

두 모델 모두 ULF/ELF wave를 핵심 메커니즘으로 가정하므로 wave 관측은 결정적 시험. SCM은 추가로 파동의 **본질**(electrostatic vs electromagnetic) 도 판별한다 — 저주파 하이브리드는 거의 정전적이지만 휘슬러는 전자기적이다.

There are two leading classes of substorm-onset models. (1) **Magnetic Reconnection (MR)**: reconnection occurs first; whistler waves emitted from the diffusion region accelerate particles to super-Alfvénic speeds (Mandt et al. 1994), and HF tearing destabilizes thin current sheets (Bulanov et al. 1992). (2) **Current Disruption (CD)**: the near-Earth (~8 $R_E$) cross-tail current is disrupted by HF cross-field instability (Lui et al. 1992) or LF ballooning (Roux et al. 1991), which couples to ion-cyclotron and lower-hybrid waves. Both classes invoke ULF/ELF waves, so SCM observations provide a critical discriminator. SCM also reveals whether the waves are predominantly electrostatic (e.g., lower-hybrid) or electromagnetic (e.g., whistler).

#### 2.2.1 Frequency Range and NEMI / 주파수 범위와 NEMI

상한 주파수는 휘슬러 컷오프 $f_{ce}=eB/(2\pi m_e)$ 에 의해 결정된다. 표준 위치별로:
- **Near-Earth plasma sheet (~8 $R_E$, $B\simeq 50$ nT):** $f_{ce}\simeq 1.4$ kHz.
- **Geostationary (6.7 $R_E$):** $f_{ce}\simeq 4$ kHz (또는 substorm growth phase에 cross-tail current가 dipole 장을 압축할 때 ~2.8 kHz).
- **5 $R_E$:** dipole에서 $f_{ce}\simeq 9$ kHz지만 CD는 이 안쪽까지 들어오지 않는다.

따라서 4 kHz 상한이 충분. Cluster가 측정한 플라즈마 시트 파동 진폭은 10–100 pT/√Hz @ 10 Hz 이므로 NEMI 사양은 < 1 pT/√Hz @ 10 Hz 로 설정 — 신호 대 잡음비 10–100배 확보.

The upper cutoff is set by the whistler ceiling $f_{ce}=eB/(2\pi m_e)$. Standard locations give: $f_{ce}\simeq 1.4$ kHz at 8 $R_E$ (B≃50 nT, plasma sheet), ~4 kHz at geostationary (6.7 $R_E$, dipole), and 9 kHz at 5 $R_E$ — but CD does not occur inside 5–6 $R_E$, and whistlers are damped/unguided above $f_{ce}/2$. So 4 kHz is sufficient. Cluster recorded plasma-sheet wave amplitudes of 10–100 pT/√Hz at 10 Hz, motivating NEMI < 1 pT/√Hz at 10 Hz to give an order-of-magnitude SNR margin.

#### 2.2.2 Other Requirements / 기타 요구사항

위성 질량 제약으로 **800 g** (센서 600 g + 전치증폭기 200 g), 전력 **< 100 mW**. 자기축 정렬 정확도 **< 1°** — 파동 polarization 결정에 필요.

Mass budget: 800 g (600 g sensors + mounting, 200 g preamplifiers). Power: < 100 mW per sensor. Magnetic-axis alignment: better than 1° to determine wave polarization.

### Part III: Description of the Instrument / 장비 설계 (Sect. 3, p. 267–270)

#### 3.1.1 Magnetic Amplification / 자기 증폭 — Eq. 1–2

서치코일은 dB/dt 센서이다. **Lenz 법칙** $\mathcal{E}=-N\, d\Phi/dt$ 로부터, 사인파 자기장 $B_{ext}\propto e^{j\omega t}$ 에 대한 유도전압 진폭은:

$$
e \;=\; N S \left( \frac{1}{L}\int_0^L \mu_{app}(l)\, dl \right) B_{ext}\, \omega \;=\; N S \langle \mu_{app}\rangle B_{ext}\, \omega
\tag{1}
$$

- $N=51{,}600$: 주권선 turn 수 (감지 권선)
- $S$: 코어 단면적 ($\pi (3.5\,\text{mm})^2 \simeq 38.5\,\text{mm}^2$)
- $\langle\mu_{app}\rangle$: 권선 길이를 따라 평균낸 겉보기 투자율
- $\omega$: 각주파수 → **응답이 주파수에 비례하므로 저주파에서 작아지는 점이 본질적 한계**.

코어 내부장과 외부장의 비율은 **demagnetizing factor $N_z$** 로 정해진다. 원기둥 코어의 경우(Bozorth & Chapin 1942; Osborn 1945):

$$
\mu_{app}(m) \;=\; \frac{B_{core}}{B_{ext}} \;=\; \frac{\mu_r}{1+(\mu_r-1)\, N_z(m)}
\tag{2}
$$

여기서 $m=L/d$ 가 길이/지름 비. THEMIS 코어는 $\mu_r \gg 1$ 이므로 $(\mu_r-1)N_z \gg 1$ 가 되어 **$\mu_{app}\simeq 1/N_z(m)$** 의 형상-제한 영역. $N_z(m)$ 은 $m$ 이 클수록 작아지므로 (가늘고 길수록) $\mu_{app}$ 가 커진다. 그러나 (i) 길이 $L$ 을 늘리면 질량·부피가 커지고, (ii) 직경 $d$ 를 줄이면 단면적 $S\propto d^2$ 도 줄어 식 (1)의 $S\mu_{app}$ 곱 자체는 별로 커지지 않는다. 결국 타협으로 **$L=170$ mm, $d=7$ mm** ($m\approx 24$) 가 채택되었다.

A search coil is a dB/dt sensor. From Faraday/Lenz law $\mathcal{E}=-N\,d\Phi/dt$, a sinusoidal field $B_{ext}\propto e^{j\omega t}$ produces an EMF given by Eq. (1). Output scales as $\omega$, so low-frequency sensitivity is intrinsically poor, requiring large $N$ and large $\mu_{app}$. The ratio $B_{core}/B_{ext}$ is governed by the demagnetizing factor $N_z$ (Eq. 2). For the THEMIS core $\mu_r \gg 1$, so $\mu_{app}\simeq 1/N_z(m)$ — a shape-limited regime where slender rods are best. Lengthening the core boosts $\mu_{app}$ but adds mass; thinning it reduces $S$. The compromise: $L=170$ mm, $d=7$ mm ($m\approx 24$).

#### 3.1.2 Electrical Modeling — RLC Circuit and Resonance / 전기 모델링과 공진 (Eq. 3, Fig. 2)

권선 자체가 인덕터·저항·분포 정전용량을 가지므로 sensor는 **RLC 직렬 공진 회로**처럼 거동한다. THEMIS의 표시값: $L_1=7.5$ H, $R_1=1$ kΩ, $C_1=55$ pF (Fig. 2). 전압원은 식 (1) 의 유도 EMF.

Sensor without feedback의 전달함수:

$$
T(j\omega) \;=\; \frac{V}{B} \;=\; \frac{-j\omega N S \mu_{app}}{(1-LC\omega^2) + jRC\omega}
\tag{3}
$$

- 저주파 ($\omega \ll \omega_0$): $T \approx -j\omega N S \mu_{app}$ — 평탄하지 않고 $\omega$에 비례.
- 공진 ($\omega = \omega_0 \equiv 1/\sqrt{LC}$): $T$ 발산 → 매우 큰 transmittance peak (Fig. 7 pink curve, ~1 kHz 부근).
- 고주파 ($\omega \gg \omega_0$): $T \propto -1/(j\omega LC) \cdot N S \mu_{app}$ → 1/$\omega$ 로 떨어짐.

값을 대입해 보면 $\omega_0=1/\sqrt{(7.5)(55\times 10^{-12})} \approx 4.92\times 10^4$ rad/s, $f_0\approx 7.8$ kHz 인데, 식 (1) 의 turn 수와 코어 자기증폭이 함께 작용하면 실제 측정 공진은 약 1 kHz 부근에 위치 (Fig. 7).

The winding behaves as a series RLC circuit (Fig. 2: $L_1=7.5$ H, $R_1=1$ kΩ, $C_1=55$ pF) driven by the EMF of Eq. (1). The bare transmittance Eq. (3) shows three regimes: rising as $\omega$ at low frequency, diverging at $\omega_0=1/\sqrt{LC}$, and rolling off as $1/\omega$ above. The pink curve in Fig. 7 (search coil without feedback) shows a sharp resonance peak near 1 kHz — clearly unsuited to wide-band wave measurement.

#### 3.1.2 (cont.) Flux Feedback / 자속 피드백 (Fig. 3)

공진 문제를 해결하기 위해 보조권선과 high-gain 전치증폭기로 음(negative) 자속 피드백을 구현한다(Fig. 3):

1. 1차 권선이 외부장에 의해 유도된 EMF를 측정.
2. 전치증폭기(PA)가 이를 증폭.
3. 출력의 일부를 피드백 저항을 통해 **2차 권선에 전류**로 주입 → 코어 안에 외부장과 **반대 방향** 자속이 형성됨.
4. 결과적으로 원래 외부장에 의한 자속이 effectively cancel되어 코어가 saturation 되지 않고, RLC dynamics가 유효하게 사라진다.

피드백 저항이 작을수록 피드백 강도가 강해져 응답이 더 평탄해지고 위상도 안정. Fig. 7의 파란 곡선이 with-feedback 응답 — 1 Hz 부근부터 공진까지 평탄한 dB/dt 응답을 보인다(저주파 1/$\omega$ rolloff는 dB/dt 의 본질).

To cancel the resonance, a **negative flux feedback loop** is implemented (Fig. 3). The PA output drives a current through a feedback resistor into the secondary winding, producing a flux opposite to that induced by $B_{ext}$. This effectively shorts out the RLC dynamics, flattening and stabilizing the amplitude and phase response. A smaller feedback resistor means stronger feedback. Fig. 7 (blue curve) shows the with-feedback response is essentially flat in dB/dt from ~1 Hz up through the resonance, then rolls off above ~1 kHz.

#### 3.2.1 SCM Antennas / 안테나 구조 (Fig. 4–5)

- 3축 직교(x, y, z) 배열.
- 1 m boom 끝에 mounted (전자기 잡음 감소).
- 2 sensor는 spin plane, 1 sensor는 spin axis 방향.
- Sensor 자체 길이 18 cm (그리고 27 cm "kin"); CETP가 7+ 미션에서 비행한 검증된 디자인의 직접 후예.
- 안테나 + mounting 질량 568 g (사양 600 g 이하 만족).
- 전기장 신호에 대한 **전자기 차폐(electrostatic shielding)** 적용.
- Mechanical–magnetic axis 차이 0.2°(FM3 X) 정도, 사양 (1°) 보다 훨씬 정밀. 평균 추정 오차 0.5°.

Antennas: tri-axial orthogonal mounting on a 1 m boom; two sensors in the spin plane, one along the spin axis. The 18 cm sensor (with a 27 cm kin) inherits the heritage flown by CETP on 7+ Earth-orbiting and interplanetary missions, including Cluster STAFF. Total antenna + mounting mass = 568 g (under 600 g spec). Electrostatic shielding suppresses E-field pickup. Measured magnetic-axis vs. mechanical-axis misalignment: ~0.2° (FM3 X-axis), with mean estimation error 0.5° — comfortably under the 1° requirement.

#### 3.2.2 Preamplifier — MCM-V Technology / MCM-V 전치증폭기 (Fig. 6)

3개의 저잡음 전치증폭기(센서축당 1개)가 IDPU 박스에 부착된 별도 전기 유닛에 들어간다. **MCM-V (Multi-Chip Module Vertical)** 신기술로:
- 회로를 작은 기능 블록으로 나누어 얇은 flexible PCB에 bare chip을 붙이고, 이를 적층해 큐브 형태로 만든다.
- 큐브 사이에 탄탈럼(tantalum) plate를 넣어 방사선 차폐(spot shielding) — 전통적 방식보다 가볍다.
- 1024-times smaller volume에 dynamic range ~100 dB.
- 1차 권선의 큰 DC 자기장(우주선 회전에 의한 ~1000 nT 의사 신호)에 견디는 low-noise input stage.

Preamplifier box: 95 × 81 × 30 mm, 200 g, 75 mW. SCM calibration signal generator도 박스 안에 내장 — 정기적 in-flight 검증.

Three low-noise preamplifiers (one per axis) are mounted in a separate unit fixed to the IDPU. They use **MCM-V (Multi-Chip Module Vertical)** technology: bare chips on thin flexible PCBs, stacked into cubes (one cube per preamp + one for power = 4 cubes total). Tantalum plates between layers provide spot radiation shielding lighter than monolithic shielding. Dynamic range ~100 dB allows weak signals to be measured against the large DC field induced by spacecraft spin (~1000 nT pseudo-signal). Mass 200 g, power 75 mW per probe — both within spec. An on-board calibration generator inside the preamp box enables routine in-flight checks.

### Part IV: Calibrations and Tests / 보정 및 테스트 (Sect. 4, p. 270–273)

지상 보정은 **Chambon-la-Forêt** 자기 관측소(파리 남쪽 ~100 km, 외부 자기 잡음이 낮은 site)의 보정 시설에서 수행되었다.

전달함수와 NEMI 측정(Fig. 7, 8):
- Fig. 7 — pink: search coil only (RLC resonance ~1 kHz 의 sharp peak). blue: with feedback (1 Hz – 1 kHz 평탄).
- Fig. 8 — pink: 전달함수 in dB V/nT (≈ 약 −5 dB at 1 kHz peak, $\sim -60$ dB at 0.1 Hz). blue: NEMI in pT/√Hz, 10 Hz 부근 0.6–0.7 pT/√Hz, 1 kHz에서 ~0.02 pT/√Hz 로 감소 (1/$f$ 잡음 감소 기여).

5기 비행 모델(FM1–FM5) 의 NEMI 일관성 — Tables 1–5:

| Probe | 10 Hz | 100 Hz | 1 kHz |
|---|---|---|---|
| FM1 X / Y / Z | 0.69 / 0.65 / 0.64 | 0.070 / 0.072 / 0.077 | 0.022 / 0.021 / 0.021 |
| FM2 X / Y / Z | 0.64 / 0.65 / 0.645 | 0.0699 / 0.0698 / 0.076 | 0.020 / 0.020 / 0.022 |
| FM3 X / Y / Z | 0.61 / 0.66 / 0.74 | 0.070 / 0.067 / 0.066 | 0.0196 / 0.016 / 0.019 |
| FM4 X / Y / Z | 0.71 / 0.76 / 0.74 | 0.079 / 0.080 / 0.077 | 0.017 / 0.019 / 0.019 |
| FM5 X / Y / Z | 0.66 / 0.70 / 0.72 | 0.065 / 0.074 / 0.069 | 0.016 / 0.016 / 0.016 |

(단위: pT/√Hz)

핵심 결과:
- 모든 FM·축에서 NEMI(10 Hz) ≤ 0.76 pT/√Hz, 사양 1 pT/√Hz 능가.
- NEMI(100 Hz) 최대 0.08 pT/√Hz, NEMI(1 kHz) 최대 0.022 pT/√Hz.
- 전달함수는 5기 사이에 1 dB 미만 차이 → multi-spacecraft 비교 분석에 적합.

**In-flight calibration**: DFB 가 9 Hz triangular wave를 secondary winding에 주입 → 1차 권선이 검출 → $B_x, B_y, B_z$ 각각 응답 확인. 기본 30초, 최대 60초. 궤도당 1회 수행해 transfer function 안정성 점검.

Ground calibration was performed at the **Chambon-la-Forêt** observatory facility. Fig. 7 demonstrates the dramatic effect of flux feedback: the bare RLC peak near 1 kHz is removed, leaving a flat dB/dt response from ~1 Hz to >1 kHz. Fig. 8 plots transfer function (pink, dBV/nT) and NEMI (blue, pT/√Hz) vs frequency. Tables 1–5 list NEMI per axis at 10 Hz, 100 Hz, 1 kHz for FM1–FM5 — all five flight models are within 1 dB of each other in transfer function. The largest NEMI is 0.76 pT/√Hz at 10 Hz (well below the 1 pT/√Hz spec), 0.08 pT/√Hz at 100 Hz, and 0.022 pT/√Hz at 1 kHz. In-flight calibration uses a 9 Hz triangular wave injected into the secondary, detected by the primary, run for 30–60 s once per orbit.

### Part V: Telemetry Modes / 텔레메트리 모드 (Table 6, p. 273–274)

위성 대역폭은 한정적이므로 SCM은 6가지 텔레메트리 모드를 조합해 운용한다:

| APID | Mode | Sampling | Notes |
|---|---|---|---|
| 440 | Filter bank (fbk) | 1/16 – 8 S/s | 6 frequency bands: [2–4]Hz, [4–8], [16–8]kHz... mean values |
| 444 | Fast survey (scf) | 2 – 256 S/s (nominal 8) | Waveform 3-axis |
| 448 | Particle burst (scp) | up to 256 S/s | scf 동일 처리 |
| 44C | Wave burst (scw) | up to 8192 S/s | scf 동일 처리 |
| 44D | Particle burst spectra (ffp) | 1/4 – 8 spec/s | Compressed FFT, 16/32/64 lines |
| 44E | Wave burst spectra (ffw) | up to 8 spec/s | 64 lines, 8 spec/s nominal |

추가로 16 bit 디지타이저, 1000 nT DC 신호에서도 saturation 방지, ~4 $R_E$ 안쪽에서는 spin frequency에서 saturation 우려가 있어 사용 위치 제한. 0.1–4 kHz 대역은 SCM (자기) + EFI (전기) 가 same manner로 처리.

The probe telemetry is shared across six SCM modes (Table 6): a Filter Bank (means in 6 bands, low rate), three waveform modes (Fast Survey, Particle Burst, Wave Burst) at increasing rates up to 8192 S/s, and two on-board FFT modes (Particle/Wave Burst Spectra) with 16/32/64 frequency lines. SCM signals are digitized at 16-bit; the dynamic range is sized to avoid saturation up to ~1000 nT AC (the spin-induced quasi-DC field saturates inside ~4 $R_E$). DFB and IDPU process SCM and EFI in the same manner across the 0.1 Hz – 4 kHz band.

### Part VI: Summary / 요약 (Sect. 5, p. 274)

- THEMIS 5기 위성 모두 동일한 주파수 응답 / NEMI / 자기축 정렬을 가진 SCM을 탑재.
- 0.1 Hz – 4 kHz 대역이 EFI + SCM 으로 동일 방식 처리.
- 모든 NEMI 측정값이 사양 만족, 5기 사이 1 dB 이내 일관성.
- 궤도당 1회 in-flight calibration.
- 모든 SCM이 정상 작동 중.

All five THEMIS SCMs are working nominally, satisfy NEMI specifications (<0.76 pT/√Hz at 10 Hz, ~0.022 pT/√Hz at 1 kHz), and have transfer functions matched within 1 dB across spacecraft. Onboard calibration is performed once per orbit.

---

## Extended Context: Induction-Coil Physics / 확장 맥락: 유도코일 물리

본 논문이 instrument paper이므로, 유도형 자기 센서의 일반 이론을 정리해 둔다.

### A. Faraday → Output Voltage / 패러데이 → 출력전압

자기 플럭스 $\Phi=\int \mathbf{B}\cdot d\mathbf{S}$. N turn 코일에서:

$$
V(t) = -N\frac{d\Phi}{dt} = -NS\frac{dB(t)}{dt}
$$

사인파 $B(t)=B_0 \cos(\omega t)$ 에 대해 $|V|=NS B_0 \omega$ 이고, 강자성 코어가 있으면 $\mu_{app}$ 만큼 곱해진다. 응답은 본질적으로 **$\omega$ 비례 (high-pass)** — fluxgate 자력계와의 기본적 차이.

For a coil of N turns, $V(t)=-NS\,dB/dt$. With a ferromagnetic core, multiply by $\mu_{app}$. Output is intrinsically high-pass (proportional to $\omega$).

### B. Demagnetizing Factor — Geometry Matters / 반자기장 계수의 형상 의존성

자성체가 자화될 때, 양 끝에 형성되는 "자유 magnetic charge"가 내부에 역방향 장 $H_d=-N_z M$을 만든다. $N_z$ 는 [0, 1] 범위의 형상 인자. 표준 결과:
- 무한히 긴 막대 (rod): $N_z\to 0$ → $\mu_{app}\to \mu_r$.
- 구 (sphere): $N_z=1/3$ → $\mu_{app}=3\mu_r/(\mu_r+2)$ → $\mu_r\to\infty$ 에서 $\mu_{app}\to 3$.
- 짧고 두꺼운 원기둥 ($m=L/d \sim 1$): $N_z\sim 0.27$ → $\mu_{app}\sim 3.7$.
- THEMIS rod ($m\approx 24$): $N_z\approx 0.0036$ → $\mu_{app}\approx 280$ (대략 추정).

따라서 형상 자체가 ~10–100배의 자기 증폭을 결정한다.

When a ferromagnet is magnetized, surface "magnetic charges" produce an internal demagnetizing field $H_d=-N_z M$. Standard limits: long rod $N_z\to 0$, sphere $N_z=1/3$, short cylinder $N_z\sim 0.27$. THEMIS's $L/d\approx 24$ gives $N_z\approx 0.0036$ and $\mu_{app}\approx 280$.

### C. NEMI — What Limits It? / NEMI를 결정하는 잡음원

세 잡음원이 결합한다:
1. **Johnson–Nyquist (열 잡음)**: 권선 저항 $R$ 의 전압 잡음 $\sqrt{4 k_B T R}$. → 입력 환산 자기 잡음 $B_n=\sqrt{4 k_B T R}/(NS\mu_{app}\omega)$. $\omega$ 에 반비례 → 저주파에서 NEMI가 커지는 본질적 이유.
2. **Preamplifier voltage noise $e_n$**: 입력 환산 $e_n/(NS\mu_{app}\omega)$. 마찬가지로 $1/\omega$ 의존.
3. **Preamplifier current noise $i_n$ × inductive impedance**: 고주파에서 dominant.

THEMIS NEMI 곡선(Fig. 8)에서 0.1 Hz 근처 NEMI ≈ 1 nT/√Hz (열·전압 잡음), 100 Hz–1 kHz 에서 ≈ 0.02 pT/√Hz minimum, 그 위로 다시 증가하는 V자 형태가 정확히 이 모델로 설명된다.

NEMI is set by Johnson noise $\sqrt{4 k_B T R}$ in the winding, preamp voltage noise $e_n$, and preamp current noise $i_n$ — the first two referred to the input scale as $1/\omega$, while $i_n$ × inductive impedance dominates at high frequency. This produces the V-shaped NEMI curve (Fig. 8) with a minimum near 1 kHz.

### D. Plasma Waves the SCM Must Detect / SCM이 검출해야 하는 플라즈마 파동

| Wave / 파동 | Frequency / 주파수 | Polarization / 편파 | Substorm role / 서브스톰 역할 |
|---|---|---|---|
| Whistler / 휘슬러 | $f_{ci} \ll f \lesssim f_{ce}$ ($\sim$ kHz) | RH circular, EM | MR 모델에서 입자 가속 |
| Ion-cyclotron / 이온 사이클로트론 | $f \sim f_{ci}$ ($\sim 0.1$ Hz) | LH, mostly EM | CD 모델 wave coupling |
| Lower-hybrid / 저주파 하이브리드 | $f_{LH}=\sqrt{f_{ci}f_{ce}}$ ($\sim$ Hz–10 Hz) | electrostatic | Cross-field current disruption |
| Ballooning / 풍선 모드 | ULF (mHz–Hz) | MHD | Pressure-driven CD trigger |
| Pc waves / Pc 파동 | mHz–수 Hz | various | Ground-coupled coherent oscillations |

Cold plasma 분산관계로부터:

- **Whistler dispersion**: $\omega = \frac{k^2 c^2 \omega_{ce}\cos\theta}{\omega_{pe}^2 + k^2 c^2}$. 평행 전파 ($\theta=0$) 에서 $\omega < \omega_{ce}$ 이고 group velocity $\propto \sqrt{\omega(\omega_{ce}-\omega)}$ — 이것이 lightning whistler의 frequency-time chirp을 만든다.
- **Ion-cyclotron R/L mode**: $k^2 c^2/\omega^2 = 1 - \omega_{pi}^2/[\omega(\omega \pm \omega_{ci})]$. L mode는 $\omega = \omega_{ci}$에서 cutoff, 강한 입자 공명.

These dispersion relations let one infer plasma parameters from observed wave frequencies, given the ambient field measured by the fluxgate magnetometer (FGM).

---

## 3. Key Takeaways / 핵심 시사점

1. **유도코일은 dB/dt 센서이며 응답은 본질적으로 $\omega$ 비례 / Induction coils are dB/dt sensors with intrinsically $\omega$-proportional response** — 이 사실이 SCM의 NEMI가 저주파에서 큰 V자 형태를 갖게 만들고, FGM (Fluxgate)와의 보완 관계를 결정한다 (DC와 1 Hz 이하는 FGM, 1 Hz 이상은 SCM 우세). The high-pass nature of induction sensing dictates the V-shaped NEMI (Fig. 8) and the FGM/SCM crossover near ~1 Hz.

2. **자기 증폭은 형상 인자 $1/N_z$ 에 의해 지배된다 / Magnetic amplification is geometry-limited by $1/N_z$** — 매우 높은 $\mu_r$ 코어에서는 $\mu_{app}\to 1/N_z(m)$ 가 상한. 따라서 미션 설계는 사실상 length-to-diameter 비 $m=L/d$ 의 최적화 문제로 환원된다. 질량(부피) 와 단면적(감도) 을 동시에 고려한 결과 THEMIS는 $m\approx 24$ 를 채택. For high-$\mu_r$ cores, $\mu_{app}\to 1/N_z(m)$, so design reduces to optimizing the length-to-diameter ratio against mass and cross-section. THEMIS chose $m\approx 24$.

3. **자속 피드백은 단순히 응답을 평탄화하는 것이 아니라, 비선형성·온도 의존성도 제거 / Flux feedback flattens response, but more importantly removes nonlinearity and temperature drift** — 코어 자화 곡선의 hysteresis와 온도 의존 $\mu_r$ 이 모두 cancel되므로, 보정 안정성이 극적으로 향상된다. Feedback also linearizes hysteresis and cancels $\mu_r$ temperature drift, dramatically improving calibration stability.

4. **NEMI 0.76 pT/√Hz @ 10 Hz는 568 g 안테나로 SQUID급 감도 / NEMI 0.76 pT/√Hz at 10 Hz is SQUID-class sensitivity in 568 g** — 지상 SQUID 자력계가 ~수 fT/√Hz 수준이지만 cryogenic 환경 필요. THEMIS는 상온·우주 환경에서 ~pT/√Hz 를 달성, mass-normalized로는 사실상 best-in-class. SQUID-class noise in a room-temperature, space-qualified, 568 g package — best mass-normalized magnetic-wave sensitivity at the time of flight.

5. **MCM-V 3D 패키징은 우주 자기 측정 산업 표준으로 정착 / MCM-V 3D packaging became an industry standard for space magnetic instrumentation** — 적층 PCB + 탄탈럼 spot shielding으로 mass와 radiation tolerance를 동시에 잡았고, 이후 MMS·Solar Orbiter·JUICE 등에 모두 변형 적용. The MCM-V approach (stacked thin PCBs + tantalum spot shielding) was widely adopted by MMS, Solar Orbiter, JUICE, and other follow-on missions.

6. **5기 위성 NEMI/transfer function 1 dB 이내 일관성이 multi-spacecraft 분석을 가능케 함 / Inter-spacecraft consistency within 1 dB enables true multi-point analyses** — 단일 위성이 아닌 5기로부터 동일 신호를 비교해야 하는 THEMIS의 핵심 미션 설계 요구. Per-axis NEMI 0.61–0.76 pT/√Hz 의 spread는 내부 비교에 충분히 작다. Without sub-dB cross-calibration, multi-point timing of substorm onset (the mission's headline result) would not have been possible.

7. **상한 4 kHz는 휘슬러 컷오프 $f_{ce}/2$ 에 맞춰진 물리 기반 결정 / The 4 kHz upper cutoff is physics-driven, set by half the electron gyrofrequency at the geostationary orbit** — 단순히 "기술적으로 가능한 한 높이" 가 아니라, $B$ 와 위치 통계로부터 도출된 합리적 사양. 8 $R_E$에서 $f_{ce}\simeq 1.4$ kHz, 6.7 $R_E$ growth phase 시 ~2.8 kHz, 5 $R_E$ 안쪽은 CD 미발생. The 4 kHz spec comes from $f_{ce}/2$ at substorm-relevant locations, not from technology limits — an example of physics-driven design.

8. **Flux feedback이 dynamic range를 100 dB까지 확보해 spin-induced 1000 nT 의사 신호와 ~pT 파동을 동시에 측정 / Flux feedback combined with 16-bit digitizers gives ~100 dB dynamic range, accommodating the 1000 nT spin-induced offset and ~pT science signal in one stream** — saturation-free wide-band wave science는 이 dynamic range가 없으면 불가능. The 100 dB dynamic range is what allows weak waves to be seen against the spin-modulated DC field — a non-trivial systems engineering achievement.

---

## 4. Mathematical Summary / 수학적 요약

### Faraday EMF in a wound coil / 권선 EMF

$$
\boxed{\; e \;=\; N S \langle\mu_{app}\rangle B_{ext} \omega \;} \quad\text{(Eq. 1)}
$$

| 기호 / Symbol | 의미 / Meaning |
|---|---|
| $N$ | 주권선 turn 수 = 51,600 |
| $S$ | 코어 단면적 ($\pi(d/2)^2 \approx 38.5\,\text{mm}^2$) |
| $\langle\mu_{app}\rangle$ | 권선 길이 평균 겉보기 투자율 |
| $B_{ext}$ | 외부 자기장 진폭 |
| $\omega = 2\pi f$ | 신호 각주파수 |

### Apparent permeability of a cylindrical core / 원기둥 코어의 겉보기 투자율

$$
\boxed{\; \mu_{app}(m) = \frac{\mu_r}{1+(\mu_r-1)N_z(m)} \;\xrightarrow[\mu_r\to\infty]{}\; \frac{1}{N_z(m)} \;} \quad\text{(Eq. 2)}
$$

| 기호 / Symbol | 의미 / Meaning |
|---|---|
| $\mu_r$ | 자성 재료의 상대 투자율 (≫ 1) |
| $N_z(m)$ | 길이/지름 비 $m=L/d$ 에 의존하는 반자기장 계수 |
| $m=L/d$ | 코어 형상비 (THEMIS: $170/7 \approx 24$) |

근사식 (slender rod, $m\gg 1$): $N_z(m) \approx \frac{\ln(2m)-1}{m^2}$ (Bozorth & Chapin 근사). $m=24$ 대입 시 $N_z\approx (\ln 48 -1)/576 \approx 2.87/576 \approx 5.0\times 10^{-3}$, 즉 $\mu_{app}\approx 200$ (수치적으로 paper 가 인용한 값들과 일치하는 대략값).

### Sensor transmittance with RLC dynamics / RLC 전달함수

$$
\boxed{\; T(j\omega) = \frac{V}{B} = \frac{-j\omega N S \mu_{app}}{(1-LC\omega^2) + jRC\omega} \;} \quad\text{(Eq. 3)}
$$

| 기호 / Symbol | 의미 / Meaning |
|---|---|
| $L$ | 권선 인덕턴스 (≈ 7.5 H) |
| $R$ | 권선 + 회로 저항 (≈ 1 kΩ) |
| $C$ | 분포 정전용량 (≈ 55 pF) |
| $\omega_0=1/\sqrt{LC}$ | 공진 각주파수 (≈ 4.92×10⁴ rad/s) |

극한 거동:

- $\omega \ll \omega_0$ (저주파): $T \approx -j\omega NS\mu_{app}$, $|T|\propto \omega$ (high-pass)
- $\omega = \omega_0$ (공진): $|T| \to NS\mu_{app}/(RC)$ — 매우 큼 (Q factor)
- $\omega \gg \omega_0$ (고주파): $T \approx -NS\mu_{app}/(jLC\omega)$, $|T|\propto 1/\omega$ (low-pass)

**Quality factor**: $Q = \omega_0 L/R = (1/R)\sqrt{L/C}$. THEMIS 값으로 $Q \approx (1/1000)\sqrt{7.5/(55\times 10^{-12})} \approx 369$ — 매우 sharp한 공진. 피드백으로 effective Q 를 1 미만으로 줄임.

### Closed-loop transfer with flux feedback / 피드백 포함 전달함수

부귀환의 일반 결과:

$$
T_{cl}(j\omega) = \frac{T(j\omega)}{1 + T(j\omega)\beta(j\omega)}
$$

여기서 $\beta$ 는 secondary winding을 통한 feedback 자속 / 입력 전압 비. 강한 feedback ($|T\beta|\gg 1$) 영역에서 $T_{cl}\approx 1/\beta$ — 외부 사양에 의해 결정되고 sensor의 $\mu_r(T)$ 변동에 무관해진다.

In the strong-feedback limit $|T\beta|\gg 1$, $T_{cl}\to 1/\beta$, set by the precision passive feedback element rather than by the temperature-dependent core $\mu_r$.

### NEMI / 등가 자기 잡음

$$
\boxed{\; B_n(\omega) = \frac{e_n^{\text{eq}}(\omega)}{NS\mu_{app}\omega} \;}
$$

여기서 $e_n^{\text{eq}}$ 는 모든 잡음원(Johnson, preamp $e_n$, $i_n Z_L$) 의 입력 환산 합. 저주파에서는 $1/\omega$ 로 발산, 고주파에서는 $i_n L \omega$ 가 dominant — V자 곡선.

| Frequency | THEMIS NEMI (worst FM) |
|---|---|
| 10 Hz | 0.76 pT/√Hz |
| 100 Hz | 0.080 pT/√Hz |
| 1 kHz | 0.022 pT/√Hz |

### Whistler-mode dispersion (parallel propagation) / 휘슬러 분산 (평행 전파)

$$
\frac{c^2 k^2}{\omega^2} = 1 - \frac{\omega_{pe}^2}{\omega(\omega - \omega_{ce})}
$$

For $\omega_{pe}\gg \omega_{ce}\gg\omega$: $\omega \approx (k^2 c^2/\omega_{pe}^2)\omega_{ce}$ — quadratic in $k$, group velocity $v_g \propto \sqrt{\omega(\omega_{ce}-\omega)}$.

### Ion-cyclotron L-mode / 이온 사이클로트론 L모드

$$
\frac{c^2 k^2}{\omega^2} = 1 - \frac{\omega_{pi}^2}{\omega(\omega + \omega_{ci})}
$$

Cutoff 가 $\omega_{ci}$ 에서 발생, 강한 입자 공명 흡수.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1831 ── Faraday: induction discovered (foundation of all coil sensors)
1942 ── Bozorth & Chapin: demagnetizing factors of rods
1945 ── Osborn: demagnetizing factors of the general ellipsoid
1958 ── Storey: VLF whistler dispersion in magnetosphere
1965 ── OGO-1: first space-borne search-coil magnetometer
1977 ── GEOS-1 search coil (CETP heritage begins)
1991 ── Roux et al.: westward traveling surge / ballooning mode (CD model)
1992 ── Lui et al.: cross-field current instability
1992 ── Bulanov et al.: HF tearing in thin current sheets
1994 ── Mandt et al.: whistler-mediated reconnection
1997 ── Cornilleau-Wehrlin et al.: Cluster STAFF design
2001 ── Ripka: "Magnetic sensors and magnetometers" textbook
2001 ── Lui: "Current controversies in magnetospheric physics" review
2003 ── Cornilleau-Wehrlin et al.: Cluster STAFF first results
2007 ── Coillot et al.: improved search-coil design (sensors letter)
2008 ── ★ Roux et al.: THEMIS SCM (this paper)
2008 ── ★ Le Contel et al.: THEMIS SCM first results
2008 ── ★ Angelopoulos et al.: THEMIS substorm onset result (Science)
2015 ── MMS launched: 4-spacecraft SCM cluster, sub-pT/√Hz
2018 ── Parker Solar Probe FIELDS SCM
2023 ── JUICE RPWI search-coil for Jupiter system
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Cornilleau-Wehrlin et al. (1997, 2003) Cluster STAFF | 직접적 설계 모태; 같은 RLC + flux feedback + MCM-V 후속 / Direct design predecessor with same RLC + feedback architecture | High — THEMIS SCM이 거의 그대로 유산을 잇는다 |
| Bozorth & Chapin (1942), Osborn (1945) | $N_z$ 계산의 수학적 기반 / Mathematical basis for demagnetizing factor $N_z$ | Foundational — Eq. (2) 의 핵심 |
| Roux et al. (1991) — westward traveling surge | 본 논문 1저자의 ballooning/CD 작업; SCM이 직접 검증해야 할 wave 모델 / Author's own ballooning work that this instrument is designed to test | High — instrument이 과학을 위해 만들어진 이유 |
| Lui et al. (1992) — cross-field current instability | CD 모델의 다른 wave 후보; SCM의 측정 대상 / The HF wave candidate for CD that SCM must detect | High |
| Mandt et al. (1994) — whistler-mediated reconnection | MR 모델의 wave 메커니즘; SCM이 식별해야 할 신호 / The MR wave mechanism SCM must identify | High |
| Bonnell et al. (2008) — THEMIS EFI | 같은 0.1 Hz–4 kHz 대역 전기장 측정; SCM과 함께 wave polarization 결정 / E-field counterpart in the same band; together they determine polarization | High — pair instrument |
| Cully et al. (2008) — THEMIS DFB | SCM 신호를 디지털화·FFT·필터뱅크 처리 / Digitizes, FFTs, filters SCM analog signals | Direct dependency |
| Taylor et al. (2008) — THEMIS IDPU | SCM 텔레메트리 모드 호스팅 / Hosts SCM telemetry modes | Direct dependency |
| Angelopoulos et al. (2008) — THEMIS mission overview | 미션 컨텍스트 및 5-spacecraft 컨셉 / Mission context and 5-probe concept | Foundational |
| Le Contel et al. (2008) — THEMIS SCM first results | 같은 호에 발표된 자매 논문, 첫 비행 데이터 / Sister paper with first in-flight data | Direct sibling — required reading pair |
| Coillot et al. (2007) | 같은 그룹의 sensor letter; 본 논문 설계 개선의 직접 출처 / Same group, direct source of design improvements | High |
| Ripka (2001) textbook | 자기 센서 일반 이론 / General magnetic sensor theory | Background reference |

---

## 7. References / 참고문헌

- A. Roux, O. Le Contel, C. Coillot, A. Bouabdellah, B. de la Porte, D. Alison, S. Ruocco, M.C. Vassal, "The Search Coil Magnetometer for THEMIS", *Space Sci. Rev.* **141**, 265–275 (2008). DOI: 10.1007/s11214-008-9455-8
- V. Angelopoulos et al., "The THEMIS mission", *Space Sci. Rev.* **141** (2008). DOI: 10.1007/s11214-008-9336-1
- J.W. Bonnell, F.S. Mozer, G.T. Delory, A.J. Hull, R.E. Ergun, C.M. Cully, V. Angelopoulos, "The Electric Field Instrument (EFI) for THEMIS", *Space Sci. Rev.* **141** (2008).
- O. Le Contel et al., "First results of THEMIS Search Coil Magnetometers (SCM)", *Space Sci. Rev.* **141** (2008). DOI: 10.1007/s11214-008-9371-y
- C.M. Cully, R.E. Ergun, K. Stevens, A. Nammari, J. Westfall, "The THEMIS digital fields board", *Space Sci. Rev.* **141** (2008). DOI: 10.1007/s11214-008-9417-1
- E. Taylor et al., "The THEMIS Instrument Data Processing Unit", *Space Sci. Rev.* **141** (2008).
- N. Cornilleau-Wehrlin et al., "The Cluster Spatio-temporal Analysis of Field Fluctuations (STAFF) experiment", *Space Sci. Rev.* **79**, 107–136 (1997).
- N. Cornilleau-Wehrlin et al., "First results obtained by the Cluster STAFF experiment", *Ann. Geophys.* **21**, 437–456 (2003).
- C. Coillot, J. Moutoussamy, P. Leroy, G. Chanteur, A. Roux, "Improvements on the design of search-coil magnetometer for space experiments", *Sens. Lett.* **5**, 1–4 (2007).
- R.M. Bozorth, D.M. Chapin, "Demagnetizing factors of rods", *J. Appl. Phys.* **13**, 320–326 (1942).
- J.A. Osborn, "Demagnetizing factors of the general ellipsoid", *Phys. Rev.* **67**, 351–357 (1945).
- P. Ripka, *Magnetic sensors and magnetometers*, Artech House (2001).
- A.T.Y. Lui, "Current controversies in magnetospheric physics", *Rev. Geophys.* **39**(4), 535–563 (2001).
- A.T.Y. Lui, R.E. Lopez, B.J. Anderson et al., "Current disruption in the near-Earth neutral sheet region", *J. Geophys. Res.* **97**, 1461 (1992).
- M.E. Mandt, R.E. Denton, J.F. Drake, "Transition to whistler mediated magnetic reconnection", *Geophys. Res. Lett.* **21**(1), 73–77 (1994).
- S.V. Bulanov, F. Pegoraro, A.S. Sakharov, "Magnetic reconnection in electron dynamics", *Phys. Fluids B* **4**, 2499–2508 (1992).
- A. Roux, S. Perraut, P. Robert et al., "Plasma sheet instability related to the westward traveling surge", *J. Geophys. Res.* **96**, 17,697 (1991).
- M. Ludlam et al., "The THEMIS magnetic cleanliness program", *Space Sci. Rev.* **141** (2008). DOI: 10.1007/s11214-008-9423-3
- P.R. Harvey et al., "The THEMIS observatory", *Space Sci. Rev.* **141** (2008).
