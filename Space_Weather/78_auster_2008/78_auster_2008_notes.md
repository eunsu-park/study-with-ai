---
title: "The THEMIS Fluxgate Magnetometer"
authors: ["H.U. Auster", "K.H. Glassmeier", "W. Magnes", "O. Aydogar", "W. Baumjohann", "D. Constantinescu", "D. Fischer", "K.H. Fornacon", "E. Georgescu", "P. Harvey", "O. Hillenmaier", "R. Kroth", "M. Ludlam", "Y. Narita", "R. Nakamura", "K. Okrafka", "F. Plaschke", "I. Richter", "H. Schwarzl", "B. Stoll", "A. Valavanoglou", "M. Wiedemann"]
year: 2008
journal: "Space Science Reviews"
doi: "10.1007/s11214-008-9365-9"
topic: Space_Weather
tags: [magnetometer, fluxgate, THEMIS, instrumentation, substorm, multi-spacecraft, calibration]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 78. The THEMIS Fluxgate Magnetometer / THEMIS 플럭스게이트 자력계

---

## 1. Core Contribution / 핵심 기여

This paper documents the design, calibration philosophy, and first half-year flight performance of the Fluxgate Magnetometer (FGM) flown identically on the five THEMIS probes launched in February 2007. The instrument resolves an apparently contradictory specification — 0.01 nT short-term sensitivity, ±25,000 nT range for attitude work near perigee, 64 Hz vector cadence, and offset stability of <0.2 nT/hour — by adopting a fully digital fluxgate architecture in which the AC signal from the pick-up coil is digitized at 32,768 Hz immediately after preamplification, and all subsequent feedback control, harmonic detection, scaling, and decimation are performed inside an Actel RT54SX72 FPGA hosting a 32-bit RISC processor. The paper presents the heritage chain from Helios/Phobos through Rosetta-Lander and VenusExpress, demonstrates that ground calibration meets every requirement at the pT/°C level, and shows two science vignettes from commissioning: a 15 pT/√Hz inflight noise statistic across 15 sensor axes and a magnetopause "string-of-pearls" multi-crossing on 7 August 2007 that yields ~67–95 km/s magnetopause oscillation speeds.

본 논문은 2007년 2월 발사된 다섯 대의 THEMIS 위성에 동일하게 탑재된 플럭스게이트 자력계(FGM)의 설계, 보정 철학, 그리고 첫 반년간 비행 성능을 정리한다. 이 기기는 0.01 nT 단기 감도, perigee에서의 자세 결정용 ±25,000 nT 범위, 64 Hz 벡터율, 시간당 0.2 nT 미만의 오프셋 안정도라는 일견 모순적인 사양을, 픽업 코일 AC 신호를 사전 증폭 직후 32,768 Hz로 디지털화하고 이후의 피드백 제어·고조파 검출·스케일링·다운샘플링을 모두 Actel RT54SX72 FPGA(32-bit RISC 프로세서 내장) 안에서 수행하는 완전 디지털 플럭스게이트 구조로 해결한다. 논문은 Helios/Phobos를 거쳐 Rosetta-Lander, VenusExpress로 이어지는 헤리티지 사슬을 보이고, 지상 보정이 pT/°C 수준에서 모든 요구를 충족함을 입증하며, 시운전 중 두 과학 사례를 제시한다: 15개 센서축에 걸친 15 pT/√Hz 비행 잡음 통계와 2007년 8월 7일 5점 magnetopause "구슬꿰미" 통과 사건에서 도출한 67–95 km/s magnetopause 진동 속도.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Science Requirements (Sect. 1–2) / 서론과 과학 요구사항

THEMIS는 perigee 1 RE, apogee 10–30 RE의 타원궤도를 도는 5기 위성 임무이다. 자력계는 태양풍·자기초·자기꼬리·내부 dipole 영역까지 모두 측정해야 하며, 다음 도전들을 동시에 해결해야 한다: (1) 방사선 벨트 빈번 통과로 인한 방사선 내성 요구, (2) 자전 안정 위성에서의 timing 정밀도 요구, (3) 자세 결정을 위한 perigee 자기장(약 25,000 nT) 측정으로 인한 광역 측정 범위 요구, (4) 통합 전자상자(common E-box) 설계로 인한 EMC 제약. 두 가지가 THEMIS 특이적이다: 단일 센서를 2 m 짧은 boom에 장착(Cluster는 5 m boom에 두 센서) — 이는 차이 분석으로 위성 교란을 제거할 수 없게 만들어 1 nT DC, 10 pT AC 미만의 자기 청결도(magnetic cleanliness) 프로그램을 강제했다(Ludlam et al. 2008).

THEMIS comprises five spacecraft on elliptical equatorial orbits with perigee at 1 RE and apogees from 10 RE (inner) to 30 RE (outer). The magnetometer must work from the solar wind through the dipole-dominated inner magnetosphere. Four challenges had to be balanced simultaneously: radiation tolerance from frequent radiation-belt traversals, precise timing on a spinning spacecraft, wide measurement range driven by the ~25,000 nT perigee field needed for attitude reconstruction, and EMC constraints imposed by the common-electronics-box concept that puts FGM electronics on a shared board with the Power Control Unit. Two features are THEMIS-specific: a single sensor on a short 2 m boom (Cluster used dual sensors on 5 m booms), which removes the option to subtract spacecraft disturbances by difference analysis, and forced a magnetic cleanliness program limiting spacecraft fields below 1 nT DC and 10 pT AC at the sensor.

The science driver is the 30-year-old debate over substorm onset location: does the expansion phase begin at a near-Earth current disruption (CD) region or at a distant X-line (NENL)? Resolution requires timing across the constellation of perturbations propagating at ~1000 km/s over scales of ~100 km, giving a temporal scale of 0.1 s — hence the requirement for ≥10 Hz vector cadence. Magnetic field changes during substorms can be as small as 1 nT, requiring 0.1 nT field resolution. To track a perturbation propagating from 15 RE to 5 RE in minutes, offset stability must hold the 0.1 nT level over that interval, giving 0.2 nT/hour offset drift. Combined with the 25,000 nT attitude requirement, the dynamic range from 0.1 nT to 25,000 nT spans 5.4 decades.

과학 목표는 30년 묵은 substorm onset 위치 논쟁이다: expansion phase가 near-Earth current disruption(CD) 영역에서 시작하는가, 아니면 distant X-line(NENL)에서 시작하는가? 해결을 위해서는 ~1000 km/s 속도로 ~100 km 규모를 가로지르는 교란을 다중 위성에서 timing 해야 하며, 이는 0.1 s 시간 분해능 요구로 이어져 ≥10 Hz 벡터율을 강제한다. Substorm 동안의 자기장 변화는 1 nT만큼 작을 수 있어 0.1 nT 분해능이 필요하다. 15 RE에서 5 RE까지 분 단위로 이동하는 교란을 추적하기 위해 그 시간 동안 오프셋이 0.1 nT 수준을 유지해야 하므로, 시간당 0.2 nT의 오프셋 drift 한계가 도출된다. 자세 결정용 25,000 nT 요구와 결합하면 동적 범위는 0.1 nT–25,000 nT, 즉 5.4 decade에 이른다.

### Part II: Instrument Description (Sect. 3) / 기기 기술

**Hardware overview (p. 238–239).** FGM은 한 개 회로기판 위에 벡터 보상된 3축 fluxgate 센서 유닛과 주로 디지털인 전자부를 통합한다. 자료 인터페이스는 두 채널: TMH(High Telemetry, 영구 128 Hz)와 TML(Low Telemetry, 4–128 Hz commandable). FGM 출력 벡터는 DCB가 제공하는 1 Hz 클록에 동기화되고, 모든 보조 전압(±8 V analog, ±5 V analog, +5 V digital, +2.5 V digital)은 PCU 경유 LVPS에서 공급된다. Table 1–2의 핵심 숫자: 센서 75 g + 하니스 150 g (60 g/m × 2.5 m boom approx) + 전자부 150 g; 센서 70 mm × 45 mm; 전력 800 mW; 24-bit 분해능에서 3 pT/LSB; 10 pT/√Hz @ 1 Hz; 시간당 1 nT 미만 오프셋 drift; 22 ppm/°C copper gain stability; <1 arcmin axes alignment knowledge.

The FGM integrates a vector-compensated three-axis fluxgate sensor and predominantly digital electronics on a single PC board. Data interface uses two channels: TMH (High Telemetry, 128 Hz permanent) and TML (Low Telemetry, 4–128 Hz commandable). Output vectors are synchronized to a 1 Hz DCB clock; secondary voltages (±8 V analog, ±5 V analog, +5 V digital, +2.5 V digital) are supplied by the LVPS via the PCU. Key numbers from Tables 1–2: sensor 75 g, harness 150 g, electronics 150 g; sensor envelope 70 mm × 45 mm; power 800 mW; 24-bit raw output yielding 3 pT/LSB; noise 10 pT/√Hz at 1 Hz; offset drift <1 nT/year; gain stability 22 ppm/°C (copper); axes alignment knowledge <1 arcmin.

**Fluxgate sensor (Sect. 3.1, p. 240–241).** 코어는 13Fe-81Ni-6Mo permalloy(20 μm 두께, 폭 2 mm 리본)를 7회 감아 제작한 두 개의 직경 13 mm/18 mm 토로이드. 850°C에서 풀림 처리 후 그 grain size(작을수록 좋음, ribbon 두께보다 훨씬 작아야)를 광학 측정. 사전 선별 ring core noise는 1 Hz에서 5 pT/√Hz 미만(Fig. 2). 두 코어는 vector-compensated set-up으로 entwined되며, 작은 코어는 X·Z 측정에, 큰 코어는 Y·Z 측정에 사용된다. 각 코어에는 두 개의 3D 코일 시스템이 있다: 안쪽의 pick-up 코일(여기 신호의 2차 고조파를 외부 자기장 비례 신호로 검출)과 바깥쪽의 Helmholtz feedback 코일(코어 위치에서 외부장을 능동 상쇄). pick-up 코일은 신호대잡음비 향상을 위해 가능한 한 코어 가까이, Helmholtz 코일은 대조적으로 더 크다. 모든 코일은 본드 코팅 동선이며, 세라믹 링·고정 부재 등 상이한 열팽창 계수의 조합을 피해 mass <40 g 달성.

The cores are two entwined toroids (13 mm and 18 mm diameter), wound from 13Fe-81Ni-6Mo permalloy ribbon (20 μm thick, 2 mm wide, 7 turns) on Inconel bobbins, annealed at 850°C to control grain size (must be much smaller than ribbon thickness). Pre-selected ring-core noise at 1 Hz is below 5 pT/√Hz (Fig. 2). The smaller core measures X and Z; the larger measures Y and Z. Each core carries two 3-D coil systems: an inner pick-up coil sensing the second harmonic of the excitation that is proportional to external B, and an outer Helmholtz feedback coil compensating the field at the core position to maintain zero-field operation. The pick-up coil is placed close for SNR; the Helmholtz coil is larger and homogenizes the compensation. Vector compensation provides additional axis-orientation stability beyond the single-axis feedback's scale stabilization. Sensor mass is held below 40 g by avoiding combinations of materials with different thermal expansion coefficients.

**Sensor electronics (Sect. 3.2, p. 241–243).** Excitation은 8192 Hz AC 전류로 두 코어를 깊은 양·음 포화로 구동(F0 = 8192 Hz). Pick-up 코일 신호는 4·F0 = 32,768 Hz로 디지털화 — 여기 주파수의 4배. 4 연속 샘플의 누적은 모든 odd harmonic(여기 → pick-up 유도 결합 성분)을 제거하며, FPGA에서 디지털 처리 후 feedback 설정이 갱신되어 Helmholtz 코일이 외부장을 거의 완전히 보상한다. 14-bit ADC(Maxwell 7872)는 ±5 V 입력에 대해 0.6 mV LSB와 0.173 mV_rms 양자화 잡음. 64 Hz 신호 대역폭에서 양자화 오차는 21.6 pT_rms 또는 3 pT/√Hz (백색 잡음 가정)로, 1 Hz에서 10 pT/√Hz 설계 목표 이내이지만 무시할 수는 없는 수준이다. **Cascaded DAC**: 12-bit DAC 1개로 ±25,000 nT 전 범위를 다루면 LSB가 12.2 nT가 되어 LSB의 절반(6 nT)인 비선형성이 허용 불가. 따라서 두 12-bit DAC을 cascade: coarse는 50,000 nT range(상위 6비트만 사용), fine은 780 nT range. Fine DAC만 사용하는 저 자기장(<400 nT) 영역에서는 비선형성을 0.23 LSB 미만으로 제한, 사전 선별 후 43 pT 수준. 400 nT 이상에서는 coarse DAC도 활성화. 디지털 플럭스게이트 진화 단계는 Table 3: Rosetta/Lander(RH 1280, DPU에서 피드백 계산), VenusExpress(RT54SX32, FPGA에서 피드백 계산이지만 자기장 계산은 DPU), THEMIS(RT54SX72, 모든 처리가 단일 FPGA).

The excitation is an 8192 Hz AC current (F0 = 8192 Hz) driving both cores deep into both saturations. Pick-up signals are digitized at 4·F0 = 32,768 Hz. Accumulating four consecutive samples cancels all odd harmonics of the excitation that couple inductively into the pick-up coil. After digital processing in the FPGA, feedback DAC settings are updated so the Helmholtz coil compensates the external field almost completely. The 14-bit ADC (Maxwell 7872) at ±5 V input range yields 0.6 mV LSB and 0.173 mV_rms quantization noise. With sensor sensitivity 0.005 mV/nT and pre-amplification of 40 dB (limited by odd harmonics), the digitization error in the 64 Hz signal bandwidth is ~21.6 pT_rms (or ~3 pT/√Hz under a white-noise assumption) — within but not negligible compared to the 10 pT/√Hz design goal at 1 Hz. **Cascaded DAC**: a single 12-bit DAC over ±25,000 nT would have a 12.2 nT LSB and an unacceptable 6 nT LSB-half non-linearity. Two cascaded 12-bit DACs solve this: a coarse DAC with 50,000 nT range (upper 6 bits only) and a fine DAC with 780 nT range. For fields <400 nT only the fine DAC is active, with non-linearity below 0.23 LSB and ~43 pT after pre-selection. Above 400 nT the coarse DAC's linearity error must be considered. Table 3 traces the digital fluxgate evolution: Rosetta/Lander (RH 1280, feedback in DPU), VenusExpress (RT54SX32, feedback in FPGA but field calculation in DPU), THEMIS (RT54SX72, all processing in a single FPGA).

**FPGA architecture (Fig. 4–5, p. 244).** Actel RT54SX72의 기능은 세 부분: (1) sensor interface — 여기 신호 시작, ADC 샘플링을 여기 클록 대비 프로그래머블 위상 차로 시작, 세 채널을 동기 샘플링, 프로그래머블 N=232(실제) 또는 256(최대) ADC 샘플 평균; (2) 32-bit RISC processor module — $\mathbf{B}_{i_0}^{\mathrm{TMH}} = k_2\,\mathrm{DAC}_{i_0-1} + k_1\,\mathrm{ADC}_{i_0}$ 식으로 자기장 벡터 계산, 새 feedback 설정($\mathrm{DAC}_{i_0} = \mathrm{DAC}_{i_0-1} + k_{fb}\,(k_1/k_2)\,\mathrm{ADC}_{i_0}$), TML decimation/필터링; (3) DCB interface — 1 Hz 클록 동기화, 24-bit 자기장 벡터 + 상태 워드 직렬 전송. 보정 모드 3종(Cal-1: 개루프 sensitivity check; Cal-2: DAC 자동 증분 linearity check; Cal-3: ADC와 DAC 값을 따로 텔레메트리하여 feedback control 분석).

The Actel RT54SX72 functionality has three parts: (1) sensor interface — initiates excitation, starts ADC sampling at programmable phase shift versus excitation, samples three channels synchronously, averages programmable N=232 (real) or 256 (maximum) ADC samples; (2) 32-bit RISC processor module — computes the field vector via $\mathbf{B}_{i_0}^{\mathrm{TMH}} = k_2\,\mathrm{DAC}_{i_0-1} + k_1\,\mathrm{ADC}_{i_0}$, updates feedback settings, and decimates/filters TML data; (3) DCB interface — synchronizes to the 1 Hz clock and transmits the 24-bit field vector plus status word. Three calibration modes: Cal-1 (open-loop sensitivity check), Cal-2 (auto-incrementing DACs for linearity), Cal-3 (separate ADC and DAC telemetry to analyze feedback control).

**Onboard data processing in IDPU (Sect. 3.3, p. 245).** IDPU의 Flight Software(FSW)는 24-bit long vector를 16-bit으로 ranging(가장 넓은 범위와 가장 낮은 분해능부터 가장 좁고 가장 높은 분해능까지 8단계 ranging). 패킷에는 2001년 1월 1일 기준 32-bit 초·16-bit 부초 timestamp가 포함된다. TMH(128 Hz 영구) + TML(가변 4–128 Hz) 두 텔레메트리 stream. FSW는 또한 attitude control packet에 8 Hz 자기장 데이터를 제공(가장 광역·가장 둔감한 range), 두 thermistor 온도(FGE 보드, 센서)를 housekeeping에 보고. **Spin fit**: TMH stream에서 한 자전 주기(3 s) 동안 32 등각 빈으로 나눈 128 Hz 데이터에 $A + B\cos\theta + C\sin\theta$ 모형을 최소제곱 fitting; 잔차 큰 점은 거부 후 재피팅 반복. 위상 shift는 spin pulse 시간이 1 Hz tick 기준으로 알려져 보정 가능.

The IDPU Flight Software (FSW) performs ranging on the 24-bit long vector to select 16 bits (8 ranges from widest/least-sensitive to narrowest/most-sensitive). Packets include a 32-bit seconds-since-2001-01-01 plus 16-bit subseconds timestamp. Two telemetry streams: TMH (128 Hz permanent) and TML (variable 4–128 Hz). FSW also feeds 8 Hz field samples to attitude control (always in the widest range) and reports two thermistor temperatures (FGE board, sensor) to housekeeping. **Spin fit**: TMH stream is fit to $A + B\cos\theta + C\sin\theta$ with 32 equiangular bins over a 3 s spin period via least squares with iterative outlier rejection. Phase shift correction is possible because spin pulse time relative to the 1 Hz tick is known.

### Part III: Instrument Calibration (Sect. 4) / 기기 보정

**Transfer function (Sect. 4.1, p. 246).** 보정된 자기장 벡터는 $\mathbf{B}_{\mathrm{fgs}} = \mathbf{M}_{\mathrm{ort}}\,(\mathbf{M}_{\mathrm{gain}}\,\mathbf{B}_{\mathrm{out}} - \mathbf{O}_{\mathrm{fgm}})$로 표현된다. Offset $\mathbf{O}_{\mathrm{fgm}}$은 약한 자기장에서 센서 회전으로 측정. Scale는 feedback 설계로 잘 정의되지만, feedback 코일 열팽창과 전기 부품 열계수, 그리고 field-/frequency-dependence를 고려해야 한다. **Cross-coupling은 무시 가능** — 디지털 설계 덕분에 misalignment는 순수 sensor property가 됨. 24개 senor position(큐브 90° 회전)에서 비직교성 3각도를 결정한 후, 두 번째 단계에서 직교 좌표계(reference fixture, Fig. 7, 10 arcsec 정밀도)에 mounting하여 6각도 변환을 측정. 모든 6각도는 비직교성 3각도(첫 단계와 일관성 검증)와 reference 회전 3각도를 포함.

The calibrated field vector is $\mathbf{B}_{\mathrm{fgs}} = \mathbf{M}_{\mathrm{ort}}\,(\mathbf{M}_{\mathrm{gain}}\,\mathbf{B}_{\mathrm{out}} - \mathbf{O}_{\mathrm{fgm}})$ with diagonal gain matrix, offset vector, and orthogonalization matrix. Offset is measured by sensor rotation in a weak field as often as practical. Scale is well defined by the feedback design but must account for thermal expansion of feedback coils, thermal coefficients of electrical parts, and field/frequency dependence. **Cross-coupling is negligible** thanks to the digital architecture, so misalignment is purely a sensor property. The non-orthogonality is determined from a linear transfer function fit to measurements at 24 sensor positions (90° cube rotations); a second step mounts the sensor in an orthogonal-reference fixture (Fig. 7, 10 arcsec precision) and measures all six angles, including the three non-orthogonality angles (verifies step one) plus the three reference rotation angles.

**Frequency response (Sect. 4.1, p. 247–248).** Sensor 출력은 여기의 2차 고조파에서 32,768 Hz로 디지털화. N=256 maximum ADC 샘플을 보간 없이 누적하여 128 Hz raw output. 그러나 feedback 갱신과 ADC 샘플링이 sequential해야 하므로 실제로 N=232만 누적. 이로 인한 amplitude/phase response:

$$G(\omega) = \frac{\sin(0.5 N\omega T)}{N\sin(0.5\omega T)}, \qquad \varphi(\omega) = -0.5 N\omega T$$

여기서 $\omega = 2\pi f$, $T = 1/32768$ s. Fig. 9는 N=256 (maximum)과 N=232 (real)의 응답을 보여주며 sequential mode에서 필터 특성이 13.24 Hz 만큼 더 높은 주파수로 shift됨을 보인다. 0.1–180 Hz 범위에서 측정된 frequency response가 검증되었다. TML 저텔레메트리 데이터는 128 Hz raw로부터 비중첩 boxcar averaging으로 도출 — DC 값이 spin modulation으로 영향받으므로 ground processing에서 보정.

The sensor output, the second harmonic of the excitation, is digitized at 32,768 Hz. Maximum N=256 ADC samples are accumulated without overlap to produce 128 Hz raw data, but because sampling and feedback update must run sequentially, only N=232 samples can be accumulated in practice. The amplitude/phase response is the boxcar formula above with $T=1/32768$ s. Figure 9 shows that the sequential-mode (N=232) filter is shifted by 13.24 Hz to higher frequencies versus N=256. Verification used sine-wave fields between 0.1 and 180 Hz generated in calibration coils; amplitude and phase were measured against the field-generating current. TML data are derived by additional non-overlapping boxcar averaging — the DC value is affected by spin modulation by the filter characteristic, which must be corrected during ground processing.

**Temperature dependence (Sect. 4.2, p. 248–251).** TU-Braunschweig에서 보드를 −20°C에서 +60°C까지 변화시키며 전자부 의존성 시험. Sensor는 Themis Sensor Control Unit(TCU, Earth field 10⁴배 차폐 ferromagnetic shield) 안에 위치. Scale value 의존성: 20,000 nT 적용 시 5 ppm/°C 미만. Linearity, noise, phase, inrush current 변화는 측정 불가, 전력 소모 변화는 5% 미만. Offset은 차폐 안에서 sensor 회전으로 측정 — feedback relay open으로 excitation·pick-up 전자부 따로 separate 가능. Fig. 11 (excitation+pick-up: ±20 pT/°C), Fig. 12 (feedback circuitry: 평균 -10 pT/°C, 오차 ±20 pT/°C), Fig. 15 (sensor 의존성: <30 pT/°C, 체계적 패턴 없음). IWF Graz의 액체질소 제어 chamber에서 −100°C ~ +65°C 시험 결과: 0–60°C에서 10 pT/√Hz 명목 잡음, −50°C에서 15–20 pT/√Hz, −100°C에서 30 pT/√Hz (사양 내). 지구 궤도 예상 sensor 온도는 약 0°C. Sensitivity는 저온에서 copper 저항 감소로 증가; phase는 ringcore inductivity의 온도 의존성으로 변화(Fig. 14: −100°C, −20°C, 60°C에서 감도 ~2.0–3.6 pT/LSB, 위상 110°–155°). Feedback 안정성을 위해 thermal expansion 계수만 고려하는 current source 사용, copper-ceramic 같은 동질 팽창 계수만 사용하여 20 ppm/°C로 제한.

Temperature dependence is the most demanding calibration. At TU-Braunschweig, electronics boards are cycled −20°C to +60°C while the sensor sits in the TCU (a ferromagnetic shield suppressing Earth field by 10⁴) equipped with coils for test fields and rotation. Scale dependency: <5 ppm/°C with 20,000 nT applied. No measurable changes for linearity, noise, phase, or inrush current; power consumption varies <5%. Offset is measured by sensor rotation inside the shield, with feedback relays open so excitation/pick-up electronics can be separated from feedback. Fig. 11 (excitation+pick-up: ±20 pT/°C), Fig. 12 (feedback circuitry: averaged −10 pT/°C with ±20 pT/°C scatter), Fig. 15 (sensor dependency: <30 pT/°C, no systematic pattern). At IWF Graz with a liquid-nitrogen chamber over −100°C to +65°C, the typical 10 pT/√Hz noise at 0–60°C rises to 15–20 pT/√Hz near −50°C and 30 pT/√Hz at −100°C — still within spec. Expected orbital sensor temperature is ~0°C. Sensitivity increases at low temperature due to lower copper resistance; phase shifts because ringcore inductivity is temperature dependent (Fig. 14). To stabilize feedback, current sources drive the feedback so only thermal-expansion coefficients matter; only matched-expansion materials (copper, aluminum, ceramic) are used, holding the temperature coefficient constant at ~20 ppm/°C across the full range.

**Functional checks under well-defined fields (Sect. 4.3, p. 251).** Calibration 두 단계: (1) Magnetsrode coil system에서 thermal control box 안에 sensor 배치, ±20,000 nT 인공장 인가, 4 Hz data rate, 60°C까지 0.3°C/min 가열 후 −70°C까지 액체질소 사전 냉각된 세라믹 블록(3.5 kg)으로 냉각, sensitivity와 직교성 모든 온도에서 검증. (2) Magson GmbH Berlin의 Jeserigerhuetten 시설에서 두 Themis sensor를 pillar에 mount, observatory magnetometer와 지구장 수평 성분 비교(perigee 1000 km로 수직 성분은 측정 불가). 90° sensor alignment 회전 후 시험 반복. 결과적으로 substorm 동안 예상되는 field change를 in-situ 검증.

Two functional checks under defined fields: (1) at the Magnetsrode coil system, a thermal control box surrounds the sensor; ±20,000 nT artificial fields are applied at 4 Hz with the box ramped to 60°C at 0.3°C/min and then cooled to −70°C using ceramic blocks (3.5 kg) pre-cooled in liquid nitrogen, verifying sensitivity and orthogonality at all temperatures; (2) at Magson Berlin's Jeserigerhuetten Test Facility, two Themis sensors mounted on a pillar are compared with a reference observatory magnetometer (only horizontal components are compared because the perigee adjustment for 1000 km altitude prevents the Themis sensor from measuring the full Earth vector); the test is repeated with sensor alignment rotated 90°. Pulsation registrations (Fig. 18) confirm in-situ field changes of the kind expected during substorm onsets.

**Spacecraft integration tests (Sect. 4.4, p. 254).** Berkeley 통합. PCB의 후반부가 처음으로 FGM과 함께 power on 되고 보조 전압이 처음으로 원래 DC/DC 컨버터에서 공급되어, 통합 환경에서 정밀 측정 검증 필수. 세 개 ferromagnetic shield(TCU의 일부)를 Graz·Braunschweig·Berkeley에 설치, Berkeley unit이 모든 통합 단계 전후 시험에 사용. Sensor를 boom에서 분리하여 TCU 내부 배치, extension cable 영향은 무시 가능 검증. Short Functional Test(SFT, 20분 절차)를 통합 동안 약 20회 반복 — overall functionality, offsets, scale values, noise, sensor-electronics balance, telemetry errors 점검. 두 개 오류 발견: (a) sensor 잡음 증가로 교체, (b) 케이블 short detected and removed.

Spacecraft integration at Berkeley brought the second half of the PCB online with the FGM for the first time, with secondary voltages provided by the original DC/DC converter for the first time — making precise measurements during integration mandatory. Three ferromagnetic shields (part of the TCU concept) were installed at Graz, Braunschweig, and Berkeley; the Berkeley unit was used for all pre/post-integration tests. The sensor was demounted from the boom and placed in the TCU, with the influence of the extension cable verified to be negligible. A 20-minute Short Functional Test (SFT) — checking functionality, offsets, scales, noise, sensor-electronics balance, and telemetry errors — was repeated ~20 times during integration. Two errors were found: a sensor was replaced for elevated noise; a cable short was detected and removed.

### Part IV: Coordinate Systems and Calibration Files (Sect. 4.5) / 좌표계와 보정 파일

Themis는 Cluster를 따른 6개 약어 좌표계를 사용(Table 5): FS(non-orthogonal sensor), FGS(orthogonal sensor), UNIT(boom aligned), SPG(spinning probe geometric), SSL(spinning sunsensor L-oriented), DSL(despun sun-oriented L). Magnetometer는 dynamic range ±25,000 nT를 24-bit digital resolution에 매핑, 변환계수 2.98 pT/bit. 송신된 16비트는 IDPU의 ranging으로 선택; range 8은 lower 16 bits, range 0은 upper 16 bits. Range 의존 변환계수: $k_r = 50000/2^{16+\mathrm{range}}$. FS에서 sensor offset $\mathbf{O}_{\mathrm{fgm}}$ 제거 후 $\mathbf{M}_{\mathrm{ort}}$로 직교화하여 FGS:

$$\mathbf{B}_{\mathrm{fgs}} = \mathbf{M}_{\mathrm{ort}}(k_r \times \mathbf{B}_{\mathrm{fs}} - \mathbf{O}_{\mathrm{fgm}})$$

Sensor 좌표계는 mechanical interface(sensor와 boom)와 probe의 관성 모멘트가 정의. Sensor alignment vs boom interface ($\mathbf{M}_{\mathrm{unit}}$)는 sensor 보정 프로그램의 일부. Boom alignment vs spacecraft ($\mathbf{M}_{\mathrm{probe}}$)는 boom 검증 절차에서 측정. Probe 좌표계로 회전: $\mathbf{B}_{\mathrm{spg}} = \mathbf{M}_{\mathrm{probe}}\,\mathbf{M}_{\mathrm{unit}}\,\mathbf{B}_{\mathrm{fgs}}$. Probe 좌표계에서 spacecraft offset $\mathbf{O}_{\mathrm{sc}}$ 추가, soft-magnetic material의 sensitivity 영향은 $\mathbf{M}_{\mathrm{scale}}$로 보상. Spin axis와 spin axis(z) + sun direction(x) 정렬을 위해 $\mathbf{M}_{\mathrm{spin}}$, $\mathbf{M}_{\mathrm{phase}}$로 회전. Spin-dependent boxcar filter delay를 위한 $\mathbf{M}_{\mathrm{filter}}$:

$$\alpha_{\mathrm{delay}} = -\pi\,\frac{f_{\mathrm{spin}}}{f_{\mathrm{sample}}}, \qquad d_{\mathrm{filter}} = \frac{f_{\mathrm{sample}}}{128} \cdot \frac{\sin(\frac{\pi}{128} f_{\mathrm{spin}})}{\sin(\pi\,\frac{f_{\mathrm{spin}}}{f_{\mathrm{sample}}})}$$

최종 SSL 좌표계 자기장:

$$\mathbf{B}_{\mathrm{ssl}} = \mathbf{M}_{\mathrm{filter}}\,\mathbf{M}_{\mathrm{phase}}\,\mathbf{M}_{\mathrm{spin}}\,\mathbf{M}_{\mathrm{scale}}\,(\mathbf{B}_{\mathrm{spg}} - \mathbf{O}_{\mathrm{sc}})$$

CalFile은 결합된 $\mathbf{M}_{\mathrm{cal}} = \mathbf{M}_{\mathrm{phase}}\,\mathbf{M}_{\mathrm{spin}}\,\mathbf{M}_{\mathrm{scale}}\,\mathbf{M}_{\mathrm{probe}}\,\mathbf{M}_{\mathrm{unit}}\,\mathbf{M}_{\mathrm{ort}}$와 결합 offset $\mathbf{O}_{\mathrm{cal}} = \mathbf{M}_{\mathrm{phase}}\,\mathbf{M}_{\mathrm{spin}}\,\mathbf{M}_{\mathrm{scale}}\,(\mathbf{M}_{\mathrm{probe}}\,\mathbf{M}_{\mathrm{unit}}\,\mathbf{M}_{\mathrm{ort}}\,\mathbf{O}_{\mathrm{fgm}} + \mathbf{O}_{\mathrm{sc}})$로 단일 변환:

$$\mathbf{B}_{\mathrm{ssl}} = \mathbf{M}_{\mathrm{filter}}(\mathbf{M}_{\mathrm{cal}}\,k_r \times \mathbf{B}_{\mathrm{fs}} - \mathbf{O}_{\mathrm{cal}})$$

THEMIS uses six coordinate frame abbreviations following the Cluster convention (Table 5). The instrument's ±25,000 nT range maps to 24-bit digital resolution with a conversion factor of 2.98 pT/bit. The 16-bit telemetered value is selected by IDPU ranging: range 8 = lower 16 bits, range 0 = upper 16 bits, with range-dependent factor $k_r = 50000/2^{16+\mathrm{range}}$. The full chain in the SSL frame combines six matrices and two offset vectors as written above. The CalFile bundles the full transformation as a single matrix $\mathbf{M}_{\mathrm{cal}}$ and offset $\mathbf{O}_{\mathrm{cal}}$. $\mathbf{M}_{\mathrm{filter}}$, $\mathbf{M}_{\mathrm{probe}}$, $\mathbf{M}_{\mathrm{unit}}$, $\mathbf{O}_{\mathrm{fgm}}$ are constant; $\mathbf{M}_{\mathrm{phase}}$, $\mathbf{M}_{\mathrm{spin}}$, $\mathbf{M}_{\mathrm{scale}}$, $\mathbf{M}_{\mathrm{ort}}$, $\mathbf{O}_{\mathrm{sc}}$ are time-dependent and updated by in-flight calibration.

### Part V: First Flight Results (Sect. 5) / 첫 비행 결과

**Inflight calibration (Sect. 5.1, p. 257–259).** Commissioning 시 모든 기본 기능 시험. Sensor-electronics balance와 sensitivity는 사전 시험 대비 변화 없음, telemetry quality와 onboard data processing 오류 없음. Boom 성공 전개 후 모든 magnetometer noise level 점검; 15 sensor 컴포넌트의 1 Hz 평균 잡음 약 12 pT/√Hz로 30 pT/√Hz 요구의 절반 미만. CalFile은 12개 transformation 요소 + spin period + 유효시간 포함. Spin frequency와 1차 고조파가 field magnitude에 없어야 한다는 사실로 4개 방정식; spin axis로 추가 2 방정식. 12 요소 중 8개 (spin plane offset 2개, spin plane scale 비, 비직교성 3 각도, spin axis 대비 방위 2 각도)는 spin tone 최소화로 결정. 나머지 4개 (spin axis offset, scale value, spin plane scale value, spin phase)는 다른 기준 필요: non-compressible waves(spin axis offset), IGRF(spin phase), curlometer 조건($\nabla\cdot\mathbf{B}=0$) 또는 균일성 조건($\mathbf{B}_1=\mathbf{B}_n$). 6개월간 모든 angle과 scale value가 10⁻⁴ 정확도로 일정, offset 안정성 12시간 0.2 nT/12h. Fig. 21: Probe A spin plane component offset의 첫 반년 — 두 offset 모두 6개월 0.2 nT 미만 변화. Fig. 22: 5개 probe의 spin plane offset 표준편차 — 최대 0.3 nT/6 month, 사양 0.2 nT/12 hours 만족. CalFile은 매일 갱신(inner spacecraft 궤도주기); 고분해능 CalFile은 요청 시 제공.

During commissioning all basic functions were tested by procedures similar to ground SFTs, with modifications for spacecraft rotation. Sensor-electronics balance and sensitivity were unchanged from preflight; telemetry quality and onboard processing were error-free. After successful boom deployment, total noise was checked at apogee for all 15 sensor components and averaged ~12 pT/√Hz at 1 Hz, less than half the 30 pT/√Hz required level. The CalFile contains 12 transformation elements plus spin period and validity. To determine the transfer function in flight, four equations come from requiring the spin frequency and its first harmonic to be absent from the field magnitude; two more come from defining one axis by the spin axis. Eight of the twelve elements (two spin-plane offsets, ratio of spin-plane scale values, three non-orthogonality angles, two angles of orientation versus spin axis) are determined by minimizing spin tones over n different field conditions. The remaining four — spin-axis offset, scale value, spin-plane scale value, spin phase — require special conditions: non-compressible waves (spin-axis offset), IGRF (spin phase), curlometer conditions ($\nabla\cdot\mathbf{B}=0$) or homogeneity ($\mathbf{B}_1=\mathbf{B}_n$). All angles and scale values stayed constant to 10⁻⁴ over 6 months; offset stability is 0.2 nT/12 hours. Fig. 21 shows Probe A spin-plane offsets varying less than 0.2 nT over half a year. Fig. 22 shows 5-probe spin-plane offset standard deviations: maximum 0.3 nT per 6 months, well within spec. CalFiles are updated daily (the inner spacecraft orbital period); higher-resolution CalFiles are available on request.

**Spacecraft interferences (Sect. 5.2, p. 259–260).** 두 종류의 0.3 nT peak-to-peak 간섭 검출. (1) Solar cell driven power management으로 spin과 동기화. spin axis component 저자기장 조건에서 spin 주파수와 그 고조파 내용을 모형 입력으로 사용; 도출된 field wavelet을 spin tone harmonics 진폭으로 scale하여 raw data에서 차감. Fig. 23: 보정 전 35 pT spin tone과 15 pT double spin tone이 4배 억제됨. (2) Particle instrument의 sectoring으로 야기, mode-dependent magnetic moment가 아닌 power profile에 의한 conducted interference. 11 Hz switch 주파수, sun pulse로 동기화되어 jitter로 주파수 dilation. FSW timing 변경으로 회피 가능하지만 진폭이 작고 영향 주파수 대역(n×11 Hz ± 2 Hz)은 SCM(Roux et al. 2008)에서 다루므로 후속 단계에서만 수정 예정.

Two types of <0.3 nT peak-to-peak interference were detected. (1) Solar-cell-driven power management synchronized to spin: a model derived from the spin-axis component at low field is scaled by spin-tone harmonic amplitudes and subtracted from the raw data. Figure 23 shows the 35 pT spin tone and 15 pT double-spin tone suppressed by a factor of four after correction; the residual periodic content reflects non-constant interference phase versus the sun pulse. (2) Conducted interference from particle-instrument sectoring at 11 Hz (switching at the 32nd of a spin period), synchronized to the sun pulse with finite-resolution jitter that smears the frequency. This will be fixed by FSW timing changes in a later mission phase since amplitude is small (0.1 nT) and the affected band (n×11 Hz ± 2 Hz) is covered by the search-coil magnetometer (Roux et al. 2008).

**Magnetopause oscillations (Sect. 5.3, p. 261–262).** 2007년 8월 7일 09:00–11:30 UT magnetopause 통과 사례. 이때 5기 위성은 injection phase, 같은 궤도(15.4 RE apogee) 위 "string-of-pearls" 구성. Probe A(string 끝)는 magnetopause 도달하지 못함. Probe B 가 09:25 UT 부근 첫 통과; Probe C, D, E가 5분 후 (서로 더 가까이 그룹화) 따라간다. 이후 90분 간 4개 leading probe 다중 통과 경험. 같은 궤도를 따라 이동하므로 공통 reference point부터 거리 vs 시간 도표 작성 가능. Probe B leading (~1 RE 앞), Probes C/D/E 그룹, Probe A closing (~1.5 RE 뒤). 각 통과의 slope에서 magnetopause 속도 도출: 평균 72 km/s(inward), −95 km/s(outward), 67 km/s 조화 진동 최대속도와 비교 가능. Total 81 single-spacecraft events이 17 crossings로 그룹화. Position-time 도표 곡선이 magnetopause 운동을 지시 — 약 2 RE 진폭, ~10분 주기 진동.

The magnetopause crossing of August 7, 2007 (09:00–11:30 UT, near the sub-solar point) is presented as a multi-spacecraft demonstration. The five probes were still in injection phase, sharing one orbit with 15.4 RE apogee in a "string of pearls". Probe A (last in the string) never reached the magnetopause. Probe B crossed first at ~09:25 UT, with Probes C, D, E (more tightly grouped) following five minutes later. In the following 90 minutes, all four leading probes experienced multiple magnetopause crossings. Because they moved along the same track, a position-time diagram (distance from a common reference vs time) shows Probe B leading at ~1 RE ahead, with C/D/E grouped, and Probe A closing at ~1.5 RE behind. Slopes give magnetopause speeds: 72 km/s inward and −95 km/s outward, comparable to the 67 km/s maximum speed predicted for harmonic oscillation. In total 81 single-spacecraft events grouped into 17 crossings; the implied magnetopause oscillation has ~2 RE amplitude and ~10 minute period.

### Part VI: Summary (Sect. 6) / 요약

THEMIS FGM은 ring core·sensor 설계, digital fluxgate 기술, 사전 임무에서 개발된 고정밀 시설로의 보정·시험에서 비롯된다. 정확하고 안정적인 자기장 측정을 near-Earth space에 제공하며, 첫 반년 운영에서 0.5 nT 미만 안정성 입증. 5점 측정은 다양한 데이터 분석 방법을 가능케 하며, 본 논문에서 제시된 magnetopause crossing 시간 이력 재구성은 그러한 한 예다.

THEMIS FGM benefits from elaborate prior work on ring cores and sensor design, digital fluxgate technology, and the high-precision facilities developed for previous missions. It provides accurate, stable magnetic field measurements in near-Earth space; stability was proven to be better than 0.5 nT during the first half year. Five-point measurements enable a number of new analysis methods, of which the magnetopause time-history reconstruction is one example.

---

## 3. Key Takeaways / 핵심 시사점

1. **Digital-at-the-front architecture is the paradigm shift / 전단 디지털화가 패러다임 전환** — Sensor pick-up 신호를 사전 증폭 직후 32,768 Hz로 ADC 처리, 모든 후속 처리를 FPGA에서 수행함으로써 analog filter, phase-sensitive integrator, 그에 따르는 EMC 취약성을 제거. 결과: 25,000 nT 동적 범위와 3 pT 분해능을 동시에 달성하면서도 75 g sensor + 800 mW 전력으로 통합 전자상자 환경에서도 동작. / Digitizing the pick-up signal directly behind the preamplifier at 32,768 Hz and performing all subsequent processing inside an FPGA eliminates analog filters, phase-sensitive integrators, and their EMC susceptibilities, yielding 25,000 nT range with 3 pT resolution at 75 g and 800 mW even in a shared-electronics-box environment.

2. **Cascaded DAC solves the dynamic-range/linearity tradeoff / 직렬 DAC이 동적범위·선형성 충돌 해소** — 단일 12-bit DAC으로 ±25,000 nT를 다루면 비선형성이 6 nT — 0.1 nT 사양과 양립 불가. 12-bit coarse(50,000 nT) + 12-bit fine(780 nT) 두 DAC 직렬화로 fine-only 영역에서 비선형성을 0.23 LSB(43 pT)까지 감소. / A single 12-bit DAC over ±25,000 nT would have ~6 nT non-linearity, incompatible with 0.1 nT science. Cascading a 50,000 nT coarse DAC and a 780 nT fine DAC reduces non-linearity to 0.23 LSB (43 pT) in the fine-only regime.

3. **Vector compensation gives both scale and orientation stability / 벡터 보상은 스케일과 방위 모두 안정화** — 단축 feedback은 scale만 안정화하지만, 3축 Helmholtz 코일이 코어 위치에서 외부장을 0으로 만들면 sensor 축의 mechanical 안정성이 추가 확보된다. 이는 22 ppm/°C copper gain stability와 <1 arcmin axes alignment 안정성의 핵심. / Single-axis feedback stabilizes scale only, but compensating the field at the core position with a three-axis Helmholtz system adds mechanical axis-orientation stability — the basis of 22 ppm/°C copper gain stability and <1 arcmin axes alignment stability.

4. **Heritage chain matters: Rosetta → VEX → THEMIS / 헤리티지 사슬이 결정적** — Table 3은 디지털 magnetometer principle 발전을 구체적으로 보인다: Rosetta-Lander(RH 1280, FPGA가 ADC/DAC 제어, DPU에서 feedback과 자기장 계산) → VenusExpress(RT54SX32, FPGA에서 feedback 계산) → THEMIS(RT54SX72, 모든 처리가 단일 FPGA). 각 단계가 mass·power·robustness를 점진 개선. / Table 3 traces specific evolutionary steps: Rosetta-Lander (RH 1280 FPGA, feedback in DPU) → VenusExpress (RT54SX32, feedback in FPGA) → THEMIS (RT54SX72, all processing in a single FPGA), each step improving mass, power, and EMC robustness.

5. **Calibration is a chain, not a step / 보정은 단계가 아닌 사슬** — 6개 행렬($\mathbf{M}_{\mathrm{filter}}\mathbf{M}_{\mathrm{phase}}\mathbf{M}_{\mathrm{spin}}\mathbf{M}_{\mathrm{scale}}\mathbf{M}_{\mathrm{probe}}\mathbf{M}_{\mathrm{unit}}\mathbf{M}_{\mathrm{ort}}$)과 두 offset 벡터($\mathbf{O}_{\mathrm{fgm}}, \mathbf{O}_{\mathrm{sc}}$)가 필요하며, 이를 단일 CalFile로 결합. 시간 의존(orbit-cadence 갱신)과 시간 독립 요소를 분리하는 설계로 일일 갱신이 충분. / Six matrices and two offsets stack into a single time-evolving CalFile. Separating time-dependent components (refreshed each orbit) from time-independent components (fixed by ground calibration) keeps in-flight maintenance manageable at one update per inner-orbit period.

6. **In-flight calibration uses physics constraints, not just statistics / 비행 보정은 통계가 아닌 물리 조건 사용** — 12 transfer-function 요소 중 8개는 spin frequency·1차 고조파를 magnitude에서 제거하는 단일 spacecraft 조건으로 결정; 나머지 4개는 비압축 wave(spin axis offset), IGRF(phase), curlometer 조건($\nabla\cdot\mathbf{B}=0$ 또는 균일성) 등 물리 조건 필요. 다중 위성 비교는 점검에 유용하나 routine input이 될 수 없다. / Eight of the twelve transfer-function elements are determined by single-spacecraft spin-tone minimization, but the remaining four require physical constraints: non-compressible waves (spin-axis offset), IGRF (spin phase), curlometer or homogeneity ($\nabla\cdot\mathbf{B}=0$ or $\mathbf{B}_1 = \mathbf{B}_n$). Multi-spacecraft comparison can verify but cannot serve as routine input.

7. **0.2 nT/12 h offset stability achieved / 0.2 nT/12시간 오프셋 안정성 달성** — Fig. 21–22가 보이듯, 5기 위성 모두 spin plane offset이 6개월에 걸쳐 0.3 nT 표준편차로 유지되며 0.2 nT/12 h 사양을 만족. 이는 substorm timing(15 RE에서 5 RE까지 1000 km/s 전파 = 약 1분)에 직접 활용 가능. / All five probes maintain spin-plane offsets within 0.3 nT std-dev over 6 months, meeting the 0.2 nT/12 h specification — directly enabling substorm propagation timing across 10 RE at 1000 km/s (~1 minute).

8. **Multi-spacecraft string-of-pearls timing reveals magnetopause oscillations / 다중 위성 구슬꿰미 timing이 magnetopause 진동 노출** — 5점 측정으로 magnetopause crossing time 이력을 distance-vs-time 도표로 재구성, 약 2 RE 진폭·10분 주기·67–95 km/s 속도의 진동을 도출. 단일 위성으로는 시공간 모호성 해결 불가. / Five-point measurements reconstruct the magnetopause crossing time history as a distance-vs-time diagram, revealing ~2 RE amplitude, ~10 minute period oscillations at 67–95 km/s — observations no single spacecraft can disambiguate from temporal change.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Calibrated field vector in FGS frame / FGS 좌표계의 보정된 자기장

Starting from the raw magnetometer output $\mathbf{B}_{\mathrm{out}}$ in digital units, the calibrated field in the orthogonal sensor frame is:

$$\boxed{\mathbf{B}_{\mathrm{fgs}} = \mathbf{M}_{\mathrm{ort}}\,(\mathbf{M}_{\mathrm{gain}}\,\mathbf{B}_{\mathrm{out}} - \mathbf{O}_{\mathrm{fgm}})}$$

| Symbol / 기호 | Meaning / 의미 |
|---|---|
| $\mathbf{B}_{\mathrm{out}}$ | Raw 24-bit magnetometer output in non-orthogonal sensor frame (FS) / 비직교 sensor 좌표계의 24비트 raw 출력 |
| $\mathbf{M}_{\mathrm{gain}}$ | Diagonal 3×3 matrix of conversion factors (digital units → nT) per axis / 축별 디지털 단위 → nT 변환 대각행렬 |
| $\mathbf{O}_{\mathrm{fgm}}$ | Sensor offset vector in nT, measured by sensor rotation in weak field / 약한 자기장에서 회전으로 측정한 센서 오프셋 |
| $\mathbf{M}_{\mathrm{ort}}$ | Orthogonalization matrix removing 3 non-orthogonality angles / 비직교성 3각도 제거 행렬 |
| $\mathbf{B}_{\mathrm{fgs}}$ | Calibrated vector in orthogonal sensor frame (FGS) / 직교 sensor 좌표계의 보정된 벡터 |

The diagonal property of $\mathbf{M}_{\mathrm{gain}}$ is crucial — it means each axis's gain factor can be scaled independently in software, which is exploited for in-flight scale calibration: $\mathbf{B}_{\mathrm{fs}} = \mathbf{M}_{\mathrm{gain}}(k)\,\mathbf{B}_{\mathrm{out}}$ where $k$ is updated by the magnetometer software. / $\mathbf{M}_{\mathrm{gain}}$이 대각행렬인 점이 중요하다 — 각 축 gain을 소프트웨어로 독립 조정 가능, 비행 보정에서 활용.

### 4.2 Digital feedback synthesis / 디지털 피드백 합성

Inside the FPGA, the magnetic field vector is reconstructed each cycle as the sum of the previous DAC contribution (large feedback field) and the residual ADC value (small remaining field):

$$\boxed{\mathbf{B}_{i_0}^{\mathrm{TMH}} = k_2\,\mathrm{DAC}_{i_0-1} + k_1\,\mathrm{ADC}_{i_0}}$$

The new DAC setting that nulls the next residual is:

$$\mathrm{DAC}_{i_0} = \mathrm{DAC}_{i_0-1} + k_{fb}\,(k_1/k_2)\,\mathrm{ADC}_{i_0}$$

| Symbol / 기호 | Meaning / 의미 |
|---|---|
| $k_1$ | ADC scaling factor (ADC LSB → field unit, basic resolution 3 pT) / ADC 단위 변환 (3 pT 기본 분해능) |
| $k_2$ | DAC scaling factor (DAC LSB → field unit; equals 25,000 nT / 2¹² ≈ 6.1 nT for the fine DAC) / DAC 단위 변환 (fine DAC의 경우 ~6.1 nT) |
| $k_{fb}$ | Feedback gain (≤1, controls loop bandwidth and stability) / 피드백 게인 |
| $\mathrm{DAC}_{i_0}$ | DAC value at cycle $i_0$, equivalent to applied feedback field / cycle $i_0$의 DAC 값 |
| $\mathrm{ADC}_{i_0}$ | ADC residual at cycle $i_0$ / cycle $i_0$의 ADC 잔차 |
| $\mathbf{B}_{i_0}^{\mathrm{TMH}}$ | High-cadence (128 Hz) field vector / 고주파(128 Hz) 자기장 벡터 |

### 4.3 Boxcar averaging filter / 박스카 평균 필터

The frequency response of the non-overlapping arithmetic mean of $N$ samples at sampling period $T$ is:

$$\boxed{G(\omega) = \frac{\sin(0.5 N\omega T)}{N\sin(0.5\omega T)}, \qquad \varphi(\omega) = -0.5 N\omega T}$$

For the 128 Hz raw output: $T = 1/32768$ s, $N = 232$ (sequential operation) or $N = 256$ (maximum). At DC, $G \to 1$. The first null is near $f = 1/(NT) \approx 141$ Hz for $N=232$, ensuring strong attenuation around the 128 Hz output rate. The filter delay is $\tau = NT/2$. The TML data are derived from 128 Hz raw via a second non-overlapping boxcar:

$$\mathbf{B}_{T_0}^{\mathrm{TML}} = \frac{s}{128}\sum_{n=1}^{128/s} \mathbf{B}_{i_0 - n + 1}^{\mathrm{TMH}}$$

where $s$ is the decimation factor giving telemetry rates 4–128 Hz. / 여기서 $s$는 다운샘플 인자, 4–128 Hz 텔레메트리율 제공.

### 4.4 Quantization noise budget / 양자화 잡음 예산

Given:
- ADC: 14-bit, ±5 V input range → LSB = 10/2¹⁴ ≈ 0.6 mV
- White-noise quantization standard deviation: $\sigma_q = \mathrm{LSB}/\sqrt{12} \approx 0.173$ mV_rms
- Sensor sensitivity: 0.005 mV/nT = 5 μV/nT
- Pre-amplification: 40 dB (factor 100)
- Effective sensitivity at ADC: 0.5 mV/nT
- Sampling/output ratio: 32,768/128 = 256

The digitization error in the 64 Hz signal bandwidth is:

$$\sigma_B = \frac{\sigma_q}{0.5 \text{ mV/nT}} \times \frac{1}{\sqrt{256}} \approx \frac{0.173}{0.5 \times 16} \text{ nT} \approx 21.6 \text{ pT}_{\mathrm{rms}}$$

Equivalent noise density: $\approx 21.6 / \sqrt{64} \approx 2.7$ pT/√Hz, conventionally reported as 3 pT/√Hz. This is just below the 10 pT/√Hz @ 1 Hz design goal, so it cannot be neglected. / 이는 1 Hz에서 10 pT/√Hz 설계 목표 바로 아래로, 무시할 수 없는 수준이다.

### 4.5 Spin fit / 스핀 적합

For a spin period $T_{\mathrm{spin}} \approx 3$ s, sampling at 128 Hz gives 384 samples per spin, binned into 32 equal-angle bins. The fit model:

$$\boxed{B(\theta) = A + B\cos\theta + C\sin\theta}$$

The DC component $A$ is the spin-axis offset; $\sqrt{B^2 + C^2}$ is the spin-plane field magnitude; $\arctan(C/B)$ is the spin-plane phase angle. Iterative outlier rejection ensures a clean fit. With 128 samples per spin a phase uncertainty of $360°/(3 \times 128) \approx 0.9°$ is achieved, just under the 1° spec. / 128 sample/spin으로 위상 불확도 약 0.9° 달성, 1° 사양 만족.

### 4.6 Filter delay correction in DSL / DSL에서의 필터 지연 보정

The boxcar filter on TML data introduces a spin-dependent delay and amplitude reduction in the spin plane. The correction matrix $\mathbf{M}_{\mathrm{filter}}$ rotates by $\alpha_{\mathrm{delay}}$ and scales by $d_{\mathrm{filter}}$:

$$\boxed{\alpha_{\mathrm{delay}} = -\pi\,\frac{f_{\mathrm{spin}}}{f_{\mathrm{sample}}}, \qquad d_{\mathrm{filter}} = \frac{f_{\mathrm{sample}}}{128}\cdot\frac{\sin(\pi f_{\mathrm{spin}}/128)}{\sin(\pi f_{\mathrm{spin}}/f_{\mathrm{sample}})}}$$

For typical spin rate $f_{\mathrm{spin}} = 1/3$ Hz and $f_{\mathrm{sample}} = 128$ Hz, $\alpha_{\mathrm{delay}} \approx -0.0082$ rad ≈ −0.47°, and $d_{\mathrm{filter}} \approx 1$ to leading order. / 일반적 spin rate 1/3 Hz, sampling 128 Hz에서 $\alpha_{\mathrm{delay}} \approx -0.47°$, $d_{\mathrm{filter}} \approx 1$.

### 4.7 Curlometer current density estimate / 컬로미터 전류밀도 추정

Although the paper invokes the curlometer constraint $\nabla\cdot\mathbf{B}=0$ as an in-flight calibration tool, the principle for current density from $n \geq 4$ simultaneous magnetometer measurements is:

$$\mu_0 \mathbf{J} = \nabla \times \mathbf{B} \approx \frac{1}{V}\sum_{\mathrm{faces}} \mathbf{B}_f \times \mathbf{S}_f$$

where the sum is over the faces of the polyhedron formed by the spacecraft and $\mathbf{S}_f$ are outward-normal area vectors. For the typical THEMIS five-spacecraft tetrahedron-plus-one configuration, four spacecraft form a tetrahedron and the fifth provides a redundancy check, giving:

$$\mu_0 J_x \approx \frac{\Delta B_z}{\Delta y} - \frac{\Delta B_y}{\Delta z}, \quad \text{etc.}$$

with finite-difference uncertainty $\sigma_J \sim \sigma_B / (\mu_0 L)$ where $L$ is the inter-spacecraft separation. / 4점 이상 동시 측정으로 ∇×B를 유한차분 추정하여 J 계산.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1960s ─── Geiger (1962): Vanguard/Explorer fluxgates                          [analog]
1970s ─── Helios mission FGM (German, IGEP heritage)                          [analog]
1980s ─── Phobos mission FGM (Russian, German)                                [analog]
1985 ──── ISEE-3 magnetometer in solar wind                                    [analog]
1994 ──── Freja (Zanetti et al.)                                              [analog]
1995 ──── Auster et al. patent digital fluxgate principle (MST 6, 477)        [DIGITAL BEGINS]
1999 ──── Equator-S (Fornacon et al.)                                         [hybrid]
2001 ──── Cluster FGM (Balogh et al.)  4 s/c, dual sensors per s/c            [analog]
2004 ──── Cassini MAG (Dougherty et al.)                                      [analog/digital]
2005 ──── Double Star (Carr et al.)                                           [hybrid]
2006 ──── VenusExpress MAG (Zhang et al.) RT54SX32                            [DIGITAL FPGA + DPU]
2007 ──── Rosetta Lander Philae ROMAP (Auster et al.)                         [DIGITAL RH 1280]
2007 ──── THEMIS launch  Feb 17, 2007 (5 s/c constellation)
2008 ──── *** Auster et al. THEMIS FGM paper ***                              [DIGITAL SINGLE FPGA]
2008 ──── Angelopoulos et al. Science: NENL X-line + dipolarization timing    [SCIENCE PAYOFF]
2010 ──── Chang'e-1 magnetometer (Chinese, digital)
2011 ──── Juno MAG (analog/digital hybrid, fluxgate + scalar)
2014 ──── Rosetta arrival at 67P, RPC-MAG (Glassmeier et al. 2007)            [digital heritage]
2015 ──── MMS Digital Fluxgate Magnetometer  (4 s/c, direct THEMIS heritage)
2018 ──── Solar Orbiter MAG  (digital, Imperial College)
2018 ──── BepiColombo MPO-MAG (digital, IGEP heritage)
2024 ──── JUICE J-MAG (digital, Imperial College + IGEP)
2027 ──── HelioSwarm (NASA, n-point digital fluxgates planned)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Auster et al. (1995, MST 6, 477) | Original digital fluxgate patent / 디지털 플럭스게이트 원리 특허 | Foundation: introduced FPGA-based digitization at the AC level / 기초: AC 레벨 FPGA 디지털화 도입 |
| Balogh et al. (2001, Ann. Geophys. 19, 1207) | Cluster FGM / Cluster FGM | Predecessor multi-spacecraft FGM, mostly analog; established the four-point benchmark THEMIS extends to five / 다중 위성 FGM 전임자, 4점 기준을 5점으로 확장 |
| Glassmeier et al. (2007a, Space Sci. Rev. 128, 649) | Rosetta RPC-MAG paper / Rosetta RPC-MAG 논문 | Direct hardware heritage, similar digital fluxgate but with DPU-based feedback / 직접 하드웨어 헤리티지, DPU 기반 피드백의 디지털 플럭스게이트 |
| Zhang et al. (2006, Planet. Space Sci.) | VenusExpress MAG / VenusExpress MAG | Intermediate digital step — feedback in FPGA but field calculation in DPU; sets stage for THEMIS single-FPGA design / 중간 디지털 단계, THEMIS 단일 FPGA의 전 단계 |
| Auster et al. (2007, Space Sci. Rev. 128, 221) | Rosetta Lander Philae ROMAP / Rosetta Lander Philae ROMAP | First full digital implementation in flight (RH 1280 FPGA) / 비행 최초 완전 디지털 구현 |
| Harvey et al. (2008, this issue) | THEMIS instrument data processing unit / THEMIS IDPU | Describes the IDPU that hosts FGM electronics — defines the EMC/integration constraints / FGM 전자부 호스팅 IDPU 기술, EMC/통합 제약 정의 |
| Angelopoulos (2008, Space Sci. Rev., this issue) | THEMIS mission overview / THEMIS 임무 개요 | Defines science requirements and constellation that drive FGM specs / 과학 요구와 constellation 정의, FGM 사양 도출 근거 |
| Roux et al. (2008) | THEMIS Search Coil Magnetometer (SCM) / THEMIS SCM | Complementary AC sensor (8 Hz–4 kHz); covers band of conducted interference (n×11 Hz) / 보완 AC 센서, conducted interference 대역 담당 |
| Ludlam et al. (2008) | THEMIS magnetic cleanliness program / THEMIS 자기 청결도 프로그램 | Describes spacecraft-level disturbance suppression (<1 nT DC, <10 pT AC) at sensor / 센서 위치에서의 위성 교란 억제 기술 |
| Mueller et al. (1998, J. Magn. Magn. Math. 177, 231) | Permalloy ring core development / Permalloy 링코어 개발 | Defines the 13Fe-81Ni-6Mo material at the heart of every modern fluxgate / 모든 현대 fluxgate의 핵심 재료 정의 |
| Russell et al. (2014/2016, MMS) | MMS Digital Fluxgate Magnetometer / MMS DFG | Direct successor: 4 s/c version of THEMIS architecture, with tighter EMC and 0.1 nT accuracy / 직접 후속, 4 위성판 THEMIS, 더 엄격한 EMC와 0.1 nT 정확도 |

---

## 7. References / 참고문헌

### Primary / 주 논문
- H.U. Auster, K.H. Glassmeier, W. Magnes, et al., "The THEMIS Fluxgate Magnetometer", Space Sci. Rev. **141**, 235–264 (2008). DOI: [10.1007/s11214-008-9365-9](https://doi.org/10.1007/s11214-008-9365-9)

### Cited / 인용
- Angelopoulos, V., "The THEMIS Mission", Space Sci. Rev. (2008, this issue). DOI: 10.1007/s11214-008-9336-1
- Auster, H.U., Lichopoj, A., Rustenbach, J., et al., "Concept and First Results of a Digital Fluxgate Magnetometer", Meas. Sci. Technol. **6**, 477–481 (1995).
- Auster, H.U., Glassmeier, K.H., et al., "RPC-MAG The Fluxgate Magnetometer in the Rosetta Plasma Consortium", Space Sci. Rev. **128**, 221–240 (2007).
- Auster, H.U., Fornacon, K.H., Georgescu, E., et al., "Calibration of flux-gate magnetometers using relative motion", Meas. Sci. Technol. **13**, 1124–1131 (2002).
- Balogh, A., Carr, C.M., Acuna, M.H., et al., "The Cluster Magnetic Field Investigation: overview of in-flight performance and initial results", Ann. Geophys. **19**, 1207–1217 (2001).
- Baumjohann, W., Haerendel, G., Treumann, R.A., et al., "Equator-S magnetic field experiment", Adv. Space Res. **24**, 77–80 (1999).
- Carr, C., Brown, P., Zhang, T.L., et al., "The Double Star magnetic field investigation: instrument design, performance and highlights of the first year's observations", Ann. Geophys. **23**, 2713–2732 (2005).
- Dougherty, M.K., Kellock, S., Southwood, D.J., et al., "The Cassini magnetic field investigation", Space Sci. Rev. **114**, 331–383 (2004).
- Fornacon, K.H., Auster, H.U., Georgescu, E., et al., "The magnetic field experiment onboard Equator-S and its scientific possibilities", Ann. Geophys. **17**, 1521–1527 (1999).
- Glassmeier, K.H., Boehnhardt, H., Koschny, D., Kührt, E., Richter, I., "The Rosetta Mission: Flying Towards the Origin of the Solar System", Space Sci. Rev. **128**, 1–21 (2007b).
- Glassmeier, K.H., Richter, I., Diedrich, A., et al., "RPC-MAG The Fluxgate Magnetometer in the Rosetta Plasma Consortium", Space Sci. Rev. **128**, 649–670 (2007a).
- Glassmeier, K.H., Motschmann, U., Dunlop, M., et al., "Cluster as a wave telescope — first results from the fluxgate magnetometer", Ann. Geophys. **19**, 1439–1448 (2001).
- Harvey, P., Taylor, E., Sterling, R., Cully, M., "The THEMIS constellation: instrument data processing unit", Space Sci. Rev. (2008, this issue).
- Ludlam, M., Angelopoulos, V., Taylor, E., et al., "The THEMIS magnetic cleanliness program", Space Sci. Rev. (2008, this issue).
- Müller, M., Lederer, T., Fornacon, K.H., Schäfer, R., "Grain structure, coercive force and induced magnetic anisotropy of permalloy (NiFeMo) annealed in the temperature range 850–1100°C", J. Magn. Magn. Math. **177**, 231–232 (1998).
- Roux, A., LeContel, O., Robert, P., et al., "The Search Coil Magnetometer for THEMIS", Space Sci. Rev. (2008, this issue).
- Zanetti, T., Potemra, R., Erlandson, R., et al., Space Sci. Rev. **70**, 465–482 (1994). DOI: 10.1007/BF00756882
- Zhang, T.L., Baumjohann, W., Delva, M., et al., "Magnetic field investigation of the Venus plasma environment: Expected new results from VenusExpress", Planet. Space Sci. **54**, 1336–1343 (2006).
