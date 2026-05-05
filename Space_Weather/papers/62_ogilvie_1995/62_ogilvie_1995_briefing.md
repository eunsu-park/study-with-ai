---
title: "Pre-Reading Briefing: SWE, A Comprehensive Plasma Instrument for the Wind Spacecraft"
paper_id: "62_ogilvie_1995"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# SWE, A Comprehensive Plasma Instrument for the Wind Spacecraft: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Ogilvie, K. W., Chornay, D. J., Fritzenreiter, R. J., Hunsaker, F., Keller, J., Lobell, J., Miller, G., Scudder, J. D., Sittler Jr., E. C., Torbert, R. B., Bodet, D., Needell, G., Lazarus, A. J., Steinberg, J. T., Tappan, J. H., Mavretic, A., and Gergin, E., "SWE, A Comprehensive Plasma Instrument for the Wind Spacecraft", Space Science Reviews 71, 55-77, 1995. DOI: 10.1007/BF00751326
**Author(s)**: K. W. Ogilvie et al. (GSFC + MIT Center for Space Research + Univ. New Hampshire + Boston Univ.)
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

This paper is the **instrument paper** for the Solar Wind Experiment (SWE) on the WIND spacecraft, the dedicated solar-wind plasma analyzer of the ISTP program. SWE is unusual in that it integrates **three complementary sensor families** under a single Data Processing Unit (DPU): (1) two MIT-style **Faraday cups (FC)** at opposite ends of a spacecraft diameter — one tilted +15° above the spin plane, the other -15° below — sweeping a modulated 200 Hz, 150 V to 8 kV energy/charge window for supersonic ion analysis; (2) two triads of small **127° cylindrical electrostatic analyzers (VEIS)** derived from ISEE-1 covering 7 V to 24.8 kV for both ions and electrons (M ≤ 1 plasmas); and (3) a dedicated **strahl detector** — a truncated toroidal analyzer (5 V to 5 kV, ~3°×±30° FOV) — designed to resolve the field-aligned suprathermal electron beam that carries the corona's collisionless heat flux.

이 논문은 ISTP 프로그램의 태양풍 플라즈마 전용 분석기인 WIND 위성의 **태양풍 실험(SWE) 측정기 설명 논문**이다. SWE는 단일 데이터 처리 장치(DPU) 하에 **세 종류의 상보적 센서군**을 통합한 점이 특이하다. (1) 위성 직경의 양 끝, 회전면에서 각각 +15°와 -15° 기울어진 두 대의 MIT 형 **패러데이 컵(FC)** — 200 Hz로 변조된 150 V부터 8 kV의 에너지/전하 창으로 초음속 이온을 측정. (2) ISEE-1에서 파생된 **127° 원통형 정전 분석기(VEIS)** 트라이어드 두 세트 — 7 V부터 24.8 kV 범위에서 마하수 ≤ 1 인 이온/전자 분포함수 측정. (3) **스트랄 검출기** — 잘린 토로이달 정전 분석기(5 V – 5 kV, 약 3°×±30° 시야)로, 코로나의 무충돌 열속을 운반하는 자기장 정렬 초열전자(strahl)를 분해. 이 세 센서가 함께 작동해 양성자 속도 ±3%, 밀도 ±10%, 열 속도 ±10%의 정밀도로 태양풍 이온의 핵심 매개변수를 도출하고, 알파/양성자 비율과 전자 분포함수를 연속 관측한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

In 1995, ISTP was assembling its multi-spacecraft constellation. WIND, launched 1 Nov 1994, was the upstream solar-wind monitor that handed off measured plasma to ground-based ISTP modelers. SWE was conceived as an integrated successor to two well-proven heritage lines: the **MIT/Bridge Faraday cup** family (Voyager 1977, IMP 7&8 1973-1976) and the **GSFC/Ogilvie-Scudder electron spectrometer** family (ISEE-1 1977). The new wrinkle was the dedicated strahl detector, which exploited the toroidal-analyzer geometry of Young et al. (1987) to capture the narrow, magnetic-field-aligned electron beam that earlier omnidirectional analyzers missed because of finite angular resolution. SWE's operating environment includes the foreshock, magnetosheath, magnetopause, and pristine solar wind near L1, where Mach numbers vary from below unity to ~20.

1995년 ISTP는 다위성 협력 미션을 구성하던 시기였다. 1994년 11월 1일 발사된 WIND는 상류 태양풍 감시 위성이었고, 측정된 플라즈마 자료를 ISTP 모델러에게 전달하는 역할을 맡았다. SWE는 두 개의 검증된 계보 — **MIT/Bridge 패러데이 컵** 계열(Voyager 1977, IMP 7&8 1973–1976)과 **GSFC/Ogilvie-Scudder 전자 분광계** 계열(ISEE-1 1977) — 의 통합 후속작으로 설계되었다. 새로운 점은 전용 스트랄 검출기로, Young 외(1987)의 토로이달 분석기 기하구조를 활용해 종전의 전방향 분석기가 제한된 각 분해능 때문에 놓쳤던 좁은 자기력선 정렬 전자 빔을 포착하는 것이다. 운영 환경은 전방 충격파역, 자기권 외피, 자기권계면, L1 근방의 순수 태양풍을 포괄하며, 마하수는 1 이하부터 약 20까지 변한다.

### 타임라인 / Timeline

```
1959 ─ Lunik 1 (Gringauz, first ion trap)
1962 ─ Mariner 2 (Neugebauer/Snyder Faraday cup, first solar wind)
1971 ─ Vasyliunas review (deep-space plasma technique compendium)
1973 ─ IMP-7 (Bellomo & Mavretic FC)
1977 ─ Voyager 1/2 (Bridge et al. PLS Faraday cups)
1977 ─ ISEE-1 (Ogilvie et al. electron spectrometer)
1979 ─ Scudder & Olbert strahl theory
1987 ─ Young et al. toroidal analyzer test (basis for strahl detector)
1991 ─ Marsch review of inner-heliosphere plasma (Helios)
1993 ─ Manuscript received (May 27)
1994 ─ WIND launch (1 Nov)
1995 ─ THIS PAPER (Space Sci. Rev. 71, 55-77)
1997 ─ ACE launch (SWEPAM = mini-SWE descendant)
2018 ─ Parker Solar Probe (SWEAP — Faraday cup heritage)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Velocity distribution function (VDF) / 속도 분포 함수**: $f(\vec{v})$ such that $f \, d^3v$ is the number density per unit volume in velocity space. Ion moments $n$, $\vec{u}$, $T$ are integrals of $f$. 입자 밀도, 흐름, 온도는 분포의 적분 모멘트.
- **Reduced distribution function / 축소 분포 함수**: $F(v_\parallel) = \iint f(v_\parallel, v_x, v_y) \, dv_x dv_y$ — the FC measures this along its normal. FC가 측정하는 1차원 분포.
- **Faraday cup principle / 패러데이 컵 원리**: A modulated suppressor grid passes only ions whose normal-component energy is in $[qV, q(V+\Delta V)]$; chopping at 200 Hz separates the signal from photo-electron DC. 200 Hz 변조 그리드로 광전자 DC를 제거.
- **Electrostatic analyzer (ESA) / 정전 분석기**: Ions/electrons curve in a radial electric field between concentric plates; only those with $E/q$ in a narrow band at fixed plate voltage exit. 동심 전극 사이 방사 전기장으로 $E/q$ 선택.
- **Geometric factor (GF) / 기하 인자**: $\text{GF} = A_{\text{eff}} \times \Omega \times \Delta E/E$ [cm² sr] determining count rate from differential flux. 카운트율과 미분속을 잇는 인자.
- **Strahl / 스트랄**: Field-aligned, narrow (a few degrees) suprathermal electron beam carrying coronal heat flux outward; remnant of the collisionless escape from corona. 코로나 열속을 운반하는 자기력선 정렬 초열전자 빔.
- **Mach number / 마하수**: $M = v/c_s$; supersonic flow ($M > 1$) collimates the ion population into a narrow cone — perfect for FC, problematic for omni-directional ESA. 초음속 흐름은 좁은 콘에 집중되어 FC에 유리.
- **Spin-stabilized spacecraft / 스핀 안정화 위성**: WIND spins at ~3 s/rev; instruments scan azimuth as the spacecraft rotates. WIND는 약 3초 주기로 회전.
- **Aberration / 광행차**: Spacecraft orbital motion (~30 km/s) deflects apparent solar-wind direction by ~4°. 위성 공전운동에 의한 약 4°의 흐름 방향 보정.
- **Heat flux $\vec{q}_e$ / 전자 열속**: $\vec{q}_e = \frac{1}{2} m_e \int (\vec{v} - \vec{u})^2 (\vec{v} - \vec{u}) \, f_e \, d^3v$. 자기장 정렬 성분이 strahl로 나타남.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| SWE | Solar Wind Experiment — full plasma sensor suite on WIND / WIND의 전체 플라즈마 측정기 |
| FC | Faraday Cup — chopped-grid current-measuring detector (150 V – 8 kV, 35 cm² effective area, ΔE/E 0.065/0.130) / 변조 그리드 전류 측정기 |
| VEIS | Vector Electron and Ion Spectrometer — two triads of 127° cylindrical ESAs (7 V – 24.8 kV, GF 4.6×10⁻⁴ cm² sr, 7.5°×6.5° FOV) / 127° 원통형 분석기 트라이어드 |
| Strahl detector | Truncated toroidal ESA (Young et al. 1987 design), 131° included angle, ±28° FOV in spin-axis plane, 6 anodes × ~5° / 잘린 토로이달 분석기 |
| DPU | Data Processing Unit — Sandia 3300 CPU; mode storage, data formatting, on-board key-parameter computation / 모드 저장 + 키 매개변수 연산 |
| Modulator grid | Square-wave 200 Hz suppressor; ΔV up to 1 kV; gates ions of selected E/q / 200 Hz 사각파 변조 그리드 |
| Logarithmic A/D | Log compander on synchronous detector output, 10-bit + 2 range bits = effective dynamic range 10⁵ / 로그 압축 A/D |
| Channeltron | Continuous-dynode electron multiplier; one each for ion and electron polarity per VEIS analyzer / 연속 다이노드 전자 증배기 |
| Channel plate (MCP) | Microchannel plate stack on strahl-detector toroid for position-sensitive readout / 다채널 플레이트 |
| UV calibrator | Single 2 W RF UV lamp, fiber-coupled to all six VEIS detectors monthly for ~1% relative-gain stability / 월간 UV 광교정 |
| Burst / Event mode | Trigger-driven high-rate buffer for shock and CME crossings / 충격파/CME용 트리거 버스트 모드 |
| Tracking mode | 14 FC velocity windows centered on previous spectrum's peak; 42 s cadence / 14개 창 추적 모드 |
| Single-spin mode | One double-window just below VDF peak; full distribution from one spin via twin peaks of cos²/sin²θ pattern / 단일 회전 모드 |
| Reduced VDF | 1-D integral $\iint f \, dv_\perp$ measured along cup normal; integrated over perpendicular plane / 컵 법선 방향 1차원 분포 |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Faraday cup current / 패러데이 컵 전류
$$
I(\theta, V, \Delta V) = q \, A_{\text{eff}}(\theta) \int_{v_1}^{v_2} v_n \, F(v_n; \theta) \, dv_n
$$
where $v_1 = \sqrt{2qV/m}$, $v_2 = \sqrt{2q(V+\Delta V)/m}$, $v_n$ is the normal component, $A_{\text{eff}}(\theta)$ is the effective collecting area at incidence angle $\theta$ (Figure 6c: ~35 cm² at $\theta = 0$, falling sharply beyond ~45°), and $F(v_n; \theta)$ is the reduced distribution along the cup normal.

여기서 $v_1, v_2$는 변조창 양 끝 속도, $A_{\text{eff}}(\theta)$는 입사각 의존 유효 면적(0°에서 ~35 cm², 45° 이상 급감), $F$는 컵 법선 방향 축소 분포함수.

### (2) Reduced distribution / 축소 분포 함수
$$
F(v_n; \hat{n}) = \iint f(v_n, v_\perp^{(1)}, v_\perp^{(2)}) \, dv_\perp^{(1)} \, dv_\perp^{(2)}
$$
The full 3-D $f(\vec{v})$ is reconstructed by tomography from $F$ measured along multiple azimuth/elevation directions ($\hat{n}$), tilted ±15° in two cups.

여러 방향($\hat{n}$)에서 측정한 $F$의 토모그래피로 3차원 $f$를 복원.

### (3) Convected Maxwellian model for FC fitting / 컨벡티드 맥스웰 모델
$$
f(\vec{v}) = n \left( \frac{m}{2\pi k_B T} \right)^{3/2} \exp\!\left[ -\frac{m(\vec{v} - \vec{u})^2}{2 k_B T} \right]
$$
A non-linear least-squares fit to FC currents extracts $n$, $\vec{u}$ (3 components), and $T$ — the four "key parameters."

비선형 최소제곱 적합으로 $n$, $\vec{u}$(3성분), $T$의 네 핵심 매개변수를 추출.

### (4) Energy/charge window / 에너지/전하 창
$$
\Delta v / v \approx \tfrac{1}{2} \, \Delta E / E
$$
For double-window FC: $\Delta E/E = 0.130$ → $\Delta v/v \approx 0.065$. For VEIS: $\Delta E/E \approx 0.06$. For strahl: $\Delta E/E \approx 0.03$.

### (5) Heat flux / 전자 열속
$$
\vec{q}_e = \tfrac{1}{2} m_e \int (\vec{v} - \vec{u})^2 (\vec{v} - \vec{u}) \, f_e(\vec{v}) \, d^3 v
$$
The strahl is the dominant contributor at large $|v_\parallel|$. Figure 1 (ISEE-1) shows $h$ jumping by ~10× across the foreshock as strahl scattering changes.

스트랄이 큰 $|v_\parallel|$ 영역에서 지배적; Figure 1은 전방 충격파역 통과 시 $h$가 약 10배 변하는 모습.

---

## 6. 읽기 가이드 / Reading Guide

**제 1독 (1 hour)**: Read Sections 1, 2, and 4 (Modes of Operation). Understand WHY: what scientific questions drove SWE's three-sensor architecture, and how mode 0 / mode 1 / burst mode correspond to different operational regimes (cruise, foreshock, shock crossing).

**1차 (1시간)**: 1·2·4장만 읽고 과학 목표와 운영 모드의 대응을 파악.

**제 2독 (2 hours)**: Read Section 3 carefully — sub-section by sub-section.
- 3.2 DPU: data flow, mode flexibility.
- 3.3 VEIS: 127° plate geometry, ΔE/E = 0.06, GF, channeltron rationale, Figure 5 contour plot of f(v∥, v⊥).
- 3.4 FC: focus on Figure 6 (cross section, modulator schematic, A_eff vs angle) and Figure 7 (chopped current vs azimuth — this is THE key figure showing how speed and temperature appear in the data).
- 3.5 Strahl: Young et al. toroidal analyzer, 131°, 6-anode read-out, ~60% mag-field coverage time.

**2차 (2시간)**: 3장을 절별로 정독. 그림 6, 7, 9를 분석해 측정 원리를 체득.

**제 3독 (1 hour)**: Re-read Section 2 with detector pictures in mind: connect the listed scientific objectives to specific sensors. Sketch the cup-current-vs-angle traces for V_sw = 320, 400, 500 km/s to internalize Figure 7's information content.

**3차 (1시간)**: 검출기 구조를 그림으로 확인한 뒤 2장의 과학 목표를 다시 매핑.

**Look out for / 주의 사항**: 
- The paper has NO derivations of the FC integral — be ready to derive Equation (1) yourself. 
- Numerical results are sparse; instead the paper presents simulated traces (Fig. 7) and ISEE-1 heritage data (Figs. 1, 5) as proof-of-concept. 
- Modes section (4) is essential for interpreting downstream data products.

---

## 7. 현대적 의의 / Modern Significance

After 30 years of operation, WIND/SWE is the **gold-standard upstream solar-wind monitor**. The Faraday-cup architecture pioneered here has been carried forward to ACE/SWEPAM (1997, simplified), Parker Solar Probe/SWEAP/SPC (2018, hot-temperature variant), and Solar Orbiter/SWA-PAS. The strahl detector concept enabled the first systematic 1-AU census of strahl pitch-angle width and its modulation by interplanetary structures (CIRs, ICMEs, switchbacks). Today's space-weather operational chain — ENLIL, EUHFORIA, BAS-CME — uses WIND/SWE key parameters as initial-condition validation, and SWE moments still anchor cross-calibration of newer L1 monitors (DSCOVR/Faraday cup, ACE/SWEPAM). The paper's FC modulation principle, 200-Hz lock-in detection, and multi-window fit pipeline remain the textbook design for supersonic plasma analyzers.

운영 30년이 지난 현재, WIND/SWE는 **상류 태양풍 감시의 표준**이다. 이 논문에서 제시된 패러데이 컵 구조는 ACE/SWEPAM(1997, 간소화), Parker Solar Probe/SWEAP/SPC(2018, 고온형), Solar Orbiter/SWA-PAS로 계승되었다. 스트랄 검출기 개념은 1 AU에서 스트랄 피치각 폭과 CIR/ICME/스위치백에 의한 변조를 체계적으로 조사할 수 있게 했다. ENLIL, EUHFORIA, BAS-CME 같은 우주기상 운영 체계는 WIND/SWE 키 매개변수를 초기·검증 자료로 사용하며, SWE 모멘트는 새 L1 모니터(DSCOVR 패러데이 컵, ACE/SWEPAM)의 교차 교정 기준이다. 200 Hz 락인 검출, 다창 적합 파이프라인, 변조 원리는 초음속 플라즈마 분석기 설계의 정석으로 남아 있다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)

### Q1. Why two Faraday cups, one tilted +15° and the other -15°?
**A.** WIND is a spinning spacecraft (~3 s/spin) with the spin axis perpendicular to the ecliptic. A single cup with normal in the spin plane would only sample the equatorial sector of the VDF as it rotates. By tilting one cup +15° and the other -15° (so the cup normals trace cones above and below the spin plane), the relative currents from the two cups give a precise measurement of the elevation angle (out-of-ecliptic flow direction) — a key ISTP quantity. The 15° offset is large enough for meaningful angular discrimination but small enough to stay within the cup's ~60° half-angle acceptance cone (Figure 6c).

WIND는 약 3초 주기로 회전(스핀축 ⊥ 황도면)하며, 한 컵만 회전면에 두면 분포함수의 적도 단면만 보게 된다. 두 컵을 ±15° 기울여 회전축 위·아래를 모두 훑게 함으로써 두 컵의 상대 전류로부터 흐름의 고도각(황도면 외 성분)을 정확히 측정. 이는 ISTP의 핵심 매개변수이다. 15°는 의미 있는 각 분해능을 주면서 컵의 ±60° 반각 수용 콘 내에 들어간다.

### Q2. How does the 200 Hz modulation reject photo-electron and secondary-electron noise?
**A.** Sunlight on the cup interior produces a steady (DC) photo-electron current whose magnitude (~nA) can swamp the wanted signal (~pA). The modulator grid potential switches between V and V+ΔV at 200 Hz, so only ions in the energy window contribute an AC current at 200 Hz; the DC photo-electron leakage stays unchopped and is filtered out by a synchronous (lock-in) detector tuned to 200 Hz. Additional grids prevent capacitive coupling from the modulator from reaching the collector. The suppressor grid biased at -130 V also prevents secondary electrons from escaping the collector.

태양광이 컵 내부를 비춰 ~nA의 DC 광전자 전류를 만든다. 신호는 ~pA로 약해 묻힐 수 있다. 변조 그리드를 V↔V+ΔV로 200 Hz로 전환하면 에너지 창 내 이온만 200 Hz AC를 만들어내고, DC 광전자는 변조되지 않아 200 Hz 동기 검출(락인)에서 제거. 변조의 용량성 결합은 추가 그리드로 차폐, 콜렉터에서 발생하는 2차 전자는 -130 V 억제 그리드로 차단.

### Q3. What is the dynamic-range strategy of the FC measurement chain?
**A.** Each collector half feeds a preamp followed by three series range amplifiers (gains 7, 46.5, 46.5 — total ~1.5×10⁴). After synchronous detection and 30 ms integration, a multiplexer picks the highest-gain UN-saturated output, then a logarithmic A/D produces 10 bits. With 2 bits identifying which range amplifier was picked, the system reaches an effective 12-bit-equivalent dynamic range of 10⁵, covering currents from 3×10⁻¹³ A (thermal noise floor at 30 ms) to 3×10⁻⁸ A. For low-flux look directions (cup pointed away from Sun), integration time stretches to 120 ms.

각 콜렉터 반쪽 → 프리앰프 → 3단 직렬 레인지 앰프(이득 7, 46.5, 46.5; 총 ~1.5×10⁴) → 동기 검파/30 ms 적분 → 멀티플렉서가 최고이득 비포화 출력 선택 → 로그 A/D 10비트 + 2비트 레인지 = 유효 동적 범위 10⁵. 전류 범위 3×10⁻¹³ A부터 3×10⁻⁸ A. 약한 방향에서는 적분 시간을 120 ms로 연장.

### Q4. Why does Figure 7 show a transition from twin-peak to single-peak structure as the modulator window moves?
**A.** Figure 7 plots FC current vs azimuth for five voltage windows scanning from below to above the bulk speed (400 km/s). When the window is **below** the peak (top panel, 309-329 km/s), only ions whose normal-component speed is in this slow band contribute — those are ions arriving at large azimuth angles from the Sun direction (so $v_n = v_{sw} \cos\theta$ is small). Result: twin peaks at ±~30°. As the window approaches the bulk speed (middle panel, 350-373), the two peaks merge. **At** the peak (373-397, fourth panel), the trace becomes flat-topped because the cup samples the full distribution including the bulk. **Above** the peak (397-423, bottom), only ions in the high-energy tail contribute, and they appear narrowly around 0° (the fastest normal-component flow). This twin-peak signature in a single spin is the basis of the "single-spin mode" — the angular separation gives temperature, the azimuth of the centroid gives flow direction.

Figure 7은 5개의 변조 창에서 azimuth(회전각)에 따른 컵 전류를 보여준다. 창이 벌크 속도보다 **느리면**(309-329 km/s), 법선 성분이 느린 이온만 통과 → 이는 태양 방향에서 큰 각도($v_n = v \cos\theta$가 작아지는)에서 오는 이온 → 결과적으로 ±30° 부근 쌍봉. 벌크에 가까워질수록 봉우리가 합쳐지고, 정확히 벌크 속도 창에서는 평탄한 분포 단면. **빠른** 창에서는 분포의 고에너지 꼬리만 잡혀 0° 부근 한 봉우리. 이 이중 봉의 각 분리 폭은 온도, 중심은 흐름 방향. 한 회전(3 s)에서 모든 정보 추출 — 단일 회전 모드.

### Q5. Why use a separate strahl detector instead of just integrating VEIS data along B?
**A.** Three reasons. (1) **Angular resolution**: VEIS analyzers have 7.5°×6.5° FOV, whereas the strahl can be only a few degrees wide. The toroidal strahl analyzer has 6 anodes each ~5°, plus 31 ms angular sweeps every 16° for finer pitch-angle sampling. (2) **Geometry**: Strahl direction varies with B; VEIS coverage in solid angle is fixed by the spin geometry, leaving gaps along B as visible in Figure 5. The strahl detector's ±28° FOV in the spin-axis plane, combined with spin rotation, dedicates ~60% of observing time to the field-aligned region. (3) **Energy resolution**: Strahl detector has ΔE/E = 0.03 (vs 0.06 for VEIS), giving twice the energy resolution to resolve the field-aligned beam's energy spectrum.

세 이유. (1) **각 분해능**: VEIS는 7.5°×6.5°이지만 스트랄은 수 도 폭. 토로이달 분석기는 ~5°씩 6 anode + 16°마다 31 ms 스윕으로 더 정밀. (2) **기하**: B 방향이 변하므로 VEIS의 입체각 커버리지는 그림 5의 빈 영역처럼 자기력선 방향에 갭이 생긴다. 스트랄 검출기는 회전축면에서 ±28°를 가져 회전과 결합 시 자기력선 방향에 ~60% 관측 시간 할당. (3) **에너지 분해능**: 스트랄 ΔE/E = 0.03 (VEIS 0.06의 두 배 정밀).

### Q6. What is the role of the UV calibrator?
**A.** Six independent channeltron detectors in two VEIS triads must have known relative gains (~1%) to combine measurements into a self-consistent 3-D distribution. A single 2 W RF UV lamp is mounted in the DPU/calibrator module; optical fibers carry UV photons to each of the six analyzers. Once a month the lamp is turned on briefly; the photoemission from each detector's photocathode area produces a known relative count rate. Because the lamp itself is the only common source, gain ratios between detectors are determined to ~1% even if the absolute lamp output drifts. This is critical because absolute 3-D distribution functions cannot be reconstructed correctly if individual detector gains drift independently.

여섯 채널트론의 상대 이득(~1%)을 알아야 6×16×6 = 576 점의 3차원 분포를 일관되게 결합할 수 있다. DPU에 단일 2 W RF UV 램프를 두고 광섬유로 6개 분석기에 분배. 월 1회 점등하면 각 검출기에서 광전자 카운트가 발생; 단일 광원이므로 램프 절대값이 변해도 검출기 간 이득 비율은 ~1% 유지. 이 교정 없이는 3차원 분포 재구성이 불가능.
