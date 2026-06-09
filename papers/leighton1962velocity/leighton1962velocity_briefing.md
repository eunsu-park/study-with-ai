# Pre-reading Briefing / 사전 읽기 브리핑

**Paper / 논문**: Velocity Fields in the Solar Atmosphere. I. Preliminary Report.
**Authors / 저자**: Robert B. Leighton, Robert W. Noyes, George W. Simon
**Year / 출판연도**: 1962
**Journal / 저널**: The Astrophysical Journal, Vol. 135, pp. 474–499
**DOI**: 10.1086/147972

---

## 핵심 기여 / Core Contribution

이 논문은 태양 대기의 속도장(velocity field)을 spectroheliograph를 Doppler 측정용으로 개조하여 체계적으로 관측한 최초의 연구입니다. 저자들은 두 가지 근본적 발견을 했습니다: (1) 태양 표면 전체에 걸쳐 수천 킬로미터 규모의 수평 대류 셀(supergranulation)이 균일하게 분포하며, 셀 중심에서 경계로 향하는 수평 유출 흐름을 보임, (2) 태양 표면의 국지적 수직 속도가 무작위 난류가 아니라 약 296초(~5분) 주기의 준진동(quasi-oscillatory) 운동을 나타냄. 이 "5분 진동"의 발견은 이후 helioseismology(일진학) 분야를 탄생시킨 우연한 발견이었습니다.

This paper represents the first systematic observation of velocity fields in the solar atmosphere using a spectroheliograph adapted for Doppler measurements. The authors made two fundamental discoveries: (1) the entire solar surface is covered by uniformly distributed large-scale horizontal convection cells (supergranulation) ~30,000 km in diameter with outward flow from center to boundary, and (2) the local vertical velocities on the solar surface are not random turbulence but exhibit quasi-oscillatory behavior with a period $T = 296 \pm 3$ sec (~5 minutes). This serendipitous discovery of the "5-minute oscillations" later gave birth to the field of helioseismology.

---

## 역사적 맥락 / Historical Context

1960년대 초, 태양 물리학은 태양 대기의 역학적 구조에 대한 이해가 제한적이었습니다. 이미 알려진 것들:

In the early 1960s, understanding of the dynamical structure of the solar atmosphere was limited. What was already known:

- **Granulation (쌀알 무늬)**: 1950년대에 Richardson & Schwarzschild가 고해상도 분광기로 ~0.37 km/s의 "난류" 속도를 측정 (Paper #4, Schwarzschild 1906에서 복사 평형 이론, 이후 대류 이론으로 발전)
- **Evershed 효과** (Paper #6, 1909): 흑점 반음영의 수평 흐름
- **자기장 관측** (Papers #5, #7): Hale이 흑점 자기장을 발견
- **Babcock의 자기장 모델** (Paper #9, 1961): 태양 자기장의 전반적 거동
- **Leighton의 이전 연구** (1959): 자기장 측정용으로 spectroheliograph를 개발 — 이것이 본 논문의 속도장 관측으로 확장됨

이 논문의 위치: Fraunhofer(Paper #2)가 흡수선을 발견하고, Kirchhoff & Bunsen(Paper #3)이 분광 분석을 확립한 이래, 분광학의 힘을 Doppler 효과에 적용하여 태양 표면 전체의 속도를 **2차원 지도로** 처음 만든 작업입니다. 특히 5분 진동의 발견은 태양 내부 구조를 탐사하는 helioseismology로 가는 문을 열었습니다.

This paper's position: Since Fraunhofer (Paper #2) discovered absorption lines and Kirchhoff & Bunsen (Paper #3) established spectral analysis, this work is the first to apply the power of spectroscopy via the Doppler effect to create **2D velocity maps** of the entire solar surface. The discovery of 5-minute oscillations opened the door to helioseismology — probing the Sun's interior structure.

**Timeline / 타임라인:**
```
1814  Fraunhofer — absorption lines
  |
1860  Kirchhoff & Bunsen — spectral analysis
  |
1906  Schwarzschild — radiative equilibrium
  |
1908  Hale — sunspot magnetic fields
  |
1950  Richardson & Schwarzschild — granulation velocity (~0.37 km/s)
  |
1959  Leighton — spectroheliograph for magnetic fields
  |
1961  Babcock — magnetic field model
  |
★ 1962  Leighton, Noyes & Simon — supergranulation + 5-min oscillations ★
  |
1970  Ulrich — p-mode interpretation → helioseismology begins
  |
1975  Deubner — confirms p-mode dispersion relation
```

---

## 필요한 배경 지식 / Prerequisites

### 물리학 / Physics

1. **Doppler 효과 / Doppler Effect**
   - 관측자를 향해 움직이는 광원의 파장은 짧아지고(blueshift), 멀어지면 길어짐(redshift)
   - A source moving toward the observer has shorter wavelength (blueshift); away = redshift
   - $\Delta\lambda / \lambda = v / c$ where $v$ is the line-of-sight velocity

2. **분광학 기초 / Spectroscopy Basics** (Papers #2, #3)
   - Fraunhofer 흡수선: 태양 스펙트럼에서 특정 원소가 빛을 흡수하는 어두운 선
   - Absorption lines: dark lines where specific elements absorb light
   - 이 논문에서 사용하는 주요 흡수선:
     - Fe λ6102, Ca λ6103 (광구 / photosphere)
     - Na D₁ λ5896 (광구 ~ 색구 / photosphere–chromosphere)
     - Ba⁺ λ4554 (광구 / photosphere)
     - Hα λ6563 (색구 / chromosphere)
     - Ca⁺ K λ8542 (색구 / chromosphere)

3. **자기상관 함수 / Autocorrelation (A-C) Function**
   - 공간적 패턴의 통계적 특성(크기, 규칙성)을 정량화하는 수학적 도구
   - A mathematical tool to quantify statistical properties (size, regularity) of spatial patterns
   - $C(s,t) = \frac{K}{A} \int\int T(x,y) T(x+s, y+t) \, dA$
   - FWHM → 패턴의 특성 크기 / characteristic size of the pattern

4. **대류 / Convection**
   - 뜨거운 유체가 상승하고 차가운 유체가 하강하는 열 전달 과정
   - Hot fluid rises, cool fluid sinks — a heat transport mechanism
   - 태양의 대류층(convection zone)에서 에너지를 표면으로 운반
   - Transports energy from the convection zone to the surface

5. **파동 물리학 / Wave Physics**
   - 진동 주기(period), 진폭(amplitude), 감쇠(damping)의 개념
   - Concepts of oscillation period, amplitude, and damping
   - 감쇠 진동: $H(\Delta t) = A \, e^{-\Delta t / \tau} (1 - \cos \omega \Delta t)$

### 선행 논문 / Prior Papers
- **Paper #3 (Kirchhoff & Bunsen, 1860)**: 분광 분석의 기초 — 흡수선이 원소를 식별하는 원리
- **Paper #6 (Evershed, 1909)**: 흑점 반음영에서의 Doppler shift를 이용한 속도장 관측의 선례

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Spectroheliograph** | 태양 표면을 특정 파장에서 촬영하는 장치. 슬릿을 태양 위로 스캔하며 한 줄씩 이미지를 만듦 / Instrument that images the Sun at a specific wavelength by scanning a slit across the solar disk |
| **Line-shifter** | 두 개의 유리 블록으로 스펙트럼 선의 적색/청색 날개를 동시에 선택 — Doppler shift를 밝기 차이로 변환 / Two glass blocks that select red/blue wings of a spectral line simultaneously, converting Doppler shifts to brightness differences |
| **Doppler plate** | 위 장치로 촬영한 사진판. 밝은/어두운 영역 = 접근/후퇴하는 물질 / Photographic plate from the line-shifter. Bright/dark regions = approaching/receding material |
| **Cancellation** | 두 시점의 Doppler plate를 사진적으로 빼기하여 변하지 않은 신호를 제거하는 기법 / Photographic subtraction of two Doppler plates to remove unchanged signals |
| **Supergranulation (초립자)** | 직경 ~30,000 km, 수명 수 시간의 대규모 대류 셀 — 이 논문에서 처음 발견 / Large convection cells ~30,000 km diameter, lifetime of hours — discovered in this paper |
| **5-minute oscillation (5분 진동)** | 태양 표면의 수직 속도가 $T \approx 296$ sec 주기로 진동하는 현상 — 이 논문에서 처음 발견 / Quasi-periodic vertical velocity oscillation on the solar surface with $T \approx 296$ sec — discovered in this paper |
| **Brightness-velocity correlation** | 밝은(뜨거운) 영역이 상승하고 어두운(차가운) 영역이 하강하는 상관관계 — 대류의 증거 / Bright (hot) regions rise, dark (cool) regions sink — evidence for convection |
| **A-C function** | Autocorrelation function — 사진판의 투과율 패턴을 자기 자신과 비교하여 특성 크기와 규칙성을 측정 / Compares a plate's transmission pattern with itself to measure characteristic size and regularity |
| **C-C function** | Cross-correlation function — 서로 다른 두 사진판을 비교 / Compares two different plates |
| **Chromospheric network** | Ca⁺ K₂ spectroheliogram에서 보이는 격자 모양 밝은 구조 — supergranulation 경계와 일치 / Grid-like bright structure seen in Ca⁺ K₂ images — coincides with supergranulation boundaries |
| **Plage (플라주)** | 활동 영역 주변의 밝은 색구 패치 — 강한 자기장과 연관 / Bright chromospheric patches near active regions — associated with strong magnetic fields |

---

## 수식 미리보기 / Equations Preview

### 1. Doppler 속도 관계 / Doppler Velocity Relation

스펙트럼 선의 파장 이동으로부터 시선 속도(line-of-sight velocity)를 구합니다:

From the wavelength shift of a spectral line, we obtain the line-of-sight velocity:

$$\delta = \frac{1}{I_0} \frac{v}{c} \lambda \frac{dI}{d\lambda}$$

여기서 $\delta$는 Doppler shift로 인한 강도 변화, $v$는 속도, $I_0$는 평균 강도, $dI/d\lambda$는 선 프로파일의 기울기입니다.

where $\delta$ is the intensity variation due to Doppler displacement, $v$ is velocity, $I_0$ is mean intensity, and $dI/d\lambda$ is the slope of the line profile.

### 2. 자기상관 함수 / Autocorrelation Function

2차원 투과율 장 $T(x,y)$의 공간 상관:

Spatial correlation of a 2D transmission field $T(x,y)$:

$$C(s,t) = \frac{K}{A} \int\int T(x,y) \, T(x+s, y+t) \, dA = K \langle T(x,y) \, T(x+s, y+t) \rangle$$

- **FWHM** → 패턴의 특성 크기 (e.g., supergranulation ~30,000 km)
- **정규화 피크 높이** $H$ → 평균 제곱 변동의 척도 / measure of mean-square variation

### 3. 밝기–속도 상관 분석 / Brightness–Velocity Correlation Analysis

적색/청색 날개에서의 강도:

Intensities on the red/blue wings:

$$I_r = I_0 [1 + \delta(x,y) + \beta(x,y)]$$
$$I_v = I_0 [1 - \delta(x,y) + \beta(x,y)]$$

여기서 $\delta$는 Doppler shift, $\beta$는 고유 밝기 변동입니다.

where $\delta$ is the Doppler shift contribution and $\beta$ is the intrinsic brightness variation.

이로부터 상관계수 $C$를 정의합니다:

From this we define the correlation coefficient $C$:

$$C = \frac{\langle \beta \, \delta' \rangle}{\langle \beta^2 \rangle^{1/2} \langle \delta'^2 \rangle^{1/2}} = \frac{\langle \beta \, v' \rangle}{\langle \beta^2 \rangle^{1/2} \langle v'^2 \rangle^{1/2}}$$

Table 1 결과: Ca 6103에서 $C = +0.50$, Na 5896에서 $C = -0.23$ → **낮은 고도에서는 밝은 곳이 상승, 높은 고도에서는 밝은 곳이 하강!**

Table 1 results: $C = +0.50$ at Ca 6103, $C = -0.23$ at Na 5896 → **at low altitudes bright regions rise; at high altitudes bright regions descend!**

### 4. 기계적 에너지 수송 / Mechanical Energy Transport

대류에 의한 에너지 수송:

Energy transport by convection:

$$\langle E \rangle = \frac{\gamma}{\gamma - 1} P_0 \left\langle \frac{\Delta T}{T} v \right\rangle$$

추정치: Fe 6102와 Ca 6103 형성 고도에서 $\langle E \rangle \sim 2$ W cm$^{-2}$

Estimate: $\langle E \rangle \sim 2$ W cm$^{-2}$ at the formation altitude of Fe 6102 and Ca 6103

### 5. 감쇠 진동 모델 / Damped Oscillation Model

5분 진동의 시간 상관을 감쇠 코사인으로 모델링:

Modeling the time correlation of 5-minute oscillations as a damped cosine:

$$H(\Delta t) = A \, e^{-\Delta t / \tau} (1 - \cos \, n\omega T)$$

- 주기 / Period: $T = 296 \pm 3$ sec
- 평균 수명 / Mean life: $\tau \sim 380$ sec (약 1.3주기 / ~1.3 periods)
- 최소 3주기 이상 추적 가능 / Can be followed for at least 3 periods

### 6. Supergranulation 셀의 시선 속도 투영 / Line-of-Sight Velocity Projection

셀 내 수평 속도 $V_\rho = F(\rho)$가 관측각 $\theta$에서 시선 방향으로 투영될 때:

When a horizontal velocity $V_\rho = F(\rho)$ within a cell is projected along the line-of-sight at angle $\theta$:

$$V_1(x,y) = \frac{1-\mu^2}{\mu} \frac{y}{\rho} F(\rho)$$

여기서 $\mu = \cos\theta$, $\rho$는 셀 중심으로부터의 거리입니다. 이 투영 효과 때문에 limb 근처에서 셀이 가장 잘 보이고, disk 중심에서는 거의 보이지 않습니다.

where $\mu = \cos\theta$ and $\rho$ is the distance from cell center. Due to this projection, cells are most visible near the limb and nearly invisible at disk center.

---

## 논문의 구조 / Paper Structure

| 섹션 / Section | 내용 / Content |
|---|---|
| Abstract | 6가지 주요 결과 요약 / Summary of 6 main results |
| I. Observational Techniques | Spectroheliograph + line-shifter 장치 설명 / Instrument description |
| II. Measurement Procedure | A-C/C-C 자기상관 장치 및 분석 방법 / Autocorrelation device and analysis methods |
| III. Sensitivity and Errors | Doppler plate의 민감도 및 노이즈 분석 / Sensitivity and noise analysis |
| IV. Results | **(핵심)** a) 대규모 셀, b) 밝기–속도 상관, c) 소규모 Doppler 장, d) 진동 운동 / **(Core)** a) Large cells, b) Brightness–velocity correlation, c) Small-scale Doppler field, d) Oscillatory motions |
| V. Discussion | 결과 해석, supergranulation과 5분 진동의 물리적 의미 / Interpretation, physical meaning |
| VI. Summary | 주요 발견 요약 / Summary of main findings |

---

## 읽기 전 질문 / Pre-reading Questions

논문을 읽으며 다음 질문들을 생각해 보세요:

Consider these questions while reading:

1. **Line-shifter는 어떻게 Doppler shift를 밝기 차이로 변환하는가?** / How does the line-shifter convert Doppler shifts into brightness differences?
2. **Supergranulation과 일반 granulation은 어떻게 다른가?** (크기, 수명, 물리적 기원) / How does supergranulation differ from ordinary granulation? (size, lifetime, physical origin)
3. **밝기–속도 상관관계가 고도에 따라 부호가 바뀌는 이유는?** / Why does the brightness–velocity correlation change sign with altitude?
4. **5분 진동이 "난류"가 아님을 어떻게 증명했는가?** / How did they prove the 5-minute oscillations are not turbulence?
5. **이 발견들이 helioseismology로 발전하게 된 경로는?** / How did these discoveries lead to the development of helioseismology?

---

## 핵심 그림 가이드 / Key Figures Guide

| Figure | 내용 / Description |
|---|---|
| Fig. 1 | 광학 시스템 개략도 — line-shifter 작동 원리 / Optical system schematic — how the line-shifter works |
| Fig. 3 | 5-cm 태양 이미지 Doppler sum plate — 태양 자전 효과 / 5-cm Doppler sum plate showing solar rotation |
| Fig. 4 | 17-cm Doppler sum plate — 대규모 및 소규모 속도장 / 17-cm plate showing large- and small-scale velocity fields |
| Fig. 6 | 대규모 속도장 (supergranulation) — 자전 보정 후 / Large-scale velocity field after rotation correction |
| Fig. 7 | A-C 함수 — supergranulation 크기 측정 (~30,000 km) / A-C function measuring supergranulation size |
| Fig. 9A,B | 밝기–속도 상관의 시각적 증거 및 A-C/C-C 곡선 / Visual evidence and A-C/C-C curves of brightness–velocity correlation |
| Fig. 10 | 다양한 흡수선에서의 singly canceled Doppler plate / Singly canceled Doppler plates in various lines |
| Fig. 14 | Doppler difference plate — 진동의 시각적 증거 / Visual evidence of oscillatory time correlation |
| Fig. 15 | A-C 피크 높이 vs Δt — 진동 패턴 / A-C peak heights vs Δt showing oscillatory pattern |
| Fig. 21 | 3개 흡수선에서의 진동 주기성 — Fe, Ca, Na / Oscillatory periodicity in three spectral lines |
| Fig. 22 | 감쇠 시간 측정: $\tau \sim 380$ sec / Damping time measurement: $\tau \sim 380$ sec |
