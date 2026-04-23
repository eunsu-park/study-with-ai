---
paper_id: 62
title: "Quiet Sun magnetic fields: an observational view"
authors: "Luis Bellot Rubio, David Orozco Suárez"
year: 2019
doi: "10.1007/s41116-018-0017-1"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: notes
tags: [quiet_sun, internetwork, polarimetry, zeeman, hanle, magnetic_fields, spectropolarimetry]
---

# Reading Notes: Quiet Sun Magnetic Fields — An Observational View
# 읽기 노트: 조용한 태양 자기장 — 관측적 관점

## Core Contribution / 핵심 기여

**English.**
Bellot Rubio & Orozco Suárez (2019) deliver the most complete observational synthesis of quiet-Sun magnetism to date. They review all diagnostics (magnetic proxies, Zeeman effect, Hanle effect), all instruments (from Babcock's 1953 photoelectric magnetograph to Hinode/SP and SUNRISE/IMaX), and the full history of discoveries and controversies over nearly five decades. Their principal contributions are fourfold. First, they demonstrate that the long-standing discrepancy between visible Fe I 630 nm (giving kG fields via line ratios) and near-IR Fe I 1565 nm (giving hG fields) has been essentially resolved in favor of weak, hectogauss fields when high-resolution, low-noise measurements and proper treatment of diffraction are applied. Second, they quantify the spatial-resolution dependence of the unsigned longitudinal flux density as |φ| ∝ exp(-1.1·r) where r is resolution in arcsec — a relation that reveals the IN fields begin to be resolved below ~1". Third, they propose a **unifying picture** in which Zeeman and Hanle effects trace the very same IN magnetic population; apparent differences reflect the vastly different angular resolutions (~5" for Hanle vs 0.16" for Hinode). Fourth, they argue that the observed IN field distributions — peaked at ~100-200 G with broad distributions extending to 1.5 kG, inclination PDF peaked at 90° (horizontal) — arise from a continuous population of small-scale magnetic loops emerging and evolving across the solar surface, possibly fed by a local small-scale dynamo.

**한국어.**
Bellot Rubio & Orozco Suárez(2019)는 현재까지 가장 완전한 조용한 태양(quiet-Sun) 자기장의 관측적 종합을 제공한다. 그들은 모든 진단법(자기 대리자, Zeeman 효과, Hanle 효과), 모든 관측기기(Babcock의 1953년 광전 magnetograph부터 Hinode/SP와 SUNRISE/IMaX까지), 그리고 거의 50년간의 발견과 논쟁의 전체 역사를 리뷰한다. 그들의 주요 기여는 네 가지이다. 첫째, 가시광 Fe I 630 nm(선 비율로 kG 자기장을 주는)와 근적외선 Fe I 1565 nm(hG 자기장을 주는) 사이의 오랜 불일치가 고해상도·저잡음 측정과 회절의 적절한 처리가 적용될 때 **약한 hectogauss급 자기장**에 유리한 쪽으로 본질적으로 해결되었음을 입증한다. 둘째, 부호 없는 종단 자속 밀도의 공간 분해능 의존성을 |φ| ∝ exp(-1.1·r)로 정량화했는데, 여기서 r은 arcsec 단위 해상도이며 — IN 자기장이 ~1" 이하에서 해결되기 시작함을 드러낸다. 셋째, Zeeman과 Hanle 효과가 **바로 같은 IN 자기 집단을 추적**한다는 통합 관점을 제안한다; 겉보기 차이는 매우 다른 각 분해능(Hanle은 ~5", Hinode는 0.16")을 반영한다. 넷째, 관측된 IN 자기장 분포 — ~100-200 G에서 피크이고 1.5 kG까지 확장되는 넓은 분포, 90°(수평)에서 피크인 경사 PDF — 가 태양 표면 전역에서 출현·진화하는 소규모 자기 loop의 연속 집단에서 기인하며, 국소 소규모 dynamo가 이를 공급할 가능성이 있다고 주장한다.

## Reading Notes / 읽기 노트

### Section 1: Introduction / 1장: 서론 (pp. 3-5)

**English.**
The quiet Sun is defined as the area outside sunspots and active regions. In continuum intensity it shows only granulation, but in polarized light it reveals a **magnetic network** (kilogauss flux concentrations at supergranular cell boundaries, organized as rosettes with size > 1") and a **magnetic internetwork (IN)** (weak flux concentrations inside supergranular cells). Supergranules have diameters of 30,000 km, lifetimes of 1-2 days, vertical upflows of 30 m/s, and horizontal flows of 300 m/s. The total flux budget is striking: network contains (6.8 ± 1.2) × 10^23 Mx, IN contains (1.1 ± 0.2) × 10^23 Mx over the entire solar surface — comparable to 6 × 10^23 Mx of active regions at solar maximum (cycle 23). This is why quiet-Sun magnetism matters for global solar energetics.

**한국어.**
조용한 태양은 흑점과 활동영역 밖의 영역으로 정의된다. 연속 스펙트럼에서는 granulation만 보이지만, 편광에서는 **자기 network**(supergranular cell 경계의 kilogauss급 자속 집중체, > 1" 크기의 rosette로 조직됨)와 **자기 internetwork(IN)**(supergranular cell 내부의 약한 자속 집중체)이 드러난다. Supergranule은 지름 30,000 km, 수명 1-2일, 수직 상승류 30 m/s, 수평류 300 m/s를 가진다. 총 자속 수지가 놀라운데: 전체 태양 표면에서 network는 (6.8 ± 1.2) × 10^23 Mx, IN은 (1.1 ± 0.2) × 10^23 Mx를 담고 있으며 — 이는 태양활동 극대기(주기 23)의 활동영역 6 × 10^23 Mx와 비슷하다. 이것이 조용한 태양 자기장이 전체 태양 에너지학에서 중요한 이유다.

### Section 2.1-2.2: Diagnostics — Magnetic Proxies and the Zeeman Effect / 2.1-2.2장: 진단법 — 자기 대리자와 Zeeman 효과 (pp. 6-23)

**English.**
Magnetic proxies include G-band bright points (0.2" structures at 430.5 nm from CH molecule evacuation), H-alpha, Ca II K, and CN band head observations. Bright points identify flux concentrations but are not one-to-one indicators: only a fraction of magnetic elements are bright.

The **Zeeman effect** splits atomic levels into 2J+1 sublevels. Wavelength shift of a σ component in a normal triplet:

$$\Delta\lambda_B = \pm \frac{e \lambda_0^2}{4\pi m_e c^2} g B = 4.6686 \times 10^{-10} \lambda_0^2 g_{\text{eff}} B$$

with Δλ_B in mÅ, λ₀ in Å, B in G. The dimensionless effective Landé factor for LS coupling is:

$$g_{\text{eff}} = \frac{1}{2}(g + g') + \frac{1}{4}(g - g')[J(J+1) - J'(J'+1)]$$

The diagnostic regime is determined by the ratio Δλ_B/Δλ_D where Δλ_D = 2(λ₀/c)√(2kT ln2/m) is the Doppler width. When Δλ_B/Δλ_D >> 1 (infrared Fe I 1564.9 nm, g=3), the σ components are clearly separated — the **strong-field regime** where B is measured directly from the peak separation. When Δλ_B/Δλ_D << 1 (most visible lines), we are in the **weak-field regime**.

**한국어.**
자기 대리자는 G-band bright point(CH 분자 소실로 430.5 nm에서 나타나는 0.2" 구조), H-alpha, Ca II K, CN band head 관측을 포함한다. Bright point는 자속 집중체를 식별하지만 일대일 지표는 아니다: 자기 요소의 일부만이 밝게 나타난다.

**Zeeman 효과**는 원자 에너지준위를 2J+1개 부준위로 분리한다. 정상 삼중항(normal triplet)에서 σ 성분의 파장 이동:

$$\Delta\lambda_B = \pm \frac{e \lambda_0^2}{4\pi m_e c^2} g B = 4.6686 \times 10^{-10} \lambda_0^2 g_{\text{eff}} B$$

단위는 Δλ_B는 mÅ, λ₀는 Å, B는 G. LS 결합에서 무차원 유효 Landé 인자는:

$$g_{\text{eff}} = \frac{1}{2}(g + g') + \frac{1}{4}(g - g')[J(J+1) - J'(J'+1)]$$

진단 영역은 비율 Δλ_B/Δλ_D에 의해 결정되며, 여기서 Δλ_D = 2(λ₀/c)√(2kT ln2/m)는 Doppler 폭이다. Δλ_B/Δλ_D >> 1(적외선 Fe I 1564.9 nm, g=3)일 때 σ 성분이 명확히 분리되며 — **강자장 영역**에서 B가 peak 분리로부터 직접 측정된다. Δλ_B/Δλ_D << 1(대부분의 가시광선)일 때는 **약자장 영역**이다.

### Section 2.2.3: Interpretation — Weak-Field Regime / 2.2.3장: 해석 — 약자장 영역 (pp. 14-18)

**English.**
The **weak-field formula** is the cornerstone diagnostic of quiet-Sun magnetism:

$$V(\lambda) = -f \, \Delta\lambda_B \, \cos\gamma \, \frac{dI_0}{d\lambda}$$

where γ is the inclination of B to line of sight, I₀ is the non-magnetic intensity profile, and f is the filling factor. This leads to the **fundamental degeneracy**: Stokes V is proportional to f·B·cos γ (the **longitudinal flux density**, in Mx/cm² if f is dimensionless). The **shape** of V equals dI₀/dλ regardless of B; only the amplitude carries flux information. Therefore from V alone one cannot separate f from B.

For linear polarization to second order in Δλ_B:

$$Q(\lambda) = -\frac{1}{4} f (\Delta\lambda_B)^2 \sin^2\gamma \cos 2\chi \frac{H''(a,v)}{H'(a,v)} \frac{dI_0}{d\lambda}$$

$$U(\lambda) = -\frac{1}{4} f (\Delta\lambda_B)^2 \sin^2\gamma \sin 2\chi \frac{H''(a,v)}{H'(a,v)} \frac{dI_0}{d\lambda}$$

Q and U depend quadratically on the transverse field B_⊥ = B sin γ and linearly on filling factor. Because Q² + U² scales as (f B_⊥²)² = f² B_⊥⁴, extracting B_⊥ from linear polarization requires knowing f from another channel. The **line ratio technique** (Stenflo 1973) uses two lines with different Landé factors (Fe I 525.021/524.705 nm, g = 3 vs 2) but otherwise identical thermodynamics. The ratio V₁/V₂ = g₁/g₂ = 3/2 in the weak-field limit; deviations diagnose stronger fields where Zeeman saturation breaks the ratio, independent of filling factor. However, this method fails if the two lines form at different heights in the presence of vertical field gradients (canopies).

**한국어.**
**약자장 공식**은 조용한 태양 자기장의 핵심 진단이다:

$$V(\lambda) = -f \, \Delta\lambda_B \, \cos\gamma \, \frac{dI_0}{d\lambda}$$

여기서 γ는 B와 시선 방향 사이의 경사각, I₀는 비자기 강도 프로파일, f는 충전 인자이다. 이로부터 **근본적 축퇴**가 발생한다: Stokes V는 f·B·cos γ에 비례한다(**종단 자속 밀도**, f가 무차원이면 Mx/cm² 단위). V의 **형태**는 B와 무관하게 dI₀/dλ와 같고; 오직 진폭만 자속 정보를 담는다. 따라서 V만으로는 f와 B를 분리할 수 없다.

Δλ_B의 이차 항까지 선편광:

$$Q(\lambda) = -\frac{1}{4} f (\Delta\lambda_B)^2 \sin^2\gamma \cos 2\chi \frac{H''(a,v)}{H'(a,v)} \frac{dI_0}{d\lambda}$$

$$U(\lambda) = -\frac{1}{4} f (\Delta\lambda_B)^2 \sin^2\gamma \sin 2\chi \frac{H''(a,v)}{H'(a,v)} \frac{dI_0}{d\lambda}$$

Q와 U는 횡단 자기장 B_⊥ = B sin γ에 이차 의존하고 충전 인자에 선형 의존한다. Q² + U² ∝ (f B_⊥²)² = f² B_⊥⁴이므로, 선편광에서 B_⊥를 추출하려면 다른 채널에서 f를 알아야 한다. **선 비율법**(Stenflo 1973)은 다른 Landé 인자를 가진 두 선(Fe I 525.021/524.705 nm, g = 3 vs 2)이지만 나머지 열역학은 동일한 경우를 이용한다. 약자장 극한에서 V₁/V₂ = g₁/g₂ = 3/2; 이로부터의 편차는 충전 인자와 무관하게 Zeeman saturation이 비율을 깨는 더 강한 자기장을 진단한다. 그러나 이 방법은 수직 자기장 경사(canopy) 존재 시 두 선이 다른 높이에서 형성되면 실패한다.

### Section 2.3: The Importance of Spatial Resolution / 2.3장: 공간 분해능의 중요성 (pp. 23-30)

**English.**
When magnetic features are unresolved, the emergent Stokes vector from a pixel is a two-component mix:

$$I = (1-f) I_{\text{NM}} + f I_M, \quad Q = fQ_M, \quad U = fU_M, \quad V = fV_M$$

The polarization signals (Q, U, V) are proportional to the filling factor f — so as spatial resolution worsens, f decreases and signals weaken, becoming swamped by noise. Additionally, **Zeeman cancellation** occurs when opposite polarities coexist: their V signals subtract. Q, U are polarity-insensitive so cancel less.

For Hinode SP (space-based, 0.32" diffraction-limited, 0.16" pixels), careful simulations using MHD models degraded by the Hinode PSF show polarization signals drop to ~45% of original values — equivalent to a **dilution factor α_d ≈ 0.55**. This is modeled as:

$$I = \alpha_d I_d + (1-\alpha_d) I_M, \quad Q = (1-\alpha_d) Q_M, \quad U = (1-\alpha_d) U_M, \quad V = (1-\alpha_d) V_M$$

80% of pixels show reduced signals (up to 80% reduction); 20% of pixels show larger signals that can carry real information. The solution is to let α_d be a free parameter in inversions or apply deconvolution using the known telescope PSF (van Noort 2012; Ruiz Cobo & Asensio Ramos 2013).

**한국어.**
자기 특징이 해결되지 않을 때, 픽셀에서 나오는 Stokes 벡터는 이성분 혼합이다:

$$I = (1-f) I_{\text{NM}} + f I_M, \quad Q = fQ_M, \quad U = fU_M, \quad V = fV_M$$

편광 신호(Q, U, V)는 충전 인자 f에 비례하므로 — 공간 분해능이 나빠지면 f가 감소하고 신호가 약해져 잡음에 묻힌다. 또한 반대 극성이 공존하면 **Zeeman 상쇄**가 일어난다: V 신호가 상쇄된다. Q, U는 극성 비민감이므로 덜 상쇄된다.

Hinode SP(우주 기반, 0.32" 회절 한계, 0.16" 픽셀)에 대해 Hinode PSF로 degrade된 MHD 모델을 이용한 정밀 시뮬레이션은 편광 신호가 원래 값의 ~45%로 떨어짐을 보인다 — **희석 인자 α_d ≈ 0.55**에 해당. 다음과 같이 모델링된다:

$$I = \alpha_d I_d + (1-\alpha_d) I_M, \quad Q = (1-\alpha_d) Q_M, \quad U = (1-\alpha_d) U_M, \quad V = (1-\alpha_d) V_M$$

픽셀의 80%는 신호 감소(최대 80% 감소)를 보이고; 20%의 픽셀은 실제 정보를 담은 더 큰 신호를 보인다. 해결책은 역변환에서 α_d를 자유 매개변수로 두거나 알려진 망원경 PSF를 이용한 deconvolution을 적용하는 것이다(van Noort 2012; Ruiz Cobo & Asensio Ramos 2013).

### Section 2.5: The Hanle Effect / 2.5장: Hanle 효과 (pp. 34-38)

**English.**
The Hanle effect modifies scattering polarization due to coherence changes between magnetic sublevels. Sensitive to fields of 0.1 B_H to 10 B_H, where the **critical Hanle field** B_H is defined by the condition that Zeeman splitting equals the natural linewidth. For Sr I 460.7 nm, B_H ≈ 20 G. For C₂ molecular lines, much smaller — hence extreme sensitivity to weak fields.

For coherent scattering of anisotropic radiation, the field alters the phase relations, producing three diagnostic signatures:
1. Depolarization of Stokes Q (reduction of amplitude).
2. Rotation of the polarization plane (U ≠ 0).
3. At disk center, forward-scattering Hanle effect creates Q, U signals in presence of weak inclined fields.

The Hanle effect "saturates" for B > 10 B_H — the depolarization is maximum and further field increase produces no additional change in amplitude, though Q/U still inform about azimuth and inclination.

Historical measurements of Sr I 460.7 nm Hanle depolarization (Stenflo 1982; Faurobert-Scholl 1993; Bommier et al. 2005; Trujillo Bueno et al. 2004) consistently indicate turbulent fields in the range 10-100 G. Trujillo Bueno et al. (2004) showed via 3D radiative transfer that an exponential distribution with ⟨B⟩ ~ 130 G reproduces center-limb variations — but only if granules harbor weaker fields (~15 G) than intergranular lanes (>150 G, reaching Hanle saturation).

**한국어.**
Hanle 효과는 자기 부준위 사이의 결맞음(coherence) 변화로 인해 산란 편광을 수정한다. 0.1 B_H에서 10 B_H 범위의 자기장에 민감하며, 여기서 **임계 Hanle 자기장** B_H는 Zeeman 분리가 자연 선폭과 같아지는 조건으로 정의된다. Sr I 460.7 nm에서 B_H ≈ 20 G. C₂ 분자선의 경우 훨씬 작아 — 약한 자기장에 극도로 민감하다.

비등방성 복사의 결맞음 산란에서 자기장은 위상 관계를 변화시켜 세 가지 진단 서명을 생성한다:
1. Stokes Q의 탈편광(진폭 감소).
2. 편광면 회전(U ≠ 0).
3. 원반 중심에서 forward-scattering Hanle 효과는 약한 기울어진 자기장 존재 시 Q, U 신호를 생성한다.

B > 10 B_H에서 Hanle 효과는 "포화"된다 — 탈편광이 최대이고 추가 자기장 증가로 진폭 변화가 없으나, Q/U는 여전히 방위각과 경사에 대한 정보를 준다.

Sr I 460.7 nm Hanle 탈편광의 역사적 측정(Stenflo 1982; Faurobert-Scholl 1993; Bommier et al. 2005; Trujillo Bueno et al. 2004)은 일관되게 10-100 G 범위의 난류 자기장을 지시한다. Trujillo Bueno et al.(2004)은 3D 복사 전달을 통해 ⟨B⟩ ~ 130 G의 지수 분포가 중심-변방 변화를 재현함을 보였다 — 단, granule이 intergranular lane(>150 G, Hanle saturation 도달)보다 약한 자기장(~15 G)을 가질 때만.

### Section 3: IN Dynamics and Evolution / 3장: IN 역학과 진화 (pp. 38-61)

**English.**
**Flux distribution**: IN fluxes follow a power law dN/dφ ∝ φ^(-1.85) from 10^16 to 10^18 Mx (Parnell et al. 2009). Combining with ephemeral regions and active regions, all magnetic features follow a single power law of slope -2.69 over 7 orders of magnitude (10^16 to 10^23 Mx) — suggesting a common dynamo origin.

**Flux budget** (Gošić 2015):
- Network: (6.8 ± 1.2) × 10^23 Mx
- IN: (1.1 ± 0.2) × 10^23 Mx — 15% of total quiet-Sun flux
- Compare active regions cycle 23 max: 6 × 10^23 Mx

**Appearance rates** (Gošić et al. 2016): 120 ± 3 Mx/cm²/day, or (3.7 ± 0.4) × 10^24 Mx/day over the whole Sun — enormous, three times larger than all the ephemeral-region flux (Schrijver et al. 1997).

**Dynamics**: IN elements appear in cell interiors, move toward network at net velocities ~0.15-0.4 km/s (with instantaneous ~3 km/s during emergence). Motion is superdiffusive with γ = 1.44-1.69 (anomalous diffusion >> normal).

**Lifetimes**: exponential distribution with decay time 230 ± 10 s (~4 min); mean 2-15 min.

**Sources**: 
- Bipolar features (small-scale loops): mean size 1"-2" loop tops, footpoint separation up to 1200 km, speeds ~2 km/s, 200 G, lifetime 12 min, 23% reach chromosphere (Martínez González & Bellot Rubio 2009).
- Unipolar flux (the dominant source).

**Sinks**:
- Flux cancellation (opposite-polarity merging).
- In-situ disappearance (fading).
- Transfer to network.

**한국어.**
**자속 분포**: IN 자속은 10^16에서 10^18 Mx까지 dN/dφ ∝ φ^(-1.85)의 멱법칙을 따른다(Parnell et al. 2009). Ephemeral region과 활동영역을 결합하면, 모든 자기 특징이 7자릿수(10^16에서 10^23 Mx)에 걸쳐 기울기 -2.69의 단일 멱법칙을 따른다 — 공통 dynamo 기원을 시사한다.

**자속 수지**(Gošić 2015):
- Network: (6.8 ± 1.2) × 10^23 Mx
- IN: (1.1 ± 0.2) × 10^23 Mx — 총 조용한 태양 자속의 15%
- 비교: 주기 23 극대기 활동영역: 6 × 10^23 Mx

**출현율**(Gošić et al. 2016): 120 ± 3 Mx/cm²/day, 또는 전체 태양에서 (3.7 ± 0.4) × 10^24 Mx/day — 엄청나게 크고, 모든 ephemeral-region 자속의 3배(Schrijver et al. 1997).

**역학**: IN 요소는 cell 내부에서 나타나 ~0.15-0.4 km/s의 순 속도로 network를 향해 이동(출현 단계 동안 순간 ~3 km/s). 운동은 γ = 1.44-1.69로 초확산(정상 확산 >> 훨씬 큼).

**수명**: 붕괴 시간 230 ± 10 s(~4분)의 지수 분포; 평균 2-15분.

**출처**:
- 쌍극 특징(소규모 loop): 평균 loop top 크기 1"-2", 족점(footpoint) 분리 최대 1200 km, 속도 ~2 km/s, 200 G, 수명 12분, 23%가 chromosphere에 도달(Martínez González & Bellot Rubio 2009).
- 단극 자속(지배적 출처).

**소멸처**:
- 자속 상쇄(반대 극성 병합).
- in-situ 사라짐(fading).
- Network로 전송.

### Section 4: Magnetic Properties of the IN / 4장: IN의 자기 특성 (pp. 61-87)

**English.**
**Stokes V profile shapes** (Sigwarth 2001 classification, ~0.3" resolution):
- Normal antisymmetric (two lobes): 15.6%
- Asymmetric (strong amplitude asymmetry): 37.7%
- One-lobed (single polarity): 13.3% + 11.1% = 24.4%
- Mixed polarity (three lobes): 0.5% + 9.9% = 10.4%
- Dynamic (blue/red hump): 9.0% + 2.9% = 11.9%

Amplitude asymmetry and area asymmetry encode vertical gradients in velocity/B; profile complexity reflects sub-resolution magnetic structuring.

**Flux density vs resolution** (SUNRISE/IMaX experiment, Sect. 4.3):
The unsigned longitudinal flux density follows:

$$\overline{|\varphi|}(r) \propto \exp(-1.1 \, r)$$

where r is spatial resolution in arcsec. Below 1", fluxes rise quickly; above 1", fluxes are roughly constant. This threshold indicates that **IN structures are starting to be resolved at ~1"**. Extrapolation to infinite resolution is unwarranted. At 0.15"-0.5" resolution, typical IN flux densities are 10-20 Mx/cm². For comparison:
- Hinode/SP (0.16"): ~11 Mx/cm² longitudinal, ~55 Mx/cm² transverse (ratio ~5!)
- Ground-based 1": ~5-10 Mx/cm²
- Ground-based 3": ~1-3 Mx/cm²

**Field strength PDF** (Orozco Suárez & Bellot Rubio 2012, Hinode deep mode, noise 3 × 10^-4 I_QS, 27.4% of pixels above 4.5σ linear threshold):
- Peak at ~100 G
- FWHM ~190 G
- Tail extending to ~1.5 kG (small bump suggests kG population)
- Average ⟨B⟩ = 170 G

**Inclination PDF**:
- Maximum at 90° (horizontal fields)
- Secondary maxima near 0° and 180° (vertical fields in intergranular lanes)
- Inclination-strength correlation: strong fields vertical, weak fields horizontal

**Filling factors**:
- Hinode/SP 0.32": peak at 25-30%, extended tail
- Ground-based 0.8": 5-10%
- Historical 1"-2": 0.5-2%
- This order-of-magnitude increase with resolution proves IN fields are resolvable.

**한국어.**
**Stokes V 프로파일 형태**(Sigwarth 2001 분류, ~0.3" 해상도):
- 정상 반대칭(두 lobe): 15.6%
- 비대칭(강한 진폭 비대칭): 37.7%
- 단일 lobe(단일 극성): 13.3% + 11.1% = 24.4%
- 혼합 극성(세 lobe): 0.5% + 9.9% = 10.4%
- 동적(청색/적색 hump): 9.0% + 2.9% = 11.9%

진폭 비대칭성과 면적 비대칭성은 속도/B의 수직 경사를 부호화하고; 프로파일 복잡성은 분해능 이하 자기 구조화를 반영한다.

**자속 밀도 vs 해상도**(SUNRISE/IMaX 실험, Sect. 4.3):
부호 없는 종단 자속 밀도는 다음을 따른다:

$$\overline{|\varphi|}(r) \propto \exp(-1.1 \, r)$$

여기서 r은 arcsec 단위 공간 분해능. 1" 이하에서는 자속이 빠르게 증가하고; 1" 이상에서는 대략 일정하다. 이 임계값은 **IN 구조가 ~1"에서 해결되기 시작함**을 지시한다. 무한 해상도로의 외삽은 보장되지 않는다. 0.15"-0.5" 해상도에서 전형적 IN 자속 밀도는 10-20 Mx/cm². 비교하면:
- Hinode/SP (0.16"): 종단 ~11 Mx/cm², 횡단 ~55 Mx/cm² (비율 ~5!)
- 지상 1": ~5-10 Mx/cm²
- 지상 3": ~1-3 Mx/cm²

**자기장 세기 PDF**(Orozco Suárez & Bellot Rubio 2012, Hinode deep mode, 잡음 3 × 10^-4 I_QS, 4.5σ 선편광 임계값 이상 픽셀 27.4%):
- ~100 G에서 피크
- FWHM ~190 G
- ~1.5 kG까지 확장되는 꼬리(작은 bump는 kG 집단 시사)
- 평균 ⟨B⟩ = 170 G

**경사 PDF**:
- 90°(수평 자기장)에서 최대
- 0°와 180° 근처 부가 최대(intergranular lane의 수직 자기장)
- 경사-세기 상관관계: 강한 자기장은 수직, 약한 자기장은 수평

**충전 인자**:
- Hinode/SP 0.32": 25-30%에서 피크, 확장된 꼬리
- 지상 0.8": 5-10%
- 역사적 1"-2": 0.5-2%
- 이 해상도에 따른 자릿수 증가는 IN 자기장이 해결 가능함을 입증한다.

### Section 4.6: The Hanle View / 4.6장: Hanle 관점 (pp. 87-92)

**English.**
Sr I 460.7 nm scattering polarization is the workhorse Hanle diagnostic in the photosphere. Strong Q/I signals at μ = 0.1 (~1.5%) decrease rapidly toward disk center (~0.1% at μ = 0.8). Hanle depolarization signatures require 10^-4 I_QS sensitivity, typically achieved by spatial-temporal averaging over ~10"-50" and minutes.

Field strength determinations from Hanle:
- Stenflo (1982): isotropic ⟨B⟩ ~ 50-100 G at h ~ 200-300 km
- Faurobert-Scholl (1993): 30-60 G at h = 150 km, 10-30 G at 250 km (height gradient!)
- Bommier et al. (2005): ⟨B⟩ = 54 G at 220 km, gradient -0.12 G/km
- Trujillo Bueno et al. (2004): exponential ⟨B⟩ ~ 130 G, granules ~15 G, intergranular lanes >150 G (Hanle-saturated)
- Shchukina & Trujillo Bueno (2011): 160 G at 60 km, 130 G at 300 km
- Milić & Faurobert (2012): rapid decrease 95 G (200 km) → 5 G (400 km)

Molecular C₂ lines indicate weaker fields (~2-15 G), likely because C₂ forms in granules only.

**Key result**: observed center-to-limb variation of Sr I 460.7 nm is consistent with an exponential distribution P(B) = (1/B̄) exp(-B/B̄) with B̄ ~ 130 G, but requires **spatial variations** — granules ≪ intergranular lanes — to simultaneously explain C₂ observations.

**한국어.**
Sr I 460.7 nm 산란 편광은 광구의 주력 Hanle 진단이다. μ = 0.1에서 강한 Q/I 신호(~1.5%)가 원반 중심으로 갈수록 빠르게 감소(μ = 0.8에서 ~0.1%). Hanle 탈편광 서명은 10^-4 I_QS 감도를 요구하며, 일반적으로 ~10"-50" 공간과 분 단위 시간 평균화로 달성된다.

Hanle로부터의 자기장 세기 결정:
- Stenflo (1982): 등방성 ⟨B⟩ ~ 50-100 G, h ~ 200-300 km
- Faurobert-Scholl (1993): h = 150 km에서 30-60 G, 250 km에서 10-30 G (높이 경사!)
- Bommier et al. (2005): 220 km에서 ⟨B⟩ = 54 G, 경사 -0.12 G/km
- Trujillo Bueno et al. (2004): 지수 ⟨B⟩ ~ 130 G, granule ~15 G, intergranular lane >150 G(Hanle 포화)
- Shchukina & Trujillo Bueno (2011): 60 km에서 160 G, 300 km에서 130 G
- Milić & Faurobert (2012): 급감 95 G (200 km) → 5 G (400 km)

분자 C₂ 선은 더 약한 자기장(~2-15 G)을 지시하는데, C₂가 granule에서만 형성되기 때문으로 추정된다.

**핵심 결과**: Sr I 460.7 nm의 관측된 중심-변방 변화는 B̄ ~ 130 G인 지수 분포 P(B) = (1/B̄) exp(-B/B̄)와 일치하지만 — granule ≪ intergranular lane의 **공간 변화**가 필요하여 C₂ 관측을 동시에 설명한다.

### Section 5: Unifying View / 5장: 통합 관점 (pp. 93-96)

**English.**
The authors' critical contribution: Zeeman (Hinode, 0.16") and Hanle (5"-50") measurements actually trace the **same IN magnetic fields**. They demonstrate this by simulating a 5 min Hanle resolution element (0.5" × 5" = 26.2 arcsec²) using 5 × 1024 Hinode/SP pixels. Within one Hanle resolution element:
- Field strength PDF: dominated by weak fields 0-200 G, tail to 1.5 kG
- ⟨B⟩ = 135 G
- 20% of fields exceed Hanle saturation (200 G for Sr I)
- 40% exceed 150 G
- Inclination PDF: peaked near 90° (horizontal)
- Azimuth PDF: flat (random)
- Filling factor PDF: peaked at 30%, extending to 0.6

These properties satisfy **all** Hanle observational constraints:
1. No Hanle rotation (azimuths random → U = 0).
2. Depolarization without strong spatial variations (each Hanle pixel samples full distribution).
3. ⟨B⟩ = 135 G consistent with Trujillo Bueno et al. (2004) ~130 G.

The paper concludes: IN fields organized on **very small scales but not turbulent at 0.32"** — they are resolved coherent structures (magnetic loops) that appear isotropic only when averaged over Hanle-scale elements.

**Origin**: the continuous emergence of small-scale magnetic loops (Martínez González & Bellot Rubio 2009; Centeno et al. 2007; Danilovic et al. 2010; Ishikawa et al. 2010) across the entire solar surface — perhaps driven by a local small-scale dynamo (Vögler & Schüssler 2007) or by the recycling of decaying active-region flux — produces the observed PDFs of field strength, inclination (random due to loop orientations), and azimuth.

**한국어.**
저자들의 결정적 기여: Zeeman(Hinode, 0.16")과 Hanle(5"-50") 측정이 실제로 **같은 IN 자기장**을 추적한다는 것. 이를 5 × 1024개 Hinode/SP 픽셀을 이용해 5분 Hanle 분해능 요소(0.5" × 5" = 26.2 arcsec²)를 시뮬레이션하여 입증한다. 한 Hanle 분해능 요소 내에서:
- 자기장 세기 PDF: 약한 자기장 0-200 G 지배, 1.5 kG까지 꼬리
- ⟨B⟩ = 135 G
- 20%의 자기장이 Hanle 포화(Sr I에서 200 G) 초과
- 40%가 150 G 초과
- 경사 PDF: 90°(수평) 근처 피크
- 방위각 PDF: 평탄(무작위)
- 충전 인자 PDF: 30%에서 피크, 0.6까지 확장

이 특성들은 **모든** Hanle 관측 제약조건을 만족한다:
1. Hanle 회전 없음(방위각 무작위 → U = 0).
2. 강한 공간 변화 없는 탈편광(각 Hanle 픽셀이 전체 분포를 샘플링).
3. ⟨B⟩ = 135 G는 Trujillo Bueno et al.(2004)의 ~130 G와 일치.

논문 결론: IN 자기장은 **매우 작은 스케일에서 조직되지만 0.32"에서 난류가 아니다** — Hanle 스케일 요소로 평균될 때만 등방성으로 보이는 해결된 결맞음 구조(자기 loop)이다.

**기원**: 태양 표면 전역에 걸친 소규모 자기 loop의 연속적 출현(Martínez González & Bellot Rubio 2009; Centeno et al. 2007; Danilovic et al. 2010; Ishikawa et al. 2010) — 국소 소규모 dynamo(Vögler & Schüssler 2007) 또는 붕괴하는 활동영역 자속 재활용에 의해 구동될 가능성이 있으며 — 관측된 자기장 세기, 경사(loop 방향으로 인한 무작위), 방위각 PDF를 생성한다.

### Section 6: Open Questions / 6장: 미결 문제 (pp. 96-105)

**English.**
1. **Center-to-limb variation**: do IN field properties change with heliocentric angle μ = cos θ? Harvey et al. (2007) found increasing polarization toward the limb. Borrero & Kobel (2013) argued true latitudinal variations exist.
2. **Height variation**: Carroll & Kopf (2008) and Danilovic et al. (2016) spatially coupled inversions find kG near continuum, sub-kG at 150 km. Need more multi-line analyses.
3. **High-resolution Hanle**: DKIST (4-m), EST (4-m) will resolve granular Hanle variations, testing Trujillo Bueno et al. (2004) predictions.
4. **Origin**: small-scale dynamo vs. recycled active-region flux — solved by solar-cycle variation studies.

**한국어.**
1. **중심-변방 변화**: IN 자기장 특성이 태양중심각 μ = cos θ에 따라 변하는가? Harvey et al.(2007)은 변방으로 가면서 편광 증가를 발견. Borrero & Kobel(2013)은 실제 위도 변화가 존재한다고 주장.
2. **높이 변화**: Carroll & Kopf(2008)와 Danilovic et al.(2016)의 공간 결합 역변환은 연속 스펙트럼 근처에서 kG, 150 km에서 sub-kG 발견. 더 많은 다선 분석 필요.
3. **고해상도 Hanle**: DKIST(4-m), EST(4-m)가 granular Hanle 변화를 해결하여 Trujillo Bueno et al.(2004) 예측을 시험할 것.
4. **기원**: 소규모 dynamo vs 재활용된 활동영역 자속 — 태양 주기 변화 연구로 해결.

## Key Takeaways / 핵심 시사점

1. **Weak-field weak-formula diagnostic / 약자장 약자 공식 진단**
   - English: The formula V(λ) = -f·Δλ_B·cos γ·dI₀/dλ shows Stokes V measures only the product f·B·cos γ (longitudinal flux density). The shape of V is B-independent — only amplitude carries information — so separating intrinsic B from filling factor f requires either line-ratio diagnostics, full Stokes inversions exploiting the non-magnetic contribution to I, or high spatial resolution.
   - 한국어: 공식 V(λ) = -f·Δλ_B·cos γ·dI₀/dλ는 Stokes V가 f·B·cos γ(종단 자속 밀도)의 곱만을 측정함을 보인다. V의 형태는 B와 무관 — 오직 진폭만 정보를 담아 — 내재적 B를 충전 인자 f로부터 분리하려면 선 비율 진단, I에 대한 비자기 기여를 활용한 전체 Stokes 역변환, 또는 고공간분해능이 필요하다.

2. **Spatial resolution drives the entire field / 공간 분해능이 전체 분야를 구동**
   - English: Every order-of-magnitude improvement in resolution (3" → 1" → 0.5" → 0.16") increased apparent filling factors by comparable amounts (0.5% → 5% → 10% → 25%). This proves the IN fields are resolvable structures, not an unresolved turbulent sea. The exp(-1.1 r) flux law captures this behavior empirically.
   - 한국어: 해상도의 자릿수 개선(3" → 1" → 0.5" → 0.16")은 비슷한 정도로 겉보기 충전 인자를 증가시켰다(0.5% → 5% → 10% → 25%). 이는 IN 자기장이 해결되지 않은 난류 바다가 아니라 **해결 가능한 구조**임을 증명한다. exp(-1.1 r) 자속 법칙이 이 거동을 경험적으로 포착한다.

3. **Weak and highly inclined: the modern consensus / 약하고 강하게 기울어진: 현대적 합의**
   - English: At Hinode resolution, the IN field strength PDF peaks at ~100 G with average 170 G; the inclination PDF peaks at 90° (horizontal). Strong fields (above 500 G) are the minority and concentrated in intergranular lanes where they tend to be more vertical. The visible-vs-near-IR controversy is essentially resolved in favor of weak fields.
   - 한국어: Hinode 해상도에서 IN 자기장 세기 PDF는 ~100 G에서 피크, 평균 170 G; 경사 PDF는 90°(수평)에서 피크. 강한 자기장(500 G 이상)은 소수이며 더 수직인 경향이 있는 intergranular lane에 집중된다. 가시광-근적외선 논쟁은 약한 자기장 쪽으로 본질적으로 해결되었다.

4. **Zeeman cancellation is real but partial / Zeeman 상쇄는 실재하지만 부분적**
   - English: Opposite polarities within a pixel cancel Stokes V but not Q/U. The existence of mixed-polarity V profiles (1-10% of pixels) and the Hanle depolarization both demonstrate that some cancellation occurs. However, high-resolution observations show Zeeman patches have well-defined signs over 0.5"-2", so cancellation is not complete at 0.16" — IN fields are largely resolved.
   - 한국어: 픽셀 내 반대 극성은 Stokes V를 상쇄하지만 Q/U는 상쇄하지 않는다. 혼합 극성 V 프로파일(픽셀의 1-10%)의 존재와 Hanle 탈편광 모두 일부 상쇄가 일어남을 입증한다. 그러나 고해상도 관측은 Zeeman 패치가 0.5"-2"에 걸쳐 잘 정의된 부호를 가짐을 보이므로, 0.16"에서 상쇄는 완전하지 않다 — IN 자기장은 대부분 해결된다.

5. **Unifying Zeeman-Hanle picture / Zeeman-Hanle 통합 관점**
   - English: The simulated ~26 arcsec² Hanle element shows ⟨B⟩ = 135 G (matching Trujillo Bueno et al. 2004's 130 G), predominantly horizontal fields with isotropic azimuths, and filling factor 30%. Zeeman and Hanle are **not** sampling different populations — they sample the same IN fields at vastly different angular resolutions.
   - 한국어: 시뮬레이션된 ~26 arcsec² Hanle 요소는 ⟨B⟩ = 135 G(Trujillo Bueno et al. 2004의 130 G와 일치), 등방성 방위각의 지배적 수평 자기장, 충전 인자 30%를 보인다. Zeeman과 Hanle는 다른 집단을 샘플링하는 것이 **아니다** — 매우 다른 각 분해능에서 같은 IN 자기장을 샘플링한다.

6. **Small-scale magnetic loops populate the IN / 소규모 자기 loop이 IN을 채운다**
   - English: Continuous emergence of Ω-loops with 1"-2" tops, 200 G fields, 12 min lifetimes at rates of 7 × 10^-4 arcsec^-2 s^-1 (Danilovic et al. 2010) produces the observed IN signatures. Half of these loops stay in the photosphere; ~23% reach the chromosphere; some reach the corona. They are the fundamental dynamical unit of the quiet Sun.
   - 한국어: 1"-2" top, 200 G 자기장, 12분 수명의 Ω-loop이 7 × 10^-4 arcsec^-2 s^-1의 비율로(Danilovic et al. 2010) 연속적으로 출현하여 관측된 IN 서명을 생성한다. 이 loop의 절반은 광구에 머물고; ~23%는 chromosphere에 도달; 일부는 corona에 도달한다. 이들이 조용한 태양의 근본 동적 단위이다.

7. **Enormous flux budget with rapid turnover / 빠른 회전을 갖는 엄청난 자속 수지**
   - English: Total quiet-Sun flux ~8 × 10^23 Mx is comparable to all active-region flux at solar maximum. But appearance rate is (3.7 ± 0.4) × 10^24 Mx/day — the entire flux is renewed every ~5 hours! This flux turnover may transfer mechanical energy to the chromosphere/corona.
   - 한국어: 총 조용한 태양 자속 ~8 × 10^23 Mx는 태양활동 극대기의 모든 활동영역 자속과 비슷하다. 그러나 출현율은 (3.7 ± 0.4) × 10^24 Mx/day — 전체 자속이 ~5시간마다 갱신된다! 이 자속 회전은 chromosphere/corona로 기계적 에너지를 전달할 수 있다.

8. **Noise is the ultimate limitation / 잡음이 궁극의 제약**
   - English: Photon noise at 10^-3 I_QS hides weak linear polarization and biases inversions toward inclined fields. Required sensitivity for reliable inference is 10^-4 I_QS, achieved only by deep integrations (~10 min) or 4-m-class telescopes (DKIST). ME inversions with dilution factor and careful noise thresholds (4.5× noise) are needed to avoid biased inclinations.
   - 한국어: 10^-3 I_QS의 광자 잡음은 약한 선편광을 숨기고 역변환을 기울어진 자기장 쪽으로 편향시킨다. 신뢰할 수 있는 추론에 필요한 감도는 10^-4 I_QS이며, 심층 적분(~10분) 또는 4-m급 망원경(DKIST)으로만 달성된다. 희석 인자와 신중한 잡음 임계값(4.5× 잡음)을 갖는 ME 역변환이 편향된 경사를 피하기 위해 필요하다.

## Mathematical Summary / 수학적 요약

### Zeeman splitting / Zeeman 분리

$$\boxed{\Delta\lambda_B = 4.6686 \times 10^{-10} \lambda_0^2 g_{\text{eff}} B}$$

- **English.** With Δλ_B in mÅ, λ₀ in Å, B in G. The effective Landé factor g_eff depends on atomic quantum numbers. For Fe I 630.25 nm, g_eff = 2.5; for Fe I 1564.85 nm, g_eff = 3.0.
- **한국어.** Δλ_B는 mÅ, λ₀는 Å, B는 G 단위. 유효 Landé 인자 g_eff는 원자 양자수에 의존. Fe I 630.25 nm에서 g_eff = 2.5; Fe I 1564.85 nm에서 g_eff = 3.0.

### Weak-field Stokes V / 약자장 Stokes V

$$\boxed{V(\lambda) = -f \, \Delta\lambda_B \, \cos\gamma \, \frac{dI_0}{d\lambda}}$$

- **English.** f is filling factor (dimensionless), γ is inclination of B to LOS. Amplitude of V is proportional to the product f·B·cos γ = longitudinal flux density (Mx/cm²).
- **한국어.** f는 충전 인자(무차원), γ는 B와 시선 방향의 경사. V의 진폭은 f·B·cos γ = 종단 자속 밀도(Mx/cm²)의 곱에 비례.

### Longitudinal flux per pixel / 픽셀당 종단 자속

$$\boxed{\varphi = \frac{\int_A \mathbf{B} \cdot d\mathbf{A}}{\int_A dA} = f B \cos\gamma}$$

- **English.** Flux per unit area. Conventionally reported in Mx/cm² (equivalent to G for filling factor f). For quiet-Sun IN: typical 10-20 Mx/cm² at 0.15"-0.5" resolution.
- **한국어.** 단위 면적당 자속. 관례적으로 Mx/cm²로 보고(충전 인자 f에 대해 G와 등가). 조용한 태양 IN에서: 0.15"-0.5" 해상도에서 일반적 10-20 Mx/cm².

### PDF of internetwork field strength / Internetwork 자기장 세기 PDF

Trujillo Bueno et al. (2004) Hanle-inferred exponential distribution:

$$\boxed{P(B) = \frac{1}{\bar{B}} \exp\left(-\frac{B}{\bar{B}}\right), \quad \bar{B} \approx 130 \text{ G}}$$

Orozco Suárez & Bellot Rubio (2012) Zeeman-inferred PDF (Hinode):
- Lognormal-like shape at small B (peak ~100 G, FWHM 190 G), with power-law tail to ~1.5 kG.

- **English.** The exponential form is motivated by theoretical dynamo models. At small B the lognormal describes continuous random field generation; the kG tail reflects convective intensification in intergranular lanes.
- **한국어.** 지수 형태는 이론적 dynamo 모델에서 유래. 작은 B에서 lognormal은 연속적 무작위 자기장 생성을 기술; kG 꼬리는 intergranular lane에서 대류 강화를 반영.

### Hanle critical field / Hanle 임계 자기장

$$\boxed{g_L \mu_B B_H = \frac{\hbar}{\tau_{\text{rad}}}}$$

or in practice,

$$B_H \approx 1.137 \times 10^{-8} / (g_L t_{\text{life}})$$

- **English.** where t_life is the natural lifetime of the upper atomic level, g_L the Landé factor. For Sr I 460.7 nm, B_H ≈ 20 G. The Hanle effect is sensitive from 0.1 B_H to 10 B_H (saturation).
- **한국어.** 여기서 t_life는 원자 상준위의 자연 수명, g_L는 Landé 인자. Sr I 460.7 nm에서 B_H ≈ 20 G. Hanle 효과는 0.1 B_H에서 10 B_H(포화)까지 민감.

### Magnetic energy density / 자기 에너지 밀도

$$\boxed{E_{\text{mag}} = \frac{B^2}{8\pi}}$$

- **English.** For B = 100 G, E_mag ≈ 400 erg/cm³. Compared to photospheric kinetic energy density ~1000 erg/cm³, quiet-Sun IN fields are slightly sub-equipartition. The equipartition field is B_eq ≈ 500 G, near which convective intensification saturates.
- **한국어.** B = 100 G에서 E_mag ≈ 400 erg/cm³. 광구 운동 에너지 밀도 ~1000 erg/cm³와 비교하면, 조용한 태양 IN 자기장은 약간 sub-equipartition. 등분배(equipartition) 자기장은 B_eq ≈ 500 G이며, 이 근처에서 대류 강화가 포화된다.

### Unsigned flux density vs resolution / 부호 없는 자속 밀도 vs 해상도

$$\boxed{\overline{|\varphi|}(r) = \varphi_0 \exp(-1.1 \, r)}$$

- **English.** r in arcsec. Empirical law from SUNRISE/IMaX spatial degradation experiments. Valid below 3"; rapid rise below 1" indicates resolved IN structures.
- **한국어.** r은 arcsec 단위. SUNRISE/IMaX 공간 degradation 실험의 경험 법칙. 3" 이하에서 유효; 1" 이하에서 급증은 해결된 IN 구조를 지시.

### Two-component mix / 이성분 혼합

$$\boxed{I = (1-f) I_{\text{NM}} + f I_M, \quad Q = fQ_M, \quad U = fU_M, \quad V = fV_M}$$

- **English.** Basic model for unresolved pixels. Polarization signals (Q, U, V) come only from the magnetic component, so they are directly proportional to f.
- **한국어.** 해결되지 않은 픽셀에 대한 기본 모델. 편광 신호(Q, U, V)는 오직 자기 성분에서만 오므로, f에 직접 비례한다.

### Worked numerical example / 수치 예제

**English.**
Consider a quiet-Sun IN pixel with B = 100 G, γ = 90° (horizontal), f = 0.3, observed in Fe I 630.25 nm (g_eff = 2.5).

- Zeeman splitting: Δλ_B = 4.6686 × 10^-10 × (6302.5)² × 2.5 × 100 = 4.64 mÅ
- Doppler width at T = 5800 K: Δλ_D ≈ 25 mÅ → ratio = 0.18 (deeply weak-field regime)
- Stokes V amplitude (at dI/dλ maximum ~0.015 per mÅ): V ∝ 0.3 × 4.64 × 0 × 0.015 = 0 (horizontal → V = 0!)
- Stokes Q amplitude: Q ∝ 0.3 × (4.64)² × 1 × (mode numerical factor ~0.1 × 0.015) ≈ 10^-4 I_QS

This illustrates why horizontal IN fields are detected via Q/U, not V — and why the Lites et al. (2008) discovery of ubiquitous transverse polarization was a breakthrough.

**한국어.**
B = 100 G, γ = 90°(수평), f = 0.3인 조용한 태양 IN 픽셀을 Fe I 630.25 nm(g_eff = 2.5)로 관측한다고 하자.

- Zeeman 분리: Δλ_B = 4.6686 × 10^-10 × (6302.5)² × 2.5 × 100 = 4.64 mÅ
- T = 5800 K에서 Doppler 폭: Δλ_D ≈ 25 mÅ → 비율 = 0.18 (깊은 약자장 영역)
- Stokes V 진폭(dI/dλ 최대 ~0.015 per mÅ에서): V ∝ 0.3 × 4.64 × 0 × 0.015 = 0 (수평 → V = 0!)
- Stokes Q 진폭: Q ∝ 0.3 × (4.64)² × 1 × (수치 인자 ~0.1 × 0.015) ≈ 10^-4 I_QS

이는 왜 수평 IN 자기장이 V가 아닌 Q/U로 검출되는지를 — 그리고 왜 Lites et al.(2008)의 보편적 횡단 편광 발견이 돌파구였는지를 — 설명한다.

## Paper in the Arc of History / 역사 속의 논문

```
┌───────────────────────────────────────────────────────────────┐
│                 Quiet Sun Magnetism Timeline                  │
│                조용한 태양 자기장 연대표                        │
└───────────────────────────────────────────────────────────────┘

1953 ───── Babcock: photoelectric magnetograph invented
           (광전 magnetograph 발명)
            │
1967 ───── Sheeley: CN band magnetic proxy
           (CN band 자기 대리자)
            │
1971/75 ── Livingston & Harvey: discovery of IN fields (10^16-17 Mx)
           (IN 자기장 발견)
            │
1973 ───── Stenflo: line ratio technique (Fe 525/524 nm)
           (선 비율법)
            │
1982 ───── Stenflo: Hanle effect on Sr I 460.7 nm → 50-100 G turbulent
           (Hanle 효과로 50-100 G 난류 자기장)
            │
1993 ───── Solanki: flux-tube review; kG network consensus
           (flux-tube 리뷰; kG network 합의)
            │
1995 ───── Lin: near-IR Fe 1565 nm → IN is weak (~500 G)
           (근적외선 → IN 약자장)
            │
1996 ───── Lites et al.: HIFs (Horizontal Internetwork Fields) discovered
           (HIF 발견)
            │
2004 ───── Trujillo Bueno et al. (Nature): hidden magnetic energy
           via Hanle; exponential PDF ⟨B⟩ ~ 130 G
           (숨은 자기 에너지; 지수 PDF ⟨B⟩ ~ 130 G)
            │
2006 ───── **Hinode launch** — 0.16" pixels, 10^-3 I_QS sensitivity
           (Hinode 발사)
            │
2007 ───── Orozco Suárez et al.: first Hinode IN inversions
           (첫 Hinode IN 역변환)
            │
2008 ───── Lites et al.: ubiquitous linear polarization in IN
           horizontal flux 55 Mx/cm² vs longitudinal 11 Mx/cm²
           (보편적 IN 선편광 발견)
            │
2009 ───── SUNRISE-I: 0.15" resolution balloon-borne observations
           (SUNRISE-I: 0.15" 해상도)
            │
2012 ───── Orozco Suárez & Bellot Rubio: deep-mode PDF
           peak 100 G, tail to 1.5 kG
           (심층 모드 PDF)
            │
2013 ───── SUNRISE-II: improved seeing-free observations
           (SUNRISE-II: 개선된 무잠상 관측)
            │
2016 ───── Gošić et al.: flux appearance rate 120 Mx/cm²/day
           (자속 출현율)
            │
2016 ───── Danilovic et al.: spatially coupled inversions
           height gradients: kG at tau=0, sub-kG at 150 km
           (공간 결합 역변환; 높이 경사)
            │
2019 ──── ★ Bellot Rubio & Orozco Suárez (THIS PAPER): unified view
          Zeeman + Hanle = same IN fields; hierarchy of loops
          (★ 통합 관점: Zeeman + Hanle = 같은 IN 자기장; loop 계층)
            │
202x ──── DKIST (4-m, 2020+), EST (4-m, 2030+): subgranular Hanle
          (DKIST/EST: sub-granular Hanle)
```

## Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Relation / 관계 |
|---|---|
| Nordlund et al. (2009) — Solar Convection | **English.** Provides the convective dynamics (granulation, supergranulation) that advect and concentrate IN fields. Quiet-Sun magnetism cannot be understood without this dynamical substrate. **한국어.** IN 자기장을 이류·집중시키는 대류 역학(granulation, supergranulation)을 제공. 이 동적 기질 없이는 조용한 태양 자기장을 이해할 수 없다. |
| Stein (2012) — Solar Surface Convection | **English.** MHD simulations of magnetoconvection provide synthetic observables to test Hinode/SP inversions. Direct reference for Orozco Suárez et al. (2007a) noise experiments. **한국어.** magnetoconvection MHD 시뮬레이션이 Hinode/SP 역변환을 시험하기 위한 합성 관측값을 제공. Orozco Suárez et al.(2007a) 잡음 실험의 직접적 참고. |
| Borrero & Ichimoto (2011) — Penumbral Magnetic Fields | **English.** Shares the full-Stokes inversion methodology (SIR, MELANIE, HELIX, MERLIN codes) and Milne-Eddington framework. Penumbra is the strong-field counterpart to the IN. **한국어.** 전체 Stokes 역변환 방법론(SIR, MELANIE, HELIX, MERLIN 코드)과 Milne-Eddington 틀을 공유. Penumbra는 IN의 강자장 대응물. |
| Rieutord & Rincon (2010) — Sun's Supergranulation | **English.** Sets the scale (30 Mm cells) and flow pattern (30 m/s upflows, 300 m/s horizontal) that organize IN field appearance and migration to network. **한국어.** IN 자기장 출현과 network로의 이주를 조직하는 스케일(30 Mm cell)과 흐름 패턴(30 m/s 상승류, 300 m/s 수평)을 설정. |
| Solanki et al. (2006) — Solar Magnetism Review | **English.** Earlier, more general review covering active regions and network. This paper is the focused IN sequel after a decade of Hinode data. **한국어.** 활동영역과 network를 다루는 이전의 더 일반적인 리뷰. 이 논문은 10년간의 Hinode 데이터 이후 IN에 집중한 후속작. |
| Trujillo Bueno et al. (2004) — Hidden Magnetic Energy (Nature) | **English.** Key Hanle result establishing ⟨B⟩ ~ 130 G exponential PDF. This 2019 review demonstrates the consistency with Zeeman measurements (Sect. 5). **한국어.** ⟨B⟩ ~ 130 G 지수 PDF를 확립한 핵심 Hanle 결과. 이 2019 리뷰는 Zeeman 측정과의 일관성을 입증(Sect. 5). |
| Martínez González & Bellot Rubio (2009) — Small-Scale Loops | **English.** Provides the dynamical unit (Ω-loops) that populate the IN, underlying the unifying picture in Sect. 5. **한국어.** IN을 채우는 동적 단위(Ω-loop)를 제공, Sect. 5의 통합 관점의 기반. |

## Standalone Test / 독립적 이해 검증

**English.** Someone who has not read the original paper but reads these notes should be able to:

1. Explain that the quiet Sun — outside sunspots — is actually filled with magnetic structures detected via polarimetry, organized as a kG network at supergranular boundaries and a weaker internetwork inside cells.
2. Use the weak-field formula V ∝ f·B·cos γ·dI/dλ to understand why Stokes V cannot separate intrinsic field from filling factor.
3. Quote that the IN contains 1.1 × 10^23 Mx of flux, renewed at 3.7 × 10^24 Mx/day — comparable to active-region flux.
4. Describe the Hinode SP resolution (0.16" pixel, 0.32" diffraction-limited) and sensitivity (10^-3 I_QS).
5. Explain why historical visible-vs-near-IR controversies existed (different sensitivities, formation heights, line ratios) and how high-resolution + proper noise treatment resolved them in favor of weak fields.
6. State the modern consensus: IN fields peak at ~100 G, average 170 G, predominantly horizontal (inclination PDF at 90°), with resolvable structure down to at least 0.16".
7. Explain the unifying view: the same IN fields produce both Zeeman (at 0.16") and Hanle (at 5"-50") signals when resolution effects are accounted for.
8. Sketch the future: DKIST and EST high-resolution Hanle will resolve granular-scale variations, potentially confirming small-scale dynamo action.

**한국어.** 원 논문을 읽지 않고 이 노트만 읽은 사람이 다음을 할 수 있어야 한다:

1. 조용한 태양 — 흑점 밖 — 이 실제로 편광계 측정으로 검출되는 자기 구조로 채워져 있으며, supergranular 경계의 kG network와 cell 내부의 더 약한 internetwork로 조직됨을 설명할 수 있다.
2. 약자장 공식 V ∝ f·B·cos γ·dI/dλ를 이용해 왜 Stokes V가 내재적 자기장과 충전 인자를 분리할 수 없는지 이해할 수 있다.
3. IN이 1.1 × 10^23 Mx의 자속을 담고 있으며, 3.7 × 10^24 Mx/day로 갱신됨 — 활동영역 자속과 비슷함 — 을 인용할 수 있다.
4. Hinode SP 해상도(0.16" 픽셀, 0.32" 회절 한계)와 감도(10^-3 I_QS)를 기술할 수 있다.
5. 역사적 가시광-근적외선 논쟁이 왜 존재했는지(다른 감도, 형성 높이, 선 비율) 및 고해상도 + 적절한 잡음 처리가 이를 약한 자기장 쪽으로 어떻게 해결했는지 설명할 수 있다.
6. 현대 합의를 진술할 수 있다: IN 자기장은 ~100 G에서 피크, 평균 170 G, 대부분 수평(경사 PDF 90°), 적어도 0.16"까지 해결 가능한 구조.
7. 통합 관점을 설명할 수 있다: 해상도 효과가 고려될 때 같은 IN 자기장이 Zeeman(0.16"에서)과 Hanle(5"-50"에서) 신호를 모두 생성.
8. 미래를 스케치할 수 있다: DKIST와 EST 고해상도 Hanle이 granular 스케일 변화를 해결하여 소규모 dynamo 작용을 잠재적으로 확인.

## References / 참고문헌

- Bellot Rubio, L., Orozco Suárez, D., "Quiet Sun magnetic fields: an observational view", Living Reviews in Solar Physics, 16:1, 2019. [DOI: 10.1007/s41116-018-0017-1]
- Lites, B. W., Kubo, M., Socas-Navarro, H., et al., "The horizontal magnetic flux of the quiet-Sun internetwork as observed with the Hinode spectro-polarimeter", ApJ, 672, 2008.
- Trujillo Bueno, J., Shchukina, N., Asensio Ramos, A., "A substantial amount of hidden magnetic energy in the quiet Sun", Nature, 430, 2004.
- Orozco Suárez, D., Bellot Rubio, L. R., del Toro Iniesta, J. C., et al., "Quiet-Sun Internetwork Magnetic Fields from the Inversion of Hinode Measurements", ApJ Lett, 670, L61-L64, 2007.
- Orozco Suárez, D., Bellot Rubio, L. R., "Milne-Eddington inversions of the quiet Sun internetwork", ApJ, 751, 2, 2012.
- Martínez González, M. J., Bellot Rubio, L. R., "Emergence of small-scale magnetic loops through the quiet solar atmosphere", ApJ, 700, 1391, 2009.
- Gošić, M., Bellot Rubio, L. R., Orozco Suárez, D., et al., "The short-term variability of the Quiet Sun", ApJ, 820, 35, 2016.
- Parnell, C. E., DeForest, C. E., Hagenaar, H. J., et al., "A power-law distribution of solar magnetic fields over more than five decades in flux", ApJ, 698, 75, 2009.
- Stenflo, J. O., "Dominant hidden turbulence in the quiet Sun" (history review), Adv. Space Res., 2010.
- Danilovic, S., Schüssler, M., Solanki, S. K., "Magnetic fields of the quiet Sun observed by SUNRISE/IMaX", A&A, 513, 2010.
- Sánchez Almeida, J., Martínez González, M. J., "The magnetism of the very quiet Sun", ASP Conf. Ser. 437, 2011.
- del Toro Iniesta, J. C., "Introduction to Spectropolarimetry", Cambridge University Press, 2003.
- Landi Degl'Innocenti, E., Landolfi, M., "Polarization in Spectral Lines", Kluwer, 2004.
