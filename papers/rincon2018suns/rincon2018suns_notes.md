---
title: "The Sun's Supergranulation (2018 Update)"
authors: ["François Rincon", "Michel Rieutord"]
year: 2018
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-018-0013-5"
topic: Living_Reviews_in_Solar_Physics
tags: [supergranulation, convection, turbulence, MHD, solar-surface, helioseismology, review]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 60. The Sun's Supergranulation (2018 Update) / 태양의 초립상 (2018 업데이트)

> An update to Paper #19 (Rieutord & Rincon 2010). Approximately 40 new references, major revision incorporating SDO/HMI observational advances and petascale numerical simulations from the 2010-2018 period.
>
> Paper #19(Rieutord & Rincon 2010)의 업데이트. 약 40개의 새 참고문헌, 2010-2018 기간의 SDO/HMI 관측 발전과 페타스케일 수치 시뮬레이션을 반영한 대규모 개정.

---

## 1. Core Contribution / 핵심 기여

**EN**: Rincon & Rieutord (2018) deliver an exhaustive, 74-page review synthesizing 60+ years of observational, theoretical, and numerical research on solar supergranulation — a quiet-Sun photospheric cellular flow pattern of horizontal scale 30-35 Mm, lifetime 24-48 h, strong horizontal rms velocity 300-400 m/s, and a much weaker vertical component (20-30 m/s). The paper's central assertion, strengthened by new SDO/HMI full-disk data and large-scale numerical simulations accumulated since the 2010 version, is that supergranulation is most plausibly a **large-scale, buoyancy-driven, near-critical nonlinear convection** phenomenon living on the large-scale side of the photospheric convection injection spectrum — rather than an independent, separately-driven process. The review systematically addresses: (i) scale hierarchy in the SCZ ($\ell_\nu \ll \ell_\eta \ll \ell_\kappa \sim H_p \sim L_G \ll L_{SG}$); (ii) Doppler, tracking, and helioseismic flow measurement; (iii) horizontal/vertical velocity anisotropy and the radically different spectra of spheroidal vs. toroidal horizontal flow components; (iv) rotational effects ($Ro_{SG} \sim 2$-3) including superrotation and anticyclonic vertical vorticity; (v) magnetic network correlation; (vi) classical Rayleigh-Bénard theory and its laminar extensions; (vii) state-of-the-art stratified and MHD simulations; and (viii) open questions on what physics sets the 30 Mm scale.

**KR**: Rincon & Rieutord (2018)는 60년 이상의 태양 초립상 연구(관측, 이론, 수치)를 종합한 74페이지의 방대한 리뷰입니다. 초립상은 조용한 태양 광구에 나타나는 수평 규모 30-35 Mm, 수명 24-48시간, 강한 수평 rms 속도 300-400 m/s, 훨씬 약한 수직 성분(20-30 m/s)의 셀 유동 패턴입니다. 2010년 버전 이후 축적된 새로운 SDO/HMI 전면 디스크 데이터와 대규모 수치 시뮬레이션 결과를 통해 강화된 본 논문의 중심 주장은, 초립상이 **광구 대류 주입 스펙트럼의 대규모 측에 존재하는 대규모 부력 구동 임계 근접(near-critical) 비선형 대류** 현상일 가능성이 가장 높다는 것입니다 — 독립적으로 구동되는 별개의 프로세스가 아니라. 리뷰는 다음을 체계적으로 다룹니다: (i) SCZ 내 규모 계층; (ii) Doppler, 추적, 일식학적 유동 측정; (iii) 수평/수직 속도 이방성과 구면조화적(spheroidal) 대 토로이달(toroidal) 수평 성분 스펙트럼의 극적 차이; (iv) 회전 효과($Ro_{SG} \sim 2$-3), 초자전 및 역회전성(anticyclonic) 수직 와도 포함; (v) 자기 네트워크 상관; (vi) 고전적 Rayleigh-Bénard 이론과 그 층류 확장; (vii) 최신 성층 및 MHD 시뮬레이션; (viii) 30 Mm 규모를 결정하는 물리에 대한 미해결 문제.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and the Supergranulation Puzzle (§1-2, pp. 3-9) / 서론과 초립상 퍼즐

**EN**: The paper opens by recapitulating the historical discovery: Avril B. Hart (1954) in Oxford reported a "noisy" velocity field superposed on the solar rotation. Hart (1956) provided the first accurate length-scale estimate (~26 Mm). Leighton, Noyes & Simon (1962) established supergranulation via Doppler imaging, and Simon & Leighton (1964) linked it to the magnetic network. After 60+ years, uncertainty persists about origin, interaction with magnetic fields, universality across stellar convection, and role in the dynamo.

§2.1 reviews the quiet-Sun dynamical landscape:
- **Solar convection zone (SCZ)**: outer 30% by radius, weakly superadiabatic down to $\sim 0.3 R_\odot$, density ratio top-to-bottom $\sim 10^6$ — strongly stratified.
- **Granulation**: intensity contrast ~15%, scale 0.5-2 Mm, velocity 0.5-1.5 km/s, lifetime 5-10 min; well-understood as buoyancy-radiation balance at the thin thermal boundary layer where the plasma becomes optically thin.
- **Supergranulation**: 30-35 Mm typical scale, essentially horizontal flow (weak signal at disc centre — Fig. 1 of the paper is a SOHO/MDI Doppler image showing this).
- **Mesogranulation**: proposed by November et al. (1981) at ~8 Mm, but high-resolution SDO/MDI/Hinode spectra (Hathaway et al. 2015; Rincon et al. 2017) show **no distinctive spectral bump** at mesoscales. The review explicitly avoids the term "mesogranulation" as a physical phenomenon and uses "mesoscale range" instead.

§2.2 introduces turbulent-scale physics:
- **Injection scale**: where buoyancy injects KE. In stratified turbulence, the Bolgiano scale $L_B \sim H_\rho \sim H_p$ is expected — near the surface this is ~1 Mm (comparable to granulation), much smaller than supergranulation.
- **Dissipation scales**: viscous $\ell_\nu \sim Re^{-3/4} L$; magnetic $\ell_\eta \sim Pm^{-3/4} \ell_\nu$ (in low-Pm Sun); thermal $\ell_\kappa$. With $Re \sim 10^{10}$-$10^{13}$, $\ell_\nu \sim 10^{-3}$ m — far beyond any current observation or simulation capability.
- **Scale ordering (near surface)**: $\ell_\nu \ll \ell_\eta \ll \ell_\kappa \sim H_p \sim H_\rho \sim L_B \sim L_G \ll L_{SG} \ll R_\odot$.

§2.3 "What about supergranulation?" is the central puzzle statement: **standard turbulent MHD convection phenomenology does NOT easily yield a privileged 30 Mm scale**. The injection scale grows self-similarly with depth; picking out ~30 Mm requires extra physics (e.g., changes in ionisation, rotation, magnetism, or nonlinear inverse cascades).

**KR**: 논문은 역사적 발견 요약으로 시작합니다. Avril B. Hart(1954)가 옥스퍼드에서 태양 자전 위에 중첩된 "잡음" 속도장을 보고했고, Hart(1956)는 최초의 정확한 규모 추정(~26 Mm)을 제시했습니다. Leighton, Noyes & Simon(1962)은 Doppler 이미징으로 초립상을 확립했고, Simon & Leighton(1964)은 자기 네트워크와의 연결을 밝혔습니다. 60년이 넘었지만 기원, 자기장과의 상호작용, 항성 대류에서의 보편성, 발전기에서의 역할에 대한 불확실성이 여전합니다.

§2.1은 조용한 태양 동역학 풍경을 검토합니다:
- **태양 대류층(SCZ)**: 반지름의 외곽 30%, $\sim 0.3 R_\odot$까지 약하게 초단열, 상하 밀도비 $\sim 10^6$ — 강한 성층.
- **입상**: 복사강도 대비 ~15%, 규모 0.5-2 Mm, 속도 0.5-1.5 km/s, 수명 5-10분; 얇은 열 경계층에서의 부력-복사 균형으로 잘 이해됨.
- **초립상**: 전형적 규모 30-35 Mm, 본질적으로 수평 유동 (논문의 Fig. 1은 SOHO/MDI Doppler 이미지로 디스크 중심에서 신호가 약함을 보여줌).
- **중간립상**: November et al.(1981)이 ~8 Mm 규모로 제안했으나, SDO/MDI/Hinode 고해상도 스펙트럼(Hathaway et al. 2015; Rincon et al. 2017)에서 **뚜렷한 스펙트럼 범프가 없음**. 본 리뷰는 물리적 현상으로서 "중간립상" 용어를 피하고 "중간 규모 대역"으로 사용.

§2.2는 난류 규모 물리를 소개합니다:
- **주입 규모**: 부력이 KE를 주입하는 곳. 성층 난류에서 Bolgiano 규모 $L_B \sim H_\rho \sim H_p$가 예상되며 — 표면 근처에서는 ~1 Mm(입상과 비슷), 초립상보다 훨씬 작음.
- **소산 규모**: 점성 $\ell_\nu \sim Re^{-3/4} L$; 자기 $\ell_\eta \sim Pm^{-3/4} \ell_\nu$(낮은 Pm 태양); 열 $\ell_\kappa$. $Re \sim 10^{10}$-$10^{13}$이므로 $\ell_\nu \sim 10^{-3}$ m — 현재의 관측이나 시뮬레이션 능력을 훨씬 초과.
- **규모 순서 (표면 근처)**: $\ell_\nu \ll \ell_\eta \ll \ell_\kappa \sim H_p \sim H_\rho \sim L_B \sim L_G \ll L_{SG} \ll R_\odot$.

§2.3 "초립상은 어떤가?"는 중심 퍼즐 서술입니다: **표준 난류 MHD 대류 현상학으로는 30 Mm 특별 규모를 쉽게 얻지 못함**. 주입 규모는 깊이에 따라 자기유사적으로 증가하므로 ~30 Mm를 골라내려면 추가 물리(이온화 변화, 회전, 자성, 비선형 역 캐스케이드)가 필요.

### Part II: Observational Characterization (§3, pp. 9-28) / 관측 특성화

**EN**: This is the meat of the 2018 update. Three flow-measurement techniques are reviewed:

**§3.1.1 Doppler imaging**: Line-of-sight velocity. Oldest method (Hart 1954). Gives horizontal component near the limb and vertical near disc centre. SOHO/MDI, SDO/HMI, Hinode are the workhorses.

**§3.1.2 Correlation and structure tracking**:
- **LCT (local correlation tracking)**: maximises image correlation between sub-images (November & Simon 1988).
- **CST (coherent structure tracking)**: segmentation + feature displacement (Roudier et al. 1999; Rieutord et al. 2007).
- **BT (ball tracking)**: floating-ball displacement on intensity surface (Potts et al. 2004).
- Resolution limit: ~2.5 Mm (below this, random granule motion swamps the signal — Rieutord et al. 2001, 2010).

**§3.1.3 Local helioseismology**: f-modes, time-distance, ring-diagram analyses. Now central for subsurface supergranulation dynamics.

**§3.1.4 Power spectra**: Fig. 4 of the paper presents: (a) line-of-sight Doppler spectra from SOHO/MDI and SDO/HMI (Williams & Pensell 2011), (b) full 3-component spectra combining CST + Doppler (Rincon et al. 2017), (c) subsurface ring-diagram spectra (Greer et al. 2015).

**§3.2 Scales and structure**:
- **Horizontal scale**: peak at $\ell = 120$-140 spherical harmonic degrees (corresponding to ~30-35 Mm), typical full width 20-75 Mm. SOHO/MDI gives ~10% larger estimates than SDO/HMI due to lower resolution (Williams et al. 2014). Leighton et al. (1962) auto-correlation: 32 Mm; Del Moro et al. (2004) mean 27 Mm; Hirzberger et al. (2008) mean ~30 Mm (>10⁵ supergranules analyzed).
- **Horizontal spectra**: spheroidal (diverging/converging) component dominates at $\ell \sim 120$; toroidal component is much weaker at that scale but rises toward smaller scales. This is a KEY 2018-era finding.
- **Lifetime**: Worden & Simon (1976) found 36 h; Wang & Zirin (1989) showed strong tracer dependence (20 h from Dopplergrams, 2 days from magnetic structures, 10 h from tracking). Parfinenko et al. (2014) wavelet analysis of MDI: 1.3 days. Hirzberger et al. (2008): 1.6-1.8 days via helioseismology. Greer et al. (2016) SDO/HMI: coherence only in first few Mm, but pattern propagates to NSSL base over ~1 month at ~40 m/s.
- **Velocities**: Horizontal 300-400 m/s (Hart 1954: 170 m/s; Simon & Leighton 1964: 300 m/s; Hathaway et al. 2002: 360 m/s). Spectral estimate: at $\lambda = 36$ Mm, $E_h \sim 500$ km$^3$/s$^2$ → $V_\lambda = \sqrt{k E_h(k)} \simeq 300$ m/s. Vertical: 20-30 m/s at supergranulation scale (Hathaway et al. 2002: 30 m/s; Duvall & Birch 2010: 4 m/s rms with 10 m/s upflows; Rincon et al. 2017: 20-30 m/s). Vertical is ≥10× smaller than horizontal — strong anisotropy.
- **Depth**: Helioseismology gives scale heights of 2-7 Mm near the surface. Anelastic $\partial_z v_z = -v_z \partial_z \ln\rho - \nabla_h \cdot \mathbf{v}_h$ gives $v_h/\lambda_h \sim v_z/\lambda_z$. Rincon et al. (2017): $\lambda_z \sim 2.5$ Mm near surface. Duvall et al. (2014), Duvall & Hanasoge (2013) claim high-speed flows to 2.3 Mm depth (240 m/s vertical, 700 m/s horizontal at 1.6 Mm — contested by DeGrave & Jackiewicz 2015, Greer et al. 2015). Greer et al. (2016): instantaneous correlation depth ~7 Mm; pattern "rains down" to NSSL base over ~1 month.

**§3.3 Intensity variations**: Contrast is very small. Goldbaum et al. (2009), Meunier et al. (2008) find centre-edge temperature drop 0.8-2.8 K. Langfellner et al. (2016) with SDO/HMI 617 nm continuum: $\Delta T = 1.1 \pm 0.1$ K. These small contrasts are *consistent with* convective origin (photospheric opacity complexities — Nordlund et al. 2009) but don't prove it.

**§3.4 Effects of rotation**: Rossby number
$$Ro = \frac{V}{2\Omega L} = (2\Omega \tau)^{-1}$$
With $\tau_{SG} \sim 1.7$ days and 27-day rotation: $Ro_{SG} \sim 2$-3 — moderate rotational influence.
- **Anticyclonic vorticity**: Gizon & Duvall (2003) showed correlation between horizontal divergence and vertical vorticity changes sign at the equator — negative in N, positive in S. Supergranules act as weak anticyclones (vertical vorticity of anticyclones changes sign at the equator). Surrounded by cyclonic vorticity at downdrafts. Confirmed by Langfellner et al. (2015b) via time-distance helioseismology and LCT on SDO/HMI: vertical component ~10 m/s in diverging cores, much weaker than horizontal divergence.
- **Superrotation**: Duvall (1980), Snodgrass & Ulrich (1990): supergranulation pattern rotates ~4% faster than plasma. Duvall & Gizon (2000), Beck & Schou (2000) confirmed via f-mode time-distance. Gizon et al. (2003): wave-like power spectrum with 6-9 day period — suggesting oscillatory convection. Disputed by Rast et al. (2004), Lisle et al. (2004) who explain via two superimposed steady mesogranulation + supergranulation flows.

**§3.5 Magnetic effects**:
- **§3.5.1** Supergranulation and magnetic network: Network strongly correlated with supergranular boundaries (Simon & Leighton 1964).
- **§3.5.2** Internetwork fields: weaker, more uniformly distributed.
- **§3.5.3** Magnetic power spectrum of the quiet photosphere.
- **§3.5.4** Supergranulation variations over the solar cycle.
- **§3.5.5** Supergranulation and active region flows.

**KR**: 이것이 2018 업데이트의 핵심입니다. 세 가지 유동 측정 기법이 검토됩니다:

**§3.1.1 Doppler 이미징**: 시선 속도. 가장 오래된 방법(Hart 1954). 림 근처에서 수평 성분, 디스크 중심에서 수직 성분 제공. SOHO/MDI, SDO/HMI, Hinode가 주력.

**§3.1.2 상관 및 구조 추적**:
- **LCT**: 하위 이미지 간 상관 최대화(November & Simon 1988).
- **CST**: 분할 + 특성 변위(Roudier et al. 1999).
- **BT**: 복사강도 표면 위 부유 볼 변위(Potts et al. 2004).
- 해상도 한계: ~2.5 Mm(이하에서는 무작위 입상 운동이 신호를 매몰).

**§3.1.3 국소 일식학**: f-모드, 시간-거리, 링 다이어그램. 이제 지하 초립상 동역학에 핵심.

**§3.1.4 파워 스펙트럼**: 논문 Fig. 4: (a) SOHO/MDI와 SDO/HMI Doppler 스펙트럼, (b) CST + Doppler 3성분 완전 스펙트럼(Rincon et al. 2017), (c) 링 다이어그램 지하 스펙트럼(Greer et al. 2015).

**§3.2 규모와 구조**:
- **수평 규모**: $\ell = 120$-140 구면조화 차수 피크(~30-35 Mm), 전형적 폭 20-75 Mm. SOHO/MDI는 낮은 해상도로 인해 ~10% 큰 추정값(Williams et al. 2014). Leighton et al.(1962) 자기상관: 32 Mm; Del Moro et al.(2004) 평균 27 Mm; Hirzberger et al.(2008): 평균 ~30 Mm(10⁵개 이상 초립상 분석).
- **수평 스펙트럼**: 구면조화적(발산/수렴) 성분이 $\ell \sim 120$에서 지배; 토로이달 성분은 그 규모에서 훨씬 약하지만 소규모로 갈수록 증가. 이것이 2018년대 핵심 발견.
- **수명**: Worden & Simon(1976): 36시간; Wang & Zirin(1989): 추적자 의존(Dopplergram 20시간, 자기 구조 2일, 추적 10시간). Parfinenko et al.(2014): 1.3일. Hirzberger et al.(2008): 1.6-1.8일. Greer et al.(2016) SDO/HMI: 첫 수 Mm에서만 결맞음, 그러나 패턴이 NSSL 바닥까지 ~1개월 동안 ~40 m/s로 전파.
- **속도**: 수평 300-400 m/s. 스펙트럼 추정: $\lambda = 36$ Mm에서 $E_h \sim 500$ km$^3$/s$^2$ → $V_\lambda \simeq 300$ m/s. 수직: 초립상 규모에서 20-30 m/s. 수직 성분은 수평의 ≥10배 작음 — 강한 이방성.
- **깊이**: 일식학으로 표면 근처 규모 높이 2-7 Mm. 비탄성 $\partial_z v_z = -v_z \partial_z \ln\rho - \nabla_h \cdot \mathbf{v}_h$으로 $v_h/\lambda_h \sim v_z/\lambda_z$. Rincon et al.(2017): 표면 근처 $\lambda_z \sim 2.5$ Mm. Greer et al.(2016): 순간 상관 깊이 ~7 Mm, 패턴이 NSSL 바닥까지 ~1개월 동안 "비처럼 내림".

**§3.3 복사강도 변화**: 대비가 매우 작음. Goldbaum et al.(2009), Meunier et al.(2008): 중심-가장자리 온도차 0.8-2.8 K. Langfellner et al.(2016) SDO/HMI 617 nm: $\Delta T = 1.1 \pm 0.1$ K. 대류 기원과 *일치*하지만 결정적 증거는 아님.

**§3.4 자전 효과**: Rossby 수 $Ro_{SG} \sim 2$-3 — 중간 정도 회전 영향.
- **역회전성 와도**: Gizon & Duvall(2003) — 수평 발산과 수직 와도의 상관이 적도에서 부호 바뀜(N에서 음, S에서 양). 초립상이 약한 역회전성 (anticyclone)으로 작용. Langfellner et al.(2015b)이 확인: 발산 중심에서 수직 성분 ~10 m/s, 수평 발산보다 훨씬 약함.
- **초자전**: 초립상 패턴이 플라즈마보다 ~4% 빠르게 자전. Gizon et al.(2003): 6-9일 주기 파형 파워 스펙트럼 — 진동 대류 시사. Rast et al.(2004)은 정상 유동 중첩으로 설명.

**§3.5 자기 효과**:
- **§3.5.1** 초립상과 자기 네트워크: 네트워크와 초립상 경계의 강한 상관.
- **§3.5.2** 네트워크 간(internetwork) 자기장: 약함, 균등 분포.
- **§3.5.3** 조용한 광구의 자기 파워 스펙트럼.
- **§3.5.4** 태양 주기에 따른 초립상 변화.
- **§3.5.5** 초립상과 활동 영역 유동.

### Part III: Classical Fluid Theory and Phenomenological Models (§4, pp. 28-38) / 고전 유체 이론과 현상학적 모델

**EN**: §4.2 reviews rotating, MHD Rayleigh-Bénard convection:
- **Formulation**: Boussinesq/anelastic equations with temperature, velocity, and magnetic field; dimensionless parameters Ra (Rayleigh), Pr (Prandtl), Pm (magnetic Prandtl), Ta (Taylor), Ch (Chandrasekhar).
- **Linear theory**: critical Rayleigh number $Ra_c$, onset wavenumber $k_c$. Rotation ($Ta \to \infty$): $Ra_c \propto Ta^{2/3}$, onset at small horizontal scales — NOT supergranular.
- **Turbulent renormalization**: effective viscosity/diffusivity are much larger than molecular values; some phenomenologies (Lord et al. 2014) apply renormalization to predict a supergranulation-scale bump.

**§4.3 Laminar convection theories of supergranulation**:
- **§4.3.1 Multiple-mode convection**: Simon & Weiss (1968) suggested supergranulation as a "preferred" large cell mode. Observationally not supported by modern spectra.
- **§4.3.2 Temperature boundary conditions**: Effect of imposed vs. fixed-flux BCs on cell size.
- **§4.3.3 Oscillatory convection**: Gizon et al. (2003) wave interpretation — requires rotation + density stratification.
- **§4.3.4 Convection, rotation and shear**: Shear stretching modifies cell aspect ratios.
- **§4.3.5 Convection and magnetic fields**: Magnetic flux bundles stabilize cell boundaries.
- **§4.3.6 Dissipative effects**: Increased effective viscosity near supergranulation-scale network.

**§4.4 Large-scale instabilities, inverse cascades and collective interactions**:
- **§4.4.1 Rip currents and large-scale instabilities**: Rast's (2003b) idea — collective instability of downdraft plumes at granular scale generates larger-scale horizontal "rip current" flows.
- **§4.4.2 Plume and fountain interactions**: Downdraft plumes (formed in the surface thermal boundary) expand self-similarly deeper, merging into larger-scale structures. Possibly related to Rieutord & Zahn (1995) plume model — cold low-entropy plumes sinking and entraining fluid.

**KR**: §4.2는 회전 MHD Rayleigh-Bénard 대류를 검토합니다:
- **정식화**: 온도, 속도, 자기장의 Boussinesq/비탄성 방정식; 무차원 수 Ra, Pr, Pm, Ta, Ch.
- **선형 이론**: 임계 Rayleigh 수 $Ra_c$, 개시 파수 $k_c$. 회전 ($Ta \to \infty$): $Ra_c \propto Ta^{2/3}$, 작은 수평 규모에서 개시 — 초립상이 아님.
- **난류 재규격화**: 유효 점성/확산이 분자값보다 훨씬 큼; Lord et al.(2014) 같은 현상학은 재규격화를 적용해 초립상 규모 범프 예측.

**§4.3 초립상의 층류 대류 이론**:
- **§4.3.1 다중 모드 대류**: Simon & Weiss(1968)의 "선호되는" 대형 셀 모드 제안. 현대 스펙트럼으로 관측 지지 안 됨.
- **§4.3.2 온도 경계 조건**: 강제 대 고정 플럭스 BC가 셀 크기에 미치는 영향.
- **§4.3.3 진동 대류**: Gizon et al.(2003) 파 해석 — 회전 + 밀도 성층 필요.
- **§4.3.4 대류, 회전, 전단**: 전단 늘임이 셀 종횡비 변화.
- **§4.3.5 대류와 자기장**: 자기 플럭스 다발이 셀 경계 안정화.
- **§4.3.6 소산 효과**: 초립상 규모 네트워크 근처의 증가된 유효 점성.

**§4.4 대규모 불안정성, 역 캐스케이드, 집합적 상호작용**:
- **§4.4.1 Rip currents와 대규모 불안정성**: Rast(2003b) — 입상 규모 하강류 플룸의 집합적 불안정이 더 큰 규모의 수평 "rip current" 유동 생성.
- **§4.4.2 플룸과 분수 상호작용**: 표면 열 경계에서 형성된 하강류 플룸이 자기유사적으로 확장, 더 큰 구조로 병합. Rieutord & Zahn(1995) 플룸 모델과 관련 가능.

### Part IV: Numerical Modelling (§5, pp. 38-54) / 수치 모델링

**EN**: The other major 2018 update. The SCZ has $Re \sim 10^{10}$-$10^{13}$ while simulations are limited to $Re \lesssim 10^4$-$10^5$ — a severe regime gap.

**§5.1 Introduction to convection simulations**:
- **§5.1.1 General potential and limitations**: Simulations cannot reach true SCZ turbulence but capture essential dynamics.
- **§5.1.2 Turbulent RB vs. Navier-Stokes turbulence**: Different regime characterizations.
- **§5.1.3 Solar convection models**: ANTARES, MURaM, Stagger, CO5BOLD, ASH, PENCIL, etc.

**§5.2 Small-scale simulations**:
- **§5.2.1 Turbulent Rayleigh-Bénard**: Canonical setup, confirms buoyancy injection phenomenology.
- **§5.2.2 Stratified convection simulations at granulation scales**: MURaM, Stagger reproduce granular patterns at ~15% intensity contrast with realistic radiative transfer.

**§5.3 Large-scale simulations**:
- **§5.3.1 Global vs. local simulations**: Tradeoffs — global spherical captures geometry/rotation; local Cartesian gives higher resolution.
- **§5.3.2 Global spherical simulations**: ASH, PENCIL, etc. Captures rotation. Featherstone & Hindman (2016) key result: spectrum strongly influenced by rotation.
- **§5.3.3 Large-scale turbulent convection in local Cartesian simulations**: Lord et al. (2014), Cossette & Rast (2016), Rincon et al. (2017). These reproduce a supergranulation-scale spectral peak when run at sufficiently large Ra and with realistic stratification. Rincon et al. (2017) supports anisotropic Bolgiano-Obukhov phenomenology at $kH \ll 1$.
- **§5.3.4 State-of-the-art local hydrodynamic Cartesian simulations**: Entropy jump thickness and magnitude at the surface are critical in setting the large-scale spectrum.
- **§5.3.5 Simulations with rotation**: Featherstone & Hindman (2016), Featherstone & Miesch — rotation shifts spectral peak.
- **§5.3.6 MHD simulations**: Ustyugov (2009), Stein et al. (2011), Karak et al. (2018). Low-Pm regime is numerically very challenging.

**KR**: 또 다른 주요 2018 업데이트. SCZ는 $Re \sim 10^{10}$-$10^{13}$인 반면 시뮬레이션은 $Re \lesssim 10^4$-$10^5$로 제한됨 — 심각한 영역 격차.

**§5.1 대류 시뮬레이션 서론**:
- **§5.1.1 잠재력과 한계**: 시뮬레이션이 실제 SCZ 난류에는 도달 못하지만 본질적 동역학은 포착.
- **§5.1.2 난류 RB 대 Navier-Stokes 난류**: 서로 다른 영역 특성화.
- **§5.1.3 태양 대류 모델**: ANTARES, MURaM, Stagger, CO5BOLD, ASH, PENCIL 등.

**§5.2 소규모 시뮬레이션**:
- **§5.2.1 난류 Rayleigh-Bénard**: 표준 셋업, 부력 주입 현상학 확인.
- **§5.2.2 입상 규모 성층 대류**: MURaM, Stagger가 현실적 복사 전달로 ~15% 복사강도 대비의 입상 패턴 재현.

**§5.3 대규모 시뮬레이션**:
- **§5.3.1 전역 대 국소 시뮬레이션**: 트레이드오프 — 전역 구면은 기하/회전 포착; 국소 직교는 고해상도.
- **§5.3.2 전역 구면 시뮬레이션**: ASH, PENCIL. Featherstone & Hindman(2016) 핵심 결과: 스펙트럼이 회전에 강하게 영향.
- **§5.3.3 국소 직교 시뮬레이션에서의 대규모 난류 대류**: Lord et al.(2014), Cossette & Rast(2016), Rincon et al.(2017). 충분히 큰 Ra와 현실적 성층으로 실행 시 초립상 규모 스펙트럼 피크 재현. Rincon et al.(2017)은 $kH \ll 1$에서 이방성 Bolgiano-Obukhov 현상학 지지.
- **§5.3.4 최신 국소 유체역학 직교 시뮬레이션**: 표면 엔트로피 점프 두께와 크기가 대규모 스펙트럼 결정에 중요.
- **§5.3.5 회전 포함 시뮬레이션**: Featherstone & Hindman(2016) — 회전이 스펙트럼 피크 이동.
- **§5.3.6 MHD 시뮬레이션**: Ustyugov(2009), Stein et al.(2011), Karak et al.(2018). 낮은 Pm 영역은 수치적으로 매우 어려움.

### Part V: Discussion and Outlook (§6, pp. 54-58) / 논의와 전망

**EN**: §6.1 Summary of observations — horizontal scale 20-70 Mm with peak at 36 Mm; strong anisotropy (horizontal 300-400 m/s >> vertical 20-30 m/s); weak temperature contrast (<3 K); rotational influence via anticyclones; magnetic network coupling.

§6.2 Physics and dynamical phenomenology — 10-15 years of progress. Strong emerging case that supergranulation is buoyancy-driven. Rincon et al. (2017) argue for a generalized Bolgiano-Obukhov phenomenology at $kH \ll 1$ based on three assumptions: (i) dominant balance between buoyancy and inertia in momentum equation; (ii) constant spectral flux of thermal variance in a nearly adiabatic well-mixed layer; (iii) "frustrated" vertical scale of variation independent of horizontal scale, of order $H$. This predicts horizontal KE spectrum that continues to rise at scales larger than granulation, with vertical KE decreasing — consistent with observations.

Three remaining credible scenarios for what sets the 30 Mm scale:
1. **Internal thermodynamic structure + entropy jump** at surface (Cossette & Rast 2016; Rincon et al. 2017).
2. **Interaction between slow large-scale convection and rotation** (Featherstone & Hindman 2016).
3. **Dynamical interactions between convection and magnetic fields** (Ustyugov 2009; Stein et al. 2011).

§6.3 Outlook — future needs: better helioseismic characterization; less controversial subsurface flow determinations; higher-Re simulations; proper treatment of low-Pm MHD; inclusion of realistic radiative transfer and rotation. Paper concludes with Fig. 18 showing global and local finite-time Lyapunov exponent (FTLE) maps revealing supergranular cell boundaries as the "skeleton" of the flow.

**KR**: §6.1 관측 요약 — 수평 규모 20-70 Mm, 피크 36 Mm; 강한 이방성(수평 300-400 m/s >> 수직 20-30 m/s); 약한 온도 대비(<3 K); 역회전성을 통한 회전 영향; 자기 네트워크 결합.

§6.2 물리와 동역학 현상학 — 10-15년의 진보. 초립상이 부력 구동이라는 강한 증거. Rincon et al.(2017)은 $kH \ll 1$에서의 일반화된 Bolgiano-Obukhov 현상학을 주장: (i) 운동량 방정식에서 부력과 관성의 지배적 균형; (ii) 거의 단열인 혼합층에서 열 분산의 일정 스펙트럼 플럭스; (iii) 수평 규모와 독립적인 $H$ 크기의 "좌절된" 수직 변동 규모. 이로써 입상보다 큰 규모에서 수평 KE 스펙트럼 증가, 수직 KE 감소 예측 — 관측과 일치.

30 Mm 규모를 결정하는 세 가지 남은 신뢰성 있는 시나리오:
1. **내부 열역학 구조 + 표면 엔트로피 점프**(Cossette & Rast 2016; Rincon et al. 2017).
2. **느린 대규모 대류와 회전의 상호작용**(Featherstone & Hindman 2016).
3. **대류와 자기장의 동역학적 상호작용**(Ustyugov 2009; Stein et al. 2011).

§6.3 전망 — 미래 필요: 더 나은 일식학적 특성화; 덜 논란이 되는 지하 유동 결정; 고-Re 시뮬레이션; 낮은 Pm MHD의 적절한 처리; 현실적 복사 전달과 회전 포함. Fig. 18은 초립상 셀 경계를 유동의 "골격"으로 드러내는 전역 및 국소 유한-시간 Lyapunov 지수(FTLE) 맵.

---

## 3. Key Takeaways / 핵심 시사점

1. **Supergranulation = 30-35 Mm cellular flow with 24-48 h lifetime / 초립상은 수명 24-48시간의 30-35 Mm 셀 유동**
   - **EN**: Since Hart (1956), the scale has been robust at ~30 Mm. SDO/HMI full-disk data (Williams et al. 2014; Hathaway et al. 2015) confirmed the peak at spherical harmonic degree $\ell = 120$-140 with no secondary mesogranulation bump. Lifetime estimates cluster around 1-2 days depending on tracer.
   - **KR**: Hart(1956) 이래로 규모는 ~30 Mm에서 견고합니다. SDO/HMI 전면 디스크 데이터는 구면조화 차수 $\ell = 120$-140에서 피크를 확인했으며, 이차적인 중간립상 범프는 없었습니다. 수명 추정치는 추적자에 따라 1-2일에 집중됩니다.

2. **Extreme flow anisotropy: horizontal (300-400 m/s) >> vertical (20-30 m/s) / 극한의 유동 이방성**
   - **EN**: The horizontal rms velocity is ≥10× larger than the vertical. Even more importantly, the spheroidal (divergent) horizontal component dominates the toroidal (vortical) at supergranulation scale, while the vertical spectrum shows a KINK (not a peak) at $\ell \sim 120$. This strong anisotropy constrains any successful theory.
   - **KR**: 수평 rms 속도가 수직의 ≥10배입니다. 더 중요한 것은 구면조화적(발산) 수평 성분이 초립상 규모에서 토로이달(와동) 성분을 지배하는 반면, 수직 스펙트럼은 $\ell \sim 120$에서 피크가 아닌 꺾임을 보입니다. 이 강한 이방성은 어떤 성공적 이론에도 제약을 가합니다.

3. **No physical meso-granulation bump / 물리적 중간립상 범프는 없음**
   - **EN**: The 2018 review formally abandons "mesogranulation" as a distinct phenomenon. Hathaway et al. (2015), Rincon et al. (2017) show a smooth spectrum in the 2-30 Mm range. The term "mesoscale" is retained as a descriptor without physical status.
   - **KR**: 2018 리뷰는 뚜렷한 현상으로서 "중간립상"을 공식적으로 폐기합니다. Hathaway et al.(2015), Rincon et al.(2017)은 2-30 Mm 범위에서 매끄러운 스펙트럼을 보입니다. "중간 규모"는 물리적 지위 없이 기술어로만 유지됩니다.

4. **Moderate rotational influence: $Ro_{SG} \sim 2$-3 / 중간 정도 회전 영향**
   - **EN**: Supergranules behave as weak anticyclones (Gizon & Duvall 2003). Vertical vorticity changes sign across the equator. Pattern superrotates ~4% faster than plasma. This shows rotation matters for supergranulation organization but does not dominate.
   - **KR**: 초립상은 약한 역회전성 (anticyclone) 으로 행동합니다(Gizon & Duvall 2003). 수직 와도가 적도를 가로질러 부호를 바꿉니다. 패턴이 플라즈마보다 ~4% 빠르게 초자전합니다. 회전이 초립상 조직에 중요하지만 지배적이지는 않음을 보여줍니다.

5. **Buoyancy-driven convective origin increasingly supported / 부력 구동 대류 기원 점점 지지됨**
   - **EN**: Recent large-scale simulations (Lord et al. 2014; Cossette & Rast 2016; Rincon et al. 2017) reproduce a supergranulation-scale spectral peak when driven purely by buoyancy from the surface thermal boundary layer. The 2018 review concludes supergranulation is "the energetically dominant convection scale on the large-scale side of the photospheric convection spectrum".
   - **KR**: 최근 대규모 시뮬레이션(Lord et al. 2014; Cossette & Rast 2016; Rincon et al. 2017)이 표면 열 경계층으로부터의 순수 부력 구동만으로 초립상 규모 스펙트럼 피크를 재현합니다. 2018 리뷰는 초립상이 "광구 대류 스펙트럼의 대규모 측에서 에너지적으로 지배적인 대류 규모"라고 결론짓습니다.

6. **The 30 Mm scale remains theoretically unexplained / 30 Mm 규모는 이론적으로 미해결**
   - **EN**: Despite observational consolidation, **no predictive theory** exists for *why* 30 Mm. Three candidate mechanisms: entropy jump structure, rotation-convection interaction, magnetic-convection feedback. A generalized anisotropic Bolgiano-Obukhov phenomenology (Rincon et al. 2017) provides a partial framework but not a full prediction of scale.
   - **KR**: 관측적 공고화에도 불구하고 *왜* 30 Mm인지에 대한 **예측 이론은 없습니다**. 세 후보 메커니즘: 엔트로피 점프 구조, 회전-대류 상호작용, 자기-대류 피드백. 일반화된 이방성 Bolgiano-Obukhov 현상학(Rincon et al. 2017)이 부분적 틀을 제공하지만 규모의 완전한 예측은 아닙니다.

7. **Deep vertical extent is controversial / 깊은 수직 확장은 논란**
   - **EN**: Duvall & Hanasoge (2013), Duvall et al. (2014) claimed supergranulation flows to ~2 Mm depth with horizontal flows of 700 m/s at 1.6 Mm. Subsequent analyses (DeGrave & Jackiewicz 2015; Greer et al. 2015) challenge these. Greer et al. (2016) find instantaneous correlation depth ~7 Mm but downward propagation to NSSL base (~30-50 Mm) over ~1 month at ~40 m/s. The vertical scale height near the surface is 2-7 Mm depending on method.
   - **KR**: Duvall & Hanasoge(2013), Duvall et al.(2014)은 ~2 Mm 깊이까지 초립상 유동과 1.6 Mm에서 700 m/s 수평 유동을 주장했습니다. 후속 분석(DeGrave & Jackiewicz 2015; Greer et al. 2015)이 이에 도전합니다. Greer et al.(2016)은 순간 상관 깊이 ~7 Mm와 ~40 m/s로 ~1개월에 걸친 NSSL 바닥까지의 하향 전파를 찾아냈습니다. 표면 근처 수직 규모 높이는 방법에 따라 2-7 Mm입니다.

8. **SDO/HMI + petascale simulations are the 2010-2018 game-changers / SDO/HMI + 페타스케일 시뮬레이션이 2010-2018 판도 전환**
   - **EN**: The main reason for the update is the dramatic improvement in data (SDO/HMI continuous long-baseline full-disk, Hinode high-resolution) and computing (large-Ra stratified simulations with realistic boundary conditions). These tools — absent or primitive in 2010 — are what allow the current convergence on the buoyancy-driven interpretation.
   - **KR**: 업데이트의 주 이유는 데이터(SDO/HMI 연속 장기 전면 디스크, Hinode 고해상도)와 컴퓨팅(현실적 경계 조건의 고-Ra 성층 시뮬레이션)의 극적 개선입니다. 이러한 도구 — 2010년에 없거나 원시적이었던 — 가 부력 구동 해석으로의 현재 수렴을 가능하게 했습니다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Rossby number (rotation parameter) / Rossby 수

$$\boxed{Ro = \frac{V}{2\Omega L} = (2\Omega \tau)^{-1}}$$

- $V$: characteristic velocity (m/s)
- $\Omega$: angular rotation rate (rad/s)
- $L$: length scale (m)
- $\tau$: timescale ($= L/V$)

**EN**: For the Sun, $\Omega \approx 2.87 \times 10^{-6}$ rad/s (27-day rotation). With $\tau_{SG} \sim 1.7$ days $\approx 1.47 \times 10^5$ s: $Ro_{SG} = (2 \times 2.87 \times 10^{-6} \times 1.47 \times 10^5)^{-1} \approx 1.18$. The paper quotes $Ro_{SG} \sim 2$-3 using slightly different conventions. **Interpretation**: $Ro \sim 1$ means Coriolis force and inertia are comparable — rotation plays a role but does not dominate (unlike $Ro \ll 1$ geostrophic regime).

**KR**: 태양의 경우 $\Omega \approx 2.87 \times 10^{-6}$ rad/s(27일 자전). $\tau_{SG} \sim 1.7$일 ≈ $1.47 \times 10^5$ s: $Ro_{SG} \approx 1.18$. 논문은 약간 다른 관례로 $Ro_{SG} \sim 2$-3을 제시. **해석**: $Ro \sim 1$은 Coriolis 힘과 관성이 비슷함을 의미 — 회전이 역할을 하지만 지배하지 않음($Ro \ll 1$ 지균 체계와 달리).

### 4.2 Anelastic mass conservation / 비탄성 질량 보존

$$\boxed{\partial_z v_z = -v_z \,\partial_z \ln\rho - \nabla_h \cdot \mathbf{v}_h}$$

- $v_z$: vertical velocity
- $\rho(z)$: background density
- $\mathbf{v}_h$: horizontal velocity field
- $\nabla_h$: horizontal divergence operator

**EN**: For incompressible flow $\nabla \cdot \mathbf{v} = 0$, this reduces to $\partial_z v_z = -\nabla_h \cdot \mathbf{v}_h$. For stratified flow, the $\partial_z \ln\rho = -1/H_\rho$ term adds a contribution. Dimensionally, if $v_h \sim V_h$ over scale $\lambda_h$, then $\partial_z v_z \sim V_h / \lambda_h$, giving a vertical scale-height estimate $\lambda_z \sim v_z / (V_h/\lambda_h) = (v_z/v_h) \lambda_h$. Used by Rincon et al. (2017) to derive $\lambda_z \sim 2.5$ Mm for supergranulation.

**KR**: 비압축 유동 $\nabla \cdot \mathbf{v} = 0$에서는 $\partial_z v_z = -\nabla_h \cdot \mathbf{v}_h$로 환원됩니다. 성층 유동에서는 $\partial_z \ln\rho = -1/H_\rho$ 항이 기여를 추가합니다. 차원적으로, $v_h \sim V_h$가 규모 $\lambda_h$에서라면 $\partial_z v_z \sim V_h / \lambda_h$, 수직 규모 높이 추정 $\lambda_z \sim (v_z/v_h) \lambda_h$. Rincon et al.(2017)이 초립상에서 $\lambda_z \sim 2.5$ Mm 유도에 사용.

### 4.3 Kinetic energy spectrum and velocity / 운동 에너지 스펙트럼과 속도

$$\boxed{V_\lambda = \sqrt{k \, E_h(k)}, \quad k = \frac{2\pi}{\lambda}}$$

- $E_h(k)$: horizontal KE spectral density (m$^3$/s$^2$ or km$^3$/s$^2$)
- $V_\lambda$: characteristic velocity at scale $\lambda$
- $k$: horizontal wavenumber (1/m)

**EN**: Extra factor of $k$ comes because $E(k)$ has units of $v^2 / k$. Worked example: at $\lambda = 36$ Mm, $k = 2\pi / (3.6 \times 10^7 \text{ m}) \approx 1.75 \times 10^{-7}$ m$^{-1}$. With $E_h \approx 500$ km$^3$/s$^2$ = $5 \times 10^{11}$ m$^3$/s$^2$: $V = \sqrt{1.75 \times 10^{-7} \times 5 \times 10^{11}} \approx \sqrt{8.73 \times 10^4} \approx 295$ m/s ≈ 300 m/s — matches direct Doppler measurements.

**KR**: $E(k)$가 $v^2/k$ 단위이므로 추가 $k$ 인자가 나옵니다. 계산 예시: $\lambda = 36$ Mm에서 $k \approx 1.75 \times 10^{-7}$ m$^{-1}$. $E_h = 5 \times 10^{11}$ m$^3$/s$^2$: $V \approx 295$ m/s ≈ 300 m/s — 직접 Doppler 측정과 일치.

### 4.4 Reynolds number and dissipation scales / Reynolds 수와 소산 규모

$$\boxed{Re = \frac{LV}{\nu}, \quad \ell_\nu \sim Re^{-3/4} L, \quad \ell_\eta \sim Pm^{-3/4} \ell_\nu, \quad Pm = \frac{\nu}{\eta}}$$

- $\nu$: kinematic viscosity (m²/s)
- $\eta$: magnetic diffusivity (m²/s)
- $Pm$: magnetic Prandtl number

**EN**: Near the solar surface: $\nu \sim 10^{-3}$ m²/s, $L \sim L_G \sim 1$ Mm, $V \sim 1$ km/s → $Re \sim 10^{12}$, so $\ell_\nu \sim 10^{-3}$ m (millimeters). Magnetic: $\eta \sim 10^2$ m²/s, $Pm \sim 10^{-5}$, so $\ell_\eta \sim Pm^{-3/4} \ell_\nu \sim (10^{-5})^{-3/4} \times 10^{-3} \approx 10^{3.75} \times 10^{-3} \approx 5.6$ m. Thermal: $Pr \sim 10^{-4}$-$10^{-6}$ in deep SCZ, so $\ell_\kappa \sim 500$ m at depth. This extreme scale hierarchy explains why no Earth-based simulation can match the Sun's true turbulent state.

**KR**: 태양 표면 근처: $\nu \sim 10^{-3}$ m²/s, $L \sim 1$ Mm, $V \sim 1$ km/s → $Re \sim 10^{12}$, $\ell_\nu \sim 10^{-3}$ m(밀리미터). 자기: $\eta \sim 10^2$ m²/s, $Pm \sim 10^{-5}$, $\ell_\eta \sim 5.6$ m. 열: 깊은 SCZ에서 $Pr \sim 10^{-4}$-$10^{-6}$, $\ell_\kappa \sim 500$ m. 이 극한의 규모 계층은 어떤 지상 시뮬레이션도 태양의 진정한 난류 상태를 맞출 수 없는 이유를 설명합니다.

### 4.5 Schwarzschild stability criterion / Schwarzschild 안정도 기준

$$\boxed{\frac{ds}{dz} < 0 \iff \left|\frac{dT}{dz}\right|_{\text{actual}} > \left|\frac{dT}{dz}\right|_{\text{ad}} \iff \text{convection}}$$

where adiabatic gradient is
$$\left(\frac{dT}{dz}\right)_{\text{ad}} = -\frac{g}{c_p} \quad (\text{for ideal gas})$$

- $s$: specific entropy
- $g$: gravitational acceleration
- $c_p$: specific heat at constant pressure

**EN**: The SCZ satisfies this criterion: actual temperature drops faster with height than adiabatic, so fluid parcels displaced upward are hotter (less dense) than surroundings and continue rising → instability. In the Sun, the superadiabatic gradient $|ds/dz|$ is very large in a thin layer just below the optical surface (the thermal boundary layer), much smaller below due to efficient convective mixing.

**KR**: SCZ는 이 기준을 만족: 실제 온도가 높이에 따라 단열보다 빠르게 감소하므로 위로 변위된 유체 덩어리가 주변보다 뜨겁고(덜 밀집) 계속 상승 → 불안정. 태양에서 초단열 기울기 $|ds/dz|$는 광학적 표면 바로 아래 얇은 층(열 경계층)에서 매우 크고, 효율적 대류 혼합으로 인해 아래에서는 훨씬 작습니다.

### 4.6 Full scale hierarchy / 전체 규모 계층

$$\boxed{\ell_\nu \ll \ell_\eta \ll \ell_\kappa \sim H_p \sim H_\rho \sim L_B \sim L_G \ll L_{SG} \ll R_\odot}$$

Near surface values:
- $\ell_\nu \sim 10^{-3}$ m (viscous)
- $\ell_\eta \sim 1$-100 m (magnetic)
- $\ell_\kappa \sim 500$ m (thermal, deep) or $\sim 1$ Mm (surface)
- $H_p \sim H_\rho \sim L_B \sim L_G \sim 1$ Mm (pressure/density scale heights, Bolgiano, granulation)
- $L_{SG} \sim 30$ Mm (supergranulation)
- $R_\odot \sim 700$ Mm

**EN**: This ~12 orders of magnitude of scale separation is the core difficulty: numerical simulations resolve ~4 decades at best, so there is always a gap between simulated and true turbulent cascade.

**KR**: 이 ~12 차수의 규모 분리가 핵심 어려움: 수치 시뮬레이션은 기껏해야 ~4 데케이드 해상; 시뮬레이션된 난류 캐스케이드와 실제 사이에는 항상 격차가 존재.

### 4.7 Bolgiano-Obukhov (anisotropic, for stratified convection)

**EN**: Rincon et al. (2017) argue for a regime at $kH \ll 1$ (wavenumber much smaller than inverse scale height) where:
- Horizontal KE spectrum: $E_h(k) \propto k^{-11/5}$ (steeper than Kolmogorov $-5/3$)
- Vertical KE spectrum: $E_z(k) \propto k^{-7/5}$ or similar, decreasing to large scales
- Temperature spectrum: $E_T(k) \propto k^{-7/5}$

The classical isotropic Bolgiano-Obukhov was derived for weakly stratified turbulence. Rincon et al. (2017) generalize it to strong anisotropy ($\lambda_z \ll \lambda_h$). **Assumptions**: (i) buoyancy-inertia balance in momentum; (ii) constant spectral thermal variance flux; (iii) $\lambda_z \sim H$ independent of $\lambda_h$.

**KR**: Rincon et al.(2017)은 $kH \ll 1$ 영역(파수가 역 규모 높이보다 훨씬 작음)에서 다음을 주장:
- 수평 KE 스펙트럼: $E_h(k) \propto k^{-11/5}$ (Kolmogorov $-5/3$보다 가파름)
- 수직 KE 스펙트럼: $E_z(k) \propto k^{-7/5}$ 정도, 대규모로 감소
- 온도 스펙트럼: $E_T(k) \propto k^{-7/5}$

고전적 등방성 Bolgiano-Obukhov는 약한 성층 난류에 대해 유도됨. Rincon et al.(2017)은 이를 강한 이방성($\lambda_z \ll \lambda_h$)으로 일반화. **가정**: (i) 운동량의 부력-관성 균형; (ii) 일정한 스펙트럼 열 분산 플럭스; (iii) $\lambda_h$와 독립적인 $\lambda_z \sim H$.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1915 ────── Plaskett — possible first detection of supergranulation "noise"
   │
1953-1954 ─ Hart (Oxford) — discovery of the noisy velocity field
   │
1956 ────── Hart — first accurate scale (~26 Mm)
   │
1962 ────── Leighton, Noyes & Simon — Doppler imaging establishes supergranulation
   │        [Paper #5 in our series]
1964 ────── Simon & Leighton — link to magnetic network
   │
1968 ────── Simon & Weiss — multiple-mode convection theory
   │
1981 ────── November et al. — proposal of mesogranulation (~8 Mm)
   │
1989 ────── Wang & Zirin — lifetime depends on tracer
   │
1995 ────── Rieutord & Zahn — plume entrainment model
   │
1996 ────── SOHO launch — MDI Doppler era begins
   │
2003 ────── Gizon & Duvall — anticyclonic vorticity, wave-like pattern
2003 ────── Rast — rip current collective instability
   │
2006 ────── Hinode launch — high-resolution SOT imaging
   │
2009 ────── Nordlund, Stein, Asplund — LRSP review on solar surface convection
   │        [Paper #15 in our series]
2010 ────── Rieutord & Rincon — LRSP review (previous version)
   │        [Paper #19 in our series]
2010 ────── SDO launch — HMI/AIA, continuous high-resolution data
   │
2014 ────── Lord et al. — turbulent renormalization prediction of SG peak
2014 ────── Duvall & Hanasoge, Duvall et al. — deep SG flows (contested)
2015 ────── Hathaway et al., Greer et al. — spectra, no mesogranulation bump
2016 ────── Cossette & Rast — surface boundary-driven large-scale convection
2016 ────── Greer et al. — SG pattern rains to NSSL base
2017 ────── Rincon et al. — 3-component SDO/HMI spectra, Bolgiano-Obukhov anisotropic
   │
★ 2018 ─── Rincon & Rieutord — THIS updated LRSP review
   │       [Paper #60 in our series, update of Paper #19]
   │
2020+ ──── DKIST first light — expected next advances
          Exascale MHD simulations — expected
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #19 (Rieutord & Rincon 2010, LRSP)** | Direct predecessor — this 2018 paper is its update | **Critical** — explicit update. ~40 new references; SDO/HMI and new simulations added; author order switched |
| **Paper #5 (Leighton et al. 1962)** | First Doppler-based establishment of supergranulation | Historical foundation — cited throughout for the 30 Mm scale |
| **Paper #15 (Nordlund, Stein & Asplund 2009, LRSP — Solar Surface Convection)** | Reviews granulation; provides the small-scale convection backdrop | Complementary — focuses on granulation (1 Mm) while this paper handles supergranulation (30 Mm) |
| **Paper #2 (Hart 1954)** | Original discovery paper | Historical — the starting point of supergranulation research |
| **Paper #12 (Rincon et al. 2005)** | Early large-scale stratified convection simulations | Cited heavily in §5 numerical modelling; provides the simulation-based motivation for buoyancy-driven SG |
| **Papers on SDO/HMI helioseismology (Greer et al. 2015, 2016)** | Subsurface flow measurements | Directly feeds §3.2.4 depth discussion; supports the "pattern rains down" picture |
| **Paper on solar dynamo (generic)** | SG-network coupling informs small-scale dynamo; magnetic network formation tied to SG boundaries | Broader context — SG may play role in quiet-Sun dynamo |
| **Paper on stellar convection (generic)** | SG may have analogs in other stars; relevant for exoplanet host star noise | Wider applicability — supergranulation as universal feature of stellar convection |

---

## 7. References / 참고문헌

**Primary paper / 본 논문**:
- Rincon, F., & Rieutord, M. (2018). "The Sun's supergranulation". *Living Reviews in Solar Physics*, 15:6. DOI: 10.1007/s41116-018-0013-5

**Predecessor (2010 version, Paper #19)**:
- Rieutord, M., & Rincon, F. (2010). "The Sun's Supergranulation". *Living Reviews in Solar Physics*, 7:2. DOI: 10.12942/lrsp-2010-2

**Historical foundations / 역사적 기초**:
- Hart, A. B. (1954). "Motions in the Sun at the photospheric level. IV. The equatorial rotation and possible velocity fields in the photosphere". *MNRAS*, 114, 17.
- Hart, A. B. (1956). "Motions in the Sun at the photospheric level. VI. Large-scale motions in the equatorial region". *MNRAS*, 116, 38.
- Leighton, R. B., Noyes, R. W., & Simon, G. W. (1962). "Velocity fields in the solar atmosphere. I. Preliminary report". *ApJ*, 135, 474.
- Simon, G. W., & Leighton, R. B. (1964). "Velocity fields in the solar atmosphere. III. Large-scale motions, the chromospheric network, and magnetic fields". *ApJ*, 140, 1120.

**Key modern observations / 핵심 현대 관측**:
- Williams, P. E., & Pesnell, W. D. (2011). "Comparisons of supergranule characteristics during the solar minima of cycles 22/23 and 23/24". *Solar Physics*, 270, 125.
- Williams, P. E., Pesnell, W. D., Beck, J. G., & Lee, S. (2014). *Solar Physics*, 289, 11.
- Hathaway, D. H., Teil, T., Norton, A. A., & Kitiashvili, I. (2015). "The Sun's photospheric convection spectrum". *ApJ*, 811, 105.
- Rincon, F., Roudier, T., Schekochihin, A. A., & Rieutord, M. (2017). "Supergranulation and multiscale flows in the solar photosphere. Global observations vs. a theory of anisotropic turbulent convection". *A&A*, 599, A69.
- Langfellner, J., Birch, A. C., & Gizon, L. (2015b). "Intensity and temperature contrasts across supergranules". *A&A*, 581, A67.
- Langfellner, J., et al. (2016). *A&A*, 596, A66.
- Greer, B. J., Hindman, B. W., Featherstone, N. A., & Toomre, J. (2015). "Helioseismic imaging of fast convective flows throughout the near-surface shear layer". *ApJL*, 803, L17.
- Greer, B. J., Hindman, B. W., & Toomre, J. (2016). *ApJ*, 824, 128.
- Duvall Jr., T. L., & Hanasoge, S. M. (2013). "Subsurface supergranular vertical flows as measured using large distance separations in time-distance helioseismology". *Solar Physics*, 287, 71.
- Duvall Jr., T. L., Hanasoge, S. M., & Chakraborty, S. (2014). *Solar Physics*, 289, 3421.
- DeGrave, K., & Jackiewicz, J. (2015). *Solar Physics*, 290, 1547.

**Rotational effects / 회전 효과**:
- Gizon, L., & Duvall Jr., T. L. (2003). "Supergranulation supports waves". *Nature*, 421, 43.
- Gizon, L., Duvall Jr., T. L., & Schou, J. (2003). "Wave-like properties of solar supergranulation". *Nature*, 421, 43.

**Theory and simulations / 이론과 시뮬레이션**:
- Lord, J. W., Cameron, R. H., Rast, M. P., Rempel, M., & Roudier, T. (2014). "The role of subsurface flows in solar surface convection: modeling the spectrum of supergranular and larger scale flows". *ApJ*, 793, 24.
- Cossette, J.-F., & Rast, M. P. (2016). "Supergranulation as the largest buoyantly driven convective scale of the Sun". *ApJL*, 829, L17.
- Featherstone, N. A., & Hindman, B. W. (2016). "The spectral amplitude of stellar convection and its scaling in the high-Rayleigh-number regime". *ApJ*, 818, 32.
- Rast, M. P. (2003b). "The scales of granulation, mesogranulation, and supergranulation". *ApJ*, 597, 1200.
- Rieutord, M., & Zahn, J.-P. (1995). "Turbulent plumes in stellar convective envelopes". *A&A*, 296, 127.
- Nordlund, Å., Stein, R. F., & Asplund, M. (2009). "Solar Surface Convection". *LRSP*, 6:2.
- Stein, R. F., et al. (2011). "Solar small-scale magnetoconvection". *Solar Physics*, 268, 271.
- Karak, B. B., et al. (2018). "Magnetic field dependence of bipolar magnetic region tilts on the Sun: Observation and model". *ApJ*, 854, 53.

**Phenomenology / 현상학**:
- Bolgiano Jr., R. (1959). "Turbulent spectra in a stably stratified atmosphere". *J. Geophys. Res.*, 64, 2226.
- Obukhov, A. M. (1959). "Effect of Archimedean forces on the structure of the temperature field in a turbulent flow". *Dokl. Akad. Nauk SSSR*, 125, 1246.
- Rincon, F. (2007). "Anisotropy, inhomogeneity and inertial-range scalings in turbulent convection". *J. Fluid Mech.*, 563, 43.
- Schekochihin, A. A., et al. (2007). *New J. Phys.*, 9, 300.

**Reviews used as prerequisites / 선행 필요 리뷰**:
- Miesch, M. S. (2005). "Large-scale dynamics of the convection zone and tachocline". *LRSP*, 2:1.
- Rieutord, M., & Rincon, F. (2010). Paper #19 (predecessor to this paper).
- Stix, M. (2004). *The Sun: An Introduction* (Springer).
