---
title: "Flare-Productive Active Regions"
authors: [Shin Toriumi, Haimin Wang]
year: 2019
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-019-0019-7"
topic: Living_Reviews_in_Solar_Physics
tags: [active-regions, solar-flares, delta-spots, NLFFF, magnetic-helicity, flux-emergence, space-weather, flare-forecasting]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 64. Flare-Productive Active Regions / 플레어 생산성 활동영역

---

## 1. Core Contribution / 핵심 기여

This Living Reviews article by Toriumi & Wang (2019) provides the most comprehensive synthesis to date of how flare-productive active regions (ARs) are born, evolve, and erupt. It weaves together five decades of observation — from Zirin & Liggett's 1987 BBSO δ-spot typology, through Hinode/SDO/GST vector magnetography, to 2018-era machine-learning flare forecasting — with the parallel thread of 3D MHD flux-emergence and data-driven coronal modelling. The review's central narrative is physical: dynamo-generated toroidal flux rises from the tachocline, is sculpted by turbulent convection into kinked, multiply-buoyant, or colliding flux systems, emerges as δ-spots with sheared polarity inversion lines (PILs) carrying large free magnetic energy ($\sim 10^{31}$–$10^{32}$ erg) and helicity, and finally releases that energy through reconnection that drives a rapid, irreversible back-reaction on the photosphere itself.

본 리뷰는 플레어 생산 AR의 탄생·진화·폭발을 통합적으로 설명한 2019년 시점의 정점이다. 1987년 Zirin–Liggett BBSO δ-spot 분류 이후 50년간의 관측, Hinode/SDO/GST 벡터 자력계 데이터, 2018년 시점의 ML 기반 예보까지를 3D MHD flux emergence 시뮬레이션과 data-driven 코로나 모델과 엮어 낸다. 중심 서사는 물리적이다: tachocline에서 발전기로 생성된 toroidal flux가 난류 대류에 의해 꼬임(kink), 다중 부력, 충돌하는 flux system으로 변형·상승하여 δ-sunspot과 shear된 PIL로 광구에 출현하고, 약 $10^{31}$–$10^{32}$ erg의 자유 자기 에너지와 helicity를 축적한 뒤 재결합으로 방출하며, 그 과정에서 광구 자체에 급격한 back-reaction을 일으킨다. 관측·이론·예보를 하나의 통일된 그림으로 꿰었다는 점이 이 논문의 기여이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Active Regions and Solar Flares (Sect. 2, pp. 5–20) / 활동영역과 태양 플레어

**Flux emergence — theory (§2.1.1, p. 7).** Parker (1955) first showed that a horizontal flux tube with internal/external pressure balance
$$ p_e = p_i + \frac{B^2}{8\pi} $$
is buoyant when $T_i=T_e$, giving buoyancy force per unit volume
$$ f_B = (\rho_e - \rho_i)g = \frac{B^2}{8\pi H_p},\qquad H_p = \frac{k_B T}{mg}. $$
In the convection zone the plasma-β $\equiv 8\pi p/B^2 \gg 1$ (≈$10^5$ at $B=10^5$ G at the base), so the rising flux is strongly deformed by convective flows. To stay coherent the tube must be twisted; Abbett et al. (2000) showed that 3D tubes need less twist than the 2D limit. Toriumi & Yokoyama (2010, 2011) introduced "two-step emergence": the tube decelerates and pancakes at $-20$ Mm before resuming rise — seamlessly reproduced in fully-compressible simulations spanning $-40$ to $+50$ Mm.

광구 아래 plasma-β가 매우 크므로(tachocline에서 $\sim 10^5$), 상승하는 flux tube는 대류 흐름에 의해 심하게 변형된다. 꼬임(twist)이 있어야 tube가 분해되지 않고 상승한다. Toriumi–Yokoyama의 two-step emergence는 $-20$ Mm에서 감속·수평 팽창(pancaking)한 뒤 광구로 재상승하는 시나리오다.

**Birth of ARs — observation (§2.1.3, p. 9).** Zwaan (1985) hierarchy: sunspots $\Phi\ge 5\times 10^{20}$ Mx, umbral field 2.9–3.3 kG (up to 4 kG); pores $2.5\times 10^{19}$–$5\times 10^{20}$ Mx at ~2 kG; below $10^{20}$ Mx only ephemeral regions. Small-scale magnetic elements of mixed polarity merge; undulatory (sea-serpent) field lines reconnect in the photosphere producing **Ellerman bombs** (Hα wing bursts) and **UV bursts** (TR lines). After emergence, an **arch filament system (AFS)** appears as parallel dark Hα fibrils.

관측 계층: ephemeral ($<10^{20}$ Mx) → pore → sunspot. 출현 초기에는 sea-serpent 재결합으로 Ellerman bomb과 UV burst가 발생하고, Hα에서는 AFS가 나타난다.

**Solar flares and CMEs (§2.2, p. 13).** GOES class = peak 1–8 Å soft X-ray flux: A/B/C/M/X = $10^{-8}/10^{-7}/10^{-6}/10^{-5}/10^{-4}$ W m$^{-2}$. Carrington flare 1859 ≈ X45 at $5\times 10^{32}$ erg. Standard flare (CSHKP) model: flux rope eruption → current sheet reconnection below → ribbons + cusp + HXR loop-top source + CME. Giant AR 12192 (Oct 2014, 2750 MSH, 6 X-class flares) produced **no CMEs** — decay index $n=-\partial \ln B_h/\partial \ln z$ stayed below the torus-instability critical value $n_c\approx 1.5$, so strong overlying fields confined the eruptions.

GOES class는 1–8 Å SXR 피크 플럭스의 로그 분류. Carrington event는 ≈X45($5\times 10^{32}$ erg). AR 12192는 X-class 6번에도 CME가 한 번도 발생하지 않은 사례로, torus instability 임계값 $n_c\approx 1.5$ 미만이 유지되어 confined flare가 된 것이다.

**Categorizations (§2.3, p. 17).** Mt. Wilson hierarchy: **α** (unipolar) → **β** (simple bipolar) → **βγ** (complex mixed) → **βδ/βγδ** (with δ component). δ-group = umbrae of opposite polarity separated <2° sharing a **common penumbra** (Künzel 1960). Sammis et al. (2000): **all ≥X4 flares occur in ARs of >1000 MSH and βγδ class**. Tian et al. (2002): 68% of 25 most-violent ARs in Cycles 22–23 violate Hale (anti-Hale). Tian et al. (2005a): ~34% of 104 δ-spots violate Hale but follow hemispheric current-helicity rule, showing strong X-class tendency. Counter-example: RGO 14886 (1947), largest spot ever (6132 MSH), was flare-quiet — simple β bipole. Fourth-largest, RGO 14585 (1946, 4279 MSH), was δ-like and great-flare-productive. Morphological/magnetic complexity, not mere size, is the decisive factor.

분류 체계: α(단극) → β(단순 쌍극) → βγ(복잡) → βδ/βγδ(δ 포함). δ-spot은 공통 penumbra 내 반대 극성의 umbra(<2°). ≥X4 플레어는 모두 >1000 MSH βγδ에서 발생한다. 최대 spot RGO 14886(1947)은 단순 β로 플레어가 거의 없었고, 네 번째로 큰 14585(1946)는 δ였고 대규모 플레어를 만들었다. **크기 > 복잡성**이 아니라 **복잡성이 결정**한다.

### Part II: Long-term, Large-scale Evolution — Observations (Sect. 3, pp. 21–47) / 장기 대규모 진화 관측

**Formation of δ-spots (§3.1).** Zirin & Liggett (1987) classified δ-formation into three types:
- **Type 1 — island δ**: complex emerges as one tight unit with intertwined dipoles (e.g., McMath 11976 in Aug 1972; NOAA 5395 in Mar 1989 that triggered the 1989 Quebec blackout via X4.5/X10 flares).
- **Type 2 — spot-satellite**: new bipole emerges near a pre-existing large spot (e.g., NOAA 10930 in Dec 2006 which produced the X3.4 flare).
- **Type 3 — collision between two separate bipoles** (quadrupolar layout; e.g., NOAA 11158 in Feb 2011 → X2.2 flare).

Toriumi et al. (2017b) surveyed all ≥M5-class flares from 2010–2016 within 45° of disk centre and added **(4) Inter-AR events** where flares occur between two independent ARs (e.g., X1.2 on 2014 Jan 7 between NOAA 11944/11943). These four categories (Spot-spot, Spot-satellite, Quadrupole, Inter-AR) now serve as the observational taxonomy for flaring ARs. Jaeggli & Norton (2016): γ/δ-class fractions rise from 10% at solar minimum to >30% at maximum — consistent with surface-collision formation at high activity.

δ-spot 형성 3+1 유형: Type 1(island, 한 번에 복잡하게), Type 2(기존 spot 옆에 satellite bipole), Type 3(두 bipole 충돌), Type 4(Inter-AR). Jaeggli–Norton(2016)에 따르면 γ/δ 비율은 극소기 10%에서 극대기 30% 이상으로 증가한다.

**Photospheric features (§3.2).**
- **§3.2.1 Strong-field, strong-gradient, highly-sheared PILs.** Transverse fields up to 4300 G (Tanaka 1991; Zirin & Wang 1993b); Okamoto & Sakurai (2018) report 6250 G in a PIL — highest ever on the Sun. Gradient $|\nabla B_z|$ up to several hundred G Mm$^{-1}$ (e.g., Wang & Li 1998; Jing et al. 2006). Shear angle ≈80°–90° at flaring PILs (Hagyard 1990). Falconer et al. (2002, 2006): PIL length with $B_t>150$ G, shear $>45°$, $|\nabla B_z|>50$ G Mm$^{-1}$ predicts CMEs. **Schrijver R-value** (2007): total unsigned flux within ~15 Mm of strong-gradient PIL; $\log R=5.0$ yields 20% probability of X-class within 24 hr.
- **§3.2.2 Flow fields and spot rotations.** Shear flows along PILs (Harvey & Harvey 1976); converging flows build flux ropes (van Ballegooijen & Martens 1989). Rotating sunspots up to 200° in 3–5 days (Brown et al. 2003). Yan et al. (2008): sunspots rotating **opposite** to global differential rotation produce more M/X-class flares.
- **§3.2.3 Helicity injection.** $H=\int \mathbf{A}\cdot\mathbf{B}\,dV$, gauge-invariant relative helicity $H_R=\int(\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV$, and photospheric flux
$$ \frac{dH_R}{dt}=2\int\left[(\mathbf{A}_p\cdot\mathbf{B})v_n-(\mathbf{A}_p\cdot\mathbf{v})B_n\right]dS $$
(emergence + shear terms). **LaBonte et al. (2007)** empirical X-class threshold: peak helicity flux $>6\times 10^{36}$ Mx$^2$ s$^{-1}$ (see Fig. 23).
- **§3.2.4 Magnetic tongues** (López Fuentes et al. 2000): yin-yang polarity extensions on both sides of PIL = surface projection of twisted emerging tube's poloidal component. McAteer et al. (2005): fractal-dimension threshold 1.2 (M-class), 1.25 (X-class). Abramenko (2005): power-law index $\alpha>2.0$ for X-producing ARs vs. $\alpha\approx 5/3$ (Kolmogorov) for flare-quiet.
- **§3.2.5 (Im)balance of electric currents.** Vertical current density
$$ j_z=\frac{c}{4\pi}\left(\frac{\partial B_y}{\partial x}-\frac{\partial B_x}{\partial y}\right). $$
Georgoulis et al. (2012): eruptive AR 10930 shows large **non-neutralized** currents along PIL, quiet AR 10940 does not. Kontogiannis et al. (2017): threshold $J_\mathrm{total}\ge 4.6\times 10^{12}$ A, $J_\mathrm{max}\ge 8\times 10^{11}$ A for X-class.

광구 특징 요약: PIL에서 강자장(최대 6250 G)·강경사·강전단(80–90°), 90° 근처로 갈수록 자유 에너지 축적. 회전 흑점은 전역 자전과 반대 방향일 때 플레어 빈도가 높다. Helicity 주입률 임계 $6\times 10^{36}$ Mx² s⁻¹, 프랙탈 차원 1.25, power-law 지수 $\alpha>2$, 비중립 전류 $\ge 4.6\times 10^{12}$ A가 X-class 기준이다.

**Atmospheric and subsurface evolution (§3.3).**
- **§3.3.1 Flux ropes, sigmoids, filaments.** Canfield et al. (1999): sigmoid/large ARs erupt 51% of the time, accounting for 65% of eruptions. Sigmoids form above sheared PILs; filaments form along the same PIL in Hα. Forward/inverse S-shape of sigmoids reflects the twist handedness, matching the hemispheric helicity rule (negative in the north, positive in the south; Pevtsov et al. 1995). NLFFF extrapolations (Fig. 26) reconstruct the sigmoid topology — a central J-shaped flux rope under a potential arcade — matching Yohkoh/XRT morphology.
- **§3.3.2 EUV broadening.** Non-thermal broadening of Fe XII, XV, XXIV lines precedes flares by hours — signature of turbulent heating in pre-eruption coronal volume. Hinode/EIS spectroscopy captures these signatures along the AR loops and has become a multi-wavelength precursor diagnostic complementing magnetogram-based predictors.
- **§3.3.3 Helioseismic signatures.** Ilonidis et al. (2011): strong acoustic perturbations in NOAA 10488 at depths 42–75 Mm up to 2 days before emergence; rise speed $\sim 0.6$ km s$^{-1}$ from 65 Mm. Statistical confirmation by Birch et al. (2013), Barnes et al. (2014) using $>100$ ARs. Additional proxies: reduced acoustic power (Hartlep 2011; Toriumi et al. 2013b), f-mode amplification (Singh et al. 2016), horizontal divergent flows (Toriumi 2012, 2014a).

시그모이드와 필라멘트는 공통 PIL 위에 형성된다. 시그모이드의 S/역-S 모양은 twist 손잡이를 반영하며 반구 helicity 규칙과 일치한다. 플레어 수 시간 전부터 EUV 비-열적 확대가 감지된다. 광구 하 42–75 Mm에서 상승 flux의 음파 신호가 출현 2일 전 검출 가능하며, acoustic power 감소·f-mode 증폭·수평 발산 흐름 등 보조 지표가 있다.

### Part III: Theoretical Aspects (Sect. 4, pp. 48–77) / 이론적 측면

The Sect. 4 discussion proceeds from general flux-emergence simulations through increasingly specific δ-spot-formation scenarios to the data-constrained/data-driven modelling that couples directly with observations.

Sect. 4는 일반 flux-emergence 시뮬레이션에서 시작해 δ-spot 형성 시나리오를 구체화하고, 마지막에 관측과 직접 결합하는 data-constrained/data-driven 모델로 이어진다.

**§4.1 Flux emergence models for δ-spot formation.**
- **§4.1.1 Kinked tube (Type 1).** A tightly twisted tube exceeds the helical kink instability threshold; the rising tube develops a knot and produces a quadrupolar PIL with common penumbra (Linton, Fisher, Longcope 1996). Appendix A narrates the history of kink-instability advocates.
- **§4.1.2 Multi-buoyant segment (Type 3).** Toriumi et al. (2014b) reproduced NOAA 11158 by rising a tube with two buoyant segments along the axis; colliding central bipoles create the sheared PIL.
- **§4.1.3 Interacting tubes.** Two interacting tubes, even in simple bipolar ARs, can leave large **non-neutralized currents** because return currents annihilate at tube-tube contact.
- **§4.1.4 Turbulent convection.** Cheung et al. (2008) 3D radiative MHD: shearing, rotation, and flux cancellation at the PIL producing flux-rope and eruption — all emerging naturally from convective turbulence.
- **§4.1.5 Unified picture.** Comparative simulations (Toriumi & Takasao 2017) assigned each category a mechanism: Spot-spot = kink-unstable tube; Spot-satellite = two interacting tubes; Quadrupole = multi-buoyant-segment; Inter-AR = two independent tubes. **Spot-spot rises fastest and accumulates most free energy.**

δ-spot 형성 4가지 시나리오: kink(Type 1), multi-buoyant segment(Type 3), interacting tubes(Type 2), independent tubes(Inter-AR). 가장 많은 자유 에너지를 축적하는 경우는 Spot-spot(kink-unstable) 시나리오다.

**§4.2 Flux cancellation models.** van Ballegooijen & Martens (1989) pioneered the flux-cancellation scenario: converging photospheric flows shear the field above the PIL, small reconnection events at the PIL produce a long overlying twisted loop (the flux rope) and a shorter submerging dip, and repeated cycles build a full prominence. This model successfully reproduces filament formation above the PIL in Hα, and thermodynamically-augmented simulations (including radiative cooling) further reproduce the dense, cold prominence condensation within the flux-rope dips.

van Ballegooijen–Martens(1989) flux cancellation 모델: PIL에서의 수렴 흐름과 소규모 재결합이 반복되며 상부 twisted loop(= flux rope)가 형성되고 아래쪽 dip이 침강한다. Hα 필라멘트 형성과 냉각 응축이 열역학 포함 시뮬레이션으로 재현된다.

**§4.3 Data-constrained/data-driven models.** Force-free $\mathbf{j}\times\mathbf{B}=0 \Rightarrow \nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B}$; potential $\nabla^2\psi=0$ with $\mathbf{B}=-\nabla\psi$ as boundary from observed $B_z$. NLFFF takes full vector-magnetogram BC. Jiang et al. (2013) data-constrained MHD of NOAA 11283 reproduced CME eruption. Inoue et al. (2018b) modelled the X9.3 flare of NOAA 12673 (the strongest of Cycle 24): multiple flux ropes along sheared PIL reconnected to form a single highly-twisted rope that erupted via torus instability. Data-driven magneto-frictional method (Yang 1986): $\mathbf{v}=(\nu c)^{-1}\mathbf{j}\times\mathbf{B}$. Hayashi et al. (2018): NLFFF + photospheric $\mathbf{E}$ from Faraday's law $\partial\mathbf{B}/\partial t=-c\nabla\times\mathbf{E}$ drives evolving coronal model. A major caveat (Leake et al. 2017): if the driving cadence exceeds $L/v_h\sim 50$ s for a 1-Mm bipole at 20 km s⁻¹, the model aliases rapidly-evolving features; HMI's 12-min cadence is already marginal for flare-triggering scales.

관측-제약·관측-구동 모델: NLFFF는 $\nabla\times\mathbf{B}=\alpha\mathbf{B}$; Jiang 2013은 AR 11283 CME 재현, Inoue 2018은 AR 12673 X9.3 flare를 다중 flux rope 병합 후 torus 불안정으로 재현. 단점: HMI 12분 케이던스는 플레어-트리거 스케일($L/v_h\sim 50$ s)에 비해 이미 한계적이다.

### Part IV: Rapid Magnetic Field Changes (Sect. 5, pp. 78–93) / 플레어 관련 급격한 자기장 변화

**§5.1 Magnetic transients.** Early observations (Tanaka 1978; Patterson 1984) reported transient LOS-polarity reversals during flares, later shown to be artifacts of spectral-line-profile changes caused by flare emission rather than real field changes (Kosovichev & Zharkova 2001; Qiu & Gary 2003). Sun et al. (2017) analyzed 135-s HMI data and modelled the Fe I 6173 Å line in non-LTE (Hong et al. 2018) to quantify the transient contribution. These transients are distinct from the permanent changes in §5.2. Xu et al. (2018) GST discovered a **real transient rotation** of 12°–20° of transverse azimuth during the M6.5 flare on 2015 June 22, co-spatial with a flare ribbon and lasting only while the ribbon crossed — a new diagnostic of the reconnection electron beam.

플레어 방출로 spectral line profile이 변해 발생하는 겉보기 극성 반전(magnetic transient)은 진짜 자기장 변화가 아니다. Sun(2017), Hong(2018) non-LTE 모델로 정량화. Xu(2018) GST는 리본 통과 시 12–20° 횡방위각이 실제로 회전하는 현상을 발견했다(재결합 전자 빔 진단).

**§5.2 Rapid, persistent magnetic field changes (back-reaction).** Hudson et al. (2008) and Fisher et al. (2012) **back-reaction / coronal implosion**: if coronal free energy drops by $\Delta E$, momentum and energy conservation demand the photospheric field become more horizontal near the flaring PIL and more vertical in the periphery — a permanent, step-wise change. Wang et al. (2012b,c) confirmed this with HMI vector data: X2.2 of NOAA 11158 showed **$\Delta B_h \approx 500$ G increase within 30 min** at PIL, with simultaneous penumbra formation at PIL (darkening) and peripheral penumbra decay. Liu et al. (2005) TRACE statistical study of six X-class events showed the same pattern. Castellanos Durán et al. (2018): 59/75 flares show LOS field changes; $A_{\Delta B_\mathrm{LOS}}=(6.03\times 10^4) F_\mathrm{GOES}^{0.67}$ (correlation $r^2=0.60$). Sun et al. (2012) NLFFF extrapolation of AR 11158 showed current density at PIL increases while higher-altitude free energy decreases — **local enhancement + global decrease** signature. The physical cause: newly reconnected, newly-formed shorter loops spanning the PIL carry the flux closer to the surface.

Back-reaction의 발견: 플레어 후 PIL 근처 수평장이 급격·영구적으로 ~500 G 증가하고, 주변부는 수직화하며 penumbra가 쇠퇴하고, 중심 PIL에서는 새로운 penumbra가 형성된다. Lorentz force 방향이 코로나 → 광구(implosion). LOS 변화 면적은 $F_\mathrm{GOES}^{0.67}$에 비례한다. PIL에서는 전류 밀도 국소 증가와 고도별 자유 에너지 감소가 공존한다.

**§5.3 Sudden sunspot rotation and flow field changes.** Wang et al. (2014) showed DAVE-tracked sunspot rotation accelerates during/after the X2.2 in AR 11158, in the same sense as the horizontal Lorentz-force change. Liu et al. (2016a) used GST to discover that flare-ribbon sweeping across AR 12371 (M6.6) drove **sunspot rotation up to 50° hr⁻¹** with Poynting and helicity fluxes **temporarily reversing sign** — direct evidence that energy propagates from corona down to photosphere during the flare. Wang et al. (2018b) reported shear-flow enhancement (up to 0.9 km s⁻¹) and penumbral-flow increase to 2 km s⁻¹ in the same AR.

DAVE flow tracking과 GST 고해상도 데이터가 플레어 중 흑점 회전 가속(최대 50°/hr)과 shear flow 증가(최대 0.9 km/s), penumbral flow 2 km/s를 포착하며 Poynting·helicity flux의 일시적 역전을 발견했다.

**§5.4 Theoretical interpretations.** Three eruption categories (Longcope & Forbes 2014): tether-cutting, break-out, loss-of-equilibrium. Tether-cutting (Moore et al. 2001) — two-step reconnection, first near the surface producing a short loop (explains post-flare $\Delta B_h$ at PIL) and longer twisted rope, then the rope erupts. Inoue et al. (2018a) MHD simulation of the X2.2 using the observed photospheric field reproduced ribbons AND the permanent $\Delta B_h$ enhancement AND peripheral penumbral decay — unified observational/theoretical picture. Aulanier (2016) simulation even exhibits the flare-driven sunspot rotation, though most simulations assume line-tying and fix footpoints.

이론 해석: tether-cutting, break-out, loss-of-equilibrium 3가지. Inoue(2018a) MHD 시뮬레이션은 리본·$\Delta B_h$·peripheral penumbra decay를 모두 재현했고, Aulanier(2016)는 흑점 회전까지 시뮬레이션으로 제시했다.

### Part V: Summary & Forecasting (Sect. 6–7, pp. 93–101) / 요약과 예보

Three discriminants of flaring ARs: (a) **size** — but not alone (RGO 14886 counterexample), (b) **complexity** — dispersed polarities, δ-structure, sheared PILs, tongues, flux ropes, sigmoids, and (c) **evolution speed** — fast flux emergence rate and its time-derivatives. Table 1 summarizes empirical X-class thresholds; Table 2 lists 13 SHARP parameters (Bobra & Couvidat 2015) with their F-scores — total unsigned current helicity $H_{c,\mathrm{total}}\propto \sum |B_z\cdot J_z|$ scores highest (3560), total magnitude of Lorentz force $F\propto\sum B^2$ (3051), total photospheric magnetic free energy $\rho_\mathrm{tot}\propto\sum(\mathbf{B}^\mathrm{Obs}-\mathbf{B}^\mathrm{Pot})^2 dA$ (2996). Forecasting has pivoted from statistical ($<2010$) to machine-learning (Bobra & Couvidat 2015; Nishizuka et al. 2017, 2018 DeFN achieving competitive performance with dynamical parameters added).

플레어 AR의 3요소: 크기·복잡성·진화속도. 가장 강력한 단일 예측변수는 total unsigned current helicity $\sum |B_z J_z|$(F=3560), 그다음 Lorentz force 총량, free-energy 밀도 순이다. Bobra–Couvidat(2015) SHARP 13-변수 ML 모델이 Leka–Barnes(2007) 통계를 대체하고 있다.

### Section 7: Discussion — outstanding questions and broader impacts / 논의

**§7.1 Outstanding questions.**
1. We still lack a direct "visual" image of subsurface emerging flux — local helioseismology has limited SNR and treats strong flux as a perturbation, which it is not.
2. The specific role of turbulent convection in generating complexity (beyond qualitative agreement) is unquantified. Access to the topmost 20 Mm requires compressible simulations, not anelastic (Hotta et al. 2014).
3. Why does extremely strong transverse field (>6000 G) appear along δ-spot PILs rather than at umbral centres?
4. Does a flux rope exist before eruption or form at the moment of eruption? NLFFF + data-constrained results favour pre-existence in flare-productive ARs, but chromospheric low-β vector data (DKIST) is needed for definitive force-free boundary conditions.

남은 질문들: (1) 지하 flux의 직접 영상 불가, (2) 난류 대류 역할의 정량화 부족, (3) PIL 초강자장(>6000 G)의 기원, (4) flux rope의 선-존재 여부.

**§7.2 Broader impacts.**
- **§7.2.1 Prediction/forecasting.** Table 2 (SHARP 13 parameters) + ML; the field has pivoted from "stochastic vs. deterministic" philosophical debate to practical ML performance competition.
- **§7.2.2 Historical extreme events.** Lundstedt et al. (2015) reconstructed 1921 "magnetograms" from Mt. Wilson drawings using torus instability thinking; transfer-learning reconstructions of vector magnetograms for Cycle 23 from LOS-only data are becoming feasible.
- **§7.2.3 Solar-stellar connection.** Stellar superflares on G-dwarfs (Maehara 2012; Shibayama 2013) and detection of stellar CMEs via type-II radio bursts (Crosley & Osten 2018) are new frontiers; confined-flare concepts from flaring AR physics (large overlying field, small decay index) help explain why strong stellar flares may NOT produce CMEs on active M-dwarfs.

넓은 영향: ML 예보 개발, 역사적 극단 사건(1921년 등) 자기장 재구성, 태양-항성 초대플레어 비교. 특히 활동성 M-dwarf의 CME 부재는 강한 덮개장에 의한 confined flare 개념으로 설명할 수 있다.

### Part VI: Benchmark Active-Region Case Studies / 벤치마크 AR 사례

The review repeatedly returns to a small set of benchmark ARs that span the observational landscape. Understanding these cases is essential for reading modern flare literature.

**NOAA 10930 (Dec 2006, X3.4).** Type-2 δ-spot: a new positive bipole emerges inside the negative umbra's penumbra and drifts east with counter-clockwise rotation (Fig. 13; Kubo et al. 2007). NLFFF extrapolation (Schrijver et al. 2008; Fig. 6) reveals a low-lying helical flux rope along the sheared PIL with the highest electric-current concentration exactly where the flare occurs. Bamba et al. (2013) resolved a "magnetic channel" of 3000-km scale at the PIL from which the flare ribbons originated — a fine-scale flare-triggering field. Min & Chae (2009) tracked the southern sunspot rotation: 0.22 km s⁻¹ counter-clockwise velocity field before the X3.4 (Fig. 21). This single AR ties together sheared PIL, magnetic channel, flux-rope formation, sunspot rotation, and rapid photospheric back-reaction.

AR 10930은 Type-2 δ-spot의 전형이다. 음극 penumbra 안으로 양극 bipole이 emerg하며 반시계 방향으로 회전했고, NLFFF 외삽은 sheared PIL에 놓인 helical flux rope를 복원한다. 3000-km 자기 채널이 플레어 리본의 출발점이었으며, 이 하나의 AR이 shear PIL, magnetic channel, flux rope, 흑점 회전, back-reaction을 모두 보여주는 교과서 사례다.

**NOAA 11158 (Feb 2011, X2.2).** Type-3 (quadrupolar) δ-spot produced by collision of two emerging bipoles P1–N1 and P2–N2 (Fig. 14; Toriumi et al. 2014b). The sheared PIL formed between the colliding P1 and N2 centroids. Wang et al. (2012b,c) used the X2.2 to first **confirm via HMI vector data** that $B_h$ at the PIL increases by $\approx 500$ G within 30 minutes — the definitive evidence for flare-driven photospheric back-reaction. Multi-buoyant-segment flux-emergence simulations (Toriumi et al. 2014b) and data-driven NLFFF models (Inoue 2015; Inoue et al. 2018a) reproduce both the polarity collision and the flare eruption.

AR 11158은 Type-3 사분극 δ-spot으로 두 bipole 충돌로 형성되었다. Wang(2012)이 HMI 벡터 데이터로 PIL 수평장 500 G 증가(30분)를 최초 확증하여 back-reaction을 결정지었다. Multi-buoyant-segment 시뮬레이션과 data-driven NLFFF 모델이 충돌·폭발을 모두 재현한다.

**NOAA 12192 (Oct 2014, 6 X-class, 0 CMEs).** Largest AR of Cycle 24 (2750 MSH; Fig. 1). Produced six X-class flares but **no CME** (Sun et al. 2015): the decay index $n=-\partial \ln B_h/\partial \ln z$ stayed below the torus-instability critical $n_c\approx 1.5$ up to great heights. This exemplifies **confined flares**: strong overlying arcade traps the flux rope and prevents it from erupting. The ratio of reconnected flux to total AR flux is smaller for confined events (Toriumi et al. 2017b). DeRosa & Barnes (2018) showed X-class flares near open-field regions erupt more often, quantifying the role of large-scale topology.

AR 12192는 Cycle 24 최대 AR(2750 MSH), X-class 6회에도 CME가 0회인 confined flare의 대표 사례. torus-불안정성 임계 $n_c\approx 1.5$ 아래로 decay index가 유지되어 flux rope가 overlying arcade에 갇혔다.

**NOAA 12673 (Sep 2017, X9.3).** The strongest flare of Cycle 24. Inoue et al. (2018b) data-constrained MHD simulation (Fig. 49) shows multiple flux ropes along the sheared PIL reconnecting with each other to merge into a single highly-twisted rope that then erupted via torus instability. The extremely rapid flux emergence rate (Sun & Norton 2017) preceded the X9.3 by only a few days — a case where **time-derivative parameters** would have flagged the risk more effectively than static ones.

AR 12673은 Cycle 24 최강 X9.3의 근원. 데이터 제약 MHD 시뮬레이션은 다수의 flux rope가 병합된 뒤 torus 불안정으로 폭발했음을 보인다. 극단적으로 빠른 flux 출현율이 플레어에 선행한 사례로, 정적 파라미터가 아닌 시간-미분 파라미터의 중요성을 강조한다.

**NOAA 12371 (Jun 2015, M6.5).** Wang et al. (2017b, 2018b) used GST at BBSO to resolve an evolving **magnetic channel** (Fig. 20) with unsigned electric-current, positive-flux, and negative-flux time series during pre-flare episodes. Liu et al. (2016a) reported flare-ribbon-driven **sunspot rotation up to 50° hr⁻¹** during the M6.6 event, with Poynting and helicity fluxes temporarily reversing sign — direct evidence that during the flare, energy flow reverses from corona-to-photosphere.

AR 12371은 GST로 관측된 magnetic channel의 미세 구조와, 플레어 중 흑점이 시간당 50°까지 회전하고 Poynting·helicity flux가 일시적으로 역전된 사례다. 코로나 → 광구 에너지 역류의 직접 증거다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Hale-class progression α → β → β-γ → β-γ-δ tracks increasing flare productivity monotonically.** — Giovanelli (1939) onward, every statistical study confirms complex spot groups produce more energetic flares. **All** ≥X4 flares occur in βγδ ARs with >1000 MSH area (Sammis 2000). Hale 분류가 복잡해질수록 플레어 강도가 커지며, ≥X4 플레어는 예외 없이 >1000 MSH βγδ AR에서 발생한다.

2. **The δ-spot is the single strongest structural flag for extreme flaring.** — Umbrae of opposite polarities sharing a common penumbra (<2° separation; Künzel 1960) concentrate free energy at a geometrically inevitable strong-field, strong-gradient, highly-sheared PIL. ~34% of δ-spots violate Hale's rule (anti-Hale), and such ARs have much stronger X-class tendency. 공통 penumbra 내 반대 극성 umbra 구조가 극단 플레어의 가장 강력한 구조적 지표다. 약 34%의 δ-spot은 Hale 규칙을 위반하고, 이들이 X-class 플레어를 더 많이 만든다.

3. **Free magnetic energy $E_\mathrm{free}\sim 10^{31}$–$10^{32}$ erg is the flare fuel, and it is identifiable via NLFFF-potential subtraction.** — NLFFF extrapolation of NOAA 10930 showed $E_\mathrm{free}$ stored in a low-lying flux rope along the sheared PIL; the X3.4 flare released this energy (Schrijver et al. 2008). Carrington (1859) had $\sim 5\times 10^{32}$ erg. 자유 자기 에너지 $10^{31-32}$ erg이 플레어의 연료이며, NLFFF − potential field 차분으로 진단 가능하다. Carrington 사건은 $5\times 10^{32}$ erg 수준.

4. **Helicity injection has an empirical X-class threshold** $dH_R/dt > 6\times 10^{36}$ Mx$^2$ s$^{-1}$ (LaBonte et al. 2007). — Measured via photospheric integral of $2[(\mathbf{A}_p\cdot\mathbf{B})v_n-(\mathbf{A}_p\cdot\mathbf{v})B_n]$ using DAVE flow tracking. Helicity accumulates, not dissipates, in ideal MHD, so its delivery rate flags an impending large flare. 광구 helicity 주입률이 $6\times 10^{36}$ Mx² s⁻¹을 넘으면 X-class 가능성이 경험적으로 높다.

5. **Non-neutralized currents along PILs are a hallmark of flaring ARs.** — Eruptive AR 10930 shows large net $j_z=(c/4\pi)(\partial_x B_y-\partial_y B_x)$ integrated over a single polarity; quiet AR 10940 does not (Georgoulis 2012). Threshold: $J_\mathrm{total}\ge 4.6\times 10^{12}$ A for X-class (Kontogiannis 2017). PIL을 따라 흐르는 비-중립 전류는 플레어 AR의 특징이며, 총량 $4.6\times 10^{12}$ A 이상이 X-class 임계다.

6. **δ-spots form through four distinct histories — Spot-spot, Spot-satellite, Quadrupole, and Inter-AR — each with a unique subsurface scenario.** — Respectively: kink-unstable single tube, two interacting tubes, multi-buoyant-segment single tube, and two independent tubes. The surface taxonomy (Toriumi et al. 2017b) maps cleanly onto simulations (Toriumi & Takasao 2017), closing the loop between observation and MHD theory. δ-spot 4-범주는 지하 4-시나리오(kink, 상호작용, 다중 부력 세그먼트, 독립 튜브)와 일대일 대응된다. 관측-이론 사이의 고리가 닫혔다.

7. **Flares drive a rapid, permanent, stepwise photospheric back-reaction.** — At the flaring PIL $B_h$ jumps by ~500 G within 30 min, peripheral penumbra decays, central penumbra forms, and the field "implodes" toward the surface. This "tail-wags-the-dog" phenomenon, doubted for decades due to ground-based seeing issues, was conclusively confirmed by HMI vector data in 2012 (Wang et al. 2012b,c). 플레어는 광구에 급격·영구·계단형 back-reaction을 남긴다. PIL 수평장 ~500 G 급증, 주변 penumbra 쇠퇴, 중심 penumbra 형성. HMI 벡터 데이터(2012)가 결정적 증거를 제공했다.

8. **Machine-learning flare forecasting now outperforms pure statistics, especially when dynamic (time-derivative) parameters are included.** — Bobra & Couvidat (2015) SHARP+SVM for M1+ flares; Nishizuka et al. (2017, 2018) DeFN added flare history and chromospheric precursors, beating static-snapshot baselines. The top predictor is $H_{c,\mathrm{total}}\propto\sum |B_z J_z|$ (F-score 3560). 정적 SHARP 파라미터 기반 ML에 시간-미분 동적 파라미터를 추가하면 예측 성능이 뚜렷이 개선된다. 최우수 예측변수는 total unsigned current helicity $\sum|B_z J_z|$.

9. **Size alone is a necessary but not sufficient condition — complexity dominates.** — The largest sunspot in recorded history, RGO 14886 (April 1947, 6132 MSH), was flare-quiet because it had a simple β bipolar structure, whereas RGO 14585 (July 1946, 4279 MSH, 4th largest ever) was a δ-configuration that produced great flares and geomagnetic storms with ground-level enhancement. This counterexample is the cornerstone of the review's argument that magnetic complexity, not spot area, is the root cause. 역사상 최대 흑점(RGO 14886, 1947)은 단순 β로 플레어가 거의 없었고, 네 번째로 큰 14585(1946)는 δ로 강력한 플레어를 만들었다. 크기가 아니라 복잡성이 본질이다.

10. **Sigmoids are eruptive precursors with clear statistical significance.** — Canfield et al. (1999) Yohkoh SXT survey: ARs with sigmoid structures erupt 51% of the time, and sigmoids account for 65% of all observed eruptions. Sigmoids form along sheared PILs and are best understood as the soft-X-ray signature of a pre-eruption flux rope. Identifying a sigmoid is therefore operationally equivalent to identifying an eruption-ready flux rope. Yohkoh SXT 통계(Canfield 1999): 시그모이드는 51% 빈도로 폭발하며 전체 폭발의 65%를 차지한다. Sheared PIL 위의 pre-eruption flux rope의 soft X-ray 흔적이다.

---

## 4. Mathematical Summary / 수학적 요약

### (a) Magnetic buoyancy and flux emergence
$$ p_e = p_i + \frac{B^2}{8\pi},\qquad f_B=(\rho_e-\rho_i)g=\frac{B^2}{8\pi H_p},\qquad H_p=\frac{k_B T}{mg}. $$
**Interpretation / 해석.** Flux tubes in thermal equilibrium are under-dense and buoyant; the buoyancy force per unit volume equals magnetic pressure divided by pressure scale height. At the base of the CZ ($B\sim 10^5$ G, $T\sim 2\times 10^6$ K), $f_B\sim B^2/(8\pi\cdot 50\text{ Mm})\approx 0.8$ dyn cm⁻³.
열평형 flux tube는 밀도가 낮아 부력을 받으며, 단위 부피당 부력 = 자기 압력/압력 척도 높이. CZ 바닥에서 $f_B\approx 0.8$ dyn cm⁻³.

### (b) Magnetic helicity and its injection
$$ H=\int_V \mathbf{A}\cdot\mathbf{B}\,dV,\qquad \mathbf{B}=\nabla\times\mathbf{A} $$
$$ H_R=\int_V (\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV \qquad\text{(gauge-invariant relative)} $$
$$ \frac{dH_R}{dt}=2\int_S \bigl[(\mathbf{A}_p\cdot\mathbf{B})v_n \;-\; (\mathbf{A}_p\cdot\mathbf{v})B_n\bigr] dS. $$
**Interpretation.** First bracket term = **emergence term** (new twisted flux crossing the photosphere); second = **shear term** (horizontal motion of existing polarities). Total accumulated helicity $\int(dH_R/dt)dt$ matches coronal $H$ from NLFFF (Park et al. 2008; Jing et al. 2012).
첫 항은 emergence, 둘째 항은 shear 기여. 축적된 값은 NLFFF로 얻은 코로나 helicity와 일치한다.

### (c) Force-free field and current density
$$ \mathbf{j}\times\mathbf{B}=0\;\Rightarrow\;\nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B},\qquad \mathbf{j}=\frac{c}{4\pi}\nabla\times\mathbf{B}. $$
Photospheric vertical current:
$$ j_z=\frac{c}{4\pi}\left(\frac{\partial B_y}{\partial x}-\frac{\partial B_x}{\partial y}\right). $$
Potential limit: $\alpha=0$, $\mathbf{B}=-\nabla\psi$, $\nabla^2\psi=0$ with BC $B_z^\mathrm{obs}$.
**Interpretation.** In the low-β corona, Lorentz force vanishes and current is field-aligned with $\alpha$ varying in space (NLFFF). NLFFF minus potential gives free energy.
저-β 코로나에서 Lorentz force = 0, 전류는 자기장에 평행($\alpha$가 공간 의존). NLFFF − potential = 자유 에너지.

### (d) Free magnetic energy and back-reaction
$$ E_\mathrm{free}=\frac{1}{8\pi}\int_V \left(B^2_\mathrm{NLFFF}-B^2_\mathrm{Pot}\right)dV. $$
Hudson (2000) back-reaction prediction: $\Delta B_h>0$ at PIL post-flare, $\Delta B_z$-periphery turns more vertical. Wang et al. (2012b): $\Delta B_h\approx 500$ G in 30 min for X2.2 (NOAA 11158).
**Interpretation.** Post-flare coronal relaxation removes $\sim 10^{31}$–$10^{32}$ erg of $E_\mathrm{free}$; by momentum conservation the photospheric field tilts toward horizontal at PIL.
코로나 완화로 $10^{31-32}$ erg의 자유 에너지가 방출되고, 운동량 보존으로 PIL 수평장이 증가한다.

### (e) Shear angle, flare index, and forecasting parameters
Shear angle $\theta_\mathrm{sh}$ = angle between observed transverse field and corresponding potential-field direction. Flaring PILs: $\theta_\mathrm{sh}\approx 80°$–$90°$.

Abramenko (2005) flare index:
$$ FI=\frac{1}{\tau}\left[100\sum I_X+10\sum I_M+1.0\sum I_C+0.1\sum I_B\right]. $$

Schrijver (2007) R-value: total unsigned flux within 15 Mm of strong-gradient PIL. $\log R=5.0 \Rightarrow 20\%$ probability of X-class in 24 hr.

Bobra & Couvidat (2015) top SHARP parameters (higher F-score = more discriminative):
| Parameter / 파라미터 | Formula / 공식 | F |
|---|---|---|
| Total unsigned current helicity | $H_{c,\mathrm{tot}}\propto\sum\|B_z\cdot J_z\|$ | 3560 |
| Total magnitude of Lorentz force | $F\propto\sum B^2$ | 3051 |
| Total photospheric free-energy density | $\rho_\mathrm{tot}\propto\sum(\mathbf{B}^\mathrm{Obs}-\mathbf{B}^\mathrm{Pot})^2 dA$ | 2996 |
| Total unsigned vertical current | $J_\mathrm{tot}=\sum\|J_z\|dA$ | 2733 |
| Total unsigned flux | $\Phi=\sum\|B_z\|dA$ | 2437 |
| Mean photospheric free energy | $\bar\rho=(1/N)\sum(\mathbf{B}^\mathrm{Obs}-\mathbf{B}^\mathrm{Pot})^2$ | 1064 |
| Fraction of area with shear $>45°$ | Area($\theta_\mathrm{sh}>45°$)/total | 740.8 |

### Worked numerical example / 정량 예제
**Case: AR 11158, X2.2 flare, 2011 Feb 15.**
- Peak $B_h$ change at PIL: $\Delta B_h\approx 500$ G within $\Delta t\approx 30$ min $=1800$ s.
- Area affected (Castellanos Durán 2018 scaling for $F_\mathrm{GOES}=2.2\times 10^{-4}$): $A\approx 6.03\times 10^4\cdot (2.2\times 10^{-4})^{0.67}\approx 6.03\times 10^4\cdot 3.7\times 10^{-3}\approx 223$ Mm².
- Energy scale: $\Delta E\approx (\Delta B_h)^2/(8\pi)\cdot V \sim (500\text{ G})^2/(8\pi)\cdot 10^{27}$ cm³ $\sim 10^{31}$ erg — of the correct order for an X2.2 event.
- Implication: observed back-reaction is energetically self-consistent with the free-energy release inferred from NLFFF extrapolation.

AR 11158 X2.2 사례: PIL 수평장이 30분 내 ~500 G 증가, 영향 면적 약 223 Mm². 에너지 규모 $\sim 10^{31}$ erg — X2.2 방출량과 자기 일관적이다.

### Second worked example / 두 번째 정량 예제
**Case: Hale-class progression and X-class probability.**
The empirical Sammis et al. (2000) result: among ARs with peak GOES classes, β spots rarely exceed M-class; βγ reach X but only in large (>500 MSH) regions; **all** ≥X4 flares require βγδ AND >1000 MSH. Converting this into a rough Bayesian posterior (taking uniform priors for illustration):

$$ P(\text{X-class} \mid \text{Hale class}) \approx \begin{cases} 0.005 & \alpha \\ 0.02 & \beta \\ 0.10 & \beta\gamma \\ 0.40 & \beta\gamma\delta, \text{ area}>1000\text{ MSH} \end{cases} $$

Adding the **R-value** (Schrijver 2007): $\log R=5.0 \Rightarrow P(X\ge 1\mid \log R=5.0, 24\text{ hr})\approx 0.20$. Multiplying by a detection window enhancement from helicity-flux threshold (LaBonte 2007): if $dH/dt>6\times 10^{36}$ Mx² s⁻¹ is also met, posterior $\gtrsim 0.5$. The review's practical message: **no single parameter** crosses 0.8 probability on its own; only the combination of Hale class + R-value + helicity flux + current helicity reaches operational forecasting confidence, which is why ML classifiers with multiple SHARP features outperform any single threshold.

두 번째 예: Hale 분류별 X-class 확률을 단순 Bayesian 사후로 표현하면 α≈0.005, β≈0.02, βγ≈0.10, βγδ+>1000 MSH≈0.40이고, R-value 및 helicity flux 기준을 추가하면 0.5 이상이다. 단일 변수는 0.8을 넘기 어려워 다변량 ML 분류기가 필요하다.

### Third worked example / 세 번째 정량 예제
**Case: Free energy budget for an X-class flare.**
Starting from $E_\mathrm{free}=\int(B^2_\mathrm{NLFFF}-B^2_\mathrm{Pot})/(8\pi)\,dV$:
- Typical X-class release: $E\sim 10^{32}$ erg.
- If distributed over coronal volume $V\sim (100\text{ Mm})^3=10^{27}$ cm³,
- required excess $\Delta(B^2)/(8\pi)\approx 10^{32}/10^{27}=10^5$ erg cm⁻³,
- $\Rightarrow \Delta B\approx\sqrt{8\pi\cdot 10^5}\approx 1600$ G.
- Since $B_\mathrm{Pot}\sim 500$ G at coronal heights, NLFFF must reach $B_\mathrm{NLFFF}\sim\sqrt{500^2+1600^2}\approx 1680$ G — only possible in highly non-potential PIL volumes.
- Consequence: measuring $E_\mathrm{free}>10^{32}$ erg via NLFFF is a sharp X-class signature but requires vector magnetogram noise suppression and force-free convergence, hence the operational appeal of proxy parameters like $H_c$ and non-neutralized current.

세 번째 예: X-class 방출 $10^{32}$ erg은 코로나 부피 $10^{27}$ cm³에 분포할 때 초과 자장 ~1600 G이 필요하므로, NLFFF 외삽으로 $E_\mathrm{free}>10^{32}$ erg을 직접 측정하는 것은 강력한 X-class 신호지만 벡터 자력계 노이즈 저감이 필수이며, 실무에서는 $H_c$·비중립 전류 같은 프록시 파라미터가 선호된다.

---

### Additional tabular summary / 추가 표 요약

**Empirical X-class thresholds (Table 1 of the review / 논문 Table 1).**
| Parameter / 파라미터 | X-class threshold / X-class 임계 | Reference / 출처 |
|---|---|---|
| Spot area | 40% of ≥1000 MSH βγδ-spots | Sammis et al. 2000 |
| PIL total unsigned flux (R-value) | 20% of $\log R=5.0$ (next 24 hr) | Schrijver 2007 |
| Fractal dimension | $\ge 1.25$ | McAteer et al. 2005 |
| Power-law index $\alpha$ | $>2.0$ | Abramenko 2005 |
| Peak helicity injection rate | $\ge 6\times 10^{36}$ Mx$^2$ s$^{-1}$ | LaBonte et al. 2007 |
| Total non-neutralized current | $\ge 4.6\times 10^{12}$ A | Kontogiannis et al. 2017 |
| Maximum non-neutralized current | $\ge 8\times 10^{11}$ A | Kontogiannis et al. 2017 |
| Normalized helicity gradient variance | 1.13 (1 day before flare) | Reinard et al. 2010 |

각 임계치는 독립 연구로 도출된 것이며, 실무 예보에서는 이들을 다변량 ML 특성으로 결합한다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1859 ─────────── Carrington/Hodgson white-light flare (X45)
1908 ─────────── Hale: sunspot magnetic fields
1919 ─────────── Hale-Nicholson polarity rule
1939 ─────────── Giovanelli: flare rate ∝ spot area & complexity
1955 ─────────── Parker: magnetic buoyancy
1960 ─────────── Künzel: δ-classification
1966 ─────────── CSHKP standard flare model (Carmichael-Sturrock-Hirayama-Kopp-Pneuman)
1985 ─────────── Zwaan: magnetic element hierarchy
1987 ─────────── Zirin & Liggett: BBSO δ-spot 3-type typology
1989 ─────────── Shibata et al.: 2D MHD flux emergence sim; van Ballegooijen & Martens: filament formation
1996 ─────────── Linton et al.: kink instability for δ-spots; Paper #27 (Shibata MHD reconnection, series)
1999 ─────────── Canfield et al.: sigmoid-eruption connection
2000 ─────────── Sammis et al.: βγδ/>1000 MSH → all ≥X4 flares
2001 ─────────── Kosovichev & Zharkova: first rapid photospheric field-change (Bastille Day)
2007 ─────────── Schrijver R-value; LaBonte et al. helicity-flux threshold
2008 ─────────── Hudson et al. back-reaction prediction; Hinode launches
2010 ─────────── SDO/HMI launches → ubiquitous vector magnetograms
2011 ─────────── Ilonidis helioseismic pre-emergence signal; NOAA 11158 X2.2
2012 ─────────── Wang et al. HMI confirms rapid ΔB_h at PIL
2014 ─────────── AR 12192 (6 X-class, no CME) — confinement puzzle
2015 ─────────── Bobra & Couvidat SHARP + ML flare forecasting; Paper #40 (van Driel-Gesztelyi & Green AR evolution, series)
2017 ─────────── Toriumi & Takasao unified simulation of 4 δ-spot types; AR 12673 X9.3
2019 ★─────────── TORIUMI & WANG LIVING REVIEW (THIS PAPER)
2020s ─────────── DKIST era, DeFN machine-learning, solar-stellar superflares
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#27 Shibata et al. 1996 (plasmoid-induced MHD reconnection)** | Provides the reconnection theory underpinning the CSHKP flare model cited throughout §2.2. The standard-model schematic in Fig. 7(a) is essentially the descendant of Shibata's plasmoid-eruption picture. 본 리뷰에서 표준 플레어 모델로 인용되는 재결합 이론의 원형. | High — defines the flare mechanism that this review's ARs host. |
| **#40 van Driel-Gesztelyi & Green 2015 (AR evolution)** | Provides the general AR life-cycle (emergence → decay → flux dispersal) on which this paper layers the flare-productive subclass. Cited in §2 as the primary historical reference for AR definition. 일반 AR 진화의 역사적 참고문헌. | High — the scaffold this review specializes. |
| Zirin & Liggett 1987 (BBSO) | The seminal δ-spot typology (3 types: island, satellite, quadrupole) directly extended by Toriumi et al. 2017b into the 4-category scheme (Spot-spot, Spot-satellite, Quadrupole, Inter-AR) adopted in §3.1. δ-spot 분류의 원형. | High — foundational observational classification. |
| Hudson et al. 2008; Fisher et al. 2012 (back-reaction) | Theoretical prediction that coronal energy release forces photospheric horizontal-field enhancement at PIL, confirmed by HMI vector-data observations (§5.2). Lorentz-force 역작용 이론. | High — explains rapid photospheric changes in §5. |
| Hale et al. 1919 (Mount Wilson classification) | Defined α/β/γ classes; Künzel 1960 added δ. Whole review uses this as the taxonomic backbone. Mount Wilson 분류의 정의. | High — taxonomy used throughout. |
| Bobra & Couvidat 2015 (SHARP + ML forecasting) | Practical machine-learning implementation of this review's photospheric predictors; Table 2 of §7.2.1 reproduces its 13-parameter F-scores. ML 예보의 구현. | High — operational culmination of the review's quantitative parameters. |
| Parker 1955 (magnetic buoyancy) | Foundational physics for flux emergence (§2.1.1, §4.1); Eq. (3) of the review. Flux tube의 부력 이론 원전. | Medium — underlies all emergence simulations. |
| Canfield et al. 1999 (sigmoid eruptivity) | Established 51%/65% sigmoid-eruption statistics used in §3.3.1. 시그모이드-폭발 통계의 근거. | Medium — key observational baseline for flux-rope criterion. |

---

### Glossary of key quantitative thresholds / 핵심 정량 임계 용어 사전

To aid operational use, the paper's quantitative thresholds are collected here for quick reference:

- **Spot area** for ≥X4 flare candidate: $>1000$ MSH = $3\times 10^9$ km² AND βγδ class (Sammis 2000).
- **Umbral field strength** of sunspots: 2900–3300 G typically, occasionally $>4000$ G (Zwaan 1985 hierarchy).
- **Transverse PIL field** at flaring δ-spots: 4300 G typical (Tanaka 1991), record 6250 G (Okamoto & Sakurai 2018).
- **PIL gradient** $|\nabla B_z|$: $>50$ G Mm⁻¹ for CME prediction (Falconer 2002); up to several hundred G Mm⁻¹ observed.
- **Shear angle** at flaring PILs: 80°–90° (Hagyard 1990).
- **R-value** strong-gradient PIL flux: $\log R\ge 5.0 \Rightarrow P(\mathrm{X}, 24\text{h})\ge 20\%$ (Schrijver 2007).
- **Helicity injection rate**: peak $dH/dt>6\times 10^{36}$ Mx² s⁻¹ for X-class (LaBonte 2007).
- **Non-neutralized current**: $J_\mathrm{tot}\ge 4.6\times 10^{12}$ A, $J_\mathrm{max}\ge 8\times 10^{11}$ A (Kontogiannis 2017).
- **Fractal dimension** of magnetogram: $\ge 1.2$ for M-class, $\ge 1.25$ for X-class (McAteer 2005).
- **Power-law index** of magnetogram spectrum: $\alpha>2.0$ for X-class (Abramenko 2005); Kolmogorov $5/3$ for flare-quiet.
- **Free magnetic energy** for X-class: $\sim 10^{31}$–$10^{32}$ erg (NLFFF-potential).
- **Back-reaction $\Delta B_h$** at PIL: ~500 G within 30 min for X2.2 (Wang 2012b).
- **Flare-induced sunspot rotation**: up to 50° hr⁻¹ (Liu 2016a, AR 12371).
- **Carrington flare** energy: $\sim 5\times 10^{32}$ erg, $\sim$X45 (Tsurutani 2003; Boteler 2006).

이 임계치들은 독립 연구에서 추출된 것으로 ML 예보의 feature engineering 기반이 된다.

### Standalone Test — can a non-reader explain this paper? / 독립 테스트

A reader who has not opened the original paper should, from these notes alone, be able to:
1. Explain **why δ-spots are flare-productive** (common-penumbra geometry forces sheared strong-gradient PILs).
2. Describe the **four δ-spot formation subsurface scenarios** (kink, multi-buoyant, interacting, independent) and their observational categories (Spot-spot, Spot-satellite, Quadrupole, Inter-AR).
3. Write down the **free-magnetic-energy formula** and state its typical flare-release scale ($10^{31-32}$ erg).
4. Write down the **helicity injection flux** and state the X-class threshold ($6\times 10^{36}$ Mx² s⁻¹).
5. Describe the **back-reaction phenomenon** and quantify it ($\Delta B_h\sim 500$ G at PIL in 30 min for X2.2 AR 11158).
6. List the **top 3 SHARP parameters** for ML flare forecasting (total current helicity, total Lorentz force, total free-energy density).
7. Place the paper between **#27 (Shibata reconnection, 1996)** and ML-era forecasting (post-2015).

본 노트만으로도 독자는 δ-spot의 플레어 생산성, 4가지 형성 시나리오, 자유 에너지·helicity 공식과 임계치, back-reaction 정량치, 예보 핵심 파라미터를 설명할 수 있어야 한다.

---

## 7. References / 참고문헌

- Toriumi, S., & Wang, H., "Flare-productive active regions," *Living Reviews in Solar Physics*, 16:3, 2019. DOI: 10.1007/s41116-019-0019-7
- Parker, E. N., "The Formation of Sunspots from the Solar Toroidal Field," *Astrophys. J.* 121, 491, 1955.
- Hale, G. E., et al., "The Magnetic Polarity of Sun-Spots," *Astrophys. J.* 49, 153, 1919.
- Künzel, H., "Zur Klassifikation von Sonnenfleckengruppen," *Astron. Nachr.* 285, 271, 1960.
- Zirin, H., & Liggett, M. A., "Delta spots and great flares," *Solar Phys.* 113, 267, 1987.
- Sammis, I., Tang, F., & Zirin, H., "The Dependence of Large Flare Occurrence on the Magnetic Structure of Sunspots," *Astrophys. J.* 540, 583, 2000.
- Schrijver, C. J., "A Characteristic Magnetic Field Pattern Associated with All Major Solar Flares and Its Use in Flare Forecasting," *Astrophys. J. Lett.* 655, L117, 2007.
- LaBonte, B. J., Georgoulis, M. K., & Rust, D. M., "Survey of Magnetic Helicity Injection in Regions Producing X-Class Flares," *Astrophys. J.* 671, 955, 2007.
- Hudson, H. S., Fisher, G. H., & Welsch, B. T., "Flare energy and magnetic field variations," *ASP Conf. Ser.* 383, 221, 2008.
- Wang, S., Liu, C., Liu, R., Deng, N., Liu, Y., & Wang, H., "Response of the Photospheric Magnetic Field to the X2.2 Flare on 2011 February 15," *Astrophys. J. Lett.* 745, L17, 2012b.
- Bobra, M. G., & Couvidat, S., "Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data with a Machine-Learning Algorithm," *Astrophys. J.* 798, 135, 2015.
- Toriumi, S., Schrijver, C. J., Harra, L. K., Hudson, H., & Nagashima, K., "Magnetic Properties of Solar Active Regions That Govern Large Solar Flares and Eruptions," *Astrophys. J.* 834, 56, 2017b.
- Linton, M. G., Longcope, D. W., & Fisher, G. H., "The Helical Kink Instability of Isolated, Twisted Magnetic Flux Tubes," *Astrophys. J.* 469, 954, 1996.
- Inoue, S., Shiota, D., Bamba, Y., & Park, S.-H., "Magnetohydrodynamic Modeling of a Solar Eruption Associated with an X9.3 Flare Observed in the Active Region 12673," *Astrophys. J.* 867, 83, 2018b.
- Shibata, K., et al., "Hydromagnetic emerging flux tubes and arch filament systems," *Astrophys. J.* 338, 471, 1989.
- van Ballegooijen, A. A., & Martens, P. C. H., "Formation and eruption of solar prominences," *Astrophys. J.* 343, 971, 1989.
- Canfield, R. C., Hudson, H. S., & McKenzie, D. E., "Sigmoidal morphology and eruptive solar activity," *Geophys. Res. Lett.* 26, 627, 1999.
- Toriumi, S., Takasao, S., Cheung, M. C. M., et al., "Comparative study of flaring active regions with high initial vertical flux injection," *Astrophys. J.* 836, 63, 2017 (Toriumi & Takasao unified simulation).
- Fisher, G. H., Bercik, D. J., Welsch, B. T., & Hudson, H. S., "Global Forces in Eruptive Solar Flares: The Lorentz Force Acting on the Solar Atmosphere and the Solar Interior," *Solar Phys.* 277, 59, 2012.
- Kontogiannis, I., Georgoulis, M. K., Park, S.-H., & Guerra, J. A., "Non-neutralized Electric Currents in Solar Active Regions and Flare Productivity," *Solar Phys.* 292, 159, 2017.
- Nishizuka, N., Sugiura, K., Kubo, Y., et al., "Deep Flare Net (DeFN) Model for Solar Flare Prediction," *Astrophys. J.* 858, 113, 2018.
- Castellanos Durán, J. S., Kleint, L., & Calvo-Mozo, B., "A statistical study of photospheric magnetic field changes during 75 solar flares," *Astrophys. J.* 852, 25, 2018.
- Liu, C., Deng, N., Lee, J., et al., "Flare Differentially Rotates Sunspot on the Sun's Surface," *Astrophys. Space Sci. Lib.* 426, 2016 (Liu et al. 2016a, AR 12371).
- Schrijver, C. J., DeRosa, M. L., Metcalf, T., et al., "Nonlinear Force-Free Field Modeling of a Solar Active Region Around the Time of a Major Flare and Coronal Mass Ejection," *Astrophys. J.* 675, 1637, 2008.
- Hagyard, M. J., "The significance of vector magnetic field measurements," *Mem. Soc. Astron. Ital.* 61, 337, 1990.
- Zwaan, C., "The emergence of magnetic flux," *Solar Phys.* 100, 397, 1985.
- Kosovichev, A. G., & Zharkova, V. V., "Magnetic Energy Release and Transients in the Solar Flare of 2000 July 14," *Astrophys. J. Lett.* 550, L105, 2001.
