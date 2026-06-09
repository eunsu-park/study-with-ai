---
title: "The Multi-Scale Nature of the Solar Wind"
authors: [Daniel Verscharen, Kristopher G. Klein, Bennett A. Maruca]
year: 2019
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-019-0021-0"
topic: Living_Reviews_in_Solar_Physics
tags: [solar-wind, kinetic-plasma, turbulence, microinstabilities, multi-scale, living-review]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 66. The Multi-Scale Nature of the Solar Wind / 태양풍의 다중 스케일 특성

---

## 1. Core Contribution / 핵심 기여

Verscharen, Klein, and Maruca (2019) provide a comprehensive 136-page Living Review establishing that the solar wind is fundamentally a **multi-scale magnetized plasma** whose dynamics and thermodynamics cannot be understood by any single theoretical framework. The review rigorously catalogues the characteristic length scales — Debye length $\lambda_D$, inertial lengths $d_j$, gyroradii $\rho_j$, mean free path $\lambda_{\text{mfp},j}$, and system size $L$ — and timescales — plasma periods $\Pi_{\omega_{pj}}$, gyroperiods $\Pi_{\Omega_j}$, collision time $\Pi_{\nu_c}$, expansion time $\tau$ — demonstrating that these span **over twelve orders of magnitude** from the coronal electron Debye length (~7 cm) to the 1-au heliocentric distance (~10¹¹ m). The central thesis is that **couplings between large-scale expansion and small-scale kinetic processes** — Coulomb collisions, wave–particle resonances (Landau and cyclotron), stochastic heating, and kinetic microinstabilities (firehose, mirror, ion-cyclotron, beam-driven A/IC and FM/W) — jointly govern the observed state of the plasma, including its non-Maxwellian distribution functions (core/beam/α-particle/halo/strahl), temperature anisotropies, differential streaming, and heat flux.

이 리뷰는 태양풍을 단일 이론으로 설명할 수 없는 본질적인 **다중 스케일 자화 플라즈마**로 확립하는 136쪽 분량의 종합 문헌이다. Debye 길이 $\lambda_D$, 관성 길이 $d_j$, 자이로 반지름 $\rho_j$, 평균 자유 경로 $\lambda_{\text{mfp},j}$, 시스템 크기 $L$로 이어지는 특성 길이 스케일과, 플라즈마 주기 $\Pi_{\omega_{pj}}$, 자이로 주기 $\Pi_{\Omega_j}$, 충돌 시간 $\Pi_{\nu_c}$, 팽창 시간 $\tau$로 이어지는 특성 시간 스케일이 코로나의 전자 Debye 길이(~7 cm)부터 1 au 거리(~10¹¹ m)에 이르기까지 **12자릿수 이상**에 걸쳐 있음을 체계적으로 정리한다. 핵심 논지는 **대형 팽창과 소형 운동론적 과정**(Coulomb 충돌, Landau·사이클로트론 공명 파동-입자 상호작용, 확률적 가열, 그리고 firehose·mirror·ion-cyclotron 등 운동 microinstability)이 **상호 결합**하여 관측되는 비Maxwellian 분포함수(코어·빔·알파·halo·strahl), 온도 비등방성, 차등 운동, 열유속을 결정한다는 것이다. 이 리뷰는 Parker Solar Probe 및 Solar Orbiter 시대의 해석 지도를 제공한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Characteristic Scales and Kinetic Framework / 특성 스케일과 운동론적 틀 (§1)

**Table 1 values at 1 au vs upper corona / 1 au와 상부 코로나의 특성값 대비**:

| Quantity / 양 | 1 au (Solar Wind) | Upper Corona (~100 Mm) |
|---|---|---|
| $n_p, n_e$ | 3 cm⁻³ | 10⁶ cm⁻³ |
| $T_p, T_e$ | 10⁵ K | 10⁶ K |
| $B$ | 3×10⁻⁵ G = 3 nT | 1 G |
| $\lambda_{\text{mfp},p}$ | 3 au | 100 Mm |
| $d_p$ | 140 km | 230 m |
| $\rho_p$ | 160 km | 13 m |
| $d_e$ | 3 km | 5 m |
| $\rho_e$ | 2 km | 30 cm |
| $\lambda_D$ | 12 m | 7 cm |
| $\Pi_{\nu_c}$ (proton collision time) | 120 d | 2 h |
| $\tau$ (expansion time) | 2.4 d | 10 min |
| $\Pi_{\Omega_p}$ (proton gyro-period) | 26 s | 660 μs |

Eq. (1) sets the expansion time $\tau \equiv L/U_r \approx 2.4$ d for $L = 1$ au and $U_r = 600$ km/s. The gyrofrequency $\Omega_j = q_j B_0/(m_j c)$ (Gaussian-cgs) gives $\Pi_{\Omega_p} \approx 26$ s. The **plasma frequency** $\omega_{pj} = \sqrt{4\pi n_{0j} q_j^2/m_j}$ yields $\Pi_{\omega_{pp}} \approx 3$ ms for protons at 1 au. The **Debye length** $\lambda_j = \sqrt{k_B T_j/(4\pi n_{0j} q_j^2)}$ is ~12 m at 1 au; the criterion $n_0 \lambda_D^3 \gg 1$ and $\lambda_D \ll L$ (Eqs. 12–13) confirms collective plasma behavior.

특성 스케일의 12자릿수 분리가 태양풍의 "본질적 다중 스케일성"을 규정한다. 1 au에서 $\rho_p \sim 160$ km, $d_p \sim 140$ km가 MHD → kinetic 전환 스케일을 설정한다.

**Vlasov–Maxwell 방정식 / Vlasov–Maxwell equations (Eqs. 20–24)**:
$$
\frac{\partial f_j}{\partial t} + \mathbf{v}\cdot\frac{\partial f_j}{\partial \mathbf{x}} + \frac{q_j}{m_j}\left(\mathbf{E} + \frac{1}{c}\mathbf{v}\times\mathbf{B}\right)\cdot\frac{\partial f_j}{\partial \mathbf{v}} = 0
$$
$$
\nabla\cdot\mathbf{E} = 4\pi\rho_c,\ \nabla\cdot\mathbf{B} = 0,\ \nabla\times\mathbf{E} = -\frac{1}{c}\frac{\partial \mathbf{B}}{\partial t},\ \nabla\times\mathbf{B} = \frac{4\pi}{c}\mathbf{j} + \frac{1}{c}\frac{\partial \mathbf{E}}{\partial t}
$$

충전 밀도와 전류는 분포함수 모멘트로 주어진다: $\rho_c = \sum_j q_j\int f_j d^3v$, $\mathbf{j} = \sum_j q_j\int \mathbf{v} f_j d^3v$ (Eqs. 25–26). 이것이 비충돌 플라즈마의 완전한 통계 기술. / The charge and current densities are moments of $f_j$, closing Vlasov–Maxwell for collisionless plasma.

**MHD closure (Eqs. 50–58)**. 모든 종 합산 후 $\rho = \sum m_j n_j$, $\mathbf{U} = \rho^{-1}\sum m_j n_j\mathbf{U}_j$, frozen-in 조건 $\mathbf{E} = -\mathbf{U}\times\mathbf{B}/c$ (Eq. 56). MHD는 (1) $\ell \lesssim L$, (2) $\ell \gg \max(d_j, \rho_j)$, (3) $t \gg \max(\Pi_{\Omega_j}, \Pi_{\omega_{pj}})$일 때만 유효. / MHD is valid only for scales well above the kinetic scales and times much longer than the gyro/plasma periods.

**Bi-Maxwellian distribution (Eq. 61)**:
$$f_{bM}(\mathbf{v}) = \frac{n_j}{\pi^{3/2}w_{\perp j}^2 w_{\|j}}\exp\left(-\frac{v_\perp^2}{w_{\perp j}^2} - \frac{(v_\| - U_{\|j})^2}{w_{\|j}^2}\right)$$

where $w_{\perp j} = \sqrt{2k_B T_{\perp j}/m_j}$ and $w_{\|j} = \sqrt{2k_B T_{\|j}/m_j}$. κ-분포 (Eq. 62)는 고에너지 꼬리가 강화된 비평형 분포: $f_\kappa \propto [1 + (2/(2\kappa-3))(v-U)^2/w^2]^{-\kappa-1}$, $\kappa > 3/2$. / The κ-distribution captures non-thermal high-energy tails observed in the electron halo.

**Ion features (§1.4.4)**: 양성자 분포는 종종 핵(core) + 빔(beam, $\Delta U \lesssim v_A$ 반-태양 방향) + α-입자(n_α ≲ 0.05 n_p, $T_\alpha \approx 4 T_{\|p}$)로 구성. Fast wind에서 $T_{\perp p} > T_{\|p}$ 비등방성이 우세. α-입자는 자기장 방향으로 $\Delta U_{\alpha p} \lesssim v_{Ap}$로 표류("surfing α-particles").

**Electron features (§1.4.5)**: 전자는 core(~95%, ~10 eV) + halo(κ-분포, ≲80 eV) + strahl(자기장 정렬 빔, ≲100 eV 반-태양 방향)로 구성. Halo와 strahl은 원일점 거리 따라 상호 변환(strahl → halo via scattering).

### Part II: Coulomb Collisions / Coulomb 충돌 (§3)

**Dimensional analysis (Eq. 92)**:
$$t_j = \frac{2^{3/2}m_j^{1/2}(k_B T_j)^{3/2}}{\pi q_j^4 n_j}$$

1 au, $n_p = 3$ cm⁻³, $T_p = 10^5$ K에서 $t_p \sim 10^8$ s ≈ 3 yr (팽창 시간 $\tau \sim 2.4$ d보다 훨씬 길다). 코로나 $n_p = 10^8$ cm⁻³, $T_p = 10^6$ K에서 $t_p \sim 350$ s (훨씬 짧음). / Coronal collisional time is dramatically shorter than solar-wind-transit collisional time — hence the transition from collisional to weakly-collisional plasma.

**Landau collision integral (Eq. 116)**:
$$\left(\frac{\delta f_j}{\delta t}\right)_{c,i} \approx \frac{2\pi q_j^2 q_i^2}{m_j}\ln\Lambda_{ji}\,\frac{\partial}{\partial \mathbf{v}_j}\cdot\left[\int d^3v_i\,\frac{\mathsf{I}_3 g_{ji}^2 - \mathbf{g}_{ji}\mathbf{g}_{ji}}{g_{ji}^3}\cdot\left(\frac{f_i(\mathbf{v}_i)}{m_j}\frac{\partial f_j}{\partial \mathbf{v}_j} - \frac{f_j(\mathbf{v}_j)}{m_i}\frac{\partial f_i}{\partial \mathbf{v}_i}\right)\right]$$

작은 각 산란 근사에서 유도되며, Coulomb 로그 $\ln\Lambda_{ji} = \ln(b_{\max}/b_{\min}) \approx 20\text{–}30$. $b_{\min} = q_j q_i/(k_B T_{ji})$, $b_{\max} = \lambda_D$. / The Coulomb integral in the small-angle limit; the Coulomb logarithm captures the log-divergent nature of small-angle scatterings.

**Coulomb number (Eq. 132)**: $N_c \equiv \tau/\tau_c = r/(U_r \tau_c)$ classifies "collisionally young" ($N_c \ll 1$) vs "collisionally old" ($N_c \gg 1$) plasma. Kasper et al. (2008, 2017)의 Brazil plots in Fig. 13 shows $|\Delta U_{\alpha p}|/v_{Ap}$, $T_\alpha/T_p$, $T_{\perp p}/T_{\|p}$, $T_{\perp\alpha}/T_{\|\alpha}$ all relax toward equilibrium as $N_c$ increases. / Coulomb number is the key observational diagnostic.

### Part III: Plasma Waves / 플라즈마 파동 (§4)

**Linear dispersion relation (Eq. 152)**: $\det[\mathcal{D}(\mathbf{k}, \omega)] = 0$ where the dispersion tensor is built from the dielectric tensor $\epsilon = 1 + \sum_j \chi_j$. Solutions give modes with complex frequency $\omega = \omega_r + i\gamma$; $\gamma < 0$ is damped, $\gamma > 0$ is unstable.

**Quasilinear diffusion (Eq. 154)**. 공명 조건 $v_{\text{res}} = (\omega_r - n\Omega_j)/k_\|$; $n = 0$ is Landau/transit-time damping, $n \neq 0$ is cyclotron damping. 입자는 반원 $(v_\| - \omega_r/k_\|)^2 + v_\perp^2 = \text{const}$에 접선 방향으로 확산. / Resonant particles diffuse along semi-circles in velocity space tangent to circles centered at $(\omega_r/k_\|, 0)$.

**Wave modes in §4.3**:
- **Large-scale Alfvén waves** (§4.3.1): $\omega = \pm |k_\||v_A^*$, δ**B** ⊥ **k**, δ**B** ⊥ **B**₀. Polarization $\delta\mathbf{U}/v_A^* = \mp \delta\mathbf{B}/B_0$. Fig. 17 shows Wind observation of perfect Alfvénic correlation for 7 h.
- **Kinetic Alfvén waves** (§4.3.2): $k_\perp\rho_p \gtrsim 1$, gyrokinetic dispersion $\omega = \pm|k_\||v_{Ap}k_\perp\rho_p/\sqrt{\beta_p + 2/(1+T_e/T_p)}$ (Eq. 165). KAWs heat ions perpendicularly at high β via stochastic heating; at low β, Landau-damp on electrons.
- **Alfvén/ion-cyclotron (A/IC) waves** (§4.3.3): $k_\|d_p \gtrsim 1$, left-hand polarized, quasi-parallel. Dispersion (Eq. 168): $\omega_r/\Omega_p = \pm(k^2 d_p^2/2)[\sqrt{1 + 4/(k^2 d_p^2)} - 1]$. Cyclotron resonance condition $\omega_r = k_\|v_\| + \Omega_p$ heats ions perpendicularly.
- **Slow modes** (§4.3.4): anti-correlated $\delta n_e$ and $\delta|\mathbf{B}|$; two kinetic branches: ion-acoustic ($\omega_r = \pm|k_\||\sqrt{(3k_B T_{\|p} + k_B T_{\|e})/m_p}$) and non-propagating mode ($\omega_r = 0$, mirror precursor).
- **Fast magnetosonic/whistler (FM/W)** (§4.3.5): positive correlation $\delta n_e$–$\delta|\mathbf{B}|$; at small scales becomes whistler with $\omega_r/|\Omega_e| = k|k_\||d_e^2/(1 + k^2d_e^2)$.

### Part IV: Plasma Turbulence / 플라즈마 난류 (§5)

**Kolmogorov spectrum (Eq. 180)**: $E(k) \sim \epsilon^{2/3}k^{-5/3}$ from $\mathcal{E} \sim (\delta U_\ell)^2$, $\tau_{nl} \sim \ell/\delta U_\ell$, $\epsilon = \mathcal{E}/\tau_{nl} = $ const.

**Observed Fig. 19 spectrum (Kiyani et al. 2015)**: $f^{-1}$ (injection, $f < 10^{-4}$ Hz), $f^{-5/3}$ (inertial, $10^{-4}$ to ~1 Hz), $f^{-2.8}$ (dissipation/kinetic, > 1 Hz). 스펙트럴 브레이크가 $f \sim 0.3$ Hz 근처에서 $\rho_p$, $d_p$ 스케일에 해당. / The MHD–kinetic break near 0.3 Hz corresponds to proton kinetic scales via Taylor's hypothesis.

**Iroshnikov–Kraichnan spectrum (Eq. 183)**: $E(k) \sim k^{-3/2}$ for wave-turbulence of counter-propagating Alfvén packets.

**Critical balance (Eq. 190)**: $\omega_r(k_\|, k_\perp) \sim k_\perp\delta U_\perp$, $\tau_{lin} \sim \tau_{nl}$. Goldreich–Sridhar의 강난류 폐쇄. Perpendicular spectrum $\sim k_\perp^{-5/3}$, parallel $\sim k_\|^{-2}$.

**Elsässer variables (Eq. 191)**: $\mathbf{z}^\pm \equiv \delta\mathbf{U} \mp \delta\mathbf{B}/\sqrt{4\pi\rho}$. Only $\mathbf{z}^+$–$\mathbf{z}^-$ 상호작용이 비선형 캐스케이드를 유발.

**Advanced topics (§5.4)**: Intermittency (PVI = $|\delta\mathbf{B}|/\sqrt{\langle\delta\mathbf{B}^2\rangle}$), current sheets driving magnetic reconnection, tearing instability cascading to even smaller scales (Loureiro & Boldyrev 2017).

### Part V: Kinetic Microinstabilities / 운동 microinstabilities (§6)

**Wave–particle instabilities driven by temperature anisotropy (§6.1.1)**. Hellinger et al. (2006) parametric threshold (Eq. 198):
$$\frac{T_{\perp j}}{T_{\|j}} = 1 + \frac{a}{(\beta_{\|j} - c)^b}$$

Fit parameters (Table 3) for $\gamma_m = 10^{-2}\Omega_p$:
- Ion-cyclotron (IC): $a = 0.649$, $b = 0.400$, $c = 0$. Drives parallel A/IC.
- Mirror: $a = 1.040$, $b = 0.633$, $c = -0.012$. Drives non-propagating oblique slow.
- Parallel firehose: $a = -0.647$, $b = 0.583$, $c = 0.713$. Drives parallel FM/W.
- Oblique firehose: $a = -1.447$, $b = 1.000$, $c = -0.148$. Drives non-propagating oblique Alfvén.

**Brazil plot (Fig. 21)**: Wind observations of $\beta_{\|p}$–$T_{\perp p}/T_{\|p}$ show that the observed distribution is bounded by mirror threshold (above) and oblique-firehose threshold (below), with IC and parallel-firehose thresholds providing weaker constraints. This directly establishes that microinstabilities regulate the solar wind's temperature-anisotropy state.

**Fluctuating-anisotropy effect (§6.3)**: Large-scale compressive fluctuations ($\delta|\mathbf{B}|/B_0 \gtrsim 0.04$) modulate $\beta_j$ and $T_{\perp j}/T_{\|j}$, pushing the plasma across instability thresholds intermittently.

**Nyquist criterion (§6.1.3, Eq. 199)**: $W_n = (2\pi i)^{-1}\oint d\omega/\det[\mathcal{D}]$ over upper half-plane; Klein et al. (2017) find ~50% of Wind intervals are unstable, ~10% with growth rates competing with turbulent cascade rate.

### Part VI: Conclusions / 결론 (§7)

**Fig. 24 coupling diagram**: Expansion → non-equilibrium features (anisotropy, beams) → microinstabilities → small-scale scattering → changes in bulk parameters; Expansion → reflection-driven waves → turbulence → cascade to small scales → Coulomb collisions → thermodynamics. All loops feed back on the large-scale evolution.

PSP (launched 2018) and Solar Orbiter (2020) are designed to directly test this multi-scale paradigm via near-Sun and polar observations.

### Part VII: In-situ Measurement Techniques / 직접 측정 기법 (§2)

**Thermal-particle instruments (§2.2)**: Three main types:
- **Faraday cups (FC)**: Multi-grid collectors that measure ion flux via modulated retarding potential. Advantage: no internal electronics, high signal-to-noise for dense populations (e.g., Wind/SWE, PSP/SPC). Limitation: mainly 1D energy cuts along spacecraft spin. / 전위차 변조를 통한 다중 격자 수집기.
- **Electrostatic analyzers (ESAs)**: Curved-plate analyzers with MCP detectors providing 3D phase-space coverage. Used by Helios/I1 E5, Wind/3DP, MMS/FPI. / 곡면 판 분석기와 MCP 검출기의 조합.
- **Mass spectrometers**: Combine E-field (ESA) and B-field or time-of-flight to separate species by m/q (e.g., ACE/SWICS, STEREO/PLASTIC). Essential for α, heavy ions. / 질량 대 전하 비 분리.

**Magnetometers (§2.4)**: Fluxgate (0–few kHz) for DC field; search-coil (few Hz–tens of kHz) for ion/electron kinetic fluctuations. Dual-fluxgate + search-coil combined on Wind/MFI, Cluster/FGM+STAFF-SC, MMS/DFG+SCM. / Fluxgate와 search-coil의 조합으로 DC부터 kHz까지 커버.

**Multi-spacecraft techniques (§2.6)**: Curlometry ($\nabla\times\mathbf{B}$ via 4 spacecraft, giving **j** from Ampère's law), wave-telescope (4-point Fourier analysis in k-space), discontinuity analysis (timing). Cluster/MMS constellations separated by ~10 km–10,000 km achieve kinetic-scale resolution. / 4-위성 편대가 곡률 및 k-공간 스펙트럼을 직접 결정.

### Part VIII: Additional Kinetic Details / 추가 운동론적 세부 사항

**Stochastic heating (§4.2.3)**: Non-resonant heating process when gyroscale fluctuations are so large that particle orbits become stochastic. Chandran et al. (2010) derived the low-β perpendicular heating rate (Eq. 166):
$$Q_\perp = c_1\frac{(\delta v_p)^3}{\rho_p}\exp\left(-\frac{c_2}{\hat\epsilon}\right)$$
with $\hat\epsilon \equiv \delta v_p/w_{\perp p}$, $c_1 \approx 0.75$, $c_2 \approx 0.34$. Hoppock et al. (2018) extended to $1 \lesssim \beta_p \lesssim 30$ (Eq. 167). / 자이로 스케일 변동이 커지면 궤도가 확률적이 되어 수직 방향 가열.

**Entropy cascade (§4.2.2)**: Schekochihin et al. (2008) framework for $dS_j/dt$; energy in $(E^2 + B^2)$ cascades to entropy fluctuations $\delta f^2/f_0$ via linear (Landau) and nonlinear (perpendicular gyro-radius scale) phase mixing. The nonlinear phase mixing decorrelates particles with different $v_\perp$ on gyroradius scales, generating small-scale velocity structure that collisions ultimately dissipate. / 엔트로피 캐스케이드: 전자장 에너지 → 엔트로피 요동 → 충돌 소산.

**Observed temperature profiles (Fig. 7)**: Fast wind $T_{\|p}$ goes from ~0.25 MK at 0.3 au down to ~0.10 MK at 1 au; $T_{\perp p}$ from ~0.7 MK to ~0.25 MK. Both fall more slowly than CGL prediction ($T_\perp \propto r^{-2}$ approximately), indicating continuous heating in the inner heliosphere. / 관측된 온도 감소가 CGL 예측보다 느림 → 지속적 가열.

**Alpha-particle energy release (Fig. 22)**: Between 0.3 and 0.4 au, $Q_{\text{flow}}$ from α-deceleration exceeds the empirical $Q_{\perp p}$ heating rate, suggesting α-driven instabilities (FM/W, A/IC) convert ~5–10% of total solar-wind energy into wave fluctuations that subsequently heat the plasma. / α-감속 에너지가 내부 태양권에서 주요 가열원.

**Radial evolution of spectral breaks**: Spectral break between injection and inertial range decreases with $r$ as $f_{b1} \propto r^{-1.5}$; inertial/dissipation break as $f_{b2} \propto r^{-1.09}$ (Bruno & Trenchi 2014). / 스펙트럴 브레이크가 거리에 따라 체계적으로 이동.

**Wave–wave instabilities (§6.2)**: Parametric decay of a parent Alfvén wave into forward slow-mode + backward Alfvén daughter, important for turbulence generation from outward-propagating Alfvén waves in fast wind. Constraint on large-amplitude magnetic fluctuations from Squire et al. (2017): $\delta B/B_0 \lesssim \sqrt{\beta_p}$ before parametric instability disrupts the wave. / Alfvén 부모파가 slow + Alfvén로 붕괴하는 파라메트릭 분해.

**PIC and hybrid simulations**: The review emphasizes that numerical simulations are essential complements to observations. Particle-in-cell (PIC) simulations resolve full kinetic dynamics but are computationally limited to small volumes; hybrid simulations (kinetic ions + fluid electrons) reach larger scales. Expanding-box simulations (Hellinger et al.) mimic solar-wind expansion in a Lagrangian frame, providing tests of instability thresholds in non-stationary plasma. / PIC, 하이브리드, expanding-box 시뮬레이션이 관측의 보완.

**Pristine solar wind selection**: The Brazil plots use "pristine" data excluding interplanetary coronal mass ejections (ICMEs) and their sheath/shock interfaces. Such careful filtering ensures the plotted distribution reflects intrinsic wind conditions, not transient disturbances. / "원시" 태양풍 선택으로 ICME 영향 배제.

### Part IX: Open Problems / 미해결 문제 (§1.4.6, §7)

The review identifies several outstanding problems that remain challenges for the PSP/SO era:
1. **Coronal heating**: The mechanisms heating the solar corona to $\sim 10^6$ K remain disputed — wave dissipation (Alfvén-wave turbulence, A/IC), reconnection nanoflares, and type-II spicules are all candidates. / 코로나 가열 메커니즘.
2. **Solar-wind acceleration**: How bulk $U_r$ reaches 300–800 km/s within ~10 R_⊙ is still debated (wave pressure, reconnection, exospheric kinetic effects). / 태양풍 가속 원인.
3. **Strahl formation**: Why electron strahl peaks at ≲100 eV with such a narrow pitch-angle, and how it scatters into halo with distance. / Strahl 형성과 산란.
4. **Fluctuating anisotropy**: Whether large-amplitude compressive modes systematically trigger microinstabilities at 1 au and beyond (§6.3). / 요동하는 비등방성.
5. **Origin of slow wind**: Closed-loop release (interchange reconnection) vs streamer-belt detachment vs other sources. / 느린 태양풍의 기원.
6. **Reconnection role**: How reconnection interrupts the turbulent cascade (Boldyrev & Loureiro 2017, Mallet et al. 2017) quantitatively. / 재결합이 캐스케이드를 중단시키는 기작.
7. **Parker spiral deviations**: Observed Parker-angle deviations at high latitudes due to α-particle deceleration and wave pressure (Verscharen et al. 2015). / Parker 나선의 편차.

### Part X: Standalone Interpretive Guide / 독립적 해석 가이드

To explain the review's core argument to a colleague who has not read the original paper:

**Step 1 — Why multi-scale?** The solar wind's collisional mean free path at 1 au (~3 au) is comparable to the system size, so ordinary fluid theories (collisional MHD, Navier–Stokes) fail. Yet the proton gyroradius (~160 km) is 10⁹ times smaller than 1 au, so full 6D Vlasov simulations across the whole heliosphere are computationally impossible. The plasma is genuinely multi-scale: no single theory captures it all. / 충돌 평균 자유 경로가 시스템 크기에 맞먹을 만큼 길지만, 자이로 반지름은 10⁹배 작음 → 단일 이론 불가.

**Step 2 — Why kinetic features persist?** With $N_c \sim 0.04$ at 1 au (from our worked example), there are only ~25 collisional times between the Sun and Earth — insufficient to fully Maxwellize the distribution. Hence the solar wind retains "memory" of coronal processes (acceleration, heating) in its non-Maxwellian features. / $N_c$가 작아 코로나 과정의 흔적이 분포 함수에 남음.

**Step 3 — Why do microinstabilities matter?** Even weakly collisional plasmas can generate their own "effective collisionality" via microinstabilities. When anisotropy $T_\perp/T_\|$ exceeds the mirror/firehose threshold, unstable waves grow on timescales $\sim \Omega_p^{-1} \sim 10$ s (far faster than $\tau \sim 2.4$ d), pitch-angle scattering particles back toward marginal stability. This is why observations in Fig. 21 cluster against the thresholds. / 불안정성이 "효과적 충돌"로 작용하여 한계 안정성 유지.

**Step 4 — Why is turbulence the energy conduit?** Kolmogorov-like cascade carries energy from injection (large-scale shear, stream interaction) through inertial range ($k^{-5/3}$) down to kinetic scales ($k_\perp\rho_p \sim 1$), where Landau damping (KAW on electrons) and cyclotron heating (A/IC on ions) convert wave energy into particle heat. The energy flux $\epsilon \sim (\delta U)^3/\ell \sim 10^{-16}$ erg/cm³/s matches empirical solar-wind heating rates. / 난류가 에너지를 대형에서 운동 스케일로 전달하고 파동-입자 공명이 가열.

**Step 5 — Why do PSP/SO matter?** All existing observations come from 0.3–100 au; the acceleration region (<0.3 au) was unmeasured until PSP (2018). PSP/SO directly test whether the multi-scale coupling paradigm scales correctly into the corona and upper heliosphere where the plasma is denser, hotter, and more magnetized. / PSP/SO가 <0.3 au 관측으로 본 패러다임을 직접 검증.

### Part XI: Quantitative Summary of Observational Highlights / 관측 하이라이트의 수치적 요약

**Helios at 0.3 au**: Proton velocity distributions (Fig. 5) show clear $T_\perp > T_\|$ anisotropy with oblate isosurfaces; fast-wind proton beams with $\Delta U/v_{Ap} \sim 1$. Proton temperatures ~0.7 MK (vs 0.25 MK at 1 au). / Helios 0.3 au에서 비등방성과 빔 관측.

**Ulysses polar orbit (1994–1997)**: Confirmed that fast wind emerges from polar coronal holes during solar minimum (Fig. 3). Density $n_p\cdot r^2$ in fast wind ~2 cm⁻³·(au)², factor ~2.5 lower than slow wind. / Ulysses가 극지방 fast wind 확인.

**Wind/MFI + SWE**: Provided the canonical Brazil plots (Fig. 13, Fig. 21) from 2.1 million data points over ~20 years. Critical for establishing both the Coulomb-number relaxation trends and the instability-threshold boundaries. / Wind가 210만 데이터로 Brazil plot의 표준 기준.

**Cluster multi-spacecraft**: Enabled direct measurement of 3D wavevector spectra (Sahraoui et al. 2010), confirming $k_\perp$-anisotropic cascade. STAFF-SC extended spectrum to electron scales. / Cluster가 3D k-스펙트럼 직접 측정.

**PSP first perihelia (Nov 2018)**: Detected magnetic field "switchbacks" — sudden reversals of $B_r$ embedded in Alfvénic flow (Bale et al. 2019). Implications for interchange reconnection and coronal-hole boundary dynamics. / PSP가 switchback 발견, interchange reconnection 시사.

---

## 3. Key Takeaways / 핵심 시사점

1. **12 orders of magnitude / 12 자릿수**: The solar wind spans from $\lambda_D \sim 7$ cm (coronal electron Debye length) to $L \sim 1$ au ($\sim 1.5\times10^{11}$ m), making it intrinsically multi-scale. No single theory (MHD alone or Vlasov alone) suffices. / 태양풍은 본질적으로 12자릿수 이상 스케일에 걸친 시스템으로, 단일 이론으로 기술 불가능.

2. **Collisions never truly vanish / 충돌은 결코 완전히 사라지지 않는다**: Though $N_c \ll 1$ at 1 au for many parameters (collisional age $A_c \sim 10^{-2}$ – 10), Coulomb collisions still modulate $T_{\perp\alpha}/T_{\|\alpha}$, $T_\alpha/T_p$, $|\Delta U_{\alpha p}|/v_{Ap}$, as revealed by the Brazil plots of Kasper et al. (2008, 2017). / 1 au에서도 약 충돌성이지만 Coulomb 충돌이 α-양성자 표류, 온도비, 비등방성을 꾸준히 이완시킴.

3. **Alfvénic turbulence is anisotropic / 알펜 난류는 비등방성**: Wind and Cluster observations confirm $k_\perp \gg k_\|$ with $E(k_\perp) \sim k_\perp^{-5/3}$ and $E(k_\|) \sim k_\|^{-2}$, matching critical-balance predictions. / 관측이 critical-balance의 비등방 스펙트럼을 직접 검증.

4. **KAW and A/IC are complementary dissipation channels / KAW과 A/IC는 상보적 에너지 소산 채널**: KAW (k_⊥ρ_p ≳ 1) dominates ion perpendicular heating at high β via stochastic heating and electron Landau damping; A/IC (k_∥d_p ≳ 1) heats ions via cyclotron resonance $\omega_r = k_\|v_\| + \Omega_p$. / KAW은 수직 가열, A/IC은 사이클로트론 공명 가열; 태양풍 가열 이중 메커니즘.

5. **Microinstabilities set the boundaries of parameter space / 미세 불안정성이 파라미터 공간의 경계를 설정**: The $\beta_{\|p}$–$T_{\perp p}/T_{\|p}$ "Brazil plot" shows the pristine solar wind tightly bounded by mirror (upper) and oblique-firehose (lower) thresholds — unstable growth rapidly drives the plasma back to marginal stability. / Mirror·firehose 문턱이 실제 관측 공간의 경계 — 불안정성이 준평형 상태를 유지.

6. **Non-Maxwellian features are universal / 비Maxwellian 특징은 보편적**: Ion beams ($\Delta U \lesssim v_{Ap}$, anti-Sunward), temperature anisotropies ($T_\perp/T_\|$ both > 1 and < 1), $\alpha$-particle drifts, and electron halo/strahl (κ-distribution) are all direct fingerprints of kinetic multi-scale physics and must be explained by any viable heating/acceleration model. / 이온 빔, α-표류, 전자 halo/strahl 등은 가열·가속 모델이 설명해야 할 보편적 관측 사실.

7. **Critical balance explains observed anisotropy / critical balance가 관측된 비등방성을 설명**: $\tau_{lin} \sim \tau_{nl}$ condition yields $k_\perp v_A \sim k_\perp\delta U$, building up $k_\perp$-anisotropy through the cascade — consistent with observed 2D+slab turbulence composition. / critical balance가 관측된 2D 우세 난류 구조와 일치.

8. **PSP/SO era will test multi-scale coupling in coronal conditions / PSP·SO 시대가 코로나 조건에서 다중 스케일 결합을 검증한다**: Parker Solar Probe's perihelion (<20 R_⊙, inside the Alfvén point) and Solar Orbiter's polar orbit will measure the same kinetic physics in dramatically different regimes, testing whether the "multi-scale paradigm" scales correctly into the near-Sun environment. / PSP·SO가 Alfvén 임계점 안쪽과 극지방에서 이 패러다임을 직접 검증.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Kinetic framework / 운동론 틀

**Vlasov–Maxwell system (Eqs. 20–26)**:
$$
\frac{\partial f_j}{\partial t} + \mathbf{v}\cdot\frac{\partial f_j}{\partial \mathbf{x}} + \frac{q_j}{m_j}\left(\mathbf{E} + \frac{\mathbf{v}\times\mathbf{B}}{c}\right)\cdot\frac{\partial f_j}{\partial \mathbf{v}} = 0
$$
coupled with Maxwell's equations and the moment integrals $\rho_c = \sum_j q_j\int f_j d^3v$, $\mathbf{j} = \sum_j q_j\int \mathbf{v} f_j d^3v$. / 6차원 위상공간 연속 방정식과 Maxwell 방정식의 결합.

**Key length and time scales**:
$$\Omega_j = \frac{q_j B_0}{m_j c},\ \omega_{pj} = \sqrt{\frac{4\pi n_{0j} q_j^2}{m_j}},\ d_j = \frac{c}{\omega_{pj}},\ \rho_j = \frac{w_{\perp j}}{|\Omega_j|},\ \lambda_D = \sqrt{\frac{k_B T_j}{4\pi n_{0j} q_j^2}}$$

Connections: $d_j = v_{A,j}/|\Omega_j|$ with $v_{A,j} = B_0/\sqrt{4\pi n_{0j}m_j}$. / 관성 길이는 Alfvén 속도와 자이로 주파수의 비.

### 4.2 MHD closure / MHD 폐쇄

Continuity, momentum, induction:
$$\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho\mathbf{U}) = 0,\quad \rho\left(\frac{\partial}{\partial t} + \mathbf{U}\cdot\nabla\right)\mathbf{U} = -\nabla P + \frac{\mathbf{j}\times\mathbf{B}}{c},\quad \frac{\partial\mathbf{B}}{\partial t} = \nabla\times(\mathbf{U}\times\mathbf{B})$$

closed by adiabatic relation $(\partial/\partial t + \mathbf{U}\cdot\nabla)(P/\rho^\kappa) = 0$. Double-adiabatic (CGL) for anisotropic pressure: $p_{\perp j}/(n_j B) =$ const, $B^2 p_{\|j}/n_j^3 =$ const (Eq. 46). / CGL 이중단열 보존량: 팽창 시 $T_\perp \propto B$, $T_\| \propto 1/n^2$.

### 4.3 Plasma dispersion and damping / 분산 관계와 감쇠

**Dispersion tensor (Eq. 151)**:
$$\mathcal{D} = \begin{pmatrix} \epsilon_{xx} - n_z^2 & \epsilon_{xy} & \epsilon_{xz} + n_x n_z \\ \epsilon_{yx} & \epsilon_{yy} - n_x^2 - n_z^2 & \epsilon_{yz} \\ \epsilon_{zx} + n_z n_x & \epsilon_{zy} & \epsilon_{zz} - n_x^2 \end{pmatrix}$$

with $\mathbf{n} \equiv \mathbf{k}c/\omega$. The dispersion relation is $\det[\mathcal{D}] = 0$. / 분산 관계는 분산 텐서의 행렬식 = 0.

**Quasilinear diffusion (Eq. 154)**:
$$\frac{\partial f_{0j}}{\partial t} = \frac{q_j^2}{8\pi^2 m_j^2}\lim_{V\to\infty}\frac{1}{V}\sum_{n=-\infty}^{+\infty}\int d^3k\,\hat{G}v_\perp\frac{1}{v_\perp}\delta(\omega_r - k_\|v_\| - n\Omega_j)\,|\psi_n|^2\hat{G}f_{0j}$$

공명 조건: $v_{\text{res}} = (\omega_r - n\Omega_j)/k_\|$. $n = 0$: Landau; $n \neq 0$: cyclotron. / Resonance: $n=0$ Landau; $n \neq 0$ cyclotron.

**Kinetic Alfvén wave (Eq. 165)**:
$$\omega = \pm\frac{|k_\||v_{Ap} k_\perp\rho_p}{\sqrt{\beta_p + 2/(1 + T_e/T_p)}}$$

Gyrokinetic limit for isotropic temperatures; valid at $k_\perp\rho_p \gtrsim 1$, $k_\perp \gg k_\|$. / KAW은 $k_\perp\rho_p \gtrsim 1$에서 유효.

**A/IC in cold plasma (Eq. 168)**:
$$\frac{\omega_r}{\Omega_p} = \pm\frac{k^2 d_p^2}{2}\left(\sqrt{1 + \frac{4}{k^2 d_p^2}} - 1\right)$$

Left-hand polarized; cutoff at $\Omega_p$; cyclotron resonance $\omega_r = k_\|v_\| + \Omega_p$. / 좌편광, $\Omega_p$에서 컷오프.

**Slow-mode ion-acoustic branch (Eq. 172)**:
$$\omega_r = \pm|k_\||\sqrt{\frac{3k_B T_{\|p} + k_B T_{\|e}}{m_p}}$$

### 4.4 Turbulence / 난류

**Inertial spectrum (Eq. 180)**: $E(k) \sim \epsilon^{2/3}k^{-5/3}$.

**Taylor hypothesis (Eq. 182)**: $f_{sc} \approx (2\pi)^{-1}\mathbf{k}\cdot\Delta\mathbf{U}$ converts spacecraft frequency to wavenumber.

**Critical balance (Eq. 190)**: $\omega_r(k_\|, k_\perp) \sim k_\perp\delta U_\perp$ implies $k_\perp \sim k_\|^{3/2}$-like anisotropy.

**Elsässer formulation (Eqs. 191–194)**:
$$\mathbf{z}^\pm = \delta\mathbf{U} \mp \delta\mathbf{B}/\sqrt{4\pi\rho},\ \frac{\partial \mathbf{z}^\pm}{\partial t} \pm (\mathbf{v}_A^*\cdot\nabla)\mathbf{z}^\pm = -(\mathbf{z}^\mp\cdot\nabla)\mathbf{z}^\pm - \frac{\nabla P_{\text{tot}}}{\rho}$$

### 4.5 Microinstability thresholds / 불안정성 문턱

**Parametric bi-Maxwellian threshold (Eq. 198)**:
$$\frac{T_{\perp j}}{T_{\|j}} = 1 + \frac{a}{(\beta_{\|j} - c)^b}$$

Coefficients from Table 3 (γ_m = 10⁻²Ω_p):

| Instability | a | b | c |
|---|---|---|---|
| Ion-cyclotron | 0.649 | 0.400 | 0 |
| Mirror | 1.040 | 0.633 | −0.012 |
| Parallel firehose | −0.647 | 0.583 | 0.713 |
| Oblique firehose | −1.447 | 1.000 | −0.148 |

**Nyquist stability criterion (Eq. 199)**:
$$W_n = \frac{1}{2\pi i}\oint \frac{d\omega}{\det[\mathcal{D}(\mathbf{k}, \omega)]}$$

$W_n = $ number of unstable modes at wavevector **k**.

### 4.6 Worked numerical example at 1 au / 1 au에서의 수치 예제

Given $n_p = 5$ cm⁻³, $T_p = 10^5$ K, $B_0 = 5$ nT, $m_p = 1.673\times10^{-24}$ g:
- Proton gyrofrequency: $\Omega_p = eB_0/(m_p c) = (4.8\times10^{-10})(5\times10^{-5})/(1.673\times10^{-24}\cdot 3\times10^{10}) \approx 0.48$ rad/s → $\Pi_{\Omega_p} = 2\pi/\Omega_p \approx 13$ s.
- Proton thermal speed: $w_{\perp p} = \sqrt{2k_B T_p/m_p} = \sqrt{2(1.38\times10^{-16})(10^5)/(1.673\times10^{-24})} \approx 4.1\times10^6$ cm/s = 41 km/s.
- Proton Larmor radius: $\rho_p = w_{\perp p}/\Omega_p \approx 41 \text{ km} / 0.48 \approx 85$ km (within factor 2 of the Table 1 value 160 km; difference due to using $T_p = 10^5$ K vs typical $T_p \sim 2\times10^5$ K in fast wind).
- Alfvén speed: $v_A = B_0/\sqrt{4\pi n_p m_p} = 5\times10^{-5}/\sqrt{4\pi\cdot 5\cdot 1.673\times10^{-24}} \approx 4.9\times10^6$ cm/s ≈ 49 km/s.
- Plasma β: $\beta_p = 8\pi n_p k_B T_p/B_0^2 = 8\pi(5)(1.38\times10^{-16})(10^5)/(5\times10^{-5})^2 \approx 0.70$.
- Spectral break frequency via Taylor: $f_b = U_r/(2\pi\rho_p) \approx (400 \text{ km/s})/(2\pi\cdot 85 \text{ km}) \approx 0.75$ Hz (consistent with observed ~0.3–1 Hz break).

### 4.7 Collisional timescale at 1 au / 1 au에서 Coulomb 충돌 시간

$$t_p = \frac{2^{3/2}m_p^{1/2}(k_B T_p)^{3/2}}{\pi e^4 n_p\ln\Lambda}$$

With $n_p = 5$ cm⁻³, $T_p = 10^5$ K, $\ln\Lambda = 23$:
$t_p \approx \frac{2^{3/2}(1.673\times10^{-24})^{1/2}(1.38\times10^{-16}\cdot 10^5)^{3/2}}{\pi(4.8\times10^{-10})^4(5)(23)} \approx 5.7\times10^6\text{ s} \approx 66\text{ days}.$

Coulomb number: $N_c = \tau/t_p = (2.4\text{ d})/(66\text{ d}) \approx 0.04$, confirming weakly collisional at 1 au. / 1 au는 $N_c \sim 0.04$로 약 충돌성.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1942 ──── Alfvén: existence of electromagnetic-hydrodynamic waves
          알펜파의 존재 증명
1946 ──── Landau: collisionless damping
          비충돌 감쇠 (Landau damping)
1956 ──── Chew, Goldberger, Low: double-adiabatic (CGL) theory
          이중단열 이론 (CGL)
1958 ──── Parker: supersonic solar wind model
          초음속 태양풍 모델
1962 ──── Mariner 2 / Neugebauer & Snyder: first in-situ SW observations
          최초 태양풍 직접 관측
1965 ──── Kraichnan: IK k^{-3/2} turbulence spectrum
          IK 난류 스펙트럼
1974 ──── Launch of Helios 1 (0.29 au perihelion)
          Helios 1 발사 (근일점 0.29 au)
1993 ──── Gary: temperature-anisotropy instability compendium
          온도 비등방성 불안정성 종합
1995 ──── Goldreich & Sridhar: critical balance in MHD turbulence
          MHD 난류의 critical balance
2006 ──── Howes et al.: gyrokinetic cascade model
          자이로운동 캐스케이드 모델
2008 ──── Kasper et al.: Coulomb number diagnostic
          Coulomb 수 진단법
2009 ──── Bale et al.: β_∥–T_⊥/T_∥ stability thresholds observed
          문턱 관측 (Brazil plot)
2015 ──── Kiyani et al.: solar-wind spectrum across 8 decades
          8자릿수에 걸친 태양풍 스펙트럼
2018 ──── PSP launch; first perihelion November 2018
          PSP 발사, 첫 근일점
★ 2019 ── Verscharen, Klein, Maruca: THIS REVIEW ★
          본 리뷰
2020 ──── Solar Orbiter launch
          Solar Orbiter 발사
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Parker (1958) | Supersonic solar wind | Foundational model that this review generalizes to multi-scale. 본 리뷰가 일반화하는 기초 모델. |
| Chew, Goldberger, Low (1956) | Double-adiabatic invariants | CGL predicts $T_{\perp} \propto B$, $T_\| \propto n^{-2}$; review shows deviations driven by microinstabilities. 관측된 CGL 편차가 불안정성 증거. |
| Goldreich & Sridhar (1995) | Critical balance | Central concept for anisotropic Alfvénic turbulence explained in §5.3. §5.3의 핵심 개념. |
| Gary (1993), Hellinger et al. (2006) | Kinetic instabilities | Threshold fits (Eq. 198, Table 3) directly imported. 문턱 식이 직접 인용됨. |
| Howes et al. (2006, 2008) | Gyrokinetic turbulence | KAW dispersion (Eq. 165) and cascade picture. KAW 분산 및 캐스케이드. |
| Kasper et al. (2008, 2017) | Collisional age | Brazil plots (Fig. 13) frame §3.3. Brazil plots이 §3.3의 틀을 제공. |
| Bale et al. (2009) | β_∥–T_⊥/T_∥ plot | Observational basis of Fig. 21. Fig. 21의 관측 기초. |
| Marsch (2006) | Kinetic solar-wind review | Previous comprehensive treatment; this review updates with 2006–2019 progress. 2006–2019 진전을 반영한 업데이트. |
| Bruno & Carbone (2013) | Solar-wind turbulence review | Complementary focus on turbulence alone. 난류만을 집중 다룬 보완 리뷰. |

---

## 7. References / 참고문헌

- Verscharen, D., Klein, K. G., & Maruca, B. A., "The multi-scale nature of the solar wind", Living Reviews in Solar Physics, 16:5, 2019. DOI: 10.1007/s41116-019-0021-0
- Parker, E. N., "Dynamics of the interplanetary gas and magnetic fields", Astrophys. J., 128, 664, 1958.
- Chew, G. F., Goldberger, M. L., & Low, F. E., "The Boltzmann equation and the one-fluid hydromagnetic equations in the absence of particle collisions", Proc. R. Soc. London A, 236, 112, 1956.
- Goldreich, P., & Sridhar, S., "Toward a theory of interstellar turbulence. II. Strong Alfvenic turbulence", Astrophys. J., 438, 763, 1995.
- Gary, S. P., "Theory of Space Plasma Microinstabilities", Cambridge Univ. Press, 1993.
- Hellinger, P., Trávníček, P., Kasper, J. C., & Lazarus, A. J., "Solar wind proton temperature anisotropy: Linear theory and WIND/SWE observations", Geophys. Res. Lett., 33, L09101, 2006.
- Howes, G. G., Cowley, S. C., Dorland, W., et al., "Astrophysical gyrokinetics: Basic equations and linear theory", Astrophys. J., 651, 590, 2006.
- Kasper, J. C., Lazarus, A. J., & Gary, S. P., "Hot solar-wind helium: Direct evidence for local heating by Alfvén-cyclotron dissipation", Phys. Rev. Lett., 101, 261103, 2008.
- Kasper, J. C., & Klein, K. G., "Strong preferential ion heating is limited to within the solar Alfvén surface", Astrophys. J. Lett., 877, L35, 2019.
- Bale, S. D., Kasper, J. C., Howes, G. G., et al., "Magnetic fluctuation power near proton temperature anisotropy instability thresholds in the solar wind", Phys. Rev. Lett., 103, 211101, 2009.
- Kiyani, K. H., Osman, K. T., & Chapman, S. C., "Dissipation and heating in solar wind turbulence", Philos. Trans. R. Soc. A, 373, 20140155, 2015.
- Marsch, E., "Kinetic physics of the solar corona and solar wind", Living Rev. Sol. Phys., 3, 1, 2006.
- Bruno, R., & Carbone, V., "The solar wind as a turbulence laboratory", Living Rev. Sol. Phys., 10, 2, 2013.
- Fox, N. J., Velli, M. C., Bale, S. D., et al., "The Solar Probe Plus mission: Humanity's first visit to our star", Space Sci. Rev., 204, 7, 2016.
- Müller, D., Marsden, R. G., St. Cyr, O. C., & Gilbert, H. R., "Solar Orbiter: Exploring the Sun-heliosphere connection", Sol. Phys., 285, 25, 2013.
- Landau, L. D., "On the vibrations of the electronic plasma", J. Phys. USSR, 10, 25, 1946.
- Alfvén, H., "Existence of electromagnetic-hydrodynamic waves", Nature, 150, 405, 1942.
- Kraichnan, R. H., "Inertial-range spectrum of hydromagnetic turbulence", Phys. Fluids, 8, 1385, 1965.
- Neugebauer, M., & Snyder, C. W., "Solar plasma experiment", Science, 138, 1095, 1962.
- Kolmogorov, A. N., "The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers", Dokl. Akad. Nauk SSSR, 30, 301, 1941.
