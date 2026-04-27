---
title: "Solar Force-Free Magnetic Fields — Reading Notes"
date: 2026-04-27
topic: Solar_Physics
paper_number: 35
authors: Thomas Wiegelmann, Takashi Sakurai
year: 2012
journal: Living Reviews in Solar Physics, 9, 5
doi: 10.12942/lrsp-2012-5
tags: [force-free, NLFFF, LFFF, potential-field, coronal-extrapolation, optimization, Grad-Rubin, magnetofrictional, vector-magnetogram, ambiguity-removal, preprocessing, review]
---

# Solar Force-Free Magnetic Fields — Reading Notes / 읽기 노트

## 1. Core Contribution / 핵심 기여

**English.** This Living Reviews article by Wiegelmann & Sakurai (2012) is the canonical comprehensive review of solar force-free magnetic field theory and the numerical methods used to reconstruct the three-dimensional coronal magnetic field from photospheric vector magnetograms. The corona is dominated by magnetic forces (plasma $\beta\ll 1$), so to leading order the Lorentz force vanishes, $\mathbf{j}\times\mathbf{B}=\mathbf{0}$, leading to either current-free *potential* fields or the more general *force-free* condition $\nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B}$ with $\mathbf{B}\cdot\nabla\alpha=0$. The review systematically covers (i) the mathematics and Seehafer–Fourier construction of linear force-free fields (LFFF, $\alpha$ constant), (ii) analytic and semi-analytic nonlinear force-free benchmarks (Low–Lou 1990, Titov–Démoulin 1999), (iii) the practical pipeline of vector-magnetogram measurement, 180° azimuth ambiguity removal, consistency checks, and preprocessing toward force-free boundaries, (iv) global properties (helicity, energy bounds, stability) of force-free fields, and (v) the five families of 3D NLFFF numerical solvers — upward integration, Grad–Rubin, MHD relaxation / magnetofrictional, optimization, and boundary-element. It synthesizes the NLFFF Consortium benchmark results (Schrijver et al. 2006, 2008; DeRosa et al. 2009), critically discusses sources of disagreement between codes, and lays out the recipe by which modern coronal magnetic-field studies estimate magnetic free energy and helicity for flare/CME forecasting.

**한국어.** Wiegelmann & Sakurai (2012)의 본 Living Reviews 논문은 태양 무력장(force-free field) 이론과 광구 벡터 자기도로부터 3차원 코로나 자기장을 재구성하는 수치 기법에 대한 표준적이고 종합적인 리뷰입니다. 코로나는 자기력이 압도적인 영역(plasma $\beta\ll 1$)이므로 Lorentz 힘이 0에 가까워야 하며, 이는 무전류(potential) 자기장 혹은 보다 일반적인 무력장 조건 $\nabla\times\mathbf{B}=\alpha(\mathbf{r})\mathbf{B}$ ($\mathbf{B}\cdot\nabla\alpha=0$)로 표현됩니다. 본 리뷰는 (i) 선형 무력장(LFFF, $\alpha$ 상수) 수학과 Seehafer–Fourier 구성, (ii) 해석적·반해석적 비선형 무력장 벤치마크 (Low–Lou 1990, Titov–Démoulin 1999), (iii) 벡터 자기도 관측, 180° azimuth 모호성 제거, 일관성 점검, 무력장 호환 preprocessing의 실용 파이프라인, (iv) 무력장의 헬리시티·에너지 상한·안정성 등 전역적 성질, (v) 3차원 NLFFF 수치 해법 다섯 종 — upward integration, Grad–Rubin, MHD relaxation/magnetofrictional, optimization, boundary-element — 을 체계적으로 정리합니다. 또한 NLFFF Consortium 벤치마크(Schrijver 등 2006, 2008; DeRosa 등 2009)를 종합·해설하고, 코드 간 불일치 원인을 논의하며, 자기 자유에너지·헬리시티 추정을 통한 플레어/CME 예보 응용의 가이드라인을 제시합니다.

---

## 2. Reading Notes / 읽기 노트

### 2.1 Section 1 — Introduction / 서론 (pp. 5–9)

**EN.** The corona's structure is "magnetic-field dominated" and accurate vector measurements are limited to the photosphere; thus the standard procedure is *extrapolation* upward from photospheric boundary conditions. Figure 1 illustrates the impact of solar storms on Earth's magnetosphere; Figure 2 (from Gary 2001) shows the famous plasma-$\beta$ height profile: $\beta\sim 0.1$–$10^2$ at the photosphere, $\beta\ll 1$ between $\sim 2$ Mm and $\sim 2$ Gm, then $\beta>1$ again in the solar wind acceleration region. Inside the low-$\beta$ sandwich, the static MHD momentum equation reduces (to leading order in $\beta$) to $\mathbf{j}\times\mathbf{B}=\mathbf{0}$ (Eq. 2). With $\mu_0\mathbf{j}=\nabla\times\mathbf{B}$ and $\nabla\cdot\mathbf{B}=0$, this is fulfilled either by $\nabla\times\mathbf{B}=0$ (current-free / potential) — Eq. (7) — or by $\mathbf{B}\parallel\nabla\times\mathbf{B}$, written $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ with $\mathbf{B}\cdot\nabla\alpha=0$ from the divergence-free condition. Potential fields require only LOS magnetograms ($B_z$ on the photosphere) and are computed by solving Laplace's equation $\Delta\phi=0$ with $\mathbf{B}=-\nabla\phi$. They give a coarse global view (e.g., the PFSS model with source surface at $\sim 2.5\,R_\odot$, Schatten et al. 1969) but contain no free energy and disagree with TRACE/STEREO loops (Schrijver et al. 2005; Sandman et al. 2009). High-$\beta$ regions are not necessarily inconsistent with force-free if pressure gradients are gravity-balanced (Neukirch 2005), but generically the force-free approach is limited to upper chromosphere and corona ($\lesssim 2.5\,R_\odot$).

**KR.** 코로나는 "자기장 지배" 영역이며 정밀 벡터 측정은 광구로 제한됩니다. 따라서 표준 절차는 광구 경계조건을 위로 *외삽*하는 것입니다. Fig. 1은 태양폭풍의 지구자기권 영향을 보여주고, Fig. 2 (Gary 2001)는 유명한 plasma $\beta$의 높이 프로파일을 제시 — 광구에서 $\beta\sim 0.1$–$10^2$, 약 2 Mm~2 Gm 구간에서 $\beta\ll 1$, 태양풍 가속 영역에서 다시 $\beta>1$. 이 저-$\beta$ 샌드위치에서 정역학 MHD 운동방정식은 선도 차수에서 $\mathbf{j}\times\mathbf{B}=\mathbf{0}$으로 환원됩니다 (Eq. 2). $\mu_0\mathbf{j}=\nabla\times\mathbf{B}$, $\nabla\cdot\mathbf{B}=0$를 대입하면, 해는 (i) $\nabla\times\mathbf{B}=0$ (무전류/퍼텐셜) — Eq. 7 — 혹은 (ii) $\mathbf{B}\parallel\nabla\times\mathbf{B}$, 즉 $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ ($\mathbf{B}\cdot\nabla\alpha=0$, divergence-free 조건에서 유도)로 분류됩니다. 퍼텐셜장은 LOS 자기도($B_z$)만 있으면 되고 $\Delta\phi=0$을 풀어 $\mathbf{B}=-\nabla\phi$로 얻습니다 (PFSS, source surface $\sim 2.5\,R_\odot$, Schatten 등 1969). 단순하지만 자유 에너지가 없고 TRACE/STEREO loop과 불일치 (Schrijver 등 2005; Sandman 등 2009). 고-$\beta$ 영역은 압력 기울기가 중력으로 상쇄되는 경우 force-free와 모순되지 않을 수 있으나(Neukirch 2005), 일반적으로 무력장 가정은 상부 채층~$2.5\,R_\odot$에 한정됩니다.

### 2.2 Section 2 — Linear Force-Free Fields / 선형 무력장 (pp. 10–13)

**EN.** Setting $\alpha\equiv$ constant in $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ and taking the curl gives the vector Helmholtz equation $\Delta\mathbf{B}+\alpha^2\mathbf{B}=0$ (Eq. 17). This is solvable by separation of variables, Green's function (Chiu & Hilton 1977), or Fourier methods (Alissandrakis 1981; Seehafer 1978). The Seehafer construction expands $B_z(x,y,0)$ on the magnetogram in a Fourier series after extending the magnetogram by anti-symmetric mirroring to enforce zero net flux on $[-L_x,L_x]\times[-L_y,L_y]$. With $\lambda_{mn}=\pi^2(m^2/L_x^2+n^2/L_y^2)$ and $r_{mn}=\sqrt{\lambda_{mn}-\alpha^2}$, the field is (Eqs. 18–20)
$$B_z=\sum_{m,n} C_{mn}\,e^{-r_{mn}z}\sin\!\frac{\pi mx}{L_x}\sin\!\frac{\pi ny}{L_y},$$
with $B_x, B_y$ given by analogous combinations of derivatives. For real, decaying solutions one needs $\alpha^2<\alpha_{\max}^2=\pi^2(L_x^{-2}+L_y^{-2})$. Normalizing by the harmonic mean $L$ gives $|\alpha|<\sqrt{2}\pi$. Setting $\alpha=0$ recovers the potential field. **Determining $\alpha$**: average $\alpha=\sum\mu_0 J_z\,\mathrm{sign}(B_z)/\sum|B_z|$ from horizontal magnetograms (Pevtsov et al. 1994; Hagino & Sakurai 2004), or fit $\alpha$ by matching projected field-line shapes against EUV/coronal images (Carcedo et al. 2003; Wiegelmann et al. 2006b; Feng et al. 2007 with STEREO). **Limitation**: optimal $\alpha$ varies among loops in the *same* active region (Pevtsov, Wheatland, Wiegelmann & Neukirch 2002), violating LFFF's premise. Marsch et al. (2004) found single-$\alpha$ adequate for some loops; Malanushenko et al. (2009) showed twist/height can still be estimated to $\sim 15\%$/$5\%$ accuracy with this inconsistent model.

**KR.** $\alpha\equiv$ 상수로 두고 $\nabla\times\mathbf{B}=\alpha\mathbf{B}$의 curl을 취하면 vector Helmholtz 방정식 $\Delta\mathbf{B}+\alpha^2\mathbf{B}=0$ (Eq. 17)이 유도됩니다. 변수분리, Green 함수(Chiu & Hilton 1977), Fourier 방법(Alissandrakis 1981; Seehafer 1978)으로 풀 수 있습니다. Seehafer 구성에서는 광구 자기도 $B_z(x,y,0)$를 반대칭 거울 연장하여 $[-L_x,L_x]\times[-L_y,L_y]$에서 알짜 자속이 0이 되도록 한 후 Fourier 급수 전개합니다. $\lambda_{mn}=\pi^2(m^2/L_x^2+n^2/L_y^2)$, $r_{mn}=\sqrt{\lambda_{mn}-\alpha^2}$로 두면 (Eqs. 18–20),
$$B_z=\sum_{m,n} C_{mn}\,e^{-r_{mn}z}\sin\!\frac{\pi mx}{L_x}\sin\!\frac{\pi ny}{L_y}.$$
실수해 및 발산하지 않으려면 $\alpha^2<\alpha_{\max}^2=\pi^2(L_x^{-2}+L_y^{-2})$. 조화평균 $L$로 정규화 시 $|\alpha|<\sqrt{2}\pi$. $\alpha=0$이면 퍼텐셜장으로 환원. **$\alpha$ 결정**: 횡자기도로부터 $\alpha=\sum\mu_0 J_z\,\mathrm{sign}(B_z)/\sum|B_z|$ 평균(Pevtsov 등 1994; Hagino & Sakurai 2004), 또는 자기력선 투영 형상을 EUV/코로나 영상에 맞춰 피팅(Carcedo 등 2003; Feng 등 2007, STEREO). **한계**: 같은 활동영역의 loop마다 최적 $\alpha$가 달라(Wiegelmann & Neukirch 2002) LFFF 가정과 모순. Marsch 등 (2004)은 일부 loop에서 단일 $\alpha$가 잘 맞는다고 보고. Malanushenko 등 (2009)은 일관성이 없어도 twist/height 추정 정확도가 $\sim 15\%$/$5\%$임을 보임.

### 2.3 Section 3 — Analytic / Semi-analytic NLFFF / 해석적·반해석적 NLFFF (pp. 14–15)

**EN.** Full 3D NLFFF analytic solutions are virtually unavailable; reductions exploit ignorable coordinates. **Lundquist (1950)**: infinitely long cylinder, $\alpha=$ const, Bessel-function flux rope. **Gold–Hoyle (1960)**: same-pitch field lines, $\alpha\neq$ const. **Low (1973)**: 1D Cartesian slab with resistive evolution. In 2D Cartesian with one ignorable coordinate, the Grad–Shafranov equation $\Delta A=-\lambda^2 f(A)$ (Eq. 21) emerges, where $A$ is the flux function and $f(A)$ the generating function. **Low & Lou (1990)** solved the spherical-coordinates Grad–Shafranov equation for axisymmetric NLFFF: $\mathbf{B}=(r\sin\theta)^{-1}[r^{-1}\partial_\theta A\,\hat r-\partial_r A\,\hat\theta+Q\,\hat\varphi]$ with $Q=Q(A)$, leading to an ODE for $P(\mu)$ when $A=P(\mu)/r^n$ and $Q=\lambda A^{1+1/n}$. The solution has a point source at the origin; rotating the symmetry axis off-axis breaks the apparent symmetry (Fig. 5 panels b–d), making it a popular benchmark for 3D NLFFF codes (Schrijver et al. 2006). **Titov–Démoulin (1999)**: an axisymmetric flux-tube of toroidal current $I$ (minor radius $a$, major radius $R$, $a\ll R$) with two magnetic monopoles $\pm q$ at depth $L$ along a line current $I_0$ embedded under the photosphere. Stable branches test extrapolation codes (Wiegelmann et al. 2006a; Valori et al. 2010); unstable branches model CME onset.

**KR.** 일반적인 3D NLFFF 해석해는 거의 없고, 무시가능 좌표를 활용합니다. **Lundquist (1950)**: 무한 원통, $\alpha=$ const, Bessel 함수 자속관. **Gold–Hoyle (1960)**: 동일 pitch 자기력선, $\alpha\neq$ const. **Low (1973)**: 1D Cartesian slab + 저항 진화. 2D Cartesian의 무시가능 좌표 1개에서는 Grad–Shafranov 방정식 $\Delta A=-\lambda^2 f(A)$ (Eq. 21)가 나옵니다. **Low & Lou (1990)**: 구면좌표 Grad–Shafranov로 축대칭 NLFFF — $\mathbf{B}=(r\sin\theta)^{-1}[r^{-1}\partial_\theta A\,\hat r-\partial_r A\,\hat\theta+Q\,\hat\varphi]$, $Q=Q(A)$, $A=P(\mu)/r^n$, $Q=\lambda A^{1+1/n}$ 가정 시 $P(\mu)$에 대한 ODE로 환원. 해는 원점에 점원이 있고, 대칭축을 회전시키면 외형적 대칭이 깨져(Fig. 5 b–d) 3D 코드 벤치마크의 표준이 됨 (Schrijver 등 2006). **Titov–Démoulin (1999)**: 광구 아래 선전류 $I_0$ 위에 토로이드 전류고리 $I$ (소반경 $a\ll$ 대반경 $R$)와 자기 monopole $\pm q$ — 안정 분기는 코드 검증 (Wiegelmann 등 2006a; Valori 등 2010), 불안정 분기는 CME 발생 모델로 사용.

### 2.4 Section 4 — Azimuth Ambiguity Removal & Consistency / 모호성 제거와 일관성 (pp. 16–24)

**EN.** Vector magnetograms invert Stokes profiles $(I,Q,U,V)$ to $(B_\parallel, B_\perp, \chi)$ where $\chi$ is azimuth — but $Q,U$ are unchanged under $\chi\to\chi+180°$, leaving a 180° ambiguity. The review surveys ambiguity-removal algorithms: **(4.3.1) acute-angle method** — choose direction nearest the potential field; **(4.3.2) improved acute-angle** with iterative correction; **(4.3.3) magnetic-pressure gradient** — exploits divergence; **(4.3.4) structure minimization** — minimize azimuth variation; **(4.3.5) non-potential calculation method** — use prior NLFFF estimate; **(4.3.6) pseudo-current**; **(4.3.7) U. Hawai'i iterative**; **(4.3.8) minimum-energy** (Metcalf 1994) — minimizes $|J_z|+|\nabla\cdot\mathbf{B}|$ via simulated annealing, generally most reliable but computationally expensive. **(4.5)** Derived $\alpha$ is sensitive to noise in transverse fields. **(4.6) Consistency** — measured photospheric field is *not* truly force-free ($\beta\sim 1$), so checks like $\sum B_xB_z=\sum B_yB_z=0$ (zero net force) and zero net torque are needed. **(4.7) Preprocessing** (Wiegelmann, Inhester & Sakurai 2006) — minimize a 4-term functional driving the magnetogram toward force-free balance, smoothness, and proximity to observation; output is a "chromospheric" boundary suitable for NLFFF codes.

**KR.** 벡터 자기도는 Stokes 프로파일 $(I,Q,U,V)$을 $(B_\parallel, B_\perp, \chi)$로 역변환하지만 $\chi\to\chi+180°$에서 $Q,U$가 동일해 180° 모호성이 발생. 리뷰는 다음 알고리즘들을 정리: **(4.3.1) acute-angle**: 퍼텐셜장과 가장 예각이 되는 방향 선택; **(4.3.2) improved acute-angle**: 반복 보정; **(4.3.3) 자기압 기울기**: 발산 활용; **(4.3.4) structure minimization**: azimuth 변화 최소화; **(4.3.5) non-potential**: 사전 NLFFF 사용; **(4.3.6) pseudo-current**; **(4.3.7) U. Hawai'i iterative**; **(4.3.8) minimum-energy** (Metcalf 1994): $|J_z|+|\nabla\cdot\mathbf{B}|$를 simulated annealing으로 최소화 — 가장 신뢰할 만하나 계산비용 큼. **(4.5)** $\alpha$는 횡자기장 잡음에 민감. **(4.6) 일관성**: 측정된 광구장은 force-free 아님 ($\beta\sim 1$), 따라서 $\sum B_xB_z=\sum B_yB_z=0$ (알짜 힘 0), 알짜 토크 0 등의 점검이 필요. **(4.7) Preprocessing** (Wiegelmann, Inhester & Sakurai 2006): 자기도를 force-free 균형·매끄러움·관측 근접성에 맞추는 4항 functional을 최소화 — 출력은 NLFFF 코드의 "채층" 경계로 적합.

### 2.5 Section 5 — NLFFF in 3D: Helicity, Energy, Stability / 헬리시티·에너지·안정성 (pp. 25–29)

**EN.** **Helicity.** Magnetic helicity $H=\int\mathbf{A}\cdot\mathbf{B}\,dV$ is gauge-invariant for closed volumes; for open volumes the relative helicity $H_R=\int(\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV$ (Berger & Field 1984; Finn & Antonsen 1985) is used, with $\mathbf{B}_p$ the potential field having the same normal flux. **Energy.** $E=\int B^2/(2\mu_0)\,dV$; the potential field has minimum energy for given $B_n$, so $E_{\rm free}=E-E_p\geq 0$ is the upper bound on extractable energy in flares/CMEs. **Maximum energy / Aly–Sturrock conjecture.** For simply connected force-free fields with all field lines closing inside the volume, $E_{\rm AS}<E_p+\Delta$ where $\Delta$ is a small open-field excess; this caps eruptions but is debated for partially-open fields. **Stability.** Force-free configurations need not be stable (Titov–Démoulin unstable branch). **Numerical stability investigations** test how preprocessing, grid resolution, and boundary treatment affect convergence (Schrijver et al. 2008; DeRosa et al. 2009).

**KR.** **Helicity.** 자기 helicity $H=\int\mathbf{A}\cdot\mathbf{B}\,dV$는 닫힌 부피에서는 게이지 불변; 열린 부피에서는 상대 helicity $H_R=\int(\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV$ (Berger & Field 1984; Finn & Antonsen 1985)를 사용, $\mathbf{B}_p$는 동일 법선자속 퍼텐셜장. **Energy.** $E=\int B^2/(2\mu_0)\,dV$; 동일 $B_n$에 대해 퍼텐셜장이 최소 에너지이므로 $E_{\rm free}=E-E_p\geq 0$이 플레어/CME에서 추출 가능한 에너지의 상한. **최대 에너지 / Aly–Sturrock 추측.** 모든 자기력선이 부피 내에서 닫히는 단순연결 무력장에서는 $E_{\rm AS}<E_p+\Delta$로 제한 — 부분적 열린장에서는 논쟁적. **안정성.** 무력장도 불안정할 수 있음 (Titov–Démoulin 불안정 분기). **수치 안정성 연구**는 preprocessing, 격자 해상도, 경계처리가 수렴에 미치는 영향 분석 (Schrijver 등 2008; DeRosa 등 2009).

### 2.6 Section 6 — Numerical Methods for NLFFF / NLFFF 수치 방법 (pp. 30–37)

**EN.** Five families:
1. **Upward integration** — directly integrate $\nabla\times\mathbf{B}=\alpha\mathbf{B}$ from $z=0$ upward. The PDE is mixed elliptic-hyperbolic; small noise grows exponentially. Damping/filtering required.
2. **Grad–Rubin** (Amari et al. 1997, 2006; Wheatland 2007) — alternates: (a) propagate $\alpha$ along field lines from one polarity (where $B_n\cdot\hat n>0$ or $<0$); (b) update $\mathbf{B}$ via Biot–Savart from the current $\mathbf{j}=\alpha\mathbf{B}/\mu_0$. Iterate to convergence. Uses only one polarity's $\alpha$ — well-posed but discards half the data.
3. **MHD relaxation / Magnetofrictional** (Yang, Sturrock & Antiochos 1986; van Ballegooijen 2004) — pseudo-time $\partial_t\mathbf{B}=\nabla\times(\mathbf{v}\times\mathbf{B})$ with $\mathbf{v}=\nu^{-1}\mathbf{j}\times\mathbf{B}/B^2$ drives Lorentz force to zero while preserving connectivity.
4. **Optimization** (Wheatland, Sturrock & Roumeliotis 2000; Wiegelmann 2004) — minimize
$$L=\int_V\left[B^{-2}|(\nabla\times\mathbf{B})\times\mathbf{B}|^2+|\nabla\cdot\mathbf{B}|^2\right]dV.$$
$L=0$ iff field is force-free and divergence-free. Steepest descent: $\partial_t\mathbf{B}=\mathbf{F}$ with $\mathbf{F}$ derived from $\delta L/\delta\mathbf{B}$. Boundary held fixed at preprocessed magnetogram.
5. **Boundary-element / Green's function** (Yan & Sakurai 2000) — surface integral formulation, asymptotic boundary conditions.

**(6.6)** *NLFFF Consortium* benchmarks (Schrijver et al. 2006 with Low–Lou; Schrijver et al. 2008, DeRosa et al. 2009 with Hinode SOT/SP for AR 10930): codes agree well on benchmarks with consistent boundary data, but disagree by factors of $\sim 2$ on real magnetograms because (i) photosphere is not force-free, (ii) field of view is too small, (iii) preprocessing & resolution differ. **(6.7)** Applications: free energy & helicity for flares (Régnier et al., Thalmann et al.), filament/sigmoid modeling, comparison to STEREO 3D loop reconstructions.

**KR.** 다섯 계열:
1. **Upward integration**: $\nabla\times\mathbf{B}=\alpha\mathbf{B}$를 $z=0$에서 위로 직접 적분. 혼합형 PDE라 작은 잡음이 지수 증가 — 감쇠/필터 필수.
2. **Grad–Rubin** (Amari 등 1997, 2006; Wheatland 2007): (a) 한 극성에서 $\alpha$를 자기력선 따라 전파, (b) $\mathbf{j}=\alpha\mathbf{B}/\mu_0$로부터 Biot–Savart로 $\mathbf{B}$ 갱신 — 반복 수렴. 한 극성의 $\alpha$만 사용해 well-posed지만 데이터 절반 폐기.
3. **MHD relaxation / Magnetofrictional** (Yang, Sturrock & Antiochos 1986; van Ballegooijen 2004): 가상시간 $\partial_t\mathbf{B}=\nabla\times(\mathbf{v}\times\mathbf{B})$, $\mathbf{v}=\nu^{-1}\mathbf{j}\times\mathbf{B}/B^2$로 Lorentz 힘을 0으로 유도, 연결성 보존.
4. **Optimization** (Wheatland, Sturrock & Roumeliotis 2000; Wiegelmann 2004):
$$L=\int_V\left[B^{-2}|(\nabla\times\mathbf{B})\times\mathbf{B}|^2+|\nabla\cdot\mathbf{B}|^2\right]dV$$
를 최소화. $L=0$ 이면 force-free + divergence-free. 최급강하: $\partial_t\mathbf{B}=\mathbf{F}$, $\mathbf{F}$는 $\delta L/\delta\mathbf{B}$에서 도출. 경계는 preprocessing된 자기도로 고정.
5. **Boundary-element / Green** (Yan & Sakurai 2000): 표면 적분식, 점근 경계조건.

**(6.6)** *NLFFF Consortium* 벤치마크 (Schrijver 등 2006: Low–Lou; Schrijver 등 2008, DeRosa 등 2009: Hinode SOT/SP의 AR 10930): 일관 경계조건 벤치마크에서는 코드 간 잘 일치하나 실제 자기도에서는 $\sim 2$배까지 불일치 — (i) 광구가 force-free 아님, (ii) FOV 협소, (iii) preprocessing/해상도 차이. **(6.7)** 응용: 플레어 자유에너지·helicity (Régnier 등, Thalmann 등), 필라멘트/시그모이드 모델링, STEREO 3D loop 재구성 비교.

### 2.7 Section 7 — Summary and Discussion / 요약 및 논의 (p. 38)

**EN.** The review concludes: (1) potential & LFFF are convenient but limited; (2) NLFFF is the right tool but currently boundary-condition-limited; (3) preprocessing is essential; (4) consortium tests show codes converge under controlled conditions; (5) future progress hinges on chromospheric vector magnetograms (e.g., DKIST), better preprocessing, larger FOVs, and joint inversion with coronal observations.

**KR.** 결론: (1) 퍼텐셜·LFFF는 편리하나 제한적, (2) NLFFF가 본 도구이나 경계조건이 결정적 한계, (3) preprocessing 필수, (4) 통제된 조건의 컨소시엄 테스트에서는 코드 수렴, (5) 향후 진보는 채층 벡터 자기도 (DKIST 등), 개선된 preprocessing, 더 큰 FOV, 코로나 관측과의 결합 역해법에 달림.

---

## 3. Key Takeaways / 핵심 시사점

1. **Force-free is justified by low plasma $\beta$, not by zero current.**
   - **EN.** The condition $\mathbf{j}\times\mathbf{B}=0$ requires only that the Lorentz force be small relative to other terms in the momentum equation, which is true wherever $\beta\ll 1$ and gravity/pressure are subdominant. Currents may be (and usually are) large; they simply flow along field lines.
   - **KR.** $\mathbf{j}\times\mathbf{B}=0$ 조건은 운동량 방정식의 다른 항보다 Lorentz 힘이 작을 것만을 요구합니다 — 즉 $\beta\ll 1$이고 중력/압력이 보조적이면 성립. 전류는 클 수 있으며 자기력선을 따라 흐를 뿐입니다.

2. **Two flavors only: potential ($\alpha=0$) and force-free ($\alpha\neq 0$).**
   - **EN.** Potential needs only $B_n$ at the boundary and gives the minimum-energy reference field; force-free needs the full vector field and contains all free energy available for eruptions.
   - **KR.** 퍼텐셜은 경계의 $B_n$만 필요하고 최소 에너지 기준장. 무력장은 전체 벡터장이 필요하며 폭발 시 방출 가능한 자유 에너지를 모두 담음.

3. **Linear FF is a stop-gap; real active regions are nonlinear.**
   - **EN.** A single $\alpha$ rarely fits all loops in an AR (Pevtsov, Wheatland), so LFFF is a quick global indicator but not a quantitative model of free energy or topology.
   - **KR.** 활동영역 내 모든 loop에 단일 $\alpha$가 맞는 경우는 드물어, LFFF는 간이 전역 지표일 뿐 자유에너지·위상 정량 모델은 아님.

4. **Photospheric data are inherently inconsistent with force-free; preprocessing is mandatory.**
   - **EN.** Net force ($\sum B_xB_z$, $\sum B_yB_z$) and net torque on the photosphere are non-zero because $\beta\sim 1$ there. NLFFF codes must condition the magnetogram (Wiegelmann, Inhester & Sakurai 2006) before extrapolating; otherwise they will not converge to a force-free state matching the boundary.
   - **KR.** 광구는 $\beta\sim 1$이라 알짜 힘과 토크가 0이 아님. NLFFF 코드는 외삽 전 자기도를 force-free 호환으로 조정해야 하며 (Wiegelmann, Inhester & Sakurai 2006), 그렇지 않으면 경계와 일치하는 force-free 해로 수렴하지 않음.

5. **180° azimuth ambiguity propagates into $\alpha$, $J_z$, and the entire NLFFF.**
   - **EN.** Stokes Q, U are invariant under azimuth flip, so transverse field direction is determined modulo 180°. Wrong choice flips the sign of $J_z=\partial_xB_y-\partial_yB_x$ in patches, producing spurious currents and wrong $\alpha$. Minimum-energy methods (Metcalf 1994) are the gold standard.
   - **KR.** Stokes Q, U는 azimuth 180° 뒤집기에 불변이므로 횡자기장 방향이 180° 모호. 잘못된 선택은 $J_z=\partial_xB_y-\partial_yB_x$ 부호를 패치 단위로 뒤집어 가짜 전류와 잘못된 $\alpha$를 생성. Minimum-energy 방법(Metcalf 1994)이 표준.

6. **Five NLFFF code families exist; they converge on benchmarks but disagree on real data.**
   - **EN.** Upward integration, Grad–Rubin, MHD relaxation, optimization, and boundary-element methods all produce close solutions for Low–Lou tests, but DeRosa et al. (2009) showed factor-of-two scatter on AR 10930 — driven by FOV size, boundary inconsistency, and resolution rather than the algorithm itself.
   - **KR.** Upward integration, Grad–Rubin, MHD relaxation, optimization, boundary-element 다섯 방법은 Low–Lou 벤치마크에서는 수렴하나 DeRosa 등(2009)의 AR 10930 실험에서는 2배 차이 — 알고리즘보다 FOV 크기, 경계 비일관성, 해상도가 주된 원인.

7. **Free magnetic energy is the eruption budget; helicity is its book-keeping.**
   - **EN.** $E_{\rm free}=E_{\rm NLFFF}-E_{\rm potential}$ caps the energy releasable in flares/CMEs (Aly–Sturrock); relative helicity $H_R$ is conserved under ideal MHD and constrains how much energy can be released without CME ejection.
   - **KR.** $E_{\rm free}=E_{\rm NLFFF}-E_{\rm potential}$은 플레어/CME 방출 가능 에너지의 상한 (Aly–Sturrock). 상대 helicity $H_R$은 이상 MHD에서 보존되며, CME 없이 방출 가능한 에너지를 제한함.

8. **Future progress requires chromospheric magnetograms and 3D coronal cross-checks.**
   - **EN.** DKIST chromospheric vector magnetograms will give a near-force-free boundary, removing the preprocessing burden. STEREO/Solar Orbiter loop tomography will validate 3D extrapolations directly.
   - **KR.** DKIST의 채층 벡터 자기도는 거의 force-free 경계를 제공해 preprocessing 부담을 줄여줌. STEREO/Solar Orbiter의 loop 토모그래피는 3D 외삽을 직접 검증.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Plasma $\beta$ / 플라즈마 베타
$$
\beta = \frac{2\mu_0 p}{B^2}.
$$
- **EN.** $p$ is gas pressure, $B$ is field magnitude. $\beta\ll 1$ ⇒ magnetic pressure dominates ⇒ Lorentz force must vanish to leading order.
- **KR.** $p$는 가스압, $B$는 자기장 크기. $\beta\ll 1$ ⇒ 자기압 지배 ⇒ Lorentz 힘은 선도 차수에서 소멸.

### 4.2 Force-free condition / 무력장 조건
$$
\mathbf{j}\times\mathbf{B}=0,\qquad \mathbf{j}=\frac{1}{\mu_0}\nabla\times\mathbf{B},\qquad \nabla\cdot\mathbf{B}=0.
$$
Equivalently:
$$
\nabla\times\mathbf{B}=\alpha(\mathbf{r})\,\mathbf{B},\qquad \mathbf{B}\cdot\nabla\alpha=0.
$$
- **EN.** Currents flow strictly along field lines; $\alpha$ is constant along each field line but may differ between field lines (NLFFF).
- **KR.** 전류는 자기력선을 따라서만 흐름. $\alpha$는 한 자기력선 위에서는 상수이나 자기력선마다 다를 수 있음 (NLFFF).

### 4.3 Potential field / 퍼텐셜장
$$
\nabla\times\mathbf{B}=0\ \Rightarrow\ \mathbf{B}=-\nabla\phi,\qquad \Delta\phi=0,\qquad \frac{\partial\phi}{\partial z}\Big|_{z=0}=-B_z^{\rm obs}.
$$
- **EN.** Solved by Green's function over a planar magnetogram or by spherical harmonics globally (PFSS, with source surface at $r_{\rm ss}\approx 2.5\,R_\odot$ where field becomes radial).
- **KR.** 평면 자기도에 대한 Green 함수 적분, 또는 PFSS의 구면조화함수 — source surface($r_{\rm ss}\approx 2.5\,R_\odot$)에서 자기장이 방사 방향이 되도록 부과.

The Green's function for the upper half-space with $B_z$ prescribed at $z=0$ is
$$
\phi(\mathbf{r})=\frac{1}{2\pi}\int\!\!\int \frac{B_z(x',y',0)}{\sqrt{(x-x')^2+(y-y')^2+z^2}}\,dx'\,dy',
$$
giving $\mathbf{B}=-\nabla\phi$.

### 4.4 Linear force-free (Helmholtz) / 선형 무력장
$$
\nabla\times\mathbf{B}=\alpha\mathbf{B}\ \Rightarrow\ \Delta\mathbf{B}+\alpha^2\mathbf{B}=0\quad(\alpha=\mathrm{const}).
$$
- **EN.** Vector Helmholtz equation. Seehafer (1978) Fourier solution for periodic-mirrored magnetogram on $[-L_x,L_x]\times[-L_y,L_y]$ uses $\lambda_{mn}=\pi^2(m^2/L_x^2+n^2/L_y^2)$ and decay rate $r_{mn}=\sqrt{\lambda_{mn}-\alpha^2}$. Validity $\alpha^2<\alpha_{\max}^2=\pi^2(L_x^{-2}+L_y^{-2})$.
- **KR.** Vector Helmholtz 방정식. Seehafer (1978) Fourier 해 — 거울 확장한 $[-L_x,L_x]\times[-L_y,L_y]$ 자기도에 대해 $\lambda_{mn}=\pi^2(m^2/L_x^2+n^2/L_y^2)$, 감쇠율 $r_{mn}=\sqrt{\lambda_{mn}-\alpha^2}$. 유효 $\alpha^2<\alpha_{\max}^2=\pi^2(L_x^{-2}+L_y^{-2})$.

### 4.5 Estimating $\alpha$ from horizontal magnetograms / $\alpha$ 추정
$$
\mu_0 J_z = \frac{\partial B_y}{\partial x}-\frac{\partial B_x}{\partial y},\qquad \alpha(x,y)=\mu_0 \frac{J_z(x,y)}{B_z(x,y)}.
$$
A spatially averaged best-fit (Hagino & Sakurai 2004):
$$
\bar\alpha = \frac{\sum \mu_0 J_z\,\mathrm{sign}(B_z)}{\sum |B_z|}.
$$
- **EN.** Sensitive to noise wherever $|B_z|$ is small — use only strong-field pixels, mask weak-field regions.
- **KR.** $|B_z|$가 작은 곳에서 잡음에 민감 — 강자장 픽셀만 사용하고 약자장 영역은 마스킹.

### 4.6 Grad–Shafranov for axisymmetric NLFFF / 축대칭 NLFFF의 Grad–Shafranov
2D Cartesian:
$$
\Delta A=-\lambda^2 f(A),\qquad \mathbf{B}=\nabla A\times\hat z+B_z(A)\hat z.
$$
Spherical Low–Lou:
$$
\frac{\partial^2 A}{\partial r^2}+\frac{1-\mu^2}{r^2}\frac{\partial^2 A}{\partial \mu^2}+Q\frac{dQ}{dA}=0,\quad Q=\lambda A^{1+1/n},\quad A=\frac{P(\mu)}{r^n}.
$$
- **EN.** Reduces 2D PDE to ODE for $P(\mu)$ — eigenvalue problem in $\lambda$ for given $n$. Solutions used as benchmark by Schrijver et al. (2006).
- **KR.** $P(\mu)$에 대한 ODE로 환원되며, 주어진 $n$에 대해 $\lambda$의 고유값 문제. Schrijver 등(2006) 벤치마크에 사용.

### 4.7 Wiegelmann optimization functional / Wiegelmann 최적화 functional
$$
L = \int_V \left[w_f\,\frac{|(\nabla\times\mathbf{B})\times\mathbf{B}|^2}{B^2} + w_d\,|\nabla\cdot\mathbf{B}|^2\right]\,dV.
$$
Functional derivative gives evolution
$$
\frac{\partial \mathbf{B}}{\partial t}=\mathbf{F}(\mathbf{B}),\qquad \mathbf{F}=\nabla\times(\Omega\times\mathbf{B})-\Omega\times(\nabla\times\mathbf{B})+\nabla(\Omega\cdot\mathbf{B})-\Omega(\nabla\cdot\mathbf{B}),
$$
with $\Omega$ encoding the local force imbalance. Boundary $\mathbf{B}|_{\partial V}$ is held fixed at the preprocessed magnetogram.
- **EN.** $L\geq 0$ by construction; $L=0$ iff field is force-free *and* divergence-free. Convergence is gradient-descent toward $L=0$.
- **KR.** $L\geq 0$, $L=0$ iff force-free 및 divergence-free. 경사하강으로 $L=0$에 수렴.

### 4.8 Magnetic energy & free energy / 자기 에너지
$$
E = \int_V \frac{B^2}{2\mu_0}\,dV,\qquad E_{\rm free}=E_{\rm NLFFF}-E_{\rm potential}\geq 0.
$$
- **EN.** Potential field minimizes $E$ for given $B_n|_{\partial V}$; the difference is available for eruptions.
- **KR.** 동일 $B_n|_{\partial V}$에 대해 퍼텐셜장이 $E$ 최소; 차이가 폭발 가용 에너지.

### 4.9 Helicity / 헬리시티
$$
H_R = \int_V (\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV
$$
(Berger & Field 1984). Conserved under ideal MHD; sets a floor on $E_{\rm AS}$ above the potential field via the "minimum-energy" theorem of Taylor (1974, 1986).
- **EN.** $H_R$ is gauge-invariant for open volumes when $\mathbf{B}_p$ has the same $B_n$ as $\mathbf{B}$ on $\partial V$.
- **KR.** 동일 $B_n|_{\partial V}$의 퍼텐셜장 $\mathbf{B}_p$에 대해 게이지 불변. 이상 MHD에서 보존.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1950   Lundquist                  | Constant-α Bessel flux rope (1D cylinder)
1954   Lust & Schlüter           | Force-free fields introduced for the corona
1957   Chandrasekhar & Kendall   | Mathematical theory of force-free fields
1958   Grad & Rubin              | Grad-Rubin scheme (well-posed BVP)
1969   Schatten et al.           | PFSS model (potential + source surface)
1972   Bineau                    | Existence proof for NLFFF BVP
1973   Low                       | 1D resistive force-free slab
1974   Taylor                    | Minimum-energy theorem (helicity-constrained)
1977   Chiu & Hilton             | LFFF Green's function in slab geometry
1978   Seehafer                  | LFFF Fourier method for magnetograms
1981   Sakurai                   | Boundary-element NLFFF formulation
1986   Yang, Sturrock & Antiochos| Magnetofrictional method
1990   Low & Lou                 | Axisymmetric NLFFF benchmark (this paper)
1994   Pevtsov, Canfield & Metcalf| Empirical α distributions
1994   Metcalf                   | Minimum-energy ambiguity removal
1997   Amari et al.              | Modern Grad-Rubin code
1999   Titov & Démoulin          | Flux-rope NLFFF equilibrium
2000   Wheatland, Sturrock,      | Optimization principle for NLFFF
       Roumeliotis
2000   Yan & Sakurai             | Boundary-element 3D NLFFF code
2004   Wiegelmann                | Optimization code with weighting
2004   van Ballegooijen          | Magnetofrictional code (CMS)
2006   Wiegelmann, Inhester      | Preprocessing of vector magnetograms
       & Sakurai
2006   Schrijver et al.          | NLFFF Consortium benchmark (Low-Lou)
2008   Hinode launch (2006/HMI   | Routine vector magnetograms
2010)
2008   Schrijver et al.          | Consortium test on AR 10930
2009   DeRosa et al.             | Critical assessment: code disagreement on real data
                                 |
2012   ★ Wiegelmann & Sakurai   | THIS LIVING REVIEW ★
                                 |
2013+  Inoue et al., Sun et al. | NLFFF for SDO/HMI, free energy mapping
2020+  DKIST, Solar Orbiter PHI | Higher-resolution / chromospheric boundaries
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Relation (EN / KR) |
|-------|--------------------|
| **#10 Parker (1988) — Coronal heating** | Defines the corona's heating problem; force-free models give the magnetic stress reservoir for nanoflare reconnection. / 코로나 가열 문제 정의; 무력장 모델은 nanoflare 재결합의 자기 응력 저장소를 제공. |
| **#33 Schou et al. (2012) — HMI** | Provides the routine vector magnetograms used as boundary conditions for NLFFF. / NLFFF 경계조건으로 쓰이는 정시 벡터 자기도 제공. |
| **Lundquist (1950)** | Constant-α Bessel solution; first analytic force-free flux rope. / 첫 해석적 무력장 자속관 (상수-α Bessel). |
| **Low & Lou (1990)** | Standard semi-analytic NLFFF benchmark for code validation. / 코드 검증의 표준 NLFFF 벤치마크. |
| **Titov & Démoulin (1999)** | Flux-rope equilibrium; tests stability and CME onset. / Flux-rope 평형 — 안정성·CME 발생 테스트. |
| **Wheatland, Sturrock & Roumeliotis (2000)** | Original optimization principle implemented in this review. / 본 리뷰에 구현된 최초의 최적화 원리. |
| **Wiegelmann (2004)** | Wiegelmann's optimization code with weighting function. / 가중함수가 있는 Wiegelmann 최적화 코드. |
| **Schrijver et al. (2006)** | NLFFF Consortium first benchmark. / NLFFF Consortium 첫 벤치마크. |
| **DeRosa et al. (2009)** | Critical assessment of NLFFF on real data — motivates Sec. 6.6 of this review. / 실제 자료에서 NLFFF의 비판적 평가 — 본 리뷰 Sec. 6.6의 동기. |
| **Berger & Field (1984)** | Defines relative helicity used throughout Sec. 5. / Sec. 5의 상대 helicity 정의. |
| **Aly (1991), Sturrock (1991)** | Aly–Sturrock conjecture on maximum energy. / 최대 에너지에 대한 Aly–Sturrock 추측. |
| **Metcalf (1994)** | Minimum-energy ambiguity-removal algorithm. / Minimum-energy 모호성 제거. |
| **Amari et al. (1997, 2006)** | Modern Grad–Rubin solver. / 현대적 Grad–Rubin 해법. |
| **Yan & Sakurai (2000)** | Boundary-element 3D NLFFF method. / 경계요소법 3D NLFFF. |
| **Schou et al. (2012)** | HMI vector magnetograms — input for modern NLFFF. / 현대 NLFFF 입력 자료. |

---

## 7. Methodology Worked Example: LFFF for a single Fourier mode / 단일 Fourier 모드 LFFF 예제

**EN.** Consider the simplest Seehafer mode: $B_z(x,y,0)=B_0\sin(kx)\sin(ky)$ with $k=\pi/L$ on $[0,L]\times[0,L]$. Then $\lambda=2k^2=2\pi^2/L^2$, and for chosen $\alpha$ with $\alpha^2<\lambda$ the decay rate is $r=\sqrt{\lambda-\alpha^2}$. The vertical field decays as $B_z=B_0 e^{-rz}\sin(kx)\sin(ky)$. For $\alpha=0$ (potential): $r_p=k\sqrt{2}\approx 1.41/L$. For $\alpha=k$: $r=k$ (slower decay). For $\alpha=k\sqrt{1.5}$: $r=k/\sqrt{2}$. Taking $L=100$ Mm, $B_0=200$ G, $\alpha=0.01\,$Mm$^{-1}$: $\alpha^2=10^{-4}$, $\lambda=2\pi^2/10^4\approx 1.97\times 10^{-3}$, so $r\approx 0.043\,$Mm$^{-1}$ — fields decay over $\sim 23$ Mm. Increasing $\alpha$ slows decay, twisting field lines and pushing energy higher.

**한국어.** 가장 단순한 Seehafer 모드: $B_z(x,y,0)=B_0\sin(kx)\sin(ky)$, $k=\pi/L$, 영역 $[0,L]^2$. $\lambda=2k^2=2\pi^2/L^2$. $\alpha^2<\lambda$이면 감쇠율 $r=\sqrt{\lambda-\alpha^2}$, 수직장 $B_z=B_0 e^{-rz}\sin(kx)\sin(ky)$. $\alpha=0$ (퍼텐셜): $r_p=k\sqrt{2}$. $\alpha=k$: $r=k$ (감쇠 둔화). $\alpha=k\sqrt{1.5}$: $r=k/\sqrt{2}$. $L=100$ Mm, $B_0=200$ G, $\alpha=0.01$ Mm$^{-1}$이면 $\alpha^2=10^{-4}$, $\lambda\approx 1.97\times 10^{-3}$, $r\approx 0.043$ Mm$^{-1}$ — 약 23 Mm 스케일로 감쇠. $\alpha$가 클수록 감쇠가 느리고 자기력선이 비틀리며 에너지가 더 높이 올라감.

---

## 8. Quantitative Results from the Review / 리뷰의 정량 결과

| Quantity | Value | Source |
|----------|-------|--------|
| Plasma $\beta$ in low corona | $\sim 10^{-2}$ | Gary (2001), Fig. 2 |
| Plasma $\beta$ in photosphere | $\sim 1$ | Gary (2001) |
| LFFF $\alpha$ range (normalized) | $|\alpha|<\sqrt{2}\pi$ | Seehafer (1978) |
| Source surface for PFSS | $r_{\rm ss}\approx 2.5\,R_\odot$ | Schatten et al. (1969) |
| LFFF twist accuracy on inconsistent data | $\sim 15\%$ | Malanushenko et al. (2009) |
| LFFF loop-height accuracy | $\sim 5\%$ | Malanushenko et al. (2009) |
| NLFFF code agreement on Low–Lou | $\lesssim 10\%$ vector error | Schrijver et al. (2006) |
| NLFFF code disagreement on AR 10930 | factor of $\sim 2$ in free energy | DeRosa et al. (2009) |
| Aly–Sturrock energy upper bound | $E_{\rm AS}\sim 1.66\,E_{\rm potential}$ (open field) | Aly (1991) |
| Free-energy fraction of typical AR | $\lesssim 50\%$ of $E_{\rm potential}$ | reviewed empirical results |
| Boundary noise threshold for $\alpha$ | $\delta B_\perp\lesssim 10$ G needed | Sec. 4.5 discussion |

---

## 9. Detailed Discussion: Why Real-Data NLFFF is Hard / 실데이터 NLFFF가 어려운 이유

### 9.1 Boundary inconsistency / 경계 비일관성
**EN.** The photospheric vector field carries gas-pressure and gravity stress that does not vanish under force-free assumptions. If the magnetogram is fed verbatim into an NLFFF code, the iterative solver finds no force-free field that matches both the line-of-sight and transverse components, and either (a) drifts the boundary (free-boundary codes) or (b) settles at a non-zero $L$ (fixed-boundary codes). Wiegelmann–Inhester–Sakurai preprocessing eliminates net force, net torque, and high-spatial-frequency noise simultaneously, mimicking the smoother chromospheric layer where $\beta\ll 1$.

**KR.** 광구 벡터장은 force-free 가정 하에서 사라지지 않는 가스압·중력 스트레스를 담고 있습니다. 자기도를 그대로 NLFFF 코드에 넣으면 수직장과 횡장을 동시에 만족하는 force-free 해를 찾지 못해, 자유경계 코드는 경계를 표류시키고 고정경계 코드는 $L>0$에 머무릅니다. Wiegelmann–Inhester–Sakurai preprocessing은 알짜 힘·토크·고주파 잡음을 동시에 제거해 $\beta\ll 1$인 매끄러운 채층에 가까운 경계를 만듭니다.

### 9.2 Field of view / 시야 효과
**EN.** Currents and helicity in an active region are not localized; magnetic flux from neighboring regions threads through any finite FOV. NLFFF codes typically assume the side and top boundaries are potential or open, which is wrong if significant flux exits the side. DeRosa et al. (2009) showed FOV truncation alone can change the inferred free energy by factors of 2–3.

**KR.** 활동영역의 전류와 helicity는 국소화되지 않으며, 이웃 영역의 자속이 어떤 유한 FOV든 관통합니다. NLFFF 코드는 보통 측면·상단 경계를 퍼텐셜이나 열린 경계로 가정하지만, 측면으로 의미 있는 자속이 빠져나가면 이는 잘못입니다. DeRosa 등(2009)은 FOV 절단만으로도 자유에너지가 2–3배 변화함을 보였습니다.

### 9.3 Resolution and noise / 해상도와 잡음
**EN.** Vector magnetograph noise (≈10–100 G in transverse field) maps to large fractional errors in $\alpha=\mu_0 J_z/B_z$ wherever $B_z$ is small. Higher resolution exposes more small-scale structure but also more noise; consortium tests showed $\sim 1$″ data plus careful masking gives the best NLFFF stability. Hinode SOT/SP (0.16″ pixels) was the dataset of record at this review's writing; SDO/HMI (0.5″) provides much wider coverage but lower resolution.

**KR.** 벡터 자기도의 잡음(횡장 약 10–100 G)은 $|B_z|$가 작은 곳에서 $\alpha=\mu_0 J_z/B_z$에 큰 상대 오차를 유발합니다. 해상도가 높아지면 미세 구조와 함께 잡음도 늘어나며, 컨소시엄 테스트는 약 1″ 자료 + 신중한 마스킹이 NLFFF 안정성에 최적임을 시사합니다. 본 리뷰 시점의 표준 자료는 Hinode SOT/SP (0.16″ pixel)였고, SDO/HMI (0.5″)는 더 넓은 FOV를 제공합니다.

### 9.4 What does "convergence" mean? / "수렴"의 의미
**EN.** Quality metrics from Schrijver et al. (2006): (i) vector correlation $C_{\rm vec}=\sum\mathbf{B}\cdot\mathbf{b}/\sqrt{\sum B^2\sum b^2}$; (ii) Cauchy–Schwarz $C_{\rm CS}=\frac{1}{N}\sum\mathbf{B}\cdot\mathbf{b}/(|\mathbf{B}||\mathbf{b}|)$; (iii) normalized vector error $E_n$, mean error $E_m$; (iv) energy ratio $\epsilon=E_{\rm code}/E_{\rm true}$; (v) divergence/force figures of merit $\langle|f_i|\rangle=\langle|\mathbf{j}\times\mathbf{B}|/(|\mathbf{j}||\mathbf{B}|)\rangle$ and $\langle|\nabla\cdot\mathbf{B}|\rangle\Delta x/B$. A "good" NLFFF reaches $C_{\rm vec}\geq 0.9$, $C_{\rm CS}\geq 0.9$, $\epsilon$ within 5% of unity, and $\langle|f|\rangle\lesssim 10^{-2}$.

**KR.** Schrijver 등(2006)의 품질 지표: (i) vector correlation $C_{\rm vec}$, (ii) Cauchy–Schwarz $C_{\rm CS}$, (iii) 정규화 벡터 오차 $E_n$, 평균 오차 $E_m$, (iv) 에너지비 $\epsilon=E_{\rm code}/E_{\rm true}$, (v) 발산/힘 figure of merit $\langle|f|\rangle$, $\langle|\nabla\cdot\mathbf{B}|\rangle\Delta x/B$. "양호한" NLFFF는 $C_{\rm vec}\geq 0.9$, $C_{\rm CS}\geq 0.9$, $\epsilon$ 1에서 5% 이내, $\langle|f|\rangle\lesssim 10^{-2}$ 수준에 도달.

### 9.5 Open questions / 열린 문제들
**EN.** (1) Self-consistent inversion combining magnetograms with EUV/coronal loops; (2) data-driven time-dependent NLFFF accounting for slow boundary evolution; (3) full-Sun spherical NLFFF connecting active regions to global PFSS; (4) integration with MHD simulations; (5) chromospheric vector magnetograms from DKIST as the natural force-free boundary.

**KR.** (1) 자기도와 EUV 코로나 loop를 결합한 자기일관 역해법; (2) 느린 경계 진화를 반영하는 자료기반 시간의존 NLFFF; (3) 활동영역과 글로벌 PFSS를 잇는 전체 태양 구면 NLFFF; (4) MHD 시뮬레이션과의 결합; (5) DKIST 채층 벡터 자기도를 자연스러운 force-free 경계로 사용.

---

## 10. Worked Numerical Example: Energy Comparison / 에너지 비교 수치 예제

**EN.** Consider a hypothetical AR with photospheric magnetogram having peak $|B_z|=2000$ G, area $100\times 100$ Mm$^2$. Total magnetic flux $\Phi\sim 2\times 10^{22}$ Mx. Modeling:

| Model | Description | Energy estimate |
|-------|-------------|-----------------|
| Potential | $\alpha=0$, lowest energy compatible with $B_n$ | $E_p\sim 10^{32}$ erg |
| LFFF, $\bar\alpha=0.005$ Mm$^{-1}$ | single twist parameter | $E_{\rm LFFF}\sim 1.1\,E_p$ |
| NLFFF (well-converged) | full $\alpha(\mathbf{r})$ | $E_{\rm NLFFF}\sim 1.3$–$1.5\,E_p$ |
| Aly–Sturrock upper bound | open-field maximum | $E_{\rm AS}\sim 1.66\,E_p$ |
| X-class flare release | typical observed | $\Delta E\sim 10^{32}$ erg |

The free energy budget $E_{\rm free}\sim 0.3$–$0.5\,E_p\sim 3$–$5\times 10^{31}$ erg is consistent with a single X-class flare. NLFFF's job is to estimate this $0.3$–$0.5$ fraction with sufficient precision (≤20% relative error) for forecasting, which requires both well-preprocessed boundaries and convergence metrics like $\langle|f|\rangle\lesssim 10^{-2}$.

**KR.** 가상의 활동영역: 광구 자기도 최대 $|B_z|=2000$ G, 면적 $100\times 100$ Mm$^2$. 총 자속 $\Phi\sim 2\times 10^{22}$ Mx. 모델별:

| 모델 | 설명 | 에너지 |
|------|------|--------|
| 퍼텐셜 | $\alpha=0$, 동일 $B_n$의 최저 에너지 | $E_p\sim 10^{32}$ erg |
| LFFF, $\bar\alpha=0.005$ Mm$^{-1}$ | 단일 비틀림 모수 | $E_{\rm LFFF}\sim 1.1\,E_p$ |
| NLFFF (잘 수렴) | $\alpha(\mathbf{r})$ 전체 | $E_{\rm NLFFF}\sim 1.3$–$1.5\,E_p$ |
| Aly–Sturrock 상한 | 열린장 최대 | $E_{\rm AS}\sim 1.66\,E_p$ |
| X급 플레어 방출 | 관측 평균 | $\Delta E\sim 10^{32}$ erg |

자유에너지 $E_{\rm free}\sim 0.3$–$0.5\,E_p\sim 3$–$5\times 10^{31}$ erg는 단일 X급 플레어와 일치. NLFFF는 예보용 $\leq 20\%$ 정확도로 이 비율을 추정해야 하며, 이를 위해 잘 preprocessing된 경계와 $\langle|f|\rangle\lesssim 10^{-2}$ 수준의 수렴 지표 모두 필요.

---

## 11. Glossary of Acronyms / 약어 사전

| Acronym | Full term (EN / KR) |
|---------|---------------------|
| FFF | Force-Free Field / 무력장 |
| LFFF | Linear FFF (constant-$\alpha$) / 선형 무력장 |
| NLFFF | Nonlinear FFF / 비선형 무력장 |
| PFSS | Potential-Field Source-Surface / 퍼텐셜장 source surface 모델 |
| BVP | Boundary Value Problem / 경계값 문제 |
| LOS | Line-Of-Sight / 시선 방향 |
| AR | Active Region / 활동영역 |
| FOV | Field Of View / 시야 |
| MHD | Magnetohydrodynamics / 자기유체역학 |
| CME | Coronal Mass Ejection / 코로나 물질 방출 |
| EUV | Extreme Ultraviolet / 극자외선 |
| HMI | Helioseismic and Magnetic Imager (SDO) / SDO의 자기·태양진동 영상기 |
| SOT/SP | Solar Optical Telescope / Spectro-Polarimeter (Hinode) / Hinode 광학·분광편광기 |
| STEREO | Solar TErrestrial RElations Observatory / 두 시점 태양 관측 위성 |
| DKIST | Daniel K. Inouye Solar Telescope / Inouye 태양망원경 |
| SOLIS | Synoptic Optical Long-term Investigations of the Sun / 미국 NSO의 종관 관측 망원경 |
| MDI | Michelson Doppler Imager (SOHO) / SOHO의 도플러 영상기 |
| TRACE | Transition Region And Coronal Explorer / 천이영역·코로나 탐사선 |

---

## 12. References / 참고문헌

- Wiegelmann, T. & Sakurai, T., "Solar Force-free Magnetic Fields", *Living Rev. Solar Phys.*, 9, 5 (2012). DOI: 10.12942/lrsp-2012-5
- Aly, J.J., "How much energy can be stored in a three-dimensional force-free magnetic field?", *ApJ*, 375, L61 (1991).
- Amari, T., Aly, J.J., Luciani, J.F., Boulmezaoud, T.Z., Mikic, Z., "Reconstructing the solar coronal magnetic field as a force-free magnetic field", *Solar Phys.*, 174, 129 (1997).
- Berger, M.A. & Field, G.B., "The topological properties of magnetic helicity", *J. Fluid Mech.*, 147, 133 (1984).
- Carcedo, L., Brown, D.S., Hood, A.W., Neukirch, T., Wiegelmann, T., "A quantitative method to optimise magnetic field line fitting of observed coronal loops", *Solar Phys.*, 218, 29 (2003).
- Chandrasekhar, S. & Kendall, P.C., "On force-free magnetic fields", *ApJ*, 126, 457 (1957).
- Chiu, Y.T. & Hilton, H.H., "Exact Green's function method of solar force-free magnetic-field computations", *ApJ*, 212, 873 (1977).
- DeRosa, M.L., et al., "A critical assessment of nonlinear force-free field modeling of the solar corona", *ApJ*, 696, 1780 (2009).
- Finn, J.M. & Antonsen, T.M., "Magnetic helicity: what is it and what is it good for?", *Comments Plasma Phys. Contr. Fus.*, 9, 111 (1985).
- Gary, G.A., "Plasma beta above a solar active region", *Solar Phys.*, 203, 71 (2001).
- Grad, H. & Rubin, H., "Hydromagnetic equilibria and force-free fields", in *Proc. 2nd UN Conf. Peaceful Uses Atomic Energy*, 31, 190 (1958).
- Hagino, M. & Sakurai, T., "Latitude variation of helicity in solar active regions", *PASJ*, 56, 831 (2004).
- Low, B.C. & Lou, Y.Q., "Modeling solar force-free magnetic fields", *ApJ*, 352, 343 (1990).
- Lundquist, S., "Magnetohydrostatic fields", *Ark. Fys.*, 2, 361 (1950).
- Malanushenko, A., Longcope, D.W., McKenzie, D.E., "Reconstructing the local twist of coronal magnetic fields", *ApJ*, 707, 1044 (2009).
- Marsch, E., Wiegelmann, T., Xia, L.D., "Coronal plasma flows and magnetic fields in solar active regions", *A&A*, 428, 629 (2004).
- Metcalf, T.R., "Resolving the 180-degree ambiguity in vector magnetic field measurements", *Solar Phys.*, 155, 235 (1994).
- Pevtsov, A.A., Canfield, R.C., Metcalf, T.R., "Patterns of helicity in solar active regions", *ApJ*, 425, L117 (1994).
- Sakurai, T., "Calculation of force-free magnetic field with non-constant α", *Solar Phys.*, 69, 343 (1981).
- Schatten, K.H., Wilcox, J.M., Ness, N.F., "A model of interplanetary and coronal magnetic fields", *Solar Phys.*, 6, 442 (1969).
- Schrijver, C.J., et al., "Nonlinear force-free modeling of coronal magnetic fields. I. A quantitative comparison of methods", *Solar Phys.*, 235, 161 (2006).
- Schrijver, C.J., et al., "Nonlinear force-free field modeling of a solar active region around the time of a major flare and CME", *ApJ*, 675, 1637 (2008).
- Seehafer, N., "Determination of constant-α force-free solar magnetic fields from magnetograph data", *Solar Phys.*, 58, 215 (1978).
- Sturrock, P.A., "Maximum energy of semi-infinite magnetic field configurations", *ApJ*, 380, 655 (1991).
- Titov, V.S. & Démoulin, P., "Basic topology of twisted magnetic configurations in solar flares", *A&A*, 351, 707 (1999).
- van Ballegooijen, A.A., "Observations and modeling of a filament on the Sun", *ApJ*, 612, 519 (2004).
- Wheatland, M.S., Sturrock, P.A., Roumeliotis, G., "An optimization approach to reconstructing force-free fields", *ApJ*, 540, 1150 (2000).
- Wiegelmann, T., "Optimization code with weighting function for the reconstruction of coronal magnetic fields", *Solar Phys.*, 219, 87 (2004).
- Wiegelmann, T., Inhester, B., Sakurai, T., "Preprocessing of vector magnetograph data for a nonlinear force-free magnetic field reconstruction", *Solar Phys.*, 233, 215 (2006).
- Yan, Y. & Sakurai, T., "New boundary integral equation representation for finite energy force-free magnetic fields in open space above the Sun", *Solar Phys.*, 195, 89 (2000).
- Yang, W.H., Sturrock, P.A., Antiochos, S.K., "Force-free magnetic fields: the magneto-frictional method", *ApJ*, 309, 383 (1986).
