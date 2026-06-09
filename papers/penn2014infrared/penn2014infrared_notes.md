---
title: "Infrared Solar Physics"
authors: Matthew J. Penn
year: 2014
journal: "Living Reviews in Solar Physics"
doi: "10.12942/lrsp-2014-2"
topic: Living_Reviews_in_Solar_Physics
tags: [infrared, zeeman_effect, spectropolarimetry, coronal_magnetic_field, dkist, solar_magnetism, CO_lines, He_I_1083, Fe_I_1565, Mg_I_12318, Fe_XIII]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 37. Infrared Solar Physics / 적외선 태양물리

---

## 1. Core Contribution / 핵심 기여

**English.** Penn (2014) delivers a textbook-length *Living Reviews* synthesis of the last two decades of solar observing in the infrared window 1000–12 400 nm. Structurally the paper divides into three conceptual pieces. First, it motivates *why* IR is worth pursuing by cataloguing four instrumental advantages (improved atmospheric seeing via $r_0 \propto \lambda^{6/5}$, reduced atmospheric scattering, less instrumental scattering, and smaller instrumental polarization) and three scientific advantages (dramatically larger Zeeman splitting $\propto g_{\rm eff}\lambda^2$, access to molecular rotation-vibration lines that form only in cool sunspot umbrae, and the ability to probe different atmospheric heights through the H$^-$ free-free continuum). It balances this against three instrumental disadvantages (larger diffraction limit, higher thermal background from 300 K optics, telluric absorption) and two scientific ones (fewer solar photons, fewer atomic absorptions). Second, it executes five deep dives into the landmark IR diagnostics: CO 4666 nm (cool "COmosphere" and thermal bifurcation), Fe I 1564.8 nm (the fully split $g=3$ workhorse that settled the kilogauss-flux-tube debate), He I 1083 nm (chromospheric/coronal-hole diagnostic), Mg I 12 318 nm (the most magnetically sensitive known line with $g_{\rm eff}\lambda \approx 12\,300$), and the coronal [Fe XIII] 1074.7 nm forbidden line (direct coronal magnetic-field measurement). Third, it looks ahead to DKIST, GREGOR/GRIS, COSMO, and space-based IR solar missions.

**한국어.** Penn(2014)은 1000–12 400 nm IR 창에서의 지난 20년 태양 관측 성과를 교과서 분량의 *Living Reviews*로 종합 정리한 논문이다. 구조적으로 세 부분으로 나뉜다. 먼저 IR이 필요한 *이유*를 정리한다 — 네 가지 기기적 장점(대기 seeing 개선 $r_0 \propto \lambda^{6/5}$, 대기 산란 감소, 기기 산란 감소, 기기 편광 감소)과 세 가지 과학적 장점(제만 분리 $\propto g_{\rm eff}\lambda^2$의 극적 증대, 차가운 흑점 본영에서만 형성되는 분자 진동-회전선 접근, H$^-$ free-free 연속체를 통한 다층 높이 진단)을 제시하고, 세 가지 기기적 단점(회절 한계 증가, 300 K 광학계 열배경, 대기 흡수대)과 두 가지 과학적 단점(광자수 감소, 원자 흡수선 감소)과 균형을 이룬다. 둘째, 다섯 개의 대표 IR 진단선을 심층 분석한다 — CO 4666 nm(차가운 "COmosphere"와 열 이분화), Fe I 1564.8 nm($g=3$로 완전 분리되는 IR 대표선; kG 자속관 논쟁 해결), He I 1083 nm(채층/코로나홀 진단), Mg I 12 318 nm($g_{\rm eff}\lambda \approx 12\,300$로 알려진 가장 자기민감한 선), [Fe XIII] 1074.7 nm(코로나 자기장 직접 측정 금지선). 셋째, DKIST, GREGOR/GRIS, COSMO, 그리고 우주 기반 IR 태양 미션의 전망을 논의한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction / 서론 (p. 5–7)

**English.** The paper opens with a colleague's rhetorical challenge ("Why does anyone still observe the Sun using visible wavelengths?") and acknowledges that night-time astronomers exploited IR long ago. Penn argues that tradition, not physics, explains the visible-dominant culture in solar observing. Three decades of wavelength (1000 nm to 1 mm) define the IR spectrum following Jefferies (1994). Table 1 offers the informal near/mid/far-IR split; Penn prefers factor-of-ten subdivisions. Detector technologies are enumerated: HgCdTe (1000–2200 nm), InGaAs (1000–1800 nm), InSb (1000–5000 nm), Si:X doped silicon (2000–30 000 nm), Ge:X (28 000–200 000 nm), PtSi diodes, Ge bolometers, and new QWIP cameras tested at McMath-Pierce (McM-P).

**한국어.** 논문은 "왜 아직도 가시광으로 태양을 보는가?"라는 동료의 수사적 도전으로 시작한다. 야간 천문학은 이미 오래전 IR을 활용해 왔음을 언급하며, 태양관측의 가시광 주도 문화는 물리가 아닌 전통 때문이라고 지적한다. Jefferies(1994)를 따라 IR은 1000 nm ~ 1 mm의 세 데케이드 파장으로 정의된다. Table 1은 비공식적 근/중/원 IR 구분을 제공하며, 저자는 10배 단위 구분을 선호한다. 검출기 기술은 HgCdTe(1000–2200 nm), InGaAs(1000–1800 nm), InSb(1000–5000 nm), Si:X doped silicon(2000–30 000 nm), Ge:X(28 000–200 000 nm), PtSi 다이오드, Ge 볼로미터, McMath-Pierce(McM-P)에서 시험 중인 QWIP 카메라 등으로 정리된다.

Figure 1 (p. 7): Atmospheric transmission from 560 to 21 000 nm at Kitt Peak, showing the J/H/K/L/M bands (1300/1600/2200/3600/5000 nm central wavelengths) established by Johnson (1962). These windows dictate *which* IR diagnostics are accessible from the ground.
Figure 1 (7쪽): Kitt Peak에서 측정된 560–21 000 nm 대기 투과율. Johnson(1962)이 정립한 J/H/K/L/M 밴드(중심 1300/1600/2200/3600/5000 nm)를 보여 주며, 지상에서 어떤 IR 진단이 가능한지 결정한다.

### Part II: Instrument Advantages / 기기적 장점 (Sec. 2.1, p. 8–11)

**2.1.1 Better atmospheric seeing / 대기 seeing 개선.**

**English.** The Fried parameter grows with wavelength as $r_0 \propto \lambda^{6/5}$ (Karo & Schneiderman 1978). Two practical benefits follow: (a) the isoplanatic patch — the angular area over which a single AO correction is valid — scales as $r_0$, so a wider corrected field is achievable in the IR; (b) the characteristic distortion time $T_{\rm AO} \propto r_0/v_{\rm atm}$ is longer, so AO loops can operate more slowly. Many infrared surveys historically achieved close to the diffraction limit without AO (Turon & Léna 1970).

**한국어.** Fried 파라미터는 파장에 따라 $r_0 \propto \lambda^{6/5}$로 증가한다(Karo & Schneiderman 1978). 두 가지 실용적 이점이 있다: (a) AO 보정이 유효한 isoplanatic 영역이 $r_0$에 비례하므로 IR에서 더 넓은 시야를 보정할 수 있다; (b) 특성 왜곡 시간 $T_{\rm AO} \propto r_0/v_{\rm atm}$이 길어져 AO 루프를 더 천천히 운용할 수 있다. 많은 IR 관측은 과거 AO 없이도 회절 한계에 근접했다(Turon & Léna 1970).

**2.1.2 Less atmospheric scattering / 대기 산란 감소.**

**English.** Rayleigh scattering scales as $\lambda^{-4}$ but at IR wavelengths particle sizes approach $\lambda$ and Mie scattering dominates (Knestrick et al. 1962 → $\lambda^{-1.7}$ empirical dependence, Figure 2). Scattered-light contamination around the solar limb and in coronagraph observations is therefore dramatically reduced in the IR.

**한국어.** Rayleigh 산란은 $\lambda^{-4}$이나 IR에서는 입자 크기가 $\lambda$에 근접해 Mie 산란이 지배적이 된다(Knestrick et al. 1962의 경험식 $\lambda^{-1.7}$, Figure 2). 따라서 태양 가장자리 주변 및 코로나그래프 관측에서 산란광 오염이 극적으로 감소한다.

**2.1.3 Less instrumental scattering / 기기 산란 감소.**

**English.** For a mirror with RMS surface roughness $\sigma$, the total integrated scatter is TIS $\propto (\sigma/\lambda)^2$ (Bennet & Porteus 1961). Moving from 1150 nm to 10 600 nm decreases diffuse scatter from surface roughness by a factor $\sim 100$. For real, dust-contaminated mirrors the reduction is about a factor of 20 (Spyak & Wolfe 1992a,b), still significant.

**한국어.** RMS 표면 거칠기 $\sigma$를 가진 거울의 총 산란은 TIS $\propto (\sigma/\lambda)^2$이다(Bennet & Porteus 1961). 1150 nm → 10 600 nm 이동 시 이상적 거울의 산란은 약 100배 감소하며, 실제 먼지 오염 거울에서도 약 20배 감소한다(Spyak & Wolfe 1992).

**2.1.4 Smaller instrumental polarization / 기기 편광 감소.**

**English.** The single-mirror Mueller matrix (Balasubramaniam et al. 1985, Eq. 1 in paper):
$$M = \begin{pmatrix} 1+X^2 & 1-X^2 & 0 & 0 \\ 1-X^2 & 1+X^2 & 0 & 0 \\ 0 & 0 & 2X\cos\tau & 2X\sin\tau \\ 0 & 0 & -2X\sin\tau & 2X\cos\tau \end{pmatrix}$$
where $X$ and $\tan\tau$ depend on the angle of incidence and the complex refractive index $n - ik$. For aluminum mirrors, $k$ increases through the IR (Rakić 1995), making $X \to 1$ and $\tau \to 0$, so the matrix becomes diagonal and instrumental polarization vanishes. Socas-Navarro et al. (2011) showed this behavior in the NSO/DST Mueller matrix from 470 to 1413 nm (Figure 3).

**한국어.** 단일 거울의 Mueller 행렬(Balasubramaniam et al. 1985)은 위 식과 같으며, $X$와 $\tan\tau$는 입사각 및 복소 굴절률 $n-ik$에 의존한다. 알루미늄 거울은 IR에서 $k$가 증가해(Rakić 1995) $X \to 1$, $\tau \to 0$이 되고, 행렬은 대각화되어 기기 편광이 사라진다. Socas-Navarro et al.(2011)이 NSO/DST Mueller 행렬을 470–1413 nm에서 측정해 이 거동을 확인했다(Figure 3).

### Part III: Instrument Disadvantages / 기기적 단점 (Sec. 2.2, p. 12)

**English.** Three drawbacks emerge: (2.2.1) the diffraction limit $\theta = 1.22\lambda/D$ grows linearly with $\lambda$, so e.g. a 1-m telescope yields 0.2″ at 1000 nm but 2.0″ at 10 000 nm. (2.2.2) A 300 K optics surface peaks via Wien's law at 9656 nm, so background thermal emission contaminates observations beyond $\sim 3000$ nm. Remedies: cryogenic cooling of detectors, filters, and feed optics below 77 K (LN$_2$); "chopping" between target and sky. (2.2.3) Telluric bands (H$_2$O, CO$_2$) block substantial swaths of the IR; transmissive optics like BK7 (cutoff 2500 nm) and fused silica (2300 nm) require replacement by CaF$_2$ or MgF$_2$ (to 6000 nm).

**한국어.** 세 가지 단점이 있다: (2.2.1) 회절 한계 $\theta = 1.22\lambda/D$가 $\lambda$에 선형 증가한다 — 1 m 망원경에서 1000 nm는 0.2″, 10 000 nm는 2.0″. (2.2.2) 300 K 광학면은 Wien 법칙으로 9656 nm에서 피크, 약 3000 nm 이상에서 열배경이 관측을 오염시킨다. 대응책은 검출기·필터·feed optics를 77 K(액체 질소) 이하로 냉각하고, 타깃-배경 사이 chopping을 수행하는 것이다. (2.2.3) 대기 흡수대(H$_2$O, CO$_2$)가 IR 상당 부분을 차단한다. BK7(차단 2500 nm), fused silica(2300 nm) 같은 투과 광학은 CaF$_2$ 또는 MgF$_2$(6000 nm까지)로 교체해야 한다.

### Part IV: Scientific Advantages / 과학적 장점 (Sec. 2.3, p. 13–15)

**2.3.1 Increased Zeeman resolution / 제만 분해능 증대.**

**English.** The central physics driver. The Zeeman splitting of sublevels scales as $g_{\rm eff}\lambda^2$, while Doppler broadening scales only linearly as $\lambda$. The ratio — a dimensionless magnetic resolution — therefore grows as $g_{\rm eff}\lambda$. Table 2 compiles magnetic sensitivities:

| Region / 영역 | Atom | $\lambda$ (nm) | $g_{\rm eff}$ | $\lambda g_{\rm eff}$ |
|---|---|---|---|---|
| Photosphere | Fe I | 525 | 3.0 | 1575 |
| Photosphere | Fe I | 630 | 2.5 | 1575 |
| Photosphere | Fe I | 1565 | 3.0 | **4695** |
| Photosphere | Ti I | 2231 | 2.5 | 5778 |
| Photosphere | Fe I | 4064 | 1.25 | 5080 |
| Photosphere | Fe I | 4137 | 2.81? | 11 625 |
| Photosphere | Mg I | 12 318 | 1.0 | **12 318** |
| Chromosphere | Ca I | 854 | 1.1 | 939 |
| Corona | [Fe XIII] | 1075 | 1.5 | 1612 |
| Corona | [Si X] | 3934 | 1.5 | 5901 |

Figure 4 compares 5250.22 Å and 15 648.54 Å Stokes V profiles at 1 kG: the 5250 line is in the weak-field regime (peak separation ≈ Voigt width), while the 15 648 line is fully resolved.

**한국어.** 중심 물리 원리. 제만 준위 분리는 $g_{\rm eff}\lambda^2$로 증가하나 도플러 폭은 $\lambda$로만 증가한다. 두 비율, 즉 무차원 자기분해능은 $g_{\rm eff}\lambda$에 비례한다. Table 2는 위와 같이 자기민감도를 정리한다. Figure 4는 5250.22 Å과 15 648.54 Å Stokes V 프로파일을 1 kG에서 비교한다: 5250선은 약자기장 영역(피크 분리 ≈ Voigt 폭), 15 648선은 완전 분해된다.

**2.3.2 Molecular rotation-vibration lines / 분자 진동-회전선.**

**English.** Using classical rotational energy $I\omega^2 \sim kT$ with $T = 6000$ K gives rotational frequencies $\omega \sim 10^4$ GHz, i.e. $\lambda \sim 25\,\mu$m — explaining why the IR spectrum is littered with molecular transitions. Because molecules survive only in the coolest solar regions (sunspot umbrae, temperature minimum), they uniquely probe these sites. Molecular lines also have Zeeman sensitivity and enable isotopic abundance measurements (e.g. $^{13}$C/$^{12}$C via CO).

**한국어.** 고전적 회전 에너지 $I\omega^2 \sim kT$에 $T=6000$ K를 대입하면 회전 주파수 $\omega \sim 10^4$ GHz, 즉 $\lambda \sim 25\,\mu$m를 얻는다. 이것이 IR 스펙트럼에 분자 전이가 풍부한 이유이다. 분자는 태양의 가장 차가운 영역(흑점 본영, 온도 최소층)에서만 살아남으므로 이들 영역의 고유한 진단자가 된다. 분자선은 제만 감도를 가지며 $^{13}$C/$^{12}$C 같은 동위원소 존재비 측정도 가능하다.

**2.3.3 Continuum probes different atmospheric heights / 연속체로 여러 대기 높이 진단.**

**English.** From 1000 to 10 000 nm the height of formation of the continuum varies from $z = -40$ km to $z = 140$ km. At $\lambda > 1600$ nm the dominant opacity switches from H$^-$ bound-free to H$^-$ free-free. VAL fit: $z_{\tau=1} = -776 - 227\log_{10}(1/\lambda_{\rm nm})$ km. Thus at 1600 nm we see $\sim 40$ km deeper than at 500 nm; magnetic fields and granulation contrast *reverse sign* between these depths (Leenaarts & Wedemeyer-Böhm 2005; Cheung et al. 2007).

**한국어.** 1000–10 000 nm에서 연속체 형성 높이는 $z = -40$ km에서 $z = 140$ km까지 변한다. 1600 nm 이상에서 주된 불투명도는 H$^-$ bound-free에서 H$^-$ free-free로 전환된다. VAL 적합식: $z_{\tau=1} = -776 - 227\log_{10}(1/\lambda_{\rm nm})$ km. 1600 nm에서는 500 nm보다 약 40 km 깊은 층을 본다. 자기장과 과립 대비가 이 구간에서 *부호 반전*된다(Leenaarts & Wedemeyer-Böhm 2005; Cheung et al. 2007).

### Part V: Scientific Disadvantages / 과학적 단점 (Sec. 2.4, p. 16)

**English.** (2.4.1) The Sun radiates as a $\sim 5800$ K blackbody peaking in the visible. At fixed spectral resolving power $R = \lambda/\Delta\lambda$, IR photon flux scales as $\lambda^{-3}$, so S/N drops as $\lambda^{-3/2}$. (2.4.2) IR photon energies of $10^{-3}$–$1$ eV require small atomic level-energy differences, which occur only in high-$n$ upper levels whose populations are typically low in solar plasma. Hence fewer and weaker atomic absorption lines — though one benefit is cleaner, less blended profiles.

**한국어.** (2.4.1) 태양은 약 5800 K 흑체로 가시광에서 피크한다. 분해능 $R = \lambda/\Delta\lambda$ 일정 조건에서 IR 광자속은 $\lambda^{-3}$로 감소 → S/N은 $\lambda^{-3/2}$로 저하. (2.4.2) $10^{-3}$–$1$ eV 에너지의 IR 광자는 작은 원자 준위 차를 요구, 이는 보통 인구밀도가 낮은 높은 $n$ 상태에서만 일어난다. 따라서 IR 원자 흡수선은 수가 적고 약하나, 대신 블렌딩이 적어 프로파일이 깨끗한 장점이 있다.

### Part VI: CO 4666 nm / CO 4666 nm (Sec. 3.1, p. 18–21)

**English.** The fundamental CO absorption lines near 4666 nm have been observed from the ground at McM-P since Hall et al. (1972) and from space with ATMOS (Farmer & Norton 1989b). The key discovery was unexpectedly strong CO absorption implying large amounts of *cool* gas at heights where chromospheric Ca II/Mg II H&K emission demanded *hot* plasma — a paradox resolved by invoking a highly dynamic, inhomogeneous atmosphere with coexisting cold "COmospheres" and hot canopies (Ayres 1981, 2002; Wedemeyer-Böhm & Steffen 2007). Figure 6 contrasts the old plane-parallel model (left) with the updated dynamic structure (right) where the COmosphere exists at $T \sim 3500$ K below a hot canopy. Heights of formation range $z = 400$–$560$ km (Clark et al. 1995; Uitenbroek et al. 1994; Ayres & Rabin 1996). Helioseismology using CO lines (Penn et al. 2011, Figure 8) measures I–V phase shifts revealing radiative relaxation frequencies and formation heights.

**한국어.** CO 기본 흡수선(4666 nm 부근)은 Hall et al.(1972) 이래 McM-P에서, ATMOS(Farmer & Norton 1989b)로 우주에서 관측되어 왔다. 핵심 발견은 예상외로 강한 CO 흡수가 *차가운* 가스의 대량 존재를 의미하며, 같은 높이의 Ca II/Mg II H&K 방출은 *뜨거운* 플라스마를 요구한다는 점이었다. 이 모순은 차가운 "COmosphere"와 뜨거운 canopy가 공존하는 고도로 역동적·비균질 대기로 해결되었다(Ayres 1981, 2002; Wedemeyer-Böhm & Steffen 2007). Figure 6은 좌: 구 평행 모형, 우: 업데이트된 동적 구조(COmosphere $T \sim 3500$ K, 뜨거운 canopy 하부)를 대조한다. 형성 높이는 400–560 km이다(Clark et al. 1995; Uitenbroek et al. 1994; Ayres & Rabin 1996). Penn et al.(2011)의 CO 헬리오사이스몰로지(Figure 8)는 I–V 위상 이동을 통해 복사 이완 주파수와 형성 높이를 유도했다.

### Part VII: Fe I 1564.8 nm / Fe I 1564.8 nm (Sec. 3.2, p. 21–24)

**English.** The *IR workhorse*. With $g_{\rm eff} = 3$ and $\lambda = 1565$ nm, this line has $g_{\rm eff}\lambda = 4695$, roughly three times the visible Fe I 6302 line. Instrumental chronology: PbS detector (Hall & Noyes 1969) → cooled InSb on FTS (Stenflo et al. 1987) → array instruments NIM, TIP, NAC, FIRS (1024×1024 HgCdTe, Jaeggli et al. 2010) → IRIM Fabry-Pérot (Cao et al. 2004). Stenflo et al. (1987) first showed that at $B \ge 1$ kG the line is *fully split*: the Zeeman σ displacement exceeds the Doppler width, so the line directly reveals $B$ rather than $B \cdot f$ (filling factor × field). This settled decades of debate: histograms of field strengths (Lin 1995) showed active-region plage fields 300–2000 G (mean 1400 G) and quiet-Sun fields 200–2000 G (mean 500 G). Khomenko et al. (2005) combining 1565 and 630 nm lines lowered the internetwork field estimate by an order of magnitude to $\sim 20$ G. Sunspot work (Kopp & Rabin 1992; Mathew et al. 2003) found a rough $B \propto T^2$ dependence consistent with horizontal magnetostatic equilibrium. Bellot Rubio et al. (2000) used SIR inversion on 1565 nm Stokes profiles for umbral helioseismology, finding velocity/field oscillation phases of 105° ± 30° (vs. adiabatic 90°).

**한국어.** *IR의 대표선*. $g_{\rm eff}=3$, $\lambda=1565$ nm → $g_{\rm eff}\lambda = 4695$로 가시광 Fe I 6302선의 약 3배 감도. 기기 계보: PbS 검출기(Hall & Noyes 1969) → FTS의 냉각 InSb(Stenflo et al. 1987) → 어레이 기기 NIM, TIP, NAC, FIRS(1024×1024 HgCdTe, Jaeggli et al. 2010) → IRIM Fabry-Pérot(Cao et al. 2004). Stenflo et al.(1987)는 $B \ge 1$ kG에서 선이 *완전 분리*됨을 최초로 보였다: 제만 σ 이동이 도플러 폭을 초과하여 $B \cdot f$ (filling factor×자기장)이 아닌 $B$ 자체를 직접 드러낸다. 이로써 수십 년간의 논쟁이 해결되었다. Lin(1995)의 자기장 세기 히스토그램은 활동영역 plage 300–2000 G(평균 1400 G), 조용한 태양 200–2000 G(평균 500 G)을 보였다. Khomenko et al.(2005)는 1565선과 630선 결합으로 네트워크 내부 자기장 추정치를 약 20 G로 한 자릿수 낮추었다. 흑점 연구(Kopp & Rabin 1992; Mathew et al. 2003)는 수평 자기정역학 평형과 일치하는 $B \propto T^2$ 관계를 발견했다. Bellot Rubio et al.(2000)은 1565 nm Stokes 프로파일의 SIR 역해법으로 본영 헬리오사이스몰로지를 수행, 속도/자기장 진동 위상차 105°±30°(단열값 90°와 대비)를 측정했다.

### Part VIII: He I 1083 nm / He I 1083 nm (Sec. 3.3, p. 24–30)

**English.** A uniquely versatile chromospheric-coronal diagnostic. Discovered on the disk by Babcock & Babcock (1934). Formation height $\sim 1.4$–$2.4$ Mm above the limb, formed by a "mixed" PR (photoionization/recombination) + collisional excitation mechanism (Andretta & Jones 1997; Centeno et al. 2009). Key diagnostics:
- **Coronal holes** appear as regions of *reduced* He I 1083 absorption (Harvey et al. 1975), reversed in sign vs. X-ray emission. The 11-year cycle of polar coronal holes has been tracked via He I 1083 (Harvey & Recely 2002).
- **Prominences** show optical thickness varying from thin to $\tau = 2$, with temperatures as low as 3750 K (Stellmacher et al. 2003); density $\sim 10^{10}$ cm$^{-3}$ (Heasley et al. 1975).
- **Flares** (Penn & Kuhn 1995; Penn 2006) reveal downflows to 100 km/s, Zeeman signatures of $\sim 735$ G in flare kernels, and filament blueshifts of 200–300 km/s.
- **Vector magnetic fields** in emerging active regions, filaments, and fibrils via full-Stokes inversion using HELIx+ and HAZEL codes (Lagg et al. 2009; Schad et al. 2012, 2013; Trujillo Bueno & Asensio Ramos 2007). Figure 13 shows magnetic fields of 100–1000 G measured between 10 and 50 Mm in a coronal condensation event.

**한국어.** 독특하게 다재다능한 채층-코로나 진단선. Babcock & Babcock(1934)에 의해 원반에서 발견. 형성 높이 1.4–2.4 Mm(가장자리 위), "혼합" PR(광이온화/재결합) + 충돌 여기 메커니즘(Andretta & Jones 1997; Centeno et al. 2009). 주요 진단:
- **코로나홀**: He I 1083 흡수가 *감소*한 영역으로 나타남(Harvey et al. 1975), X선 방출과 부호 반대. 11년 주기의 극 코로나홀을 He I 1083으로 추적(Harvey & Recely 2002).
- **프로미넌스**: 광학 두께가 얇은 것부터 $\tau=2$까지, 온도 최저 3750 K(Stellmacher et al. 2003); 밀도 $\sim 10^{10}$ cm$^{-3}$(Heasley et al. 1975).
- **플레어**(Penn & Kuhn 1995; Penn 2006): 100 km/s 하향류, 플레어 kernel의 $\sim 735$ G 제만 서명, 필라멘트의 200–300 km/s 블루시프트.
- **벡터 자기장**: HELIx+와 HAZEL 코드로 풀-Stokes 역해(Lagg et al. 2009; Schad et al. 2012, 2013; Trujillo Bueno & Asensio Ramos 2007). Figure 13은 코로나 응축 이벤트에서 10–50 Mm 고도의 100–1000 G 자기장 측정을 보여 준다.

### Part IX: Mg I 12 318 nm — The Most Sensitive Probe / Mg I 12 318 nm — 가장 민감한 자기 탐침 (Sec. 3.4, p. 30–34)

**English.** Identified by Chang & Noyes (1983) as a high-$\ell$ Rydberg transition. With $g_{\rm eff} = 1.0$ and $\lambda = 12\,318$ nm, $g_{\rm eff}\lambda = 12\,318$ — the largest known in the solar spectrum. Observations require the McM-P telescope due to low solar flux and high thermal background. Detector evolution: arsenic-doped silicon photodiode (Brault & Noyes 1983) → Celeste instrument's 128×128 Si:As BIB array (McCabe et al. 2003). Formation height $z \approx 400$ km (upper photosphere), confirmed by eclipse measurements and NLTE modelling (Carlsson et al. 1992). Deming et al. (1988) saw Zeeman-split emission profiles in sunspots with fields 850–1400 G. Hewagama et al. (1993) first full-Stokes at 12 318 nm. Moran et al. (2000) showed different vertical field gradients in sunspot/plage between 1565 and 12 318 nm (Figure 15). Jennings et al. (2002) produced field-strength maps using Zeeman-shifted σ components directly — bypassing Stokes V saturation — and found flare magnetic energies exceeding X-ray luminosity.

**한국어.** Chang & Noyes(1983)에 의해 높은 $\ell$ Rydberg 전이로 식별됨. $g_{\rm eff}=1.0$, $\lambda=12\,318$ nm → $g_{\rm eff}\lambda = 12\,318$로 태양 스펙트럼에서 가장 큰 값. 낮은 태양 flux와 높은 열배경 때문에 McM-P 망원경이 주된 관측지이다. 검출기 진화: arsenic-doped silicon 광다이오드(Brault & Noyes 1983) → Celeste의 128×128 Si:As BIB 어레이(McCabe et al. 2003). 형성 높이 $z \approx 400$ km(상부 광구), 일식 측정과 NLTE 모델(Carlsson et al. 1992)로 확인. Deming et al.(1988)은 흑점에서 제만 분리된 방출 프로파일(850–1400 G)을 관측했다. Hewagama et al.(1993)는 12 318 nm 최초 풀-Stokes 관측. Moran et al.(2000)은 흑점/plage에서 1565 nm와 12 318 nm 사이의 수직 자기장 경사 차이를 보였다(Figure 15). Jennings et al.(2002)은 제만 이동된 σ 성분으로 직접 자기장 지도를 만들어 Stokes V 포화를 우회, 플레어 자기 에너지가 X선 광도를 초과함을 발견했다.

### Part X: Coronal Measurements / 코로나 관측 (Sec. 3.5, p. 34–38)

**English.** The [Fe XIII] 1074.7 nm coronal forbidden line, first seen by Lyot in 1936 and identified by Edlén in 1942, is the flagship IR coronal diagnostic. Table 3 lists key coronal IR lines: Fe XIII 1074.7/1079.8, Si X 1430, S XI 1920, Si IX 2584, Fe IX 2855, Mg VIII 3028, Si IX 3934, Mg VII 5502/9031. The 1075/1080 pair ratio is electron-density sensitive (Flower & Pineau des Forêts 1973; Penn et al. 1994 Figure 17). Direct magnetic-field measurements: Kuhn (1995) upper limit 40 G from 1075 Stokes V; Lin et al. (2000) found 33 G; Lin et al. (2004) mapped active-region fields $\sim 4$ G (Figure 20). COMP (Tomczyk et al. 2008) measures Stokes I, Q, U, V at three wavelength positions with 4.5″ pixels over 1.05–1.35 $R_\odot$. Doppler-shift oscillations with rms velocity 300 m/s, periods near 5 min, are interpreted as outward-propagating magneto-acoustic kink waves carrying $\sim 100$ erg cm$^{-2}$ s$^{-1}$ (Tomczyk & McIntosh 2009, Figure 21).

**한국어.** [Fe XIII] 1074.7 nm 코로나 금지선은 Lyot이 1936년 최초 관측, Edlén이 1942년 식별한 대표 IR 코로나 진단선이다. Table 3은 주요 코로나 IR 선(Fe XIII 1074.7/1079.8, Si X 1430, S XI 1920, Si IX 2584, Fe IX 2855, Mg VIII 3028, Si IX 3934, Mg VII 5502/9031)을 정리한다. 1075/1080 선비는 전자밀도에 민감하다(Flower & Pineau des Forêts 1973; Penn et al. 1994 Figure 17). 자기장 직접 측정: Kuhn(1995)는 1075 Stokes V로 40 G 상한; Lin et al.(2000)은 33 G; Lin et al.(2004)은 활동영역 자기장 지도 $\sim 4$ G(Figure 20). COMP(Tomczyk et al. 2008)는 4.5″ 픽셀로 1.05–1.35 $R_\odot$ 구간에서 세 파장 위치의 Stokes I, Q, U, V를 측정한다. rms 속도 300 m/s, 주기 5분의 도플러 진동은 $\sim 100$ erg cm$^{-2}$ s$^{-1}$를 운반하는 외향 자기-음향 kink 파로 해석된다(Tomczyk & McIntosh 2009, Figure 21).

### Part XI: Miscellaneous / 기타 (Sec. 3.6, p. 39–43)

**English.** (3.6.1) Granulation contrast *reverses* with height: in the IR continuum at deep layers, granule centers are bright; higher up (visible/UV continuum) the reverse is true. Penn (2008) imaged granulation at 1100/1600/2200 nm (Figure 22) with unexpectedly large 2200 nm K-band contrast. (3.6.2) Molecular spectropolarimetry of OH 1541 nm, CN 1100 nm and FeH 1006/990 nm reveals anomalous "negative Landé-$g$" Stokes V profiles explained by Paschen-Back effects (Berdyugina & Solanki 2001; Asensio Ramos et al. 2005). (3.6.3) Mn I 1526 nm hyperfine splitting (Asensio Ramos et al. 2007) allows detection of fields down to 80 G and resolves filling-factor ambiguity. (3.6.4) Ti I 2231 nm lines probe coolest umbral parts (500–1400 G) where TiO molecules form (Rüedi et al. 1998; Penn et al. 2003).

**한국어.** (3.6.1) 과립 대비가 *높이에 따라 반전*된다: IR 연속체의 깊은 층에서는 과립 중심이 밝고, 상부(가시광/UV 연속체)에서는 반대이다. Penn(2008)은 1100/1600/2200 nm에서 과립을 촬영(Figure 22), 2200 nm K-band에서 예상외로 큰 대비를 발견했다. (3.6.2) OH 1541 nm, CN 1100 nm, FeH 1006/990 nm 분자 분광편광은 Paschen-Back 효과로 설명되는 "음의 Landé-$g$" 이상 Stokes V 프로파일을 보인다(Berdyugina & Solanki 2001; Asensio Ramos et al. 2005). (3.6.3) Mn I 1526 nm 초미세 분리(Asensio Ramos et al. 2007)는 80 G까지 자기장 검출을 가능케 하고 filling-factor 모호성을 해결한다. (3.6.4) Ti I 2231 nm 선은 TiO 분자가 형성되는 가장 차가운 본영(500–1400 G)을 진단한다(Rüedi et al. 1998; Penn et al. 2003).

### Part XII: Future Prospects / 미래 전망 (Sec. 4, p. 44–47)

**English.** (4.1) New instruments: **DKIST/ATST** (4-m all-reflecting, Haleakalā) with Cryo-NIRSP and DL-NIRSP will deliver 0.08″ at 1565 nm, 3× the resolution of McM-P/NST; **GREGOR** (1.5-m Tenerife) with **GRIS** spectrograph (1000–2300 nm); **Cyra** on NST; **COSMO** (large coronagraph for [Fe XIII]) — more synoptic than DKIST Cryo-NIRSP. SPIES instrument demonstrated simultaneous 1083/1565 nm spectropolarimetry using a $64\times32$ fiber array feeding a 2k×2k detector. (4.2) Next key wavelengths: 4000 nm (weak Fe I lines at 4135 nm target for Cyra) and 5000–10 000 nm (mostly unexplored). (4.3) Spectropolarimetry of molecules — underdeveloped theory. (4.4) Space-based solar IR: only one upcoming JAXA IR spectropolarimeter; SIRE mission (Deming et al. 1991) was proposed but not flown. A Hinode-class mission with 3× sensitivity would be revolutionary.

**한국어.** (4.1) 신기기: **DKIST/ATST**(4-m 전반사, 할레아칼라)의 Cryo-NIRSP와 DL-NIRSP는 1565 nm에서 0.08″ 해상도(McM-P/NST의 3배); **GREGOR**(1.5-m, 테네리페)의 **GRIS** 분광기(1000–2300 nm); NST의 **Cyra**; **COSMO**([Fe XIII] 대형 코로나그래프, DKIST Cryo-NIRSP보다 synoptic). SPIES 기기는 $64\times32$ 광섬유 어레이와 2k×2k 검출기로 1083/1565 nm 동시 분광편광을 시연했다. (4.2) 다음 핵심 파장: 4000 nm(Cyra가 타깃하는 4135 nm 부근 약한 Fe I 선) 및 5000–10 000 nm(거의 미탐구). (4.3) 분자 분광편광 — 이론 미성숙. (4.4) 우주 기반 태양 IR: JAXA의 단 하나의 IR 분광편광기 제안만 존재; SIRE 미션(Deming et al. 1991)은 제안되었으나 비행하지 않음. Hinode 대비 3배 민감도 미션이 혁명적일 것이다.

---

## 3. Key Takeaways / 핵심 시사점

1. **IR Zeeman splitting scales as $\lambda^2$ while Doppler broadening scales as $\lambda$ — magnetic resolution therefore grows as $g_{\rm eff}\lambda$, making IR the preferred regime for accurate solar magnetic field measurements.** / **IR 제만 분리는 $\lambda^2$, 도플러 폭은 $\lambda$로 증가하므로 자기분해능은 $g_{\rm eff}\lambda$에 비례. 따라서 IR은 정확한 태양 자기장 측정에 최적 영역이다.**
   English. Practically, the Fe I 1565 nm line ($g=3$) is fully split at 1 kG, letting observers measure $B$ directly without the filling-factor ambiguity that plagued the pre-1987 visible-only era. Moving from 500 nm to 5000 nm alone improves magnetic sensitivity tenfold.
   한국어. 실제로 Fe I 1565 nm 선($g=3$)은 1 kG에서 완전 분리되어 filling-factor 모호성 없이 $B$를 직접 측정할 수 있게 한다(1987년 이전 가시광 시대의 한계 해소). 500 nm → 5000 nm만으로도 자기민감도가 10배 향상된다.

2. **The CO 4666 nm lines revealed a "COmosphere" — regions of cool ($\sim 3500$ K) plasma coexisting with hot canopy structures at chromospheric heights.** / **CO 4666 nm 선은 채층 고도에서 뜨거운 canopy와 공존하는 차가운($\sim 3500$ K) "COmosphere" 영역을 드러냈다.**
   English. This resolved the Ca II/Mg II H&K vs. CO absorption paradox and forced revision of static plane-parallel atmospheric models into dynamic, inhomogeneous frameworks (Wedemeyer-Böhm et al. 2007).
   한국어. 이는 Ca II/Mg II H&K 방출과 CO 흡수의 모순을 해결하고 정적 평행 대기 모델을 동적·비균질 모델로 바꾸는 계기가 되었다(Wedemeyer-Böhm et al. 2007).

3. **Mg I 12 318 nm remains the most magnetically sensitive line known in the solar spectrum, with $g_{\rm eff}\lambda = 12\,318$.** / **Mg I 12 318 nm는 $g_{\rm eff}\lambda = 12\,318$로 태양 스펙트럼에서 알려진 가장 자기민감한 선이다.**
   English. Zeeman-shifted σ components can be used *directly* (bypassing Stokes V saturation issues) to map field strength distributions, revealing flare magnetic energies orders of magnitude larger than X-ray luminosity (Jennings et al. 2002).
   한국어. 제만 이동된 σ 성분을 직접 사용해(Stokes V 포화 우회) 자기장 분포를 지도화할 수 있고, 플레어 자기 에너지가 X선 광도를 여러 자릿수 초과함을 밝혔다(Jennings et al. 2002).

4. **He I 1083 nm is uniquely versatile — a single line diagnoses coronal holes (reduced absorption), prominences ($T \sim 3750$ K, $n \sim 10^{10}$ cm$^{-3}$), flare dynamics (100 km/s downflows), and chromospheric vector magnetic fields.** / **He I 1083 nm는 독특하게 다재다능하다 — 단일 선으로 코로나홀(흡수 감소), 프로미넌스($T \sim 3750$ K, $n \sim 10^{10}$ cm$^{-3}$), 플레어 역학(100 km/s 하향류), 채층 벡터 자기장을 모두 진단한다.**
   English. Full-Stokes inversions with HELIx+ and HAZEL codes (Lagg et al. 2009; Schad et al. 2013) exploit the Hanle effect to measure coronal-height magnetic fields up to 1000 G at 10 Mm.
   한국어. HELIx+와 HAZEL(Lagg et al. 2009; Schad et al. 2013)의 풀-Stokes 역해는 Hanle 효과를 활용해 10 Mm에서 1000 G까지의 코로나 고도 자기장을 측정한다.

5. **[Fe XIII] 1074.7 nm is the primary tool for direct coronal magnetic-field measurement; fields of $\sim 4$ G above active regions are routinely measured with COMP.** / **[Fe XIII] 1074.7 nm는 코로나 자기장 직접 측정의 주된 도구이며, COMP로 활동영역 상공 약 4 G 자기장이 일상적으로 측정된다.**
   English. Tomczyk et al. (2008) detected propagating transverse waves (kink modes) carrying $\sim 100$ erg cm$^{-2}$ s$^{-1}$, far short of the $10^6$ erg cm$^{-2}$ s$^{-1}$ required for coronal heating — suggesting unresolved small-scale modes carry more energy.
   한국어. Tomczyk et al.(2008)은 $\sim 100$ erg cm$^{-2}$ s$^{-1}$를 운반하는 전파 횡파(kink mode)를 검출했으나, 코로나 가열에 필요한 $10^6$ erg cm$^{-2}$ s$^{-1}$에 크게 못 미쳐 미해결 소규모 모드가 더 큰 에너지를 운반함을 시사한다.

6. **The H$^-$ free-free continuum opacity at $\lambda > 1600$ nm enables IR continuum observations to probe specific photospheric heights from $z = -40$ km to $z = +140$ km, where granulation contrast reverses sign.** / **$\lambda > 1600$ nm의 H$^-$ free-free 연속체 불투명도는 IR 연속체 관측으로 $z = -40$ km ~ $+140$ km의 특정 광구 높이를 진단케 하며, 이 구간에서 과립 대비가 부호 반전한다.**
   English. Models by Leenaarts & Wedemeyer-Böhm (2005) and Cheung et al. (2007) predict reverse granulation at $z \sim 130$ km — observable in the 2200 nm K-band (Penn 2008).
   한국어. Leenaarts & Wedemeyer-Böhm(2005)와 Cheung et al.(2007) 모델은 $z \sim 130$ km에서 역과립을 예측하며, 이는 2200 nm K-band에서 관측 가능하다(Penn 2008).

7. **IR observing trades spatial resolution and photon flux for magnetic sensitivity, atmospheric seeing stability, and reduced scattering/polarization — a favorable compromise for polarimetric science.** / **IR 관측은 공간 해상도와 광자 flux를 자기 민감도·seeing 안정성·산란/편광 감소와 교환한다. 편광 과학에는 유리한 절충이다.**
   English. The $\lambda^{-3/2}$ S/N penalty and the $\theta \propto \lambda$ diffraction limit are offset by the $\lambda^2$ Zeeman gain, $\lambda^{6/5}$ seeing improvement, $\sim \lambda^{-2}$ scattering reduction, and $\sim \lambda^{-2}$ instrumental polarization reduction.
   한국어. $\lambda^{-3/2}$의 S/N 손실과 $\theta \propto \lambda$ 회절 한계는 $\lambda^2$ 제만 이득, $\lambda^{6/5}$ seeing 개선, $\sim \lambda^{-2}$ 산란 감소, $\sim \lambda^{-2}$ 기기 편광 감소로 상쇄된다.

8. **DKIST (4-m Haleakalā) will deliver 0.08″ diffraction-limited spectropolarimetry at 1565 nm — a factor-of-three improvement over any previous facility — opening quiet-Sun magnetoconvection and coronal magnetometry.** / **DKIST(4 m, 할레아칼라)는 1565 nm에서 0.08″ 회절한계 분광편광을 제공하여 기존 시설 대비 3배 개선, 조용한 태양 자기대류와 코로나 자기장 연구를 개척할 것이다.**
   English. Cryo-NIRSP (cryogenic IR) and DL-NIRSP (dual-beam imaging spectropolarimeter) observe 1000–5000 nm. COSMO's large coronagraph complements DKIST with synoptic [Fe XIII] coverage.
   한국어. Cryo-NIRSP(극저온 IR)와 DL-NIRSP(이중빔 영상 분광편광기)는 1000–5000 nm를 관측한다. COSMO의 대형 코로나그래프는 [Fe XIII]의 synoptic 관측으로 DKIST를 보완한다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Zeeman splitting (IR advantage) / 제만 분리(IR 이점)

The wavelength displacement of Zeeman σ components from line center in a longitudinal field $B$:

$$\Delta\lambda_B = \frac{e}{4\pi m_e c^2}\, g_{\rm eff}\, \lambda^2\, B = 4.67\times 10^{-5}\, g_{\rm eff}\, \lambda_{\rm nm}^2\, B_{\rm G}\ [\mathrm{m\AA}]$$

- $e$, $m_e$, $c$: electron charge, mass, speed of light / 전자 전하·질량·광속
- $g_{\rm eff}$: effective Landé factor (atomic structure) / 유효 Landé 인자
- $\lambda$: rest wavelength / 정지 파장 — **the key: squared dependence gives IR its magnetic edge** / 제곱 의존성이 IR 자기 이점의 핵심
- $B$: longitudinal magnetic field / 시선방향 자기장

**Worked numerical example / 수치 예시:**
- Visible Fe I 5250.22 Å, $g_{\rm eff} = 3$, $B = 1000$ G:
  $$\Delta\lambda = 4.67\times 10^{-5} \times 3 \times (525)^2 \times 1000 \approx 38.6\ \mathrm{m\AA}$$
  Doppler width $\sim 25$ mÅ → ratio $\sim 1.5$ (weak-field regime).
- IR Fe I 1564.85 nm, $g_{\rm eff} = 3$, $B = 1000$ G:
  $$\Delta\lambda = 4.67\times 10^{-5} \times 3 \times (1565)^2 \times 1000 \approx 343\ \mathrm{m\AA}$$
  Doppler width $\sim 75$ mÅ → ratio $\sim 4.6$ (**fully split**).
- Mg I 12 318 nm, $g_{\rm eff} = 1$, $B = 1000$ G:
  $$\Delta\lambda = 4.67\times 10^{-5} \times 1 \times (12\,318)^2 \times 1000 \approx 7090\ \mathrm{m\AA}$$
  — enormous splitting even at 1 kG.

### 4.2 Diffraction limit / 회절 한계

$$\theta = 1.22\,\frac{\lambda}{D},\qquad \theta[\mathrm{arcsec}] \approx \frac{0.2\,\lambda_{\rm nm}}{D_{\rm mm}}$$

- For DKIST ($D = 4000$ mm) at $\lambda = 1565$ nm: $\theta \approx 0.2 \times 1565 / 4\,000\,000 \times 10^6 \approx 0.078''$.
- 4 m DKIST에서 1565 nm: $\theta \approx 0.078''$로 McM-P/NST의 약 1/3.

### 4.3 Planck function & Rayleigh-Jeans limit / 플랑크 함수와 Rayleigh-Jeans 극한

Full form:
$$B_\lambda(T) = \frac{2hc^2}{\lambda^5}\frac{1}{\exp(hc/\lambda k_B T)-1}$$

Rayleigh-Jeans limit ($\lambda k_B T \gg hc$):
$$B_\lambda(T) \xrightarrow{\lambda \to \infty} \frac{2ckT}{\lambda^4}$$

At fixed spectral resolving power $R = \lambda/\Delta\lambda$ the detected photon number per unit time per unit area per bin is:
$$N_\gamma \propto \frac{B_\lambda \Delta\lambda}{h c/\lambda} \propto \lambda^{-3}\ (\text{Rayleigh-Jeans})$$
so signal-to-noise decreases as $N_\gamma^{1/2} \propto \lambda^{-3/2}$ — the fundamental IR photon-flux penalty.

**Example:** For solar $T \sim 5800$ K, at $\lambda = 1565$ nm: $hc/\lambda k_B T = 19877.6/(1565 \times 10^{-9} \cdot 5800 \cdot 8.617\times 10^{-5}\ \mathrm{eV/K}) = \ldots \approx 1.59$ (not yet RJ); at 12 $\mu$m: $\approx 0.207$ (clearly RJ).

### 4.4 Fried parameter & AO timescale / Fried 파라미터와 AO 시간규모

$$r_0(\lambda) \propto \lambda^{6/5}$$

Isoplanatic patch $\theta_0 \propto r_0$, coherence time $T_{\rm AO} \propto r_0/v_{\rm atm}$. From 500 nm to 1565 nm, $r_0$ grows by $(1565/500)^{6/5} \approx 4.0$.

### 4.5 H-minus opacity and height of formation / H-minus 불투명도와 형성 높이

VAL fit for $\tau_\lambda = 1$ height:
$$z_{\tau=1}(\lambda) \approx -776 - 227\log_{10}\left(\frac{1}{\lambda_{\rm nm}}\right)\ [\mathrm{km}]$$
(valid 2000–20 000 nm).

At 1600 nm: $z_{\tau=1} \approx -776 - 227 \log_{10}(1/1600) = -776 + 727 \approx -49$ km (below $\tau_{500}=1$).
At 10 000 nm: $z_{\tau=1} \approx -776 + 908 \approx 132$ km.

### 4.6 Mueller matrix for a single mirror / 단일 거울의 Mueller 행렬

$$M = \begin{pmatrix} 1+X^2 & 1-X^2 & 0 & 0 \\ 1-X^2 & 1+X^2 & 0 & 0 \\ 0 & 0 & 2X\cos\tau & 2X\sin\tau \\ 0 & 0 & -2X\sin\tau & 2X\cos\tau \end{pmatrix}$$
In the limit $X \to 1$, $\tau \to 0$ (IR for aluminum): $M \to \mathrm{diag}(2, 2, 2, 2)$ — purely diagonal, no instrumental polarization cross-talk.

### 4.7 CO molecular formation / CO 분자 형성

Classical rotational energy balance:
$$\tfrac{1}{2}I\omega^2 \sim \tfrac{1}{2}k_B T \quad\Rightarrow\quad \omega \sim \sqrt{k_B T / (m r^2)}$$

For $T = 6000$ K, $m = m_{\rm O}$, $r = r_{\rm OH}$: $\omega \sim 10^4$ GHz → $\lambda \sim 25\,\mu$m, confirming why molecular rotation-vibration bands populate the IR spectrum.

### 4.8 Total integrated scatter (TIS) / 총 적분 산란

$$\mathrm{TIS} \approx (\sigma/\lambda)^2$$

For observed solar scattering: $\mathrm{TIS} = 0.026 + 0.06\,(1+\lambda/1000\,\mathrm{nm})\,e^{-\lambda/1000\,\mathrm{nm}}$ (Johnson 1972).

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1800 ─── Herschel: 적외선 발견 (온도계) / Discovery of IR radiation by thermometer
   │
1869 ─── Janssen: He 발견 (일식, visible D3선) / Helium discovered in eclipse
   │
1934 ─── Babcock & Babcock: He I 1083 nm on disk / 원반에서 He I 1083 흡수선 관측
   │
1936 ─── Lyot: Pic du Midi에서 [Fe XIII] 1075 nm 관측 / First coronal IR line
   │
1942 ─── Edlén: [Fe XIII] as Fe^12+ 식별 / Identification as highly-ionized iron
   │
1962 ─── Johnson: J/H/K/L/M 밴드 정립 / Johnson's IR photometry bands established
   │
1969 ─── Hall & Noyes: HF/Fe I 1565 nm PbS 검출기 관측 / Early IR detector solar work
   │
1971 ─── Harvey & Hall: He I 1083 nm 자기장 측정 / First He I 1083 magnetograph
   │
1975 ─── Harvey & Hall: Fe I 1565 nm 흑점 관측 / Sunspot observations at 1565 nm
   │
1976 ─── Vernazza, Avrett, Loeser: VAL 대기 모델 / VAL semi-empirical atmosphere
   │
1983 ─── Brault & Noyes: Mg I 12 318 nm 방출선 식별 / Mg I 12 μm emission identified
   │
1987 ─── Stenflo et al.: Fe I 1565 nm 완전 분리 / Fully-resolved Zeeman splitting
   │       (IR 태양 자기학의 전환점 / turning point of IR solar magnetism)
   │
1991 ─── Hawaii 일식: IR 코로나 서베이 / Mauna Kea eclipse IR coronal survey
   │
1992 ─── IAU Symposium 154 "Infrared Solar Physics" (Jefferies 편집)
   │       McM-P 첫 InSb 어레이 / First InSb array at McM-P
   │
2001 ─── ATST 과학 요구사항 수락 / ATST science requirements accepted
   │       (후에 DKIST로 개명 / later renamed DKIST)
   │
2008 ─── Tomczyk et al.: COMP [Fe XIII] 코로나 파동 검출 / CoMP Alfvén-like waves
   │
2010 ─── Jaeggli et al.: NSO/DST FIRS 가동 / FIRS 1024×1024 HgCdTe instrument
   │
2014 ─── ★ Penn: 본 리뷰 / THIS REVIEW — comprehensive LRSP synthesis ★
   │       DKIST 건설 중, GREGOR/GRIS 초기 자료 / DKIST under construction
   │
2019 ─── DKIST 첫빛 / DKIST first light on Haleakalā
   │
2020+ ── COSMO 추진, Cryo-NIRSP 관측 시작 / COSMO proposal; Cryo-NIRSP science
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Vernazza, Avrett & Loeser (1976) — "Structure of the solar chromosphere III"** | Provides the VAL semi-empirical atmospheric model; Penn uses it extensively for height-of-formation analysis in §2.3.3 and Fig. 5. / Penn이 §2.3.3과 Fig. 5에서 형성 높이 분석의 기반으로 사용하는 VAL 반경험 대기 모델. | **Foundational** — every radiative transfer comparison in the review stands on VAL. / 이 리뷰의 모든 복사전달 비교의 기반. |
| **Stenflo, Solanki & Harvey (1987) — "Diagnostics of solar magnetic flux tubes using a Fourier transform spectrometer"** | First demonstration of fully-split Zeeman profiles of Fe I 1565 nm for $B \ge 1$ kG, as reproduced in Figure 4. / Fe I 1565 nm의 1 kG 이상 완전 분리 Zeeman 프로파일 최초 시연; Figure 4에 재현. | **Paradigm-setting** — established IR as the solution to the filling-factor ambiguity. / IR이 filling-factor 모호성의 해법임을 확립. |
| **Ayres (2002) — "Does the Sun have a full-time COmosphere?"** | The "COmosphere" concept that motivates the paradigm shift from static to dynamic chromosphere models (Figure 6). / 정적 → 동적 채층 모델 전환을 유발한 "COmosphere" 개념(Figure 6). | **Key paradigm** — §3.1 is structured around this thermal-bifurcation debate. / §3.1의 핵심 주제. |
| **Tomczyk et al. (2008) — "Alfvén waves in the solar corona"** | COMP observation of propagating transverse waves in [Fe XIII] 1074.7 nm Doppler shifts, central to §3.5. / [Fe XIII] 1074.7 nm 도플러 이동의 전파 횡파 관측; §3.5 핵심 결과. | **Direct data input** — the single most cited coronal-magnetometry result. / 코로나 자기장 측정의 대표 사례. |
| **Judge et al. (2002) — "Spectroscopic detection of the 3.934 μm line of [Si IX]"** | Discovery paper for [Si IX] 3934 nm as a future coronal magnetic-field probe (Figure 19). / [Si IX] 3934 nm의 코로나 자기장 진단 잠재력(Figure 19). | **Forward-looking** — informs Penn's future-prospects section. / Penn의 미래 전망의 근거. |
| **Trujillo Bueno & Asensio Ramos (2007) — "Influence of atomic polarization and horizontal radiation transfer on the Hanle-effect signatures of He I 10830"** | The theoretical framework for He I 1083 nm spectropolarimetry that Penn cites in §3.3.7. / Penn이 §3.3.7에서 인용하는 He I 1083 nm 분광편광의 이론 기반. | **Theory backbone** for chromospheric magnetometry with He I. / He I 채층 자기학의 이론 기초. |
| **Jaeggli et al. (2010) — "FIRS: a new instrument for photospheric and chromospheric studies at the DST"** | Description of the FIRS multi-slit spectropolarimeter at NSO/DST, representative of the modern IR instrumentation era. / NSO/DST의 FIRS 다중슬릿 분광편광기, 현대 IR 기기 대표. | **Instrument flagship** — context for the "now" of the review. / 이 리뷰 "현재"의 기기적 맥락. |
| **Rimmele & Marino (2011) — "Solar adaptive optics"** (Living Reviews) | The AO Living Review complementing Penn's §2.1.1 on atmospheric seeing. / Penn §2.1.1의 대기 seeing 보완하는 AO Living Review. | **Companion review** cited explicitly by Penn for AO details. / Penn이 AO 상세 참조. |

---

## 7. References / 참고문헌

- Penn, M. J., "Infrared Solar Physics", *Living Reviews in Solar Physics*, **11**, 2 (2014). [DOI:10.12942/lrsp-2014-2]
- Vernazza, J. E., Avrett, E. H., & Loeser, R., "Structure of the solar chromosphere. II. The underlying photosphere and temperature-minimum region", *Astrophys. J. Suppl.*, **30**, 1–60 (1976).
- Stenflo, J. O., Solanki, S. K., & Harvey, J. W., "Diagnostics of solar magnetic flux tubes using a Fourier transform spectrometer", *Astron. Astrophys.*, **173**, 167–179 (1987).
- Jefferies, J. T., "Overview of Infrared Solar Physics", IAU Symposium 154, Kluwer (1994).
- Ayres, T. R., "Does the Sun have a full-time COmosphere?", *Astrophys. J.*, **575**, 1104–1115 (2002).
- Wedemeyer-Böhm, S. & Steffen, M., "CO-mosphere simulation with time-dependent chemistry", *Astron. Astrophys.*, **462**, L31–L35 (2007).
- Harvey, J. & Hall, D., "Magnetic fields measured with the 10830 Å He I line", IAU Symp. 43, 279 (1971).
- Tomczyk, S. et al., "Alfvén waves in the solar corona", *Science*, **317**, 1192–1196 (2007).
- Tomczyk, S. & McIntosh, S. W., "Time-distance seismology of the solar corona with CoMP", *Astrophys. J.*, **697**, 1384–1391 (2009).
- Jennings, D. E. et al., "Solar magnetic field studies using the 12 micron emission lines. IV. Observations of a Delta Region Solar Flare", *Astrophys. J.*, **568**, 1043–1048 (2002).
- Lin, H., Penn, M. J., & Tomczyk, S., "A new precise measurement of the coronal magnetic field strength", *Astrophys. J. Lett.*, **541**, L83 (2000); follow-up Lin et al. *Astrophys. J. Lett.* **613**, L177 (2004).
- Kuhn, J. R., Penn, M. J., & Mann, I., "The near-infrared coronal spectrum", *Astrophys. J. Lett.*, **456**, L67 (1996).
- Judge, P. G. et al., "Spectroscopic detection of the 3.934-micron line of [Si IX] in the solar corona", *Astrophys. J. Lett.*, **576**, L157 (2002).
- Asensio Ramos, A., Martínez González, M. J., López Ariste, A., Trujillo Bueno, J., Collados, M., "A near-infrared line of Mn I as a diagnostic tool of the average magnetic energy in the solar photosphere", *Astrophys. J.*, **659**, 829–847 (2007).
- Trujillo Bueno, J. & Asensio Ramos, A., "Influence of atomic polarization on He I 10830", *Astrophys. J.*, **655**, 642–667 (2007).
- Penn, M. J. & Kuhn, J. R., "He I 1083 nm observations of a flare", *Astrophys. J. Lett.*, **441**, L51 (1995).
- Rimmele, T. R. & Marino, J., "Solar adaptive optics", *Living Reviews in Solar Physics*, **8**, 2 (2011).
- Keil, S. L. et al., "The Advanced Technology Solar Telescope", ASP Conf. Ser. **236**, 597 (2001). [DKIST/ATST]
- Schmidt, W. et al., "The 1.5 meter solar telescope GREGOR", *Astron. Nachr.*, **333**, 796 (2012).
- Collados, M. et al., "GRIS: The GREGOR Infrared Spectrograph", *Astron. Nachr.*, **333**, 872 (2012).
- Jaeggli, S. A. et al., "FIRS: a new instrument for photospheric and chromospheric studies at the DST", *Mem. Soc. Astron. Ital.*, **81**, 763 (2010).
- Leenaarts, J. & Wedemeyer-Böhm, S., "Reverse granulation in the solar photosphere", *Astron. Astrophys.*, **431**, 687 (2005).
- Cheung, M. C. M. et al., "Origin of the reversed granulation in the solar photosphere", *Astron. Astrophys.*, **461**, 1163 (2007).
- Bellot Rubio, L. R. et al., "Oscillations in sunspot umbra from inversion of infrared Stokes profiles", *Astrophys. J.*, **534**, 989 (2000).
- Khomenko, E. V. et al., "Magnetic flux in the internetwork quiet Sun", *Astron. Astrophys.*, **436**, L27 (2005).
- Lagg, A. et al., "Internetwork horizontal magnetic fields in the quiet Sun chromosphere", ASP Conf. Ser. **415**, 327 (2009).
- Schad, T. A. et al., "Full-disk vector magnetic measurements using He I 1083 nm", *Astrophys. J.*, **768**, 111 (2013).
- Hall, D. N. B. & Noyes, R. W., "Observation of hydrogen fluoride in sunspots", *Astrophys. Lett.*, **4**, 143 (1969).
- Penn, M. J. & Kuhn, J. R., "He I 1083 nm spectropolarimetry of active regions", *Astrophys. J.*, **441**, L51 (1995).
- Brault, J. & Noyes, R., "Solar emission lines near 12 microns", *Astrophys. J.*, **269**, L61 (1983).
- Deming, D. et al., "Solar magnetic field studies using the 12 micron emission lines. I.", *Astrophys. J.*, **333**, 978 (1988).
- Lin, H., "On the distribution of the solar magnetic fields", *Astrophys. J.*, **446**, 421 (1995).
- Moran, T. et al., "Simultaneous measurements of sunspot magnetic fields at 1565 and 12 318 nm", *Astrophys. J.*, **543**, 509 (2000).
- Mathew, S. K. et al., "Sunspot magnetic field inversions including molecular lines", *Astron. Astrophys.*, **410**, 695 (2003).
- Harvey, J. W. & Recely, F., "Polar coronal holes during cycles 22 and 23", *Solar Phys.*, **211**, 31 (2002).
- Kopp, G. & Rabin, D., "Relation between magnetic field and temperature in sunspots", *Solar Phys.*, **141**, 253 (1992).
- Penn, M. J. (2008), "Imaging granulation at near-infrared wavelengths", Solar Physics.
- Berdyugina, S. V. & Solanki, S. K., "Zeeman-split OH lines in sunspot spectra", *Astron. Astrophys.*, **380**, L5 (2001).
- Carlsson, M. et al., "The formation of the Mg I emission features near 12 microns", *Astron. Astrophys.*, **253**, 567 (1992).

---

## Appendix A: Detector Technology Reference Table / 부록 A: 검출기 기술 참조 표

**English.** The infrared detector landscape for solar observations is rich and continuously evolving. This table summarizes the detectors mentioned by Penn (2014):

| Detector / 검출기 | Material / 소재 | Wavelength range / 파장 범위 | Notes / 비고 |
|---|---|---|---|
| Silicon CCD/CMOS | Si | 400–1100 nm | Cutoff at $\sim 1100$ nm (Si becomes transparent) / 1100 nm에서 Si 투명화로 감도 소멸 |
| PbS photoconductor | Lead sulfide | 1000–3500 nm | 1960s–1980s single-element detectors / 1960–80년대 단일 소자 |
| InGaAs | InGaAs | 1000–1800 nm | Near-IR arrays / 근적외 어레이 |
| InSb | Indium antimonide | 1000–5000 nm | Cryogenic (77 K); FTS era workhorse / 극저온(77 K); FTS 시대 주력 |
| HgCdTe | Mercury-cadmium-telluride | 1000–2200 nm (tunable) | FIRS, NAC arrays (1024×1024) / FIRS, NAC 어레이 |
| Si:X doped Si | Arsenic-doped Si BIB | 2000–30 000 nm | Celeste (128×128 Si:As) / Celeste |
| Ge:X doped Ge | Doped germanium | 28 000–200 000 nm | Far-IR / 원적외 |
| PtSi | Platinum silicide | 1000–5000 nm | Low dark current / 저 암전류 |
| Ge bolometer | Cryogenic Ge | Broadband thermal | Historical eclipse experiments / 과거 일식 실험 |
| QWIP | Quantum-well infrared photodetector | Tunable (8–9 μm typical) | Emerging tech at McM-P / McM-P에서 신기술 시험 |

**한국어.** IR 검출기는 태양 관측에서 계속 진화하고 있다. Penn(2014)에 언급된 검출기를 정리하면 위 표와 같다. Si CCD는 1100 nm에서 한계에 도달하므로, 그 이상의 파장을 보려면 HgCdTe·InSb·Si:X·Ge:X 등 고체 IR 검출기가 필수이다. 최신 QWIP 카메라는 바이어스 전압 조절로 파장 응답을 바꿀 수 있어 유연성이 크다.

## Appendix B: Key Solar IR Telescope Facilities / 부록 B: 주요 태양 IR 망원경 시설

**English.** The review cites numerous facilities; this summary highlights the most important:

- **McMath-Pierce Solar Facility (McM-P), Kitt Peak AZ, USA**: Historic all-reflecting 1.6-m main mirror; ideal for IR (no transmissive optics); home to NAC, Celeste, FTS, and Cyra instruments. Workhorse for Mg I 12 318 nm science (only facility capable until recently).
- **Dunn Solar Telescope (DST), Sunspot NM, USA**: 0.76-m telescope with state-of-the-art AO and FIRS spectropolarimeter (1024×1024 HgCdTe, multi-slit grating design).
- **New Solar Telescope (NST), Big Bear CA, USA**: 1.6-m telescope with planned Cyra cooled-grating IR spectrograph (3000–5000 nm).
- **GREGOR, Tenerife, Canary Islands**: 1.5-m telescope at Observatorio del Teide; GRIS spectrograph operating 1000–2300 nm (spectropolarimetry to 1800 nm). Added in the 16 June 2014 revision.
- **DKIST (Daniel K. Inouye Solar Telescope, formerly ATST), Haleakalā HI, USA**: 4-m all-reflecting; Cryo-NIRSP (cryogenic IR spectropolarimeter) and DL-NIRSP; designed for 0.08″ at 1565 nm.
- **COMP (Coronal Multichannel Polarimeter), Mauna Loa HI, USA**: 20-cm coronagraph measuring [Fe XIII] 1074.7 nm full-Stokes at 4.5″ pixels over 1.05–1.35 $R_\odot$.
- **COSMO (planned)**: Large coronagraph project building on COMP for synoptic [Fe XIII] observations.

**한국어.** 리뷰에서 언급된 주요 시설을 요약하면 다음과 같다:

- **McMath-Pierce 태양 시설(McM-P), 킷피크 애리조나**: 전반사 1.6-m 주경; 투과 광학 없어 IR에 이상적; NAC, Celeste, FTS, Cyra 기기의 본거지. Mg I 12 318 nm 과학의 주력 시설.
- **Dunn 태양망원경(DST), 선스팟 뉴멕시코**: 0.76-m, 최첨단 AO와 FIRS 분광편광기(1024×1024 HgCdTe, 다중슬릿 회절격자).
- **New Solar Telescope(NST), 빅베어 캘리포니아**: 1.6-m, Cyra(3000–5000 nm 냉각 회절격자 분광기) 장착 예정.
- **GREGOR, 테네리페 카나리아 제도**: 1.5-m; GRIS 분광기(1000–2300 nm). 2014년 6월 개정에서 추가됨.
- **DKIST(구 ATST), 할레아칼라 하와이**: 4-m 전반사; Cryo-NIRSP과 DL-NIRSP; 1565 nm에서 0.08″ 설계.
- **COMP, 마우나로아 하와이**: 20-cm 코로나그래프, [Fe XIII] 1074.7 nm 풀-Stokes를 4.5″ 픽셀로 측정.
- **COSMO(계획)**: COMP를 계승한 대형 코로나그래프.

## Appendix C: Numerical Comparison — Magnetic Resolution by Line / 부록 C: 선별 자기 분해능 수치 비교

**English.** To make the IR advantage concrete, here is a side-by-side worked comparison at $B = 1000$ G for selected lines:

| Line | $\lambda$ (nm) | $g_{\rm eff}$ | Zeeman $\Delta\lambda$ (mÅ) | Doppler $\sim$ (mÅ) | Resolution ratio |
|---|---|---|---|---|---|
| Fe I 5250.22 Å | 525 | 3.0 | 38.6 | 25 | 1.54 (weak-field) |
| Fe I 6302 Å | 630 | 2.5 | 46.3 | 30 | 1.54 |
| Fe I 1564.85 nm | 1565 | 3.0 | 343 | 75 | 4.6 (**fully split**) |
| Ti I 2231 nm | 2231 | 2.5 | 581 | 105 | 5.5 |
| Mg I 12 318 nm | 12 318 | 1.0 | 7089 | 580 | 12.2 |
| [Fe XIII] 1074.7 nm | 1075 | 1.5 | 81 | (thermal broad ~800) | 0.10 (line broad, still useful) |

**한국어.** IR의 이점을 구체화하기 위해 $B = 1000$ G에서 여러 선의 수치를 비교하면 위와 같다. Fe I 1565 nm는 저장 도플러 폭의 약 4.6배로 완전 분리되며, Mg I 12 318 nm는 12배 이상으로 극적 분리를 보인다. [Fe XIII] 1075 nm는 선폭이 매우 넓어(코로나 고온 열 폭) 비율은 작으나 편광 측정으로 자기장을 추출할 수 있다.
