---
title: "Reading Notes — Small-Scale Solar Magnetic Fields (de Wijn et al. 2009)"
date: 2026-04-27
completed_date: 2026-04-27
topic: Solar_Physics
paper_number: 30
authors: "A.G. de Wijn, J.O. Stenflo, S.K. Solanki, S. Tsuneta"
year: 2009
journal: "Space Science Reviews 144, 275-315"
doi: "10.1007/s11214-008-9473-6"
status: completed
tags: [quiet_sun, internetwork, small_scale_magnetism, hanle_effect, zeeman_effect, magnetic_carpet, hinode, polar_field, convective_collapse, local_dynamo]
---

# Reading Notes — Small-Scale Solar Magnetic Fields
# 읽기 노트 — 소규모 태양 자기장

**Citation.** A.G. de Wijn, J.O. Stenflo, S.K. Solanki, S. Tsuneta, *Space Sci. Rev.* **144**, 275–315 (2009). DOI: 10.1007/s11214-008-9473-6.

---

## 1. Core Contribution / 핵심 기여

**English.** This review consolidates the post-Hinode picture of the quiet-Sun photospheric magnetic field at the smallest observable scales — the *internetwork* regime that fills supergranular cell interiors. Combining Zeeman, Hanle, and infrared/visible diagnostics with Hinode SOT-SP and 3D MHD simulations (Vögler, Schüssler), the authors argue that the quiet Sun is not magnetically empty but threaded by a tangled "magnetic carpet" of mixed polarities. Internetwork field carries $\sim 4$ orders of magnitude more emerging flux per unit area than active regions and is dominated by *horizontal* (not vertical) field, with $\langle |B_h| \rangle / \langle |B_v| \rangle \sim 5$ from Lites et al. (2008). Average flux densities in network and internetwork are $\sim 11$ G and $\sim 1$–$10$ G respectively, but Hanle measurements imply substantial unresolved turbulent flux ($\sim 60$–$100$ G *intrinsic* equivalent isotropic strength). The emerging consensus invokes a local turbulent dynamo, with implications for chromospheric and coronal heating.

**한국어.** 본 리뷰는 가장 작은 관측 가능 스케일 — 초입자 셀 내부를 채우는 *네트워크간(internetwork)* 영역 — 에서의 조용한 태양 광구 자기장에 대한 Hinode 이후의 그림을 종합합니다. Zeeman, Hanle, 적외선/가시광 진단을 Hinode SOT-SP 및 3D MHD 시뮬레이션(Vögler, Schüssler)과 결합하여, 저자들은 조용한 태양이 자기적으로 비어있지 않고 혼합 극성의 얽힌 "자기 카펫"으로 짜여 있다고 주장합니다. 네트워크간 자기장은 활동 영역보다 단위 면적당 약 4차수 더 많은 emerging flux를 운반하며, *수평* 자기장이 *수직* 자기장보다 우세 — Lites et al.(2008)에서 $\langle |B_h| \rangle / \langle |B_v| \rangle \sim 5$. 네트워크와 네트워크간의 평균 자속 밀도는 각각 $\sim 11$ G와 $\sim 1$–$10$ G이지만, Hanle 측정은 상당한 미분해 난류 자속(등방성 강도 환산 $\sim 60$–$100$ G)이 존재함을 시사합니다. 떠오르는 합의는 국소 난류 다이나모를 호출하며, 채층·코로나 가열에 대한 함의를 갖습니다.

---

## 2. Reading Notes / 읽기 노트

### 2.1 Section 1 — Introduction (pp. 275–276) / 1장 — 서론

**English.** The Sun is magnetic on every scale, from active regions ($\sim 10^4$ km) down to the diffraction limit of the best telescopes ($\sim 100$ km). Granular convection sweeps flux into intergranular lanes producing kilogauss flux concentrations (bright points, faculae). In active regions these cluster as plages; in the quiet Sun supergranular flows concentrate them into the *magnetic network*. *Internetwork* covers most of the solar surface, contains four orders of magnitude more emerging flux per area than active regions, and was historically ignored due to lack of sensitivity. New instruments (Hinode SOT-SP since 2006), seeing-mitigation (adaptive optics, image reconstruction), and 3D MHD simulations have made systematic study possible.

**한국어.** 태양은 모든 스케일에서 자기적입니다 — 활동 영역($\sim 10^4$ km)부터 최고 망원경의 회절 한계($\sim 100$ km)까지. 입자 대류는 자속을 입자간 lane으로 휩쓸어 kilogauss 자속 집중(bright point, facula)을 만듭니다. 활동 영역에서는 plage로 모이고, 조용한 태양에서는 초입자 흐름이 이를 *자기 네트워크*로 모읍니다. *네트워크간*은 태양 표면 대부분을 덮고, 단위 면적당 활동 영역보다 4차수 많은 emerging flux를 가지며, 감도 부족으로 역사적으로 무시되었습니다. 새 장비(2006년 이후 Hinode SOT-SP), 시상 완화(적응광학, 이미지 재구성), 3D MHD 시뮬레이션이 체계적 연구를 가능하게 했습니다.

### 2.2 Section 2.1 — Magnetic Flux in the Quiet Sun (pp. 277–280) / 2.1장 — 조용한 태양의 자속

**Methods (2.1.1).**

**English.** Three diagnostic families compete:
- **Brightness proxies** (G band, Ca II H, K core, CN bandhead) — high spatial resolution, low sensitivity, biased toward kG fields.
- **Zeeman effect** — quantitative on $B_\parallel$ via Stokes V; can detect fluxes $\sim 10^{16}$ Mx but cancels for mixed polarities within the resolution element.
- **Hanle effect** — modifies resonance scattering polarisation; sensitive to weak ($\sim 1$–$100$ G) *isotropic* fields invisible to Zeeman.

**한국어.** 세 가지 진단군이 경쟁합니다:
- **밝기 대리 진단**(G 밴드, Ca II H, K 코어, CN bandhead) — 고공간 분해능, 저감도, kG 영역에 편향.
- **Zeeman 효과** — Stokes $V$로 $B_\parallel$를 정량화; $\sim 10^{16}$ Mx 자속까지 검출하지만 분해 요소 내 반대 극성이 상쇄.
- **Hanle 효과** — 공명 산란 편광을 변조; Zeeman에 보이지 않는 약한($\sim 1$–$100$ G) *등방성* 자기장에 민감.

**Measurements (2.1.2).**

**English.** Network elements show *exponential* flux distribution (Schrijver et al. 1997; Hagenaar 2001) with sensitivity floor $2\times 10^{18}$ Mx, while Wang et al. (1995) reported a non-power-law for both network and internetwork down to $\sim 10^{16}$ Mx. Zirin (1987): internetwork flux emergence rate is $\sim 100\times$ larger than ephemeral regions, which is itself $\sim 100\times$ that of normal active regions; thus internetwork dominates the *emergence* budget but its decay is also rapid. Pre-Hinode Zeeman measurements gave $\langle B \rangle \sim 2$–$5$ G; Domínguez Cerdeña (2003a,b) and Khomenko (2005a) found $\sim 20$ G at $0.5''$ resolution. Hinode/SP at $0.32''$ gave $\langle B \rangle = 11$ G (Lites et al. 2008), about half of Domínguez Cerdeña's value, attributed to seeing degradation in earlier ground-based work. Hanle constraints (Stenflo 1982; Trujillo Bueno et al. 2004; Bommier 2005) yield $10$–$60$ G average and exponential PDF with e-folding $130$ G — implying *more* than half the surface carries fields $> 60$ G even when the mean is $\sim 60$ G.

**한국어.** 네트워크 요소는 감도 한계 $2\times 10^{18}$ Mx까지 *지수* 분포를 보입니다(Schrijver 1997; Hagenaar 2001). Wang et al.(1995)은 네트워크와 네트워크간 모두에서 $\sim 10^{16}$ Mx까지 비-멱법칙을 보고. Zirin(1987): 네트워크간 자속 emergence 비율은 단명 영역(ephemeral region)보다 $\sim 100\times$, 단명 영역은 정상 활동 영역보다 $\sim 100\times$; 따라서 네트워크간이 *emergence* 예산을 지배하나 소멸도 빠릅니다. Hinode 이전 Zeeman 측정은 $\langle B \rangle \sim 2$–$5$ G; Domínguez Cerdeña(2003a,b), Khomenko(2005a)는 $0.5''$ 분해능에서 $\sim 20$ G. Hinode/SP $0.32''$에서 $\langle B \rangle = 11$ G(Lites et al. 2008), 이전 지상 관측의 절반 — 시상 저하 효과로 해석. Hanle 제약(Stenflo 1982; Trujillo Bueno 2004; Bommier 2005)은 $10$–$60$ G 평균과 e-folding $130$ G 지수 PDF — 평균이 $\sim 60$ G일 때조차 표면 절반 이상이 $> 60$ G 자기장을 가짐을 시사.

### 2.3 Section 2.2 — Field Strength (pp. 280–283) / 2.2장 — 자기장 세기

**English.** Network is intrinsically *kilogauss* (Stenflo 1973; Solanki & Stenflo 1984), confirmed by infrared 1.56 μm Fe I observations probing the deep photosphere. Internetwork is more controversial:
- **Infrared 1.56 μm** Fe I lines (Lin 1995; Solanki et al. 1996; Khomenko 2003, 2005b; Martínez González 2007) → $\langle B \rangle \lesssim 600$ G, *weak fields*.
- **Visible 630 nm** Fe I lines (Sánchez Almeida; Socas-Navarro 2003; Lites & Socas-Navarro 2004) → mixed strong+weak, with strong fields dominating the inversions.

The infrared/visible discrepancy reflects different sensitivity: the 1.56 μm pair is in the complete-Zeeman-splitting regime (signal $\propto$ flux, so weak fields are visible); the 630 nm pair is in the incomplete regime (signal $\propto B^2$, so strong fields dominate). MISMA inversions (Sánchez Almeida) bridge them by allowing both to coexist on micro-scales. Domínguez Cerdeña et al. (2006a) using both wavelengths simultaneously find a positive $B$–flux correlation: stronger fields in higher-flux pixels, consistent with convective collapse.

**한국어.** 네트워크는 본질적으로 *kilogauss*(Stenflo 1973; Solanki & Stenflo 1984), 적외선 1.56 μm Fe I 관측으로 확인 — 깊은 광구를 탐사. 네트워크간은 더 논쟁적:
- **적외선 1.56 μm** Fe I 선(Lin 1995; Solanki 1996; Khomenko 2003, 2005b; Martínez González 2007) → $\langle B \rangle \lesssim 600$ G, *약자기장*.
- **가시광 630 nm** Fe I 선(Sánchez Almeida; Socas-Navarro 2003; Lites & Socas-Navarro 2004) → 강+약 혼합, 인버전에서 강자기장 우세.

적외선/가시광 불일치는 감도 차이를 반영: 1.56 μm 쌍은 완전 Zeeman 분리 영역(신호 ∝ 자속, 약자기장 보임); 630 nm 쌍은 불완전 영역(신호 ∝ $B^2$, 강자기장 우세). MISMA 인버전은 양쪽을 미시스케일에서 공존시켜 연결합니다. Domínguez Cerdeña et al.(2006a)는 두 파장 동시 사용으로 양의 $B$-자속 상관관계를 발견 — 고자속 픽셀에서 강자기장, 대류 붕괴와 일치.

### 2.4 Section 2.3 — Horizontal Fields (p. 284) / 2.3장 — 수평 자기장

**English.** Martin (1988) first noted internetwork features visible from disk centre to limb, suggesting both vertical and horizontal components. Lites et al. (1996) found arcsec-scale, minute-lifetime horizontal fields. Center-to-limb variation of Stokes V (Meunier 1998; Martínez González 2008) and Hanle SOLIS data (Harvey 2007) point to a "seething" horizontal field of $1$–$2$ G on $2.5$–$5''$ scales, varying within minutes. Hinode (Orozco Suárez 2007a,b) finds a peak at $90°$ inclination — i.e., predominantly horizontal. **Lites et al. (2008): horizontal flux is $5\times$ vertical flux**, with $\langle B_h \rangle \approx 50$–$60$ G — comparable to Hanle estimates. Martínez González et al. (2007) reconstructed small Ω-loops emerging in granules carrying $\sim 10^{17}$ Mx each.

**한국어.** Martin(1988)은 디스크 중심부터 가장자리까지 보이는 네트워크간 feature를 처음 보고 — 수직과 수평 성분 모두 시사. Lites et al.(1996)은 arcsec 규모, 분 단위 수명의 수평 자기장 발견. Stokes $V$ 중심-가장자리 변화(Meunier 1998; Martínez González 2008)와 Hanle SOLIS 데이터(Harvey 2007)는 $2.5$–$5''$ 스케일에서 $1$–$2$ G "seething" 수평 자기장이 분 단위로 변화함을 지시. Hinode(Orozco Suárez 2007a,b)는 경사각 $90°$의 피크 — 즉 주로 수평. **Lites et al.(2008): 수평 자속이 수직의 $5\times$**, $\langle B_h \rangle \approx 50$–$60$ G — Hanle 추정값과 비슷. Martínez González et al.(2007)은 입자에서 emerge하는 $\sim 10^{17}$ Mx의 작은 Ω-루프를 재구성.

### 2.5 Section 2.4 — Source of Internetwork Fields (p. 284) / 2.4장 — 네트워크간 자기장의 기원

**English.** Three (non-exclusive) hypotheses:
1. *Deep dynamo emergence* — flux generated below the surface (e.g., at the tachocline) rises and is shredded into small bipoles.
2. *Recycled active-region flux* — old flux ground down by convection refills the quiet Sun.
3. *Local turbulent dynamo* — granular and supergranular convective turbulence amplifies a tangled small-scale field. Vögler & Schüssler (2007) MHD simulations demonstrate this works in surface conditions.

The horizontal-dominant geometry (Lites 2008) and the high turnover rate (carpet replacement timescale) most naturally fit a local dynamo, but recycling cannot be excluded.

**한국어.** 세 가지(상호 배타적이지 않은) 가설:
1. *심층 다이나모 emergence* — 표면 아래(예: 타코클라인)에서 생성된 자속이 떠올라 작은 양극성으로 잘게 쪼개짐.
2. *재순환 활동 영역 자속* — 대류로 분쇄된 오래된 자속이 조용한 태양을 다시 채움.
3. *국소 난류 다이나모* — 입자·초입자 대류 난류가 얽힌 소규모 자기장을 증폭. Vögler & Schüssler(2007) MHD 시뮬레이션에서 표면 조건에서 작동 입증.

수평 우세 기하(Lites 2008)와 빠른 turnover(카펫 교체 시간 척도)는 국소 다이나모와 가장 자연스럽게 부합하지만, 재순환을 배제할 수 없습니다.

### 2.6 Section 3 — Horizontal Fields (continued) / 3장 — 수평 자기장(계속)

**English.** Detailed Hinode statistics confirm internetwork is dominated by short-lived ($\lesssim 10$ min) horizontal patches. The flux probability density function in the quiet Sun shows little change with centre-to-limb distance, supporting near-isotropic geometry. Combined with Hanle constraints, the unresolved-flux budget remains substantial: even after the Hinode revolution, $\gtrsim 50\%$ of quiet-Sun flux may be hidden below the spatial resolution (i.e., truly turbulent).

**한국어.** Hinode 통계는 네트워크간이 단명($\lesssim 10$ 분) 수평 패치로 지배됨을 확인. 조용한 태양의 자속 PDF는 디스크 중심-가장자리 거리에 따른 변화가 거의 없어 거의 등방성 기하를 지지. Hanle 제약과 결합하면 미분해 자속 예산은 여전히 상당: Hinode 혁명 이후에도 조용한 태양 자속의 $\gtrsim 50\%$가 공간 분해능 아래(진정한 난류)에 숨어있을 수 있음.

### 2.7 Section 4 — Polar Field / 4장 — 극자기장

**English.** Hinode resolved kilogauss patches in the polar caps with sizes of arcseconds. They are unipolar (matching the global 11-year cycle polarity), spatially compact, and contain a substantial fraction of the open flux that becomes the heliospheric magnetic field. Their existence as discrete strong-field elements at high latitudes constrains dynamo and flux-transport models.

**한국어.** Hinode는 극관에서 arcsec 크기의 kilogauss 패치를 분해했습니다. 단극성(11년 주기 극성과 일치), 공간적으로 조밀하며, 태양권 자기장이 되는 개방 자속의 상당 부분을 포함합니다. 고위도에서 강자기장 요소의 이산적 존재는 다이나모 및 자속 수송 모델을 제약합니다.

### 2.8 Section 5 — Strong Vertical Field Concentrations / 5장 — 강한 수직 자기장 집중

**English.** Convective collapse (Spruit 1979; Parker 1978): a flux tube with $B \sim 500$ G is unstable to a downflow that evacuates the tube, increasing $B$ to equipartition with the gas pressure ($B^2/8\pi \sim p$), giving $\sim 1.5$ kG. Bright points and faculae are observational signatures: G-band brightenings, $0.1''$ structures with kG fields. Domínguez Cerdeña 2006a's $B$-flux relation is consistent with this mechanism: low-flux features are weak (collapse incomplete or unstable), high-flux features are kG.

**한국어.** 대류 붕괴(Spruit 1979; Parker 1978): $B \sim 500$ G의 자속관은 하강류에 불안정 — 가스를 배출하여 $B$를 가스압과의 등분배($B^2/8\pi \sim p$)까지 증가, $\sim 1.5$ kG. Bright point와 facula가 관측 시그니처: G 밴드 밝기, $0.1''$ 구조의 kG 자기장. Domínguez Cerdeña 2006a의 $B$-자속 관계가 이 메커니즘과 일치: 저자속 feature는 약함(붕괴 불완전·불안정), 고자속은 kG.

### 2.9 Section 6 — Discussion: Unresolved Fields / 6장 — 토론: 미분해 자기장

**English.** The "unresolved-flux puzzle" remains. Estimates by combining Zeeman flux maps (resolved) and Hanle isotropic-equivalent strength (unresolved) suggest the *true* mean flux density of the quiet Sun is $\sim 100$ G, of which only $\sim 10$–$20$ G is resolved by Hinode at $0.32''$. This unresolved energy budget is large enough — when dissipated by reconnection — to power chromospheric heating and possibly contribute to coronal heating in quiet regions.

**한국어.** "미분해 자속 퍼즐"이 남아있습니다. Zeeman 자속 맵(분해됨)과 Hanle 등방성 환산 강도(미분해)의 결합 추정은 조용한 태양의 *진짜* 평균 자속 밀도가 $\sim 100$ G이며, Hinode $0.32''$에서 분해되는 것은 $\sim 10$–$20$ G에 불과함을 시사. 이 미분해 에너지 예산은 — reconnection으로 소산될 때 — 채층 가열을 구동하고 조용한 영역의 코로나 가열에 기여하기에 충분합니다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Quiet Sun is magnetically alive / 조용한 태양은 자기적으로 살아있다.**
   *EN.* The quiet Sun is far from a "field-free" region: it carries 4 orders of magnitude more emerging flux per unit area than active regions and hosts a continuously renewed magnetic carpet.
   *KR.* 조용한 태양은 "자기장 없는" 영역과는 거리가 멉니다: 활동 영역보다 단위 면적당 4차수 많은 emerging flux를 운반하고 끊임없이 갱신되는 자기 카펫을 호스트합니다.

2. **Horizontal field dominates internetwork / 네트워크간은 수평 자기장이 우세.**
   *EN.* Hinode/SP measures $\langle |B_h| \rangle / \langle |B_v| \rangle \approx 5$ — a factor that *reverses* the classical vertical-flux-tube paradigm at small scales.
   *KR.* Hinode/SP는 $\langle |B_h| \rangle / \langle |B_v| \rangle \approx 5$를 측정 — 작은 스케일에서 고전적 수직 자속관 패러다임을 *역전*시키는 비율.

3. **Mean flux density depends on technique / 평균 자속 밀도는 기법에 의존한다.**
   *EN.* Zeeman→$\sim 11$ G, infrared Zeeman→$\langle B \rangle$ hectogauss with low filling, Hanle→$60$–$100$ G isotropic-equivalent. They probe different aspects of the same tangled field.
   *KR.* Zeeman→$\sim 11$ G, 적외선 Zeeman→낮은 filling factor의 hectogauss $\langle B \rangle$, Hanle→$60$–$100$ G 등방성 환산. 동일한 얽힌 자기장의 서로 다른 측면을 탐사.

4. **Network flux follows exponential PDF, internetwork is debated / 네트워크 자속은 지수 PDF, 네트워크간은 논쟁 중.**
   *EN.* Network elements: $N(\Phi) \propto \exp(-\Phi/\Phi_0)$ with cutoff $\sim 2\times 10^{18}$ Mx (Hagenaar 2001). Internetwork: log-normal or non-power-law down to $10^{16}$ Mx (Wang 1995).
   *KR.* 네트워크 요소: $N(\Phi) \propto \exp(-\Phi/\Phi_0)$, 컷오프 $\sim 2\times 10^{18}$ Mx(Hagenaar 2001). 네트워크간: $10^{16}$ Mx까지 로그정규 또는 비-멱법칙(Wang 1995).

5. **IR vs. visible discrepancy is a signal-to-noise effect / 적외선 vs. 가시광 불일치는 신호 대 잡음 효과.**
   *EN.* The 1.56 μm Fe I pair (complete Zeeman splitting) sees weak fields well; 630 nm (incomplete splitting) is biased toward strong fields because the signal scales as $B^2$.
   *KR.* 1.56 μm Fe I 쌍(완전 Zeeman 분리)은 약자기장을 잘 봅니다; 630 nm(불완전 분리)는 신호가 $B^2$로 스케일하므로 강자기장에 편향.

6. **Convective collapse explains kG concentrations / 대류 붕괴가 kG 집중을 설명한다.**
   *EN.* Equipartition $B^2/8\pi \sim p$ with photospheric $p \sim 10^5$ dyn/cm² gives $B \sim 1.5$ kG, matching observed bright-point fields. The $B$-flux correlation supports this.
   *KR.* 광구압 $p \sim 10^5$ dyn/cm²와의 등분배 $B^2/8\pi \sim p$는 $B \sim 1.5$ kG 산출 — 관측된 bright point 자기장과 일치. $B$-자속 상관관계가 이를 지지.

7. **Local turbulent dynamo is plausible / 국소 난류 다이나모가 그럴듯하다.**
   *EN.* Vögler & Schüssler (2007) MHD simulations show granular convection alone can amplify a seed field to observed levels, naturally producing horizontal-dominated mixed-polarity geometry.
   *KR.* Vögler & Schüssler(2007) MHD 시뮬레이션은 입자 대류만으로도 시드 자기장을 관측 수준까지 증폭 가능함을 보여 — 수평 우세 혼합 극성 기하를 자연스럽게 생성.

8. **Implications for chromospheric/coronal heating / 채층·코로나 가열 함의.**
   *EN.* The unresolved-flux energy budget, dissipated via small-scale reconnection, is comparable to chromospheric radiative losses ($\sim 10^7$ erg/cm²/s) — making the magnetic carpet a candidate heating agent for quiet regions.
   *KR.* 미분해 자속 에너지 예산이 소규모 reconnection으로 소산되면 채층 복사 손실($\sim 10^7$ erg/cm²/s)과 비슷 — 자기 카펫을 조용한 영역의 가열 후보로 만듭니다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Zeeman Effect / Zeeman 효과

**English.** A spectral line at wavelength $\lambda$ in a field $B$ splits into $\sigma$ and $\pi$ components separated by:
**한국어.** 자기장 $B$ 안의 파장 $\lambda$ 분광선은 다음만큼 분리되는 $\sigma$와 $\pi$ 성분으로 갈라집니다:

$$
\Delta \lambda_B = \frac{e}{4 \pi m_e c}\, g\, \lambda^2\, B
\;\approx\; 4.67\times 10^{-13}\, g\, \lambda^2[\text{Å}]\, B[\text{G}]\;\;[\text{Å}]
$$

- $e, m_e, c$ — electron charge/mass/light speed (전자 전하/질량/광속).
- $g$ — Landé factor (Landé 인자).
- $\lambda$ — line wavelength (선의 파장).
- $B$ — magnetic field strength (자기장 세기).

Stokes $V$ amplitude in the *weak-field* limit: $V \propto g \lambda^2 \frac{\partial I}{\partial \lambda} B_\parallel$. Linear polarisation: $Q, U \propto B_\perp^2$.
약자기장 한계의 Stokes $V$ 진폭: $V \propto g \lambda^2 \frac{\partial I}{\partial \lambda} B_\parallel$. 선편광: $Q, U \propto B_\perp^2$.

### 4.2 Convective Collapse Equipartition / 대류 붕괴 등분배

$$
\frac{B_{\text{eq}}^2}{8\pi} = p_{\text{gas}}
\;\;\Rightarrow\;\; B_{\text{eq}} = \sqrt{8\pi p_{\text{gas}}}
$$

**English.** With photospheric $p_{\text{gas}} \approx 1.2 \times 10^5$ dyn/cm² (at $\tau_{500}=1$): $B_{\text{eq}} \approx 1.74$ kG, matching observed bright-point fields. The exact value drops with height; at the temperature minimum it is a few hundred gauss.
**한국어.** 광구 $p_{\text{gas}} \approx 1.2 \times 10^5$ dyn/cm²($\tau_{500}=1$): $B_{\text{eq}} \approx 1.74$ kG — 관측된 bright point 자기장과 일치. 정확한 값은 고도에 따라 감소 — 온도 최저점에서는 수백 가우스.

### 4.3 Magnetic Flux and Filling Factor / 자속과 filling factor

$$
\Phi = f \cdot B \cdot A
$$

- $\Phi$ — total flux in pixel (Mx) (픽셀 내 총 자속).
- $f$ — filling factor (0–1) (충전인자).
- $B$ — intrinsic field strength (G) (본질 자기장 세기).
- $A$ — pixel area (cm²) (픽셀 면적).

**English.** A $0.32''$ Hinode pixel at disk centre $\approx (230\text{ km})^2 \approx 5.3\times 10^{14}$ cm². With $\Phi = 10^{17}$ Mx and $B = 1$ kG: $f \approx 0.19$.
**한국어.** 디스크 중심에서 Hinode $0.32''$ 픽셀 $\approx (230\text{ km})^2 \approx 5.3\times 10^{14}$ cm². $\Phi = 10^{17}$ Mx, $B = 1$ kG일 때: $f \approx 0.19$.

### 4.4 Flux Distributions / 자속 분포

Network exponential (Hagenaar 2001):
$$
N(\Phi)\, d\Phi = N_0\, \exp\!\left(-\Phi / \Phi_0\right)\, d\Phi/\Phi_0
$$

Trujillo Bueno et al. (2004) PDF for internetwork field strength:
$$
\mathrm{PDF}(B) = \frac{1}{B_e}\, \exp(-B/B_e),\;\; B_e \approx 130\,\text{G}
$$

Lognormal flux distribution (often used for combined network+internetwork):
$$
p(\Phi) = \frac{1}{\Phi \sigma\sqrt{2\pi}} \exp\!\left[-\frac{(\ln\Phi - \mu)^2}{2\sigma^2}\right]
$$

with $\langle \ln \Phi \rangle = \mu$, $\mathrm{Var}(\ln\Phi) = \sigma^2$.

### 4.5 Magnetic Carpet Replacement Timescale / 자기 카펫 교체 시간 척도

**English.** Defined by total photospheric flux divided by emergence rate:
**한국어.** 전체 광구 자속을 emergence 비율로 나눈 값:

$$
\tau_{\text{rep}} = \frac{\Phi_{\text{tot}}}{\dot{\Phi}_{\text{emrg}}}
$$

Order-of-magnitude estimate: with $\langle B \rangle = 11$ G over $A_\odot = 6 \times 10^{22}$ cm² → $\Phi_{\text{tot}} \approx 6.6 \times 10^{23}$ Mx. Internetwork emergence rate $\sim 100$ Mx/cm²/day = $6 \times 10^{24}$ Mx/day for the whole Sun → $\tau_{\text{rep}} \approx 0.1$ day $\sim 2$–$3$ hours. (Hagenaar 2001 finds $\sim 14$ hours for ephemeral regions plus internetwork.)
크기 차수 추정: $\langle B \rangle = 11$ G가 $A_\odot = 6 \times 10^{22}$ cm² 위에서 → $\Phi_{\text{tot}} \approx 6.6 \times 10^{23}$ Mx. 네트워크간 emergence 비율 $\sim 100$ Mx/cm²/일 = 전체 태양에서 $6 \times 10^{24}$ Mx/일 → $\tau_{\text{rep}} \approx 0.1$ 일 $\sim 2$–$3$ 시간. (Hagenaar 2001은 단명 영역+네트워크간에 대해 $\sim 14$ 시간 산출.)

### 4.6 Hanle Effect Sensitivity / Hanle 효과 감도

**English.** The Hanle critical field for a transition is:
**한국어.** 천이의 Hanle 임계 자기장:

$$
B_H = \frac{1.137\times 10^{-7}}{t_{\text{life}} g_J}\;\;[\text{G}]
$$

with $t_{\text{life}}$ the upper-level lifetime (s) and $g_J$ the Landé factor of the upper level. For Sr I 4607 Å, $t_{\text{life}} \approx 6\,\text{ns}$, $g_J = 1$ → $B_H \approx 19$ G — exactly the regime probed by Trujillo Bueno's depolarisation measurements.
$t_{\text{life}}$는 상위 준위 수명(초), $g_J$는 상위 준위 Landé 인자. Sr I 4607 Å의 경우 $t_{\text{life}} \approx 6\,\text{ns}$, $g_J = 1$ → $B_H \approx 19$ G — Trujillo Bueno의 탈편광 측정이 탐사하는 정확한 영역.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1908 ───────── Hale: sunspot Zeeman polarisation
                       │
1959 ───────── Babcock: solar dynamo magnetograph
                       │
1971 ───────── Livingston & Harvey: internetwork discovery
                       │
1973 ───────── Stenflo: kilogauss network flux tubes (infrared Zeeman)
                       │
1979 ───────── Spruit: convective collapse
                       │
1987 ───────── Stenflo: Hanle effect on quiet Sun (limit < 100 G)
                       │
1995 ───────── Lin: 1.56 μm internetwork (hectogauss)
                       │
2003 ───────── Trujillo Bueno: Hanle ⇒ unresolved 130 G fields
                       │
2006 ───────── Hinode launch (SOT-SP at 0.32")
                       │
2008 ───────── Lites et al.: horizontal:vertical = 5
                       │
★ 2009 ★ ───── de Wijn, Stenflo, Solanki, Tsuneta REVIEW (this paper)
                       │
2010s ───────── SUNRISE, GREGOR, DKIST: sub-100 km photospheric resolution
```

**English.** This review crystallised the post-Hinode picture: the quiet Sun as a horizontally-dominated magnetic carpet probably maintained by a local turbulent dynamo. It became the standard reference for the next decade of work on chromospheric heating and the photospheric dynamo.

**한국어.** 본 리뷰는 Hinode 이후의 그림을 구체화: 국소 난류 다이나모로 유지되는 수평 우세 자기 카펫으로서의 조용한 태양. 이후 10년간 채층 가열과 광구 다이나모 연구의 표준 참고문헌이 되었습니다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Connection (EN) | 연결 (KR) |
|-------|-----------------|-----------|
| Babcock 1961 (#5) | Provides the global dynamo backdrop; this paper challenges it with a *local* surface dynamo. | 전역 다이나모 배경 제공; 본 논문은 *국소* 표면 다이나모로 도전. |
| Spruit 1981 (#10 — flux tubes) | Convective collapse mechanism is invoked here for kG concentrations. | 대류 붕괴 메커니즘을 kG 집중에 호출. |
| Title & Schrijver 1998 (magnetic carpet) | Coined the magnetic carpet concept; this review extends to internetwork scale. | 자기 카펫 개념 창안; 본 리뷰는 네트워크간 스케일로 확장. |
| Trujillo Bueno et al. 2004 | Hanle constraint on $130$ G PDF; fundamental input to this review's unresolved-flux budget. | $130$ G PDF의 Hanle 제약; 미분해 자속 예산의 기본 입력. |
| Lites et al. 2008 | Headline horizontal:vertical = 5 result; central data point of Section 3. | 핵심 수평:수직 = 5 결과; 3장의 중심 데이터. |
| Vögler & Schüssler 2007 | MHD simulation showing local dynamo viability at surface. | 표면에서 국소 다이나모 작동성을 보인 MHD 시뮬레이션. |
| Domínguez Cerdeña et al. 2003a, 2006a | Pre-Hinode quiet-Sun Zeeman analyses; revised down by Hinode. | Hinode 이전 조용한 태양 Zeeman 분석; Hinode로 하향 수정. |
| Aschwanden 2005 (coronal heating) | Magnetic carpet provides candidate energy reservoir for quiet-Sun heating. | 자기 카펫이 조용한 태양 가열의 에너지 저장소 후보 제공. |

---

## 6.5 Worked Numerical Examples / 수치 워크드 예제

### Example A — Hinode pixel flux budget / Hinode 픽셀 자속 예산

**English.** Take a single Hinode SOT-SP pixel of $0.16'' \times 0.16''$. At disk centre 1 arcsecond corresponds to $\sim 725$ km, so the pixel is approximately $116$ km on a side, giving area $A_{\text{pix}} = (1.16\times 10^7\,\text{cm})^2 \approx 1.35\times 10^{14}$ cm². If the average internetwork field $\langle B \rangle = 11$ G fills the pixel uniformly, the *apparent* flux is $\Phi_{\text{app}} = \langle B \rangle \cdot A_{\text{pix}} \approx 1.5 \times 10^{15}$ Mx — well below the typical $10^{16}$ Mx detection floor of Stokes V analyses, which is why averaging or stacking is required.

**한국어.** Hinode SOT-SP 픽셀 $0.16'' \times 0.16''$ 하나를 가정합니다. 디스크 중심에서 1 arcsec는 $\sim 725$ km, 픽셀 한 변은 약 $116$ km, 면적 $A_{\text{pix}} = (1.16\times 10^7\,\text{cm})^2 \approx 1.35\times 10^{14}$ cm². 픽셀에 평균 네트워크간 자기장 $\langle B \rangle = 11$ G가 균일하게 차면 *겉보기* 자속은 $\Phi_{\text{app}} = \langle B \rangle \cdot A_{\text{pix}} \approx 1.5 \times 10^{15}$ Mx — Stokes V 분석의 전형적 검출 한계 $10^{16}$ Mx보다 낮으므로 평균화·중첩이 필요합니다.

### Example B — Convective collapse equipartition trace / 대류 붕괴 등분배 추적

**English.** A flux tube starts with $B_0 = 500$ G and area $A_0$ at $\tau_{500}=1$ where $p_0 = 1.2\times 10^5$ dyn/cm². A downflow evacuates the tube, dropping internal density and pressure. Pressure balance with the external medium requires the *external* gas to push the tube to a new equilibrium $B_1$ such that $B_1^2/(8\pi) = p_0$ (assuming negligible internal gas pressure). Solve: $B_1 = \sqrt{8\pi \cdot 1.2\times 10^5} = \sqrt{3.02\times 10^6} \approx 1738$ G. Flux conservation $B_0 A_0 = B_1 A_1$ gives $A_1/A_0 = B_0/B_1 \approx 0.288$ — the tube cross-section *shrinks* by a factor 3.5. This matches the observed bright-point sizes ($\sim 0.1''$ in plages) being smaller than their proxy areas in MHD simulations.

**한국어.** 자속관이 $\tau_{500}=1$에서 $B_0 = 500$ G, 면적 $A_0$로 시작 — $p_0 = 1.2\times 10^5$ dyn/cm². 하강류가 자속관을 비우고 내부 밀도·압력 감소. 외부와의 압력 평형은 *외부* 가스가 자속관을 새 평형 $B_1$($B_1^2/(8\pi) = p_0$, 내부 가스압 무시)으로 밀어줌을 요구. $B_1 = \sqrt{8\pi \cdot 1.2\times 10^5} = \sqrt{3.02\times 10^6} \approx 1738$ G. 자속 보존 $B_0 A_0 = B_1 A_1$ → $A_1/A_0 = B_0/B_1 \approx 0.288$ — 자속관 단면이 3.5배 *수축*. 관측된 plage bright point 크기($\sim 0.1''$)가 MHD 시뮬레이션에서 대리 영역보다 작은 것과 일치.

### Example C — Carpet replacement timescale / 카펫 교체 시간 척도

**English.** Hagenaar (2001) reports an emergence rate of bipoles totalling $\dot{\Phi} \sim 4\times 10^{24}$ Mx/day across the visible hemisphere. Total quiet-Sun unsigned flux at $\langle B \rangle = 11$ G is $\Phi_{\text{tot}} \approx 11 \cdot 6\times 10^{22} = 6.6\times 10^{23}$ Mx. Then $\tau_{\text{rep}} \approx 6.6\times 10^{23}/4\times 10^{24} \approx 0.17$ day $\approx 4$ hours. With unresolved Hanle flux ($\sim 100$ G), $\Phi_{\text{tot}}$ scales up by $\sim 10\times$, giving $\tau_{\text{rep}} \sim 40$ hours — still much shorter than the 11-year cycle, demonstrating decoupling between local dynamo and global cycle.

**한국어.** Hagenaar(2001)는 가시 반구에 걸친 양극성 emergence 비율을 $\dot{\Phi} \sim 4\times 10^{24}$ Mx/일로 보고. $\langle B \rangle = 11$ G에서 조용한 태양 총 unsigned 자속 $\Phi_{\text{tot}} \approx 11 \cdot 6\times 10^{22} = 6.6\times 10^{23}$ Mx. $\tau_{\text{rep}} \approx 6.6\times 10^{23}/4\times 10^{24} \approx 0.17$ 일 $\approx 4$ 시간. 미분해 Hanle 자속($\sim 100$ G) 포함 시 $\Phi_{\text{tot}}$이 $\sim 10\times$ 증가하여 $\tau_{\text{rep}} \sim 40$ 시간 — 여전히 11년 주기보다 훨씬 짧아 국소 다이나모와 전역 주기의 분리를 시사.

### Example D — Hanle vs. Zeeman cross-check / Hanle vs. Zeeman 교차 점검

**English.** Suppose half a $0.5''$ resolution element (area $\sim 1.3\times 10^{15}$ cm²) is filled with $+B = 100$ G and the other half with $-B = 100$ G. Net Stokes V signal: zero (cancellation). Hanle signal from depolarisation: positive, sensitive to $|B|$ regardless of sign. So Zeeman magnetograms report $\Phi \approx 0$, while Hanle yields $\langle |B| \rangle = 100$ G. The disparity directly visualises the unresolved-flux puzzle.

**한국어.** $0.5''$ 분해 요소(면적 $\sim 1.3\times 10^{15}$ cm²)의 절반이 $+B = 100$ G, 나머지 절반이 $-B = 100$ G로 차있다고 가정. 순 Stokes V 신호: 영(상쇄). 탈편광에서 오는 Hanle 신호: 양수, 부호와 관계없이 $|B|$에 민감. Zeeman magnetogram은 $\Phi \approx 0$을 보고하지만 Hanle은 $\langle |B| \rangle = 100$ G를 산출. 이 불일치가 미분해 자속 퍼즐을 직접 시각화합니다.

### Example E — Lognormal flux distribution moments / 로그정규 자속 분포 모멘트

**English.** For a lognormal $p(\Phi)$ with $\mu = \ln(10^{17})$ Mx and $\sigma = 1.0$, the median is $e^{\mu} = 10^{17}$ Mx, the mean is $e^{\mu + \sigma^2/2} \approx 1.65\times 10^{17}$ Mx, and the mode is $e^{\mu - \sigma^2} \approx 3.7\times 10^{16}$ Mx. The right tail extends to $\sim 10^{19}$ Mx (about 0.13% of features), corresponding to ephemeral active regions. This single-parameter family captures both the bulk of internetwork features and the rare strong events.

**한국어.** $\mu = \ln(10^{17})$ Mx, $\sigma = 1.0$인 로그정규 $p(\Phi)$의 경우, 중앙값은 $e^{\mu} = 10^{17}$ Mx, 평균은 $e^{\mu + \sigma^2/2} \approx 1.65\times 10^{17}$ Mx, 최빈값은 $e^{\mu - \sigma^2} \approx 3.7\times 10^{16}$ Mx. 우측 꼬리는 $\sim 10^{19}$ Mx까지 (전체 feature의 약 0.13%) — 단명 활동 영역에 해당. 이 단일 매개변수 분포가 네트워크간 대다수와 드문 강한 이벤트 모두를 포착합니다.

---

## 7. References / 참고문헌

- de Wijn, A.G., Stenflo, J.O., Solanki, S.K., Tsuneta, S., "Small-Scale Solar Magnetic Fields", *Space Sci. Rev.* **144**, 275–315 (2009). DOI: 10.1007/s11214-008-9473-6.
- Livingston, W., Harvey, J., "The Solar Magnetic Field — Internetwork Component", *BAAS* **3**, 258 (1971); *BAAS* **7**, 346 (1975).
- Stenflo, J.O., "Magnetic-field structure of the photospheric network", *Solar Phys.* **32**, 41 (1973).
- Spruit, H.C., "Convective collapse of flux tubes", *Solar Phys.* **61**, 363 (1979).
- Hagenaar, H.J., "Ephemeral regions on a sequence of full-disk magnetograms", *ApJ* **555**, 448 (2001).
- Trujillo Bueno, J., Shchukina, N., Asensio Ramos, A., "A substantial amount of hidden magnetic energy in the quiet Sun", *Nature* **430**, 326 (2004).
- Lites, B.W., et al., "The horizontal magnetic flux of the quiet-Sun internetwork as observed with the Hinode SP", *ApJ* **672**, 1237 (2008).
- Domínguez Cerdeña, I., Sánchez Almeida, J., Kneer, F., *ApJ* **582**, L55 (2003a); *ApJ* **636**, 496 (2006a).
- Khomenko, E.V., Collados, M., Solanki, S.K., Lagg, A., Trujillo Bueno, J., *A&A* **408**, 1115 (2003); *A&A* **442**, 1059 (2005a).
- Martínez González, M.J., Collados, M., Ruiz Cobo, B., Solanki, S.K., *A&A* **469**, L39 (2007).
- Orozco Suárez, D., et al., *ApJ* **670**, L61 (2007a); *PASJ* **59**, S837 (2007b).
- Vögler, A., Schüssler, M., "A solar surface dynamo", *A&A* **465**, L43 (2007).
- Tsuneta, S., et al., "The polar magnetic field as observed with Hinode/SOT", *ApJ* **688**, 1374 (2008).
- Schrijver, C.J., Title, A.M., et al., "The dynamic quiet Solar magnetic field", *ApJ* **487**, 424 (1997).
- Wang, J., et al., "The flux distribution of the magnetic network", *Solar Phys.* **160**, 277 (1995).
- Zirin, H., "Weak solar fields and their connection to the solar cycle", *Solar Phys.* **110**, 101 (1987).
- Spruit, H.C., "A model of the solar convection zone", *Solar Phys.* **34**, 277 (1974) — predecessor of the convective collapse paper.
- Asensio Ramos, A., Trujillo Bueno, J., "Quiet Sun magnetic fields from the Mn I infrared line", *ApJ* **636**, L113 (2007).
- Sánchez Almeida, J., Landi Degl'Innocenti, E., Martínez Pillet, V., Lites, B.W., "MISMA — Micro-Structured Magnetic Atmosphere", *ApJ* **466**, 537 (1996).
- Bommier, V., Derouich, M., Landi Degl'Innocenti, E., Molodij, G., Sahal-Bréchot, S., "Hanle effect by depolarisation", *A&A* **432**, 295 (2005).
- Centeno, R., et al., "Emergence of small-scale magnetic loops through the quiet solar atmosphere", *ApJ* **666**, L137 (2007).
