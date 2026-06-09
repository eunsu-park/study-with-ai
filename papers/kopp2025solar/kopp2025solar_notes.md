---
title: "Reading Notes: Solar Irradiance Measurements"
paper_id: "85"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: notes
---

# Solar Irradiance Measurements — Reading Notes / 읽기 노트

**Paper**: Kopp, G. "Solar Irradiance Measurements", *Living Reviews in Solar Physics*, 22:1 (2025). DOI: 10.1007/s41116-025-00040-5
**Author**: Greg Kopp (LASP, University of Colorado Boulder)
**Length**: 102 pages, Living Review article

---

## 1. 핵심 기여 / Core Contribution

이 Living Review는 1978년 이후 47년간의 우주 탑재 태양복사조도 (solar irradiance) 측정을 역사적, 기기공학적, 자료 처리적 관점에서 총망라한다. TSI (Total Solar Irradiance)와 SSI (Spectral Solar Irradiance) 모두에 대해 (1) 기후 요구 측정 정밀도/안정도, (2) 개별 기기 (NIMBUS-7/ERB, ACRIM-1/2/3, SoHO/VIRGO, SORCE/TIM, TSIS-1/TIM 등)의 설계·교정·degradation, (3) 합성 시계열 (ACRIM/PMOD/RMIB/Community Consensus composites)의 구성 방법과 차이, (4) 분·시·27-일·11-년·세기 단위 변동의 원인 (흑점 darkening, 광반 brightening)을 다룬다. Kopp 자신이 주도한 SORCE/TIM의 낮은 TSI 값 (1361 W/m^2) 발표가 기존 ~1366 W/m^2 값을 대체하게 된 과정을 상세히 기록한다.

This Living Review comprehensively surveys 47 years of space-borne solar-irradiance measurements from historical, instrumental, and data-processing perspectives. For both TSI and SSI it covers (1) climate-driven measurement requirements for absolute accuracy and long-term stability, (2) design, calibration, and on-orbit degradation of individual instruments (NIMBUS-7/ERB, ACRIM-1/2/3, SoHO/VIRGO, SORCE/TIM, TSIS-1/TIM, etc.), (3) construction and intercomparison of composite records (ACRIM/PMOD/RMIB/Community Consensus), and (4) causes of variability across timescales from minutes (convection/oscillations) to centuries (solar evolution) — dominated on solar-rotation and cycle timescales by sunspot darkening and facular brightening. The review documents how Kopp's own SORCE/TIM-established lower TSI value (1361 W/m^2) superseded the earlier ~1366 W/m^2 scale.

본 논문의 특별한 가치는 각 기기의 **측정 방법론 세부사항**과 **교정 체인** (NIST → TRF → 비행 기기)에 대한 투명한 기술, 그리고 측정 기록의 한계 (기기 안정도 ≥ 현재 세기 단위 태양 변동)에 대한 솔직한 평가에 있다.
The special value of this paper lies in its transparent documentation of each instrument's methodology and calibration chain (NIST → TRF → flight instrument), and its honest assessment of the record's limitations: current instrument stabilities are comparable to or larger than the expected secular solar variability, which is why the record cannot yet definitively detect century-scale trends.

---

## 2. Reading Notes / 읽기 노트

### 2.1 Section 1: 태양복사조도의 정의 / Solar Irradiance Definitions (pp.3-8)

**TSI 정의**: 1 AU (149,598,500 km)에서 태양으로부터 단위 면적당 공간·스펙트럼 적분된 복사력 (radiant power per unit area). 지구 기후를 움직이는 총 에너지의 99.978%를 공급한다 (Table 1). 나머지 ~0.022%는 2차 원천 (지열 0.09 W/m^2, 우주 복사 8.5E-6 W/m^2 등)으로 TSI의 ~1/2700 수준.

**TSI definition**: Spatially and spectrally integrated radiant power per unit area from the Sun at 1 AU. Provides 99.978% of the total energy powering Earth's climate (Table 1). Remaining ~0.022% from secondary sources (geothermal 0.09 W/m^2, cosmic background 3.1E-6 W/m^2, etc.) totaling a factor ~2700 lower than TSI itself.

**회색체 평형**:
$$
S\cdot e\pi R^2 = 4\pi R^2 \cdot e\sigma T^4 \;\Rightarrow\; T = \left(\frac{S}{4\sigma}\right)^{1/4}
$$
$S=1361$ W/m^2, $\sigma=5.670\times 10^{-8}$ W/m^2/K^4를 대입하면 $T = 278$ K (5 °C). 그러나 실제 지구 평균 표면온도는 290 K인데, 이는 비-회색 대기 (greenhouse effect)로 인해 지표가 10-15 K 더 따뜻해야 에너지 균형을 이루기 때문.

**Gray-body equilibrium**: Plugging $S=1361$ W/m^2, $\sigma=5.670\times 10^{-8}$ W/m^2/K^4 gives $T=278$ K (5 °C). Actual mean surface temperature is 290 K; the 10-15 K offset is the greenhouse contribution from non-gray atmospheric absorption.

**태양 광구 광도**: 1 AU 구면 적분 $\Rightarrow L_\odot = 3.828\times 10^{26}$ W. 광구 표면 플럭스 $6.293\times 10^7$ W/m^2 (반지름 $R_\odot = 695{,}700$ km 사용), 유효온도 $T_\mathrm{eff}=5772.0\pm 0.8$ K (Prša et al. 2016), G2V 황색왜성.

**Photospheric luminosity**: Spherical integration at 1 AU gives $L_\odot = 3.828\times 10^{26}$ W. Surface flux of $6.293\times 10^7$ W/m^2 using $R_\odot = 695{,}700$ km implies $T_\mathrm{eff}=5772.0\pm 0.8$ K — a G2V yellow-dwarf.

**분광 복사조도 (SSI)**: 파장별 복사조도. 5772 K 흑체 스펙트럼에서 Fraunhofer 흡수선, 크로모스피어/코로나 방출선 (UV/EUV/X-ray) 추가. 200-2400 nm 적분 SSI $\approx 1309$ W/m^2 (TSI의 ~96%), 나머지 ~4%는 주로 적외선 (IR).

**SSI**: Wavelength-resolved irradiance. A 5772 K blackbody continuum with Fraunhofer absorption lines and chromospheric/coronal emission lines (UV/EUV/X-ray). The 200-2400 nm integral is about 1309 W/m^2 (~96% of TSI); remaining ~4% in IR.

### 2.2 Section 2.1-2.2: 지상 관측사와 측정 요구사항 / Pre-spacecraft History & Requirements (pp.8-14)

| 연도 / Year | 관측자 / Observer | 방법 / Method | TSI 값 / Reported value (W/m^2) |
|---|---|---|---|
| 1837 | Pouillet | Flask calorimetry | 1227 (after atm. corr.), raw 1361 |
| 1881 | Langley | Mt. Whitney bolometer | 2903 (too high, data-analysis error) |
| 1902-1962 | Abbot | Smithsonian ground stations | 1357 |
| 1968 | Willson | Active cavity radiometer, balloon | 1369 ± 0.6% |
| 1976 | Willson | Rocket (ACR IV) | 1368 ± 0.5% |
| 1977 | WRR (PMOD/WRC) | 6-cavity ground ensemble | Reference scale (later found 0.34% high vs SI) |

**측정 요구사항** (Table 2 in paper): 11년 주기 ~0.1%와 세기 단위 <0.1% 변동 검출 필요. Instrument-to-instrument trend detection에는 0.001%/yr 안정도 필요. 절대 정확도 <0.01%가 장기 공백 허용치를 결정.

**Requirements** (Table 2): Detect 0.1% solar cycle variation and <0.1% centennial trend ⇒ absolute accuracy $<0.01\%$ and stability $<0.001\%$/yr.

### 2.3 Section 2.3: 비행 기기 설계 / Spaceflight Instrument Design (pp.14-18)

**전기열치환 복사계 (ESR)**: 모든 우주 TSI 기기의 표준. 흑체 cavity를 전기저항 가열로 안정 온도에 유지하다가 shutter로 태양광을 입사시키면 전기 가열 전력이 감소. 이 감소분이 입사 복사 전력과 같음. $10^{-6}$ 수준 전력 측정 정밀도.

**Electrical-substitution radiometer (ESR)**: Standard detector for all space-based TSI instruments. A dark cavity is held at stable temperature by electrical heating; when a shutter admits sunlight, the heater power drops by an amount equal to the incident radiant power. Electrical power measured to $\sim 10^{-6}$.

**Aperture**: 금속 knife-edge (CTE $\sim 2\times 10^{-5}$) 또는 Si photolithographic, NIST 교정 불확도 $\sim 2.5\times 10^{-5}$. 회절 손실 계산 (Kopp et al. 2005) 및 산란 경험적 측정.

**Spectral selection (SSI)**: 세 가지 표준 — filter (단순, UV 선호), prism (파장 의존 굴절률, folded/unfolded), grating (회절 간섭, overlap orders 주의).

**Degradation tracking**: Redundant (duty-cycled) channels. Primary vs rarely-used secondary 비교. UV로 인해 black paint의 bond-breaking이 발생, 민감도 변화. TSIS-1/TIM은 NiP cavity로 $0.00025\%$/yr 열화만 보임 (역대 최소).

### 2.4 Section 2.4: 우주 탑재 TSI 기기 / Space-borne TSI Instruments (pp.18-42)

**주요 기기 타임라인**:
```
1978-1993  NIMBUS-7/ERB       TSI ~1376 W/m^2 (high)
1980-1989  SMM/ACRIM-1        First multi-channel, 0.005%/decade
1984-2003  ERBS/ERBE          Single-channel, bi-weekly
1991-2000  UARS/ACRIM-2
1996-      SoHO/VIRGO         L1, longest single instrument (~28 yr)
2000-2013  ACRIMSat/ACRIM-3
2003-2020  SORCE/TIM          New lower TSI 1361 W/m^2
2013-2019  TCTE/TIM           Bridge
2017-      TSIS-1/TIM (ISS)   Best stability 0.00025%/yr
2022-      FY-3E (China)
2022-2023  CTIM (CubeSat)
2025       TSIS-2 (planned)
```

**SORCE/TIM 혁신**: 20° apex conical cavity를 entering sunlight 쪽으로 향하게 하여 산란광 감소, Kopp & Lawrence 2005에서 10× 정확도 개선 보고. 4-way redundant ESR로 degradation tracking. 2003년 발표된 1361 W/m^2 값은 당시 consensus value ~1366 W/m^2보다 약 0.35% 낮아 큰 논란 유발.

**SORCE/TIM innovation**: 20°-apex conical cavities facing the entering sunlight (rather than back-to-back geometry) reduce internal-instrument scatter by ~10×. 4-way redundant ESRs enable degradation tracking. The 2003-announced 1361 W/m^2 value was ~0.35% lower than the then-consensus ~1366 W/m^2, triggering major debate.

**해결 과정 (Section 2.4.12)**:
- 2005: NIST/NASA TSI calibration workshop
- 2007: TRF (TSI Radiometer Facility) 완성 (Kopp et al. 2007). 태양 출력 수준에서 진공 상태 end-to-end 교정.
- 2010: PICARD/PREMOS 발사, TRF-교정 비행, SORCE/TIM 낮은 값 확인
- 2011: ACRIM-3 재처리로 0.5% 낮춤 (Willson 2014)
- 2012: Fehlmann et al. - WRR scale이 SI보다 0.34% 높음 확인
- 2014: VIRGO 재처리로 낮은 값에 수렴
- 결과: 새 IAU 합의값 1361 W/m^2 (Prša et al. 2016)

### 2.5 Section 2.5: SSI 기기와 reference spectra / SSI Instruments (pp.42-71)

**Reference spectra**:
- **ATLAS-3** (Thuillier et al. 2003/2004): SOLSPEC/ATLAS shuttle flights + EURECA/SOSP, 200-2400 nm, 0.1% quoted accuracy (later found up to 8% high in NIR)
- **SOLAR-ISS-V2.0** (Meftah et al. 2020): ISS/SOLAR/SOLSPEC 2008 solar min, <0.1 nm below 1000 nm
- **WHI/SIRS** (Woods et al. 2009): Whole Heliosphere Interval 2008, 0.1-2400 nm, Carrington Rotation 2068
- **SAO2010** (Chance & Kurucz 2010): 0.04 nm resolution, 200-1000 nm
- **TSIS-1 HSRS** (Coddington et al. 2023): Hybrid reference spectrum, 115 nm-200 μm, 0.3% accuracy 460-2365 nm. 현재 가장 정확한 reference.

**주요 SSI 시계열 기기**:
| Instrument | Period | Wavelength |
|---|---|---|
| NIMBUS-7 SBUV | 1978- | 160-400 nm |
| UARS/SOLSTICE | 1991-2001 | 115-420 nm |
| UARS/SUSIM | 1991-2005 | 115-410 nm |
| SORCE/SOLSTICE | 2003-2020 | 115-320 nm |
| SORCE/SIM | 2003-2020 | 240-2400 nm |
| SCIAMACHY | 2002-2012 | 240-2380 nm |
| TSIS-1/SIM | 2018- | 200-2400 nm (best) |

**UV 변동성이 큼**: Lyα (121.6 nm) factor 2, FUV (115-200 nm) >10%, UV (<300 nm) up to 50%. UV <300 nm는 11년 주기 TSI 변동의 ~30%를 기여하지만, UV 자체 변동성은 훨씬 큼.

### 2.6 Section 3: TSI 합성 시계열 / TSI Composites (pp.71-78)

**세 전통적 합성**:
1. **ACRIM composite** (Willson & Mordvinov 2003): ACRIMs (73%) + NIMBUS-7 (16%) + VIRGO (11%). 원 데이터 보정 없이 사용. 1986-1996 solar min 사이 **증가** 0.005%/yr 보고.
2. **PMOD composite** (Fröhlich 2006): NIMBUS-7/ERB + VIRGO + ACRIM modifications. 1986-1996 사이 감소/no significant trend 0.0012% ± 0.0008%/yr.
3. **RMIB composite** (Dewitte et al. 2004, updates 2016): ERB + ACRIMs + SOLCON + SOVAs + VIRGO + SORCE/TIM, daisy-chained. 1986-1996 사이 증가 ~0.15 ± 0.35 W/m^2.

**세 합성의 트렌드 차이**는 "ACRIM Gap" (1989-1992) 동안 어느 기기 (NIMBUS-7 vs ERBE)를 선택하고 어떻게 보정하느냐에 기인. 이 차이는 실제 세기 단위 태양 변동보다 **기기 안정도 부족**의 징후일 가능성이 큼.

**Community Consensus TSI Composite** (Dudok de Wit et al. 2017): ISSI 팀이 도출. 데이터 기반 (data-driven), maximum-likelihood, scale-wise weighting. 개인 편향 제거. http://spot.colorado.edu/~koppg/TSI 제공. 시간 의존 불확도 포함 (현재는 ~0.1 W/m^2).

### 2.7 Section 4: 변동성 / Solar-Irradiance Variability (pp.78-89)

#### 분-시 (3-10 min): 표면 대류 (granulation) + p-mode 진동 superposition, ~0.01% 수준. Large flares (X17, 28 Oct 2003)로 +0.028% 일시 증가 기록. GOES XRS에서 본 것의 100배 에너지가 TSI에서 나타남.

#### 27일 회전주기: 흑점 passage로 -0.1% ~ -0.34% 일시 감소 (Oct 2003). 광반은 UV에서 밝지만 가시광에서는 sunspot darkening만 우세. 시간 평균 후에는 광반이 흑점을 **이김** (net brightening during high activity).

#### 11년 주기 (solar cycle): ~0.1% peak-to-peak. PMOD composite는 Cycle 21 (1980 peak) 0.082%, Cycle 22 0.077%, Cycle 23 0.096%, Cycle 24 0.063%. TSI minimum ~1360.5 W/m^2, maximum ~1362 W/m^2.

**SATIRE 기반 재구성**: Sunspot + facular magnetic flux → TSI 변동의 83-93% 설명 (PMOD/SORCE composites). Chapman et al. ground-based photometry로 95% 설명.

**SSI-TSI 민감도** (Kopp et al. 2024):
$$
\frac{\Delta SSI}{SSI}\approx \frac{5}{8\lambda}\cdot\frac{\Delta TSI}{TSI}\quad (\lambda\ \text{in }\mu m)
$$
$\lambda=0.4$ μm에서 ΔSSI/SSI $\approx 1.56\times$ ΔTSI/TSI, $\lambda=0.65$에서 ~1×, $\lambda>2$ μm에서 ~0.3-0.5×. UV (<400 nm)는 흑점 효과 약해서 깨짐.

#### 세기 단위: 직접 측정 불가능. 모델 (Wang et al. 2005, NRLTSI2, SATIRE, Steinhilber 2009, Lean 2018) → Maunder Minimum (1645-1715) 동안 현재 대비 -0.05% ~ -0.1% 추정. $-0.1\%$는 $-1.36$ W/m^2 TSI ⇒ radiative forcing $-0.24$ W/m^2 (1/4 factor, albedo) ⇒ $\Delta T\approx 0.1$ K per W/m^2 민감도 가정시 $\Delta T \approx -0.024$ K 수준 (직접). 증폭 메커니즘 (UV/stratospheric ozone, Gray et al. 2010)을 감안하면 지역적으로 더 클 수 있음.

#### Milankovitch (21,000-420,000 yr) & 태양 진화 (billion-yr): 관측 범위 밖. 태양은 현재 광도 증가율 0.009%/Myr, 초기 Sun은 현재의 72%.

### 2.8 Section 5: 미래 개선 / Future Measurement Needs (pp.89-92)

- **Stability**: <0.001%/yr 필요, TSIS-1 이미 달성 중
- **Accuracy**: <0.01%, TSIS-2에서 TRF 기반 교정
- **Duration**: 연속 기록 유지 critical — TSIS-1/TIM-TSIS-2 overlap 필수
- **Precision**: 작은 변동 연구를 위한 노이즈 감소

예정 임무: **TSIS-2** (2025), **TRUTHS** (~2026, ESA/UKSA, SI-traceable transfer standard), **CLARREO** (~2023), **Libera** (ERB 후속)

---

## 3. Key Takeaways / 핵심 시사점

### 3.1 TSI는 "태양 상수"가 아니다 / TSI is not a "solar constant"

Abbot이 이미 1950년대에 ~0.1% 변동을 주장했으나 당시 반박됨. SMM/ACRIM-1 (1980-1988) 데이터가 11년 주기와 **동위상** (in-phase) 변동을 결정적으로 보임: 태양 활동이 **높을수록** TSI가 높다. 직관과 반대 (흑점이 많은데 왜 밝을까?)는 광반 brightening이 흑점 darkening을 능가하기 때문.

Abbot claimed ~0.1% variability already in the 1950s but was disputed at the time. SMM/ACRIM-1 (1980-1988) decisively established that TSI varies **in-phase** with the 11-year cycle — higher activity means higher TSI. The counter-intuitive result (more sunspots, brighter Sun) is because facular brightening dominates sunspot darkening on the time-averaged full disk.

### 3.2 측정 기록의 핵심은 **중첩 기기** / The key is overlapping instruments

단일 기기로는 0.001%/yr 안정도 장기 달성 불가능. Daisy-chaining으로 기기 교체 시 scale 차이 보정. 1978년 이후 공백 없는 47년 기록은 "uninterrupted series of overlapping instruments" 덕분. 향후 TSIS-1/TSIS-2 overlap이 마찬가지로 결정적.

No single instrument achieves 0.001%/yr over decades. Daisy-chaining corrects for scale offsets at each instrument transition. The 47-year uninterrupted record since 1978 depends entirely on overlap between successive instruments. TSIS-1/TSIS-2 overlap will be similarly critical.

### 3.3 SORCE/TIM의 낮은 TSI 값은 **더 정확한 것이었다** / The SORCE/TIM lower value was the correct one

2003년 SORCE/TIM 1361 W/m^2 발표 시 community는 이를 잘못된 것으로 여겼다. 2007년 TRF 시설 완성 후 SI-traceable ground calibration으로 WRR scale 자체가 0.34% 높았음이 밝혀짐 (Fehlmann et al. 2012). ACRIM-3 (2011) 및 VIRGO (2014) 재처리가 SORCE/TIM 값에 수렴. **과학의 자기-수정 과정**의 모범 사례.

When SORCE/TIM announced 1361 W/m^2 in 2003, the community initially viewed it as erroneous. After the TRF facility (2007) enabled SI-traceable calibrations at solar power levels in vacuum, the WRR scale itself was found to be 0.34% high (Fehlmann et al. 2012). ACRIM-3 (2011) and VIRGO (2014) reprocessings converged on the SORCE/TIM value. A textbook example of science self-correction.

### 3.4 합성 시계열의 트렌드 차이는 **태양이 아니라 기기** / Composite trend differences reflect instrument instability, not the Sun

PMOD (flat trend), ACRIM (increasing), RMIB (slightly increasing) 사이의 1986-1996 minimum-to-minimum 차이는 ~0.15 W/m^2. 각 composite의 PI가 선호하는 기기 선택과 보정 방법 차이에서 기인. 통계적 독립 접근 (Community Consensus, Dudok de Wit 2017) 및 frequency-dependent data fusion (Montillet et al. 2022)이 개인 편향을 제거.

The 1986-1996 minimum-to-minimum trend differences between PMOD (flat), ACRIM (+0.005%/yr), and RMIB (+0.15 W/m^2) composites amount to ~0.15 W/m^2 and arise from each PI's choice of instrument weighting and corrections. Statistically independent approaches (Community Consensus; frequency-dependent data fusion) eliminate these personal biases.

### 3.5 UV 변동이 기후-태양 커플링에서 비례 이상으로 중요 / UV variability punches above its weight in Sun-climate coupling

UV (<300 nm)는 TSI의 ~8%에 불과하지만 11년 주기 TSI 변동의 ~30%를 기여 (Rottman 2005). Lyα 2× factor 변화, FUV >10%, UV (<300) up to 50%. 성층권 오존, QBO, 열권 온도에 영향. Kopp et al. 2024의 $\Delta SSI/SSI \approx (5/8\lambda)(\Delta TSI/TSI)$ 관계가 핵심 단순화.

UV (<300 nm) contributes only ~8% of TSI but ~30% of TSI's solar-cycle variability. Lyα varies by factor of 2, FUV by >10%, UV (<300 nm) up to 50%. Drives stratospheric ozone, QBO, thermospheric temperature. The Kopp et al. 2024 approximation $\Delta SSI/SSI \approx (5/8\lambda)(\Delta TSI/TSI)$ captures the wavelength dependence in visible/NIR.

### 3.6 현재 기록은 **세기 단위 태양 변동을 검출할 수 없다** / Current record cannot detect secular solar trends

47년 기록의 기기 안정도는 잘 해야 0.0009%/yr (PMOD claim). 세기 단위 변동 검출을 위한 요구사항 <0.001%/yr에 근접하지만 duration이 부족. Maunder Minimum 재현 시 TSI 감소 -0.05% ~ -0.1% 예상인데, 이는 현 stability 불확도보다 약간 큰 수준. TSIS-2/TRUTHS/CLARREO가 향후 이 문제 해결의 열쇠.

The 47-year record's instrument stability is at best 0.0009%/yr (PMOD claim), close to the <0.001%/yr requirement but the duration is insufficient. A Maunder-Minimum-like variation would give $\Delta TSI$ of $-0.05\%$ to $-0.1\%$, marginally above stability uncertainties. TSIS-2, TRUTHS, and CLARREO are key to resolving this.

### 3.7 SSI 측정은 TSI보다 $10-15\times$ 뒤처져 있다 / SSI measurements lag TSI by 10-15×

TSI 정확도 ~0.016%, 안정도 ~0.0012%/yr vs SSI 정확도 >0.24%, 안정도 >0.01%/yr (McClintock et al. 2005 requirements 5%/0.5%/yr). UV <300 nm 외에는 1979-2002 동안 기록 공백. TSIS-1/SIM이 TSI 수준 안정도에 처음 근접. 기후 모델의 spectral resolution 입력 품질이 제한.

TSI accuracy ~0.016%, stability ~0.0012%/yr vs SSI accuracy >0.24%, stability >0.01%/yr. UV (<300 nm) continuous since 1979 but broadband UV-NIR began only in 2003 (SCIAMACHY). TSIS-1/SIM is the first to approach TSI-level stability. Climate-model spectral-forcing input quality is limited by this.

### 3.8 Flares와 Convection은 **기후 관련 없음** / Flares and convection are climate-irrelevant

X17 flare (28 Oct 2003) TSI 증가 0.028%는 관측사 최대였으나 duration <1 hr ⇒ 기후 시스템의 수-시 반응 시간척도 이하. 대류 (3-10 min)와 p-mode 진동 마찬가지. 태양 기후 forcing은 27-일 이상 timescale에서만 유효.

The X17 flare (28 Oct 2003) produced the largest TSI excursion (+0.028%) ever observed, but its <1-hour duration is below the climate system's response timescale. Convection (3-10 min) and p-mode oscillations likewise. Solar climate forcing operates only on timescales >27 days.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 TSI와 Solar Luminosity

$$
S_0 = 1361\ \mathrm{W\,m^{-2}}\quad\text{(IAU 2015, 1 AU)}
$$
$$
L_\odot = 4\pi D_\mathrm{AU}^2 \cdot S_0 = 4\pi\,(1.496\times 10^{11})^2 \cdot 1361 = 3.828\times 10^{26}\ \mathrm{W}
$$

광구 플럭스 / Photospheric surface flux:
$$
F_\mathrm{phot} = \frac{L_\odot}{4\pi R_\odot^2} = \frac{3.828\times 10^{26}}{4\pi\,(6.957\times 10^8)^2} = 6.293\times 10^7\ \mathrm{W\,m^{-2}}
$$

Effective temperature via Stefan-Boltzmann:
$$
F_\mathrm{phot} = \sigma T_\mathrm{eff}^4 \;\Rightarrow\; T_\mathrm{eff} = \left(\frac{6.293\times 10^7}{5.670\times 10^{-8}}\right)^{1/4} = 5772\ \mathrm{K}
$$

### 4.2 지구 평형 온도 / Earth Equilibrium Temperature

흡수된 태양 입력 = 방출된 복사
Absorbed solar = Emitted radiation:
$$
(1-A_\mathrm{albedo})\cdot S_0 \cdot \pi R_E^2 = \varepsilon \sigma T_E^4 \cdot 4\pi R_E^2
$$
$$
T_E = \left[\frac{(1-A)S_0}{4\varepsilon\sigma}\right]^{1/4}
$$
$A=0.29$, $\varepsilon=1$ 회색체 가정: $T_E = 255$ K (-18 °C). 실제 290 K는 온실효과 +35 K. $\varepsilon=0.612$ (non-gray 조정) 시 Kopp 논문의 278 K 값 나옴.

Using $A=0.29$, $\varepsilon=1$: $T_E = 255$ K (-18 °C); actual 290 K gives +35 K greenhouse. Kopp's 278 K uses different $\varepsilon$ convention.

### 4.3 기후 복사강제 / Climate Radiative Forcing

$$
\Delta F = \frac{1-A}{4}\cdot \Delta\mathrm{TSI}
$$
$1/4$는 spherical geometry ($\pi R^2 / 4\pi R^2$). $\Delta TSI = 1$ W/m^2 ⇒ $\Delta F = 0.18$ W/m^2. 기후 감도 $\lambda_c \sim 0.5-1$ K per W/m^2 (IPCC).

간단화된 감도 (본 리뷰 맥락): $\Delta T \approx 0.1$ K per $\Delta\mathrm{TSI}$ W/m^2 (Lean & Matthes). 11년 주기 $\Delta TSI \approx 1.36$ W/m^2 (0.1%) ⇒ $\Delta T_\mathrm{cycle} \approx 0.1$ °C.

### 4.4 Degradation 보정 / Degradation correction

Primary channel sensitivity $S_p(t)$ vs less-used reference $S_r(t)$:
$$
S_p(t) = S_p(0)\cdot[1 - f(E_p(t))]
$$
where $E_p(t) = \int_0^t \mathrm{(sun exposure time)}\,dt'$. Periodic cross-calibrations give $f$. TSIS-1/TIM $f \leq 0.0013\%$ total over 7 years.

### 4.5 합성 시계열 scaling / Composite stitching

각 기기 $i$가 시간 $t$에서 TSI 값 $X_i(t)$와 불확도 $\sigma_i(t)$를 보고. Community Consensus (Dudok de Wit 2017):
$$
\hat{X}(t) = \frac{\sum_i X_i(t)/\sigma_i^2(t)}{\sum_i 1/\sigma_i^2(t)}
$$
with additional frequency-scale decomposition (wavelet) for common-mode noise removal.

### 4.6 SATIRE 기반 재구성 / SATIRE-based reconstruction

SATIRE: Spectral And Total Irradiance REconstruction. 자기장 필링 factor로 표면 성분 분리:
$$
F_\mathrm{TSI}(t) = \alpha_q(t) F_q + \alpha_s(t) F_s + \alpha_p(t) F_p + \alpha_n(t) F_n
$$
where subscripts $q$=quiet Sun, $s$=sunspots (umbra+penumbra), $p$=plage (faculae), $n$=network. $\alpha_i(t)$는 magnetogram에서, $F_i$는 static 1D atmosphere models (ATLAS9 계열). 11년 주기 TSI 변동의 83-93% 재현.

### 4.7 SSI–TSI 민감도 근사 / SSI-TSI sensitivity (Kopp et al. 2024)

$$
\boxed{\frac{\Delta\mathrm{SSI}(\lambda)}{\mathrm{SSI}(\lambda)} \approx \frac{5}{8\lambda}\cdot\frac{\Delta\mathrm{TSI}}{\mathrm{TSI}},\quad \lambda\ \text{in }\mu\mathrm{m}}
$$

적용 가능 범위: $\lambda \gtrsim 0.4$ μm (가시-NIR). UV에서는 흑점 darkening이 UV에서 약하므로 관계가 깨짐. 해석: $\lambda=0.4$ μm ⇒ 계수 1.56 (SSI가 TSI의 1.56× 변동), $\lambda=0.65$ (피크 방출) ⇒ 0.96, $\lambda=2$ μm ⇒ 0.31.

### 4.8 측정 요구사항 / Requirements (Table 2)

| Parameter | Requirement |
|---|---|
| Absolute Accuracy (TSI) | $< 0.01\%$ |
| Stability (TSI) | $< 0.001\%\ \mathrm{yr}^{-1}$ |
| Absolute Accuracy (SSI) | $\sim 1\%$ |
| Stability (SSI) | $\sim 0.1\%\ \mathrm{yr}^{-1}$ |

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1837 Pouillet pyrheliometer (ground)
  |  1227 W/m^2 (with atm. correction)
1881 Langley Mt. Whitney bolometer
  |  2903 W/m^2 (erroneously high)
1902-1962 Abbot Smithsonian ground program
  |  1357 W/m^2, 1st claim of ~0.1% variability
1977 PMOD/WRC establishes WRR scale (later found 0.34% high)
1978 *** NIMBUS-7/ERB launch — space-borne TSI era begins ***
1980 SMM/ACRIM-1 (Willson) — multi-channel, in-phase 11-yr variation
1988 Willson & Hudson — cycle variation confirmed
1989-1992 "ACRIM Gap" (Challenger delayed ACRIM-2)
1996 SoHO/VIRGO at L1 (longest single-instrument record)
2003 *** SORCE/TIM — new lower TSI 1361 W/m^2 announced ***
2007 TRF facility (SI-traceable ground calibration; Kopp et al.)
2011 ACRIM-3 data lowered by 0.5% after TRF calibration
2012 Fehlmann et al.: WRR is 0.34% high vs SI
2014 VIRGO reprocessed, converges on 1361 W/m^2
2016 Prša et al.: IAU accepts 1361 W/m^2 as nominal
2017 *** TSIS-1/TIM (ISS) — 0.00025%/yr degradation ***
2017 Dudok de Wit: Community Consensus Composite
2023 Coddington: TSIS-1 HSRS reference spectrum
2024 Kopp: SSI-TSI sensitivity approximation
2025 *** This review (Kopp) — synthesis of 47-yr record ***
2025 TSIS-2 planned launch
~2026 TRUTHS (ESA/UKSA)
~2030+ CLARREO, Libera

Future: Multi-century stable TSI/SSI record enabling
        secular solar trend detection; improved climate models
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper | Connection / 연결 |
|---|---|
| **Paper #11 Haigh (2007)** "The Sun and the Earth's Climate" | 태양-기후 커플링의 물리 기제를 논의, 본 논문은 그 입력 데이터 (TSI/SSI)의 측정 인프라 제공 / Haigh discusses Sun-climate physics; Kopp provides the measurement infrastructure for the inputs |
| Paper #4 Sheeley (2005) CMEs/solar cycle | 11년 주기 활동의 CME/현상 측면 vs Kopp의 복사조도 측면 / Activity-cycle phenomena (CMEs) vs radiative output (TSI/SSI) |
| Paper #42 Hathaway (2015) solar cycle | Sunspot number ↔ TSI 광반/흑점 contributions의 기반 / Sunspot record is the proxy basis for historical TSI reconstructions |
| Paper #48-52 SATIRE/irradiance models | Krivova, Unruh, Solanki SATIRE 재구성의 관측적 근거 / Observational basis for SATIRE irradiance reconstructions |
| Paper #72 Usoskin (2023) solar activity history | Cosmogenic isotope based pre-telescopic irradiance; Kopp는 그 calibration을 제공 / Cosmogenic-isotope reconstructions need the Kopp-era irradiance record for calibration |
| IPCC AR6 (2021) | 태양 forcing 입력; Community Consensus Composite가 권장 / Solar-forcing input to IPCC uses Community Consensus TSI Composite |
| Paper #11 Haigh (2007), Gray et al. (2010) | UV-stratosphere coupling 메커니즘, Kopp의 SSI 측정이 물리 계산 입력 / UV-stratospheric mechanisms; Kopp's SSI measurements feed those calculations |

---

## 7. References / 참고문헌

- Kopp, G. "Solar Irradiance Measurements", *Living Reviews in Solar Physics*, 22:1 (2025). DOI: 10.1007/s41116-025-00040-5
- Kopp, G. & Lawrence, G. "The Total Irradiance Monitor (TIM)", *Sol. Phys.* 230, 91-109 (2005)
- Kopp, G. & Lean, J.L. "A new, lower value of total solar irradiance", *GRL* 38, L01706 (2011)
- Kopp, G. et al. "The TSI Radiometer Facility (TRF)", *Proc. SPIE* 6677 (2007)
- Prša, A. et al. "Nominal values for selected solar and planetary quantities: IAU 2015 Resolution B3", *AJ* 152, 41 (2016)
- Fröhlich, C. "Solar irradiance variability since 1978", *Space Sci. Rev.* 125, 53-65 (2006)
- Willson, R.C. & Mordvinov, A.V. "Secular total solar irradiance trend during solar cycles 21-23", *GRL* 30, 1199 (2003)
- Dudok de Wit, T. et al. "Methodologies to understand solar irradiance variability", *GRL* 44, 1196 (2017) [Community Consensus Composite]
- Fehlmann, A. et al. "Fourth World Radiometric Reference to SI radiometric scale comparison", *Metrologia* 49, S34 (2012)
- Coddington, O. et al. "The TSIS-1 Hybrid Solar Reference Spectrum", *Earth Space Sci.* 8 (2021); updated 2023
- Meftah, M. et al. "SOLAR-ISS: A new reference solar spectrum", *Astron. Astrophys.* 611, A1 (2018)
- Haigh, J.D. "The Sun and the Earth's Climate", *Living Reviews in Solar Physics* 4:2 (2007) [Paper #11]
- Stephens, G.L. et al. "The changing nature of Earth's reflected sunlight", *Proc. R. Soc. A* 478 (2022)
- Krivova, N.A. et al. "Reconstruction of solar spectral irradiance since the Maunder minimum", *JGR Space Phys.* 115, A12112 (2010)
- Thuillier, G. et al. "The solar spectral irradiance from 200 to 2400 nm (ATLAS reference)", *Sol. Phys.* 214, 1-22 (2003)
- Solanki, S.K., Krivova, N.A., Haigh, J.D. "Solar Irradiance Variability and Climate", *ARA&A* 51, 311-351 (2013)
- IPCC AR6 WG1 (2021)
- Lean, J.L. "Evolution of total solar irradiance during the past three decades", *GRL* 37 (2010)
- Dewitte, S. et al. "Measurement and uncertainty of long term total solar irradiance trend", *Sol. Phys.* 224, 209-216 (2004)
- Chatzistergos, T. et al. "Ca II K historical data...", *A&A* 609, A92 (2018)
- Montillet, J.-P. et al. "A new TSI composite based on an optimal frequency-dependent statistical method", *JSWSC* 12, 9 (2022)

---

## Appendix A. 계측기 기술 상세 / Instrument Technical Details

### A.1 Aperture 정밀도 / Aperture precision

- Al knife-edge: CTE $\sim 2\times 10^{-5}$/K; thermal area correction $0.0004\%$/°C
- Photolithographic Si: Same uncertainty level, smaller size, used in SSI & compact TSI instruments
- NIST area measurements: $\sim 2.5\times 10^{-5}$ relative uncertainty for TSI-class apertures
- Diffraction correction: typical $<1.8\%$ (Shirley et al. 2002)
- Scatter from aperture edges: empirically measured, $\sim 5\times 10^{-4}$ typical

### A.2 Redundant channel duty cycles / 중복 채널 사용 비율

- VIRGO PMO6: 150:1 (primary:secondary)
- VIRGO DIARAD: 1200:1
- ACRIM-1/2/3: 3 back-to-back cavity pairs, variable duty cycle
- SORCE/TIM: 4-way redundancy with different duty cycles
- TSIS-1/TIM: Similar 4-way design, carbon-nanotube interior coating

### A.3 TSI 기기 측정 정확도 순위 / Ranking of TSI instrument accuracy

From Table 3 (short-term uncertainty in W/m^2):
1. TSIS-1/TIM: 0.032
2. VIRGO V8: 0.021
3. SORCE/TIM: 0.028
4. ACRIM-3: 0.032
5. PREMOS: 0.033
6. NIMBUS-7/ERB: 0.110 (worst)

TSIS-1/TIM has the lowest **degradation** rate of any prior instrument at $0.00025\%$/yr.

### A.4 WRR to SI scale correction / WRR-to-SI 스케일 보정

Fehlmann et al. (2012) demonstrated WRR reads $0.34 \pm 0.18\%$ (= 4.6 W/m^2) high compared to the absolute SI scale. This one correction alone explains most of the offset between older instruments (calibrated to WRR) and SORCE/TIM (calibrated via TRF to SI). Instruments directly calibrated on WRR: VIRGO, PREMOS. Instruments directly calibrated via TRF: SORCE/TIM, TCTE/TIM, TSIS-1/TIM, CTIM.

---

## Appendix B. Numerical examples / 수치 예제

### B.1 지구 평형 온도 계산 / Equilibrium temperature

Given $S_0 = 1361$ W/m^2, $A = 0.29$, $\varepsilon = 1$:
$$
T_E = \left[\frac{(1-0.29)\cdot 1361}{4\cdot 5.670\times 10^{-8}}\right]^{1/4} = \left[\frac{966.3}{2.268\times 10^{-7}}\right]^{1/4} = (4.261\times 10^9)^{1/4} = 255.3\ \mathrm{K}
$$

### B.2 11년 주기 복사 강제 / 11-yr cycle forcing

$\Delta\mathrm{TSI}_\mathrm{cycle} \approx 1.36$ W/m^2 (0.1% of 1361):
$$
\Delta F_\mathrm{top} = \frac{1-0.29}{4}\cdot 1.36 = 0.241\ \mathrm{W\,m^{-2}}
$$
With climate sensitivity ~0.5 K per W/m^2 (IPCC range 0.4-1.2): $\Delta T_\mathrm{cycle}\approx 0.12$ K — consistent with observed ~0.1 K global surface-T variation in-phase with cycle (Lean & Matthes 2017).

### B.3 Maunder Minimum forcing estimate

$\Delta\mathrm{TSI}_\mathrm{MM}\approx -0.1\%$ ($\sim -1.36$ W/m^2) per SATIRE, NRLTSI2:
$$
\Delta F_\mathrm{MM}\approx -0.24\ \mathrm{W\,m^{-2}};\quad \Delta T_\mathrm{MM}\approx -0.12\ \mathrm{K}\ \text{(direct)}
$$
UV/stratospheric amplification (Gray et al. 2010) can double this regionally, especially in northern Europe (Eddy 1976, Little Ice Age correlation).

### B.4 SSI 민감도 적용 / SSI sensitivity applied

At 11-year cycle, $\Delta\mathrm{TSI}/\mathrm{TSI} = 10^{-3}$ (0.1%).
- $\lambda = 0.4$ μm: $\Delta SSI/SSI \approx 5/(8\cdot 0.4)\cdot 10^{-3} = 1.56\times 10^{-3}$ (0.156%)
- $\lambda = 0.65$ μm: $\approx 0.96\times 10^{-3}$ (0.096%)
- $\lambda = 2$ μm: $\approx 0.31\times 10^{-3}$ (0.031%)
- $\lambda = 0.12$ μm (Lyα): formula breaks; observed factor 2 (100% variation)

---

## Appendix C. Open questions / 미해결 문제

1. **세기 단위 secular trend 존재 여부**: 현재 ±0.1 W/m^2 수준 불확도. TSIS-1/TSIS-2 시대로 해결 가능?
   Existence of centennial secular TSI trend: ±0.1 W/m^2 uncertainty; resolvable in TSIS-1/TSIS-2 era?
2. **Maunder Minimum TSI 수준**: -0.05% vs -0.1% vs -0.3% 모델 간 차이.
   Maunder-Minimum TSI level: -0.05% to -0.3% across models.
3. **SSI 세기 단위 재구성 품질**: UV-visible-NIR 상대 기여.
   Quality of SSI centennial reconstructions: relative UV-visible-NIR contributions.
4. **기후-태양 증폭 메커니즘의 정량화**: UV-stratosphere, TSI-ocean, cloud response.
   Quantification of climate-solar amplification: UV-stratosphere, TSI-ocean, cloud response.
5. **Pre-spacecraft reconstructions의 calibration**: sunspot records, Ca II K, cosmogenic isotopes를 통합하는 best practice.
   Calibration of pre-spacecraft reconstructions integrating sunspots, Ca II K, cosmogenic isotopes.

