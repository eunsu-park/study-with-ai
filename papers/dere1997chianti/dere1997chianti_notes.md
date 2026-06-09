---
title: "CHIANTI — an atomic database for emission lines. I. Wavelengths greater than 50 Å"
authors: [K. P. Dere, E. Landi, H. E. Mason, B. C. Monsignori Fossi, P. R. Young]
year: 1997
journal: "Astronomy and Astrophysics Supplement Series, 125, 149–173"
doi: "10.1051/aas:1997368"
topic: Solar Observation / Atomic database
tags: [CHIANTI, emission lines, atomic data, contribution function, optically thin plasma]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 63. CHIANTI — an atomic database for emission lines. I. Wavelengths greater than 50 Å / CHIANTI — 방출선용 원자 데이터베이스. I. 50 Å 이상의 파장

---

## 1. Core Contribution / 핵심 기여

### English
This is the founding paper of **CHIANTI**, an atomic database created to synthesise the EUV/UV/X-ray emission line spectra of optically thin astrophysical plasmas. For each ion the database stores three transparent ASCII files: an energy-level file (`.elvlc`), a radiative-data file with wavelengths and weighted oscillator strengths or A-values (`.wgfa`), and an electron-collision-strength file containing 5-point spline fits to the thermally averaged effective collision strengths $\Upsilon_{ij}(T_e)$ in the Burgess & Tully (1992) scaled form (`.splups`). Version 1 covers wavelengths longer than 50 Å for cosmically abundant elements from H to Ni and is essentially complete for that range, drawing energy levels primarily from the NIST critical compilation (Martin et al. 1995), supplementing missing data with theoretical SSTRUCT (UCL) calculations, and adopting collision data preferentially from Close-Coupling / R-matrix calculations (Iron Project, QUB R-matrix) over older Distorted Wave or Coulomb-Bethe results, and replacing the long-used Van Regemorter $\bar g$ approximation whenever possible.

The paper bundles five intentional design goals — easy to update, easy to distribute (anonymous FTP at CDS Strasbourg), transparent to the user, accuracy maintained by visual inspection of every collision dataset using the `BURLY` IDL routines, and a data/programming structure suitable for downstream code development. The companion IDL package solves statistical equilibrium for level populations, builds synthetic spectra, and supplies temperature/density diagnostics; it was integrated into SOHO/CDS analysis software by Pike & Del Zanna. CHIANTI thereby unified a fragmented landscape of competing spectral codes (Landini & Monsignori Fossi 1970/1990, Mewe 1972, Mewe & Gronenschild 1981, Raymond & Smith 1977, Kato 1976, Stern 1978, Gaetz & Salpeter 1983, Arnaud & Rothenflug 1985 ionisation balance) and became the de-facto international standard for solar coronal spectroscopy through SOHO, TRACE, Hinode/EIS, SDO/AIA, IRIS, and Solar Orbiter EUI/SPICE.

### 한국어
이 논문은 광학적으로 얇은(optically thin) 천체 플라즈마의 EUV/UV/X-ray 방출선 스펙트럼을 합성하기 위한 원자 데이터베이스인 **CHIANTI**의 창립 논문이다. 데이터베이스는 이온별로 세 가지 투명한 ASCII 파일을 저장한다: 에너지 준위(`.elvlc`), 파장과 가중 발진자 강도/A 계수를 담은 복사 데이터(`.wgfa`), 그리고 Burgess & Tully (1992) 스케일링 형태의 5-point spline으로 압축된 열평균 유효 충돌 강도 $\Upsilon_{ij}(T_e)$ 파일(`.splups`)이다. 버전 1은 우주적으로 풍부한 원소(H–Ni)에 대해 50 Å 이상 영역을 사실상 완전히 커버하며, 에너지 준위는 NIST 임계 편찬(Martin et al. 1995)을 우선 사용하고 부족한 부분은 UCL의 SSTRUCT로 보완한다. 충돌 데이터는 Close-Coupling/R-matrix(Iron Project, QUB R-matrix) 결과를 Distorted Wave나 Coulomb-Bethe 결과보다 우선 채택하며, 가능한 한 오래된 Van Regemorter $\bar g$ 근사를 대체했다.

이 논문은 다섯 가지 설계 목표를 명시한다: (1) 갱신 용이성, (2) 익명 FTP(CDS 스트라스부르)를 통한 자유로운 배포, (3) 사용자에 대한 투명성, (4) `BURLY` IDL 루틴으로 모든 충돌 데이터를 시각 검사하여 정확도 유지, (5) 후속 프로그램 개발에 적합한 데이터·프로그래밍 구조. 동봉된 IDL 패키지는 통계 평형을 풀어 준위 인구를 계산하고, 합성 스펙트럼과 온도/밀도 진단을 제공하며, Pike & Del Zanna에 의해 SOHO/CDS 분석 소프트웨어에 통합되었다. 이로써 CHIANTI는 Landini & Monsignori Fossi (1970/1990), Mewe (1972), Mewe & Gronenschild (1981), Raymond & Smith (1977), Kato (1976), Stern (1978), Gaetz & Salpeter (1983), Arnaud & Rothenflug (1985) 이온화 평형 등 분산되어 있던 분광 코드를 통합하여, SOHO, TRACE, Hinode/EIS, SDO/AIA, IRIS, Solar Orbiter EUI/SPICE 시대 태양 코로나 분광학의 사실상 국제 표준이 되었다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Abstract & §1 Introduction (p. 149) / 초록 및 서론

#### English
The abstract states the scope plainly: **a comprehensive set of accurate atomic data is required to interpret astrophysical and solar emission line spectra, and CHIANTI provides the energy levels, wavelengths, radiative data, and electron collisional excitation data needed for cosmically abundant ions**. Collision data are stored according to the Burgess & Tully (1992) scaling. Version 1 is "essentially complete for specifying the emission spectrum at wavelengths greater than 50 Å". A list of observed lines between 50 and 1100 Å has been compiled and compared to CHIANTI predictions. The IDL package supplies optically thin synthetic spectra, density/temperature diagnostics, and emission measure routines. The atomic database and IDL routines are distributed via anonymous FTP.

§1 explains the historical motivation. Many spectral codes (Landini & Monsignori Fossi 1970/1990, Tucker & Koren 1971, Mewe 1972, Mewe & Gronenschild 1981, Kato 1976, Raymond & Smith 1977, Stern et al. 1978, Gaetz & Salpeter 1983) were assembled when accurate collision strengths were lacking, so they relied heavily on the Van Regemorter (1962) $\bar g$ formula or its variants. Mason (1996a) provides a comparison and critique of these codes. Computer technology has now made reliable calculations widely available (Brickhouse et al. 1995, Monsignori Fossi & Landini 1994a, Mewe et al. 1995). The five explicit goals of CHIANTI are: (1) ready to update, (2) easy to distribute, (3) transparent to the end user, (4) accuracy maintained by visually examining as much of the input data as possible, (5) data/programming structure that facilitates the development of programs by end users. The basic unit of the database is the individual ion: each ion has its own directory containing an energy level file, a wavelength + radiative rate file, and a file with fits to collision strengths. Energy levels come predominantly from NIST (Martin et al. 1995), supplemented by theoretical estimates where measurements are missing. All collision data (collision strengths $\Omega$ and effective collision strengths $\Upsilon$) have been visually inspected and rescaled following Burgess & Tully (1992). Element abundances are user-specifiable; ionisation equilibrium uses Arnaud & Rothenflug (1985). The database currently reproduces the optically thin emission line spectrum at $\lambda > 50$ Å for $N_e \lesssim 10^{15}$ cm$^{-3}$. There is no long-wavelength limit, but neutrals are not included so the database becomes less comprehensive at long wavelengths. Cosmically abundant elements from H through Ni are included; Tables 1 and 2 list which ionisation stages are present. The IDL routines were integrated into the SOHO/CDS analysis software by Pike & Del Zanna.

#### 한국어
초록은 그 범위를 명확히 한다: **천체·태양 방출선 스펙트럼을 해석하려면 정확한 원자 데이터의 종합 집합이 필요하며, CHIANTI는 우주적으로 풍부한 이온의 에너지 준위, 파장, 복사 데이터, 전자 충돌 여기 데이터를 제공한다**. 충돌 데이터는 Burgess & Tully (1992) 스케일링 형식으로 저장된다. 버전 1은 50 Å 이상에서 "방출 스펙트럼 명세에 사실상 완전"하다. 50–1100 Å 영역의 관측선 목록이 정리되어 CHIANTI 예측과 비교되었다. IDL 패키지는 광학적으로 얇은 합성 스펙트럼, 밀도/온도 진단, 방출 측도 루틴을 공급한다. 원자 데이터베이스와 IDL 루틴은 익명 FTP로 배포된다.

§1은 역사적 동기를 설명한다. 많은 분광 코드(Landini & Monsignori Fossi 1970/1990, Tucker & Koren 1971, Mewe 1972, Mewe & Gronenschild 1981, Kato 1976, Raymond & Smith 1977, Stern et al. 1978, Gaetz & Salpeter 1983)는 정확한 충돌 강도가 없던 시기에 만들어졌기에 Van Regemorter (1962)의 $\bar g$ 공식 또는 그 변형에 크게 의존했다. Mason (1996a)이 이 코드들을 비교·비평했다. 이제 컴퓨터 기술 발전으로 신뢰할 수 있는 계산이 대량 가능해졌다(Brickhouse et al. 1995, Monsignori Fossi & Landini 1994a, Mewe et al. 1995). CHIANTI의 명시적 다섯 가지 목표는: (1) 갱신 용이, (2) 배포 용이, (3) 사용자 투명성, (4) 가능한 한 많은 입력 데이터를 시각 검사하여 정확도 유지, (5) 후속 프로그램 개발을 촉진하는 데이터·프로그래밍 구조. 데이터베이스의 기본 단위는 개별 이온으로, 각 이온은 에너지 준위 파일, 파장+복사 천이율 파일, 충돌 강도 fitting 파일을 담은 디렉터리를 갖는다. 에너지 준위는 주로 NIST(Martin et al. 1995)에서 가져오고 부족하면 이론적 추정치로 보완한다. 모든 충돌 데이터($\Omega$, $\Upsilon$)는 시각 검사를 거쳐 Burgess & Tully (1992)에 따라 재스케일링되었다. 원소 풍부도는 사용자가 지정할 수 있고, 이온화 평형은 Arnaud & Rothenflug (1985) 사용. 현재 $\lambda > 50$ Å, $N_e \lesssim 10^{15}$ cm$^{-3}$ 영역에서 광학적으로 얇은 방출선 스펙트럼을 재현한다. 장파장 한계는 없으나 중성은 포함되지 않아 장파장에서 포괄성이 떨어진다. H–Ni 우주적 풍부 원소가 포함되며, Table 1·2에 어느 이온 단계가 있는지 나열되어 있다. IDL 루틴은 Pike & Del Zanna에 의해 SOHO/CDS 분석 소프트웨어에 통합되었다.

### Part II: §2 Emission Line Intensities (p. 150) / 방출선 세기

#### English
The single-line emissivity (power per unit volume in erg cm$^{-3}$ s$^{-1}$) at wavelength $\lambda_{i,j}$ is given by Eq. (1):

$$
\epsilon_{i,j} = N_j(X^{+m})\, A_{j,i}\, \frac{hc}{\lambda_{i,j}}
$$

where $A_{j,i}$ (s$^{-1}$) is the spontaneous emission coefficient, $N_e$ is the electron number density (cm$^{-3}$), and $N_j(X^{+m})$ is the population (cm$^{-3}$) of upper level $j$ of ion $X^{+m}$.

The upper-level population is decomposed into a chain of dimensionless factors (Eq. 2):

$$
N_j(X^{+m}) = \frac{N_j(X^{+m})}{N(X^{+m})} \cdot \frac{N(X^{+m})}{N(X)} \cdot \frac{N(X)}{N(H)} \cdot \frac{N(H)}{N_e} \cdot N_e
$$

The factors are: (i) the **level population fraction**, computed by statistical equilibrium and depending on $T_e, N_e$; (ii) the **ion fraction** $N(X^{+m})/N(X)$, predominantly a function of $T_e$, taken from Arnaud & Rothenflug (1985); (iii) the **element abundance** $N(X)/N(H)$, which varies by astrophysical source and FIP-bias; (iv) the **hydrogen-to-electron ratio** $N(H)/N_e \approx 0.83$ for fully ionised H+He plasma. The flux at Earth is then (Eq. 3):

$$
I(\lambda_{i,j}) = \frac{1}{4\pi R^2}\int_V \epsilon_{i,j}\, dV \quad [\text{erg cm}^{-2}\text{ s}^{-1}]
$$

The paper notes that in low-density plasmas, collisional excitation is faster than ionisation/recombination, so collisional excitation dominates the population of excited states. The low-lying level populations can therefore be treated separately from the ionisation/recombination problem. The upper-level population $N_j$ is obtained from the **statistical equilibrium equations** for a number of low-lying levels, including all important collisional and radiative excitation/de-excitation mechanisms (Eq. 4):

$$
N_j\!\left[N_e\sum_i C^e_{j,i} + N_p\sum_i C^p_{j,i} + \sum_{i>j} R_{j,i} + \sum_{i<j} A_{j,i}\right] = \sum_i N_i\!\left[N_e C^e_{i,j} + N_p C^p_{i,j}\right] + \sum_{i>j} N_i A_{i,j} + \sum_{i<j} N_i R_{i,j}
$$

with $C^e_{j,i}, C^p_{j,i}$ the electron/proton collisional excitation rate coefficients (cm$^3$ s$^{-1}$), $R_{j,i}$ the stimulated absorption rate coefficient (s$^{-1}$), and $A_{j,i}$ the spontaneous radiation transition probability (s$^{-1}$). LHS = total rate out of $j$, RHS = total rate into $j$; in steady state these balance.

**Important caveat for v1**: proton excitation rates ($C^p$) are NOT yet included in this release, although they are important for fine-structure transitions in highly ionised systems. An extensive review with recommended proton-excitation data was compiled by Copeland et al. (1996) and is slated for inclusion. Reviews of line-ratio diagnostics are referenced (Mason & Monsignori Fossi 1994; Dwivedi 1994; Doschek 1997; Keenan 1996).

#### 한국어
파장 $\lambda_{i,j}$에서 광학적으로 얇은 단일 방출선의 emissivity(단위 부피당 일률, erg cm$^{-3}$ s$^{-1}$)는 식 (1)로 주어진다:

$$
\epsilon_{i,j} = N_j(X^{+m})\, A_{j,i}\, \frac{hc}{\lambda_{i,j}}
$$

여기서 $A_{j,i}$(s$^{-1}$)는 자발 복사 계수, $N_e$는 전자 수밀도(cm$^{-3}$), $N_j(X^{+m})$는 이온 $X^{+m}$의 윗준위 $j$ 인구밀도(cm$^{-3}$)이다.

윗준위 인구는 무차원 인수의 곱으로 분해된다(식 2):

$$
N_j(X^{+m}) = \frac{N_j(X^{+m})}{N(X^{+m})} \cdot \frac{N(X^{+m})}{N(X)} \cdot \frac{N(X)}{N(H)} \cdot \frac{N(H)}{N_e} \cdot N_e
$$

각 인수는: (i) **준위 인구 비율**(통계 평형으로 계산, $T_e, N_e$의 함수), (ii) **이온 비율** $N(X^{+m})/N(X)$(주로 $T_e$ 함수, Arnaud & Rothenflug 1985), (iii) **원소 풍부도** $N(X)/N(H)$(천체 종류와 FIP 효과에 따라 변동), (iv) **수소/전자 비** $N(H)/N_e \approx 0.83$(완전 이온화 H+He 플라즈마). 지구에서의 flux는(식 3):

$$
I(\lambda_{i,j}) = \frac{1}{4\pi R^2}\int_V \epsilon_{i,j}\, dV
$$

논문은 저밀도 플라즈마에서는 충돌 여기 시간 척도가 이온화/재결합보다 짧아 들뜬 상태의 인구를 충돌 여기가 지배함을 지적한다. 따라서 저준위 인구를 이온화/재결합 문제와 분리해 다룰 수 있다. 윗준위 $N_j$는 모든 주요 충돌·복사 여기/소거 메커니즘을 포함한 저준위들의 **통계 평형 방정식**(식 4)으로 얻는다:

$$
N_j\!\left[N_e\sum_i C^e_{j,i} + N_p\sum_i C^p_{j,i} + \sum_{i>j} R_{j,i} + \sum_{i<j} A_{j,i}\right] = \sum_i N_i\!\left[N_e C^e_{i,j} + N_p C^p_{i,j}\right] + \sum_{i>j} N_i A_{i,j} + \sum_{i<j} N_i R_{i,j}
$$

여기서 $C^e_{j,i}, C^p_{j,i}$는 전자/양성자 충돌 여기율 계수(cm$^3$ s$^{-1}$), $R_{j,i}$는 유도 흡수율 계수(s$^{-1}$), $A_{j,i}$는 자발 복사 천이 확률(s$^{-1}$). 좌변=$j$를 떠나는 총 비율, 우변=$j$로 들어오는 총 비율. 정상상태에서 같다.

**v1의 중요한 한계**: 양성자 여기율($C^p$)은 본 release에 아직 포함되지 않았다. 그러나 고도 이온화 계의 미세구조 천이에 중요하며, Copeland et al. (1996)이 추천 데이터를 포괄적으로 정리해 곧 포함될 예정이다. 선 비율 진단에 대한 리뷰는 Mason & Monsignori Fossi (1994), Dwivedi (1994), Doschek (1997), Keenan (1996)을 참고.

### Part III: §3 Atomic Data (pp. 151–154) / 원자 데이터

#### §3.1 Energy levels / 에너지 준위

##### English
Energy levels are taken from NIST (Martin et al. 1995). Where NIST values are unavailable, theoretical energies are calculated using the **UCL SSTRUCT program** (Eissner et al. 1974). These are carefully adjusted to give the authors' "best estimates" of the predicted energies for individual ions; if no reliable energy is available, the target value used in the atomic scattering calculation is taken instead. Energy levels are usually arranged so that higher energy → higher index, but the authors found it useful to use a **fixed ordering for the LSJ levels throughout an iso-electronic sequence** to aid data evaluation, interpolation, and extrapolation along the sequence.

##### 한국어
에너지 준위는 NIST(Martin et al. 1995)에서 가져온다. NIST에 없으면 **UCL SSTRUCT 프로그램**(Eissner et al. 1974)으로 이론값을 계산하며, 저자들이 "최적 추정치"로 판단한 값으로 신중히 조정한다. 신뢰할 만한 에너지가 전혀 없으면 원자 산란 계산에서 사용된 target 값을 쓴다. 일반적으로 높은 에너지가 높은 인덱스를 갖도록 배열하지만, 데이터 평가·내삽·외삽을 돕기 위해 **iso-electronic sequence 전체에 대해 LSJ 준위에 고정 순서**를 사용한다.

#### §3.2 Radiative data / 복사 데이터

##### English
Each ion has one file containing wavelengths, weighted oscillator strengths $gf$, and Einstein A values. Wavelengths are calculated from observed energy levels; if observed energies are unavailable, the wavelength is calculated from theoretical levels and stored as a **negative value** to flag it as predicted rather than reliable. All wavelengths are vacuum wavelengths. A values are generally taken from the literature. **Transitions with branching ratio less than $10^{-5}$ have been removed from the distributed file but are retained in the master files.**

For ions where literature radiative data are unavailable, the authors used SSTRUCT to calculate theoretical energy levels and radiative data. SSTRUCT computes E1 (electric dipole), E2 (electric quadrupole), M1 (magnetic dipole), and M2 (magnetic quadrupole) transition probabilities. SSTRUCT was used in three main cases for CHIANTI:
1. **Ions with no electronic version of the radiative data** (data only on paper): SSTRUCT data was checked against the original (e.g., Fe X).
2. **Ions where the previously calculated radiative data could be improved** (e.g., Fe IX).
3. **Individual transitions for which radiative data was unavailable** (e.g., Fe XIII).

Where SSTRUCT is used, the configurations used in the calculation are documented inside the radiative data files.

##### 한국어
각 이온에는 파장, 가중 발진자 강도 $gf$, Einstein A 값을 담은 파일이 있다. 파장은 관측된 에너지 준위로부터 계산하며, 관측값이 없을 때는 이론 준위로 계산하고 **음수 값**으로 저장하여 신뢰할 수 있는 값과 구별한다. 모든 파장은 진공 파장. A 값은 보통 문헌에서 가져오며, **분지비(branching ratio)가 $10^{-5}$ 미만인 천이는 배포 파일에서 제거**하되 마스터 파일에는 보존한다.

문헌에 복사 데이터가 없으면 SSTRUCT로 이론 에너지 준위와 복사 데이터를 계산한다. SSTRUCT는 E1(전기 쌍극자), E2(전기 사극자), M1(자기 쌍극자), M2(자기 사극자) 천이 확률을 계산한다. CHIANTI에서 SSTRUCT는 세 가지 주된 경우에 사용되었다:
1. **전자 형태의 복사 데이터가 없는 이온**(논문 인쇄본에만 존재): SSTRUCT 결과를 원본과 대조 검증(예: Fe X).
2. **기존 계산을 개선할 수 있는 이온**(예: Fe IX).
3. **개별 천이의 복사 데이터가 없을 때**(예: Fe XIII).

SSTRUCT 사용 시 계산에 사용된 configuration이 복사 데이터 파일에 문서화된다.

#### §3.3 Electron collisional data / 전자 충돌 데이터

##### English
The electron collisional excitation rate coefficient (cm$^3$ s$^{-1}$) for a Maxwellian electron velocity distribution at temperature $T_e$ (K) is (Eq. 5):

$$
C^e_{i,j} = \frac{8.63 \times 10^{-6}}{T_e^{1/2}}\, \frac{\Upsilon_{i,j}(T_e)}{\omega_i}\, \exp\!\left(-\frac{E_{i,j}}{k T_e}\right)
$$

where $\omega_i$ is the statistical weight of level $i$, $E_{i,j}$ is the energy difference between levels $i$ and $j$, $k$ is the Boltzmann constant, and $\Upsilon_{i,j}$ is the thermally-averaged effective collision strength (Eq. 6):

$$
\Upsilon_{i,j}(T_e) = \int_0^{\infty} \Omega_{i,j}\, \exp\!\left(-\frac{E_j}{k T_e}\right)\, d\!\left(\frac{E_j}{k T_e}\right)
$$

$E_j$ is the energy of the scattered electron relative to the final energy state of the ion. The collision strength $\Omega$ is symmetric, dimensionless: $\Omega_{i,j} = \Omega_{j,i}$. It relates to the electron excitation cross-section by $\sigma_{i,j} = \pi a_0^2 (\Omega_{i,j}/\omega_i)/E$, where $\pi a_0^2$ is the area of the first Bohr orbit and $E$ is the incident electron energy in Rydbergs. De-excitation rates are obtained by detailed balance.

**Approximations and their accuracy**: The four main approximations used for electron-ion scattering are
- **Distorted Wave (DW)**: neglects channel coupling; valid where the scattering electron sees a central potential (more than a few times ionised). UCL DW program (Eissner & Seaton 1972) used by Nussbaumer, Flower, Mason, Bhatia and co-workers. **Generally accurate to ~25%.**
- **Coulomb Bethe (CBe)** (Burgess & Sheorey 1974): valid for high partial wave values, when the scattering electron does not penetrate the target.
- **Coulomb Born (CBO)**: takes no account of distortion; CBO with exchange (CBOX) includes exchange.
- **Close-Coupling (CC)**: solves coupled integro-differential equations; the most accurate but most expensive. The R-matrix package (Burke et al. 1971; Berrington et al. 1978) at QUB is the standard CC code. **Generally accurate to better than 10%.** Resonance structures (which can dominate forbidden and intersystem transitions) are accounted for only by CC. Even though resonances enhance excitation rates, **radiation damping** can reduce their effect for highly ionised systems — this is acknowledged but allowing for it is "not straightforward".

Relativistic effects matter for highly ionised species: for low ionisation, transformation from LS to intermediate coupling (IC) is straightforward (algebraic), but for higher ionisation the LS→IC transformation must include relativistic perturbations to the non-relativistic Hamiltonian. Scattering calculations are often done with the target in LS coupling (with one-body terms), then transformed using **Term Coupling Coefficients (TCCs)** from an IC structure calculation (Saraph 1972). A relativistic DW code (Sampson, Zhang and co-workers; Zhang & Sampson 1994) based on the multi-configuration Dirac-Fock program of Grant et al. (1980) does not calculate resonance structure or channel coupling. A Dirac-Fock R-matrix (DARC) package was developed by Norrington & Grant (1987). For the iron ions and elements with $Z \le 26$, comparisons between DARC, relativistic Breit-Pauli R-matrix, and R-matrix with TCCs generally agree.

Reviews of recent advances in laboratory determinations of electron excitation rate coefficients are given by Henry (1993) and Dunn (1992).

The semi-empirical **Van Regemorter (1962) $\bar g$ approximation** for electric dipole transitions is (Eq. 7):

$$
\Omega_{i,j} = \frac{8\pi}{\sqrt 3}\, \omega_j\, f_{i,j}\, \frac{I_H}{E_{i,j}}\, \bar g
$$

Younger & Wiese (1979) assessed it: for $\Delta n = 0$ allowed transitions in alkali-like ions, $\bar g$ is within 25% of unity. For $\Delta n \ne 0$, $\bar g$ varies from 0.05 to 0.7 — **no general approximation possible**. Sampson & Zhang (1992) re-assessed: a very poor approximation for $\Delta n \ge 1$, especially given today's accurate atomic data; they recommended abandoning it. The Van Regemorter formula is also unsuitable for forbidden and intercombination transitions, which often have narrow resonances dominating the rate. Mewe (1972) and Mewe & Gronenschild (1981) abandoned it in favour of a polynomial $A_n E^{-n}$ + $A_5 \log E$ functional form, which provides the correct $\sim \log E$ functional form at high $E$.

##### 한국어
Maxwell 전자 속도 분포(온도 $T_e$, K)에서의 전자 충돌 여기율 계수(cm$^3$ s$^{-1}$)는 식 (5)로 주어진다:

$$
C^e_{i,j} = \frac{8.63 \times 10^{-6}}{T_e^{1/2}}\, \frac{\Upsilon_{i,j}(T_e)}{\omega_i}\, \exp\!\left(-\frac{E_{i,j}}{k T_e}\right)
$$

여기서 $\omega_i$는 준위 $i$의 통계 가중치, $E_{i,j}$는 두 준위의 에너지 차, $k$는 Boltzmann 상수, $\Upsilon_{i,j}$는 열평균 유효 충돌 강도(식 6):

$$
\Upsilon_{i,j}(T_e) = \int_0^{\infty} \Omega_{i,j}\, \exp\!\left(-\frac{E_j}{k T_e}\right)\, d\!\left(\frac{E_j}{k T_e}\right)
$$

$E_j$는 이온의 최종 상태 기준 산란 전자 에너지. 충돌 강도 $\Omega$는 대칭, 무차원: $\Omega_{i,j} = \Omega_{j,i}$. 단면적과 $\sigma_{i,j} = \pi a_0^2 (\Omega_{i,j}/\omega_i)/E$로 연결되며, $\pi a_0^2$은 첫 Bohr 궤도 면적, $E$는 입사 전자 에너지(Rydberg). 소거율은 상세 평형(detailed balance)으로 얻는다.

**근사법과 정확도**: 전자-이온 산란의 주요 4가지 근사:
- **Distorted Wave (DW)**: 채널 결합 무시; 산란 전자가 중심 퍼텐셜을 보는 경우(이온화 단계가 몇 단계 이상)에서 유효. UCL DW 프로그램(Eissner & Seaton 1972)이 Nussbaumer, Flower, Mason, Bhatia 등에 의해 광범위하게 사용. **대체로 ~25% 정확도.**
- **Coulomb Bethe (CBe)** (Burgess & Sheorey 1974): 고 부분파, 산란 전자가 표적을 침투하지 않을 때 유효.
- **Coulomb Born (CBO)**: 왜곡 무시; 교환 포함 시 CBOX.
- **Close-Coupling (CC)**: 결합 적분-미분 방정식을 풀이; 가장 정확하나 가장 비싸다. QUB의 R-matrix 패키지(Burke et al. 1971; Berrington et al. 1978)가 표준. **대체로 10% 이내 정확도.** 공명(resonance) 구조는 CC에서만 처리되며, 금지·intercombination 천이를 지배할 수 있다. 공명이 여기율을 증가시키지만, 고이온화에서는 **radiation damping**이 그 효과를 줄일 수 있다 — 인정되지만 처리는 "단순치 않다".

상대론적 효과는 고이온화에서 중요: 저이온화에서는 LS→IC(중간 결합) 변환이 대수적으로 간단하나, 고이온화에서는 비상대론 Hamiltonian에 대한 상대론적 교란을 포함해야 한다. 산란 계산은 보통 LS 결합(일체 항 일부 포함) 표적으로 수행한 뒤 IC 구조 계산의 **Term Coupling Coefficient(TCC)** (Saraph 1972)로 변환한다. 상대론적 DW 코드(Sampson, Zhang; Zhang & Sampson 1994)는 Grant et al. (1980)의 MCDF 프로그램 기반으로, 공명 구조나 채널 결합은 계산하지 않는다. Norrington & Grant (1987)의 Dirac-Fock R-matrix(DARC)도 사용된다. 철 이온과 $Z \le 26$ 원소에 대해 DARC, 상대론적 Breit-Pauli R-matrix, TCC 포함 R-matrix 비교는 일반적으로 잘 일치.

실험실 전자 여기율 계수 측정의 최근 진보 리뷰는 Henry (1993)와 Dunn (1992).

전기 쌍극자 천이에 대한 반경험적 **Van Regemorter (1962) $\bar g$ 근사**(식 7):

$$
\Omega_{i,j} = \frac{8\pi}{\sqrt 3}\, \omega_j\, f_{i,j}\, \frac{I_H}{E_{i,j}}\, \bar g
$$

Younger & Wiese (1979): 알칼리류 이온 $\Delta n = 0$ 허용 천이에서 $\bar g$는 1의 25% 이내. $\Delta n \ne 0$에서는 0.05–0.7로 변동 → **일반 근사 불가**. Sampson & Zhang (1992): $\Delta n \ge 1$에 매우 부정확하며, 정확한 원자 데이터가 풍부한 오늘날 폐기 권고. 또한 금지·intercombination 천이에 부적합(좁은 공명 지배). Mewe (1972)와 Mewe & Gronenschild (1981)는 이를 폐기하고 다항식 $A_n E^{-n}$ + $A_5 \log E$ 형식을 도입(고에너지에서 올바른 $\sim \log E$ 형태 제공).

#### §3.4 Assessment of electron excitation data / 전자 여기 데이터 평가

##### English
Considerable effort has been put into assessing published atomic data. A workshop at Abingdon in March 1992, sponsored by the SOHO CDS and SUMER projects, produced reviews on each isoelectronic sequence from H-like to Ne-like, plus several Si and S ions and Fe I–Fe XXVI, published in a single volume of Atomic Data and Nuclear Data Tables (ed. Lang 1994). Together with the Itikawa (1991) bibliography and Pradhan & Gallagher (1992) compilation, these provide a comprehensive survey. New $\Omega$ and $\Upsilon$ calculations continue to appear.

The **iron ions** pose a difficult challenge for electron scattering calculations because of the complexity required to represent the $n = 3$ configurations accurately. **Mason (1994)** assessed available publications for the coronal ions Fe IX–Fe XIV. She found that existing electron excitation data are severely limited in accuracy. Subsequently new data for **Fe X and Fe XI have been calculated by Bhatia & Doschek (1995a, 1996)**, which are used in CHIANTI v1. Serious inadequacies remain in the current atomic data; very accurate work using CC with relativistic targets is underway as part of the **Iron Project (Hummer et al. 1993)** and is intended for inclusion in future CHIANTI releases.

**Burgess & Tully (1992) scaling — the heart of CHIANTI's data philosophy**: Electron excitation data are provided by authors in many different formats; some give $\Omega$ as a function of energy, others give $\Upsilon$ as a function of temperature. The task of CHIANTI was to gather these together and present in a compressed, easily accessible format. Burgess & Tully (1992) provide a method based on scaling the incident electron energy and the collision strength so they both fall within a finite range. Each transition type (allowed E1 dipole, forbidden, intercombination) is treated differently. For dipole (allowed) transitions:

$$
x = 1 - \frac{\ln(C)}{\ln(X - 1 + C)}, \qquad y = \Omega \ln(X - 1 + e)
$$

where $X = E/E_{\text{th}}$ is the colliding electron energy in threshold units, $C$ is an adjustable parameter chosen to suit the case, and $x \in [0, 1]$ (threshold to infinite energy). The high energy limit for $\Omega$ and high temperature limit for $\Upsilon$ is from the Coulomb-Bethe approximation: $4 \omega_i f_{i,j}/E_{i,j}$, where $\omega_i f_{i,j}$ is the weighted dipole oscillator strength and $E_{i,j}$ is the energy difference in Rydbergs.

Burgess & Tully's interactive graphical programs were written in BBC BASIC (not portable; PC emulator only). The CHIANTI authors developed IDL routines based on the same concept and methods, called **`BURLY`** (from BURgess..tulLY). Scaled values can often be approximated by a straight line, making extrapolation to threshold and infinite energy straightforward. The collision strength over the full energy range is needed to determine the integration over a Maxwellian, carried out with a **Gauss-Laguerre method**. Collision strengths are finally expressed as a **5-point spline fit** to the scaled collision strengths. Burgess et al. (1997) developed a method for obtaining infinite energy/temperature values for the collisional data from the Coulomb-Born approximation; this is hoped to be incorporated into future CHIANTI fitting programs. The authors emphasise that **all collisional data, each ion and transition, should be visually examined**.

Several problems with $\Omega$ and $\Upsilon$ calculations were uncovered. The version of UCL DW used for most of the DW calculations required that the incident electron energy be higher than the energy of the highest target level — so DW values of $\Omega$ at threshold (which often dominates the average at low $T_e$) had to be extrapolated. Another issue: insufficient partial waves due to computing cost cause $\Omega$ to NOT approach the correct Bethe high-energy limit for allowed transitions; this happens with both DW and CC. Examples have been uncovered and dealt with by truncating published values at some energy/temperature and interpolating to the high-energy limit. Care must be taken that the high partial wave contribution is accurately accounted for in both dipole and non-dipole transitions; the BT scaling allows visual diagnosis of whether scattering calculations tend to their correct high-energy/temperature limits. Burgess et al. (1996) use the Coulomb-Born approximation to determine high-energy limits for non-dipole transitions.

##### 한국어
공개된 원자 데이터 평가에 상당한 노력이 들었다. SOHO CDS·SUMER 프로젝트 후원으로 1992년 3월 Abingdon에서 워크숍이 열렸고, H-like부터 Ne-like까지 각 isoelectronic sequence의 리뷰, Si·S 이온 다수, Fe I–Fe XXVI 리뷰가 Atomic Data and Nuclear Data Tables 한 권(ed. Lang 1994)으로 출간되었다. Itikawa (1991) 참고문헌 목록과 Pradhan & Gallagher (1992) 편찬과 함께 종합적 조사가 가능. 새로운 $\Omega$, $\Upsilon$ 계산은 계속 등장하고 있다.

**철 이온**은 $n = 3$ configuration의 복잡성 때문에 전자 산란 계산에 매우 어렵다. **Mason (1994)**이 코로나 이온 Fe IX–Fe XIV의 출간 데이터를 평가했고, 기존 전자 여기 데이터의 정확도가 심각히 제한적임을 발견했다. 이후 **Fe X와 Fe XI에 대한 새로운 데이터가 Bhatia & Doschek (1995a, 1996)**에 의해 계산되어 CHIANTI v1에 포함되었다. 현재 원자 데이터에는 심각한 부족이 남아 있으며, 상대론적 표적을 사용한 매우 정확한 CC 작업이 **Iron Project (Hummer et al. 1993)**의 일환으로 진행 중이며 향후 CHIANTI 릴리스에 포함될 예정.

**Burgess & Tully (1992) 스케일링 — CHIANTI 데이터 철학의 심장**: 전자 여기 데이터는 저자마다 형식이 달라서, 어떤 이는 $\Omega$를 에너지 함수로, 어떤 이는 $\Upsilon$를 온도 함수로 제공한다. CHIANTI의 임무는 이를 수집하여 압축·접근 용이한 형태로 제시하는 것이었다. Burgess & Tully (1992)는 입사 전자 에너지와 충돌 강도를 모두 유한 범위에 매핑하는 스케일링 방법을 제공한다. 각 천이 유형(허용 E1 쌍극자, 금지, intercombination)은 다르게 처리된다. 쌍극자(허용) 천이의 경우:

$$
x = 1 - \frac{\ln(C)}{\ln(X - 1 + C)}, \qquad y = \Omega \ln(X - 1 + e)
$$

여기서 $X = E/E_{\text{th}}$는 threshold 단위 입사 전자 에너지, $C$는 사례에 맞게 조정되는 파라미터, $x \in [0, 1]$(threshold→무한대 에너지). $\Omega$의 고에너지 극한 및 $\Upsilon$의 고온 극한은 Coulomb-Bethe 근사에서: $4 \omega_i f_{i,j}/E_{i,j}$, $\omega_i f_{i,j}$는 가중 쌍극자 발진자 강도, $E_{i,j}$는 Rydberg 단위 에너지 차.

Burgess & Tully의 대화형 그래픽 프로그램은 BBC BASIC으로 작성되어 휴대성이 낮았다(PC 에뮬레이터 전용). CHIANTI 저자들은 동일 개념·방법을 따르는 IDL 루틴 **`BURLY`**(BURgess..tulLY)을 개발. 스케일링된 값은 종종 직선으로 근사 가능하여 threshold와 무한 에너지로의 외삽이 쉽다. 충돌 강도는 전 에너지 영역에서 필요하며, **Gauss-Laguerre 방법**으로 Maxwellian 적분을 수행한다. 최종적으로 충돌 강도는 스케일링된 값에 대한 **5-point spline fit**으로 표현된다. Burgess et al. (1997)이 Coulomb-Born에서 무한 에너지/온도 값을 얻는 방법을 개발했고, 향후 CHIANTI fitting 프로그램에 포함될 예정. 저자들은 **모든 충돌 데이터, 모든 이온과 천이를 시각 검사해야 한다**고 강조.

$\Omega$, $\Upsilon$ 계산의 여러 문제가 드러났다. 대부분의 DW 계산에 사용된 UCL DW 버전은 입사 전자 에너지가 최고 표적 준위 에너지보다 높아야 했기에 — threshold에서의 DW $\Omega$ 값(저온 평균을 종종 지배)을 외삽해야 했다. 또 다른 문제: 계산 비용 때문에 부분파가 부족하면 $\Omega$가 허용 천이에 대한 올바른 Bethe 고에너지 극한에 접근하지 못한다(DW와 CC 모두). 예시들이 발견되어, 출간 값을 어떤 에너지/온도에서 절단하고 고에너지 극한으로 내삽하는 방식으로 처리했다. 쌍극자·비쌍극자 천이 모두에서 고 부분파 기여를 정확히 처리해야 하며, BT 스케일링은 산란 계산이 올바른 고에너지/고온 극한으로 향하는지 시각적으로 진단해 준다. Burgess et al. (1996)은 비쌍극자 천이의 고에너지 극한을 Coulomb-Born으로 결정한다.

### Part IV: §4 The CHIANTI database (pp. 154+) / CHIANTI 데이터베이스

#### English
The goal is to reproduce emission spectrum at $\lambda \gtrsim 1$ Å. Initial v1 is essentially complete for $\lambda > 50$ Å, motivated by the immediate analysis needs of EUVE, Hubble, and SOHO. **Ions in the He-like sequence are NOT included** in v1, and **He II is the only H-like ion included** — these are slated for the next release. Many transitions at $\lambda < 50$ Å are also in the database but are not complete at shorter wavelengths.

§4.1 **Hydrogen sequence: He II** — 25 fine-structure levels of 1s, 2$l$, 3$l$, 4$l$, 5$l$ configurations. Energies from Kelly (1987). Radiative constants ($gf$, $A$) from Wiese et al. (1966) for dipole transitions. For 1s $^2$S$_{1/2}$ – 2s $^2$S$_{1/2}$ magnetic dipole and two-photon E1 transitions, A values from Parpia & Johnson (1972). R-matrix collision strengths for $n = 1$–5 levels from Aggarwal et al. (1991b), combined in the threshold region with the close-coupling values of Unnikrishnan et al. (1991) at 4.4–14.71 Ryd.

§4.2 **Lithium isoelectronic sequence** — atomic structure simple, no intersystem transitions, no metastable levels (so no density-sensitive ratios but interpretation of emission measure simplified). 2s–2p transitions are strong, providing diagnostics from C IV (transition region) through Mg X (quiet corona) to Fe XXIV (flares). Excitation data accurate (McWhirter 1994). For C IV, N V, O VI, Mg X, Al XI, Si XII, S XIV, Ar XVI, Ca XVIII, Fe XXIV, Ni XXVI, configurations 2s$^2$2s, 2s$^2$2p, 2s$^2$3s, 2s$^2$3p, 2s$^2$3d are included. Zhang et al. (1990) provide relativistic DW collision strengths. Lab measurements of C IV 2s–2p excitation rates by Savin et al. (1995) agree with the 9-state CC calculations of Burke (1992). The **Mg X 1s$^2$2s $^2$S$_{1/2}$ – 1s$^2$3p $^2$P$_{3/2}$** dipole transition is shown in **Fig. 1** as the example of Burgess–Tully scaling: upper-left = original $\Omega(E)$, upper-right = scaled $\Omega(x)$, lower-left = $\Upsilon(T)$, lower-right = scaled $\Upsilon(x)$ — the spline fit deviation averages **0.2%**.

§4.3 **Beryllium isoelectronic sequence** — has metastable levels (2s2p $^3$P$_{0,1,2}$ in the first excited config). Useful electron density diagnostics for the solar transition region since early Skylab work (Gabriel & Jordan 1972; Dupree et al. 1976; Dere & Mason 1981). C III diagnostics extensively explored (Doschek 1997). Channel coupling and resonance effects of $n = 3$ states on $n = 2$ to $n = 2$ transitions important — R-matrix recommended. **§4.3.1 C III**: 6 configurations (2s$^2$, 2s2p, 2p$^2$, 2s3s, 2s3p, 2s3d), 20 fine-structure levels. Energies from NIST. $A$ values from Bhatia & Kastner (1992). Collision data for 2s$^2$, 2s2p and 2p$^2$ levels from R-matrix calculations of Berrington et al. (1985); $n = 2$–3 and $n = 3$–3 from Berrington et al. (1989). LS coupling scheme; collision strengths scaled to fine-structure levels using statistical weights. **§4.3.2 N IV** — 6 configurations, 20 fine-structure levels. NIST + Ramsbottom et al. (1994). For some transitions effective collision strengths below $T_e = 10^{3.6}$ K omitted because the 5-point spline cannot reproduce the complex $T_e$ dependence — but $T_{\max}$ for N IV $\approx 10^{5.5}$ K so this has no consequence. **§4.4 O V**: configurations 2s$^2$, 2s2p, 2p$^2$, 2s3s, 2s3p, 2s3d. Zhang & Sampson (1992) and updates by Kato (1996) used.

§4.4.1 **Ne VII, Mg IX, Si XI** — 8 configurations, 46 fine-structure levels (some unidentified). Zhang & Sampson (1992) E1 oscillator strengths and A values; Muhlethaler & Nussbaumer (1976) for forbidden/intercombination; relativistic DW collision strengths for $n = 2$ from Zhang & Sampson (1992); $n = 2$ to $n = 3$ probabilities and Coulomb-Born-exchange from Sampson et al. (1984). R-matrix data for Ne VII to be assessed in next release. §4.4.2 **Al X, S XIII, Ar XV, Ca XVII, Ni XXV** — only 2s$^2$, 2s2p, 2p$^2$ (10 fine-structure levels). For S XIII and higher, $n = 3$ transitions are at $\lambda < 50$ Å so $n = 3$ levels not yet included (computational cost in level-population solver). §4.4.3 **Fe XXIII** — 9 configurations, 30 fine-structure levels. NIST + Bhatia + Zhang & Sampson (1992).

§4.5 **Boron isoelectronic sequence** — diagnostic for solar atmosphere (Vernazza & Mason 1978). For low ionisations, transitions from metastable $^4$P$_J$ in 2s2p$^2$ at ~1400 Å are primary diagnostic for transition region electron pressure (Dere et al. 1982). For coronal ions (Mg VIII, Si X, Ar XIV, Ca XVI), relative population change in ground term 2s$^2$2p $^2$P$_J$ is reflected in UV transition intensities from 2s2p$^2$. Used to determine electron densities in solar flares (Dere et al. 1979). X-ray and XUV lines from Fe XXII arise from transitions between excited 2s2p$^2$, 2s$^2$3$l$ and ground 2s$^2$4$l$ configurations (Mason & Storey 1980), recorded in flares and other astrophysical sources (Dupree et al. 1993; Monsignori et al. 1994b). Useful for $N_e$ if $N_e > 10^{12}$ cm$^{-3}$. Sampson et al. (1994) reviewed B-like collisional data; CB exchange with relativistic corrections, generally agree with IC DW (UCL). Now superseded by R-matrix and fully relativistic DW. **§4.5.1 C II** — configurations 2s$^2$2p, 2s2p$^2$, 2p$^3$, 2s$^2$3s, 2s$^2$3p; energies complete from Kelly (1987). $f$ and $A$ from Dankwort & Trefftz (1978), Nussbaumer & Storey (1981), Lennon et al. (1985), Wiese & Fuhr (1995). Fang et al. (1993) lab radiative transition probabilities. R-matrix collision strengths from Blum & Pradhan (1992). **§4.5.2 N III** — 20 fine-structure levels of 2s$^2$2p, 2s2p$^2$, 2p$^3$, 2s$^2$3s, 2s$^2$3p, 2s$^2$3d. Stafford et al. (1993) oscillator strengths; Stafford et al. (1994) R-matrix Maxwellian-averaged collision strengths. Average deviation of spline fit ~0.5%. **§4.5.3 O IV, Ne VI, Mg VIII, Al IX, Si X, S XII, Ar XIV, Ca XVI, Fe XXII** — 125 fine-structure levels of 2s$^2$2p, 2s2p$^2$, 2p$^3$ and 2$l$ 2$l'$ 3$l''$. NIST (Martin et al. 1995) + Edlén (1981). For Si X, Zhang's (1995) confirmation that the $^2$S$_{1/2}$ and $^2$P$_{1/2}$ labels published by Dankwort & Trefftz (1978) need to be exchanged because of a level crossing somewhere between Z = 20 (Ca) and Z = 26 (Fe). Zhang et al. (1994) R-matrix Maxwellian-averaged collision strengths between 15 fine-structure levels in 2s$^2$2p, 2s2p$^2$, 2p$^3$. For high temperatures (10$^7$ K), 10% agreement with RDW (Zhang & Sampson 1994) and R-matrix (BP+TCC) (Zhang & Pradhan 1994). **§4.5.4 Ni XXIV** — same level scheme; observed energies from NIST and Edlén (1981).

§4.6 **Carbon isoelectronic sequence** — lines studied in transition region and corona (continued in further pages, beyond p. 158).

#### 한국어
목표는 $\lambda \gtrsim 1$ Å에서 방출 스펙트럼을 재현하는 것. 초기 v1은 EUVE, Hubble, SOHO의 즉각적 분석 수요에 맞춰 $\lambda > 50$ Å에서 사실상 완전. **He-like sequence 이온은 v1 미포함**, **H-like은 He II만 포함** — 다음 릴리스에서 추가 예정. $\lambda < 50$ Å 천이도 일부 포함되나 단파장에서는 완전치 않다.

§4.1 **수소 sequence: He II** — 1s, 2$l$, 3$l$, 4$l$, 5$l$의 25 fine-structure 준위. 에너지는 Kelly (1987). $gf$, $A$ 값은 쌍극자 천이에 대해 Wiese et al. (1966). 1s $^2$S$_{1/2}$ – 2s $^2$S$_{1/2}$ M1과 두 광자 E1은 Parpia & Johnson (1972). $n = 1$–5 R-matrix 충돌 강도는 Aggarwal et al. (1991b), threshold 영역에서는 4.4–14.71 Ryd의 Unnikrishnan et al. (1991) close-coupling 값과 결합.

§4.2 **리튬 isoelectronic sequence** — 원자 구조 단순, intersystem 천이 없음, metastable 준위 없음(밀도 민감 비율 없으나 방출 측도 해석 단순화). 2s–2p 천이가 강해 C IV(천이영역)부터 Mg X(조용한 코로나), Fe XXIV(플레어)까지 진단 제공. 여기 데이터 정확(McWhirter 1994). C IV, N V, O VI, Mg X, Al XI, Si XII, S XIV, Ar XVI, Ca XVIII, Fe XXIV, Ni XXVI에 대해 2s$^2$2s, 2s$^2$2p, 2s$^2$3s, 2s$^2$3p, 2s$^2$3d configuration 포함. Zhang et al. (1990) 상대론적 DW 충돌 강도. C IV 2s–2p 여기율에 대한 Savin et al. (1995) 실험은 Burke (1992)의 9-state CC와 일치. **Mg X 1s$^2$2s $^2$S$_{1/2}$ – 1s$^2$3p $^2$P$_{3/2}$** 쌍극자 천이가 **Fig. 1**의 BT 스케일링 예시: 좌상=원본 $\Omega(E)$, 우상=스케일링 $\Omega(x)$, 좌하=$\Upsilon(T)$, 우하=스케일링 $\Upsilon(x)$. spline fit 평균 편차 **0.2%**.

§4.3 **베릴륨 isoelectronic sequence** — metastable 준위(첫 들뜬 configuration의 2s2p $^3$P$_{0,1,2}$) 보유. 초기 Skylab 시대부터 태양 천이영역의 유용한 전자 밀도 진단 제공(Gabriel & Jordan 1972; Dupree et al. 1976; Dere & Mason 1981). C III 진단 광범위 탐구(Doschek 1997). $n = 3$ 상태의 채널 결합·공명 효과가 $n = 2$ 천이에 중요 → R-matrix 권장. **§4.3.1 C III**: 6 configurations (2s$^2$, 2s2p, 2p$^2$, 2s3s, 2s3p, 2s3d), 20 fine-structure 준위. 에너지는 NIST. $A$ 값은 Bhatia & Kastner (1992). 2s$^2$, 2s2p, 2p$^2$ 충돌 데이터는 Berrington et al. (1985) R-matrix; $n = 2$–3, $n = 3$–3은 Berrington et al. (1989). LS 결합; 통계 가중치로 fine-structure로 스케일링. **§4.3.2 N IV** — 6 configurations, 20 fine-structure 준위. NIST + Ramsbottom et al. (1994). 일부 천이에 대해 $T_e < 10^{3.6}$ K 유효 충돌 강도 생략(5-point spline이 복잡한 $T_e$ 의존성 표현 불가) — N IV $T_{\max} \approx 10^{5.5}$ K이므로 문제 없음. **§4.4 O V**: 2s$^2$, 2s2p, 2p$^2$, 2s3s, 2s3p, 2s3d. Zhang & Sampson (1992)과 Kato (1996) 갱신.

§4.4.1 **Ne VII, Mg IX, Si XI** — 8 configurations, 46 fine-structure(일부 미동정). Zhang & Sampson (1992) E1 발진자 강도·$A$; Muhlethaler & Nussbaumer (1976) 금지/intercombination; $n = 2$ 상대론적 DW from Zhang & Sampson; $n = 2$–3 확률과 Coulomb-Born-exchange는 Sampson et al. (1984). Ne VII R-matrix 데이터는 다음 릴리스 평가. §4.4.2 **Al X, S XIII, Ar XV, Ca XVII, Ni XXV** — 2s$^2$, 2s2p, 2p$^2$만(10 fine-structure). S XIII 이상에서는 $n = 3$ 천이가 $\lambda < 50$ Å이므로 $n = 3$ 미포함(준위 인구 풀이 비용). §4.4.3 **Fe XXIII** — 9 configurations, 30 fine-structure. NIST + Bhatia + Zhang & Sampson (1992).

§4.5 **붕소 isoelectronic sequence** — 태양 대기 진단(Vernazza & Mason 1978). 저이온화에서 2s2p$^2$의 metastable $^4$P$_J$로부터의 ~1400 Å 천이가 천이영역 전자 압력 측정(Dere et al. 1982). 코로나 이온(Mg VIII, Si X, Ar XIV, Ca XVI)에서는 바닥 항 2s$^2$2p $^2$P$_J$ 인구 변화가 2s2p$^2$로부터의 UV 천이 세기에 반영. 태양 플레어 전자 밀도 결정(Dere et al. 1979). Fe XXII의 X-ray·XUV 라인은 들뜬 2s2p$^2$, 2s$^2$3$l$과 바닥 2s$^2$4$l$ 사이 천이(Mason & Storey 1980). $N_e > 10^{12}$ cm$^{-3}$에서 유용. Sampson et al. (1994)이 B-like 충돌 데이터 리뷰; 상대론적 보정 포함 CB 교환, IC DW(UCL)와 일치. 현재 R-matrix와 완전 상대론적 DW로 대체. **§4.5.1 C II**, **§4.5.2 N III** (spline fit 평균 편차 ~0.5%), **§4.5.3 O IV, Ne VI, Mg VIII, Al IX, Si X, S XII, Ar XIV, Ca XVI, Fe XXII** (NIST + Edlén 1981; Si X 라벨 이슈), **§4.5.4 Ni XXIV** 순으로 상세.

§4.6 **탄소 isoelectronic sequence** — 천이영역과 코로나 스펙트럼에서 연구(p. 158 이후 계속).

### Part V: Worked numerical example — Fe IX 171.07 Å contribution function / 작업 예제 — Fe IX 171.07 Å 기여 함수

#### English
The Fe IX 3p$^6$ $^1$S$_0$ – 3p$^5$3d $^1$P$_1$ transition at **171.07 Å** is the dominant contributor near the EUI/FSI 17.4 nm passband used by Abbo et al. (2025, paper #61). Tracing how CHIANTI v1 produces $G(T_e, n_e)$ for this line:

1. **Energy levels** (`fe_9.elvlc`): Fe IX is Ar-like (18 electrons). The ground configuration 3p$^6$ has level 1 ($^1$S$_0$, $\omega_1 = 1$). The excited 3p$^5$3d configuration includes $^1$P$_1$ at $\sim 5.85 \times 10^5$ cm$^{-1}$ ($\omega = 3$), giving $\lambda_{1,j} = 10^8/5.85 \times 10^5 \approx 1709$ Å$^{-1}$ for the wavenumber → 171.07 Å vacuum (Storey & Zeippen for nearby ions, or Edlén-derived values stored in `.elvlc`).

2. **A value** (`fe_9.wgfa`): The $^1$P$_1$ → $^1$S$_0$ allowed E1 transition has $A_{j,1} \sim 7 \times 10^{10}$ s$^{-1}$ (taken from SSTRUCT improvement on previous calculations, since Fe IX was one of the explicit cases where SSTRUCT was used to improve existing data — see §3.2). Photon energy $hc/\lambda = (1240\,\mathrm{eV\,nm})/17.107\,\mathrm{nm} \approx 72.5$ eV $= 1.16 \times 10^{-10}$ erg.

3. **Collision data** (`fe_9.splups`): Bhatia & Doschek (1995a, 1996) calculated new collision data for Fe X, Fe XI, used in v1; Fe IX uses earlier published data. The effective collision strength $\Upsilon_{1,j}(T_e)$ for the resonance line is tabulated as 5-point spline coefficients in scaled $x \in [0,1]$. At $T_e = 10^{6.0}$ K (Fe IX $T_{\max}$), $\Upsilon \sim 1$–2 for this allowed transition.

4. **Excitation rate coefficient** from Eq. 5: with $\omega_i = 1$, $E_{ij}/(kT_e)$ at $T_e = 10^6$ K: $E_{ij} = 72.5$ eV $\Rightarrow 8.4 \times 10^5$ K $\Rightarrow E_{ij}/kT_e = 0.84$ → $\exp(-0.84) \approx 0.43$.

$$
C^e_{1,j} \approx \frac{8.63 \times 10^{-6}}{(10^6)^{1/2}}\, \frac{1.5}{1}\, \times 0.43 \approx 5.6 \times 10^{-9}\,\mathrm{cm}^3\,\mathrm{s}^{-1}
$$

5. **Coronal-limit upper-level fraction**: at low density (corona), the $^1$P$_1$ excited population is set by the balance $N_1 N_e C^e_{1,j} = N_j A_{j,1}$ (assuming pure radiative decay back to the ground). Therefore
   $$
   \frac{N_j}{N_1} \approx \frac{N_e C^e_{1,j}}{A_{j,1}} \approx \frac{10^9 \times 5.6 \times 10^{-9}}{7 \times 10^{10}} \approx 8 \times 10^{-11}
   $$
   for $N_e = 10^9$ cm$^{-3}$ — confirming the corona's near-ground-state population.

6. **Ion fraction** $N(\mathrm{Fe~IX})/N(\mathrm{Fe})$ from Arnaud & Rothenflug (1985) peaks near $T_e \approx 10^{5.9}$ K with value $\sim 0.4$.

7. **Element abundance**: with coronal Fe abundance $N(\mathrm{Fe})/N(\mathrm{H}) \approx 4 \times 10^{-5}$ and $N(\mathrm{H})/N_e \approx 0.83$:

8. **Contribution function**:

$$
G(T_e, n_e) = \frac{hc}{\lambda} \cdot \frac{A_{j,1}\, N_j/N(\mathrm{Fe~IX})}{N_e} \cdot \frac{N(\mathrm{Fe~IX})}{N(\mathrm{Fe})} \cdot \frac{N(\mathrm{Fe})}{N(\mathrm{H})} \cdot \frac{N(\mathrm{H})}{N_e}
$$

Numerically near $T_e \sim 10^{5.9}$ K, $G$ for Fe IX 171.07 Å peaks near $\sim 10^{-25}$ erg cm$^3$ s$^{-1}$ sr$^{-1}$ (typical CHIANTI textbook value). When integrated against the FSI 17.4 nm bandpass throughput (Abbo et al. 2025, paper #61), this is the dominant contribution to the response function $R(T_e)$ near $\log T \approx 5.9$ — exactly the temperature regime where FSI traces the quiet corona.

#### 한국어
Fe IX 3p$^6$ $^1$S$_0$ – 3p$^5$3d $^1$P$_1$ 천이(**171.07 Å**)는 Abbo et al. (2025, paper #61)이 사용하는 EUI/FSI 17.4 nm 통과대역 근처의 지배적 기여자이다. CHIANTI v1이 이 라인의 $G(T_e, n_e)$를 어떻게 만드는지 추적하면:

1. **에너지 준위** (`fe_9.elvlc`): Fe IX는 Ar-like(18 전자). 바닥 configuration 3p$^6$의 준위 1($^1$S$_0$, $\omega_1 = 1$). 들뜬 3p$^5$3d configuration의 $^1$P$_1$이 $\sim 5.85 \times 10^5$ cm$^{-1}$($\omega = 3$) → 진공 파장 171.07 Å(`.elvlc`에 저장된 NIST/Edlén 유래 값).

2. **A 값** (`fe_9.wgfa`): $^1$P$_1$ → $^1$S$_0$ 허용 E1 천이의 $A_{j,1} \sim 7 \times 10^{10}$ s$^{-1}$. Fe IX는 §3.2에서 SSTRUCT로 기존 데이터를 개선한 명시적 사례. 광자 에너지 $hc/\lambda = 1240\,\mathrm{eV\,nm}/17.107\,\mathrm{nm} \approx 72.5$ eV = $1.16 \times 10^{-10}$ erg.

3. **충돌 데이터** (`fe_9.splups`): Bhatia & Doschek (1995a, 1996)가 Fe X, Fe XI 새 데이터 계산(v1 사용); Fe IX는 이전 출간 데이터. 공명선의 유효 충돌 강도 $\Upsilon_{1,j}(T_e)$는 스케일링된 $x \in [0,1]$에 대한 5-point spline 계수로 표 형태 저장. $T_e = 10^{6.0}$ K(Fe IX $T_{\max}$)에서 이 허용 천이의 $\Upsilon \sim 1$–2.

4. **여기율 계수**(식 5): $\omega_i = 1$, $E_{ij}/kT_e = 0.84$ → $\exp(-0.84) \approx 0.43$:
$$
C^e_{1,j} \approx \frac{8.63 \times 10^{-6}}{(10^6)^{1/2}}\, \frac{1.5}{1}\, \times 0.43 \approx 5.6 \times 10^{-9}\,\mathrm{cm}^3\,\mathrm{s}^{-1}
$$

5. **코로나 극한 윗준위 비율**: 저밀도(코로나)에서 $^1$P$_1$ 인구는 $N_1 N_e C^e_{1,j} = N_j A_{j,1}$ 균형으로 결정. 따라서
$$
\frac{N_j}{N_1} \approx \frac{N_e C^e_{1,j}}{A_{j,1}} \approx \frac{10^9 \times 5.6 \times 10^{-9}}{7 \times 10^{10}} \approx 8 \times 10^{-11}
$$
($N_e = 10^9$ cm$^{-3}$) — 코로나가 거의 바닥상태 인구임을 확인.

6. **이온 비율** $N(\mathrm{Fe~IX})/N(\mathrm{Fe})$ from Arnaud & Rothenflug (1985)는 $T_e \approx 10^{5.9}$ K에서 ~0.4로 정점.

7. **원소 풍부도**: 코로나 Fe 풍부도 $N(\mathrm{Fe})/N(\mathrm{H}) \approx 4 \times 10^{-5}$, $N(\mathrm{H})/N_e \approx 0.83$.

8. **기여 함수**:

$$
G(T_e, n_e) = \frac{hc}{\lambda} \cdot \frac{A_{j,1}\, N_j/N(\mathrm{Fe~IX})}{N_e} \cdot \frac{N(\mathrm{Fe~IX})}{N(\mathrm{Fe})} \cdot \frac{N(\mathrm{Fe})}{N(\mathrm{H})} \cdot \frac{N(\mathrm{H})}{N_e}
$$

수치적으로 $T_e \sim 10^{5.9}$ K 근처에서 Fe IX 171.07 Å의 $G$는 $\sim 10^{-25}$ erg cm$^3$ s$^{-1}$ sr$^{-1}$ 정도로 정점(전형적 CHIANTI 교과서 값). FSI 17.4 nm 통과대역 throughput(Abbo et al. 2025, paper #61)에 대해 적분하면, 이는 $\log T \approx 5.9$ 부근 $R(T_e)$의 지배적 기여 — 정확히 FSI가 조용한 코로나를 추적하는 온도 영역.

---

## 3. Key Takeaways / 핵심 시사점

1. **Single transparent ASCII format unifies a fragmented community.** / **단일 투명 ASCII 형식이 분산된 커뮤니티를 통합한다.**
   English: Each ion lives in its own directory with three small ASCII files (`.elvlc`, `.wgfa`, `.splups`); any user can open them in a text editor, validate them, or contribute corrections. This was a novel break from black-box spectral codes (Mewe, Raymond–Smith) whose data were buried inside FORTRAN routines.
   한국어: 각 이온은 세 개의 작은 ASCII 파일(`.elvlc`, `.wgfa`, `.splups`)을 담은 디렉터리를 갖는다. 누구나 텍스트 에디터로 열어 검증·수정 기여 가능. FORTRAN 루틴 안에 데이터가 숨어 있던 블랙박스 분광 코드(Mewe, Raymond–Smith)와의 결별이었다.

2. **Burgess–Tully scaling makes "visual quality control" of every collision dataset feasible.** / **Burgess–Tully 스케일링이 모든 충돌 데이터의 시각적 품질 관리를 가능하게 한다.**
   English: By mapping $E \in [E_{\text{th}}, \infty)$ to $x \in [0, 1]$ and pinning the asymptote to the Bethe limit, BT scaling compresses any collision strength into a finite plot where deviations from physical behaviour are obvious. The `BURLY` IDL companion makes this an interactive, day-to-day workflow — a 5-point spline reproduces $\Upsilon$ to better than 1% (0.2% for Mg X, 0.5% for N III).
   한국어: $E \in [E_{\text{th}}, \infty)$를 $x \in [0, 1]$로 매핑하고 점근선을 Bethe 극한에 고정함으로써, BT 스케일링은 어떤 충돌 강도도 유한한 plot으로 압축하여 물리적 거동에서의 이탈을 명백히 만든다. `BURLY` IDL 동봉 패키지가 이를 일상적 대화형 작업으로 가능하게 한다 — 5-point spline이 $\Upsilon$를 1% 미만으로 재현(Mg X 0.2%, N III 0.5%).

3. **Modular factorisation of $N_j(X^{+m})$ separates atomic physics from astrophysics.** / **$N_j(X^{+m})$의 모듈식 인수분해가 원자물리와 천체물리를 분리한다.**
   English: Eq. (2) splits the upper-level density into level population (atomic physics: SE solver), ion fraction (atomic physics: ionisation balance), abundance (astrophysics: chemical composition), and electron density (astrophysics: structure). Each layer is replaceable independently — the user can swap in different abundance sets or ionisation balance calculations without touching the database.
   한국어: 식 (2)는 윗준위 밀도를 준위 인구(원자물리: SE 풀이), 이온 비율(원자물리: 이온화 평형), 풍부도(천체물리: 화학 조성), 전자 밀도(천체물리: 구조)로 나눈다. 각 층은 독립적으로 교체 가능 — 사용자는 데이터베이스를 건드리지 않고 다른 풍부도 집합이나 이온화 평형 계산을 끼워 넣을 수 있다.

4. **R-matrix / Iron Project supersedes Van Regemorter.** / **R-matrix / Iron Project가 Van Regemorter를 대체한다.**
   English: The 1962 $\bar g$ approximation that powered the previous generation of spectral codes is shown to be unreliable beyond $\Delta n = 0$ alkali-like cases (Younger & Wiese 1979 found 0.05 ≤ $\bar g$ ≤ 0.7 for $\Delta n \ne 0$). CHIANTI replaces it with R-matrix (~10% accuracy) and DW (~25%) wherever possible; the Iron Project's CC + relativistic targets is the future roadmap.
   한국어: 이전 세대 분광 코드의 토대였던 1962년 $\bar g$ 근사는 $\Delta n = 0$ 알칼리류를 벗어나면 신뢰할 수 없다(Younger & Wiese 1979: $\Delta n \ne 0$에 대해 $\bar g$가 0.05–0.7로 변동). CHIANTI는 가능한 한 R-matrix(~10% 정확)와 DW(~25%)로 대체하며, Iron Project의 상대론적 표적 CC가 미래 로드맵.

5. **Iron ions get explicit special attention because of their dominance in the corona.** / **철 이온은 코로나 지배성 때문에 명시적 특별 주목을 받는다.**
   English: Fe IX–XIV are the main coronal diagnostics in EUV; Mason (1994) explicitly assessed these and found published data severely limited. Bhatia & Doschek (1995a, 1996) re-calculated Fe X and Fe XI for v1; Fe IX, Fe XIII used SSTRUCT improvements. This precise lineage matters: the Fe IX 171 Å, Fe X 174–177 Å, Fe XI 180 Å, Fe XII 195 Å lines drive every modern EUV passband (TRACE, EIT, AIA, EUI/FSI).
   한국어: Fe IX–XIV는 EUV의 주요 코로나 진단; Mason (1994)이 명시적으로 평가하여 출간 데이터가 심각히 제한적임을 발견. Bhatia & Doschek (1995a, 1996)이 v1을 위해 Fe X, Fe XI 재계산; Fe IX, Fe XIII는 SSTRUCT 개선. 이 정확한 계보가 중요하다: Fe IX 171 Å, Fe X 174–177 Å, Fe XI 180 Å, Fe XII 195 Å 라인이 모든 현대 EUV 통과대역(TRACE, EIT, AIA, EUI/FSI)을 추동한다.

6. **Open distribution by anonymous FTP prefigures FAIR data.** / **익명 FTP 공개 배포가 FAIR 데이터를 예시한다.**
   English: At a time when most spectral codes were sent by physical media on request, CHIANTI's distribution via anonymous FTP at CDS Strasbourg (`cdsarc.u-strasbg.fr`) and the WWW (`http://cdsweb.u-strasbg.fr/`) embedded an "update from anywhere, by anyone" workflow. This is a 1990s prototype of today's FAIR (Findable, Accessible, Interoperable, Reusable) data culture; the modern CHIANTI website and the Python port ChiantiPy continue this lineage.
   한국어: 대부분의 분광 코드가 요청에 따라 물리적 매체로 전송되던 시기에, CHIANTI는 CDS 스트라스부르의 익명 FTP(`cdsarc.u-strasbg.fr`)와 WWW(`http://cdsweb.u-strasbg.fr/`)로 배포되어 "어디서나, 누구나 갱신" 워크플로우를 내장했다. 이는 오늘날 FAIR(Findable, Accessible, Interoperable, Reusable) 데이터 문화의 1990년대 프로토타입; 현대 CHIANTI 웹사이트와 Python 포팅 ChiantiPy가 이 계보를 잇는다.

7. **The contribution function $G(T_e, n_e)$ is the universal currency for instrument response.** / **기여 함수 $G(T_e, n_e)$는 기기 응답 함수의 보편 통화이다.**
   English: Although not written explicitly in this paper, the natural composition of Eqs. (1)–(2)–(5) is the per-line contribution function. SDO/AIA, Hinode/EIS, IRIS, Solar Orbiter EUI/SPICE, and EIT all build their temperature response curves $R(T_e)$ as $\sum_{\text{lines}} G(T_e)\, T(\lambda)$ where $T(\lambda)$ is the wavelength-dependent throughput. Paper #61 (Abbo+ 2025) is a direct application: their FSI 17.4 nm $R(T_e)$ uses CHIANTI v10.1's $G(T_e)$.
   한국어: 본 논문에 명시적으로 쓰이진 않았지만, 식 (1)–(2)–(5)의 자연스러운 합성이 라인별 기여 함수. SDO/AIA, Hinode/EIS, IRIS, Solar Orbiter EUI/SPICE, EIT 모두 온도 응답 곡선 $R(T_e) = \sum_{\text{lines}} G(T_e)\, T(\lambda)$ ($T(\lambda)$는 파장 의존 throughput)로 만든다. Paper #61 (Abbo+ 2025)이 직접 응용: FSI 17.4 nm $R(T_e)$가 CHIANTI v10.1의 $G(T_e)$ 사용.

8. **Explicit caveats are baked into v1 — proton rates, He sequence, $\lambda < 50$ Å.** / **명시적 한계가 v1에 내장 — 양성자 율, He sequence, $\lambda < 50$ Å.**
   English: The paper does not oversell. Proton rates (important for fine-structure transitions in highly ionised systems) are explicitly missing from v1 (Copeland et al. 1996 review cited as the upcoming source). All He-sequence ions and all H-like ions except He II are missing. $\lambda < 50$ Å incomplete. $N_e < 10^{15}$ cm$^{-3}$ regime only. This honesty about scope is itself a model for community data products.
   한국어: 논문은 과대 광고하지 않는다. 양성자 율(고이온화 미세구조 천이에 중요)은 v1에서 명시적으로 누락(Copeland et al. 1996 리뷰가 향후 소스로 인용). He sequence 이온 전체와 H-like 중 He II 제외 전체 누락. $\lambda < 50$ Å 미완성. $N_e < 10^{15}$ cm$^{-3}$ 영역만. 범위에 대한 이런 정직함 자체가 커뮤니티 데이터 제품의 모범.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Optically thin line emissivity / 광학적으로 얇은 선의 emissivity (Eq. 1)
$$
\boxed{\;\epsilon_{i,j} = N_j(X^{+m})\, A_{j,i}\, \frac{hc}{\lambda_{i,j}}\;}\quad [\mathrm{erg\,cm^{-3}\,s^{-1}}]
$$
- $\epsilon_{i,j}$: power emitted per unit volume in transition $j \to i$ / 천이 $j \to i$의 단위 부피당 일률.
- $N_j(X^{+m})$: upper-level population density of ion $X^{+m}$ (cm$^{-3}$) / 윗준위 인구밀도.
- $A_{j,i}$: Einstein spontaneous emission coefficient (s$^{-1}$) / Einstein 자발 복사 계수.
- $hc/\lambda_{i,j}$: photon energy / 광자 에너지.

### 4.2 Upper-level population factorisation / 윗준위 인구 인수분해 (Eq. 2)
$$
\boxed{\;N_j(X^{+m}) = \frac{N_j(X^{+m})}{N(X^{+m})} \cdot \frac{N(X^{+m})}{N(X)} \cdot \frac{N(X)}{N(H)} \cdot \frac{N(H)}{N_e} \cdot N_e\;}
$$
- Factor 1 / 인수 1: level population fraction (SE solver, $f(T_e, N_e)$) / 준위 인구 비율.
- Factor 2 / 인수 2: ion fraction (Arnaud & Rothenflug 1985, $\sim f(T_e)$) / 이온 비율.
- Factor 3 / 인수 3: element abundance (user-specifiable) / 원소 풍부도.
- Factor 4 / 인수 4: $N(H)/N_e \approx 0.83$ for fully ionised H+He / 완전 이온화 H+He에서 ~0.83.

### 4.3 Flux at Earth / 지구에서의 flux (Eq. 3)
$$
\boxed{\;I(\lambda_{i,j}) = \frac{1}{4\pi R^2}\int_V \epsilon_{i,j}\, dV\;}\quad [\mathrm{erg\,cm^{-2}\,s^{-1}}]
$$
- $V$: emission volume / 방출 부피; $R$: Earth-to-source distance / 지구–소스 거리.

### 4.4 Statistical equilibrium / 통계 평형 (Eq. 4)
$$
\boxed{\;
N_j\!\left[N_e\!\sum_i C^e_{j,i} + N_p\!\sum_i C^p_{j,i} + \!\!\sum_{i>j} R_{j,i} + \!\!\sum_{i<j} A_{j,i}\right]
=
\sum_i N_i\!\left[N_e C^e_{i,j} + N_p C^p_{i,j}\right] + \!\!\sum_{i>j} N_i A_{i,j} + \!\!\sum_{i<j} N_i R_{i,j}
\;}
$$
- LHS / 좌변: total rate out of level $j$ / 준위 $j$를 떠나는 총 비율.
- RHS / 우변: total rate into level $j$ / 준위 $j$로 들어오는 총 비율.
- $C^e, C^p$: electron/proton collisional rate coefficients (cm$^3$ s$^{-1}$) / 전자·양성자 충돌율 계수.
- $R_{j,i}$: stimulated absorption rate (s$^{-1}$); negligible for low-density / 저밀도에서 무시.
- $A_{j,i}$: spontaneous radiation rate (s$^{-1}$) / 자발 복사율.
- v1 omits $C^p$ / v1에서 양성자 율 누락.

### 4.5 Maxwellian-averaged rate coefficient / Maxwell 평균 율 계수 (Eq. 5)
$$
\boxed{\;
C^e_{i,j}(T_e) = \frac{8.63\times 10^{-6}}{T_e^{1/2}}\,\frac{\Upsilon_{i,j}(T_e)}{\omega_i}\,\exp\!\left(-\frac{E_{i,j}}{kT_e}\right)
\;}
$$
- Constant $8.63\times 10^{-6}$: from $\sqrt{2\pi\hbar^4/(km_e^3)}$ collected with $\pi a_0^2$ / 보통 단위계 정수.
- $\omega_i$: statistical weight of lower level ($2J_i + 1$) / 아래준위 통계 가중치.
- $\Upsilon_{i,j}$: thermally-averaged effective collision strength / 열평균 유효 충돌 강도 (Eq. 6 below):

$$
\boxed{\;
\Upsilon_{i,j}(T_e) = \int_0^{\infty} \Omega_{i,j}\, \exp\!\left(-\frac{E_j}{kT_e}\right)\, d\!\left(\frac{E_j}{kT_e}\right)
\;}
$$
- $E_j$: scattered electron energy relative to final ion state / 최종 이온 상태 기준 산란 전자 에너지.
- De-excitation by detailed balance: $C^e_{j,i} = (\omega_i/\omega_j) C^e_{i,j} \exp(+E_{i,j}/kT_e)$.
- $\Omega_{i,j}$ symmetric, dimensionless; $\sigma = \pi a_0^2 (\Omega/\omega_i)/E$ / 단면적과의 관계.

### 4.6 Van Regemorter $\bar g$ approximation / Van Regemorter $\bar g$ 근사 (Eq. 7)
$$
\boxed{\;
\Omega_{i,j} = \frac{8\pi}{\sqrt 3}\,\omega_j\, f_{i,j}\, \frac{I_H}{E_{i,j}}\,\bar g
\;}
$$
- $f_{i,j}$: oscillator strength / 발진자 강도.
- $I_H$: hydrogen ionisation potential / 수소 이온화 에너지.
- $\bar g$: effective Gaunt factor; ~1 within 25% only for $\Delta n = 0$ alkali-like / $\Delta n = 0$ 알칼리류에서만 25% 이내.
- Used as fallback when no R-matrix/CC/DW data exist; replaced wherever possible / R-matrix/CC/DW가 없을 때 폴백, 가능한 한 대체.

### 4.7 Burgess–Tully scaling for dipole transitions / 쌍극자 천이의 BT 스케일링
$$
\boxed{\;
x = 1 - \frac{\ln(C)}{\ln(X - 1 + C)},\qquad y = \Omega\,\ln(X - 1 + e)\;}
$$
- $X = E/E_{\text{th}}$: incident electron energy in threshold units / threshold 단위 입사 에너지.
- $C$: case-adjusted scale parameter / 사례별 스케일 파라미터.
- $x \in [0, 1]$: maps threshold ($x=0$) to infinity ($x=1$) / threshold→무한대 매핑.
- High-energy limit (Coulomb-Bethe) / 고에너지 극한: $\Omega \to 4\omega_i f_{i,j}/E_{i,j}$ (Rydberg) at $x = 1$.
- 5-point spline fit reproduces $\Upsilon$ to <1% (0.2% Mg X, 0.5% N III) / 5-point spline이 1% 미만 재현.
- Different scaling forms for forbidden / intercombination transitions (paper §3.4) / 금지·intercombination 천이는 다른 스케일링.

### 4.8 Contribution function (composite, derived) / 기여 함수 (합성, 유도)
$$
\boxed{\;
G(T_e, n_e) = \frac{hc}{\lambda_{i,j}}\,\frac{A_{j,i}\, N_j(X^{+m})/N(X^{+m})}{N_e}\,\frac{N(X^{+m})}{N(X)}\,\frac{N(X)}{N(H)}\,\frac{N(H)}{N_e}\;}
$$
$$
I(\lambda_{i,j}) = \frac{1}{4\pi}\int G(T_e, n_e)\, n_e^2\, d\ell = \frac{1}{4\pi}\int G(T_e)\,\mathrm{DEM}(T_e)\, dT_e
$$
- $\mathrm{DEM}(T_e) = n_e^2 \, d\ell/dT_e$: differential emission measure / 미분 방출 측도.
- IDL routine `g_of_t.pro` computes $G$ / IDL 루틴 `g_of_t.pro`가 계산.
- Units: erg cm$^3$ s$^{-1}$ sr$^{-1}$ (per-line) / 라인별 단위.
- Used by paper #61 (Abbo+ 2025) for FSI 17.4 nm $R(T_e)$ / paper #61이 FSI 17.4 nm $R(T_e)$ 산출에 사용.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1928 ──── Edlén & Grotrian: Coulomb-style energy level systematics
1942 ──── Edlén: identifies coronal "forbidden" lines as Fe X, Fe XIV — birth of coronal spectroscopy
                 (ends the 1869 "coronium" mystery)
1955 ──── Pottasch: first DEM-style temperature analyses of solar EUV
1962 ──── Van Regemorter: g-bar approximation for collision strengths
1970 ──── Landini & Monsignori Fossi: first comprehensive coronal spectral code
1971 ──── Burke et al.: R-matrix package developed at QUB Belfast
1972 ──── Mewe: spectral code with polynomial collision-strength fits
1972 ──── Eissner & Seaton: UCL DW program
1972 ──── Eissner et al.: SSTRUCT atomic structure program (UCL)
1974 ──── Burgess & Sheorey: Coulomb-Bethe high-partial-wave formulation
1976 ──── Kato: ATOMDB-precursor spectral code
1978 ──── Berrington et al.: extended R-matrix package
1981 ──── Mewe & Gronenschild: extended Mewe formulation
1985 ──── Arnaud & Rothenflug: standard ionisation equilibrium tables
1987 ──── DARC (Dirac-Fock R-matrix): Norrington & Grant
1991 ──── Yohkoh launched (SXT, BCS) — soft X-ray spectroscopy era begins
1992 ──── EUVE launched
1992 ──── Burgess & Tully: scaling method for collision strengths (BBC BASIC)
1992 ──── Abingdon SOHO CDS/SUMER atomic data workshop (March 1992)
1993 ──── Iron Project consortium begins (Hummer et al.)
1994 ──── ADNDT special volume on isoelectronic sequences (ed. Lang)
1994 ──── Mason: review/critique of plasma emission codes; Fe IX–XIV assessment
1995 ──── NIST critical compilation of energy levels (Martin et al.)
1995 ──── SOHO launched (CDS, SUMER, UVCS, EIT)
1995-96 ──── Bhatia & Doschek: new Fe X, Fe XI calculations for CHIANTI
1997 ──── ★ Dere et al.: CHIANTI v1 published — THIS PAPER
                 (received July 1996, accepted November 1996)
... ──── Continuous evolution
2003 ──── CHIANTI v4 (Young et al.): proton rates, photoexcitation added
2009 ──── CHIANTI v6 (Dere et al.): full ionisation/recombination unified
2015 ──── CHIANTI v8: improved Fe coronal lines for Hinode/EIS, IRIS era
2021 ──── CHIANTI v10 (Del Zanna et al.): non-Maxwellian distributions supported
2023 ──── CHIANTI v10.1 (Dere et al.) [paper #64] — used by paper #61
2025 ──── Abbo et al. [paper #61]: Solar Orbiter/FSI 17.4 nm R(T_e) uses CHIANTI v10.1
```

The 1997 paper sits at the inflection point between (a) the previous era of independently-developed FORTRAN spectral codes with hard-coded atomic data (Mewe, Raymond–Smith, Landini & Monsignori Fossi) and (b) the modern era of community-curated, version-controlled, openly-distributed atomic databases. Its 25+ year continuous evolution tracks every major EUV/X-ray solar mission.

1997년 논문은 (a) 하드코딩 원자 데이터를 가진 독립 개발 FORTRAN 분광 코드 시대(Mewe, Raymond–Smith, Landini & Monsignori Fossi)와 (b) 커뮤니티 큐레이션·버전 관리·공개 배포 원자 데이터베이스의 현대 사이의 변곡점에 위치한다. 25년 이상의 연속적 진화가 모든 주요 EUV/X-ray 태양 임무를 추적한다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#61 Abbo et al. (2025)** — Solar Orbiter/EUI FSI 17.4 nm response function $R(T_e)$ | Direct downstream user / 직접 다운스트림 사용자 | Abbo+ 2025 sums CHIANTI's $G(T_e, n_e)$ over every spectral line within the FSI 17.4 nm bandpass to build $R(T_e)$. Fe IX 171 Å, Fe X 174–177 Å, Fe XI 180 Å, Fe XII 195 Å (the Bhatia & Doschek 1995/96 data introduced in CHIANTI v1) dominate. Without CHIANTI's modular factorisation (Eq. 2) and BT-scaled $\Upsilon$, the FSI temperature response cannot be computed. / FSI 17.4 nm 통과대역 안 모든 라인의 $G(T_e, n_e)$를 합산해 $R(T_e)$ 구축. CHIANTI v1에 도입된 Bhatia & Doschek 1995/96 데이터의 Fe IX–XII 라인이 지배. |
| **#64 Dere et al. (2023)** — CHIANTI v10.1 | Continuous evolution of the same database / 같은 데이터베이스의 연속 진화 | Same lead author, same database, 26 years later. v10.1 retains the file-format philosophy and BT scaling philosophy of v1, while expanding to non-Maxwellian distributions, more accurate Fe coronal lines, modern Iron Project results, and unified ionisation/recombination. The version actually used by paper #61. / 같은 lead 저자, 같은 데이터베이스, 26년 후. 파일 형식과 BT 스케일링 철학은 v1을 유지하면서 비-Maxwellian 분포, 정확도 높은 Fe 코로나 라인, 현대 Iron Project, 일원화된 이온화/재결합으로 확장. paper #61이 실제 사용. |
| **#65 Mason (1994)** — review/assessment of Fe IX–Fe XIV electron excitation data | Foundational diagnostic methodology by same author; cited explicitly in §3.4 / 같은 저자의 기초 진단 방법론; §3.4에 명시 인용 | H. E. Mason is co-author of this paper AND author of the 1994 Fe IX–XIV assessment that is explicitly cited in §3.4 as motivating the inclusion of Bhatia & Doschek (1995a, 1996) Fe X and Fe XI data in CHIANTI v1. Mason (1994) directly shaped which iron coronal data CHIANTI v1 used. / H. E. Mason은 본 논문 공저자이자 §3.4에 명시 인용된 1994년 Fe IX–XIV 평가 논문 저자. CHIANTI v1이 어떤 철 코로나 데이터를 사용할지 직접 결정. |
| **Burgess & Tully (1992)** — scaling method for collision strengths | Methodological foundation / 방법론적 기초 | Provides the entire data-storage philosophy of CHIANTI: scaled $\Omega(x)$ and $\Upsilon(x)$ on $[0,1]$ with Bethe limit pinned at $x = 1$, enabling 5-point spline storage and visual inspection. The IDL companion `BURLY` is named in homage. / CHIANTI의 데이터 저장 철학 전체 제공: $[0,1]$에서 스케일링된 $\Omega(x)$, $\Upsilon(x)$를 Bethe 극한에 고정. 5-point spline 저장과 시각 검사 가능. IDL 동봉 패키지 `BURLY`가 헌사로 명명. |
| **Mason & Monsignori Fossi (1994)** — extensive review of line-ratio diagnostics | Theoretical companion / 이론적 동반 논문 | Cited in §2 as the comprehensive review of line-ratio diagnostics that this database is designed to enable computationally. Two co-authors of CHIANTI v1 (Mason, Monsignori Fossi) wrote this review three years earlier, articulating the diagnostic methodology that CHIANTI operationalises. / §2에서 본 데이터베이스가 계산 가능하게 만들고자 하는 선 비율 진단 종합 리뷰로 인용. CHIANTI v1 공저자 둘(Mason, Monsignori Fossi)이 3년 전 작성, CHIANTI가 운용하는 진단 방법론 명시. |
| **Arnaud & Rothenflug (1985)** — ionisation equilibrium tables | Required external input / 필수 외부 입력 | Eq. (2) factor 2 — ion fraction $N(X^{+m})/N(X)$ — is taken from Arnaud & Rothenflug (1985). Without this, the database cannot map atomic physics output to observable line intensities. / 식 (2)의 인수 2(이온 비율)는 Arnaud & Rothenflug에서 가져옴. 없으면 원자물리 출력을 관측 가능한 선 세기에 매핑 불가. |
| **NIST (Martin et al. 1995)** — critical compilation of atomic energy levels | Primary energy level source / 주요 에너지 준위 소스 | §3.1 specifies NIST as the primary source of all energy levels in CHIANTI; SSTRUCT only fills gaps. The NIST compilation's accuracy directly determines CHIANTI's wavelength accuracy. / §3.1: 모든 에너지 준위의 주요 소스로 NIST 명시; SSTRUCT는 빈 곳만 채움. NIST 정확도가 CHIANTI 파장 정확도를 직접 결정. |
| **Iron Project (Hummer et al. 1993)** — international CC/R-matrix consortium | Future-data pipeline / 미래 데이터 파이프라인 | Cited in §3.4 as the source of the very accurate CC + relativistic-target Fe data slated for inclusion in future CHIANTI releases. The Iron Project is the long-term solution to the "iron data inadequacy" Mason (1994) flagged. / §3.4에 인용; 향후 CHIANTI 릴리스에 포함될 매우 정확한 CC + 상대론적 표적 Fe 데이터의 소스. Mason (1994)이 지적한 "철 데이터 부족"의 장기적 해법. |

---

## 7. References / 참고문헌

- K. P. Dere, E. Landi, H. E. Mason, B. C. Monsignori Fossi, P. R. Young, "CHIANTI — an atomic database for emission lines. I. Wavelengths greater than 50 Å", *Astronomy and Astrophysics Supplement Series*, **125**, 149–173 (1997). DOI: [10.1051/aas:1997368](https://doi.org/10.1051/aas:1997368).

### Cited in the paper / 본 논문 인용
- Aggarwal et al. 1991b, 1992 — He II R-matrix collision strengths.
- Arnaud & Rothenflug 1985, A&AS, 60, 425 — ionisation equilibrium tables.
- Berrington et al. 1978, 1985, 1989 — R-matrix package and Be-like calculations.
- Bhatia & Doschek 1995a, 1996 — Fe X, Fe XI collision strengths used in v1.
- Bhatia & Kastner 1992 — C III A values.
- Brickhouse et al. 1995 — recent atomic-parameter calculations for iron ions.
- Burgess & Tully 1992, A&A, 254, 436 — scaling method (foundational).
- Burgess et al. 1996, 1997 — Coulomb-Born high-energy limits for non-dipole transitions.
- Burgess & Sheorey 1974 — Coulomb-Bethe high-partial-wave formulation.
- Burke et al. 1971 — R-matrix package (QUB).
- Copeland et al. 1996 — review of theoretical proton excitation rates.
- Dere & Mason 1981; Dere et al. 1979, 1982 — Be-like and B-like density diagnostics.
- Doschek 1997 — review of plasma diagnostics.
- Dunn 1992; Henry 1993 — reviews of laboratory electron-excitation rate measurements.
- Dwivedi 1994 — review of EUV diagnostics.
- Edlén 1981 — energy levels along isoelectronic sequences.
- Eissner et al. 1974; Eissner & Seaton 1972 — SSTRUCT and UCL DW programs.
- Gaetz & Salpeter 1983 — older spectral code.
- Grant et al. 1980 — multi-configuration Dirac-Fock atomic structure.
- Hummer et al. 1993 — Iron Project.
- Itikawa 1991 — atomic-data bibliography.
- Kato 1976, 1996 — earlier spectral code; updates to O V.
- Keenan 1996 — bibliography of spectroscopic diagnostics.
- Kelly 1987 — atomic energy level tables.
- Lang (ed.) 1994, ADNDT special volume — Abingdon workshop reviews.
- Landini & Monsignori Fossi 1970, 1990 — earlier coronal spectral codes.
- Martin et al. 1995 — NIST critical compilation of energy levels.
- Mason 1994 — Fe IX–XIV assessment (paper #65 in this study).
- Mason 1996a — comparison/critique of plasma emission codes.
- Mason & Monsignori Fossi 1994 — review of line-ratio diagnostics.
- Mason & Storey 1980 — Fe XXII X-ray and XUV transitions.
- Mewe 1972; Mewe & Gronenschild 1981 — earlier polynomial-fit spectral codes.
- Mewe et al. 1995 — recent atomic-parameter calculations.
- Monsignori Fossi & Landini 1994a — recent atomic-parameter calculations.
- Norrington & Grant 1987 — DARC (Dirac-Fock R-matrix).
- Pradhan & Gallagher 1992 — atomic-data compilation.
- Ramsbottom et al. 1994, 1995 — N IV R-matrix.
- Raymond & Smith 1977 — earlier spectral code.
- Saraph 1972 — Term Coupling Coefficients (LS → IC transformation).
- Sampson & Zhang 1992; Sampson et al. 1984, 1986, 1994; Zhang & Sampson 1992, 1994, 1995; Zhang & Pradhan 1994; Zhang 1995 — relativistic DW programme and Li-/Be-/B-like collisional data.
- Stafford et al. 1993, 1994 — N III oscillator strengths and R-matrix.
- Stern et al. 1978 — earlier spectral code.
- Tucker & Koren 1971 — earlier spectral code.
- Unnikrishnan et al. 1991 — He II close-coupling collision strengths.
- Van Regemorter 1962 — $\bar g$ approximation.
- Vernazza & Mason 1978 — B-like solar atmospheric diagnostics.
- Wiese et al. 1966; Wiese & Fuhr 1995 — atomic transition probabilities.
- Younger & Wiese 1979 — assessment of $\bar g$ approximation accuracy.

### Companion / context references / 동반·맥락 참고
- Del Zanna, G. & Mason, H. E., "Solar UV and X-ray spectral diagnostics", *Living Reviews in Solar Physics*, **15**, 5 (2018). [Modern review of CHIANTI-based diagnostics]
- Dere, K. P., et al. (CHIANTI v10.1, 2023) — see paper #64 in this study.
- Abbo, L., et al., Solar Orbiter EUI/FSI 17.4 nm response (2025) — see paper #61 in this study.
