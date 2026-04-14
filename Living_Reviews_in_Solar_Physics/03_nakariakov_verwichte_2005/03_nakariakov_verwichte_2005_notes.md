---
title: "Coronal Waves and Oscillations"
authors: Valery M. Nakariakov, Erwin Verwichte
year: 2005
journal: "Living Reviews in Solar Physics, Vol. 2, 3"
topic: Living Reviews in Solar Physics / MHD Waves
tags: [MHD, coronal seismology, kink oscillations, sausage mode, Alfvén waves, coronal loops, TRACE, SOHO, wave damping, resonant absorption, phase mixing]
status: completed
date_started: 2026-04-08
date_completed: 2026-04-08
---

# Coronal Waves and Oscillations — Reading Notes
# 코로나 파동과 진동 — 읽기 노트

---

## 핵심 기여 / Core Contribution

이 리뷰는 1990년대 후반~2000년대 초반 SOHO와 TRACE 우주선이 가져온 관측 혁명 이후, 태양 코로나에서 검출된 다양한 MHD 파동과 진동 현상을 최초로 체계적으로 분류하고 이론적으로 해석한 포괄적 리뷰입니다. 저자들은 kink, sausage, acoustic, torsional 등 네 가지 주요 진동 모드를 관측 증거와 함께 정리하고, 자기 실린더 모델에 기반한 MHD 분산 관계 이론을 통해 각 모드의 물리적 특성을 설명합니다. 가장 혁신적인 기여는 **MHD coronal seismology** — 코로나 파동의 관측 특성(주기, 파장, 감쇠율)을 MHD 이론의 분산 관계와 결합하여 코로나 자기장 강도, 수송 계수, 미세 구조 등 직접 측정이 불가능한 물리량을 추정하는 기법 — 의 이론적·관측적 기초를 확립한 것입니다.

This review is the first comprehensive classification and theoretical interpretation of the diverse MHD wave and oscillation phenomena detected in the solar corona following the observational revolution brought by SOHO and TRACE spacecraft in the late 1990s–early 2000s. The authors systematically organize four major oscillation modes — kink, sausage, acoustic, and torsional — with supporting observational evidence, and explain the physical characteristics of each mode through MHD dispersion relation theory based on the magnetic cylinder model. The most innovative contribution is establishing the theoretical and observational foundations of **MHD coronal seismology** — a technique that combines observed wave properties (period, wavelength, damping rate) with MHD dispersion relations to estimate otherwise unmeasurable coronal parameters such as magnetic field strength, transport coefficients, and fine-scale structuring.

---

## 읽기 노트 / Reading Notes

### §1 Introduction — MHD 코로나 지진학의 개념 / The Concept of MHD Coronal Seismology

태양 코로나의 자기적으로 지배되는 플라즈마($\beta \ll 1$)는 탄성적이고 압축 가능한 매질로, 다양한 종류의 파동을 지원합니다. 이온 Larmor 반지름(< 1 m)과 자이로주기(< $10^{-4}$ s)보다 훨씬 큰 파장과 주기를 가진 파동은 MHD 프레임워크로 기술됩니다. 이 파동들의 주기는 수 초에서 수 분 범위이며, 현대 관측 장비의 시간·공간 분해능으로 검출 가능합니다.

The magnetically dominated plasma ($\beta \ll 1$) of the solar corona is an elastic, compressible medium supporting various types of waves. Waves with wavelengths and periods much larger than the ion Larmor radius (< 1 m) and gyroperiod (< $10^{-4}$ s) are described within the MHD framework. These waves have typical periods in the range of a few seconds to several minutes, well covered by the temporal and spatial resolution of modern observational tools.

코로나 물리학의 핵심 미해결 문제 — 코로나 가열, 태양풍 가속, 태양 플레어 — 에 답하려면 코로나의 물리적 조건과 매개변수에 대한 상세한 지식이 필요하지만, 코로나 자기장의 정확한 값은 여전히 미지입니다. Zeeman 분리, 자기회전 공명 방출 등 직접적 방법에는 본질적인 어려움이 있고, 부피 및 전단 점성, 저항률, 열전도 등의 수송 계수도 크기 자릿수 이내로 측정되지 않습니다.

Key unsolved problems in coronal physics — coronal heating, solar wind acceleration, solar flares — require detailed knowledge of coronal conditions, but the exact value of the coronal magnetic field remains unknown. Direct methods (Zeeman splitting, gyroresonant emission) have intrinsic difficulties, and transport coefficients (volume/shear viscosity, resistivity, thermal conduction) are not measured even within an order of magnitude.

**MHD 코로나 지진학**은 이 간극을 메우는 새로운 도구입니다. 코로나 파동의 관측 가능한 특성(주기, 파장, 진폭, 감쇠율)을 이론적 모델(분산 관계, 진화 방정식)과 결합하면, 자기장 강도, 수송 계수 등의 미지 매개변수를 결정할 수 있습니다. 이는 태양 내부의 음향 진단인 helioseismology와 철학적으로 유사하지만, Alfvén, slow, fast 세 가지 파동 모드를 기반으로 하여 훨씬 풍부한 진단 잠재력을 가집니다.

**MHD coronal seismology** fills this gap. By combining observationally measurable wave properties (period, wavelength, amplitude, damping rate) with theoretical models (dispersion relations, evolutionary equations), unknown parameters such as magnetic field strength and transport coefficients can be determined. Philosophically similar to helioseismology (acoustic diagnostics of the solar interior), but based on three different wave modes — Alfvén, slow, and fast — making the approach richer in diagnostic potential.

이 기법의 기원은 Uchida (1970, global seismology)와 Roberts et al. (1984, local seismology)까지 거슬러 올라가며, Nakariakov & Ofman (2001)이 최초로 kink 진동에서 코로나 자기장을 추정하는 데 적용했습니다.

The method traces back to Uchida (1970, global seismology) and Roberts et al. (1984, local seismology), with the first practical application by Nakariakov & Ofman (2001) estimating coronal magnetic fields from kink oscillations.

---

### §2 Properties of MHD Modes of Plasma Structures — MHD 이론의 기초 / Theoretical Foundations

#### §2.1 MHD modes of a straight cylinder — 직선 실린더의 MHD 모드

코로나 구조(루프, 플룸, 필라멘트 등)는 자기 실린더로 모델링됩니다. 반지름 $a$인 원통형 자기 플럭스 튜브를 고려합니다:

Coronal structures (loops, plumes, filaments) are modeled as magnetic cylinders. Consider a cylindrical magnetic flux tube of radius $a$:

- **내부 / Inside** ($r < a$): 밀도 $\rho_0$, 압력 $p_0$, 자기장 $B_0 \mathbf{e}_z$
  Density $\rho_0$, pressure $p_0$, magnetic field $B_0 \mathbf{e}_z$
- **외부 / Outside** ($r > a$): 밀도 $\rho_e$, 압력 $p_e$, 자기장 $B_e \mathbf{e}_z$
  Density $\rho_e$, pressure $p_e$, magnetic field $B_e \mathbf{e}_z$
- **평형 조건 / Equilibrium**: $p_0 + B_0^2/2\mu_0 = p_e + B_e^2/2\mu_0$

네 가지 특성 속도가 정의됩니다:
Four characteristic speeds are defined:

| 속도 / Speed | 공식 / Formula | 물리적 역할 / Physical role |
|---|---|---|
| Sound speed $C_s$ | $(\gamma p_0/\rho_0)^{1/2}$ | 가스 압력파 속도 / Gas pressure wave speed |
| Alfvén speed $C_A$ | $B_0/(\mu_0\rho_0)^{1/2}$ | 자기 장력파 속도 / Magnetic tension wave speed |
| Tube speed $C_T$ | $C_s C_A/(C_A^2+C_s^2)^{1/2}$ | Slow 모드의 가이드 속도 / Guided speed of slow modes |
| Kink speed $C_K$ | $[(\rho_0 C_{A0}^2+\rho_e C_{Ae}^2)/(\rho_0+\rho_e)]^{1/2}$ | Kink 모드의 위상 속도 / Phase speed of kink modes |

MHD 방정식을 평형 주위에서 선형화하면 분산 관계(Eq. 7, Edwin & Roberts 1983)를 얻습니다:

Linearizing MHD equations around equilibrium yields the dispersion relation (Eq. 7, Edwin & Roberts 1983):

$$\rho_e(\omega^2 - k_z^2 C_{Ae}^2)\kappa_0 \frac{I_m'(\kappa_0 a)}{I_m(\kappa_0 a)} + \rho_0(k_z^2 C_{A0}^2 - \omega^2)\kappa_e \frac{K_m'(\kappa_e a)}{K_m(\kappa_e a)} = 0$$

여기서 $I_m$, $K_m$은 수정 Bessel 함수이며, $m$은 방위각 모드 번호입니다:

Where $I_m$, $K_m$ are modified Bessel functions, and $m$ is the azimuthal mode number:

- **$m = 0$: sausage 모드** — 축대칭, 루프 단면 팽창/수축
  Axisymmetric, loop cross-section expansion/contraction
- **$m = 1$: kink 모드** — 루프 전체의 횡방향 변위
  Transverse displacement of the entire loop
- **$m \geq 2$: flute/ballooning 모드** — 고차 방위각 구조
  Higher-order azimuthal structure

Figure 3의 분산 다이어그램(이 논문의 핵심 그림)은 위상 속도 $\omega/k_z$를 무차원 파수 $k_z a$의 함수로 보여줍니다. MHD 모드의 위상 속도는 두 밴드에 존재합니다:

Figure 3's dispersion diagram (the key figure of this paper) shows phase speed $\omega/k_z$ as a function of dimensionless wavenumber $k_z a$. Phase speeds of MHD modes exist in two bands:

1. **Fast band**: $C_{A0}$와 $C_{Ae}$ 사이 ($C_{A0} < C_{Ae}$일 때) — fast magnetoacoustic modes
   Between $C_{A0}$ and $C_{Ae}$ (when $C_{A0} < C_{Ae}$) — fast magnetoacoustic modes
2. **Slow band**: $C_{T0}$와 $C_{s0}$ 사이 — slow magnetoacoustic modes
   Between $C_{T0}$ and $C_{s0}$ — slow magnetoacoustic modes

Torsional Alfvén 파동은 분산이 없는 $\omega/k_z = C_{A0}$ 직선으로 나타납니다.

Torsional Alfvén waves appear as the non-dispersive line $\omega/k_z = C_{A0}$.

**Body modes vs surface modes**: 모든 trapped 모드는 body mode(내부에서 진동, 외부에서 감쇠)입니다. Surface mode(내부에서도 감쇠)는 특정 조건에서만 존재합니다.

**Body modes vs surface modes**: All trapped modes are body modes (oscillatory inside, evanescent outside). Surface modes (evanescent inside too) exist only under specific conditions.

**Leaky modes**: mode localization 조건을 완화하면(외부로 에너지 방사 허용) 복소 고유진동수를 가진 leaky mode가 존재합니다. 분산 관계에서 $K_m(x)$을 Hankel 함수로 대체합니다.

**Leaky modes**: Relaxing the mode localization condition (allowing energy radiation into external medium) gives leaky modes with complex eigenfrequencies. In the dispersion relation, $K_m(x)$ is replaced by Hankel functions.

#### §2.2 MHD continua — MHD 연속체

플라즈마 밀도가 자기장에 수직으로 연속적으로 변할 때, 각 자기면에서의 Alfvén 고유진동수 $C_A(r)|k_z|$도 연속적으로 변합니다. 이것이 **Alfvén 연속체**입니다. 마찬가지로 **slow (cusp) 연속체** $C_T(r)|k_z|$도 존재합니다.

When plasma density varies continuously across the magnetic field, the Alfvén eigenfrequency $C_A(r)|k_z|$ at each magnetic surface also varies continuously. This is the **Alfvén continuum**. Similarly, a **slow (cusp) continuum** $C_T(r)|k_z|$ exists.

**공진 흡수 / Resonant absorption** (§2.2.1):

전체적(global) 파동 모드의 주파수 $\omega$가 Alfvén 연속체 내에 위치할 때, 특정 반지름 $r_A$에서 $\omega = C_A(r_A)|k_z|$인 공진이 발생합니다. 에너지가 전체 모드에서 국소 Alfvén 모드로 세속적으로(secularly) 전달되며, 이 과정을 **모드 변환(mode conversion)**이라 합니다. 공진층에서 섭동 진폭이 발산하므로 산일을 고려해야 합니다.

When a global mode frequency $\omega$ lies within the Alfvén continuum, a resonance occurs at the specific radius $r_A$ where $\omega = C_A(r_A)|k_z|$. Energy transfers secularly from the global mode to the local Alfvén mode — a process called **mode conversion**. The perturbation amplitude diverges at the resonant layer, requiring dissipation.

핵심 결과 — 얇은 경계층($\ell \ll a$)을 가진 약산일 루프에서 kink 파동의 감쇠 시간 (Ruderman & Roberts 2002):

Key result — damping time for a kink wave in a weakly dissipative loop with a thin boundary layer ($\ell \ll a$) (Ruderman & Roberts 2002):

$$\frac{\tau}{P} = \frac{2}{\pi}\left(\frac{\ell}{a}\right)^{-1}\left(\frac{\rho_0 + \rho_e}{\rho_0 - \rho_e}\right)$$

이 식은 점성 계수를 포함하지 않습니다 — 감쇠는 산일이 아닌 모드 변환에 의한 것입니다. 관측된 감쇠를 설명하려면 $\ell/a \approx 0.23$ 정도가 필요합니다.

This expression does not contain the viscosity coefficient — the damping is due to mode conversion, not dissipation. To explain observed damping, $\ell/a \approx 0.23$ is needed.

중요한 제한 사항: sausage 모드($m=0$)에서는 자기음향 방정식과 Alfvén 방정식이 디커플링되어 Alfvén 공진을 통한 공진 흡수가 일어나지 않습니다.

Important caveat: For sausage modes ($m=0$), the magnetoacoustic and Alfvén equations decouple, so resonant absorption through Alfvén resonance does not operate.

**위상 혼합 / Phase mixing** (§2.2.2):

전체 모드 대신, 각 자기면에서 독립적으로 Alfvén 파동이 여기되는 경우를 고려합니다. 각 자기면의 Alfvén 속도 $C_A(x)$가 다르므로, 초기에 평면이었던 파동이 점차 기울어지면서 횡방향 경사가 급격히 증가합니다. 이 작은 스케일의 경사는 산일을 크게 강화합니다.

Instead of a global mode, consider Alfvén waves excited independently on each magnetic surface. Since the Alfvén speed $C_A(x)$ differs on each surface, an initially plane wave gradually tilts, steepening transverse gradients. These small-scale gradients greatly enhance dissipation.

Heyvaerts & Priest (1983)의 감쇠 법칙:

Decay law by Heyvaerts & Priest (1983):

$$V_y(t) \propto V_y(0) \exp\left\{-\frac{\nu k_z^2}{6}\left[\frac{dC_A(x)}{dx}\right]^2 t^3\right\}$$

감쇠 시간은 $\text{Re}^{1/3}$에 비례하며(공진 흡수와 동일한 스케일링), 균일 매질에서의 $\text{Re}$ 비례보다 훨씬 빠릅니다.

Damping time is proportional to $\text{Re}^{1/3}$ (same scaling as resonant absorption), much faster than the $\text{Re}$ proportionality in a homogeneous medium.

#### §2.3 Zero plasma-β density profiles — 제로 플라즈마-β 밀도 프로파일

$\beta = 0$ (자기 압력 >> 가스 압력)인 극한에서, symmetric Epstein profile을 가진 자기 슬랩을 고려합니다:

In the $\beta = 0$ limit (magnetic pressure >> gas pressure), consider a magnetic slab with a symmetric Epstein profile:

$$\rho_0 = \rho_{\max}\text{sech}^2\left(\frac{x}{a}\right) + \rho_\infty$$

이 프로파일에서는 정확한 해석적 해를 구할 수 있으며, kink와 sausage 모드의 고유함수와 분산 관계가 명시적으로 주어집니다. 특히 sausage 모드의 군속도는 프로파일의 가파름에 크게 영향을 받으며, 이것이 fast wave train의 형태를 결정합니다.

With this profile, exact analytical solutions can be obtained, and eigenfunctions and dispersion relations for kink and sausage modes are given explicitly. Notably, the sausage mode group speed is strongly affected by profile steepness, which determines the shape of fast wave trains.

#### §2.4 Effects of twisting — 비틀림의 효과

자기장의 비틀림은 다양한 MHD 모드를 선형적으로 결합시킵니다. 약한 비틀림($K \ll 1$)에서 torsional 모드의 분산 관계:

Twisting of the magnetic field linearly couples various MHD modes. For weak twist ($K \ll 1$), the torsional mode dispersion relation:

$$\omega^2 \approx C_A^2\left(1+\frac{K\beta}{2}\right)k_z^2\left(1+\frac{a^2K^2\beta^2(1-\beta)}{16}k_z^2\right)$$

비틀림은 분산과 위상 속도 변화를 유발하며, 비틀린 실린더에서의 torsional 모드는 순수 비압축성이 아닌, 압축 성분(밀도 섭동)도 동반합니다.

Twisting introduces dispersion and phase speed modification; torsional modes in a twisted cylinder are not purely incompressible but also carry compressive components (density perturbations).

---

### §3 Kink Oscillations of Coronal Loops — 코로나 루프의 킹크 진동

#### §3.1 TRACE observations — TRACE 관측

**1998년 7월 14일**: TRACE가 활동 영역 AR 8270에서 태양 플레어 직후 코로나 루프의 감쇠하는 횡방향 진동을 최초로 spatially resolved하게 관측했습니다 (Aschwanden et al. 1999; Nakariakov et al. 1999). 171 Å과 195 Å 라인에서 관측되었습니다.

**14 July 1998**: TRACE first observed spatially resolved decaying transverse oscillations of coronal loops shortly after a solar flare in active region AR 8270 (Aschwanden et al. 1999; Nakariakov et al. 1999), in both 171 Å and 195 Å lines.

Nakariakov et al. (1999)의 정량적 분석 결과:

Quantitative analysis by Nakariakov et al. (1999):

| 관측량 / Observable | 값 / Value |
|---|---|
| 진동 주기 $P$ / Oscillation period | $4.3 \pm 0.9$ min ($\approx 256$ s) |
| 주파수 / Frequency | $\approx 4$ mHz |
| 변위 진폭 / Displacement amplitude | $\sim$ several Mm |
| 루프 길이 $2L/\pi$ / Loop footpoint distance | $\approx 83$ Mm |
| 루프 단면 직경 $2a$ / Loop cross-section diameter | $\approx 1$ Mm |
| e-folding 감쇠 시간 $\tau$ / Decay time | $14.5 \pm 2.7$ min |
| 위상 속도 추정 $\omega/k$ / Phase speed estimate | $1020 \pm 132$ km s$^{-1}$ |

루프 변위의 시간 진화는 $A\sin(\omega t + \phi)\exp(-\lambda t)$로 잘 맞습니다 (Figure 10). 파장이 루프 단면보다 훨씬 길므로 장파장 한계가 적용되어 위상 속도가 kink speed $C_K$에 접근합니다.

The temporal evolution of loop displacement is well fit by $A\sin(\omega t + \phi)\exp(-\lambda t)$ (Figure 10). Since the wavelength is much longer than the loop cross-section, the long-wavelength limit applies and the phase speed approaches the kink speed $C_K$.

진동의 특성: 서로 다른 루프들은 위상이 동기화되지 않고, 루프 꼭대기 부근에서 진폭이 최대입니다. 이는 **kink global standing mode**로 해석됩니다. 일부 루프에서는 고차 공간 고조파도 관측되었습니다.

Characteristics: Different loops are not synchronized in phase, and amplitude is maximum near the loop apex. This is interpreted as a **kink global standing mode**. Higher spatial harmonics were also observed in some loops.

수직 편광 kink 진동도 발견되었습니다 (Wang & Solanki 2004) — 수평 편광과 달리 루프 길이가 변하므로 밀도 변화를 유발하여 강한 압축 성분을 가집니다.

Vertically polarized kink oscillations were also discovered (Wang & Solanki 2004) — unlike horizontal polarization, this changes the loop length, causing density variations and a significant compressive component.

#### §3.2 Non-TRACE observations — 비-TRACE 관측

Kink 모드는 라디오파에서도 관측됩니다. 횡방향 진동이 자기장-시선 각도 $\theta$를 주기적으로 변화시켜 **gyrosynchrotron 방출**을 변조합니다:

Kink modes can also be observed in radio. Transverse oscillations periodically change the angle $\theta$ between magnetic field and line-of-sight, modulating **gyrosynchrotron emission**:

$$I_f \approx 3.3 \times 10^{-24}\frac{BN}{2\pi} \times 10^{-0.52\delta}(\sin\theta)^{-0.43+0.65\delta}\left(\frac{f}{f_B}\right)^{1.22-0.90\delta}$$

Asai et al. (2001)은 Nobeyama에서 6.6초 주기의 마이크로파 준주기 맥동을 관측하여 fast kink mode로 해석했습니다 (밀도 $4.5 \times 10^{16}$ m$^{-3}$, 루프 길이 16 Mm).

Asai et al. (2001) observed 6.6 s period microwave quasi-periodic pulsations with Nobeyama, interpreted as a fast kink mode (density $4.5 \times 10^{16}$ m$^{-3}$, loop length 16 Mm).

또한 Doppler shift의 주기적 변조를 통해서도 kink 모드를 검출할 수 있습니다 (Koutchmy et al. 1983: 300, 80, 43초 주기).

Kink modes can also be detected through periodic modulation of the Doppler shift (Koutchmy et al. 1983: 300, 80, 43 s periods).

#### §3.3 Determination of coronal magnetic fields — 코로나 자기장 결정

**이 논문의 가장 중요한 응용 결과입니다.** Low plasma-$\beta$ 한계에서 kink speed는:

**This is the most important applied result of the paper.** In the low plasma-$\beta$ limit, the kink speed is:

$$C_K \approx \left(\frac{2}{1+\rho_e/\rho_0}\right)^{1/2}C_{A0}$$

Alfvén 속도는 자기장과 밀도로 정의되므로, kink speed $C_K$를 관측적으로 측정하고 밀도비 $\rho_e/\rho_0$를 매개변수로 취하면 루프 내부의 Alfvén 속도와 자기장을 결정할 수 있습니다.

Since the Alfvén speed is defined by magnetic field and density, observationally measuring $C_K$ and taking the density ratio $\rho_e/\rho_0$ as a parameter allows determination of the internal Alfvén speed and magnetic field.

실용 공식 / Practical formula:

$$B_0 \approx 1.02 \times 10^{-12}\frac{d\sqrt{\mu n_0}\sqrt{1+n_e/n_0}}{P}$$

$B_0$: 자기장(G), $d$: footpoint 간 거리(m), $n_0$: 수밀도(m$^{-3}$), $P$: 주기(s), $\mu = 1.27$.

$B_0$: magnetic field (G), $d$: footpoint distance (m), $n_0$: number density (m$^{-3}$), $P$: period (s), $\mu = 1.27$.

1998년 7월 14일 이벤트에 적용: $\rho_e/\rho_0 = 0.1$이라 가정하면, $C_A = 756 \pm 100$ km s$^{-1}$, $B_0 = 13 \pm 9$ G를 얻습니다. 이것이 kink 진동에서 추정된 **최초의 코로나 자기장 값**입니다.

Applied to the 14 July 1998 event: assuming $\rho_e/\rho_0 = 0.1$, we get $C_A = 756 \pm 100$ km s$^{-1}$ and $B_0 = 13 \pm 9$ G. This is the **first coronal magnetic field estimate from kink oscillations**.

#### §3.4 Decay of the oscillations — 진동의 감쇠

Kink 진동의 빠른 감쇠(3~4주기 내에 소멸)의 원인은 활발한 논쟁 주제입니다. 고전적 점성이나 저항률에 의한 직접적 산일로는 관측된 감쇠 시간을 설명할 수 없습니다 ($\text{Re} \sim 10^{14}$이므로 산일 시간이 너무 김).

The rapid damping of kink oscillations (dying out within 3–4 periods) is an actively debated topic. Direct dissipation by classical viscosity or resistivity cannot explain observed decay times (since $\text{Re} \sim 10^{14}$, the dissipation time is far too long).

네 가지 감쇠 메커니즘이 제안되었습니다:

Four damping mechanisms have been proposed:

| 메커니즘 / Mechanism | 스케일링 / Scaling | 장단점 / Pros & Cons |
|---|---|---|
| **공진 흡수 / Resonant absorption** | $\tau \propto P$ (Eq. 34) | 점성 불필요, $\ell/a \approx 0.23$ 필요 / No viscosity needed, requires $\ell/a \approx 0.23$ |
| **위상 혼합 / Phase mixing** | $\tau \propto P^{2/3}$ (Eq. 35) | $\text{Re}$ 의존, 고전적 $\nu$로는 너무 느림 / Re-dependent, too slow with classical $\nu$ |
| **코로나 누설 / Coronal leakage** | $\tau \propto L^2 P$ (Eq. 36) | 대형 루프에서 너무 느림 / Too slow for large loops |
| **Footpoint 누설 / Footpoint leakage** | $\tau \propto (h/L)^{-1}$ (Eq. 37) | 관측보다 ~5배 느림 / ~5× slower than observed |

관측적 스케일링 분석 ($\tau \propto P^{1.12\pm0.36}$)은 공진 흡수($\tau \propto P$)를 약간 선호하지만, 위상 혼합도 배제할 수 없습니다. 데이터가 아직 불충분합니다.

Observational scaling analysis ($\tau \propto P^{1.12\pm0.36}$) slightly favors resonant absorption ($\tau \propto P$), but phase mixing cannot be excluded. Data remain insufficient.

#### §3.5 Alternative mechanisms — 대안적 메커니즘

**LCR 회로 모델**: 코로나 루프를 전류를 흘리는 전기 회로(인덕턴스 $\mathcal{L}$, 커패시턴스 $\mathcal{C}$)로 모사합니다. 진동 주기: $P = (2\pi/c)(\mathcal{L}\mathcal{C})^{1/2}$. 이 모델에서는 감쇠가 매우 작을 것으로 예측됩니다.

**LCR circuit model**: Models a coronal loop as an electric circuit carrying current (inductance $\mathcal{L}$, capacitance $\mathcal{C}$). Oscillation period: $P = (2\pi/c)(\mathcal{L}\mathcal{C})^{1/2}$. This model predicts very small damping.

**자기 재연결에 의한 진동**: 전류 운반 루프의 합체 과정에서 진동이 발생할 수 있으며, 최소 주기 $P = 2\pi C_{s0}\varepsilon / C_{A0}$. 이 진동은 본질적으로 비선형입니다.

**Oscillations from magnetic reconnection**: Oscillations can arise from coalescence of current-carrying loops, with minimum period $P = 2\pi C_{s0}\varepsilon / C_{A0}$. These oscillations are inherently nonlinear.

---

### §4 Sausage Oscillations of Coronal Loops — 코로나 루프의 소시지 진동

Fast magnetoacoustic sausage mode ($m=0$)는 루프 단면과 플라즈마 밀도의 축대칭 변동을 수반합니다. 주요 특성:

The fast magnetoacoustic sausage mode ($m=0$) involves axisymmetric perturbations of loop cross-section and plasma density. Key characteristics:

- **장파장 cutoff**가 존재: 파수 $k_z$가 cutoff 값 $k_{zc}$ 이하이면 trapped sausage mode가 존재하지 않습니다.
  A **long-wavelength cutoff** exists: trapped sausage mode does not exist for wavenumbers $k_z$ below the cutoff $k_{zc}$.

$$k_{zc} a = j_0 \left[\frac{(C_{s0}^2+C_{A0}^2)(C_{Ae}^2-C_{T0}^2)}{(C_{Ae}^2-C_{A0}^2)(C_{Ae}^2-C_{s0}^2)}\right]^{1/2}$$

여기서 $j_0 \approx 2.40$은 Bessel 함수 $J_0$의 첫 번째 영점입니다.
Where $j_0 \approx 2.40$ is the first zero of Bessel function $J_0$.

- **Global sausage mode의 존재 조건**: 루프가 충분히 두껍고 밀도가 높아야 합니다:
  **Condition for global sausage mode**: The loop must be sufficiently thick and dense:

$$\frac{L}{2a} < \frac{\pi}{2j_0}\frac{C_{Ae}}{C_{A0}} \approx 0.65\sqrt{\frac{\rho_0}{\rho_e}}$$

이 조건은 플레어 루프에서 충족될 수 있습니다 ($\rho_0/\rho_e > 17$ 필요).

This condition can be satisfied in flaring loops (requires $\rho_0/\rho_e > 17$).

Nakariakov et al. (2003)은 Nobeyama 관측의 14~17초 마이크로파 맥동을 global sausage mode로 해석했습니다 ($L = 25$ Mm, 루프 너비 $\sim 6$ Mm).

Nakariakov et al. (2003) interpreted 14–17 s microwave pulsations from Nobeyama as a global sausage mode ($L = 25$ Mm, loop width $\sim 6$ Mm).

---

### §5 Acoustic Oscillations of Coronal Loops — 코로나 루프의 음향 진동

#### §5.1 Global acoustic mode — 전역 음향 모드

SOHO/SUMER이 Fe XIX와 Fe XXI 코로나 방출선에서 준주기적 강도/Doppler shift 진동을 발견했습니다 (Kliem et al. 2002; Wang et al. 2002, 2003). 이 진동은 약 6 MK의 뜨거운 플라즈마와 연관됩니다.

SOHO/SUMER discovered quasi-periodic oscillations of intensity and Doppler shift in the coronal emission lines Fe XIX and Fe XXI (Kliem et al. 2002; Wang et al. 2002, 2003). These oscillations are associated with hot plasma at about 6 MK.

관측 특성:

Observational properties:

| 속성 / Property | 값 / Value |
|---|---|
| 주기 / Period | 7–31 min |
| 감쇠 시간 / Decay time | 5.7–36.8 min |
| 초기 Doppler 속도 / Initial Doppler velocity | up to 200 km s$^{-1}$ |
| 음속에 해당하는 온도 / Temperature corresponding to sound speed | $\sim 6$ MK ($C_s \approx 370$ km s$^{-1}$) |
| 강도 변동 ↔ Doppler shift 위상차 / Intensity-Doppler phase lag | $\sim 1/4$ period |

이론적으로 이 진동은 **global standing acoustic mode**로 해석됩니다:

Theoretically, these oscillations are interpreted as a **global standing acoustic mode**:

$$V_z(s,t) \propto \cos\left(\frac{\pi C_s}{L}t\right)\cos\left(\frac{\pi}{L}s\right), \quad \rho(s,t) \propto \sin\left(\frac{\pi C_s}{L}t\right)\sin\left(\frac{\pi}{L}s\right)$$

진동 주기: $P = 2L/C_s$. 빠른 감쇠는 뜨거운 플라즈마의 높은 열전도도에 의한 산일로 설명됩니다.

Oscillation period: $P = 2L/C_s$. Rapid damping is explained by dissipation from the high thermal conductivity of hot plasma.

#### §5.2 Second standing harmonics — 제2 정상 고조파

Nakariakov et al. (2004b)은 코로나 루프에 대한 충격적 에너지 입력에 대한 자연 반응으로 제2 음향 고조파가 나타남을 보였습니다:

Nakariakov et al. (2004b) showed that the second acoustic harmonics appears as a natural response to impulsive energy deposition in a coronal loop:

$$V_x(s,t) = A\cos\left(\frac{2\pi C_s}{L}t\right)\sin\left(\frac{2\pi}{L}s\right)$$

밀도 섭동은 루프 꼭대기에서 최대, 종방향 속도 섭동은 노드를 가집니다. 이 모드는 10~300초 주기의 준주기 맥동을 설명할 수 있습니다.

Density perturbation peaks at the loop apex, while longitudinal velocity perturbation has a node there. This mode can explain quasi-periodic pulsations with periods of 10–300 s.

---

### §6 Propagating Acoustic Waves — 전파하는 음향 파동

#### §6.1 Observational results — 관측 결과

코로나 파동 활동의 가장 흔한 유형 중 하나: 열린/닫힌 코로나 자기 구조에서 관측되는 느리게 전파하는 강도 교란입니다. **stroboscopic method**(시간-거리 맵)으로 검출됩니다.

One of the most common types of coronal wave activity: slow propagating intensity disturbances observed in both open and closed coronal magnetic structures. Detected via the **stroboscopic method** (time-distance maps).

최초 관측: Ofman et al. (1997, 1998b) — 코로나 홀에서 $\sim 1.9 R_\odot$ 높이에서 $\sim 9$분 주기의 편광 밝기 변동을 SOHO/UVCS로 검출했습니다.

First detection: Ofman et al. (1997, 1998b) — detected $\sim 9$ min period polarized brightness fluctuations in coronal holes at $\sim 1.9 R_\odot$ with SOHO/UVCS.

EUV 전파 교란의 관측적 특성 요약:

Summary of observational properties of EUV propagating disturbances:

- 투영 전파 속도 / Projected propagation speed: 35–165 km s$^{-1}$
- 강도 진폭 / Intensity amplitude: < 10% (밀도 < 5% / density < 5%)
- 주기 / Period: 140–420 s (준주기적 / quasi-periodic)
- 대부분 상향 전파만 검출 / Mostly only upward propagation detected
- 흑점 위 루프에서 $\sim 3$분, 비흑점 루프에서 $\sim 5$분 주기 경향
  $\sim 3$ min periods in loops above sunspots, $\sim 5$ min in non-sunspot loops

전파 방향과 속도, 압축적 특성은 이들이 **slow magnetoacoustic wave**임을 강하게 시사합니다.

Propagation direction, speed, and compressive nature strongly suggest these are **slow magnetoacoustic waves**.

#### §6.2 Theoretical modelling — 이론적 모델링

성층화된 코로나 구조에서의 종방향 파동 전파는 확장된 Burgers 방정식으로 기술됩니다:

Longitudinal wave propagation in stratified coronal structures is described by the extended Burgers equation:

$$\frac{\partial A}{\partial s} - a_1 A - a_2\frac{\partial^2 A}{\partial \xi^2} + a_3 A\frac{\partial A}{\partial \xi} = 0$$

- $a_1$: 성층화 효과 (파동 증폭) / Stratification effects (wave amplification)
- $a_2$: 열전도와 점성에 의한 산일 / Dissipation by thermal conductivity and viscosity
- $a_3$: 비선형 효과 / Nonlinear effects

이론과 관측의 비교가 양호하며, 비단열 과정(가열/냉각의 경쟁)의 효율을 추정할 수 있습니다. 관측된 파동이 운반하는 에너지는 코로나 가열에 불충분하지만, 저주파 광대역 파동 스펙트럼은 충분한 가열률을 제공할 수 있습니다.

Theory-observation comparison is satisfactory, and the efficiency of non-adiabatic processes (competition between heating and cooling) can be estimated. While the observed waves carry insufficient energy for coronal heating, a broadband low-frequency wave spectrum could provide sufficient heating rates.

#### §6.3 Propagating slow waves as a tool for coronal seismology — 전파 느린 파동을 이용한 코로나 지진학

느린 파동은 자기장선을 따라 전파하며 국소 음속으로 이동합니다. 음속은 온도의 제곱근에 비례하므로, 전파 속도의 관측은 온도와 단열 지수 $\gamma$에 대한 정보를 줍니다.

Slow waves propagate along field lines at the local sound speed. Since sound speed is proportional to the square root of temperature, observed propagation speeds give information about temperature and the adiabatic index $\gamma$.

다중 밴드 관측(171 Å과 195 Å)에서 다른 밴드의 전파 교란 상관계수가 거리에 따라 감소하는 현상은 루프의 미세 구조(다중 온도 자기 실의 다발, 또는 횡방향 온도 프로파일)를 시사합니다.

In multi-band observations (171 Å and 195 Å), the decreasing correlation coefficient of propagating disturbances in different bandpasses with distance suggests sub-resolution structuring of the loop (bundle of magnetic threads at varying temperatures, or a transverse temperature profile).

---

### §7 Propagating Fast Waves — 전파하는 빠른 파동

#### §7.1 Propagating fast waves in coronal loops — 코로나 루프에서의 빠른 파동 전파

Fast 파동의 파장은 구조의 크기보다 훨씬 짧아야 하므로, 높은 시간 분해능(~1초)이 필요합니다. TRACE/EIT의 20~30초 케이던스로는 부족합니다.

Fast wave wavelengths must be much shorter than the structure size, requiring high cadence (~1 s). TRACE/EIT's 20–30 s cadence is insufficient.

Williams et al. (2001, 2002)과 Katsiyannis et al. (2003)이 개기일식 관측에서 SECIS 장비로 빠르게 전파하는 압축 파열(wave train)을 발견했습니다: 속도 $\sim 2100$ km s$^{-1}$, 주기 $\sim 6$초.

Williams et al. (2001, 2002) and Katsiyannis et al. (2003) discovered rapidly propagating compressible wave trains during a total solar eclipse with the SECIS instrument: speed $\sim 2100$ km s$^{-1}$, period $\sim 6$ s.

분산 관계에 의해, 충격적으로 생성된 파동은 특징적인 준주기 파열로 발전하며, wavelet 분석에서 **"tadpole" 시그니처**를 나타냅니다 — 주기가 시간에 따라 감소하는 패턴입니다.

According to the dispersion relation, impulsively generated waves evolve into characteristic quasi-periodic wave trains, showing **"tadpole" signatures** in wavelet analysis — a pattern of decreasing period with time.

#### §7.2 Propagating fast kink waves in open structures — 열린 구조에서의 빠른 kink 파동 전파

Verwichte et al. (2005)가 supra-arcade에서 하향 이동하는 **tadpole** 구조의 꼬리 부분에서 횡방향 진동(fast kink wave)을 최초로 관측했습니다. 위상 속도: 200~700 km s$^{-1}$, 주기: 90~220초, 파장: 20~40 Mm.

Verwichte et al. (2005) first observed transverse oscillations (fast kink waves) in the tails of sunward-moving **tadpole** structures in supra-arcades. Phase speeds: 200–700 km s$^{-1}$, periods: 90–220 s, wavelengths: 20–40 Mm.

---

### §8 Torsional Modes — 비틀림 모드

Alfvén 파동(비틀린 실린더에서의 torsional 모드)은 비압축성이므로 밀도를 섭동하지 않고, 따라서 EUV 방출 강도를 변조하지 않습니다. 검출 방법:

Alfvén waves (torsional modes of untwisted cylinders) are incompressible, do not perturb density, and therefore do not modulate EUV emission intensity. Detection methods:

1. **Doppler shift 변동**: 장주기 파동은 분해된 Doppler shift로 검출 가능
   Doppler shift variations: long-period waves detectable as resolved Doppler shifts
2. **비열 선폭 / Non-thermal line broadening**: 단주기 파동은 발한선의 비열 성분 증가로 간접 검출
   Short-period waves detected indirectly through increased non-thermal emission line width

$$T_{\text{eff}} = T_i + \mathcal{C}\frac{m_i}{2k_B}\langle V_{\text{LOS}}^2\rangle$$

Banerjee et al. (1998), Doyle et al. (1999) 등은 코로나 홀에서 비열 선폭이 $\sim 1.2 R_\odot$까지 증가한 후 $\sim 1.5 R_\odot$에서 감소하다가 다시 증가하는 패턴을 발견했습니다. 이는 Alfvén 파동의 존재와 비선형 파동 전복(wave overturning)을 시사합니다.

Banerjee et al. (1998), Doyle et al. (1999) found non-thermal line widths in coronal holes that grow up to $\sim 1.2 R_\odot$, decrease to $\sim 1.5 R_\odot$, then grow again — suggesting the presence of Alfvén waves and nonlinear wave overturning.

n차 정상 torsional 모드의 공명 주기: $P = 2L/nC_{A0}$.
Resonant period of the $n$-th standing torsional mode: $P = 2L/nC_{A0}$.

---

### §9 Conclusions — 결론

최근 달성된 시간·공간 분해능 덕분에 코로나 파동과 진동의 **체계적(systematic)** 관측 연구가 가능해졌습니다. 1980년대 초에 확립된 자기 실린더 MHD 모드 이론이 관측 결과 해석의 이론적 기초를 제공하며, 관측과 이론의 결합이 MHD 코로나 지진학이라는 새로운 연구 분야를 열었습니다.

Recently achieved temporal and spatial resolution has made **systematic** observational studies of coronal waves and oscillations possible. The theory of MHD modes of a magnetic cylinder, developed in the early 1980s, provides the theoretical basis for interpreting observations, and the combination of observation and theory has opened the new field of MHD coronal seismology.

리뷰 범위 밖의 중요 주제: coronal Moreton/EIT wave, 코로나 밝은 점의 진동, sub-second 라디오 맥동, 비선형 효과 등.

Important topics outside the review scope: coronal Moreton/EIT waves, oscillations in coronal bright points, sub-second radio pulsations, nonlinear effects.

---

## 핵심 시사점 / Key Takeaways

1. **MHD 코로나 지진학은 코로나 플라즈마의 새로운 진단 도구이다.** 직접 측정이 불가능한 코로나 자기장, 수송 계수, 미세 구조를 파동 관측으로부터 추정할 수 있다. 이는 helioseismology와 유사하지만 세 종류의 파동 모드(Alfvén, fast, slow)를 활용하여 더 풍부한 정보를 제공한다.
   **MHD coronal seismology is a new diagnostic tool for coronal plasma.** Coronal magnetic field, transport coefficients, and fine structuring — unmeasurable directly — can be estimated from wave observations. Similar to helioseismology but exploiting three wave modes (Alfvén, fast, slow) for richer information.

2. **Kink 진동에서 최초로 코로나 자기장이 추정되었다 (~13 G).** $C_K = 2L/P$ 관계와 밀도비 가정을 통해 Alfvén 속도와 자기장을 결정하는 실용적 공식이 확립되었으며, 이는 코로나 물리학에서 오랫동안 미해결이었던 자기장 측정 문제에 대한 돌파구이다.
   **Coronal magnetic field was first estimated from kink oscillations (~13 G).** A practical formula was established using $C_K = 2L/P$ and density ratio assumptions, representing a breakthrough for the long-standing problem of coronal magnetic field measurement.

3. **Kink 진동의 빠른 감쇠 메커니즘은 여전히 논쟁 중이다.** 공진 흡수(모드 변환), 위상 혼합, 누설 모드 등이 경쟁 메커니즘이며, 관측 데이터는 아직 결정적이지 않다. 감쇠율 자체가 코로나 미세 구조($\ell/a$ 비율)를 진단하는 독립적 도구가 된다.
   **The rapid damping mechanism of kink oscillations remains debated.** Resonant absorption (mode conversion), phase mixing, and leaky modes are competing mechanisms, and observational data are not yet conclusive. The damping rate itself serves as an independent diagnostic of coronal fine structure ($\ell/a$ ratio).

4. **Edwin & Roberts (1983)의 분산 관계는 코로나 MHD 파동 이론의 기초이다.** 자기 실린더 모델에서 도출된 이 분산 관계가 kink, sausage, 모든 MHD 모드의 존재 조건과 특성을 결정한다. Figure 3의 분산 다이어그램은 관측 해석의 필수적 참조 프레임이다.
   **The Edwin & Roberts (1983) dispersion relation is the foundation of coronal MHD wave theory.** This dispersion relation from the magnetic cylinder model determines the existence conditions and properties of all MHD modes. Figure 3's dispersion diagram is the essential reference frame for observation interpretation.

5. **Sausage 모드는 장파장 cutoff를 가지며, 충분히 두껍고 밀도가 높은 루프에서만 global 모드가 존재한다.** 이 제약 조건($L/2a < 0.65\sqrt{\rho_0/\rho_e}$)은 마이크로파 맥동의 해석과 루프의 물리적 특성 추정에 핵심적이다.
   **Sausage modes have a long-wavelength cutoff, and the global mode exists only in sufficiently thick, dense loops.** This constraint ($L/2a < 0.65\sqrt{\rho_0/\rho_e}$) is essential for interpreting microwave pulsations and estimating loop physical properties.

6. **전파하는 느린 파동은 코로나에서 가장 흔한 파동 현상이며, Burgers 방정식으로 기술된다.** 성층화, 산일, 비선형 효과의 경쟁을 통해 파동 진폭의 진화가 결정되며, 관측과 이론의 비교로 비단열 과정의 효율을 추정할 수 있다.
   **Propagating slow waves are the most common wave phenomenon in the corona, described by the Burgers equation.** Wave amplitude evolution is determined by competition among stratification, dissipation, and nonlinearity; comparison of observation and theory allows estimation of non-adiabatic process efficiency.

7. **Fast 파동의 분산 진화는 특징적인 "tadpole" wavelet 시그니처를 만든다.** 충격적으로 생성된 fast magnetoacoustic 펄스가 분산에 의해 주기 변조된 파열로 발전하며, 이 형태가 루프의 밀도 프로파일 정보를 담고 있다.
   **Dispersive evolution of fast waves creates characteristic "tadpole" wavelet signatures.** Impulsively generated fast magnetoacoustic pulses evolve into period-modulated wave trains through dispersion, and this shape encodes information about the loop's density profile.

8. **Torsional 모드는 관측적으로 가장 검출하기 어렵지만, 코로나 가열의 핵심 후보이다.** 비압축성이어서 강도를 변조하지 않고, Doppler shift나 비열 선폭을 통해서만 간접 검출 가능하다. 먼 거리까지 에너지를 운반할 수 있어 열린 자기 구조의 가열에 중요하다.
   **Torsional modes are the most observationally elusive but are key candidates for coronal heating.** Being incompressible, they do not modulate intensity and are detectable only indirectly via Doppler shift or non-thermal line broadening. They can transport energy over long distances, important for heating open magnetic structures.

---

## 수학적 요약 / Mathematical Summary

### 코로나 MHD 파동의 완전한 체계 / Complete Framework of Coronal MHD Waves

**1단계: 평형 / Step 1: Equilibrium**

자기 실린더 ($r < a$: 내부, $r > a$: 외부):
Magnetic cylinder ($r < a$: inside, $r > a$: outside):

$$p_0 + \frac{B_0^2}{2\mu_0} = p_e + \frac{B_e^2}{2\mu_0}$$

**2단계: 선형 섭동 방정식 / Step 2: Linearized Perturbation Equations**

$$D\frac{d}{dr}(r\xi_r) = (C_A^2+C_s^2)(\omega^2-C_T^2 k_z^2)\left(\kappa^2+\frac{m^2}{r^2}\right)r\,\delta P_{\text{tot}}$$

$$\frac{d}{dr}(\delta P_{\text{tot}}) = \rho_0(\omega^2-C_A^2 k_z^2)\xi_r$$

여기서 / Where:
$$D = \rho_0(C_A^2+C_s^2)(\omega^2-C_A^2 k_z^2)(\omega^2-C_T^2 k_z^2)$$

$$\kappa^2(\omega) = -\frac{(\omega^2-C_s^2 k_z^2)(\omega^2-C_A^2 k_z^2)}{(C_s^2+C_A^2)(\omega^2-C_T^2 k_z^2)}$$

**3단계: 분산 관계 / Step 3: Dispersion Relation**

경계 조건(총 압력과 변위의 연속성)을 적용:
Applying boundary conditions (continuity of total pressure and displacement):

$$\rho_e(\omega^2-k_z^2 C_{Ae}^2)\kappa_0\frac{I_m'(\kappa_0 a)}{I_m(\kappa_0 a)} + \rho_0(k_z^2 C_{A0}^2-\omega^2)\kappa_e\frac{K_m'(\kappa_e a)}{K_m(\kappa_e a)} = 0$$

**4단계: 코로나 지진학 응용 / Step 4: Coronal Seismology Application**

- Kink 진동 ($m=1$): $C_K = 2L/P$ → $B_0 = \sqrt{\mu_0\rho_0}\,C_{A0} \approx \sqrt{2\mu_0 L}/P \cdot \sqrt{\rho_0(1+\rho_e/\rho_0)}$
- 감쇠로부터 $\ell/a$ 추정: $\ell/a = (2/\pi)(P/\tau)[(\rho_0+\rho_e)/(\rho_0-\rho_e)]$
- Sausage 모드: $P_{\text{GSM}} = 2L/C_{\text{ph}}$, 존재 조건으로 밀도비 하한 추정
  Sausage mode: $P_{\text{GSM}} = 2L/C_{\text{ph}}$, lower bound on density ratio from existence condition
- Acoustic 모드: $P = 2L/C_s$, 음속으로부터 온도 추정
  Acoustic mode: $P = 2L/C_s$, temperature estimation from sound speed

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1942  Alfvén — MHD 파동 이론 제안
         MHD wave theory proposed
  |
1970  Uchida — MHD 코로나 지진학 개념 최초 제안
         First concept of MHD coronal seismology
  |
1975  Zaitsev & Stepanov — leaky mode 해석
         Leaky mode analysis
  |
1981  Roberts — 자기 슬랩/실린더의 MHD 모드
         MHD modes of magnetic slab/cylinder
  |
1983  Edwin & Roberts — 완전한 분산 다이어그램
         Complete dispersion diagram
  |     Heyvaerts & Priest — Alfvén 파동 위상 혼합 제안
         Alfvén wave phase mixing proposed
  |
1984  Roberts et al. — 국소 코로나 지진학 제안, fast wave train 예측
         Local coronal seismology proposed, fast wave train predicted
  |
1995  SOHO 발사 / SOHO launched
  |
1997  Ofman et al. — 코로나 홀에서 전파 느린 파동 최초 관측
         First propagating slow wave detection in coronal holes
  |
1998  TRACE 발사 → kink 진동 최초 관측 (7월 14일)
         TRACE launched → First kink oscillation observation (July 14)
  |     Thompson et al. — EIT 파동 발견
         EIT wave discovery
  |
1999  Nakariakov et al. — kink 진동의 정량적 분석
         Quantitative analysis of kink oscillations
  |
2001  Nakariakov & Ofman — kink 진동으로 자기장 최초 추정 (13 G)
         First magnetic field estimate from kink oscillations (13 G)
  |     Williams et al. — 일식에서 fast wave train 최초 관측
         First fast wave train observation in eclipse
  |
2002  Ruderman & Roberts — 공진 흡수 감쇠 이론 (Eq. 34)
         Resonant absorption damping theory (Eq. 34)
  |     Wang et al. — SUMER 음향 진동 발견
         SUMER acoustic oscillation discovery
  |
2003  Nakariakov et al. — global sausage mode 관측
         Global sausage mode observation
  |
>>> 2005  Nakariakov & Verwichte — 이 리뷰 <<<
           This review
  |
  v
[향후: SDO/AIA (2010), 고분해능 관측, 비선형 효과]
[Future: SDO/AIA (2010), high-resolution observations, nonlinear effects]
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 연결 논문 / Connected Paper | 관계 / Relationship | 영향 / Impact |
|---|---|---|
| Edwin & Roberts (1983) | 이론적 기초 — 자기 실린더 분산 관계 / Theoretical foundation — magnetic cylinder dispersion relation | 이 리뷰의 모든 모드 해석이 이 분산 관계에 기반 / All mode interpretations in this review are based on this |
| Roberts et al. (1984) | 국소 코로나 지진학 제안, fast wave train 예측 / Local coronal seismology proposal, fast wave train prediction | §3.3, §7의 이론적 출발점 / Theoretical starting point for §3.3 and §7 |
| Nakariakov et al. (1999) | kink 진동의 최초 관측적 정량 분석 / First observational quantitative analysis of kink oscillations | 이 리뷰의 핵심 관측 결과, 감쇠 논쟁의 출발점 / Core observational result of this review, starting point of damping debate |
| Nakariakov & Ofman (2001) | kink 진동에서 자기장 최초 추정 / First magnetic field estimate from kink oscillations | §3.3의 중심 결과 / Central result of §3.3 |
| Ruderman & Roberts (2002) | 공진 흡수 감쇠 시간 해석적 공식 / Analytical formula for resonant absorption damping time | §3.4의 핵심 이론적 결과 / Key theoretical result of §3.4 |
| LRSP #1 Wood (2004) | 태양풍과 코로나 파동의 연결 / Connection between solar wind and coronal waves | 코로나 가열, 태양풍 가속 동기 제공 / Motivation for coronal heating and solar wind acceleration |
| LRSP #2 Miesch (2005) | 태양 내부 역학 (helioseismology와의 유사성) / Solar interior dynamics (analogy with helioseismology) | MHD 코로나 지진학 ↔ helioseismology 비교의 배경 / Context for MHD coronal seismology ↔ helioseismology comparison |

---

## 참고문헌 / References

- Aschwanden, M. J., Fletcher, L., Schrijver, C. J., & Alexander, D. (1999). "Coronal Loop Oscillations Observed with the Transition Region and Coronal Explorer." *ApJ*, 520, 880.
- Edwin, P. M. & Roberts, B. (1983). "Wave propagation in a magnetic cylinder." *Sol. Phys.*, 88, 179.
- Heyvaerts, J. & Priest, E. R. (1983). "Coronal heating by phase-mixed shear Alfvén waves." *A&A*, 117, 220.
- Nakariakov, V. M., Ofman, L., DeLuca, E. E., Roberts, B., & Davila, J. M. (1999). "TRACE observation of damped coronal loop oscillations." *Science*, 285, 862.
- Nakariakov, V. M. & Ofman, L. (2001). "Determination of the coronal magnetic field by coronal loop oscillations." *A&A*, 372, L53.
- Roberts, B., Edwin, P. M., & Benz, A. O. (1984). "On coronal oscillations." *ApJ*, 279, 857.
- Ruderman, M. S. & Roberts, B. (2002). "The damping of coronal loop oscillations." *ApJ*, 577, 475.
- Uchida, Y. (1970). "Diagnosis of coronal magnetic structure by flare-associated hydromagnetic disturbances." *PASJ*, 22, 341.
