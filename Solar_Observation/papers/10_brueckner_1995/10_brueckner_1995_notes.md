---
title: "The Large Angle Spectroscopic Coronagraph (LASCO)"
authors: "G.E. Brueckner, R.A. Howard, M.J. Koomen, C.M. Korendyke, D.J. Michels, J.D. Moses, D.G. Socker, K.P. Dere, P.L. Lamy, A. Llebaria, M.V. Bout, R. Schwenn, G.M. Simnett, D.K. Bedford, C.J. Eyles"
year: 1995
journal: "Solar Physics, Vol. 162, pp. 357–402"
doi: "10.1007/BF00733434"
topic: Solar Observation
tags: [LASCO, SOHO, coronagraph, C1, C2, C3, CME, Thomson scattering, Fabry-Perot, externally occulted, internally occulted, Lyot, stray light, pB, white light corona, NRL]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 10. The Large Angle Spectroscopic Coronagraph (LASCO) / 대각도 분광 코로나그래프 (LASCO)

---

## 1. Core Contribution / 핵심 기여

LASCO(Large Angle Spectroscopic Coronagraph)는 SOHO 위성에 탑재된 **3중 코로나그래프 시스템**으로, C1 (1.1–3 $R_\odot$), C2 (1.5–6 $R_\odot$), C3 (3.7–30 $R_\odot$)의 세 망원경이 겹치는 시야각(FOV)을 통해 태양 표면 바로 위부터 30 태양 반경까지의 코로나를 연속적으로 관측한다. C1은 **거울형 내부 차폐(internally occulted) Lyot 코로나그래프**로 Fabry-Perot 간섭계를 장착하여 최초의 우주 기반 코로나 분광 관측 능력을 제공하였다. C2와 C3는 **외부 차폐(externally occulted)** 코로나그래프로, 특히 C2의 톱니형(serrated) 외부 차폐판은 이전 우주 코로나그래프 대비 **1차수(order of magnitude) 이상** 개선된 산란광 억제를 달성하였다. LASCO는 미국 해군연구소(NRL), 프랑스 마르세유 LAS, 독일 린다우 MPI, 영국 버밍엄 대학의 국제 공동 개발 기기이며, EIT와 LEB(LASCO/EIT Electronics Box) 전자부를 공유한다. 1996년 과학 관측 시작 이후 **40,000개 이상의 CME**와 **4,000개 이상의 혜성**을 발견하여 코로나 물리학과 태양풍 연구에 혁명적 기여를 하였다.

LASCO (Large Angle Spectroscopic Coronagraph) is a **triple coronagraph system** on the SOHO satellite, covering the solar corona from just above the solar surface out to 30 solar radii through three telescopes with overlapping fields of view: C1 (1.1–3 $R_\odot$), C2 (1.5–6 $R_\odot$), and C3 (3.7–30 $R_\odot$). C1 is a **mirror-based internally occulted Lyot coronagraph** equipped with a Fabry-Perot interferometer, providing the first spaceborne spectroscopic coronagraph capability. C2 and C3 are **externally occulted** coronagraphs; notably, C2's serrated external occulter achieves stray light suppression improved by **more than an order of magnitude** over previous spaceborne coronagraphs. LASCO was jointly developed by NRL (USA), LAS Marseille (France), MPI Lindau (Germany), and the University of Birmingham (UK), sharing the LEB (LASCO/EIT Electronics Box) with EIT. Since the start of science operations in 1996, LASCO has discovered **over 40,000 CMEs** and **more than 4,000 comets**, revolutionizing coronal physics and solar wind research.

---

## 2. Reading Notes / 읽기 노트

### §1 Introduction (pp. 357–358)

논문은 코로나그래프의 역사적 발전을 추적하며 시작한다. Lyot(1930)가 최초의 지상 코로나그래프를 발명하였고, Tousey(1963)가 로켓에서 최초의 외부 차폐 코로나그래프를 비행시켰다. 이후 OSO-7(1971), Skylab(1973), P78-1/Solwind(1979), SMM(1980)에서 우주 코로나그래프가 운용되었다. 이전 기기들의 주요 한계는 다음과 같다: (1) 외부 차폐 방식은 내부 시야 분해능이 좋지 않고, (2) 렌즈 기반 Lyot 코로나그래프는 색수차와 렌즈 결함 문제가 있으며, (3) 분광 관측 능력이 없었다. LASCO는 이러한 한계를 세 가지 겹치는 FOV를 가진 3중 망원경 시스템으로 극복하며, 특히 C1에 거울형 Lyot 설계와 Fabry-Perot 간섭계를 결합하여 최초의 우주 기반 코로나 분광 관측을 실현한다.

The paper begins by tracing the historical development of coronagraphs. Lyot (1930) invented the first ground-based coronagraph, and Tousey (1963) flew the first externally occulted coronagraph on a rocket. Subsequently, space coronagraphs were operated on OSO-7 (1971), Skylab (1973), P78-1/Solwind (1979), and SMM (1980). Key limitations of previous instruments include: (1) externally occulted designs have poor inner-field resolution, (2) lens-based Lyot coronagraphs suffer from chromatic aberration and lens defects, and (3) no spectroscopic capability existed. LASCO overcomes these limitations with a triple-telescope system with overlapping FOVs, and particularly realizes the first spaceborne coronagraph spectroscopy by combining a mirror Lyot design with a Fabry-Perot interferometer in C1.

### §2 Optical Design (pp. 358–366)

#### §2.1 C1 — Mirror Lyot Coronagraph / C1 — 거울형 Lyot 코로나그래프

C1은 전통적인 렌즈 Lyot 코로나그래프의 거울 버전이다(Fig. 2 참조). 광학 경로는 다음과 같다:

C1 is a mirror version of the traditional lens Lyot coronagraph (see Fig. 2). The optical path is as follows:

1. **A0 (입구 조리개 / Entrance aperture)**: 직경 4.7 cm. 태양광이 입사.
   - Diameter 4.7 cm. Sunlight enters.
2. **M1 (주경 / Primary mirror)**: 비축 포물경(off-axis parabolic), 초점 거리 75 cm. 태양상을 형성.
   - Off-axis parabolic mirror, 75 cm focal length. Forms the solar image.
3. **M2 (시야 거울 / Field mirror)**: 구면경(spherical), 중앙에 구멍이 있음(1.1 $R_\odot$에 해당). 코로나 빛은 반사하고, 광구 빛은 구멍을 통해 빠져나감.
   - Spherical mirror with central hole (equivalent to 1.1 $R_\odot$). Reflects coronal light; photospheric light exits through the hole.
4. **M4 (거부경 / Rejection mirror)**: M2 구멍을 통과한 광구 빛을 기기 외부로 배출.
   - Ejects photospheric light passing through M2 hole out of the instrument.
5. **M3 (재영상 거울 / Reimaging mirror)**: 비축 포물경, M2로부터 반사된 코로나 상을 재결상.
   - Off-axis parabolic, reimages the corona reflected from M2.
6. **A1 (Lyot stop)**: M1의 회절 패턴을 차단하는 핵심 산란광 억제 요소.
   - Key stray light suppression element blocking the diffraction pattern of M1.
7. **Fabry-Perot 간섭계**: Lyot stop 뒤의 평행광 경로에 배치.
   - Placed in the collimated beam behind the Lyot stop.
8. **TL (텔레포토 렌즈 / Telephoto lens)**: 5매 렌즈, 유효 초점 거리 76.8 cm. 최종 상을 CCD에 형성.
   - 5-element lens, effective focal length 76.8 cm. Forms the final image on the CCD.

**C1 핵심 사양 / C1 key specifications**:
- FOV: 1.1–3 $R_\odot$ (1024² CCD에 내접 / inscribed in 1024² CCD)
- 픽셀 스케일 / Pixel scale: 5.6"/pixel
- **동적 영상(Dynamic imaging)**: M1 위에 피에조(piezo) 장치가 장착되어 1.4" 간격으로 틸트 가능. 4회 노출로 2배 분해능 달성 (Nyquist 한계 극복).
  - M1 mounted on piezo actuators for 1.4" step tilts. 4 exposures achieve 2× resolution (overcoming Nyquist limit).

#### §2.2 C2 and C3 — Externally Occulted Coronagraphs / C2와 C3 — 외부 차폐 코로나그래프

C2와 C3는 고전적인 외부 차폐 설계를 따른다(Fig. 3 참조). 광학 경로는:

C2 and C3 follow the classical externally occulted design (see Fig. 3). The optical path is:

1. **D1 (외부 차폐판 / External occulter)**: 태양 직사광을 차단.
   - Blocks direct sunlight.
2. **A1 (입구 조리개 / Entrance aperture)**: D1 가장자리의 회절광을 제한.
   - Limits diffracted light from D1 edge.
3. **O1 (대물렌즈 / Objective lens)**: 코로나 상을 형성.
   - Forms the coronal image.
4. **D2 (시야 차단판 / Field stop)**: O1이 만든 내부 태양상을 추가 차단(internal occulter).
   - Additional occulting of the internal solar image formed by O1.
5. **O2 (시야 렌즈 / Field lens)**: 빛을 릴레이 렌즈로 전달.
   - Relays light to the relay lens.
6. **A3 (Lyot stop)**: A1의 회절 패턴을 차단.
   - Blocks diffraction pattern of A1.
7. **O3 (릴레이 렌즈 / Relay lens)**: 최종 코로나 상을 CCD에 형성.
   - Forms the final coronal image on the CCD.
8. **F/P (필터/편광기 / Filters/Polarizers)**: O2와 O3 사이에 위치.
   - Located between O2 and O3.

**C2 외부 차폐판 설계 / C2 external occulter design**: 160개의 다이아몬드 가공 나사산(threads)을 가진 테이퍼 원뿔형(tapered cone) 톱니 구조. 이 혁신적인 톱니형(serrated) 설계는 회절에 의한 산란광을 극적으로 감소시킨다.

C2's external occulter features a tapered cone with 160 diamond-machined threads in a serrated design. This innovative serrated structure dramatically reduces diffraction-induced stray light.

**C3 외부 차폐판 설계 / C3 external occulter design**: 스핀들 위에 3개의 원판(disks)을 배치한 다중 차폐 설계.

C3's external occulter uses a multi-disk design with 3 disks mounted on a spindle.

#### Table I — Three Coronagraph Summary / 3중 코로나그래프 요약

| 파라미터 / Parameter | C1 | C2 | C3 |
|---|---|---|---|
| FOV ($R_\odot$) | 1.1–3 | 1.5–6 | 3.7–30 |
| 차폐 방식 / Occulting type | Internal | External | External |
| 분광 / Spectral capability | Fabry-Perot | Broadband | Broadband |
| 광학 / Optics | Mirror (반사경) | Lens (굴절) | Lens (굴절) |
| 픽셀 스케일 / Pixel scale (") | 5.6 | 11.4 | 56 |
| 밝기 범위 상한 / Brightness upper ($B_\odot$) | $2 \times 10^{-5}$ | $2 \times 10^{-7}$ | $3 \times 10^{-9}$ |
| 밝기 범위 하한 / Brightness lower ($B_\odot$) | $2 \times 10^{-8}$ | $5 \times 10^{-10}$ | $1 \times 10^{-11}$ |

#### §2.3 Stray Light Performance / 산란광 성능 (Fig. 4)

**Fig. 4**는 LASCO의 핵심 성능 지표를 보여주는 그림이다: C1/C2/C3의 측정된 산란광 수준을 K+F 코로나(전자 산란 + 먼지 산란) 및 이전 코로나그래프(OSO-7, Skylab, Solwind, SMM)와 비교한다.

**Fig. 4** is the key performance figure: it compares the measured stray light levels of C1/C2/C3 against the K+F corona (electron scattering + dust scattering) and previous coronagraphs (OSO-7, Skylab, Solwind, SMM).

- **C2/C3**: 이전 코로나그래프 대비 **최소 1차수(order of magnitude) 이상** 낮은 산란광 달성. C2는 2.2 $R_\odot$까지 코로나가 산란광 위에서 검출 가능.
  - At least **one order of magnitude lower** stray light than previous coronagraphs. C2 corona detectable above stray light down to 2.2 $R_\odot$.
- **C1**: Fe XIV 방출선이 1.8 $R_\odot$에서 산란광보다 밝음. 3 $R_\odot$에서 산란광 수준 ~$10^{-8}$ $B_\odot$.
  - Fe XIV emission line brighter than stray light at 1.8 $R_\odot$. Stray light level ~$10^{-8}$ $B_\odot$ at 3 $R_\odot$.

#### §2.4 Spatial Resolution / 공간 분해능 (Figs. 5, 6)

- **C1**: 픽셀 크기(5.6")에 의해 제한됨. 분해능 ~11" (2픽셀). 동적 영상(dynamic imaging)으로 개선 가능.
  - Pixel-limited at 5.6". Resolution ~11" (2 pixels). Improvable by dynamic imaging.
- **C2**: 내부 가장자리에서 광학 분해능 ~20", 외부에서는 픽셀 제한(~23") (Fig. 5).
  - Optical resolution ~20" at inner edge, pixel-limited (~23") at outer edge (Fig. 5).
- **C3**: 내부에서 ~200", 외부에서 픽셀 제한(~112") (Fig. 6).
  - ~200" at inner edge, pixel-limited (~112") at outer edge (Fig. 6).

### §3 Detailed Design: C1 (pp. 366–370)

#### 거울 제작 / Mirror Fabrication

M1과 M3는 Zerodur 기판의 비축 포물경(off-axis parabolic mirror)이다:

M1 and M3 are off-axis parabolic mirrors on Zerodur substrates:

- **마이크로 거칠기(Micro-roughness)**: M1 = 0.084 nm rms, M3 = 0.104 nm rms. 이는 극도로 높은 표면 품질이다.
  - M1 = 0.084 nm rms, M3 = 0.104 nm rms. This is extremely high surface quality.
- **코팅 / Coating**: Al + SiO₂ (반사율 92% / 92% reflectivity)
- **M1 피에조 마운트 / M1 piezo mount**: ±1 arcmin 틸트 범위, 1.4" 스텝. CCD 픽셀 크기의 1/4에 해당하여 dynamic imaging에 활용.
  - ±1 arcmin tilt range, 1.4" steps. Corresponds to 1/4 of CCD pixel size for dynamic imaging.

**M2 (Field mirror)**: 스테인리스 강(stainless steel) 기판, 중앙 구멍에 방사형 ND 필터(radial neutral density filter) 장착. 이 ND 필터는 광구에 가까운 밝은 영역의 동적 범위를 확장하는 역할을 한다.

**M2 (Field mirror)**: Stainless steel substrate with radial neutral density (ND) filter at the central hole. This ND filter extends the dynamic range in the bright region near the photosphere.

#### 필터/편광기 휠 / Filter/Polarizer Wheels

**F wheel (필터 휠)**: Na I, Fe XIV (530.3 nm), Ca XV (564.9 nm), Fe X (637.4 nm), Orange continuum

**P wheel (편광기 휠)**: 3개 편광기(0°, ±60°), Clear, Hα (656.3 nm)

편광도(pB, polarized brightness) 관측은 3개 편광기의 연속 촬영을 통해 수행되며, 이는 코로나 전자 밀도를 결정하는 핵심 관측이다.

Polarized brightness (pB) observations are performed through consecutive imaging with 3 polarizers, which is the key observation for determining coronal electron density.

#### Dynamic Imaging (Fig. 9) / 동적 영상

M1의 피에조 틸트를 이용하여 4회 노출(각각 1.4" 이동)을 수행한 후 합성하면, 유효 분해능이 ~5.6"에서 ~2.8"로 개선된다. Fig. 9는 이 기법의 시연 결과를 보여준다.

By performing 4 exposures (each shifted by 1.4" via M1 piezo tilt) and combining them, effective resolution improves from ~5.6" to ~2.8". Fig. 9 demonstrates this technique.

### §4 Fabry-Perot Interferometer (pp. 370–373)

C1의 Fabry-Perot 간섭계는 **피에조 구동(piezo-scanned)**, **정전용량 모니터링(capacitance-monitored)** 방식이다. Lyot stop 뒤의 평행광(collimated beam) 경로에 위치하며, 다양한 코로나 방출선의 도플러 속도를 측정할 수 있다.

C1's Fabry-Perot interferometer is **piezo-scanned** and **capacitance-monitored**. Positioned in the collimated beam behind the Lyot stop, it can measure Doppler velocities of various coronal emission lines.

#### Table III — Fabry-Perot Spectral Lines / Fabry-Perot 분광선

| 이온/선 / Ion/Line | 파장 / Wavelength (nm) | FWHM (nm) | 속도 범위 / Velocity range (km/s) |
|---|---|---|---|
| Fe XIV (녹색선 / Green line) | 530.3 | 0.065 | ±430 |
| Ca XV | 564.9 | 0.059 | ±485 |
| Na I (D선 / D line) | 589.0 | 0.072 | — |
| Fe X (적색선 / Red line) | 637.4 | 0.085 | ±500 |
| Hα | 656.3 | 0.104 | ±512 |
| 백색광 / White light | 530–640 | 0.065 | — |

Fe XIV 530.3 nm은 코로나에서 가장 밝은 방출선으로, 코로나 온도 ~2 MK를 추적한다. Na I D선은 먼지 코로나(F-corona)의 프라운호퍼 선 관측에 사용된다.

Fe XIV 530.3 nm is the brightest coronal emission line, tracing coronal temperatures ~2 MK. The Na I D line is used for observing the Fraunhofer line in the dust corona (F-corona).

#### OCC (Optical Control Channel) / 광학 제어 채널

Fabry-Perot의 간격(gap) 드리프트를 실시간으로 보정하기 위해 OCC가 사용된다. 이는 기준 레이저 파장을 모니터링하여 피에조 전압을 피드백 제어한다.

The OCC is used to correct Fabry-Perot gap drift in real time. It monitors a reference laser wavelength and provides feedback control to the piezo voltage.

#### Table IV — Expected Velocity Precision / 예상 속도 정밀도

논문의 Table IV는 다양한 높이에서의 속도 측정 정밀도를 제시한다. 예를 들어, Fe XIV 530.3 nm에서:

Table IV presents velocity measurement precision at various heights. For example, at Fe XIV 530.3 nm:

| 높이 / Height ($R_\odot$) | 속도 정밀도 / Velocity precision (km/s) |
|---|---|
| 1.1 | ~5 |
| 1.5 | ~15 |
| 2.0 | ~50 |

이는 코로나 루프 내 플라즈마 흐름(10–100 km/s)과 CME 초기 가속(수십 km/s)을 검출하기에 충분한 정밀도이다.

This precision is sufficient to detect plasma flows in coronal loops (10–100 km/s) and early CME acceleration (tens of km/s).

### §5 Detailed Design: C2 (pp. 373–378)

#### External Occulter / 외부 차폐판

C2의 외부 차폐판은 LASCO의 핵심 기술 혁신 중 하나이다:

C2's external occulter is one of LASCO's key technological innovations:

- **톱니형 원뿔 설계 / Serrated cone design**: 160개의 다이아몬드 가공 나사산(diamond-machined threads)으로 이루어진 테이퍼 원뿔. 톱니 구조는 회절 패턴을 무작위화(randomize)하여 특정 방향으로의 산란광 집중을 방지한다.
  - Tapered cone with 160 diamond-machined threads. The serrated structure randomizes the diffraction pattern, preventing concentration of stray light in specific directions.
- **A1에서의 산란광 억제 / Stray light rejection at A1**: $1.5 \times 10^{-5}$ $B_\odot$. 이는 입구 조리개에서의 1차 회절 억제 수준이다.
  - $1.5 \times 10^{-5}$ $B_\odot$. This is the first-order diffraction suppression level at the entrance aperture.

#### Objective Lens O1 / 대물렌즈 O1

이중렌즈(doublet), 반사 방지 코팅(anti-reflection coated), 광학 접합(optical contact), **초정밀 연마(superpolished)**. 렌즈 표면의 미세 결함에 의한 산란을 최소화하기 위해 초정밀 연마가 필수적이다.

Doublet, anti-reflection coated, optical contact, **superpolished**. Superpolishing is essential to minimize scattering from microscopic surface defects on the lens.

#### Internal Occulter D2 / 내부 차폐판 D2

O1이 형성한 태양상보다 **10% 과잉 차폐(over-occultation)**. 이는 차폐판 위치 오차에 대한 여유를 제공하고, 가장자리 회절을 추가로 억제한다.

**10% over-occultation** relative to the solar image formed by O1. This provides margin for occulter positioning errors and further suppresses edge diffraction.

#### Fig. 11 — Vignetting Function / 비네팅 함수

C2의 내부 가장자리 근처에서는 비네팅(vignetting)이 발생한다. 비네팅 함수는 약 1.5–2.0 $R_\odot$ 범위에서 0에서 1로 급격히 증가하며, 이 보정은 데이터 처리 시 반드시 적용해야 한다.

Vignetting occurs near C2's inner edge. The vignetting function increases sharply from 0 to 1 in the range ~1.5–2.0 $R_\odot$, and this correction must be applied during data processing.

#### Fig. 13 — Mueller Matrix / Mueller 행렬

C2의 편광 특성을 나타내는 Mueller 행렬 항이 제시된다. 이는 pB(polarized brightness) 관측의 정량적 보정에 필수적이다. 기기 편광(instrumental polarization)은 수 % 수준이며, 이를 보정해야 정확한 코로나 전자 밀도를 도출할 수 있다.

Mueller matrix terms characterizing C2's polarization properties are presented. These are essential for quantitative correction of pB (polarized brightness) observations. Instrumental polarization is at the few-percent level, and correcting for it is necessary to derive accurate coronal electron densities.

#### Fig. 14 — C2 Stray Light Profile / C2 산란광 프로파일

C2의 산란광 프로파일을 K+F 코로나와 비교한 결과, **2.2 $R_\odot$까지 코로나가 산란광 위에서 검출 가능**하다. 이는 외부 차폐 코로나그래프로서는 매우 우수한 성능이다.

Comparing C2's stray light profile with the K+F corona, the **corona is detectable above stray light down to 2.2 $R_\odot$**. This is excellent performance for an externally occulted coronagraph.

### §7 C3 Coronagraph (pp. 380–385)

C3는 가장 넓은 시야각(3.7–30 $R_\odot$)을 가진 코로나그래프로, 태양에서 먼 외부 코로나와 태양풍 구조를 관측한다.

C3 has the widest field of view (3.7–30 $R_\odot$), observing the far outer corona and solar wind structures.

#### Table V — C3 Key Parameters / C3 주요 파라미터

| 파라미터 / Parameter | 값 / Value |
|---|---|
| 전체 길이 / Total length | 889 mm |
| 유효 초점 거리 / Effective focal length | 77.6 mm |
| F-number | f/9.3 |
| 픽셀 스케일 / Pixel scale | 56"/pixel |
| 산란광 수준 / Stray light level | $1 \times 10^{-12}$ $B_\odot$ |
| 입구 조리개 A0 / Entrance aperture A0 | 110 mm |
| 유효 조리개 A1 / Effective aperture A1 | 9.6 mm |

C3의 산란광 수준 $10^{-12}$ $B_\odot$는 30 $R_\odot$에서의 코로나/태양풍 밝기(~$10^{-11}$ $B_\odot$)보다 한 차수 낮아, 이 극도로 먼 영역까지 관측이 가능하다.

C3's stray light level of $10^{-12}$ $B_\odot$ is one order of magnitude below the corona/solar wind brightness at 30 $R_\odot$ (~$10^{-11}$ $B_\odot$), enabling observations to this extremely distant region.

#### External Occulter / 외부 차폐판

3개의 원판(disks)이 스핀들 위에 장착된 다중 차폐 설계. C2의 톱니형과 다른 고전적인 다중 디스크 방식이지만, 넓은 FOV에 적합하다.

Multi-disk design with 3 disks on a spindle. Unlike C2's serrated design, this is a classical multi-disk approach, but suited for the wide FOV.

#### Objective / 대물렌즈

**단일렌즈(singlet)** — 반사 방지 코팅 없음(no AR coating). 이는 의도적인 설계로, AR 코팅에 의한 고스트 이미지(ghost)를 방지하기 위함이다. 과잉 차폐(over-occultation)는 3.7 $R_\odot$까지이다.

**Singlet** — no AR coating. This is intentional, to prevent ghost images caused by AR coating. Over-occultation extends to 3.7 $R_\odot$.

#### Table VI — C3 Color Filters / C3 색 필터

| 필터 / Filter | 파장 범위 / Wavelength range (nm) |
|---|---|
| Blue | 420–520 |
| Orange | 540–640 |
| Light Red | 620–780 |
| Deep Red | 730–835 |
| Hα | 656.3 |
| IR | 860–1050 |
| Clear | 400–850 |
| Polarizers | — |

다양한 색 필터는 K-코로나(전자 산란, 연속 스펙트럼)와 F-코로나(먼지 산란, 프라운호퍼 흡수선)를 분리하는 데 사용된다. Blue 필터에서는 K-코로나가 우세하고, IR 필터에서는 F-코로나가 우세하다.

The various color filters are used to separate the K-corona (electron scattering, continuum spectrum) from the F-corona (dust scattering, Fraunhofer absorption lines). K-corona dominates in the Blue filter, while F-corona dominates in the IR filter.

### §8 Coalignment (pp. 385–386)

세 코로나그래프는 하나의 정밀 가공 알루미늄 케이스(339.5 × 323 × 1362 mm)에 통합되어 있다. C1은 케이스의 한쪽 절반에, C2와 C3는 다른 쪽 절반에 배치된다. 세 망원경 간의 정렬 정밀도는 **60 arcsec** 이내이다. 이 정도의 정렬은 겹치는 FOV 영역에서 세 기기의 관측을 매끄럽게 연결하기에 충분하다.

The three coronagraphs are integrated into a single precision-milled aluminum case (339.5 × 323 × 1362 mm). C1 occupies one half, while C2 and C3 occupy the other half. Alignment accuracy between the three telescopes is within **60 arcsec**. This alignment is sufficient to seamlessly connect observations from the three instruments in the overlapping FOV regions.

### §9 Detectors (pp. 386–388)

세 코로나그래프 모두 동일한 CCD 사양을 사용한다:

All three coronagraphs use the same CCD specifications:

| 파라미터 / Parameter | 값 / Value |
|---|---|
| 포맷 / Format | 1024 × 1024 pixels |
| 픽셀 크기 / Pixel size | 21 μm |
| CCD 타입 / Type | **Front-illuminated** (전면 조사형) |
| 제조사 / Manufacturer | Tektronix (SITe) |
| 동작 모드 / Mode | MPP (Multi-Pinned Phase) |
| 운용 온도 / Operating temperature | −80°C |
| 양자 효율 / QE | 0.3–0.5 (500–700 nm) |
| Full well capacity | ~150,000 e⁻ |
| CTE | >0.999999 |
| ADC | 14-bit |
| 게인 / Gain | 15–20 e⁻/DN |
| 판독 시간 / Readout time | ~22 s |

**중요**: EIT는 후면 조사형(back-illuminated) CCD를 사용하지만, LASCO는 **전면 조사형(front-illuminated)**을 사용한다. 이는 LASCO가 가시광(500–700 nm)을 관측하기 때문에 전면 조사형으로도 충분한 QE(30–50%)를 달성할 수 있기 때문이다. 반면 EIT는 EUV(<30 nm)를 관측하므로 후면 조사형의 높은 QE가 필수적이다.

**Important**: EIT uses back-illuminated CCDs, but LASCO uses **front-illuminated** CCDs. This is because LASCO observes visible light (500–700 nm) where front-illuminated CCDs achieve sufficient QE (30–50%). In contrast, EIT observes EUV (<30 nm) where the higher QE of back-illuminated CCDs is essential.

### §10 LEB Electronics (pp. 388–396)

LEB(LASCO/EIT Electronics Box)는 LASCO의 세 코로나그래프와 EIT를 **모두 제어하는 공유 전자부**이다. 이는 우주선 무게와 전력을 절약하기 위한 설계이지만, 운용상의 제약도 수반한다.

The LEB (LASCO/EIT Electronics Box) is the **shared electronics unit controlling all three LASCO coronagraphs and EIT**. This design saves spacecraft mass and power but entails operational constraints.

#### Hardware / 하드웨어

- **CPU**: Sandia SA3000, 32-bit, 15 MHz, **방사선 내성(radiation-hardened)**
  - Radiation-hardened Sandia SA3000, 32-bit, 15 MHz
- **보조 프로세서 / Co-processor**: Intel 8031 — 원격 명령(telecommand) 및 하우스키핑(housekeeping) 담당
  - Intel 8031 for telecommand and housekeeping
- **메모리 / Memory**: 주 메모리 12 MB RAM + 예비 6 MB RAM (총 18 MB)
  - Primary 12 MB RAM + redundant 6 MB RAM (total 18 MB)
- **EEPROM**: 코드 저장용. 비행 중 소프트웨어 업데이트 가능.
  - For code storage. Enables in-flight software updates.
- **메커니즘 / Mechanisms**: 총 23개 (필터 휠, 편광기 휠, 도어, 차폐판 등)
  - Total 23 mechanisms (filter wheels, polarizer wheels, doors, occulters, etc.)
- **열 제어 / Thermal control**: PID 방식

#### LEB Programs (LP) / LEB 프로그램

스케줄링 시스템은 4개의 큐(queue)를 사용한다:

The scheduling system uses 4 queues:

1. **Dormant** (휴면): 비활성 프로그램
2. **Wait** (대기): 특정 조건(시간, 이벤트) 대기 중
3. **Ready** (준비): 실행 가능하지만 CPU 할당 대기 중
4. **Current** (실행): 현재 실행 중

이 구조는 여러 관측 시퀀스를 동시에 관리할 수 있게 하며, 4대의 망원경이 하나의 CPU를 효율적으로 공유하도록 한다.

This structure enables managing multiple observation sequences simultaneously, allowing 4 telescopes to efficiently share a single CPU.

#### Image Handling / 이미지 처리

이미지는 **32 × 32 픽셀 블록** 단위로 처리된다. 2 MB 이미지 버퍼를 통해 읽어들인 후 압축하여 다운링크 대기열에 넣는다.

Images are processed in **32 × 32 pixel blocks**. They are read through a 2 MB image buffer, compressed, and queued for downlink.

#### Compression Suite / 압축 방법

LASCO의 압축은 두 단계로 구성된다: **기하학적 압축(geometric compression)**과 **코딩 압축(coding compression)**.

LASCO's compression consists of two stages: **geometric compression** and **coding compression**.

**기하학적 압축 / Geometric compression** (데이터량 감소 / data reduction):

| 방법 / Method | 압축비 / Compression ratio | 설명 / Description |
|---|---|---|
| Occulter masking | ×1.3–1.5 | 차폐된 영역의 데이터 제거 / Remove data in occulted region |
| BadColAvg | — | 불량 열 보정 / Bad column correction |
| InterMin | — | 최소값 간 보간 / Interpolation between minima |
| Mask | — | 관심 영역 외 제거 / Remove outside region of interest |
| Subregion | — | 부분 영역만 전송 / Transmit only subregion |
| PixSum (n×n) | — | n×n 픽셀 합산 / n×n pixel summation |
| RadialSpoke | ×5.7–9 | 방사형 스포크 샘플링으로 FOV 가장자리 압축 / Radial spoke sampling to compress FOV edges |

**코딩 압축 / Coding compression** (비트 감소 / bit reduction):

| 방법 / Method | 압축비 / Ratio | 타입 / Type |
|---|---|---|
| Rice | ~×2 | Lossless (무손실) |
| ADCT (Adaptive DCT) | ×15–20 | Lossy (손실) |
| DivideBy2 | — | 비트 감소 / Bit reduction |
| SquareRoot | — | 동적 범위 압축 / Dynamic range compression |
| Difference | — | 차분 영상 / Difference imaging |
| Summed | — | 합산 영상 / Summed imaging |

**평균 종합 압축비 / Average overall compression ratio**: ~×10

이를 통해 제한된 텔레메트리(~5.2 kbit/s for LASCO+EIT combined)에서 효율적인 과학 데이터 전송이 가능하다.

This enables efficient science data transmission within the limited telemetry (~5.2 kbit/s for LASCO+EIT combined).

#### Image Cadence Calculation / 영상 케이던스 계산

$$\text{Compressed image size} = \frac{1024^2 \times 14 \, \text{bits}}{10 \, (\text{compression})} = 1{,}468{,}006 \, \text{bits} \approx 1.4 \, \text{Mbit}$$

$$\text{Transmission time per image} = \frac{1.4 \times 10^6 \, \text{bits}}{5200 \, \text{bits/s}} \approx 269 \, \text{s} \approx 4.5 \, \text{min}$$

4대의 망원경이 공유하므로, 각 망원경당 약 6분에 한 장의 1024² 압축 이미지를 전송할 수 있다. 하루 약 **240장**의 이미지를 전송한다 (4대 합산).

Since 4 telescopes share the link, each telescope can transmit approximately one 1024² compressed image every 6 minutes. This yields approximately **240 images per day** (combined for all 4 telescopes).

---

## 3. Key Takeaways / 핵심 시사점

1. **3중 코로나그래프 — 완전한 시야각 커버리지 / Triple coronagraph — complete FOV coverage**: C1(1.1–3), C2(1.5–6), C3(3.7–30 $R_\odot$)의 겹치는 FOV는 태양 표면 직상부부터 30 태양 반경까지 연속적 관측을 가능하게 한다. 이는 CME의 발생, 가속, 전파 전 과정을 하나의 기기 세트로 추적할 수 있다는 의미이다.
   - The overlapping FOVs of C1 (1.1–3), C2 (1.5–6), and C3 (3.7–30 $R_\odot$) enable continuous observation from just above the solar surface to 30 solar radii. This means a single instrument suite can track the entire CME lifecycle: initiation, acceleration, and propagation.

2. **C1 거울형 Lyot 혁신 / C1 mirror Lyot innovation**: 전통적 렌즈 Lyot 코로나그래프의 색수차와 렌즈 결함 문제를 비축 포물경(off-axis parabolic mirror)으로 해결하였다. 이는 내부 차폐 코로나그래프의 새로운 설계 패러다임을 제시하였으나, 1998년 SOHO 일시 상실 사고 후 C1이 복구되지 못해 이 설계의 장기 우주 검증은 이루어지지 못하였다.
   - Solved the chromatic aberration and lens defect problems of traditional lens Lyot coronagraphs with off-axis parabolic mirrors. This presented a new design paradigm for internally occulted coronagraphs, but since C1 was not recovered after SOHO's temporary loss in 1998, long-term space validation of this design was not achieved.

3. **C2 톱니형 외부 차폐판 / C2 serrated external occulter**: 160개 다이아몬드 가공 나사산의 톱니 구조는 이전 코로나그래프(Skylab, SMM) 대비 1차수 이상의 산란광 억제를 달성하였다. 이 설계는 이후 STEREO/COR2에 직접 계승되었다.
   - The serrated structure with 160 diamond-machined threads achieved more than one order of magnitude stray light suppression over previous coronagraphs (Skylab, SMM). This design was directly inherited by STEREO/COR2.

4. **Fabry-Perot 분광 능력 / Fabry-Perot spectroscopic capability**: Fe XIV (530.3 nm)에서 ±430 km/s 범위, ~5 km/s 정밀도의 도플러 속도 측정이 가능하였다. 이는 최초의 우주 기반 코로나 분광 관측으로, 코로나 역학 연구의 새로운 장을 열었다.
   - Doppler velocity measurements with ±430 km/s range and ~5 km/s precision were possible at Fe XIV (530.3 nm). As the first spaceborne coronal spectroscopy, this opened a new chapter in coronal dynamics research.

5. **산란광 성능 — 역대 최고 / Stray light performance — best ever at launch**: Fig. 4의 핵심 메시지는 C2/C3가 이전 모든 우주 코로나그래프보다 산란광이 낮다는 것이다. C3는 $10^{-12}$ $B_\odot$ 수준까지 도달하여 30 $R_\odot$에서의 외부 코로나 관측을 가능하게 하였다.
   - Fig. 4's key message is that C2/C3 have lower stray light than all previous spaceborne coronagraphs. C3 reaches the $10^{-12}$ $B_\odot$ level, enabling outer corona observations at 30 $R_\odot$.

6. **EIT와의 LEB 공유 — 공학적 절충 / Shared LEB with EIT — engineering tradeoff**: 하나의 전자부(SA3000 CPU, 18 MB RAM)가 4대의 망원경을 제어한다. 이는 무게·전력을 절약하지만, 동시 관측 능력에 제약을 준다. 하루 ~240장이라는 제한된 이미지 수는 이 공유 구조의 직접적 결과이다.
   - A single electronics unit (SA3000 CPU, 18 MB RAM) controls 4 telescopes. This saves mass and power but constrains simultaneous observation capability. The limited ~240 images/day is a direct consequence of this shared architecture.

7. **광범위한 압축 기법 / Extensive compression suite**: 기하학적 압축(Occulter masking, RadialSpoke 등)과 코딩 압축(Rice lossless, ADCT lossy)의 2단계 체계로 평균 ~10×의 압축비를 달성한다. 특히 RadialSpoke(×5.7–9)는 코로나그래프에 특화된 독창적 방법이다.
   - A two-stage system of geometric compression (Occulter masking, RadialSpoke, etc.) and coding compression (Rice lossless, ADCT lossy) achieves an average ~10× compression ratio. RadialSpoke (×5.7–9) is a particularly inventive method specific to coronagraphs.

8. **1998년 C1 상실과 C2/C3의 30년 운용 / 1998 loss of C1 and 30-year operation of C2/C3**: SOHO의 일시적 통신 상실(1998년 6월) 후 C1의 Fabry-Perot은 복구되지 못하였다. 그러나 C2와 C3는 2025년 현재까지 30년간 운용되어 40,000개 이상의 CME를 발견하였으며, CME 카탈로그(CDAW)의 기반이 되었다. 이는 우주 기기 설계의 놀라운 신뢰성과 내구성의 증거이다.
   - After SOHO's temporary loss of contact (June 1998), C1's Fabry-Perot was not recovered. However, C2 and C3 have operated for 30 years as of 2025, discovering over 40,000 CMEs and forming the basis of the CME catalog (CDAW). This is remarkable evidence of space instrument reliability and durability.

---

## 4. Mathematical Summary / 수학적 요약

### Thomson Scattering and Polarized Brightness / Thomson 산란과 편광 밝기

코로나의 자유 전자에 의한 Thomson 산란은 코로나그래프가 관측하는 K-코로나의 물리적 기반이다. 편광 밝기(pB)는 다음과 같이 정의된다:

Thomson scattering by free coronal electrons is the physical basis of the K-corona observed by coronagraphs. Polarized brightness (pB) is defined as:

$$pB = B_T - B_R$$

여기서 / Where:
- $B_T$ = 접선 방향 편광 밝기 / tangential polarization brightness
- $B_R$ = 방사 방향 편광 밝기 / radial polarization brightness

시선(line-of-sight) 방향으로 적분하면 / Integrating along the line of sight:

$$pB(r) = \frac{\pi \sigma_T B_\odot}{2} \int_{-\infty}^{+\infty} n_e(l) \left[ (1 - u) A(r, l) + u B(r, l) \right] dl$$

여기서 / Where:
- $\sigma_T = 6.65 \times 10^{-25}$ cm² — Thomson 산란 단면적 / Thomson scattering cross-section
- $n_e$ = 전자 밀도 / electron density
- $u$ = limb darkening 계수 / limb darkening coefficient
- $A(r, l)$, $B(r, l)$ = Van de Hulst (1950)의 기하학적 함수 / geometric functions from Van de Hulst (1950)

LASCO C2/C3에서 pB 관측은 3개 편광기(0°, +60°, −60°)의 연속 촬영으로 수행된다. Stokes 파라미터를 통해:

In LASCO C2/C3, pB observations are performed by consecutive imaging with 3 polarizers (0°, +60°, −60°). Through Stokes parameters:

$$I = \frac{2}{3}(I_0 + I_{+60} + I_{-60})$$

$$Q = \frac{2}{3}(2I_0 - I_{+60} - I_{-60})$$

$$U = \frac{2}{3}(I_{+60} - I_{-60})$$

$$pB = \sqrt{Q^2 + U^2}$$

### Fabry-Perot Transmission / Fabry-Perot 투과 함수

$$T(\lambda) = \frac{1}{1 + F \sin^2\!\left(\dfrac{2\pi n d}{\lambda}\right)}$$

여기서 / Where:
- $F = \frac{4R}{(1-R)^2}$ — 핀에스 계수(finesse coefficient), $R$은 거울 반사율 / finesse coefficient, $R$ is mirror reflectivity
- $n$ = 간격 매질의 굴절률 (공기/진공: $n \approx 1$) / refractive index of gap medium (air/vacuum: $n \approx 1$)
- $d$ = Fabry-Perot 간격(gap) / Fabry-Perot gap spacing
- $\lambda$ = 파장 / wavelength

FWHM(반값전폭)은 / FWHM (full width at half maximum):

$$\Delta\lambda_{\text{FWHM}} = \frac{\lambda^2}{nd} \cdot \frac{1}{\pi\sqrt{F}} = \frac{\text{FSR}}{\mathcal{F}}$$

여기서 FSR은 자유 스펙트럼 영역(free spectral range), $\mathcal{F}$는 핀에스(finesse)이다. Fe XIV 530.3 nm에서 FWHM = 0.065 nm이므로:

Where FSR is the free spectral range and $\mathcal{F}$ is the finesse. At Fe XIV 530.3 nm, FWHM = 0.065 nm, so:

$$\frac{\Delta v}{c} = \frac{\Delta\lambda}{\lambda} \implies \Delta v = c \cdot \frac{0.065}{530.3} = 3 \times 10^5 \times 1.226 \times 10^{-4} \approx 36.8 \, \text{km/s}$$

이는 하나의 분광 채널의 속도 분해능이다. 여러 채널을 스캔하여 ±430 km/s 범위의 선 프로파일을 재구성한다.

This is the velocity resolution of a single spectral channel. Multiple channels are scanned to reconstruct the line profile over a ±430 km/s range.

### Stray Light from Diffraction / 회절에 의한 산란광

외부 차폐 코로나그래프에서 산란광의 주요 원인은 차폐판 가장자리의 회절이다. 원형 차폐판(직경 $D$)에 의한 회절 밝기는 대략:

In externally occulted coronagraphs, the primary source of stray light is diffraction at the occulter edge. Diffracted brightness from a circular occulter (diameter $D$) is approximately:

$$B_{\text{stray}} \propto \left(\frac{\lambda}{D}\right)^2 B_\odot$$

톱니형(serrated) 차폐판은 회절 패턴을 분산(defocus)시켜 이 수준을 더욱 낮춘다. C2의 160-thread 설계는 경험적으로 이 이론값 대비 추가 ~1 차수 억제를 달성한다.

Serrated occulters defocus the diffraction pattern, further lowering this level. C2's 160-thread design empirically achieves an additional ~1 order of magnitude suppression beyond this theoretical value.

### Vignetting Function for C2 / C2 비네팅 함수

C2의 내부 가장자리에서 비네팅은 기하학적으로 결정된다. 비네팅 함수 $V(r)$는 외부 차폐판 D1, 입구 조리개 A1, 내부 차폐판 D2의 상대적 크기와 위치에 의해 결정되며, 대략:

At C2's inner edge, vignetting is geometrically determined. The vignetting function $V(r)$ is determined by the relative sizes and positions of external occulter D1, entrance aperture A1, and internal occulter D2, approximately:

$$V(r) = \frac{A_{\text{unvignetted}}(r)}{A_{\text{total}}}$$

$r < 1.5 \, R_\odot$에서 $V = 0$ (완전 차폐), $r > 2.2 \, R_\odot$에서 $V \approx 1$ (비네팅 없음). 이 보정은 정량적 코로나 밝기 측정에 필수적이다.

At $r < 1.5 \, R_\odot$, $V = 0$ (fully occulted); at $r > 2.2 \, R_\odot$, $V \approx 1$ (no vignetting). This correction is essential for quantitative coronal brightness measurements.

### Pixel Scale Calculation / 픽셀 스케일 계산

각 코로나그래프의 픽셀 스케일은 유효 초점 거리 $f$와 CCD 픽셀 크기 $p$ (= 21 μm)로부터 결정된다:

The pixel scale of each coronagraph is determined from the effective focal length $f$ and CCD pixel size $p$ (= 21 μm):

$$\theta = \frac{p}{f} \times \frac{180°}{\pi} \times 3600 \, \text{("/pixel)}$$

**C1**: $f = 768$ mm

$$\theta_{C1} = \frac{0.021}{768} \times 206265 \approx 5.6 \, \text{"/pixel}$$

**C2**: $f = 380$ mm (논문에서 유추 / inferred from paper)

$$\theta_{C2} = \frac{0.021}{380} \times 206265 \approx 11.4 \, \text{"/pixel}$$

**C3**: $f = 77.6$ mm

$$\theta_{C3} = \frac{0.021}{77.6} \times 206265 \approx 55.8 \approx 56 \, \text{"/pixel}$$

### Compression and Cadence / 압축과 케이던스

**비압축 이미지 크기 / Uncompressed image size**:

$$S_{\text{raw}} = 1024^2 \times 14 \, \text{bits} = 14{,}680{,}064 \, \text{bits} \approx 14.7 \, \text{Mbit}$$

**10× 압축 후 / After 10× compression**:

$$S_{\text{comp}} = \frac{14.7 \, \text{Mbit}}{10} = 1.47 \, \text{Mbit}$$

**전송 시간 / Transmission time** (5.2 kbit/s 공유 대역폭 / shared bandwidth):

$$t = \frac{1.47 \times 10^6}{5200} \approx 283 \, \text{s} \approx 4.7 \, \text{min}$$

**일일 이미지 수 / Daily image count**:

$$N = \frac{86400 \, \text{s/day}}{283 \, \text{s/image}} \approx 305 \, \text{images/day}$$

실제로는 하우스키핑 데이터, 오버헤드 등을 고려하면 ~240 images/day이다.

In practice, accounting for housekeeping data, overhead, etc., this yields ~240 images/day.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1930  ├─ Lyot — 최초의 지상 코로나그래프 발명 / First ground-based coronagraph invented
      │    └─ 렌즈 기반 Lyot 설계의 시작 / Beginning of lens-based Lyot design
      │
1963  ├─ Tousey — 최초의 로켓 외부 차폐 코로나그래프 / First rocket externally occulted coronagraph
      │
1971  ├─ OSO-7 — 최초의 궤도 코로나그래프 / First orbital coronagraph
      │    └─ 최초의 CME 관측 (Tousey 1973) / First CME observation
      │
1973  ├─ Skylab/ATM — 최초의 고품질 우주 코로나그래프 / First high-quality space coronagraph
      │    └─ 외부 차폐, 필름 기록, 수동 운용 / External occulter, film, manual operation
      │
1979  ├─ P78-1/Solwind — 최초의 자동화 우주 코로나그래프 / First automated space coronagraph
      │
1980  ├─ SMM/Coronagraph-Polarimeter (C/P) — pB 관측 시작 / pB observations begin
      │    └─ 산란광 수준: ~10⁻⁸ at 3 R☉ / Stray light: ~10⁻⁸ at 3 R☉
      │
1995  ├─ ★ SOHO/LASCO 발사 — 3중 코로나그래프 + 분광 능력 ★ [본 논문]
      │    ★ Triple coronagraph + spectroscopic capability ★ [THIS PAPER]
      │    ├─ C1: 거울형 Lyot + Fabry-Perot (최초 우주 분광 코로나그래프)
      │    │   Mirror Lyot + Fabry-Perot (first spaceborne spectroscopic coronagraph)
      │    ├─ C2: 톱니형 외부 차폐판 (산란광 1차수 개선)
      │    │   Serrated external occulter (1 order of magnitude stray light improvement)
      │    └─ C3: 30 R☉까지 관측 (역대 최대 FOV)
      │        Observations to 30 R☉ (largest FOV ever)
      │
1996  ├─ SOHO 과학 관측 시작 / SOHO science operations begin
      │
1998  ├─ SOHO 일시 상실 → C1 Fabry-Perot 영구 손실 / SOHO temporary loss → C1 FP permanently lost
      │    └─ C2/C3는 복구 후 계속 운용 / C2/C3 recovered and continued
      │
2006  ├─ STEREO/COR1+COR2 — LASCO 설계를 쌍둥이 우주선으로 / LASCO design on twin spacecraft
      │    └─ COR2의 톱니형 차폐판은 C2에서 직접 계승 / COR2 serrated occulter inherited from C2
      │
2013  ├─ ISON 혜성 — LASCO C3로 태양 근접 관측 / Comet ISON observed with LASCO C3
      │    └─ LASCO의 4000+ 혜성 발견 중 하나 / One of 4000+ comets discovered by LASCO
      │
2020  ├─ Solar Orbiter/Metis — 다중 채널 코로나그래프 (UV + VL)
      │    Solar Orbiter/Metis — multi-channel coronagraph (UV + VL)
      │
2024  ├─ PROBA-3/ASPIICS — 위성 편대를 이용한 거대 외부 차폐 코로나그래프
      │    Formation-flying giant external occulter coronagraph
      │
 현재  └─ LASCO C2/C3 — 30년 연속 운용, CME 카탈로그의 표준
 Now     LASCO C2/C3 — 30 years of continuous operation, standard for CME catalogs
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| # | 논문 / Paper | 연결 / Connection |
|---|---|---|
| 7 | Tomczyk et al. (2016) — COSMO/K-Cor | LASCO C1(1.1–3 $R_\odot$)과 동일한 영역을 지상에서 관측하는 지상 코로나그래프. C1 상실 후 이 영역의 관측 공백을 지상 관측으로 보완. / Ground-based coronagraph observing the same region as LASCO C1 (1.1–3 $R_\odot$). Fills the observational gap after C1 loss with ground-based observations. |
| 8 | Domingo, Fleck, & Poland (1995) — SOHO overview | LASCO는 SOHO의 12개 탑재체 중 하나. 미션 궤도(L1), 텔레메트리, 운용 개념의 맥락을 제공. / LASCO is one of SOHO's 12 instruments. Provides context for mission orbit (L1), telemetry, and operations concept. |
| 9 | Delaboudinière et al. (1995) — EIT | LASCO와 LEB 전자부를 공유하는 형제 기기. 동일한 CPU(SA3000), 메모리(18 MB), 텔레메트리를 4대의 망원경이 분할 사용. EIT의 디스크 관측과 LASCO의 코로나 관측이 결합하여 CME의 디스크-코로나 연결을 추적. / Sibling instrument sharing the LEB electronics. Same CPU (SA3000), memory (18 MB), and telemetry shared by 4 telescopes. EIT disk observations combined with LASCO corona observations enable tracking the disk-to-corona connection of CMEs. |
| 11 | Handy et al. (1999) — TRACE | LASCO가 외부 코로나에서 CME를 관측하는 동안, TRACE는 디스크 위에서 활동 영역의 EUV 미세 구조를 관측. 시간적 연계 관측(coordinated campaigns)이 빈번하게 수행됨. / While LASCO observes CMEs in the outer corona, TRACE observes EUV fine structure of active regions on disk. Coordinated campaigns were frequently conducted. |
| 12 | Lemen et al. (2012) — SDO/AIA | SDO/AIA의 12초 케이던스 전 디스크 EUV 영상과 LASCO의 코로나 관측을 결합하면 CME 발생 메커니즘의 가장 완전한 관측 데이터를 제공. / Combining SDO/AIA's 12-second cadence full-disk EUV imaging with LASCO's coronagraph observations provides the most complete observational data on CME initiation mechanisms. |
| 13 | Scherrer et al. (2012) — SDO/HMI (Paper #13 in series) | 자기장 관측(HMI)과 코로나 구조(LASCO)의 결합은 CME의 자기적 기원을 이해하는 핵심. LASCO 헤일로 CME의 소스 영역을 HMI 자기장 맵에서 식별. / Combining magnetic field observations (HMI) with coronal structure (LASCO) is key to understanding the magnetic origin of CMEs. Source regions of LASCO halo CMEs identified in HMI magnetic maps. |
| — | Skylab/ATM (1973) | LASCO의 직접적 선조. 외부 차폐 코로나그래프의 우주 비행 유산을 물려받음. LASCO는 Skylab 대비 산란광을 1차수 이상 개선. / Direct ancestor of LASCO. Inherited the spaceflight heritage of externally occulted coronagraphs. LASCO improved stray light by more than 1 order of magnitude over Skylab. |
| — | SMM/C-P (1980) | LASCO 이전의 가장 진보된 우주 코로나그래프. pB 관측 기법을 개척하였으며, LASCO C2/C3가 이를 계승 및 발전시킴. / Most advanced space coronagraph before LASCO. Pioneered pB observation techniques, which LASCO C2/C3 inherited and advanced. |
| — | STEREO/COR1+COR2 (2006) | LASCO C2의 톱니형 외부 차폐판 설계를 COR2가 직접 계승. 쌍둥이 우주선에서 스테레오 코로나그래프 관측을 실현하여 CME의 3D 구조를 최초로 측정. / COR2 directly inherited C2's serrated external occulter design. Realized stereoscopic coronagraph observations from twin spacecraft, first measuring CME 3D structure. |

---

## 7. References / 참고문헌

- Brueckner, G.E., et al., "The Large Angle Spectroscopic Coronagraph (LASCO)," Solar Physics, Vol. 162, pp. 357–402, 1995. [DOI: 10.1007/BF00733434](https://doi.org/10.1007/BF00733434)
- Domingo, V., Fleck, B., & Poland, A.I., "The SOHO Mission: An Overview," Solar Physics, Vol. 162, pp. 1–37, 1995. [DOI: 10.1007/BF00733425](https://doi.org/10.1007/BF00733425)
- Delaboudinière, J.-P., et al., "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission," Solar Physics, Vol. 162, pp. 291–312, 1995. [DOI: 10.1007/BF00733432](https://doi.org/10.1007/BF00733432)
- Lyot, B., "La couronne solaire étudiée en dehors des éclipses," Zeitschrift für Astrophysik, Vol. 5, pp. 73–95, 1932.
- Tousey, R., "The Extreme Ultraviolet Spectrum of the Sun," Space Science Reviews, Vol. 2, pp. 3–69, 1963.
- Van de Hulst, H.C., "The Electron Density of the Solar Corona," Bulletin of the Astronomical Institutes of the Netherlands, Vol. 11, pp. 135–150, 1950.
- Howard, R.A., et al., "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)," Space Science Reviews, Vol. 136, pp. 67–115, 2008. [DOI: 10.1007/s11214-008-9341-4](https://doi.org/10.1007/s11214-008-9341-4)
- Tomczyk, S., et al., "Coronal Solar Magnetism Observatory (COSMO) Technical Report," NCAR Technical Note, NCAR/TN-523+STR, 2016.
- Lemen, J.R., et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," Solar Physics, Vol. 275, pp. 17–40, 2012. [DOI: 10.1007/s11207-011-9776-8](https://doi.org/10.1007/s11207-011-9776-8)
- Scherrer, P.H., et al., "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)," Solar Physics, Vol. 275, pp. 207–227, 2012. [DOI: 10.1007/s11207-011-9834-2](https://doi.org/10.1007/s11207-011-9834-2)
- Handy, B.N., et al., "The Transition Region and Coronal Explorer," Solar Physics, Vol. 187, pp. 229–260, 1999. [DOI: 10.1023/A:1005166902804](https://doi.org/10.1023/A:1005166902804)
- Yashiro, S., et al., "A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft," Journal of Geophysical Research, Vol. 109, A07105, 2004. [DOI: 10.1029/2003JA010282](https://doi.org/10.1029/2003JA010282)
- Rochus, P., et al., "The Solar Orbiter EUI instrument: The Extreme Ultraviolet Imager," Astronomy & Astrophysics, Vol. 642, A8, 2020. [DOI: 10.1051/0004-6361/201936663](https://doi.org/10.1051/0004-6361/201936663)
