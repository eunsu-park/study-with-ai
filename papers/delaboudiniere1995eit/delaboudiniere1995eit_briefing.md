---
title: "Pre-Reading Briefing: EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission"
paper_id: "09_delaboudiniere_1995"
topic: Solar Observation
date: 2026-04-16
type: briefing
---

# EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Delaboudinière, J.-P., et al. (1995). "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission." *Solar Physics*, Vol. 162, pp. 291–312.
**Author(s)**: Jean-Pierre Delaboudinière (IAS, Orsay, France) + 24 co-authors
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

이 논문은 SOHO에 탑재된 **EIT(Extreme-ultraviolet Imaging Telescope)**의 설계, 광학 원리, 성능 사양을 상세히 기술합니다. EIT는 4개의 EUV 파장 밴드(Fe IX/X 171 Å, Fe XII 195 Å, Fe XV 284 Å, He II 304 Å)에서 태양 전면(full-disk) 영상을 1024×1024 CCD로 촬영하는 기기입니다. 핵심 혁신은 **수직 입사(normal-incidence) 다층 코팅 거울**을 사용하여 특정 EUV 파장만 반사하는 기술입니다. EIT는 서로 다른 온도의 코로나 플라즈마를 동시에 영상화할 수 있는 최초의 우주 기기로서, 이후 STEREO/EUVI, SDO/AIA 등 모든 EUV 태양 영상기의 원형(prototype)이 되었습니다.

This paper describes the design, optical principles, and performance specifications of **EIT (Extreme-ultraviolet Imaging Telescope)** on SOHO. EIT images the full solar disk in four EUV bandpasses (Fe IX/X 171 Å, Fe XII 195 Å, Fe XV 284 Å, He II 304 Å) using a 1024×1024 CCD. The key innovation is **normal-incidence multilayer-coated mirrors** that selectively reflect specific EUV wavelengths. EIT was the first space instrument to simultaneously image coronal plasma at different temperatures, becoming the prototype for all subsequent EUV solar imagers (STEREO/EUVI, SDO/AIA).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

EUV(극자외선, 100–1200 Å) 영역은 태양 코로나와 천이 영역(transition region)의 핵심 진단 파장대입니다. 이 파장 범위의 방출선은 $10^4$–$10^7$ K 온도의 플라즈마에서 발생하므로, 채층에서 코로나까지의 온도 구조를 직접 관측할 수 있습니다.

The EUV (extreme ultraviolet, 100–1200 Å) range is the key diagnostic wavelength band for the solar corona and transition region. Emission lines in this range originate from plasma at $10^4$–$10^7$ K, enabling direct observation of the temperature structure from chromosphere to corona.

1990년대 이전의 EUV 관측은 주로 **경사 입사(grazing incidence)** 광학계를 사용했습니다. 경사 입사 거울은 제작이 복잡하고 영상 품질이 낮았습니다. 1980년대에 **다층 코팅(multilayer coating)** 기술이 발전하면서, 수직 입사 거울로 특정 EUV 파장을 높은 반사율로 반사할 수 있게 되었습니다. 이 기술 돌파구가 EIT를 가능하게 했습니다.

Before the 1990s, EUV observations relied on **grazing-incidence** optics, which were complex to fabricate and had poor imaging quality. In the 1980s, advances in **multilayer coating** technology enabled normal-incidence mirrors to reflect specific EUV wavelengths with high reflectivity. This breakthrough made EIT possible.

### 타임라인 / Timeline

```
1946  ── 최초의 로켓 UV 태양 관측 (Baum et al.)
         First rocket UV solar observation
         │
1973  ── Skylab/ATM — 최초의 우주 태양 EUV/X선 관측 (사진 필름 사용)
         First space solar EUV/X-ray (photographic film)
         │
1980  ── SMM — 태양 극대기 관측 (CCD 없음)
         │
1981  ── Spiller: 다층 코팅으로 EUV 수직 입사 반사 실현
         Multilayer coatings enable normal-incidence EUV reflection
         │
1985  ── Underwood & Barbee: 다층 코팅 거울로 최초의 태양 EUV 영상
         First solar EUV image with multilayer mirror (rocket flight)
         │
1988  ── SOHO 기기 선정 — EIT 선정 (PI: Delaboudinière, IAS Orsay)
         SOHO instrument selection — EIT selected
         │
1991  ── MSSTA 로켓 비행 — 다층 코팅 텔레스코프 어레이로 다중 파장 EUV 영상
         Multi-Spectral Solar Telescope Array rocket flight
         │
1993  ── Yohkoh/SXT — X선 태양 영상 (연 X선, 경사 입사 거울)
         │
1995  ── ★ Delaboudinière et al.: EIT 기기 논문 출판 ★
      │  12월: SOHO 발사, EIT 첫 빛(first light)
         │
1996  ── EIT 일상 관측 시작 — "EIT waves" 발견 (코로나 충격파)
         Routine observations begin — discovery of "EIT waves"
         │
2006  ── STEREO/EUVI — EIT 설계를 기반으로 한 쌍안 EUV 영상기
         │
2010  ── SDO/AIA — EIT의 직접적 후계자 (7 EUV 채널, 12초 케이던스)
         Direct successor (7 EUV channels, 12s cadence)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 EUV 방출선과 형성 온도 / EUV Emission Lines and Formation Temperature

코로나 플라즈마는 특정 온도에서 특정 이온의 방출선을 생성합니다:

Coronal plasma emits specific spectral lines at specific temperatures:

| 파장 / Wavelength | 이온 / Ion | 형성 온도 / $T_{\text{form}}$ | 코로나 구조 / Coronal Feature |
|---|---|---|---|
| 171 Å | Fe IX/X | $\sim 1 \times 10^6$ K | 조용한 코로나, 코로나 루프 / Quiet corona, coronal loops |
| 195 Å | Fe XII | $\sim 1.5 \times 10^6$ K | 활동 영역 코로나 / Active region corona |
| 284 Å | Fe XV | $\sim 2 \times 10^6$ K | 활동 영역 핵심 / Active region core |
| 304 Å | He II | $\sim 8 \times 10^4$ K | 채층, 천이 영역 / Chromosphere, transition region |

이 4개 밴드를 선택한 이유: 철(Fe) 이온의 여러 이온화 단계가 서로 다른 온도에 대응하므로, 단일 원소로 $10^4$–$10^7$ K 범위를 커버할 수 있습니다.

### 3.2 다층 코팅 원리 / Multilayer Coating Principle

다층 코팅은 고굴절률 물질(예: Mo)과 저굴절률 물질(예: Si)을 번갈아 증착하여 구성합니다:

Multilayer coatings alternate high-Z material (e.g., Mo) and low-Z material (e.g., Si):

$$n\lambda = 2d\sin\theta$$

여기서 $d$는 이중층 주기(bilayer period), $\theta$는 입사각, $n$은 반사 차수입니다.

수직 입사($\theta = 90°$)에서 $\lambda = 2d$이므로, $d \approx 85$ Å이면 171 Å를 반사합니다. 전형적으로 20–40쌍의 Mo/Si 이중층을 쌓아 반사율 ~30–40%를 달성합니다.

At normal incidence ($\theta = 90°$), $\lambda = 2d$, so $d \approx 85$ Å reflects 171 Å. Typically 20–40 Mo/Si bilayer pairs achieve ~30–40% reflectivity.

### 3.3 Ritchey-Chrétien 광학계 / Ritchey-Chrétien Optics

EIT는 Ritchey-Chrétien 망원경 설계를 사용합니다. 이것은 쌍곡면 주경(primary)과 쌍곡면 부경(secondary)으로 구성되어, 넓은 시야에서 코마(coma)와 구면 수차를 동시에 제거합니다. HST도 같은 설계입니다.

EIT uses a Ritchey-Chrétien telescope design: hyperbolic primary and hyperbolic secondary mirrors, correcting both coma and spherical aberration over a wide field. HST uses the same design.

### 3.4 이전 논문과의 연결 / Connection to Previous Papers

- **#8 Domingo et al. (1995) — SOHO**: EIT는 SOHO의 12개 기기 중 하나. 이 논문은 EIT를 상세히 기술.
- **#7 Tomczyk et al. (2016) — COSMO**: EIT의 EUV 관측은 가시광/적외선 코로나 관측과 상보적 (충돌 과정 $n_e^2$ vs 복사 과정).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **EUV (Extreme Ultraviolet)** | 파장 100–1200 Å의 극자외선. 대기에 흡수되어 우주에서만 관측 가능. / Wavelength 100–1200 Å. Absorbed by atmosphere, observable only from space. |
| **Normal incidence** | 거울 표면에 수직으로 입사. 다층 코팅으로 EUV 반사 가능. / Light hitting mirror surface perpendicularly. Multilayer coatings enable EUV reflection. |
| **Multilayer coating** | Mo/Si 등 고/저 Z 물질을 번갈아 증착. Bragg 반사 원리로 특정 파장만 반사. / Alternating high/low-Z materials (Mo/Si). Bragg reflection selects specific wavelengths. |
| **Ritchey-Chrétien** | 쌍곡면 주경+부경의 반사 망원경. 넓은 시야에서 코마 제거. / Hyperbolic primary+secondary reflector telescope. Coma-free over wide FOV. |
| **Quadrant design** | EIT의 4분할 거울 설계. 거울을 4등분하여 각각 다른 다층 코팅을 적용. / EIT's mirror divided into 4 sectors, each with different multilayer coating. |
| **CCD (Charge-Coupled Device)** | 실리콘 기반 광검출기. EUV 광자를 직접 검출 (back-illuminated). / Silicon-based photodetector. Directly detects EUV photons (back-thinned). |
| **Backside-illuminated CCD** | CCD 뒷면에서 빛을 받아 전극 구조에 의한 흡수 손실을 제거. EUV에 필수. / Light enters from back side, eliminating absorption by electrode structures. Essential for EUV. |
| **Formation temperature** | 특정 이온이 가장 많이 존재하는 온도. 방출선 강도가 최대인 온도. / Temperature at which a specific ion is most abundant. Temperature of peak emission. |
| **Differential Emission Measure (DEM)** | 온도별 방출 기여도. 다중 밴드 관측에서 재구성 가능. / Emission contribution as a function of temperature. Reconstructable from multi-band observations. |
| **EIT wave** | 코로나를 가로지르는 대규모 파동 현상. EIT에 의해 최초 발견 (1997). / Large-scale wave propagating across the corona. First discovered by EIT (1997). |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Bragg 반사 조건 / Bragg Reflection Condition

$$n\lambda = 2d\sin\theta$$

수직 입사($\theta = 90°$)에서: $\lambda = 2d$ (1차 반사, $n=1$)

- $d$: 이중층 주기 (bilayer period) [Å]
- 171 Å 반사: $d \approx 85$ Å Mo/Si
- 195 Å 반사: $d \approx 98$ Å Mo/Si
- 284 Å 반사: $d \approx 142$ Å Mo/Si (또는 SiC/Si)
- 304 Å 반사: $d \approx 152$ Å Mo/Si

### 5.2 다층 코팅 반사율 / Multilayer Reflectivity

각 표면에서의 반사율 $R$은 이중층 수 $N$에 따라 증가:

$$R \approx R_{\text{single}} \cdot N^2 \quad (\text{thin film limit, small } R_{\text{single}})$$

실제로는 흡수와 계면 거칠기(interface roughness)에 의해 포화됩니다. 전형적인 값:
- Mo/Si at 171 Å: $R \approx 0.35$ (per surface, 40쌍)
- 2-반사 시스템: $R_{\text{total}} = R_1 \times R_2 \approx 0.12$

### 5.3 공간 분해능 / Spatial Resolution

$$\theta_{\text{pixel}} = \frac{\text{pixel size}}{f_{\text{eff}}} = \frac{21\,\mu\text{m}}{1650\,\text{mm}} \approx 2.6''$$

EIT의 유효 초점 거리 $f_{\text{eff}} = 1650$ mm, CCD 픽셀 크기 21 μm → 2.6" 각분해능.

### 5.4 FOV 계산 / Field of View

$$\text{FOV} = 1024 \times 2.6'' = 2662'' \approx 44.4' \approx 0.74°$$

이것은 태양 전면(지름 ~32')을 충분히 커버하며, 림 위 ~6'까지 관측 가능합니다.

---

## 6. 읽기 가이드 / Reading Guide

### 구조 / Structure

이 논문은 22페이지의 기기 논문입니다. 주요 섹션:

1. **Introduction (§1)**: EUV 관측의 과학적 동기, 다층 코팅 기술 발전사
2. **Optical Design (§2)**: Ritchey-Chrétien 설계, 4분할 거울, 필터
3. **Multilayer Coatings (§3)**: Mo/Si 다층 코팅 제작과 성능
4. **CCD Detector (§4)**: 후면 조사(back-illuminated) CCD 사양
5. **Electronics and Operations (§5)**: 데이터 처리, 관측 모드
6. **Calibration (§6)**: 지상 교정과 궤도상 교정 계획
7. **Expected Performance (§7)**: 예상 감도, 케이던스, 영상 품질

### 읽기 전략 / Reading Strategy

1. **핵심 집중 (§2–3)**: 광학 설계와 다층 코팅이 이 논문의 핵심 혁신. 4분할 거울이 왜 필요한지, 각 사분면의 코팅이 어떻게 다른지 이해.
2. **§4 CCD**: 후면 조사 CCD가 EUV에 왜 필수적인지 이해.
3. **§7**: 예상 성능 수치를 SDO/AIA와 비교하면서 읽기.
4. **나머지**: 가볍게 스캔.

### 핵심 Figure 목록 / Key Figures

- **광학 설계도**: Ritchey-Chrétien 레이아웃과 4분할 구조
- **다층 코팅 반사율 곡선**: 각 밴드의 스펙트럼 반사 프로파일
- **EUV 태양 영상**: 4개 밴드의 실제 (또는 예상) 태양 영상

---

## 7. 현대적 의의 / Modern Significance

EIT는 태양 EUV 영상 관측의 패러다임을 확립했습니다:

EIT established the paradigm for solar EUV imaging:

1. **4-밴드 EUV 영상의 표준화**: EIT의 171/195/284/304 Å 밴드 선택이 이후 모든 EUV 영상기의 기준이 됨. SDO/AIA는 이를 7개(+2 UV) 채널로 확장.
   EIT's 171/195/284/304 Å band selection became the standard. SDO/AIA expanded to 7 (+2 UV) channels.

2. **"EIT waves" 발견**: 1997년 EIT로 발견된 코로나 충격파(coronal bright front)는 CME 연구의 새로운 영역을 개척.
   "EIT waves" discovered in 1997 opened a new field in CME research.

3. **다층 코팅 기술의 우주 검증**: EIT의 성공적 운용이 수직 입사 EUV 광학의 우주 환경 내구성을 입증 → EUVI, AIA, Solar Orbiter/EUI로 이어짐.
   EIT's successful operation validated normal-incidence EUV optics in space → EUVI, AIA, Solar Orbiter/EUI.

4. **코로나 가열 연구의 도구**: 다중 온도 영상으로 DEM(Differential Emission Measure) 분석 가능 → 코로나 가열 메커니즘 연구의 핵심 관측 기반.
   Multi-temperature imaging enabled DEM analysis → key observational foundation for coronal heating studies.

5. **이 시리즈에서의 위치**: #8(SOHO 개요)의 EIT 섹션을 상세 확장. #12(AIA)로 직접 연결되는 기술적 계보의 시작점.
   Detailed expansion of EIT section from #8 (SOHO overview). Starting point of the technological lineage leading to #12 (AIA).

---

## Q&A

### Q1: SOHO Table II의 Bit Rate의 의미 — 데이터 생성률인가 전송률인가?

**답**: Table II의 Bit Rate는 각 기기가 OBDH(On-Board Data Handling) 버스에 전송하는 **과학 데이터 생성률(telemetry allocation)**입니다. 우주선 전체 텔레메트리 예산(40 kbit/s)에서 각 기기에 할당된 몫이며, 실제 지상 전송률(DSN 다운링크)과는 다릅니다.

**Answer**: The Bit Rate in Table II is each instrument's **science data generation rate (telemetry allocation)** on the OBDH bus. It is the share allocated from the total 40 kbit/s spacecraft telemetry budget, not the actual ground downlink rate.

데이터 흐름: 기기(연속 생성, 24h) → OBDH 버스 → 온보드 저장(SSR 2Gbit + 테이프 1Gbit) → DSN 다운링크(접촉 시에만, ~12.8h/day). 접촉 없는 시간에는 SSR에 버퍼링됩니다.

Data flow: Instruments (continuous, 24h) → OBDH bus → on-board storage (SSR 2Gbit + tape 1Gbit) → DSN downlink (contact only, ~12.8h/day). Data is buffered on SSR during non-contact periods.

괄호 안의 값(예: MDI 5 (+160))은 고속 텔레메트리 모드에서의 추가 할당을 의미합니다. MDI는 연속 모드에서 온보드 압축/공간 평균으로 5 kbit/s에 맞추고, 연간 60일만 고속 모드(160 kbit/s)로 전체 해상도 영상을 전송합니다.

Values in parentheses (e.g., MDI 5 (+160)) indicate additional allocation in high-rate telemetry mode. MDI uses on-board compression/spatial averaging to fit 5 kbit/s in continuous mode, transmitting full-resolution images at 160 kbit/s for only 60 days/year.

### Q2: SUMER, CDS, EIT, UVCS의 차이 — 관측 대상, 방법, 파장

| | **SUMER** | **CDS** | **EIT** | **UVCS** |
|---|---|---|---|---|
| **유형/Type** | UV 분광기 | EUV 분광기 | EUV 영상기 | UV 코로나그래프+분광기 |
| **파장/Wavelength** | 500–1600 Å | 150–800 Å | 171/195/284/304 Å | Ly-α 1216 Å, O VI 1032 Å |
| **온도/Temp range** | $10^4$–$10^7$ K | $10^5$–$10^7$ K | $10^4$–$2\times10^6$ K | $10^5$–$10^6$ K |
| **관측 대상/Target** | 채층, 천이영역 (disk) | 천이영역, 코로나 (disk) | 전일면 (disk+limb) | 외부 코로나 (1.3–10 $R_\odot$) |
| **공간 분해능/Resolution** | 1.5″ | 3″ | 2.6″/pixel | ~7″ |
| **분광 분해능** | λ/Δλ = 18,800–40,000 | λ/Δλ = 2,000–10,000 | 없음 (협대역 필터) | λ/Δλ ~ 수천 |
| **핵심 측정/Key measurement** | 도플러 속도(1 km/s), 난류 | 온도, 전자 밀도 (선 비율) | 코로나 형태학 (전체 맥락) | 태양풍 유출 속도 (Doppler dimming) |

**핵심 차이**: EIT는 유일한 **전일면 영상기** (분광 없음), SUMER/CDS는 **슬릿 분광기** (높은 분광 분해능, 좁은 FOV), UVCS는 **림 위 전용** 코로나그래프+분광기 (1.3 $R_\odot$ 이상).

**Key difference**: EIT is the only **full-disk imager** (no spectroscopy), SUMER/CDS are **slit spectrometers** (high spectral resolution, narrow FOV), UVCS is a **limb-only** coronagraph+spectrometer (above 1.3 $R_\odot$).

4개 기기가 서로 다른 공간 영역과 물리량을 측정하여 채층–천이영역–코로나–태양풍의 연결을 완성합니다. EIT가 "어디를 볼지" 전체 맥락을 제공하면, SUMER/CDS가 그 지점의 정밀 물리 진단을, UVCS가 태양풍 가속 영역을 관측합니다.

The four instruments measure different spatial domains and physical quantities, completing the chromosphere–TR–corona–solar wind connection. EIT provides the "where to look" context, SUMER/CDS perform precision diagnostics at those locations, and UVCS observes the solar wind acceleration region.

### Q3: 데이터 압축, 포인팅 정밀도, 텔레메트리 수신 확인

**(1) 데이터 압축**: 논문에 명시적 "compression" 언급은 없음. MDI는 온보드에서 "spatial and temporal averages"(공간/시간 평균)로 데이터를 축약하여 5 kbit/s에 맞춤 (원시 ~170 kbit/s). 현대적 압축 알고리즘이 아닌 과학적 데이터 축약(비닝, 평균, 선택적 전송)에 의존.

**(2) 포인팅 정밀도**: (a) 자세 센서(FPSS, SSU, Gyros)를 기기와 같은 PLM에 배치하여 구조적 일체성 확보, (b) PLM-SVM 열적 분리(단열 와셔, 20°C±5°C), (c) Reaction wheel로 무진동 미세 제어 (thruster는 8주마다 궤도 유지에만 사용), (d) L1 궤도의 완만한 속도 변화(±16 m/s/day)로 빈번한 기동 불필요.

**(3) 텔레메트리 수신 확인**: SOHO는 실시간 ACK/NAK를 사용하지 않음. 대신: (a) **순방향 오류 정정(FEC)**: Convolutional + Reed-Solomon 이중 코딩으로 오류 자동 복구, (b) **프레임 시퀀스 번호**로 누락 감지, (c) **3 Gbit 온보드 저장**으로 필요시 재전송 가능, (d) 지상 명령으로 재생/삭제 제어.
