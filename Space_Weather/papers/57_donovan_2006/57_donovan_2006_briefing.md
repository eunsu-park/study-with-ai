---
title: "Pre-Reading Briefing: The THEMIS All-Sky Imaging Array — System Design and Initial Results from the Prototype Imager"
paper_id: "57_donovan_2006"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# The THEMIS All-Sky Imaging Array — System Design and Initial Results / 사전 읽기 브리핑

**Paper**: Donovan, E., Mende, S., Jackel, B., Frey, H., Syrjäsuo, M., Voronkov, I., Trondsen, T., Peticolas, L., Angelopoulos, V., Harris, S., Greffen, M., Connors, M., 2006. The THEMIS all-sky imaging array — system design and initial results from the prototype imager. *Journal of Atmospheric and Solar-Terrestrial Physics*, 68, 1472–1487. DOI: 10.1016/j.jastp.2005.03.027
**Author(s)**: Eric Donovan (lead) and the THEMIS GBO team
**Year**: 2006

---

## 1. 핵심 기여 / Core Contribution

**English**: This paper documents the system-level design of the THEMIS All-Sky Imager (ASI) ground-based array — twenty white-light, panchromatic, CCD-based imagers blanketing North America at auroral latitudes — and demonstrates with prototype data from Athabasca Geophysical Observatory that the array meets the THEMIS mission's two core ground-based requirements: pinpointing the **substorm onset meridian to better than 1°** and the **onset time to better than 10 s**. By choosing a 3-second cadence, ~1 km zenith pixel scale, and panchromatic (no narrow-band filter) optics on the cost-effective Starlight Express MX716 CCD camera, the team produced an instrument simultaneously sensitive enough to detect onset arcs at low geomagnetic latitudes and inexpensive enough to deploy 20 copies. The Athabasca prototype event of 4 October 2003 (~0619:30 UT) reveals two scientific previews: (i) the breakup arc develops nearly stationary, ~800 km wavelength **wavelike azimuthal structure** during the late growth phase, and (ii) the brightening unfolds nearly **simultaneously** along the entire arc visible in the FOV.

**한국어**: 본 논문은 THEMIS 임무의 지상 보조 관측망인 **All-Sky Imager(ASI) array**의 시스템 설계를 문서화하고, 프로토타입(Athabasca) 자료로 임무 요구 사양 — **서브스톰 onset 자오선 ≤1°, onset 시각 ≤10초** — 을 만족함을 입증한다. 캐나다 전역에 20대의 백색광(panchromatic) CCD 이미저를 배치하여 GPS 동기화된 3초 cadence와 천정에서 약 1 km 공간 해상도를 달성하였다. 이미저는 어안렌즈(fish-eye objective) + telecentric 렌즈 + Sony ICX249AL CCD를 결합한 저가 상용 카메라(Starlight Express MX716)를 기반으로 한다. 2003년 10월 4일 ~0619:30 UT의 프로토타입 데이터는 (1) growth phase 후반에 onset 직전 약 **800 km 파장의 정지파(wavelike) azimuthal 구조**가 발달하고, (2) breakup 시점에 시야 전체의 arc가 **거의 동시에** 밝아지는 두 가지 과학적 결과를 보여준다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English**: By the early 2000s, the substorm onset problem — does the breakup begin near-Earth (Current Disruption, CD) or in the mid-tail (Near-Earth Neutral Line, NENL)? — had been argued for two decades on the basis of fragmentary single-spacecraft conjunctions and isolated ground stations. The community recognized that resolving the question demanded simultaneous in-situ measurements bracketing the CD (~6–10 RE) and NENL (~20–30 RE) regions plus *continent-scale* ground-based optical/magnetic coverage so the azimuthal evolution could be tracked. NASA approved the THEMIS MIDEX mission (PI: Angelopoulos, UC Berkeley) for launch in 2006, with five identical satellites lining up over central Canada every four sidereal days. Without a matching ground array, the in-situ data alone could not separate radial from azimuthal substorm motion. The THEMIS ground-based observatory (GBO) program was thus conceived as a *mission-critical* component, not merely a complement.

**한국어**: 2000년대 초까지 서브스톰 **onset 문제** — breakup이 근지구(Current Disruption, CD ~6–10 RE)에서 시작되는가, 중간 magnetotail(NENL ~20–30 RE)에서 시작되는가 — 는 단일 위성 conjunction과 산발적 지상관측만으로는 결론이 나지 않았다. THEMIS는 5기의 동일 위성을 4 sidereal days마다 캐나다 자오선 위에서 conjunction 시키는 NASA MIDEX 미션으로, 2006년 발사를 목표로 승인되었다. 그러나 in-situ 자료만으로는 radial 진화와 azimuthal 진화를 분리할 수 없었기에, 대륙 규모의 지상 광학/자기 관측망(GBO)이 **미션 필수 요소**로 설계되었다. 본 논문은 그 GBO의 핵심인 ASI array의 설계 사양과 첫 결과를 제시한다.

### 타임라인 / Timeline

```
1964 ─ Akasofu: substorm 4단계 모델 (growth/breakup/expansion/recovery)
1970 ─ McPherron: 3-stage 위상 모델 + current wedge
1977 ─ Akasofu textbook "Physics of Magnetospheric Substorms"
1984 ─ Hones: NENL 모델 정립
1989 ─ Baker & Wing: AACGM 좌표계 (ASI mosaic에 사용)
1992 ─ Lui et al.: CD 모델 정립
1996 ─ Baker et al.: NENL 모델 종합 리뷰
1999 ─ Mende et al.: IMAGE FUV 위성 imager
2003 ─ Voronkov et al.: growth-phase arc의 wavelike 구조 보고 (MSP 30s)
2003.05 ─ Athabasca THEMIS 프로토타입 ASI 가동 시작
2003.10.04 ─ 본 논문의 핵심 prototype event (~0619:30 UT)
2004 ─ Frey et al.: IMAGE FUV onset 통계 (2437 events)
2006 ★ Donovan et al. (본 논문): THEMIS ASI 시스템 설계 + 첫 사례
2007.02.17 ─ THEMIS 5기 위성 발사
2008+ ─ ASI array가 26+개 사이트로 확장, "THEMIS GBO" 데이터 시작
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English**:
- **Substorm phenomenology**: growth phase (energy storage, equatorward arc motion), breakup (auroral brightening, current wedge), expansive phase (poleward expansion), recovery phase. Familiarity with Akasofu (1964), McPherron (1970), Rostoker et al. (1980).
- **CD vs. NENL models**: where in the magnetotail breakup originates and how the inner-plasma-sheet current wedge relates to mid-tail reconnection.
- **AACGM coordinates**: Altitude-Adjusted Corrected Geomagnetic latitude/MLT used to map ionospheric features to geomagnetic coordinates (Baker & Wing 1989).
- **Auroral emissions**: 557.7 nm green line (O¹S, ~110 km, fast-electron-driven), 630.0 nm red line (O¹D, ~250 km, soft-electron-driven), and the panchromatic broadband response that integrates them.
- **CCD camera fundamentals**: quantum efficiency, on-chip binning, anti-blooming, readout noise, exposure time, GPS time synchronization.
- **Fish-eye optical projection**: equidistant (f-θ) projection mapping zenith angle θ to image radius r = f·θ, and the relation between pixel position, look direction, and ionospheric mapping.
- **Magnetometer signatures**: H-bay (negative ground-magnetic X deflection from westward electrojet), Pi2 pulsations (40–150 s mid-latitude bursts at onset), riometer absorption (~30 MHz cosmic-noise drop indicating energetic electron precipitation).
- **Keograms / ewograms**: two-dimensional (time, position) summaries of all-sky imagery built by averaging meridional or zonal slices.

**한국어**:
- **서브스톰 현상학**: growth/breakup/expansive/recovery 단계별 자오선 지자기 신호 — Akasofu (1964), McPherron (1970), Rostoker et al. (1980).
- **CD vs. NENL 모델**: 근지구 CD에서 onset이 시작되는지, 중간 자기꼬리 NENL에서 시작되는지의 논쟁.
- **AACGM 좌표계**: 고도 보정 지자기 좌표 — Baker & Wing (1989).
- **오로라 방출선**: 557.7 nm 녹색선(O¹S, ~110 km), 630.0 nm 적색선(O¹D, ~250 km), panchromatic 광대역 응답.
- **CCD 카메라 원리**: quantum efficiency, on-chip binning, readout time, GPS 시각 동기화.
- **Fish-eye projection**: 등거리 투영 r = f·θ, 픽셀 ↔ 시선 방향 ↔ 전리권 좌표 매핑.
- **자력계 시그니처**: H-bay, Pi2 pulsation, 리오미터 흡수.
- **Keogram/ewogram**: all-sky 이미지의 자오선/방위방향 시간 단면.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **THEMIS** | Time History of Events and Macroscale Interactions during Substorms — NASA MIDEX 임무, 5위성 + 20 GBO. NASA MIDEX class mission with 5 satellites + 20 ground observatories. |
| **ASI (All-Sky Imager)** | 어안렌즈로 하늘 전체(180° FOV)를 촬영하는 광학 영상기. Fish-eye, ~180° FOV optical imager. |
| **GBO** | Ground-Based Observatory — magnetometer + ASI 등을 갖춘 지상관측소. Ground site housing ASI + magnetometer. |
| **Panchromatic / White-light** | 좁은 필터 없이 가시광 전 대역 응답 (~400–700 nm). Broadband response without narrow filter. |
| **Telecentric lens** | 주광선이 광축에 평행하도록 동공이 무한대에 위치한 렌즈 — CCD에 균일한 입사각 보장. Lens with pupil at infinity producing chief rays parallel to axis; ensures uniform incidence on CCD. |
| **CCD on-chip binning** | 인접 픽셀(예: 2×2)을 칩 내부에서 합산해 SNR↑·해상도↓·readout 빠르게. Combining adjacent pixels at chip level for higher SNR and faster readout. |
| **Quantum efficiency (QE)** | 입사 광자 → 광전자 전환 비율 (이 카메라: ~70 % @ 600 nm). Photon-to-electron conversion ratio. |
| **Keogram** | (시간, 자오선 elevation) 2D plot — all-sky 이미지의 N–S 단면 시계열. (time, meridional elevation) plot of all-sky N–S slice. |
| **Ewogram** | (시간, east–west) 단면 — 본 논문이 도입한 변형 keogram. East–west keogram introduced in this paper. |
| **AACGM** | Altitude-Adjusted Corrected Geomagnetic 좌표 — 자력선 기반 위도/MLT. Magnetic-field-line-based latitude/MLT system. |
| **MLT (Magnetic Local Time)** | 자기 자정으로부터 측정한 자기 지방시. Magnetic local time. |
| **Pseudobreakup** | 작은 규모로 일어나 본격 expansive phase로 발전하지 않는 breakup. Localized brightening that fails to develop into full substorm. |
| **Current Disruption (CD)** | 근지구 자기꼬리(L ~6–10)에서 cross-tail current가 차단되어 dipolarization이 일어나는 영역/모델. Near-Earth current wedge formation region/model. |
| **NENL** | Near-Earth Neutral Line — 자기꼬리 ~20–30 RE에서의 재결합 위치/모델. Mid-tail reconnection site/model. |
| **Riometer** | 30 MHz 우주잡음 흡수로 D-region 전자침투를 측정하는 계측기. 30 MHz cosmic-noise absorption monitor of energetic-electron precipitation. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Fish-eye equidistant projection / 어안렌즈 등거리 투영**:
$$ r = f \cdot \theta $$
**English**: r is image radius from optical center, θ is zenith angle of incoming ray, f is effective focal length. Unlike pinhole (r = f·tan θ which diverges at θ→90°), the equidistant model keeps r finite at the horizon and is the standard assumption for all-sky imagers.
**한국어**: r은 광축 중심으로부터 이미지 반경, θ는 입사광의 천정각, f는 유효 초점거리. 핀홀 모델(r = f·tan θ)이 horizon에서 발산하는 반면, 등거리 모델은 horizon에서도 유한하여 ASI에 적합.

**(2) Ionospheric ground-projection / 전리권 지상 투영**:
$$ d \approx h \cdot \tan\theta \quad\text{or}\quad d \approx \frac{h \cdot \sin\theta}{\sqrt{1-(R_E/(R_E+h))^2 \sin^2\theta}}\;(\text{spherical}) $$
**English**: d is the horizontal ground distance from the imager to the auroral footprint, h ≈ 110 km is the assumed emission altitude, θ is zenith angle. Using h = 110 km gives ~1 km/pixel at zenith (256-pixel-wide image, 180° FOV → ~0.7°/pixel × 110 km × tan(0.7°) ≈ 1.3 km).
**한국어**: d는 이미저로부터 오로라 지상투영점까지의 수평 거리, h ≈ 110 km는 가정한 발광 고도. 256 픽셀 폭 이미지에서 천정 픽셀 크기는 약 1 km.

**(3) Imager FOV at altitude h / 고도 h에서의 시야 반경**:
$$ R_\text{FOV} \approx h \cdot \tan(\theta_\text{max}) $$
**English**: For maximum useful zenith angle θ_max ≈ 75–80° and h = 110 km, R_FOV ≈ 410–625 km — the radius of the red FOV circles in Fig. 3 of the paper. With 20 sites spaced so adjacent FOVs touch, the array spans ~5000 km east-west across North America.
**한국어**: θ_max ≈ 75–80°, h = 110 km이면 단일 ASI 시야는 반경 ~500 km. 인접 사이트가 맞닿도록 배치된 20개로 북미 전역 ~5000 km 동서 범위를 커버.

**(4) Data volume per imager per year / 이미저당 연간 데이터량**:
$$ V = N_\text{pix}^2 \cdot B \cdot f_s \cdot T_\text{obs} $$
**English**: N_pix = 256, B = 16 bit = 2 byte, f_s = 1/3 s⁻¹, T_obs ≈ 7.9 h/night × 365 ≈ 2880 h ≈ 1.04×10⁷ s. V ≈ 256²·2·(1/3)·1.04×10⁷ ≈ 4.5×10¹¹ B ≈ **450 GB/yr**, matching the ~500 GB/yr quoted in the paper.
**한국어**: 256² 픽셀, 16-bit, 3초 cadence, 연 7.9 h × 365일 가동 → 약 450 GB/yr/이미저, 20대 합산 ~10 TB/yr.

**(5) Azimuthal wavelength of breakup-arc structure / breakup arc 방위 파장**:
$$ \lambda_\text{azim,iono} \approx 50\ \text{km},\quad \lambda_\text{azim,equator} \approx 800\ \text{km} $$
**English**: The wavelike azimuthal structure observed at 0619:30 UT has λ ≈ 50 km in the ionosphere; mapped along stretched dipole field lines into the equatorial magnetosphere, this corresponds to ~800 km — providing a direct constraint on the cross-tail instability driving the onset.
**한국어**: 전리권에서 50 km 파장의 azimuthal 구조는 늘어진 자기력선을 따라 적도 자기권으로 매핑하면 ~800 km에 해당 — onset을 일으키는 cross-tail 불안정성의 길이 척도를 제약.

---

## 6. 읽기 가이드 / Reading Guide

**English**:
1. **§1 Introduction (pp. 1472–1475)**: Skim if you remember Akasofu/McPherron substorm phases and the CD vs. NENL controversy. Pay attention to **Fig. 1** — this defines the multi-instrument signature suite that the THEMIS ASI is designed to extend (MSP keogram + Churchill-line magnetometers + GOES dipolarization).
2. **§2 Instrumentation and data (pp. 1476–1478)**: This is the engineering core. Read carefully — note the 1° / 10 s science requirements, the choice of 3-s cadence (not the requirement-floor 10 s), the 256×256 cropped subframe with 2×2 binning, the ~70 % QE @ 600 nm Sony ICX249AL CCD, the **mechanical aluminum sunshade** decision (vs. hot-mirror), and the data-volume budget (~10 TB/yr total).
3. **§3 Coverage of the onset region (pp. 1478–1479)**: Notice how the team used Frey et al. (2004)'s 2437 IMAGE-FUV onsets to fit an elliptical region containing 70 % of onsets and then computed *fractional coverage as a function of UT* (**Fig. 5**). Conclusion: ≥80 % coverage for ~8 h every night.
4. **§4 Prototype event (pp. 1479–1484)**: The science teaser. Trace the multi-instrument cross-validation: CANOPUS MSP red line (lobe reconnection signature) → CANOPUS magnetometer chain (onset meridian) → riometer (dispersionless injection at Fort Smith) → GOES (dipolarization) → Athabasca ASI (azimuthal wave structure + simultaneous brightening). The **ewogram** in Fig. 7 bottom panel and the **stack plot of column-integrated brightness** in Fig. 8 are the key new diagnostics.
5. **§5 Discussion (pp. 1484–1485)**: The paper's theoretical bet — onset begins as an *azimuthally-stretched, near-monochromatic, non-dispersive* structure ("current and/or flow shear" instability). Connect this to balloon and bursty bulk flow instability literature.

**Reading order suggestion**: Abstract → Fig. 1 → Fig. 3 (site map) → §2 → §4 + Figs. 6–8 → §3 → Discussion. Skip the references on first pass.

**한국어**:
1. **§1 Introduction**: Akasofu/McPherron 단계 모델과 CD vs. NENL 논쟁이 익숙하면 빠르게 훑기. **Fig. 1**의 다중 계측기 신호 패턴(MSP keogram + 자력계 + GOES dipolarization)이 THEMIS ASI가 확장하려는 baseline.
2. **§2 Instrumentation**: 핵심 공학부. 1°/10초 요구 → 3초 cadence 선택, 2×2 binning + 256×256 crop, Sony ICX249AL CCD(QE ~70 %), 알루미늄 sunshade, 연 10 TB 데이터량 등을 정독.
3. **§3 Coverage**: Frey et al. (2004)의 2437 onset 통계로 70 % 영역을 타원으로 fit하고 UT별 coverage 계산. 결론: 매일 ~8 h 동안 ≥80 % coverage.
4. **§4 Prototype event**: 다중 계측기 교차검증 흐름을 따라 읽기. **ewogram(Fig. 7 하단)** 과 **column 적분 brightness 스택(Fig. 8)** 이 새로운 진단 도구.
5. **§5 Discussion**: "방위로 늘어진, 거의 단색, 비분산성" 구조 가설과 cross-tail 불안정성과의 연결을 음미.

읽기 순서 권장: Abstract → Fig. 1 → Fig. 3 → §2 → §4 + Fig. 6–8 → §3 → Discussion.

---

## 7. 현대적 의의 / Modern Significance

**English**: The THEMIS ASI array became operational with the satellite launch in February 2007 and immediately delivered the data set that resolved the CD/NENL debate in favor of a *near-Earth onset followed by tail reconnection* sequence (Angelopoulos et al., *Science*, 2008 — using exactly this ASI infrastructure). The 1 km / 3 s panchromatic dataset is still in active use for studies of: (i) auroral beads and the ballooning/flow-shear instability that this paper anticipates in §5; (ii) STEVE and SAID/SAR-arc science (the wide FOV captures sub-auroral as well as auroral emissions); (iii) cross-validation of SuperMAG, SuperDARN, and EISCAT 3D measurements; (iv) machine-learning training data for auroral classification (extending Syrjäsuo & Donovan 2004 cited herein); (v) inputs for assimilative ionospheric models. The system architecture — *cheap commercial CCD + custom optics + GPS sync + commodity satellite uplink* — became the template for follow-on arrays (REGO, NORSTAR, TREx) and is conceptually echoed by global lightning, all-sky meteor, and exoplanet survey networks.

**한국어**: THEMIS ASI array는 2007년 2월 위성 발사와 함께 본격 가동되어, 곧바로 Angelopoulos et al. (*Science*, 2008)이 CD/NENL 논쟁을 "near-Earth onset 후 tail reconnection" 순서로 정리하는 데 결정적 기여를 했다. 1 km / 3 s panchromatic 자료는 현재까지도 (1) 본 논문 §5가 예고한 ballooning/flow-shear 불안정성과 auroral beads 연구, (2) STEVE 및 SAID/SAR-arc 발견·연구, (3) SuperMAG·SuperDARN·EISCAT 3D 교차검증, (4) 오로라 분류용 머신러닝 학습 자료, (5) 동화형 전리권 모델 입력 자료로 활용된다. *저가 상용 CCD + 맞춤 광학 + GPS 동기 + 상용 위성 업링크* 라는 시스템 아키텍처는 이후 REGO·NORSTAR·TREx 등 후속 array의 표준 설계가 되었다.

---

## Q&A

(읽기 세션 중 추가됨 / Populated during reading session)

**Q1. 왜 narrow-band 필터(예: 557.7 nm) 대신 panchromatic을 선택했는가?**
- **English**: Three reasons. (1) **Sensitivity**: a 5 nm bandpass loses ~95 % of the auroral broadband flux, requiring ~20× longer exposure or much more expensive intensified cameras to maintain SNR at the lower-latitude (Athabasca) sites where onset arcs are dimmer. (2) **Cost & reliability**: removing the filter wheel eliminates a moving part, simplifies thermal control, and brings the camera price into commercial territory (Starlight Express MX716 ≪ $10k vs. ICCD ~$100k). (3) **Science scope**: panchromatic captures both 557.7 nm onset arcs *and* 427.8 nm proton aurora, 630 nm cusp/SAR-arc emissions in one instrument — useful for the broader auroral science program. The trade-off is loss of altitude/species discrimination, which is recovered partially by co-located CANOPUS MSP red-line photometers.
- **한국어**: 세 가지 이유. (1) **감도**: 5 nm bandpass는 광대역 오로라 광속의 ~95 %를 잃어, 어두운 저위도(Athabasca) onset arc 검출을 위해 ~20× 긴 노출이나 훨씬 비싼 ICCD가 필요. (2) **비용·신뢰성**: 필터휠 제거로 movable part 감소, 열 제어 간소화, 카메라가 상용 가격대(Starlight Express MX716 ≪ $10k vs. ICCD ~$100k)로 진입. (3) **과학 범위**: 557.7 nm onset arc + 427.8 nm 양성자 오로라 + 630 nm cusp/SAR-arc를 하나의 기기로 모두 포착. 단점인 고도/종류 구분 손실은 공동 배치된 CANOPUS MSP red-line 광도계로 일부 회복.

**Q2. 왜 GPS 동기화가 mission-critical인가?**
- **English**: The 1° onset-meridian requirement equates to ~110 km on the ground at 65° magnetic latitude (since 1° of longitude ≈ 110 km × cos(latitude) ≈ 47 km, and including obliquity factors it works out to roughly the spacing between adjacent ASI sites). To unambiguously identify which station first sees the brightening, all stations must agree on time to better than the cadence (3 s). In practice, GPS receivers deliver ≪ 1 ms accuracy, so timing is dominated by exposure mid-time uncertainty (~0.5 s). Without GPS, NTP over commercial satellite links would degrade to seconds-level offsets, mixing cause and effect across the array.
- **한국어**: 1° onset 자오선 요구는 65° 자기위도에서 지상 ~110 km(인접 ASI 간격에 해당). 어느 사이트가 먼저 brightening을 봤는지 확정하려면 모든 사이트가 cadence(3 s)보다 정확하게 시각이 일치해야 한다. GPS 수신기는 ≪ 1 ms 정확도를 주므로 노출 중심 시각 불확정도(~0.5 s)가 지배. GPS 없이 NTP만 쓰면 상용 위성 링크 지연으로 초 단위 어긋남이 발생해 cause/effect 순서를 혼동할 수 있다.

**Q3. Athabasca 사례에서 왜 "fully developed substorm"이라 결론지었는가? (cf. pseudobreakup)**
- **English**: Three converging signatures. (1) **Red-line poleward motion**: the CANOPUS MSP at Fort Smith / Gillam shows the 630 nm separatrix moving poleward — a marker of lobe-line reconnection per Blanchard et al. (1995). Pseudobreakups don't reach this. (2) **Dispersionless injection at LANL 1994-084**: the riometer absorption spike at Fort Smith plus dispersed ion injection arriving after 0620 UT at the geosynchronous spacecraft (which sits at ~1600 MLT) confirms global current-wedge formation. (3) **Persistent dipolarization at GOES**: the magnetic-inclination jump at GOES is sustained, not transient. Together these meet the multi-criterion definition of full onset, distinguishing this from a pseudobreakup that would show only localized brightening and rapid recovery.
- **한국어**: 세 가지 수렴 시그니처. (1) **Red-line의 자극 방향 확장**: Fort Smith/Gillam MSP의 630 nm separatrix가 자극(poleward)으로 이동 — Blanchard et al. (1995) 기준 lobe reconnection 표지. Pseudobreakup은 여기까지 가지 않는다. (2) **LANL 1994-084의 dispersionless injection**: Fort Smith 리오미터 흡수 spike와 0620 UT 이후 위성에서 분산된 이온 주입이 관측되어 global current wedge 형성을 확인. (3) **GOES의 지속적 dipolarization**: 자기경사 점프가 일시적이지 않고 지속. 이들이 종합되어 full onset의 다중 기준을 만족.

**Q4. 800 km 적도 면 azimuthal 파장은 어떻게 mapping되는가?**
- **English**: At Athabasca (61.5° MLAT, L ≈ 4.4 in dipole), the equatorial radial distance of the foot point's field-line apex is ~4.4 RE. Using ionospheric latitude width of 50 km / (R_E × cos λ) ≈ 50/(6371·cos 61.5°) ≈ 1.65×10⁻² rad and the corresponding equatorial azimuthal arc length 1.65×10⁻² × 4.4 RE × cos(0°) ≈ 462 km in a pure dipole. Donovan et al. quote ~800 km because the late-growth-phase magnetosphere is *stretched* — the actual L-shell reaches ~7–10 RE, and the magnetic-flux-tube cross-section expands accordingly. So the factor ~1.7× difference between dipole (~460 km) and stretched (~800 km) mapping is itself a constraint on the magnetotail topology at onset.
- **한국어**: Athabasca (61.5° MLAT, dipole L ≈ 4.4)에서 전리권 50 km는 적도면에서 dipole이라면 ~462 km. 본 논문이 ~800 km를 인용한 것은 growth phase 후반 자기권이 *stretch*되어 실제 L-shell이 ~7–10 RE까지 늘어나고 자력선 단면이 확장되기 때문. dipole(~460 km)과 stretched(~800 km) 매핑의 ~1.7× 차이 자체가 onset 시점의 magnetotail topology 제약 조건이 된다.
