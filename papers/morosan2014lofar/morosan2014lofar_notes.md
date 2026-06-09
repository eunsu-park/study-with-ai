---
title: "LOFAR Tied-Array Imaging of Type III Solar Radio Bursts"
authors: D. E. Morosan, P. T. Gallagher, P. Zucca, R. Fallows, E. P. Carley, G. Mann, et al.
year: 2014
journal: "Astronomy & Astrophysics"
doi: "10.1051/0004-6361/201423936"
topic: Solar_Observation
tags: [LOFAR, type_III_burst, radio_burst, tied_array, CME, coronal_density, plasma_emission]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 28. LOFAR Tied-Array Imaging of Type III Solar Radio Bursts / LOFAR Tied-Array 이미징을 이용한 Type III 태양 전파 폭발 관측

---

## 1. Core Contribution / 핵심 기여

**English.** Morosan et al. (2014) present the first use of LOFAR's tied-array beam-formed observing mode to image solar Type III radio bursts in the 30-90 MHz band with ~83 ms temporal and 12.5 kHz spectral resolution. 126 simultaneous beams were pointed at the Sun in a honeycomb pattern covering the station's ~3.3° field of view (≲5 R_sun around the solar centre). Over a 30-minute interval on 28 February 2013, more than 30 Type III bursts were detected. Each beam produced a high-cadence dynamic spectrum, enabling the authors to build frequency-resolved radio images by interpolating intensity across the beam grid. This recovers the spatial information that interferometric LOFAR output (limited to ~1 image per second) cannot. Their central discovery is twofold: (i) several Type III bursts emit at altitudes near 4 R_sun at 30 MHz — far higher than the 1-D radial density models of Newkirk (1961), Mann et al. (1999), and Zucca et al. (2014) would allow; and (ii) these high-altitude bursts follow non-radial trajectories whose southern flank is cospatial with the leg of a slow CME (~250 km/s) imaged by LASCO/C3. The authors conclude that CME-driven compression of a coronal streamer locally enhanced the electron density (required n_e ~ 3.3×10^6 cm^-3 at 32.5 MHz vs background ~10^5 cm^-3), and that CME expansion deflected radial magnetic fields to non-radial directions, channelling the electron beams accordingly.

**한국어.** Morosan et al. (2014)는 LOFAR의 tied-array 빔 형성 관측 모드를 이용하여 30-90 MHz 대역에서 Type III 태양 전파 폭발을 ~83 ms 시간 분해능 및 12.5 kHz 스펙트럼 분해능으로 영상화한 최초의 연구이다. 태양 중심 주변 ≲5 R_sun 영역을 덮기 위해 ~3.3°의 스테이션 시야각을 벌집 패턴으로 배열한 126개의 동시 빔이 태양을 지향하였다. 2013년 2월 28일 30분간 30개 이상의 Type III 폭발이 탐지되었다. 각 빔은 고-시간 분해능 동적 스펙트럼을 생성하였고, 저자들은 빔 격자에서 강도를 보간함으로써 주파수 분해 전파 이미지를 구성할 수 있었다. 이는 (초당 ~1 이미지로 제한된) LOFAR의 표준 간섭계 출력으로는 회복할 수 없는 공간 정보를 복원한다. 본 논문의 핵심 발견은 두 가지이다: (i) 여러 Type III 폭발이 30 MHz에서 ~4 R_sun 근처에서 방출되며, 이는 Newkirk (1961), Mann et al. (1999), Zucca et al. (2014)의 1차원 방사형 밀도 모델 예측을 크게 상회한다; (ii) 이러한 고고도 폭발은 비방사형 궤적을 따르며, 그 남쪽 플랭크는 LASCO/C3로 관측된 느린 CME (~250 km/s)의 다리와 공간적으로 일치한다. 저자들은 CME에 의한 코로나 스트리머 압축이 국소 전자 밀도를 상승시켰고 (32.5 MHz에서 n_e ~ 3.3×10^6 cm^-3이 필요; 배경 ~10^5 cm^-3), CME 팽창이 방사형 자기장을 비방사형으로 편향시켜 전자 빔을 그에 맞게 유도했다고 결론 지었다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Type III Bursts and the Metre-Wave Gap / 서론 — Type III 폭발과 m파 관측 공백

**English (p. 1-2).** The paper opens with the Wild (1963) five-class taxonomy of solar radio bursts (I-V). Among these, Type III is the most rapidly drifting: frequencies sweep from high to low at rates from -1 MHz/s (at 20 MHz) up to -20 MHz/s (at 100 MHz) (Abranin et al. 1990; Mann & Klassen 2002). Observed frequencies span an enormous range — from kHz (Krupar et al. 2010) up to 8 GHz (Ma et al. 2012) — but most bursts concentrate below 150 MHz (Saint-Hilaire et al. 2013). Brightness temperatures reach up to 10^12 K for coronal bursts. The canonical emission mechanism (Lin 1974): electron beams accelerated during magnetic reconnection or shock episodes escape along open field lines; faster electrons outpace slower ones, producing a bump-on-tail velocity distribution; this instability drives Langmuir (electron plasma) waves (Robinson et al. 1993); Langmuir waves couple nonlinearly into electromagnetic radiation at the local plasma frequency f_p and its second harmonic 2f_p (Bastian, Benz & Gary 1998). Higher harmonics are rare (Zlotnik et al. 1998). Li & Cairns (2013) proposed that plasmas with kappa-distributed suprathermal tails can enhance fundamental emission intensity above Maxwellian plasmas. The Sun was previously imaged at kHz-10 MHz by Ulysses/WIND/STEREO and at ≥150 MHz by the Nancay Radioheliograph; the 10-150 MHz range — the critical middle corona — remained spatially unconstrained apart from a few bursts imaged by Culgoora at 80 MHz (Wild 1967). The arrival of LOFAR (van Haarlem et al. 2013) and MWA (Tingay et al. 2013), full-band arrays spanning 10-240 MHz (LOFAR) with collecting areas ~10^4-10^5 m^2, finally opens this observational window.

**한국어 (p. 1-2).** 논문은 Wild (1963)의 다섯 종류 태양 전파 폭발 분류(I-V)로 시작한다. 이 중 Type III가 가장 빠른 드리프트를 보인다: 주파수가 고→저로 -1 MHz/s (20 MHz)에서 -20 MHz/s (100 MHz)의 비율로 변화한다 (Abranin et al. 1990; Mann & Klassen 2002). 관측 주파수 범위는 매우 넓어 kHz (Krupar et al. 2010)부터 8 GHz (Ma et al. 2012)까지 이르지만, 대부분은 150 MHz 이하에 집중된다 (Saint-Hilaire et al. 2013). 코로나 Type III의 밝기 온도는 10^12 K에 이를 수 있다. 표준 방출 기작은 (Lin 1974): 자기 재결합이나 충격파로 가속된 전자 빔이 개방 자기력선을 따라 탈출; 빠른 전자가 느린 전자를 앞지르며 bump-on-tail 속도 분포를 형성; 이 불안정성이 Langmuir 파(전자 플라스마 파)를 생성 (Robinson et al. 1993); Langmuir 파가 비선형적으로 국소 플라스마 진동수 f_p 및 2배 고조파 2f_p에서 전자기파로 변환 (Bastian, Benz & Gary 1998). 더 높은 고조파는 희귀하다 (Zlotnik et al. 1998). Li & Cairns (2013)는 kappa 분포 초열(suprathermal) 꼬리를 갖는 플라스마가 Maxwell 플라스마보다 기본파 방출 강도를 강화할 수 있다고 제안했다. 태양은 이전에 kHz-10 MHz에서 Ulysses/WIND/STEREO에 의해, ≥150 MHz에서 Nancay Radioheliograph에 의해 이미징되었으나, 10-150 MHz 대역(핵심 중간 코로나)은 Culgoora가 80 MHz에서 관측한 일부 폭발 외에는 공간적으로 제약되지 않은 상태였다 (Wild 1967). LOFAR (van Haarlem et al. 2013)와 MWA (Tingay et al. 2013) — 10-240 MHz (LOFAR)의 전 대역을 ~10^4-10^5 m^2의 수집 면적으로 다루는 배열 — 의 등장으로 이 관측 창이 드디어 열렸다.

### Part II: LOFAR Instrument and Tied-Array Mode / LOFAR 기기와 Tied-Array 모드

**English (Sect. 2.1, p. 2-3).** LOFAR consists of thousands of dipole antennas organized as 24 core stations and 16 remote stations in the Netherlands, plus 8 international stations across Europe. Two antenna types cover different bands: Low Band Antennas (LBAs) for 10-90 MHz and High Band Antennas (HBAs) for 110-240 MHz. LOFAR's collecting area is ~35,000 m^2 at 30 MHz, yielding a sensitivity two orders of magnitude above previous metre-wave heliographs (Mann et al. 2011). For this observation, only LBAs of the 24 core stations (maximum baseline ~2 km, located in Exloo, Netherlands) were used. The observing mode is beam-formed rather than interferometric: per-station voltage streams are coherently combined into "tied-array beams" with coefficient weights w_n and phase delays encoding the desired pointing. This preserves the full time-frequency resolution of the raw sampling: 12.5 kHz × 83 ms in this case, orders of magnitude finer than standard LOFAR interferometric imaging (~1 image/s). 126 beams were deployed in a honeycomb (hexagonal close-pack) pattern with inter-beam spacing ~14' and individual beam FWHMs of 7' (at 90 MHz) to 21' (at 30 MHz). At 45 MHz the beams are tangent; below 45 MHz they overlap. Radio frequency interference (RFI) below 30 MHz forced the analysis band to 30-90 MHz. No absolute calibrator was used, so intensities are in arbitrary units.

**한국어 (Sect. 2.1, p. 2-3).** LOFAR는 24개 코어 스테이션과 네덜란드 내 16개 원격 스테이션, 그리고 유럽 전역의 8개 국제 스테이션에 분포된 수천 개의 다이폴 안테나로 구성된다. 두 안테나 유형이 서로 다른 대역을 담당한다: 10-90 MHz의 Low Band Antenna (LBA)와 110-240 MHz의 High Band Antenna (HBA). LOFAR의 수집 면적은 30 MHz에서 ~35,000 m^2로, 이전 m파 헬리오그래프보다 감도가 2차수 정도 높다 (Mann et al. 2011). 이 관측에서는 네덜란드 Exloo에 위치한 24개 코어 스테이션의 LBA (최대 기선 ~2 km)만 사용되었다. 관측 모드는 간섭계가 아닌 빔 형성(beam-formed) 방식이다: 각 스테이션의 전압 스트림이 계수 w_n과 원하는 지향 방향을 인코딩한 위상 지연으로 결맞음 합산되어 "tied-array 빔"을 형성한다. 이로써 원시 샘플링의 전체 시간-주파수 분해능 — 이 경우 12.5 kHz × 83 ms — 이 보존되며, 이는 표준 LOFAR 간섭계 이미징(~1 image/s)보다 차수가 다르게 세밀하다. 126개 빔이 빔 간격 ~14'의 벌집(육각 밀집) 패턴으로 배치되었고, 개별 빔 FWHM은 7' (90 MHz) - 21' (30 MHz)이다. 45 MHz에서 빔이 서로 접하며, 45 MHz 이하에서는 겹친다. 30 MHz 이하의 RFI로 인해 분석 대역은 30-90 MHz로 제한되었다. 절대 검정자는 사용되지 않아 강도는 임의 단위로 표현된다.

### Part III: Data Analysis and Imaging Method / 자료 분석과 이미징 방법

**English (Sect. 2.2, p. 3).** Each of the 126 beams produces a 30-90 MHz dynamic spectrum. To form a radio image at a particular frequency-time slice, the intensity in each of the 126 dynamic spectra at that (f, t) is taken as a "macro-pixel" placed at the beam's sky position — yielding a honeycomb-sampled intensity map. The authors average over 5 MHz in frequency and 1 s in time to suppress spectral noise and stabilize morphology. Since the honeycomb sampling is sparse, a smooth quintic interpolation is applied to produce a regularly gridded 2-D image. From each such image a centroid (brightness-weighted position) is extracted and converted from arcmin on sky into solar radii from Sun centre. Centroids across frequencies (5 MHz bins from 30-35, 35-40, …, 55-60 MHz) build distance-frequency plots (Fig. 3); centroids across time build distance-time plots (Fig. 6) from which radial velocities are measured via the slope.

Error sources: (i) beam FWHM of 7'-21' → ~1.3 R_sun positional uncertainty at 30 MHz; (ii) ionospheric refraction at the Sun's low elevation (~30°) estimated following Stewart & McLean (1982) and Mercier (1996) at ~2' maximum — negligible compared to the beam. The quiet-Sun maximum-intensity point coincided with the centre beam to within 0.5 R_sun across 30-90 MHz, confirming that no absolute astrometric correction was needed.

**한국어 (Sect. 2.2, p. 3).** 126개 빔 각각이 30-90 MHz 동적 스펙트럼을 생성한다. 특정 주파수-시간 슬라이스에서의 전파 이미지를 형성하기 위해, 126개 동적 스펙트럼 각각의 (f, t)에서의 강도를 빔의 하늘 위치에 놓인 "매크로 픽셀"로 취하여, 벌집 샘플된 강도 맵을 얻는다. 저자들은 스펙트럼 잡음을 억제하고 형태를 안정화하기 위해 주파수에서 5 MHz, 시간에서 1 s로 평균한다. 벌집 샘플링이 성긴 분포이므로, 균일한 격자의 2D 이미지를 생성하기 위해 매끄러운 quintic 보간을 적용한다. 각 이미지에서 중심(밝기 가중 위치)이 추출되어 하늘의 arcmin 단위에서 태양 중심으로부터의 태양 반경 단위로 변환된다. 주파수별 중심(30-35, 35-40, …, 55-60 MHz의 5 MHz 빈)은 거리-주파수 플롯 (Fig. 3)을 구성하고, 시간별 중심은 거리-시간 플롯 (Fig. 6)을 구성하여 기울기로부터 방사 속도를 측정한다.

오차 원천: (i) 빔 FWHM 7'-21' → 30 MHz에서 ~1.3 R_sun의 위치 불확실성; (ii) 태양 저각(~30°)에서의 전리층 굴절 — Stewart & McLean (1982)과 Mercier (1996)에 따라 최대 ~2'로 추정되어 빔에 비하면 무시 가능. 잠잠한 태양의 최대 강도 지점이 30-90 MHz에 걸쳐 중심 빔과 0.5 R_sun 이내에서 일치하여, 절대 천체측정 보정은 필요하지 않음이 확인되었다.

### Part IV: Type III Spectral Characteristics / Type III 스펙트럼 특성

**English (Sect. 3.1, p. 3-4).** During the 30-minute run on 2013 Feb 28 13:00-13:30 UT, 32 Type III bursts were identified above an intensity threshold (Fig. 1b). The GOES X-ray flux (Fig. 1c) shows no flares in the preceding or concurrent interval; only minor B-class flares occurred afterward. However, coronagraphs on STEREO and SOHO/LASCO detected several CMEs in the hours before the LOFAR observation, apparently originating from the far side of the Sun. The Type IIIs are best described as a Type III storm — a quasi-continuous episode of bursts not obviously triggered by a flare.

Measured drift rates (df/dt) range from -2 to -17 MHz/s in the 30-60 MHz band, with mean -7 MHz/s. This is consistent with Mann & Klassen (2002), who reported -11 MHz/s at 40-70 MHz. Frequency-to-density conversion is performed via Eq. (1):

$$
f = n\,f_p = n\,C\,\sqrt{N_e}\ \mathrm{Hz},\quad C = 8980\ \mathrm{Hz\ cm^{3/2}}
$$

Radial velocity via Eq. (2):

$$
v = \frac{2\sqrt{N_e}}{C}\left(\frac{dN_e}{dr}\right)^{-1}\frac{df}{dt}
$$

The authors assume harmonic (n=2) emission throughout: the sources appear at altitudes where harmonic radiation dominates because fundamental emission suffers stronger free-free absorption (Bastian et al. 1998). This choice halves the implied local density (n_e = (f/(2C))^2 instead of (f/C)^2).

**한국어 (Sect. 3.1, p. 3-4).** 2013년 2월 28일 13:00-13:30 UT의 30분 관측 동안, 강도 임계값 이상에서 32개의 Type III 폭발이 식별되었다 (Fig. 1b). GOES X선 플럭스 (Fig. 1c)는 이전 혹은 동시 구간에 플레어가 없음을 보이며, 그 이후에만 소규모 B급 플레어가 발생하였다. 다만, STEREO와 SOHO/LASCO의 코로나그래프가 LOFAR 관측 전 수 시간 이내에 태양 뒷면에서 발원한 것으로 보이는 여러 CME를 포착하였다. 이러한 Type III는 플레어에 의해 명확히 촉발되지 않은 준연속 폭발 시퀀스인 Type III 스톰(storm)으로 가장 잘 설명된다.

측정된 드리프트 율(df/dt)은 30-60 MHz 대역에서 -2에서 -17 MHz/s까지 범위이며 평균 -7 MHz/s이다. 이는 40-70 MHz에서 -11 MHz/s를 보고한 Mann & Klassen (2002)과 일치한다. 주파수-밀도 변환은 Eq. (1)을 통해 수행되고, 방사 속도는 Eq. (2)를 통해 계산된다. 저자들은 전 구간에서 고조파 (n=2) 방출을 가정한다: 소스가 나타나는 고도에서는 기본파가 자유-자유 흡수를 더 강하게 받기 때문에 고조파 방사가 지배적이다 (Bastian et al. 1998). 이 선택은 암시되는 국소 밀도를 절반으로 줄인다 (n_e = (f/(2C))^2 대신 (f/C)^2가 아닌).

### Part V: Type III Spatial Characteristics and the 2-D Density Map / Type III 공간 특성과 2D 밀도 지도

**English (Sect. 3.2, p. 4-5).** Fig. 2 shows tied-array images of Burst 1 at three frequencies, 50-55, 40-45, and 30-35 MHz, separated by 1 s. Lower frequencies image the source at larger heights, confirming an outward-moving beam. Fig. 3a plots all 32-burst centroids across 30-60 MHz; Fig. 3b shows the Zucca et al. (2014) 2-D electron density map for the same epoch, constructed by combining SDO/AIA differential emission measure (DEM, 1-1.3 R_sun), LASCO/C2 polarized brightness (2.5-5 R_sun), and a plane-parallel + spherically-symmetric model in between (1.3-2.5 R_sun). Contours on the map mark the density levels required for 30 MHz and 60 MHz harmonic emission (n=2 in Eq. 1): n_e = 2.8×10^6 cm^-3 (30 MHz) and 1.1×10^7 cm^-3 (60 MHz). Most centroids straddle these contours as expected, but several bursts — notably Burst 2 — lie far outside them at heights approaching 4 R_sun. At Burst 2's position, the Zucca map gives n_e ~ 10^5 cm^-3 — an order of magnitude below the 3.3×10^6 cm^-3 required for emission at 32.5 MHz. Something beyond the quiet-streamer corona is needed.

**한국어 (Sect. 3.2, p. 4-5).** Fig. 2는 Burst 1의 tied-array 이미지를 50-55, 40-45, 30-35 MHz의 세 주파수에서 1초 간격으로 보여준다. 낮은 주파수가 더 높은 고도에서 소스를 영상화하며, 이는 외향 이동 빔임을 확인한다. Fig. 3a는 30-60 MHz에 걸친 32개 폭발 모든 중심을 플롯하며, Fig. 3b는 동일 시점에 대한 Zucca et al. (2014)의 2D 전자 밀도 지도를 보여준다. 이 지도는 SDO/AIA 차등 방출 측정(DEM, 1-1.3 R_sun), LASCO/C2 편광 밝기(2.5-5 R_sun), 그리고 그 사이(1.3-2.5 R_sun)의 평면-병렬 + 구형 대칭 모델의 결합으로 구성된다. 지도 위 등치선은 30 MHz 및 60 MHz 고조파 방출에 필요한 밀도 수준을 표시한다 (Eq. 1에서 n=2): n_e = 2.8×10^6 cm^-3 (30 MHz)과 1.1×10^7 cm^-3 (60 MHz). 대부분의 중심은 예상대로 이 등치선의 양쪽에 위치하지만, 여러 폭발 — 특히 Burst 2 — 은 등치선에서 멀리 벗어나 4 R_sun에 근접한 고도에 놓여 있다. Burst 2 위치에서 Zucca 지도는 n_e ~ 10^5 cm^-3을 제공하며, 이는 32.5 MHz 방출에 필요한 3.3×10^6 cm^-3보다 한 차수 낮다. 잠잠한 스트리머 코로나 이상의 무언가가 필요하다.

### Part VI: High-Altitude Type III Bursts and the CME Connection / 고고도 Type III 폭발과 CME의 연관

**English (Sect. 3.3, p. 6).** Fig. 4 compares burst trajectories with Newkirk, Mann, and Zucca density curves in the distance-frequency plane. Bursts 1 and 3 have slopes steeper than all three models, suggesting locally steeper density gradients than assumed. Burst 2 sits at markedly higher altitudes (2.6-4.1 R_sun at 30-60 MHz) well above all models. Fig. 5 overlays the LOFAR 40-45 MHz contour of Burst 2 at 13:09:01 UT on a LASCO/C3 running-difference image at 13:30:23 UT — a clear CME is expanding southward, and its southern leg aligns spatially with the Burst 2 contours. Because the occulting-disk pylon obscures the line-of-sight radial to Burst 2 in LASCO, a direct density measurement there is impossible, but the streamer compression scenario is physically motivated.

Two mechanisms could raise the local density to ≥3×10^6 cm^-3 at ~4 R_sun: (i) direct plasma compression along the CME's outer surface; (ii) CME expansion impinging on a pre-existing streamer, compressing the streamer and raising its internal density. The latter is favoured because the authors observe streamer activity before and after the CME. The CME can simultaneously deflect the streamer and its embedded open field lines, producing the observed non-radial trajectories. This is consistent with numerical simulations of CME-streamer interaction by Bemporad et al. (2010).

**한국어 (Sect. 3.3, p. 6).** Fig. 4는 거리-주파수 평면에서 폭발 궤적을 Newkirk, Mann, Zucca 밀도 곡선과 비교한다. Burst 1과 3은 세 모델 모두보다 가파른 기울기를 보이며, 이는 가정된 것보다 국소적으로 더 가파른 밀도 경사를 시사한다. Burst 2는 30-60 MHz에서 2.6-4.1 R_sun의 현저하게 높은 고도에 위치하여 모든 모델 위에 있다. Fig. 5는 13:09:01 UT의 Burst 2의 LOFAR 40-45 MHz 등치선을 13:30:23 UT의 LASCO/C3 차분 이미지 위에 겹쳐 보여준다 — 남쪽으로 팽창하는 CME가 명확히 보이며, 그 남쪽 다리는 Burst 2 등치선과 공간적으로 정렬된다. LASCO의 차폐판 지주가 Burst 2 방향의 시선을 가리기 때문에 그곳에서의 직접 밀도 측정은 불가능하지만, 스트리머 압축 시나리오는 물리적으로 타당하다.

~4 R_sun에서 국소 밀도를 ≥3×10^6 cm^-3로 상승시킬 수 있는 두 기작: (i) CME 외곽면을 따른 직접 플라스마 압축; (ii) CME 팽창이 기존 스트리머에 충돌하여 스트리머를 압축하고 내부 밀도를 상승시킴. 후자가 선호되는데, 저자들이 CME 전후에 스트리머 활동을 관측했기 때문이다. CME는 스트리머와 거기에 내장된 개방 자기력선을 동시에 편향시켜 관측된 비방사형 궤적을 생성할 수 있다. 이는 Bemporad et al. (2010)의 CME-스트리머 상호작용 수치 모의와 일치한다.

### Part VII: Type III Kinematics — Velocities Higher Than Expected / Type III 운동학 — 예상보다 빠른 속도

**English (Sect. 3.4, p. 6-7).** Fig. 6 shows distance-time plots for Bursts 1, 2, 3 with error bars set by the FWHM of the beams. Linear fits give radial velocities:
- Burst 1: 0.51 ± 0.18 c
- Burst 2: 0.27 ± 0.11 c
- Burst 3: 0.58 ± 0.17 c

Table 1 summarizes the comparison with density-model-derived velocities via Eq. (2):
- Burst 1: Mann 0.11 c, Newkirk 0.20 c, Zucca 0.14 c (LOFAR: 0.51 c)
- Burst 2: Mann 0.05 c, Newkirk 0.09 c, Zucca 0.05 c (LOFAR: 0.27 c)
- Burst 3: Mann 0.07 c, Newkirk 0.12 c, Zucca 0.08 c (LOFAR: 0.58 c)

LOFAR imaging velocities exceed model-predicted values by factors of ~2-10. The standard literature range for Type III velocities is 0.05-0.30 c (Dulk et al. 1987; Lin et al. 1981, 1986), and Wild et al. (1959) already noted some bursts above 0.3 c averaging 0.45 c. The high velocities here are unlikely to be a beam-size artifact because the source displacements exceed the beam FWHM, and these FWHM-based errors are already included in the quoted uncertainties. The implication is that the density models systematically underestimate local densities and, therefore, overestimate the density gradient's role in drift-rate interpretation — particularly in CME-compressed environments.

**한국어 (Sect. 3.4, p. 6-7).** Fig. 6은 빔의 FWHM로 설정된 오차 막대를 갖는 Burst 1, 2, 3의 거리-시간 플롯을 보여준다. 선형 적합으로 얻은 방사 속도:
- Burst 1: 0.51 ± 0.18 c
- Burst 2: 0.27 ± 0.11 c
- Burst 3: 0.58 ± 0.17 c

Table 1은 Eq. (2)를 통한 밀도 모델 기반 속도와의 비교를 요약한다:
- Burst 1: Mann 0.11 c, Newkirk 0.20 c, Zucca 0.14 c (LOFAR: 0.51 c)
- Burst 2: Mann 0.05 c, Newkirk 0.09 c, Zucca 0.05 c (LOFAR: 0.27 c)
- Burst 3: Mann 0.07 c, Newkirk 0.12 c, Zucca 0.08 c (LOFAR: 0.58 c)

LOFAR 이미징 속도가 모델 예측값을 ~2-10배 초과한다. Type III 속도의 표준 문헌 범위는 0.05-0.30 c (Dulk et al. 1987; Lin et al. 1981, 1986)이며, Wild et al. (1959)은 이미 0.3 c를 넘어 평균 0.45 c에 이르는 일부 폭발을 지적한 바 있다. 여기서의 높은 속도가 빔 크기 아티팩트일 가능성은 낮은데, 소스 변위가 빔 FWHM을 초과하며 이 FWHM 기반 오차가 이미 인용된 불확실성에 포함되어 있기 때문이다. 이것이 함의하는 바는 밀도 모델이 국소 밀도를 체계적으로 과소평가하고, 따라서 드리프트 율 해석에서 밀도 경사의 역할을 과대평가한다는 것이며 — 특히 CME로 압축된 환경에서 그렇다.

### Part VII-b: A Closer Look at Burst 1, 2, 3 / Burst 1, 2, 3의 상세 비교

**English.** To cement the quantitative contrast between the three imaged bursts, the following merged picture is useful:

- **Burst 1** (seen in Beam 4, triangle marker in Fig. 2): start distance 1.18 ± 0.7 R_sun at 50-55 MHz, end distance 2.90 ± 1.2 R_sun at 30-35 MHz; LOFAR velocity 0.51 c. This burst's start is close to the solar limb, implying an origin in the low corona and relatively normal propagation through the ambient streamer structure. Its velocity is high for a standard Type III but not unprecedented.
- **Burst 2** (seen in Beam 24, square marker in Fig. 2): start distance 2.64 ± 0.7 R_sun at 50-55 MHz, end distance 4.06 ± 1.2 R_sun at 30-35 MHz; LOFAR velocity 0.27 c. This is the key anomalous burst — starting high and ending higher. The density required (3-5×10^6 cm^-3 across the band) cannot be met by any quiet-coronal model at those heights.
- **Burst 3** (imaged at 13:10 UT from Beam 24): start 1.16 ± 0.7 R_sun, end 2.77 ± 1.2 R_sun; LOFAR velocity 0.58 c — the fastest of the three. Its trajectory appears more radial than Burst 2's.

The imaging geometry allows the authors to distinguish these bursts unambiguously: two simultaneous Type IIIs at different sky positions (triangle vs square beams) produce distinguishable patches in the macro-pixel interpolation, which a single-beam spectrometer could not separate.

**한국어.** 세 개의 영상화된 폭발 사이의 정량적 대비를 확고히 하기 위해 다음 통합 그림이 유용하다:

- **Burst 1** (Beam 4에서 관측, Fig. 2의 삼각형 마커): 50-55 MHz에서 시작 거리 1.18 ± 0.7 R_sun, 30-35 MHz에서 종료 거리 2.90 ± 1.2 R_sun; LOFAR 속도 0.51 c. 이 폭발의 시작은 태양 림에 가까워 저고도 코로나에서 기원하며 주변 스트리머 구조를 통한 비교적 일반적인 전파를 시사한다. 속도는 표준 Type III보다 빠르지만 전례가 없는 것은 아니다.
- **Burst 2** (Beam 24에서 관측, Fig. 2의 사각형 마커): 50-55 MHz에서 시작 거리 2.64 ± 0.7 R_sun, 30-35 MHz에서 종료 거리 4.06 ± 1.2 R_sun; LOFAR 속도 0.27 c. 핵심 이상 폭발로 — 높게 시작하여 더 높게 끝난다. 필요 밀도 (대역 전체에 걸쳐 3-5×10^6 cm^-3)는 해당 고도에서 어떤 정상 코로나 모델로도 충족될 수 없다.
- **Burst 3** (13:10 UT에 Beam 24에서 영상화): 시작 1.16 ± 0.7 R_sun, 종료 2.77 ± 1.2 R_sun; LOFAR 속도 0.58 c — 세 폭발 중 가장 빠르다. 궤적은 Burst 2보다 더 방사형으로 보인다.

이미징 기하는 저자들이 이 폭발들을 명확히 구별할 수 있게 한다: 서로 다른 하늘 위치의 두 동시 Type III (삼각형 대 사각형 빔)는 macro-pixel 보간에서 구별 가능한 패치를 생성하는데, 단일 빔 분광기로는 분리할 수 없는 것이다.

### Part VIII: Discussion — Acceleration Mechanism and Methodological Outlook / 논의 — 가속 기작과 방법론 전망

**English (Sect. 4, p. 7).** Two candidate acceleration mechanisms for the high-altitude electron beams:
1. **CME shock**: Carley et al. (2013) showed herringbone-generating electrons at ~0.15 c associated with a CME-shock Type II burst. However, the 28 Feb 2013 CME was slow (v_CME ~ 250 km/s). The local Alfvén speed at 4 R_sun is estimated at ~220 km/s, yielding Alfvén Mach number ≲1.1 — barely super-Alfvénic. Such a weak shock is inefficient at accelerating particles to the ~0.1 c required to produce Type III-like beams.
2. **Side reconnection along CME flanks**: Bemporad et al. (2010) identified multiple reconnection sites between expanding CMEs and surrounding streamers, confirmed by numerical simulations. If the southern leg of the CME sits adjacent to an open field line, reconnection-accelerated electrons can escape into the compressed streamer, producing the observed high-altitude Type IIIs along non-radial paths set by field deflection.

The authors favour scenario (2) given the slow CME. They also note technical improvements: adding LOFAR remote stations would shrink the beams from 7'-21' to ~3.5'-10', enabling finer localization, though ionospheric effects become more significant at smaller scales. Simultaneous LBA+HBA operation would allow full 10-240 MHz spectral imaging. The tied-array method is immediately applicable to Type II, Type IV, and other transient phenomena within LOFAR's band.

**한국어 (Sect. 4, p. 7).** 고고도 전자 빔의 가속 기작에 대한 두 후보:
1. **CME 충격파**: Carley et al. (2013)은 CME 충격파 Type II 폭발에 수반된 ~0.15 c의 herringbone 전자를 보였다. 그러나 2013년 2월 28일의 CME는 느렸다 (v_CME ~ 250 km/s). 4 R_sun에서의 국소 Alfvén 속도는 ~220 km/s로 추정되어, Alfvén Mach 수가 ≲1.1로 간신히 초-Alfvén적이다. 이러한 약한 충격파는 Type III와 유사한 빔을 만드는 데 필요한 ~0.1 c로 입자를 가속하는 데 비효율적이다.
2. **CME 플랭크를 따른 측면 재결합**: Bemporad et al. (2010)은 팽창하는 CME와 주변 스트리머 사이의 다수 재결합 지점을 확인하였고, 이는 수치 모의로 검증되었다. CME의 남쪽 다리가 개방 자기력선에 인접해 있다면, 재결합 가속 전자가 압축된 스트리머 내부로 탈출하여 자기장 편향에 의해 비방사형 경로를 따르는 관측된 고고도 Type III를 생성할 수 있다.

느린 CME를 고려할 때 저자들은 시나리오 (2)를 선호한다. 또한 기술적 개선을 언급한다: LOFAR 원격 스테이션 추가로 빔을 7'-21'에서 ~3.5'-10'로 축소하여 더 정밀한 위치 파악이 가능할 것이나, 더 작은 스케일에서는 전리층 효과가 더 유의미해진다. 동시적 LBA+HBA 운영은 10-240 MHz 전 대역 스펙트럼 이미징을 가능하게 한다. tied-array 방법은 LOFAR 대역 내의 Type II, Type IV 및 기타 과도 현상에 즉시 적용 가능하다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Tied-array imaging converts beam-formed voltages into spatial maps / Tied-array 이미징은 빔 형성 전압을 공간 지도로 변환한다** — By pointing 126 simultaneous beams at the Sun in a honeycomb, the authors retain the raw time-frequency resolution (~83 ms × 12.5 kHz) while also recovering imaging capability, breaking the standard LOFAR interferometric bottleneck of ~1 image/s. / 126개 빔을 벌집 패턴으로 태양에 동시 지향함으로써, 저자들은 원시 시간-주파수 분해능 (~83 ms × 12.5 kHz)을 유지하면서 이미징 능력을 복원하여 LOFAR 표준 간섭계의 ~1 image/s 병목을 돌파한다.

2. **30 MHz bursts can occur at 4 R_sun — contradicting 1-D density models / 30 MHz 폭발이 4 R_sun에서 발생할 수 있어 1D 밀도 모델과 모순된다** — Burst 2 at ~4 R_sun requires n_e ~ 3.3×10^6 cm^-3 for harmonic emission, yet the background Zucca map there has only ~10^5 cm^-3. Quiet coronal density models systematically fail in CME-perturbed regions. / Burst 2는 ~4 R_sun에서 고조파 방출에 n_e ~ 3.3×10^6 cm^-3을 요구하나, 그곳의 배경 Zucca 지도는 ~10^5 cm^-3에 불과하다. 정상 코로나 밀도 모델은 CME가 교란한 영역에서 체계적으로 실패한다.

3. **Type III trajectories need not be radial / Type III 궤적이 반드시 방사형이어야 할 이유는 없다** — Burst 2's non-radial propagation tracks the deflected field of a slow CME's southern flank. Radial density models (Newkirk, Mann) implicitly assume radial trajectories; the 2-D Zucca map is necessary but still insufficient without a CME update. / Burst 2의 비방사형 전파는 느린 CME의 남쪽 플랭크에서 편향된 자기장을 추적한다. 방사형 밀도 모델 (Newkirk, Mann)은 암묵적으로 방사 궤적을 가정하며, 2D Zucca 지도는 필요하지만 CME 업데이트 없이는 여전히 불충분하다.

4. **Imaging-derived velocities exceed model-derived velocities by factors of 2-10 / 이미징 기반 속도가 모델 기반 속도를 2-10배 초과한다** — LOFAR tied-array gives 0.27-0.58 c for Bursts 1-3, while Eq. (2) with Mann/Newkirk/Zucca densities gives 0.05-0.20 c. Direct position tracking sidesteps the density-gradient assumption and is more reliable in perturbed coronae. / LOFAR tied-array는 Burst 1-3에 대해 0.27-0.58 c를 주는 반면, Mann/Newkirk/Zucca 밀도를 이용한 Eq. (2)는 0.05-0.20 c를 준다. 직접 위치 추적은 밀도 경사 가정을 우회하며 교란된 코로나에서 더 신뢰할 수 있다.

5. **Slow CMEs cannot drive strong shocks but can still accelerate particles via side reconnection / 느린 CME는 강한 충격파를 구동할 수 없지만 측면 재결합을 통해 여전히 입자를 가속할 수 있다** — With v_CME ~ 250 km/s and local v_A ~ 220 km/s, the Alfvén Mach number is ~1.1, too weak for efficient shock acceleration. Bemporad et al. (2010)-style flank reconnection explains the electron beams. / v_CME ~ 250 km/s 및 국소 v_A ~ 220 km/s에서 Alfvén Mach 수는 ~1.1로 효율적인 충격 가속에는 너무 약하다. Bemporad et al. (2010) 스타일의 측면 재결합이 전자 빔을 설명한다.

6. **Harmonic (n=2) emission is the natural assumption at high altitudes / 고고도에서는 고조파(n=2) 방출 가정이 자연스럽다** — Free-free absorption depth scales as n_e^2 T^(-3/2) f^(-2); at lower densities (higher altitudes) fundamental emission is still partly absorbed while 2f_p escapes more readily. The n=2 choice halves the inferred n_e vs n=1. / 자유-자유 흡수 깊이는 n_e^2 T^(-3/2) f^(-2)로 스케일링된다; 낮은 밀도(높은 고도)에서도 기본파는 여전히 부분적으로 흡수되는 반면 2f_p는 더 쉽게 탈출한다. n=2 선택은 n=1 대비 추정 n_e를 절반으로 한다.

7. **A Type III storm can occur in the absence of detectable flares / Type III 스톰은 탐지 가능한 플레어 없이도 발생할 수 있다** — GOES shows no flares during the observation, yet 32 bursts appear. This reinforces Dulk's (1985) observation that ~90% of Type IIIs occur without associated flares or CMEs on the visible disk; electron acceleration at small spatial scales is pervasive. / 관측 중 GOES는 플레어를 보이지 않지만 32개 폭발이 나타난다. 이는 Type III의 ~90%가 가시면의 동반 플레어나 CME 없이 발생한다는 Dulk (1985)의 관측을 강화하며, 작은 공간 스케일에서의 전자 가속이 도처에 존재함을 보여준다.

8. **The tied-array technique generalizes to other radio bursts / Tied-array 기법은 다른 전파 폭발로 일반화된다** — The method is not Type III-specific: the same 126-beam honeycomb can image Type II shock-driven bursts, Type IV continuum, moving-sources, and noise storms. It defined the observational template for subsequent LOFAR solar campaigns. / 이 방법은 Type III에만 특화된 것이 아니다: 동일한 126-빔 벌집은 Type II 충격파 구동 폭발, Type IV 연속체, 이동 소스, 잡음 폭풍을 영상화할 수 있다. 이는 후속 LOFAR 태양 캠페인의 관측 템플릿을 정의했다.

9. **Ionospheric effects are negligible at LOFAR beam scales (~arcmin) but will matter at sub-arcmin / 전리층 효과는 LOFAR 빔 스케일(~arcmin)에서는 무시 가능하지만 sub-arcmin에서는 중요해진다** — Stewart & McLean (1982) and Mercier (1996) style corrections give ≲2' at 30 MHz and Sun elevation 30°, well below the 21' beam FWHM. For the next generation of arrays (sub-arcmin resolution at 30 MHz), ionospheric modelling becomes essential. / Stewart & McLean (1982)과 Mercier (1996) 스타일의 보정은 30 MHz, 태양 고도 30°에서 ≲2'를 제공하며, 이는 21' 빔 FWHM보다 훨씬 작다. 차세대 배열(30 MHz에서 sub-arcmin 분해능)의 경우 전리층 모델링이 필수적이 된다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Plasma frequency / 플라스마 진동수

**English.** The fundamental natural oscillation of a cold electron plasma (with immobile ions) has angular frequency ω_p = √(n_e e^2 / (ε_0 m_e)). Converting to Hz and n_e in cm^-3 gives the practical form:

$$
f_p = \frac{1}{2\pi}\sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}} \approx 8980\,\sqrt{N_e}\ \mathrm{Hz},\qquad N_e\ \mathrm{in\ cm^{-3}}.
$$

Observed emission frequency: f = n f_p, n=1 (fundamental) or n=2 (harmonic). Inverting:

$$
N_e = \left(\frac{f}{n\cdot C}\right)^2,\qquad C = 8980\ \mathrm{Hz\ cm^{3/2}}.
$$

**한국어.** 고정된 이온을 가진 차가운 전자 플라스마의 기본 자연 진동은 각진동수 ω_p = √(n_e e^2 / (ε_0 m_e))를 갖는다. Hz 단위와 cm^-3 단위의 n_e로 변환하면 실용적 형태가 된다. 관측 방출 주파수 f = n f_p (n=1 기본파 또는 n=2 고조파). 역산으로 국소 밀도를 유도할 수 있다.

**Numerical examples / 수치 예:**

| n_e (cm^-3) | f_p (MHz) | Altitude (R_sun, Newkirk) | Band |
|---|---|---|---|
| 10^10 | 898 | ~1.02 | not observed (optical depth too high) |
| 10^9 | 284 | ~1.1 | HBA high |
| 10^8 | 89.8 | ~1.3 | HBA low / LBA high |
| 10^7 | 28.4 | ~2.0 | LBA |
| 10^6 | 8.98 | ~3.5 | below LBA, RFI-dominated |
| 10^5 | 2.84 | ~5 | space-based only |
| 10^4 | 0.898 | ~10-20 | STEREO/WIND |

### 4.2 Frequency drift rate / 주파수 드리프트 율

**English.** A Type III electron beam moving radially at speed v through a density profile n_e(r) has:

$$
\frac{df}{dt} = \frac{\partial f}{\partial N_e}\frac{dN_e}{dr}\frac{dr}{dt}
= \frac{n\,C}{2\sqrt{N_e}}\cdot\frac{dN_e}{dr}\cdot v.
$$

Solving for v yields Eq. (2):

$$
v = \frac{2\sqrt{N_e}}{n\,C}\left(\frac{dN_e}{dr}\right)^{-1}\frac{df}{dt}.
$$

(The paper writes v = 2√N_e / C × (dN_e/dr)^-1 × df/dt absorbing n into the harmonic number convention; for n=2 the 2 in the numerator and the n=2 cancel partially.)

**한국어.** 밀도 프로파일 n_e(r)을 속도 v로 방사 이동하는 Type III 전자 빔은 위의 식을 따른다. v에 대해 풀면 Eq. (2)가 된다 (논문은 n을 고조파 수 관행에 흡수시켜 수식을 쓴다; n=2일 때 분자의 2와 n=2가 부분적으로 상쇄된다).

**Numerical example / 수치 예.** At f = 40 MHz (so f_p = 20 MHz for harmonic), n_e = (20×10^6 / 8980)^2 ≈ 4.96×10^6 cm^-3. Using Newkirk's n_e(r) = n_0 × 10^(4.32/r) with n_0 = 4.2×10^4 cm^-3, |dn_e/dr| ≈ n_e × (4.32 ln 10) / r^2. At r ≈ 1.8 R_sun, |dn_e/dr| ≈ 3.5×10^6 cm^-3/R_sun = 3.5×10^6 / (7×10^10 cm) ≈ 5×10^-5 cm^-4. For df/dt = -7 MHz/s, v ≈ (2 × √(5×10^6) / 8980) / (5×10^-5) × (7×10^6) ≈ 3×10^9 cm/s ≈ 0.1 c — consistent with Dulk's classical range. The LOFAR direct measurement gives 2-5× larger values, implying either a steeper local gradient or non-radial geometry not captured by a radial model.

### 4.3 Tied-array beam pattern / Tied-array 빔 패턴

**English.** Given N_stn stations at positions r_n (n=1, …, N_stn) on the ground, the tied-array beam response to a plane wave arriving from direction n̂(θ,φ) is:

$$
W(\theta,\phi) = \sum_{n=1}^{N_\mathrm{stn}} w_n\, e^{i\,\mathbf{k}(\theta,\phi)\cdot\mathbf{r}_n},
$$

where k = (2π/λ) n̂ is the wave vector and w_n are complex weights (amplitude + phase steering). For uniform weights and N_stn = 24 stations with maximum baseline B ~ 2 km at λ = 10 m (30 MHz), the main lobe has angular width θ_FWHM ~ 1.22 λ/B ≈ 6×10^-3 rad ≈ 21' — matching the reported FWHM. At 90 MHz (λ=3.3 m), θ_FWHM ~ 2×10^-3 rad ≈ 7'.

**한국어.** 지상의 위치 r_n (n=1, …, N_stn)에 놓인 N_stn개 스테이션이 방향 n̂(θ,φ)에서 도착하는 평면파에 대해 가지는 tied-array 빔 응답은 위 식과 같다. k = (2π/λ) n̂은 파벡터, w_n은 복소 가중치(진폭 + 위상 조향)이다. λ = 10 m (30 MHz)에서 최대 기선 B ~ 2 km, N_stn = 24인 경우 주 로브의 각폭은 θ_FWHM ~ 1.22 λ/B ≈ 6×10^-3 rad ≈ 21'로 보고된 FWHM과 일치한다. 90 MHz (λ=3.3 m)에서는 θ_FWHM ~ 7'.

**Grating lobe note / 격자 로브 주석.** Because LOFAR core station positions are not on a regular grid, grating lobes are suppressed; this is why a 24-station core can form clean pencil beams over a 3.3° FoV.

### 4.4 Langmuir wave → radio wave conversion / Langmuir 파 → 전파 변환

**English.** The bump-on-tail instability of the electron beam generates electrostatic Langmuir waves at ω_L ≈ ω_p. Three-wave coupling (L + L → T for harmonic emission at 2f_p; L + S → T for fundamental with an ion-acoustic mediator S) produces electromagnetic waves that escape through the corona. Growth rates and frequencies obey:

$$
\omega_p^2 = \frac{n_e e^2}{\varepsilon_0 m_e},\quad \omega_L \approx \omega_p\left(1 + \frac{3 k_L^2 v_{th}^2}{2\omega_p^2}\right)\ \text{(Bohm-Gross)},
$$

with v_th the electron thermal speed. The bump-on-tail distribution condition ∂f_e/∂v > 0 is satisfied in a narrow velocity window around the beam speed v_b.

**한국어.** 전자 빔의 bump-on-tail 불안정성은 ω_L ≈ ω_p에서 정전기적 Langmuir 파를 생성한다. 3파 결합 (2f_p에서의 고조파 방출에 대해 L + L → T; 이온-음향 매개자 S와의 기본파에 대해 L + S → T)이 코로나를 통해 탈출하는 전자기파를 생성한다. 성장률과 주파수는 위 식을 따른다 (Bohm-Gross 분산). v_th는 전자 열 속도이며, bump-on-tail 분포 조건 ∂f_e/∂v > 0은 빔 속도 v_b 주변의 좁은 속도 창에서 만족된다.

### 4.5 Fundamental vs harmonic emission / 기본파 대 고조파 방출

**English.** Free-free (bremsstrahlung) absorption optical depth in a thermal plasma scales as:

$$
\tau_\mathrm{ff} \propto \frac{n_e^2}{T^{3/2} f^2}\,L,
$$

with L the path length. At fundamental frequency f_p the wave propagates near the plasma cutoff — its group velocity ≈ 0 near emission, so optical path is effectively long. Harmonic emission at 2f_p propagates much more freely. Practical rule: in the low corona with n_e ≥ 10^8 cm^-3 (f_p ≥ 90 MHz), fundamental emission is strongly absorbed and we mostly see 2f_p. The authors' assumption of n=2 for 30-60 MHz bursts at ≥2 R_sun is the conservative, standard choice.

**한국어.** 열 플라스마에서의 자유-자유(제동복사) 흡수 광학 깊이는 τ_ff ∝ n_e^2 / (T^(3/2) f^2) × L로 스케일링된다. 여기서 L은 경로 길이이다. 기본파 진동수 f_p에서 파는 플라스마 차단 근처에서 전파되며 — 방출 지점 근처에서의 군속도가 ≈ 0이므로 광학 경로가 사실상 길다. 2f_p에서의 고조파 방출은 훨씬 자유롭게 전파된다. 실용적 규칙: n_e ≥ 10^8 cm^-3 (f_p ≥ 90 MHz)의 저고도 코로나에서 기본파 방출은 강하게 흡수되고, 주로 2f_p가 관측된다. 저자들의 ≥2 R_sun에서 30-60 MHz 폭발에 대한 n=2 가정은 보수적이고 표준적인 선택이다.

### 4.6 Coronal density models used / 사용된 코로나 밀도 모델

**English.** All three models are radial — n_e = n_e(r) — and differ only in their assumed base density and scale height.

- **Newkirk (1961)**: n_e(r) = 4.2×10^4 × 10^(4.32/r) cm^-3, where r is in R_sun from Sun centre. An exponential-in-1/r profile fit to streamer brightness. Gives the highest densities at mid-heights among the three models (hence the largest predicted altitudes in Table 1).
- **Mann et al. (1999)**: derived from hydrostatic equilibrium with constant temperature: n_e(r) = n_0 exp[A/r × (R_sun/R_ref − R_sun/r)], typically with A ~ 13.6 for T ≈ 1.4 MK. Gives the lowest densities and therefore the lowest altitude predictions.
- **Zucca et al. (2014)**: 2-D map combining DEM-derived densities in the low corona (1-1.3 R_sun) with polarized-brightness LASCO/C2 inversions (2.5-5 R_sun) and a plane-parallel + spherical model in between. The only model that is spatially 2-D but still static during the 30 min observation — limited by the 20 min LASCO cadence.

For Burst 2 at ~4 R_sun: Newkirk gives n_e ≈ 1.5×10^6, Mann ≈ 4×10^5, Zucca ≈ 1×10^5 cm^-3 along the Burst 2 trajectory — none meeting the 3.3×10^6 cm^-3 required at 32.5 MHz.

**한국어.** 세 모델은 모두 방사형 — n_e = n_e(r) — 이며, 가정된 기저 밀도와 스케일 높이만 다르다.

- **Newkirk (1961)**: n_e(r) = 4.2×10^4 × 10^(4.32/r) cm^-3, r은 태양 중심으로부터의 R_sun 단위. 스트리머 밝기에 적합된 1/r의 지수 프로파일. 세 모델 중 중간 고도에서 가장 높은 밀도를 주며 (따라서 Table 1에서 가장 큰 예측 고도를 제공한다).
- **Mann et al. (1999)**: 일정 온도 정수압 평형으로부터 유도: n_e(r) = n_0 exp[A/r × (R_sun/R_ref − R_sun/r)], 일반적으로 T ≈ 1.4 MK에 대해 A ~ 13.6. 가장 낮은 밀도를 주어 가장 낮은 고도 예측을 제공한다.
- **Zucca et al. (2014)**: 저고도 코로나(1-1.3 R_sun)의 DEM 유도 밀도와 편광 밝기 LASCO/C2 역변환(2.5-5 R_sun), 그리고 그 사이의 평면-병렬 + 구형 모델을 결합한 2D 지도. 공간적으로 2D인 유일한 모델이지만 30분 관측 동안 여전히 정적 — 20분 LASCO 주기에 제한된다.

~4 R_sun의 Burst 2에 대해: Burst 2 궤적을 따라 Newkirk는 n_e ≈ 1.5×10^6, Mann ≈ 4×10^5, Zucca ≈ 1×10^5 cm^-3를 주며, 32.5 MHz에서 필요한 3.3×10^6 cm^-3를 충족하는 것은 없다.

### 4.7 Type III beam travel to Earth / Type III 전자 빔의 지구 도달

**English.** A beam with v_b = 0.3 c travelling along an interplanetary Parker spiral of length ~1.1 AU reaches Earth in t = 1.1 AU / 0.3c = 1.1×1.5×10^11 m / (0.3×3×10^8 m/s) ≈ 1830 s ≈ 30 min. The corresponding kHz-MHz in-situ emissions can be detected by WIND and STEREO spacecraft minutes to tens of minutes after the LOFAR coronal burst. This chain (LOFAR coronal Type III → Parker-spiral beam → in-situ Langmuir waves and radio emission) is a powerful multi-point probe of the interplanetary magnetic field topology.

**한국어.** v_b = 0.3 c의 빔이 길이 ~1.1 AU의 행성간 Parker 나선을 따라 이동하면 t = 1.1 AU / 0.3c = 1.1×1.5×10^11 m / (0.3×3×10^8 m/s) ≈ 1830 s ≈ 30분에 지구에 도달한다. 해당 kHz-MHz 현장(in-situ) 방출은 LOFAR 코로나 폭발 후 수 분-수십 분 후에 WIND 및 STEREO 우주선으로 탐지될 수 있다. 이 연쇄(LOFAR 코로나 Type III → Parker 나선 빔 → 현장 Langmuir 파 및 전파 방출)는 행성간 자기장 위상의 강력한 다점 탐사 도구이다.

### 4.8 Worked example — drift rate to altitude / 실전 예제 — 드리프트 율에서 고도로

**English.** Consider a Type III burst drifting from 60 to 30 MHz in 5 seconds (df/dt = -6 MHz/s, typical LOFAR observation). Assuming harmonic emission (n=2):
- At f = 60 MHz: n_e = (60×10^6/(2×8980))^2 = (3340)^2 ≈ 1.12×10^7 cm^-3 → altitude (Newkirk) ~1.6 R_sun.
- At f = 30 MHz: n_e = (30×10^6/(2×8980))^2 = (1670)^2 ≈ 2.79×10^6 cm^-3 → altitude (Newkirk) ~2.2 R_sun.
- Radial displacement Δr ≈ 0.6 R_sun ≈ 4.2×10^10 cm in 5 s → v ≈ 8.4×10^9 cm/s ≈ 0.28 c.

This is the standard Type III range. If LOFAR tied-array imaging instead gives Δr ≈ 1.5 R_sun in 5 s, we infer v ≈ 0.7 c or a steeper-than-Newkirk local gradient. The choice between these interpretations is precisely what direct imaging allows us to disambiguate.

**한국어.** Type III 폭발이 60에서 30 MHz로 5초 동안 드리프트한다고 가정하자 (df/dt = -6 MHz/s, 전형적인 LOFAR 관측). 고조파 방출 가정 (n=2):
- f = 60 MHz에서: n_e = (60×10^6/(2×8980))^2 = (3340)^2 ≈ 1.12×10^7 cm^-3 → 고도 (Newkirk) ~1.6 R_sun.
- f = 30 MHz에서: n_e = (30×10^6/(2×8980))^2 = (1670)^2 ≈ 2.79×10^6 cm^-3 → 고도 (Newkirk) ~2.2 R_sun.
- 방사 변위 Δr ≈ 0.6 R_sun ≈ 4.2×10^10 cm / 5 s → v ≈ 8.4×10^9 cm/s ≈ 0.28 c.

이것이 표준 Type III 범위이다. LOFAR tied-array 이미징이 대신 5초 동안 Δr ≈ 1.5 R_sun을 주면 v ≈ 0.7 c 또는 Newkirk보다 가파른 국소 경사로 추론된다. 이 해석들 사이의 선택이 바로 직접 이미징이 가능하게 하는 것이다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1950 --- Wild: Type III discovery (Austr. J. Sci. Res.)
      |     > "rapid frequency drift from high to low"
1961 --- Newkirk: 1D radial coronal density model (ApJ)
      |     > n_e(r) = 4.2×10^4 × 10^(4.32/r) cm^-3
1963 --- Wild: 5-class radio burst taxonomy (I, II, III, IV, V)
1967 --- Wild: Culgoora Radioheliograph 80 MHz imaging
1974 --- Lin: Type III = open-field-line electron beams (Space Sci. Rev.)
1975 --- Melrose: Type I burst theory
1985 --- Dulk: "90% of Type IIIs occur without flares" (ARA&A)
      |     > sets the scene for storm-type activity
1993 --- Robinson, Willes, Cairns: Langmuir wave + three-wave theory (ApJ)
1995 --- Reiner+: Ulysses triangulation, Type III on Parker spirals
1998 --- Bastian, Benz, Gary: solar radio ARA&A review
1999 --- Mann+: hydrostatic coronal density model (A&A)
2002 --- Mann, Klassen: drift-rate statistics at 40-70 MHz
2010 --- Bemporad+: multiple side reconnection during CME (ApJ)
2011 --- Stappers+: LOFAR tied-array beam-forming for pulsars (A&A)
2013 --- van Haarlem+: LOFAR commissioning paper (A&A)
2013 --- Carley+: CME-shock herringbone acceleration (Nature Phys)
2013 --- Saint-Hilaire+: 10,000 NRH Type III statistical study
2014 --- Zucca+: 2D DEM+pB coronal density map (A&A)
2014 --- MOROSAN+ --- LOFAR tied-array Type III imaging [THIS PAPER]
      |     > 126 beams, 83ms/12.5 kHz resolution, CME-deflected bursts
2015+ --- Reid+, Kontar+: Type III fine structure with LOFAR
2017+ --- Mann, Vocks+: broader LOFAR solar campaigns
2020+ --- MWA + LOFAR joint campaigns; sub-arcminute Type III imaging
2025+ --- SKA-Low era: ≤0.5' resolution at 50 MHz anticipated
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Wild (1950), Austr. J. Sci. Res. | First characterization of Type III drift / Type III 드리프트 최초 특성화 | Foundational taxonomy this paper works within / 본 논문이 기반하는 기초 분류 |
| Newkirk (1961), ApJ | 1D radial density model used in Eq. (2) / Eq. (2)에 사용된 1D 방사 밀도 모델 | One of three reference models compared with LOFAR imaging / LOFAR 이미징과 비교한 세 기준 모델 중 하나 |
| Mann, Jansen, MacDowall+ (1999), A&A | Alternative hydrostatic density model / 대안 정수압 밀도 모델 | Second reference model in Table 1 / Table 1의 두 번째 기준 모델 |
| Zucca, Carley, Bloomfield, Gallagher (2014), A&A | 2D DEM+pB density map used for context / 맥락으로 사용된 2D DEM+pB 밀도 지도 | Primary 2D comparison; motivates the non-radial analysis / 주요 2D 비교; 비방사형 분석의 동기 |
| Bastian, Benz, Gary (1998), ARA&A | Review establishing fundamental vs harmonic physics / 기본파 대 고조파 물리를 확립한 리뷰 | Justifies n=2 harmonic assumption / n=2 고조파 가정의 근거 |
| Robinson, Willes, Cairns (1993), ApJ | Langmuir wave theory / Langmuir 파 이론 | Underpins Type III emission mechanism / Type III 방출 기작의 기반 |
| Stappers, Hessels, Alexov+ (2011), A&A | LOFAR tied-array pulsar mode / LOFAR tied-array 펄서 모드 | Original tied-array technique repurposed for solar / 태양용으로 재목적화된 원래의 tied-array 기법 |
| van Haarlem+ (2013), A&A | LOFAR instrument description / LOFAR 기기 기술 | Instrumental reference / 기기적 참고 |
| Bemporad, Soenen+ (2010), ApJ | CME side-reconnection simulation / CME 측면 재결합 모의 | Provides mechanism for non-radial Type IIIs / 비방사형 Type III의 기작 제공 |
| Carley, Long, Byrne+ (2013), Nature Phys | Herringbone acceleration at CME shocks / CME 충격파에서의 herringbone 가속 | Comparison case for particle acceleration / 입자 가속에 대한 비교 사례 |
| Saint-Hilaire, Vilmer, Kerdraon (2013), ApJ | Statistical study of 10,000 Type IIIs at NRH / NRH에서 10,000개 Type III의 통계 연구 | Context for spectral characteristics / 스펙트럼 특성의 맥락 |
| Mann, Klassen (2002), ESA SP | Drift-rate statistics at 40-70 MHz / 40-70 MHz에서의 드리프트 율 통계 | Benchmark for drift rate comparison / 드리프트 율 비교의 기준 |

---

## 7. References / 참고문헌

### Primary paper / 주요 논문

- Morosan, D. E., Gallagher, P. T., Zucca, P., et al., "LOFAR tied-array imaging of Type III solar radio bursts", A&A 568, A67, 2014. DOI: 10.1051/0004-6361/201423936

### Cited references / 인용 참고문헌

- Abranin, E. P., Bazelyan, L. L., & Tsybko, Y. G., Sov. Astron., 34, 74, 1990.
- Bastian, T. S., Benz, A. O., & Gary, D. E., ARA&A, 36, 131, 1998.
- Bemporad, A., Soenen, A., Jacobs, C., Landini, F., & Poedts, S., ApJ, 718, 251, 2010.
- Brueckner, G. E., Howard, R. A., Koomen, M. J., et al., Sol. Phys., 162, 357, 1995.
- Carley, E. P., Long, D. M., Byrne, J. P., et al., Nature Physics, 9, 811, 2013.
- Dulk, G. A., ARA&A, 23, 169, 1985.
- Dulk, G. A., Goldman, M. V., Steinberg, J. L., & Hoang, S., A&A, 173, 366, 1987.
- Dulk, G. A., Leblanc, Y., Bastian, T. S., & Bougeret, J.-L., JGR, 105, 27343, 2000.
- Kerdraon, A., & Delouis, J.-M., in Coronal Physics from Radio and Space Observations, Lecture Notes in Physics 483, 192, 1997.
- Lemen, J. R., Title, A. M., Akin, D. J., et al., Sol. Phys., 275, 17, 2012.
- Li, B., & Cairns, I. H., Sol. Phys., 289, 951, 2013.
- Lin, R. P., Space Sci. Rev., 16, 189, 1974.
- Lin, R. P., Potter, D. W., Gurnett, D. A., & Scarf, F. L., ApJ, 251, 364, 1981.
- Lin, R. P., Levedahl, W. K., Lotko, W., Gurnett, D. A., & Scarf, F. L., ApJ, 308, 954, 1986.
- Mann, G., Jansen, F., MacDowall, R. J., Kaiser, M. L., & Stone, R. G., A&A, 348, 614, 1999.
- Mann, G., & Klassen, A., in Solar Variability: From Core to Outer Frontiers, ESA SP 506, 245, 2002.
- Mann, G., Vocks, C., & Breitling, F., Planetary, Solar and Heliospheric Radio Emissions (PRE VII), 507, 2011.
- McLean, D. J., & Labrum, N. R., Solar Radiophysics: Studies of emission from the Sun at metre wavelengths, Cambridge Univ. Press, 1985.
- Mercier, C., Ann. Geophys., 14, 42, 1996.
- Newkirk, G. Jr., ApJ, 133, 983, 1961.
- Reiner, M. J., Fainberg, J., & Stone, R. G., Science, 270, 461, 1995.
- Reiner, M. J., Goetz, K., Fainberg, J., et al., Sol. Phys., 259, 255, 2009.
- Robinson, P. A., Willes, A. J., & Cairns, I. H., ApJ, 408, 720, 1993.
- Saint-Hilaire, P., Vilmer, N., & Kerdraon, A., ApJ, 762, 60, 2013.
- Stappers, B. W., Hessels, J. W. T., Alexov, A., et al., A&A, 530, A80, 2011.
- Stewart, R. T., & McLean, D. J., Proc. Astron. Soc. Australia, 4, 386, 1982.
- Stone, R. G., Bougeret, J. L., Caldwell, J., et al., A&AS, 92, 291, 1992.
- Thejappa, G., & MacDowall, R. J., ApJ, 720, 1395, 2010.
- Tingay, S. J., Oberoi, D., Cairns, I., et al., J. Phys. Conf. Ser., 440, 012033, 2013.
- van Haarlem, M. P., Wise, M. W., Gunst, A. W., et al., A&A, 556, A2, 2013.
- Wild, J. P., Austr. J. Sci. Res. A, 3, 541, 1950.
- Wild, J. P., Radiotekhnika, 1, 1963.
- Wild, J. P., Sheridan, K. V., & Neylan, A. A., Aust. J. Phys., 12, 369, 1959.
- Zlotnik, E. Y., Klassen, A., Klein, K.-L., Aurass, H., & Mann, G., A&A, 331, 1087, 1998.
- Zucca, P., Carley, E. P., Bloomfield, D. S., & Gallagher, P. T., A&A, 564, A47, 2014.

### Follow-up context / 후속 맥락

- Reid, H. A. S., & Ratcliffe, H., "A review of solar Type III bursts", Research in Astronomy and Astrophysics, 14, 773, 2014.
- Morosan, D. E., et al., "LOFAR observations of fine spectral structure in Type III bursts", A&A, 580, A65, 2015.
- Kontar, E. P., Yu, S., Kuznetsov, A. A., et al., "Imaging spectroscopy of solar radio burst fine structures", Nature Communications, 8, 1515, 2017.
