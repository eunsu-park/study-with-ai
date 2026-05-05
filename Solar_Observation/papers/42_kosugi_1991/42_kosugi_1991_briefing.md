---
title: "Pre-Reading Briefing: The Hard X-ray Telescope (HXT) for the SOLAR-A Mission"
paper_id: "42_kosugi_1991"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Hard X-ray Telescope (HXT) for the SOLAR-A Mission: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kosugi, T., Makishima, K., Murakami, T., Sakao, T., Dotani, T., Inda, M., Kai, K., Masuda, S., Nakajima, H., Ogawara, Y., Sawa, M., and Shibasaki, K., "The Hard X-ray Telescope (HXT) for the SOLAR-A Mission", *Solar Physics* **136**, 17–36, 1991. DOI: 10.1007/BF00151693
**Author(s)**: T. Kosugi *et al.* (12 co-authors, ISAS / Univ. Tokyo / NAOJ)
**Year**: 1991

---

## 1. 핵심 기여 / Core Contribution

이 논문은 일본의 SOLAR-A (발사 후 YOHKOH로 개명) 위성에 탑재된 Hard X-ray Telescope (HXT)의 기기 설계와 영상 합성 원리를 제시한다. HXT는 64개의 독립적인 부시준기 (subcollimator)를 사용해 태양 X-ray 휘도 분포의 64개 푸리에 성분을 동시에 측정하고, MEM (maximum entropy method)과 modified CLEAN으로 이미지를 재구성하는 세계 최초의 푸리에 합성형 (Fourier-synthesis) X-ray 영상 망원경이다. 각도 분해능 ~5″, 시간 분해능 0.5 s, 4개 에너지 대역 (15(19)–24, 24–35, 35–57, 57–100 keV)으로, 30 keV 이상에서 처음으로 태양 플레어의 영상 관측을 가능케 했다.

This paper describes the instrument design and image-synthesis principle of the Hard X-ray Telescope (HXT) aboard Japan's SOLAR-A satellite (renamed YOHKOH after launch). HXT is the world's first Fourier-synthesis X-ray imager: 64 independent subcollimators each measure one generalized complex Fourier component of the solar X-ray brightness distribution, which is reconstructed via MEM (maximum entropy method) or modified CLEAN. With ~5″ angular resolution, 0.5 s temporal resolution, and four energy bands spanning 15(19)–100 keV, HXT enabled the first imaging observations of solar flares above 30 keV — directly probing nonthermal electron acceleration sites.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1980년대 후반은 태양 플레어 연구의 전환기였다. 이전 위성 SMM (HXIS, 1980)과 HINOTORI (1981)는 hard X-ray 영상 관측의 가능성을 보여주었으나, 모두 30 keV 이하 1–2개 에너지 대역에 한정되었고 각도/시간 분해능도 부족했다. 비열적 (nonthermal) 전자 가속 메커니즘과 자기장 위상학과의 관계는 여전히 미스터리였다. SOLAR-A 미션은 ISAS의 두 번째 태양 플레어 전용 위성으로, HXT (hard X-ray imaging) + SXT (soft X-ray imaging) + BCS (Bragg crystal spectrometer) + WBS/GRS (wideband / gamma-ray spectrometer)의 4개 기기로 플레어를 종합 관측하는 야심찬 프로젝트였다.

The late 1980s marked a turning point in solar-flare physics. Prior missions — SMM (HXIS, 1980) and HINOTORI (1981) — had demonstrated the feasibility of hard X-ray imaging but were limited to one or two energy bands below 30 keV with modest angular and temporal resolution. The mechanism of nonthermal electron acceleration and its relation to magnetic topology remained mysterious. SOLAR-A was ISAS's second flare-dedicated satellite, combining HXT (hard X-ray imaging), SXT (soft X-ray imaging), BCS (Bragg crystal spectrometer), and WBS/GRS (wideband/gamma-ray spectrometers) for comprehensive flare diagnostics.

### 타임라인 / Timeline

```
1972 — Frieden: Maximum Entropy Method (MEM) 통계적 영상 재구성 / statistical reconstruction
1974 — Högbom: CLEAN deconvolution algorithm for radio interferometry / 전파 간섭계용 CLEAN
1978 — Makishima et al.: Multi-pitch modulation collimator (MPMC) 제안 / MPMC concept
1980 — SMM HXIS: 최초 hard X-ray 영상 (이미징 시준기) / first hard X-ray imaging (HXIS)
1981 — HINOTORI: 회전 bigrid 변조 시준기, 1D→2D / rotating-bigrid 1D-to-2D imager
1988 — Prince et al.: Fourier-transform telescope with PSDs / 위치민감 검출기 푸리에 망원경
1991 — *HXT (this paper) — 64 SC, 4-band 푸리에 합성 영상기*
1991 (Aug) — YOHKOH 발사 / YOHKOH launch
1994 — Masuda flare: above-the-loop-top hard X-ray source 발견 / discovery
2002 — RHESSI: 회전 변조 시준기로 푸리에 영상 계승 / inherits Fourier imaging via RMC
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Bremsstrahlung (제동복사)**: 고에너지 전자가 이온과 충돌할 때 방출하는 hard X-ray. 비열적 전자의 직접 진단 / direct diagnostic of nonthermal electrons
- **Modulation collimator (변조 시준기)**: 두 개의 평행 격자 사이에 광원 위치에 따른 투과율을 만들어 위치 정보를 주파수 영역에서 부호화 / encodes source position in transmission modulation
- **Fourier transform & visibility (가시도)**: 라디오 간섭계 (van Cittert–Zernike) 와 동일하게 변조 시준기의 측정값은 광원 분포의 푸리에 성분 / each measurement is one Fourier component
- **MEM (maximum entropy method)**: 부족한 (k,θ) 샘플링에서도 양의 비음 (positive) 해를 선호하는 정규화 영상 재구성 / regularised reconstruction
- **CLEAN algorithm**: 점원 (point source)을 반복적으로 차감하며 잔차에 추가하는 deconvolution / iterative point-source subtraction
- **NaI(Tl) scintillator**: 광자가 이온화/여기를 일으키며 가시광 펄스를 만들고 PMT가 증폭 — hard X-ray 검출의 표준
- **(u,v) plane sampling**: 각 부시준기 = (u,v) 한 점, 더 많은 점일수록 좋은 영상 / sampling pattern determines PSF
- **Pulse-height analysis**: PMT 출력 펄스의 진폭이 광자 에너지에 비례 — 4개 에너지 채널 분리 / distinguishes energy channels

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Subcollimator (SC) | 한 쌍의 평행 텅스텐 격자 + 검출기. 하나의 변조 패턴을 만든다 / A pair of parallel tungsten grids + detector forming one modulation pattern |
| Bigrid modulation collimator | 두 격자가 동일 슬릿/피치를 갖는 구조; 광원 각도에 따라 삼각형 투과율 / two-grid collimator with triangular transmission |
| Cosine/Sine pair | 슬릿 위치를 1/4 피치 시프트한 부시준기 쌍, 90° 위상 차로 복소 푸리에 성분 측정 / quarter-pitch shifted pair giving 90° phase, yields complex Fourier component |
| Wave number k | 변조 패턴의 공간 주파수, 기본 주기 (2'06″ = 2.1 arcmin)의 정수배 / spatial frequency in units of fundamental period 2'06″ |
| Fanbeam element | 1D 부채살 모양 변조; 위상 차이 90° 4개 위상으로 1D 위치 측정 (k=1,2) / 1D fan-beam modulator at low k |
| Fourier element | 코사인-사인 쌍, 2D 영상에 필요한 복소 푸리에 성분 (k=3..8) / cosine–sine pair providing complex Fourier component |
| MEM | Maximum Entropy Method — 부족한 측정에서 음수 없이 부드러운 영상 재구성 / sparse-sampling regularised image reconstruction |
| CLEAN | 점원 모델을 반복 차감하여 dirty map을 정리; HXT는 modified version 사용 / iterative deconvolution adapted for HXT |
| Synthesis aperture | (u,v)에서 가장 낮은 k=1 에 대응하는 시야 단위 ~ 2'06″ / FOV unit corresponding to k=1 |
| HXA (HX Aspect system) | 가시광 CCD로 X-ray 광축의 위치를 1–2″ 정확도로 결정 / visible-light CCD aspect system for ~1″ pointing |
| Pulse pile-up | 검출기 시간상수보다 짧은 간격으로 광자가 들어오면 합쳐져 잘못 측정 / overlapping pulses misregistered |
| Pulse-height analysis | 펄스 진폭에서 광자 에너지를 추정해 채널 (L, M1, M2, H) 분리 / energy binning via pulse amplitude |

---

## 5. 수식 미리보기 / Equations Preview

### Eq. 1 — 변조 시준기 투과 함수 / Modulation transmission of one SC
$$F_c(k\rho), \quad \rho = X\cos\theta + Y\sin\theta$$
- $k$: 정수 파수 (wave number, 1..8) / spatial wave number in units of fundamental
- $\theta$: 격자의 자세각 / position angle of grid pattern
- $(X,Y)$: 천구상 좌표 (기본 주기로 정규화) / sky coordinates normalised to fundamental period
- $F_c$: 삼각형 투과 패턴 (DC 무시 시 cos과 유사) / triangular transmission, cosine-like once DC is dropped

### Eq. 2 — 사인 짝 / Sine partner
$$F_s(k\rho) = F_c(k\rho - \pi/2)$$
같은 $(k,\theta)$이지만 슬릿이 1/4 피치 시프트되어 90° 위상 차이.

### Eq. 3, 4 — 광자 카운트가 측정하는 것 / What the photon counts measure
$$b_c(k,\theta) = A\!\!\int B(X,Y)\,F_c(k\rho)\,dX\,dY,\quad b_s(k,\theta) = A\!\!\int B(X,Y)\,F_s(k\rho)\,dX\,dY$$
- $B(X,Y)$: 복원하고 싶은 X-ray 휘도 분포 / X-ray brightness distribution
- $A$: 부시준기 유효 면적 / effective area
- $b_c+ib_s$ → 한 (k,θ) 점에서의 복소 푸리에 성분 / yields one complex Fourier component

### Eq. 5 — 에너지 분해능 / Energy resolution
$$\Delta E/E \sim 1.3\,E^{-1/2}\quad (E\text{ in keV})$$
NaI(Tl)의 전형적인 광자통계-제한 분해능 / photon-statistics-limited resolution of NaI(Tl).

### Eq. 6 — 비틀림 강성 / Tube twist stiffness
$$\theta(\text{arcmin}) = 0.025\,T\,(\text{kg m})$$
CFRP 미터링 튜브의 토크-비틀림 관계, 격자 정렬에서 가장 위험한 변형 모드 / torque-to-twist relation, the most damaging deformation mode.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1 (Introduction)**: SMM/HINOTORI의 한계와 SOLAR-A의 동기. 빠르게 통독 / read briefly for motivation.
- **Section 2 (HXT Overview) + Table I**: HXT를 SMM HXIS, HINOTORI imager와 비교한 표 — 가장 중요한 한 장. 64 SC, 4 채널, 5″, 0.5 s, 70 cm² 숫자를 기억하기.
- **Section 3 (Design and Image Synthesis Principles)**: 핵심. Eq. 1–4, Figure 2 (코사인/사인 짝의 기하), Figure 3 ((u,v) 점 배치 — 16 fanbeam + 48 Fourier element = 64 SC), Figure 4 (MEM 시뮬레이션 결과). 이 섹션을 가장 천천히 읽기 / read slowly.
- **Section 4 (The Instrument)**: 격자 (Table II), 검출기 (NaI(Tl) 5 mm + PMT), 신호 처리 (Figure 8), HXA (시야계 1–2″). 공학적 디테일 — 처음에는 훑고 나중에 참고 / engineering detail; skim first.
- **Section 4.3.1 + Table III**: 4 PC 채널 (L, M1, M2, H) 정의를 기억.
- **Section 5 (Final Remarks)**: 4가지 결정적 캘리브레이션 (그리드 정밀, 64 SC 정렬, 변조 패턴 측정, gain 1%) 요약.

키 그림 / Key figures: **Fig. 2** (Fourier synthesis principle), **Fig. 3** ((u,v) coverage), **Fig. 4** (MEM image-synthesis simulations), **Fig. 5a** (64 SC layout), **Fig. 7** (spectral response).

---

## 7. 현대적 의의 / Modern Significance

HXT는 1991년 발사된 YOHKOH 위성의 핵심 기기로 작동하며 14년 동안 (~2001 재진입) 수천 건의 플레어 영상을 제공했다. 가장 유명한 발견은 **Masuda flare (1994)**: 자기 루프의 꼭대기 위 (above-the-loop-top)에서 비열적 hard X-ray source가 발견되어 고전적 thick-target 모델을 재고하게 한 결정적 증거가 되었다. HXT의 푸리에 합성 원리는 2002년 발사된 **RHESSI** (Lin et al. 2002)의 Rotating Modulation Collimator로 직접 계승되어 9개 격자로 100 keV–17 MeV까지 영상을 확장했다. 또한 변조 시준기 + 푸리에 합성은 STIX (Solar Orbiter, 2020) 까지 이어져 X-ray 영상의 사실상 표준 방법론이 되었다.

HXT served as a flagship instrument on YOHKOH (1991–2001), providing thousands of flare images over 14 years. Its most famous outcome was the **Masuda flare (1994)** — the discovery of an "above-the-loop-top" nonthermal hard X-ray source that forced reconsideration of the classical thick-target model. The Fourier-synthesis principle was directly inherited by **RHESSI** (Lin et al. 2002), which used 9 rotating modulation collimators to extend imaging up to 17 MeV, and lives on in **STIX** aboard Solar Orbiter (2020). Modulation-collimator + Fourier-synthesis has effectively become the standard methodology for hard X-ray imaging of compact astrophysical sources.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
