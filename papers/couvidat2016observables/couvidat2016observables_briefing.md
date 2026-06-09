---
paper_id: 31
topic: Solar_Observation
date: 2026-04-23
type: briefing
title: "Observables Processing for HMI on SDO"
authors: "S. Couvidat, J. Schou, J.T. Hoeksema, R.S. Bogart, R.I. Bush, T.L. Duvall Jr., Y. Liu, A.A. Norton, P.H. Scherrer"
year: 2016
doi: "10.1007/s11207-016-0957-3"
journal: "Solar Physics"
---

# Pre-Reading Briefing / 사전 학습 브리핑

## 1. Paper Identification / 논문 개요

**한국어:**
이 논문은 SDO(Solar Dynamics Observatory) 위성에 탑재된 HMI(Helioseismic and Magnetic Imager) 장비의 Level-1 필터그램(filtergram)으로부터 Level-1.5 관측량(observables) — 도플러 속도, 시선방향 자기장, 연속복사 강도, 선폭, 선깊이, Stokes [I, Q, U, V] — 을 산출하는 두 개의 독립적인 자료처리 파이프라인을 상세하게 기술한다. Couvidat 등은 MDI-like 알고리즘(Michelson Doppler Imager 계승), 편광 보정, 파장 캘리브레이션, CCD 비선형성 보정, 왜곡 보정, 롤각 계산 등 HMI의 5년간(2010년 5월 ~ 2015년 4월) 운영 경험을 총망라하여 전 과정의 설계와 교정 전략을 문서화한다.

**English:**
This paper documents the two independent data-processing pipelines that convert HMI Level-1 filtergrams taken by SDO into Level-1.5 observables: Doppler velocity, line-of-sight (LoS) magnetic field, continuum intensity, line width, line depth, and the Stokes polarization parameters [I, Q, U, V]. Couvidat et al. provide a comprehensive account of the MDI-like algorithm (inherited from the SOHO/MDI instrument), polarization calibration, wavelength calibration, CCD nonlinearity correction, distortion correction, and roll-angle measurement, summarizing five years (May 2010 - April 2015) of HMI prime-mission operational experience.

## 2. Context & Motivation / 맥락과 동기

**한국어:**
HMI는 태양 광구의 자기장과 속도장을 연속적으로 관측해 우주날씨, 일조진동학(helioseismology), 자기 활동의 장기 변동을 연구하는 SDO의 핵심 장비이다. HMI는 6,173 Å(Fe I)의 좁은 대역 필터그램을 2대의 4096 x 4096 CCD 카메라로 45초 간격으로 수집하며, 프라임 미션 동안 약 8400만 장의 필터그램을 기록하였다(설계 목표의 99.86 %). 이 방대한 데이터 스트림으로부터 과학적으로 활용 가능한 관측량을 일관된 품질로 산출하기 위해서는 복잡한 교정 및 변환 절차가 필요하며, 본 논문은 그 방법론과 남아 있는 계통 오차를 정량화한다.

**English:**
HMI is SDO's core instrument for continuous photospheric observations of the magnetic field and velocity field, enabling space-weather forecasting, helioseismology, and long-term studies of solar magnetic activity. HMI acquires narrow-band filtergrams around the Fe I 6173 Å line every 45 seconds using two 4096 x 4096 CCD cameras; during the prime mission, 84 million filtergrams were recorded (99.86 % of the design goal). Converting this data stream into scientifically usable observables with uniform quality requires an elaborate calibration and transformation pipeline. This paper documents that methodology and quantifies its remaining systematic errors.

## 3. Prerequisites / 선수 지식

**한국어:**
- **Fourier 해석의 기초**: MDI-like 알고리즘은 주기 T = 412.8 mÅ의 이산 Fourier 계수(1차, 2차) 추정에 기반한다.
- **Zeeman 효과**: 6173 Å Fe I 흡수선의 Landé 인자 gL = 2.5를 이용해 시선방향 자기장을 LCP/RCP 도플러 속도 차이로부터 환산한다.
- **Stokes 벡터 편광 측정**: I, Q, U, V 네 개의 편광 상태와 변조/복조 행렬 개념.
- **Milne-Eddington 대기 모형 및 VFISV 인버전**: 완전한 벡터 자기장은 Stokes 프로파일을 ME 대기에서 풀어 얻는다(Borrero+ 2011).
- **Lyot/Michelson 필터 시스템**: HMI의 투과 프로파일은 두 개의 Michelson 간섭계와 하나의 조정형 Lyot 요소로 구성된다.
- **SOHO/MDI 배경**: HMI의 LoS 파이프라인은 MDI 알고리즘을 직접 승계하지만 동적 영역(dynamic range)이 다르다.

**English:**
- **Fourier analysis basics**: the MDI-like algorithm estimates the first- and second-order Fourier coefficients of the spectral line profile over a period T = 412.8 mÅ.
- **Zeeman effect**: the Fe I 6173 Å line has Landé factor gL = 2.5; the LoS magnetic flux density is derived from the LCP-RCP Doppler-velocity difference.
- **Stokes vector polarimetry**: the four Stokes parameters I, Q, U, V and the modulation/demodulation matrix formalism.
- **Milne-Eddington atmosphere and VFISV inversion**: the full vector magnetic field is obtained by fitting the Stokes profiles with a Milne-Eddington model (Borrero+ 2011).
- **Lyot/Michelson filter system**: HMI's transmission is defined by two Michelson interferometers and a tunable Lyot element.
- **SOHO/MDI heritage**: HMI's LoS pipeline directly inherits the MDI algorithm, but operates with a different dynamic range.

## 4. Key Vocabulary / 주요 용어

| Term | 한국어 | Definition / 정의 |
|------|--------|-------------------|
| **Filtergram** | 필터그램 | 특정 파장/편광으로 촬영한 단일 CCD 이미지 / a single CCD image at a particular wavelength and polarization |
| **Observable** | 관측량 | 여러 필터그램에서 유도한 물리량(도플러, 자기장 등) / a physical quantity derived from filtergrams |
| **MDI-like algorithm** | MDI 유사 알고리즘 | 6파장 샘플에서 첫/두번째 Fourier 계수를 추정해 속도 등을 계산 / estimates Fourier coefficients from 6-wavelength samples |
| **Framelist** | 프레임 리스트 | CCD가 반복하는 파장/편광 시퀀스 / the repeating sequence of exposures |
| **Stokes [I,Q,U,V]** | 스토크스 벡터 | 전자기파 편광 상태를 기술하는 4 성분 / 4 components describing polarization state |
| **VFISV** | VFISV | Very Fast Inversion of the Stokes Vector, ME 인버전 코드 / Milne-Eddington inversion code |
| **LoS / Vector pipeline** | 시선/벡터 파이프라인 | 45초 LoS 또는 720초 벡터 관측량 파이프라인 / 45-s LoS or 720-s vector observables pipeline |
| **Level-1 / Level-1.5** | 레벨-1 / 레벨-1.5 | 교정 필터그램 / 관측량 / calibrated filtergram / derived observables |
| **DRMS / JSOC** | DRMS / JSOC | Data Record Management System / Joint Science Operations Center |
| **Phase map** | 위상 맵 | 조정 필터 요소의 픽셀별 위상 비균일성 지도 / pixel-wise phase non-uniformity map |
| **I-ripple** | 강도 잔물결 | 조정 필터 결함으로 발생하는 강도 변화 / intensity variation from tunable-filter imperfections |
| **Look-up table (LUT)** | 룩업 테이블 | 원시 Fourier 속도를 실제 속도로 보정하는 사전 시뮬레이션 테이블 / table mapping raw to true velocity |

## 5. Paper Structure / 논문 구조

**한국어:**
- §1 서론: HMI 프라임 미션 요약 및 동기
- §2 관측량 산출: Level-1 생성(§2.1), 두 파이프라인(§2.2), 필터그램 선택/정렬/왜곡(§2.3~§2.6), 편광(§2.7), 필터(§2.8), MDI-like 알고리즘(§2.9), 다항식 속도 보정(§2.10)
- §3 성능과 오차: 롤에 대한 민감도(§3.1), 24시간 변동(§3.2), LoS 알고리즘 오차(§3.3), Stokes 인버전 오차(§3.4), 온도 의존성(§3.5), PSF 보정(§3.6)
- §4 요약 및 결론 + 부록(DRMS/JSOC 데이터 관리)

**English:**
- §1 Introduction: HMI prime-mission overview and motivation
- §2 Observables computation: Level-1 production (§2.1), two pipelines (§2.2), filtergram selection/alignment/distortion (§2.3-§2.6), polarization (§2.7), filters (§2.8), MDI-like algorithm (§2.9), polynomial velocity correction (§2.10)
- §3 Performance and errors: roll sensitivity (§3.1), 24-hour variations (§3.2), LoS algorithm errors (§3.3), Stokes inversion errors (§3.4), temperature dependence (§3.5), PSF correction (§3.6)
- §4 Summary and conclusion + Appendix on DRMS/JSOC data management

## 6. Q&A / 질의응답

_Questions and answers to be filled in during reading._
_읽는 동안 질문과 답을 채워 넣는다._

### Q1.

### A1.

### Q2.

### A2.

## 7. Reading Strategy / 읽기 전략

**한국어:**
1. §2.9(MDI-like 알고리즘)를 핵심으로 삼아 수식(4)~(14)을 단계별로 따라가며 Fourier 계수 → 도플러 속도 → 자기장 유도 과정을 이해한다.
2. §2.7(편광)과 §2.8(필터)에서 기기 교정의 복잡성(편광 PSF, 위상 맵, I-ripple)을 파악한다.
3. §3(오차)에서 24시간 변동과 강한 자기장 영역에서의 포화가 왜 발생하는지 이해한다.
4. 보조 논문으로 Scherrer+ 2012(HMI 개관), Schou+ 2012a(설계/지상 교정), Borrero+ 2011(VFISV)을 참조한다.

**English:**
1. Treat §2.9 (MDI-like algorithm) as the centerpiece and follow Equations (4)-(14) step by step to understand Fourier coefficients -> Doppler velocity -> magnetic field.
2. In §2.7 (polarization) and §2.8 (filters), grasp the complexity of instrumental calibration (polarization PSF, phase maps, I-ripple).
3. In §3 (errors), understand why the 24-hour variations and saturation in strong-field regions occur.
4. Consult supporting papers: Scherrer+ 2012 (HMI overview), Schou+ 2012a (design and ground calibration), Borrero+ 2011 (VFISV).
