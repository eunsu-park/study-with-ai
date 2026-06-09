---
paper_id: 29
topic: Solar_Observation
date: 2026-04-23
type: briefing
title: "The Nobeyama Radioheliograph"
authors: "Nakajima, H., Nishio, M., Enome, S., Shibasaki, K., Takano, T., Hanaoka, Y., Torii, C., Sekiguchi, H., Bushimata, T., Kawashima, S., Shinohara, N., Irimajiri, Y., Koshiishi, H., Kosugi, T., Shiomi, Y., Sawa, M., Kai, K."
year: 1994
venue: "Proceedings of the IEEE, Vol. 82, No. 5, pp. 705-713"
doi: "10.1109/5.284737"
---

# Pre-reading Briefing / 사전 브리핑

## 1. One-paragraph Summary / 한 문단 요약

**English.** This 1994 paper describes the design, construction, and first-light performance of the Nobeyama Radioheliograph (NoRH), a dedicated solar radio interferometer built at the Nobeyama Radio Observatory, Japan. Operating at 17 GHz with 84 parabolic dish antennas (80 cm diameter) arranged in a T-shaped array of 490 m E-W and 220 m N-S extent, NoRH achieves 10 arcsecond spatial resolution, 1 second temporal resolution (50 ms for flares), a 40 arcminute field of view that covers the whole Sun, and dynamic range exceeding 20 dB for snapshots and 30 dB for rotational synthesis. The instrument's key innovations include phase-stable optical-fiber signal transmission, custom CMOS gate-array LSI 1-bit complex correlators, and a modified CLEAN algorithm tailored for the solar disk and extended components. Routine 8-hour daily observations began in late June 1992, enabling statistical studies of solar flares, filament disappearances, and CME precursors via gyrosynchrotron and thermal bremsstrahlung radio imaging.

**한국어.** 이 1994년 논문은 일본 Nobeyama 전파 관측소에 건설된 태양 전용 전파 간섭계인 Nobeyama Radioheliograph (NoRH)의 설계, 건설, 초기 성능을 기술한다. 17 GHz에서 동작하며 80 cm 직경의 84개 포물선 안테나를 동서 490 m × 남북 220 m의 T자형 배열로 구성하여, 10 arcsecond의 공간 분해능, 1 s 시간 분해능 (플레어는 50 ms), 태양 전체를 덮는 40 arcminute 시야, 그리고 스냅샷 20 dB / 회전 합성 30 dB의 동적 범위를 달성한다. 핵심 기술 혁신은 위상 안정 광섬유 신호 전송, 맞춤형 CMOS 게이트 어레이 LSI 1-bit 복소 상관기, 태양 원반과 확장 성분에 맞춰진 수정된 CLEAN 알고리즘이다. 1992년 6월 말부터 하루 8시간 정기 관측을 시작하여, 자이로싱크로트론 및 열제동복사 전파 영상화를 통한 태양 플레어, 필라멘트 소실, CME 전조현상의 통계적 연구를 가능하게 하였다.

## 2. Why This Paper Matters / 이 논문이 중요한 이유

**English.** NoRH was the first radio instrument purposefully designed to produce full-disk solar snapshot images at high cadence. Before NoRH, solar radio imaging relied on sharing time on general-purpose instruments (VLA, Westerbork) that prioritized narrow field-of-view high-sensitivity observations of compact sources. By dedicating an entire interferometer to the Sun and optimizing the array for snapshot imaging (not rotational synthesis), NoRH enabled statistical surveys of flare radio emission that were previously impossible. Its 1-second cadence maps captured electron acceleration at impulsive flare onset, non-thermal gyrosynchrotron bursts, quiescent filament structures, and the coronal magnetic field through polarization. Together with the Yohkoh hard X-ray telescope (launched 1991), NoRH formed half of the flagship observational duo for flare physics in the 1990s-2000s.

**한국어.** NoRH는 전태양 스냅샷 영상을 높은 관측 주기로 생성하기 위해 의도적으로 설계된 최초의 전파 관측기기이다. NoRH 이전에는 태양 전파 영상화가 범용 관측기기(VLA, Westerbork)의 시간을 공유하는 방식에 의존하였는데, 이들은 좁은 시야의 고감도 콤팩트 소스 관측을 우선시했다. 전체 간섭계를 태양에 전념시키고 회전 합성이 아닌 스냅샷 영상화를 위해 배열을 최적화함으로써, NoRH는 이전에는 불가능했던 플레어 전파 방출의 통계 조사를 가능하게 하였다. 1초 관측 주기 영상은 임펄시브 플레어 개시에서의 전자 가속, 비열 자이로싱크로트론 폭발, 정온 필라멘트 구조, 편광을 통한 코로나 자기장을 포착하였다. 1991년 발사된 Yohkoh 경 X선 망원경과 함께, NoRH는 1990~2000년대 플레어 물리학의 주력 관측 쌍을 형성하였다.

## 3. Prerequisites / 선수 지식

### 3.1 Mathematics / 수학
- **Fourier analysis / 푸리에 해석**: 2D Fourier transform, convolution theorem, sampling theorem. The van Cittert-Zernike theorem relates interferometric visibility to the sky brightness Fourier transform. / 2D 푸리에 변환, 콘볼루션 정리, 샘플링 정리. 반 시테르트-제르니케 정리는 간섭 가시도를 천구 밝기의 푸리에 변환과 관련시킨다.
- **Complex analysis / 복소 해석**: Complex correlation $V = \langle E_1 E_2^* \rangle$, phase and amplitude representation. / 복소 상관 $V = \langle E_1 E_2^* \rangle$, 위상-진폭 표현.
- **Linear algebra / 선형대수**: Matrix inversion for calibration, least-squares redundancy solutions. / 보정을 위한 행렬 역변환, 중복 기저선 최소 제곱해.

### 3.2 Physics / 물리
- **Plasma radio emission mechanisms / 플라즈마 전파 방출 기전**: thermal bremsstrahlung (free-free), gyrosynchrotron, plasma emission. / 열제동복사, 자이로싱크로트론, 플라즈마 방출.
- **Solar flare physics / 태양 플레어 물리**: magnetic reconnection, electron acceleration, chromospheric evaporation. / 자기 재결합, 전자 가속, 채층 증발.
- **Optical depth and brightness temperature / 광학 깊이와 밝기 온도**: $T_B$ at 17 GHz: quiet Sun ~10⁴ K, active regions ~10⁵ K, flares 10⁶-10⁹ K. / 17 GHz 밝기 온도: 정온태양 ~10⁴ K, 활동영역 ~10⁵ K, 플레어 10⁶-10⁹ K.

### 3.3 Radio astronomy / 전파천문학
- **Interferometer basics / 간섭계 기초**: baseline, u-v plane, visibility, synthesized beam (dirty beam), PSF. / 기저선, u-v 평면, 가시도, 합성 빔 (dirty beam), PSF.
- **Array design / 배열 설계**: aperture synthesis, redundant baselines, snapshot vs. rotational synthesis. / 개구 합성, 중복 기저선, 스냅샷 vs. 회전 합성.
- **Calibration / 보정**: CLEAN deconvolution (Högbom 1974), closure phases, self-calibration. / CLEAN 디컨볼루션, 폐쇄 위상, 자가보정.
- **Prerequisite papers / 선행 논문**: Paper #26 (Bastian et al. 1998, Radio Emission from Solar Flares review) gives scientific context; Högbom (1974) introduces CLEAN; Jennison (1958) defines closure phases. / 논문 #26 (Bastian et al. 1998 태양 플레어 전파 방출 리뷰)가 과학적 맥락을 제공; Högbom (1974)은 CLEAN 도입; Jennison (1958)은 폐쇄 위상을 정의.

## 4. Key Vocabulary / 핵심 용어

| Term (EN) | Term (KO) | Meaning |
|-----------|-----------|---------|
| Radioheliograph | 전파태양관측기 | Radio interferometer dedicated to imaging the Sun / 태양 영상화 전용 전파 간섭계 |
| T-array | T자형 배열 | Antenna layout with perpendicular E-W and N-S arms meeting at origin / 원점에서 만나는 수직한 E-W 및 N-S 암으로 구성된 안테나 배치 |
| Baseline | 기저선 | Vector distance between two antennas; determines one u-v sample / 두 안테나 간 벡터 거리; 하나의 u-v 샘플 결정 |
| Visibility | 가시도 | Complex cross-correlation of signals from antenna pair / 안테나 쌍 신호의 복소 교차상관 |
| u-v coverage | u-v 덮개 | Set of baseline vectors projected on plane perpendicular to source direction / 광원 방향에 수직한 평면에 투영된 기저선 벡터 집합 |
| Dirty beam / PSF | 더티 빔 / 점확산함수 | Inverse FT of u-v sampling function; image of a point source / u-v 샘플링 함수의 역 FT; 점광원의 영상 |
| CLEAN | 클린 | Deconvolution algorithm (Högbom 1974) removing sidelobes from dirty map / 더티 맵에서 사이드로브를 제거하는 디컨볼루션 알고리즘 |
| Closure phase | 폐쇄 위상 | Sum of visibility phases around antenna triangle; antenna-independent / 안테나 삼각형 주위 가시도 위상의 합; 안테나 독립적 |
| Gyrosynchrotron | 자이로싱크로트론 | Radio emission from mildly relativistic electrons spiraling in magnetic field / 자기장에서 나선 운동하는 약한 상대론적 전자의 전파 방출 |
| Optical fiber | 광섬유 | Phase-stable single-mode transmission of LO and IF signals / LO 및 IF 신호의 위상 안정 단일모드 전송 |
| HEMT | HEMT | High-Electron-Mobility Transistor low-noise amplifier / 고전자이동도 트랜지스터 저잡음 증폭기 |
| Walsh function | 월쉬 함수 | Set of binary orthogonal functions used for 180° phase switching / 180° 위상 전환에 쓰이는 이진 직교 함수 집합 |
| Van Vleck correction | 반 블렉 보정 | Nonlinearity correction for 1-bit correlator quantization / 1-bit 상관기 양자화에 대한 비선형성 보정 |
| Redundant baseline | 중복 기저선 | Multiple antenna pairs with identical baseline; enables self-calibration / 동일한 기저선을 갖는 다중 안테나 쌍; 자가보정 가능 |
| sfu | sfu (solar flux unit) | $1\,\text{sfu} = 10^{-22}\,\text{W}\,\text{m}^{-2}\,\text{Hz}^{-1}$ |

## 5. Central Questions to Answer / 핵심 질문

**English.**
1. Why does NoRH use a T-shaped array rather than a VLA-like Y-shape, cross, or circular array?
2. What sets the fundamental spacing $d = 1.528\,\text{m}$ and the maximum baseline 488.96 m?
3. How does the 1-bit correlator work, and what is the $2/\pi$ sensitivity penalty?
4. Why are phase-stable optical fibers and a phase-locked local oscillator essential?
5. How does the modified CLEAN algorithm handle the dominant solar disk plus extended active regions?
6. What kinds of solar physics does 17 GHz imaging uniquely enable?
7. How is a 20-30 dB dynamic range achieved given the huge brightness contrast between the disk and flare cores?

**한국어.**
1. NoRH는 왜 VLA와 같은 Y자형, 십자형, 원형이 아닌 T자형 배열을 채택했는가?
2. 기본 간격 $d = 1.528\,\text{m}$와 최대 기저선 488.96 m는 무엇이 결정하는가?
3. 1-bit 상관기는 어떻게 동작하며, $2/\pi$ 감도 손실은 무엇인가?
4. 왜 위상 안정 광섬유와 위상 고정 국부 발진기가 필수적인가?
5. 수정된 CLEAN 알고리즘은 지배적 태양 원반 + 확장 활동영역을 어떻게 처리하는가?
6. 17 GHz 영상화는 어떤 종류의 태양 물리를 고유하게 가능하게 하는가?
7. 원반과 플레어 코어 사이의 큰 밝기 대비에도 20-30 dB 동적 범위를 어떻게 달성하는가?

## 6. Reading Strategy / 읽기 전략

**English.** This is an instrument paper that mixes engineering details (optical fibers, CMOS LSI, temperature control) with interferometer theory and scientific results. First pass: read Sections I, II (array), V (performance), and skim the scientific discussion. Second pass: dive into Section III receiver system and Section IV calibration/CLEAN, with special attention to Figs. 3 (block diagram), 5 (correlator chip), 6 (dirty vs. clean image), and 7 (flare time evolution). Pay attention to Table 1 numbers (resolution, cadence, dynamic range, sensitivity) because they set the scientific niche. Keep a running list of engineering innovations (optical fiber, 1-bit correlator, HEMT, Walsh modulation, Peltier-stabilized PLL) and relate them back to phase stability requirements.

**한국어.** 본 논문은 공학적 세부사항(광섬유, CMOS LSI, 온도 제어)과 간섭계 이론 및 과학 결과를 혼합한 관측기기 논문이다. 1차 통독: 서론 I장, 배열 II장, 성능 V장을 읽고 과학 논의는 훑어본다. 2차 정독: 수신기 III장과 보정/CLEAN IV장을 파고들며, 특히 Fig. 3 (블록도), Fig. 5 (상관기 칩), Fig. 6 (dirty vs. clean), Fig. 7 (플레어 시간 진화)을 주목한다. Table 1의 수치 (분해능, 관측주기, 동적 범위, 감도)에 주의하라 — 이들이 과학적 활용 범위를 결정한다. 공학적 혁신 항목(광섬유, 1-bit 상관기, HEMT, Walsh 변조, Peltier 안정화 PLL)의 목록을 유지하면서 위상 안정성 요건으로 되돌아 연결시켜라.

## 7. Q&A / 질의응답

*Questions and answers will be populated during and after reading.*
*질문과 답변은 읽기 중 및 읽기 후에 채워진다.*

### Q1. [Placeholder]
**A1.** [To be filled]

### Q2. [Placeholder]
**A2.** [To be filled]

---

## References / 참고문헌

- Nakajima, H. et al., "The Nobeyama Radioheliograph", Proc. IEEE, 82(5), 705-713, 1994. DOI: 10.1109/5.284737
- Bastian, T. S., Benz, A. O., Gary, D. E., "Radio Emission from Solar Flares", ARA&A, 36, 131-188, 1998. (Paper #26 in this study)
- Högbom, J. A., "Aperture synthesis with a non-regular distribution of interferometer baselines", A&AS, 15, 417, 1974.
- Jennison, R. C., "A phase sensitive interferometer technique...", MNRAS, 118, 276, 1958.
- Thompson, A. R., Moran, J. M., Swenson, G. W., "Interferometry and Synthesis in Radio Astronomy", Wiley, 1986.
