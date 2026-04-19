---
title: "Speckle interferometry with adaptive optics corrected solar data"
authors: Friedrich Wöger, Oskar von der Lühe, Kevin Reardon
year: 2008
journal: "Astronomy & Astrophysics, 488, 375–381"
doi: "10.1051/0004-6361:200809894"
topic: Solar Observation
tags: [speckle-interferometry, post-processing, KISIP, AO-post-processing, bispectrum, Knox-Thompson, spectral-ratio, photometric-accuracy, parallel-computing]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 21. Speckle Interferometry with Adaptive Optics Corrected Solar Data / AO 보정 태양 데이터의 스펙클 간섭법

---

## 1. Core Contribution / 핵심 기여

이 논문은 **KISIP(Kiepenheuer-Institut Speckle Interferometry Package)** 의 알고리즘, 구현, 성능을 종합적으로 보고한다. KISIP은 von der Lühe(1984, 1993)가 정립한 태양 스펙클 간섭법을 **C 언어로 재작성**하고 **MPI 기반 병렬화**하여, 15 fps로 촬영된 ~100장짜리 speckle burst를 near-real-time으로 재구축할 수 있게 만들었다. 저자들은 두 가지 푸리에 위상 복원 알고리즘(**Knox–Thompson** 과 **Triple Correlation/Bispectrum**)의 수렴 특성과 계산 복잡도를 비교하고, 푸리에 진폭은 von der Lühe의 **spectral ratio 기법**으로 seeing 파라미터 $r_0$를 추정한 뒤 Labeyrie 추정자로 calibrate한다. 핵심 검증으로는 DST(지상, AO+speckle)와 **Hinode/SOT(우주, seeing-free)** 가 **동일 과립(granulation) 영역을 동시 관측**한 데이터를 비교하여, KISIP으로 재구축된 DST 이미지가 공간 해상도와 **광도 정확도(photometric accuracy)** 측면에서 우주 기반 기준에 근접함을 입증한다. 또한 노드 수에 따른 스케일링 실험으로 다중 코어 프로세서 시대에 real-time speckle 재구축이 실현 가능함을 보였다. 이 알고리즘은 GREGOR와 DKIST 파이프라인의 표준이 되었다.

This paper presents the algorithms, implementation, and performance of the **Kiepenheuer-Institut Speckle Interferometry Package (KISIP)**. KISIP rewrites von der Lühe's (1984, 1993) solar speckle framework in **C with MPI parallelism**, enabling near-real-time reconstruction of ~100-frame speckle bursts recorded at 15 fps. The authors compare two Fourier-phase algorithms — **Knox–Thompson (KT)** and **Triple-Correlation / Bispectrum (TC)** — in convergence and cost, and calibrate the Fourier **amplitude** via the **spectral-ratio** technique (von der Lühe 1984) that retrieves $r_0$ from the observed data, feeding the Labeyrie estimator. The key validation compares **simultaneous ground-based DST (AO + speckle)** and **space-based Hinode/SOT (seeing-free)** observations of the same granulation field, showing that the KISIP-reconstructed DST images approach the space-based benchmark in **both spatial resolution and photometric accuracy**. Scaling experiments confirm real-time feasibility on commodity multi-core hardware. KISIP became the de-facto standard pipeline for GREGOR and DKIST.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

핵심 논점: **AO 보정은 항상 부분적**이다. Strehl ratio가 기껏해야 0.3–0.5 수준이라 회절 한계 이미지를 얻으려면 반드시 post-processing이 필요하다. 두 가지 경로가 경쟁한다:

- **Blind deconvolution 계열** — MFBD, MOMFBD (van Noort et al. 2005).
- **Speckle interferometry 계열** — Labeyrie(1970), Knox–Thompson(1974), Weigelt(1977/79), Lohmann et al.(1983), von der Lühe(1984/93).

KISIP은 후자를 상속·확장. 스펙클의 장점: 한 burst = ~100 frames @ 15 fps → 하루 수백 GB → **전송·저장 부담 때문에 관측소에서 near-real-time 축약 필요**. 이 실용적 문제 의식이 KISIP의 동기다.

**Key tension: AO correction is always partial (Strehl 0.3–0.5), so post-processing is mandatory.** Two paths compete: blind deconvolution (MFBD/MOMFBD) and speckle. KISIP extends the speckle tradition. Practical driver: a burst is ~100 frames at 15 fps, generating hundreds of GB/day — the operational need for **near-real-time on-site reduction** motivates the C/MPI rewrite.

---

### Part II: Algorithmic Details (§2) / 알고리즘 상세

#### 2.1 Imaging model / 결상 모델

푸리에 영역에서:

$$
I_k(\vec{q}) = O(\vec{q})\, H_k(\vec{q}) + N_k(\vec{q})
$$

$I_k$: frame $k$ 스펙트럼, $O$: 물체 스펙트럼(추정 대상), $H_k$: 순시 OTF, $N_k$: 노이즈.

Fourier-domain imaging model with unknown object spectrum $O$, random instantaneous OTF $H_k$, and noise $N_k$.

#### 2.2 Amplitude: Labeyrie estimator / 진폭: Labeyrie 추정자

$$
\langle |I_k(\vec{q})|^2 \rangle = |O(\vec{q})|^2 \cdot \langle |H_k(\vec{q})|^2 \rangle + \text{noise}
$$

STF $\langle |H|^2 \rangle$가 알려지면 $|O|$를 복원할 수 있다. KISIP은 STF 모델(Korff 1973)을 seeing 파라미터 $r_0$의 함수로 표현하고, 이 $r_0$를 **spectral-ratio** 로부터 추정.

Knowing the **STF** $\langle |H|^2 \rangle$ — modelled (Korff 1973) as a function of $r_0$ — recovers $|O|$. KISIP estimates $r_0$ from the data via spectral ratio.

#### 2.3 Spectral ratio (von der Lühe 1984) / 분광 비율 방법

장·단노출 파워 스펙트럼의 비율:

$$
\epsilon(\vec{q}) = \frac{|\langle I_k(\vec{q}) \rangle|^2}{\langle |I_k(\vec{q})|^2 \rangle}
$$

$\epsilon$은 $\vec{q}$와 $r_0$만의 함수(물체 의존성 소거)이므로, 관측된 $\epsilon(\vec{q})$ 곡선을 이론 모델에 맞추면 해당 burst의 **effective $r_0$** 를 얻는다. 태양은 확장 광원이라 점광원 calibrator를 못 쓰기 때문에 이 self-calibration 방법이 결정적.

The **spectral ratio** $\epsilon(\vec{q})$ — power-spectrum ratio of long- to short-exposure spectra — depends only on $\vec{q}$ and $r_0$ (object-independent). Fitting the observed curve to the theoretical model yields the burst's effective $r_0$ — a self-calibration essential for the Sun, which has no point-source calibrator.

#### 2.4 Phase: Knox–Thompson / 위상: Knox–Thompson

$$
K(\vec{q}, \Delta\vec{q}) = \langle I(\vec{q}) \cdot I^*(\vec{q}+\Delta\vec{q}) \rangle
$$

작은 shift $\Delta\vec{q}$를 사용한 교차 파워 스펙트럼이 물체 위상차의 정보를 담는다. 위상을 인접 주파수로부터 **적분**하여 복원. 계산량 $\mathcal{O}(N^2)$ per shift, 간단하나 노이즈 누적.

**KT** uses cross-spectra at small shifts; object phase is recovered by integrating phase differences from neighbor frequencies. $\mathcal{O}(N^2)$ per shift, simple but accumulates noise.

#### 2.5 Phase: Triple correlation / 위상: 삼중 상관

$$
B(\vec{q}_1, \vec{q}_2) = \langle I(\vec{q}_1)\, I(\vec{q}_2)\, I^*(\vec{q}_1 + \vec{q}_2) \rangle
$$

Bispectrum $B$는 **phase closure** (항공 VLBI와 같은 원리)를 만족하여 seeing 위상이 소거된다. 물체 bispectrum phase를 추출하고 recursive하게 $O(\vec{q})$의 위상을 복원. 계산량 $\mathcal{O}(N^4)$ (4D array) 그러나 견고함. KISIP은 기본 TC, 옵션으로 KT 제공.

The **bispectrum** enjoys **phase closure** (analogous to aperture-synthesis radio interferometry): seeing phases cancel. Recursive reconstruction yields object phases. $\mathcal{O}(N^4)$ (4D array) but robust. KISIP uses TC by default with KT as option.

#### 2.6 Anisoplanatic tiling / 등각편차 타일링

전체 FOV는 $\theta_0$(~5″)를 넘으므로 **작은 타일(보통 4–8″ × 4–8″)** 로 나누어 각 타일마다 독립적으로 재구축 후 mosaic. 타일 경계에서 apodization으로 블렌딩.

FOV exceeds $\theta_0$ (~5″), so the image is split into small isoplanatic tiles (~4–8″) reconstructed independently and mosaicked with apodized blending.

#### 2.7 Parallelization / 병렬화

MPI로 타일·burst별 자연스러운 데이터 병렬. C로 재작성해 메모리·캐시 최적. 마스터-슬레이브 구조: 마스터는 작업 분배, 슬레이브는 각 타일의 FFT·bispectrum·적분 수행.

Natural data parallelism over tiles/bursts via MPI; a C rewrite optimizes memory and cache. Master–slave: master distributes work, slaves do FFTs, bispectrum assembly, and phase integration per tile.

---

### Part III: Scalability (§3) / 확장성 평가

저자들은 노드 수(core 수) $P$에 따른 실행 시간을 측정. 이상적 speedup $T(1)/T(P) = P$를 기대. 결과:

- **Knox–Thompson**: $\mathcal{O}(N^2)$, 메모리 가벼움 → 거의 선형 scaling, ~16 cores까지 효율적.
- **Triple correlation**: $\mathcal{O}(N^4)$ 이지만 통신량이 더 크고 로드 밸런싱이 필요. 8–16 노드에서 효율 ~70–80%.
- 2008년 기준 16-core 머신에서 burst(100 frames, 500×500 px)를 **수 초** 내에 처리. 15 fps 입력 속도에 맞춰 실시간 가능.

Scaling measurements with core count $P$: KT scales nearly linearly up to ~16 cores (lighter memory). TC is heavier in communication and load balancing; achieves 70–80% efficiency at 8–16 cores. A 100-frame 500×500-px burst reconstructs in **seconds** on 16 cores (2008), matching the 15-fps input rate for real-time use.

---

### Part IV: Photometric Accuracy via DST + Hinode (§4) / DST + Hinode로 광도 정확도 검증

이 논문의 **가장 결정적인 검증**:

1. **DST**(지상, 76-cm) — IBIS 또는 G-band에서 AO(DST AO 시스템) + 스펙클 burst.
2. **Hinode/SOT**(우주, 50-cm) — seeing 없는 회절 한계 이미지.
3. **동시 관측** — 같은 granulation 시야를 수 초 이내에 cross-match.
4. **비교 지표**: (a) 공간 파워 스펙트럼, (b) 과립 대비, (c) 픽셀 대 픽셀 상관.

**결과**:
- KISIP 재구축 후 DST 이미지의 공간 해상도가 Hinode에 거의 필적(~0.15″ 수준).
- **Photometric accuracy**: 과립 대비(RMS contrast) 차이 수 % 수준. 이전의 speckle 재구축 중 일부가 과대 대비를 보인 것과 대조.
- Fourier 진폭의 seeing 보정이 제대로 되면 **정량적 광도**까지 복원 가능함을 증명.

The decisive validation: simultaneous DST (76 cm, ground, AO+speckle) and Hinode/SOT (50 cm, space, seeing-free) observations of the same granulation field. Metrics compared: spatial power spectra, granule contrast, pixel-to-pixel correlation. **Results**: KISIP-reconstructed DST images reach near-Hinode resolution (~0.15″), and the **RMS granule contrast matches Hinode within a few percent** — demonstrating that KISIP recovers not only sharpness but also **quantitative photometry**, unlike some earlier speckle reductions that over-amplified high-frequency power.

---

### Part V: AO-Corrected Speckle Considerations / AO-보정 스펙클 고려사항

AO가 있는 경우 핵심 이슈:

1. **STF 왜곡** — AO가 저차 파면을 제거하여 $\langle |H|^2 \rangle$의 low-frequency 부분을 강화. 미보정한 Korff STF를 적용하면 저주파에서 진폭 과소, 고주파에서 과대.
2. **Spectral ratio는 여전히 유효** — $r_0$ 추정이 effective seeing(AO 이후 잔차)을 자동으로 반영.
3. **부분 보정 체제** — $\langle |H_{\rm AO}|^2 \rangle$에 결정론적(AO 보정분) + 스펙클(잔차) 항이 공존.
4. 실무적으로 KISIP은 AO burst를 그대로 처리하되 STF 모델을 AO 조건에 맞게 재조정.

AO changes the STF — low frequencies are boosted because AO removes low-order aberrations. Applying the uncorrected Korff STF biases the amplitude. The spectral-ratio estimator still works because it recovers effective residual $r_0$ after AO. The corrected STF has coexisting deterministic (AO-corrected) and speckle (residual) terms. KISIP handles AO bursts by adapting the STF model to the effective residual seeing.

---

### Part V-bis: KISIP Pipeline — Step by Step / KISIP 파이프라인 단계별 정리

논문의 기술 전체 흐름을 정리하면 다음 8단계다 / The full KISIP pipeline can be summarized as 8 stages:

1. **Burst acquisition** — 15 fps로 ~80–100 frames 캡처. 노출시간은 $\tau_0$보다 짧게 (수 ms). AO 잠금 상태에서 촬영. / Capture ~80–100 frames at 15 fps with exposure shorter than $\tau_0$ (few ms), AO locked.
2. **Pre-processing** — dark, flat, bad-pixel, destretching (low-order image-motion 제거). / Detector calibration plus destretching to remove low-order image motion.
3. **Anisoplanatic tiling** — FOV를 isoplanatic patch(~4–8″) 타일로 분할, apodization window 적용. / Partition FOV into isoplanatic tiles with apodized windows.
4. **Fourier transform** — 타일마다 FFT. 병렬화 포인트. / FFT per tile — a key parallelization stage.
5. **Spectral-ratio fit** — $\epsilon(\vec{q})$ 계산 후 Korff 모델로 effective $r_0$ 추정. / Fit $\epsilon(\vec{q})$ to the Korff model to extract effective $r_0$.
6. **Amplitude reconstruction** — Labeyrie estimator + 추정된 STF. / Recover $|O(\vec{q})|$ via Labeyrie with the fitted STF.
7. **Phase reconstruction** — Bispectrum 또는 Knox–Thompson로 $\arg O(\vec{q})$ 복원. / Recover $\arg O(\vec{q})$ via bispectrum (default) or KT.
8. **Inverse FFT + mosaicking** — 각 타일 복원 이미지를 블렌딩하여 최종 FOV 조립. / Inverse FFT and mosaic tiles with smooth blending.

KT와 TC 중 선택은 **계산 예산 + SNR 요구**에 따라 결정. Preview에는 KT, science 최종본은 TC.

Selection between KT and TC depends on **compute budget and SNR requirements**: KT for preview, TC for science-grade final product.

### Part V-ter: Comparison with MOMFBD / MOMFBD와의 비교

| 측면 / Aspect | Speckle (KISIP) | MOMFBD |
|---|---|---|
| 통계 모델 / Statistical model | 고전 난류 통계 + Korff STF | Zernike-mode 파라미터적 PSF |
| 위상 복원 / Phase | Bispectrum closure (model-free) | 최적화로 PSF와 object 동시 추정 |
| Multi-wavelength | 파장별 독립 처리 | 공동(joint) 복원 지원 |
| 계산 비용 / Cost | O(N² to N⁴), 병렬 잘됨 | 반복 최적화, 무겁다 |
| Frame 수 / Frames | 80–100 필요 | 더 적게 가능 (10–20) |
| 견고성 / Robustness | 높음 (통계 평균화) | 중간 (local minima 위험) |
| 속도 / Speed | **near-real-time** | slow, 배치 처리 |

현대 관측소는 두 방법을 모두 유지하여 상황에 맞게 선택. / Modern observatories keep both and select per situation.

### Part VI: Discussion & Conclusions (§5) / 토의·결론

- KISIP은 near-real-time 재구축을 실용화하여 **데이터 볼륨을 100배 압축** — 관측소 운용 혁신.
- **Bispectrum이 기본**, Knox–Thompson은 빠른 preview용.
- **MOMFBD와 상보** — MOMFBD는 multi-wavelength 공동 복원이 강점, 스펙클은 위상 통계 견고성과 속도가 강점.
- 제한: burst당 frame 수가 적으면 statistics 부족(최소 ~80–100 frames 권장). 장노출에서는 과립 생멸(~5 min)로 시간 해상도 희생.
- 향후: MCAO 시대의 wide-field speckle, adaptive tiling, GPU 포팅.

KISIP enables near-real-time reduction, compressing data by ~100× and transforming observatory operations. Bispectrum is the default; KT is a fast preview mode. It complements MOMFBD (which excels at multi-wavelength joint reconstruction). Limits: bursts need ≥80–100 frames for robust statistics, and 5-minute granule lifetimes cap integration time. Future directions: wide-field speckle in the MCAO era, adaptive tiling, GPU implementation.

---

## 3. Key Takeaways / 핵심 시사점

1. **AO is necessary but insufficient / AO만으로는 부족** — 부분 보정(Strehl 0.3–0.5)만 제공하므로 스펙클이나 MOMFBD 후처리가 회절 한계 이미지의 표준 경로다. / Partial AO correction (Strehl 0.3–0.5) always requires post-processing (speckle or MOMFBD) to reach the diffraction limit.

2. **Spectral ratio enables self-calibration / 분광 비율이 자기 보정을 가능케 함** — 태양은 점광원 calibrator가 없어 Labeyrie 추정자를 바로 쓸 수 없는데, von der Lühe의 $\epsilon(\vec{q}) = |\langle I \rangle|^2 / \langle |I|^2 \rangle$ 가 burst 자체로부터 $r_0$를 복원한다. 이것이 태양 스펙클 방법의 결정적 돌파구다. / With no point-source calibrator available, von der Lühe's spectral-ratio $\epsilon(\vec{q})$ extracts $r_0$ directly from the burst — the enabling breakthrough for solar speckle.

3. **Bispectrum phase closure beats Knox–Thompson / 삼중 상관이 Knox-Thompson보다 견고** — KT는 계산은 저렴하나 위상 적분 시 노이즈 누적. TC는 phase closure로 seeing 위상을 해석적으로 소거하여 더 견고. KISIP은 TC가 기본. / TC achieves analytical cancellation of seeing phases via phase closure; KT accumulates noise in phase integration. KISIP uses TC by default.

4. **Anisoplanatism forces tiled reconstruction / 등각편차가 타일링을 강제** — $\theta_0 \sim 5″$ 제약으로 FOV를 작은 타일(4–8″)로 분할 후 독립 재구축·모자이크. MCAO 보편화까지의 과도기적 해법. / Isoplanatic-patch size forces tiled reconstruction with apodized blending — a pragmatic solution in the era before widespread MCAO.

5. **Parallelism + C rewrite = real-time / 병렬화 + C 재작성 = 실시간** — 멀티코어 서버에서 15 fps burst를 수 초 내 재구축 가능. 관측소 데이터 볼륨을 100배 축소하여 운용을 근본적으로 바꿈. / On a multi-core server KISIP keeps pace with 15 fps bursts, compressing data ~100× and transforming observatory operations.

6. **Photometric accuracy validated against Hinode / Hinode로 광도 정확도 검증** — 동시 관측 비교에서 KISIP 출력의 과립 RMS 대비가 Hinode와 수 % 내 일치. 스펙클 재구축이 **샤프니스 뿐 아니라 정량적 광도**를 보존함이 입증됨. / DST + Hinode simultaneous comparison shows granule RMS contrast agrees to within a few percent — speckle reconstruction preserves not just sharpness but also quantitative photometry.

7. **AO modifies STF, but spectral-ratio still works / AO는 STF를 바꾸지만 분광 비율은 여전히 작동** — AO가 파면 통계를 변형해 Korff의 고전 STF는 편향되지만, spectral-ratio가 effective residual $r_0$를 자동 추출해 STF 모델을 AO 조건에 맞춘다. / AO biases the classical Korff STF, but the spectral-ratio estimator self-adjusts by recovering the effective residual $r_0$ after AO.

8. **Complementary to MOMFBD / MOMFBD와 상보적** — 스펙클은 위상 통계 견고성·속도, MOMFBD는 multi-wavelength 공동 복원. 현대 파이프라인(CRISP, IBIS)은 둘 다 지원. KISIP은 GREGOR·DKIST 표준. / Speckle favors statistical robustness and speed; MOMFBD favors joint multi-wavelength reconstruction. Modern pipelines (CRISP, IBIS) support both; KISIP is the standard at GREGOR and DKIST.

9. **Burst statistics set the floor / Burst 통계가 성능 하한을 결정** — 프레임 수 $N$이 부족하면 $|O|$와 $\arg O$ 모두 SNR이 급감. 권장 $N \gtrsim 80$. 그러나 과립 수명(~5분)과 채층 진화 시간 스케일이 $N$의 상한을 제한 → 관측 설계의 핵심 trade-off. / Burst size $N \gtrsim 80$ is needed for SNR; meanwhile the 5-minute granule lifetime and chromospheric evolution cap $N$ from above — a central observational trade-off.

10. **Near-real-time changes how observatories operate / 준-실시간 처리가 관측소 운용을 바꿈** — burst → 단일 복원 이미지 ~100× 데이터 압축. 저장·전송 비용 절감, 즉각적 품질 피드백으로 관측 효율 향상. DKIST에 필수적. / Near-real-time reduction compresses data by ~100×, reduces storage/transfer costs, and provides instant quality feedback — now essential at DKIST scale.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Imaging equation

$$
I_k(\vec{q}) = O(\vec{q})\, H_k(\vec{q}) + N_k(\vec{q})
$$

### 4.2 Labeyrie amplitude estimator

$$
\boxed{|\hat{O}(\vec{q})|^2 = \frac{\langle |I_k(\vec{q})|^2 \rangle - \langle |N_k(\vec{q})|^2 \rangle}{\langle |H_k(\vec{q})|^2 \rangle_{\rm STF}}}
$$

### 4.3 Spectral ratio for $r_0$ estimation

$$
\epsilon(\vec{q}) = \frac{|\langle I_k(\vec{q}) \rangle|^2}{\langle |I_k(\vec{q})|^2 \rangle} = \frac{|T(\vec{q})|^2}{\langle |H(\vec{q})|^2 \rangle} = E(\vec{q}, r_0)
$$

$T(\vec{q})$: telescope diffraction OTF. $E(\vec{q}, r_0)$: Korff 모델 함수. 피팅으로 $r_0$ 추정.

### 4.4 Knox–Thompson cross spectrum

$$
K(\vec{q}, \Delta\vec{q}) = \langle I_k(\vec{q})\, I_k^*(\vec{q}+\Delta\vec{q}) \rangle
$$

Object phase recovery:

$$
\arg K(\vec{q}, \Delta\vec{q}) \approx \arg O(\vec{q}) - \arg O(\vec{q}+\Delta\vec{q})
$$

Integration along a path in $\vec{q}$-plane recovers $\arg O$.

### 4.5 Bispectrum / Triple correlation

$$
\boxed{B(\vec{q}_1, \vec{q}_2) = \langle I_k(\vec{q}_1)\, I_k(\vec{q}_2)\, I_k^*(\vec{q}_1 + \vec{q}_2) \rangle}
$$

Phase closure:

$$
\arg B = \arg O(\vec{q}_1) + \arg O(\vec{q}_2) - \arg O(\vec{q}_1 + \vec{q}_2)
$$

Recursive phase reconstruction:

$$
\arg O(\vec{q}_1 + \vec{q}_2) = \arg O(\vec{q}_1) + \arg O(\vec{q}_2) - \arg B(\vec{q}_1, \vec{q}_2)
$$

Weighted average over many $(\vec{q}_1, \vec{q}_2)$ pairs improves SNR.

### 4.6 Korff STF model (schematic)

$$
\langle |H(\vec{q})|^2 \rangle = |T(\vec{q})|^2 \cdot e^{-D_\phi(\lambda f \vec{q})} + |T(\vec{q})|^2 \cdot A(\vec{q}, r_0)
$$

- First term: coherent (long-exposure) part;
- Second term: speckle (incoherent) component.
- $D_\phi$: wavefront structure function (Fried: $D_\phi(r) = 6.88 (r/r_0)^{5/3}$).

### 4.7 Worked example / 수치 예제

500 nm, $r_0 = 10$ cm, $D = 76$ cm:
- Long-exposure cutoff ≈ $r_0/\lambda = 0.2 \times 10^6$ rad$^{-1}$ → ~0.4″ seeing-limited res.
- Short-exposure (speckle) cutoff ≈ $D/\lambda = 1.52 \times 10^6$ → 0.15″ diffraction limit.
- Stop uncertainty improvement: speckle amplitude SNR improves as $\sqrt{N}$; $N=100$ frames → 10× SNR gain vs single frame.
- Bispectrum phase SNR ≈ $\sqrt{N}\, c^2$ with granule contrast $c$; $c=0.08$, $N=100$ → $\sim$0.06 (rough order-of-magnitude).

### 4.8 Complexity per burst / 번들당 계산 복잡도

| Step | Complexity | 비고 / Note |
|---|---|---|
| FFT per frame | $\mathcal{O}(N_{\rm pix}^2 \log N_{\rm pix})$ | $N$ frames × tiles → 병렬 가능 |
| Spectral-ratio | $\mathcal{O}(N_{\rm pix}^2)$ | 1D 피팅 후 table lookup |
| Labeyrie amplitude | $\mathcal{O}(N_{\rm pix}^2)$ | 픽셀 단위 나눗셈 |
| Knox–Thompson | $\mathcal{O}(N_{\rm pix}^2 \cdot N_{\rm shift})$ | $N_{\rm shift}$ = 이웃 수 |
| Bispectrum | $\mathcal{O}(N_{\rm pix}^4)$ | 메모리 $\mathcal{O}(N_{\rm pix}^4)$, 주된 병목 |

Bispectrum의 4D 배열이 병목 → KISIP은 symmetry 이용하여 유효 공간을 줄이고, redundant pair를 평균화하여 SNR 동시 향상.

The **bispectrum's 4D array** is the main bottleneck. KISIP exploits symmetries to reduce the effective space and averages redundant pairs to simultaneously raise SNR.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1970    Labeyrie — stellar speckle interferometry (amplitude)
1973    Korff — speckle transfer function formalism
1974    Knox & Thompson — phase reconstruction via cross spectra
1977    Weigelt — triple correlation concept
1983    Lohmann, Weigelt, Wirnitzer — bispectrum image reconstruction
1984    von der Lühe — spectral ratio for solar speckle
1993    von der Lühe — complete solar speckle framework
1999    Rimmele — real-time solar AO
2005    van Noort et al. — MOMFBD (alternative)
2006    Mikurda & von der Lühe — precursor KISIP implementation
2008    ★ Wöger, von der Lühe, Reardon — THIS PAPER (KISIP, AO+speckle, photometric accuracy)
2011    Rimmele & Marino — SAO review
2012    GREGOR first light — KISIP in pipeline
2020    DKIST first light — KISIP as standard post-processing
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Labeyrie (1970) "Attainment of Diffraction-Limited Resolution..." | Foundational speckle concept / 스펙클 원리 | **Direct ancestor** — KISIP amplitude estimator is the Labeyrie estimator. / KISIP 진폭 추정자의 기원. |
| Knox & Thompson (1974) "Recovery of images from atmospherically degraded..." | Phase via cross spectra / 교차 스펙트럼 위상 | **Implemented as KT option** in KISIP. / KISIP의 KT 모드. |
| Weigelt (1977); Lohmann et al. (1983) "Speckle masking" | Bispectrum / 삼중 상관 | **KISIP default phase algorithm**. / KISIP 기본 위상 알고리즘. |
| von der Lühe (1984) "Estimating Fried's parameter..." | Spectral ratio method / 분광 비율 | **Self-calibration backbone** — solves the "no calibrator" problem for extended sources. / 확장 광원 calibrator 문제의 해결. |
| von der Lühe (1993) "Speckle imaging of solar small scale structure" | Full solar speckle pipeline / 전체 태양 스펙클 파이프라인 | **Direct progenitor** that this paper C-rewrites and parallelizes. / 본 논문이 C로 재작성·병렬화한 원본. |
| #20 Rimmele & Marino (2011) SAO review | AO context / AO 맥락 | **Complementary** — Rimmele describes AO, Wöger describes what to do with AO-corrected frames. / Rimmele는 AO 제공, Wöger는 후처리 제공. |
| van Noort et al. (2005) MOMFBD | Competing post-processing / 경쟁 후처리 | **Complementary alternative** — speckle vs blind deconvolution. / 스펙클 vs 블라인드 디컨볼루션. |
| Korff (1973) "Analysis of a method for obtaining diffraction-limited..." | STF formalism / STF 수식 | **STF model used inside KISIP**. / KISIP이 내부에서 사용하는 STF 모델. |
| Scharmer et al. (2002) "Dark cores in sunspot penumbral filaments" | SST science application / SST 과학 응용 | **Canonical result** enabled by AO + speckle/MOMFBD. / AO + 후처리가 가능케 한 대표 발견. |

---

## 7. References / 참고문헌

- Wöger, F., von der Lühe, O., & Reardon, K., "Speckle interferometry with adaptive optics corrected solar data", *A&A*, 488, 375 (2008). [DOI: 10.1051/0004-6361:200809894]
- Labeyrie, A., "Attainment of Diffraction-Limited Resolution in Large Telescopes by Fourier Analysis of Speckle Patterns in Star Images", *A&A*, 6, 85 (1970).
- Korff, D., "Analysis of a method for obtaining near-diffraction-limited information in the presence of atmospheric turbulence", *JOSA*, 63, 971 (1973).
- Knox, K. T. & Thompson, B. J., "Recovery of images from atmospherically degraded short-exposure photographs", *ApJ*, 193, L45 (1974).
- Weigelt, G. P., "Modified astronomical speckle interferometry 'speckle masking'", *Opt. Commun.*, 21, 55 (1977).
- Lohmann, A. W., Weigelt, G., & Wirnitzer, B., "Speckle masking in astronomy: triple correlation theory and applications", *Appl. Opt.*, 22, 4028 (1983).
- von der Lühe, O., "Estimating Fried's parameter from a time series of an arbitrary resolved object imaged through atmospheric turbulence", *JOSA A*, 1, 510 (1984).
- von der Lühe, O., "Speckle imaging of solar small scale structure. I.", *A&A*, 268, 374 (1993).
- Mikurda, K. & von der Lühe, O., "High-resolution speckle masking reconstruction with the Kiepenheuer-Institut speckle interferometry package", *Solar Phys.*, 235, 31 (2006).
- van Noort, M., Rouppe van der Voort, L., & Löfdahl, M. G., "Solar Image Restoration by use of MOMFBD", *Solar Phys.*, 228, 191 (2005).
- Rimmele, T. R. & Marino, J., "Solar Adaptive Optics", *Living Rev. Solar Phys.*, 8, 2 (2011).
