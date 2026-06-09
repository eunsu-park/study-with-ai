---
title: "Cryo-CARE: Content-Aware Image Restoration for Cryo-Transmission Electron Microscopy Data"
authors: Tim-Oliver Buchholz, Mareike Jordan, Gaia Pigino, Florian Jug
year: 2019
journal: "Proc. IEEE International Symposium on Biomedical Imaging (ISBI)"
doi: "10.1109/ISBI.2019.8759519"
topic: Low-SNR Imaging / Self-Supervised Deep Denoising
tags: [cryo-care, cryo-em, cryo-tem, electron-tomography, noise2noise, content-aware-restoration, care, even-odd-splitting, dose-fractionation, p2p, t2t, u-net, buchholz-jug]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 15. Cryo-CARE: Content-Aware Image Restoration for Cryo-TEM Data / 극저온 투과전자현미경 데이터를 위한 컨텐츠-기반 영상 복원

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **Lehtinen et al. (2018)의 Noise2Noise 학습 패러다임**을 **저온 투과전자현미경(cryo-TEM)** 영상 복원에 *처음으로 성공적으로 적용*했다. Cryo-TEM은 생체 분자를 native 상태에서 *원자에 가까운* 분해능으로 관찰할 수 있지만, 시료의 빔 손상(beam damage)을 막기 위해 *극저용량* 전자선만 쓸 수밖에 없으므로 SNR이 매우 낮다. CARE(Weigert et al., 2017)와 같은 supervised 딥러닝 디노이저는 *고용량 ground-truth*를 요구하지만 cryo-TEM에선 그것 자체가 *물리적으로 불가능*하다.

저자들은 cryo-TEM 데이터의 **이중 노이즈 실현(two independent noisy realisations)**을 cryo-EM 작업 흐름에서 *공짜로* 얻을 수 있음을 관찰했다:

1. **P2P-tap (projection-to-projection, tilt-angle pairs)**: tilt 시리즈에서 인접 두 tilt-angle은 거의 같은 시료를 (약간 다른 각도에서) 본 두 noisy projection.
2. **P2P-ip (image pairs)**: 절반 dose의 두 독립 노출을 직접 획득.
3. **P2P-df (dose-fractionated movies)**: 직접 검출기(K2 등)의 dose-fractionation 모드에서 모든 짝수 프레임과 홀수 프레임을 따로 합치면 *동일 시료의 두 독립 노이즈 영상*이 만들어진다.
4. **T2T-eoa / T2T-df (tomogram-to-tomogram)**: tilt-angle 절반(짝/홀수 또는 even/odd dose-fractionation)에서 *별도의* 3D tomogram을 ETOMO·IMOD로 재구성한 뒤, 이 두 *3D 볼륨* 자체를 N2N의 입력/타겟으로 사용 → 3D U-Net 학습.

이렇게 얻은 페어로 U-Net (depth 2, kernel 3, MSE loss)을 학습한 결과 raw projection·tomogram의 **대비가 극적으로 향상**되어 비전문가 시각 검토와 자동 분할(*Chlamydomonas reinhardtii* outer dynein arm detection)의 정밀도와 재현율이 모두 *유의미하게* 개선된다 (Fig. 5). Fourier shell correlation (FSC)으로 측정한 정량적 분해능도 raw·NAD baseline 대비 거의 모든 spatial frequency에서 우월하다 (Fig. 2). T2T 방식은 P2P-df로 복원된 tilt-angle을 다시 재구성할 때 발생하는 *missing-wedge artefact 증폭* 문제도 회피한다 (Fig. 4).

핵심적 기여는 (i) cryo-TEM에서 N2N을 위한 **데이터 페어 생성 프로토콜의 종합적 카탈로그** (P2P-tap/ip/df, T2T-eoa/df), (ii) 3D U-Net을 직접 tomogram에 적용하는 *T2T 회귀*, (iii) *automated downstream analysis 가속*이라는 *실질적* 영향 시연이다. 이 논문은 이후 cryo-EM 분야에서 *디폴트 데이터 전처리*로 자리잡게 되며, structural biology 워크플로우에 deep denoising을 도입한 결정적 다리이다.

### English
The paper is the first successful application of **Noise2Noise** (Lehtinen et al., 2018) to **cryo-transmission electron microscopy (cryo-TEM)**. Cryo-TEM is rate-limited by beam damage and therefore acquires data at extremely low electron dose, producing very low-SNR images. Supervised deep denoisers such as CARE (Weigert et al., 2017) require high-SNR ground truth, which is *physically unobtainable* for cryo-TEM. Cryo-CARE shows that the N2N principle (zero-mean noisy targets preserve the conditional mean) maps naturally onto cryo-TEM workflows, where independent noisy realisations of *the same specimen* are essentially free:

1. **P2P-tap** uses adjacent tilt angles from a tilt series.
2. **P2P-ip** acquires two half-dose images.
3. **P2P-df** splits even/odd frames of a dose-fractionated movie (Gatan K2 direct detector + MotionCor2 alignment).
4. **T2T-eoa** reconstructs two independent tomograms from even-/odd-numbered tilt angles via ETOMO/IMOD, then trains a *3D* U-Net on the 3D volumes (Tomo2Tomo, T2T).
5. **T2T-df** reconstructs even/odd dose-fractionation versions of every tilt angle, then reconstructs two tomograms.

Trained U-Nets (depth 2, 3×3 kernel, MSE loss, 1000 random 128×128 patches) recover dramatic contrast and resolution improvements: the Fourier shell correlation curve (Fig. 2) of cryo-CARE T2T-df dominates the raw and the NAD (non-linear anisotropic diffusion, Frangakis-Hegerl 2001) baseline across nearly all spatial frequencies. The T2T scheme avoids missing-wedge artefacts that arise from P2P-restored tilt angles re-reconstructed independently (Fig. 4). Crucially, the authors quantify a downstream impact: precision and recall of automated outer-dynein-arm (ODA) segmentation on *Chlamydomonas reinhardtii* axonemes both improve substantially after cryo-CARE preprocessing (Fig. 5). The work sketches the now-standard recipe of (i) cataloguing N2N-compatible pair-generation protocols for a given imaging modality, (ii) supplying both 2D and 3D U-Nets, (iii) demonstrating that denoising accelerates *downstream* analysis — not just visual quality.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / 서론

#### 한국어
- Cryo-TEM은 생체 분자를 native 상태에서 거의 원자 분해능으로 관찰할 수 있는 핵심 기술 (Knapek-Dubochet 1980 — 빔 손상). 그러나 *전자 투여량 제한* → 매우 noisy.
- 미세조직학자(microscopist)들은 보통 defocus를 늘려 contrast를 높이지만 *분해능 손실*. 좋은 디노이저는 low-dose, low-defocus 모드를 가능하게 함.
- Fluorescence microscopy 용 CARE (Weigert+, 2017)는 깨끗한 *고노출* 페어를 요구 → cryo-TEM에는 *원리적으로* 적용 불가 (시료가 빔 손상으로 변함).
- N2N (Lehtinen+ 2018, ref [3])이 등장 → *깨끗한 타겟 필요 없음*, 두 독립 noisy realisation만 있으면 됨.
- **본 논문의 기여**: cryo-TEM 작업 흐름에서 N2N 학습용 페어를 어떻게 만들 것인가, 어떻게 *2D projection*과 *3D tomogram* 둘 다에 적용할 것인가, 그리고 *자동 분석*에 미치는 영향까지 보임.

#### English
Cryo-TEM acquires at low dose to avoid beam damage, hence very low SNR. CARE-style supervised denoisers need clean reference, which is impossible here. N2N's relaxation makes cryo-TEM denoising feasible. The paper systematically maps cryo-EM acquisition modes onto N2N's required two independent noisy realisations.

---

### Part II: §2 Approach and Methods / 접근과 방법

#### 한국어 — Network and training protocol
- U-Net (Ronneberger+ 2015), depth 2, 3×3 conv, 마지막 layer linear activation, **per-pixel MSE loss**.
- 학습 페어 10%는 validation으로 보유.
- Open-source CARE framework 위에 구현.

#### 한국어 — §2.1 P2P (single projection) variants
**(a) P2P-tap (tomographic adjacent-pair).**
- 보존된 tilt-series만 있을 때 (한 tilt-angle당 1장만 측정) 적용. IMOD로 align/register 후 *인접 두 tilt-angle* projection을 input/target으로 사용.
- 1000 random patches of 128×128 → train. 두 tilt-angle 각각에 적용 후 결과는 두 영상 모두에 사용 가능.
- *문제*: 인접 tilt-angle은 시료를 약간 다른 각도에서 봄 → *blur* 발생 (Fig. 1c).

**(b) P2P-ip (image pairs).**
- 같은 위치를 두 번 (각각 절반 dose) 측정 → 두 독립 noisy 영상. 더 깨끗한 페어.

**(c) P2P-df (dose-fractionation movies).**
- 직접 검출기 K2의 movie mode에서 짝/홀수 프레임을 따로 합쳐 align (MotionCor2) → 두 독립 noisy projection. *총 dose는 그대로 유지*하면서 induced beam damage가 *두 영상에 공유*된다는 추가 장점.

#### 한국어 — §2.2 T2T (tomogram-to-tomogram) variants
**(a) T2T-eoa (even-odd acquisitions).**
- 모든 tilt-angle을 *번호 짝수/홀수*로 분할 → *데이터 독립* 두 tomogram 재구성 (ETOMO/IMOD).
- 두 3D 볼륨에서 1200 random sub-volume $64\times64\times64$ → 3D U-Net 학습.
- 두 복원 tomogram을 각각 통과시킨 뒤 per-voxel 평균 → 최종.

**(b) T2T-df (dose-fractionation tomograms).**
- 각 tilt-angle을 dose-fractionation으로 짝/홀수 분할 후 align → 두 *동일 각도* projection 페어 → 두 tomogram 재구성. *각 시점 sampling이 더 dense* 하므로 T2T-eoa보다 결과가 좋음 (Fig. 2 FSC).

#### 한국어 — §2.3 Automated Downstream Analysis
- *Chlamydomonas reinhardtii* axoneme의 outer dynein arm (ODA) 검출 워크플로우.
- U-Net (학습은 PEET-refined manually annotated 1 tomogram)으로 dense segmentation → Otsu threshold → connected component → size filtering. ODA의 hand-annotated 학습 383 / 테스트 712.
- 데이터 augmentation 없음 — denoising이 그 역할.

#### English
A U-Net of depth 2, kernel 3 with linear final activation, trained with per-pixel MSE, is applied at three levels:
- **Single projections (P2P)** with three pair-construction variants: adjacent tilt angles (tap), explicit half-dose pairs (ip), and even/odd dose-fractionation frames (df).
- **3D tomograms (T2T)** with two variants: even/odd tilt angles (eoa) or even/odd dose-fractionation (df). T2T-df is the highest quality because it preserves the angular sampling.
- **Downstream automated analysis**: a separate U-Net for ODA detection on *C. reinhardtii* axonemes.

---

### Part III: §3 Results / 결과

#### 한국어 — §3.1 P2P Restoration (Fig. 1)
- P2P-tap: tilt-angle 페어 → restored projection이 *blurry* (구조가 인접 각도에서 다른 위치에 있기 때문). Fig. 1c.
- P2P-ip / P2P-df: 같은 시료 dual exposure / dose-fractionation 분할 → 훨씬 sharp (Fig. 1d, e).
- *그러나* P2P-df로 복원된 tilt-angle을 *다시* tomogram으로 재구성하면 missing-wedge artefact 증폭 (Fig. 4) — 신경망이 비선형이라 tilt-angle별 amplitude consistency가 깨짐.

#### 한국어 — §3.2 T2T Restoration (Fig. 2, Fig. 3)
- T2T-eoa: 절반 tilt-angle만 사용한 두 tomogram이 각자 reduced angular sampling. 그래도 raw 대비 분해능·대비 향상.
- T2T-df: 모든 tilt-angle에서 dose-fractionation 분할 → 각 tomogram이 full angular sampling. **Fig. 2 FSC**: T2T-df > T2T-eoa > NAD baseline > raw 거의 모든 분해능 대역에서.
- EMPIAR-10110 (publicly available, dose-fractionated 데이터 없음)에서도 P2P-tap·T2T-eoa 적용 가능 (Fig. 3).
- P2P 결과를 직접 tomogram 재구성에 사용시 missing-wedge가 두드러짐 → T2T 권장 (Fig. 4).

#### 한국어 — §3.3 Automated Analysis (Fig. 5)
- Raw vs cryo-CARE T2T-df 복원 데이터 → 동일 ODA 검출 파이프라인 적용.
- 결과 (precision-recall curve, Fig. 5 below): cryo-CARE 사용 시 *모든* segment-size threshold에서 PR이 위쪽으로 이동. False positives (orange voxels) 줄어들고 true positives (turquoise) 증가.
- *결론*: 디노이저는 시각 품질뿐 아니라 *학습 데이터·후처리에서 인간의 라벨링 부담을 줄인다*.

#### English
P2P-tap blurs because adjacent tilt angles see slightly displaced projections. P2P-df is sharper but cannot be safely re-tomogrammed (Fig. 4). T2T-df, training a 3D U-Net on two tomograms reconstructed from even/odd dose-fractionation halves, gives the best Fourier shell correlation across nearly all bands (Fig. 2). On real automated *C. reinhardtii* ODA segmentation (Fig. 5), cryo-CARE-restored data dominates raw across precision-recall, demonstrating real-world workflow benefit.

---

### Part IV: §4 Discussion / 논의

#### 한국어
- Cryo-CARE는 EM expert가 manual investigation 전에 사용할 수 있는 *간단·강력한* contour-aware tomographic restoration tool.
- T2T가 P2P보다 일반적으로 권장됨 (re-tomography 시 artefact 회피).
- *데이터를 microscope 자체가 만든다* — 사람의 라벨링 불필요. 다른 end-to-end 파이프라인 (segmentation 등)이 풍부한 라벨링 데이터를 필요로 하는 것과 차별화.
- Cryo-EM 워크플로우 전반에 빠르게 adoption될 것으로 예상 (검증된 대로 됨).

#### English
Cryo-CARE acts as an *unsupervised preprocessing stage* upstream of expert investigation and automated pipelines. T2T is generally preferred over P2P for tomographic data because it preserves missing-wedge geometry. The "training data is generated by the microscope itself" framing is the practical kernel: no human annotations required.

---

### Part V: Datasets used / 사용된 데이터셋

#### 한국어
1. **TOMO110 (저자들이 직접 획득)**: 300 kV Thermo Fisher cryo-TEM Titan Halo + Gatan K2 직접 검출기. *Chlamydomonas reinhardtii* cilia 시료를 dose-fractionation movie 모드로 측정. P2P-df / T2T-df 변형 모두 가능.
2. **EMPIAR-10110 (공개 데이터셋)**: EMPIAR (Iudin+ 2016) 데이터베이스에서 공개된 tilt 시리즈. dose-fractionation 데이터 없음 → P2P-tap, T2T-eoa만 적용.
3. **Patch sampling**: 1000 random 128×128 patches per training run (P2P), 1200 random 64×64×64 sub-volumes (T2T).
4. **Validation**: 추출된 학습 패치의 10%를 validation으로 보유 (overfitting 모니터링).
5. **Binning**: P2P 결과는 unbinned data 위에 계산, T2T는 6× binned 데이터 위에 계산 (3D 메모리 절약).

#### English
- TOMO110 (authors' own): 300 kV Titan Halo + K2; *C. reinhardtii* cilia in dose-fractionation. Used for P2P-df / T2T-df.
- EMPIAR-10110 (public): no dose-fractionation; only P2P-tap and T2T-eoa applicable.
- 1000 random 128×128 patches per P2P run; 1200 64×64×64 sub-volumes per T2T run.
- 10% validation hold-out; T2T computed on 6× binned data.

---

### Part VI: Practical recipe (workflow) / 실용 레시피 (워크플로우)

#### 한국어
실제 cryo-EM 시설에서 cryo-CARE 적용 단계:

```
1. tilt-series acquisition (with K2 dose-fractionation, if available)
2. MotionCor2 alignment per tilt-angle
3. Even/odd frame splitting → two half-projections per tilt-angle
4. IMOD/ETOMO tomogram reconstruction × 2 (one from each half) [T2T-df]
   OR direct P2P training on per-tilt-angle pairs [P2P-df]
5. Random sub-volume / patch extraction (1000–1200 patches)
6. U-Net training (depth 2, kernel 3, MSE, ~hours on single GPU)
7. Inference: apply to both tomograms (T2T) or both projections (P2P), per-pixel/voxel average
8. Downstream: visual inspection, automated segmentation, particle picking
```

각 단계는 기존 cryo-EM 파이프라인에 쉽게 끼워넣을 수 있도록 설계 — 그 점이 이 논문의 *실용적* 천재성.

#### English
The seven-step workflow above is designed to slot into any cryo-EM lab's existing pipeline (Tomo110 lab tested as proof of principle). Step 7 (per-voxel averaging of two restored tomograms) reduces remaining N2N variance by ~$\sqrt 2$.

---

## 3. Key Takeaways / 핵심 시사점

1. **N2N transfers cleanly to cryo-TEM / N2N은 cryo-TEM에 깔끔히 이식된다** — N2N의 핵심 가정 (independent noisy pairs of the same latent) cryo-TEM 데이터 획득 모드에서 *자연스럽게* 만족된다 (split detector frames, even/odd tilt angles, half-dose pairs). 알고리즘 변경은 사실상 zero.

2. **Pair-construction matters / 페어 구성이 결정적** — Adjacent tilt-angle 페어는 blur (다른 시점), half-dose 페어는 더 sharp, dose-fractionation 페어는 *induced beam damage가 두 영상에 공유*되어 가장 권장. 페어가 *얼마나 같은 latent를 보고 있느냐* 가 PSNR·해상도를 좌우.

3. **3D T2T avoids re-tomography artefacts / 3D T2T는 재-토모그래피 아티팩트를 회피** — P2P로 tilt-angle 개별 디노이즈 후 tomogram 재구성하면 비선형 네트워크가 amplitude consistency를 깨고 missing-wedge artefact를 증폭. T2T로 *3D 볼륨 자체*를 N2N의 단위로 사용하면 회피 (Fig. 4).

4. **Quantitative resolution gain confirmed by FSC / 정량적 분해능 향상이 FSC로 확인** — T2T-df의 Fourier shell correlation은 raw·NAD baseline 대비 거의 전 대역에서 우월 (Fig. 2). 단순 cosmetic 향상이 아니라 *진짜 정보 회복*임을 입증.

5. **Downstream analysis benefits / 하위 분석 파이프라인이 함께 좋아진다** — *C. reinhardtii* ODA 자동 분할의 precision-recall curve가 모든 size threshold에서 위로 이동 (Fig. 5). 디노이저의 *진짜 효용*은 PSNR이 아니라 후속 작업 정확도다.

6. **Decouples preprocessing from analysis / 전처리와 분석을 분리** — End-to-end pipeline은 거대 라벨 데이터를 요구하지만 cryo-CARE는 *microscope가 라벨 없이 자동 학습 데이터를 생성*. 이 분리는 cryo-EM 외에도 의료영상·천문학 등 ground-truth가 비싼 모든 분야에 적용되는 패턴.

7. **U-Net is the right architecture / U-Net이 적절한 구조** — Multi-scale receptive field가 cryo-TEM의 다양한 scale 구조 (단백질·축사·필라멘트)를 동시에 다룰 수 있다. Depth 2의 작은 U-Net으로도 충분 — over-fit 위험 낮음.

8. **From idea to community standard / 아이디어에서 표준으로** — 본 논문 이후 cryo-EM tomogram preprocessing 표준 단계로 자리잡았고, *Topaz-Denoise* (Bepler+ 2020), *IsoNet*, *DeepDeWedge* 등 후속 cryo-EM 디노이저들이 모두 cryo-CARE 패턴을 확장.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 N2N principle (inherited from paper #16) / 노이즈투노이즈 원리
$$
\theta^* = \arg\min_\theta \sum_{i} \bigl\|f_\theta(\hat x_i) - \hat y_i\bigr\|_2^2,\qquad \hat x_i = s_i + n_i,\;\hat y_i = s_i + n'_i,\; n_i \perp n'_i,\;\mathbb E[n_i] = \mathbb E[n'_i] = 0
$$
$\Rightarrow f_{\theta^*}(\hat x) = \mathbb E[\hat y \mid \hat x] = \mathbb E[s\mid \hat x]$.

### 4.2 Pair-construction protocols / 페어 구성 프로토콜
다섯 가지 cryo-TEM 페어 구성:
| Protocol | Inputs | Required mode |
|----------|--------|--------------|
| P2P-tap | adjacent tilt-angles | any tilt-series |
| P2P-ip  | two half-dose acquisitions | dual exposure |
| P2P-df  | even / odd movie frames | dose-fractionation, K2 detector |
| T2T-eoa | even / odd tilt angles → 2 tomograms | any tilt-series |
| T2T-df  | dose-fractionation per tilt → 2 tomograms | dose-fractionation |

### 4.3 3D U-Net training objective / 3D U-Net 학습 목표
T2T variants에서:
$$
\mathcal L_{T2T}(\theta) = \frac{1}{B}\sum_{b=1}^{B}\bigl\|f_\theta(V^{\mathrm{even}}_b) - V^{\mathrm{odd}}_b\bigr\|_2^2
$$
$V_b$는 $64\times 64\times 64$ sub-volume, $B=1200$ random sub-volumes.

### 4.4 Fourier Shell Correlation / 푸리에 셸 상관
정량적 분해능 척도. 두 tomogram 사이의 spatial-frequency-domain correlation:
$$
\mathrm{FSC}(k) = \frac{\sum_{|\mathbf k|\in [k, k+\Delta k]} \hat V_1(\mathbf k)\,\hat V_2^*(\mathbf k)}{\sqrt{\sum |\hat V_1|^2 \cdot \sum |\hat V_2|^2}}
$$
0.143 또는 0.5 임계값에서 cutoff resolution. cryo-CARE T2T-df FSC는 raw 보다 모든 대역에서 위.

### 4.5 Worked example: ODA detection / 작동 예시: ODA 검출
*C. reinhardtii* axoneme tomogram에서 outer dynein arm 검출.
- 학습 데이터: 1 hand-annotated tomogram + PEET refinement.
- 평가: 383 train / 712 test ODA.
- Workflow: U-Net dense segmentation → Otsu threshold → connected components → filter by size in voxels.
- 결과 (Fig. 5): cryo-CARE 적용 시 PR-curve 모든 segment-size에서 위로 이동.

### 4.6 Relationship with paper #16 / 논문 #16과의 관계
이 논문은 Lehtinen et al. (2018)의 *직접 구현 사례*. 새로운 *손실 함수*나 *학습 알고리즘*은 없으나 (i) cryo-EM 데이터 페어 *생성 카탈로그*, (ii) 2D→3D 확장, (iii) downstream impact 측정 — 이 세 측면에서 응용을 *완성*시킨다.

### 4.7 Beam-damage sharing argument / 빔 손상 공유 논증
P2P-df의 핵심 장점: dose-fractionation의 짝/홀 프레임은 *동일 시료*를 본 두 측정. 빔 손상은 누적 dose의 함수이므로 split point까지의 *induced beam damage*는 *두 영상에 공유*된다. 즉:
$$
s^{\mathrm{even}}(t_{\mathrm{split}}) = s^{\mathrm{odd}}(t_{\mathrm{split}})
$$
이는 N2N의 핵심 가정 $s_{\mathrm{input}} = s_{\mathrm{target}}$ (signal identical)을 *정확히* 만족시킨다. P2P-ip의 두 노출은 *시간차*가 있어 추가 빔 손상이 한쪽에만 발생할 수 있어 약간 열등.

### 4.8 Comparison with NAD baseline / NAD 베이스라인과의 비교
NAD (Frangakis-Hegerl 2001)는 $L = u_t - \nabla\cdot(g(|\nabla u|)\nabla u) = 0$ 형태의 PDE 디노이저. *국소* 그래디언트만 사용 → high-resolution 정보 보존 한계. cryo-CARE는 학습된 *비국소* prior로 수십 nm scale의 단백질·필라멘트 구조를 회복.

### 4.9 Why FSC is the right metric / 왜 FSC가 적절한 척도인가
일반 영상에서는 PSNR이 표준이지만 cryo-EM tomogram은:
1. *깨끗 ground truth가 없음* → PSNR 계산 불가.
2. *공간 주파수 별 정보량이 중요* (단백질 secondary structure는 ~10 Å, tertiary는 ~5 Å).
FSC는 두 *독립* 복원의 spatial-frequency-domain correlation으로 *분해능*을 직접 측정. cryo-CARE T2T-df의 FSC가 모든 대역에서 raw·NAD를 dominate한다는 결과 (Fig. 2)는 *real* 정보 복원의 강력한 증거.

### 4.10 Architectural choices summary / 구조 선택 요약
| Choice | Value | Why |
|--------|-------|-----|
| Architecture | U-Net depth 2 | Multi-scale; small enough to avoid overfitting on 1000 patches |
| Kernel | 3×3 (2D) / 3×3×3 (3D) | Standard, fast |
| Final activation | Linear | Regression to original intensity range (no saturation) |
| Loss | Per-pixel MSE | Matches N2N's L_2 derivation |
| Patch size | 128×128 (2D) / 64³ (3D) | Receptive field >> object scale of interest |
| Patches | 1000 (2D) / 1200 (3D) | Empirically sufficient |
| Validation hold-out | 10% | Standard practice |

### 4.11 Comparison with paper #17 (Noise2Void) on same data / 같은 데이터에서 #17 N2V와 비교
cryo-CARE는 페어 데이터 (P2P / T2T) 가 *반드시* 필요. dose-fractionation이 안 되거나 single tilt-series만 있으면 적용 불가. Krull et al. (paper #17, N2V)은 single 영상 만으로 학습 → cryo-CARE보다 *데이터 요구가 적지만* PSNR도 약 1–2 dB 낮음. 두 방법은 *상호 보완적*이며 cryo-EM 워크플로우에서 데이터 가용성에 따라 선택.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1968       DeRosier-Klug — first electron tomography (3D from tilt series)
1980       Knapek-Dubochet — beam damage limits dose
1996       Kremer-Mastronarde-McIntosh — IMOD (tomogram reconstruction)
2001       Frangakis-Hegerl — non-linear anisotropic diffusion (NAD) baseline
2013       Li+ — K2 direct detector + dose fractionation (Nature Methods)
2015       Ronneberger+ — U-Net (architecture used)
2016       Hasinoff+ — burst photography (zero-mean noise framing)
2017       Weigert+ — CARE (supervised, requires clean targets)
2018       LEHTINEN+ — Noise2Noise (paper #16)
                            ↳ proves clean targets are not necessary
2019 ★★    BUCHHOLZ+ — CRYO-CARE (THIS PAPER)
                            ↳ first N2N-based deep denoiser for cryo-TEM
                            ↳ catalogues P2P/T2T pair construction
2019       Krull+ — Noise2Void (paper #17)
                            ↳ generalises further to single-image
2020       Bepler+ — Topaz-Denoise (Nature Methods) — single-particle EM
2022       IsoNet, DeepDeWedge — fight missing-wedge with deep priors
2024+      Cryo-CARE remains a *default* preprocessing in many pipelines
```

이 논문은 cryo-EM 분야가 *deep denoising을 표준으로 채택한 결정적 분기점*. 이전엔 NAD·median 같은 단순 필터가 사용됐지만 본 논문 이후 N2N 계열이 빠르게 확산.

This paper is the **decisive moment when cryo-EM adopted deep denoising as a standard preprocessing step**. Before, NAD and median filters were the norm; after, N2N-style methods rapidly proliferated.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Lehtinen+ (2018)** Noise2Noise (paper #16) | Theoretical foundation | Cryo-CARE is a direct application — adopts N2N's loss/architecture/principle and supplies the cryo-EM-specific pair-construction recipes. |
| **Weigert+ (2017)** CARE | Software framework reused | Cryo-CARE is built on the open-source CARE codebase; "CARE" in the name acknowledges this. The key change is moving from supervised (clean-target) to N2N (noisy-target). |
| **Ronneberger+ (2015)** U-Net | Network architecture | Adopts the now-standard biomedical-imaging U-Net (depth 2 here for cryo-TEM compactness). |
| **Frangakis-Hegerl (2001)** NAD | Classical baseline | The non-linear anisotropic diffusion baseline that cryo-CARE outperforms across the FSC; classical PDE-based denoising. |
| **Kremer+ (1996)** IMOD / ETOMO | Tomogram reconstruction tool | Used to reconstruct the two independent tomograms in T2T-eoa and T2T-df. |
| **Li+ (2013)** K2 detector + dose-fractionation | Hardware enabler | Dose-fractionation is what allows the P2P-df / T2T-df even-odd splitting; cryo-CARE's best variants depend on this hardware. |
| **Krull+ (2019)** Noise2Void (paper #17) | Concurrent generalisation | N2V drops the paired-noise requirement entirely; on cryo-TEM data N2V also works (paper #17 Fig. 4 row 3) — direct comparison/extension. |
| **Buades+ (2005)** NL-means (paper #4) | Self-similarity baseline | Still used as a non-deep baseline for cryo-EM tomograms; cryo-CARE is its content-aware learned successor. |
| **Bepler+ (2020)** Topaz-Denoise | Direct successor | Single-particle cryo-EM denoiser that extends cryo-CARE's recipe to particle picking. |

---

### Implementation status notes / 구현 상태 노트

#### 한국어
- 저자들이 발표 후 *오픈소스 cryo-CARE 패키지* 공개 (PyTorch + CSBDeep). cryo-EM 사용자에게 권장 입력: aligned tilt-series + dose-fractionation movie.
- TOMO110 데이터는 2019년 시점 비공개; EMPIAR-10110은 공개 → 본 노트북에서는 *합성 데이터로 N2N 원리 자체를 시연* (cryo-EM 시료 자체는 다루지 않음).
- 후속 도구: Topaz-Denoise (Bepler+ 2020), IsoNet (Liu+ 2022), DeepDeWedge (Wiedemann 2023) 모두 cryo-CARE의 페어 구성 패턴을 확장.

#### English
The authors released an open-source cryo-CARE package (PyTorch + CSBDeep). Standard input is aligned tilt-series with dose-fractionation movies. TOMO110 was internal at publication; EMPIAR-10110 is the public reference. Successor tools (Topaz-Denoise, IsoNet, DeepDeWedge) extend the pair-construction pattern.

### Why this paper became standard practice / 이 논문이 표준이 된 이유

#### 한국어
1. **Drop-in compatibility**: 기존 cryo-EM 파이프라인 (MotionCor2, IMOD/ETOMO)에 *추가 단계*로 끼워넣을 수 있음 → 진입 장벽 zero.
2. **No annotation cost**: ground truth 라벨 또는 깨끗 영상 *전혀* 불필요 → 시설 운영 부담 zero.
3. **Reproducible quantitative gain**: FSC 곡선과 PR-curve 모두에서 *측정 가능한 개선* → 추측이 아닌 검증.
4. **Open implementation**: CARE/CSBDeep 위에 구현되어 *재현·확장 용이* → 커뮤니티가 빠르게 변형·개선.

#### English
The combination of zero-cost integration, no annotation requirement, measurable quantitative gain, and an open implementation made cryo-CARE the de facto standard preprocessing step in cryo-EM tomography by 2021–2022.

---

## 7. References / 참고문헌

- Buchholz, T.-O., Jordan, M., Pigino, G., & Jug, F. "Cryo-CARE: Content-Aware Image Restoration for Cryo-Transmission Electron Microscopy Data", *Proc. IEEE ISBI*, 502–506 (2019). [DOI: 10.1109/ISBI.2019.8759519]
- Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. "Noise2Noise: Learning Image Restoration without Clean Data", *Proc. ICML*, 2018. [arXiv:1803.04189]
- Weigert, M., Schmidt, U., Boothe, T., et al. "Content-Aware Image Restoration: Pushing the Limits of Fluorescence Microscopy", *bioRxiv* p. 236463 (2017).
- Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation", *Proc. MICCAI*, 234–241 (2015).
- Knapek, E., & Dubochet, J. "Beam damage to organic material is considerably reduced in cryo-electron microscopy", *J. Mol. Biol.*, 141(2), 147–161 (1980).
- Frangakis, A. S., & Hegerl, R. "Noise reduction in electron tomographic reconstructions using nonlinear anisotropic diffusion", *J. Struct. Biol.*, 135(3), 239–250 (2001).
- Kremer, J. R., Mastronarde, D. N., & McIntosh, J. R. "Computer visualization of three-dimensional image data using IMOD", *J. Struct. Biol.*, 116(1), 71–76 (1996).
- Li, X., Mooney, P., Zheng, S., et al. "Electron counting and beam-induced motion correction enable near-atomic-resolution single-particle cryo-EM", *Nature Methods*, 10(6), 584–590 (2013).
- Zheng, S., Palovcak, E., Armache, J.-P., Cheng, Y., & Agard, D. "Anisotropic correction of beam-induced motion for improved single-particle cryo-EM" (MotionCor2), *bioRxiv* (2016).
- Iudin, A., Korir, P. K., Salavert-Torres, J., Kleywegt, G. J., & Patwardhan, A. "EMPIAR: a public archive for raw electron microscopy image data", *Nature Methods*, 13(5), 387 (2016).
- Otsu, N. "A threshold selection method from gray-level histograms", *IEEE Trans. SMC*, 9(1), 62–66 (1979).
