# Low-SNR Imaging / 저신호잡음비 영상

## Overview / 개요
A study track on **image restoration and faint-signal enhancement in the low signal-to-noise regime** — where per-pixel SNR is ≲ a few and the signal is buried in comparable-or-larger background noise, often without clean ground truth. Covers classical denoising (Anscombe-stabilized BM3D, NLM, wavelet/curvelet/shearlet shrinkage), self-supervised deep learning (Noise2Noise/Void/Self family), diffusion-based inverse-problem solvers (DDRM, DPS), faint-signal visual enhancement (NRGF, MGN, à-trous coronagraph filtering), cosmic-ray rejection (L.A.Cosmic, deepCR), and low-light deep learning (LLNet, MIRNet, Zero-DCE, EnlightenGAN). Born from the LOLIPOP project's algorithm survey, but the techniques generalise to coronagraph/auroral/cryo-EM/microscopy data and any photon-starved scientific imaging.

**저신호잡음비 regime에서의 영상 복원과 약한 신호 강조**를 다루는 학습 트랙. per-pixel SNR ≲ 수 단위, 신호가 비슷하거나 큰 배경 잡음에 묻혀 있고 clean ground truth 가 없는 경우가 표적. 고전 denoising (Anscombe-안정화 BM3D, NLM, wavelet/curvelet/shearlet shrinkage), 자기지도 딥러닝 (Noise2Noise/Void/Self 계열), diffusion 기반 역문제 풀이 (DDRM, DPS), 약한 신호 시각화 (NRGF, MGN, à-trous coronagraph), cosmic-ray rejection (L.A.Cosmic, deepCR), 저조도 딥러닝 (LLNet, MIRNet, Zero-DCE, EnlightenGAN) 을 망라. LOLIPOP 프로젝트의 알고리즘 서베이에서 출발했지만, 코로나그래프·오로라·cryo-EM·현미경·광자 부족 과학 영상 전반으로 일반화 가능.

## Learning Roadmap / 학습 로드맵

### Phase 1 (Tier B basics): Classical Patch-Based and Transform-Domain Denoising / 고전 패치·변환 영역 denoising (1994–2013)
- Wavelet shrinkage thresholding (VisuShrink, SureShrink, BayesShrink) / wavelet shrinkage 임계법
- Curvelet, contourlet, shearlet — directional sparsity / 방향성 희소성
- Non-local means (NLM) and BM3D / 비국소 평균 및 BM3D
- BM4D / V-BM4D for volumetric and video data / 볼륨 및 비디오 데이터

### Phase 2 (Tier A — Poisson stats): Photon-Statistics-Aware Denoising / 광자 통계 기반 denoising (1948–2013)
- Anscombe / generalized Anscombe variance-stabilizing transforms / 분산 안정화 변환
- Optimal Anscombe inversion for Poisson–Gaussian / Poisson–Gaussian 최적 역변환
- PURE-LET (unbiased risk for mixed noise) / 혼합 잡음 PURE-LET
- Poisson NL means / Poisson NLM

### Phase 3 (Tier A — DL): Self-Supervised Deep Learning Denoising / 자기지도 딥러닝 denoising (2018–2022)
- Noise2Noise lineage (training without clean ground truth) / clean GT 없는 학습
- Blind-spot networks (Noise2Void, Noise2Self) / 블라인드 스팟 네트워크
- Self-supervised single-image (Self2Self, Neighbor2Neighbor, Blind2Unblind, R2R) / 단일 영상 자기지도
- Domain reference: Cryo-CARE (cryo-EM regime origin) / Cryo-CARE (regime 기원 reference)

### Phase 4 (Tier B — astronomy): Outlier and Cosmic-Ray Rejection / 이상치 및 cosmic-ray 제거 (2001–2020)
- L.A.Cosmic — Laplacian edge detection / Laplacian 엣지 검출
- deepCR — CNN-based cosmic-ray rejection / CNN 기반 cosmic-ray 제거

### Phase 5 (Tier C): Diffusion-Based Inverse-Problem Restoration / Diffusion 모형 기반 역문제 복원 (2022–2023)
- DDRM, DPS — pretrained-prior posterior sampling / 사전훈련 prior 사후표본
- DiffPIR — plug-and-play with diffusion / Diffusion plug-and-play
- DiffLL — wavelet-domain low-light diffusion / Wavelet 영역 저조도 diffusion
- Caveats: prior hallucination, injection-recovery validation / Prior hallucination, injection-recovery 검증

### Phase 6 (Tier D): Faint-Signal Enhancement (Solar/Astronomy) / 약한 신호 강조 (태양·천문 특화) (2003–2022)
- à-trous / undecimated wavelet transform / à-trous 웨이블릿 변환
- NRGF, MGN — coronal radial filtering / 코로나 동경 필터링
- SiRGraF — radial gradient filter / 동경 기울기 필터
- Wavelet-packets equalization for coronagraphs / 코로나그래프 wavelet packets 균등화
- Robust PCA — low-rank + sparse decomposition / 저랭크 + 희소 분해

### Phase 7 (Tier E): Low-Light Deep Learning / 저조도 딥러닝 (2017–2023)
- LLNet, MIRNet — paired supervised low-light enhancement / 쌍 지도 저조도 강화
- Zero-DCE, EnlightenGAN — zero-reference / unpaired / zero-reference / 비쌍 학습
- SCI, SNR-aware — fast / SNR-conditioned architectures / 빠른 SNR-조건화 구조

## Directory Structure / 디렉토리 구조
```
Low_SNR_Imaging/
├── papers/          # Curated paper reading list and per-paper notes / 논문 리딩 리스트 및 논문별 노트
├── notes/           # Theory and concept notes (Markdown) / 이론 및 개념 노트
├── notebooks/       # Practice and implementation (Jupyter) / 실습 및 구현
├── scripts/         # Standalone Python scripts / 독립 실행 스크립트
├── data/            # Sample datasets (synthetic Poisson-Gaussian, real low-SNR) / 샘플 데이터셋
└── README.md        # This file / 이 파일
```

## Source / 출처
The initial 41-paper curation is extracted from the LOLIPOP project's `LOW_SNR_WEAK_SIGNAL_REFERENCES.md` (audited 2026-04-30, 4 verification agents, 181 algorithm rows). Eight Tier-D entries (NRGF, MGN, à-trous coronagraph, SiRGraF, RPCA-related) are also relevant to `Solar_Observation/` Phase 7 and are cross-referenced from there.

초기 41편 큐레이션은 LOLIPOP 프로젝트의 `LOW_SNR_WEAK_SIGNAL_REFERENCES.md` (2026-04-30 audit, 4 verification agent, 181 알고리즘 행) 에서 추출. Tier-D 8편 (NRGF, MGN, à-trous coronagraph, SiRGraF, RPCA 관련) 은 `Solar_Observation/` Phase 7 과 연관성이 있어 cross-reference 되어 있음.

## Status / 상태
**Active** — Initial scaffolding 2026-05-01. / 2026-05-01 초기 스캐폴드 완료.
