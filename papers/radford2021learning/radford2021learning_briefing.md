---
title: "Pre-Reading Briefing: Learning Transferable Visual Models From Natural Language Supervision (CLIP)"
paper_id: "36_radford_2021"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# Learning Transferable Visual Models From Natural Language Supervision (CLIP): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. arXiv:2103.00020.
**Author(s)**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever (OpenAI)
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

**한국어**
CLIP(Contrastive Language–Image Pre-training)은 **인터넷에서 수집한 4억 개의 (이미지, 텍스트) 쌍**(WIT, WebImageText 데이터셋)을 사용하여 이미지 인코더와 텍스트 인코더를 **대조 학습(contrastive learning)** 으로 동시에 훈련합니다. 이전의 컴퓨터 비전 시스템은 미리 정해진 객체 카테고리 집합(예: ImageNet 1000 클래스)을 예측하도록 학습되어 있어 새로운 시각 개념에 일반화하기 어려웠습니다. CLIP은 이 한계를 깨고, **자연어로 묘사 가능한 모든 시각 개념을 zero-shot으로 분류**할 수 있게 합니다. ImageNet에서 학습 예제를 단 하나도 사용하지 않고도 76.2%의 zero-shot 정확도를 달성하며, 이는 ResNet-50 (1.28M 라벨로 학습) 수준의 성능에 해당합니다. 또한 30개 이상의 데이터셋(Food101, Pets, OCR, 동영상 인식 등)에서 강력한 전이 성능과 분포 변화에 대한 견고성(robustness)을 보입니다.

**English**
CLIP (Contrastive Language–Image Pre-training) jointly trains an image encoder and a text encoder via **contrastive learning** on **400 million (image, text) pairs** scraped from the internet (the WIT or WebImageText dataset). Previous computer vision systems were trained to predict a fixed set of object categories (e.g., ImageNet's 1000 classes), which severely limited their ability to generalize to new visual concepts. CLIP breaks this bottleneck and enables **zero-shot classification of any visual concept describable in natural language**. Without using a single ImageNet training example, CLIP attains 76.2% zero-shot top-1 accuracy on ImageNet — matching the original ResNet-50 trained on 1.28M labels. Across 30+ benchmarks (Food101, Pets, OCR, action recognition, etc.) CLIP transfers strongly and is dramatically more robust to natural distribution shifts (e.g., +51.2% on ImageNet-R, +74.4% on ImageNet-A vs. ResNet-101).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2010년대 컴퓨터 비전은 ImageNet 같은 **고비용 라벨 데이터셋**을 중심으로 발전했습니다. AlexNet (2012, paper #13), VGG, ResNet (2015, paper #20)은 모두 ImageNet 분류기로 사전학습된 후 다른 task로 fine-tuning됐습니다. 이 패러다임은 (1) 1000개 카테고리에 갇혀 있고 (2) 새 task마다 라벨 수집과 fine-tuning이 필요한 한계가 있었습니다. 동시에 NLP에서는 GPT-2 (paper #29), GPT-3, BERT가 **거대한 raw text**로 사전학습한 후 zero/few-shot transfer로 옮겨가는 혁명을 일으켰습니다. 비전이 NLP의 길을 따를 수 있을까요? CLIP은 그 답입니다.

**English**
Throughout the 2010s, computer vision was anchored by **costly labeled datasets** like ImageNet. AlexNet (2012, paper #13), VGG, and ResNet (2015, paper #20) were pre-trained as ImageNet classifiers and fine-tuned for downstream tasks. This paradigm was (1) locked into 1000 categories and (2) required new labels for every new task. Meanwhile NLP was being upended by GPT-2 (paper #29), GPT-3, and BERT — pre-train on **vast raw text**, then zero/few-shot transfer. Could vision follow? CLIP says yes.

### 타임라인 / Timeline

```
2009  ImageNet (Deng et al.) — supervised CV foundation
2012  AlexNet (paper #13) — deep CNN era begins
2015  ResNet (paper #20) — 152-layer networks
2017  Transformer (paper #25) — attention everywhere
2018  BERT, GPT-1 — NLP self-supervised pre-training
2019  GPT-2 (paper #29) — zero-shot NLP transfer
2019  VirTex, ICMLM — caption-based pre-training (limited scale)
2020  ViT (paper #31), SimCLR (paper #32) — vision transformers, contrastive vision
2020  ConVIRT (Zhang et al.) — contrastive image-text in medical
2021★ CLIP (this paper) — 400M pairs, zero-shot ImageNet 76.2%
2021  DALL-E, ALIGN — text-to-image, even larger image-text
2022  Flamingo, BLIP — multimodal foundation models
2023  GPT-4V, LLaVA — large multimodal models
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **신경망 기초**: CNN (paper #13, #20), Transformer (paper #25)
- **대조 학습**: SimCLR (paper #32) — InfoNCE 손실, positive/negative 쌍, temperature scaling
- **ViT**: Vision Transformer (paper #31) — 이미지를 패치로 나눠 Transformer로 처리
- **확률 & 정보 이론**: softmax, cross-entropy, mutual information의 직관
- **언어 모델**: BPE 토큰화, 트랜스포머 인코더, [SOS]/[EOS] 토큰
- **Optimizer**: Adam (paper #18), cosine learning rate schedule

**English**
- **Neural network basics**: CNNs (papers #13, #20), Transformers (paper #25)
- **Contrastive learning**: SimCLR (paper #32) — InfoNCE loss, positive/negative pairs, temperature scaling
- **ViT**: Vision Transformer (paper #31) — split images into patches, feed to a Transformer
- **Probability & info theory**: softmax, cross-entropy, mutual-information intuition
- **Language modeling**: BPE tokenization, transformer encoders, [SOS]/[EOS] tokens
- **Optimizer**: Adam (paper #18), cosine LR schedule

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Contrastive Learning** | 매칭되는 쌍은 가까이, 매칭되지 않는 쌍은 멀리 임베딩하도록 학습 / Train embeddings so matched pairs are close and unmatched pairs are far |
| **Zero-shot Transfer** | 다운스트림 task의 학습 데이터를 전혀 사용하지 않고 분류 / Classify without using any task-specific training data |
| **WIT (WebImageText)** | OpenAI가 인터넷에서 수집한 400M (이미지, 텍스트) 쌍 / 400M (image, text) pairs scraped from the web |
| **Image Encoder** | 이미지를 임베딩 벡터로 변환 (ResNet 또는 ViT) / Maps images to embeddings (ResNet or ViT) |
| **Text Encoder** | 텍스트를 임베딩 벡터로 변환 (63M 파라미터 Transformer) / Maps text to embeddings (63M-parameter Transformer) |
| **Symmetric InfoNCE** | 이미지→텍스트, 텍스트→이미지 방향 cross-entropy의 평균 / Mean of image→text and text→image cross-entropy losses |
| **Temperature τ** | 로짓 스케일링 파라미터; CLIP은 log-parameterized로 학습 / Logit scaling, learned in log-space |
| **Prompt Engineering** | "a photo of a {label}" 같은 템플릿으로 분류 정확도 개선 / Templates like "a photo of a {label}" boost zero-shot accuracy |
| **Prompt Ensembling** | 여러 prompt의 텍스트 임베딩을 평균하여 견고성 향상 / Average embeddings across many prompts for robustness |
| **Linear Probe** | 사전학습 feature 위에 logistic regression만 학습 / Train logistic regression on frozen features |
| **Effective Robustness** | 분포 외 정확도가 in-distribution 정확도로부터 예측되는 선보다 얼마나 위에 있는가 / How far OOD accuracy lies above the line predicted by in-distribution accuracy |
| **ViT-L/14** | Patch size 14의 ViT-Large (CLIP의 최고 모델) / ViT-Large with patch 14 (best CLIP model) |

---

## 5. 수식 미리보기 / Equations Preview

**한국어**

**1. 코사인 유사도 (cosine similarity)**:
$$\text{sim}(I, T) = \frac{I \cdot T}{\|I\| \|T\|}$$
- L2-normalized 임베딩의 내적으로 효율적으로 계산.

**2. 대칭 InfoNCE 손실 (symmetric contrastive loss)**:
$$\mathcal{L}_{i \to t} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j)/\tau)}$$
$$\mathcal{L}_{t \to i} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(T_i, I_j)/\tau)}$$
$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i})$$

**3. Zero-shot 분류 (zero-shot classification)**:
$$P(y=k \mid x) = \frac{\exp(\text{sim}(I_x, T_k)/\tau)}{\sum_{k'=1}^{K} \exp(\text{sim}(I_x, T_{k'})/\tau)}$$
- $T_k$ = "a photo of a {class$_k$}"의 텍스트 임베딩.

**English**
The three equations above formalize CLIP's training and inference. Cosine similarity is computed between L2-normalized embeddings; the symmetric InfoNCE loss treats both modalities as queries; and zero-shot classification reuses the same softmax-over-similarities at test time, replacing class labels with text embeddings of natural-language prompts.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
1. **§1 Introduction**: NLP의 sclable pretraining이 비전에 적용 가능한가? (★)
2. **§2 Approach**: 데이터셋(WIT 400M), 학습 목표(contrastive vs predictive), 모델 선택. **Figure 1, Figure 3** 의사코드는 반드시 이해. (★★★)
3. **§3.1 Zero-Shot Transfer**: prompt engineering, prompt ensembling, 27개 데이터셋 비교. (★★★)
4. **§3.2 Representation Learning**: linear probe로 SOTA 비교. ViT-L/14가 EfficientNet-L2 NS를 능가. (★★)
5. **§3.3 Robustness to Natural Distribution Shift**: zero-shot CLIP이 robustness gap을 75% 줄임. (★★)
6. **§3.4 Comparison to Human Performance**: Oxford Pets에서 인간 vs CLIP. (시간 있을 때)
7. **§4 Data Overlap**: train-test 중첩 분석. (시간 있을 때)
8. **§7 Broader Impacts**: bias, surveillance, fairness. (★)

**English**
Prioritize §1, §2, and §3.1–§3.3. Pseudocode in Figure 3 is the entire CLIP training loop in ~15 lines — make sure you can re-derive it. The robustness section (§3.3, Figures 13–14) is a profound finding: zero-shot CLIP is much more robust than ImageNet-trained models on natural shifts.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
CLIP은 **multimodal foundation model 시대의 시작**입니다. 그 영향은 광범위합니다:
- **Text-to-Image generation**: DALL-E, Stable Diffusion, Imagen 모두 CLIP 임베딩을 텍스트 조건 신호로 사용.
- **Open-vocabulary detection/segmentation**: GroundingDINO, OWL-ViT, SAM이 자연어 쿼리로 이미지에서 영역을 찾음.
- **Vision-language models**: Flamingo, BLIP, LLaVA, GPT-4V는 모두 CLIP-style vision encoder 위에 구축.
- **Retrieval & search**: 임베딩 공간에서 텍스트로 이미지 검색 (역도 가능).
- **Robotics & embodied AI**: 자연어 task description을 시각적 reward로 변환.

CLIP의 메시지는 단순합니다: **데이터를 충분히 모으고, 자연어로 감독 신호를 주면, 손에 쥔 분류 태깅을 넘어선 일반 시각 표현이 학습된다**. 이는 비전 분야의 GPT-2 모먼트입니다.

**English**
CLIP marks the **beginning of the multimodal foundation model era**. Its influence is everywhere: DALL-E/Stable Diffusion/Imagen condition on CLIP text embeddings; open-vocabulary detection (GroundingDINO, OWL-ViT) and segmentation use CLIP-derived encoders; vision-language models (Flamingo, BLIP, LLaVA, GPT-4V) build on CLIP-style vision backbones; cross-modal retrieval and robotics rewards leverage the joint embedding space. The core message — *gather web-scale image-text data, train contrastively, and you get a transferable, robust, open-vocabulary vision system* — is the GPT-2 moment for computer vision.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
