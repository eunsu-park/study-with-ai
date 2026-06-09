# Study with AI / AI와 함께 공부하기

A personal study project for learning diverse subjects from beginner to expert level, with AI (Claude) as a study partner.
AI(Claude)를 학습 파트너로 삼아 초급부터 전문가 수준까지 다양한 주제를 학습하는 개인 프로젝트입니다.

The core method: **read landmark papers chronologically, understand the ideas, and implement them in code** — building deep knowledge from first principles.
핵심 방법: **핵심 논문을 시간순으로 읽고, 아이디어를 이해하며, 코드로 구현** — 기본 원리에서부터 깊은 지식을 쌓아갑니다.

## Topics / 주제

| Topic / 주제 | Description / 설명 | Papers / 논문 | Status / 상태 |
|---|---|---|---|
| [Artificial Intelligence](Artificial_Intelligence/) | From McCulloch-Pitts neurons (1943) to modern LLMs / McCulloch-Pitts 뉴런(1943)부터 현대 LLM까지 | 40 | **Active** |
| [Solar Physics](Solar_Physics/) | From Galileo's sunspots (1613) to Parker Solar Probe / Galileo 흑점(1613)부터 Parker Solar Probe까지 | 38 | **Active** |
| [Space Weather](Space_Weather/) | From Birkeland's aurora (1908) to ML forecasting / Birkeland 극광(1908)부터 ML 예보까지 | 81 | **Active** |
| [Solar Observation](Solar_Observation/) | Ground-based & space-based solar instruments / 지상 및 우주 태양 관측 기기 | 80 | **Active** |
| [Living Reviews in Solar Physics](Living_Reviews_in_Solar_Physics/) | LRSP journal reviews (2004–present) / LRSP 저널 리뷰 (2004–현재) | 87+ | **Active** |
| [Low-SNR Imaging](Low_SNR_Imaging/) | Image restoration & faint-signal enhancement in the strong-noise regime / 강잡음·약신호 regime 영상 복원 | 45 | **Active** |
| [Helioseismology & Asteroseismology](Helioseismology_Asteroseismology/) | Solar & stellar oscillations as probes of stellar interiors / 태양·항성 진동을 항성 내부 탐침으로 | 7 | **Active** |
| [Magnetic Reconnection & Eruption](Magnetic_Reconnection_Eruption/) | Reconnection theory and the flares/CMEs it powers / 재결합 이론 및 플레어·CME | 10 | **Active** |
| [Heliosphere & Solar Wind](Heliosphere_Solar_Wind/) | Solar wind, heliosphere, in-situ from Mariner 2 to PSP / 태양풍·헬리오스피어, in-situ 관측 | 5 | **Active** |
| [Plasma Spectroscopy & Diagnostics](Plasma_Spectroscopy_Diagnostics/) | DEM, atomic data (CHIANTI), line-ratio diagnostics, abundances / DEM·원자 데이터·진단·abundance | 20 | **Active** |
| [Numerical MHD Simulation](Numerical_MHD_Simulation/) | MHD codes, schemes, divergence cleaning, AMR / MHD 코드·수치 기법·divergence cleaning·AMR | 1 | **Active** |

## How It Works / 학습 방법

### Paper-Driven Learning / 논문 기반 학습

Each topic is structured around a curated list of landmark papers, read in chronological order. For each paper, the workflow is:
각 주제는 시간순으로 읽는 엄선된 핵심 논문 리스트로 구성됩니다. 각 논문의 워크플로우:

1. **Prepare / 준비** — Review prerequisites, historical context, and key vocabulary / 전제 조건, 역사적 맥락, 핵심 용어 검토
2. **Read / 읽기** — Read the original paper / 원본 논문 읽기
3. **Discuss / 토론** — Clarify difficult passages and answer questions / 어려운 부분 명확히 하고 질문 답변
4. **Summarize / 요약** — Write structured notes capturing key insights / 핵심 통찰을 담은 구조화된 노트 작성
5. **Implement / 구현** — Reproduce key algorithms or results in code (when applicable) / 핵심 알고리즘 또는 결과를 코드로 재현

### Content Types / 콘텐츠 유형

| Type / 유형 | Format / 형식 | Location / 위치 |
|---|---|---|
| Paper reading notes / 논문 읽기 노트 | Markdown | `papers/<paper_name>/notes.md` |
| Code implementations / 코드 구현 | Jupyter Notebook | `papers/<paper_name>/implementation.ipynb` |
| Theory notes / 이론 노트 | Markdown | `notes/` |
| Practice & analysis / 실습 및 분석 | Jupyter Notebook | `notebooks/` |
| Utility scripts / 유틸리티 스크립트 | Python / IDL | `scripts/` |

## Project Structure / 프로젝트 구조

All papers live in a single flat `papers/` folder, named by BibTeX-style
citekey (`<surname><year><keyword>`). Topic grouping is expressed through the
README index and the `topics/` Maps-of-Content, not the directory tree.

모든 논문은 단일 평면 `papers/` 폴더에 BibTeX 스타일 citekey
(`<성><연도><키워드>`)로 저장됩니다. 주제 구분은 폴더가 아니라 README 색인과
`topics/` MOC로 표현됩니다.

```
StudyWithAI/
├── CLAUDE.md              # Project rules / 프로젝트 규칙
├── README.MD              # This file — auto-generated paper index / 자동 생성 색인
├── papers/                # ALL papers, flat, by citekey / 모든 논문 (평면, citekey)
│   ├── vaswani2017attention/
│   │   ├── vaswani2017attention_notes.md
│   │   ├── vaswani2017attention_briefing.md
│   │   ├── vaswani2017attention_implementation.ipynb
│   │   └── vaswani2017attention_paper.pdf   # gitignored (re-download via DOI)
│   └── ... (409 papers)
├── topics/                # Per-topic Maps-of-Content / 주제별 MOC
│   ├── artificial-intelligence.md
│   └── ...
├── <Topic>/papers/reading_list.md  # Curated reading lists (data source) / 큐레이션 목록
├── docs/                  # WORKFLOW, SKILLS, IMPROVEMENTS, MCP_SETUP ...
└── scripts/               # CLI tooling
    ├── flatten_migrate.py # Flatten migration / 평면화 마이그레이션
    └── gen_index.py       # Regenerate README index + topics MOCs / 색인 재생성
```

## Current Progress / 현재 진행 상황

| Topic / 주제 | Progress / 진행 |
|---|---|
| [Artificial Intelligence / 인공지능](#artificial-intelligence) | 40 / 69 |
| [Solar Physics / 태양물리학](#solar-physics) | 53 / 60 |
| [Space Weather / 우주기상](#space-weather) | 81 / 81 |
| [Solar Observation / 태양관측](#solar-observation) | 80 / 81 |
| [Living Reviews in Solar Physics / 리빙 리뷰](#living-reviews-solar-physics) | 87 / 87 |
| [Low-SNR Imaging / 저신호대잡음 영상](#low-snr-imaging) | 45 / 45 |
| [Helioseismology & Asteroseismology / 일진동·성진동학](#helioseismology-asteroseismology) | 5 / 7 |
| [Magnetic Reconnection & Eruption / 자기재결합·분출](#magnetic-reconnection-eruption) | 8 / 10 |
| [Heliosphere & Solar Wind / 태양권·태양풍](#heliosphere-solar-wind) | 4 / 6 |
| [Plasma Spectroscopy & Diagnostics / 플라즈마 분광·진단](#plasma-spectroscopy-diagnostics) | 4 / 20 |
| [Numerical MHD Simulation / 수치 MHD 시뮬레이션](#numerical-mhd-simulation) | 0 / 1 |
| **Total / 합계** | **407 / 467** |

<a id="artificial-intelligence"></a>

### Artificial Intelligence / 인공지능 (40 / 69)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | A Logical Calculus of the Ideas Immanent in Nervous Activity | 1943 | ✅ | [📝 notes](papers/mcculloch1943logical/mcculloch1943logical_notes.md) · [💻 code](papers/mcculloch1943logical/mcculloch1943logical_implementation.ipynb) · [📄 pdf](papers/mcculloch1943logical/mcculloch1943logical_paper.pdf) |
| 2 | Computing Machinery and Intelligence | 1950 | ✅ | [📝 notes](papers/turing1950computing/turing1950computing_notes.md) · [💻 code](papers/turing1950computing/turing1950computing_implementation.ipynb) · [📄 pdf](papers/turing1950computing/turing1950computing_paper.pdf) |
| 3 | The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain | 1958 | ✅ | [📝 notes](papers/rosenblatt1958perceptron/rosenblatt1958perceptron_notes.md) · [💻 code](papers/rosenblatt1958perceptron/rosenblatt1958perceptron_implementation.ipynb) · [📄 pdf](papers/rosenblatt1958perceptron/rosenblatt1958perceptron_paper.pdf) |
| 4 | Perceptrons (book) | 1969 | ✅ | [📝 notes](papers/minsky1969perceptrons/minsky1969perceptrons_notes.md) · [💻 code](papers/minsky1969perceptrons/minsky1969perceptrons_implementation.ipynb) · [📄 pdf](papers/minsky1969perceptrons/minsky1969perceptrons_paper.pdf) |
| 5 | Neural Networks and Physical Systems with Emergent Collective Computational Abilities | 1982 | ✅ | [📝 notes](papers/hopfield1982neural/hopfield1982neural_notes.md) · [💻 code](papers/hopfield1982neural/hopfield1982neural_implementation.ipynb) · [📄 pdf](papers/hopfield1982neural/hopfield1982neural_paper.pdf) |
| 6 | Learning Representations by Back-propagating Errors | 1986 | ✅ | [📝 notes](papers/rumelhart1986learning/rumelhart1986learning_notes.md) · [💻 code](papers/rumelhart1986learning/rumelhart1986learning_implementation.ipynb) · [📄 pdf](papers/rumelhart1986learning/rumelhart1986learning_paper.pdf) |
| 7 | Backpropagation Applied to Handwritten Zip Code Recognition | 1989 | ✅ | [📝 notes](papers/lecun1989backpropagation/lecun1989backpropagation_notes.md) · [💻 code](papers/lecun1989backpropagation/lecun1989backpropagation_implementation.ipynb) · [📄 pdf](papers/lecun1989backpropagation/lecun1989backpropagation_paper.pdf) |
| 8 | Support-Vector Networks | 1995 | ✅ | [📝 notes](papers/cortes1995supportvector/cortes1995supportvector_notes.md) · [💻 code](papers/cortes1995supportvector/cortes1995supportvector_implementation.ipynb) · [📄 pdf](papers/cortes1995supportvector/cortes1995supportvector_paper.pdf) |
| 9 | Long Short-Term Memory | 1997 | ✅ | [📝 notes](papers/hochreiter1997long/hochreiter1997long_notes.md) · [💻 code](papers/hochreiter1997long/hochreiter1997long_implementation.ipynb) · [📄 pdf](papers/hochreiter1997long/hochreiter1997long_paper.pdf) |
| 10 | Gradient-Based Learning Applied to Document Recognition | 1998 | ✅ | [📝 notes](papers/lecun1998gradientbased/lecun1998gradientbased_notes.md) · [💻 code](papers/lecun1998gradientbased/lecun1998gradientbased_implementation.ipynb) · [📄 pdf](papers/lecun1998gradientbased/lecun1998gradientbased_paper.pdf) |
| 11 | Random Forests | 2001 | ✅ | [📝 notes](papers/breiman2001random/breiman2001random_notes.md) · [💻 code](papers/breiman2001random/breiman2001random_implementation.ipynb) · [📄 pdf](papers/breiman2001random/breiman2001random_paper.pdf) |
| 12 | A Fast Learning Algorithm for Deep Belief Nets | 2006 | ✅ | [📝 notes](papers/hinton2006fast/hinton2006fast_notes.md) · [💻 code](papers/hinton2006fast/hinton2006fast_implementation.ipynb) · [📄 pdf](papers/hinton2006fast/hinton2006fast_paper.pdf) |
| 13 | ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | 2012 | ✅ | [📝 notes](papers/krizhevsky2012imagenet/krizhevsky2012imagenet_notes.md) · [💻 code](papers/krizhevsky2012imagenet/krizhevsky2012imagenet_implementation.ipynb) · [📄 pdf](papers/krizhevsky2012imagenet/krizhevsky2012imagenet_paper.pdf) |
| 14 | Efficient Estimation of Word Representations in Vector Space (Word2Vec) | 2013 | ✅ | [📝 notes](papers/mikolov2013efficient/mikolov2013efficient_notes.md) · [💻 code](papers/mikolov2013efficient/mikolov2013efficient_implementation.ipynb) · [📄 pdf](papers/mikolov2013efficient/mikolov2013efficient_paper.pdf) |
| 15 | Auto-Encoding Variational Bayes (VAE) | 2013 | ✅ | [📝 notes](papers/kingma2013autoencoding/kingma2013autoencoding_notes.md) · [💻 code](papers/kingma2013autoencoding/kingma2013autoencoding_implementation.ipynb) · [📄 pdf](papers/kingma2013autoencoding/kingma2013autoencoding_paper.pdf) |
| 16 | Generative Adversarial Nets (GANs) | 2014 | ✅ | [📝 notes](papers/goodfellow2014generative/goodfellow2014generative_notes.md) · [💻 code](papers/goodfellow2014generative/goodfellow2014generative_implementation.ipynb) · [📄 pdf](papers/goodfellow2014generative/goodfellow2014generative_paper.pdf) |
| 17 | Neural Machine Translation by Jointly Learning to Align and Translate | 2014 | ✅ | [📝 notes](papers/bahdanau2014neural/bahdanau2014neural_notes.md) · [💻 code](papers/bahdanau2014neural/bahdanau2014neural_implementation.ipynb) · [📄 pdf](papers/bahdanau2014neural/bahdanau2014neural_paper.pdf) |
| 18 | Adam: A Method for Stochastic Optimization | 2014 | ✅ | [📝 notes](papers/kingma2014adam/kingma2014adam_notes.md) · [💻 code](papers/kingma2014adam/kingma2014adam_implementation.ipynb) · [📄 pdf](papers/kingma2014adam/kingma2014adam_paper.pdf) |
| 19 | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | 2015 | ✅ | [📝 notes](papers/ioffe2015batch/ioffe2015batch_notes.md) · [💻 code](papers/ioffe2015batch/ioffe2015batch_implementation.ipynb) · [📄 pdf](papers/ioffe2015batch/ioffe2015batch_paper.pdf) |
| 20 | Deep Residual Learning for Image Recognition (ResNet) | 2015 | ✅ | [📝 notes](papers/he2015deep/he2015deep_notes.md) · [💻 code](papers/he2015deep/he2015deep_implementation.ipynb) · [📄 pdf](papers/he2015deep/he2015deep_paper.pdf) |
| 21 | Layer Normalization | 2016 | ✅ | [📝 notes](papers/ba2016layer/ba2016layer_notes.md) · [💻 code](papers/ba2016layer/ba2016layer_implementation.ipynb) · [📄 pdf](papers/ba2016layer/ba2016layer_paper.pdf) |
| 22 | Playing Atari with Deep Reinforcement Learning (DQN) | 2013 | ✅ | [📝 notes](papers/mnih2013playing/mnih2013playing_notes.md) · [💻 code](papers/mnih2013playing/mnih2013playing_implementation.ipynb) · [📄 pdf](papers/mnih2013playing/mnih2013playing_paper.pdf) |
| 23 | Mastering the game of Go with deep neural networks and tree search (AlphaGo) | 2016 | ✅ | [📝 notes](papers/silver2016mastering/silver2016mastering_notes.md) · [💻 code](papers/silver2016mastering/silver2016mastering_implementation.ipynb) · [📄 pdf](papers/silver2016mastering/silver2016mastering_paper.pdf) |
| 24 | Proximal Policy Optimization Algorithms (PPO) | 2017 | ✅ | [📝 notes](papers/schulman2017proximal/schulman2017proximal_notes.md) · [💻 code](papers/schulman2017proximal/schulman2017proximal_implementation.ipynb) · [📄 pdf](papers/schulman2017proximal/schulman2017proximal_paper.pdf) |
| 25 | Attention Is All You Need | 2017 | ✅ | [📝 notes](papers/vaswani2017attention/vaswani2017attention_notes.md) · [💻 code](papers/vaswani2017attention/vaswani2017attention_implementation.ipynb) · [📄 pdf](papers/vaswani2017attention/vaswani2017attention_paper.pdf) |
| 26 | Semi-Supervised Classification with Graph Convolutional Networks | 2017 | ✅ | [📝 notes](papers/kipf2017semisupervised/kipf2017semisupervised_notes.md) · [💻 code](papers/kipf2017semisupervised/kipf2017semisupervised_implementation.ipynb) · [📄 pdf](papers/kipf2017semisupervised/kipf2017semisupervised_paper.pdf) |
| 27 | Neural Architecture Search with Reinforcement Learning | 2017 | ✅ | [📝 notes](papers/zoph2017neural/zoph2017neural_notes.md) · [💻 code](papers/zoph2017neural/zoph2017neural_implementation.ipynb) · [📄 pdf](papers/zoph2017neural/zoph2017neural_paper.pdf) |
| 28 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 2018 | ✅ | [📝 notes](papers/devlin2018bert/devlin2018bert_notes.md) · [💻 code](papers/devlin2018bert/devlin2018bert_implementation.ipynb) · [📄 pdf](papers/devlin2018bert/devlin2018bert_paper.pdf) |
| 29 | Language Models are Unsupervised Multitask Learners (GPT-2) | 2019 | ✅ | [📝 notes](papers/radford2019language/radford2019language_notes.md) · [💻 code](papers/radford2019language/radford2019language_implementation.ipynb) · [📄 pdf](papers/radford2019language/radford2019language_paper.pdf) |
| 30 | The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | 2019 | ✅ | [📝 notes](papers/frankle2019lottery/frankle2019lottery_notes.md) · [💻 code](papers/frankle2019lottery/frankle2019lottery_implementation.ipynb) · [📄 pdf](papers/frankle2019lottery/frankle2019lottery_paper.pdf) |
| 31 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT) | 2020 | ✅ | [📝 notes](papers/dosovitskiy2020image/dosovitskiy2020image_notes.md) · [💻 code](papers/dosovitskiy2020image/dosovitskiy2020image_implementation.ipynb) · [📄 pdf](papers/dosovitskiy2020image/dosovitskiy2020image_paper.pdf) |
| 32 | A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) | 2020 | ✅ | [📝 notes](papers/chen2020simple/chen2020simple_notes.md) · [💻 code](papers/chen2020simple/chen2020simple_implementation.ipynb) · [📄 pdf](papers/chen2020simple/chen2020simple_paper.pdf) |
| 33 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG) | 2020 | ✅ | [📝 notes](papers/lewis2020retrievalaugmented/lewis2020retrievalaugmented_notes.md) · [💻 code](papers/lewis2020retrievalaugmented/lewis2020retrievalaugmented_implementation.ipynb) · [📄 pdf](papers/lewis2020retrievalaugmented/lewis2020retrievalaugmented_paper.pdf) |
| 34 | Language Models are Few-Shot Learners (GPT-3) | 2020 | ✅ | [📝 notes](papers/brown2020language/brown2020language_notes.md) · [💻 code](papers/brown2020language/brown2020language_implementation.ipynb) · [📄 pdf](papers/brown2020language/brown2020language_paper.pdf) |
| 35 | Denoising Diffusion Probabilistic Models (DDPM) | 2020 | ✅ | [📝 notes](papers/ho2020denoising/ho2020denoising_notes.md) · [💻 code](papers/ho2020denoising/ho2020denoising_implementation.ipynb) · [📄 pdf](papers/ho2020denoising/ho2020denoising_paper.pdf) |
| 36 | Learning Transferable Visual Models From Natural Language Supervision (CLIP) | 2021 | ✅ | [📝 notes](papers/radford2021learning/radford2021learning_notes.md) · [💻 code](papers/radford2021learning/radford2021learning_implementation.ipynb) · [📄 pdf](papers/radford2021learning/radford2021learning_paper.pdf) |
| 37 | Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2) | 2021 | ✅ | [📝 notes](papers/jumper2021highly/jumper2021highly_notes.md) · [💻 code](papers/jumper2021highly/jumper2021highly_implementation.ipynb) · [📄 pdf](papers/jumper2021highly/jumper2021highly_paper.pdf) |
| 38 | Training Language Models to Follow Instructions with Human Feedback (InstructGPT) | 2022 | ✅ | [📝 notes](papers/ouyang2022training/ouyang2022training_notes.md) · [💻 code](papers/ouyang2022training/ouyang2022training_implementation.ipynb) · [📄 pdf](papers/ouyang2022training/ouyang2022training_paper.pdf) |
| 39 | Scaling Laws for Neural Language Models | 2020 | ✅ | [📝 notes](papers/kaplan2020scaling/kaplan2020scaling_notes.md) · [💻 code](papers/kaplan2020scaling/kaplan2020scaling_implementation.ipynb) · [📄 pdf](papers/kaplan2020scaling/kaplan2020scaling_paper.pdf) |
| 40 | Constitutional AI: Harmlessness from AI Feedback | 2022 | ✅ | [📝 notes](papers/bai2022constitutional/bai2022constitutional_notes.md) · [💻 code](papers/bai2022constitutional/bai2022constitutional_implementation.ipynb) · [📄 pdf](papers/bai2022constitutional/bai2022constitutional_paper.pdf) |
| 41 | Technical Note: Q-Learning | 1992 | ⬜ | [📄 pdf](papers/watkins1992technical/watkins1992technical_paper.pdf) |
| 42 | Improving Neural Networks by Preventing Co-adaptation of Feature Detectors (Dropout) | 2012 | ⬜ | [📄 pdf](papers/hinton2012improving/hinton2012improving_paper.pdf) |
| 43 | Sequence to Sequence Learning with Neural Networks | 2014 | ⬜ | [📄 pdf](papers/sutskever2014sequence/sutskever2014sequence_paper.pdf) |
| 44 | Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG) | 2014 | ⬜ | [📄 pdf](papers/simonyan2014very/simonyan2014very_paper.pdf) |
| 45 | Going Deeper with Convolutions (GoogLeNet / Inception v1) | 2015 | ⬜ | [📄 pdf](papers/szegedy2015going/szegedy2015going_paper.pdf) |
| 46 | U-Net: Convolutional Networks for Biomedical Image Segmentation | 2015 | ⬜ | [📄 pdf](papers/ronneberger2015unet/ronneberger2015unet_paper.pdf) |
| 47 | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | 2015 | ⬜ | [📄 pdf](papers/ren2017faster/ren2017faster_paper.pdf) |
| 48 | You Only Look Once: Unified, Real-Time Object Detection (YOLO) | 2016 | ⬜ | [📄 pdf](papers/redmon2016you/redmon2016you_paper.pdf) |
| 49 | Trust Region Policy Optimization (TRPO) | 2015 | ⬜ | [📄 pdf](papers/schulman2015trust/schulman2015trust_paper.pdf) |
| 50 | A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go through Self-Play (AlphaZero) | 2018 | ⬜ | [📄 pdf](papers/silver2018general/silver2018general_paper.pdf) |
| 51 | Improved Protein Structure Prediction Using Potentials from Deep Learning (AlphaFold 1) | 2020 | ⬜ | [📄 pdf](papers/senior2020improved/senior2020improved_paper.pdf) |
| 52 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) | 2019 | ⬜ | [📄 pdf](papers/raffel2019exploring/raffel2019exploring_paper.pdf) |
| 53 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | 2021 | ⬜ | [📄 pdf](papers/fedus2021switch/fedus2021switch_paper.pdf) |
| 54 | LoRA: Low-Rank Adaptation of Large Language Models | 2021 | ⬜ | [📄 pdf](papers/hu2021lora/hu2021lora_paper.pdf) |
| 55 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 2022 | ⬜ | [📄 pdf](papers/wei2022chainofthought/wei2022chainofthought_paper.pdf) |
| 56 | Emergent Abilities of Large Language Models | 2022 | ⬜ | [📄 pdf](papers/wei2022emergent/wei2022emergent_paper.pdf) |
| 57 | Training Compute-Optimal Large Language Models (Chinchilla) | 2022 | ⬜ | [📄 pdf](papers/hoffmann2022training/hoffmann2022training_paper.pdf) |
| 58 | PaLM: Scaling Language Modeling with Pathways | 2022 | ⬜ | [📄 pdf](papers/chowdhery2022palm/chowdhery2022palm_paper.pdf) |
| 59 | ReAct: Synergizing Reasoning and Acting in Language Models | 2022 | ⬜ | [📄 pdf](papers/yao2022react/yao2022react_paper.pdf) |
| 60 | High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion) | 2022 | ⬜ | [📄 pdf](papers/rombach2022highresolution/rombach2022highresolution_paper.pdf) |
| 61 | GPT-4 Technical Report | 2023 | ⬜ | [📄 pdf](papers/achiam2023gpt4/achiam2023gpt4_paper.pdf) |
| 62 | LLaMA: Open and Efficient Foundation Language Models | 2023 | ⬜ | [📄 pdf](papers/touvron2023llama/touvron2023llama_paper.pdf) |
| 63 | Toolformer: Language Models Can Teach Themselves to Use Tools | 2023 | ⬜ | [📄 pdf](papers/schick2023toolformer/schick2023toolformer_paper.pdf) |
| 64 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | 2023 | ⬜ | [📄 pdf](papers/yao2023tree/yao2023tree_paper.pdf) |
| 65 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO) | 2023 | ⬜ | [📄 pdf](papers/rafailov2023direct/rafailov2023direct_paper.pdf) |
| 66 | Mistral 7B | 2023 | ⬜ | [📄 pdf](papers/jiang2023mistral/jiang2023mistral_paper.pdf) |
| 67 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2023 | ⬜ | [📄 pdf](papers/gu2023mamba/gu2023mamba_paper.pdf) |
| 68 | Mixtral of Experts | 2024 | ⬜ | [📄 pdf](papers/jiang2024mixtral/jiang2024mixtral_paper.pdf) |
| 69 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2025 | ⬜ | [📄 pdf](papers/deepseek2025deepseekr1/deepseek2025deepseekr1_paper.pdf) |

<a id="solar-physics"></a>

### Solar Physics / 태양물리학 (53 / 60)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | Istoria e Dimostrazioni Intorno Alle Macchie Solari (Letters on Sunspots) | 1613 | ✅ | [📝 notes](papers/galilei1613istoria/galilei1613istoria_notes.md) · [💻 code](papers/galilei1613istoria/galilei1613istoria_implementation.ipynb) · [📄 pdf](papers/galilei1613istoria/galilei1613istoria_paper.pdf) |
| 2 | Bestimmung des Brechungs- und Farbenzerstreuungs-Vermogens verschiedener Glasarten (Determination of the Refractive and Dispersive Power of Different Kinds of Glass) | 1814 | ✅ | [📝 notes](papers/vonfraunhofer1814bestimmung/vonfraunhofer1814bestimmung_notes.md) · [💻 code](papers/vonfraunhofer1814bestimmung/vonfraunhofer1814bestimmung_implementation.ipynb) · [📄 pdf](papers/vonfraunhofer1814bestimmung/vonfraunhofer1814bestimmung_paper.pdf) |
| 3 | Chemische Analyse durch Spectralbeobachtungen (Chemical Analysis by Spectral Observations) | 1860 | ✅ | [📝 notes](papers/kirchhoff1860chemische/kirchhoff1860chemische_notes.md) · [💻 code](papers/kirchhoff1860chemische/kirchhoff1860chemische_implementation.ipynb) · [📄 pdf](papers/kirchhoff1860chemische/kirchhoff1860chemische_paper.pdf) |
| 4 | Uber das Gleichgewicht der Sonnenatmosphare (On the Equilibrium of the Solar Atmosphere) | 1906 | ✅ | [📝 notes](papers/schwarzschild1906gleichgewicht/schwarzschild1906gleichgewicht_notes.md) · [💻 code](papers/schwarzschild1906gleichgewicht/schwarzschild1906gleichgewicht_implementation.ipynb) · [📄 pdf](papers/schwarzschild1906gleichgewicht/schwarzschild1906gleichgewicht_paper.pdf) |
| 5 | On the Probable Existence of a Magnetic Field in Sun-Spots | 1908 | ✅ | [📝 notes](papers/hale1908probable/hale1908probable_notes.md) · [💻 code](papers/hale1908probable/hale1908probable_implementation.ipynb) · [📄 pdf](papers/hale1908probable/hale1908probable_paper.pdf) |
| 6 | Radial Movement in Sun-Spots | 1909 | ✅ | [📝 notes](papers/evershed1909radial/evershed1909radial_notes.md) · [💻 code](papers/evershed1909radial/evershed1909radial_implementation.ipynb) · [📄 pdf](papers/evershed1909radial/evershed1909radial_paper.pdf) |
| 7 | The Law of Sun-Spot Polarity | 1925 | ✅ | [📝 notes](papers/hale1925law/hale1925law_notes.md) · [💻 code](papers/hale1925law/hale1925law_implementation.ipynb) · [📄 pdf](papers/hale1925law/hale1925law_paper.pdf) |
| 8 | Existence of Electromagnetic-Hydrodynamic Waves | 1942 | ✅ | [📝 notes](papers/alfven1942existence/alfven1942existence_notes.md) · [💻 code](papers/alfven1942existence/alfven1942existence_implementation.ipynb) · [📄 pdf](papers/alfven1942existence/alfven1942existence_paper.pdf) |
| 9 | The Topology of the Sun's Magnetic Field and the 22-Year Cycle | 1961 | ✅ | [📝 notes](papers/babcock1961topology/babcock1961topology_notes.md) · [💻 code](papers/babcock1961topology/babcock1961topology_implementation.ipynb) · [📄 pdf](papers/babcock1961topology/babcock1961topology_paper.pdf) |
| 10 | A Magneto-Kinematic Model of the Solar Cycle | 1969 | ✅ | [📝 notes](papers/leighton1969magnetokinematic/leighton1969magnetokinematic_notes.md) · [💻 code](papers/leighton1969magnetokinematic/leighton1969magnetokinematic_implementation.ipynb) · [📄 pdf](papers/leighton1969magnetokinematic/leighton1969magnetokinematic_paper.pdf) |
| 11 | Biermann 1951 (Comet Tails) → migrated |  | ↗️ migrated | [📝 notes](papers/biermann1951kometenschweife/biermann1951kometenschweife_notes.md) · [💻 code](papers/biermann1951kometenschweife/biermann1951kometenschweife_implementation.ipynb) · [📄 pdf](papers/biermann1951kometenschweife/biermann1951kometenschweife_paper.pdf) |
| 12 | Parker 1958 (Solar Wind Theory) → migrated |  | ↗️ migrated | [📝 notes](papers/parker1958dynamics/parker1958dynamics_notes.md) · [💻 code](papers/parker1958dynamics/parker1958dynamics_implementation.ipynb) · [📄 pdf](papers/parker1958dynamics/parker1958dynamics_paper.pdf) |
| 13 | Leighton+ 1962 (5-minute oscillation) → migrated |  | ↗️ migrated | [📝 notes](papers/leighton1962velocity/leighton1962velocity_notes.md) · [💻 code](papers/leighton1962velocity/leighton1962velocity_implementation.ipynb) · [📄 pdf](papers/leighton1962velocity/leighton1962velocity_paper.pdf) |
| 14 | Ulrich 1970 (p-mode interpretation) → migrated |  | ↗️ migrated | [📝 notes](papers/ulrich1970fiveminute/ulrich1970fiveminute_notes.md) · [💻 code](papers/ulrich1970fiveminute/ulrich1970fiveminute_implementation.ipynb) · [📄 pdf](papers/ulrich1970fiveminute/ulrich1970fiveminute_paper.pdf) |
| 15 | Deubner 1975 (k-ω diagram) → migrated |  | ↗️ migrated | [📝 notes](papers/deubner1975observations/deubner1975observations_notes.md) · [💻 code](papers/deubner1975observations/deubner1975observations_implementation.ipynb) · [📄 pdf](papers/deubner1975observations/deubner1975observations_paper.pdf) |
| 16 | Christensen-Dalsgaard+ 1996 (Solar Modeling) → migrated |  | ↗️ migrated | [📝 notes](papers/christensendalsgaard1996current/christensendalsgaard1996current_notes.md) · [💻 code](papers/christensendalsgaard1996current/christensendalsgaard1996current_implementation.ipynb) · [📄 pdf](papers/christensendalsgaard1996current/christensendalsgaard1996current_paper.pdf) |
| 17 | Schou+ 1998 (Differential Rotation, MDI) → migrated |  | ↗️ migrated | [📝 notes](papers/schou1998helioseismic/schou1998helioseismic_notes.md) · [💻 code](papers/schou1998helioseismic/schou1998helioseismic_implementation.ipynb) · [📄 pdf](papers/schou1998helioseismic/schou1998helioseismic_paper.pdf) |
| 18 | Giovanelli 1946 (Chromospheric Flare Theory) → migrated |  | ↗️ migrated | [📝 notes](papers/giovanelli1946theory/giovanelli1946theory_notes.md) · [💻 code](papers/giovanelli1946theory/giovanelli1946theory_implementation.ipynb) · [📄 pdf](papers/giovanelli1946theory/giovanelli1946theory_paper.pdf) |
| 19 | Parker 1957 (Sweet-Parker Reconnection) → migrated |  | ↗️ migrated | [📝 notes](papers/parker1957sweets/parker1957sweets_notes.md) · [💻 code](papers/parker1957sweets/parker1957sweets_implementation.ipynb) · [📄 pdf](papers/parker1957sweets/parker1957sweets_paper.pdf) |
| 20 | Petschek 1964 (Fast Reconnection) → migrated |  | ↗️ migrated | [📝 notes](papers/petschek1964magnetic/petschek1964magnetic_notes.md) · [💻 code](papers/petschek1964magnetic/petschek1964magnetic_implementation.ipynb) · [📄 pdf](papers/petschek1964magnetic/petschek1964magnetic_paper.pdf) |
| 21 | Parker 1988 (Nanoflares) → migrated |  | ↗️ migrated | [📝 notes](papers/parker1988nanoflares/parker1988nanoflares_notes.md) · [💻 code](papers/parker1988nanoflares/parker1988nanoflares_implementation.ipynb) · [📄 pdf](papers/parker1988nanoflares/parker1988nanoflares_paper.pdf) |
| 22 | Gosling 1993 (Solar Flare Myth) → migrated |  | ↗️ migrated | [📝 notes](papers/gosling1993solar/gosling1993solar_notes.md) · [💻 code](papers/gosling1993solar/gosling1993solar_implementation.ipynb) · [📄 pdf](papers/gosling1993solar/gosling1993solar_paper.pdf) |
| 23 | Aulanier & Démoulin 1998 (Prominence Flux Rope) → migrated |  | ↗️ migrated | [📝 notes](papers/aulanier1998magnetic/aulanier1998magnetic_notes.md) · [💻 code](papers/aulanier1998magnetic/aulanier1998magnetic_implementation.ipynb) · [📄 pdf](papers/aulanier1998magnetic/aulanier1998magnetic_paper.pdf) |
| 24 | TRACE Observation of Damped Coronal Loop Oscillations: Implications for Coronal Heating | 1999 | ✅ | [📝 notes](papers/nakariakov1999trace/nakariakov1999trace_notes.md) · [💻 code](papers/nakariakov1999trace/nakariakov1999trace_implementation.ipynb) · [📄 pdf](papers/nakariakov1999trace/nakariakov1999trace_paper.pdf) |
| 25 | Antiochos+ 1999 (CME Breakout Model) → migrated |  | ↗️ migrated | [📝 notes](papers/antiochos1999model/antiochos1999model_notes.md) · [💻 code](papers/antiochos1999model/antiochos1999model_implementation.ipynb) · [📄 pdf](papers/antiochos1999model/antiochos1999model_paper.pdf) |
| 26 | Kliem & Török 2006 (Torus Instability) → migrated |  | ↗️ migrated | [📝 notes](papers/kliem2006torus/kliem2006torus_notes.md) · [💻 code](papers/kliem2006torus/kliem2006torus_implementation.ipynb) · [📄 pdf](papers/kliem2006torus/kliem2006torus_paper.pdf) |
| 27 | Reames 1999 (SEP two-class paradigm) → migrated |  | ↗️ migrated | [📝 notes](papers/reames1999particle/reames1999particle_notes.md) · [💻 code](papers/reames1999particle/reames1999particle_implementation.ipynb) · [📄 pdf](papers/reames1999particle/reames1999particle_paper.pdf) |
| 28 | Solar and Stellar Magnetic Activity (monograph) | 2000 | ✅ | [📝 notes](papers/schrijver2000solar/schrijver2000solar_notes.md) · [💻 code](papers/schrijver2000solar/schrijver2000solar_implementation.ipynb) |
| 29 | Kaiser+ 2008 (STEREO Mission) → see canonical |  | ↗️ migrated | [📝 notes](papers/kaiser2008stereo/kaiser2008stereo_notes.md) · [💻 code](papers/kaiser2008stereo/kaiser2008stereo_implementation.ipynb) · [📄 pdf](papers/kaiser2008stereo/kaiser2008stereo_paper.pdf) |
| 30 | Small-Scale Solar Magnetic Fields | 2009 | ✅ | [📝 notes](papers/dewijn2009smallscale/dewijn2009smallscale_notes.md) · [💻 code](papers/dewijn2009smallscale/dewijn2009smallscale_implementation.ipynb) · [📄 pdf](papers/dewijn2009smallscale/dewijn2009smallscale_paper.pdf) |
| 31 | Charbonneau 2010 (Dynamo Models, LRSP) → see canonical |  | ↗️ migrated | [📝 notes](papers/charbonneau2010dynamo/charbonneau2010dynamo_notes.md) · [💻 code](papers/charbonneau2010dynamo/charbonneau2010dynamo_implementation.ipynb) · [📄 pdf](papers/charbonneau2010dynamo/charbonneau2010dynamo_paper.pdf) |
| 32 | A New, Lower Value of Total Solar Irradiance: Evidence and Climate Significance | 2011 | ✅ | [📝 notes](papers/kopp2011lower/kopp2011lower_notes.md) · [💻 code](papers/kopp2011lower/kopp2011lower_implementation.ipynb) · [📄 pdf](papers/kopp2011lower/kopp2011lower_paper.pdf) |
| 33 | Pesnell+ 2012 (SDO Mission) → see canonical |  | ↗️ migrated | [📝 notes](papers/pesnell2012solar/pesnell2012solar_notes.md) · [💻 code](papers/pesnell2012solar/pesnell2012solar_implementation.ipynb) · [📄 pdf](papers/pesnell2012solar/pesnell2012solar_paper.pdf) |
| 34 | Emslie+ 2012 (Flare Global Energetics) → migrated |  | ↗️ migrated | [📝 notes](papers/emslie2012global/emslie2012global_notes.md) · [💻 code](papers/emslie2012global/emslie2012global_implementation.ipynb) · [📄 pdf](papers/emslie2012global/emslie2012global_paper.pdf) |
| 35 | Wiegelmann & Sakurai 2012 (NLFFF, LRSP) → see canonical |  | ↗️ migrated | [📝 notes](papers/wiegelmann2012solar/wiegelmann2012solar_notes.md) · [💻 code](papers/wiegelmann2012solar/wiegelmann2012solar_implementation.ipynb) · [📄 pdf](papers/wiegelmann2012solar/wiegelmann2012solar_paper.pdf) |
| 36 | Woods+ 2012 (SDO/EVE) → see canonical |  | ↗️ migrated | [📝 notes](papers/woods2012extreme/woods2012extreme_notes.md) · [💻 code](papers/woods2012extreme/woods2012extreme_implementation.ipynb) · [📄 pdf](papers/woods2012extreme/woods2012extreme_paper.pdf) |
| 37 | Fox+ 2016 (Parker Solar Probe Mission) → see canonical |  | ↗️ migrated | [📝 notes](papers/fox2016solar/fox2016solar_notes.md) · [💻 code](papers/fox2016solar/fox2016solar_implementation.ipynb) · [📄 pdf](papers/fox2016solar/fox2016solar_paper.pdf) |
| 38 | Kasper+ 2021 (PSP Crosses Alfvén Surface) → migrated |  | ↗️ migrated | [📝 notes](papers/kasper2021parker/kasper2021parker_notes.md) · [💻 code](papers/kasper2021parker/kasper2021parker_implementation.ipynb) · [📄 pdf](papers/kasper2021parker/kasper2021parker_paper.pdf) |
| 39 | Evidence for the Solar Cycle in the Sunspot and Aurora Records of Goryer Dynasty | 1998 | ✅ | [📝 notes](papers/yang1998evidence/yang1998evidence_notes.md) · [💻 code](papers/yang1998evidence/yang1998evidence_implementation.ipynb) · [📄 pdf](papers/yang1998evidence/yang1998evidence_paper.pdf) |
| 40 | The Sunspot and Auroral Activity Cycle Derived from Korean Historical Records of the 11th-18th Century | 2004 | ✅ | [📝 notes](papers/lee2004sunspot/lee2004sunspot_notes.md) · [💻 code](papers/lee2004sunspot/lee2004sunspot_implementation.ipynb) · [📄 pdf](papers/lee2004sunspot/lee2004sunspot_paper.pdf) |
| 41 | Features of great minima in solar activity during the last 1000 years | 2007 | ✅ | [📝 notes](papers/lee2007features/lee2007features_notes.md) · [💻 code](papers/lee2007features/lee2007features_implementation.ipynb) · [📄 pdf](papers/lee2007features/lee2007features_paper.pdf) |
| 42 | Solar activities and climate change during the last millennium recorded in Korean chronicles | 2019 | ✅ | [📝 notes](papers/yang2019solar/yang2019solar_notes.md) · [💻 code](papers/yang2019solar/yang2019solar_implementation.ipynb) · [📄 pdf](papers/yang2019solar/yang2019solar_paper.pdf) |
| 43 | A Signature of Cosmic-Ray Increase in AD 774-775 from Tree Rings in Japan | 2012 | ✅ | [📝 notes](papers/miyake2012signature/miyake2012signature_notes.md) · [💻 code](papers/miyake2012signature/miyake2012signature_implementation.ipynb) · [📄 pdf](papers/miyake2012signature/miyake2012signature_paper.pdf) |
| 44 | The Maunder Minimum: The Reign of Louis XIV Appears to Have Been a Time of Real Anomaly in the Behavior of the Sun | 1976 | ⬜ | [📄 pdf](papers/eddy1976maunder/eddy1976maunder_paper.pdf) |
| 45 | Does a Nonmagnetic Solar Chromosphere Exist? | 1995 | ⬜ | [📄 pdf](papers/carlsson1995does/carlsson1995does_paper.pdf) |
| 46 | Time Variability of the "Quiet" Sun Observed with TRACE. II. Physical Parameters, Temperature Evolution, and Energetics of EUV Nanoflares | 2000 | ⬜ | [📄 pdf](papers/aschwanden2000time/aschwanden2000time_paper.pdf) |
| 47 | Bahcall+ 2001 (Standard Solar Model) → migrated |  | ↗️ migrated | [📄 pdf](papers/bahcall2001/bahcall2001_paper.pdf) |
| 48 | Sunspots: An Overview | 2003 | ⬜ | [📄 pdf](papers/solanki2003sunspots/solanki2003sunspots_paper.pdf) |
| 49 | Allred+ 2005 (RADYN Flare RHD) → migrated |  | ↗️ migrated | [📄 pdf](papers/allred2005/allred2005_paper.pdf) |
| 50 | On Solving the Coronal Heating Problem | 2006 | ⬜ | [📄 pdf](papers/klimchuk2006solving/klimchuk2006solving_paper.pdf) |
| 51 | De Pontieu+ 2007 (Chromospheric Alfvén Waves) → migrated |  | ↗️ migrated | [📄 pdf](papers/depontieu2007pontieu/depontieu2007pontieu_paper.pdf) |
| 52 | Evidence for Alfvén Waves in Solar X-Ray Jets | 2007 | ⬜ | [📄 pdf](papers/cirtain2007evidence/cirtain2007evidence_paper.pdf) |
| 53 | Kosugi+ 2007 (Hinode Mission) → see canonical |  | ↗️ migrated | [📝 notes](papers/kosugi2007hinode/kosugi2007hinode_notes.md) · [💻 code](papers/kosugi2007hinode/kosugi2007hinode_implementation.ipynb) · [📄 pdf](papers/kosugi2007hinode/kosugi2007hinode_paper.pdf) |
| 54 | Asplund+ 2009 (Solar Chemical Composition) → migrated |  | ↗️ migrated | [📄 pdf](papers/asplund2009/asplund2009_paper.pdf) |
| 55 | Rempel+ 2009 (3D Sunspot Simulation) → migrated |  | ↗️ migrated | [📄 pdf](papers/rempel2009/rempel2009_paper.pdf) |
| 56 | Hanasoge+ 2012 (Anomalously Weak Convection) → migrated |  | ↗️ migrated | [📄 pdf](papers/hanasoge2012/hanasoge2012_paper.pdf) |
| 57 | De Pontieu+ 2014 (IRIS Mission) → see canonical |  | ↗️ migrated | [📝 notes](papers/depontieu2014interface/depontieu2014interface_notes.md) · [💻 code](papers/depontieu2014interface/depontieu2014interface_implementation.ipynb) · [📄 pdf](papers/depontieu2014interface/depontieu2014interface_paper.pdf) |
| 58 | Müller+ 2020 (Solar Orbiter Mission) → see canonical |  | ↗️ migrated | [📝 notes](papers/muller2020solar/muller2020solar_notes.md) · [💻 code](papers/muller2020solar/muller2020solar_implementation.ipynb) · [📄 pdf](papers/muller2020solar/muller2020solar_paper.pdf) |
| 59 | Rimmele+ 2020 (DKIST Observatory) → see canonical |  | ↗️ migrated | [📝 notes](papers/rimmele2020daniel/rimmele2020daniel_notes.md) · [💻 code](papers/rimmele2020daniel/rimmele2020daniel_implementation.ipynb) · [📄 pdf](papers/rimmele2020daniel/rimmele2020daniel_paper.pdf) |
| 60 | Extreme-UV Quiet Sun Brightenings Observed by the Solar Orbiter/EUI ("Campfires") | 2021 | ⬜ | [📄 pdf](papers/berghmans2021extremeuv/berghmans2021extremeuv_paper.pdf) |

<a id="space-weather"></a>

### Space Weather / 우주기상 (81 / 81)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | The Norwegian Aurora Polaris Expedition 1902–1903 | 1908 | ✅ | [📝 notes](papers/birkeland1908norwegian/birkeland1908norwegian_notes.md) · [💻 code](papers/birkeland1908norwegian/birkeland1908norwegian_implementation.ipynb) · [📄 pdf](papers/birkeland1908norwegian/birkeland1908norwegian_paper.pdf) |
| 2 | A New Theory of Magnetic Storms, Part I: The Initial Phase | 1931 | ✅ | [📝 notes](papers/chapman1931theory/chapman1931theory_notes.md) · [💻 code](papers/chapman1931theory/chapman1931theory_implementation.ipynb) · [📄 pdf](papers/chapman1931theory/chapman1931theory_paper.pdf) |
| 3 | Geomagnetism (monograph) | 1940 | ✅ | [📝 notes](papers/chapman1940geomagnetism/chapman1940geomagnetism_notes.md) · [💻 code](papers/chapman1940geomagnetism/chapman1940geomagnetism_implementation.ipynb) · [📄 pdf](papers/chapman1940geomagnetism/chapman1940geomagnetism_paper.pdf) |
| 4 | Dynamics of the Interplanetary Gas and Magnetic Fields | 1958 | ✅ | [📝 notes](papers/parker1958dynamics/parker1958dynamics_notes.md) · [💻 code](papers/parker1958dynamics/parker1958dynamics_implementation.ipynb) · [📄 pdf](papers/parker1958dynamics/parker1958dynamics_paper.pdf) |
| 5 | Observation of High Intensity Radiation by Satellites 1958 Alpha and Gamma | 1958 | ✅ | [📝 notes](papers/vanallen1958observation/vanallen1958observation_notes.md) · [💻 code](papers/vanallen1958observation/vanallen1958observation_implementation.ipynb) · [📄 pdf](papers/vanallen1958observation/vanallen1958observation_paper.pdf) |
| 6 | Interplanetary Magnetic Field and the Auroral Zones | 1961 | ✅ | [📝 notes](papers/dungey1961interplanetary/dungey1961interplanetary_notes.md) · [💻 code](papers/dungey1961interplanetary/dungey1961interplanetary_implementation.ipynb) · [📄 pdf](papers/dungey1961interplanetary/dungey1961interplanetary_paper.pdf) |
| 7 | A Unifying Theory of High-Latitude Geophysical Phenomena and Geomagnetic Storms | 1961 | ✅ | [📝 notes](papers/axford1961unifying/axford1961unifying_notes.md) · [💻 code](papers/axford1961unifying/axford1961unifying_implementation.ipynb) · [📄 pdf](papers/axford1961unifying/axford1961unifying_paper.pdf) |
| 8 | The Development of the Auroral Substorm | 1964 | ✅ | [📝 notes](papers/akasofu1964development/akasofu1964development_notes.md) · [💻 code](papers/akasofu1964development/akasofu1964development_implementation.ipynb) · [📄 pdf](papers/akasofu1964development/akasofu1964development_paper.pdf) |
| 9 | The Earth's Magnetic Tail | 1965 | ✅ | [📝 notes](papers/ness1965earths/ness1965earths_notes.md) · [💻 code](papers/ness1965earths/ness1965earths_implementation.ipynb) · [📄 pdf](papers/ness1965earths/ness1965earths_paper.pdf) |
| 10 | Satellite Studies of Magnetospheric Substorms on August 15, 1968 | 1973 | ✅ | [📝 notes](papers/mcpherron1973satellite/mcpherron1973satellite_notes.md) · [💻 code](papers/mcpherron1973satellite/mcpherron1973satellite_implementation.ipynb) · [📄 pdf](papers/mcpherron1973satellite/mcpherron1973satellite_paper.pdf) |
| 11 | An Empirical Relationship between Interplanetary Conditions and Dst | 1975 | ✅ | [📝 notes](papers/burton1975empirical/burton1975empirical_notes.md) · [💻 code](papers/burton1975empirical/burton1975empirical_implementation.ipynb) · [📄 pdf](papers/burton1975empirical/burton1975empirical_paper.pdf) |
| 12 | The Causes of Convection in the Earth's Magnetosphere: A Review of Developments During the IMS | 1982 | ✅ | [📝 notes](papers/cowley1982causes/cowley1982causes_notes.md) · [💻 code](papers/cowley1982causes/cowley1982causes_implementation.ipynb) · [📄 pdf](papers/cowley1982causes/cowley1982causes_paper.pdf) |
| 13 | Mapping Electrodynamic Features of the High-Latitude Ionosphere from Localized Observations (AMIE) | 1988 | ✅ | [📝 notes](papers/richmond1988mapping/richmond1988mapping_notes.md) · [💻 code](papers/richmond1988mapping/richmond1988mapping_implementation.ipynb) · [📄 pdf](papers/richmond1988mapping/richmond1988mapping_paper.pdf) |
| 14 | A Magnetospheric Magnetic Field Model with a Warped Tail Current Sheet | 1989 | ✅ | [📝 notes](papers/tsyganenko1989magnetospheric/tsyganenko1989magnetospheric_notes.md) · [💻 code](papers/tsyganenko1989magnetospheric/tsyganenko1989magnetospheric_implementation.ipynb) · [📄 pdf](papers/tsyganenko1989magnetospheric/tsyganenko1989magnetospheric_paper.pdf) |
| 15 | What Is a Geomagnetic Storm? | 1994 | ✅ | [📝 notes](papers/gonzalez1994what/gonzalez1994what_notes.md) · [💻 code](papers/gonzalez1994what/gonzalez1994what_implementation.ipynb) · [📄 pdf](papers/gonzalez1994what/gonzalez1994what_paper.pdf) |
| 16 | Response of the Thermosphere and Ionosphere to Geomagnetic Storms | 1994 | ✅ | [📝 notes](papers/fullerrowell1994response/fullerrowell1994response_notes.md) · [💻 code](papers/fullerrowell1994response/fullerrowell1994response_implementation.ipynb) · [📄 pdf](papers/fullerrowell1994response/fullerrowell1994response_paper.pdf) |
| 17 | Effects of the March 1989 Solar Activity | 1989 | ✅ | [📝 notes](papers/allen1989effects/allen1989effects_notes.md) · [📄 pdf](papers/allen1989effects/allen1989effects_paper.pdf) |
| 18 | Improvement in the Prediction of Solar Wind Conditions Using Near-Real Time Solar Magnetic Field Updates (WSA) | 2000 | ✅ | [📝 notes](papers/arge2000improvement/arge2000improvement_notes.md) · [💻 code](papers/arge2000improvement/arge2000improvement_implementation.ipynb) · [📄 pdf](papers/arge2000improvement/arge2000improvement_paper.pdf) |
| 19 | Effects of the Sun on the Earth's Environment and the Consequences for Mankind | 2000 | ✅ | [📝 notes](papers/baker2000effects/baker2000effects_notes.md) · [💻 code](papers/baker2000effects/baker2000effects_implementation.ipynb) · [📄 pdf](papers/baker2000effects/baker2000effects_paper.pdf) |
| 20 | GPS and Ionospheric Scintillations | 2007 | ✅ | [📝 notes](papers/kintner2007gps/kintner2007gps_notes.md) · [💻 code](papers/kintner2007gps/kintner2007gps_implementation.ipynb) · [📄 pdf](papers/kintner2007gps/kintner2007gps_paper.pdf) |
| 21 | Modeling 3-D Solar Wind Structure | 2003 | ✅ | [📝 notes](papers/odstrcil2003modeling/odstrcil2003modeling_notes.md) · [💻 code](papers/odstrcil2003modeling/odstrcil2003modeling_implementation.ipynb) · [📄 pdf](papers/odstrcil2003modeling/odstrcil2003modeling_paper.pdf) |
| 22 | The Global Dayside Ionospheric Uplift and Enhancement Associated with Interplanetary Electric Fields | 2004 | ✅ | [📝 notes](papers/tsurutani2004global/tsurutani2004global_notes.md) · [💻 code](papers/tsurutani2004global/tsurutani2004global_implementation.ipynb) · [📄 pdf](papers/tsurutani2004global/tsurutani2004global_paper.pdf) |
| 23 | Introduction to Violent Sun–Earth Connection Events of October–November 2003 | 2005 | ✅ | [📝 notes](papers/gopalswamy2005introduction/gopalswamy2005introduction_notes.md) · [💻 code](papers/gopalswamy2005introduction/gopalswamy2005introduction_implementation.ipynb) · [📄 pdf](papers/gopalswamy2005introduction/gopalswamy2005introduction_paper.pdf) |
| 24 | Space Weather: Terrestrial Perspective | 2007 | ✅ | [📝 notes](papers/pulkkinen2007space/pulkkinen2007space_notes.md) · [💻 code](papers/pulkkinen2007space/pulkkinen2007space_implementation.ipynb) · [📄 pdf](papers/pulkkinen2007space/pulkkinen2007space_paper.pdf) |
| 25 | Geomagnetic Storms and Their Impacts on the U.S. Power Grid | 2010 | ✅ | [📝 notes](papers/kappenman2010geomagnetic/kappenman2010geomagnetic_notes.md) · [💻 code](papers/kappenman2010geomagnetic/kappenman2010geomagnetic_implementation.ipynb) · [📄 pdf](papers/kappenman2010geomagnetic/kappenman2010geomagnetic_paper.pdf) |
| 26 | Radiation Belt Dynamics: The Importance of Wave-Particle Interactions | 2010 | ✅ | [📝 notes](papers/thorne2010radiation/thorne2010radiation_notes.md) · [💻 code](papers/thorne2010radiation/thorne2010radiation_implementation.ipynb) · [📄 pdf](papers/thorne2010radiation/thorne2010radiation_paper.pdf) |
| 27 | The THEMIS Mission | 2008 | ✅ | [📝 notes](papers/angelopoulos2008themis/angelopoulos2008themis_notes.md) · [💻 code](papers/angelopoulos2008themis/angelopoulos2008themis_implementation.ipynb) · [📄 pdf](papers/angelopoulos2008themis/angelopoulos2008themis_paper.pdf) |
| 28 | Science Objectives and Rationale for the Radiation Belt Storm Probes Mission (Van Allen Probes) | 2013 | ✅ | [📝 notes](papers/mauk2013science/mauk2013science_notes.md) · [💻 code](papers/mauk2013science/mauk2013science_implementation.ipynb) · [📄 pdf](papers/mauk2013science/mauk2013science_paper.pdf) |
| 29 | A Major Solar Eruptive Event in July 2012: Defining Extreme Space Weather Scenarios | 2013 | ✅ | [📝 notes](papers/baker2013major/baker2013major_notes.md) · [💻 code](papers/baker2013major/baker2013major_implementation.ipynb) · [📄 pdf](papers/baker2013major/baker2013major_paper.pdf) |
| 30 | An Impenetrable Barrier to Ultrarelativistic Electrons in the Van Allen Radiation Belts | 2014 | ✅ | [📝 notes](papers/baker2014impenetrable/baker2014impenetrable_notes.md) · [💻 code](papers/baker2014impenetrable/baker2014impenetrable_implementation.ipynb) · [📄 pdf](papers/baker2014impenetrable/baker2014impenetrable_paper.pdf) |
| 31 | Space Weather Operations: The NOAA Space Weather Prediction Center | 2015 | ✅ | [📝 notes](papers/biesecker2015space/biesecker2015space_notes.md) · [💻 code](papers/biesecker2015space/biesecker2015space_implementation.ipynb) |
| 32 | Geomagnetically Induced Currents: Science, Engineering, and Applications Readiness | 2017 | ✅ | [📝 notes](papers/pulkkinen2017geomagnetically/pulkkinen2017geomagnetically_notes.md) · [💻 code](papers/pulkkinen2017geomagnetically/pulkkinen2017geomagnetically_implementation.ipynb) · [📄 pdf](papers/pulkkinen2017geomagnetically/pulkkinen2017geomagnetically_paper.pdf) |
| 33 | The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting | 2019 | ✅ | [📝 notes](papers/camporeale2019challenge/camporeale2019challenge_notes.md) · [💻 code](papers/camporeale2019challenge/camporeale2019challenge_implementation.ipynb) · [📄 pdf](papers/camporeale2019challenge/camporeale2019challenge_paper.pdf) |
| 34 | A Comparison of Flare Forecasting Methods | 2019 | ✅ | [📝 notes](papers/leka2019comparison/leka2019comparison_notes.md) · [💻 code](papers/leka2019comparison/leka2019comparison_implementation.ipynb) · [📄 pdf](papers/leka2019comparison/leka2019comparison_paper.pdf) |
| 35 | Solar Wind Prediction Using Deep Learning | 2020 | ✅ | [📝 notes](papers/upendran2020solar/upendran2020solar_notes.md) · [💻 code](papers/upendran2020solar/upendran2020solar_implementation.ipynb) · [📄 pdf](papers/upendran2020solar/upendran2020solar_paper.pdf) |
| 36 | Extreme Space-Weather Events and the Solar Cycle | 2021 | ✅ | [📝 notes](papers/owens2021extreme/owens2021extreme_notes.md) · [💻 code](papers/owens2021extreme/owens2021extreme_implementation.ipynb) · [📄 pdf](papers/owens2021extreme/owens2021extreme_paper.pdf) |
| 37 | Extended Lead-Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning | 2026 | ✅ | [📝 notes](papers/billcliff2026extended/billcliff2026extended_notes.md) · [💻 code](papers/billcliff2026extended/billcliff2026extended_implementation.ipynb) · [📄 pdf](papers/billcliff2026extended/billcliff2026extended_paper.pdf) |
| 38 | Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification | 2024 | ✅ | [📝 notes](papers/abduallah2024prediction/abduallah2024prediction_notes.md) · [💻 code](papers/abduallah2024prediction/abduallah2024prediction_implementation.ipynb) · [📄 pdf](papers/abduallah2024prediction/abduallah2024prediction_paper.pdf) |
| 39 | Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning | 2026 | ✅ | [📝 notes](papers/wang2026daily/wang2026daily_notes.md) · [💻 code](papers/wang2026daily/wang2026daily_implementation.ipynb) · [📄 pdf](papers/wang2026daily/wang2026daily_paper.pdf) |
| 40 | Automatic 3D Reconstruction of Coronal Mass Ejections Based on Dual-viewpoint Observations and Machine Learning | 2025 | ✅ | [📝 notes](papers/lin2025automatic/lin2025automatic_notes.md) · [💻 code](papers/lin2025automatic/lin2025automatic_implementation.ipynb) · [📄 pdf](papers/lin2025automatic/lin2025automatic_paper.pdf) |
| 41 | CMEGNets: A self-supervised framework for coronal mass ejection detection & region segmentation | 2026 | ✅ | [📝 notes](papers/guesmi2026cmegnets/guesmi2026cmegnets_notes.md) · [💻 code](papers/guesmi2026cmegnets/guesmi2026cmegnets_implementation.ipynb) · [📄 pdf](papers/guesmi2026cmegnets/guesmi2026cmegnets_paper.pdf) |
| 42 | The Advanced Composition Explorer (ACE) | 1998 | ✅ | [📝 notes](papers/stone1998advanced/stone1998advanced_notes.md) · [💻 code](papers/stone1998advanced/stone1998advanced_implementation.ipynb) · [📄 pdf](papers/stone1998advanced/stone1998advanced_paper.pdf) |
| 43 | The WIND Spacecraft and Its Early Scientific Results | 1997 | ✅ | [📝 notes](papers/ogilvie1997wind/ogilvie1997wind_notes.md) · [💻 code](papers/ogilvie1997wind/ogilvie1997wind_implementation.ipynb) · [📄 pdf](papers/ogilvie1997wind/ogilvie1997wind_paper.pdf) |
| 44 | Deep Space Climate Observatory: The DSCOVR Mission | 2012 | ✅ | [📝 notes](papers/burt2012deep/burt2012deep_notes.md) · [💻 code](papers/burt2012deep/burt2012deep_implementation.ipynb) · [📄 pdf](papers/burt2012deep/burt2012deep_paper.pdf) |
| 45 | The GOES-R Series: A New Generation of Geostationary Environmental Satellites (Chapter 1) | 2019 | ✅ | [📝 notes](papers/goodman2019goesr/goodman2019goesr_notes.md) · [💻 code](papers/goodman2019goesr/goodman2019goesr_implementation.ipynb) · [📄 pdf](papers/goodman2019goesr/goodman2019goesr_paper.pdf) |
| 46 | Operational Uses of the GOES Energetic Particle Detectors | 1996 | ✅ | [📝 notes](papers/onsager1996operational/onsager1996operational_notes.md) · [💻 code](papers/onsager1996operational/onsager1996operational_implementation.ipynb) · [📄 pdf](papers/onsager1996operational/onsager1996operational_paper.pdf) |
| 47 | Imaging Results from Dynamics Explorer 1 | 1988 | ✅ | [📝 notes](papers/frank1988imaging/frank1988imaging_notes.md) · [💻 code](papers/frank1988imaging/frank1988imaging_implementation.ipynb) · [📄 pdf](papers/frank1988imaging/frank1988imaging_paper.pdf) |
| 48 | An Ultraviolet Auroral Imager for the Viking Spacecraft | 1987 | ✅ | [📝 notes](papers/anger1987ultraviolet/anger1987ultraviolet_notes.md) · [💻 code](papers/anger1987ultraviolet/anger1987ultraviolet_implementation.ipynb) · [📄 pdf](papers/anger1987ultraviolet/anger1987ultraviolet_paper.pdf) |
| 49 | A Far Ultraviolet Imager for the International Solar-Terrestrial Physics Mission (Polar/UVI) | 1995 | ✅ | [📝 notes](papers/torr1995far/torr1995far_notes.md) · [💻 code](papers/torr1995far/torr1995far_implementation.ipynb) · [📄 pdf](papers/torr1995far/torr1995far_paper.pdf) |
| 50 | IMAGE Mission Overview | 2000 | ✅ | [📝 notes](papers/burch2000image/burch2000image_notes.md) · [💻 code](papers/burch2000image/burch2000image_implementation.ipynb) · [📄 pdf](papers/burch2000image/burch2000image_paper.pdf) |
| 51 | Far Ultraviolet Imaging from the IMAGE Spacecraft. 1. System Design | 2000 | ✅ | [📝 notes](papers/mende2000far/mende2000far_notes.md) · [💻 code](papers/mende2000far/mende2000far_implementation.ipynb) · [📄 pdf](papers/mende2000far/mende2000far_paper.pdf) |
| 52 | Substorm Onset Observations by IMAGE-FUV | 2004 | ✅ | [📝 notes](papers/frey2004substorm/frey2004substorm_notes.md) · [💻 code](papers/frey2004substorm/frey2004substorm_implementation.ipynb) · [📄 pdf](papers/frey2004substorm/frey2004substorm_paper.pdf) |
| 53 | GUVI: A Hyperspectral Imager for Geospace (TIMED/GUVI) | 2004 | ✅ | [📝 notes](papers/paxton2004guvi/paxton2004guvi_notes.md) · [💻 code](papers/paxton2004guvi/paxton2004guvi_implementation.ipynb) · [📄 pdf](papers/paxton2004guvi/paxton2004guvi_paper.pdf) |
| 54 | Special Sensor Ultraviolet Spectrographic Imager (SSUSI): An Instrument Description | 1992 | ✅ | [📝 notes](papers/paxton1992special/paxton1992special_notes.md) · [💻 code](papers/paxton1992special/paxton1992special_implementation.ipynb) · [📄 pdf](papers/paxton1992special/paxton1992special_paper.pdf) |
| 55 | The Fast Auroral SnapshoT (FAST) Mission | 1998 | ✅ | [📝 notes](papers/carlson1998fast/carlson1998fast_notes.md) · [💻 code](papers/carlson1998fast/carlson1998fast_implementation.ipynb) · [📄 pdf](papers/carlson1998fast/carlson1998fast_paper.pdf) |
| 56 | The THEMIS Array of Ground-Based Observatories for the Study of Auroral Substorms | 2008 | ✅ | [📝 notes](papers/mende2008themis/mende2008themis_notes.md) · [💻 code](papers/mende2008themis/mende2008themis_implementation.ipynb) · [📄 pdf](papers/mende2008themis/mende2008themis_paper.pdf) |
| 57 | The THEMIS All-Sky Imaging Array — System Design and Initial Results from the Prototype Imager | 2006 | ✅ | [📝 notes](papers/donovan2006themis/donovan2006themis_notes.md) · [💻 code](papers/donovan2006themis/donovan2006themis_implementation.ipynb) · [📄 pdf](papers/donovan2006themis/donovan2006themis_paper.pdf) |
| 58 | DARN/SuperDARN: A Global View of the Dynamics of High-Latitude Convection | 1995 | ✅ | [📝 notes](papers/greenwald1995darn/greenwald1995darn_notes.md) · [💻 code](papers/greenwald1995darn/greenwald1995darn_implementation.ipynb) · [📄 pdf](papers/greenwald1995darn/greenwald1995darn_paper.pdf) |
| 59 | EISCAT: An Updated Description of Technical Characteristics and Operational Capabilities | 1983 | ✅ | [📝 notes](papers/folkestad1983eiscat/folkestad1983eiscat_notes.md) · [💻 code](papers/folkestad1983eiscat/folkestad1983eiscat_implementation.ipynb) · [📄 pdf](papers/folkestad1983eiscat/folkestad1983eiscat_paper.pdf) |
| 60 | First Observations From the TREx Spectrograph: The Optical Spectrum of STEVE and the Picket Fence Phenomena | 2019 | ✅ | [📝 notes](papers/gillies2019first/gillies2019first_notes.md) · [💻 code](papers/gillies2019first/gillies2019first_implementation.ipynb) · [📄 pdf](papers/gillies2019first/gillies2019first_paper.pdf) |
| 61 | The WIND Magnetic Field Investigation (Wind/MFI) | 1995 | ✅ | [📝 notes](papers/lepping1995wind/lepping1995wind_notes.md) · [💻 code](papers/lepping1995wind/lepping1995wind_implementation.ipynb) · [📄 pdf](papers/lepping1995wind/lepping1995wind_paper.pdf) |
| 62 | SWE: A Comprehensive Plasma Instrument for the Wind Spacecraft | 1995 | ✅ | [📝 notes](papers/ogilvie1995swe/ogilvie1995swe_notes.md) · [💻 code](papers/ogilvie1995swe/ogilvie1995swe_implementation.ipynb) · [📄 pdf](papers/ogilvie1995swe/ogilvie1995swe_paper.pdf) |
| 63 | A Three-Dimensional Plasma and Energetic Particle Investigation for the Wind Spacecraft (Wind/3DP) | 1995 | ✅ | [📝 notes](papers/lin1995threedimensional/lin1995threedimensional_notes.md) · [💻 code](papers/lin1995threedimensional/lin1995threedimensional_implementation.ipynb) · [📄 pdf](papers/lin1995threedimensional/lin1995threedimensional_paper.pdf) |
| 64 | WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft | 1995 | ✅ | [📝 notes](papers/bougeret1995waves/bougeret1995waves_notes.md) · [💻 code](papers/bougeret1995waves/bougeret1995waves_implementation.ipynb) · [📄 pdf](papers/bougeret1995waves/bougeret1995waves_paper.pdf) |
| 65 | The ACE Magnetic Fields Experiment (ACE/MAG) | 1998 | ✅ | [📝 notes](papers/smith1998ace/smith1998ace_notes.md) · [💻 code](papers/smith1998ace/smith1998ace_implementation.ipynb) · [📄 pdf](papers/smith1998ace/smith1998ace_paper.pdf) |
| 66 | Solar Wind Electron Proton Alpha Monitor (SWEPAM) for ACE | 1998 | ✅ | [📝 notes](papers/mccomas1998solar/mccomas1998solar_notes.md) · [💻 code](papers/mccomas1998solar/mccomas1998solar_implementation.ipynb) · [📄 pdf](papers/mccomas1998solar/mccomas1998solar_paper.pdf) |
| 67 | The Solar Isotope Spectrometer (ACE/SIS) | 1998 | ✅ | [📝 notes](papers/stone1998solar/stone1998solar_notes.md) · [💻 code](papers/stone1998solar/stone1998solar_implementation.ipynb) · [📄 pdf](papers/stone1998solar/stone1998solar_paper.pdf) |
| 68 | The Cosmic-Ray Isotope Spectrometer (ACE/CRIS) | 1998 | ✅ | [📝 notes](papers/stone1998cosmicray/stone1998cosmicray_notes.md) · [💻 code](papers/stone1998cosmicray/stone1998cosmicray_implementation.ipynb) · [📄 pdf](papers/stone1998cosmicray/stone1998cosmicray_paper.pdf) |
| 69 | The Ultra-Low-Energy Isotope Spectrometer (ACE/ULEIS) | 1998 | ✅ | [📝 notes](papers/mason1998ultralowenergy/mason1998ultralowenergy_notes.md) · [💻 code](papers/mason1998ultralowenergy/mason1998ultralowenergy_implementation.ipynb) · [📄 pdf](papers/mason1998ultralowenergy/mason1998ultralowenergy_paper.pdf) |
| 70 | Electron, Proton, and Alpha Monitor on ACE (ACE/EPAM) | 1998 | ✅ | [📝 notes](papers/gold1998electron/gold1998electron_notes.md) · [💻 code](papers/gold1998electron/gold1998electron_implementation.ipynb) · [📄 pdf](papers/gold1998electron/gold1998electron_paper.pdf) |
| 71 | SWICS and SWIMS on ACE: Solar and Interstellar Composition via Solar Wind and Pickup Ions | 1998 | ✅ | [📝 notes](papers/gloeckler1998swics/gloeckler1998swics_notes.md) · [💻 code](papers/gloeckler1998swics/gloeckler1998swics_implementation.ipynb) · [📄 pdf](papers/gloeckler1998swics/gloeckler1998swics_paper.pdf) |
| 72 | High Energy Neutral Atom Imager (IMAGE/HENA) | 2000 | ✅ | [📝 notes](papers/mitchell2000high/mitchell2000high_notes.md) · [💻 code](papers/mitchell2000high/mitchell2000high_implementation.ipynb) · [📄 pdf](papers/mitchell2000high/mitchell2000high_paper.pdf) |
| 73 | Medium Energy Neutral Atom Imager (IMAGE/MENA) | 2000 | ✅ | [📝 notes](papers/pollock2000medium/pollock2000medium_notes.md) · [💻 code](papers/pollock2000medium/pollock2000medium_implementation.ipynb) · [📄 pdf](papers/pollock2000medium/pollock2000medium_paper.pdf) |
| 74 | The Low-Energy Neutral Atom Imager for IMAGE (LENA) | 2000 | ✅ | [📝 notes](papers/moore2000lowenergy/moore2000lowenergy_notes.md) · [💻 code](papers/moore2000lowenergy/moore2000lowenergy_implementation.ipynb) · [📄 pdf](papers/moore2000lowenergy/moore2000lowenergy_paper.pdf) |
| 75 | The Extreme Ultraviolet Imager Investigation for IMAGE (EUV) | 2000 | ✅ | [📝 notes](papers/sandel2000extreme/sandel2000extreme_notes.md) · [💻 code](papers/sandel2000extreme/sandel2000extreme_implementation.ipynb) · [📄 pdf](papers/sandel2000extreme/sandel2000extreme_paper.pdf) |
| 76 | The Radio Plasma Imager Investigation on IMAGE (RPI) | 2000 | ✅ | [📝 notes](papers/reinisch2000radio/reinisch2000radio_notes.md) · [💻 code](papers/reinisch2000radio/reinisch2000radio_implementation.ipynb) · [📄 pdf](papers/reinisch2000radio/reinisch2000radio_paper.pdf) |
| 77 | The THEMIS Mission (Satellite Suite Overview) | 2008 | ✅ | [📝 notes](papers/angelopoulos2008themis/angelopoulos2008themis_notes.md) · [💻 code](papers/angelopoulos2008themis/angelopoulos2008themis_implementation.ipynb) · [📄 pdf](papers/angelopoulos2008themis/angelopoulos2008themis_paper.pdf) |
| 78 | The THEMIS Fluxgate Magnetometer (THEMIS/FGM) | 2008 | ✅ | [📝 notes](papers/auster2008themis/auster2008themis_notes.md) · [💻 code](papers/auster2008themis/auster2008themis_implementation.ipynb) · [📄 pdf](papers/auster2008themis/auster2008themis_paper.pdf) |
| 79 | The Search Coil Magnetometer for THEMIS (THEMIS/SCM) | 2008 | ✅ | [📝 notes](papers/roux2008search/roux2008search_notes.md) · [💻 code](papers/roux2008search/roux2008search_implementation.ipynb) · [📄 pdf](papers/roux2008search/roux2008search_paper.pdf) |
| 80 | The THEMIS ESA Plasma Instrument and In-flight Calibration (THEMIS/ESA) | 2008 | ✅ | [📝 notes](papers/mcfadden2008themis/mcfadden2008themis_notes.md) · [💻 code](papers/mcfadden2008themis/mcfadden2008themis_implementation.ipynb) · [📄 pdf](papers/mcfadden2008themis/mcfadden2008themis_paper.pdf) |
| 81 | The Electric Field Instrument (EFI) for THEMIS | 2008 | ✅ | [📝 notes](papers/bonnell2008electric/bonnell2008electric_notes.md) · [💻 code](papers/bonnell2008electric/bonnell2008electric_implementation.ipynb) · [📄 pdf](papers/bonnell2008electric/bonnell2008electric_paper.pdf) |

<a id="solar-observation"></a>

### Solar Observation / 태양관측 (80 / 81)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | The McMath Solar Telescope of Kitt Peak National Observatory | 1964 | ✅ | [📝 notes](papers/pierce1964mcmath/pierce1964mcmath_notes.md) · [💻 code](papers/pierce1964mcmath/pierce1964mcmath_implementation.ipynb) · [📄 pdf](papers/pierce1964mcmath/pierce1964mcmath_paper.pdf) |
| 2 | An Evacuated Tower Telescope | 1964 | ✅ | [📝 notes](papers/dunn1964evacuated/dunn1964evacuated_notes.md) · [💻 code](papers/dunn1964evacuated/dunn1964evacuated_implementation.ipynb) · [📄 pdf](papers/dunn1964evacuated/dunn1964evacuated_paper.pdf) |
| 3 | The 1-meter Swedish Solar Telescope | 2003 | ✅ | [📝 notes](papers/scharmer2003meter/scharmer2003meter_notes.md) · [💻 code](papers/scharmer2003meter/scharmer2003meter_implementation.ipynb) · [📄 pdf](papers/scharmer2003meter/scharmer2003meter_paper.pdf) |
| 4 | The 1.6 m Off-Axis New Solar Telescope (NST) in Big Bear | 2012 | ✅ | [📝 notes](papers/goode2012offaxis/goode2012offaxis_notes.md) · [💻 code](papers/goode2012offaxis/goode2012offaxis_implementation.ipynb) · [📄 pdf](papers/goode2012offaxis/goode2012offaxis_paper.pdf) |
| 5 | The Global Oscillation Network Group (GONG) Project | 1996 | ✅ | [📝 notes](papers/harvey1996global/harvey1996global_notes.md) · [💻 code](papers/harvey1996global/harvey1996global_implementation.ipynb) · [📄 pdf](papers/harvey1996global/harvey1996global_paper.pdf) |
| 6 | BiSON Performance | 1996 | ✅ | [📝 notes](papers/chaplin1996bison/chaplin1996bison_notes.md) · [💻 code](papers/chaplin1996bison/chaplin1996bison_implementation.ipynb) · [📄 pdf](papers/chaplin1996bison/chaplin1996bison_paper.pdf) |
| 7 | Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory | 2016 | ✅ | [📝 notes](papers/tomczyk2016scientific/tomczyk2016scientific_notes.md) · [💻 code](papers/tomczyk2016scientific/tomczyk2016scientific_implementation.ipynb) · [📄 pdf](papers/tomczyk2016scientific/tomczyk2016scientific_paper.pdf) |
| 8 | The SOHO Mission: An Overview | 1995 | ✅ | [📝 notes](papers/domingo1995soho/domingo1995soho_notes.md) · [💻 code](papers/domingo1995soho/domingo1995soho_implementation.ipynb) · [📄 pdf](papers/domingo1995soho/domingo1995soho_paper.pdf) |
| 9 | EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission | 1995 | ✅ | [📝 notes](papers/delaboudiniere1995eit/delaboudiniere1995eit_notes.md) · [💻 code](papers/delaboudiniere1995eit/delaboudiniere1995eit_implementation.ipynb) · [📄 pdf](papers/delaboudiniere1995eit/delaboudiniere1995eit_paper.pdf) |
| 10 | The Large Angle Spectroscopic Coronagraph (LASCO) | 1995 | ✅ | [📝 notes](papers/brueckner1995large/brueckner1995large_notes.md) · [💻 code](papers/brueckner1995large/brueckner1995large_implementation.ipynb) · [📄 pdf](papers/brueckner1995large/brueckner1995large_paper.pdf) |
| 11 | The Transition Region and Coronal Explorer | 1999 | ✅ | [📝 notes](papers/handy1999transition/handy1999transition_notes.md) · [💻 code](papers/handy1999transition/handy1999transition_implementation.ipynb) · [📄 pdf](papers/handy1999transition/handy1999transition_paper.pdf) |
| 12 | The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO) | 2012 | ✅ | [📝 notes](papers/lemen2012atmospheric/lemen2012atmospheric_notes.md) · [💻 code](papers/lemen2012atmospheric/lemen2012atmospheric_implementation.ipynb) · [📄 pdf](papers/lemen2012atmospheric/lemen2012atmospheric_paper.pdf) |
| 13 | The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO) | 2012 | ✅ | [📝 notes](papers/scherrer2012helioseismic/scherrer2012helioseismic_notes.md) · [💻 code](papers/scherrer2012helioseismic/scherrer2012helioseismic_implementation.ipynb) · [📄 pdf](papers/scherrer2012helioseismic/scherrer2012helioseismic_paper.pdf) |
| 14 | The Solar Optical Telescope for the Hinode Mission: An Overview | 2008 | ✅ | [📝 notes](papers/tsuneta2008solar/tsuneta2008solar_notes.md) · [💻 code](papers/tsuneta2008solar/tsuneta2008solar_implementation.ipynb) · [📄 pdf](papers/tsuneta2008solar/tsuneta2008solar_paper.pdf) |
| 15 | The EUV Imaging Spectrometer for Hinode | 2007 | ✅ | [📝 notes](papers/culhane2007euv/culhane2007euv_notes.md) · [💻 code](papers/culhane2007euv/culhane2007euv_implementation.ipynb) · [📄 pdf](papers/culhane2007euv/culhane2007euv_paper.pdf) |
| 16 | The Interface Region Imaging Spectrograph (IRIS) | 2014 | ✅ | [📝 notes](papers/depontieu2014interface/depontieu2014interface_notes.md) · [💻 code](papers/depontieu2014interface/depontieu2014interface_implementation.ipynb) · [📄 pdf](papers/depontieu2014interface/depontieu2014interface_paper.pdf) |
| 17 | EUVI: The STEREO-SECCHI Extreme Ultraviolet Imager | 2004 | ✅ | [📝 notes](papers/wuelser2004euvi/wuelser2004euvi_notes.md) · [📄 pdf](papers/wuelser2004euvi/wuelser2004euvi_paper.pdf) |
| 18 | The Solar Orbiter SPICE Instrument — An Extreme UV Imaging Spectrometer | 2020 | ✅ | [📝 notes](papers/anderson2020solar/anderson2020solar_notes.md) · [💻 code](papers/anderson2020solar/anderson2020solar_implementation.ipynb) · [📄 pdf](papers/anderson2020solar/anderson2020solar_paper.pdf) |
| 19 | The Solar Orbiter EUI Instrument: The Extreme Ultraviolet Imager | 2020 | ✅ | [📝 notes](papers/rochus2020solar/rochus2020solar_notes.md) · [💻 code](papers/rochus2020solar/rochus2020solar_implementation.ipynb) · [📄 pdf](papers/rochus2020solar/rochus2020solar_paper.pdf) |
| 20 | Solar Adaptive Optics | 2011 | ✅ | [📝 notes](papers/rimmele2011solar/rimmele2011solar_notes.md) · [💻 code](papers/rimmele2011solar/rimmele2011solar_implementation.ipynb) · [📄 pdf](papers/rimmele2011solar/rimmele2011solar_paper.pdf) |
| 21 | Speckle Interferometry with Adaptive Optics Corrected Solar Data | 2008 | ✅ | [📝 notes](papers/woger2008speckle/woger2008speckle_notes.md) · [💻 code](papers/woger2008speckle/woger2008speckle_implementation.ipynb) · [📄 pdf](papers/woger2008speckle/woger2008speckle_paper.pdf) |
| 22 | CRISP Spectropolarimetric Imaging of Penumbral Fine Structure | 2008 | ✅ | [📝 notes](papers/scharmer2008crisp/scharmer2008crisp_notes.md) · [💻 code](papers/scharmer2008crisp/scharmer2008crisp_implementation.ipynb) · [📄 pdf](papers/scharmer2008crisp/scharmer2008crisp_paper.pdf) |
| 23 | The Daniel K. Inouye Solar Telescope — Observatory Overview | 2020 | ✅ | [📝 notes](papers/rimmele2020daniel/rimmele2020daniel_notes.md) · [💻 code](papers/rimmele2020daniel/rimmele2020daniel_implementation.ipynb) · [📄 pdf](papers/rimmele2020daniel/rimmele2020daniel_paper.pdf) |
| 24 | The Visible Spectro-Polarimeter of the Daniel K. Inouye Solar Telescope | 2022 | ✅ | [📝 notes](papers/dewijn2022visible/dewijn2022visible_notes.md) · [💻 code](papers/dewijn2022visible/dewijn2022visible_implementation.ipynb) · [📄 pdf](papers/dewijn2022visible/dewijn2022visible_paper.pdf) |
| 25 | The Diffraction-Limited Near-Infrared Spectropolarimeter (DL-NIRSP) of the Daniel K. Inouye Solar Telescope (DKIST) | 2022 | ✅ | [📝 notes](papers/jaeggli2022diffractionlimited/jaeggli2022diffractionlimited_notes.md) · [💻 code](papers/jaeggli2022diffractionlimited/jaeggli2022diffractionlimited_implementation.ipynb) · [📄 pdf](papers/jaeggli2022diffractionlimited/jaeggli2022diffractionlimited_paper.pdf) |
| 26 | Radio Emission from Solar Flares | 1998 | ✅ | [📝 notes](papers/bastian1998radio/bastian1998radio_notes.md) · [💻 code](papers/bastian1998radio/bastian1998radio_implementation.ipynb) · [📄 pdf](papers/bastian1998radio/bastian1998radio_paper.pdf) |
| 27 | Solar Science with the Atacama Large Millimeter/Submillimeter Array — A New View of Our Sun | 2016 | ✅ | [📝 notes](papers/wedemeyer2016solar/wedemeyer2016solar_notes.md) · [💻 code](papers/wedemeyer2016solar/wedemeyer2016solar_implementation.ipynb) · [📄 pdf](papers/wedemeyer2016solar/wedemeyer2016solar_paper.pdf) |
| 28 | LOFAR Tied-Array Imaging of Type III Solar Radio Bursts | 2014 | ✅ | [📝 notes](papers/morosan2014lofar/morosan2014lofar_notes.md) · [💻 code](papers/morosan2014lofar/morosan2014lofar_implementation.ipynb) · [📄 pdf](papers/morosan2014lofar/morosan2014lofar_paper.pdf) |
| 29 | The Nobeyama Radioheliograph | 1994 | ✅ | [📝 notes](papers/nakajima1994nobeyama/nakajima1994nobeyama_notes.md) · [💻 code](papers/nakajima1994nobeyama/nakajima1994nobeyama_implementation.ipynb) · [📄 pdf](papers/nakajima1994nobeyama/nakajima1994nobeyama_paper.pdf) |
| 30 | Cheung+ 2015 (Sparse DEM for AIA) → migrated |  | ↗️ migrated | [📝 notes](papers/cheung2015thermal/cheung2015thermal_notes.md) · [💻 code](papers/cheung2015thermal/cheung2015thermal_implementation.ipynb) · [📄 pdf](papers/cheung2015thermal/cheung2015thermal_paper.pdf) |
| 31 | Observables Processing for the Helioseismic and Magnetic Imager Instrument on the Solar Dynamics Observatory | 2016 | ✅ | [📝 notes](papers/couvidat2016observables/couvidat2016observables_notes.md) · [💻 code](papers/couvidat2016observables/couvidat2016observables_implementation.ipynb) · [📄 pdf](papers/couvidat2016observables/couvidat2016observables_paper.pdf) |
| 32 | Multi-Frame Blind Deconvolution with Linear Equality Constraints | 2002 | ✅ | [📝 notes](papers/lofdahl2002multiframe/lofdahl2002multiframe_notes.md) · [💻 code](papers/lofdahl2002multiframe/lofdahl2002multiframe_implementation.ipynb) · [📄 pdf](papers/lofdahl2002multiframe/lofdahl2002multiframe_paper.pdf) |
| 33 | Inversion of Stokes Profiles | 1992 | ✅ | [📝 notes](papers/cobo1992inversion/cobo1992inversion_notes.md) · [💻 code](papers/cobo1992inversion/cobo1992inversion_implementation.ipynb) · [📄 pdf](papers/cobo1992inversion/cobo1992inversion_paper.pdf) |
| 34 | An Instrument to Measure Coronal Emission Line Polarization | 2008 | ✅ | [📝 notes](papers/tomczyk2008instrument/tomczyk2008instrument_notes.md) · [💻 code](papers/tomczyk2008instrument/tomczyk2008instrument_implementation.ipynb) · [📄 pdf](papers/tomczyk2008instrument/tomczyk2008instrument_paper.pdf) |
| 35 | The Solar Dynamics Observatory (SDO) | 2012 | ✅ | [📝 notes](papers/pesnell2012solar/pesnell2012solar_notes.md) · [💻 code](papers/pesnell2012solar/pesnell2012solar_implementation.ipynb) · [📄 pdf](papers/pesnell2012solar/pesnell2012solar_paper.pdf) |
| 36 | The Solar Orbiter mission — Science overview | 2020 | ✅ | [📝 notes](papers/muller2020solar/muller2020solar_notes.md) · [💻 code](papers/muller2020solar/muller2020solar_implementation.ipynb) · [📄 pdf](papers/muller2020solar/muller2020solar_paper.pdf) |
| 37 | Identification of Coronal Holes on AIA/SDO Images Using Unsupervised Machine Learning | 2022 | ✅ | [📝 notes](papers/inceoglu2022identification/inceoglu2022identification_notes.md) · [💻 code](papers/inceoglu2022identification/inceoglu2022identification_implementation.ipynb) · [📄 pdf](papers/inceoglu2022identification/inceoglu2022identification_paper.pdf) |
| 60 | The Chinese H-alpha Solar Explorer (CHASE) mission: An overview | 2022 | ✅ | [📝 notes](papers/li2022chinese/li2022chinese_notes.md) · [💻 code](papers/li2022chinese/li2022chinese_implementation.ipynb) · [📄 pdf](papers/li2022chinese/li2022chinese_paper.pdf) |
| 61 | Combined coronal observations of the streamer belt with Metis and EUI instruments on Solar Orbiter | 2025 | ✅ | [📝 notes](papers/abbo2025combined/abbo2025combined_notes.md) · [💻 code](papers/abbo2025combined/abbo2025combined_implementation.ipynb) · [📄 pdf](papers/abbo2025combined/abbo2025combined_paper.pdf) |
| 62 | The electron density of the solar corona | 1950 | ✅ | [📝 notes](papers/vandehulst1950electron/vandehulst1950electron_notes.md) · [💻 code](papers/vandehulst1950electron/vandehulst1950electron_implementation.ipynb) · [📄 pdf](papers/vandehulst1950electron/vandehulst1950electron_paper.pdf) |
| 63 | Dere+ 1997 (CHIANTI v1) → migrated |  | ↗️ migrated | [📝 notes](papers/dere1997chianti/dere1997chianti_notes.md) · [💻 code](papers/dere1997chianti/dere1997chianti_implementation.ipynb) · [📄 pdf](papers/dere1997chianti/dere1997chianti_paper.pdf) |
| 64 | Dere+ 2023 (CHIANTI v10.1) → migrated |  | ↗️ migrated | [📝 notes](papers/dere2023chianti/dere2023chianti_notes.md) · [💻 code](papers/dere2023chianti/dere2023chianti_implementation.ipynb) · [📄 pdf](papers/dere2023chianti/dere2023chianti_paper.pdf) |
| 65 | Mason & Monsignori-Fossi 1994 (VUV Diagnostics) → migrated |  | ↗️ migrated | [📝 notes](papers/mason1994spectroscopic/mason1994spectroscopic_notes.md) · [💻 code](papers/mason1994spectroscopic/mason1994spectroscopic_implementation.ipynb) · [📄 pdf](papers/mason1994spectroscopic/mason1994spectroscopic_paper.pdf) |
| 81 | The Heliospheric Imagers Onboard the STEREO Mission | 2009 | ⬜ | [📄 pdf](papers/eyles2009heliospheric/eyles2009heliospheric_paper.pdf) |
| 38 | The Soft X-ray Telescope for the Solar-A Mission (Yohkoh/SXT) | 1991 | ✅ | [📝 notes](papers/tsuneta1991soft/tsuneta1991soft_notes.md) · [💻 code](papers/tsuneta1991soft/tsuneta1991soft_implementation.ipynb) · [📄 pdf](papers/tsuneta1991soft/tsuneta1991soft_paper.pdf) |
| 39 | The Reuven Ramaty High-Energy Solar Spectroscopic Imager (RHESSI) | 2002 | ✅ | [📝 notes](papers/lin2002reuven/lin2002reuven_notes.md) · [💻 code](papers/lin2002reuven/lin2002reuven_implementation.ipynb) · [📄 pdf](papers/lin2002reuven/lin2002reuven_paper.pdf) |
| 40 | The Solar Probe Plus Mission: Humanity's First Visit to Our Star (Parker Solar Probe) | 2016 | ✅ | [📝 notes](papers/fox2016solar/fox2016solar_notes.md) · [💻 code](papers/fox2016solar/fox2016solar_implementation.ipynb) · [📄 pdf](papers/fox2016solar/fox2016solar_paper.pdf) |
| 41 | The SOLAR-A Mission: An Overview (Yohkoh) | 1991 | ✅ | [📝 notes](papers/ogawara1991solara/ogawara1991solara_notes.md) · [💻 code](papers/ogawara1991solara/ogawara1991solara_implementation.ipynb) · [📄 pdf](papers/ogawara1991solara/ogawara1991solara_paper.pdf) |
| 42 | The Hard X-ray Telescope (HXT) for the SOLAR-A Mission | 1991 | ✅ | [📝 notes](papers/kosugi1991hard/kosugi1991hard_notes.md) · [💻 code](papers/kosugi1991hard/kosugi1991hard_implementation.ipynb) · [📄 pdf](papers/kosugi1991hard/kosugi1991hard_paper.pdf) |
| 43 | The Solar Oscillations Investigation — Michelson Doppler Imager (SOHO/MDI) | 1995 | ✅ | [📝 notes](papers/scherrer1995solar/scherrer1995solar_notes.md) · [💻 code](papers/scherrer1995solar/scherrer1995solar_implementation.ipynb) · [📄 pdf](papers/scherrer1995solar/scherrer1995solar_paper.pdf) |
| 44 | The Coronal Diagnostic Spectrometer for the Solar and Heliospheric Observatory (SOHO/CDS) | 1995 | ✅ | [📝 notes](papers/harrison1995coronal/harrison1995coronal_notes.md) · [💻 code](papers/harrison1995coronal/harrison1995coronal_implementation.ipynb) · [📄 pdf](papers/harrison1995coronal/harrison1995coronal_paper.pdf) |
| 45 | The Ultraviolet Coronagraph Spectrometer for the Solar and Heliospheric Observatory (SOHO/UVCS) | 1995 | ✅ | [📝 notes](papers/kohl1995ultraviolet/kohl1995ultraviolet_notes.md) · [💻 code](papers/kohl1995ultraviolet/kohl1995ultraviolet_implementation.ipynb) · [📄 pdf](papers/kohl1995ultraviolet/kohl1995ultraviolet_paper.pdf) |
| 46 | SWAN: A Study of Solar Wind Anisotropies on SOHO with Lyman Alpha Sky Mapping | 1995 | ✅ | [📝 notes](papers/bertaux1995swan/bertaux1995swan_notes.md) · [💻 code](papers/bertaux1995swan/bertaux1995swan_implementation.ipynb) · [📄 pdf](papers/bertaux1995swan/bertaux1995swan_paper.pdf) |
| 47 | The Hinode (Solar-B) Mission: An Overview | 2007 | ✅ | [📝 notes](papers/kosugi2007hinode/kosugi2007hinode_notes.md) · [💻 code](papers/kosugi2007hinode/kosugi2007hinode_implementation.ipynb) · [📄 pdf](papers/kosugi2007hinode/kosugi2007hinode_paper.pdf) |
| 48 | The X-Ray Telescope (XRT) for the Hinode Mission | 2007 | ✅ | [📝 notes](papers/golub2007xray/golub2007xray_notes.md) · [💻 code](papers/golub2007xray/golub2007xray_implementation.ipynb) · [📄 pdf](papers/golub2007xray/golub2007xray_paper.pdf) |
| 49 | The STEREO Mission: An Introduction | 2008 | ✅ | [📝 notes](papers/kaiser2008stereo/kaiser2008stereo_notes.md) · [💻 code](papers/kaiser2008stereo/kaiser2008stereo_implementation.ipynb) · [📄 pdf](papers/kaiser2008stereo/kaiser2008stereo_paper.pdf) |
| 50 | Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI) | 2008 | ✅ | [📝 notes](papers/howard2008sun/howard2008sun_notes.md) · [💻 code](papers/howard2008sun/howard2008sun_implementation.ipynb) · [📄 pdf](papers/howard2008sun/howard2008sun_paper.pdf) |
| 51 | Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO) | 2012 | ✅ | [📝 notes](papers/woods2012extreme/woods2012extreme_notes.md) · [💻 code](papers/woods2012extreme/woods2012extreme_implementation.ipynb) · [📄 pdf](papers/woods2012extreme/woods2012extreme_paper.pdf) |
| 52 | The FIELDS Instrument Suite for Solar Probe Plus (PSP/FIELDS) | 2016 | ✅ | [📝 notes](papers/bale2016fields/bale2016fields_notes.md) · [💻 code](papers/bale2016fields/bale2016fields_implementation.ipynb) · [📄 pdf](papers/bale2016fields/bale2016fields_paper.pdf) |
| 53 | The Wide-Field Imager for Solar Probe Plus (PSP/WISPR) | 2016 | ✅ | [📝 notes](papers/vourlidas2016widefield/vourlidas2016widefield_notes.md) · [💻 code](papers/vourlidas2016widefield/vourlidas2016widefield_implementation.ipynb) · [📄 pdf](papers/vourlidas2016widefield/vourlidas2016widefield_paper.pdf) |
| 54 | Solar Wind Electrons Alphas and Protons (SWEAP) Investigation for PSP | 2016 | ✅ | [📝 notes](papers/kasper2016solar/kasper2016solar_notes.md) · [💻 code](papers/kasper2016solar/kasper2016solar_implementation.ipynb) · [📄 pdf](papers/kasper2016solar/kasper2016solar_paper.pdf) |
| 55 | Integrated Science Investigation of the Sun (IS☉IS) for PSP | 2016 | ✅ | [📝 notes](papers/mccomas2016integrated/mccomas2016integrated_notes.md) · [💻 code](papers/mccomas2016integrated/mccomas2016integrated_implementation.ipynb) · [📄 pdf](papers/mccomas2016integrated/mccomas2016integrated_paper.pdf) |
| 56 | The Polarimetric and Helioseismic Imager on Solar Orbiter (PHI) | 2020 | ✅ | [📝 notes](papers/solanki2020polarimetric/solanki2020polarimetric_notes.md) · [💻 code](papers/solanki2020polarimetric/solanki2020polarimetric_implementation.ipynb) · [📄 pdf](papers/solanki2020polarimetric/solanki2020polarimetric_paper.pdf) |
| 57 | Metis: the Solar Orbiter Visible Light and Ultraviolet Coronal Imager | 2020 | ✅ | [📝 notes](papers/antonucci2020metis/antonucci2020metis_notes.md) · [💻 code](papers/antonucci2020metis/antonucci2020metis_implementation.ipynb) · [📄 pdf](papers/antonucci2020metis/antonucci2020metis_paper.pdf) |
| 58 | The Solar Orbiter Heliospheric Imager (SoloHI) | 2020 | ✅ | [📝 notes](papers/howard2020solar/howard2020solar_notes.md) · [💻 code](papers/howard2020solar/howard2020solar_implementation.ipynb) · [📄 pdf](papers/howard2020solar/howard2020solar_paper.pdf) |
| 59 | The Spectrometer/Telescope for Imaging X-rays (STIX) on Solar Orbiter | 2020 | ✅ | [📝 notes](papers/krucker2020spectrometer/krucker2020spectrometer_notes.md) · [💻 code](papers/krucker2020spectrometer/krucker2020spectrometer_implementation.ipynb) · [📄 pdf](papers/krucker2020spectrometer/krucker2020spectrometer_paper.pdf) |
| 66 | Pottasch 1963 (EM Foundation) → migrated |  | ↗️ migrated | — |
| 67 | Jefferies+ 1972 (Bivariate DEM Formalism) → migrated |  | ↗️ migrated | — |
| 68 | Withbroe 1975 (XUV Analysis) → migrated |  | ↗️ migrated | — |
| 69 | Craig & Brown 1976 (Ill-Posed DEM Inversion) → migrated |  | ↗️ migrated | — |
| 70 | Sylwester+ 1980 (Withbroe-Sylwester Method) → migrated |  | ↗️ migrated | — |
| 71 | Monsignori-Fossi & Landini 1991 (Arcetri Iterative DEM) → migrated |  | ↗️ migrated | — |
| 72 | Judge+ 1997 (Information-theoretic Limits) → migrated |  | ↗️ migrated | — |
| 73 | Kashyap & Drake 1998 (MCMC DEM) → migrated |  | ↗️ migrated | — |
| 74 | Weber+ 2004 (xrt_dem_iterative2) → migrated |  | ↗️ migrated | — |
| 75 | Hannah & Kontar 2012 (demreg) → migrated |  | ↗️ migrated | — |
| 76 | Aschwanden+ 2013 (Automated AIA DEM) → migrated |  | ↗️ migrated | — |
| 77 | Plowman+ 2013 (Fast Basis-Pursuit DEM) → migrated |  | ↗️ migrated | — |
| 78 | Aschwanden+ 2015 (DEM Benchmark) → migrated |  | ↗️ migrated | — |
| 79 | Morgan & Pickering 2019 (SITES) → migrated |  | ↗️ migrated | — |
| 80 | Massa+ 2023 (RML DEM) → migrated |  | ↗️ migrated | — |

<a id="living-reviews-solar-physics"></a>

### Living Reviews in Solar Physics / 리빙 리뷰 (87 / 87)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | Astrospheres and Solar-like Stellar Winds | 2004 | ✅ | [📝 notes](papers/wood2004astrospheres/wood2004astrospheres_notes.md) · [💻 code](papers/wood2004astrospheres/wood2004astrospheres_implementation.ipynb) · [📄 pdf](papers/wood2004astrospheres/wood2004astrospheres_paper.pdf) |
| 2 | Large-Scale Dynamics of the Convection Zone and Tachocline | 2005 | ✅ | [📝 notes](papers/miesch2005largescale/miesch2005largescale_notes.md) · [💻 code](papers/miesch2005largescale/miesch2005largescale_implementation.ipynb) · [📄 pdf](papers/miesch2005largescale/miesch2005largescale_paper.pdf) |
| 3 | Coronal Waves and Oscillations | 2005 | ✅ | [📝 notes](papers/nakariakov2005coronal/nakariakov2005coronal_notes.md) · [💻 code](papers/nakariakov2005coronal/nakariakov2005coronal_implementation.ipynb) · [📄 pdf](papers/nakariakov2005coronal/nakariakov2005coronal_paper.pdf) |
| 4 | Surface Evolution of the Sun's Magnetic Field: A Historical Review of the Flux-Transport Mechanism | 2005 | ✅ | [📝 notes](papers/sheeley2005surface/sheeley2005surface_notes.md) · [💻 code](papers/sheeley2005surface/sheeley2005surface_implementation.ipynb) · [📄 pdf](papers/sheeley2005surface/sheeley2005surface_paper.pdf) |
| 5 | Local Helioseismology | 2005 | ✅ | [📝 notes](papers/gizon2005local/gizon2005local_notes.md) · [💻 code](papers/gizon2005local/gizon2005local_implementation.ipynb) · [📄 pdf](papers/gizon2005local/gizon2005local_paper.pdf) |
| 6 | Topological Methods for the Analysis of Solar Magnetic Fields | 2005 | ✅ | [📝 notes](papers/longcope2005topological/longcope2005topological_notes.md) · [💻 code](papers/longcope2005topological/longcope2005topological_implementation.ipynb) · [📄 pdf](papers/longcope2005topological/longcope2005topological_paper.pdf) |
| 7 | Starspots: A Key to the Stellar Dynamo | 2005 | ✅ | [📝 notes](papers/berdyugina2005starspots/berdyugina2005starspots_notes.md) · [💻 code](papers/berdyugina2005starspots/berdyugina2005starspots_implementation.ipynb) · [📄 pdf](papers/berdyugina2005starspots/berdyugina2005starspots_paper.pdf) |
| 8 | Kinetic Physics of the Solar Corona and Solar Wind | 2006 | ✅ | [📝 notes](papers/marsch2006kinetic/marsch2006kinetic_notes.md) · [💻 code](papers/marsch2006kinetic/marsch2006kinetic_implementation.ipynb) · [📄 pdf](papers/marsch2006kinetic/marsch2006kinetic_paper.pdf) |
| 9 | Space Weather: The Solar Perspective | 2006 | ✅ | [📝 notes](papers/schwenn2006space/schwenn2006space_notes.md) · [💻 code](papers/schwenn2006space/schwenn2006space_implementation.ipynb) · [📄 pdf](papers/schwenn2006space/schwenn2006space_paper.pdf) |
| 10 | Space Weather: Terrestrial Perspective | 2007 | ✅ | [📝 notes](papers/pulkkinen2007space/pulkkinen2007space_notes.md) · [💻 code](papers/pulkkinen2007space/pulkkinen2007space_implementation.ipynb) · [📄 pdf](papers/pulkkinen2007space/pulkkinen2007space_paper.pdf) |
| 11 | The Sun and the Earth's Climate | 2007 | ✅ | [📝 notes](papers/haigh2007sun/haigh2007sun_notes.md) · [💻 code](papers/haigh2007sun/haigh2007sun_implementation.ipynb) · [📄 pdf](papers/haigh2007sun/haigh2007sun_paper.pdf) |
| 12 | The Sun in Time: Activity and Environment | 2007 | ✅ | [📝 notes](papers/gudel2007sun/gudel2007sun_notes.md) · [💻 code](papers/gudel2007sun/gudel2007sun_implementation.ipynb) · [📄 pdf](papers/gudel2007sun/gudel2007sun_paper.pdf) |
| 13 | Flare Observations | 2008 | ✅ | [📝 notes](papers/benz2008flare/benz2008flare_notes.md) · [💻 code](papers/benz2008flare/benz2008flare_implementation.ipynb) · [📄 pdf](papers/benz2008flare/benz2008flare_paper.pdf) |
| 14 | Stellar Chromospheric Activity | 2008 | ✅ | [📝 notes](papers/hall2008stellar/hall2008stellar_notes.md) · [💻 code](papers/hall2008stellar/hall2008stellar_implementation.ipynb) · [📄 pdf](papers/hall2008stellar/hall2008stellar_paper.pdf) |
| 15 | Solar Interior Rotation and its Variation | 2009 | ✅ | [📝 notes](papers/howe2009solar/howe2009solar_notes.md) · [💻 code](papers/howe2009solar/howe2009solar_implementation.ipynb) · [📄 pdf](papers/howe2009solar/howe2009solar_paper.pdf) |
| 16 | Solar Surface Convection | 2009 | ✅ | [📝 notes](papers/nordlund2009solar/nordlund2009solar_notes.md) · [💻 code](papers/nordlund2009solar/nordlund2009solar_implementation.ipynb) · [📄 pdf](papers/nordlund2009solar/nordlund2009solar_paper.pdf) |
| 17 | Coronal Holes | 2009 | ✅ | [📝 notes](papers/cranmer2009coronal/cranmer2009coronal_notes.md) · [💻 code](papers/cranmer2009coronal/cranmer2009coronal_implementation.ipynb) · [📄 pdf](papers/cranmer2009coronal/cranmer2009coronal_paper.pdf) |
| 18 | Magnetic Fields in the Solar Convection Zone | 2009 | ✅ | [📝 notes](papers/fan2009magnetic/fan2009magnetic_notes.md) · [💻 code](papers/fan2009magnetic/fan2009magnetic_implementation.ipynb) · [📄 pdf](papers/fan2009magnetic/fan2009magnetic_paper.pdf) |
| 19 | The Sun's Supergranulation | 2010 | ✅ | [📝 notes](papers/rieutord2010suns/rieutord2010suns_notes.md) · [💻 code](papers/rieutord2010suns/rieutord2010suns_implementation.ipynb) · [📄 pdf](papers/rieutord2010suns/rieutord2010suns_paper.pdf) |
| 20 | Dynamo Models of the Solar Cycle | 2010 | ✅ | [📝 notes](papers/charbonneau2010dynamo/charbonneau2010dynamo_notes.md) · [💻 code](papers/charbonneau2010dynamo/charbonneau2010dynamo_implementation.ipynb) · [📄 pdf](papers/charbonneau2010dynamo/charbonneau2010dynamo_paper.pdf) |
| 21 | Wave Modeling of the Solar Wind | 2010 | ✅ | [📝 notes](papers/ofman2010wave/ofman2010wave_notes.md) · [💻 code](papers/ofman2010wave/ofman2010wave_implementation.ipynb) · [📄 pdf](papers/ofman2010wave/ofman2010wave_paper.pdf) |
| 22 | Coronal Mass Ejections: Models and Their Observational Basis | 2011 | ✅ | [📝 notes](papers/chen2011coronal/chen2011coronal_notes.md) · [💻 code](papers/chen2011coronal/chen2011coronal_implementation.ipynb) · [📄 pdf](papers/chen2011coronal/chen2011coronal_paper.pdf) |
| 23 | Solar Adaptive Optics | 2011 | ✅ | [📝 notes](papers/rimmele2011solar/rimmele2011solar_notes.md) · [💻 code](papers/rimmele2011solar/rimmele2011solar_implementation.ipynb) · [📄 pdf](papers/rimmele2011solar/rimmele2011solar_paper.pdf) |
| 24 | Sunspot Modeling: From Simplified Models to Radiative MHD Simulations | 2011 | ✅ | [📝 notes](papers/rempel2011sunspot/rempel2011sunspot_notes.md) · [💻 code](papers/rempel2011sunspot/rempel2011sunspot_implementation.ipynb) · [📄 pdf](papers/rempel2011sunspot/rempel2011sunspot_paper.pdf) |
| 25 | Magnetic Structure of Sunspots | 2011 | ✅ | [📝 notes](papers/borrero2011magnetic/borrero2011magnetic_notes.md) · [💻 code](papers/borrero2011magnetic/borrero2011magnetic_implementation.ipynb) · [📄 pdf](papers/borrero2011magnetic/borrero2011magnetic_paper.pdf) |
| 26 | Solar Stereoscopy and Tomography | 2011 | ✅ | [📝 notes](papers/aschwanden2011solar/aschwanden2011solar_notes.md) · [💻 code](papers/aschwanden2011solar/aschwanden2011solar_implementation.ipynb) · [📄 pdf](papers/aschwanden2011solar/aschwanden2011solar_paper.pdf) |
| 27 | Solar Flares: Magnetohydrodynamic Processes | 2011 | ✅ | [📝 notes](papers/shibata2011solar/shibata2011solar_notes.md) · [💻 code](papers/shibata2011solar/shibata2011solar_implementation.ipynb) · [📄 pdf](papers/shibata2011solar/shibata2011solar_paper.pdf) |
| 28 | Observations of Cool-Star Magnetic Fields | 2012 | ✅ | [📝 notes](papers/reiners2012observations/reiners2012observations_notes.md) · [💻 code](papers/reiners2012observations/reiners2012observations_implementation.ipynb) · [📄 pdf](papers/reiners2012observations/reiners2012observations_paper.pdf) |
| 29 | Coronal Mass Ejections: Observations | 2012 | ✅ | [📝 notes](papers/webb2012coronal/webb2012coronal_notes.md) · [💻 code](papers/webb2012coronal/webb2012coronal_implementation.ipynb) · [📄 pdf](papers/webb2012coronal/webb2012coronal_paper.pdf) |
| 30 | Solar Surface Magneto-Convection | 2012 | ✅ | [📝 notes](papers/stein2012solar/stein2012solar_notes.md) · [💻 code](papers/stein2012solar/stein2012solar_implementation.ipynb) · [📄 pdf](papers/stein2012solar/stein2012solar_paper.pdf) |
| 31 | The Sun's Global Photospheric and Coronal Magnetic Fields: Observations and Models | 2012 | ✅ | [📝 notes](papers/mackay2012suns/mackay2012suns_notes.md) · [💻 code](papers/mackay2012suns/mackay2012suns_implementation.ipynb) · [📄 pdf](papers/mackay2012suns/mackay2012suns_paper.pdf) |
| 32 | The Solar Wind as a Turbulence Laboratory | 2013 | ✅ | [📝 notes](papers/bruno2013solar/bruno2013solar_notes.md) · [💻 code](papers/bruno2013solar/bruno2013solar_implementation.ipynb) · [📄 pdf](papers/bruno2013solar/bruno2013solar_paper.pdf) |
| 33 | Solar Modulation of Cosmic Rays | 2013 | ✅ | [📝 notes](papers/potgieter2013solar/potgieter2013solar_notes.md) · [💻 code](papers/potgieter2013solar/potgieter2013solar_implementation.ipynb) · [📄 pdf](papers/potgieter2013solar/potgieter2013solar_paper.pdf) |
| 34 | Reconstruction and Prediction of Variations in the Open Solar Magnetic Flux and Interplanetary Conditions | 2013 | ✅ | [📝 notes](papers/lockwood2013reconstruction/lockwood2013reconstruction_notes.md) · [💻 code](papers/lockwood2013reconstruction/lockwood2013reconstruction_implementation.ipynb) · [📄 pdf](papers/lockwood2013reconstruction/lockwood2013reconstruction_paper.pdf) |
| 35 | The Heliospheric Magnetic Field | 2013 | ✅ | [📝 notes](papers/owens2013heliospheric/owens2013heliospheric_notes.md) · [💻 code](papers/owens2013heliospheric/owens2013heliospheric_implementation.ipynb) · [📄 pdf](papers/owens2013heliospheric/owens2013heliospheric_paper.pdf) |
| 36 | Solar Prominences: Observations | 2014 | ✅ | [📝 notes](papers/parenti2014solar/parenti2014solar_notes.md) · [💻 code](papers/parenti2014solar/parenti2014solar_implementation.ipynb) · [📄 pdf](papers/parenti2014solar/parenti2014solar_paper.pdf) |
| 37 | Infrared Solar Physics | 2014 | ✅ | [📝 notes](papers/penn2014infrared/penn2014infrared_notes.md) · [💻 code](papers/penn2014infrared/penn2014infrared_implementation.ipynb) · [📄 pdf](papers/penn2014infrared/penn2014infrared_paper.pdf) |
| 38 | Flux Emergence (Theory) | 2014 | ✅ | [📝 notes](papers/cheung2014flux/cheung2014flux_notes.md) · [💻 code](papers/cheung2014flux/cheung2014flux_implementation.ipynb) · [📄 pdf](papers/cheung2014flux/cheung2014flux_paper.pdf) |
| 39 | Coronal Loops: Observations and Modeling of Confined Plasma | 2014 | ✅ | [📝 notes](papers/reale2014coronal/reale2014coronal_notes.md) · [💻 code](papers/reale2014coronal/reale2014coronal_implementation.ipynb) · [📄 pdf](papers/reale2014coronal/reale2014coronal_paper.pdf) |
| 40 | Evolution of Active Regions | 2015 | ✅ | [📝 notes](papers/vandrielgesztelyi2015evolution/vandrielgesztelyi2015evolution_notes.md) · [💻 code](papers/vandrielgesztelyi2015evolution/vandrielgesztelyi2015evolution_implementation.ipynb) · [📄 pdf](papers/vandrielgesztelyi2015evolution/vandrielgesztelyi2015evolution_paper.pdf) |
| 41 | The FIP and Inverse FIP Effects in Solar and Stellar Coronae | 2015 | ✅ | [📝 notes](papers/laming2015fip/laming2015fip_notes.md) · [💻 code](papers/laming2015fip/laming2015fip_implementation.ipynb) · [📄 pdf](papers/laming2015fip/laming2015fip_paper.pdf) |
| 42 | Large-scale Globally Propagating Coronal Waves | 2015 | ✅ | [📝 notes](papers/warmuth2015largescale/warmuth2015largescale_notes.md) · [💻 code](papers/warmuth2015largescale/warmuth2015largescale_implementation.ipynb) · [📄 pdf](papers/warmuth2015largescale/warmuth2015largescale_paper.pdf) |
| 43 | The Solar Cycle | 2015 | ✅ | [📝 notes](papers/hathaway2015solar/hathaway2015solar_notes.md) · [💻 code](papers/hathaway2015solar/hathaway2015solar_implementation.ipynb) · [📄 pdf](papers/hathaway2015solar/hathaway2015solar_paper.pdf) |
| 44 | Solar Magnetism in the Polar Regions | 2015 | ✅ | [📝 notes](papers/petrie2015solar/petrie2015solar_notes.md) · [💻 code](papers/petrie2015solar/petrie2015solar_implementation.ipynb) · [📄 pdf](papers/petrie2015solar/petrie2015solar_paper.pdf) |
| 45 | Oscillations and Waves in Sunspots | 2015 | ✅ | [📝 notes](papers/khomenko2015oscillations/khomenko2015oscillations_notes.md) · [💻 code](papers/khomenko2015oscillations/khomenko2015oscillations_implementation.ipynb) · [📄 pdf](papers/khomenko2015oscillations/khomenko2015oscillations_paper.pdf) |
| 46 | Solar Coronal Plumes | 2015 | ✅ | [📝 notes](papers/poletto2015solar/poletto2015solar_notes.md) · [💻 code](papers/poletto2015solar/poletto2015solar_implementation.ipynb) · [📄 pdf](papers/poletto2015solar/poletto2015solar_paper.pdf) |
| 47 | Interaction Between Convection and Pulsation | 2015 | ✅ | [📝 notes](papers/houdek2015interaction/houdek2015interaction_notes.md) · [💻 code](papers/houdek2015interaction/houdek2015interaction_implementation.ipynb) · [📄 pdf](papers/houdek2015interaction/houdek2015interaction_paper.pdf) |
| 48 | Solar and Stellar Photospheric Abundances | 2016 | ✅ | [📝 notes](papers/prieto2016solar/prieto2016solar_notes.md) · [💻 code](papers/prieto2016solar/prieto2016solar_implementation.ipynb) · [📄 pdf](papers/prieto2016solar/prieto2016solar_paper.pdf) |
| 49 | Global Seismology of the Sun | 2016 | ✅ | [📝 notes](papers/basu2016global/basu2016global_notes.md) · [💻 code](papers/basu2016global/basu2016global_implementation.ipynb) · [📄 pdf](papers/basu2016global/basu2016global_paper.pdf) |
| 50 | Large Gradual Solar Energetic Particle Events | 2016 | ✅ | [📝 notes](papers/desai2016large/desai2016large_notes.md) · [💻 code](papers/desai2016large/desai2016large_implementation.ipynb) · [📄 pdf](papers/desai2016large/desai2016large_paper.pdf) |
| 51 | Inversion of the Radiative Transfer Equation for Polarized Light | 2016 | ✅ | [📝 notes](papers/iniesta2016inversion/iniesta2016inversion_notes.md) · [💻 code](papers/iniesta2016inversion/iniesta2016inversion_implementation.ipynb) · [📄 pdf](papers/iniesta2016inversion/iniesta2016inversion_paper.pdf) |
| 52 | A History of Solar Activity over Millennia | 2017 | ✅ | [📝 notes](papers/usoskin2017history/usoskin2017history_notes.md) · [💻 code](papers/usoskin2017history/usoskin2017history_implementation.ipynb) · [📄 pdf](papers/usoskin2017history/usoskin2017history_paper.pdf) |
| 53 | Magnetism, Dynamo Action and the Solar-Stellar Connection | 2017 | ✅ | [📝 notes](papers/brun2017magnetism/brun2017magnetism_notes.md) · [💻 code](papers/brun2017magnetism/brun2017magnetism_implementation.ipynb) · [📄 pdf](papers/brun2017magnetism/brun2017magnetism_paper.pdf) |
| 54 | Coronal Mass Ejections and Their Sheath Regions in Interplanetary Space | 2017 | ✅ | [📝 notes](papers/kilpua2017coronal/kilpua2017coronal_notes.md) · [💻 code](papers/kilpua2017coronal/kilpua2017coronal_implementation.ipynb) · [📄 pdf](papers/kilpua2017coronal/kilpua2017coronal_paper.pdf) |
| 55 | Solar Wind Stream Interaction Regions Throughout the Heliosphere | 2018 | ✅ | [📝 notes](papers/richardson2018solar/richardson2018solar_notes.md) · [💻 code](papers/richardson2018solar/richardson2018solar_implementation.ipynb) · [📄 pdf](papers/richardson2018solar/richardson2018solar_paper.pdf) |
| 56 | Space–Time Structure and Wavevector Anisotropy in Space Plasma Turbulence | 2018 | ✅ | [📝 notes](papers/narita2018space/narita2018space_notes.md) · [💻 code](papers/narita2018space/narita2018space_implementation.ipynb) · [📄 pdf](papers/narita2018space/narita2018space_paper.pdf) |
| 57 | Prominence Oscillations | 2018 | ✅ | [📝 notes](papers/arregui2018prominence/arregui2018prominence_notes.md) · [💻 code](papers/arregui2018prominence/arregui2018prominence_implementation.ipynb) · [📄 pdf](papers/arregui2018prominence/arregui2018prominence_paper.pdf) |
| 58 | Extended MHD Modeling of the Steady Solar Corona and the Solar Wind | 2018 | ✅ | [📝 notes](papers/gombosi2018extended/gombosi2018extended_notes.md) · [💻 code](papers/gombosi2018extended/gombosi2018extended_implementation.ipynb) · [📄 pdf](papers/gombosi2018extended/gombosi2018extended_paper.pdf) |
| 59 | Solar UV and X-ray Spectral Diagnostics | 2018 | ✅ | [📝 notes](papers/zanna2018solar/zanna2018solar_notes.md) · [💻 code](papers/zanna2018solar/zanna2018solar_implementation.ipynb) · [📄 pdf](papers/zanna2018solar/zanna2018solar_paper.pdf) |
| 60 | The Sun's Supergranulation (update) | 2018 | ✅ | [📝 notes](papers/rincon2018suns/rincon2018suns_notes.md) · [💻 code](papers/rincon2018suns/rincon2018suns_implementation.ipynb) · [📄 pdf](papers/rincon2018suns/rincon2018suns_paper.pdf) |
| 61 | Solar Prominences: Theory and Models | 2018 | ✅ | [📝 notes](papers/gibson2018solar/gibson2018solar_notes.md) · [💻 code](papers/gibson2018solar/gibson2018solar_implementation.ipynb) · [📄 pdf](papers/gibson2018solar/gibson2018solar_paper.pdf) |
| 62 | Quiet Sun Magnetic Fields: An Observational View | 2019 | ✅ | [📝 notes](papers/rubio2019quiet/rubio2019quiet_notes.md) · [💻 code](papers/rubio2019quiet/rubio2019quiet_implementation.ipynb) · [📄 pdf](papers/rubio2019quiet/rubio2019quiet_paper.pdf) |
| 63 | Coronal Bright Points | 2019 | ✅ | [📝 notes](papers/madjarska2019coronal/madjarska2019coronal_notes.md) · [💻 code](papers/madjarska2019coronal/madjarska2019coronal_implementation.ipynb) · [📄 pdf](papers/madjarska2019coronal/madjarska2019coronal_paper.pdf) |
| 64 | Flare-Productive Active Regions | 2019 | ✅ | [📝 notes](papers/toriumi2019flareproductive/toriumi2019flareproductive_notes.md) · [💻 code](papers/toriumi2019flareproductive/toriumi2019flareproductive_implementation.ipynb) · [📄 pdf](papers/toriumi2019flareproductive/toriumi2019flareproductive_paper.pdf) |
| 65 | Asteroseismology of Solar-Type Stars | 2019 | ✅ | [📝 notes](papers/garcia2019asteroseismology/garcia2019asteroseismology_notes.md) · [💻 code](papers/garcia2019asteroseismology/garcia2019asteroseismology_implementation.ipynb) · [📄 pdf](papers/garcia2019asteroseismology/garcia2019asteroseismology_paper.pdf) |
| 66 | The Multi-Scale Nature of the Solar Wind | 2019 | ✅ | [📝 notes](papers/verscharen2019multiscale/verscharen2019multiscale_notes.md) · [💻 code](papers/verscharen2019multiscale/verscharen2019multiscale_implementation.ipynb) · [📄 pdf](papers/verscharen2019multiscale/verscharen2019multiscale_paper.pdf) |
| 67 | Historical Sunspot Records | 2020 | ✅ | [📝 notes](papers/arlt2020historical/arlt2020historical_notes.md) · [💻 code](papers/arlt2020historical/arlt2020historical_implementation.ipynb) · [📄 pdf](papers/arlt2020historical/arlt2020historical_paper.pdf) |
| 68 | Solar Cycle Prediction | 2020 | ✅ | [📝 notes](papers/petrovay2020solar/petrovay2020solar_notes.md) · [💻 code](papers/petrovay2020solar/petrovay2020solar_implementation.ipynb) · [📄 pdf](papers/petrovay2020solar/petrovay2020solar_paper.pdf) |
| 69 | The Parker Problem: Existence of Smooth Force-Free Fields and Coronal Heating | 2020 | ✅ | [📝 notes](papers/pontin2020parker/pontin2020parker_notes.md) · [💻 code](papers/pontin2020parker/pontin2020parker_implementation.ipynb) · [📄 pdf](papers/pontin2020parker/pontin2020parker_paper.pdf) |
| 70 | Radiation Hydrodynamics in Simulations of the Solar Atmosphere | 2020 | ✅ | [📝 notes](papers/leenaarts2020radiation/leenaarts2020radiation_notes.md) · [💻 code](papers/leenaarts2020radiation/leenaarts2020radiation_implementation.ipynb) · [📄 pdf](papers/leenaarts2020radiation/leenaarts2020radiation_paper.pdf) |
| 71 | Dynamo Models of the Solar Cycle (update) | 2020 | ✅ | [📝 notes](papers/charbonneau2020dynamo/charbonneau2020dynamo_notes.md) · [💻 code](papers/charbonneau2020dynamo/charbonneau2020dynamo_implementation.ipynb) · [📄 pdf](papers/charbonneau2020dynamo/charbonneau2020dynamo_paper.pdf) |
| 72 | Solar Force-Free Magnetic Fields | 2021 | ✅ | [📝 notes](papers/wiegelmann2021solar/wiegelmann2021solar_notes.md) · [💻 code](papers/wiegelmann2021solar/wiegelmann2021solar_implementation.ipynb) · [📄 pdf](papers/wiegelmann2021solar/wiegelmann2021solar_paper.pdf) |
| 73 | Solar Structure and Evolution | 2021 | ✅ | [📝 notes](papers/christensendalsgaard2021solar/christensendalsgaard2021solar_notes.md) · [💻 code](papers/christensendalsgaard2021solar/christensendalsgaard2021solar_implementation.ipynb) · [📄 pdf](papers/christensendalsgaard2021solar/christensendalsgaard2021solar_paper.pdf) |
| 74 | The Evolution of the Solar Wind | 2021 | ✅ | [📝 notes](papers/vidotto2021evolution/vidotto2021evolution_notes.md) · [💻 code](papers/vidotto2021evolution/vidotto2021evolution_implementation.ipynb) · [📄 pdf](papers/vidotto2021evolution/vidotto2021evolution_paper.pdf) |
| 75 | Space Weather: The Solar Perspective (update) | 2021 | ✅ | [📝 notes](papers/temmer2021space/temmer2021space_notes.md) · [💻 code](papers/temmer2021space/temmer2021space_implementation.ipynb) · [📄 pdf](papers/temmer2021space/temmer2021space_paper.pdf) |
| 76 | Magnetic Fields in the Solar Convection Zone (update) | 2021 | ✅ | [📝 notes](papers/fan2021magnetic/fan2021magnetic_notes.md) · [💻 code](papers/fan2021magnetic/fan2021magnetic_implementation.ipynb) · [📄 pdf](papers/fan2021magnetic/fan2021magnetic_paper.pdf) |
| 77 | Magnetic Reconnection: MHD Theory and Modelling | 2022 | ✅ | [📝 notes](papers/pontin2022magnetic/pontin2022magnetic_notes.md) · [💻 code](papers/pontin2022magnetic/pontin2022magnetic_implementation.ipynb) · [📄 pdf](papers/pontin2022magnetic/pontin2022magnetic_paper.pdf) |
| 78 | Extreme Solar Events | 2022 | ✅ | [📝 notes](papers/cliver2022extreme/cliver2022extreme_notes.md) · [💻 code](papers/cliver2022extreme/cliver2022extreme_implementation.ipynb) · [📄 pdf](papers/cliver2022extreme/cliver2022extreme_paper.pdf) |
| 79 | Surface and Interior Meridional Circulation in the Sun | 2022 | ✅ | [📝 notes](papers/hanasoge2022surface/hanasoge2022surface_notes.md) · [💻 code](papers/hanasoge2022surface/hanasoge2022surface_implementation.ipynb) · [📄 pdf](papers/hanasoge2022surface/hanasoge2022surface_paper.pdf) |
| 80 | Waves in the Lower Solar Atmosphere: The Dawn of Next-Generation Solar Telescopes | 2023 | ✅ | [📝 notes](papers/jess2023waves/jess2023waves_notes.md) · [💻 code](papers/jess2023waves/jess2023waves_implementation.ipynb) · [📄 pdf](papers/jess2023waves/jess2023waves_paper.pdf) |
| 81 | A History of Solar Activity over Millennia (update) | 2023 | ✅ | [📝 notes](papers/usoskin2023history/usoskin2023history_notes.md) · [💻 code](papers/usoskin2023history/usoskin2023history_implementation.ipynb) · [📄 pdf](papers/usoskin2023history/usoskin2023history_paper.pdf) |
| 82 | Models for the Long-Term Variations of Solar Activity | 2023 | ✅ | [📝 notes](papers/karak2023models/karak2023models_notes.md) · [💻 code](papers/karak2023models/karak2023models_implementation.ipynb) · [📄 pdf](papers/karak2023models/karak2023models_paper.pdf) |
| 83 | Machine Learning in Solar Physics | 2023 | ✅ | [📝 notes](papers/ramos2023machine/ramos2023machine_notes.md) · [💻 code](papers/ramos2023machine/ramos2023machine_implementation.ipynb) · [📄 pdf](papers/ramos2023machine/ramos2023machine_paper.pdf) |
| 84 | Stellar Flares | 2024 | ✅ | [📝 notes](papers/kowalski2024stellar/kowalski2024stellar_notes.md) · [💻 code](papers/kowalski2024stellar/kowalski2024stellar_implementation.ipynb) · [📄 pdf](papers/kowalski2024stellar/kowalski2024stellar_paper.pdf) |
| 85 | Solar Irradiance Measurements | 2025 | ✅ | [📝 notes](papers/kopp2025solar/kopp2025solar_notes.md) · [💻 code](papers/kopp2025solar/kopp2025solar_implementation.ipynb) · [📄 pdf](papers/kopp2025solar/kopp2025solar_paper.pdf) |
| 86 | Coronal Dimmings and What They Tell Us About Solar and Stellar Coronal Mass Ejections | 2025 | ✅ | [📝 notes](papers/veronig2025coronal/veronig2025coronal_notes.md) · [💻 code](papers/veronig2025coronal/veronig2025coronal_implementation.ipynb) · [📄 pdf](papers/veronig2025coronal/veronig2025coronal_paper.pdf) |
| 87 | Modeling Multiphase Plasma in the Corona: Prominences and Rain | 2025 | ✅ | [📝 notes](papers/keppens2025modeling/keppens2025modeling_notes.md) · [💻 code](papers/keppens2025modeling/keppens2025modeling_implementation.ipynb) · [📄 pdf](papers/keppens2025modeling/keppens2025modeling_paper.pdf) |

<a id="low-snr-imaging"></a>

### Low-SNR Imaging / 저신호대잡음 영상 (45 / 45)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | Ideal Spatial Adaptation by Wavelet Shrinkage (VisuShrink) | 1994 | ✅ | [📝 notes](papers/donoho1994ideal/donoho1994ideal_notes.md) · [💻 code](papers/donoho1994ideal/donoho1994ideal_implementation.ipynb) · [📄 pdf](papers/donoho1994ideal/donoho1994ideal_paper.pdf) |
| 2 | Adapting to Unknown Smoothness via Wavelet Shrinkage (SureShrink) | 1995 | ✅ | [📝 notes](papers/donoho1995adapting/donoho1995adapting_notes.md) · [💻 code](papers/donoho1995adapting/donoho1995adapting_implementation.ipynb) · [📄 pdf](papers/donoho1995adapting/donoho1995adapting_paper.pdf) |
| 3 | Adaptive Wavelet Thresholding for Image Denoising and Compression (BayesShrink) | 2000 | ✅ | [📝 notes](papers/chang2000adaptive/chang2000adaptive_notes.md) · [💻 code](papers/chang2000adaptive/chang2000adaptive_implementation.ipynb) · [📄 pdf](papers/chang2000adaptive/chang2000adaptive_paper.pdf) |
| 4 | A Non-Local Algorithm for Image Denoising (NLM) | 2005 | ✅ | [📝 notes](papers/buades2005nonlocal/buades2005nonlocal_notes.md) · [💻 code](papers/buades2005nonlocal/buades2005nonlocal_implementation.ipynb) · [📄 pdf](papers/buades2005nonlocal/buades2005nonlocal_paper.pdf) |
| 5 | The Contourlet Transform: An Efficient Directional Multiresolution Image Representation | 2005 | ✅ | [📝 notes](papers/do2005contourlet/do2005contourlet_notes.md) · [💻 code](papers/do2005contourlet/do2005contourlet_implementation.ipynb) · [📄 pdf](papers/do2005contourlet/do2005contourlet_paper.pdf) |
| 6 | Fast Discrete Curvelet Transforms | 2006 | ✅ | [📝 notes](papers/candes2006fast/candes2006fast_notes.md) · [💻 code](papers/candes2006fast/candes2006fast_implementation.ipynb) · [📄 pdf](papers/candes2006fast/candes2006fast_paper.pdf) |
| 7 | Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering (BM3D) | 2007 | ✅ | [📝 notes](papers/dabov2007image/dabov2007image_notes.md) · [💻 code](papers/dabov2007image/dabov2007image_implementation.ipynb) · [📄 pdf](papers/dabov2007image/dabov2007image_paper.pdf) |
| 8 | Sparse Directional Image Representations using the Discrete Shearlet Transform | 2008 | ✅ | [📝 notes](papers/easley2008sparse/easley2008sparse_notes.md) · [💻 code](papers/easley2008sparse/easley2008sparse_implementation.ipynb) · [📄 pdf](papers/easley2008sparse/easley2008sparse_paper.pdf) |
| 9 | Video Denoising, Deblocking, and Enhancement Through Separable 4-D Nonlocal Spatiotemporal Transforms (V-BM4D) | 2012 | ✅ | [📝 notes](papers/maggioni2012video/maggioni2012video_notes.md) · [💻 code](papers/maggioni2012video/maggioni2012video_implementation.ipynb) · [📄 pdf](papers/maggioni2012video/maggioni2012video_paper.pdf) |
| 10 | Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction (BM4D) | 2013 | ✅ | [📝 notes](papers/maggioni2013nonlocal/maggioni2013nonlocal_notes.md) · [💻 code](papers/maggioni2013nonlocal/maggioni2013nonlocal_implementation.ipynb) · [📄 pdf](papers/maggioni2013nonlocal/maggioni2013nonlocal_paper.pdf) |
| 11 | The Transformation of Poisson, Binomial and Negative-Binomial Data (Anscombe Transform) | 1948 | ✅ | [📝 notes](papers/anscombe1948transformation/anscombe1948transformation_notes.md) · [💻 code](papers/anscombe1948transformation/anscombe1948transformation_implementation.ipynb) · [📄 pdf](papers/anscombe1948transformation/anscombe1948transformation_paper.pdf) |
| 12 | Poisson NL Means: Unsupervised Non-local Means for Poisson Noise | 2010 | ✅ | [📝 notes](papers/deledalle2010poisson/deledalle2010poisson_notes.md) · [💻 code](papers/deledalle2010poisson/deledalle2010poisson_implementation.ipynb) · [📄 pdf](papers/deledalle2010poisson/deledalle2010poisson_paper.pdf) |
| 13 | Image Denoising in Mixed Poisson–Gaussian Noise (PURE-LET) | 2011 | ✅ | [📝 notes](papers/luisier2011image/luisier2011image_notes.md) · [💻 code](papers/luisier2011image/luisier2011image_implementation.ipynb) · [📄 pdf](papers/luisier2011image/luisier2011image_paper.pdf) |
| 14 | Optimal Inversion of the Generalized Anscombe Transformation for Poisson–Gaussian Noise | 2013 | ✅ | [📝 notes](papers/makitalo2013optimal/makitalo2013optimal_notes.md) · [💻 code](papers/makitalo2013optimal/makitalo2013optimal_implementation.ipynb) · [📄 pdf](papers/makitalo2013optimal/makitalo2013optimal_paper.pdf) |
| 15 | Cryo-CARE: Content-Aware Image Restoration for Cryo-Electron Tomography | 2019 | ✅ | [📝 notes](papers/buchholz2019cryocare/buchholz2019cryocare_notes.md) · [💻 code](papers/buchholz2019cryocare/buchholz2019cryocare_implementation.ipynb) · [📄 pdf](papers/buchholz2019cryocare/buchholz2019cryocare_paper.pdf) |
| 16 | Noise2Noise: Learning Image Restoration without Clean Data | 2018 | ✅ | [📝 notes](papers/lehtinen2018noise2noise/lehtinen2018noise2noise_notes.md) · [💻 code](papers/lehtinen2018noise2noise/lehtinen2018noise2noise_implementation.ipynb) · [📄 pdf](papers/lehtinen2018noise2noise/lehtinen2018noise2noise_paper.pdf) |
| 17 | Noise2Void — Learning Denoising from Single Noisy Images | 2019 | ✅ | [📝 notes](papers/krull2019noise2void/krull2019noise2void_notes.md) · [💻 code](papers/krull2019noise2void/krull2019noise2void_implementation.ipynb) · [📄 pdf](papers/krull2019noise2void/krull2019noise2void_paper.pdf) |
| 18 | Noise2Self: Blind Denoising by Self-Supervision | 2019 | ✅ | [📝 notes](papers/batson2019noise2self/batson2019noise2self_notes.md) · [💻 code](papers/batson2019noise2self/batson2019noise2self_implementation.ipynb) · [📄 pdf](papers/batson2019noise2self/batson2019noise2self_paper.pdf) |
| 19 | Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image | 2020 | ✅ | [📝 notes](papers/quan2020self2self/quan2020self2self_notes.md) · [💻 code](papers/quan2020self2self/quan2020self2self_implementation.ipynb) · [📄 pdf](papers/quan2020self2self/quan2020self2self_paper.pdf) |
| 20 | Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images | 2021 | ✅ | [📝 notes](papers/huang2021neighbor2neighbor/huang2021neighbor2neighbor_notes.md) · [💻 code](papers/huang2021neighbor2neighbor/huang2021neighbor2neighbor_implementation.ipynb) · [📄 pdf](papers/huang2021neighbor2neighbor/huang2021neighbor2neighbor_paper.pdf) |
| 21 | Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising (R2R) | 2021 | ✅ | [📝 notes](papers/pang2021recorruptedtorecorrupted/pang2021recorruptedtorecorrupted_notes.md) · [💻 code](papers/pang2021recorruptedtorecorrupted/pang2021recorruptedtorecorrupted_implementation.ipynb) · [📄 pdf](papers/pang2021recorruptedtorecorrupted/pang2021recorruptedtorecorrupted_paper.pdf) |
| 22 | Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots | 2022 | ✅ | [📝 notes](papers/wang2022blind2unblind/wang2022blind2unblind_notes.md) · [💻 code](papers/wang2022blind2unblind/wang2022blind2unblind_implementation.ipynb) · [📄 pdf](papers/wang2022blind2unblind/wang2022blind2unblind_paper.pdf) |
| 23 | Cosmic-Ray Rejection by Laplacian Edge Detection (L.A.Cosmic) | 2001 | ✅ | [📝 notes](papers/vandokkum2001cosmicray/vandokkum2001cosmicray_notes.md) · [💻 code](papers/vandokkum2001cosmicray/vandokkum2001cosmicray_implementation.ipynb) · [📄 pdf](papers/vandokkum2001cosmicray/vandokkum2001cosmicray_paper.pdf) |
| 24 | deepCR: Cosmic Ray Rejection with Deep Learning | 2020 | ✅ | [📝 notes](papers/zhang2020deepcr/zhang2020deepcr_notes.md) · [💻 code](papers/zhang2020deepcr/zhang2020deepcr_implementation.ipynb) · [📄 pdf](papers/zhang2020deepcr/zhang2020deepcr_paper.pdf) |
| 25 | Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser | 2021 | ✅ | [📝 notes](papers/kadkhodaie2021stochastic/kadkhodaie2021stochastic_notes.md) · [💻 code](papers/kadkhodaie2021stochastic/kadkhodaie2021stochastic_implementation.ipynb) · [📄 pdf](papers/kadkhodaie2021stochastic/kadkhodaie2021stochastic_paper.pdf) |
| 26 | Denoising Diffusion Restoration Models (DDRM) | 2022 | ✅ | [📝 notes](papers/kawar2022denoising/kawar2022denoising_notes.md) · [💻 code](papers/kawar2022denoising/kawar2022denoising_implementation.ipynb) · [📄 pdf](papers/kawar2022denoising/kawar2022denoising_paper.pdf) |
| 27 | Cold Diffusion: Inverting Arbitrary Image Transforms without Noise | 2023 | ✅ | [📝 notes](papers/bansal2023cold/bansal2023cold_notes.md) · [💻 code](papers/bansal2023cold/bansal2023cold_implementation.ipynb) · [📄 pdf](papers/bansal2023cold/bansal2023cold_paper.pdf) |
| 28 | Diffusion Posterior Sampling for General Noisy Inverse Problems (DPS) | 2023 | ✅ | [📝 notes](papers/chung2023diffusion/chung2023diffusion_notes.md) · [💻 code](papers/chung2023diffusion/chung2023diffusion_implementation.ipynb) · [📄 pdf](papers/chung2023diffusion/chung2023diffusion_paper.pdf) |
| 29 | Ambient Diffusion: Learning Clean Distributions from Corrupted Data | 2023 | ✅ | [📝 notes](papers/daras2023ambient/daras2023ambient_notes.md) · [💻 code](papers/daras2023ambient/daras2023ambient_implementation.ipynb) · [📄 pdf](papers/daras2023ambient/daras2023ambient_paper.pdf) |
| 30 | Denoising Diffusion Models for Plug-and-Play Image Restoration (DiffPIR) | 2023 | ✅ | [📝 notes](papers/zhu2023denoising/zhu2023denoising_notes.md) · [💻 code](papers/zhu2023denoising/zhu2023denoising_implementation.ipynb) · [📄 pdf](papers/zhu2023denoising/zhu2023denoising_paper.pdf) |
| 31 | Low-Light Image Enhancement with Wavelet-Based Diffusion Models (DiffLL) | 2023 | ✅ | [📝 notes](papers/jiang2023lowlight/jiang2023lowlight_notes.md) · [💻 code](papers/jiang2023lowlight/jiang2023lowlight_implementation.ipynb) · [📄 pdf](papers/jiang2023lowlight/jiang2023lowlight_paper.pdf) |
| 32 | A Wavelet Packets Equalization Technique to Reveal the Multiple Spatial-Scale Nature of Coronal Structures | 2003 | ✅ | [📝 notes](papers/stenborg2003wavelet/stenborg2003wavelet_notes.md) · [💻 code](papers/stenborg2003wavelet/stenborg2003wavelet_implementation.ipynb) · [📄 pdf](papers/stenborg2003wavelet/stenborg2003wavelet_paper.pdf) |
| 33 | Gray and Color Image Contrast Enhancement by the Curvelet Transform | 2003 | ✅ | [📝 notes](papers/starck2003gray/starck2003gray_notes.md) · [💻 code](papers/starck2003gray/starck2003gray_implementation.ipynb) · [📄 pdf](papers/starck2003gray/starck2003gray_paper.pdf) |
| 34 | A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft (CDAW Catalog) | 2004 | ✅ | [📝 notes](papers/yashiro2004catalog/yashiro2004catalog_notes.md) · [💻 code](papers/yashiro2004catalog/yashiro2004catalog_implementation.ipynb) · [📄 pdf](papers/yashiro2004catalog/yashiro2004catalog_paper.pdf) |
| 35 | The Depiction of Coronal Structure in White-Light Images (NRGF) | 2006 | ✅ | [📝 notes](papers/morgan2006depiction/morgan2006depiction_notes.md) · [💻 code](papers/morgan2006depiction/morgan2006depiction_implementation.ipynb) · [📄 pdf](papers/morgan2006depiction/morgan2006depiction_paper.pdf) |
| 36 | The Undecimated Wavelet Decomposition and Its Reconstruction (à-trous) | 2007 | ✅ | [📝 notes](papers/starck2007undecimated/starck2007undecimated_notes.md) · [💻 code](papers/starck2007undecimated/starck2007undecimated_implementation.ipynb) · [📄 pdf](papers/starck2007undecimated/starck2007undecimated_paper.pdf) |
| 37 | Robust Principal Component Analysis? (RPCA) | 2011 | ✅ | [📝 notes](papers/candes2011robust/candes2011robust_notes.md) · [💻 code](papers/candes2011robust/candes2011robust_implementation.ipynb) · [📄 pdf](papers/candes2011robust/candes2011robust_paper.pdf) |
| 38 | Multi-Scale Gaussian Normalization for Solar Image Processing (MGN) | 2014 | ✅ | [📝 notes](papers/morgan2014multiscale/morgan2014multiscale_notes.md) · [💻 code](papers/morgan2014multiscale/morgan2014multiscale_implementation.ipynb) · [📄 pdf](papers/morgan2014multiscale/morgan2014multiscale_paper.pdf) |
| 39 | SiRGraF: A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images | 2022 | ✅ | [📝 notes](papers/patel2022sirgraf/patel2022sirgraf_notes.md) · [💻 code](papers/patel2022sirgraf/patel2022sirgraf_implementation.ipynb) · [📄 pdf](papers/patel2022sirgraf/patel2022sirgraf_paper.pdf) |
| 40 | LLNet: A Deep Autoencoder Approach to Natural Low-Light Image Enhancement | 2017 | ✅ | [📝 notes](papers/lore2017llnet/lore2017llnet_notes.md) · [💻 code](papers/lore2017llnet/lore2017llnet_implementation.ipynb) · [📄 pdf](papers/lore2017llnet/lore2017llnet_paper.pdf) |
| 41 | Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE) | 2020 | ✅ | [📝 notes](papers/guo2020zeroreference/guo2020zeroreference_notes.md) · [💻 code](papers/guo2020zeroreference/guo2020zeroreference_implementation.ipynb) · [📄 pdf](papers/guo2020zeroreference/guo2020zeroreference_paper.pdf) |
| 42 | Learning Enriched Features for Real Image Restoration and Enhancement (MIRNet) | 2020 | ✅ | [📝 notes](papers/zamir2020learning/zamir2020learning_notes.md) · [💻 code](papers/zamir2020learning/zamir2020learning_implementation.ipynb) · [📄 pdf](papers/zamir2020learning/zamir2020learning_paper.pdf) |
| 43 | EnlightenGAN: Deep Light Enhancement without Paired Supervision | 2021 | ✅ | [📝 notes](papers/jiang2021enlightengan/jiang2021enlightengan_notes.md) · [💻 code](papers/jiang2021enlightengan/jiang2021enlightengan_implementation.ipynb) · [📄 pdf](papers/jiang2021enlightengan/jiang2021enlightengan_paper.pdf) |
| 44 | Toward Fast, Flexible, and Robust Low-Light Image Enhancement (SCI) | 2022 | ✅ | [📝 notes](papers/ma2022toward/ma2022toward_notes.md) · [💻 code](papers/ma2022toward/ma2022toward_implementation.ipynb) · [📄 pdf](papers/ma2022toward/ma2022toward_paper.pdf) |
| 45 | SNR-Aware Low-Light Image Enhancement | 2022 | ✅ | [📝 notes](papers/xu2022snraware/xu2022snraware_notes.md) · [💻 code](papers/xu2022snraware/xu2022snraware_implementation.ipynb) · [📄 pdf](papers/xu2022snraware/xu2022snraware_paper.pdf) |

<a id="helioseismology-asteroseismology"></a>

### Helioseismology & Asteroseismology / 일진동·성진동학 (5 / 7)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | Velocity Fields in the Solar Atmosphere. I. Preliminary Report 🔗 | 1962 | ✅ | [📝 notes](papers/leighton1962velocity/leighton1962velocity_notes.md) · [💻 code](papers/leighton1962velocity/leighton1962velocity_implementation.ipynb) · [📄 pdf](papers/leighton1962velocity/leighton1962velocity_paper.pdf) |
| 2 | The Five-Minute Oscillations on the Solar Surface 🔗 | 1970 | ✅ | [📝 notes](papers/ulrich1970fiveminute/ulrich1970fiveminute_notes.md) · [💻 code](papers/ulrich1970fiveminute/ulrich1970fiveminute_implementation.ipynb) · [📄 pdf](papers/ulrich1970fiveminute/ulrich1970fiveminute_paper.pdf) |
| 3 | Observations of Low Wavenumber Nonradial Eigenmodes of the Sun 🔗 | 1975 | ✅ | [📝 notes](papers/deubner1975observations/deubner1975observations_notes.md) · [💻 code](papers/deubner1975observations/deubner1975observations_implementation.ipynb) · [📄 pdf](papers/deubner1975observations/deubner1975observations_paper.pdf) |
| 4 | The Current State of Solar Modeling 🔗 | 1996 | ✅ | [📝 notes](papers/christensendalsgaard1996current/christensendalsgaard1996current_notes.md) · [💻 code](papers/christensendalsgaard1996current/christensendalsgaard1996current_implementation.ipynb) · [📄 pdf](papers/christensendalsgaard1996current/christensendalsgaard1996current_paper.pdf) |
| 5 | Helioseismic Studies of Differential Rotation in the Solar Envelope by the Solar Oscillations Investigation Using the Michelson Doppler Imager 🔗 | 1998 | ✅ | [📝 notes](papers/schou1998helioseismic/schou1998helioseismic_notes.md) · [💻 code](papers/schou1998helioseismic/schou1998helioseismic_implementation.ipynb) · [📄 pdf](papers/schou1998helioseismic/schou1998helioseismic_paper.pdf) |
| 6 | Solar Models: Current Epoch and Time Dependences, Neutrinos, and Helioseismological Properties 🔗 | 2001 | ⬜ | [📄 pdf](papers/bahcall2001/bahcall2001_paper.pdf) |
| 7 | Anomalously Weak Solar Convection 🔗 | 2012 | ⬜ | [📄 pdf](papers/hanasoge2012/hanasoge2012_paper.pdf) |

<a id="magnetic-reconnection-eruption"></a>

### Magnetic Reconnection & Eruption / 자기재결합·분출 (8 / 10)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | A Theory of Chromospheric Flares 🔗 | 1946 | ✅ | [📝 notes](papers/giovanelli1946theory/giovanelli1946theory_notes.md) · [💻 code](papers/giovanelli1946theory/giovanelli1946theory_implementation.ipynb) · [📄 pdf](papers/giovanelli1946theory/giovanelli1946theory_paper.pdf) |
| 2 | Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids 🔗 | 1957 | ✅ | [📝 notes](papers/parker1957sweets/parker1957sweets_notes.md) · [💻 code](papers/parker1957sweets/parker1957sweets_implementation.ipynb) · [📄 pdf](papers/parker1957sweets/parker1957sweets_paper.pdf) |
| 3 | Magnetic Field Annihilation 🔗 | 1964 | ✅ | [📝 notes](papers/petschek1964magnetic/petschek1964magnetic_notes.md) · [💻 code](papers/petschek1964magnetic/petschek1964magnetic_implementation.ipynb) · [📄 pdf](papers/petschek1964magnetic/petschek1964magnetic_paper.pdf) |
| 4 | Nanoflares and the Solar X-Ray Corona 🔗 | 1988 | ✅ | [📝 notes](papers/parker1988nanoflares/parker1988nanoflares_notes.md) · [💻 code](papers/parker1988nanoflares/parker1988nanoflares_implementation.ipynb) · [📄 pdf](papers/parker1988nanoflares/parker1988nanoflares_paper.pdf) |
| 5 | The Solar Flare Myth 🔗 | 1993 | ✅ | [📝 notes](papers/gosling1993solar/gosling1993solar_notes.md) · [💻 code](papers/gosling1993solar/gosling1993solar_implementation.ipynb) · [📄 pdf](papers/gosling1993solar/gosling1993solar_paper.pdf) |
| 6 | 3-D Magnetic Configurations Supporting Prominences. I. The Natural Presence of Lateral Feet 🔗 | 1998 | ✅ | [📝 notes](papers/aulanier1998magnetic/aulanier1998magnetic_notes.md) · [💻 code](papers/aulanier1998magnetic/aulanier1998magnetic_implementation.ipynb) · [📄 pdf](papers/aulanier1998magnetic/aulanier1998magnetic_paper.pdf) |
| 7 | A Model for Solar Coronal Mass Ejections (Magnetic Breakout) 🔗 | 1999 | ✅ | [📝 notes](papers/antiochos1999model/antiochos1999model_notes.md) · [💻 code](papers/antiochos1999model/antiochos1999model_implementation.ipynb) · [📄 pdf](papers/antiochos1999model/antiochos1999model_paper.pdf) |
| 8 | Torus Instability 🔗 | 2006 | ✅ | [📝 notes](papers/kliem2006torus/kliem2006torus_notes.md) · [💻 code](papers/kliem2006torus/kliem2006torus_implementation.ipynb) · [📄 pdf](papers/kliem2006torus/kliem2006torus_paper.pdf) |
| 9 | Radiative Hydrodynamic Models of the Optical and Ultraviolet Emission from Solar Flares 🔗 | 2005 | ⬜ | [📄 pdf](papers/allred2005/allred2005_paper.pdf) |
| 10 | Global Energetics of Solar Flares 🔗 | 2012 | ⬜ | [📝 notes](papers/emslie2012global/emslie2012global_notes.md) · [💻 code](papers/emslie2012global/emslie2012global_implementation.ipynb) · [📄 pdf](papers/emslie2012global/emslie2012global_paper.pdf) |

<a id="heliosphere-solar-wind"></a>

### Heliosphere & Solar Wind / 태양권·태양풍 (4 / 6)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | Kometenschweife und solare Korpuskularstrahlung (Comet Tails and Solar Corpuscular Radiation) 🔗 | 1951 | ✅ | [📝 notes](papers/biermann1951kometenschweife/biermann1951kometenschweife_notes.md) · [💻 code](papers/biermann1951kometenschweife/biermann1951kometenschweife_implementation.ipynb) · [📄 pdf](papers/biermann1951kometenschweife/biermann1951kometenschweife_paper.pdf) |
| 2 | Dynamics of the Interplanetary Gas and Magnetic Fields 🔗 | 1958 | ✅ | [📝 notes](papers/parker1958dynamics/parker1958dynamics_notes.md) · [💻 code](papers/parker1958dynamics/parker1958dynamics_implementation.ipynb) · [📄 pdf](papers/parker1958dynamics/parker1958dynamics_paper.pdf) |
| 3 | Particle Acceleration at the Sun and in the Heliosphere 🔗 | 1999 | ✅ | [📝 notes](papers/reames1999particle/reames1999particle_notes.md) · [💻 code](papers/reames1999particle/reames1999particle_implementation.ipynb) · [📄 pdf](papers/reames1999particle/reames1999particle_paper.pdf) |
| 4 | Chromospheric Alfvénic Waves Strong Enough to Power the Solar Wind 🔗 | 2007 | ⬜ | [📄 pdf](papers/depontieu2007pontieu/depontieu2007pontieu_paper.pdf) |
| 5 | Parker Solar Probe Enters the Magnetically Dominated Solar Corona 🔗 | 2021 | ⬜ | [📝 notes](papers/kasper2021parker/kasper2021parker_notes.md) · [💻 code](papers/kasper2021parker/kasper2021parker_implementation.ipynb) · [📄 pdf](papers/kasper2021parker/kasper2021parker_paper.pdf) |
| 6 | Identification of a single plasma parcel during a radial alignment of the Parker Solar Probe and Solar Orbiter | 2024 | ✅ | [📝 notes](papers/berriot2024identification/berriot2024identification_notes.md) · [💻 code](papers/berriot2024identification/berriot2024identification_implementation.ipynb) · [📄 pdf](papers/berriot2024identification/berriot2024identification_paper.pdf) |

<a id="plasma-spectroscopy-diagnostics"></a>

### Plasma Spectroscopy & Diagnostics / 플라즈마 분광·진단 (4 / 20)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | The Lower Solar Corona: Interpretation of the Ultraviolet Spectrum | 1963 | ⬜ | — |
| 2 | The Interpretation of Total Line Intensities from Optically Thin Gases. I: A General Method | 1972 | ⬜ | — |
| 3 | The Analysis of XUV Emission Lines | 1975 | ⬜ | — |
| 4 | Fundamental Limitations of X-ray Spectra as Diagnostics of Plasma Temperature Structure | 1976 | ⬜ | — |
| 5 | Spectroscopic Diagnostics in the VUV for Solar and Stellar Plasmas 🔗 | 1994 | ✅ | [📝 notes](papers/mason1994spectroscopic/mason1994spectroscopic_notes.md) · [💻 code](papers/mason1994spectroscopic/mason1994spectroscopic_implementation.ipynb) · [📄 pdf](papers/mason1994spectroscopic/mason1994spectroscopic_paper.pdf) |
| 6 | CHIANTI — An Atomic Database for Emission Lines. I. Wavelengths Greater than 50 Å 🔗 | 1997 | ✅ | [📝 notes](papers/dere1997chianti/dere1997chianti_notes.md) · [💻 code](papers/dere1997chianti/dere1997chianti_implementation.ipynb) · [📄 pdf](papers/dere1997chianti/dere1997chianti_paper.pdf) |
| 7 | CHIANTI — An Atomic Database for Emission Lines. XVII. Version 10.1 🔗 | 2023 | ✅ | [📝 notes](papers/dere2023chianti/dere2023chianti_notes.md) · [💻 code](papers/dere2023chianti/dere2023chianti_implementation.ipynb) · [📄 pdf](papers/dere2023chianti/dere2023chianti_paper.pdf) |
| 8 | Multi-Temperature Analysis of Solar X-Ray Line Emission (Withbroe–Sylwester Method) | 1980 | ⬜ | — |
| 9 | Models for Inner Corona Parameters (Iterative DEM via Arcetri Code) | 1991 | ⬜ | — |
| 10 | Fundamental Limitations of Emission-Line Spectra as Diagnostics of Plasma Temperature and Density Structure | 1997 | ⬜ | — |
| 11 | Markov-Chain Monte Carlo Reconstruction of Emission Measure Distributions | 1998 | ⬜ | — |
| 12 | Temperature Diagnostics with Multichannel Imaging Telescopes (xrt_dem_iterative2) | 2004 | ⬜ | — |
| 13 | Differential Emission Measures from the Regularized Inversion of Hinode and SDO Data (demreg) | 2012 | ⬜ | — |
| 14 | Automated Temperature and Emission Measure Analysis of Coronal Loops with SDO/AIA | 2013 | ⬜ | — |
| 15 | Fast Differential Emission Measure Inversion of Solar Coronal Data | 2013 | ⬜ | — |
| 16 | Thermal Diagnostics with the AIA on SDO: A Validated Method for DEM Inversions (Sparse DEM) 🔗 | 2015 | ✅ | [📝 notes](papers/cheung2015thermal/cheung2015thermal_notes.md) · [💻 code](papers/cheung2015thermal/cheung2015thermal_implementation.ipynb) · [📄 pdf](papers/cheung2015thermal/cheung2015thermal_paper.pdf) |
| 17 | Benchmark Test of Differential Emission Measure Codes | 2015 | ⬜ | — |
| 18 | SITES: Solar Iterative Temperature Emission Solver for DEM Inversion of EUV Observations | 2019 | ⬜ | — |
| 19 | Robust Construction of Differential Emission Measure Profiles using a Regularized Maximum Likelihood Method | 2023 | ⬜ | — |
| 20 | The Chemical Composition of the Sun 🔗 | 2009 | ⬜ | [📄 pdf](papers/asplund2009/asplund2009_paper.pdf) |

<a id="numerical-mhd-simulation"></a>

### Numerical MHD Simulation / 수치 MHD 시뮬레이션 (0 / 1)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | Penumbral Structure and Outflows in Simulated Sunspots 🔗 | 2009 | ⬜ | [📄 pdf](papers/rempel2009/rempel2009_paper.pdf) |

## Language / 언어

- Conversation / 대화: Korean (한국어)
- Documents / 문서: Bilingual Korean/English (한/영 병기)
- Code comments / 코드 주석: Google-style English

## Tools / 도구

- **Languages**: Python, IDL
- **Key libraries**: NumPy, SciPy, Matplotlib, PyTorch, scikit-learn
- **AI assistant**: Claude (Anthropic)
- **MCP servers**: paper-search-mcp, pdf-reader-mcp, jupyter-notebook-mcp ([setup guide](docs/MCP_SETUP.md))

## License / 라이선스

This project is licensed under the [MIT License](LICENSE).
이 프로젝트는 [MIT 라이선스](LICENSE)에 따라 제공됩니다.
