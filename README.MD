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

```
StudyWithAI/
├── CLAUDE.md              # Project rules / 프로젝트 규칙
├── README.MD              # This file / 이 파일
├── docs/
│   ├── WORKFLOW.md        # Detailed workflow / 상세 워크플로우
│   ├── IMPROVEMENTS.md    # Improvement ideas / 개선 아이디어
│   └── MCP_SETUP.md       # MCP server setup / MCP 서버 설정
├── Artificial_Intelligence/
├── Solar_Physics/
├── Space_Weather/
├── Solar_Observation/
└── Living_Reviews_in_Solar_Physics/
```

## Current Progress / 현재 진행 상황

### Artificial Intelligence — Paper Reading / 논문 읽기 (40 / 69)

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | McCulloch & Pitts — "A Logical Calculus of Ideas Immanent in Nervous Activity" | 1943 | ✅ Done |
| 2 | Turing — "Computing Machinery and Intelligence" | 1950 | ✅ Done |
| 3 | Rosenblatt — "The Perceptron" | 1958 | ✅ Done |
| 4 | Minsky & Papert — *Perceptrons* | 1969 | ✅ Done |
| 5 | Hopfield — "Neural Networks and Physical Systems with Emergent Collective Computational Abilities" | 1982 | ✅ Done |
| 6 | Rumelhart, Hinton & Williams — "Learning Representations by Back-propagating Errors" | 1986 | ✅ Done |
| 7 | LeCun et al. — "Backpropagation Applied to Handwritten Zip Code Recognition" | 1989 | ✅ Done |
| 8 | Cortes & Vapnik — "Support-Vector Networks" | 1995 | ✅ Done |
| 9 | Hochreiter & Schmidhuber — "Long Short-Term Memory" | 1997 | ✅ Done |
| 10 | LeCun et al. — "Gradient-Based Learning Applied to Document Recognition" | 1998 | ✅ Done |
| 11 | Breiman — "Random Forests" | 2001 | ✅ Done |
| 12 | Hinton, Osindero & Teh — "A Fast Learning Algorithm for Deep Belief Nets" | 2006 | ✅ Done |
| 13 | Krizhevsky, Sutskever & Hinton — "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet) | 2012 | ✅ Done |
| 14 | Mikolov — "Efficient Estimation of Word Representations in Vector Space (Word2Vec)" | 2013 | ✅ Done |
| 15 | Kingma — "Auto-Encoding Variational Bayes (VAE)" | 2013 | ✅ Done |
| 16 | Goodfellow — "Generative Adversarial Nets (GANs)" | 2014 | ✅ Done |
| 17 | Bahdanau — "Neural Machine Translation by Jointly Learning to Align and Translate" | 2014 | ✅ Done |
| 18 | Kingma — "Adam: A Method for Stochastic Optimization" | 2014 | ✅ Done |
| 19 | Ioffe — "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" | 2015 | ✅ Done |
| 20 | He — "Deep Residual Learning for Image Recognition (ResNet)" | 2015 | ✅ Done |
| 21 | Ba — "Layer Normalization" | 2016 | ✅ Done |
| 22 | Mnih — "Playing Atari with Deep Reinforcement Learning (DQN)" | 2013 | ✅ Done |
| 23 | Silver — "Mastering the game of Go with deep neural networks and tree search (AlphaGo)" | 2016 | ✅ Done |
| 24 | Schulman — "Proximal Policy Optimization Algorithms (PPO)" | 2017 | ✅ Done |
| 25 | Vaswani — "Attention Is All You Need" | 2017 | ✅ Done |
| 26 | Kipf — "Semi-Supervised Classification with Graph Convolutional Networks" | 2017 | ✅ Done |
| 27 | Zoph — "Neural Architecture Search with Reinforcement Learning" | 2017 | ✅ Done |
| 28 | Devlin — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" | 2018 | ✅ Done |
| 29 | Radford — "Language Models are Unsupervised Multitask Learners (GPT-2)" | 2019 | ✅ Done |
| 30 | Frankle — "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" | 2019 | ✅ Done |
| 31 | al. — "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)" | 2020 | ✅ Done |
| 32 | Chen — "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)" | 2020 | ✅ Done |
| 33 | Lewis — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)" | 2020 | ✅ Done |
| 34 | al. — "Language Models are Few-Shot Learners (GPT-3)" | 2020 | ✅ Done |
| 35 | Ho — "Denoising Diffusion Probabilistic Models (DDPM)" | 2020 | ✅ Done |
| 36 | Radford — "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" | 2021 | ✅ Done |
| 37 | Jumper — "Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2)" | 2021 | ✅ Done |
| 38 | al. — "Training Language Models to Follow Instructions with Human Feedback (InstructGPT)" | 2022 | ✅ Done |
| 39 | Kaplan — "Scaling Laws for Neural Language Models" | 2020 | ✅ Done |
| 40 | al. — "Constitutional AI: Harmlessness from AI Feedback" | 2022 | ✅ Done |
| 41–40 | (See [full reading list](Artificial_Intelligence/papers/reading_list.md)) | 2013–2022 | — |

### Solar Physics — Paper Reading / 논문 읽기 (19 / 60)
> _2026-05-01 Phase B 토픽 재정의 후 helioseismology, reconnection, heliosphere, plasma diagnostics 관련 항목은 신규 토픽으로 이관됨. 본 토픽에는 일반 태양 물리, 자기장, 다이나모, 채층/코로나, 흑점, 태양 활동 역사가 남음._

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Galileo — Sunspot observations / 흑점 관측 | 1613 | ✅ Done |
| 2 | Fraunhofer — Solar absorption lines / 태양 흡수선 | 1814 | ✅ Done |
| 3 | Kirchhoff & Bunsen — Spectral analysis / 분광 분석 | 1860 | ✅ Done |
| 4 | Schwarzschild — Radiative equilibrium / 복사 평형 | 1906 | ✅ Done |
| 5 | Hale — Magnetic field in sunspots / 흑점 자기장 | 1908 | ✅ Done |
| 6 | Evershed — Radial movement in sunspots / 흑점 방사 운동 | 1909 | ✅ Done |
| 7 | Hale & Nicholson — Law of sun-spot polarity / 흑점 극성 법칙 | 1925 | ✅ Done |
| 8 | Alfvén — "Existence of Electromagnetic-Hydrodynamic Waves" / 전자기유체역학 파동의 존재 | 1942 | ✅ Done |
| 9 | Babcock — "The Topology of the Sun's Magnetic Field and the 22-Year Cycle" / 태양 다이나모 모델 | 1961 | ✅ Done |
| 10 | Leighton — "A Magneto-Kinematic Model of the Solar Cycle" / 운동학적 다이나모 모델 | 1969 | ✅ Done |
| 11 | Biermann — "Kometenschweife und solare Korpuskularstrahlung" / 혜성 꼬리와 태양 입자 복사 | 1951 | ✅ Done |
| 12 | Parker — "Dynamics of the Interplanetary Gas and Magnetic Fields" / 행성간 가스와 자기장의 역학 | 1958 | ✅ Done |
| 13 | Leighton, Noyes & Simon — "Velocity Fields in the Solar Atmosphere" / 태양 대기의 속도장 | 1962 | ✅ Done |
| 14 | Ulrich — "The Five-Minute Oscillations on the Solar Surface" | 1970 | ✅ Done |
| 15 | Deubner — "Observations of Low Wavenumber Nonradial Eigenmodes of the Sun" | 1975 | ✅ Done |
| 16 | al. — "The Current State of Solar Modeling" | 1996 | ✅ Done |
| 17 | al. — "Helioseismic Studies of Differential Rotation in the Solar Envelope by the Solar Oscillations Investigation Using the Michelson Doppler Imager" | 1998 | ✅ Done |
| 18 | Giovanelli — "A Theory of Chromospheric Flares" | 1946 | ✅ Done |
| 19 | Parker — "Sweet's Mechanism for Merging Magnetic Fields in Conducting Fluids" | 1957 | ✅ Done |
| 20 | Petschek — "Magnetic Field Annihilation" | 1964 | ✅ Done |
| 21 | Parker — "Nanoflares and the Solar X-Ray Corona" | 1988 | ✅ Done |
| 22 | Gosling — "The Solar Flare Myth" | 1993 | ✅ Done |
| 23 | Aulanier — "Magnetic Dips in the Solar Corona and Prominences" | 1998 | ✅ Done |
| 24 | Nakariakov — "Determination of Coronal Loop Parameters from Coronal Seismology" | 1999 | ✅ Done |
| 25 | Antiochos — "A Model for Solar Coronal Mass Ejections" | 1999 | ✅ Done |
| 26 | Kliem — "Torus Instability" | 2006 | ✅ Done |
| 27 | Reames — "Particle Acceleration at the Sun and in the Heliosphere" | 1999 | ✅ Done |
| 28 | Schrijver — "Solar and Stellar Magnetic Activity (monograph)" | 2000 | ✅ Done |
| 29 | al. — "The STEREO Mission: An Introduction" | 2008 | ✅ Done |
| 30 | Wijn — "Small-Scale Solar Magnetic Fields" | 2009 | ✅ Done |
| 31 | Charbonneau — "Dynamo Models of the Solar Cycle" | 2010 | ✅ Done |
| 32 | Kopp — "A New, Lower Value of Total Solar Irradiance: Evidence and Climate Significance" | 2011 | ✅ Done |
| 33 | Pesnell — "The Solar Dynamics Observatory (SDO)" | 2012 | ✅ Done |
| 34 | al. — "Global Energetics of Solar Flares" | 2012 | ✅ Done |
| 35 | Wiegelmann — "Solar Force-Free Magnetic Fields" | 2012 | ✅ Done |
| 36 | al. — "Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of Science Objectives, Instrument Design, Data Products, and Model Developments" | 2012 | ✅ Done |
| 37 | al. — "The Solar Probe Plus Mission: Humanity's First Visit to Our Star" | 2016 | ✅ Done |
| 38 | al. — "Parker Solar Probe Enters the Magnetically Dominated Solar Corona" | 2021 | ✅ Done |
| 39–38 | (See [full reading list](Solar_Physics/papers/reading_list.md)) | 1970–2021 | — |

### Space Weather — Paper Reading / 논문 읽기 (81 / 81)

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Birkeland — "The Norwegian Aurora Polaris Expedition" | 1908 | ✅ Done |
| 2 | Chapman & Ferraro — "A New Theory of Magnetic Storms" | 1931 | ✅ Done |
| 3 | Chapman & Bartels — *Geomagnetism* | 1940 | ✅ Done |
| 4 | Parker — "Dynamics of the Interplanetary Gas and Magnetic Fields" | 1958 | ✅ Done |
| 5 | Van Allen et al. — "Observation of High Intensity Radiation by Satellites" | 1958 | ✅ Done |
| 6 | Dungey — "Interplanetary Magnetic Field and the Auroral Zones" | 1961 | ✅ Done |
| 7 | Axford & Hines — "A Unifying Theory of High-Latitude Geophysical Phenomena" | 1961 | ✅ Done |
| 8 | Akasofu — "The Development of the Auroral Substorm" | 1964 | ✅ Done |
| 9 | Ness — "The Earth's Magnetic Tail" | 1965 | ✅ Done |
| 10 | McPherron, Russell & Aubry — "Phenomenological Model for Substorms" | 1973 | ✅ Done |
| 11 | Burton, McPherron & Russell — "An Empirical Relationship between Interplanetary Conditions and Dst" | 1975 | ✅ Done |
| 12 | Cowley — "The Causes of Convection in the Earth's Magnetosphere" | 1982 | ✅ Done |
| 13 | Richmond & Kamide — "Mapping Electrodynamic Features of the High-Latitude Ionosphere (AMIE)" | 1988 | ✅ Done |
| 14 | Tsyganenko — "A Magnetospheric Magnetic Field Model with a Warped Tail Current Sheet" | 1989 | ✅ Done |
| 15 | Gonzalez — "What Is a Geomagnetic Storm?" | 1994 | ✅ Done |
| 16 | Fuller-Rowell — "Response of the Thermosphere and Ionosphere to Geomagnetic Storms" | 1994 | ✅ Done |
| 17 | Allen — "Effects of the March 1989 Solar Activity" | 1989 | ✅ Done |
| 18 | Arge — "Improvement in the Prediction of Solar Wind Conditions Using Near-Real Time Solar Magnetic Field Updates (WSA)" | 2000 | ✅ Done |
| 19 | Baker — "Effects of the Sun on the Earth's Environment and the Consequences for Mankind" | 2000 | ✅ Done |
| 20 | Kintner — "GPS and Ionospheric Scintillations" | 2007 | ✅ Done |
| 21 | Odstrcil — "Modeling 3-D Solar Wind Structure" | 2003 | ✅ Done |
| 22 | Tsurutani — "The Global Dayside Ionospheric Uplift and Enhancement Associated with Interplanetary Electric Fields" | 2004 | ✅ Done |
| 23 | Gopalswamy — "Introduction to Violent Sun–Earth Connection Events of October–November 2003" | 2005 | ✅ Done |
| 24 | Pulkkinen — "Space Weather: Terrestrial Perspective" | 2007 | ✅ Done |
| 25 | Kappenman — "Geomagnetic Storms and Their Impacts on the U.S. Power Grid" | 2010 | ✅ Done |
| 26 | Thorne — "Radiation Belt Dynamics: The Importance of Wave-Particle Interactions" | 2010 | ✅ Done |
| 27 | Angelopoulos — "The THEMIS Mission" | 2008 | ✅ Done |
| 28 | Mauk — "Science Objectives and Rationale for the Radiation Belt Storm Probes Mission (Van Allen Probes)" | 2013 | ✅ Done |
| 29 | al. — "A Major Solar Eruptive Event in July 2012: Defining Extreme Space Weather Scenarios" | 2013 | ✅ Done |
| 30 | Baker — "An Impenetrable Barrier to Ultrarelativistic Electrons in the Van Allen Radiation Belts" | 2014 | ✅ Done |
| 31 | al. — "Space Weather Operations: The NOAA Space Weather Prediction Center" | 2015 | ✅ Done |
| 32 | al. — "Geomagnetically Induced Currents: Science, Engineering, and Applications Readiness" | 2017 | ✅ Done |
| 33 | Camporeale — "The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting" | 2019 | ✅ Done |
| 34 | Leka — "A Comparison of Flare Forecasting Methods" | 2019 | ✅ Done |
| 35 | al. — "Solar Wind Prediction Using Deep Learning" | 2020 | ✅ Done |
| 36–60 | (See [full reading list](Space_Weather/papers/reading_list.md)) | 1989–2026 | — |

### Solar Observation — Paper Reading / 논문 읽기 (61 / 80)
> _2026-05-01 DEM·CHIANTI·atomic data 관련 항목 (Phase 7 일부, Phase 8 전체) 은 Plasma_Spectroscopy_Diagnostics 로 이관됨._

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Pierce — "The McMath Solar Telescope of Kitt Peak National Observatory" | 1964 | ✅ Done |
| 2 | Dunn — "An Evacuated Tower Telescope" | 1964 | ✅ Done |
| 3 | Scharmer et al. — "The 1-meter Swedish Solar Telescope" | 2003 | ✅ Done |
| 4 | Goode — "The New Solar Telescope at Big Bear Solar Observatory" | 2012 | ✅ Done |
| 5 | Harvey — "The Global Oscillation Network Group (GONG) Project" | 1996 | ✅ Done |
| 6 | Chaplin — "BiSON — The Birmingham Solar Oscillations Network" | 1996 | ✅ Done |
| 7 | Tomczyk — "Scientific Objectives and Capabilities of the Coronal Solar Magnetism Observatory" | 2016 | ✅ Done |
| 8 | Domingo — "The SOHO Mission: An Overview" | 1995 | ✅ Done |
| 9 | Delaboudiniere — "EIT: Extreme-Ultraviolet Imaging Telescope for the SOHO Mission" | 1995 | ✅ Done |
| 10 | Brueckner — "The Large Angle Spectroscopic Coronagraph (LASCO)" | 1995 | ✅ Done |
| 11 | al. — "The Transition Region and Coronal Explorer" | 1999 | ✅ Done |
| 12 | Lemen — "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)" | 2012 | ✅ Done |
| 13 | Scherrer — "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)" | 2012 | ✅ Done |
| 14 | Tsuneta — "The Solar Optical Telescope for the Hinode Mission: An Overview" | 2008 | ✅ Done |
| 15 | Culhane — "The EUV Imaging Spectrometer for Hinode" | 2007 | ✅ Done |
| 16–40 | (See [full reading list](Solar_Observation/papers/reading_list.md)) | 2012–2022 | — |

### Living Reviews in Solar Physics — Paper Reading / 논문 읽기 (87 / 87+)

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Wood — "Astrospheres and Solar-like Stellar Winds" | 2004 | ✅ Done |
| 2 | Miesch — "Large-Scale Dynamics of the Convection Zone and Tachocline" | 2005 | ✅ Done |
| 3 | Nakariakov & Verwichte — "Coronal Waves and Oscillations" | 2005 | ✅ Done |
| 4 | Sheeley — "Surface Evolution of the Sun's Magnetic Field" | 2005 | ✅ Done |
| 5 | Gizon & Birch — "Local Helioseismology" | 2005 | ✅ Done |
| 6 | Longcope — "Topological Methods for the Analysis of Solar Magnetic Fields" | 2005 | ✅ Done |
| 7 | Berdyugina — "Starspots: A Key to the Stellar Dynamo" | 2005 | ✅ Done |
| 8 | Marsch — "Kinetic Physics of the Solar Corona and Solar Wind" | 2006 | ✅ Done |
| 9 | Schwenn — "Space Weather: The Solar Perspective" | 2006 | ✅ Done |
| 10 | Pulkkinen — "Space Weather: Terrestrial Perspective" | 2007 | ✅ Done |
| 11 | Haigh — "The Sun and the Earth's Climate" | 2007 | ✅ Done |
| 12 | Güdel — "The Sun in Time: Activity and Environment" | 2007 | ✅ Done |
| 13 | Benz — "Flare Observations" | 2008 | ✅ Done |
| 14 | Hall — "Stellar Chromospheric Activity" | 2008 | ✅ Done |
| 15 | Howe — "Solar Interior Rotation and its Variation" | 2009 | ✅ Done |
| 16 | Nordlund — "Solar Surface Convection" | 2009 | ✅ Done |
| 17 | Cranmer — "Coronal Holes" | 2009 | ✅ Done |
| 18 | Fan — "Magnetic Fields in the Solar Convection Zone" | 2009 | ✅ Done |
| 19 | Rieutord — "The Sun's Supergranulation" | 2010 | ✅ Done |
| 20 | Charbonneau — "Dynamo Models of the Solar Cycle" | 2010 | ✅ Done |
| 21 | Ofman — "Wave Modeling of the Solar Wind" | 2010 | ✅ Done |
| 22 | Chen — "Coronal Mass Ejections: Models and Their Observational Basis" | 2011 | ✅ Done |
| 23 | Rimmele — "Solar Adaptive Optics" | 2011 | ✅ Done |
| 24 | Rempel — "Sunspot Modeling: From Simplified Models to Radiative MHD Simulations" | 2011 | ✅ Done |
| 25 | Borrero — "Magnetic Structure of Sunspots" | 2011 | ✅ Done |
| 26 | Aschwanden — "Solar Stereoscopy and Tomography" | 2011 | ✅ Done |
| 27 | Shibata — "Solar Flares: Magnetohydrodynamic Processes" | 2011 | ✅ Done |
| 28 | Reiners — "Observations of Cool-Star Magnetic Fields" | 2012 | ✅ Done |
| 29 | Webb — "Coronal Mass Ejections: Observations" | 2012 | ✅ Done |
| 30 | Stein — "Solar Surface Magneto-Convection" | 2012 | ✅ Done |
| 31 | Mackay — "The Sun's Global Photospheric and Coronal Magnetic Fields: Observations and Models" | 2012 | ✅ Done |
| 32 | Bruno — "The Solar Wind as a Turbulence Laboratory" | 2013 | ✅ Done |
| 33 | Potgieter — "Solar Modulation of Cosmic Rays" | 2013 | ✅ Done |
| 34 | Lockwood — "Reconstruction and Prediction of Variations in the Open Solar Magnetic Flux and Interplanetary Conditions" | 2013 | ✅ Done |
| 35 | Owens — "The Heliospheric Magnetic Field" | 2013 | ✅ Done |
| 36 | Parenti — "Solar Prominences: Observations" | 2014 | ✅ Done |
| 37 | Penn — "Infrared Solar Physics" | 2014 | ✅ Done |
| 38 | Cheung — "Flux Emergence (Theory)" | 2014 | ✅ Done |
| 39 | Reale — "Coronal Loops: Observations and Modeling of Confined Plasma" | 2014 | ✅ Done |
| 40 | Driel-Gesztelyi — "Evolution of Active Regions" | 2015 | ✅ Done |
| 41 | Laming — "The FIP and Inverse FIP Effects in Solar and Stellar Coronae" | 2015 | ✅ Done |
| 42 | Warmuth — "Large-scale Globally Propagating Coronal Waves" | 2015 | ✅ Done |
| 43 | Hathaway — "The Solar Cycle" | 2015 | ✅ Done |
| 44 | Petrie — "Solar Magnetism in the Polar Regions" | 2015 | ✅ Done |
| 45 | Khomenko — "Oscillations and Waves in Sunspots" | 2015 | ✅ Done |
| 46 | Poletto — "Solar Coronal Plumes" | 2015 | ✅ Done |
| 47 | Houdek — "Interaction Between Convection and Pulsation" | 2015 | ✅ Done |
| 48 | Prieto — "Solar and Stellar Photospheric Abundances" | 2016 | ✅ Done |
| 49 | Basu — "Global Seismology of the Sun" | 2016 | ✅ Done |
| 50 | Desai — "Large Gradual Solar Energetic Particle Events" | 2016 | ✅ Done |
| 51 | Iniesta — "Inversion of the Radiative Transfer Equation for Polarized Light" | 2016 | ✅ Done |
| 52 | Usoskin — "A History of Solar Activity over Millennia" | 2017 | ✅ Done |
| 53 | Brun — "Magnetism, Dynamo Action and the Solar-Stellar Connection" | 2017 | ✅ Done |
| 54 | Kilpua — "Coronal Mass Ejections and Their Sheath Regions in Interplanetary Space" | 2017 | ✅ Done |
| 55 | Richardson — "Solar Wind Stream Interaction Regions Throughout the Heliosphere" | 2018 | ✅ Done |
| 56 | Narita — "Space–Time Structure and Wavevector Anisotropy in Space Plasma Turbulence" | 2018 | ✅ Done |
| 57 | Arregui — "Prominence Oscillations" | 2018 | ✅ Done |
| 58 | Gombosi — "Extended MHD Modeling of the Steady Solar Corona and the Solar Wind" | 2018 | ✅ Done |
| 59 | Zanna — "Solar UV and X-ray Spectral Diagnostics" | 2018 | ✅ Done |
| 60 | Rincon — "The Sun's Supergranulation (update)" | 2018 | ✅ Done |
| 61 | Gibson — "Solar Prominences: Theory and Models" | 2018 | ✅ Done |
| 62 | Rubio — "Quiet Sun Magnetic Fields: An Observational View" | 2019 | ✅ Done |
| 63 | Madjarska — "Coronal Bright Points" | 2019 | ✅ Done |
| 64 | Toriumi — "Flare-Productive Active Regions" | 2019 | ✅ Done |
| 65 | García — "Asteroseismology of Solar-Type Stars" | 2019 | ✅ Done |
| 66 | Verscharen — "The Multi-Scale Nature of the Solar Wind" | 2019 | ✅ Done |
| 67 | Arlt — "Historical Sunspot Records" | 2020 | ✅ Done |
| 68 | Petrovay — "Solar Cycle Prediction" | 2020 | ✅ Done |
| 69 | Pontin — "The Parker Problem: Existence of Smooth Force-Free Fields and Coronal Heating" | 2020 | ✅ Done |
| 70 | Leenaarts — "Radiation Hydrodynamics in Simulations of the Solar Atmosphere" | 2020 | ✅ Done |
| 71 | Charbonneau — "Dynamo Models of the Solar Cycle (update)" | 2020 | ✅ Done |
| 72 | Wiegelmann — "Solar Force-Free Magnetic Fields" | 2021 | ✅ Done |
| 73 | Christensen-Dalsgaard — "Solar Structure and Evolution" | 2021 | ✅ Done |
| 74 | Vidotto — "The Evolution of the Solar Wind" | 2021 | ✅ Done |
| 75 | Temmer — "Space Weather: The Solar Perspective (update)" | 2021 | ✅ Done |
| 76 | Fan — "Magnetic Fields in the Solar Convection Zone (update)" | 2021 | ✅ Done |
| 77 | Pontin — "Magnetic Reconnection: MHD Theory and Modelling" | 2022 | ✅ Done |
| 78 | Cliver — "Extreme Solar Events" | 2022 | ✅ Done |
| 79 | Hanasoge — "Surface and Interior Meridional Circulation in the Sun" | 2022 | ✅ Done |
| 80 | Jess — "Waves in the Lower Solar Atmosphere: The Dawn of Next-Generation Solar Telescopes" | 2023 | ✅ Done |
| 81 | Usoskin — "A History of Solar Activity over Millennia (update)" | 2023 | ✅ Done |
| 82 | Karak — "Models for the Long-Term Variations of Solar Activity" | 2023 | ✅ Done |
| 83 | Ramos — "Machine Learning in Solar Physics" | 2023 | ✅ Done |
| 84 | Kowalski — "Stellar Flares" | 2024 | ✅ Done |
| 85 | Kopp — "Solar Irradiance Measurements" | 2025 | ✅ Done |
| 86 | Veronig — "Coronal Dimmings and What They Tell Us About Solar and Stellar Coronal Mass Ejections" | 2025 | ✅ Done |
| 87 | Keppens — "Modeling Multiphase Plasma in the Corona: Prominences and Rain" | 2025 | ✅ Done |
| 88–87 | (See [full reading list](Living_Reviews_in_Solar_Physics/papers/reading_list.md)) | 2008–2025 | — |

### Low-SNR Imaging — Paper Reading / 논문 읽기 (0 / 45)

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Donoho & Johnstone — *Ideal Spatial Adaptation by Wavelet Shrinkage (VisuShrink)* | 1994 | ⬜ Not started |
| 2 | Donoho & Johnstone — *Adapting to Unknown Smoothness via Wavelet Shrinkage (SureShrink)* | 1995 | ⬜ Not started |
| 3 | Chang, Yu & Vetterli — *Adaptive Wavelet Thresholding (BayesShrink)* | 2000 | ⬜ Not started |
| 4 | Buades, Coll & Morel — *A Non-Local Algorithm for Image Denoising (NLM)* | 2005 | ⬜ Not started |
| 5 | Do & Vetterli — *The Contourlet Transform* | 2005 | ⬜ Not started |
| 6 | Candès et al. — *Fast Discrete Curvelet Transforms* | 2006 | ⬜ Not started |
| 7 | Dabov et al. — *BM3D — Sparse 3-D Transform-Domain Collaborative Filtering* | 2007 | ⬜ Not started |
| 8 | Easley, Labate & Lim — *Discrete Shearlet Transform* | 2008 | ⬜ Not started |
| 9 | Maggioni et al. — *V-BM4D — Spatiotemporal Video Denoising* | 2012 | ⬜ Not started |
| 10 | Maggioni et al. — *BM4D — Volumetric Denoising* | 2013 | ⬜ Not started |
| 11 | Anscombe — *The Transformation of Poisson, Binomial and Negative-Binomial Data* | 1948 | ⬜ Not started |
| 12 | Deledalle et al. — *Poisson NL Means* | 2010 | ⬜ Not started |
| 13 | Luisier, Blu & Unser — *PURE-LET (Poisson–Gaussian)* | 2011 | ⬜ Not started |
| 14 | Mäkitalo & Foi — *Optimal Generalized Anscombe Inversion* | 2013 | ⬜ Not started |
| 15 | Buchholz et al. — *Cryo-CARE* | 2019 | ⬜ Not started |
| 16 | Lehtinen et al. — *Noise2Noise* | 2018 | ⬜ Not started |
| 17 | Krull, Buchholz & Jug — *Noise2Void* | 2019 | ⬜ Not started |
| 18 | Batson & Royer — *Noise2Self* | 2019 | ⬜ Not started |
| 19 | Quan et al. — *Self2Self with Dropout* | 2020 | ⬜ Not started |
| 20 | Huang et al. — *Neighbor2Neighbor* | 2021 | ⬜ Not started |
| 21 | Pang et al. — *R2R (Recorrupted-to-Recorrupted)* | 2021 | ⬜ Not started |
| 22 | Wang et al. — *Blind2Unblind* | 2022 | ⬜ Not started |
| 23 | van Dokkum — *L.A.Cosmic — Cosmic-Ray Rejection* | 2001 | ⬜ Not started |
| 24 | Zhang & Bloom — *deepCR* | 2020 | ⬜ Not started |
| 25 | Kadkhodaie & Simoncelli — *Stochastic Solutions via Implicit Denoiser Prior* | 2021 | ⬜ Not started |
| 26 | Kawar et al. — *DDRM* | 2022 | ⬜ Not started |
| 27 | Bansal et al. — *Cold Diffusion* | 2023 | ⬜ Not started |
| 28 | Chung et al. — *DPS — Diffusion Posterior Sampling* | 2023 | ⬜ Not started |
| 29 | Daras et al. — *Ambient Diffusion* | 2023 | ⬜ Not started |
| 30 | Zhu et al. — *DiffPIR* | 2023 | ⬜ Not started |
| 31 | Jiang et al. — *DiffLL — Wavelet-Domain Low-Light Diffusion* | 2023 | ⬜ Not started |
| 32 | Stenborg & Cobelli — *Wavelet Packets Equalization for Coronal Structures* | 2003 | ⬜ Not started |
| 33 | Starck et al. — *Curvelet Contrast Enhancement* | 2003 | ⬜ Not started |
| 34 | Yashiro et al. — *CDAW LASCO CME Catalog* | 2004 | ⬜ Not started |
| 35 | Morgan, Habbal & Woo — *NRGF — Normalizing Radial Graded Filter* | 2006 | ⬜ Not started |
| 36 | Starck, Fadili & Murtagh — *Undecimated Wavelet (à-trous)* | 2007 | ⬜ Not started |
| 37 | Candès et al. — *Robust PCA* | 2011 | ⬜ Not started |
| 38 | Morgan & Druckmüller — *MGN — Multi-Scale Gaussian Normalization* | 2014 | ⬜ Not started |
| 39 | Patel et al. — *SiRGraF — Simple Radial Gradient Filter* | 2022 | ⬜ Not started |
| 40 | Lore, Akintayo & Sarkar — *LLNet* | 2017 | ⬜ Not started |
| 41 | Guo et al. — *Zero-DCE* | 2020 | ⬜ Not started |
| 42 | Zamir et al. — *MIRNet* | 2020 | ⬜ Not started |
| 43 | Jiang et al. — *EnlightenGAN* | 2021 | ⬜ Not started |
| 44 | Ma et al. — *SCI — Self-Calibrated Illumination* | 2022 | ⬜ Not started |
| 45 | Xu et al. — *SNR-Aware Low-Light Enhancement* | 2022 | ⬜ Not started |

### Helioseismology & Asteroseismology — Paper Reading / 논문 읽기 (5 / 7)
> _Phase B 이관 (2026-05-01) 후 Solar_Physics 에서 7편 이관. 자세한 목록은 [reading_list](Helioseismology_Asteroseismology/papers/reading_list.md) 참조._

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Leighton+ — *Velocity Fields in the Solar Atmosphere (5-min oscillation)* | 1962 | ✅ Done |
| 2 | Ulrich — *The Five-Minute Oscillations on the Solar Surface* | 1970 | ✅ Done |
| 3 | Deubner — *Observations of Low Wavenumber Nonradial Eigenmodes of the Sun* | 1975 | ✅ Done |
| 4 | Christensen-Dalsgaard+ — *The Current State of Solar Modeling* | 1996 | ✅ Done |
| 5 | Schou+ — *Helioseismic Studies of Differential Rotation (MDI)* | 1998 | ✅ Done |
| 6 | Bahcall+ — *Solar Models: Neutrinos and Helioseismological Properties* | 2001 | ⬜ Not started |
| 7 | Hanasoge+ — *Anomalously Weak Solar Convection* | 2012 | ⬜ Not started |

### Magnetic Reconnection & Eruption — Paper Reading / 논문 읽기 (8 / 10)
> _Phase B 이관 (2026-05-01) 후 Solar_Physics 에서 10편 이관. 자세한 목록은 [reading_list](Magnetic_Reconnection_Eruption/papers/reading_list.md) 참조._

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Giovanelli — *A Theory of Chromospheric Flares* | 1946 | ✅ Done |
| 2 | Parker — *Sweet's Mechanism (Sweet-Parker Reconnection)* | 1957 | ✅ Done |
| 3 | Petschek — *Magnetic Field Annihilation (Fast Reconnection)* | 1964 | ✅ Done |
| 4 | Parker — *Nanoflares and the Solar X-Ray Corona* | 1988 | ✅ Done |
| 5 | Gosling — *The Solar Flare Myth* | 1993 | ✅ Done |
| 6 | Aulanier & Démoulin — *3-D Magnetic Configurations Supporting Prominences* | 1998 | ✅ Done |
| 7 | Antiochos+ — *A Model for Solar Coronal Mass Ejections (Breakout)* | 1999 | ✅ Done |
| 8 | Kliem & Török — *Torus Instability* | 2006 | ✅ Done |
| 9 | Allred+ — *Radiative Hydrodynamic Models of Flare Emission (RADYN)* | 2005 | ⬜ Not started |
| 10 | Emslie+ — *Global Energetics of Solar Flares* | 2012 | ⬜ Not started |

### Heliosphere & Solar Wind — Paper Reading / 논문 읽기 (3 / 5)
> _Phase B 이관 (2026-05-01) 후 Solar_Physics 에서 5편 이관. PSP/Solar Orbiter 미션·기기 paper 는 Solar_Observation 유지 (Decision 3). 자세한 목록은 [reading_list](Heliosphere_Solar_Wind/papers/reading_list.md)._

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Biermann — *Comet Tails and Solar Corpuscular Radiation* | 1951 | ✅ Done |
| 2 | Parker — *Dynamics of the Interplanetary Gas (Solar Wind Theory)* | 1958 | ✅ Done |
| 3 | Reames — *Particle Acceleration at the Sun and in the Heliosphere* | 1999 | ✅ Done |
| 4 | De Pontieu+ — *Chromospheric Alfvénic Waves Powering the Solar Wind* | 2007 | ⬜ Not started |
| 5 | Kasper+ — *Parker Solar Probe Enters the Magnetically Dominated Solar Corona* | 2021 | ⬜ Not started |

### Plasma Spectroscopy & Diagnostics — Paper Reading / 논문 읽기 (4 / 20)
> _Phase B 이관 (2026-05-01) 후 Solar_Observation Phase 7–8 (#30, #63–80) 19편 + Solar_Physics #54 (Asplund 2009) 이관. 자세한 목록은 [reading_list](Plasma_Spectroscopy_Diagnostics/papers/reading_list.md)._

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Pottasch — *Lower Solar Corona: UV Spectrum Interpretation (EM foundation)* | 1963 | ⬜ Not started |
| 2 | Jefferies+Orrall+Zirker — *Total Line Intensities from Optically Thin Gases (Bivariate DEM)* | 1972 | ⬜ Not started |
| 3 | Withbroe — *The Analysis of XUV Emission Lines (modern DEM formalism)* | 1975 | ⬜ Not started |
| 4 | Craig & Brown — *Fundamental Limitations of X-ray Spectra (ill-posed proof)* | 1976 | ⬜ Not started |
| 5 | Mason & Monsignori-Fossi — *Spectroscopic Diagnostics in the VUV* | 1994 | ✅ Done |
| 6 | Dere+ — *CHIANTI v1: An Atomic Database for Emission Lines* | 1997 | ✅ Done |
| 7 | Dere+ — *CHIANTI v10.1 Update* | 2023 | ✅ Done |
| 16 | Cheung+ — *Sparse DEM for AIA (validated method)* | 2015 | ✅ Done |
| 8–19 | DEM algorithms (Sylwester 1980, Monsignori-Fossi 1991, Judge 1997, Kashyap-Drake 1998, Weber 2004, Hannah-Kontar 2012, Aschwanden 2013/2015, Plowman 2013, Morgan-Pickering 2019, Massa 2023) | 1980–2023 | ⬜ Not started (except #16) |
| 20 | Asplund+ — *The Chemical Composition of the Sun* | 2009 | ⬜ Not started |

### Numerical MHD Simulation — Paper Reading / 논문 읽기 (0 / 1)
> _Phase B 이관 (2026-05-01) 후 Solar_Physics #55 Rempel 2009 이관. LRSP 의 MHD-simulation 종설 다수 cross-reference. 신규 method paper 큐레이션 진행 중. [reading_list](Numerical_MHD_Simulation/papers/reading_list.md)._

| # | Paper / 논문 | Year / 연도 | Status / 상태 |
|---|---|---|---|
| 1 | Rempel+ — *Penumbral Structure and Outflows in Simulated Sunspots* | 2009 | ⬜ Not started |

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
