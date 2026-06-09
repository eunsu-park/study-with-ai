# AI Paper Reading List / AI 논문 읽기 목록

A curated list of landmark papers in Artificial Intelligence, organized by conceptual flow.
Each paper builds on the concepts introduced by earlier ones.

인공지능의 주요 논문을 개념 흐름에 따라 정리한 목록입니다.
각 논문은 이전 논문에서 소개된 개념을 기반으로 합니다.

---

## Phase 1: Foundations (1943–1969) / 기초 (1943–1969)

### 1. A Logical Calculus of the Ideas Immanent in Nervous Activity
- **Authors**: Warren McCulloch, Walter Pitts
- **Year**: 1943
- **DOI**: 10.1007/BF02478259
- **Why it matters**: The very first mathematical model of an artificial neuron. Showed that networks of simple binary threshold units can compute any logical function — laying the theoretical foundation for all neural networks. / 최초의 인공 뉴런 수학적 모델입니다. 단순한 이진 임계값 유닛의 네트워크가 모든 논리 함수를 계산할 수 있음을 보여주었으며, 모든 신경망의 이론적 기초를 놓았습니다.
- **Prerequisites**: Basic logic (AND, OR, NOT gates) / 기초 논리학 (AND, OR, NOT 게이트)
- **Status**: [x]

### 2. Computing Machinery and Intelligence
- **Authors**: Alan Turing
- **Year**: 1950
- **DOI**: 10.1093/mind/LIX.236.433
- **Why it matters**: Posed the fundamental question "Can machines think?" and proposed the Turing Test. Anticipated and addressed major objections to machine intelligence decades before they became mainstream debates. / "기계가 생각할 수 있는가?"라는 근본적인 질문을 던지고 Turing Test를 제안했습니다. 기계 지능에 대한 주요 반론을 수십 년 전에 예견하고 다루었습니다.
- **Prerequisites**: None (philosophical paper, highly accessible) / 없음 (철학 논문, 접근성 높음)
- **Status**: [x]

### 3. The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain
- **Authors**: Frank Rosenblatt
- **Year**: 1958
- **DOI**: 10.1037/h0042519
- **Why it matters**: Introduced the perceptron — the first trainable neural network model. Demonstrated that a machine could learn from data rather than being explicitly programmed, sparking enormous excitement about AI. / Perceptron을 소개했습니다 — 최초의 학습 가능한 신경망 모델입니다. 기계가 명시적으로 프로그래밍되지 않고 데이터로부터 학습할 수 있음을 보여주어, AI에 대한 엄청난 기대를 불러일으켰습니다.
- **Prerequisites**: Paper #1; basic linear algebra (vectors, dot products) / 논문 #1; 기초 선형대수학 (벡터, 내적)
- **Status**: [x]

### 4. Perceptrons (book)
- **Authors**: Marvin Minsky, Seymour Papert
- **Year**: 1969
- **Why it matters**: Rigorously proved the limitations of single-layer perceptrons (e.g., inability to learn XOR). Widely credited with triggering the first "AI winter" by dampening enthusiasm for neural networks. Understanding these limitations motivates multi-layer networks. / 단층 perceptron의 한계(예: XOR 학습 불가)를 엄밀히 증명했습니다. 신경망에 대한 열기를 식혀 첫 번째 "AI 겨울"을 촉발한 것으로 널리 알려져 있습니다. 이러한 한계를 이해하는 것이 다층 네트워크의 동기가 됩니다.
- **Prerequisites**: Paper #3; basic set theory and logic / 논문 #3; 기초 집합론 및 논리학
- **Status**: [x]

---

## Phase 2: Revival and Classical Machine Learning (1982–2001) / 부흥 및 고전 기계학습 (1982–2001)

### 5. Neural Networks and Physical Systems with Emergent Collective Computational Abilities
- **Authors**: John Hopfield
- **Year**: 1982
- **DOI**: 10.1073/pnas.79.8.2554
- **Why it matters**: Introduced Hopfield networks — recurrent networks with an energy function that can store and retrieve patterns (associative memory). Bridged physics and neural computation, reigniting interest in neural networks. / Hopfield 네트워크를 소개했습니다 — 패턴을 저장하고 검색할 수 있는 에너지 함수를 가진 순환 네트워크(연상 기억)입니다. 물리학과 신경 계산을 연결하여 신경망에 대한 관심을 다시 불러일으켰습니다.
- **Prerequisites**: Papers #1, #3; basic physics concepts (energy minimization) / 논문 #1, #3; 기초 물리학 개념 (에너지 최소화)
- **Status**: [x]

### 6. Learning Representations by Back-propagating Errors
- **Authors**: David Rumelhart, Geoffrey Hinton, Ronald Williams
- **Year**: 1986
- **DOI**: 10.1038/323533a0
- **Why it matters**: Popularized backpropagation — the algorithm that makes training multi-layer neural networks practical. This single paper enabled the entire field of deep learning. It directly addressed the limitations Minsky & Papert identified. / Backpropagation을 대중화했습니다 — 다층 신경망 학습을 실용적으로 만든 알고리즘입니다. 이 단일 논문이 딥러닝 분야 전체를 가능하게 했습니다. Minsky & Papert이 밝힌 한계를 직접 해결했습니다.
- **Prerequisites**: Paper #4; calculus (chain rule), linear algebra (matrix multiplication) / 논문 #4; 미적분학 (연쇄 법칙), 선형대수학 (행렬 곱셈)
- **Status**: [x]

### 7. Backpropagation Applied to Handwritten Zip Code Recognition
- **Authors**: Yann LeCun et al.
- **Year**: 1989
- **DOI**: 10.1162/neco.1989.1.4.541
- **Why it matters**: First successful application of backpropagation to a real-world problem using convolutional neural networks (CNNs). Introduced weight sharing and local receptive fields — the architecture that now dominates computer vision. / Convolutional neural network(CNN)을 사용하여 backpropagation을 실제 문제에 최초로 성공적으로 적용했습니다. 가중치 공유와 국소 수용 영역을 도입했으며, 이 아키텍처가 현재 컴퓨터 비전을 지배하고 있습니다.
- **Prerequisites**: Paper #6; convolution operation basics / 논문 #6; 합성곱 연산 기초
- **Status**: [x]

### 8. Support-Vector Networks
- **Authors**: Corinna Cortes, Vladimir Vapnik
- **Year**: 1995
- **DOI**: 10.1007/BF00994018
- **Why it matters**: Introduced Support Vector Machines (SVMs) with the kernel trick. SVMs dominated machine learning for over a decade and brought rigorous statistical learning theory into practice. Understanding SVMs gives crucial insight into margins, generalization, and the bias-variance tradeoff. / Kernel trick을 적용한 Support Vector Machine(SVM)을 소개했습니다. SVM은 10년 이상 기계학습을 지배했으며 엄밀한 통계적 학습 이론을 실무에 도입했습니다. SVM을 이해하면 마진, 일반화, 편향-분산 트레이드오프에 대한 중요한 통찰을 얻을 수 있습니다.
- **Prerequisites**: Linear algebra, basic optimization (Lagrange multipliers) / 선형대수학, 기초 최적화 (Lagrange 승수법)
- **Status**: [x]

### 9. Long Short-Term Memory
- **Authors**: Sepp Hochreiter, Jurgen Schmidhuber
- **Year**: 1997
- **DOI**: 10.1162/neco.1997.9.8.1735
- **Why it matters**: Solved the vanishing gradient problem in recurrent neural networks by introducing gating mechanisms. LSTMs became the dominant architecture for sequence modeling (text, speech, time series) for nearly two decades. / 게이팅 메커니즘을 도입하여 순환 신경망의 기울기 소실 문제를 해결했습니다. LSTM은 거의 20년간 시퀀스 모델링(텍스트, 음성, 시계열)의 지배적 아키텍처가 되었습니다.
- **Prerequisites**: Paper #6; RNN basics (understanding of vanishing gradients) / 논문 #6; RNN 기초 (기울기 소실에 대한 이해)
- **Status**: [x]

### 10. Gradient-Based Learning Applied to Document Recognition
- **Authors**: Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner
- **Year**: 1998
- **DOI**: 10.1109/5.726791
- **Why it matters**: The definitive CNN paper. Presented LeNet-5 in full detail with thorough experiments on handwritten digit recognition. Also introduced the concept of end-to-end trainable systems and graph transformer networks. / 결정적인 CNN 논문입니다. LeNet-5를 상세히 제시하고 필기 숫자 인식에 대한 철저한 실험을 수행했습니다. 또한 종단 간 학습 가능한 시스템과 graph transformer network 개념을 도입했습니다.
- **Prerequisites**: Papers #6, #7; convolution operations / 논문 #6, #7; 합성곱 연산
- **Status**: [x]

### 11. Random Forests
- **Authors**: Leo Breiman
- **Year**: 2001
- **DOI**: 10.1023/A:1010933404324
- **Why it matters**: Introduced Random Forests — an ensemble method combining decision trees with bagging and random feature selection. Remains one of the most robust and widely-used algorithms in practice, especially for tabular data. / Random Forest를 소개했습니다 — 의사결정 트리에 배깅과 랜덤 특징 선택을 결합한 앙상블 방법입니다. 특히 테이블형 데이터에서 가장 견고하고 널리 사용되는 알고리즘 중 하나로 남아 있습니다.
- **Prerequisites**: Decision trees, bootstrap sampling concepts / 의사결정 트리, 부트스트랩 샘플링 개념
- **Status**: [x]

---

## Phase 3: Deep Learning Revolution (2006–2016) / 딥러닝 혁명 (2006–2016)

### 12. A Fast Learning Algorithm for Deep Belief Nets
- **Authors**: Geoffrey Hinton, Simon Osindero, Yee-Whye Teh
- **Year**: 2006
- **DOI**: 10.1162/neco.2006.18.7.1527
- **Why it matters**: Demonstrated that deep networks could be effectively trained using layer-wise unsupervised pre-training (Restricted Boltzmann Machines). This paper is widely credited with igniting the modern deep learning revolution. / 계층별 비지도 사전학습(Restricted Boltzmann Machine)을 사용하여 심층 네트워크를 효과적으로 학습할 수 있음을 보여주었습니다. 이 논문은 현대 딥러닝 혁명의 시작으로 널리 인정받고 있습니다.
- **Prerequisites**: Papers #5, #6; probability theory, Boltzmann distributions / 논문 #5, #6; 확률론, Boltzmann 분포
- **Status**: [x]

### 13. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **Year**: 2012
- **DOI**: 10.1145/3065386
- **Why it matters**: Won ImageNet 2012 by a massive margin using a deep CNN trained on GPUs. Proved that deep learning could dramatically outperform traditional methods at scale. This moment is considered the "big bang" of practical deep learning. / GPU로 학습한 심층 CNN을 사용하여 ImageNet 2012에서 압도적 차이로 우승했습니다. 딥러닝이 대규모에서 전통적 방법을 극적으로 능가할 수 있음을 입증했습니다. 이 순간은 실용적 딥러닝의 "빅뱅"으로 여겨집니다.
- **Prerequisites**: Papers #7, #10; GPU computing basics, ReLU activation / 논문 #7, #10; GPU 컴퓨팅 기초, ReLU 활성화 함수
- **Status**: [x]

### 14. Efficient Estimation of Word Representations in Vector Space (Word2Vec)
- **Authors**: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
- **Year**: 2013
- **DOI**: arXiv:1301.3781
- **Why it matters**: Showed that simple neural networks trained on large text corpora produce word embeddings capturing semantic relationships ("king - man + woman ≈ queen"). Revolutionized NLP and made distributed representations mainstream. / 대규모 텍스트 코퍼스에서 학습한 간단한 신경망이 의미적 관계를 포착하는 단어 임베딩을 생성함을 보여주었습니다("king - man + woman ≈ queen"). NLP를 혁신하고 분산 표현을 주류로 만들었습니다.
- **Prerequisites**: Paper #6; basic NLP concepts, softmax function / 논문 #6; 기초 NLP 개념, softmax 함수
- **Status**: [x]

### 15. Auto-Encoding Variational Bayes (VAE)
- **Authors**: Diederik Kingma, Max Welling
- **Year**: 2013
- **DOI**: arXiv:1312.6114
- **Why it matters**: Introduced Variational Autoencoders — a principled probabilistic framework for learning latent representations and generating new data. Combined deep learning with Bayesian inference via the reparameterization trick. / Variational Autoencoder를 소개했습니다 — 잠재 표현 학습과 새로운 데이터 생성을 위한 원리적 확률 프레임워크입니다. Reparameterization trick을 통해 딥러닝과 Bayesian 추론을 결합했습니다.
- **Prerequisites**: Papers #6, #12; probability theory, KL divergence, Bayesian inference basics / 논문 #6, #12; 확률론, KL 발산, Bayesian 추론 기초
- **Status**: [x]

### 16. Generative Adversarial Nets (GANs)
- **Authors**: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- **Year**: 2014
- **DOI**: arXiv:1406.2661
- **Why it matters**: Introduced a radically new generative framework — two networks (generator and discriminator) competing in a minimax game. Spawned an enormous research field and enabled photorealistic image generation. / 근본적으로 새로운 생성 프레임워크를 도입했습니다 — 두 네트워크(생성자와 판별자)가 미니맥스 게임에서 경쟁합니다. 거대한 연구 분야를 탄생시키고 사실적인 이미지 생성을 가능하게 했습니다.
- **Prerequisites**: Papers #6, #13; game theory basics, training dynamics / 논문 #6, #13; 기초 게임 이론, 학습 역학
- **Status**: [x]

### 17. Neural Machine Translation by Jointly Learning to Align and Translate
- **Authors**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- **Year**: 2014
- **DOI**: arXiv:1409.0473
- **Why it matters**: Introduced the attention mechanism for sequence-to-sequence models. Instead of compressing an entire input into a fixed-length vector, the model learns to focus on relevant parts of the input. This idea became the foundation of Transformers. / Sequence-to-sequence 모델에 attention 메커니즘을 도입했습니다. 전체 입력을 고정 길이 벡터로 압축하는 대신, 모델이 입력의 관련 부분에 집중하는 것을 학습합니다. 이 아이디어가 Transformer의 기초가 되었습니다.
- **Prerequisites**: Papers #6, #9; encoder-decoder architecture / 논문 #6, #9; 인코더-디코더 아키텍처
- **Status**: [x]

### 18. Adam: A Method for Stochastic Optimization
- **Authors**: Diederik Kingma, Jimmy Ba
- **Year**: 2014
- **DOI**: arXiv:1412.6980
- **Why it matters**: Proposed the Adam optimizer — combining momentum and adaptive learning rates. Became the de facto default optimizer for deep learning due to its robustness and ease of use. / Adam optimizer를 제안했습니다 — 모멘텀과 적응적 학습률을 결합합니다. 견고성과 사용 편의성으로 인해 딥러닝의 사실상 기본 optimizer가 되었습니다.
- **Prerequisites**: Paper #6; SGD, momentum, gradient descent variants / 논문 #6; SGD, 모멘텀, 경사 하강법 변형
- **Status**: [x]

### 19. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- **Authors**: Sergey Ioffe, Christian Szegedy
- **Year**: 2015
- **DOI**: arXiv:1502.03167
- **Why it matters**: Introduced batch normalization, which stabilizes and accelerates training of deep networks by normalizing layer inputs. Became a near-universal component of modern architectures and a key prerequisite for training very deep networks like ResNet. / 배치 정규화를 도입하여 레이어 입력을 정규화함으로써 심층 네트워크의 학습을 안정화하고 가속화했습니다. 현대 아키텍처의 거의 보편적 구성 요소가 되었으며 ResNet 같은 초심층 네트워크 학습의 핵심 전제가 되었습니다.
- **Prerequisites**: Papers #13, #18; gradient flow, training dynamics / 논문 #13, #18; 기울기 흐름, 학습 역학
- **Status**: [x]

### 20. Deep Residual Learning for Image Recognition (ResNet)
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year**: 2015
- **DOI**: arXiv:1512.03385
- **Why it matters**: Introduced skip connections (residual learning), enabling training of networks with 100+ layers. Solved the degradation problem and won ImageNet 2015. ResNet remains a backbone architecture in computer vision. / Skip connection(잔차 학습)을 도입하여 100층 이상의 네트워크 학습을 가능하게 했습니다. 성능 저하 문제를 해결하고 ImageNet 2015에서 우승했습니다. ResNet은 컴퓨터 비전의 백본 아키텍처로 남아 있습니다.
- **Prerequisites**: Papers #13, #18, #19; deep network training challenges / 논문 #13, #18, #19; 심층 네트워크 학습 과제
- **Status**: [x]

### 21. Layer Normalization
- **Authors**: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton
- **Year**: 2016
- **DOI**: arXiv:1607.06450
- **Why it matters**: Proposed layer normalization as an alternative to batch normalization, particularly suited for RNNs and Transformers where batch statistics are unreliable. Became the standard normalization technique in all Transformer architectures. / 배치 정규화의 대안으로 레이어 정규화를 제안했으며, 배치 통계가 신뢰할 수 없는 RNN과 Transformer에 특히 적합합니다. 모든 Transformer 아키텍처의 표준 정규화 기법이 되었습니다.
- **Prerequisites**: Papers #9, #19; normalization statistics across dimensions / 논문 #9, #19; 차원 간 정규화 통계
- **Status**: [x]

---

## Phase 3.5: Deep Reinforcement Learning / 딥 강화학습 (2013–2017)

### 22. Playing Atari with Deep Reinforcement Learning (DQN)
- **Authors**: Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al. (DeepMind)
- **Year**: 2013 (workshop), 2015 (Nature)
- **DOI**: arXiv:1312.5602
- **Why it matters**: Combined deep neural networks with Q-learning to learn directly from raw pixels, achieving human-level performance on Atari games. Launched the modern deep reinforcement learning revolution and demonstrated end-to-end RL at scale. / 심층 신경망과 Q-learning을 결합하여 원시 픽셀에서 직접 학습하여 Atari 게임에서 인간 수준 성능을 달성했습니다. 현대 딥 강화학습 혁명을 시작하고 대규모 종단 간 RL을 시연했습니다.
- **Prerequisites**: Papers #10, #13; Markov decision processes, Q-learning basics / 논문 #10, #13; Markov 결정 과정, Q-learning 기초
- **Status**: [x]

### 23. Mastering the game of Go with deep neural networks and tree search (AlphaGo)
- **Authors**: David Silver, Aja Huang, Chris J. Maddison et al. (DeepMind)
- **Year**: 2016
- **DOI**: 10.1038/nature16961
- **Why it matters**: Combined deep learning with Monte Carlo tree search and reinforcement learning to defeat the world champion in Go. Demonstrated that AI could master domains previously thought to require human intuition. / 딥러닝을 Monte Carlo 트리 탐색 및 강화학습과 결합하여 바둑 세계 챔피언을 이겼습니다. AI가 인간의 직관이 필요하다고 여겨졌던 영역을 정복할 수 있음을 보여주었습니다.
- **Prerequisites**: Paper #22; policy networks, value networks, MCTS / 논문 #22; 정책 네트워크, 가치 네트워크, MCTS
- **Status**: [x]

### 24. Proximal Policy Optimization Algorithms (PPO)
- **Authors**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **Year**: 2017
- **DOI**: arXiv:1707.06347
- **Why it matters**: Introduced PPO — simple, stable, and efficient policy gradient method that became the de facto standard for policy optimization. PPO is the RL algorithm used in RLHF for training ChatGPT/InstructGPT. / PPO를 도입했습니다 — 간단하고 안정적이며 효율적인 정책 경사 방법으로, 정책 최적화의 사실상 표준이 되었습니다. PPO는 ChatGPT/InstructGPT 학습을 위한 RLHF에 사용되는 RL 알고리즘입니다.
- **Prerequisites**: Paper #22; policy gradient methods, trust region optimization / 논문 #22; 정책 경사 방법, 신뢰 영역 최적화
- **Status**: [x]

---

## Phase 4: The Transformer Era (2017–Present) / Transformer 시대 (2017–현재)

### 25. Attention Is All You Need
- **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Lukasz Kaiser, Illia Polosukhin
- **Year**: 2017
- **DOI**: 10.5555/3295222.3295349
- **Why it matters**: Introduced the Transformer architecture — replacing recurrence entirely with self-attention. Enabled massive parallelization and became the foundation for virtually all modern AI: LLMs, vision transformers, and multimodal models. Arguably the most influential AI paper of the decade. / Transformer 아키텍처를 도입했습니다 — 순환 구조를 self-attention으로 완전히 대체했습니다. 대규모 병렬화를 가능하게 하고 사실상 모든 현대 AI(LLM, vision transformer, 멀티모달 모델)의 기초가 되었습니다. 아마도 10년간 가장 영향력 있는 AI 논문입니다.
- **Prerequisites**: Papers #17, #18, #21; matrix operations, positional encoding / 논문 #17, #18, #21; 행렬 연산, 위치 인코딩
- **Status**: [x]

### 26. Semi-Supervised Classification with Graph Convolutional Networks
- **Authors**: Thomas N. Kipf, Max Welling
- **Year**: 2017
- **DOI**: arXiv:1609.02907
- **Why it matters**: Introduced Graph Convolutional Networks (GCNs), extending deep learning to non-Euclidean graph-structured data. Opened the field of geometric deep learning, fundamental for molecular chemistry, social networks, and physics simulations. / Graph Convolutional Network(GCN)을 도입하여 딥러닝을 비유클리드 그래프 구조 데이터로 확장했습니다. 분자 화학, 소셜 네트워크, 물리 시뮬레이션에 필수적인 기하학적 딥러닝 분야를 열었습니다.
- **Prerequisites**: Papers #6, #20; graph theory basics, spectral graph theory / 논문 #6, #20; 그래프 이론 기초, 스펙트럼 그래프 이론
- **Status**: [x]

### 27. Neural Architecture Search with Reinforcement Learning
- **Authors**: Barret Zoph, Quoc V. Le
- **Year**: 2017
- **DOI**: arXiv:1611.01578
- **Why it matters**: Used an RNN controller trained with RL to automatically discover neural network architectures. Pioneered the field of automated machine learning (AutoML) and challenged the assumption that architecture design requires human expertise. / RL로 학습된 RNN 컨트롤러를 사용하여 자동으로 신경망 아키텍처를 탐색했습니다. 자동화된 기계학습(AutoML) 분야를 개척하고 아키텍처 설계에 인간 전문 지식이 필요하다는 가정에 도전했습니다.
- **Prerequisites**: Papers #9, #22; architecture search spaces, controller networks / 논문 #9, #22; 아키텍처 탐색 공간, 컨트롤러 네트워크
- **Status**: [x]

### 28. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Year**: 2018
- **DOI**: 10.18653/v1/N19-1423
- **Why it matters**: Showed that bidirectional pre-training on unlabeled text followed by fine-tuning produces state-of-the-art results across NLP tasks. Popularized the pre-train → fine-tune paradigm that reshaped the entire field. / 비지도 텍스트에 대한 양방향 사전학습 후 미세조정이 NLP 작업 전반에서 최고 성능을 달성함을 보여주었습니다. 전체 분야를 재편한 사전학습 → 미세조정 패러다임을 대중화했습니다.
- **Prerequisites**: Paper #25; masked language modeling, transfer learning concepts / 논문 #25; 마스크 언어 모델링, 전이 학습 개념
- **Status**: [x]

### 29. Language Models are Unsupervised Multitask Learners (GPT-2)
- **Authors**: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- **Year**: 2019
- **DOI**: NO_DOI
- **Why it matters**: Demonstrated that a large autoregressive language model trained on diverse internet text can perform multiple NLP tasks without explicit fine-tuning (zero-shot). Revealed the emergent capabilities of scale. / 다양한 인터넷 텍스트로 학습한 대규모 자기회귀 언어 모델이 명시적 미세조정 없이(zero-shot) 여러 NLP 작업을 수행할 수 있음을 보여주었습니다. 규모의 창발적 능력을 밝혔습니다.
- **Prerequisites**: Paper #25; autoregressive modeling, byte-pair encoding / 논문 #25; 자기회귀 모델링, 바이트 쌍 인코딩
- **Status**: [x]

### 30. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
- **Authors**: Jonathan Frankle, Michael Carlin
- **Year**: 2019
- **DOI**: arXiv:1803.03635
- **Why it matters**: Showed that dense networks contain sparse subnetworks ("winning tickets") that can be trained to full accuracy when isolated. Challenged assumptions about overparameterization and opened the field of neural network pruning. / 밀집 네트워크가 분리 시 전체 정확도로 학습될 수 있는 희소 하위 네트워크("당첨 티켓")를 포함한다는 것을 보여주었습니다. 과매개변수화에 대한 가정에 도전하고 신경망 가지치기 분야를 열었습니다.
- **Prerequisites**: Papers #13, #20; network pruning, training dynamics / 논문 #13, #20; 네트워크 가지치기, 학습 역학
- **Status**: [x]

### 31. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)
- **Authors**: Alexey Dosovitskiy et al.
- **Year**: 2020
- **DOI**: arXiv:2010.11929
- **Why it matters**: Applied the Transformer architecture directly to image patches, achieving competitive or superior performance to CNNs. Unified the architectures for vision and language, opening the door to multimodal models. / Transformer 아키텍처를 이미지 패치에 직접 적용하여 CNN과 동등하거나 우수한 성능을 달성했습니다. 비전과 언어의 아키텍처를 통합하여 멀티모달 모델의 문을 열었습니다.
- **Prerequisites**: Papers #20, #25; image patch embedding / 논문 #20, #25; 이미지 패치 임베딩
- **Status**: [x]

### 32. A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
- **Authors**: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- **Year**: 2020
- **DOI**: arXiv:2002.05709
- **Why it matters**: Introduced a simple yet effective framework for self-supervised contrastive learning. Showed that learned representations without labels can rival supervised learning, advancing the self-supervised paradigm. / 간단하면서도 효과적인 자기지도 대조 학습 프레임워크를 도입했습니다. 라벨 없이 학습된 표현이 지도 학습에 필적할 수 있음을 보여주며, 자기지도 학습 패러다임을 발전시켰습니다.
- **Prerequisites**: Papers #20, #13; data augmentation, contrastive loss (InfoNCE) / 논문 #20, #13; 데이터 증강, 대조 손실 (InfoNCE)
- **Status**: [x]

### 33. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)
- **Authors**: Patrick Lewis, Ethan Perez et al.
- **Year**: 2020
- **DOI**: arXiv:2005.11401
- **Why it matters**: Combined parametric (language model) and non-parametric (retrieval) memory for knowledge-intensive generation tasks. RAG became a foundational technique for grounding LLMs in factual knowledge, directly relevant to modern AI applications. / 지식 집약적 생성 작업을 위해 매개변수(언어 모델)와 비매개변수(검색) 메모리를 결합했습니다. RAG는 LLM을 사실적 지식에 접지(grounding)하는 기초 기법이 되어, 현대 AI 응용에 직접 관련됩니다.
- **Prerequisites**: Papers #25, #28; information retrieval, dense passage retrieval / 논문 #25, #28; 정보 검색, 밀집 구절 검색
- **Status**: [x]

### 34. Language Models are Few-Shot Learners (GPT-3)
- **Authors**: Tom Brown et al.
- **Year**: 2020
- **DOI**: arXiv:2005.14165
- **Why it matters**: Scaled GPT to 175B parameters and demonstrated remarkable few-shot and in-context learning abilities without gradient updates. Showed that scale itself can be a path to general-purpose AI capabilities. / GPT를 1,750억 개 파라미터로 확장하여 기울기 업데이트 없이 놀라운 few-shot 및 in-context 학습 능력을 보여주었습니다. 규모 자체가 범용 AI 능력으로의 경로가 될 수 있음을 보여주었습니다.
- **Prerequisites**: Paper #29; scaling laws, in-context learning / 논문 #29; 스케일링 법칙, in-context 학습
- **Status**: [x]

### 35. Denoising Diffusion Probabilistic Models (DDPM)
- **Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Year**: 2020
- **DOI**: arXiv:2006.11239
- **Why it matters**: Made diffusion models practical for high-quality image generation, rivaling GANs. The forward-reverse denoising framework became the foundation for DALL-E 2, Stable Diffusion, and modern generative AI. / Diffusion 모델을 고품질 이미지 생성에 실용화하여 GAN에 필적하게 했습니다. 순방향-역방향 denoising 프레임워크가 DALL-E 2, Stable Diffusion 및 현대 생성 AI의 기초가 되었습니다.
- **Prerequisites**: Papers #15, #16; Markov chains, score matching, noise schedules / 논문 #15, #16; Markov 체인, score matching, noise schedule
- **Status**: [x]

### 36. Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- **Authors**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh et al. (OpenAI)
- **Year**: 2021
- **DOI**: arXiv:2103.00020
- **Why it matters**: Trained a vision-language model on 400M image-text pairs using contrastive learning, achieving remarkable zero-shot transfer to diverse visual tasks. Established the multimodal pre-training paradigm underpinning DALL-E, GPT-4V, and modern multimodal AI. / 4억 이미지-텍스트 쌍에 대한 대조 학습으로 비전-언어 모델을 학습시켜, 다양한 시각 작업에 대해 놀라운 zero-shot 전이를 달성했습니다. DALL-E, GPT-4V 및 현대 멀티모달 AI를 뒷받침하는 멀티모달 사전학습 패러다임을 확립했습니다.
- **Prerequisites**: Papers #25, #31, #32; contrastive learning, vision-language alignment / 논문 #25, #31, #32; 대조 학습, 비전-언어 정렬
- **Status**: [x]

### 37. Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2)
- **Authors**: John Jumper, Richard Evans, Alexander Pritzel et al. (DeepMind)
- **Year**: 2021
- **DOI**: 10.1038/s41586-021-03819-2
- **Why it matters**: Solved the 50-year protein folding problem using attention mechanisms and equivariant architectures. Demonstrated that deep learning can make fundamental scientific breakthroughs, not just engineering improvements. / Attention 메커니즘과 등변 아키텍처를 사용하여 50년간의 단백질 접힘 문제를 해결했습니다. 딥러닝이 공학적 개선뿐 아니라 근본적인 과학적 돌파구를 만들 수 있음을 보여주었습니다.
- **Prerequisites**: Paper #25; protein structure basics, attention mechanisms, equivariance / 논문 #25; 단백질 구조 기초, attention 메커니즘, 등변성
- **Status**: [x]

---

## Phase 5: Alignment, Scale, and Frontiers (2022–Present) / 정렬, 규모, 그리고 프론티어 (2022–현재)

### 38. Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
- **Authors**: Long Ouyang et al.
- **Year**: 2022
- **DOI**: 10.5555/3600270.3602281
- **Why it matters**: Introduced RLHF (Reinforcement Learning from Human Feedback) for aligning language models with human intent. The technique behind ChatGPT — making LLMs helpful, harmless, and honest. Bridged the gap between raw capability and practical usability. / 언어 모델을 인간 의도에 맞추기 위한 RLHF(인간 피드백 강화학습)를 도입했습니다. ChatGPT 뒤의 기술로, LLM을 유용하고 무해하며 정직하게 만듭니다. 원시적 능력과 실용적 사용성 사이의 격차를 해소했습니다.
- **Prerequisites**: Papers #24, #29, #34; reinforcement learning basics (reward models, PPO) / 논문 #24, #29, #34; 강화학습 기초 (보상 모델, PPO)
- **Status**: [x]

### 39. Scaling Laws for Neural Language Models
- **Authors**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- **Year**: 2020
- **DOI**: arXiv:2001.08361
- **Why it matters**: Empirically established power-law relationships between model performance and compute, data, and parameters. Provides a principled framework for deciding how to allocate resources when training large models. / 모델 성능과 연산량, 데이터, 파라미터 간의 멱법칙 관계를 실증적으로 확립했습니다. 대규모 모델 학습 시 자원 배분을 결정하기 위한 원리적 프레임워크를 제공합니다.
- **Prerequisites**: Paper #34; basic statistics, power laws / 논문 #34; 기초 통계학, 멱법칙
- **Status**: [x]

### 40. Constitutional AI: Harmlessness from AI Feedback
- **Authors**: Yuntao Bai et al.
- **Year**: 2022
- **DOI**: arXiv:2212.08073
- **Why it matters**: Proposed using AI feedback guided by a set of principles ("constitution") to train helpful and harmless models, reducing reliance on human labelers. A key development in AI alignment and safety research. / 일련의 원칙("헌법")에 의해 안내되는 AI 피드백을 사용하여 유용하고 무해한 모델을 학습시키는 방법을 제안하여, 인간 라벨러에 대한 의존을 줄였습니다. AI 정렬 및 안전 연구의 핵심 발전입니다.
- **Prerequisites**: Paper #38; RLHF concepts / 논문 #38; RLHF 개념
- **Status**: [x]

---

## Legend / 범례
- `[ ]` Not started / 시작 전
- `[~]` In progress / 진행 중
- `[x]` Completed / 완료

---

## Phase 6: Expansion (Backfill & Frontier) / 확장 (보완 및 최신, 2026-04)

This section adds important papers backfilled into earlier phases and the most recent frontier work (2023–2025). Papers are listed chronologically; they slot into the conceptual phases above.

이 섹션은 이전 단계에 보완할 중요 논문과 최근 최전선 연구(2023–2025)를 추가합니다. 시기순으로 정리되어 있으며, 위의 개념 단계에 속합니다.

### 41. Technical Note: Q-Learning
- **Authors**: Christopher J.C.H. Watkins, Peter Dayan
- **Year**: 1992
- **DOI**: 10.1023/A:1022676722315
- **Why it matters**: Formalized Q-learning — a model-free, off-policy reinforcement learning algorithm — and proved its convergence to the optimal action-value function under standard conditions. Q-learning underpins DQN and the entire deep RL revolution. / Q-learning(모델-프리, off-policy 강화학습 알고리즘)을 형식화하고 표준 조건 하에서 최적 행동-가치 함수로의 수렴을 증명했습니다. Q-learning은 DQN과 딥 RL 혁명 전반의 기초가 됩니다.
- **Prerequisites**: Markov decision processes, Bellman equation / Markov 결정 과정, Bellman 방정식
- **Status**: [ ]

### 42. Improving Neural Networks by Preventing Co-adaptation of Feature Detectors (Dropout)
- **Authors**: Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov
- **Year**: 2012
- **DOI**: arXiv:1207.0580
- **Why it matters**: Introduced dropout — randomly omitting hidden units during training — as a powerful regularizer that prevents co-adaptation of features. Became a near-universal technique to combat overfitting in deep networks (full JMLR version: Srivastava et al., 2014). / Dropout — 학습 중 은닉 유닛을 무작위로 제외 — 을 도입하여 특징의 공동 적응을 방지하는 강력한 정규화 기법으로 제시했습니다. 딥 네트워크에서 과적합을 방지하는 거의 보편적인 기법이 되었습니다 (전체 JMLR 버전: Srivastava et al., 2014).
- **Prerequisites**: Papers #6, #13; overfitting, regularization concepts / 논문 #6, #13; 과적합, 정규화 개념
- **Status**: [ ]

### 43. Sequence to Sequence Learning with Neural Networks
- **Authors**: Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- **Year**: 2014
- **DOI**: arXiv:1409.3215
- **Why it matters**: Introduced the encoder-decoder framework using LSTMs for general sequence-to-sequence tasks (notably machine translation). Established the seq2seq paradigm that the attention mechanism (#17) and Transformer (#25) later refined. / LSTM을 사용한 인코더-디코더 프레임워크를 일반 시퀀스-투-시퀀스 작업(특히 기계번역)에 도입했습니다. Attention 메커니즘(#17)과 Transformer(#25)가 후에 발전시킨 seq2seq 패러다임을 확립했습니다.
- **Prerequisites**: Papers #6, #9; RNN/LSTM basics, machine translation / 논문 #6, #9; RNN/LSTM 기초, 기계번역
- **Status**: [ ]

### 44. Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)
- **Authors**: Karen Simonyan, Andrew Zisserman
- **Year**: 2014
- **DOI**: arXiv:1409.1556
- **Why it matters**: Demonstrated that depth alone — using small 3x3 convolution filters stacked in 16-19 layer networks — yields substantial accuracy gains on ImageNet. The simple, uniform VGG architecture became a widely used backbone and a key step toward truly deep CNNs. / 16-19층 네트워크에 작은 3x3 합성곱 필터를 쌓은 깊이만으로도 ImageNet에서 상당한 정확도 향상을 얻을 수 있음을 입증했습니다. 단순하고 균일한 VGG 아키텍처는 널리 사용되는 백본이 되었으며 진정한 심층 CNN으로 가는 핵심 단계였습니다.
- **Prerequisites**: Papers #10, #13; convolutional architectures / 논문 #10, #13; 합성곱 아키텍처
- **Status**: [ ]

### 45. Going Deeper with Convolutions (GoogLeNet / Inception v1)
- **Authors**: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
- **Year**: 2015
- **DOI**: 10.1109/CVPR.2015.7298594
- **Why it matters**: Introduced the Inception module — multi-scale parallel convolutions with 1x1 bottlenecks — winning ILSVRC 2014. Demonstrated that careful architectural design (sparsity, multi-scale features) can match or beat brute-force depth at far lower computational cost. / Inception 모듈 — 1x1 병목과 다중 스케일 병렬 합성곱 — 을 도입하여 ILSVRC 2014에서 우승했습니다. 신중한 아키텍처 설계(희소성, 다중 스케일 특징)가 훨씬 적은 계산 비용으로 무차별적 깊이를 능가할 수 있음을 입증했습니다.
- **Prerequisites**: Papers #13, #44; 1x1 convolutions, network-in-network / 논문 #13, #44; 1x1 합성곱, network-in-network
- **Status**: [ ]

### 46. U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Authors**: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **Year**: 2015
- **DOI**: 10.1007/978-3-319-24574-4_28
- **Why it matters**: Introduced an encoder-decoder CNN with symmetric skip connections for dense pixel-wise prediction, achieving state-of-the-art results on biomedical segmentation with very few training images. U-Net became the canonical architecture for semantic segmentation and the backbone of modern diffusion models. / 조밀한 픽셀별 예측을 위한 대칭 skip connection을 가진 인코더-디코더 CNN을 도입하여, 매우 적은 학습 이미지로 생의학 분할에서 최첨단 결과를 달성했습니다. U-Net은 의미론적 분할의 정형 아키텍처이자 현대 diffusion 모델의 백본이 되었습니다.
- **Prerequisites**: Papers #10, #13; semantic segmentation, transposed convolutions / 논문 #10, #13; 의미론적 분할, 전치 합성곱
- **Status**: [ ]

### 47. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- **Authors**: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- **Year**: 2015 (arXiv); 2017 (TPAMI)
- **DOI**: 10.1109/TPAMI.2016.2577031
- **Why it matters**: Introduced the Region Proposal Network (RPN), unifying proposal generation and detection into a single end-to-end trainable network. Established the two-stage object detection paradigm that dominated benchmarks for years. / Region Proposal Network(RPN)을 도입하여 제안 생성과 탐지를 단일 종단간 학습 가능한 네트워크로 통합했습니다. 수년간 벤치마크를 지배한 2단계 객체 탐지 패러다임을 확립했습니다.
- **Prerequisites**: Papers #13, #20; R-CNN/Fast R-CNN, object detection basics / 논문 #13, #20; R-CNN/Fast R-CNN, 객체 탐지 기초
- **Status**: [ ]

### 48. You Only Look Once: Unified, Real-Time Object Detection (YOLO)
- **Authors**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **Year**: 2016
- **DOI**: 10.1109/CVPR.2016.91
- **Why it matters**: Reframed object detection as a single regression problem solved by one CNN forward pass, achieving real-time inference (45 FPS). Pioneered the one-stage detector family that now powers most production object-detection systems. / 객체 탐지를 단일 CNN forward pass로 해결되는 단일 회귀 문제로 재정의하여 실시간 추론(45 FPS)을 달성했습니다. 현재 대부분의 프로덕션 객체 탐지 시스템을 구동하는 1단계 탐지기 계열을 개척했습니다.
- **Prerequisites**: Papers #13, #47; bounding box regression, IoU / 논문 #13, #47; 경계 상자 회귀, IoU
- **Status**: [ ]

### 49. Trust Region Policy Optimization (TRPO)
- **Authors**: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel
- **Year**: 2015
- **DOI**: arXiv:1502.05477
- **Why it matters**: Introduced a policy gradient method that constrains updates to a trust region (KL divergence bound), guaranteeing monotonic improvement and training stability. TRPO directly motivated PPO (#24) — the algorithm behind RLHF. / KL 발산 제약을 사용해 정책 업데이트를 신뢰 영역에 제한함으로써 단조 개선과 학습 안정성을 보장하는 정책 경사법을 도입했습니다. TRPO는 PPO(#24, RLHF의 핵심 알고리즘)에 직접적 영감을 주었습니다.
- **Prerequisites**: Paper #22; policy gradients, KL divergence, natural gradients / 논문 #22; 정책 경사, KL 발산, natural gradient
- **Status**: [ ]

### 50. A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go through Self-Play (AlphaZero)
- **Authors**: David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou et al. (DeepMind)
- **Year**: 2018
- **DOI**: 10.1126/science.aar6404
- **Why it matters**: Generalized AlphaGo (#23) into a single algorithm that learned chess, shogi, and Go from scratch using only self-play and the rules of the game — defeating the strongest specialized engines. Demonstrated that domain-agnostic RL + search can master diverse perfect-information games. / AlphaGo(#23)를 일반화하여 자기 대국과 게임 규칙만으로 체스, 쇼기, 바둑을 처음부터 학습한 단일 알고리즘을 만들었으며, 가장 강력한 전문 엔진들을 모두 이겼습니다. 도메인 비의존적 RL + 탐색이 다양한 완전 정보 게임을 정복할 수 있음을 입증했습니다.
- **Prerequisites**: Paper #23; MCTS, self-play, residual networks / 논문 #23; MCTS, 자기 대국, residual network
- **Status**: [ ]

### 51. Improved Protein Structure Prediction Using Potentials from Deep Learning (AlphaFold 1)
- **Authors**: Andrew W. Senior, Richard Evans, John Jumper et al. (DeepMind)
- **Year**: 2020
- **DOI**: 10.1038/s41586-019-1923-7
- **Why it matters**: Won CASP13 (2018) by predicting inter-residue distance distributions with deep neural networks and converting them into 3D structures via gradient descent. Established the deep-learning approach to protein folding that AlphaFold 2 (#37) later perfected. / 심층 신경망으로 잔기 간 거리 분포를 예측하고 경사 하강을 통해 3D 구조로 변환하는 방식으로 CASP13(2018)에서 우승했습니다. AlphaFold 2(#37)가 후에 완성한 단백질 접힘에 대한 딥러닝 접근법을 확립했습니다.
- **Prerequisites**: Papers #20, #25; protein structure basics, distance geometry / 논문 #20, #25; 단백질 구조 기초, 거리 기하학
- **Status**: [ ]

### 52. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)
- **Authors**: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
- **Year**: 2019 (arXiv); 2020 (JMLR)
- **DOI**: arXiv:1910.10683
- **Why it matters**: Recast every NLP task as text-to-text and exhaustively studied design choices in pre-training (objectives, architectures, datasets) at scale. T5 popularized the unified text-to-text paradigm and produced the C4 dataset that became a standard pre-training corpus. / 모든 NLP 작업을 text-to-text로 재구성하고 사전학습의 설계 선택(목적함수, 아키텍처, 데이터셋)을 대규모로 철저히 연구했습니다. T5는 통합 text-to-text 패러다임을 대중화하고 표준 사전학습 코퍼스가 된 C4 데이터셋을 만들었습니다.
- **Prerequisites**: Papers #25, #28; encoder-decoder Transformer, denoising objectives / 논문 #25, #28; encoder-decoder Transformer, denoising 목적함수
- **Status**: [ ]

### 53. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- **Authors**: William Fedus, Barret Zoph, Noam Shazeer
- **Year**: 2021
- **DOI**: arXiv:2101.03961
- **Why it matters**: Simplified Mixture-of-Experts (MoE) routing to a single expert per token, enabling scaling to trillion-parameter models with comparable per-token compute to dense models. MoE became a key technique behind modern large models (Mixtral, GPT-4 rumored, Gemini). / Mixture-of-Experts(MoE) 라우팅을 토큰당 단일 expert로 단순화하여, 밀집 모델과 비슷한 토큰당 연산량으로 1조 매개변수 모델까지 확장할 수 있게 했습니다. MoE는 현대 대형 모델(Mixtral, GPT-4 추정, Gemini)의 핵심 기법이 되었습니다.
- **Prerequisites**: Papers #25, #39; MoE basics, distributed training / 논문 #25, #39; MoE 기초, 분산 학습
- **Status**: [ ]

### 54. LoRA: Low-Rank Adaptation of Large Language Models
- **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Year**: 2021
- **DOI**: arXiv:2106.09685
- **Why it matters**: Showed that fine-tuning large language models can be done by training only low-rank update matrices, reducing trainable parameters by 10,000x with no inference latency. LoRA became the dominant parameter-efficient fine-tuning method (PEFT) for adapting LLMs in practice. / 대형 언어 모델의 미세조정을 저랭크 업데이트 행렬만 학습함으로써 수행할 수 있음을 보여, 학습 가능 매개변수를 1만 배 줄이고 추론 지연도 없게 만들었습니다. LoRA는 실무에서 LLM을 적응시키는 매개변수-효율적 미세조정(PEFT)의 지배적 방법이 되었습니다.
- **Prerequisites**: Paper #28, #34; matrix rank, fine-tuning / 논문 #28, #34; 행렬 랭크, 미세조정
- **Status**: [ ]

### 55. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors**: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- **Year**: 2022
- **DOI**: arXiv:2201.11903
- **Why it matters**: Showed that prompting LLMs to produce intermediate reasoning steps before the final answer dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks. Chain-of-thought (CoT) became a fundamental prompting technique and the seed for "reasoning" models like o1/R1. / LLM이 최종 답변 전 중간 추론 단계를 생성하도록 유도하면 산술, 상식, 기호적 추론 작업에서 극적으로 성능이 향상됨을 보여주었습니다. Chain-of-thought(CoT)는 기본적인 프롬프팅 기법이자 o1/R1 같은 "추론" 모델의 씨앗이 되었습니다.
- **Prerequisites**: Paper #34; in-context learning, prompt engineering / 논문 #34; in-context 학습, 프롬프트 엔지니어링
- **Status**: [ ]

### 56. Emergent Abilities of Large Language Models
- **Authors**: Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph et al.
- **Year**: 2022
- **DOI**: arXiv:2206.07682
- **Why it matters**: Catalogued capabilities that appear unpredictably and discontinuously past certain scale thresholds (arithmetic, multi-step reasoning, instruction following). Sparked extensive debate on whether scaling alone unlocks qualitatively new abilities or whether "emergence" is a metric artifact. / 특정 규모 임계값을 넘어서야 예측 불가능하고 불연속적으로 나타나는 능력(산술, 다단계 추론, 지시 따르기)을 정리했습니다. 단순 규모 확대가 질적으로 새로운 능력을 잠금 해제하는지, 아니면 "창발"이 지표상의 인공물인지에 대한 광범위한 논쟁을 촉발했습니다.
- **Prerequisites**: Papers #34, #39; benchmarks, scaling curves / 논문 #34, #39; 벤치마크, 스케일링 곡선
- **Status**: [ ]

### 57. Training Compute-Optimal Large Language Models (Chinchilla)
- **Authors**: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch et al. (DeepMind)
- **Year**: 2022
- **DOI**: arXiv:2203.15556
- **Why it matters**: Revised the Kaplan et al. scaling laws (#39) by showing that compute-optimal training requires scaling parameters and tokens roughly equally. A 70B Chinchilla trained on 1.4T tokens beat a 280B Gopher trained on far fewer tokens. Reshaped LLM training budgets industry-wide. / Kaplan et al. 스케일링 법칙(#39)을 수정하여, 연산-최적 학습은 매개변수와 토큰을 거의 동등하게 확장해야 함을 보였습니다. 1.4T 토큰으로 학습된 70B Chinchilla가 훨씬 적은 토큰으로 학습된 280B Gopher를 이겼습니다. 업계 전반의 LLM 학습 예산을 재편했습니다.
- **Prerequisites**: Paper #39; scaling laws, compute budgets / 논문 #39; 스케일링 법칙, 연산 예산
- **Status**: [ ]

### 58. PaLM: Scaling Language Modeling with Pathways
- **Authors**: Aakanksha Chowdhery, Sharan Narang, Jacob Devlin et al. (Google)
- **Year**: 2022
- **DOI**: arXiv:2204.02311
- **Why it matters**: Scaled a dense decoder Transformer to 540B parameters trained on 6,144 TPU v4 chips, demonstrating breakthrough capability on hundreds of language tasks and substantial gains on multi-step reasoning. Showcased Google's Pathways system for efficient large-scale training. / 밀집 디코더 Transformer를 540B 매개변수로 확장하고 6,144개 TPU v4 칩으로 학습시켜, 수백 가지 언어 작업에서 획기적 성능과 다단계 추론에서 상당한 향상을 보여주었습니다. Google의 Pathways 시스템을 효율적 대규모 학습에 시연했습니다.
- **Prerequisites**: Papers #25, #34; large-scale distributed training / 논문 #25, #34; 대규모 분산 학습
- **Status**: [ ]

### 59. ReAct: Synergizing Reasoning and Acting in Language Models
- **Authors**: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- **Year**: 2022
- **DOI**: arXiv:2210.03629
- **Why it matters**: Combined chain-of-thought reasoning with action-taking (tool/API calls) in an interleaved Thought-Act-Observation loop. ReAct established the canonical "agent loop" pattern that powers modern LLM agents and tool-use frameworks (LangChain, AutoGPT). / Chain-of-thought 추론과 행동 수행(도구/API 호출)을 Thought-Act-Observation 루프로 교차 결합했습니다. ReAct는 현대 LLM 에이전트와 도구 사용 프레임워크(LangChain, AutoGPT)를 구동하는 정형 "에이전트 루프" 패턴을 확립했습니다.
- **Prerequisites**: Papers #34, #55; tool use, in-context learning / 논문 #34, #55; 도구 사용, in-context 학습
- **Status**: [ ]

### 60. High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)
- **Authors**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer
- **Year**: 2022
- **DOI**: 10.1109/CVPR52688.2022.01042
- **Why it matters**: Moved diffusion to a learned compressed latent space, drastically reducing compute requirements while preserving quality. Latent Diffusion Models (LDMs) became Stable Diffusion — the open-weights model that brought high-quality generative imagery to the masses. / Diffusion을 학습된 압축 잠재 공간으로 이동시켜, 품질을 유지하면서 연산 요구량을 극적으로 줄였습니다. Latent Diffusion Models(LDM)는 고품질 생성 이미지를 대중에게 가져온 오픈-웨이트 모델인 Stable Diffusion이 되었습니다.
- **Prerequisites**: Papers #15, #35; autoencoders, cross-attention conditioning / 논문 #15, #35; 오토인코더, cross-attention 조건화
- **Status**: [ ]

### 61. GPT-4 Technical Report
- **Authors**: OpenAI (Josh Achiam, Steven Adler, Sandhini Agarwal et al.)
- **Year**: 2023
- **DOI**: arXiv:2303.08774
- **Why it matters**: Reported GPT-4 — a multimodal (image+text in, text out) model with human-level performance on many professional and academic benchmarks. Notable for demonstrating predictable scaling and for what it deliberately did not disclose (architecture, parameter count, training data), shaping the closed-frontier-model era. / GPT-4 — 다양한 전문 및 학술 벤치마크에서 인간 수준 성능을 보이는 멀티모달(이미지+텍스트 입력, 텍스트 출력) 모델 — 을 보고했습니다. 예측 가능한 스케일링을 시연한 점, 그리고 의도적으로 공개하지 않은 정보(아키텍처, 매개변수 수, 학습 데이터)로 인해 폐쇄형 프론티어 모델 시대를 형성했습니다.
- **Prerequisites**: Papers #29, #34, #38; multimodal models, capability evaluation / 논문 #29, #34, #38; 멀티모달 모델, 능력 평가
- **Status**: [ ]

### 62. LLaMA: Open and Efficient Foundation Language Models
- **Authors**: Hugo Touvron, Thibaut Lavril, Gautier Izacard et al. (Meta AI)
- **Year**: 2023
- **DOI**: arXiv:2302.13971
- **Why it matters**: Released a family of 7B-65B parameter models trained on public data only, with the 13B model outperforming GPT-3 on most benchmarks. The leaked weights ignited the open-weights LLM ecosystem (Alpaca, Vicuna, etc.) and reshaped academic and indie research access. / 공개 데이터만으로 학습된 7B-65B 매개변수 모델 패밀리를 출시했으며, 13B 모델은 대부분의 벤치마크에서 GPT-3을 능가했습니다. 유출된 가중치는 오픈-웨이트 LLM 생태계(Alpaca, Vicuna 등)를 점화시키고 학술 및 인디 연구 접근성을 재편했습니다.
- **Prerequisites**: Papers #25, #34; pre-training data curation, RoPE / 논문 #25, #34; 사전학습 데이터 큐레이션, RoPE
- **Status**: [ ]

### 63. Toolformer: Language Models Can Teach Themselves to Use Tools
- **Authors**: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- **Year**: 2023
- **DOI**: arXiv:2302.04761
- **Why it matters**: Showed that an LLM can teach itself when and how to call APIs (calculator, search, translation, etc.) through self-supervised data augmentation, with only a handful of demonstrations. Foundation for modern function-calling capabilities in production LLMs. / LLM이 자기지도 데이터 증강을 통해 단 몇 개의 시연만으로 API(계산기, 검색, 번역 등)를 언제 어떻게 호출할지 스스로 학습할 수 있음을 보여주었습니다. 프로덕션 LLM의 현대적 function-calling 기능의 기초입니다.
- **Prerequisites**: Papers #34, #59; self-supervised learning, perplexity-based filtering / 논문 #34, #59; 자기지도 학습, perplexity 기반 필터링
- **Status**: [ ]

### 64. Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- **Authors**: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- **Year**: 2023
- **DOI**: arXiv:2305.10601
- **Why it matters**: Generalized chain-of-thought into tree search over partial solutions, with the LLM evaluating intermediate states and choosing branches to explore or backtrack from. Demonstrated significant gains on tasks requiring planning (Game of 24, creative writing, crosswords). / Chain-of-thought를 부분 해에 대한 트리 탐색으로 일반화하여, LLM이 중간 상태를 평가하고 탐색하거나 되돌아갈 분기를 선택하도록 했습니다. 계획이 필요한 작업(Game of 24, 창작, 크로스워드)에서 상당한 성능 향상을 보였습니다.
- **Prerequisites**: Papers #55, #59; tree search, evaluation prompts / 논문 #55, #59; 트리 탐색, 평가 프롬프트
- **Status**: [ ]

### 65. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)
- **Authors**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year**: 2023
- **DOI**: arXiv:2305.18290
- **Why it matters**: Showed that the standard RLHF objective can be solved in closed form, eliminating the need for a separate reward model and PPO. DPO replaces complex RLHF pipelines (#38) with a simple classification loss, dramatically simplifying preference-tuning of LLMs. / 표준 RLHF 목적함수가 닫힌 형태로 풀릴 수 있음을 보여, 별도의 보상 모델과 PPO가 필요없게 만들었습니다. DPO는 복잡한 RLHF 파이프라인(#38)을 단순 분류 손실로 대체하여 LLM의 선호도 튜닝을 극적으로 단순화시켰습니다.
- **Prerequisites**: Papers #24, #38; RLHF, Bradley-Terry preference model / 논문 #24, #38; RLHF, Bradley-Terry 선호 모델
- **Status**: [ ]

### 66. Mistral 7B
- **Authors**: Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch et al. (Mistral AI)
- **Year**: 2023
- **DOI**: arXiv:2310.06825
- **Why it matters**: Demonstrated that careful engineering (grouped-query attention, sliding-window attention, byte-fallback BPE) lets a 7B model outperform LLaMA 2 13B and approach LLaMA 1 34B. Set a new bar for compute-efficient open-weights models and the architectural template for many later 7B-class models. / 신중한 엔지니어링(grouped-query attention, sliding-window attention, byte-fallback BPE)으로 7B 모델이 LLaMA 2 13B를 능가하고 LLaMA 1 34B에 근접할 수 있음을 시연했습니다. 연산-효율적 오픈-웨이트 모델의 새 기준을 세우고 이후 많은 7B급 모델의 아키텍처 템플릿이 되었습니다.
- **Prerequisites**: Papers #25, #62; GQA, sliding window attention / 논문 #25, #62; GQA, sliding window attention
- **Status**: [ ]

### 67. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors**: Albert Gu, Tri Dao
- **Year**: 2023
- **DOI**: arXiv:2312.00752
- **Why it matters**: Introduced selective state-space models (SSMs) that match Transformer quality on language modeling while scaling linearly (vs. quadratically) in sequence length, with 5x higher inference throughput. The leading post-Transformer architectural challenger and an active line of research. / 언어 모델링에서 Transformer 품질을 따라잡으면서 시퀀스 길이에 대해 선형으로(2차가 아닌) 확장되며 5배 높은 추론 처리량을 가진 선택적 상태공간 모델(SSM)을 도입했습니다. Transformer 이후의 주요 아키텍처 도전자이자 활발한 연구 흐름입니다.
- **Prerequisites**: Papers #9, #25; state-space models, S4 / 논문 #9, #25; 상태공간 모델, S4
- **Status**: [ ]

### 68. Mixtral of Experts
- **Authors**: Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux et al. (Mistral AI)
- **Year**: 2024
- **DOI**: arXiv:2401.04088
- **Why it matters**: A sparse Mixture-of-Experts (8x7B, ~13B active) open-weights model that matched or beat LLaMA 2 70B and GPT-3.5 on most benchmarks. Brought MoE (#53) into mainstream practice and validated sparse models for production deployment. / 희소 Mixture-of-Experts(8x7B, ~13B 활성) 오픈-웨이트 모델로, 대부분의 벤치마크에서 LLaMA 2 70B와 GPT-3.5를 따라잡거나 능가했습니다. MoE(#53)를 주류 실무로 가져왔고 프로덕션 배포를 위한 희소 모델을 검증했습니다.
- **Prerequisites**: Papers #53, #66; MoE routing, expert balancing / 논문 #53, #66; MoE 라우팅, expert 균형
- **Status**: [ ]

### 69. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Authors**: DeepSeek-AI
- **Year**: 2025
- **DOI**: arXiv:2501.12948
- **Why it matters**: Demonstrated that pure reinforcement learning (without supervised reasoning data) can elicit emergent long chain-of-thought, self-verification, and reflection in LLMs — matching OpenAI o1-class performance. Open-weights release transformed the public landscape for reasoning-capable LLMs. / 순수 강화학습(지도 추론 데이터 없이)으로 LLM에서 긴 chain-of-thought, 자가 검증, 반성과 같은 창발적 능력을 유도할 수 있음을 시연하여 OpenAI o1급 성능에 도달했습니다. 오픈-웨이트 공개는 추론 가능 LLM의 공개 환경을 변혁시켰습니다.
- **Prerequisites**: Papers #24, #38, #55; RLHF, GRPO, reasoning evaluation / 논문 #24, #38, #55; RLHF, GRPO, 추론 평가
- **Status**: [ ]
