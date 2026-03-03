# AI Paper Reading List

A curated list of landmark papers in Artificial Intelligence, organized chronologically by era.
Read in order — each paper builds on the concepts introduced by earlier ones.

---

## Phase 1: Foundations (1943–1969)

### 1. A Logical Calculus of the Ideas Immanent in Nervous Activity
- **Authors**: Warren McCulloch, Walter Pitts
- **Year**: 1943
- **Why it matters**: The very first mathematical model of an artificial neuron. Showed that networks of simple binary threshold units can compute any logical function — laying the theoretical foundation for all neural networks.
- **Prerequisites**: Basic logic (AND, OR, NOT gates)
- **Status**: [x]

### 2. Computing Machinery and Intelligence
- **Authors**: Alan Turing
- **Year**: 1950
- **Why it matters**: Posed the fundamental question "Can machines think?" and proposed the Turing Test. Anticipated and addressed major objections to machine intelligence decades before they became mainstream debates.
- **Prerequisites**: None (philosophical paper, highly accessible)
- **Status**: [ ]

### 3. The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain
- **Authors**: Frank Rosenblatt
- **Year**: 1958
- **Why it matters**: Introduced the perceptron — the first trainable neural network model. Demonstrated that a machine could learn from data rather than being explicitly programmed, sparking enormous excitement about AI.
- **Prerequisites**: Paper #1; basic linear algebra (vectors, dot products)
- **Status**: [ ]

### 4. Perceptrons (book)
- **Authors**: Marvin Minsky, Seymour Papert
- **Year**: 1969
- **Why it matters**: Rigorously proved the limitations of single-layer perceptrons (e.g., inability to learn XOR). Widely credited with triggering the first "AI winter" by dampening enthusiasm for neural networks. Understanding these limitations motivates multi-layer networks.
- **Prerequisites**: Paper #3; basic set theory and logic
- **Status**: [ ]

---

## Phase 2: Revival and Classical Machine Learning (1982–2001)

### 5. Neural Networks and Physical Systems with Emergent Collective Computational Abilities
- **Authors**: John Hopfield
- **Year**: 1982
- **Why it matters**: Introduced Hopfield networks — recurrent networks with an energy function that can store and retrieve patterns (associative memory). Bridged physics and neural computation, reigniting interest in neural networks.
- **Prerequisites**: Papers #1, #3; basic physics concepts (energy minimization)
- **Status**: [ ]

### 6. Learning Representations by Back-propagating Errors
- **Authors**: David Rumelhart, Geoffrey Hinton, Ronald Williams
- **Year**: 1986
- **Why it matters**: Popularized backpropagation — the algorithm that makes training multi-layer neural networks practical. This single paper enabled the entire field of deep learning. It directly addressed the limitations Minsky & Papert identified.
- **Prerequisites**: Paper #4; calculus (chain rule), linear algebra (matrix multiplication)
- **Status**: [ ]

### 7. Backpropagation Applied to Handwritten Zip Code Recognition
- **Authors**: Yann LeCun et al.
- **Year**: 1989
- **Why it matters**: First successful application of backpropagation to a real-world problem using convolutional neural networks (CNNs). Introduced weight sharing and local receptive fields — the architecture that now dominates computer vision.
- **Prerequisites**: Paper #6; convolution operation basics
- **Status**: [ ]

### 8. Support-Vector Networks
- **Authors**: Corinna Cortes, Vladimir Vapnik
- **Year**: 1995
- **Why it matters**: Introduced Support Vector Machines (SVMs) with the kernel trick. SVMs dominated machine learning for over a decade and brought rigorous statistical learning theory into practice. Understanding SVMs gives crucial insight into margins, generalization, and the bias-variance tradeoff.
- **Prerequisites**: Linear algebra, basic optimization (Lagrange multipliers)
- **Status**: [ ]

### 9. Long Short-Term Memory
- **Authors**: Sepp Hochreiter, Jürgen Schmidhuber
- **Year**: 1997
- **Why it matters**: Solved the vanishing gradient problem in recurrent neural networks by introducing gating mechanisms. LSTMs became the dominant architecture for sequence modeling (text, speech, time series) for nearly two decades.
- **Prerequisites**: Paper #6; RNN basics (understanding of vanishing gradients)
- **Status**: [ ]

### 10. Gradient-Based Learning Applied to Document Recognition
- **Authors**: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
- **Year**: 1998
- **Why it matters**: The definitive CNN paper. Presented LeNet-5 in full detail with thorough experiments on handwritten digit recognition. Also introduced the concept of end-to-end trainable systems and graph transformer networks.
- **Prerequisites**: Papers #6, #7; convolution operations
- **Status**: [ ]

### 11. Random Forests
- **Authors**: Leo Breiman
- **Year**: 2001
- **Why it matters**: Introduced Random Forests — an ensemble method combining decision trees with bagging and random feature selection. Remains one of the most robust and widely-used algorithms in practice, especially for tabular data.
- **Prerequisites**: Decision trees, bootstrap sampling concepts
- **Status**: [ ]

---

## Phase 3: Deep Learning Revolution (2006–2016)

### 12. A Fast Learning Algorithm for Deep Belief Nets
- **Authors**: Geoffrey Hinton, Simon Osindero, Yee-Whye Teh
- **Year**: 2006
- **Why it matters**: Demonstrated that deep networks could be effectively trained using layer-wise unsupervised pre-training (Restricted Boltzmann Machines). This paper is widely credited with igniting the modern deep learning revolution.
- **Prerequisites**: Papers #5, #6; probability theory, Boltzmann distributions
- **Status**: [ ]

### 13. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **Year**: 2012
- **Why it matters**: Won ImageNet 2012 by a massive margin using a deep CNN trained on GPUs. Proved that deep learning could dramatically outperform traditional methods at scale. This moment is considered the "big bang" of practical deep learning.
- **Prerequisites**: Papers #7, #10; GPU computing basics, ReLU activation
- **Status**: [ ]

### 14. Efficient Estimation of Word Representations in Vector Space (Word2Vec)
- **Authors**: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
- **Year**: 2013
- **Why it matters**: Showed that simple neural networks trained on large text corpora produce word embeddings capturing semantic relationships ("king - man + woman ≈ queen"). Revolutionized NLP and made distributed representations mainstream.
- **Prerequisites**: Paper #6; basic NLP concepts, softmax function
- **Status**: [ ]

### 15. Auto-Encoding Variational Bayes (VAE)
- **Authors**: Diederik Kingma, Max Welling
- **Year**: 2013
- **Why it matters**: Introduced Variational Autoencoders — a principled probabilistic framework for learning latent representations and generating new data. Combined deep learning with Bayesian inference via the reparameterization trick.
- **Prerequisites**: Papers #6, #12; probability theory, KL divergence, Bayesian inference basics
- **Status**: [ ]

### 16. Generative Adversarial Nets (GANs)
- **Authors**: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- **Year**: 2014
- **Why it matters**: Introduced a radically new generative framework — two networks (generator and discriminator) competing in a minimax game. Spawned an enormous research field and enabled photorealistic image generation.
- **Prerequisites**: Papers #6, #13; game theory basics, training dynamics
- **Status**: [ ]

### 17. Neural Machine Translation by Jointly Learning to Align and Translate
- **Authors**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- **Year**: 2014
- **Why it matters**: Introduced the attention mechanism for sequence-to-sequence models. Instead of compressing an entire input into a fixed-length vector, the model learns to focus on relevant parts of the input. This idea became the foundation of Transformers.
- **Prerequisites**: Papers #6, #9; encoder-decoder architecture
- **Status**: [ ]

### 18. Adam: A Method for Stochastic Optimization
- **Authors**: Diederik Kingma, Jimmy Ba
- **Year**: 2014
- **Why it matters**: Proposed the Adam optimizer — combining momentum and adaptive learning rates. Became the de facto default optimizer for deep learning due to its robustness and ease of use.
- **Prerequisites**: Paper #6; SGD, momentum, gradient descent variants
- **Status**: [ ]

### 19. Deep Residual Learning for Image Recognition (ResNet)
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year**: 2015
- **Why it matters**: Introduced skip connections (residual learning), enabling training of networks with 100+ layers. Solved the degradation problem and won ImageNet 2015. ResNet remains a backbone architecture in computer vision.
- **Prerequisites**: Papers #13, #18; batch normalization, deep network training challenges
- **Status**: [ ]

---

## Phase 4: The Transformer Era (2017–Present)

### 20. Attention Is All You Need
- **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Łukasz Kaiser, Illia Polosukhin
- **Year**: 2017
- **Why it matters**: Introduced the Transformer architecture — replacing recurrence entirely with self-attention. Enabled massive parallelization and became the foundation for virtually all modern AI: LLMs, vision transformers, and multimodal models. Arguably the most influential AI paper of the decade.
- **Prerequisites**: Papers #17, #18; matrix operations, positional encoding
- **Status**: [ ]

### 21. BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding
- **Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Year**: 2018
- **Why it matters**: Showed that bidirectional pre-training on unlabeled text followed by fine-tuning produces state-of-the-art results across NLP tasks. Popularized the pre-train → fine-tune paradigm that reshaped the entire field.
- **Prerequisites**: Paper #20; masked language modeling, transfer learning concepts
- **Status**: [ ]

### 22. Language Models are Unsupervised Multitask Learners (GPT-2)
- **Authors**: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- **Year**: 2019
- **Why it matters**: Demonstrated that a large autoregressive language model trained on diverse internet text can perform multiple NLP tasks without explicit fine-tuning (zero-shot). Revealed the emergent capabilities of scale.
- **Prerequisites**: Paper #20; autoregressive modeling, byte-pair encoding
- **Status**: [ ]

### 23. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)
- **Authors**: Alexey Dosovitskiy et al.
- **Year**: 2020
- **Why it matters**: Applied the Transformer architecture directly to image patches, achieving competitive or superior performance to CNNs. Unified the architectures for vision and language, opening the door to multimodal models.
- **Prerequisites**: Papers #19, #20; image patch embedding
- **Status**: [ ]

### 24. Language Models are Few-Shot Learners (GPT-3)
- **Authors**: Tom Brown et al.
- **Year**: 2020
- **Why it matters**: Scaled GPT to 175B parameters and demonstrated remarkable few-shot and in-context learning abilities without gradient updates. Showed that scale itself can be a path to general-purpose AI capabilities.
- **Prerequisites**: Paper #22; scaling laws, in-context learning
- **Status**: [ ]

### 25. Denoising Diffusion Probabilistic Models (DDPM)
- **Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Year**: 2020
- **Why it matters**: Made diffusion models practical for high-quality image generation, rivaling GANs. The forward-reverse denoising framework became the foundation for DALL-E 2, Stable Diffusion, and modern generative AI.
- **Prerequisites**: Papers #15, #16; Markov chains, score matching, noise schedules
- **Status**: [ ]

---

## Phase 5: Alignment, Scale, and Frontiers (2022–Present)

### 26. Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
- **Authors**: Long Ouyang et al.
- **Year**: 2022
- **Why it matters**: Introduced RLHF (Reinforcement Learning from Human Feedback) for aligning language models with human intent. The technique behind ChatGPT — making LLMs helpful, harmless, and honest. Bridged the gap between raw capability and practical usability.
- **Prerequisites**: Papers #22, #24; reinforcement learning basics (reward models, PPO)
- **Status**: [ ]

### 27. Scaling Laws for Neural Language Models
- **Authors**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- **Year**: 2020
- **Why it matters**: Empirically established power-law relationships between model performance and compute, data, and parameters. Provides a principled framework for deciding how to allocate resources when training large models.
- **Prerequisites**: Paper #24; basic statistics, power laws
- **Status**: [ ]

### 28. Constitutional AI: Harmlessness from AI Feedback
- **Authors**: Yuntao Bai et al. (Anthropic)
- **Year**: 2022
- **Why it matters**: Proposed using AI feedback guided by a set of principles ("constitution") to train helpful and harmless models, reducing reliance on human labelers. A key development in AI alignment and safety research.
- **Prerequisites**: Paper #26; RLHF concepts
- **Status**: [ ]

---

## Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Completed
