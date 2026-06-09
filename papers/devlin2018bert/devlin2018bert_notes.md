---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: [Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova]
year: 2018
journal: "NAACL-HLT 2019"
doi: "10.18653/v1/N19-1423"
topic: Artificial_Intelligence
tags: [transformer, pre-training, language-model, masked-language-model, NSP, fine-tuning, NLP, transfer-learning]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 28. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding / 언어 이해를 위한 깊은 양방향 Transformer 사전학습

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 **BERT(Bidirectional Encoder Representations from Transformers)** 를 제시합니다. BERT는 Transformer encoder를 **양방향(deeply bidirectional)** 으로 사전학습하여, 모든 layer가 좌우 문맥을 동시에 conditioning하도록 설계된 표현 학습 모델입니다. 핵심 혁신은 두 가지 비지도 사전학습 과제의 결합입니다. 첫째, **Masked Language Model(MLM)** — Cloze task(Taylor, 1953)에서 영감을 받아 입력 토큰의 15%를 무작위로 가린 뒤 양쪽 문맥으로부터 그 토큰을 예측합니다. 이는 표준 단방향 LM의 "양방향이 되면 자기 자신을 본다" 라는 trivial leakage 문제를 우회합니다. 둘째, **Next Sentence Prediction(NSP)** — 두 문장 A, B의 쌍이 실제 연속(IsNext)인지 무작위 쌍(NotNext)인지를 50:50 비율로 학습하여, 문장 간 관계를 포착합니다. 사전학습은 BooksCorpus(800M words) + English Wikipedia(2,500M words) 위에서 1M step, batch 256 sequences × 512 tokens로 수행됩니다. 사전학습된 모델은 단순히 task별 출력 layer 하나만 추가해 **모든 파라미터를 end-to-end fine-tuning** 함으로써 11개 NLP 벤치마크에서 SOTA를 갱신했습니다: GLUE 80.5(+7.7%), MultiNLI 86.7%, SQuAD 1.1 F1 93.2(+1.5), SQuAD 2.0 F1 83.1(+5.1), SWAG 86.3%(+27.1). 모델 크기는 BERT-Base(L=12, H=768, A=12, 110M params)와 BERT-Large(L=24, H=1024, A=16, 340M params) 두 가지를 보고합니다.

**English**
This paper introduces **BERT (Bidirectional Encoder Representations from Transformers)**, a representation-learning model that pre-trains a Transformer encoder in a **deeply bidirectional** manner so that **every layer jointly conditions on both left and right context**. The core innovation is the combination of two unsupervised pre-training tasks. First, the **Masked Language Model (MLM)** — inspired by the Cloze task (Taylor, 1953) — randomly masks 15% of input tokens and predicts them from their two-sided context. This circumvents the "self-leakage" problem that ordinary bidirectional LMs would encounter (a token would trivially see itself through deep layers). Second, **Next Sentence Prediction (NSP)** — given two sentences A and B paired 50/50 as IsNext or NotNext, the model learns inter-sentence relationships useful for QA and NLI. Pre-training is done on BooksCorpus (800M words) + English Wikipedia (2,500M words) for 1M steps with batch 256 sequences × 512 tokens. The pre-trained model is then fine-tuned **end-to-end** on each downstream task by adding only one task-specific output layer. BERT advances the state of the art on 11 NLP benchmarks: GLUE 80.5 (+7.7% absolute), MultiNLI 86.7%, SQuAD 1.1 F1 93.2 (+1.5), SQuAD 2.0 F1 83.1 (+5.1), SWAG 86.3% (+27.1). Two model sizes are reported: BERT-Base (L=12, H=768, A=12, 110M params) and BERT-Large (L=24, H=1024, A=16, 340M params).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 NLP에서 사전학습 표현(pre-trained representations)을 다운스트림 과제에 적용하는 두 갈래의 전략을 대비합니다.

- **Feature-based** (예: ELMo, Peters et al., 2018a): 사전학습된 표현을 task-specific 모델 architecture에 추가 입력 특징으로 제공.
- **Fine-tuning** (예: OpenAI GPT, Radford et al., 2018): 최소한의 task-specific 파라미터만 도입하고, 사전학습된 모든 파라미터를 미세조정.

두 접근 모두 사전학습 시 **단방향 언어 모델 목적함수**를 공유합니다. 저자들은 이것이 fine-tuning 접근의 핵심 한계라고 주장합니다. 예를 들어 GPT는 left-to-right Transformer로, self-attention의 모든 토큰이 좌측 문맥만 볼 수 있습니다. 이는 **문장 수준 과제에는 sub-optimal**이고, **token-level 과제(특히 SQuAD QA에서 답 span 예측)** 처럼 양쪽 문맥이 결정적인 경우 매우 해롭습니다.

BERT의 해결책: **Masked Language Model(MLM)** 사전학습 목적. 입력 토큰의 일부를 무작위로 가리고, 모델이 그 토큰의 원래 vocabulary id를 예측합니다. 이 목적은 양쪽 문맥을 융합한 표현을 학습 가능하게 하며, 따라서 **deep bidirectional Transformer**를 사전학습할 수 있습니다. 추가로 **next sentence prediction** 으로 텍스트 쌍 표현도 함께 학습합니다.

논문의 명시적 기여 3가지:
1. 언어 표현을 위한 **양방향 사전학습의 중요성** 입증.
2. 사전학습된 표현이 **무거운 task-specific 구조의 필요성을 줄임**을 입증; BERT는 sentence-level과 token-level 과제 모두에서 task-specific 구조를 능가하는 **첫 번째 fine-tuning 기반 표현 모델**.
3. 11개 NLP 과제에서 SOTA 갱신; 코드와 사전학습 모델 공개 (https://github.com/google-research/bert).

**English**
The authors contrast two strategies for applying pre-trained representations:

- **Feature-based** (e.g., ELMo, Peters et al., 2018a): the pre-trained representations are added as features to a task-specific architecture.
- **Fine-tuning** (e.g., OpenAI GPT, Radford et al., 2018): minimal task-specific parameters are introduced; all pre-trained parameters are fine-tuned.

Both share **unidirectional language-modeling objectives** during pre-training. The authors argue this is the key limitation of the fine-tuning approach. GPT, for instance, uses a left-to-right Transformer where every token in self-attention can only attend to its left context. This is **sub-optimal for sentence-level tasks** and **harmful for token-level tasks** (e.g., predicting answer spans in SQuAD QA), where both directions matter.

BERT's remedy: the **Masked Language Model (MLM)** pre-training objective. Some input tokens are randomly masked, and the model must predict their original vocabulary ids. This objective fuses left and right context and thus enables pre-training a **deep bidirectional Transformer**. An auxiliary **next-sentence prediction** task jointly pre-trains text-pair representations.

The paper makes three explicit contributions:
1. Demonstrates the **importance of bidirectional pre-training** for language representations.
2. Shows that pre-trained representations **reduce the need for heavy task-specific architectures**; BERT is the **first fine-tuning-based representation model** to surpass task-specific architectures on a large suite of both sentence-level and token-level tasks.
3. Advances the state of the art on 11 NLP tasks; code and pre-trained models released at https://github.com/google-research/bert.

### Part II: Related Work (§2) / 관련 연구

**한국어**
세 갈래의 사전학습 전략이 다뤄집니다.

**§2.1 Unsupervised Feature-based Approaches.** word2vec, GloVe 같은 정적 word embedding부터 ELMo의 contextual embedding까지. ELMo는 left-to-right LM과 right-to-left LM을 **독립적으로** 학습한 뒤, 각 토큰의 표현을 두 LSTM의 hidden state concatenation으로 만듭니다. 이는 **shallowly bidirectional** 입니다 — 두 방향이 깊이 결합되지 않습니다. ELMo는 QA(SQuAD), 감성 분석(SST), NER 등 여러 벤치마크에서 SOTA를 달성했습니다. Melamud et al. (2016)은 LSTM으로 양방향 cloze 과제를 시도했지만 feature-based이고 깊은 양방향성은 아닙니다.

**§2.2 Unsupervised Fine-tuning Approaches.** Dai & Le (2015), ULMFiT (Howard & Ruder, 2018), OpenAI GPT (Radford et al., 2018) 등이 사전학습된 contextual encoder를 다운스트림에 fine-tune합니다. GPT는 GLUE에서 당시 SOTA를 달성했습니다.

**§2.3 Transfer Learning from Supervised Data.** NLI(Conneau et al., 2017), MT(McCann et al., 2017) 같은 대규모 supervised data에서의 전이도 효과적임을 인정. 컴퓨터 비전의 ImageNet 사전학습이 좋은 비유.

**English**
Three strands of prior work are discussed.

**§2.1 Unsupervised Feature-based Approaches.** From static word embeddings (word2vec, GloVe) to ELMo's contextual embeddings. ELMo trains a left-to-right LSTM and a right-to-left LSTM **independently**, then concatenates their hidden states for each token. This is **shallowly bidirectional** — the two directions are not deeply fused. ELMo set SOTA on QA (SQuAD), sentiment (SST), and NER. Melamud et al. (2016) explored a bidirectional cloze task with LSTMs but the model is feature-based and not deeply bidirectional.

**§2.2 Unsupervised Fine-tuning Approaches.** Dai & Le (2015), ULMFiT (Howard & Ruder, 2018), and OpenAI GPT (Radford et al., 2018) fine-tune pre-trained contextual encoders on downstream tasks. GPT held the GLUE SOTA at the time.

**§2.3 Transfer Learning from Supervised Data.** Effective transfer also exists from large supervised datasets such as NLI (Conneau et al., 2017) and MT (McCann et al., 2017). Computer-vision ImageNet pre-training serves as a useful analogy.

### Part III: BERT (§3) / BERT 모델

**한국어**
BERT는 두 단계로 구성됩니다: **pre-training**과 **fine-tuning**. 사전학습 동안 unlabeled data에 대한 두 가지 task로 학습하며, fine-tuning에서는 사전학습된 파라미터로 초기화한 뒤 다운스트림 과제의 labeled data로 모든 파라미터를 학습합니다. **distinctive feature**는 다양한 과제에 걸쳐 통일된 architecture를 사용한다는 것 — 사전학습 architecture와 최종 다운스트림 architecture 사이에 거의 차이가 없습니다.

**Model Architecture (§3 본문).** Vaswani et al. (2017)의 multi-layer bidirectional Transformer encoder를 거의 그대로 사용. 표기:
- $L$: Transformer block 수 (layers)
- $H$: hidden size
- $A$: self-attention head 수
- feed-forward filter size: $4H$ (즉 H=768이면 3072, H=1024이면 4096)

두 모델 크기:
- **BERT-Base**: $L=12,\ H=768,\ A=12$, 총 110M 파라미터 — GPT와 비교 목적으로 동일 크기로 설정
- **BERT-Large**: $L=24,\ H=1024,\ A=16$, 총 340M 파라미터

GPT와의 결정적 차이: **BERT의 self-attention은 양방향**, GPT는 좌측만 attention할 수 있는 constrained self-attention.

**Input/Output Representations (§3 "Input/Output Representations").** 단일 문장과 문장 쌍을 모두 동일 token sequence로 처리할 수 있어야 합니다. "Sentence"는 임의의 연속 텍스트 span을 의미하며, 실제 언어학적 문장이 아닙니다.
- **WordPiece tokenizer** (Wu et al., 2016): 30,000 vocab.
- 모든 sequence의 첫 token은 `[CLS]`. 이 토큰의 최종 hidden state $C \in \mathbb{R}^H$가 분류 과제의 aggregate representation으로 사용됩니다.
- 두 문장은 `[SEP]`로 분리되며, 학습된 segment embedding $E_A, E_B$로도 구별됩니다.
- 입력 임베딩은 token + segment + position 임베딩의 **합** (Figure 2):

$$E_i = E^{\text{tok}}_i + E^{\text{seg}}_i + E^{\text{pos}}_i$$

**§3.1 Pre-training BERT.**

**Task #1: Masked LM.** 표준 LM은 좌→우 또는 우→좌만 가능; 단순한 양방향 conditioning은 다층 신경망에서 토큰이 자기 자신을 간접적으로 보게 만듭니다. 해결: **입력 토큰의 15%를 무작위로 가린 뒤 그 위치의 원래 토큰을 예측**하는 cloze 스타일 목적. 마스킹된 위치의 final hidden state만 vocabulary 위 softmax로 보내며, 전체 입력을 재구성하지 않습니다(denoising autoencoder와의 대비점).

mismatch 문제: `[MASK]` 토큰은 fine-tuning에서는 등장하지 않음. 이를 완화하기 위해 **80/10/10 마스킹 전략** 사용. 마스킹 후보 15% 중에서:
- 80% → `[MASK]` 로 대체
- 10% → 무작위 vocabulary 토큰으로 대체
- 10% → 변경하지 않음 (그러나 그 위치도 예측 대상)

이렇게 하면 Transformer encoder는 **어느 토큰을 예측해야 할지 모르는 상태에서 모든 입력 토큰의 distributional contextual representation을 유지**해야 합니다. 또한 random 대체는 전체의 1.5%(=15%×10%)에 불과해 언어 이해 능력에 거의 해를 주지 않습니다.

수식적으로 MLM 손실:

$$
\mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x}; \theta)
$$

여기서 $\mathcal{M}$은 마스킹된 위치 집합, $\tilde{x}$는 마스킹/교체된 입력.

**Task #2: Next Sentence Prediction.** 문장 간 관계(QA, NLI에서 결정적)는 LM이 직접 포착하지 못합니다. 이를 학습하기 위해, 입력의 50%는 (A, A의 실제 다음 문장 B)로 IsNext 라벨, 나머지 50%는 (A, 무작위 문장 B)로 NotNext 라벨로 만듭니다. `[CLS]` 토큰의 최종 hidden state $C$가 NSP에 사용됩니다 — 이 single neuron 분류기가 최종 모델에서 97-98%의 NSP accuracy를 달성합니다(footnote 5). NSP는 §5.1 ablation에서 QA와 NLI 성능에 매우 유익함이 확인됩니다.

**Pre-training data.** BooksCorpus(800M words) + English Wikipedia(2,500M words; 본문/리스트/표/헤더 제외, **document-level corpus**). Document-level이 중요한 이유: 긴 contiguous sequence를 추출해야 NSP와 long-range coherence를 학습할 수 있기 때문.

**§3.2 Fine-tuning BERT.** Self-attention은 single text와 text-pair를 통일된 방식으로 처리할 수 있으므로 fine-tuning은 단순합니다. 각 과제마다 input/output을 적절히 매핑하고 모든 파라미터를 end-to-end 학습:
- 입력: sentence A,B는 (1) paraphrasing의 sentence pair, (2) NLI의 hypothesis-premise, (3) QA의 question-passage, (4) text classification의 (text, ∅)에 매핑
- 출력: token-level task는 token representation $T_i$를 output layer로 (예: NER, QA span); classification task는 `[CLS]` representation $C$를 (예: NLI, sentiment).

Fine-tuning은 사전학습보다 훨씬 저렴 — single Cloud TPU에서 1시간, GPU에서 몇 시간이면 모든 결과 재현 가능 (e.g., SQuAD는 30분 fine-tuning으로 Dev F1 91.0%).

**English**
BERT has two stages: **pre-training** and **fine-tuning**. During pre-training the model is trained on unlabeled data over two self-supervised tasks. For fine-tuning, the model is initialized with pre-trained parameters and all of them are updated using the labeled data of the downstream task. A **distinctive feature** is the unified architecture across tasks — minimal difference between the pre-trained architecture and the final downstream architecture.

**Model Architecture (§3 main).** A multi-layer bidirectional Transformer encoder almost identical to Vaswani et al. (2017). Notation:
- $L$: number of Transformer blocks (layers)
- $H$: hidden size
- $A$: number of self-attention heads
- feed-forward filter size: $4H$ (so 3072 for H=768, 4096 for H=1024)

Two reported sizes:
- **BERT-Base**: $L=12,\ H=768,\ A=12$, **110M total params** — chosen identical to GPT for fair comparison
- **BERT-Large**: $L=24,\ H=1024,\ A=16$, **340M total params**

The decisive difference vs GPT: **BERT uses bidirectional self-attention**, whereas GPT uses constrained self-attention where every token attends only to its left.

**Input/Output Representations.** A single token sequence must unambiguously encode both single sentences and sentence pairs. "Sentence" here means an arbitrary span of contiguous text, not a linguistic sentence.
- **WordPiece tokenizer** (Wu et al., 2016) with 30,000 vocab.
- The first token of every sequence is `[CLS]`. Its final hidden state $C \in \mathbb{R}^H$ serves as the aggregate sequence representation for classification.
- Sentence pairs are separated by `[SEP]` and additionally distinguished by learned segment embeddings $E_A, E_B$.
- Input embedding is the **sum** of token, segment, and position embeddings (Figure 2):

$$E_i = E^{\text{tok}}_i + E^{\text{seg}}_i + E^{\text{pos}}_i$$

**§3.1 Pre-training BERT.**

**Task #1: Masked LM.** Standard LMs are L→R or R→L only; naive bidirectional conditioning lets a multi-layer network indirectly see each token. The fix: a **cloze-style objective that masks 15% of input tokens at random and predicts the originals** at those positions. Only the masked positions feed a softmax over the vocabulary; we do not reconstruct the entire input (contrast with denoising autoencoders).

**Mismatch problem**: the `[MASK]` token never appears at fine-tuning time. Mitigation: an **80/10/10 masking strategy**. Of the 15% mask candidates:
- 80% → replace with `[MASK]`
- 10% → replace with a random vocabulary token
- 10% → leave unchanged (but still predict)

This forces the Transformer encoder to **maintain a distributional contextual representation of every input token, since it does not know which positions will be predicted**. Random replacement happens for only 1.5% (=15%×10%) of tokens, which empirically does not hurt language understanding.

Formally, the MLM loss is:

$$
\mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x}; \theta)
$$

with $\mathcal{M}$ the masked positions and $\tilde{x}$ the corrupted input.

**Task #2: Next Sentence Prediction.** Inter-sentence relationships (crucial for QA and NLI) are not captured directly by language modeling. To learn them: 50% of training pairs are (A, B = the actual next sentence) labeled IsNext; the other 50% are (A, B = random sentence) labeled NotNext. The `[CLS]` final hidden state $C$ is used for NSP — this single classifier reaches 97–98% NSP accuracy in the final model (footnote 5). The §5.1 ablation later shows NSP is very helpful for QA and NLI.

**Pre-training data.** BooksCorpus (800M words) + English Wikipedia (2,500M words; only running text, no lists/tables/headers — a **document-level corpus**, which is critical for extracting long contiguous sequences and learning NSP/long-range coherence).

**§3.2 Fine-tuning BERT.** Self-attention naturally handles single text and text pairs with a unified mechanism, so fine-tuning is simple. For each task, swap in the right input/output and update all parameters end-to-end:
- Input mapping: sentence A,B becomes (1) sentence pair in paraphrasing, (2) hypothesis-premise in NLI, (3) question-passage in QA, (4) (text, ∅) in text classification.
- Output: token-level tasks feed token representations $T_i$ to an output layer (e.g., NER, QA span); classification tasks feed `[CLS]` representation $C$ (e.g., NLI, sentiment).

Fine-tuning is far cheaper than pre-training — every result can be reproduced in **at most 1 hour on a single Cloud TPU or a few hours on a GPU** (e.g., SQuAD reaches Dev F1 91.0% in ~30 minutes of fine-tuning).

### Part IV: Experiments (§4) / 실험

**한국어**
11개 NLP 과제 결과를 보고합니다.

**§4.1 GLUE.** General Language Understanding Evaluation (Wang et al., 2018a)은 9개 다양한 NLU 과제. fine-tuning input은 §3 그대로 사용; `[CLS]` representation $C$ 위에 새로운 weight $W \in \mathbb{R}^{K \times H}$만 추가(K = label 개수). 손실은 $\log\,\mathrm{softmax}(CW^\top)$.

설정: batch 32, 3 epochs. learning rate는 {5e-5, 4e-5, 3e-5, 2e-5} 중 Dev에서 최적값 선택. BERT-Large는 작은 데이터셋에서 fine-tuning이 불안정해 random restart 후 best dev model 선택.

**Table 1 결과 (Test)**:
| System | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| **BERT-Base** | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | **79.6** |
| **BERT-Large** | 86.7/85.9 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | **82.1** |

BERT-Base는 모든 과제에서 GPT를 앞서며 평균 4.5% 향상. BERT-Large는 7.0% 향상. GLUE 리더보드 점수는 BERT-Large 80.5 vs GPT 72.8 (+7.7%).

**§4.2 SQuAD v1.1.** 100k crowd-sourced QA 쌍; 답은 항상 passage 내 contiguous span. Question은 segment A, passage는 segment B로 packing. Fine-tuning에서 도입되는 **유일한 새 파라미터**: start vector $S \in \mathbb{R}^H$, end vector $E \in \mathbb{R}^H$. 위치 $i$가 답의 시작일 확률:

$$
P_i^{(\text{start})} = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}
$$

end도 마찬가지. span 점수는 $S \cdot T_i + E \cdot T_j$ ($j \ge i$); 학습 손실은 정답 start/end의 log-likelihood 합. 3 epochs, lr=5e-5, batch=32.

**Table 2 결과**:
- BERT-Large (Single): Dev EM 84.1 / F1 90.9, Test F1 — / —
- BERT-Large + TriviaQA pre-fine-tune (Single): Test EM 85.1, F1 91.8
- BERT-Large + TriviaQA (Ensemble): Test EM 87.4, **F1 93.2**
- 인간 성능: F1 91.2 — BERT가 인간을 능가.
- Ensemble은 prior best ensemble을 +1.5 F1, single은 +1.3 F1 능가.

**§4.3 SQuAD v2.0.** Passage에 답이 없을 수도 있는 더 현실적인 과제. no-answer는 `[CLS]` 위치를 start=end로 두는 방식으로 처리. no-answer 점수: $s_{\text{null}} = S \cdot C + E \cdot C$. 최선의 non-null span 점수 $\hat{s}_{i,j} = \max_{j \ge i} S \cdot T_i + E \cdot T_j$가 $s_{\text{null}} + \tau$ 보다 클 때만 답 출력 ($\tau$는 dev set에서 F1 최대화로 선택). 2 epochs, lr=5e-5, batch=48.

**Table 3 결과**: BERT-Large Test EM 80.0 / F1 83.1 — 이전 best system 대비 **+5.1 F1** 향상.

**§4.4 SWAG.** 113k commonsense inference 문장-쌍 완성 과제. 4개 후보 continuation 중 가장 그럴듯한 것 선택. 각 후보마다 (A=주어진 sentence, B=후보)를 BERT에 넣고 `[CLS]` representation $C$와 학습되는 vector의 dot product로 점수화 후 4-way softmax. 3 epochs, lr=2e-5, batch=16.

**Table 4 결과**: BERT-Large Test 86.3 — ESIM+ELMo 대비 **+27.1**, GPT 대비 **+8.3**.

**English**
The paper reports results on 11 NLP tasks.

**§4.1 GLUE.** GLUE (Wang et al., 2018a) consists of 9 diverse NLU tasks. The fine-tuning input uses §3 directly; the only new parameters are a classification weight $W \in \mathbb{R}^{K \times H}$ on top of `[CLS]` representation $C$ (K = number of labels). Loss is $\log\,\mathrm{softmax}(CW^\top)$.

Setup: batch 32, 3 epochs, learning rate searched in {5e-5, 4e-5, 3e-5, 2e-5} on Dev. BERT-Large fine-tuning is sometimes unstable on small datasets, so the authors run random restarts and pick the best Dev model.

**Table 1 (Test)**:
| System | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| **BERT-Base** | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | **79.6** |
| **BERT-Large** | 86.7/85.9 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | **82.1** |

BERT-Base beats GPT on every task, with an average gain of 4.5%; BERT-Large gains 7.0%. The GLUE leaderboard score is **BERT-Large 80.5 vs GPT 72.8 (+7.7% absolute)**.

**§4.2 SQuAD v1.1.** 100k crowd-sourced QA pairs; the answer is always a contiguous span in the passage. Question is packed as segment A, passage as segment B. The **only new parameters introduced at fine-tuning** are a start vector $S \in \mathbb{R}^H$ and an end vector $E \in \mathbb{R}^H$. The probability that position $i$ is the answer start is:

$$
P_i^{(\text{start})} = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}
$$

Analogous for end. The span score is $S \cdot T_i + E \cdot T_j$ with $j \ge i$. Training loss is the sum of log-likelihoods of correct start and end positions. 3 epochs, lr=5e-5, batch=32.

**Table 2 results**:
- BERT-Large (Single): Dev EM 84.1 / F1 90.9
- BERT-Large + TriviaQA pre-fine-tune (Single): Test EM 85.1 / F1 91.8
- BERT-Large + TriviaQA (Ensemble): Test EM 87.4 / **F1 93.2**
- Human: F1 91.2 — BERT exceeds human performance.
- Ensemble beats prior best ensemble by +1.5 F1; single by +1.3 F1.

**§4.3 SQuAD v2.0.** A more realistic task allowing passages with no answer. No-answer is encoded by putting start=end at the `[CLS]` position. The no-answer score is $s_{\text{null}} = S \cdot C + E \cdot C$. The best non-null score $\hat{s}_{i,j} = \max_{j \ge i} S \cdot T_i + E \cdot T_j$ is output only when $\hat{s}_{i,j} > s_{\text{null}} + \tau$, with $\tau$ chosen on Dev to maximize F1. 2 epochs, lr=5e-5, batch=48.

**Table 3**: BERT-Large Test EM 80.0 / F1 83.1 — **+5.1 F1** over previous best.

**§4.4 SWAG.** 113k commonsense inference sentence-pair completion examples; pick the most plausible continuation among 4. For each candidate, (A=given sentence, B=candidate) is fed to BERT and scored by the dot product between `[CLS]` representation $C$ and a learned vector, followed by a 4-way softmax. 3 epochs, lr=2e-5, batch=16.

**Table 4**: BERT-Large Test 86.3 — **+27.1 over ESIM+ELMo, +8.3 over GPT**.

### Part V: Ablation Studies (§5) / 어블레이션

**한국어**

**§5.1 Effect of Pre-training Tasks.** BERT-Base architecture로 동일 데이터/스킴 하에서 사전학습 task 변형:
- **No NSP**: MLM은 그대로, NSP만 제거.
- **LTR & No NSP**: MLM 대신 standard left-to-right LM, NSP도 제거. (GPT와 거의 비교 가능, 단 데이터/입력/fine-tune 스킴은 BERT와 동일.)

**Table 5 (Dev)**:
| Tasks | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
|---|---|---|---|---|---|
| BERT-Base | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| No NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & No NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| + BiLSTM | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

관찰:
- NSP 제거는 QNLI/MNLI/SQuAD에서 유의미하게 성능 하락.
- LTR-only는 MRPC와 SQuAD에서 큰 폭으로 하락(SQuAD F1 -10.6). BiLSTM을 위에 얹으면 SQuAD는 회복되지만 GLUE는 더 악화. 이는 deep bidirectionality가 ELMo 스타일의 shallow 양방향(LTR+RTL concatenation)보다 본질적으로 강력함을 시사.

**§5.2 Effect of Model Size.** 다양한 (L, H, A) 조합을 동일 hyperparam/절차로 사전학습.

**Table 6**: 5 random restart의 Dev set 평균.
| #L | #H | #A | LM ppl | MNLI-m | MRPC | SST-2 |
|---|---|---|---|---|---|---|
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 3 | 5.24 | 80.6 | 82.2 | 90.7 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | 86.6 | 87.8 | 93.7 |

크기를 키울수록 모든 4개 task에서 단조 향상 — 심지어 3,600 example의 작은 MRPC에서도. 저자들은 **충분히 사전학습된 모델은 fine-tuning을 통해 small task data에서도 더 큰 표현력의 이득을 본다**고 가설을 세웁니다(작은 random task head만 추가되므로). 이는 prior feature-based 연구가 "더 큰 모델은 작은 task에 도움 안 됨" 이라고 보고했던 것과 대조적입니다.

**§5.3 Feature-based Approach with BERT.** BERT를 fine-tuning이 아닌 feature extractor로 사용. CoNLL-2003 NER에서 활성화 추출 후 고정, 그 위에 768-dim BiLSTM 2층 + 분류기.

**Table 7 (CoNLL-2003 NER F1)**:
| Approach | Dev F1 |
|---|---|
| Fine-tune BERT-Large | 96.6 |
| Fine-tune BERT-Base | 96.4 |
| Embeddings only | 91.0 |
| Last hidden | 94.9 |
| Second-to-last hidden | 95.6 |
| Concat last 4 hidden | 96.1 |
| Weighted sum last 4 | 95.9 |
| Weighted sum all 12 | 95.5 |

상위 4 layer concat은 fine-tuning에 0.3 F1 차이로 근접 — BERT가 feature-based로도 효과적임을 보여줍니다.

**Appendix C 어블레이션**:
- **C.1 Number of Training Steps**: 1M step이 500k step 대비 MNLI에서 +1.0% 추가 향상. MLM은 LTR보다 약간 느리게 수렴하지만 절대 성능에서 거의 즉시 LTR을 능가 (Figure 5).
- **C.2 Masking Strategy**: 80/10/10이 최선. 100% MASK는 feature-based NER에서 큰 손실, 100% RND는 fine-tuning과 feature-based 모두 악화.

**English**

**§5.1 Effect of Pre-training Tasks.** Using BERT-Base architecture with identical data/fine-tune scheme:
- **No NSP**: MLM only, drop NSP.
- **LTR & No NSP**: standard left-to-right LM (no MLM, no NSP); roughly comparable to GPT but with BERT's data/input/fine-tune scheme.

**Table 5 (Dev)**:
| Tasks | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
|---|---|---|---|---|---|
| BERT-Base | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| No NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & No NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| + BiLSTM | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

Observations:
- Removing NSP significantly hurts QNLI/MNLI/SQuAD.
- LTR-only collapses on MRPC and especially SQuAD (F1 −10.6). Adding a BiLSTM on top recovers SQuAD partially but worsens GLUE — confirming that **deep bidirectionality is essentially stronger than ELMo-style shallow bidirectionality (LTR+RTL concatenation)**.

**§5.2 Effect of Model Size.** Varying (L, H, A) under identical hyper-params and procedure.

**Table 6** (Dev, average of 5 fine-tune restarts):
| #L | #H | #A | LM ppl | MNLI-m | MRPC | SST-2 |
|---|---|---|---|---|---|---|
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 3 | 5.24 | 80.6 | 82.2 | 90.7 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | 86.6 | 87.8 | 93.7 |

Scaling improves every task monotonically — even on MRPC, which has only 3,600 training examples. The authors hypothesize that **once a model is sufficiently pre-trained, fine-tuning enables small downstream tasks to benefit from larger, more expressive representations** (since only a small randomly initialized head is added). This contrasts with earlier feature-based reports that bigger pre-trained models did not help small tasks.

**§5.3 Feature-based Approach with BERT.** Use BERT as a fixed feature extractor on CoNLL-2003 NER: extract activations, freeze them, feed to a 2-layer 768-dim BiLSTM + classifier.

**Table 7 (CoNLL-2003 NER F1)**:
| Approach | Dev F1 |
|---|---|
| Fine-tune BERT-Large | 96.6 |
| Fine-tune BERT-Base | 96.4 |
| Embeddings only | 91.0 |
| Last hidden | 94.9 |
| Second-to-last hidden | 95.6 |
| Concat last 4 hidden | 96.1 |
| Weighted sum last 4 | 95.9 |
| Weighted sum all 12 | 95.5 |

Concatenating the top 4 layers is within 0.3 F1 of fine-tuning — BERT works well in feature-based mode too.

**Appendix C ablations**:
- **C.1 Training Steps**: 1M steps gives +1.0% MNLI over 500k. MLM converges slightly slower than LTR but **almost immediately surpasses LTR in absolute accuracy** (Figure 5).
- **C.2 Masking strategy**: 80/10/10 is best. 100% MASK hurts feature-based NER substantially; 100% RND hurts both modes.

### Part VI: Conclusion (§6) / 결론

**한국어**
주요 기여는 풍부한 **unsupervised pre-training**의 가치를 일반화하는 것 — 특히 **deep bidirectional architecture** 가 동일 사전학습 모델로 광범위한 NLP 과제를 해결할 수 있게 한다는 점입니다. 양방향성과 두 사전학습 task가 개선의 대부분을 만들어냈음이 §5.1로 입증됩니다.

**English**
The major contribution is generalizing the value of rich **unsupervised pre-training** — and showing that **deep bidirectional architecture** lets a single pre-trained model tackle a broad set of NLP tasks. §5.1 demonstrates that bidirectionality and the two pre-training tasks account for the bulk of the improvement.

---

## 3. Key Takeaways / 핵심 시사점

1. **Deep bidirectionality > shallow bidirectionality** — ELMo는 LTR과 RTL LSTM을 독립 학습 후 concat하는 shallow 양방향이지만, BERT는 모든 layer에서 양방향 self-attention으로 conditioning합니다. §5.1의 ablation에서 LTR+BiLSTM 보강조차 BERT의 deep bidirectional MLM에 미치지 못함을 보여줍니다 (SQuAD F1 84.9 vs 88.5).
   *Deep bidirectionality dominates ELMo-style shallow concatenation. ELMo independently trains LTR and RTL LSTMs and concatenates; BERT lets every layer condition on both sides via bidirectional self-attention. The §5.1 ablation shows that even an LTR+BiLSTM hybrid trails the deep bidirectional MLM on SQuAD (F1 84.9 vs 88.5).*

2. **MLM은 양방향 conditioning의 leakage 문제를 우회한다** — 다층 양방향 LM은 토큰이 자기 자신을 간접적으로 본다는 trivial leakage가 있습니다. MLM은 입력의 15%를 가린 뒤 그 위치만 예측함으로써 이 문제를 풀고, 80/10/10 마스킹으로 fine-tuning과의 mismatch를 추가로 완화합니다.
   *MLM resolves the bidirectional leakage problem. A naive deep bidirectional LM has trivial self-leakage: a token can see itself through layered self-attention. MLM masks 15% and predicts only those positions; the 80/10/10 mix further mitigates the pre-training/fine-tuning mismatch.*

3. **NSP는 sentence-pair task에서 명확히 유익하다** — §5.1에서 NSP 제거는 QNLI/MNLI/SQuAD에서 명확한 성능 하락을 가져옵니다. 비록 RoBERTa(2019) 등 이후 연구에서 NSP의 가치가 재평가되긴 했지만, BERT의 시점에서는 분명한 이득이었습니다.
   *NSP empirically helps sentence-pair tasks at BERT's scale. §5.1 shows clear drops on QNLI, MNLI, SQuAD when NSP is removed. Later work (RoBERTa, 2019) re-evaluated NSP's value, but at the BERT stage it was a meaningful gain.*

4. **사전학습된 BERT는 가벼운 task head만으로도 11개 과제 SOTA를 달성한다** — 분류는 `[CLS]` 위에 $W \in \mathbb{R}^{K \times H}$만 추가, QA는 start/end vector $S, E \in \mathbb{R}^H$만 추가. task-specific architecture engineering의 시대를 사실상 종식시킨 결과입니다.
   *A lightweight task head suffices. Classification adds only $W \in \mathbb{R}^{K \times H}$ over `[CLS]`; QA adds only start/end vectors $S, E \in \mathbb{R}^H$. This effectively ended the era of heavy task-specific architecture engineering.*

5. **모델 크기 scaling이 작은 데이터셋에도 도움이 된다** — Table 6은 단조 향상을 보여주며, 3,600 예제의 MRPC에서도 BERT-Large가 BERT-Base보다 우수합니다. 저자의 가설: 사전학습 표현이 충분히 풍부하면 작은 task head는 그 풍부함을 활용할 수 있다.
   *Scaling helps even small downstream tasks. Table 6 shows monotonic gains, and BERT-Large beats BERT-Base on MRPC (3.6k examples). Hypothesis: when pre-trained representations are rich enough, the small task head can exploit them.*

6. **입력 표현은 token + segment + position 임베딩의 단순한 합** — 학습된 (sinusoidal이 아닌) position embedding을 사용하며 최대 길이 512. 단순함에도 불구하고 sentence-pair task를 통합적으로 처리하기에 충분.
   *Input is the simple sum of token, segment, and position embeddings. Position embeddings are learned (not sinusoidal), max length 512. Despite simplicity, it unifies single-sentence and sentence-pair tasks cleanly.*

7. **Fine-tuning은 사전학습보다 한 자릿수 이상 저렴하다** — pre-training은 BERT-Large 기준 16 TPU × 4일이지만, downstream fine-tuning은 Cloud TPU 1대로 1시간 이내 또는 GPU 몇 시간이면 모든 결과 재현 가능. 이는 산업 채택을 가능케 한 핵심 실용 요인.
   *Fine-tuning is dramatically cheaper than pre-training. Pre-training BERT-Large needs 16 TPUs × 4 days, but every downstream result reproduces in ≤1 hour on a single Cloud TPU or a few GPU-hours — the practical factor that enabled industrial adoption.*

8. **BERT는 fine-tuning과 feature-based 모두에서 효과적** — §5.3에서 frozen 상위 4 hidden layer를 concat하면 NER에서 fine-tuning에 0.3 F1 차이로 근접 (96.1 vs 96.4). 즉 사전학습 모델 자체가 풍부한 contextual feature임을 시사합니다.
   *BERT works in both fine-tuning and feature-based modes. Concatenating the top 4 frozen hidden layers reaches within 0.3 F1 of fine-tuning on CoNLL-2003 NER (96.1 vs 96.4) — confirming that the pre-trained model itself is a rich contextual feature extractor.*

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Input Embedding / 입력 임베딩

각 토큰 $i$의 입력 표현은 token, segment, position 세 임베딩의 합:

$$
\boxed{ \quad E_i = E^{\text{tok}}_i + E^{\text{seg}}_i + E^{\text{pos}}_i \quad }
$$

| 기호 / Symbol | 정의 / Definition |
|---|---|
| $E^{\text{tok}}_i \in \mathbb{R}^H$ | WordPiece token embedding (vocab 30k) / WordPiece token embedding (30k vocab) |
| $E^{\text{seg}}_i \in \{E_A, E_B\} \subset \mathbb{R}^H$ | sentence A/B 표시 segment embedding / segment embedding marking sentence A or B |
| $E^{\text{pos}}_i \in \mathbb{R}^H$ | 학습된 position embedding ($i \in [0, 511]$) / learned position embedding |

### 4.2 Multi-Head Self-Attention (recap)

각 head $h$에서:

$$
\text{head}_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right) V_h, \quad d_k = H / A
$$

$$
\text{MHA}(X) = [\text{head}_1; \dots; \text{head}_A] W^O
$$

각 head는 hidden을 $A$개로 분할하여 (BERT-Base $A=12$, $d_k=64$) 병렬 attention을 수행. / Each head splits hidden into $A$ shards (Base: $A=12$, $d_k=64$) for parallel attention.

### 4.3 Transformer Block

각 layer $\ell$:

$$
\begin{aligned}
\tilde{H}^{(\ell)} &= \mathrm{LayerNorm}\!\big(H^{(\ell-1)} + \text{MHA}(H^{(\ell-1)})\big) \\
H^{(\ell)} &= \mathrm{LayerNorm}\!\big(\tilde{H}^{(\ell)} + \text{FFN}(\tilde{H}^{(\ell)})\big) \\
\text{FFN}(x) &= \mathrm{GELU}(x W_1 + b_1) W_2 + b_2
\end{aligned}
$$

BERT는 ReLU 대신 **GELU** 사용 (Hendrycks & Gimpel, 2016). FFN 내부 차원 = $4H$.

### 4.4 MLM Loss / MLM 손실

마스킹된 위치 집합 $\mathcal{M} \subset \{1, \dots, n\}$ ($|\mathcal{M}| \approx 0.15n$)에 대해:

$$
\boxed{ \quad \mathcal{L}_{\text{MLM}}(\theta) = - \sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x}; \theta) \quad }
$$

여기서 $\tilde{x}$는 80/10/10 규칙으로 변형된 입력. $P(x_i \mid \tilde{x}; \theta) = \mathrm{softmax}(W_{\text{vocab}} T_i + b)_{x_i}$, $T_i$는 위치 $i$의 최종 hidden state, $W_{\text{vocab}} \in \mathbb{R}^{V \times H}$ (output projection은 input token embedding과 weight tying 가능).

| 기호 / Symbol | 정의 / Definition |
|---|---|
| $\mathcal{M}$ | 마스킹된 위치 집합, 입력의 ~15% / set of masked positions, ~15% of input |
| $\tilde{x}$ | 80/10/10 규칙으로 변형된 입력 / corrupted input by 80/10/10 rule |
| $x_i$ | 원래 vocabulary token id / original vocabulary token id |
| $T_i \in \mathbb{R}^H$ | 위치 $i$의 최종 hidden state / final hidden state at position $i$ |
| $W_{\text{vocab}} \in \mathbb{R}^{V \times H}$ | vocabulary projection matrix |

### 4.5 NSP Loss / NSP 손실

`[CLS]`의 최종 hidden state $C \in \mathbb{R}^H$ 위에 binary classifier $W_{NSP} \in \mathbb{R}^{2 \times H}$:

$$
\boxed{ \quad \mathcal{L}_{\text{NSP}}(\theta) = - \log P(y_{ns} \mid C; \theta), \quad P(y_{ns} \mid C) = \mathrm{softmax}(W_{NSP} C)_{y_{ns}} \quad }
$$

$y_{ns} \in \{\texttt{IsNext}, \texttt{NotNext}\}$, 50:50 비율로 학습 데이터 생성.

### 4.6 Total Pre-training Loss / 총 사전학습 손실

$$
\mathcal{L}_{\text{pre-train}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

(단순 합; 별도 weighting 없음.)

### 4.7 SQuAD Span Prediction / SQuAD 답 위치 예측

추가 파라미터: start vector $S \in \mathbb{R}^H$, end vector $E \in \mathbb{R}^H$.

$$
P_i^{(\text{start})} = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}, \qquad
P_i^{(\text{end})} = \frac{e^{E \cdot T_i}}{\sum_j e^{E \cdot T_j}}
$$

Span $(i, j)$의 score: $S \cdot T_i + E \cdot T_j$ ($j \ge i$). 학습 손실:

$$
\mathcal{L}_{\text{SQuAD}} = - \log P_{i^*}^{(\text{start})} - \log P_{j^*}^{(\text{end})}
$$

$i^*, j^*$는 정답 span의 시작/끝.

**SQuAD 2.0 unanswerable** 처리:
- $s_{\text{null}} = S \cdot C + E \cdot C$ (no-answer score; $C$ = `[CLS]` hidden state)
- $\hat{s}_{i,j} = \max_{j \ge i} (S \cdot T_i + E \cdot T_j)$ (best non-null score)
- 답 출력 조건: $\hat{s}_{i,j} > s_{\text{null}} + \tau$, $\tau$는 dev에서 F1 최대화.

### 4.8 Worked Numerical Example / 수치 예시

**예시: MLM의 80/10/10 마스킹.**

원문: `my dog is hairy`

15% 마스킹 후보 = 4 토큰 × 0.15 = 0.6 토큰. 실제 한 sequence에서 모든 토큰이 후보가 될 수 있고 평균 15%가 마스킹된다. 4번째 토큰 `hairy`가 후보로 선택되었다고 하자. 그러면:

| 확률 / Probability | 변환 / Transform | 결과 / Result |
|---|---|---|
| 80% | `hairy` → `[MASK]` | `my dog is [MASK]` |
| 10% | `hairy` → 무작위 토큰 (예: `apple`) | `my dog is apple` |
| 10% | `hairy` 유지 | `my dog is hairy` |

세 경우 모두에서 위치 4의 final hidden state $T_4$로 원래 토큰 `hairy`를 예측. 이 절차로 인해 Transformer는 **어느 위치가 예측 대상인지 모르므로 모든 입력 토큰의 풍부한 contextual representation을 유지해야 합니다**.

**파라미터 수 추정 (BERT-Base, L=12, H=768, A=12)**:
- Token embedding: $30000 \times 768 \approx 23M$
- Position embedding: $512 \times 768 \approx 0.4M$
- Segment embedding: $2 \times 768 \approx 0.001M$
- Per layer self-attention: $4 \times H^2 = 4 \times 768^2 \approx 2.36M$ (Q,K,V,O projections)
- Per layer FFN: $2 \times H \times 4H = 8H^2 \approx 4.72M$
- Per layer LN/biases: ~수 K
- Per layer 합계 ≈ 7.1M, $\times 12$ ≈ 85M
- 합계 ≈ 23 + 0.4 + 85 ≈ **약 109M ≈ 110M** ✓

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1953 ── Cloze procedure (Taylor)                     │ Reading-test fill-in-the-blank
                                                     │ → MLM의 직접적 영감
                                                     │
2003 ── Neural LM (Bengio)                           │ First neural language model
                                                     │
2013 ── word2vec (Mikolov)                           │ Distributed word representations
2014 ── GloVe (Pennington)                           │ Co-occurrence word vectors
                                                     │
2015 ── Seq2Seq + Attention (Bahdanau)               │ Soft alignment
2015 ── Semi-supervised seq learning (Dai & Le)      │ Early LM pre-train + fine-tune
                                                     │
2016 ── context2vec (Melamud)                        │ BiLSTM cloze (feature-based)
                                                     │
2017 ── Transformer (Vaswani) ◀────────── paper 27   │ Self-attention encoder/decoder
                                                     │
2018 ── ELMo (Peters)                                │ Shallow bi-directional features
2018 ── ULMFiT (Howard & Ruder)                      │ LM transfer for classification
2018 ── GPT-1 (Radford)                              │ LTR Transformer fine-tune
        ┌─────────────────────────────────────┐
2018 ── │  BERT (Devlin) ◀── THIS PAPER       │      │ Deep bidirectional + MLM + NSP
        └─────────────────────────────────────┘
                                                     │
2019 ── XLNet (Yang)                                 │ Permutation LM
2019 ── RoBERTa (Liu)                                │ Better trained BERT, drops NSP
2019 ── ALBERT (Lan)                                 │ Parameter sharing
2019 ── DistilBERT (Sanh)                            │ Knowledge distillation
                                                     │
2020 ── ELECTRA (Clark)                              │ Replaced-token-detection
2020 ── T5 (Raffel)                                  │ Text-to-text unification
2020 ── GPT-3 (Brown)                                │ 175B param scaling
                                                     │
2021 ── DeBERTa (He)                                 │ Disentangled attention
2021 ── BEiT, MAE                                    │ MLM idea → vision (masked images)
2021 ── HuBERT, wav2vec 2.0                          │ MLM idea → speech
                                                     │
2022 ── ChatGPT (decoder-only mainstream)            │ Encoders still backbone for
                                                     │ search/retrieval/embeddings
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Vaswani et al. 2017 — Attention Is All You Need (paper 27) | BERT의 architecture는 이 논문의 Transformer encoder를 거의 그대로 차용 / BERT's architecture is essentially this paper's Transformer encoder | **Direct prerequisite / 직접 선행 논문**: BERT를 이해하려면 self-attention, multi-head attention, position encoding을 먼저 이해해야 함 / Cannot understand BERT without first understanding self-attention |
| Peters et al. 2018a — ELMo | feature-based, shallowly bidirectional 사전학습 표현 / Feature-based, shallow bidirectional contextualized embeddings | **Direct competitor / 직접 비교 대상**: BERT는 ELMo의 shallow 양방향성을 deep 양방향으로 일반화 / BERT generalizes ELMo's shallow bidirectionality to deep bidirectionality |
| Radford et al. 2018 — GPT-1 | LTR Transformer + fine-tuning 패러다임 / LTR Transformer + fine-tuning paradigm | **Direct competitor / 직접 비교 대상**: BERT-Base는 GPT와 동일 크기로 설정해 양방향성의 효과를 isolate / BERT-Base has the same size as GPT to isolate the effect of bidirectionality |
| Howard & Ruder 2018 — ULMFiT | text classification 위한 LM fine-tuning | 사전학습+fine-tuning 패러다임의 초기 성공 사례 / Early success of pre-train + fine-tune paradigm |
| Taylor 1953 — Cloze procedure | 빈칸 채우기 reading test / Fill-in-the-blank reading comprehension | **MLM의 직접적 영감** / Direct inspiration for MLM |
| Liu et al. 2019 — RoBERTa | BERT 후속, 더 긴 학습 + 더 큰 batch + NSP 제거로 성능 향상 / Follow-up: longer training, bigger batches, drops NSP | BERT의 ablation을 재평가; NSP의 가치가 과대평가되었을 수 있음을 시사 / Re-evaluates BERT ablations; suggests NSP value may be overestimated |
| Lan et al. 2019 — ALBERT | parameter sharing + factorized embedding으로 BERT 압축 / Parameter sharing + factorized embedding compression | BERT 계열 효율화 / Efficiency variant of the BERT family |
| Clark et al. 2020 — ELECTRA | MLM 대신 replaced-token detection으로 더 sample-efficient / Replaced-token detection as a more sample-efficient pre-training objective | MLM 대안 사전학습 목적 / Alternative to MLM as pre-training objective |
| Raffel et al. 2020 — T5 | 모든 NLP 과제를 text-to-text로 통일 / Unifies all NLP tasks as text-to-text | encoder-only(BERT) vs encoder-decoder(T5) vs decoder-only(GPT) 이분법의 정립 / Cements the encoder-only vs encoder-decoder vs decoder-only taxonomy |

---

## 7. References / 참고문헌

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT 2019, pp. 4171–4186. [DOI: 10.18653/v1/N19-1423; arXiv:1810.04805]
- Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.
- Peters, M. E. et al. "Deep Contextualized Word Representations." NAACL 2018a (ELMo).
- Radford, A. et al. "Improving Language Understanding by Generative Pre-Training." OpenAI Tech Report 2018 (GPT-1).
- Taylor, W. L. "Cloze Procedure: A New Tool for Measuring Readability." Journalism Bulletin, 1953.
- Howard, J. & Ruder, S. "Universal Language Model Fine-tuning for Text Classification." ACL 2018 (ULMFiT).
- Wu, Y. et al. "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation." arXiv:1609.08144, 2016 (WordPiece).
- Hendrycks, D. & Gimpel, K. "Gaussian Error Linear Units (GELUs)." arXiv:1606.08415, 2016.
- Wang, A. et al. "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding." EMNLP BlackboxNLP 2018a.
- Rajpurkar, P. et al. "SQuAD: 100,000+ Questions for Machine Comprehension of Text." EMNLP 2016.
- Zellers, R. et al. "SWAG: A Large-scale Adversarial Dataset for Grounded Commonsense Inference." EMNLP 2018.
- Liu, Y. et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv:1907.11692, 2019.
- Melamud, O. et al. "context2vec: Learning Generic Context Embedding with Bidirectional LSTM." CoNLL 2016.
- Lan, Z. et al. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." ICLR 2020.
- Clark, K. et al. "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators." ICLR 2020.
- Raffel, C. et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR 2020 (T5).
- Source code & pre-trained models: https://github.com/google-research/bert
