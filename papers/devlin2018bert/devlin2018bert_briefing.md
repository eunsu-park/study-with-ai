---
title: "Pre-Reading Briefing: BERT — Pre-training of Deep Bidirectional Transformers for Language Understanding"
paper_id: "28_devlin_2018"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT 2019, pp. 4171–4186.
**Author(s)**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)
**Year**: 2018 (preprint) / 2019 (NAACL)
**DOI**: 10.18653/v1/N19-1423 (arXiv:1810.04805)

---

## 1. 핵심 기여 / Core Contribution

**한국어**
BERT(Bidirectional Encoder Representations from Transformers)는 **Transformer encoder**를 **양방향(deep bidirectional)** 으로 사전학습(pre-training)하는 표현 학습 모델입니다. 핵심은 두 가지 비지도 사전학습 과제입니다. 첫째, **Masked Language Model(MLM)** — 입력 토큰의 15%를 무작위로 마스킹하고, 모델이 해당 토큰을 좌우 양쪽 문맥(context)으로부터 예측하도록 합니다. 이는 표준 단방향(left-to-right) 언어 모델의 한계를 깨고, 모든 layer에서 좌우 문맥을 동시에 conditioning할 수 있게 만듭니다. 둘째, **Next Sentence Prediction(NSP)** — 두 문장 A, B가 주어졌을 때 B가 A의 실제 다음 문장인지 무작위 문장인지를 분류합니다. 이는 QA, NLI 같은 문장 쌍 과제에 유용한 표현을 학습합니다. BERT는 11개 NLP 벤치마크에서 SOTA를 갱신했습니다: GLUE 80.5(+7.7%), MultiNLI 86.7%, SQuAD 1.1 F1 93.2, SQuAD 2.0 F1 83.1.

**English**
BERT (Bidirectional Encoder Representations from Transformers) is a representation-learning model that pre-trains a **Transformer encoder** in a **deeply bidirectional** manner. Its contribution rests on two unsupervised pre-training tasks. First, the **Masked Language Model (MLM)** randomly masks 15% of input tokens and forces the model to predict them using **both left and right context simultaneously**, breaking the unidirectionality constraint of standard left-to-right language models and enabling all Transformer layers to jointly condition on both directions. Second, **Next Sentence Prediction (NSP)** trains the model to classify whether sentence B is the actual next sentence after sentence A — useful for sentence-pair tasks like Question Answering (QA) and Natural Language Inference (NLI). BERT advances the state of the art on 11 NLP benchmarks: GLUE 80.5 (+7.7% absolute), MultiNLI 86.7%, SQuAD 1.1 F1 93.2, SQuAD 2.0 F1 83.1.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2017년 Vaswani et al.의 Transformer 등장 이후, NLP는 빠르게 사전학습-미세조정(pre-train then fine-tune) 패러다임으로 이동했습니다. 2018년 시점, 두 갈래의 사전학습 방법이 경쟁했습니다: (1) **feature-based** 접근 — ELMo(Peters et al., 2018a)는 좌→우, 우→좌 LSTM을 독립적으로 학습한 뒤 그 표현을 task-specific 모델에 입력으로 추가; (2) **fine-tuning** 접근 — OpenAI GPT(Radford et al., 2018)는 left-to-right Transformer를 사전학습한 뒤 모든 파라미터를 다운스트림 과제에 미세조정. 두 접근 모두 **단방향 언어 모델 목적함수**를 사용하기 때문에 token-level 과제(예: QA의 answer span 예측)에서 양쪽 문맥을 활용하지 못한다는 한계가 있었습니다. BERT는 이 단방향 제약을 깨는 것을 목표로 합니다.

**English**
After Vaswani et al.'s 2017 Transformer, NLP rapidly shifted to the **pre-train then fine-tune** paradigm. By 2018, two competing strategies dominated: (1) the **feature-based** approach — ELMo (Peters et al., 2018a) trained left-to-right and right-to-left LSTMs independently and concatenated their hidden states as additional features for task-specific architectures; (2) the **fine-tuning** approach — OpenAI GPT (Radford et al., 2018) pre-trained a left-to-right Transformer and fine-tuned all parameters on downstream tasks. Both approaches relied on **unidirectional language-modeling objectives**, which is sub-optimal for token-level tasks (e.g., predicting answer spans in QA) where both left and right context matter. BERT explicitly targets this unidirectionality limitation.

### 타임라인 / Timeline

```
2013 ── word2vec (Mikolov)            : static word embeddings
2014 ── GloVe (Pennington)            : co-occurrence-based word vectors
2015 ── Seq2Seq + Attention (Bahdanau)
2017 ── Transformer (Vaswani)         : self-attention replaces RNN
2018 ── ELMo (Peters)                 : context-sensitive features (BiLSTM)
2018 ── ULMFiT (Howard & Ruder)       : LM fine-tuning for text classification
2018 ── GPT-1 (Radford)               : LTR Transformer + fine-tuning
2018 ── BERT (Devlin) ◀── this paper  : bidirectional Transformer + MLM + NSP
2019 ── GPT-2 (Radford)               : larger LTR Transformer, zero-shot
2019 ── XLNet, RoBERTa, ALBERT        : variants of bidirectional pre-training
2020 ── T5, GPT-3                     : unification + scaling
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Transformer encoder**: multi-head self-attention, position-wise feed-forward, residual connections, layer normalization (Vaswani et al., 2017 — paper 27)
- **언어 모델(Language Model)**: 토큰 시퀀스의 확률을 추정하는 모델; 표준 LM은 $P(x_t \mid x_{<t})$로 좌측 문맥만 사용
- **Word embeddings & subword tokenization**: word2vec, GloVe, WordPiece (30,000 vocab)
- **Cross-entropy loss & softmax**
- **Fine-tuning vs feature extraction** 차이
- **GLUE/SQuAD 벤치마크 구조**: GLUE는 9개 분류/회귀 과제, SQuAD 1.1은 추출형 QA, SQuAD 2.0은 답이 없을 수도 있는 QA

**English**
- **Transformer encoder**: multi-head self-attention, position-wise feed-forward, residuals, layer norm (Vaswani et al., 2017 — paper 27)
- **Language modeling**: estimating $P(x_t \mid x_{<t})$ for token sequences; standard LMs are unidirectional
- **Word embeddings & subword tokenization**: word2vec, GloVe, WordPiece (30k vocab)
- **Cross-entropy loss & softmax**
- Distinction between **fine-tuning** and **feature-based** transfer
- **GLUE/SQuAD benchmark structure**: GLUE = 9 classification/regression tasks; SQuAD 1.1 = extractive QA; SQuAD 2.0 = QA with possibly unanswerable questions

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **MLM (Masked LM)** | 입력 토큰의 15%를 마스킹하고 양방향 문맥으로부터 예측 / Predict 15% of randomly masked tokens using both-side context (Cloze task style) |
| **NSP (Next Sentence Prediction)** | 두 문장이 실제 연속인지 분류하는 이진 과제 / Binary task: is sentence B the actual next sentence after A? |
| **WordPiece** | subword tokenizer (vocab 30,000); rare words decomposed into pieces (e.g., "playing" → "play" + "##ing") / Subword tokenizer with 30k vocab |
| **`[CLS]` token** | 모든 입력 시퀀스의 첫 번째 위치 특수 토큰; 분류 과제의 aggregate representation으로 사용 / Special first token; final hidden state used as sequence-level classification feature |
| **`[SEP]` token** | 두 문장을 구분하거나 시퀀스 끝을 표시하는 특수 토큰 / Separator token between sentence pairs |
| **Segment embedding** | 토큰이 sentence A인지 B인지 표시하는 학습된 임베딩 ($E_A, E_B$) / Learned embedding ($E_A, E_B$) marking which sentence a token belongs to |
| **Position embedding** | 학습된 위치 임베딩 (sinusoidal이 아닌 learned), 최대 512 / Learned (not sinusoidal) positional embedding, max length 512 |
| **Fine-tuning** | 사전학습된 모든 파라미터를 다운스트림 과제 데이터로 함께 학습 / Update all pre-trained parameters end-to-end on downstream task |
| **Feature-based** | 사전학습 모델은 동결, 그 hidden state만 task-specific 모델 입력으로 사용 (ELMo 스타일) / Freeze pre-trained model, use its activations as input features (ELMo-style) |
| **GLUE** | General Language Understanding Evaluation; 9개 이질적 NLU 과제 모음 / Benchmark of 9 diverse NLU tasks |
| **SQuAD** | Stanford Question Answering Dataset; 1.1은 답이 항상 존재, 2.0은 unanswerable 포함 / Extractive QA dataset; 2.0 adds unanswerable questions |
| **Cloze task** | 빈칸 채우기 문제; MLM의 영감 (Taylor 1953) / Fill-in-the-blank task; MLM is its modern incarnation |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 입력 표현 / Input Representation

$$
E_i = E^{tok}_i + E^{seg}_i + E^{pos}_i
$$

각 토큰의 입력 임베딩은 token, segment, position 임베딩의 합. / Each input embedding is the sum of token, segment, and position embeddings.

### 5.2 Multi-Head Self-Attention (recap)

$$
\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

### 5.3 MLM 손실 / MLM Loss

$$
\mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x}; \theta)
$$

여기서 $\mathcal{M}$은 마스킹된 위치 집합 (입력의 15%), $\tilde{x}$는 마스킹된 입력. / $\mathcal{M}$ is the masked positions (15%), $\tilde{x}$ the corrupted input.

### 5.4 NSP 손실 / NSP Loss

$$
\mathcal{L}_{\text{NSP}} = - \log P(y_{ns} \mid C; \theta), \quad y_{ns} \in \{\texttt{IsNext}, \texttt{NotNext}\}
$$

`[CLS]` 토큰의 최종 hidden state $C$ 위에 sigmoid/softmax 분류 head. / Classification head over the final `[CLS]` hidden state $C$.

### 5.5 SQuAD Span 예측 / SQuAD Span Prediction

$$
P_i^{(\text{start})} = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}, \qquad
P_i^{(\text{end})} = \frac{e^{E \cdot T_i}}{\sum_j e^{E \cdot T_j}}
$$

start vector $S$, end vector $E$를 새로 도입하여 span 시작/끝을 dot-product로 점수화. / Introduce start vector $S$ and end vector $E$; score span endpoints by dot product with token representations.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
1. **Section 1 (Introduction)**: 단방향 vs 양방향 사전학습의 차이를 명확히 이해하세요. 왜 LTR이 token-level QA에 부적합한지가 핵심 동기.
2. **Section 2 (Related Work)**: ELMo와 GPT의 차이, 그리고 BERT가 두 접근의 어떤 점을 결합/개선하는지 비교.
3. **Section 3 (BERT)**: 가장 중요. MLM의 80/10/10 마스킹 전략, NSP 데이터 생성, `[CLS]`/`[SEP]`/segment embedding 구조를 그림 1, 2와 함께 정독.
4. **Section 4 (Experiments)**: 11개 과제 결과를 표 1–4로 확인. GLUE 평균, SQuAD F1, SWAG accuracy의 절대적 향상폭에 주목.
5. **Section 5 (Ablation)**: §5.1 양방향성의 중요성(NSP 제거, LTR 대체 효과), §5.2 model size의 영향, §5.3 feature-based 사용법.
6. **Appendix A.1**: MLM의 80/10/10 의 구체 예시(`my dog is hairy`)는 직관 형성에 매우 유용.

**English**
1. **Section 1**: Understand the unidirectional-vs-bidirectional pre-training distinction. The motivation hinges on why LTR is sub-optimal for token-level QA.
2. **Section 2**: Compare ELMo and GPT carefully — note what BERT inherits from each and what it changes.
3. **Section 3**: Most important. Read alongside Figures 1 & 2: the 80/10/10 masking strategy, NSP data construction, `[CLS]`/`[SEP]`/segment-embedding architecture.
4. **Section 4**: Tables 1–4 list the 11-task results. Focus on absolute improvements on GLUE Avg, SQuAD F1, and SWAG accuracy.
5. **Section 5**: §5.1 (importance of bidirectionality via No-NSP and LTR ablations), §5.2 (effect of model size), §5.3 (BERT as a feature extractor).
6. **Appendix A.1**: The concrete `my dog is hairy` example for the 80/10/10 mask is very helpful for intuition.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
BERT는 **사전학습-미세조정 패러다임을 NLP의 표준**으로 확립한 분수령적 논문입니다. 이 논문 이후 RoBERTa, ALBERT, ELECTRA, DeBERTa 등 수많은 변형이 등장했고, encoder-only 모델 계열의 기반이 되었습니다. 또한 MLM 아이디어는 vision (BEiT, MAE), speech (HuBERT, wav2vec 2.0), code (CodeBERT) 등 다른 modality로 광범위하게 전파되었습니다. 한편 GPT 계열의 decoder-only 생성 모델이 ChatGPT 시대에 주류가 되었지만, encoder-style BERT는 여전히 검색(query-document encoding), classification, embedding 산업 응용의 백본입니다. **"사전학습된 표현 + 가벼운 task head"** 라는 단순한 레시피의 위력을 결정적으로 입증했다는 점이 가장 큰 유산입니다.

**English**
BERT is the **watershed paper that cemented the pre-train then fine-tune paradigm as NLP's default**. It spawned a vast family of variants — RoBERTa, ALBERT, ELECTRA, DeBERTa — and became the foundation for the encoder-only model lineage. The MLM idea also propagated to other modalities: vision (BEiT, MAE), speech (HuBERT, wav2vec 2.0), code (CodeBERT). While the GPT-style decoder-only generative line has taken the spotlight in the ChatGPT era, encoder-style BERT remains the backbone of industrial search (query-document encoding), classification, and embedding services. Its enduring legacy is the decisive empirical demonstration that **"pre-trained representations + a lightweight task head"** suffices for state-of-the-art performance across diverse NLP tasks.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
