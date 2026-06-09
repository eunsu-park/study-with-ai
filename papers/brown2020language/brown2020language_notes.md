---
title: "Language Models are Few-Shot Learners"
authors: [Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei]
year: 2020
journal: "Advances in Neural Information Processing Systems (NeurIPS) 2020"
doi: "arXiv:2005.14165"
topic: Artificial_Intelligence
tags: [language-model, GPT-3, in-context-learning, few-shot, transformer, scaling, foundation-model, NLP]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 34. Language Models are Few-Shot Learners (GPT-3) / 언어 모델은 Few-Shot 학습자

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 OpenAI가 학습시킨 **1,750억 파라미터 autoregressive transformer 언어 모델 GPT-3**를 발표하고, 이 모델이 **gradient update나 fine-tuning 없이** 단지 자연어 instruction과 0–100개의 demonstration을 프롬프트에 넣어주는 것만으로 수십 가지 NLP 작업을 수행할 수 있음을 보입니다. 저자들은 이 패러다임을 세 가지 설정으로 구분합니다 — **zero-shot** ($K=0$, 자연어 지시만), **one-shot** ($K=1$), **few-shot** ($K$가 context window에 들어가는 만큼; 보통 10–100). 모든 경우에 모델 가중치 $\theta$는 추론 시간에 단 한 번도 업데이트되지 않으며, "학습"은 forward pass 안에서 prompt의 패턴을 인식·일반화하는 형태로 일어납니다 — 이를 **in-context learning**이라 부릅니다.

핵심 결과는 두 가지입니다. **첫째**, GPT-3는 fine-tuning 없이도 LAMBADA cloze test에서 86.4% (이전 SOTA 대비 +18%), TriviaQA closed-book QA에서 71.2% few-shot, PTB perplexity 20.5 (이전 35.8 대비 -43%), 2-digit 덧셈 100%, 5개 prompt만으로 SuperGLUE에서 fine-tuned BERT-Large를 능가하는 등의 강력한 성능을 보입니다. **둘째**, 그리고 더 중요한 것은, 125M부터 175B까지 **8개 모델 사이즈** (3 orders of magnitude)에 걸쳐 학습한 결과, **모델 크기가 커질수록 few-shot의 zero-shot 대비 우위가 증가**한다는 점입니다 — 즉 큰 모델은 단순히 더 잘하는 것이 아니라 **in-context learning 자체를 더 잘 한다**. 이는 사전학습이 암묵적으로 "learning to learn" 메타학습 outer loop 역할을 한다는 가설(Figure 1.1)을 정량적으로 뒷받침합니다.

**English**
This paper introduces **GPT-3**, a 175-billion-parameter autoregressive transformer language model trained by OpenAI, and demonstrates that it can perform dozens of NLP tasks **with no fine-tuning and no gradient updates whatsoever** — purely through a natural-language instruction and 0–100 input/output demonstrations placed in the prompt. The authors crystallize three settings: **zero-shot** ($K=0$, instruction only), **one-shot** ($K=1$), **few-shot** ($K$ as many demonstrations as fit in the 2048-token context, typically 10–100). In all cases the weights $\theta$ are frozen at inference; "learning" happens entirely within a forward pass that recognizes and generalizes the pattern in the prompt. The authors term this **in-context learning**.

There are two headline findings. **First**, without any fine-tuning GPT-3 reaches LAMBADA 86.4% (+18 over prior SOTA), TriviaQA 71.2% few-shot, PTB perplexity 20.5 (down from 35.8), 100% on 2-digit addition, and beats a fine-tuned BERT-Large on SuperGLUE with only ~5 prompts per task. **Second** — and this is the deeper finding — across 8 model sizes spanning 125M to 175B parameters (3 orders of magnitude), **the gap between few-shot and zero-shot grows with model scale**. Larger models do not merely perform tasks better; they become measurably better **at in-context learning itself**. This quantitatively supports the framing (Figure 1.1) that pretraining acts as an implicit "learning-to-learn" outer loop.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1, pp 3–6) / 서론

**한국어**
저자들은 NLP의 지난 5년을 4단계로 정리합니다 — (1) word vectors (word2vec, GloVe), (2) contextual RNN representations (ELMo), (3) pretrained transformer fine-tuning (BERT, GPT-1), (4) 그리고 이제 task-agnostic in-context learning. 핵심 문제 제기: pretrain+fine-tune 패러다임은 **여전히 task별로 수천∼수만 개의 라벨 데이터가 필요**하다. 이는 (a) 실용적으로 비싸고, (b) 좁은 fine-tuning distribution은 spurious correlation을 학습하여 OOD generalization이 나쁘며, (c) 인간은 새 작업을 짧은 자연어 지시("이 문장이 슬픈지 말해줘")나 1–2개의 예시만으로 학습한다는 점에서 인간과 다릅니다.

해결의 단서는 **meta-learning** + **scale**. GPT-2(2019)가 "learning to learn"을 제안했지만 NaturalQuestions 4%, CoQA 55 F1로 SOTA 대비 35점 이상 뒤처졌습니다. 이를 단순히 "능력 부족"으로 보고, **Kaplan et al. (2020)의 scaling laws**를 신뢰하여 **두 자릿수 더 키운 모델**을 학습하면 어떻게 될지 검증합니다.

Figure 1.1은 패러다임을 한 그림으로 요약합니다 — outer loop는 SGD를 통한 unsupervised pretraining, inner loop는 단일 sequence 안에 반복되는 sub-task 패턴(예: $5+8=13, 7+2=9, \ldots$)을 학습하는 forward pass입니다. Figure 1.2는 핵심 발견을 한 장으로 보여줍니다 — symbol removal task에서 175B 모델은 in-context examples 수($K$)가 늘어날수록 가파르게 정확도가 오르지만, 13B와 1.3B 모델은 거의 평평합니다. 즉 **ICL의 기울기 자체가 scale에 dependent**.

**English**
The authors frame the last five years of NLP as four eras — (1) word vectors (word2vec, GloVe), (2) contextual RNN representations (ELMo), (3) pretrained transformer fine-tuning (BERT, GPT-1), and (4) task-agnostic in-context learning, the regime they push into. The core complaint is that pretrain+fine-tune **still requires thousands to tens of thousands of labeled examples per task**: (a) practically expensive, (b) narrow fine-tuning distributions induce spurious correlations and poor OOD generalization, and (c) humans learn new tasks from a short instruction or 1–2 demonstrations.

The proposed remedy combines **meta-learning** with **scale**. GPT-2 (2019) hinted at "language models as multitask learners" but trailed SOTA by 35+ points on CoQA. The authors take this as a capacity problem and trust **Kaplan et al.'s scaling laws** to predict that 100× more parameters will close the gap.

Figure 1.1 captures the framing in a single image: the outer loop is unsupervised pretraining via SGD, the inner loop is a forward pass that recognizes a repeated sub-task pattern (e.g., $5+8=13, 7+2=9, \ldots$) inside one sequence. Figure 1.2 shows the headline finding — on a symbol-removal task, the 175B model's accuracy grows steeply with the number of in-context demonstrations $K$, while the 13B and 1.3B curves are nearly flat. **The slope of in-context learning itself depends on scale.**

### Part II: Approach (§2, pp 6–10) / 접근 방법

#### 2.1 Model and Architecture / 모델과 아키텍처

**한국어**
GPT-3는 GPT-2와 본질적으로 동일한 아키텍처입니다 — modified initialization, pre-normalization (LayerNorm before sublayer), reversible BPE tokenization. **단 한 가지 변경점**은 transformer 레이어가 **alternating dense and locally banded sparse attention** 패턴을 사용한다는 점 (Sparse Transformer, Child et al. 2019). 즉 짝수 layer는 표준 dense attention, 홀수 layer는 banded sparse attention을 사용하여 $O(n^2)$ 비용을 일부 완화합니다.

Table 2.1이 8개 모델을 정리합니다:

| Model | $n_\text{params}$ | $n_\text{layers}$ | $d_\text{model}$ | $n_\text{heads}$ | $d_\text{head}$ | Batch | LR |
|---|---|---|---|---|---|---|---|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | $6 \times 10^{-4}$ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | $3 \times 10^{-4}$ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | $2.5 \times 10^{-4}$ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | $2 \times 10^{-4}$ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | $1.6 \times 10^{-4}$ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | $1.2 \times 10^{-4}$ |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 128 | 2M | $1 \times 10^{-4}$ |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **128** | **3.2M** | **$0.6 \times 10^{-4}$** |

모든 모델은 context window $n_\text{ctx}=2048$, feed-forward $d_\text{ff}=4 d_\text{model}$, **300B token**으로 학습합니다. 175B 모델은 V100 클러스터에서 model parallelism (depth + width 양방향)으로 학습.

**English**
Architecturally GPT-3 is essentially GPT-2: modified initialization, pre-LN, reversible BPE. **The only change** is that transformer layers use **alternating dense and locally banded sparse attention** patterns (Sparse Transformer, Child et al. 2019) — even layers are dense, odd layers are banded sparse, partially mitigating $O(n^2)$ cost.

Table 2.1 catalogs the 8 models. The largest, GPT-3 175B, has 96 layers, $d_\text{model}=12288$, 96 heads, $d_\text{head}=128$, batch size 3.2M tokens, learning rate $0.6 \times 10^{-4}$. All models share $n_\text{ctx}=2048$, $d_\text{ff}=4 d_\text{model}$, and are trained for **300B tokens**. The 175B model uses model parallelism along both depth and width on a V100 cluster provided by Microsoft.

#### 2.2 Training Dataset / 학습 데이터셋

**한국어**
사전학습 코퍼스는 **5개 데이터셋의 가중 혼합**입니다 (Table 2.2):

| Dataset | Tokens | Weight | Epochs (at 300B) |
|---|---|---|---|
| Common Crawl (filtered) | 410B | 60% | 0.44 |
| WebText2 (Reddit links) | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia (English) | 3B | 3% | 3.4 |
| **Total** | **499B** | 100% | — |

Common Crawl은 41 shard (2016–2019), 45TB plaintext에서 시작하여 (1) WebText, Books1/2, Wikipedia 같은 high-quality reference에 대한 유사도 분류기로 필터링하고, (2) document-level fuzzy deduplication을 적용해 570GB / ~410B BPE token으로 줄였습니다. **품질이 높은 데이터(Wikipedia, Books1)는 epoch 수를 더 많이**, 큰 저품질 데이터(Common Crawl, Books2)는 epoch 1 미만으로 가중치를 둡니다.

데이터 contamination 문제(test set이 학습 데이터에 우연히 포함됨)를 인정하며 §4에서 정량 분석합니다.

**English**
The pretraining corpus is a **weighted mixture of 5 datasets** (Table 2.2). Common Crawl was processed by (1) a high-quality-reference similarity classifier, and (2) document-level fuzzy deduplication, reducing 45TB → 570GB ≈ 410B BPE tokens. **Higher-quality datasets are sampled more times per epoch**: Wikipedia 3.4× and Books1 1.9×, while Common Crawl gets only 0.44 epochs over the 300B-token training run. The authors candidly note bug in deduplication left some test-set overlap, and analyze its impact in §4.

#### 2.3 Training Process / 학습 과정

**한국어**
모든 모델 300B token, gradient noise scale (McCandlish 2018)로 batch size 결정, larger model → larger batch + smaller LR. Half-precision training, ZeRO-style memory optimization (Dhariwal). 175B 모델은 약 **3640 PetaFLOP/s-days**의 컴퓨트를 소비 (Figure 2.2) — GPT-2의 1000배.

**English**
All models train on 300B tokens. Batch size is set via gradient-noise-scale measurements (McCandlish 2018); larger models use larger batches and smaller learning rates. Mixed-precision plus ZeRO-style memory sharding. Total compute for 175B ≈ **3,640 PetaFLOP/s-days**, roughly 1000× GPT-2.

#### 2.4 Evaluation / 평가

**한국어**
- **Few-shot**: training set에서 $K \in [10, 100]$개 example을 1–2 newline으로 구분하여 prompt 구성. 일부 task는 description set이 없어 dev set에서 conditioning을 가져옴.
- **Multiple choice**: 각 옵션의 LM likelihood 비교. ARC/OpenBookQA/RACE는 unconditional probability로 정규화: $P(c|\text{ctx}) / P(c|\text{"Answer: "})$.
- **Binary classification**: 0/1 대신 "True"/"False"로 framing.
- **Free-form**: beam search (width 4, length penalty $\alpha=0.6$). F1, BLEU, exact match로 채점.
- 175B는 너무 커서 SuperGLUE/TriviaQA/PiQA 외에는 dev set 결과를 보고.

**English**
- **Few-shot conditioning**: $K \in [10, 100]$ training examples concatenated with 1–2 newlines as separator.
- **Multiple choice**: compare per-token likelihood of each option; for ARC/OpenBookQA/RACE, normalize by unconditional probability $P(c|\text{"Answer: "})$.
- **Binary classification**: relabel as "True"/"False" rather than 0/1.
- **Free-form**: beam search, beam width 4, length penalty 0.6. Scored with F1, BLEU, or exact match.
- Test-server submissions only on a handful of tasks; for most, dev-set numbers are reported.

### Part III: Results (§3, pp 10–28) / 결과

#### 3.1 Language Modeling, Cloze, Completion

**한국어**
- **PTB**: zero-shot perplexity **20.5** (이전 SOTA 35.8, -15.3 절대치, 새 SOTA).
- **LAMBADA** (long-range word prediction): zero-shot **76.2%**, one-shot 72.5%, **few-shot 86.4%** (이전 SOTA 68.0%). Few-shot에서는 fill-in-the-blank cloze 형식 사용:
  > "Alice was friends with Bob. Alice went to visit her friend ___. → Bob"
- **HellaSwag** (adversarial completion): few-shot 79.3% (fine-tuned 1.5B 베이스라인 75.4 능가, 그러나 ALUM SOTA 85.6에는 못 미침).
- **StoryCloze**: zero-shot 83.2%, few-shot ($K=70$) 87.7%.

**English**
- **PTB** zero-shot perplexity **20.5** (prior 35.8) — new SOTA.
- **LAMBADA**: zero-shot **76.2%** / few-shot **86.4%** (+18 over prior). Few-shot uses fill-in-the-blank cloze format which lets the model infer "exactly one word" is the answer.
- **HellaSwag**: 79.3% few-shot, beats a fine-tuned 1.5B baseline (75.4) but under ALUM (85.6).
- **StoryCloze**: 83.2% zero / 87.7% few ($K=70$).

#### 3.2 Closed-Book Question Answering

**한국어**
검색·외부 텍스트 없이 모델 파라미터만으로 답하는 QA. Table 3.3:

| Setting | NaturalQS | WebQS | TriviaQA |
|---|---|---|---|
| RAG (fine-tuned, open-book) | 44.5 | 45.5 | 68.0 |
| T5-11B+SSM (closed-book) | 36.6 | 44.7 | 60.5 |
| GPT-3 Zero | 14.6 | 14.4 | 64.3 |
| GPT-3 One | 23.0 | 25.3 | 68.0 |
| GPT-3 Few | 29.9 | 41.5 | **71.2** |

TriviaQA에서 zero-shot만으로 fine-tuned T5-11B+SSM(60.5)을 14.2점 능가, few-shot 71.2는 retrieval 없이 RAG도 능가하는 SOTA.

**English**
Closed-book QA is harder than open-book — no retrieval allowed. On TriviaQA, GPT-3 zero-shot already beats fine-tuned T5-11B+SSM by 14.2 points; few-shot reaches 71.2, beating even the open-domain retrieval-augmented RAG (68.0). On Natural Questions, GPT-3 underperforms (29.9 vs 36.6) — the authors hypothesize NQ's fine-grained Wikipedia trivia is OOD for GPT-3's broad pretraining.

#### 3.3 Translation

**한국어**
Table 3.4 — WMT BLEU. GPT-3 학습 데이터는 93% 영어, 7% 다언어. unsupervised NMT와 비교:

| | En→Fr | Fr→En | En→De | De→En | En→Ro | Ro→En |
|---|---|---|---|---|---|---|
| Supervised SOTA | **45.6** | 35.0 | **41.2** | 40.2 | **38.5** | **39.9** |
| Few-shot GPT-3 | 32.6 | **39.2** | 29.7 | **40.6** | 21.0 | **39.5** |

**English로 번역할 때** 강함 (Fr→En 39.2, De→En 40.6, Ro→En 39.5는 supervised SOTA 근접/능가). **English에서 다른 언어로** 갈 때 약함. 이는 GPT-3가 본질적으로 English LM임을 반영. En→Ro 21.0은 BPE가 English 위주로 만들어진 영향.

**English**
On WMT, **translating into English** is strong (Fr→En 39.2, De→En 40.6, Ro→En 39.5), beating prior unsupervised NMT by ~5 BLEU and approaching supervised SOTA. **Translating out of English** is weak — En→Ro is only 21.0 BLEU, partly because the BPE tokenizer was built for English. GPT-3 is fundamentally an English LM.

#### 3.4 Winograd-Style / 3.5 Common Sense

**한국어**
- **WSC273 Winograd**: zero/one/few = 88.3/89.7/88.6 (fine-tuned SOTA 90.1과 1–2점 차).
- **Winogrande XL** (adversarial): 70.2/73.2/77.7 (fine-tuned RoBERTa 79.0, SOTA 84.6).
- **PIQA**: 81.0/80.5/82.8 — fine-tuned SOTA 79.4를 모든 setting에서 능가 (단, 일부 contamination 의심으로 별표).
- **ARC Challenge**: 51.4/53.2/51.5 (fine-tuned UnifiedQA 78.5에 한참 못 미침).
- **OpenBookQA**: 57.6/58.8/65.4.

**English**
On Winograd (88.6 few) GPT-3 is within 1–2 points of fine-tuned SOTA. On adversarial Winogrande (77.7 few) it beats fine-tuned RoBERTa-Large but trails T5 SOTA. PIQA few-shot 82.8 is a new SOTA but contamination is suspected. ARC and OpenBookQA show large gaps (~25 points) to fine-tuned UnifiedQA, indicating commonsense scientific knowledge is not yet absorbed.

#### 3.6 Reading Comprehension

**한국어**
| | CoQA | DROP | QuAC | SQuAD2 | RACE-h | RACE-m |
|---|---|---|---|---|---|---|
| Fine-tuned SOTA | **90.7** | **89.1** | **74.4** | **93.0** | **90.0** | **93.1** |
| GPT-3 Few-shot | 85.0 | 36.5 | 44.3 | 69.8 | 46.8 | 58.1 |

**CoQA에서는 인간 베이스라인 3점 이내**(85.0 vs 88-90), 그러나 DROP/QuAC/RACE에서는 큰 격차. 자유형식 dialog (CoQA)는 강하고, structured/discrete reasoning (DROP)이나 multiple-choice exam (RACE)은 약함.

**English**
GPT-3 is within 3 points of human on free-form CoQA (85.0) but lags badly on DROP (numerical reasoning), QuAC (structured dialog), and RACE (school exam multiple-choice). The pattern: free-form completion strong; tasks needing fine-grained discrete reasoning or long-range comparison weak.

#### 3.7 SuperGLUE

**한국어**
Table 3.8 — 32 examples per task in few-shot context.

| | SuperGLUE Avg | BoolQ | CB Acc | COPA | RTE | WSC | ReCoRD F1 |
|---|---|---|---|---|---|---|---|
| Fine-tuned SOTA | **89.0** | **91.0** | **96.9** | **94.8** | **92.5** | **93.8** | **93.3** |
| Fine-tuned BERT-Large | 69.0 | 77.4 | 83.6 | 70.6 | 71.7 | 64.6 | 72.0 |
| GPT-3 Few-shot | 71.8 | 76.4 | 75.6 | **92.0** | 69.0 | 80.1 | 91.1 |

**Few-shot GPT-3 (32 examples)가 fine-tuned BERT-Large (125K examples)를 평균에서 능가**. COPA 92.0과 ReCoRD 91.1은 SOTA에 근접. WiC만 49.4%(거의 random) — 단어 의미 비교 형태의 태스크에서 약함. Figure 3.8 핵심: $K$ examples per task만 8개여도 BERT-Large를 추월.

**English**
GPT-3 few-shot (32 examples per task, 256 total) beats fine-tuned BERT-Large (~125K examples) on the SuperGLUE average. COPA 92.0 and ReCoRD F1 91.1 are near-SOTA. The notable failure is WiC at 49.4% — chance — suggesting GPT-3 struggles with "are these two uses of the same word the same sense?" comparison tasks.

#### 3.8 NLI (Natural Language Inference)

**한국어**
- RTE (in SuperGLUE): few-shot 69.0 (fine-tuned BERT 71.7 근처).
- **ANLI** (adversarial 3 rounds): 모든 GPT-3보다 작은 모델이 ~33% (random); 175B만 R3에서 40%로 random에서 SOTA 절반 거리만큼 진전. NLI는 LM에 여전히 어려운 작업.

**English**
On RTE GPT-3 ~ fine-tuned BERT. **ANLI Round 3**: every model smaller than 175B is at random chance; only 175B reaches 40%, half the gap to SOTA. NLI remains hard.

#### 3.9 Synthetic and Qualitative Tasks / 합성 작업

**한국어 (가장 흥미로운 섹션)**

**3.9.1 Arithmetic**: Table 3.9 — 자연어 ("Q: What is 48 plus 76? A:"):

| Setting | 2D+ | 2D- | 3D+ | 3D- | 4D+ | 4D- | 5D+ | 5D- | 2Dx | 1DC |
|---|---|---|---|---|---|---|---|---|---|---|
| Zero | 76.9 | 58.0 | 34.2 | 48.3 | 4.0 | 7.5 | 0.7 | 0.8 | 19.8 | 9.8 |
| One | 99.6 | 86.4 | 65.5 | 78.7 | 14.0 | 14.0 | 3.5 | 3.8 | 27.4 | 14.3 |
| Few | **100.0** | 98.9 | 80.4 | 94.2 | 25.5 | 26.8 | 9.3 | 9.9 | 29.2 | 21.3 |

**Few-shot 175B는 2-digit 덧셈 100%, 3-digit 덧셈 80.4%**. Memorization 가능성을 spot-check — 학습 데이터에서 "$<NUM_1> + <NUM_2> = $" 검색 결과 단 17/2000 (0.8%) match. 즉 진짜 계산 중. Figure 3.10이 보여주는 강력한 메시지: **13B에서 175B 사이에 큰 phase transition**(2-digit ~10% → 100%).

**3.9.2 Word Scrambling** (CL=cycle, A1/A2=anagrams, RI=random insertion, RW=reversed): few-shot CL 37.9, A2 39.7, RI 67.2, RW 0.44 (역순은 거의 못함).

**3.9.3 SAT Analogies**: 374개 college entrance 문제에서 few-shot 65.2% (대학 지원자 평균 57%, random 20%).

**3.9.4 News Article Generation**: 80명에게 "이 200-word article은 사람이 썼나, 모델이 썼나?" 식별. 175B 모델 식별 정확도 **52% (chance에 근접)** — Table 3.11. Figure 3.13이 보여주듯, **모델 크기가 커질수록 인간이 model-generated text를 식별하기 더 어려워짐** (멱법칙 곡선).

**3.9.5 Learning and Using Novel Words**: "Gigamuru는 일본 악기다. Gigamuru를 사용한 문장 예시는: → I have a Gigamuru that my uncle gave me as a gift." 단 1–5개 예시로 새로운 단어를 문장에서 사용 (Figure 3.16).

**English (the most striking section)**

**3.9.1 Arithmetic**: GPT-3 175B few-shot achieves **100% on 2-digit addition, 98.9% 2-digit subtraction, 80.4% 3-digit addition, 94.2% 3-digit subtraction**, 29.2% 2-digit multiplication, 21.3% on 1-digit composite (e.g. "6+(4*8)=38"). Critically, only 17/2000 problems were exact matches in training data (0.8%) — GPT-3 is genuinely computing, not memorizing. Figure 3.10 shows a striking phase transition between 13B and 175B.

**3.9.2 Word Scrambling**: Few-shot 175B unscrambles random insertions 67.2%, anagrams 39.7%, cycle-letters 37.9%, but reverses words at 0.44% (essentially failing).

**3.9.3 SAT Analogies**: 65.2% few-shot vs 57% college applicant average and 20% random.

**3.9.4 News Article Generation**: Humans correctly identify 200-word GPT-3 175B articles as machine-generated only **52% of the time** — barely above 50% chance. Figure 3.13: identification accuracy follows a power-law decay with model size.

**3.9.5 Novel Word Usage**: GPT-3 fluently uses fictional words like "Gigamuru" or "screeg" in plausible new sentences after seeing 1–5 examples (Figure 3.16).

### Part IV: Contamination, Limitations, Broader Impacts (§4–§6, pp 29–39)

**한국어**
**§4 Contamination**: train/test 13-gram overlap 분석. Wikipedia LM 4종, CBT는 거의 완전 contamination → 보고에서 제외. LAMBADA, PIQA, Winograd는 일부 contamination이지만 clean subset 대비 0.5% 미만 차이로 영향 미미. PTB는 인터넷 이전 데이터라 깨끗함.

**§5 Limitations**:
1. **Text synthesis 약점**: 긴 글에서 의미적 반복, 일관성 손실, non-sequitur.
2. **WiC, ANLI 등 비교형 task 약함** — 두 sentence/word 비교가 약함.
3. **Bidirectional 부재**: GPT-3는 left-to-right autoregressive. fill-in-the-blank류 task는 본질적으로 불리.
4. **Pretraining objective의 한계**: 모든 token 동일 가중. 무엇이 중요한지 학습 못함.
5. **Sample efficiency 낮음**: 수백 B token 학습은 인간이 평생 보는 텍스트보다 많음.
6. **Few-shot이 "학습"인지 "인식"인지 불명**: 학습 분포에서 본 task인지 새로 학습한 것인지 구분 불가.
7. **Inference cost**: 175B 모델은 상용 배포에 비현실적. Distillation 미답.
8. **Interpretability, calibration, bias 부족**.

**§6 Broader Impacts**:
- **§6.1 Misuse**: misinformation, phishing, spam의 자동화. 현재까지 advanced threat actor의 LLM 채택 미관찰.
- **§6.2 Bias**: gender — 388개 직업 중 83%가 male leaning ("The {detective} was a"). 직업 편향 평균 -1.11 (neutral), -2.14 (competent), -1.15 (incompetent). race — 'Asian' sentiment 높음, 'Black' 낮음. religion — 'violent', 'terrorism', 'terrorist'가 Islam과 가장 자주 동시 출현.
- **§6.3 Energy**: 175B 학습은 GPU-PetaFLOP/s-days 수천대. 그러나 amortized 추론 비용은 페이지당 ~$0.4kWhr.

**English**
**§4 Contamination**: Most benchmarks have negligible contamination effect (<0.5% difference between full and clean subsets). Only Wikipedia LM tasks and CBT were nearly fully contaminated (excluded from reporting). PTB, predating the modern internet, was clean.

**§5 Limitations**: weak text synthesis at length; weak on comparison tasks (WiC, ANLI); no bidirectionality (handicaps fill-in-the-blank, comprehension); pretraining objective weights all tokens equally; sample-inefficient (orders of magnitude more text than a human ever sees); ambiguity about whether few-shot is "learning" or "recognition"; expensive inference; poor interpretability; calibration and bias issues.

**§6 Broader Impacts**: Misuse (phishing, misinformation) is plausible but not yet observed in advanced threat actors. Gender bias: 83% of 388 occupations male-leaning. Race: 'Asian' high sentiment, 'Black' low. Religion: 'violence', 'terrorism' co-occur most with Islam. Training cost: thousands of PetaFLOP/s-days; amortized inference is cheap (~$0.4 kWh per 100 pages).

### Part V: Conclusion (§8) / 결론

**한국어**
저자들은 175B 매개변수 LM이 zero/one/few-shot에서 강한 성능을 보이며, 일부에서는 fine-tuned SOTA에 근접/능가함을 다시 강조. fine-tuning 없이 매끄러운 scaling이 관측됨. 한계와 사회적 영향을 인정. 결론: **거대 언어 모델이 적응적이고 일반적인 언어 시스템을 만드는 데 중요한 ingredient일 수 있다**.

**English**
The conclusion reiterates: 175B parameters yield strong zero/one/few-shot performance, sometimes matching or exceeding fine-tuned SOTA, with smooth scaling and no fine-tuning. Despite real limitations and social risks, **very large LMs may be a key ingredient in adaptable, general-purpose language systems**.

---

## 3. Key Takeaways / 핵심 시사점

1. **In-context learning is a real capability that emerges with scale / In-context learning은 scale과 함께 출현하는 실제 능력**
   — 한국어: 125M부터 175B까지 8개 모델을 같은 데이터로 학습한 결과, few-shot이 zero-shot 대비 더 빨리 개선됨 (Figure 1.3). 즉 큰 모델은 단순히 더 잘하는 것이 아니라 prompt에서 학습하는 능력 자체가 더 좋다.
   — English: Across 8 models on identical data, few-shot performance improves *faster* than zero-shot as parameters grow. Large models are not just better; they are measurably better at the meta-task of learning from a prompt.

2. **No fine-tuning, no gradient updates / Fine-tuning도 gradient도 없음**
   — 한국어: 동일한 175B 가중치 하나로 번역, QA, 산술, 패턴 인식, 작문 등 수십 작업 수행. 이는 inference architecture의 패러다임 전환 — 작업당 모델이 아니라 작업당 prompt.
   — English: A single frozen 175B weight set handles translation, QA, arithmetic, pattern recognition, writing — a paradigm shift from "one model per task" to "one prompt per task."

3. **Scaling laws extrapolate two more orders of magnitude / Scaling laws가 두 자릿수 더 외삽됨**
   — 한국어: $L = 2.57 C^{-0.048}$ (compute에 대한 멱법칙)이 GPT-2 이후 두 orders of magnitude 더 매끄럽게 이어짐 (Figure 3.1). 이는 추가 scaling이 예측 가능한 이득을 줄 것임을 시사.
   — English: The Kaplan-2020 power law $L = 2.57 C^{-0.048}$ continues smoothly two more decades of compute (Figure 3.1), suggesting predictable gains from further scaling — a thesis that drove the next 3+ years of LLM research.

4. **Phase transitions in arithmetic and unscrambling / 산술과 unscrambling에서의 phase transition**
   — 한국어: 13B → 175B 사이 2-digit 덧셈이 ~10%에서 100%로 급격히 변화 (Figure 3.10). 작은 모델에는 거의 없던 능력이 큰 모델에서 갑자기 등장 — 이후 "emergent abilities" 논쟁의 출발점.
   — English: 2-digit addition jumps from ~10% to 100% between 13B and 175B (Figure 3.10). Capabilities essentially absent in smaller models suddenly appear at scale — the seed of the later "emergent abilities" debate.

5. **GPT-3 generates news indistinguishable from humans / GPT-3는 인간 식별 불가능한 뉴스 생성**
   — 한국어: 80명이 200-word 기사를 식별한 정확도 **52%** (random 50%). Figure 3.13의 멱법칙 곡선은 model size 증가에 따라 인간의 식별 능력이 매끄럽게 감소함을 보임. 이는 §6 Broader Impacts (misinformation 위험)의 핵심 근거.
   — English: Humans identify GPT-3 175B 200-word articles only 52% of the time (50% chance). Figure 3.13's power-law decay shows human detection ability falling smoothly with model size — the empirical foundation for misinformation concerns.

6. **Bidirectional architectures may be missing / Bidirectional 구조의 부재**
   — 한국어: WiC 49.4% (random), ANLI ~chance, RACE 약함 — autoregressive (left-to-right) 가정이 비교형/회고형 task에서 한계. BERT-style이나 T5-style bidirectional 사전학습이 더 나을 수 있음을 시인.
   — English: WiC at chance (49.4%), ANLI near chance, weak RACE. Left-to-right autoregression handicaps comparison and look-back tasks. The authors explicitly conjecture bidirectional pretraining at GPT-3 scale would be stronger.

7. **Training data quality matters more than quantity / 데이터 품질이 양보다 중요**
   — 한국어: Common Crawl 410B token이 weight 60%인데 epoch 0.44만 학습. Wikipedia 3B token은 epoch 3.4번 학습. 데이터 mix 가중치는 약간의 overfitting을 감수하는 대신 품질을 높이는 trade-off.
   — English: Common Crawl (410B tokens) gets 60% weight but only 0.44 epochs; Wikipedia (3B tokens) gets 3.4 epochs. The mix accepts mild overfitting for quality. This insight foreshadowed Chinchilla (2022) which found GPT-3 was data-undertrained.

8. **Data contamination is a real but small effect / 데이터 contamination은 실재하지만 영향 미미**
   — 한국어: §4의 13-gram overlap 분석은 대부분 벤치마크에서 0.5% 미만 차이. 단 몇 데이터셋(Wikipedia LM, CBT)은 거의 전체 contamination이라 보고에서 제외. 인터넷 스케일 사전학습의 새로운 평가 방법론 문제 제기.
   — English: A 13-gram overlap analysis showed <0.5% difference between full and clean subsets for most benchmarks; a few datasets (Wikipedia LM, CBT) were near-totally contaminated and excluded. This raised a methodological alarm for the entire field about benchmarking on internet-scale pretrained models.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Pretraining Objective / 사전학습 목적함수

$$\mathcal{L}_\text{pre}(\theta) = - \mathbb{E}_{x \sim \mathcal{D}}\left[ \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \right]$$

- $\theta$: 1.75 × 10¹¹ parameters in the 175B model.
- $\mathcal{D}$: weighted mixture (CC 60%, WT2 22%, B1 8%, B2 8%, Wiki 3%).
- $T$: training context length (2048 tokens).
- Total tokens trained: 300B.

**한국어**: 표준 좌→우 next-token prediction. 모든 fine-tuning, RL, instruction tuning 없이 이 한 가지 손실만 최적화.
**English**: Standard left-to-right next-token prediction — no fine-tuning, no RL, no instruction tuning. Just this one objective on 300B tokens.

### 4.2 In-Context Learning Formalism / In-Context Learning 정식화

Given a task description $T$ and $K$ demonstrations $\{(x_i, y_i)\}_{i=1}^{K}$, the few-shot prediction for a test input $x_*$ is:

$$\hat{y}_* = \arg\max_y \; p_\theta\!\left(y \;\Big|\; \underbrace{T}_\text{instruction} \,\Vert\, \underbrace{(x_1, y_1) \,\Vert\, \cdots \,\Vert\, (x_K, y_K)}_\text{$K$ demonstrations} \,\Vert\, x_*\right)$$

where $\Vert$ denotes string concatenation (with a separator like "\n\n" or "###").

**Key constraint**: $\theta$ is fixed throughout. There is no inner gradient. The "learning" is implicit pattern induction inside one forward pass.

**한국어**: $\theta$ 고정, 외부 SGD loop 없음. Inner-loop "학습"은 forward pass 안의 attention pattern matching.
**English**: $\theta$ never changes; no outer SGD loop at inference. The "inner loop" is just whatever computation happens inside one forward pass conditioned on the prompt.

### 4.3 Compute Scaling Law (Empirical) / 컴퓨트 스케일링 법칙 (실증)

From Figure 3.1 of the paper, validation loss as a function of compute $C$ (in PetaFLOP/s-days) follows:

$$L(C) \approx 2.57 \cdot C^{-0.048}$$

**한국어**: $C$를 10배 늘리면 $L$이 약 $10^{-0.048} \approx 0.895$배 감소 (10.5% reduction). 175B GPT-3 (3640 PFLOP/s-days)의 예측: $L \approx 2.57 \cdot 3640^{-0.048} \approx 1.94$.
**English**: 10× compute → ~10.5% loss reduction. For GPT-3 175B at 3640 PFLOP/s-days, the formula predicts $L \approx 1.94$.

### 4.4 Worked Example: Few-Shot Translation Prompt Trace / Few-Shot 번역 프롬프트 추적

다음은 Figure 2.1에서 가져온 Fr 번역 prompt의 token-level trace입니다 / Token-level trace of the French-translation prompt from Figure 2.1:

```
INPUT (concatenated, ~30 tokens):
┌───────────────────────────────────────────────────────────┐
│ Translate English to French:                              │  ← T (instruction)
│                                                            │
│ sea otter => loutre de mer                                 │  ← (x_1, y_1)
│ peppermint => menthe poivrée                               │  ← (x_2, y_2)
│ plush giraffe => girafe peluche                            │  ← (x_3, y_3)
│ cheese =>                                                  │  ← x_* (test prompt)
└───────────────────────────────────────────────────────────┘

FORWARD PASS (frozen θ_175B):
  1. Tokenize → [Translate, _English, _to, _French, :, \n, \n,
                 sea, _otter, _=>, _l, outre, _de, _mer, \n,
                 ...
                 cheese, _=>]
  2. Embed → 12288-dim vectors per token
  3. 96 transformer layers (alternating dense / banded sparse attention)
  4. Final LayerNorm → unembed → logits over 50257 BPE vocab
  5. Sample next token → "_from" ❌  OR  "_fromage" ✓

NO BACKWARD PASS. No gradient. No weight update.
```

**관찰 / Observations**:
- Demonstration 패턴은 "English_word => French_word" 형식을 establish하여 model이 'cheese =>' 다음에 French token이 와야 함을 induce.
- Pattern induction은 attention head가 Q-K dot product를 통해 in-context "lookup"을 수행한다고 해석 가능 (induction heads, Olsson et al. 2022).
- 이 경우 정답: "_fromage" (BPE 1-token).

**한국어**: 위 trace에서 알 수 있듯, GPT-3에서 "few-shot learning"은 실제로는 **conditional generation의 prompt engineering**이며, gradient descent의 단 한 번도 일어나지 않습니다.
**English**: As the trace shows, "few-shot learning" in GPT-3 is really **prompt-engineered conditional generation**; there is literally not one step of gradient descent at test time.

### 4.5 Architecture Hyperparameters Recap / 아키텍처 하이퍼파라미터 정리

- Per-layer compute: $\approx 12 \cdot d_\text{model}^2 \cdot n_\text{ctx}$ FLOPs/token (attention + FFN, ignoring sparse savings).
- Total parameters: $n_\text{params} \approx 12 \cdot n_\text{layers} \cdot d_\text{model}^2$.
- For 175B: $12 \cdot 96 \cdot 12288^2 \approx 1.74 \times 10^{11}$ ✓.
- Forward FLOPs per token: $2 \cdot n_\text{params} \approx 3.5 \times 10^{11}$ FLOPs ≈ 350 GFLOPs.
- Training FLOPs: $6 \cdot n_\text{params} \cdot 300 \times 10^9 \approx 3.14 \times 10^{23}$ FLOPs ≈ 3640 PFLOP-days.

**한국어**: 위 공식들로 175B 모델 파라미터 수와 학습 컴퓨트가 일치하는지 검증 가능. Kaplan scaling laws의 stable 법칙들.
**English**: These formulas verify the 175B parameter count and 3640 PFLOP/s-day training cost — Kaplan's scaling-laws bookkeeping.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
Pre-2017 ──────────────────────────────────────────────────────────
2013 ─ word2vec (Mikolov)                — distributed word vectors
2014 ─ Seq2Seq (Sutskever)                — neural translation
2015 ─ Attention (Bahdanau)               — content-based addressing

The Transformer Era ───────────────────────────────────────────────
2017 ─ Transformer (Vaswani)              — Attention Is All You Need
2018 ─ GPT-1 (Radford) 117M               — pretrain + fine-tune
2018 ─ BERT (Devlin) 340M                 — bidirectional masked LM
2019 ─ GPT-2 (Radford) 1.5B               — zero-shot LM, "multitask learners"
2019 ─ T5 (Raffel) 11B                    — text-to-text fine-tune
2019 ─ Sparse Transformer (Child)         — banded + strided attention
2020 ─ Scaling Laws (Kaplan #39)          — L(N) ∝ N^-α, predicts gains

★ 2020 ─ GPT-3 (Brown, this paper) 175B   — in-context learning + scaling
★         → "language models are few-shot learners"

The LLM Era (post-GPT-3) ──────────────────────────────────────────
2021 ─ "Foundation Models" (Bommasani)    — terminology coined for GPT-3 era
2021 ─ Codex / GitHub Copilot             — GPT-3 fine-tuned on code
2022 ─ Chinchilla (Hoffmann)              — GPT-3 was data-undertrained
2022 ─ PaLM (Chowdhery) 540B              — 3× GPT-3, beats it
2022 ─ Chain-of-Thought (Wei)             — prompting beyond i.i.d. examples
2022 ─ InstructGPT (Ouyang) / RLHF        — GPT-3 + human feedback
2022 ─ ChatGPT                            — InstructGPT-style productized
2023 ─ GPT-4 / Claude / LLaMA             — multimodal, longer context
2024+ ─ Reasoning models (o1, R1)         — RL on chain-of-thought
```

**한국어**: GPT-3는 transformer 10년 (2017–2027)의 한가운데에 위치. 그 이전까지의 연구는 GPT-3로 수렴하고, 그 이후의 연구는 모두 GPT-3를 출발점으로 삼습니다. "before GPT-3"와 "after GPT-3"는 NLP에서 진정한 분기점입니다.

**English**: GPT-3 sits at the midpoint of the transformer decade (2017–2027). Everything before converges into GPT-3; everything after takes GPT-3 as the starting point. "Before GPT-3" and "after GPT-3" is a genuine dividing line in NLP.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Vaswani et al. 2017, *Transformer* | Architectural backbone — exact same Transformer block, just 96 layers and $d_\text{model}=12288$ / 동일한 Transformer 블록을 96 layer로 깊게 쌓음 | 핵심 / Foundational |
| Radford et al. 2018, *GPT-1* | Same recipe (causal LM fine-tune), 117M parameters / 동일한 causal LM 패러다임의 117M 버전 | 직계 선조 / Direct ancestor |
| Devlin et al. 2018, *BERT* | Bidirectional alternative; GPT-3 §5 explicitly admits BERT-style pretraining might be stronger / 양방향 대안. GPT-3 §5에서 BERT 스타일이 더 강할 수 있다고 인정 | 패러다임 라이벌 / Paradigm rival |
| Radford et al. 2019, *GPT-2* | Direct predecessor; same architecture, 100× smaller; introduced "LMs as multitask learners" framing / 직계 전임자, 동일 구조 1/100 크기, "다중작업 학습자" 프레이밍 도입 | 직계 선조 / Direct predecessor |
| Child et al. 2019, *Sparse Transformer* | Provides the "alternating dense and locally banded sparse attention" used in GPT-3 layers / GPT-3 레이어의 alternating dense/sparse attention 패턴 출처 | 핵심 컴포넌트 / Key component |
| Raffel et al. 2020, *T5* | Contemporary 11B text-to-text alternative; GPT-3 reframes everything as completion instead / 동시대 11B text-to-text 대안. GPT-3는 모두 completion으로 reframe | 동시대 비교 / Contemporary contrast |
| **Kaplan et al. 2020, *Scaling Laws (#39)*** | **Predictive backbone — gave OpenAI confidence to scale 100×** / **OpenAI가 100× scaling 결정의 이론적 근거** | **이론 토대 / Theoretical foundation** |
| Hoffmann et al. 2022, *Chinchilla* | Showed GPT-3 was data-undertrained: 70B model with 1.4T tokens beats 175B with 300B tokens / GPT-3가 데이터 부족 상태였음을 보임 | 후속 비판 / Successor critique |
| Wei et al. 2022, *Emergent Abilities* | Formalized the "phase transition" patterns first observed in GPT-3 arithmetic (Figure 3.10) / GPT-3의 산술 phase transition을 일반화 | 직접 후속 / Direct follow-up |
| Wei et al. 2022, *Chain-of-Thought* | Built directly on GPT-3 prompt format; showed step-by-step reasoning prompts unlock arithmetic / GPT-3 prompt 형식 위에서 추론 prompting 발전 | 프롬프팅 후속 / Prompting follow-up |
| Ouyang et al. 2022, *InstructGPT* | RLHF fine-tunes GPT-3 to follow instructions; the bridge from GPT-3 to ChatGPT / GPT-3에 RLHF를 적용하여 ChatGPT로 가는 다리 | 산업 후속 / Production follow-up |

---

## 7. References / 참고문헌

- Brown, T. B., et al. "Language Models are Few-Shot Learners". *NeurIPS 2020*. arXiv:2005.14165. https://arxiv.org/abs/2005.14165
- Vaswani, A., et al. "Attention Is All You Need". *NeurIPS 2017*. arXiv:1706.03762.
- Radford, A., et al. "Improving Language Understanding by Generative Pre-Training" (GPT-1). 2018. OpenAI tech report.
- Radford, A., et al. "Language Models are Unsupervised Multitask Learners" (GPT-2). 2019. OpenAI tech report.
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL 2019*. arXiv:1810.04805.
- Child, R., et al. "Generating Long Sequences with Sparse Transformers". 2019. arXiv:1904.10509.
- Raffel, C., et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5). *JMLR 2020*. arXiv:1910.10683.
- Kaplan, J., et al. "Scaling Laws for Neural Language Models". 2020. arXiv:2001.08361. **[Paper #39 in this study sequence]**
- Hoffmann, J., et al. "Training Compute-Optimal Large Language Models" (Chinchilla). *NeurIPS 2022*. arXiv:2203.15556.
- Wei, J., et al. "Emergent Abilities of Large Language Models". *TMLR 2022*. arXiv:2206.07682.
- Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". *NeurIPS 2022*. arXiv:2201.11903.
- Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT). *NeurIPS 2022*. arXiv:2203.02155.
- Bommasani, R., et al. "On the Opportunities and Risks of Foundation Models". 2021. arXiv:2108.07258.
- McCandlish, S., et al. "An Empirical Model of Large-Batch Training". 2018. arXiv:1812.06162.
