---
title: "Language Models are Unsupervised Multitask Learners (GPT-2)"
authors: [Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever]
year: 2019
journal: "OpenAI Technical Report"
doi: "OpenAI tech report (no DOI)"
topic: Artificial_Intelligence
tags: [language-model, transformer, GPT-2, zero-shot, WebText, scaling, BPE, unsupervised, multitask]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 29. Language Models are Unsupervised Multitask Learners / 언어 모델은 비지도 멀티태스크 학습자이다

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문(GPT-2)은 단일한 large-scale autoregressive 언어 모델이, **어떤 task-specific fine-tuning이나 supervised data도 사용하지 않고** 다양한 NLP task를 zero-shot으로 수행할 수 있음을 보입니다. 핵심 주장은 두 가지입니다. **첫째**, 언어 모델링은 본질적으로 multitask learning입니다 — 충분히 큰 자연어 corpus는 번역, 요약, QA, 독해 등의 task를 자연스러운 시연(demonstration)으로 포함하므로, 이 corpus에 대한 maximum likelihood 학습은 암묵적으로 multitask 학습이 됩니다. 형식적으로, 모든 NLP task는 $p(\text{output} \mid \text{input}, \text{task})$의 추정으로 환원됩니다. **둘째**, 모델 용량(117M → 345M → 762M → 1542M 파라미터)이 커질수록 zero-shot 성능이 **log-linear**하게 향상됩니다. 저자들은 새로 만든 **WebText**(Reddit ≥3 karma의 outbound link, 8M 문서, 40GB)에서 학습한 1.5B 파라미터 모델 GPT-2가 8개 LM 벤치마크 중 7개에서 SOTA를 zero-shot으로 달성함을 보입니다(LAMBADA 8.6 PPL, WikiText-2 18.34 PPL, PTB 35.76 PPL, CBT 공통명사 93.3%, Winograd 70.7%, CoQA 55 F1, WMT-14 Fr-En 11.5 BLEU). 가장 큰 모델조차 WebText에 underfit한다는 사실은, 더 큰 모델이 더 잘 할 것이라는 강력한 신호입니다 — 이는 GPT-3/Chinchilla로 이어지는 scaling era의 출발점이 됩니다.

**English**
This paper (GPT-2) demonstrates that a single large-scale autoregressive language model can perform a wide range of NLP tasks **zero-shot — without any task-specific fine-tuning or supervised data**. Two claims anchor the paper. **First**, language modeling is implicitly multitask: a sufficiently large natural-language corpus contains naturally occurring demonstrations of translation, summarization, QA, and reading comprehension, so maximum-likelihood training on the corpus is implicitly multitask. Formally, every NLP task reduces to estimating $p(\text{output} \mid \text{input}, \text{task})$. **Second**, capacity scales: as parameter count grows from 117M to 1.5B, zero-shot performance improves **log-linearly**. The authors curate **WebText** (40GB / 8M documents from Reddit-recommended outbound links with ≥3 karma) and train a 1.5B-parameter Transformer decoder, GPT-2, that achieves state-of-the-art zero-shot on 7 of 8 LM benchmarks (LAMBADA 8.6 PPL, WikiText-2 18.34 PPL, PTB 35.76 PPL, CBT common nouns 93.3%, Winograd 70.7%, CoQA 55 F1, WMT-14 Fr-En 11.5 BLEU). Crucially, the 1.5B model still underfits WebText, signaling that larger models will continue to improve — directly setting the stage for GPT-3 and the scaling-laws era.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 현재 ML 시스템의 **brittleness** 문제를 지적합니다. Recht et al. (2018)은 image classifier가 distribution shift에 매우 민감함을 보였고, Jia & Liang (2017)은 reading comprehension 시스템이 adversarial perturbation에 무너짐을 보였습니다. 저자들의 진단: 현재 시스템은 "competent generalists"가 아니라 "narrow experts"이며, 이는 **single-task training on single-domain data**에 기인합니다.

해결책으로 **multitask learning**(Caruana 1997)이 제안되어 왔지만, NLP에서는 아직 미성숙합니다 — McCann et al. (2018)의 decaNLP는 10개 (dataset, objective) pair, Bowman et al. (2018)은 17개로 학습했지만, meta-learning 관점에서 보면 각 (dataset, objective) pair는 단 하나의 학습 example입니다. 현 ML 시스템이 task에 일반화하려면 수백~수천 개의 example이 필요한데, 이는 (dataset, objective) pair를 그만큼 brute-force로 만드는 것이 비현실적임을 의미합니다.

저자의 대안 경로: **language modeling 만으로 task 학습이 가능한가?** Schwartz et al. (2017)이 commonsense reasoning에, Radford et al. (2017)이 sentiment analysis에 이를 시연한 바 있습니다. 본 논문은 zero-shot setting에서 LM의 multitask 능력을 측정합니다.

**English**
The authors open by diagnosing current ML systems as **brittle** — Recht et al. (2018) showed image classifiers fail under distribution shift, and Jia & Liang (2017) showed reading comprehension systems collapse under adversarial perturbations. The diagnosis: current systems are "narrow experts" rather than "competent generalists," because they are trained on **single tasks with single-domain data**.

Multitask learning (Caruana 1997) is the proposed remedy, but in NLP it remains immature: McCann et al. (2018)'s decaNLP used 10 (dataset, objective) pairs and Bowman et al. (2018) used 17, but from a meta-learning perspective each pair is just one training example. Since current ML systems need hundreds to thousands of examples to generalize, brute-forcing comparable numbers of (dataset, objective) pairs is impractical.

The authors' alternative: can **language modeling alone** learn tasks? Schwartz et al. (2017) demonstrated this for commonsense reasoning; Radford et al. (2017) for sentiment analysis. This paper tests it broadly in a **zero-shot** setting.

### Part II: Approach (§2) / 접근 방법

**한국어**
모든 LM의 기반은 sequence symbol $(s_1, \ldots, s_n)$의 결합 확률을 conditional 곱으로 분해하는 것입니다 (Jelinek & Mercer 1980; Bengio et al. 2003):

$$p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \ldots, s_{i-1}) \tag{1}$$

이 분해는 tractable sampling/density estimation을 가능하게 하며, Transformer (Vaswani et al. 2017) 같은 모델이 conditional 추정에 매우 강력함이 입증됐습니다.

**핵심 통찰**: single task 학습은 conditional $p(\text{output} \mid \text{input})$ 추정이지만, **general system**은 task 자체에도 조건화되어야 합니다:

$$p(\text{output} \mid \text{input}, \text{task})$$

McCann et al. (2018)은 task를 자연어 sequence로 표현할 수 있음을 보였습니다 — 예: "(translate to french, english text, french text)". 이러한 sequence를 single LM이 학습하면, task-conditioning을 자연어로 prompt할 수 있게 됩니다.

**중요한 관찰**: supervised objective는 unsupervised objective와 같지만 sequence의 한 부분(subset)에서만 평가됩니다. 따라서 unsupervised objective의 global minimum은 supervised objective의 global minimum이기도 합니다. "충분히 큰 LM이 자연어로 표현된 task를 자연 텍스트의 reading 과정에서 학습한다"는 가설은, WebText에서 자연스럽게 등장하는 영어↔프랑스어 번역 예시(Table 1) 등으로 뒷받침됩니다.

**English**
All LMs factorize the joint over symbols $(s_1, \ldots, s_n)$ as a product of conditionals (Jelinek & Mercer 1980; Bengio et al. 2003):

$$p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \ldots, s_{i-1}) \tag{1}$$

This permits tractable sampling and density estimation. Transformers (Vaswani 2017) have proven exceptionally strong at the conditional estimation.

**Key insight**: single-task learning estimates $p(\text{output} \mid \text{input})$, but a **general system** must condition on the task too:

$$p(\text{output} \mid \text{input}, \text{task})$$

McCann et al. (2018) showed tasks can be expressed as natural-language sequences — e.g., `(translate to french, english text, french text)`. Training a single LM on such sequences makes task-conditioning emergent through natural-language prompts.

**Critical observation**: the supervised objective is the same as the unsupervised one but evaluated on only a subset of the sequence. So the global minimum of the unsupervised objective is also the global minimum of the supervised one. The hypothesis — that a sufficiently large LM learns naturally specified tasks while reading text — is supported by examples like Table 1, which shows English↔French translation pairs occurring in WebText.

### Part III: Training Dataset — WebText (§2.1) / 학습 데이터셋

**한국어**
저자들은 다양성과 자연스러움을 모두 갖춘 corpus를 원합니다. Common Crawl은 magnitude가 충분하지만 quality issue가 심각합니다 — Trinh & Le (2018)는 "거의 알아볼 수 없는" 문서가 많다고 보고했습니다. 저자들도 같은 문제를 관찰했습니다.

**해결책: WebText**
- **Quality filter via human curation**: Reddit에서 ≥3 karma를 받은 outbound link만 수집. Karma는 "다른 user가 링크를 흥미롭다/교육적이다/재밌다고 생각했다"는 휴리스틱.
- **규모**: 45M 링크의 텍스트 부분집합. de-duplication과 휴리스틱 cleaning 후 **8M+ 문서, ~40GB 텍스트**.
- **시점**: Dec 2017 이전 링크만 사용 (cutoff).
- **Wikipedia 제외**: 다른 데이터셋의 흔한 source여서 train/test overlap 우려. 분석 단계의 confounding 회피.
- **추출**: Dragnet (Peters & Lecocq 2013) + Newspaper 라이브러리로 HTML에서 텍스트 추출.

**핵심 차별점**: 단일 도메인(News, Wiki, fiction)에 의존하지 않고, Reddit user들이 큐레이션한 **다양한 domain의 자연 텍스트**를 모았습니다.

**English**
The authors want a corpus that is both **diverse** and **natural**. Common Crawl has the scale but suffers severe quality issues — Trinh & Le (2018) found many documents "whose content are mostly unintelligible." The authors observed the same.

**Solution: WebText**
- **Quality filter via human curation**: scrape only outbound Reddit links with ≥3 karma — karma serves as a heuristic for "other users found this link interesting, educational, or funny."
- **Scale**: text subset of 45M links. After de-duplication and heuristic cleaning: **8M+ documents, ~40GB of text**.
- **Cutoff**: links from before Dec 2017.
- **Wikipedia removed** — common across datasets, so removing it avoids analysis confounds.
- **Extraction**: Dragnet (Peters & Lecocq 2013) + the Newspaper library for HTML→text.

**Key differentiator**: rather than relying on a single domain (news, Wikipedia, fiction), WebText is **diverse natural text curated by Reddit users**.

### Part IV: Input Representation — Byte-Level BPE (§2.2) / 입력 표현

**한국어**
이상적인 LM은 **임의의 string에 확률을 부여**할 수 있어야 합니다. 기존 LM은 lower-casing, tokenization, OOV token 등의 전처리로 인해 model-able string 공간이 제한됩니다.

**Byte-level 접근의 매력**: UTF-8 바이트 sequence를 처리하면 모든 string을 cover. 그러나 Al-Rfou et al. (2018)에 따르면 byte-level LM은 word-level LM에 비해 large-scale에서 경쟁력 부족.

**Byte Pair Encoding (BPE) (Sennrich et al. 2015)**: character와 word 수준 사이 — 빈번한 symbol sequence는 word-like, 드문 것은 character-like. 그러나 일반 BPE는 Unicode code point 위에서 동작하며 multi-symbol token 추가 전에도 vocabulary가 130,000+에 달합니다(byte-level의 256 vs.).

**저자들의 해결**: byte-level BPE 사용. **단, byte-category 경계를 넘는 merge 차단**. 예: `dog`, `dog!`, `dog?`, `dog.` 같이 다양한 punctuation 결합이 vocabulary 슬롯을 낭비하지 않도록. 단 spaces는 예외 처리해 compression 효율을 크게 향상시키되 단어 분절은 최소화.

**결과**:
- **Vocabulary**: 50,257 (byte-level의 base 256 + BPE merges)
- **모든 Unicode string에 확률 할당 가능** — 전처리/tokenization/vocabulary와 무관하게 모든 데이터셋에서 평가 가능.

**English**
An ideal LM should **assign probability to any string**. Existing LMs restrict the space via lowercasing, tokenization, and OOV tokens.

**Byte-level appeal**: processing UTF-8 byte sequences covers all strings. But Al-Rfou et al. (2018) found byte-level LMs uncompetitive with word-level at large scale.

**Byte Pair Encoding (BPE) (Sennrich 2015)** is a middle ground — frequent symbol sequences become word-like, rare ones character-like. Reference BPE implementations operate on Unicode code points, where vocabularies of 130,000+ arise even before multi-symbol tokens (vs. just 256 base symbols at byte level).

**The authors' fix**: byte-level BPE, **but block merges that cross byte-category boundaries**. E.g., variants `dog.`, `dog!`, `dog?` shouldn't waste vocabulary slots. Spaces are handled as a curated exception to preserve compression efficiency while minimizing word fragmentation.

**Result**:
- **Vocabulary**: 50,257 tokens (256 byte-level base + BPE merges).
- **Probability over any Unicode string** — evaluation on any dataset regardless of preprocessing/tokenization.

### Part V: Model Architecture (§2.3) / 모델 구조

**한국어**
GPT-2는 GPT-1 (Radford 2018)을 base로 한 **decoder-only Transformer**이며, 다음 modification이 있습니다:

| Modification | 설명 |
|---|---|
| **Pre-LayerNorm** | LayerNorm을 각 sub-block의 **input**으로 이동 (pre-activation ResNet 스타일, He et al. 2016). 학습 안정성 향상. |
| **Final LayerNorm** | 마지막 self-attention block 뒤에 LayerNorm 추가. |
| **Residual scaling** | residual layer 가중치를 $1/\sqrt{N}$로 초기화 ($N$ = residual layer 수). residual path 누적 폭주 방지. |
| **Vocabulary expansion** | 50,257 (byte-level BPE) |
| **Context size** | 512 → 1024 tokens |
| **Batch size** | 512 (대형) |

**4가지 모델 크기 / Four model sizes (Table 2)**:

| Parameters | Layers | $d_{\text{model}}$ |
|---|---|---|
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| **1542M (GPT-2)** | 48 | 1600 |

- 117M ≈ original GPT
- 345M ≈ BERT-large (Devlin 2018)
- 1542M = "GPT-2"

학습률은 5% held-out WebText에서 perplexity가 가장 좋도록 manual tuning. 모든 모델이 **WebText에 underfit** — 더 학습하면 perplexity가 더 좋아집니다.

**English**
GPT-2 is a **decoder-only Transformer** based on GPT-1 (Radford 2018), with the following modifications:

| Modification | Description |
|---|---|
| **Pre-LayerNorm** | LN moved to the **input** of each sub-block, à la pre-activation ResNets (He 2016). Improves training stability at depth. |
| **Final LayerNorm** | Extra LN added after the last self-attention block. |
| **Residual scaling** | Residual weights scaled by $1/\sqrt{N}$ at init ($N$ = number of residual layers) to prevent activation buildup along the residual path. |
| **Vocabulary expansion** | 50,257 (byte-level BPE) |
| **Context size** | 512 → 1024 tokens |
| **Batch size** | 512 (large) |

**Four model sizes (Table 2)**:

| Parameters | Layers | $d_{\text{model}}$ |
|---|---|---|
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| **1542M (GPT-2)** | 48 | 1600 |

- 117M ≈ original GPT
- 345M ≈ BERT-large (Devlin 2018)
- 1542M = "GPT-2"

Learning rate manually tuned for best perplexity on a 5% held-out WebText sample. All models still **underfit WebText** — more training would improve perplexity further.

### Part VI: Experiments (§3) / 실험

**한국어**

#### 3.1 Language Modeling (Table 3)
8개 표준 LM dataset에서 zero-shot 평가. invertible de-tokenizer로 PTB-style artifact 제거 시 PPL 2.5–5 개선.

| Dataset | SOTA | GPT-2 (1.5B) |
|---|---|---|
| LAMBADA (PPL) | 99.8 | **8.63** |
| LAMBADA (ACC) | 59.23 | **63.24** |
| CBT-CN (ACC) | 85.7 | **93.30** |
| CBT-NE (ACC) | 82.3 | **89.05** |
| WikiText-2 (PPL) | 39.14 | **18.34** |
| **PTB (PPL)** | 46.54 | **35.76** |
| enwik8 (BPC) | 0.99 | **0.93** |
| text8 (BPC) | 1.08 | **0.98** |
| WikiText-103 (PPL) | 18.3 | **17.48** |
| 1BW (PPL) | **21.8** | 42.16 |

→ 8개 중 7개에서 SOTA. **1BW만 실패** — 가장 큰 데이터셋이지만 sentence-level shuffling으로 long-range structure가 모두 파괴됨.

#### 3.2 Children's Book Test (CBT)
빈칸 채우기(cloze) — 10개 후보 중 정답 선택. GPT-2: **공통명사 93.3%**, named entity 89.1% (둘 다 SOTA, human은 ~96%).

#### 3.3 LAMBADA
50+ 토큰 context로 마지막 단어 예측 — long-range dependency 측정. SOTA 99.8 → **8.6 PPL** (10x 개선!), accuracy 19% → 52.66%, stop-word filter 추가시 63.24%.

#### 3.4 Winograd Schema Challenge
Commonsense reasoning. 273 examples. SOTA 63.7 → **70.70%**.

#### 3.5 Reading Comprehension (CoQA)
Conversational QA, 7 domains. Greedy decoding from GPT-2 conditioned on (document, conversation history, "A:") token: **55 F1** (zero-shot). Supervised SOTA(BERT)는 89 F1이지만, GPT-2는 baseline 4개 중 3개를 fine-tuning 없이 능가.

#### 3.6 Summarization (CNN/DailyMail)
"TL;DR:" 토큰을 article 뒤에 붙이고 top-$k=2$ sampling으로 100 tokens 생성. 처음 3 sentence를 summary로 사용. ROUGE F1:

| | R-1 | R-2 | R-L | R-AVG |
|---|---|---|---|---|
| Bottom-Up Sum (SOTA) | 41.22 | 18.68 | 38.34 | 32.75 |
| Lede-3 baseline | 40.38 | 17.66 | 36.62 | 31.55 |
| **GPT-2 TL;DR** | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2 no hint | 21.58 | 4.03 | 19.47 | 15.03 |

→ 약하지만, "TL;DR" hint 제거시 6.4 ROUGE 하락 — task-specific behavior가 자연어 prompt로 발화됨을 시연.

#### 3.7 Translation (WMT-14)
Few-shot prompting: `english sentence = french sentence` 형태의 예시쌍 후 `english sentence =`. 
- WMT-14 En-Fr: **5 BLEU** (word-by-word baseline보다 약함)
- WMT-14 Fr-En: **11.5 BLEU** (수 unsupervised baseline 능가, 단 unsupervised SOTA 33.5에는 한참 못 미침)
- WebText에서 non-English 페이지를 **명시적으로 제거**했음에도 작동 — 10MB의 우연한 프랑스어 fragment만으로 학습됨.

#### 3.8 Question Answering (Natural Questions)
Few-shot QA pair prompts. Exact-match accuracy **4.1%**(SQUAD-style). 117M 모델은 1.0% (frequency baseline)도 못 넘음 — 용량이 결정적. 30개 most confident answers는 calibration 양호 (Table 5).

**English**

#### 3.1 Language Modeling (Table 3)
Zero-shot evaluation across 8 standard LM datasets, with invertible de-tokenizers removing PTB-style artifacts (gain of 2.5-5 PPL).

| Dataset | SOTA | GPT-2 (1.5B) |
|---|---|---|
| LAMBADA (PPL) | 99.8 | **8.63** |
| LAMBADA (ACC) | 59.23 | **63.24** |
| CBT-CN (ACC) | 85.7 | **93.30** |
| CBT-NE (ACC) | 82.3 | **89.05** |
| WikiText-2 (PPL) | 39.14 | **18.34** |
| **PTB (PPL)** | 46.54 | **35.76** |
| enwik8 (BPC) | 0.99 | **0.93** |
| text8 (BPC) | 1.08 | **0.98** |
| WikiText-103 (PPL) | 18.3 | **17.48** |
| 1BW (PPL) | **21.8** | 42.16 |

→ SOTA on 7/8. The only failure (1BW) is the largest dataset but its sentence-level shuffling destroys long-range structure.

#### 3.2 Children's Book Test (CBT)
A cloze task picking from 10 candidates. GPT-2: **93.3% common nouns**, 89.1% named entities (both SOTA; human ~96%).

#### 3.3 LAMBADA
Predict the last word given ≥50 tokens of context — measures long-range dependency. SOTA 99.8 → **8.6 PPL** (10× improvement), accuracy 19% → 52.66%, with a stop-word filter 63.24%.

#### 3.4 Winograd Schema Challenge
Commonsense reasoning over 273 examples. SOTA 63.7 → **70.70%**.

#### 3.5 Reading Comprehension (CoQA)
Conversational QA across 7 domains. Greedy decoding from GPT-2 conditioned on (document, conversation history, "A:"): **55 F1** zero-shot. Supervised SOTA (BERT) is 89 F1, but GPT-2 beats 3 of 4 baselines without fine-tuning.

#### 3.6 Summarization (CNN/DailyMail)
Append "TL;DR:" to the article, top-$k=2$ sample 100 tokens, take the first 3 sentences as summary. ROUGE F1:

| | R-1 | R-2 | R-L | R-AVG |
|---|---|---|---|---|
| Bottom-Up Sum (SOTA) | 41.22 | 18.68 | 38.34 | 32.75 |
| Lede-3 baseline | 40.38 | 17.66 | 36.62 | 31.55 |
| **GPT-2 TL;DR** | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2 no hint | 21.58 | 4.03 | 19.47 | 15.03 |

→ Weak but striking: removing the hint drops ROUGE 6.4, demonstrating that task-specific behavior is invoked by a natural-language prompt.

#### 3.7 Translation (WMT-14)
Few-shot prompting: pairs `english sentence = french sentence`, then `english sentence =`.
- WMT-14 En-Fr: **5 BLEU** (worse than word-by-word baseline).
- WMT-14 Fr-En: **11.5 BLEU** (beats several unsupervised baselines; unsupervised SOTA 33.5 is far away).
- Surprising because non-English pages were **explicitly removed** from WebText — only ~10MB of accidental French remained.

#### 3.8 Question Answering (Natural Questions)
Few-shot QA prompting. Exact-match accuracy **4.1%** (SQUAD-style). The 117M model barely reaches 1.0% (the frequency baseline) — capacity is the dominant factor. Top-30 confident answers are well-calibrated (Table 5).

### Part VII: Generalization vs Memorization (§4) / 일반화 vs 암기

**한국어**
Recht et al. (2019)이 image dataset의 train/test overlap (CIFAR-10에서 ~3.3%)을 지적한 맥락에서, 저자들은 WebText train set의 8-grams를 **Bloom filter**(false positive ≤ $10^{-8}$)로 인덱싱하고, 평가 데이터셋의 8-gram이 얼마나 overlap하는지 측정합니다.

| Dataset | own train (%) | WebText train (%) |
|---|---|---|
| PTB | 2.67 | 0.88 |
| WikiText-2 | 0.66 | **1.63** |
| enwik8 | 7.50 | 6.31 |
| text8 | 2.34 | **3.94** |
| WikiText-103 | 9.09 | 2.42 |
| 1BW | 13.19 | 3.75 |

**관찰**:
- WebText overlap은 평균 3.2% — 일반 LM dataset의 자체 train/test overlap과 비슷하거나 적음.
- LAMBADA: 평균 1.2% overlap. Overlap 있는 example이 ~2 PPL 더 잘함. Overlap 제거 시 PPL 8.6→8.7, accuracy 63.2%→62.9% — 미미한 영향.
- Winograd: 273 schema 중 10개만 8-gram overlap, 그 중 1개만 정답을 spoil.
- CoQA: news 도메인의 15%가 WebText에 있으나, 정답 자체는 cutoff 이후 발표되어 trivially-leaked는 아님.

**결론**: overlap의 영향은 작지만 일관됨. n-gram 기반 dedup이 standard practice가 되어야 함. **GPT-2는 여전히 WebText를 underfit** — train과 test set의 perplexity가 함께 개선되며, 모델이 더 커져야 한다는 신호.

**English**
In the spirit of Recht et al. (2019)'s exposure of train/test overlap in image datasets (~3.3% in CIFAR-10), the authors index 8-grams of WebText train via **Bloom filters** (false-positive ≤ $10^{-8}$) and measure how many test 8-grams hit them.

| Dataset | own train (%) | WebText train (%) |
|---|---|---|
| PTB | 2.67 | 0.88 |
| WikiText-2 | 0.66 | **1.63** |
| enwik8 | 7.50 | 6.31 |
| text8 | 2.34 | **3.94** |
| WikiText-103 | 9.09 | 2.42 |
| 1BW | 13.19 | 3.75 |

**Observations**:
- WebText overlap averages 3.2% — comparable to or smaller than each dataset's own train/test overlap.
- LAMBADA: 1.2% average overlap. Overlapping examples score ~2 PPL better. Excluding overlap: PPL 8.6→8.7, accuracy 63.2%→62.9%, marginal.
- Winograd: only 10 of 273 schemas had 8-gram overlap, only 1 spoiled.
- CoQA news domain: 15% in WebText, but answers were released after the cutoff so leakage is limited.

**Conclusion**: overlap effect is small but consistent — n-gram dedup should be standard. **GPT-2 still underfits WebText** — train and test perplexity continue improving in tandem.

---

## 3. Key Takeaways / 핵심 시사점

1. **Tasks-as-text, not tasks-as-heads** — 모든 NLP task를 자연어 prompt + 입력으로 표현해 단일 LM이 conditional 분포로 처리. 이는 fine-tuning 없는 multitask learning의 새 패러다임. / Frame all NLP tasks as natural-language prompts so a single LM can handle them via conditional distributions — a new paradigm of fine-tuning-free multitasking.

2. **Scale begets generalization (log-linear)** — 117M → 1.5B에서 zero-shot 성능이 선형적으로 (로그 스케일에서) 개선됩니다. 이는 GPT-3 (Brown 2020)와 Chinchilla (Hoffmann 2022) scaling laws의 직접적 motivation. / Zero-shot performance scales log-linearly with parameters from 117M to 1.5B — direct motivation for GPT-3 and Chinchilla scaling laws.

3. **WebText: data quality via human-curation proxy** — Reddit ≥3 karma는 user-curation 휴리스틱. 단순 Common Crawl보다 quality가 훨씬 좋고, 단일 도메인보다 다양함. / Reddit ≥3 karma is a clever proxy for human curation: better quality than Common Crawl, more diverse than single-domain corpora.

4. **Byte-level BPE: universal tokenization** — UTF-8 byte 위에서 동작하므로 모든 Unicode string에 확률 부여. category-경계 merge 차단으로 vocab 효율 보존. 이후 모든 modern LM의 표준이 됨. / Byte-level BPE assigns probability to any Unicode string and, with cross-category merge blocking, maintains vocab efficiency — now standard in all modern LMs.

5. **Pre-LayerNorm + residual scaling = stable deep training** — 48-layer transformer를 안정적으로 학습하기 위한 architectural tweak. pre-LN과 $1/\sqrt{N}$ residual scaling은 후일 모든 large transformer의 표준이 됨. / Pre-LN and $1/\sqrt{N}$ residual init are the architectural keys to stable 48-layer training; both became standard in all subsequent large transformers.

6. **Prompt engineering is born** — "TL;DR:" hint 하나가 ROUGE를 6.4 올림. 자연어로 task를 specify하는 것이 가능함을 시연 — ChatGPT/InstructGPT의 씨앗. / A single "TL;DR:" hint adds 6.4 ROUGE — natural-language task specification works, planting the seed for ChatGPT and InstructGPT.

7. **Capacity is the bottleneck, not data leakage** — 8-gram overlap 분석에서 train/test overlap의 영향은 미미. 117M 모델은 NQ에서 1%, 1.5B 모델은 4.1% — 정확도 차이는 거의 전적으로 capacity. / Bloom-filter analysis shows train/test overlap has marginal effect; the 117M→1.5B accuracy jump on NQ (1%→4.1%) is almost entirely capacity-driven.

8. **Underfitting at 1.5B → bigger is better** — WebText에서 1.5B 모델조차 underfit. 더 큰 모델, 더 긴 학습이 더 잘 할 것이라는 강력한 신호. / Even the 1.5B model underfits WebText — a strong signal that bigger and longer training will do better, fulfilled in GPT-3 and beyond.

---

## 4. Mathematical Summary / 수학적 요약

### Core formulation / 핵심 정식화

**Step 1**: Autoregressive joint factorization (Eq. 1)
$$p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \ldots, s_{i-1})$$
sequence 결합 확률을 left-to-right conditional의 곱으로 분해. tractable sampling/density estimation 보장. / Decomposes the joint into a product of left-to-right conditionals; permits tractable sampling and density estimation.

**Step 2**: General task as conditional distribution
$$p(\text{output} \mid \text{input}, \text{task})$$
여기서 task는 자연어 sequence — "translate to french", "summarize:", "Q:...A:" 등. supervised objective는 unsupervised objective의 subset이므로 둘의 global minimum이 일치. / Task is a natural-language descriptor; supervised objective is a subset of the unsupervised one, so global minima coincide.

**Step 3**: Transformer decoder forward pass
하나의 decoder block에서, $X \in \mathbb{R}^{T \times d}$ 입력에 대해 / For one decoder block on input $X \in \mathbb{R}^{T \times d}$:

$$\tilde{X} = \mathrm{LN}(X) \quad (\text{pre-LN, GPT-2의 변경점})$$
$$Y = X + \mathrm{MultiHead}(\tilde{X})$$
$$\tilde{Y} = \mathrm{LN}(Y)$$
$$Z = Y + \mathrm{MLP}(\tilde{Y})$$

여기서 multi-head attention은 / Multi-head attention is:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases} \quad (\text{causal mask})$$

causal mask $M$이 미래 token attention을 차단해 left-to-right factorization을 강제. / Causal mask $M$ blocks future-token attention, enforcing the autoregressive factorization.

**Step 4**: Output distribution
$$p(s_t \mid s_{<t}) = \mathrm{softmax}(W_E^\top h_t)$$
final hidden state $h_t$에 token embedding matrix $W_E$를 transposed하게 곱(weight tying)해 vocabulary 분포 산출. / Final hidden state is projected through transposed token embedding (weight tying) to produce a vocab distribution.

**Step 5**: Loss = cross-entropy = negative log-likelihood
$$\mathcal{L} = -\sum_{t=1}^{T} \log p(s_t \mid s_{<t}) = -\sum_{t=1}^{T} \log \mathrm{softmax}(W_E^\top h_t)_{s_t}$$

**Step 6**: Perplexity (evaluation metric)
$$\mathrm{PPL}(x) = \exp\!\left(\frac{\mathcal{L}}{T}\right) = \exp\!\left(-\frac{1}{T}\sum_t \log p(s_t \mid s_{<t})\right)$$

낮을수록 좋음. uniform 분포에서 vocabulary $V$는 PPL = $V$. / Lower is better; uniform over vocab $V$ gives PPL = $V$.

**Step 7**: Residual scaling at init
$$W_{\text{residual}} \sim \mathcal{N}\!\left(0, \frac{\sigma^2}{N}\right)$$
$N$ residual layer가 누적될 때, $\mathrm{Var}(\sum_l y_l) \propto N \cdot \sigma^2/N = \sigma^2$로 안정. / With $N$ residuals, total variance stays $\sim \sigma^2$ instead of growing with depth.

### Worked example: zero-shot QA prompt / 워크스루 — zero-shot QA prompt

**한국어**
다음 zero-shot QA를 고려:
```
Q: Who wrote the book the origin of species?
A:
```
GPT-2는 conditional $p(\text{answer} \mid \text{prompt})$를 추정. 학습 corpus(WebText) 어디에도 이 정확한 prompt-answer pair는 없지만, 다음과 같은 자연 텍스트는 풍부하게 등장합니다:
- 위키백과 스타일: "Charles Darwin wrote *On the Origin of Species* in 1859."
- 인용 구문: "...the book the origin of species, written by Charles Darwin..."

LM 목적 함수는 이런 contexts에서 "Charles Darwin"이 다음 token이 될 확률을 끌어올리도록 학습. zero-shot에서는 prompt structure만 제공하면 추론. GPT-2 1.5B는 "Charles Darwin"에 83.4% 확률을 부여 (Table 5의 top answer).

**English**
Consider the zero-shot QA:
```
Q: Who wrote the book the origin of species?
A:
```
GPT-2 estimates $p(\text{answer} \mid \text{prompt})$. The exact prompt-answer pair never appears in WebText, but related natural text does:
- Wikipedia-style: "Charles Darwin wrote *On the Origin of Species* in 1859."
- Citation form: "...the book the origin of species, written by Charles Darwin..."

LM training pushes up the probability of "Charles Darwin" as the next token in such contexts. With only the prompt structure provided, GPT-2 1.5B assigns 83.4% probability to "Charles Darwin" (top entry in Table 5).

### Parameter count: GPT-2 1.5B / 매개변수 수 계산

per-layer (with $d = 1600$, FFN inner dim $= 4d = 6400$):
- attention: $4 \times d^2 = 4 \times 1600^2 \approx 10.24$M (Q, K, V, O projections)
- FFN: $2 \times d \times 4d = 8d^2 \approx 20.48$M
- LN params: ~6400 (negligible)
- per-layer total: ~30.7M
- 48 layers: ~1473M
- token embedding: $50{,}257 \times 1600 \approx 80$M
- positional embedding: $1024 \times 1600 \approx 1.6$M
- **Total ≈ 1542M** ✓

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1980 ─ Jelinek & Mercer: Interpolated estimation (n-gram LM)
        │
1997 ─ Caruana: Multitask Learning (theoretical foundation)
        │
2003 ─ Bengio et al.: Neural Probabilistic Language Model
        │
2013 ─ Mikolov et al.: word2vec (paper #21)
        │
2014 ─ Sutskever et al.: Seq2Seq learning
        │
2014 ─ Bahdanau et al.: Attention (paper #17)
        │
2015 ─ Sennrich et al.: Byte Pair Encoding (BPE)
        │
2017 ─ Vaswani et al.: Transformer (paper #25) ★
        │       ─ self-attention 기반 완전 신경망
        │
2018 ─ Peters et al.: ELMo — contextualized word reps
2018 ─ Radford et al.: GPT-1 — decoder-only pre-train + fine-tune
2018 ─ Devlin et al.: BERT — bidirectional pre-train + fine-tune
2018 ─ McCann et al.: decaNLP — natural-language task spec
        │
2019 ─ ★★★ Radford et al.: GPT-2 (this paper) ★★★
        │       ─ zero-shot generalist via scale + diverse data
        │       ─ 117M / 345M / 762M / 1.5B
        │
2020 ─ Brown et al.: GPT-3 (175B, in-context few-shot)
        │
2020 ─ Kaplan et al.: Scaling Laws for Neural LMs
        │
2022 ─ Hoffmann et al.: Chinchilla (compute-optimal scaling)
2022 ─ Ouyang et al.: InstructGPT (RLHF)
2022 ─ ChatGPT release
        │
2023 ─ GPT-4 (multimodal)
2024 ─ Claude 3, Gemini, Llama 3 — descendants of GPT-2's recipe
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#17 Bahdanau et al. (2014) — Attention** | self-attention의 직계 조상. GPT-2의 모든 inter-token mixing은 attention으로 실행. / Direct ancestor of self-attention; all inter-token mixing in GPT-2 uses attention. | High — 메커니즘적 기반 / Mechanistic foundation |
| **#19 Ioffe & Szegedy (2015) — Batch Norm** | LayerNorm은 BN의 sequence-friendly 버전. GPT-2는 pre-LN으로 LN의 위치를 옮긴 것이 핵심 변경. / LayerNorm is BN's sequence-friendly cousin; GPT-2's pre-LN move is its key tweak. | High — 학습 안정성 / Training stability |
| **#20 He et al. (2015) — ResNet** | residual connection이 48-layer transformer의 학습을 가능하게 함. residual scaling $1/\sqrt{N}$은 ResNet 스타일. / Residual connections enable 48-layer transformer training; $1/\sqrt{N}$ scaling echoes ResNet philosophy. | High — depth-stable training |
| **#25 Vaswani et al. (2017) — Transformer** | GPT-2의 architecture base. encoder 없는 decoder-only 변형. / Architectural base of GPT-2 — decoder-only variant. | Critical — 직접 부모 / Direct parent |
| **#26 Kipf & Welling (2017) — GCN** | Transformer를 fully-connected graph의 GAT/GCN으로 보는 통합적 시각의 한 축. / One pole of the unifying view that sees transformers as GATs on fully-connected graphs. | Medium — conceptual bridge |
| **#27 GPT-1 (Radford 2018, implied)** | GPT-2의 직접 전신. 같은 architecture + scale up + zero-shot 평가. / Direct predecessor — same architecture, scaled up, zero-shot evaluation. | Critical — 가장 직접적 선조 / Most direct predecessor |
| **#28 BERT (Devlin 2018, implied)** | 같은 시기의 경쟁/대응 — bidirectional masked LM vs. autoregressive LM. GPT-2의 345M ≈ BERT-large. / Contemporaneous counterpoint — masked LM vs. autoregressive; GPT-2 345M ≈ BERT-large size. | High — 패러다임 비교 / Paradigm contrast |
| **GPT-3 (Brown 2020, future)** | GPT-2 recipe의 100배 scaleup. in-context learning을 zero-shot에서 few-shot으로 일반화. / 100× scaleup of GPT-2's recipe; generalizes zero-shot to few-shot in-context learning. | Critical — 직계 후손 / Direct descendant |
| **Kaplan et al. (2020) — Scaling Laws** | GPT-2의 log-linear improvement 곡선을 이론화/정량화. / Theorizes and quantifies GPT-2's log-linear improvement curves. | High — empirical → theory |
| **Sennrich et al. (2015) — BPE** | byte-level BPE의 base algorithm. GPT-2가 byte-level + category-boundary blocking으로 확장. / Base algorithm; GPT-2 extends with byte-level operation and category-boundary blocking. | High — tokenization foundation |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI Technical Report. https://openai.com/research/better-language-models
- Code: https://github.com/openai/gpt-2

### Direct predecessors / 직접 선조
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training* (GPT-1). OpenAI Tech Report.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv:1810.04805*.
- Peters, M. E., Neumann, M., Iyyer, M., et al. (2018). Deep contextualized word representations (ELMo). *NAACL 2018*.

### Multitask & meta-learning / 멀티태스크 및 메타학습
- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41–75.
- McCann, B., Keskar, N. S., Xiong, C., & Socher, R. (2018). The natural language decathlon: Multitask learning as question answering (decaNLP). *arXiv:1806.08730*.
- Bowman, S. R., et al. (2018). Looking for ELMo's friends: Sentence-level pretraining beyond language modeling. *arXiv:1812.10860*.
- Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning. *ICML 2017*.

### Tokenization / 토큰화
- Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine translation of rare words with subword units (BPE). *ACL 2016*.
- Gillick, D., Brunk, C., Vinyals, O., & Subramanya, A. (2015). Multilingual language processing from bytes. *NAACL 2016*.
- Al-Rfou, R., Choe, D., et al. (2018). Character-level language modeling with deeper self-attention. *AAAI 2019*.

### Architecture / 구조
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. *ECCV 2016*.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.

### Datasets & evaluation / 데이터셋 및 평가
- Reddy, S., Chen, D., & Manning, C. D. (2018). CoQA: A conversational question answering challenge. *TACL 2019*.
- Paperno, D., Kruszewski, G., Lazaridou, A., et al. (2016). The LAMBADA dataset. *ACL 2016*.
- Hill, F., Bordes, A., Chopra, S., & Weston, J. (2015). The Children's Book Test. *ICLR 2016*.
- Levesque, H. J., Davis, E., & Morgenstern, L. (2012). The Winograd schema challenge. *KR 2012*.
- Trinh, T. H., & Le, Q. V. (2018). A simple method for commonsense reasoning. *arXiv:1806.02847*.
- See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. *ACL 2017*.
- Nallapati, R., Zhou, B., et al. (2016). Abstractive text summarization using sequence-to-sequence RNNs (CNN/DailyMail). *CoNLL 2016*.
- Artetxe, M., Labaka, G., & Agirre, E. (2017). Unsupervised neural machine translation. *ICLR 2018*.
- Kwiatkowski, T., et al. (2019). Natural Questions. *TACL 2019*.

### Generalization & memorization / 일반화 및 암기
- Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet classifiers generalize to ImageNet? *ICML 2019*.
- Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2018). Do CIFAR-10 classifiers generalize to CIFAR-10? *arXiv:1806.00451*.
- Jia, R., & Liang, P. (2017). Adversarial examples for evaluating reading comprehension systems. *EMNLP 2017*.

### Future descendants / 미래 후속
- Brown, T. B., et al. (2020). Language models are few-shot learners (GPT-3). *NeurIPS 2020*.
- Kaplan, J., McCandlish, S., et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361*.
- Hoffmann, J., et al. (2022). Training compute-optimal large language models (Chinchilla). *NeurIPS 2022*.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback (InstructGPT). *NeurIPS 2022*.
