---
title: "Efficient Estimation of Word Representations in Vector Space"
authors: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
year: 2013
journal: "arXiv:1301.3781"
doi: "arXiv:1301.3781"
topic: Artificial Intelligence
tags: [word2vec, word-embeddings, CBOW, skip-gram, NLP, distributed-representations, language-model]
status: completed
date_started: 2026-04-14
date_completed: 2026-04-14
---

# 14. Efficient Estimation of Word Representations in Vector Space / 벡터 공간에서의 효율적인 단어 표현 추정

---

## 1. Core Contribution / 핵심 기여

이 논문은 대규모 텍스트 코퍼스에서 고품질 단어 벡터(word vector)를 학습하기 위한 두 가지 새로운 log-linear 모델 아키텍처를 제안합니다: **CBOW(Continuous Bag-of-Words)**와 **Skip-gram**. 핵심 통찰은 기존 neural network language model(NNLM)에서 계산 병목이 되는 **비선형 hidden layer를 제거**하면, 모델의 표현력은 다소 줄어들지만 훨씬 더 많은 데이터에서 훨씬 더 빠르게 학습할 수 있다는 것입니다. 이를 통해 16억 단어 데이터셋에서 하루 이내에 고품질 word vector를 학습할 수 있었으며, 학습된 벡터는 $\text{vector}("King") - \text{vector}("Man") + \text{vector}("Woman") \approx \text{vector}("Queen")$과 같은 놀라운 선형 규칙성(linear regularity)을 보여줍니다.

This paper proposes two new log-linear model architectures for learning high-quality word vectors from large text corpora: **CBOW (Continuous Bag-of-Words)** and **Skip-gram**. The key insight is that **removing the computationally expensive non-linear hidden layer** from existing NNLM reduces expressiveness somewhat but allows training on far more data far more quickly. This enabled learning high-quality word vectors from a 1.6 billion word dataset in less than a day. The learned vectors exhibit remarkable linear regularities such as $\text{vector}("King") - \text{vector}("Man") + \text{vector}("Woman") \approx \text{vector}("Queen")$.

저자들은 새로운 Semantic-Syntactic Word Relationship 테스트셋을 설계하여 word vector의 품질을 체계적으로 평가했습니다. 이 테스트셋은 8,869개의 의미적 질문과 10,675개의 구문적 질문으로 구성되며, "A:B = C:?" 형태의 유추 문제를 벡터 산술로 풀 수 있는지 측정합니다. 실험 결과 Skip-gram은 의미적 관계에서, CBOW는 구문적 관계에서 상대적으로 강점을 보였으며, 두 모델 모두 기존의 NNLM, RNNLM보다 훨씬 적은 학습 시간으로 동등하거나 더 나은 정확도를 달성했습니다.

The authors designed a new Semantic-Syntactic Word Relationship test set to systematically evaluate word vector quality. This test set consists of 8,869 semantic questions and 10,675 syntactic questions, measuring whether analogy problems of the form "A:B = C:?" can be solved via vector arithmetic. Experiments showed Skip-gram excels at semantic relationships while CBOW is relatively stronger at syntactic ones, and both models achieve comparable or better accuracy than NNLM/RNNLM with far less training time.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Goals / 소개 및 목표 (Sections 1–1.2)

#### 문제 인식 / Problem Recognition

2013년 당시 대부분의 NLP 시스템은 단어를 **원자적 단위(atomic unit)**로 취급했습니다. 단어는 어휘 목록(vocabulary)의 인덱스에 불과했고, "cat"과 "dog"가 의미적으로 유사하다는 정보를 모델이 활용할 수 없었습니다. N-gram 모델은 단순하고 강건하여 수조 단어 규모의 데이터에도 적용 가능했지만, 이런 단순함의 대가로 단어 간 유사성을 전혀 포착하지 못했습니다.

In 2013, most NLP systems treated words as **atomic units** — mere indices in a vocabulary. The similarity between "cat" and "dog" was invisible to the model. N-gram models were simple and robust enough to scale to trillions of words, but this simplicity came at the cost of completely missing inter-word similarities.

반면 neural network 기반 language model은 단어를 연속적인 벡터로 표현(distributed representation)하여 유사한 단어가 벡터 공간에서 가까이 위치하게 할 수 있었지만, 학습에 **막대한 계산 비용**이 소요되었습니다. 예를 들어 RNNLM을 학습하는 데 한 대의 CPU로 약 8주가 걸렸습니다.

On the other hand, neural network-based language models could represent words as continuous vectors (distributed representations) where similar words cluster together in vector space, but training required **enormous computational cost** — for example, training an RNNLM took about 8 weeks on a single CPU.

#### 논문의 목표 / Paper's Goals

저자들의 목표는 명확합니다: **수십억 단어, 수백만 어휘 규모**의 데이터셋에서도 고품질 word vector를 학습할 수 있는 기술을 개발하는 것. 기존 어떤 아키텍처도 이 규모에서 성공적으로 학습된 적이 없었으며(word vector 차원 50–100), 저자들은 이를 300–1000차원까지 확장하고자 했습니다.

The authors' goal is clear: develop techniques to learn high-quality word vectors from datasets with **billions of words and millions of vocabulary entries**. No previous architecture had been successfully trained at this scale (with vector dimensionality of only 50–100), and the authors aimed to scale up to 300–1000 dimensions.

특히 주목할 점은 저자들이 품질 측정 기준으로 **벡터 간 선형 규칙성(linear regularities)**을 제시한 것입니다. 단순히 유사한 단어끼리 가까이 있는 것(proximity)을 넘어서, $\text{vector}("King") - \text{vector}("Man") + \text{vector}("Woman") \approx \text{vector}("Queen")$처럼 **방향(direction)**이 특정 관계를 인코딩하는지를 평가합니다. 이는 Rumelhart et al. (1986, 논문 #6)이 제시한 distributed representation의 이상적 특성 — 의미적 관계가 벡터 공간의 기하학적 구조로 반영되는 것 — 을 구체적으로 실현한 것입니다.

Notably, the authors propose **linear regularities between vectors** as the quality metric. Beyond simple proximity of similar words, they evaluate whether **directions** in vector space encode specific relationships, like $\text{vector}("King") - \text{vector}("Man") + \text{vector}("Woman") \approx \text{vector}("Queen")$. This concretely realizes the ideal property of distributed representations proposed by Rumelhart et al. (1986, Paper #6) — semantic relationships reflected in the geometric structure of vector space.

---

### Part II: Existing Model Architectures / 기존 모델 아키텍처 (Section 2)

저자들은 모든 모델의 학습 복잡도를 하나의 통일된 프레임워크로 비교합니다:

The authors compare all models' training complexity within a unified framework:

$$O = E \times T \times Q$$

여기서 $E$는 epoch 수, $T$는 학습 단어 수, $Q$는 아키텍처별 복잡도입니다. $E$와 $T$는 모든 모델에 공통이므로, 결국 **$Q$의 최소화**가 설계 목표가 됩니다.

Where $E$ is number of epochs, $T$ is number of training words, and $Q$ is per-architecture complexity. Since $E$ and $T$ are shared across all models, the design goal reduces to **minimizing $Q$**.

#### 2.1 Feedforward NNLM (Bengio et al., 2003)

NNLM의 구조는 4개의 층으로 구성됩니다:

NNLM consists of four layers:

1. **Input layer**: $N$개의 이전 단어를 1-of-V(one-hot) 인코딩으로 입력
   - $N$ previous words encoded as 1-of-V (one-hot) vectors
2. **Projection layer** ($P$): $N \times D$ 차원. 각 단어의 one-hot 벡터에 공유 projection 행렬을 곱하여 $D$차원 벡터로 변환한 후 concatenate
   - Dimensionality $N \times D$. Each word's one-hot vector is multiplied by a shared projection matrix to produce a $D$-dimensional vector, then concatenated
3. **Hidden layer** ($H$): 비선형 활성화 함수(tanh 등) 적용. 보통 500–1000 유닛
   - Non-linear activation (tanh, etc.). Typically 500–1000 units
4. **Output layer**: vocabulary 전체에 대한 확률 분포 계산 (softmax)
   - Probability distribution over entire vocabulary (softmax)

학습 복잡도는:

Training complexity:

$$Q_{\text{NNLM}} = N \times D + N \times D \times H + H \times V$$

여기서 **지배적 항은 $H \times V$**입니다. 예를 들어 $H = 500$, $V = 1{,}000{,}000$이면 이 항만 5억 번의 연산이 필요합니다. Hierarchical softmax를 사용하면 $H \times V$를 $H \times \log_2(V) \approx H \times 20$으로 줄일 수 있지만, 여전히 **$N \times D \times H$ 항**이 남습니다. $N=10, D=500, H=500$이면 이 항은 250만으로, projection과 hidden layer 사이의 **dense 행렬 곱셈**이 여전히 병목입니다.

The **dominant term is $H \times V$**. For example, with $H = 500$ and $V = 1{,}000{,}000$, this term alone requires 500 million operations. Hierarchical softmax can reduce $H \times V$ to $H \times \log_2(V) \approx H \times 20$, but the **$N \times D \times H$ term** remains. With $N=10, D=500, H=500$, this term is 2.5 million — the **dense matrix multiplication** between projection and hidden layers is still a bottleneck.

#### 2.2 Recurrent NNLM (RNNLM)

RNNLM은 NNLM의 두 가지 한계를 극복하고자 제안되었습니다:

RNNLM was proposed to overcome two limitations of NNLM:

1. **고정된 context 길이**: NNLM은 $N$개의 단어만 볼 수 있지만, RNNLM은 hidden state를 통해 이론적으로 무한한 과거를 참조 가능
   - **Fixed context length**: NNLM sees only $N$ words, but RNNLM can theoretically reference infinite past through hidden state
2. **Projection layer 불필요**: 단어가 직접 hidden layer에 연결됨
   - **No projection layer needed**: Words connect directly to hidden layer

복잡도는:

Complexity:

$$Q_{\text{RNNLM}} = H \times H + H \times V$$

$H \times H$ 항은 recurrent connection(hidden → hidden)에서 발생합니다. $H \times V$를 hierarchical softmax로 줄이면 $H \times H$가 지배적이 됩니다. $H$가 word vector 차원 $D$와 같으므로, word vector의 차원을 높이면 복잡도가 $D^2$로 증가합니다.

The $H \times H$ term comes from recurrent connections (hidden → hidden). With hierarchical softmax reducing $H \times V$, $H \times H$ dominates. Since $H$ equals word vector dimensionality $D$, increasing vector dimensions raises complexity as $D^2$.

#### 핵심 관찰 / Key Observation

두 기존 모델의 복잡도를 분석한 결과, **비선형 hidden layer가 가장 큰 계산 병목**이라는 것이 저자들의 핵심 관찰입니다. NNLM에서는 $N \times D \times H$ 항, RNNLM에서는 $H \times H$ 항이 이에 해당합니다. 이 관찰이 Section 3의 새로운 모델 설계를 이끄는 핵심 동기가 됩니다.

The key observation from analyzing both existing models is that the **non-linear hidden layer is the biggest computational bottleneck** — $N \times D \times H$ in NNLM and $H \times H$ in RNNLM. This observation directly motivates the new model designs in Section 3.

---

### Part III: New Log-linear Models — CBOW and Skip-gram / 새로운 Log-linear 모델 (Section 3)

이 섹션이 논문의 **핵심**입니다. 저자들의 전략은 명확합니다: neural network에서 가장 매력적인 부분은 **단어를 연속 벡터로 표현하는 능력**인데, 이를 위해 반드시 복잡한 비선형 모델이 필요한 것은 아니다. 더 단순한 모델로도 단어 벡터를 잘 학습할 수 있다면, 그 단순함 덕분에 훨씬 더 많은 데이터를 처리할 수 있다.

This section is the **core** of the paper. The authors' strategy is clear: the most attractive aspect of neural networks is their **ability to represent words as continuous vectors**, but this doesn't necessarily require complex non-linear models. If simpler models can learn word vectors well, their simplicity allows processing far more data.

#### 3.1 CBOW (Continuous Bag-of-Words)

CBOW는 NNLM에서 세 가지 핵심적인 단순화를 적용합니다:

CBOW applies three key simplifications to NNLM:

1. **Hidden layer 제거**: 비선형 활성화 함수가 있는 hidden layer를 완전히 제거
   - **Remove hidden layer**: Completely remove the hidden layer with non-linear activation
2. **Projection layer 공유**: 모든 단어가 **동일한** projection 행렬을 공유할 뿐 아니라, 각 단어의 projection 결과를 concatenate하지 않고 **평균(average/sum)**을 취함
   - **Share projection layer**: All words share the **same** projection matrix, and instead of concatenating each word's projection, take the **average/sum**
3. **양방향 context**: 과거 $N$개 단어뿐 아니라 **미래의 단어도** context로 사용
   - **Bidirectional context**: Use not only past $N$ words but also **future words** as context

이 세 가지 변경의 결과, CBOW는 **단어 순서를 무시**합니다(bag-of-words). Projection 결과를 concatenate하면 순서 정보가 보존되지만, 합산/평균을 취하면 순서가 사라집니다. "the cat sat on" → "sat"이나 "on sat the cat" → "sat" 모두 같은 projection을 생성합니다. 이는 모델의 표현력을 제한하지만, 복잡도를 극적으로 낮춥니다.

As a result of these three changes, CBOW **ignores word order** (bag-of-words). Concatenating projections preserves order information, but summing/averaging destroys it. "the cat sat on" → "sat" and "on sat the cat" → "sat" produce the same projection. This limits the model's expressiveness but dramatically reduces complexity.

복잡도:

Complexity:

$$Q_{\text{CBOW}} = N \times D + D \times \log_2(V)$$

비교하면 NNLM의 $N \times D \times H + H \times V$ 대신 $D \times \log_2(V)$만 남습니다. $D=300, V=1{,}000{,}000$이면 $D \times \log_2(V) \approx 300 \times 20 = 6{,}000$으로, NNLM의 수백만–수억 대비 **수천 배 이상 효율적**입니다.

Compared to NNLM's $N \times D \times H + H \times V$, only $D \times \log_2(V)$ remains. With $D=300, V=1{,}000{,}000$: $D \times \log_2(V) \approx 300 \times 20 = 6{,}000$, which is **thousands of times more efficient** than NNLM's millions-to-hundreds of millions.

학습 목표는 context 단어들의 벡터 평균이 주어졌을 때 중심 단어를 정확히 예측하는 것입니다. 4개의 과거 단어와 4개의 미래 단어를 사용했을 때 최고 성능을 얻었다고 보고합니다.

The training objective is to correctly predict the center word given the average of context word vectors. Best performance was reported using 4 past and 4 future words.

#### 3.2 Skip-gram

Skip-gram은 CBOW의 **역방향**입니다:

Skip-gram is the **reverse** of CBOW:

- **CBOW**: context → target word (주변 단어들로 중심 단어 예측)
- **Skip-gram**: target word → context (중심 단어로 주변 단어들 예측)

구체적으로, 각 학습 단어에 대해 log-linear classifier에 입력하고, 범위 $C$ 내의 주변 단어들을 예측합니다. 먼 단어일수록 관련성이 낮으므로, 1부터 $C$ 사이에서 랜덤하게 $R$을 선택하고, 현재 단어의 앞뒤 $R$개 단어를 예측 대상으로 사용합니다. 이렇게 하면 먼 단어에 자연스럽게 **낮은 가중치**를 부여하는 효과가 있습니다(가까운 단어가 더 자주 학습 대상이 됨).

Specifically, for each training word, it inputs the word to a log-linear classifier and predicts surrounding words within range $C$. Since distant words are less related, a random $R$ is chosen between 1 and $C$, and $R$ words before and after the current word are used as prediction targets. This naturally gives **lower weight** to distant words (closer words are more frequently used as targets).

복잡도:

Complexity:

$$Q_{\text{Skip-gram}} = C \times (D + D \times \log_2(V))$$

$C$는 최대 context 거리로, 실험에서 $C = 10$을 사용했습니다. $C$에 비례하여 복잡도가 증가하므로 CBOW보다 느리지만, 여전히 NNLM/RNNLM보다 훨씬 효율적입니다.

$C$ is the maximum context distance, set to $C = 10$ in experiments. Complexity scales linearly with $C$, making Skip-gram slower than CBOW, but still far more efficient than NNLM/RNNLM.

#### CBOW vs. Skip-gram: 핵심 차이 / Key Differences

| 측면 / Aspect | CBOW | Skip-gram |
|---|---|---|
| 방향 / Direction | 문맥 → 중심 단어 / Context → center word | 중심 단어 → 문맥 / Center word → context |
| 속도 / Speed | 더 빠름 / Faster | 더 느림 / Slower |
| 빈출 단어 / Frequent words | 강점 (여러 context가 평균화) / Strength (multiple contexts averaged) | 약점 상대적 / Relatively weaker |
| 희귀 단어 / Rare words | 약점 (평균에 묻힘) / Weakness (lost in averaging) | 강점 (각 출현마다 독립 학습) / Strength (each occurrence trained independently) |
| 구문적 관계 / Syntactic | 더 강함 / Stronger | 상대적 약함 / Relatively weaker |
| 의미적 관계 / Semantic | 상대적 약함 / Relatively weaker | 더 강함 / Stronger |

CBOW가 빈출 단어에 강하고 구문적 관계를 잘 포착하는 이유는 **평균화(averaging)** 때문입니다. 자주 등장하는 단어일수록 더 많은 context에서 평균이 계산되어 안정적인 벡터를 학습합니다. 반면 Skip-gram은 각 출현마다 독립적으로 학습하므로, 드물게 등장하는 단어라도 그 한 번의 출현에서 충실히 학습됩니다.

CBOW's strength with frequent words and syntactic relationships comes from **averaging**: the more often a word appears, the more contexts contribute to a stable averaged vector. Skip-gram trains independently on each occurrence, so even rare words are faithfully learned from their few appearances.

#### Training Details / 학습 세부사항

논문의 실험적 성공 뒤에는 여러 가지 중요한 학습 세부사항이 숨어 있습니다. 이런 실용적 디테일들이 Word2Vec을 단순한 이론적 제안이 아닌 **재현 가능하고 널리 채택된 도구**로 만들었습니다.

Behind the paper's experimental success lie several important training details. These practical details transformed Word2Vec from a mere theoretical proposal into a **reproducible and widely adopted tool**.

**학습률 스케줄 / Learning Rate Schedule**: 초기 학습률 $\alpha = 0.025$에서 시작하여 학습이 진행됨에 따라 **선형적으로 0에 가깝게 감소(linearly decayed)**시킵니다. 이는 학습 초반에는 빠르게 수렴하고, 후반에는 미세 조정하여 안정적인 최종 벡터를 얻기 위함입니다. 고정 학습률 대비 이 스케줄이 상당한 성능 향상을 가져옵니다.

**Learning Rate Schedule**: Starting from initial learning rate $\alpha = 0.025$, it is **linearly decayed** toward 0 as training progresses. This allows fast convergence early on and fine-tuning later for stable final vectors. This schedule yields significant performance gains over a fixed learning rate.

**빈출 단어 서브샘플링 / Subsampling of Frequent Words**: "the", "a", "in" 같은 매우 빈번한 단어는 정보량이 적으면서 학습 데이터의 상당 부분을 차지합니다. 후속 논문(Mikolov et al., 2013b)에서 도입된 서브샘플링은 각 단어를 확률적으로 제거합니다:

**Subsampling of Frequent Words**: Very frequent words like "the", "a", "in" carry little information yet dominate the training data. Subsampling, introduced in the follow-up paper (Mikolov et al., 2013b), probabilistically discards each word:

$$P_{\text{keep}}(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

여기서 $f(w_i)$는 단어 $w_i$의 출현 빈도, $t$는 임계값(보통 $10^{-5}$)입니다. 예를 들어 빈도 0.01인 "the"는 약 97%가 제거되지만, 빈도 $10^{-5}$ 이하의 희귀 단어는 전혀 제거되지 않습니다. 이 기법은 학습 속도를 2-10배 향상시키면서 동시에 희귀 단어의 정확도를 높입니다.

Where $f(w_i)$ is the frequency of word $w_i$ and $t$ is a threshold (typically $10^{-5}$). For example, "the" with frequency 0.01 has about 97% of its occurrences removed, while rare words with frequency $\leq 10^{-5}$ are never removed. This technique speeds up training 2-10x while simultaneously improving accuracy for rare words.

**Window 크기의 영향 / Window Size Effects**: context window 크기는 학습되는 관계의 성격에 큰 영향을 미칩니다. **큰 window**(예: 10-15)는 더 넓은 주제적 맥락을 포착하여 **의미적(semantic) 유사성**이 강한 벡터를 생성합니다 — "dog"와 "cat"이 가까워짐. **작은 window**(예: 2-5)는 인접 단어의 문법적 역할을 포착하여 **구문적(syntactic) 유사성**이 강한 벡터를 생성합니다 — "running"과 "walking"이 가까워짐.

**Window Size Effects**: Context window size significantly affects the nature of learned relationships. **Large windows** (e.g., 10-15) capture broader topical context, producing vectors with stronger **semantic similarity** — "dog" and "cat" become closer. **Small windows** (e.g., 2-5) capture grammatical roles of adjacent words, producing vectors with stronger **syntactic similarity** — "running" and "walking" become closer.

---

### Part IV: Results & Evaluation / 결과 및 평가 (Section 4)

#### 4.1 Semantic-Syntactic Word Relationship Test Set

저자들이 설계한 평가 테스트셋은 14가지 관계 유형을 포함합니다:

The evaluation test set designed by the authors includes 14 relationship types:

- **의미적 관계 (5가지) / Semantic (5 types)**: 수도-국가(Athens-Greece), 도시-주(Chicago-Illinois), 화폐(Angola-kwanza), 성별(brother-sister), 전체 수도
  - Capital-country, city-state, currency, gender, all capitals
- **구문적 관계 (9가지) / Syntactic (9 types)**: 형용사→부사(apparent-apparently), 반의어(ethical-unethical), 비교급(great-greater), 최상급(easy-easiest), 현재분사(think-thinking), 국적 형용사(Switzerland-Swiss), 과거시제(walking-walked), 복수형 명사(mouse-mice), 복수형 동사(work-works)
  - Adjective→adverb, opposite, comparative, superlative, present participle, nationality adjective, past tense, plural nouns, plural verbs

총 8,869개 의미적 질문과 10,675개 구문적 질문. 평가 기준은 **정확 일치(exact match)**입니다 — 코사인 거리로 가장 가까운 단어가 정답과 정확히 일치해야 합니다. 동의어도 오답으로 처리되므로 100% 정확도는 사실상 불가능합니다.

Total of 8,869 semantic and 10,675 syntactic questions. The evaluation criterion is **exact match** — the word with the closest cosine distance must exactly match the answer. Synonyms are counted as wrong, making 100% accuracy practically impossible.

#### 4.2 차원과 데이터 크기의 영향 / Impact of Dimensionality and Data Size

Table 2의 결과는 중요한 통찰을 제공합니다:

Table 2's results provide important insights:

- **차원을 높이거나 데이터를 늘리면 정확도가 향상**되지만, 일정 지점 이후 수확체감(diminishing returns)이 발생
  - **Increasing dimensions or data improves accuracy**, but with diminishing returns past a certain point
- **차원과 데이터 크기를 동시에 늘려야** 최대 효과를 얻음. 50차원+783M 단어(23.2%)보다 300차원+783M 단어(50.4%)가 훨씬 나음
  - **Both dimensions and data size must increase together** for maximum effect. 300-dim + 783M words (50.4%) far outperforms 50-dim + 783M words (23.2%)
- 수식 4에 의해, 학습 데이터를 2배로 늘리는 것과 벡터 차원을 2배로 늘리는 것의 계산 비용 증가가 거의 동일
  - Per Equation 4, doubling training data and doubling vector dimensionality result in nearly the same computational cost increase

#### 4.3 모델 아키텍처 비교 / Model Architecture Comparison (Table 3)

동일한 학습 데이터(LDC 코퍼스, 320M 단어)와 동일한 차원(640)에서의 비교:

Comparison with same training data (LDC corpus, 320M words) and same dimensionality (640):

| 모델 / Model | Semantic [%] | Syntactic [%] | MSR Word Relatedness |
|---|---|---|---|
| RNNLM | 9 | 36 | 35 |
| NNLM | 23 | 53 | 47 |
| **CBOW** | **24** | **64** | **61** |
| **Skip-gram** | **55** | **59** | **56** |

주목할 점:

Key observations:

1. **NNLM이 RNNLM보다 상당히 우수**: RNNLM의 word vector는 hidden layer에 직접 연결되어 있어, word vector와 hidden state가 혼재된 표현을 학습함. 반면 NNLM은 projection layer가 별도로 존재하여 더 깨끗한 word vector를 학습
   - **NNLM significantly outperforms RNNLM**: RNNLM's word vectors are directly connected to the hidden layer, learning a mixed representation. NNLM's separate projection layer learns cleaner word vectors
2. **Skip-gram의 semantic 정확도가 압도적(55%)**: 모든 다른 모델의 2배 이상. 각 단어를 독립적으로 학습하는 Skip-gram의 특성상 의미적 관계를 더 잘 포착
   - **Skip-gram's semantic accuracy is dominant (55%)**: More than double all others. Skip-gram's independent per-word training better captures semantic relationships
3. **CBOW의 syntactic 정확도가 최고(64%)**: 단어 순서를 무시하지만 context 평균화가 문법적 패턴을 잘 포착
   - **CBOW's syntactic accuracy is highest (64%)**: Despite ignoring word order, context averaging captures grammatical patterns well

#### 4.4 대규모 병렬 학습 / Large-Scale Parallel Training (Table 6)

Google News 6B 데이터셋에서 DistBelief 프레임워크를 사용한 대규모 학습 결과:

Large-scale training results on Google News 6B dataset using DistBelief framework:

| 모델 / Model | 차원 / Dim | Semantic [%] | Syntactic [%] | Total [%] | 학습 시간 / Training Time |
|---|---|---|---|---|---|
| NNLM | 100 | 34.2 | 64.5 | 50.8 | 14일 × 180 CPU |
| CBOW | 1000 | 57.3 | 68.9 | 63.7 | 2일 × 140 CPU |
| Skip-gram | 1000 | 66.1 | 65.1 | 65.6 | 2.5일 × 125 CPU |

이 결과가 논문의 핵심 메시지를 압축합니다: **CBOW와 Skip-gram은 NNLM보다 10배 이상 적은 계산 자원으로 더 높은 정확도를 달성합니다.** NNLM은 14일 × 180 CPU = 2,520 CPU-days가 필요한 반면, Skip-gram은 2.5일 × 125 CPU = 312.5 CPU-days로 충분합니다.

These results compress the paper's core message: **CBOW and Skip-gram achieve higher accuracy with more than 10× less computational resources than NNLM.** NNLM requires 14 days × 180 CPUs = 2,520 CPU-days, while Skip-gram needs only 2.5 days × 125 CPUs = 312.5 CPU-days.

#### 4.5 Microsoft Sentence Completion Challenge

Skip-gram vector를 RNNLM과 결합했을 때 기존 최고 성능(55.4%)을 넘어 **58.9%**를 달성했습니다. 이는 word vector가 독립적인 태스크뿐 아니라 **다른 모델과 상보적(complementary)**으로 결합될 수 있음을 보여줍니다.

Combining Skip-gram vectors with RNNLM achieved **58.9%**, surpassing the previous best (55.4%). This demonstrates that word vectors can be **complementarily combined** with other models, not just used independently.

---

### Supplement: Negative Sampling / 보충: Negative Sampling (Mikolov et al., 2013b)

**참고 / Note**: Negative sampling은 이 논문 자체에서 제안된 것이 아니라, 같은 해 후속 논문 "Distributed Representations of Words and Phrases and their Compositionality"(Mikolov et al., 2013b)에서 도입되었습니다. 그러나 Word2Vec의 실용적 성공에 결정적 역할을 했으며 이후 사실상 표준 학습 방법이 되었으므로, 이 논문의 맥락에서 함께 이해하는 것이 중요합니다.

**Note**: Negative sampling was not proposed in this paper itself, but introduced in the same-year follow-up paper "Distributed Representations of Words and Phrases and their Compositionality" (Mikolov et al., 2013b). However, it played a decisive role in Word2Vec's practical success and became the de facto standard training method, making it important to understand in the context of this paper.

#### 동기: Hierarchical Softmax의 한계 / Motivation: Limitations of Hierarchical Softmax

이 논문에서 사용한 hierarchical softmax는 $O(V)$를 $O(\log V)$로 줄여주지만 여전히 단점이 있습니다: (1) Huffman 트리 구조를 사전에 구축해야 하고, (2) 빈출 단어와 희귀 단어의 경로 길이 차이가 학습 불균형을 야기할 수 있으며, (3) 구현이 복잡합니다.

The hierarchical softmax used in this paper reduces $O(V)$ to $O(\log V)$ but still has drawbacks: (1) Huffman tree structure must be pre-built, (2) path length differences between frequent and rare words can cause training imbalances, and (3) implementation is complex.

#### Negative Sampling의 목적 함수 / Negative Sampling Objective Function

Negative sampling(NEG)은 훨씬 단순한 접근법을 제안합니다. 전체 vocabulary에 대한 확률 분포를 계산하는 대신, **실제 context에 등장하는 단어(positive sample)와 무작위로 선택한 $k$개의 단어(negative samples)만 구별**하면 됩니다:

Negative sampling (NEG) proposes a far simpler approach. Instead of computing a probability distribution over the entire vocabulary, it only needs to **distinguish actual context words (positive samples) from $k$ randomly chosen words (negative samples)**:

$$\log \sigma(\mathbf{v}'_{w_O}{}^T \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{v}'_{w_i}{}^T \mathbf{v}_{w_I})]$$

여기서 $w_I$는 입력 단어, $w_O$는 실제 context 단어, $\sigma$는 시그모이드 함수입니다. 첫 번째 항은 실제 쌍의 내적을 최대화(이 두 단어는 함께 등장하므로 벡터가 유사해야 함)하고, 두 번째 항은 무작위로 뽑힌 단어들의 내적을 최소화(이 단어들은 관련 없으므로 벡터가 달라야 함)합니다.

Where $w_I$ is the input word, $w_O$ is the actual context word, and $\sigma$ is the sigmoid function. The first term maximizes the dot product for real pairs (these words co-occur, so their vectors should be similar), and the second term minimizes the dot product for randomly sampled words (these are unrelated, so their vectors should differ).

#### Noise Distribution의 선택 / Choice of Noise Distribution

Negative sample을 뽑는 noise distribution은 다음과 같이 정의됩니다:

The noise distribution for drawing negative samples is defined as:

$$P_n(w) = \frac{U(w)^{3/4}}{Z}$$

여기서 $U(w)$는 단어 $w$의 unigram 빈도, $Z$는 정규화 상수입니다. 왜 **3/4 지수**를 사용할까요? 균등 분포($U(w)^0$)를 쓰면 희귀 단어가 너무 자주 negative sample로 뽑히고, 원래 빈도($U(w)^1$)를 쓰면 빈출 단어가 지나치게 많이 뽑힙니다. 3/4 지수는 이 둘 사이의 중간점으로, **희귀 단어에게 더 많은 학습 기회를 주면서도 빈출 단어를 완전히 무시하지 않는** 균형을 달성합니다. 예를 들어 빈도 0.01인 단어와 빈도 $10^{-6}$인 단어의 비율은 원래 $10^4$이지만, 3/4 지수 적용 후 약 $10^3$으로 줄어듭니다.

Where $U(w)$ is the unigram frequency of word $w$ and $Z$ is a normalization constant. Why the **3/4 exponent**? Using a uniform distribution ($U(w)^0$) would oversample rare words as negatives, while using raw frequency ($U(w)^1$) would oversample frequent words. The 3/4 exponent strikes a balance, **giving rare words more training opportunities without completely ignoring frequent words**. For example, the ratio between a word with frequency 0.01 and one with frequency $10^{-6}$ is originally $10^4$ but reduces to about $10^3$ after the 3/4 exponent.

#### Negative Sample 수 $k$의 선택 / Choosing the Number of Negative Samples $k$

$k$의 최적값은 데이터셋 크기에 따라 달라집니다:

The optimal $k$ depends on dataset size:

- **소규모 데이터셋**: $k = 5$-$20$. 데이터가 적으면 각 학습 단계에서 더 많은 negative sample을 봐야 안정적인 gradient를 얻을 수 있음
  - **Small datasets**: $k = 5$-$20$. With less data, more negative samples per step are needed for stable gradients
- **대규모 데이터셋**: $k = 2$-$5$. 데이터가 충분히 많으면 각 단어를 여러 번 반복 학습하므로 적은 수의 negative sample로도 충분
  - **Large datasets**: $k = 2$-$5$. With sufficient data, each word is encountered many times, so fewer negative samples suffice

Negative sampling은 hierarchical softmax보다 **구현이 간단하고, 속도가 빠르며, 특히 Skip-gram에서 더 나은 벡터 품질**을 제공하여 Word2Vec의 기본 학습 방법이 되었습니다.

Negative sampling became Word2Vec's default training method because it is **simpler to implement, faster, and produces better vector quality especially with Skip-gram** compared to hierarchical softmax.

---

### Part V: Learned Relationships / 학습된 관계 (Section 5)

Table 8은 논문에서 가장 인상적인 결과를 보여줍니다. 벡터 산술로 발견할 수 있는 관계의 범위가 놀랍도록 넓습니다:

Table 8 shows the paper's most impressive results. The range of relationships discoverable through vector arithmetic is remarkably broad:

- **지리**: France - Paris + Italy = Rome, Japan - Tokyo = (국가-수도 관계 / country-capital relationship)
- **크기 관계**: big - bigger + small = smaller (비교급 / comparative)
- **직업**: Einstein - scientist + Messi = midfielder
- **정치**: Sarkozy - France + Berlusconi = Italy
- **기업**: Microsoft - Windows + Google = Android, IBM - Linux + Apple = iPhone
- **문화**: Japan - sushi + Germany = bratwurst

정확도가 약 60%에 불과하다는 점(exact match 기준)은 한계이지만, 관계 벡터(relationship vector)를 하나의 예시 쌍 대신 **10개의 예시 쌍을 평균**하면 약 10% 절대 정확도 향상을 얻었다고 보고합니다.

While accuracy is only about 60% (by exact match), the authors report that **averaging 10 example pairs** for the relationship vector instead of using a single pair yields about 10% absolute accuracy improvement.

#### Worked Example: 벡터 산술의 실제 동작 / How Vector Arithmetic Actually Works

"King - Man + Woman = Queen" 연산이 구체적으로 어떻게 진행되는지 살펴봅시다. 300차원 공간에서 (여기서는 이해를 위해 3차원으로 단순화):

Let's trace how "King - Man + Woman = Queen" concretely works. In a 300-dimensional space (simplified to 3 dimensions here for illustration):

$$\mathbf{v}_{\text{King}} = [0.8, 0.6, 0.3]$$
$$\mathbf{v}_{\text{Man}} = [0.5, 0.7, 0.1]$$
$$\mathbf{v}_{\text{Woman}} = [0.5, 0.2, 0.8]$$

연산을 수행하면:

Performing the arithmetic:

$$\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} = [0.8 - 0.5 + 0.5, \; 0.6 - 0.7 + 0.2, \; 0.3 - 0.1 + 0.8] = [0.8, 0.1, 1.0]$$

이 결과 벡터와 vocabulary 내 모든 단어의 코사인 유사도를 계산하여 가장 높은 단어를 찾습니다. 만약 $\mathbf{v}_{\text{Queen}} = [0.8, 0.1, 0.9]$라면, 이 벡터와의 코사인 유사도가 가장 높을 것입니다.

This result vector is compared against all words in the vocabulary via cosine similarity to find the closest match. If $\mathbf{v}_{\text{Queen}} = [0.8, 0.1, 0.9]$, it would have the highest cosine similarity with this result vector.

#### 기하학적 해석: 왜 이것이 작동하는가 / Geometric Interpretation: Why Does This Work?

핵심은 **관계가 방향으로 인코딩된다**는 것입니다. $\mathbf{v}_{\text{Man}} - \mathbf{v}_{\text{Woman}}$이 만드는 벡터는 "성별(gender)" 방향을 나타냅니다. 마찬가지로 $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Queen}}$도 같은 "성별" 방향을 가리킵니다. 이 두 벡터가 벡터 공간에서 **거의 평행(parallel)**하다는 것은, 성별이라는 관계가 royalty 맥락과 일반적 맥락에서 동일한 방향으로 인코딩되어 있음을 의미합니다.

The key is that **relationships are encoded as directions**. The vector $\mathbf{v}_{\text{Man}} - \mathbf{v}_{\text{Woman}}$ represents the "gender" direction. Similarly, $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Queen}}$ points in the same "gender" direction. These two vectors being **approximately parallel** in vector space means gender is encoded in the same direction regardless of whether the context is royalty or general.

이것은 Rumelhart et al. (1986, 논문 #6)의 family tree 실험과 직접적으로 연결됩니다. 1986년의 소규모 실험에서 nationality, generation 같은 의미적 특성이 hidden unit에서 자연스럽게 창발했듯이, Word2Vec은 이를 **수십억 단어 규모에서 수백 차원의 벡터 공간으로 확장**한 것입니다. 차이점은 1986년에는 명시적으로 구조화된 데이터(family tree)에서 작동했지만, Word2Vec은 비구조화된 자연어 텍스트에서 동일한 현상을 관찰했다는 것입니다.

This connects directly to Rumelhart et al. (1986, Paper #6)'s family tree experiment. Just as the 1986 small-scale experiment showed semantic features like nationality and generation emerging naturally in hidden units, Word2Vec **scales this to billions of words in hundred-dimensional vector space**. The difference is that 1986 worked on explicitly structured data (family trees), while Word2Vec observes the same phenomenon in unstructured natural language text.

#### 실패 모드: 벡터 산술이 작동하지 않는 경우 / Failure Modes: When Vector Arithmetic Doesn't Work

벡터 산술이 항상 성공하는 것은 아닙니다. 주요 실패 원인은 다음과 같습니다:

Vector arithmetic doesn't always succeed. Major failure causes include:

1. **다의어(Polysemous words)**: "bank"는 "은행"과 "강둑" 두 가지 의미를 갖지만, Word2Vec은 단어당 **하나의 벡터만** 학습합니다. 결과적으로 "bank" 벡터는 두 의미의 중간점에 위치하여 어느 쪽 유추에서도 부정확해집니다
   - "bank" means both "financial institution" and "river bank," but Word2Vec learns **only one vector** per word. The "bank" vector ends up at an intermediate point between both meanings, making it inaccurate for either analogy

2. **희귀 단어(Rare words)**: 학습 데이터에서 적게 등장하는 단어는 불안정한 벡터를 갖습니다. 충분한 context를 보지 못해 벡터가 정확한 위치로 수렴하지 못합니다
   - Words appearing rarely in training data have unstable vectors. Without enough contexts, vectors fail to converge to accurate positions

3. **추상적 관계(Abstract relationships)**: "democracy is to freedom as dictatorship is to ?" 같은 추상적이고 다차원적인 관계는 단일 방향 벡터로 포착하기 어렵습니다. 벡터 산술은 **선형 관계**에 가장 잘 작동하며, 복잡한 비선형 관계에는 한계가 있습니다
   - Abstract, multi-dimensional relationships like "democracy is to freedom as dictatorship is to ?" are difficult to capture as a single direction vector. Vector arithmetic works best for **linear relationships** and has limitations with complex non-linear ones

#### 10-pair 평균화 기법: 왜 효과적인가 / The 10-Pair Averaging Technique: Why It Works

단일 예시 쌍(예: King-Queen)에서 추출한 관계 벡터에는 **노이즈**가 포함되어 있습니다. King과 Queen의 차이에는 "성별" 외에도 역사적 맥락, 코퍼스 내 등장 패턴 등의 잡음이 섞입니다. 10개의 예시 쌍 — (King, Queen), (man, woman), (brother, sister), (uncle, aunt), ... — 의 관계 벡터를 **평균**하면, 공통 요소인 "성별" 방향은 강화되고, 각 쌍에 고유한 노이즈는 상쇄됩니다. 이는 통계학의 기본 원리와 같습니다: $n$개 표본의 평균은 표준오차를 $1/\sqrt{n}$으로 줄입니다.

The relationship vector extracted from a single example pair (e.g., King-Queen) contains **noise**. The difference between King and Queen captures not only "gender" but also historical context, corpus-specific co-occurrence patterns, and other noise. **Averaging** relationship vectors from 10 example pairs — (King, Queen), (man, woman), (brother, sister), (uncle, aunt), ... — strengthens the common element ("gender" direction) while canceling pair-specific noise. This follows the basic statistical principle: the mean of $n$ samples reduces standard error by $1/\sqrt{n}$.

---

## 3. Key Takeaways / 핵심 시사점

1. **단순한 모델 + 대규모 데이터 > 복잡한 모델 + 소규모 데이터 / Simple model + large data > Complex model + small data** — Hidden layer를 제거한 log-linear 모델이 NNLM보다 더 나은 word vector를 학습할 수 있었던 이유는, 단순함 덕분에 10배 이상 많은 데이터를 처리할 수 있었기 때문. 이 원칙은 이후 GPT, BERT 등 대규모 언어 모델의 "scaling law"로 발전 / The log-linear model without a hidden layer learned better word vectors than NNLM because its simplicity allowed processing 10× more data. This principle evolved into the "scaling laws" of GPT, BERT, and modern LLMs

2. **계산 복잡도는 설계의 1급 시민 / Computational complexity is a first-class design citizen** — 저자들은 모델 설계 시 정확도가 아닌 $O = E \times T \times Q$의 $Q$를 최소화하는 것을 최우선 목표로 설정. 이는 당시로서는 급진적인 접근이었으며, "모델이 충분히 빠르다면 더 많은 데이터를 쓸 수 있고, 그것이 정확도를 높이는 가장 효과적인 방법"이라는 통찰을 반영 / The authors prioritized minimizing $Q$ in $O = E \times T \times Q$ over accuracy. This reflected the insight that "if the model is fast enough, you can use more data, and that's the most effective way to improve accuracy"

3. **Word vector의 선형 규칙성은 자연스럽게 창발 / Linear regularities of word vectors emerge naturally** — 모델에 "의미적 관계를 학습하라"고 명시적으로 지시한 것이 아님. 단순히 주변 단어를 예측하는 과정에서 벡터 공간에 구조가 자연스럽게 형성됨. 이는 neural network의 학습이 단순한 패턴 매칭이 아니라 **의미적 구조의 추출**임을 보여줌 / The model was never explicitly told to "learn semantic relationships." Structure emerged naturally from simply predicting surrounding words, showing that neural network learning extracts **semantic structure**, not just pattern matching

4. **Hierarchical softmax가 확장성의 핵심 / Hierarchical softmax is key to scalability** — 어휘 크기 $V$에 대한 복잡도를 $O(V)$에서 $O(\log V)$로 줄이는 hierarchical softmax(Huffman 트리 기반)가 없었다면, 수백만 어휘 규모에서의 학습은 불가능. Huffman 코딩을 사용하여 빈출 단어에 짧은 경로를 할당하는 것은 추가적인 효율성 / Without hierarchical softmax (Huffman tree-based) reducing vocabulary complexity from $O(V)$ to $O(\log V)$, training at million-word vocabulary scale would be impossible. Using Huffman coding to assign shorter paths to frequent words adds further efficiency

5. **CBOW와 Skip-gram은 상보적 / CBOW and Skip-gram are complementary** — CBOW는 구문적 관계에 강하고 빠르며, Skip-gram은 의미적 관계에 강하고 희귀 단어를 잘 처리. 이들은 경쟁 관계가 아니라 용도에 따라 선택할 수 있는 도구 / CBOW is stronger at syntactic relationships and faster; Skip-gram is stronger at semantic relationships and handles rare words better. They're not competitors but tools to choose based on use case

6. **차원과 데이터는 함께 확장해야 / Dimensions and data must scale together** — 차원만 높이거나 데이터만 늘리면 수확체감이 빠르게 발생. 두 가지를 동시에 확장해야 최대 효과. 이 관찰은 이후 scaling law 연구의 선구적 결과 / Increasing only dimensions or only data leads to rapid diminishing returns. Both must scale together for maximum effect — a precursor to later scaling law research

7. **Pre-trained representation의 전이 가능성 / Transferability of pre-trained representations** — Word2Vec으로 학습한 벡터를 다른 태스크(sentiment analysis, paraphrase detection, sentence completion 등)에 성공적으로 전이할 수 있었으며, 이는 "pre-training → fine-tuning" 패러다임의 시작 / Word2Vec vectors were successfully transferred to other tasks (sentiment analysis, paraphrase detection, sentence completion), marking the beginning of the "pre-training → fine-tuning" paradigm

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 모델 복잡도 비교 프레임워크 / Model Complexity Comparison Framework

모든 모델의 학습 비용은 다음과 같이 표현됩니다:

All models' training costs are expressed as:

$$O = E \times T \times Q$$

| 변수 / Variable | 의미 / Meaning | 전형적 값 / Typical Value |
|---|---|---|
| $E$ | Epoch 수 / Number of epochs | 3–50 (보통 1–3) |
| $T$ | 학습 단어 수 / Training words | Up to $10^9$ |
| $Q$ | 아키텍처별 복잡도 / Per-architecture complexity | 아래 참조 / See below |

### 4.2 아키텍처별 복잡도 $Q$ / Per-Architecture Complexity $Q$

| 모델 / Model | 복잡도 $Q$ / Complexity $Q$ | 지배적 항 / Dominant Term |
|---|---|---|
| NNLM | $N \times D + N \times D \times H + H \times V$ | $H \times V$ (with full softmax) |
| RNNLM | $H \times H + H \times V$ | $H \times V$ (with full softmax) |
| **CBOW** | $N \times D + D \times \log_2(V)$ | $N \times D$ |
| **Skip-gram** | $C \times (D + D \times \log_2(V))$ | $C \times D \times \log_2(V)$ |

- $N$: context window size (보통 10)
- $D$: word vector 차원 (50–1000)
- $H$: hidden layer size (500–1000)
- $V$: vocabulary size (up to $10^6$)
- $C$: Skip-gram의 최대 context 거리 (보통 5–10)

### 4.3 CBOW 학습 / CBOW Training

**입력**: context 단어 $\{w_{t-N/2}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+N/2}\}$의 벡터 평균

**Input**: Average of context word vectors $\{w_{t-N/2}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+N/2}\}$

$$\mathbf{h} = \frac{1}{N}\sum_{i \in \text{context}} \mathbf{v}_{w_i}$$

**출력**: hierarchical softmax를 통해 target word $w_t$의 확률 계산

**Output**: Probability of target word $w_t$ via hierarchical softmax

$$P(w_t | \text{context}) = \prod_{j=1}^{L(w_t)} \sigma\left(\text{sign}(n(w_t, j)) \cdot \mathbf{h}^T \mathbf{v}'_{n(w_t,j)}\right)$$

여기서 $L(w_t)$는 Huffman 트리에서 $w_t$까지의 경로 길이, $n(w_t, j)$는 경로상의 $j$번째 내부 노드, $\sigma$는 시그모이드 함수입니다.

Where $L(w_t)$ is the path length to $w_t$ in the Huffman tree, $n(w_t, j)$ is the $j$-th internal node on the path, and $\sigma$ is the sigmoid function.

### 4.4 Skip-gram 학습 / Skip-gram Training

**입력**: 중심 단어 $w_t$의 벡터 $\mathbf{v}_{w_t}$

**Input**: Center word $w_t$'s vector $\mathbf{v}_{w_t}$

**출력**: context 범위 내의 각 단어 $w_{t+j}$ ($-C \leq j \leq C, j \neq 0$)에 대한 확률

**Output**: Probability for each word $w_{t+j}$ ($-C \leq j \leq C, j \neq 0$) within context range

**학습 목표 / Training Objective**:

$$\max \frac{1}{T}\sum_{t=1}^{T}\sum_{-C \leq j \leq C, j \neq 0} \log P(w_{t+j} | w_t)$$

### 4.5 단어 유추 벡터 산술 / Word Analogy Vector Arithmetic

"$a$ is to $b$ as $c$ is to $d$" 관계에서 $d$를 찾는 방법:

Finding $d$ in the relationship "$a$ is to $b$ as $c$ is to $d$":

$$d = \arg\max_{w \in V \setminus \{a,b,c\}} \cos\left(\mathbf{v}_w, \; \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c\right)$$

코사인 유사도:

Cosine similarity:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}| \cdot |\mathbf{v}|}$$

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1986        1997         2003         2008          2010         2013         2013          2014
  |           |            |            |             |            |            |             |
  ▼           ▼            ▼            ▼             ▼            ▼            ▼             ▼
Backprop   LSTM        NNLM      Collobert      RNNLM      Word2Vec    Word2Vec v2      GloVe
(#6)       (#9)      (Bengio)    & Weston     (Mikolov)   ★ THIS ★    (neg sampling)  (Pennington)
  |           |            |            |             |            |            |             |
분산표현    시퀀스       최초의       NLP 통합      순환 신경망    log-linear   구/문장 수준    행렬 분해
학습의     모델링의    neural LM   neural arch   language     word vector  word vector     관점의
기반       기반                    for NLP       model                                  word vec

          2017            2018            2018           2019+
            |               |               |              |
            ▼               ▼               ▼              ▼
        Transformer       ELMo            BERT          GPT-2/3
        (Vaswani)       (Peters)        (Devlin)        (OpenAI)
            |               |               |              |
        self-attention    문맥화된         양방향          대규모
        메커니즘으로      word vector     pre-training    생성형
        RNN 대체        (다의어 해결)    + fine-tuning    언어 모델
```

#### Static에서 Contextual Embedding으로의 전환 / The Shift from Static to Contextual Embeddings

Word2Vec은 각 단어에 **하나의 고정된 벡터(static embedding)**를 할당합니다. 이는 강력하지만 근본적인 한계가 있습니다: "bank"가 "river bank"에서든 "bank account"에서든 동일한 벡터를 갖습니다.

Word2Vec assigns **one fixed vector (static embedding)** per word. This is powerful but has a fundamental limitation: "bank" has the same vector whether in "river bank" or "bank account."

**ELMo(2018)**는 이 문제를 해결한 첫 번째 주요 모델입니다. 양방향 LSTM을 사용하여 **문맥에 따라 달라지는 벡터(contextualized embedding)**를 생성합니다. "bank" 주변에 "river"가 있으면 "강둑" 의미의 벡터가, "account"가 있으면 "은행" 의미의 벡터가 생성됩니다.

**ELMo (2018)** was the first major model to solve this problem. Using bidirectional LSTMs, it generates **context-dependent vectors (contextualized embeddings)**. If "river" surrounds "bank," a "river bank" vector is produced; if "account" surrounds it, a "financial institution" vector is produced.

**Transformer(2017)**는 RNN/LSTM의 순차적 처리를 **self-attention 메커니즘**으로 대체하여 병렬 처리를 가능하게 했습니다. 이는 단순히 구조적 혁신이 아니라, Word2Vec이 보여준 "더 단순한 모델 + 더 많은 데이터" 원칙의 연장선에 있습니다 — attention은 RNN보다 단순하면서도 더 효과적으로 long-range dependency를 포착합니다.

**Transformer (2017)** replaced RNN/LSTM's sequential processing with **self-attention mechanism**, enabling parallelization. This is not merely a structural innovation but an extension of Word2Vec's principle of "simpler model + more data" — attention is simpler than RNN yet captures long-range dependencies more effectively.

**BERT(2018)**와 이후의 GPT 시리즈는 Word2Vec이 개척한 **"pre-training → downstream task" 패러다임**을 극대화했습니다. Word2Vec이 단어 수준의 pre-trained representation을 제공했다면, BERT/GPT는 문장/문서 수준의 pre-trained representation을 제공하며, 이는 NLP의 거의 모든 태스크에서 패러다임 전환을 일으켰습니다.

**BERT (2018)** and subsequent GPT series maximized the **"pre-training then downstream task" paradigm** pioneered by Word2Vec. Where Word2Vec provided word-level pre-trained representations, BERT/GPT provide sentence/document-level pre-trained representations, causing a paradigm shift across nearly all NLP tasks.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #6 Rumelhart, Hinton, Williams (1986) — Backpropagation | Word2Vec은 backpropagation과 SGD로 학습됨. 또한 distributed representation의 개념적 기원 / Word2Vec is trained with backpropagation and SGD. Also the conceptual origin of distributed representations | 직접적 선행 연구. 1986년에 제시된 "분산 표현" 아이디어가 27년 후 대규모로 실현 / Direct predecessor. The "distributed representation" idea from 1986 realized at scale 27 years later |
| #9 Hochreiter & Schmidhuber (1997) — LSTM | RNNLM 비교 대상의 기반 아키텍처. LSTM/RNN 기반 language model이 Word2Vec의 비교 baseline / Foundation architecture for RNNLM comparison baselines | RNNLM은 hidden layer의 recurrent connection 때문에 $H^2$ 복잡도를 가짐 — Word2Vec이 이를 극복 / RNNLM has $H^2$ complexity due to recurrent connections — Word2Vec overcomes this |
| Bengio et al. (2003) — NNLM | Word2Vec의 가장 직접적인 선행 연구. NNLM에서 hidden layer를 제거하여 CBOW를 도출 / Most direct predecessor. CBOW derived by removing hidden layer from NNLM | NNLM의 projection layer 개념이 Word2Vec의 embedding layer로 직결. NNLM의 계산 병목($H \times V$)을 분석하여 해결 / NNLM's projection layer concept directly leads to Word2Vec's embedding layer. NNLM's bottleneck ($H \times V$) analyzed and solved |
| Collobert & Weston (2008) — NLP from Scratch | Word vector를 다양한 NLP 태스크에 전이하는 아이디어의 선구자 / Pioneer of transferring word vectors to various NLP tasks | Word2Vec은 이 아이디어를 더 효율적이고 범용적인 형태로 실현 / Word2Vec realized this idea in a more efficient and universal form |
| Mikolov et al. (2013) — Distributed Representations of Words and Phrases | 이 논문의 직접적 후속 연구. Negative sampling, subsampling 등 Word2Vec의 핵심 학습 기법 도입 / Direct follow-up. Introduces negative sampling, subsampling — key training techniques for Word2Vec | Hierarchical softmax의 대안인 negative sampling을 제시하여 Skip-gram을 더욱 효율적으로 / Presents negative sampling as an alternative to hierarchical softmax, making Skip-gram even more efficient |
| Pennington et al. (2014) — GloVe | Word2Vec의 Skip-gram이 암묵적으로 단어 동시출현 행렬을 분해한다는 것을 보여주고, 이를 명시적 행렬 분해로 재구성 / Showed that Skip-gram implicitly factorizes word co-occurrence matrix, and reformulated it as explicit matrix factorization | Word2Vec과 전통적인 count-based 방법(LSA 등)의 이론적 연결을 확립 / Established theoretical connection between Word2Vec and traditional count-based methods (LSA, etc.) |

---

## 7. References / 참고문헌

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
- Bengio, Y., Ducharme, R., Vincent, P. (2003). "A Neural Probabilistic Language Model." JMLR, 3:1137-1155.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). "Distributed Representations of Words and Phrases and their Compositionality." NIPS 2013.
- Collobert, R. & Weston, J. (2008). "A Unified Architecture for Natural Language Processing." ICML 2008.
- Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation." EMNLP 2014.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning internal representations by error-propagating errors." Nature, 323:533-536.
- Mikolov, T. (2012). "Statistical Language Models based on Neural Networks." PhD thesis, Brno University of Technology.
- Morin, F. & Bengio, Y. (2005). "Hierarchical Probabilistic Neural Network Language Model." AISTATS 2005.
- Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017.
- Peters, M. et al. (2018). "Deep contextualized word representations." NAACL 2018.
- Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
