---
title: "Scaling Laws for Neural Language Models"
authors: [Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei]
year: 2020
journal: "arXiv preprint"
doi: "arXiv:2001.08361"
topic: Artificial_Intelligence
tags: [scaling-laws, language-model, transformer, compute-efficient-training, power-law, GPT, sample-efficiency]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 39. Scaling Laws for Neural Language Models / 신경망 언어 모델의 스케일링 법칙

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 Transformer 기반 자기회귀 언어 모델의 **cross-entropy 손실이 모델 크기 $N$, 데이터셋 크기 $D$, 학습 컴퓨트 $C$의 단순한 멱법칙(power law) 함수**임을 7자리 이상의 규모에 걸쳐 경험적으로 입증합니다. 세 가지 핵심 멱법칙:

$$L(N) = (N_c/N)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$
$$L(D) = (D_c/D)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}$$
$$L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \quad C_c^{\min} \approx 3.1 \times 10^8 \text{ PF-days}$$

이 법칙들은 모델 모양(깊이, 너비, attention head 수)에는 거의 무관하며, 두 변수의 동시 의존성은 결합식 $L(N, D) = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$로 기술됩니다. **가장 영향력 있는 결론**: 고정된 컴퓨트 예산에서 손실을 최소화하려면 매우 큰 모델을 적당한 데이터에 훈련시키되 수렴 훨씬 전에 멈춰야 합니다 — 컴퓨트 10배당 모델 5배(≈$C^{0.73}$), 배치 1.7배($C^{0.24}$), step 1.07배($C^{0.03}$). 이는 GPT-3(2020) 175B 매개변수 모델의 직접적 동기가 되었으며, 현대 LLM 시대의 정량적 출발점이 되었습니다. **주의**: 본 논문의 "모델 크기 > 데이터" 결론은 이후 Chinchilla(Hoffmann et al., 2022)에 의해 학습률 스케줄 결함과 작은 모델에 불리한 설정 때문이라며 **약 1:1 비율로 함께 키워야 한다**는 정정이 제시됩니다.

**English**
This paper empirically demonstrates that the **cross-entropy loss of Transformer autoregressive language models is a simple power-law function of model size $N$, dataset size $D$, and training compute $C$**, with trends spanning more than seven orders of magnitude. The three central power laws are:

$$L(N) = (N_c/N)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$
$$L(D) = (D_c/D)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}$$
$$L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \quad C_c^{\min} \approx 3.1 \times 10^8 \text{ PF-days}$$

These laws are nearly independent of model shape (depth, width, attention heads), and the joint dependence is captured by $L(N, D) = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$. **The most consequential conclusion**: under a fixed compute budget, to minimize loss one should train very large models on a modest amount of data and stop well before convergence — for every 10× compute increase, model size grows 5× ($\propto C^{0.73}$), batch 1.7× ($\propto C^{0.24}$), and steps barely 1.07× ($\propto C^{0.03}$). This directly motivated GPT-3 (175B parameters, 2020) and inaugurated the quantitative era of modern LLMs. **Caveat**: this paper's "model > data" conclusion was later challenged by Chinchilla (Hoffmann et al., 2022), which attributed the bias to a learning-rate schedule that disadvantaged small models and showed that **model and data should grow roughly 1:1**, not 5:1.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Summary (§1) / 서론과 요약

**한국어**
저자들은 언어가 AI 연구에 자연스러운 영역(reasoning을 효율적으로 표현하고 평가)이며, 인터넷의 막대한 텍스트는 비지도 학습(unsupervised learning) 데이터를 풍부하게 제공한다는 점에서 출발합니다. 이미 GPT-2 등 SOTA 모델이 인간 수준에 근접하는 분야가 등장했습니다.

핵심 질문: **언어 모델 성능이 모델 아키텍처, 모델 크기, 컴퓨트 파워, 데이터 가용성에 어떻게 의존하는가?** 본 연구는 Transformer 아키텍처에 초점을 맞춰 7자리 이상의 규모에서 정밀한 멱법칙을 발견합니다.

**§1.1 Summary of key findings (Transformer 언어 모델)**:

1. **Performance depends strongly on scale, weakly on shape / 성능은 규모에는 강하게, 모양에는 약하게 의존**
   세 요인 — 매개변수 수 $N$ (임베딩 제외), 데이터셋 크기 $D$, 학습 컴퓨트 $C$ — 이 결정적. 깊이 vs 너비 같은 다른 하이퍼파라미터는 합리적 범위 내에서 거의 영향 없음.

2. **Smooth power laws / 매끄러운 멱법칙**
   $N, D, C$ 각각에 대해 다른 두 요인이 병목이 아닐 때 6자리 이상 멱법칙 관계가 유지됨. 상한에서도 이탈 없음.

3. **Universality of overfitting / 과적합의 보편성**
   $N$과 $D$를 함께 키우면 예측 가능하게 향상되나, 한쪽만 키우면 수익 체감. 페널티는 비율 $N^{0.74}/D$에 의존 — **모델 8배 키우면 데이터는 5배만 늘리면 됨**.

4. **Universality of training / 학습의 보편성**
   학습 곡선이 모델 크기에 거의 무관한 매개변수의 멱법칙을 따름 — 학습 초반만 보고도 장기 손실 예측 가능.

5. **Transfer improves with test performance / 전이는 성능에 비례**
   학습 분포와 다른 텍스트로 평가해도 학습 검증 손실과 강한 상관 — 분포 차이는 일정 오프셋만 추가.

6. **Sample efficiency / 표본 효율성**
   큰 모델은 같은 성능에 더 적은 샘플과 적은 step으로 도달.

7. **Convergence is inefficient / 수렴까지 학습은 비효율적**
   고정 컴퓨트 예산에서 최적 성능은 **매우 큰 모델을 수렴 훨씬 전에 멈추는 것**으로 달성. 데이터 요구량은 $D \sim C^{0.27}$로 매우 천천히 증가.

8. **Optimal batch size / 최적 배치 크기**
   손실의 멱함수로만 결정되며 (모델 크기와 무관), gradient noise scale로 결정 가능. 가장 큰 모델에서는 수렴 시 약 1–2M 토큰.

**§1.2 Summary of Scaling Laws** (식 1.1, 1.2, 1.3 — 본 논문의 결론):
- $L(N) = (N_c/N)^{\alpha_N}$, $\alpha_N \sim 0.076$, $N_c \sim 8.8 \times 10^{13}$ (수렴까지 학습, 충분한 데이터)
- $L(D) = (D_c/D)^{\alpha_D}$, $\alpha_D \sim 0.095$, $D_c \sim 5.4 \times 10^{13}$ (충분히 큰 모델, early stop)
- $L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}$, $\alpha_C^{\min} \sim 0.050$, $C_c^{\min} \sim 3.1 \times 10^8$ PF-days

정확한 수치는 토큰화에 의존하므로 근본적 의미는 없습니다 — 지수만 보편적입니다. 임계 배치 크기:
$$B_{\text{crit}}(L) = \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

**English**
The authors begin from the observation that language is a natural domain for AI (most reasoning is expressible/evaluable in language) and that the internet provides abundant unsupervised training data. SOTA models like GPT-2 already approach human-level performance in some tasks.

The key question: **how does language modeling performance depend on architecture, model size, compute, and data?** Focusing on the Transformer, the authors discover precise power laws spanning >7 orders of magnitude.

The eight headline findings (above) and the three power laws (Eqs. 1.1–1.3) are summarized in Section 1.2. Numerical values of $N_c, D_c, C_c$ depend on the tokenizer and hence have no fundamental meaning — only the exponents are universal. Critical batch size also follows a power law in the loss: $B_{\text{crit}}(L) = B_*/L^{1/\alpha_B}$, with $B_* \approx 2 \times 10^8$ tokens and $\alpha_B \approx 0.21$. The critical batch size approximately doubles for every 13% decrease in loss.

### Part II: Background and Methods (§2) / 배경과 방법론

**한국어**

**§2.1 Parameter and Compute Scaling of Transformers**
- 모델 매개변수화: $n_{\text{layer}}$ (층 수), $d_{\text{model}}$ (residual stream 차원), $d_{\text{ff}}$ (FFN 중간 차원), $d_{\text{attn}}$ (attention 출력 차원), $n_{\text{heads}}$, $n_{\text{ctx}} = 1024$.
- **비임베딩 매개변수 수** (이 논문의 $N$):
$$N \approx 2 d_{\text{model}} n_{\text{layer}} (2 d_{\text{attn}} + d_{\text{ff}}) = 12 n_{\text{layer}} d_{\text{model}}^2$$
표준 설정 $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$일 때.
- **Forward pass 컴퓨트**:
$$C_{\text{forward}} \approx 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$$
2배는 multiply-accumulate 연산. $d_{\text{model}} \gg n_{\text{ctx}}/12$이면 컨텍스트 종속 항은 무시 가능.
- **총 학습 컴퓨트** (forward + backward $\approx 2\times$):
$$C \approx 6 N B S \quad [\text{FLOPs per training token}]$$
$B$ = 배치 크기, $S$ = 매개변수 업데이트 수.

**§2.2 Training Procedures**
- Adam optimizer, $2.5 \times 10^5$ steps, batch 512 sequences × 1024 tokens.
- 1B+ 매개변수 모델은 메모리 제약으로 Adafactor 사용.
- Learning rate: 3000-step linear warmup + cosine decay to zero. 수렴 시 결과는 LR 스케줄에 거의 무관.

**§2.3 Datasets**
- **WebText2**: 원래 WebText(GPT-2)의 확장. Reddit 링크 (3+ karma), 2017년 12월~2018년 10월. 20.3M 문서, 96GB, $1.62 \times 10^{10}$ words. Reversible BPE tokenizer로 $2.29 \times 10^{10}$ tokens. $6.6 \times 10^8$ 토큰 테스트 셋.
- **Transfer 평가**: Books Corpus, Common Crawl, Wikipedia, Internet Books.

**English**

**§2.1 Parameter and Compute Scaling**
Non-embedding parameter count: $N \approx 12 n_{\text{layer}} d_{\text{model}}^2$ when $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$. Forward FLOPs per token: $C_{\text{forward}} \approx 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$. Training compute (forward+backward): $C \approx 6 N B S$.

**§2.2 Training**
Adam, $2.5 \times 10^5$ steps, batch 512 × 1024. Adafactor for 1B+ models. LR: 3000-step warmup + cosine decay. Results at convergence are LR-schedule-insensitive.

**§2.3 Datasets**
WebText2: $2.29 \times 10^{10}$ training tokens via BPE. Test on Books, Common Crawl, Wikipedia, Internet Books for transfer.

### Part III: Empirical Results and Basic Power Laws (§3) / 경험적 결과와 기본 멱법칙

**한국어**

**모델 변동 범위**:
- 모델 크기: 768 ~ 1.5B (비임베딩) 매개변수
- 데이터셋: 22M ~ 23B 토큰
- Shape: depth, width, heads, FFN 차원
- Context length: 주로 1024
- Batch size: 주로 $2^{19}$, 일부 변동

**§3.1 Approximate Transformer Shape and Hyperparameter Independence**

Figure 5 (가장 결정적인 결과): 비임베딩 매개변수 $N$ 고정 시
- **Feed-forward ratio** $d_{\text{ff}}/d_{\text{model}}$: $10^0$~$10^1$ 범위에서 1% 손실 변동
- **Aspect ratio** $d_{\text{model}}/n_{\text{layer}}$: 40배 변화에 손실 1% 미만 변동 — $(n_{\text{layer}}, d_{\text{model}}) = (6, 4288)$이 표준 $(48, 1600)$ 모델의 3% 이내
- **Attention head dimension** $d_{\text{model}}/n_{\text{heads}}$: 거의 영향 없음

**§3.2 Performance with Non-Embedding Parameter Count $N$**

Figure 6 좌: 임베딩 포함 시 깊이가 매우 다른 모델들이 다른 곡선을 그림.
Figure 6 우: 임베딩 제외 시 모든 모델이 단일 멱법칙으로 수렴 — **임베딩 행렬은 작게 만들어도 손실에 영향 없음**.

이는 식 (3.1)/(1.1)을 정당화: $L(N) \approx (N_c/N)^{\alpha_N}$.

**§3.2.1 Comparing to LSTMs and Universal Transformers**
Figure 7 좌: LSTM은 작은 모델에서는 Transformer와 비슷하나, 큰 모델에서 Transformer가 우월.
Figure 7 우: LSTM은 100 토큰 이후 per-token 손실이 plateau — long context를 활용 못 함. Transformer는 전체 컨텍스트에 걸쳐 개선.

**§3.2.2 Generalization Among Data Distributions**
Figure 8 좌: WebText2 학습 모델을 다른 분포에서 평가 — 모두 부드러운 멱법칙. 일정한 오프셋만 더해짐.
Figure 8 우: **Transfer 손실은 in-distribution 검증 손실에만 의존**, 학습 phase에 무관.

**§3.3 Performance with Dataset Size and Compute**

데이터 크기에 대한 적합:
$$L(D) \approx (D_c/D)^{\alpha_D}, \quad \alpha_D \approx 0.095$$
(36-layer, 1280-dim 모델을 데이터 부분집합에 학습하고 손실 정체 시 중단.)

컴퓨트에 대한 (naive, fixed-batch) 적합:
$$L(C) \approx (C_c/C)^{\alpha_C}, \quad \alpha_C \approx 0.057$$
(이는 $L(C_{\min})$의 $\alpha_C^{\min} \approx 0.050$과 다름 — 후자가 진정한 최적.)

**English**

Section 3 documents: (i) shape independence (Figure 5: aspect ratio varies 40× with negligible effect; FF ratio and head dimension also nearly invariant). (ii) Power law in $N$ (non-embedding parameters) — embedding-inclusive plots are confounded by depth-dependent embedding sizes. (iii) Transformers asymptotically dominate LSTMs because LSTM per-token loss plateaus after ~100 tokens (Figure 7). (iv) Transfer loss to other text distributions tracks training-distribution loss with a constant offset (Figure 8). (v) Empirical $L(D)$ and $L(C)$ fits.

### Part IV: Charting the Infinite Data Limit and Overfitting (§4) / 무한 데이터 한계와 과적합

**한국어**

**§4.1 Proposed $L(N, D)$ Equation**

선택한 매개변수화 (식 4.1 = 식 1.5):
$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

**유도 원리** (3가지):
1. **토큰화 불변성**: 어휘 변경은 손실에 전체적 rescale만 가해야 함 — $N_c, D_c$의 rescale로 흡수 가능.
2. **한계 일치**: $N \to \infty$이면 $L(D)$, $D \to \infty$이면 $L(N)$로 환원되어야 함.
3. **$D \to \infty$에서 해석성**: $1/D$의 정수 거듭제곱으로 series expansion 가능해야 함.

세 번째 원리는 더 추측적이지만, 과적합이 데이터셋의 분산이나 신호 대 잡음비와 관련되며 이는 $1/D$로 스케일링한다는 직관에 기반합니다.

**§4.2 Results**

10% dropout으로 정규화, early stopping. **Table 2 적합값**:
| Parameter | $\alpha_N$ | $\alpha_D$ | $N_c$ | $D_c$ |
|---|---|---|---|---|
| Value | 0.076 | 0.103 | $6.4 \times 10^{13}$ | $1.8 \times 10^{13}$ |

(이 값은 $L(\infty, D)$나 $L(N, \infty)$만으로 적합한 §3 값과 약간 다름 — 동시 적합이라 자연스러움.)

**과적합 정량화**:
$$\delta L(N, D) \equiv \frac{L(N, D)}{L(N, \infty)} - 1 \approx \left(1 + \left(\frac{N}{N_c}\right)^{\alpha_N/\alpha_D} \frac{D_c}{D}\right)^{\alpha_D} - 1$$

Figure 9 우: $\delta L$이 단일 변수 $N^{\alpha_N/\alpha_D}/D = N^{0.74}/D$의 함수로 collapse — **universal overfitting curve**.

**과적합 회피 조건** (random seed 변동 0.02 이내):
$$D \gtrsim (5 \times 10^3) \cdot N^{0.74}$$

이는 **모델 크기를 sublinearly 키워야 한다**는 의미: $N$ 8배 → $D$ $8^{0.74} \approx 5$배.

**English**

The proposed $L(N,D)$ form (Eq. 4.1) is justified by three principles: (1) tokenization-induced rescaling, (2) limiting reduction to $L(N)$ and $L(D)$, (3) analyticity in $1/D$ at $D=\infty$. Fitted values give $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c = 6.4\times 10^{13}$, $D_c = 1.8\times 10^{13}$. The overfitting penalty $\delta L$ collapses onto a single curve in the variable $N^{0.74}/D$, giving the rule $D \gtrsim 5\times 10^3 \cdot N^{0.74}$ to keep overfitting below random-seed variance.

### Part V: Scaling Laws with Model Size and Training Time (§5) / 모델 크기와 학습 시간

**한국어**

**§5.1 Adjustment for Training at $B_{\text{crit}}(L)$**

McCandlish et al. (2018)의 critical batch size 이론을 적용. 어떤 손실 $L$에 도달하기 위한 step 수 $S$와 처리 데이터 $E = BS$는:
$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1$$

이는 critical batch size를 정의:
$$B_{\text{crit}}(L) \equiv \frac{E_{\min}}{S_{\min}}$$

Figure 10: 여러 모델 크기에서 $B_{\text{crit}}$이 손실에만 의존 (모델 크기 무관). 멱법칙:
$$B_{\text{crit}}(L) \approx \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

조정된 정의:
$$S_{\min}(S) \equiv \frac{S}{1 + B_{\text{crit}}(L)/B} \quad (\text{at } B \gg B_{\text{crit}})$$
$$C_{\min}(C) \equiv \frac{C}{1 + B/B_{\text{crit}}(L)} \quad (\text{at } B \ll B_{\text{crit}})$$

**§5.2 Results for $L(N, S_{\min})$**

식 (5.6):
$$L(N, S_{\min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}$$

Table 3 적합값: $\alpha_N = 0.077$, $\alpha_S = 0.76$, $N_c = 6.5 \times 10^{13}$, $S_c = 2.1 \times 10^3$.

이는 학습 곡선이 모델 크기와 거의 무관한 매개변수의 멱법칙 형태를 가짐을 보입니다 (Figure 4 우, Figure 11).

**§5.3 Lower Bound on Early Stopping Step**

$L(N, S_{\min})$로부터 데이터가 제한적일 때 early stopping이 일어나야 할 step의 하한:
$$S_{\text{stop}}(N, D) \gtrsim \frac{S_c}{[L(N, D) - L(N, \infty)]^{1/\alpha_S}}$$

**English**

§5 introduces the critical batch size theory (McCandlish et al.). $B_{\text{crit}}(L) = B_*/L^{1/\alpha_B}$ depends only on loss, not on model size (Figure 10). Adjusted $S_{\min}$ and $C_{\min}$ measure steps and compute "at the optimal batch size". The learning-curve fit $L(N, S_{\min})$ uses $\alpha_S \approx 0.76$, $S_c \approx 2.1 \times 10^3$. From this, early-stopping is bounded below by Eq. (5.7).

### Part VI: Optimal Allocation of the Compute Budget (§6) / 최적 컴퓨트 예산 배분

**한국어**

**§6.1 Optimal Performance and Allocations**

식 (5.5)로 $C_{\min}$을 정의하고 $L(C_{\min})$을 적합 (Figure 13):
$$L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050$$

**최적 모델 크기** (Figure 14, 식 6.1):
$$N(C_{\min}) \propto C_{\min}^{0.73}$$

$C_{\min} = 6 N B_{\text{crit}} S$이고 $B_{\text{crit}} \propto L^{-4.8} \propto C_{\min}^{0.24}$이므로:
$$S_{\min} \propto C_{\min}^{0.03}$$

즉, **최적 학습 step 수는 컴퓨트와 거의 무관**! (지수가 0과 거의 같음.)

**핵심 통찰** (Figure 12, 3): 컴퓨트 10억 배 증가 시
- **모델 크기**: 1,000,000배 이상 증가 ($10^9 \times 0.73 \approx 6 \times 10^6$배)
- **배치 크기**: 100배 증가
- **Serial steps**: 10배 미만 증가
- **데이터**: 비교적 천천히 (재사용 거의 없음)

**§6.2 Predictions from $L(N, S_{\min})$**

이론적 예측:
$$\alpha_C^{\min} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx 0.054$$

이는 Figure 13의 경험값 0.050과 잘 일치.

**§6.3 Contradictions and a Conjecture**

**중요한 자기 모순**:
- $L(C_{\min}) \propto C_{\min}^{-0.050}$: 매우 천천히 감소
- 데이터 증가율: $D \propto N^{0.74} \propto C_{\min}^{0.54}$ (overfitting 회피)
- 한 epoch 학습 시: $D(C_{\min}) = 2C_{\min}/(6 N(C_{\min})) \approx 4 \times 10^{10} (C_{\min}/\text{PF-Day})^{0.26}$
- 데이터 부족 시 손실은 $L(D) \propto D^{-0.095}$ → $L \propto C_{\min}^{-0.03}$

이 두 예측은 결국 교차합니다 (Figure 15):
$$C^* \sim 10^4 \text{ PF-Days}, \quad N^* \sim 10^{12}, \quad D^* \sim 10^{12} \text{ tokens}, \quad L^* \sim 1.7 \text{ nats/token}$$

**해석**: 이 점에서 멱법칙이 깨질 것이 분명. $L^*$는 자연어의 entropy-per-token 추정치일 가능성. 이 지점 너머에서는 더 이상 "compute-efficient"하게 큰 모델을 키우는 것이 불가능.

**역사적 후속**: 이 conjecture는 **Chinchilla(Hoffmann et al., 2022)**가 부분적으로 검증/반박. Chinchilla는 본 논문의 LR 스케줄이 작은 모델에 불리해 $\alpha_N$, $\alpha_D$가 편향됐다고 주장. 새로 적합한 결과 $N \propto C^{0.5}$, $D \propto C^{0.5}$ — 모델과 데이터를 **거의 1:1로 키워야** 한다는 결론.

**English**

§6.1 derives the compute-efficient frontier: $N(C_{\min}) \propto C_{\min}^{0.73}$, $S_{\min} \propto C_{\min}^{0.03}$. §6.2 predicts $\alpha_C^{\min} = 1/(1/\alpha_S + 1/\alpha_B + 1/\alpha_N) \approx 0.054$, matching the empirical 0.050. §6.3 identifies a self-contradiction: $L(C_{\min})$ decreases as $C^{-0.050}$ but the dataset can grow at most one epoch's worth, predicting $L \propto C^{-0.03}$ once data-bottlenecked. The crossover at $C^* \sim 10^4$ PF-Days, $N^* \sim 10^{12}$, $D^* \sim 10^{12}$, $L^* \sim 1.7$ nats/token is conjectured to mark the breakdown of these laws and possibly the entropy-per-token of natural language. Hoffmann et al. (2022, "Chinchilla") later argued that Kaplan's LR schedule biased the exponents and proposed a 1:1 model:data scaling.

### Part VII: Related Work and Discussion (§7–§8) / 관련 연구 및 논의

**한국어**

**Related Work (§7)**: 멱법칙은 다양한 곳에서 발생 — 밀도 추정, random forest 모델 등. 이전 연구로 BB01, Goo01, HNA17, HAD19. 특히 HNA17은 dataset 크기와 모델 크기 사이 super-linear 스케일링을 발견한 반면 본 논문은 sub-linear ($N^{0.74}$ vs $D$). EfficientNet (TL19)은 이미지 모델에서 depth/width의 지수적 스케일링을 권장하지만, 언어 모델에서는 너비/깊이 비율이 크게 영향 없음. RRBS19b는 본 논문과 비슷한 ansatz로 다양한 데이터셋의 스케일링을 연구.

**Discussion (§8)**:
- 본 결과는 단순한 관찰을 넘어 예측적 프레임워크를 제공.
- 결과를 ideal gas law의 유사물로 해석 가능 — microscopic 세부와 무관한 macroscopic 통일성.
- 다른 generative 작업(이미지, 오디오, 비디오, 멀티모달)에도 확장 가능성. 실제로 후속 연구로 검증됨.
- 큰 모델의 sample efficiency는 sparsity, model parallelism, growing networks 같은 기술과 결합 시 더 가속될 수 있음.

**English**

§7 relates the work to density estimation power laws, EfficientNet (which favors exponential depth/width scaling for images, in contrast to language model insensitivity to shape), Hestness et al. (super-linear data scaling — opposite to this paper's sub-linear), and concurrent Rosenfeld et al. (similar ansatz). §8 frames the laws as analogous to the ideal gas law (universal macroscopic relations independent of micro-details) and conjectures extension to images, audio, video, and other generative settings — later borne out by follow-up work.

### Part VIII: Appendices Summary / 부록 요약

**한국어**

**Appendix A (Summary of Power Laws)**: Table 4–6은 모든 멱법칙을 한 페이지로 정리.
- $\alpha_N = 0.076$, $\alpha_D = 0.095$, $\alpha_C = 0.057$, $\alpha_C^{\min} = 0.050$, $\alpha_B = 0.21$, $\alpha_S = 0.76$
- 컴퓨트 효율적 학습: $N_{\text{opt}} \propto C_{\min}^{0.73}$, $B \propto C_{\min}^{0.24}$, $S_{\min} \propto C_{\min}^{0.03}$, $D \propto C_{\min}^{0.27}$

**Appendix B (Empirical Model of Compute-Efficient Frontier)**: $L(N, S)$로부터 모든 컴퓨트 효율 결과를 미적분으로 유도.

**Appendix C (Caveats)**: 결론의 한계 — LR 스케줄, 작은 모델 학습 부족, 토큰화 의존성 등.

**Appendix D (Supplemental Figures)**: 추가 그림 — context length 스케일링, recurrent Transformer, training curves의 상세 분석.

**English**

Appendix A tabulates all exponents and prefactors. Appendix B derives all compute-efficient relations analytically from $L(N, S)$. Appendix C lists caveats including LR-schedule sensitivity and insufficient small-model training (the very issue Chinchilla later flagged). Appendix D provides supplemental figures.

---

## 3. Key Takeaways / 핵심 시사점

1. **멱법칙은 보편적이다 / Power laws are universal**
   $L \propto N^{-\alpha_N}, D^{-\alpha_D}, C^{-\alpha_C^{\min}}$가 7자리 이상의 규모에서 유지. 이탈 신호 없음 (자연어의 entropy 한계 전까지). 이는 **딥러닝이 정량적 과학으로 다뤄질 수 있음**을 의미합니다. / Loss obeys $L \propto N^{-\alpha_N}, D^{-\alpha_D}, C^{-\alpha_C^{\min}}$ over 7+ orders of magnitude with no deviation. This makes deep learning quantitatively predictable.

2. **모양은 거의 무관, 규모가 전부 / Shape barely matters; scale is everything**
   Aspect ratio 40배 변화에도 손실 1% 미만. 이는 architecture engineering보다 **scale up이 더 효과적**임을 시사 (LLM 시대의 핵심 메시지). / Aspect ratio can vary 40× with <1% loss change. This implies scaling beats architecture tuning — a defining message of the LLM era.

3. **고정 컴퓨트 예산에서는 큰 모델 + 적은 데이터 + 조기 종료 / For fixed compute: large models, modest data, early stop**
   $N \propto C^{0.73}$, $S \propto C^{0.03}$: 컴퓨트 10배당 모델 5배, step 1.07배. **수렴은 비효율**. 이것이 GPT-3 175B의 직접적 동기. / $N \propto C^{0.73}$, $S \propto C^{0.03}$ — for 10× compute, model grows 5×, steps barely 1.07×. Convergence is inefficient. This directly motivated GPT-3's 175B.

4. **큰 모델은 표본 효율적 / Larger models are more sample-efficient**
   Figure 2: 큰 모델이 같은 손실에 도달하는 데 더 적은 step과 데이터를 사용. 이는 "data scarcity가 문제이므로 작은 모델"이라는 직관과 정반대. / Figure 2: larger models reach a target loss with fewer steps and data — counter to the intuition "use small models when data is scarce".

5. **Overfitting의 보편 곡선: $\delta L \sim f(N^{0.74}/D)$ / Universal overfitting curve**
   모델 8배 키우면 데이터 5배만 늘리면 됨. 모델과 데이터를 동시에 sublinearly 함께 키워야. / Scaling model 8× requires only 5× more data. Both should grow, but sub-linearly together.

6. **Chinchilla 정정: 1:1 비율이 더 정확 / Chinchilla correction: 1:1 ratio is more accurate**
   본 논문의 "모델 > 데이터" 결론은 LR 스케줄 결함과 작은 모델의 학습 부족 때문. Chinchilla(Hoffmann 2022)는 $N \propto C^{0.5}$, $D \propto C^{0.5}$로 정정. 결과: GPT-3는 너무 작은 데이터에 학습됐고, Chinchilla 70B가 GPT-3 175B를 능가. / The "model > data" conclusion was biased by Kaplan et al.'s LR schedule. Chinchilla (2022) corrected to $N \propto C^{0.5}$, $D \propto C^{0.5}$. Implication: GPT-3 was undertrained; Chinchilla 70B beats GPT-3 175B.

7. **Critical batch size: 손실로 결정 / Critical batch size depends only on loss**
   $B_{\text{crit}} \propto L^{-4.8}$. 모델 크기 무관 — gradient noise scale로 측정 가능. 큰 모델 학습 시 1–2M 토큰이 적절. / $B_{\text{crit}} \propto L^{-4.8}$, independent of model size. For largest models, ~1–2M tokens is optimal at convergence.

8. **외삽의 한계와 자연어의 entropy 추측 / Extrapolation breaks down at language entropy bound**
   $C^* \sim 10^4$ PF-days, $N^* \sim 10^{12}$, $L^* \sim 1.7$ nats/token에서 멱법칙들이 모순. $L^*$는 자연어의 진정한 per-token entropy일 가능성. / Power laws contradict each other at $C^* \sim 10^4$ PF-days, $N^* \sim 10^{12}$, $L^* \sim 1.7$ nats/token. $L^*$ may be the true per-token entropy of natural language.

---

## 4. Mathematical Summary / 수학적 요약

### 핵심 정의 / Core definitions

**한국어**
- $N$: 비임베딩 매개변수 수
- $D$: 학습 데이터셋 크기 (tokens)
- $C$: 학습 컴퓨트 (PF-days, $1 \text{ PF-day} = 8.64 \times 10^{19}$ FLOPs)
- $L$: cross-entropy 손실 (nats)
- $B$: 배치 크기 (tokens), $S$: 매개변수 업데이트 수

기본 관계:
$$N \approx 12 n_{\text{layer}} d_{\text{model}}^2$$
$$C \approx 6 N B S$$
$$C_{\text{forward}} \approx 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$$

### 세 가지 기본 멱법칙 / Three basic power laws

**(1) $L(N)$, 수렴까지 학습, 충분한 데이터**:
$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$

**해석**: 모델 크기 $k$배 → 손실 $k^{-0.076}$배. $k=10$ → 손실 $\times 0.84$ (16% 감소).

**(2) $L(D)$, 충분히 큰 모델, early stopping**:
$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}$$

**해석**: 데이터 $k$배 → 손실 $k^{-0.095}$배. $k=10$ → 손실 $\times 0.80$ (20% 감소).

**(3) $L(C_{\min})$, 최적 모델, 충분한 데이터**:
$$L(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \quad C_c^{\min} \approx 3.1 \times 10^8 \text{ PF-days}$$

### 결합 멱법칙 / Joint power laws

**$L(N, D)$**:
$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

한계: $D \to \infty$ → $L(N, \infty) = (N_c/N)^{\alpha_N}$. $N \to \infty$ → $L(\infty, D) = (D_c/D)^{\alpha_D}$.

**$L(N, S_{\min})$** (학습 곡선):
$$L(N, S_{\min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}, \quad \alpha_S \approx 0.76, \quad S_c \approx 2.1 \times 10^3$$

### Critical batch size / 임계 배치 크기

$$B_{\text{crit}}(L) = \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

조정된 정의:
$$S_{\min}(S) = \frac{S}{1 + B_{\text{crit}}(L)/B}, \quad C_{\min}(C) = \frac{C}{1 + B/B_{\text{crit}}(L)}$$

### 컴퓨트 효율적 경계 / Compute-efficient frontier

이론적 예측:
$$\alpha_C^{\min} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx 0.054$$

자원 배분:
$$N \propto C_{\min}^{p_N}, \quad p_N = \alpha_C^{\min}/\alpha_N \approx 0.73$$
$$B \propto C_{\min}^{p_B}, \quad p_B = \alpha_C^{\min}/\alpha_B \approx 0.24$$
$$S \propto C_{\min}^{p_S}, \quad p_S = \alpha_C^{\min}/\alpha_S \approx 0.03$$
$$D = B \cdot S \propto C_{\min}^{0.27}$$

### Worked example / 구체적 예시: GPT-3 크기로 외삽

**한국어**
GPT-3는 $N = 1.75 \times 10^{11}$ 비임베딩 매개변수 (전체는 $\sim$임베딩 포함). $L(N)$ 외삽:
$$L_{\text{GPT-3}} \approx \left(\frac{8.8 \times 10^{13}}{1.75 \times 10^{11}}\right)^{0.076} = 503^{0.076}$$

계산: $\ln(503) \approx 6.22$, $0.076 \times 6.22 \approx 0.473$, $e^{0.473} \approx 1.60$ nats/token.

비교: 본 논문 추정 자연어 entropy $L^* \sim 1.7$ → GPT-3는 이 한계에 매우 가까움.

**최적 컴퓨트**: GPT-3급 모델의 컴퓨트 추정 — $N \propto C^{0.73}$이므로
$$C \propto N^{1/0.73} \approx N^{1.37}$$
$N$이 GPT-2 (1.5B) 대비 100배이므로 $C$는 $100^{1.37} \approx 500$배 증가 필요.

**English**
For GPT-3 with $N = 1.75 \times 10^{11}$ non-embedding parameters, extrapolating $L(N)$:
$$L_{\text{GPT-3}} \approx (8.8 \times 10^{13} / 1.75 \times 10^{11})^{0.076} = 503^{0.076} \approx 1.60 \text{ nats/token}$$
This is remarkably close to the conjectured natural-language entropy bound of $L^* \sim 1.7$ nats/token. The compute scales as $C \propto N^{1.37}$, so 100× the parameters of GPT-2 demand ~500× the compute.

### Overfitting penalty / 과적합 페널티

$$\delta L(N, D) \equiv \frac{L(N, D)}{L(N, \infty)} - 1 \approx \left(1 + \left(\frac{N}{N_c}\right)^{\alpha_N/\alpha_D} \frac{D_c}{D}\right)^{\alpha_D} - 1$$

데이터 요구 한계 (random seed 변동 0.02 이내):
$$D \gtrsim 5 \times 10^3 \cdot N^{0.74}$$

### Early stopping lower bound / 조기 종료 하한

$$S_{\text{stop}}(N, D) \gtrsim \frac{S_c}{[L(N, D) - L(N, \infty)]^{1/\alpha_S}}$$

### 자기 모순과 추측 / Contradiction and conjecture

$L(D(C_{\min})) \propto D^{-0.095}$, $D(C_{\min}) \propto C_{\min}^{0.26}$ → $L \propto C_{\min}^{-0.025}$.
$L(C_{\min}) \propto C_{\min}^{-0.050}$.

두 예측이 교차하는 지점:
$$C^* \sim 10^4 \text{ PF-Days}, \quad N^* \sim 10^{12}, \quad D^* \sim 10^{12} \text{ tokens}, \quad L^* \sim 1.7 \text{ nats/token}$$

### Parameter table / 매개변수 표

| Symbol | Value | Meaning |
|---|---|---|
| $\alpha_N$ | 0.076 | 모델 크기 지수 / Model-size exponent |
| $\alpha_D$ | 0.095 | 데이터 크기 지수 / Data-size exponent |
| $\alpha_C^{\min}$ | 0.050 | 최적 컴퓨트 지수 / Optimal-compute exponent |
| $\alpha_B$ | 0.21 | 임계 배치 지수 / Critical-batch exponent |
| $\alpha_S$ | 0.76 | step 지수 / Step exponent |
| $N_c$ | $8.8 \times 10^{13}$ | 매개변수 스케일 / Parameter scale |
| $D_c$ | $5.4 \times 10^{13}$ | 데이터 스케일 (tokens) / Data scale |
| $C_c^{\min}$ | $3.1 \times 10^8$ PF-days | 컴퓨트 스케일 / Compute scale |
| $B_*$ | $2 \times 10^8$ tokens | 배치 스케일 / Batch scale |
| $S_c$ | $2.1 \times 10^3$ steps | step 스케일 / Step scale |

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1948 ─ Shannon: Mathematical Theory of Communication
        │       (entropy of natural language ~1 bit/character)
        │
1990s ─ Statistical NLP, n-gram language models
        │
2003 ─ Bengio et al.: Neural Probabilistic Language Model
        │
2013 ─ Mikolov et al.: word2vec (paper #21)
        │
2014 ─ Sutskever et al.: Seq2Seq, Bahdanau et al.: Attention (#17)
        │
2017 ─ Vaswani et al.: Transformer (paper #25)
        │       ★ 본 논문의 아키텍처 / Architecture studied here
        │
2018 ─ Devlin et al.: BERT
2018 ─ Radford et al.: GPT-1
2018 ─ McCandlish et al.: Empirical Model of Large-Batch Training
        │       ★ Critical batch size 개념 / Critical batch size concept
        │
2019 ─ Radford et al.: GPT-2 (1.5B params)
        │       ─ "더 크면 더 좋다"의 일화적 증거
        │
2019 ─ Hestness et al. (HAD19): Beyond Human-Level Accuracy
        │       ─ 데이터 스케일링의 초기 연구
        │
2020 ─ ★★★ Kaplan et al.: Scaling Laws (this paper) ★★★
        │       ─ 멱법칙 정량화 / Quantitative power laws
        │       ─ Compute-efficient frontier
        │
2020 ─ Brown et al.: GPT-3 (175B params, paper #34)
        │       ─ Scaling Laws의 직접 적용
        │
2020 ─ Henighan et al.: Scaling Laws for Autoregressive Generative Modeling
        │       ─ 다른 modality (이미지, 비디오, 수학)로 확장
        │
2021 ─ Hernandez et al.: Scaling Laws for Transfer
        │
2022 ─ ★ Hoffmann et al.: Chinchilla ★
        │       ─ 본 논문의 결론을 정정: 모델:데이터 = 1:1
        │       ─ 70B Chinchilla가 175B GPT-3 능가
        │
2022 ─ Sorscher et al.: Beyond Neural Scaling Laws
2023 ─ LLaMA, GPT-4: Chinchilla-optimal 학습
2024 ─ Wei et al.: Emergent Abilities (멱법칙 너머의 현상)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#13 Krizhevsky et al. (2012) — AlexNet** | "큰 모델 + 큰 데이터"의 첫 증거. 본 논문은 이를 정량화. / First evidence that scale matters; this paper quantifies it. | High — scaling 패러다임의 시작 / Origin of the scaling paradigm |
| **#17 Bahdanau et al. (2014) — Attention** | Transformer의 attention 메커니즘 기반. / Foundation for Transformer attention. | Medium — 아키텍처 계보 / Architectural lineage |
| **#18 Kingma & Ba (2014) — Adam** | 본 논문의 표준 optimizer. / Standard optimizer used here. | Medium — 실험 도구 / Practical tool |
| **#25 Vaswani et al. (2017) — Transformer** | 본 논문이 분석하는 정확한 아키텍처. / The exact architecture studied. | Critical — direct dependency |
| **#34 Brown et al. (2020) — GPT-3** | 본 논문의 처방대로 175B 모델 훈련. Few-shot 능력의 출현. / GPT-3 is a direct application of these scaling laws (175B params, modest epochs). | Critical — 직접 적용 / Direct application |
| **Hoffmann et al. (2022) — Chinchilla** | 본 논문의 모델:데이터 = 5:1 결론을 1:1로 정정. LR 스케줄 결함 지적. / Corrects this paper's 5:1 to 1:1, citing LR-schedule bias. | Critical — 정정/도전 / Direct correction |
| **McCandlish et al. (2018) — Empirical Model of Large-Batch Training** | Critical batch size 개념의 출처. 본 논문 §5.1의 기반. / Source of critical batch size concept used in §5.1. | High — 이론적 도구 / Theoretical foundation |
| **Hestness et al. (2017) — Deep Learning Scaling is Predictable** | 데이터 스케일링의 초기 연구; 본 논문은 super-linear 대신 sub-linear 발견. / Earlier scaling study; this paper finds sub-linear instead of HAD17's super-linear. | High — 직접 비교 / Direct comparison |
| **Henighan et al. (2020) — Scaling Laws for Autoregressive Generative Modeling** | 본 논문 결과를 이미지/비디오/수학으로 확장. 같은 저자 그룹. / Extends these laws to images/video/math. Same group. | High — 직접 후속 / Direct follow-up |
| **#26 Kipf & Welling (2017) — GCN** | Sub-linear scaling 사이의 수학적 유사성 (제한 영역에서 power laws). / Similar phenomenology of power-law growth in restricted regimes. | Low — 개념적 유사성 / Conceptual analogy |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.

### Direct successors / 직접 후속
- Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners (GPT-3). *NeurIPS 2020*. arXiv:2005.14165.
- Henighan, T., Kaplan, J., Katz, M., et al. (2020). Scaling Laws for Autoregressive Generative Modeling. *arXiv:2010.14701*.
- Hernandez, D., Kaplan, J., Henighan, T., & McCandlish, S. (2021). Scaling Laws for Transfer. *arXiv:2102.01293*.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). *NeurIPS 2022*. arXiv:2203.15556.

### Theoretical foundations / 이론적 기반
- McCandlish, S., Kaplan, J., Amodei, D., & OpenAI Dota Team. (2018). An Empirical Model of Large-Batch Training. *arXiv:1812.06162*.
- Vaswani, A., et al. (2017). Attention is All You Need. *NIPS 2017*.
- Radford, A., Wu, J., Child, R., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2). OpenAI technical report.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.

### Related scaling work / 관련 스케일링 연구
- Hestness, J., Narang, S., Ardalani, N., et al. (2017). Deep Learning Scaling is Predictable, Empirically. *arXiv:1712.00409*.
- Hestness, J., Ardalani, N., & Diamos, G. (2019). Beyond Human-Level Accuracy: Computational Challenges in Deep Learning. *PPoPP 2019*.
- Rosenfeld, J. S., Rosenfeld, A., Belinkov, Y., & Shavit, N. (2019). A Constructive Prediction of the Generalization Error Across Scales. *ICLR 2020*. arXiv:1909.12673.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.

### Optimization and infrastructure / 최적화와 인프라
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
- Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. *ICML 2018*.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units (BPE). *ACL 2016*.

### Datasets / 데이터셋
- WebText: Radford et al. (2019), GPT-2 paper.
- Books Corpus: Zhu, Y., Kiros, R., Zemel, R., et al. (2015). Aligning Books and Movies. *ICCV 2015*.

### Modern context / 현대 맥락
- Wei, J., et al. (2022). Emergent Abilities of Large Language Models. *TMLR*. arXiv:2206.07682.
- Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. S. (2022). Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning. *NeurIPS 2022*.
- Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.
