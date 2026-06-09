---
title: "Highly Accurate Protein Structure Prediction with AlphaFold"
authors: [John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, Demis Hassabis]
year: 2021
journal: "Nature, Vol. 596, 583–589"
doi: "10.1038/s41586-021-03819-2"
topic: Artificial_Intelligence
tags: [protein-structure, alphafold, attention, evoformer, invariant-point-attention, equivariance, MSA, end-to-end, casp14, deepmind]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 37. Highly Accurate Protein Structure Prediction with AlphaFold / AlphaFold를 이용한 고정밀 단백질 구조 예측

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 1972년 Anfinsen이 제기한 이래 50년간 풀리지 않은 **단백질 구조 예측 문제(protein folding problem)**에서 최초로 **원자 수준 정확도(atomic accuracy)**에 도달한 컴퓨터 알고리즘 **AlphaFold 2**를 제시합니다. 2020년 12월 CASP14(14th Critical Assessment of Structure Prediction)에서 87개 도메인에 대해 backbone Cα r.m.s.d. 중앙값 **0.96 Å (95% 신뢰구간 0.85–1.16 Å)**를 달성했는데, 이는 차순위 방법 2.8 Å, 그리고 탄소 원자 직경 1.4 Å보다도 작은 오차입니다. all-atom 정확도 1.5 Å는 X선 결정학과 NMR 실험 구조와 견줄 만한 수준이며, 구조 미해결인 새로운 fold에 대해서도 견고하게 작동합니다. 핵심 아키텍처는 두 부분입니다: (i) **Evoformer** — MSA representation $(s, r, c)$과 pair representation $(r, r, c)$을 48개 블록에 걸쳐 동시에 갱신하는 신경망 trunk. row-wise gated self-attention with pair bias, column-wise self-attention, outer-product mean(MSA→pair), 그리고 **삼각 부등식 기반 4종 연산**(triangle multiplicative update outgoing/incoming, triangle attention starting/ending)으로 구성. (ii) **Structure Module** — 잔기를 회전+병진의 강체 frame $(R_i, t_i)$로 보고, **Invariant Point Attention(IPA)**으로 좌표를 직접 회귀하는 8개 블록(weight-shared). 학습은 ~170k PDB 구조와 ~350k Uniclust30 서열에 대한 self-distillation으로 진행되며, 손실은 **FAPE(Frame Aligned Point Error)**를 중심으로 한 다중 항입니다. 출력 직전 전체 네트워크를 3회 **recycling**하여 점진 정제를 수행하고, 잔기별 신뢰도(**pLDDT**)와 도메인 packing 신뢰도(**PAE/pTM**)를 함께 출력합니다.

**English**
This paper introduces **AlphaFold 2**, the first computational method to reach **atomic-accuracy** on the 50-year-old protein-folding problem (Anfinsen 1972). At CASP14 (December 2020), AlphaFold 2 achieved a median backbone Cα r.m.s.d. of **0.96 Å (95% CI 0.85–1.16 Å)** across 87 domains — versus 2.8 Å for the next-best entry, and below the diameter of a carbon atom (~1.4 Å). All-atom accuracy was 1.5 Å, comparable to experimental X-ray and NMR structures, and the model generalises to novel folds. The architecture has two trunks: (i) the **Evoformer** — 48 blocks that jointly update an MSA representation $(s, r, c)$ and a pair representation $(r, r, c)$ via row-wise gated self-attention biased by the pair stack, column-wise self-attention, outer-product mean (MSA → pair), and four **triangle-inequality-aware** operations (triangle multiplicative update outgoing/incoming, triangle attention starting/ending node); and (ii) the **Structure Module** — 8 weight-shared blocks that treat each residue as a rigid frame $(R_i, t_i)$ and regress its rotation and translation through **Invariant Point Attention (IPA)**, an SE(3)-invariant attention with 3D-point queries/keys/values produced in each residue's local frame. Training combines ~170k PDB chains with ~350k Uniclust30 sequences whose structures were predicted by an earlier model and filtered by confidence (**self-distillation**). The main loss is **FAPE** (Frame-Aligned Point Error) — a clipped per-frame, per-atom L1 distance — supplemented by auxiliary distogram, masked-MSA (BERT-style), and confidence (lDDT, TM-score) losses. The full network is **recycled** three times for progressive refinement, and produces per-residue confidence (**pLDDT**) plus pairwise Predicted Aligned Error (**PAE**, summarised as **pTM**) alongside the structure.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (p. 583) / 서론

**한국어**
저자들은 단백질이 약 1억 개의 알려진 서열을 가지지만 실험 결정 구조는 약 100,000개에 불과하다는 통계로 시작합니다 — 약 1,000:1의 격차. 구조 결정은 한 단백질당 수개월~수년이 걸리며, 이를 컴퓨터로 메우는 것이 50년 묵은 과제입니다. 두 갈래의 전통적 접근:

1. **물리 시뮬레이션**: 분자동역학(MD), force field 기반 에너지 최소화. 이론적으로는 Anfinsen의 원리에 부합하지만, 중간 크기의 단백질에서도 conformation 공간이 너무 커서 계산적으로 불가능. 또한 force field 자체가 근사이므로 정확도 한계가 있음.
2. **진화 정보(MSA)와 bioinformatics**: 동족 단백질에서 잔기 쌍의 공진화(coevolution)를 분석하여 contact을 추정 → 거리 제약 → 3D 좌표. CASP13(2018)의 AlphaFold 1, trRosetta가 이 파이프라인의 정점이었음. 그러나 close homologue가 없으면 무력하고, atomic accuracy에 미치지 못함.

저자들의 결론: AlphaFold 2는 **첫 번째로 원자 수준 정확도를 일관되게 제공하는 방법**이며, CASP14에서 이를 blind test로 입증함. 핵심은 (i) MSA와 pairwise feature를 공동 임베딩하는 새로운 아키텍처, (ii) 종단 간 학습이 가능한 새로운 출력 표현과 손실, (iii) 새로운 equivariant attention, (iv) 중간 손실을 통한 iterative refinement, (v) MSA를 활용한 마스킹 손실, (vi) self-distillation, (vii) 정확도 자기 추정.

**English**
The introduction frames the gap: ~$10^8$ known sequences vs ~$10^5$ experimental structures (a ~1000:1 ratio), with months-to-years per structure determination. Two classical paths existed: **physical simulation** (theoretically grounded but intractable beyond small proteins, and limited by force-field approximations) and **bioinformatics** using evolutionary covariation in MSAs (productive but stuck below atomic accuracy and weak when no close homologue exists). The authors claim AlphaFold 2 is the first method to **regularly produce predictions with atomic accuracy even when no similar structure is known**, validated at CASP14. Seven listed innovations: (i) joint MSA/pair architecture, (ii) end-to-end output representation and loss, (iii) new equivariant attention, (iv) intermediate losses for iterative refinement, (v) BERT-style masked MSA loss, (vi) self-distillation on unannotated sequences, (vii) confidence self-estimation.

### Part II: Performance on CASP14 (Fig. 1, p. 584) / CASP14 성능

**한국어**
- **Fig. 1a**: 87개 protein domain 에 대한 r.m.s.d.$_{95}$ 중앙값. AlphaFold = 0.96 Å, 다른 상위 그룹들은 모두 2.8 Å 이상. 그래프에서 AlphaFold만 단독으로 낮은 막대.
- **Fig. 1b**: T1049 (PDB 6YF) — 예측(파랑)과 실험(녹색)이 거의 완벽 중첩. r.m.s.d.$_{95}$ = 0.8 Å, TM-score = 0.93.
- **Fig. 1c**: T1056 (PDB 6YJ1) — Zn 결합 site의 side chain까지 정확. AlphaFold는 zinc ion 자체를 예측하지 않지만 그 주변 cysteine side chain 위치까지 0.59 Å 이내로 맞춤.
- **Fig. 1d**: T1044 (PDB 6VR4, 2,180 residue 단일 chain) — 거대 단백질이지만 r.m.s.d.$_{95}$ = 2.2 Å, TM-score = 0.96. 대형 단백질에도 작동.
- **Fig. 1e**: 전체 아키텍처. 입력 서열 → (genetic database search → MSA) + (structure database search → templates) → MSA representation $(s, r, c)$ + Pair representation $(r, r, c)$ → **Evoformer (48 blocks)** → Single representation $(r, c)$ + Pair representation → **Structure module (8 blocks)** → 3D structure. 그리고 **Recycling (3회)**.

**Recently released PDB chains (Fig. 2)**: 학습 데이터 cutoff 이후 PDB에 등재된 단백질 3,144개에 대해 backbone r.m.s.d. 중앙값 1.46 Å (95% CI 1.40–1.56 Å). 즉, 새로운 단백질에서도 견고하게 작동.

**Figure 2b**: 신뢰도(pLDDT)가 실제 lDDT-Cα를 잘 예측 (Pearson r=0.76). Figure 2c–d: pTM도 TM-score를 정확히 예측 (r=0.85).

**English**
- **Fig. 1a**: median r.m.s.d.$_{95}$ across 87 CASP14 domains. AlphaFold ≈ 0.96 Å vs ≥ 2.8 Å for every other top group.
- **Fig. 1b–d**: example predictions (T1049 r.m.s.d.$_{95}$ = 0.8 Å; T1056 zinc-binding site predicted to 0.59 Å within 8 Å of Zn; T1044 a 2,180-residue chain predicted at 2.2 Å with TM-score 0.96).
- **Fig. 1e**: end-to-end architecture (sequence → MSA + pair via templates → 48-block Evoformer → 8-block Structure Module → 3D structure, with 3× recycling).
- **Fig. 2**: median 1.46 Å on 3,144 recent PDB chains (post-cutoff); pLDDT correlates with lDDT-Cα at r=0.76; pTM with TM-score at r=0.85. The model **knows when it is right**.

### Part III: AlphaFold Network Overview (p. 584-585) / 네트워크 개요

**한국어**
- **입력**: 1차 아미노산 서열 (길이 $r = N_{\text{res}}$).
- **MSA construction**: jackhmmer + HHblits로 동족 서열 검색 → MSA representation $(N_{\text{seq}} \times N_{\text{res}} \times c)$, $c=256$ (default).
- **Templates**: HHsearch로 기존 PDB 구조에서 부분 정렬을 찾고 pair representation 초기값으로 사용.
- **Pair representation**: $(N_{\text{res}} \times N_{\text{res}} \times c)$, $c=128$.
- **Trunk (Evoformer)**: 48개 블록, weight 비공유. MSA와 pair를 동시에 진화시킴.
- **Structure Module**: 8개 블록, weight 공유. 잔기 frame $(R_i, t_i)$를 회귀.
- **Recycling**: 출력을 다시 입력에 합쳐 3회 반복.

**핵심 통찰 (저자 인용)**: "구체적인 구조 가설이 Evoformer 블록 초기에 형성되며 이후 지속적으로 정제됨" — 즉, 네트워크가 깊어질수록 막연한 분포에서 점점 확정적인 구조로 수렴. Fig. 4b의 trajectory는 이를 시각화.

**English**
Input is the primary amino-acid sequence. Genetic-database search (jackhmmer, HHblits) builds the MSA $(N_{\text{seq}} \times N_{\text{res}} \times 256)$, while structure-database search (HHsearch) provides templates that initialise the pair representation $(N_{\text{res}} \times N_{\text{res}} \times 128)$. The trunk is the Evoformer (48 blocks, no weight sharing) which jointly updates MSA and pair representations. The Structure Module (8 weight-shared blocks) converts these into rotation+translation per residue and predicts atom positions. The whole network is recycled 3 times. The authors observe that "a concrete structural hypothesis arises early in the Evoformer and is continuously refined" — the residue-gas trajectory visualises this (Fig. 4b).

### Part IV: Evoformer (p. 585-586, Fig. 3a-c) / Evoformer 블록 상세

**한국어**
Evoformer의 핵심 아이디어는 **단백질 구조 예측을 3D 공간에서의 그래프 추론 문제**로 보는 것입니다 — pair representation이 잔기 사이 가장자리(edge)이고, 가장자리들이 삼각 부등식 등 기하 제약을 만족해야 합니다.

**Evoformer 블록 내부 (Fig. 3a, 9개 모듈, 순차 적용)**:

1. **Row-wise gated self-attention with pair bias**: MSA의 같은 시퀀스 내(같은 row) 잔기들 사이의 attention. **pair representation에서 온 logit이 attention에 bias로 더해짐** — 이것이 pair → MSA 정보 흐름의 핵심.
   - 입력: MSA $(s, r, c_m)$, pair $(r, r, c_z)$
   - 한 row(=하나의 시퀀스)에 대해: $\mathrm{Att}_{ij}^s \propto \mathrm{exp}\big( q_i^\top k_j / \sqrt{c_m} + b_{ij} \big)$, 여기서 $b_{ij}$는 pair representation의 linear projection.
   - "Gated" — sigmoid gate로 출력 흐름 제어.

2. **Column-wise gated self-attention**: MSA의 같은 column(같은 잔기 위치) 내 시퀀스들 사이의 attention. 어느 시퀀스가 어떤 시퀀스와 진화적으로 관련이 깊은지 학습.

3. **Transition (MSA)**: position-wise feed-forward (4× hidden, ReLU 또는 유사).

4. **Outer product mean (MSA → pair)**: 각 시퀀스에서 두 잔기 column $m_{si}, m_{sj}$의 외적을 시퀀스 차원 $s$로 평균하여 pair representation에 더함.
   $$z_{ij} \mathrel{+}= \mathrm{Linear}\!\left(\mathrm{mean}_{s}\, \mathrm{Linear}(m_{si}) \otimes \mathrm{Linear}(m_{sj})\right)$$
   - Coevolution을 직접 인코딩. 기존 AlphaFold 1과 달리 매 블록마다 이 연산이 적용됨.

5. **Triangle multiplicative update — outgoing edges**:
   $$z_{ij} \mathrel{+}= \mathrm{Linear}\!\left(\sum_k a_{ik} \odot b_{jk}\right)$$
   - 두 outgoing 엣지 $i \to k$, $j \to k$로부터 $i \to j$를 갱신. attention보다 저렴한 대체로 개발됨.

6. **Triangle multiplicative update — incoming edges**:
   $$z_{ij} \mathrel{+}= \mathrm{Linear}\!\left(\sum_k a_{ki} \odot b_{kj}\right)$$
   - 두 incoming 엣지로부터 $i \to j$를 갱신.

7. **Triangle self-attention — around starting node**: 잔기 $i$를 시작점으로 하는 모든 엣지 $(i, j)$, $(i, k)$ 사이의 self-attention. 즉 동일한 head(start) 잔기 주변 엣지들을 함께 다룸.

8. **Triangle self-attention — around ending node**: 동일한 끝점 잔기 주변.

9. **Transition (pair)**.

이 9개 연산이 한 블록을 이루고, 48개 블록이 직렬 연결됨(weight 비공유).

**왜 삼각형인가?** 거리는 삼각 부등식 $d_{ij} \le d_{ik} + d_{kj}$를 만족해야 함. pair representation을 거리/방향 행렬로 해석한다면, 어떤 두 엣지가 결정되면 세 번째 엣지에 강한 제약이 있음. Triangle multiplicative update와 triangle attention은 이를 명시적 inductive bias로 인코딩.

**English**
The Evoformer views structure prediction as **graph inference in 3D**, where pair representation entries are edges of a residue graph that must satisfy geometric constraints (chiefly the triangle inequality on distances). One block applies 9 modules in order:

1. **Row-wise gated self-attention with pair bias** — each MSA row attends within itself; logits include a learned bias from the pair stack (this is the principal pair → MSA channel).
2. **Column-wise gated self-attention** — across sequences at the same residue position, learning evolutionary relationships.
3. **MSA Transition** — position-wise feed-forward (~4× hidden).
4. **Outer-product mean (MSA → pair)** — outer products of MSA columns averaged over sequences, written into pair (encodes coevolution; applied every block, unlike AF1).
5. **Triangle multiplicative update — outgoing**: $z_{ij} \mathrel{+}= \mathrm{Linear}(\sum_k a_{ik} \odot b_{jk})$.
6. **Triangle multiplicative update — incoming**: $z_{ij} \mathrel{+}= \mathrm{Linear}(\sum_k a_{ki} \odot b_{kj})$.
7. **Triangle attention around starting node** — attention over edges sharing a common starting residue.
8. **Triangle attention around ending node** — attention over edges sharing a common ending residue.
9. **Pair Transition**.

48 blocks total, no weight sharing. The triangle operations bake the **triangle-inequality** constraint directly into the inductive bias: knowing two edges of a residue triangle constrains the third.

### Part V: Structure Module (p. 586-587, Fig. 3d-f) / Structure Module 상세

**한국어**
Structure Module은 Evoformer의 출력 **single representation** $(r, c)$ (MSA의 첫 row)과 pair representation을 받아, 잔기마다 backbone frame $(R_i, t_i) \in SE(3)$과 side-chain 토션각 $\chi_1, \chi_2, \chi_3, \chi_4$를 회귀합니다.

**초기화**: 모든 잔기의 frame을 identity rotation + 원점 — 이를 저자들은 **"residue gas"**라 부름 (Fig. 3e). 모든 잔기가 원점에 겹쳐 있는 비물리적 상태.

**8개 블록(weight 공유) 각각 (Fig. 3d)**:
1. **IPA module (Invariant Point Attention)**: single representation을 pair에 의해 bias된 SE(3)-invariant attention으로 갱신. 핵심 연산.
2. **Predict relative rotations and translations**: 갱신된 single representation에서 각 잔기의 relative frame update를 회귀.
3. **Compose with previous frames**: $(R_i, t_i) \leftarrow (R_i \cdot \Delta R_i, R_i \cdot \Delta t_i + t_i)$.
4. **Predict $\chi$ angles & compute all atom positions**: side chain 토션 4개를 sin/cos로 예측 → 표준 amino acid 기하학을 사용해 모든 heavy atom 좌표 계산.

블록을 8회 반복하면 residue gas → folded structure로 진화.

**IPA의 수학** (논문 Methods + Supplementary):

각 head는 표준 attention의 query/key/value 외에 **3D 점**들을 추가로 생성합니다:
- $\vec{q}_i^{hp}, \vec{k}_i^{hp} \in \mathbb{R}^3$: head $h$, point $p$, 잔기 $i$의 local frame에서 좌표.

이 점들을 전역(global) frame으로 변환: $T_i \cdot \vec{q}_i^{hp} = R_i \vec{q}_i^{hp} + t_i$.

attention logit:
$$a_{ij}^h = \frac{1}{\sqrt{c}} q_i^{h\top} k_j^h \;-\; \frac{\gamma^h}{2} \sum_p \|T_i \cdot \vec{q}_i^{hp} - T_j \cdot \vec{k}_j^{hp}\|^2 \;+\; b_{ij}^h$$

세 항 해석:
- 첫째 항: 표준 dot-product attention. 추상 임베딩 유사도.
- 둘째 항: **3D 점들이 가까울수록** attention이 강해짐 (음의 거리 제곱 페널티). $\gamma^h$는 학습 가능 head별 가중치.
- 셋째 항: pair representation의 logit bias.

attention 적용:
$$o_i^h = \sum_j \mathrm{softmax}_j(a_{ij}^h) \cdot v_j^h \quad\text{(scalar value)}$$
$$\vec{o}_i^{hp} = T_i^{-1} \cdot \sum_j \mathrm{softmax}_j(a_{ij}^h) \cdot (T_j \cdot \vec{v}_j^{hp}) \quad\text{(point value)}$$

마지막에 다시 $T_i^{-1}$로 잔기 $i$의 local frame으로 가져옴 — 이것이 SE(3) **invariance**의 핵심. 전체 단백질을 회전·병진해도 IPA 출력은 변하지 않음. 이는 결과적으로 좌표 예측 자체는 **equivariant**가 됨 (입력 회전 → 출력 회전).

**FAPE 손실 (Fig. 3f)**:
$$\mathcal{L}_{\text{FAPE}} = \frac{1}{N_{\text{frames}} N_{\text{atoms}}} \sum_{i=1}^{N_{\text{frames}}} \sum_{j=1}^{N_{\text{atoms}}} \min\!\big(d_{\max}, \, \|T_i^{-1} x_j - \hat{T}_i^{-1} \hat{x}_j\|\big)$$

여기서 $T_i, x_j$는 예측, $\hat{T}_i, \hat{x}_j$는 정답. clipping $d_{\max} = 10$ Å.

**왜 FAPE가 영리한가?**
- 각 frame $i$에서 본 모든 atom $j$의 위치 오차를 평가 → 측정이 $N_{\text{frames}} \times N_{\text{atoms}}$ 회 이루어지므로 dense supervision.
- side-chain의 방향성과 chirality가 자연스럽게 학습됨 (RMSD는 mirror image에 무력).
- $d_{\max}$ clipping으로 outlier에 robust.

**Equivariance & Iterative refinement**:
- IPA의 attention 자체는 frame에 invariant. frame 갱신 연산은 SE(3) equivariant.
- 전체 backbone을 끊어서(즉, peptide bond 기하 제약을 일시적으로 무시하고) 모든 잔기를 동시에 자유롭게 갱신 — 복잡한 loop closure 문제를 회피.
- 최종 단계에서 Amber force field로 gradient descent relaxation. 정확도(GDT, lDDT)는 거의 변하지 않으나 stereochemistry violation을 제거.

**English**
The Structure Module ingests the single representation $(r, c)$ (first row of MSA) and the pair representation, and regresses a backbone frame $(R_i, t_i) \in SE(3)$ plus four side-chain torsion angles $\chi_1\ldots\chi_4$ per residue. Frames initialise to identity at the origin — the authors call this the **"residue gas"** (Fig. 3e), a non-physical state where all residues overlap at $\mathbf{0}$. Each of 8 weight-shared blocks (Fig. 3d) does:

1. **IPA** updates the single representation with SE(3)-invariant attention biased by the pair stack.
2. **Predict relative $(\Delta R_i, \Delta t_i)$** from the updated single rep.
3. **Compose** with the running frame.
4. **Predict $\chi$ angles** and compute all heavy-atom positions using ideal amino-acid geometry.

**IPA mathematics**: each head produces, in addition to scalar Q/K/V, **3D point** queries/keys/values $\vec{q}_i^{hp}, \vec{k}_i^{hp}, \vec{v}_i^{hp} \in \mathbb{R}^3$ in residue $i$'s local frame. Logits:
$$a_{ij}^h = \tfrac{1}{\sqrt{c}} q_i^{h\top} k_j^h - \tfrac{\gamma^h}{2} \sum_p \|T_i \cdot \vec{q}_i^{hp} - T_j \cdot \vec{k}_j^{hp}\|^2 + b_{ij}^h$$
Three terms: standard dot-product, **squared 3D-point distance penalty** (closer points → higher attention), and pair-bias. Outputs include scalar values and point values that are pulled back to residue $i$'s local frame via $T_i^{-1}$ — making the operation SE(3)-invariant. Coordinate prediction is then SE(3)-**equivariant**.

**FAPE loss**: $\mathcal{L}_{\text{FAPE}} = \frac{1}{N_{\text{frames}} N_{\text{atoms}}} \sum_{i,j} \min(d_{\max}, \|T_i^{-1} x_j - \hat{T}_i^{-1} \hat{x}_j\|)$ with $d_{\max}=10$ Å. Three virtues: (a) dense supervision ($N_{\text{frames}} \times N_{\text{atoms}}$ measurements), (b) chirality is forced because frames distinguish mirror images (RMSD does not), (c) clipping handles outliers.

The frames are updated independently each step — the chain is allowed to "break" temporarily (peptide bonds violated) so that all residues can be refined locally without solving expensive loop-closure constraints. Final Amber force-field relaxation removes stereochemistry violations without changing GDT.

### Part VI: Training (p. 587-588) / 학습

**한국어**
**데이터**:
- **PDB**: ~170,000개 chain (~25%는 single-chain, 나머지는 multi-chain의 individual chain). 학습 데이터 cutoff: 2018년 4월(처음) → 2020년 5월.
- **Self-distillation**: Uniclust30(~350,000개 다양한 unannotated sequences)에 대해 첫 학습 모델로 구조를 예측하고, 신뢰도 높은 것만(filtered) 추가 학습 데이터로 사용. **결정적**: 평균 GDT를 약 +5점 향상시킴 (Fig. 4a).
- **MSA depth**: 100~10,000 시퀀스가 일반적. <30 시퀀스에서 정확도 급감 (Fig. 5a).

**손실 함수 (다중 항 가중합)**:
- FAPE (주 손실, backbone + side chain).
- Auxiliary distogram head: pair representation에서 $C_\beta$-$C_\beta$ 거리 분포를 예측하는 loss.
- Masked-MSA(BERT-style): MSA의 일부 토큰을 mask 후 복원. 진화 통계 학습 강제.
- pLDDT head: per-residue lDDT-Cα를 예측하는 cross-entropy.
- TM-score(pTM) head: pairwise predicted aligned error (PAE) 분포에서 expected TM-score를 계산.
- Structural violation loss: peptide bond 기하 제약(C-N 거리, $\omega$ 각, atom clash)에 대한 penalty (fine-tuning에서 활성화).

**Recycling**: 출력 frame과 pair representation을 입력에 추가하여 전체 forward pass를 3회 반복. gradient는 마지막 cycle만 흐름 → 메모리 절약. Fig. 4a의 ablation: recycling 제거 시 GDT가 ~10점 하락.

**Optimization**: Adam, batch size 128, gradient clipping. 학습은 16 TPUv3 cores로 약 1주일.

**Ablation results (Fig. 4a)** — GDT 차이 vs baseline:
| Component removed | CASP14 GDT difference | PDB lDDT-Cα difference |
|---|---|---|
| With self-distillation | +5 | +1 |
| **Baseline** | 0 | 0 |
| No templates | -0.5 | -0.5 |
| No auxiliary distogram head | -1 | -1 |
| No raw MSA (use pairwise frequencies) | -3 | -2 |
| No IPA (use direct projection) | -3 | -2 |
| No auxiliary masked-MSA head | -1 | -0.5 |
| No recycling | -10 | -2 |
| No triangles, biasing, gating | -5 | -2 |
| No end-to-end gradients (auxiliary heads only) | -10 | -3 |
| No IPA & no recycling | -20 | -4 |

**English**
**Data**: ~170k PDB chains (training cutoff initially April 2018, updated to May 2020). **Self-distillation** on ~350k filtered Uniclust30 sequences whose structures were predicted by an initial model and kept only when high-confidence — adds ~+5 GDT (Fig. 4a). MSA depth matters: accuracy degrades sharply below ~30 sequences (Fig. 5a).

**Losses (weighted sum)**: FAPE (main), distogram head, BERT-style masked MSA reconstruction, pLDDT (per-residue lDDT-Cα prediction), pTM (from PAE distribution), and structural violation penalties (peptide bond geometry, atom clashes).

**Recycling**: feed final outputs (frames, pair) back as inputs for 3 cycles; gradient flows only through the last cycle (memory-efficient). Removing recycling drops CASP14 GDT by ~10 points.

**Training**: Adam, batch size 128, gradient clipping, ~1 week on 16 TPUv3 cores.

**Key ablation findings**: removing IPA, recycling, end-to-end gradients, or triangle operations each costs many GDT points; combining "no IPA & no recycling" alone costs ~20 GDT — the architecture is genuinely synergistic.

### Part VII: Interpreting the Network (p. 587, Fig. 4b) / 네트워크 해석

**한국어**
저자들은 48개 Evoformer 블록 각각에 대해 별도의 structure module head를 학습시켜, 블록 단위로 "현재까지의 구조 예측"을 시각화했습니다 — 192개 중간 구조의 trajectory(48 blocks × 4 recycle cycles).

**Fig. 4b 결과**:
- T1024 (LmrP): 첫 몇 블록 안에 거의 정답 구조에 도달하고 이후 미세 조정.
- T1044, T1064 (SARS-CoV-2 ORF8): 50+ 블록 동안 큰 구조 변화 → 여러 가설을 탐색하는 양상.
- T1064는 마지막 직전까지도 GDT가 50 미만으로 머물다 급격히 90+로 점프.

**해석**: 네트워크가 단순히 lookup이 아니라 **반복적 가설 형성과 정제** 과정을 수행. residue gas → 거시 구조 → 미세 정제. 깊이 자체가 알고리즘적 자원을 제공하는 셈.

**English**
The authors trained separate structure-module heads for each of the 48 Evoformer blocks, providing a 192-frame (48 blocks × 4 recycling cycles) trajectory of the network's evolving structural hypothesis. Easy targets (T1024) converge in a few blocks; hard ones (T1064 = SARS-CoV-2 ORF8) reorganise secondary structure for many blocks before settling. The trajectory is monotonically improving once past the early phase — depth is genuinely doing **iterative hypothesis refinement**, not just lookup.

### Part VIII: Limitations & Discussion (p. 588-589) / 한계와 논의

**한국어**
**한계**:
1. **MSA depth < 30**: 정확도가 급감. 매우 새로운 단백질 패밀리에 무력.
2. **Cross-chain contacts**: heteromer에서 chain 간 접촉이 많은 단백질은 단독 chain 예측이 부정확. AlphaFold-Multimer에서 부분 해결.
3. **Ligand, cofactor**: Zn, heme 등 cofactor 의존 fold는 좌표는 맞추지만 cofactor 자체는 예측 안 함.
4. **Dynamics, multiple conformations**: 단일 정적 구조만 출력. allosteric ensemble이나 disordered region은 표현 한계.
5. **메모리/연산**: 1,000 잔기 단백질 1개 모델 예측에 ~1 GPU minute (V100). 매우 긴 단백질은 도메인 단위로 나눠야 함.

**Strengths over physical methods**:
- Hydrogen bond term을 명시적으로 넣지 않아도 학습으로 효과적으로 형성됨.
- Hand-crafted feature 최소화 — PDB의 raw 데이터에서 직접 학습.

**Implications**:
- AlphaFold DB(2021)는 인간 프로테옴 + ~200M UniProt 서열 구조 공개.
- cryo-EM map 해석, 신약 표적 발굴, de novo 단백질 디자인에 즉시 적용.
- 더 일반적으로, **dual-representation + cross-talk** 패턴은 다른 도메인(날씨, 분자, 신소재)으로 확장 가능.

**English**
**Limitations**: (1) accuracy drops sharply for MSA depth < 30; (2) heteromer chains with many cross-chain contacts are weakly predicted (addressed by AlphaFold-Multimer); (3) cofactors (Zn, heme) themselves are not modelled though their binding-site geometry is; (4) only a single static structure — no ensembles or disordered regions; (5) ~1 GPU-minute per 1k-residue model on V100, requiring domain-splitting for very long chains. **Strengths** over physical methods: hydrogen bonding emerges without an explicit term; hand-crafted features are minimised; the model handles missing physical context (cofactors, oligomeric state) gracefully. **Implications**: AlphaFold DB (2021) covers the human proteome and ~200M UniProt sequences; the dual-representation + cross-talk pattern is now standard in other domains (GraphCast for weather, AlphaFold 3 for ligands/DNA/RNA via diffusion, ESMFold).

---

## 3. Key Takeaways / 핵심 시사점

1. **50년 묵은 문제의 완결 / The protein folding problem is essentially solved**
   CASP14에서 0.96 Å backbone 정확도는 X선 결정학과 견줄 수 있는 수준이며, 단백질 구조 예측이 더 이상 "과학적 미해결 문제"가 아닌 도구임을 의미합니다. / 0.96 Å backbone accuracy at CASP14 places computational prediction at the level of experimental crystallography — turning structure prediction from an open scientific problem into a routine tool.

2. **이중 표현(MSA + pair)과 그들의 상호 갱신이 핵심 / Dual representations with constant cross-talk is the architectural core**
   MSA만 또는 pair만 사용하는 single-tower 모델은 정확도가 크게 떨어집니다. Outer-product mean(MSA→pair)과 row-wise attention with pair bias(pair→MSA)가 매 블록마다 정보를 양방향 전달. / Single-tower variants (MSA-only or pair-only) lose substantial accuracy. The bidirectional flow — outer-product mean (MSA → pair) and pair-biased row attention (pair → MSA) — is applied every block.

3. **삼각 부등식이 직접 inductive bias가 됨 / Triangle inequality is encoded as inductive bias**
   Triangle multiplicative update와 triangle attention은 잔기 트리플 $(i, j, k)$의 거리 일관성을 직접 강제. ablation에서 이를 제거하면 GDT가 ~5점 하락. / Removing triangle operations costs ~5 GDT — geometric consistency is not an emergent property but an architectural choice.

4. **Equivariance를 invariant한 attention으로 달성 / Equivariance achieved via SE(3)-invariant attention**
   IPA는 잔기마다 local frame을 두고 거기서 3D 점을 생성, 다른 잔기로 전송할 때만 global frame으로 변환. attention은 SE(3) invariant이고, 전체 좌표 예측은 equivariant. 복잡한 SE(3)-equivariant network 이론 없이 단순 trick으로 달성. / IPA generates 3D points in each residue's local frame and only momentarily moves to global coordinates — making attention SE(3)-invariant and coordinate prediction equivariant, without any heavy group-theoretic machinery.

5. **FAPE는 단순 RMSD보다 훨씬 강력 / FAPE is far more powerful than naive RMSD**
   $N_{\text{frames}} \times N_{\text{atoms}}$ 측정으로 dense supervision을 제공하고, 모든 local frame에서 본 거리이기에 chirality와 side-chain orientation이 자연스럽게 학습됨. clipping으로 outlier에 robust. / Frame-aligned errors give dense ($N_{\text{frames}} \times N_{\text{atoms}}$) supervision, naturally learn chirality and orientation, and clip outliers — a much richer signal than a single RMSD.

6. **Recycling은 단순하지만 결정적 / Recycling is simple but decisive**
   네트워크의 출력을 다시 입력으로 넣어 3회 반복하는 것만으로 GDT가 ~10점 향상. 추가 매개변수 없이 효과적인 깊이 증가. / Looping the network outputs back into its inputs three times — with no extra parameters — buys ~10 GDT. Effective depth without parameter cost.

7. **Self-distillation으로 PDB의 한계를 돌파 / Self-distillation breaks past the PDB ceiling**
   170k PDB만으로 학습한 baseline 위에 350k unannotated 서열로 self-distill하면 추가 +5 GDT. PDB가 모든 자연 단백질의 다양성을 대표하지 않음을 시사. / Self-distillation on 350k unannotated sequences adds +5 GDT over a 170k-PDB-only baseline — the PDB does not span natural protein diversity.

8. **신뢰도 자기 추정이 실용성의 열쇠 / Self-estimated confidence makes the model usable**
   pLDDT (per-residue) 와 PAE/pTM (pairwise/global)가 실제 정확도와 강한 상관 (r ≥ 0.76, 0.85). 사용자는 어디를 믿을지 안다. / pLDDT and PAE/pTM strongly correlate with true accuracy (r ≥ 0.76 and 0.85). The model knows where it is trustworthy — essential for downstream use.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Tensor shapes / 텐서 모양

| Symbol | Shape | Meaning |
|---|---|---|
| MSA representation $m$ | $(s, r, c_m)$ | $s$ sequences × $r$ residues × $c_m=256$ |
| Pair representation $z$ | $(r, r, c_z)$ | residue-residue edges, $c_z=128$ |
| Single representation | $(r, c_s)$ | first row of MSA after Evoformer |
| Backbone frames | $(r, 3 \times 3)$ rotation + $(r, 3)$ translation | per-residue rigid frame |

### 4.2 Row-wise gated self-attention with pair bias / 페어 바이어스가 있는 행 단위 self-attention

For each MSA row $s$ (한 시퀀스에 대해):
$$\text{Attn}_{ij}^{s,h} = \mathrm{softmax}_j\!\left(\frac{1}{\sqrt{c}} q_{si}^{h\top} k_{sj}^h + b_{ij}^h\right)$$
$$\tilde{m}_{si}^h = \sum_j \text{Attn}_{ij}^{s,h} \, v_{sj}^h$$
$$m_{si} \mathrel{+}= \mathrm{Linear}\!\big(\sigma(g_{si}) \odot \mathrm{concat}_h \tilde{m}_{si}^h\big)$$

여기서 $b_{ij}^h = \mathrm{Linear}_h(z_{ij})$는 pair representation에서 온 bias, $g_{si}$는 학습된 gate. **Pair → MSA 정보 흐름의 본체**.

### 4.3 Column-wise gated self-attention / 열 단위 self-attention

For each MSA column $i$ (한 잔기 위치에 대해):
$$\text{Attn}_{st}^{i,h} = \mathrm{softmax}_t\!\left(\frac{1}{\sqrt{c}} q_{si}^{h\top} k_{ti}^h\right), \quad m_{si} \mathrel{+}= \mathrm{(gated\ aggregation)}$$

bias 없음 — 시퀀스간 진화 관계만으로 attention.

### 4.4 Outer-product mean (MSA → pair)

$$z_{ij} \mathrel{+}= \mathrm{Linear}\!\left(\mathrm{flatten}\!\Big( \frac{1}{s}\sum_{s'} \mathrm{Linear}_a(m_{s'i}) \otimes \mathrm{Linear}_b(m_{s'j}) \Big)\right)$$

차원 분석: $(c_a) \otimes (c_b) = c_a c_b$ → flatten → linear → $c_z$. coevolution을 명시적으로 추출.

### 4.5 Triangle multiplicative update / 삼각 곱셈 갱신

Outgoing edges:
$$z_{ij} \mathrel{+}= \mathrm{Linear}\!\Big(\sigma(\text{gate}) \odot \sum_k a_{ik} \odot b_{jk}\Big), \quad a_{ik} = \mathrm{Linear}_a(z_{ik}),\; b_{jk} = \mathrm{Linear}_b(z_{jk})$$

Incoming edges (반대 방향의 합):
$$z_{ij} \mathrel{+}= \mathrm{Linear}\!\Big(\sigma(\text{gate}) \odot \sum_k a_{ki} \odot b_{kj}\Big)$$

**삼각 부등식과의 관계**: $z_{ij}$가 거리 분포의 logit이라면, $i$-$k$와 $j$-$k$ 거리가 결정될 때 $i$-$j$ 거리는 $|d_{ik} - d_{jk}| \le d_{ij} \le d_{ik} + d_{jk}$. 곱셈 form은 두 변수의 결합을 효과적으로 모델링.

### 4.6 Triangle self-attention / 삼각 self-attention

Around starting node $i$:
$$z_{ij} \mathrel{+}= \sum_k \mathrm{softmax}_k\!\left(\frac{1}{\sqrt{c}} q_{ij}^\top k_{ik} + b_{jk}\right) v_{ik}$$

같은 시작점 $i$를 공유하는 엣지들끼리 attention. extra logit bias $b_{jk}$ 가 있어 "missing edge"를 채워 넣는 효과 (저자 표현).

### 4.7 Invariant Point Attention (IPA) / 불변 점 주목

핵심 식 (head 인덱스 생략):
$$a_{ij} = \mathrm{softmax}_j\!\left(\frac{w_C}{\sqrt{c}} q_i^\top k_j \;-\; \frac{w_C \gamma}{2}\sum_p \|T_i \vec{q}_i^p - T_j \vec{k}_j^p\|^2 \;+\; w_L b_{ij}\right)$$

$w_C, w_L$은 정규화 상수 ($w_C = \sqrt{1/(c_h + N_{\text{point}}\gamma_p)}$ 등 — Supplementary 1.8.1).

**Scalar value 합성**:
$$o_i = \sum_j a_{ij} \, v_j$$

**Point value 합성** (잔기 $j$의 local frame 점을 잔기 $i$의 local frame으로 가져옴):
$$\vec{o}_i^p = T_i^{-1} \cdot \sum_j a_{ij} \, (T_j \cdot \vec{v}_j^p)$$

**SE(3) invariance 증명**:
전역 회전·병진 $T_g$ 적용 시 모든 frame이 $T_g T_i$로 변환됨. 점 거리 $\|T_g T_i \vec{q}_i^p - T_g T_j \vec{k}_j^p\|^2 = \|T_g (T_i \vec{q}_i^p - T_j \vec{k}_j^p)\|^2 = \|T_i \vec{q}_i^p - T_j \vec{k}_j^p\|^2$ ($T_g$의 회전은 거리 보존). 따라서 logit, 그리고 attention weight $a_{ij}$는 변하지 않음. point value 합성도 $(T_g T_i)^{-1} (T_g T_j) = T_i^{-1} T_j$로 cancel out.

### 4.8 FAPE Loss / 프레임 정렬 점 오차

$$\mathcal{L}_{\text{FAPE}} = \frac{1}{Z} \sum_{i=1}^{N_{\text{frames}}} \sum_{j=1}^{N_{\text{atoms}}} \min\!\big(d_{\max},\, \|T_i^{-1} \cdot x_j - \hat{T}_i^{-1} \cdot \hat{x}_j\|_2\big)$$

- $T_i = (R_i, t_i)$ 예측 frame, $\hat{T}_i$ 정답 frame.
- $T_i^{-1} \cdot x = R_i^\top (x - t_i)$.
- Clipping: $d_{\max} = 10$ Å (training), $d_{\max} = 30$ Å (fine-tuning).
- 정규화 $Z = N_{\text{frames}} \cdot N_{\text{atoms}} \cdot d_{\max}$ (논문 식에 따라).

### 4.9 Worked example: One refinement step / 한 단계 정제 워크스루

5-잔기 토이 단백질을 가정합니다.

**초기 상태 (residue gas)**:
- $T_i^{(0)} = (I, \mathbf{0})$ for all $i = 1, \ldots, 5$. 모든 잔기가 원점에 겹쳐 있음.
- single representation $s_i^{(0)} \in \mathbb{R}^{c_s}$는 Evoformer 출력.

**IPA 한 head, $N_{\text{point}}=1$ 가정**:
1. 각 잔기에서 query/key/value scalar $q_i, k_i, v_i \in \mathbb{R}^c$와 point $\vec{q}_i, \vec{k}_i, \vec{v}_i \in \mathbb{R}^3$ 생성. 처음에는 모든 frame이 identity이므로 $T_i \vec{q}_i = \vec{q}_i$.
2. 가정: 학습된 결과 $\vec{q}_i = (i, 0, 0)$, $\vec{k}_j = (j, 0, 0)$. 그러면 거리 $\|(i, 0, 0) - (j, 0, 0)\| = |i - j|$.
3. attention logit (head $\gamma = 1$, scalar 항·bias 무시):
$$a_{ij} \approx -\tfrac{1}{2}(i - j)^2$$
4. 잔기 $i=3$의 attention weight (softmax over $j=1..5$):
   - $j=3$에 가장 큰 weight (logit 0), $j=2,4$ 다음, $j=1,5$ 가장 작음.
5. point value 합성:
$$\vec{o}_3 = \sum_j a_{3j} \cdot \vec{v}_j$$
6. 이 출력을 단일 표현에 더함, 다음으로 $(\Delta R_3, \Delta t_3)$를 예측. 가정: $\Delta t_3 = (0, 0.5, 0)$로 잔기 3을 살짝 위로 이동.
7. frame 갱신: $T_3^{(1)} = (\Delta R_3 R_3^{(0)}, \Delta R_3 t_3^{(0)} + \Delta t_3) = (I, (0, 0.5, 0))$.
8. 8 블록을 반복하면 잔기들이 점차 분리되어 helix/sheet 패턴으로 정렬.

**FAPE 평가** (간단화: $N_{\text{atoms}} = N_{\text{frames}} = 5$, atom $j$ = 잔기 $j$의 Cα):
- 정답: $\hat{x}_j = (j, 0, 0)$ (선형 helix 가정), $\hat{T}_j = (I, (j, 0, 0))$.
- 예측 step 1: $T_3 = (I, (0, 0.5, 0))$, $x_3 = (0, 0.5, 0)$. 모든 $T_i, T_{\ne 3}$는 origin.
- frame $i=1$에서 본 atom $j=3$ 오차: $\|T_1^{-1} x_3 - \hat{T}_1^{-1} \hat{x}_3\| = \|(0, 0.5, 0) - (3-1, 0, 0)\| = \|(−2, 0.5, 0)\| \approx 2.06$ Å. clip 안 됨.
- 8 블록 후: 잔기들이 1Å 간격으로 분리되며 모든 frame-atom pair에서 오차 ~0.5 Å 이하로 수렴 (이상적인 경우).

**핵심 통찰**: FAPE는 단순 RMSD와 달리 모든 frame-atom pair($N^2$ 개)를 본다. 잔기 1의 frame에서 잔기 5의 위치, 잔기 3의 frame에서 잔기 5의 위치 등이 모두 일관되어야 함 → side-chain 방향과 chirality가 자연스럽게 학습됨.

### 4.10 Parameter and compute budget / 매개변수와 연산량

- 전체 모델 ~21M 매개변수 (Evoformer ~16M + Structure Module ~5M, 추정).
- Training: 16 TPUv3 cores × ~1주.
- Inference: 384-residue 단백질 ~1 GPU-minute (V100).
- Self-distillation: ~350k 시퀀스 × ~1분 = ~6 GPU-days/cycle.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958 ─ Kendrew: First X-ray structure of myoglobin (Nobel 1962)
        │
1972 ─ Anfinsen: Thermodynamic hypothesis (sequence → structure)
        │
1973 ─ Anfinsen Nobel lecture: "Principles that govern the folding..."
        │
1986 ─ Rumelhart et al.: Backpropagation (paper #6)
        │
1994 ─ CASP1 begins (biennial blind benchmark)
        │
1997 ─ Hochreiter & Schmidhuber: LSTM (paper #9)
        │
2003 ─ Rosetta (Baker lab) — fragment assembly, the de-facto standard
        │
2009 ─ Marks et al.: Direct coupling analysis on MSAs
        │
2012 ─ Krizhevsky: AlexNet (paper #13) — CNN revolution
        │
2014 ─ Bahdanau: Attention (paper #17)
        │
2017 ─ Vaswani: Transformer (paper #25) — direct ancestor of Evoformer
        │
2017 ─ Kipf & Welling: GCN (paper #26) — graph reasoning
        │
2018 ─ AlphaFold 1 (CASP13): distance map → optimization, GDT ~58
        │
2020 ─ Senior et al.: AlphaFold 1 paper in Nature
        │
2020 Dec ─ ★★★ AlphaFold 2 wins CASP14 with median GDT 92.4 ★★★
        │
2021 Jul ─ Jumper et al. (this paper) published in Nature
        │
2021 Jul ─ Tunyasuvunakool et al.: human proteome (companion paper)
        │
2022 ─ AlphaFold DB: ~200M structures (>99% UniProt)
        │
2023 ─ ESMFold (Meta) — language-model-only, ~14× faster
2023 ─ RFdiffusion (Baker lab) — generative protein design
        │
2024 ─ AlphaFold 3 — ligands, DNA, RNA via diffusion
2024 ─ Hassabis & Jumper share Nobel Prize in Chemistry
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#6 Rumelhart et al. (1986) — Backpropagation** | AlphaFold 학습은 표준 backprop + Adam / Standard backprop and Adam underpin all training | High — 학습의 기초 / Foundational |
| **#13 Krizhevsky et al. (2012) — AlexNet** | "Deep learning이 도메인을 깨뜨릴 수 있다" 패러다임 / Established the DL-breaks-domain paradigm | Medium — 정신적 선조 / Spiritual ancestor |
| **#17 Bahdanau et al. (2014) — Attention** | Evoformer와 IPA가 attention을 핵심으로 사용 / Attention is the central operation in Evoformer and IPA | High — 직접 후예 / Direct descendant |
| **#18 Kingma & Ba (2014) — Adam** | AlphaFold 학습 옵티마이저 / Optimizer for AlphaFold training | Medium — 실용 도구 / Practical tool |
| **#19 Ioffe & Szegedy (2015) — Batch Norm** | LayerNorm이 Evoformer 모든 attention 후에 사용 / LayerNorm follows every attention block | Medium — 정규화 전통 / Same normalisation tradition |
| **#20 He et al. (2015) — ResNet** | Residual connection이 Evoformer/Structure Module 전체에 사용 / Residuals are pervasive | High — 깊은 모델의 필수 / Required for depth |
| **#25 Vaswani et al. (2017) — Transformer** | Evoformer의 attention 패턴은 Transformer의 axial/triangular 변형 / Evoformer uses axial + triangular variants of Transformer attention | Very High — 직접 청사진 / Direct blueprint |
| **#26 Kipf & Welling (2017) — GCN** | Pair representation을 잔기 그래프의 엣지로 보는 graph 추론 관점 / Pair representation is the edge feature of a residue graph | High — graph 관점 / Graph-reasoning perspective |
| **#27 Goodfellow et al. (2014) — GAN** (if curated) | Self-distillation은 GAN과 다르지만 generative-discriminative 사이클의 사촌 / Self-distillation is a distant cousin of GAN cycles | Low — 영감 수준 / Inspirational only |
| **AlphaFold 1 (Senior et al., 2020)** | 직접 선조. distance map만 예측 → AlphaFold 2가 end-to-end 좌표 예측으로 도약 / Direct predecessor — distance maps replaced by end-to-end coordinate prediction | Very High — 직계 / Direct lineage |
| **AlphaFold 3 (Abramson et al., 2024)** | Diffusion-based 일반화. ligand, DNA, RNA 추가 / Diffusion-based generalisation; adds ligands, DNA, RNA | High — 직계 후예 / Direct successor |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589. DOI: 10.1038/s41586-021-03819-2.
- Code: https://github.com/google-deepmind/alphafold
- AlphaFold DB: https://alphafold.ebi.ac.uk

### Direct predecessors / 직접 선조
- Senior, A. W., et al. (2020). Improved protein structure prediction using potentials from deep learning. *Nature*, 577, 706–710. (AlphaFold 1)
- Yang, J., et al. (2020). Improved protein structure prediction using predicted interresidue orientations. *PNAS*, 117, 1496–1503. (trRosetta)
- Anfinsen, C. B. (1973). Principles that govern the folding of protein chains. *Science*, 181, 223–230.

### Foundational architecture / 기초 아키텍처
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Bahdanau, D., et al. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.

### Coevolution & MSA methods / 공진화와 MSA 방법
- Marks, D. S., et al. (2011). Protein 3D structure computed from evolutionary sequence variation. *PLoS ONE*, 6, e28766.
- Jones, D. T., et al. (2012). PSICOV: precise structural contact prediction using sparse inverse covariance estimation. *Bioinformatics*, 28, 184–190.
- Steinegger, M., et al. (2019). Protein-level assembly increases protein sequence recovery from metagenomic samples manyfold. *Nature Methods*, 16, 603–606.
- Mirdita, M., et al. (2017). Uniclust databases of clustered and deeply annotated protein sequences. *NAR*, 45, D170–D176.

### Datasets / 데이터셋
- wwPDB Consortium (2019). Protein Data Bank: the single global archive for 3D macromolecular structure data. *NAR*, 47, D520–D528.
- CASP14: Kryshtafovych, A., et al. (2021). Critical assessment of methods of protein structure prediction (CASP14). *Proteins*, 89, 1607–1617.

### Evaluation metrics / 평가 지표
- Zemla, A. (2003). LGA: a method for finding 3D similarities in protein structures. *NAR*, 31, 3370–3374. (GDT)
- Mariani, V., et al. (2013). lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. *Bioinformatics*, 29, 2722–2728.
- Zhang, Y., & Skolnick, J. (2004). Scoring function for automated assessment of protein structure template quality. *Proteins*, 57, 702–710. (TM-score)

### Companion / 후속
- Tunyasuvunakool, K., et al. (2021). Highly accurate protein structure prediction for the human proteome. *Nature*, 596, 590–596.
- Evans, R., et al. (2021). Protein complex prediction with AlphaFold-Multimer. *bioRxiv*.
- Abramson, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630, 493–500.
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379, 1123–1130. (ESMFold)

### Force field for relaxation / 이완용 힘장
- Hornak, V., et al. (2006). Comparison of multiple Amber force fields and development of improved protein backbone parameters. *Proteins*, 65, 712–725.

### Conceptually related deep learning papers / 개념적 관련 논문
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*. (Masked LM objective)
- Xie, Q., et al. (2020). Self-training with noisy student improves ImageNet classification. *CVPR 2020*. (Self-distillation)
- Carreira, J., et al. (2016). Human pose estimation with iterative error feedback. *CVPR 2016*. (Iterative refinement, recycling)
