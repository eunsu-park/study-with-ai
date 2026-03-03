---
title: "A Logical Calculus of the Ideas Immanent in Nervous Activity"
authors: Warren McCulloch, Walter Pitts
year: 1943
journal: Bulletin of Mathematical Biophysics, 5, 115–133
topic: Foundations of Neural Networks
tags: [neuron model, boolean logic, threshold logic, computational neuroscience]
status: completed
date_started: 2026-03-01
date_completed: 2026-03-01
---

# Paper #1: A Logical Calculus of the Ideas Immanent in Nervous Activity

## Paper Info
- **Authors**: Warren S. McCulloch (neurophysiologist), Walter Pitts (logician, mathematician)
- **Published**: 1943, Bulletin of Mathematical Biophysics
- **Pages**: 17 pages
- **Citation**: McCulloch, W.S., Pitts, W. (1943). "A Logical Calculus of the Ideas Immanent in Nervous Activity." *Bulletin of Mathematical Biophysics*, 5, 115–133.

---

## Core Contribution (One-paragraph summary)

McCulloch and Pitts showed that biological neurons, simplified as binary threshold units operating in discrete time, can be described using propositional logic. They proved that any logical function (AND, OR, NOT, and their combinations) can be realized by a network of such neurons, and that networks with feedback loops (circles) are equivalent to Turing machines — making the nervous system, in principle, a universal computer.

---

## Reading Notes

### Section 1: Introduction (pp. 99–101)

**Neurophysiological background:**
- The nervous system is a network of neurons, each with a soma and an axon
- Synapses connect the axon of one neuron to the soma of another
- A neuron has a **threshold** — excitation must exceed it to fire an impulse
- The impulse is **all-or-none**: the neuron either fires completely or not at all
- Once fired, a neuron enters a **refractory period** (~0.5 ms) where it cannot fire again
- **Synaptic delay**: >0.5 ms between input arrival and neuron response

**Two types of synapses:**
- **Excitatory**: promotes firing (contributes positively toward threshold)
- **Inhibitory**: prevents firing (the paper introduces "absolute inhibition" — a single inhibitory input completely prevents firing regardless of excitatory inputs)

**Key insight — Neuron as proposition:**
- Because neural activity is all-or-none, the activity of any neuron at any moment can be represented as a proposition (true = firing, false = not firing)
- Relations among neural activities correspond to relations among propositions in logic

### Section 2: The Theory — Nets Without Circles (pp. 101–108)

**The five fundamental assumptions:**
1. The activity of the neuron is an "all-or-none" process
2. A fixed number of synapses must be excited within the period of latent addition to excite a neuron (this number = threshold), independent of previous activity and position
3. The only significant delay in the nervous system is synaptic delay
4. The activity of any inhibitory synapse absolutely prevents excitation of the neuron at that time
5. The structure of the net does not change with time (no learning!)

**Notation:**
- Neurons labeled $c_1, c_2, \ldots, c_n$
- $N_i(t)$ = proposition that neuron $c_i$ fires at time $t$
- The action of neuron $c_i$ is denoted $N_i$
- Peripheral afferents = input neurons from the external world (no axons synapsing on them)

**Key concept — Temporal Propositional Expressions (TPE):**
- A TPE is a logical formula that describes neural activity over time
- Built from predicates $p(z_1)$ using disjunction, conjunction, negation, and the functor $S$ (which shifts time by one step)

**Theorem 1**: Every net of order 0 (no circles) can be described in terms of TPEs.
- This means feed-forward networks can be fully characterized by propositional logic expressions

**Theorem 2**: Every TPE is realizable by a net of order zero.
- Converse of Theorem 1: for any logical expression, you can build a neural network that computes it

**Theorem 3**: A complex sentence (logical expression) is a TPE if and only if it is false when all its atomic constituents are false (no "negated terms" only).

**Figure 1 — The key diagrams (p. 105):**
These show how simple neuron networks realize basic logical operations:
- **(a)**: Simple relay — $N_2(t) = N_1(t-1)$ (neuron 2 fires if neuron 1 fired one step ago)
- **(b)**: OR gate — $N_3(t) = N_1(t-1) \lor N_2(t-1)$ (fire if either input fired)
- **(c)**: AND gate — $N_3(t) = N_1(t-1) \land N_2(t-1)$ (fire only if both inputs fired)
- **(d)**: AND-NOT — $N_3(t) = N_1(t-1) \land \lnot N_2(t-1)$ (fire if input 1 fired but input 2 did not — uses inhibition)
- **(e)–(i)**: More complex combinations including temporal patterns and memory-like behavior

**Practical example — Heat/cold sensation (p. 106):**
- A cold object briefly held to skin → sensation of heat (transient cooling illusion)
- Longer application → sensation of cold only
- Modeled with heat receptor $N_1$, cold receptor $N_2$, and neurons $N_3$ (heat sensation), $N_4$ (cold sensation)
- Shows how network structure can create complex temporal perceptions from simple inputs

**Equivalence theorems (Theorems 4–7):**
- **Theorem 4**: Relative inhibition (raising threshold) and absolute inhibition are equivalent in the extended sense
- **Theorem 5**: Extinction (temporary inability after firing) is equivalent to absolute inhibition
- **Theorem 6**: Facilitation and temporal summation can be replaced by spatial summation
- **Theorem 7**: Alterable synapses (learning!) can be replaced by circles (feedback loops)
  - This is remarkable: it suggests that the effect of learning can be modeled by fixed networks with feedback

### Section 3: The Theory — Nets with Circles (pp. 108–113)

**What are circles?**
- Feedback loops where a chain of neurons forms a cycle ($c_1 \to c_2 \to \ldots \to c_p \to c_1$)
- Activity can reverberate around these loops indefinitely
- This introduces **memory**: the network can reference its own past states

**The order of a net** = the number of neurons in its largest cyclic set (i.e., the size of the feedback loop)

**Key results:**
- **Theorem 8**: For nets with circles, expressions can still be written in terms of TPEs of the cyclic neurons + the actions of other neurons
- **Theorems 9–10**: Characterize which classes of behavior ("prehensible classes") can be realized by nets with circles

**The Turing equivalence (p. 113):**
> "Every net, if furnished with a tape, scanners connected to afferents, and suitable efferents to perform the necessary motor-operations, can compute only such numbers as can a Turing machine; second, that each of the latter numbers can be computed by such a net."

This is the paper's most profound result: **McCulloch-Pitts neural networks are computationally equivalent to Turing machines** — they can compute anything that is computable.

### Section 4: Consequences (pp. 113–115)

**Philosophical implications:**
- Since the nervous system is equivalent to a logical calculus, "mental" activity can be rigorously deduced from neurophysiology
- Causality in the nervous system is **irreciprocal** (unlike in statistics) — the net's specification determines the next state from the current state
- Circular (feedback) networks make reference to the past **indefinite** — knowledge is inherently incomplete and uncertain
- The authors draw an analogy: hallucinations, delusions, and confusion can be understood as properties of the network, not as mysterious mental phenomena
- "Mind" is no longer "more ghostly than a ghost" — it can be understood in terms of the network

---

## Key Takeaways

1. **Neuron as logic gate**: By simplifying the biological neuron to a binary threshold unit, McCulloch and Pitts created a bridge between neuroscience and mathematical logic. This abstraction — crude yet powerful — is the ancestor of all artificial neurons.

2. **Universality**: Any Boolean function can be computed by a network of McCulloch-Pitts neurons (Theorems 1–2). With feedback loops, these networks become Turing-complete (can compute anything computable).

3. **Five assumptions that launched a field**: The simplifications (all-or-none, fixed threshold, synaptic delay, absolute inhibition, fixed structure) made the math tractable. Modern neural networks relax most of these — especially the fixed structure (learning!).

4. **No learning in this model**: Assumption 5 explicitly forbids structural change. The paper acknowledges learning exists (Theorem 7 shows alterable synapses can be replaced by feedback loops), but does not model it directly. This limitation would be addressed by Rosenblatt's perceptron (1958) and Hebb's learning rule (1949).

5. **Inhibition is fundamental**: The absolute inhibition assumption (assumption 4) is critical for computing NOT and therefore for universal computation. Without it, only monotone functions would be computable.

6. **Time is discrete**: The model operates in discrete time steps synchronized by synaptic delay. This is both a simplification and a precursor to the discrete time steps used in modern recurrent neural networks.

7. **Philosophical boldness**: The paper doesn't just propose a model — it makes a philosophical claim that the mind can be understood as a logical machine. This was radical in 1943.

---

## Historical Context

- **Before this paper**: No mathematical model of neural computation existed. The brain was studied purely through biology and philosophy.
- **After this paper**: Opened the door for:
  - Hebb's learning rule (1949) — how connections strengthen
  - Rosenblatt's Perceptron (1958) — adding learning to the McCulloch-Pitts model
  - Von Neumann's computer architecture — influenced by this paper's logical framework
  - The entire field of artificial neural networks

---

## Connections to Modern AI

| McCulloch-Pitts (1943) | Modern Neural Networks |
|---|---|
| Binary output (0 or 1) | Continuous activations (sigmoid, ReLU, etc.) |
| Fixed threshold | Learnable weights and biases |
| No learning | Backpropagation, gradient descent |
| Absolute inhibition | Negative weights |
| Discrete time steps | Continuous or batched computation |
| Logical equivalence proofs | Empirical/statistical performance |

---

## References
- McCulloch, W.S., Pitts, W. (1943). "A Logical Calculus of the Ideas Immanent in Nervous Activity." *Bulletin of Mathematical Biophysics*, 5, 115–133. [DOI: 10.1007/BF02478259]
- Carnap, R. (1938). *The Logical Syntax of Language*. New York: Harcourt-Brace.
- Hilbert, D. and Ackermann, W. (1927). *Grundzüge der Theoretischen Logik*. Berlin: Springer.
- Russell, B. and Whitehead, A.N. (1925). *Principia Mathematica*. Cambridge University Press.
