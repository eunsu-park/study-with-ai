---
title: "Computing Machinery and Intelligence"
authors: Alan M. Turing
year: 1950
journal: "Mind, 59(236), 433–460"
topic: Philosophy of AI
tags: [turing test, imitation game, machine intelligence, philosophy of mind, learning machines]
status: completed
date_started: 2026-02-28
date_completed: 2026-02-28
---

# Paper #2: Computing Machinery and Intelligence

## Paper Info
- **Author**: Alan M. Turing
- **Published**: 1950, *Mind*, Vol. 59, No. 236, pp. 433–460
- **Pages**: 28 pages
- **Citation**: Turing, A.M. (1950). "Computing Machinery and Intelligence." *Mind*, 59(236), 433–460.

---

## Core Contribution (One-paragraph summary)

Rather than attempting to define "thinking," Turing replaced the question "Can machines think?" with a concrete, operational test: the Imitation Game (now known as the Turing Test). He then systematically addressed nine objections to the possibility of machine intelligence, and proposed that the most promising path forward is not to directly program adult intelligence, but to create a "child machine" that can *learn* — anticipating machine learning, reinforcement learning, and even evolutionary computation by decades.

---

## Reading Notes

### Section 1: The Imitation Game (pp. 433–434)

**The core move — replacing "thinking" with a test:**
- "Can machines think?" is too vague — the words "machine" and "think" are ambiguous
- Instead, Turing proposes a game with three players:
  - **A** (the machine, trying to deceive)
  - **B** (a human, trying to help)
  - **C** (the interrogator, trying to tell them apart)
- Communication is text-only (teleprinter) — no voice, no appearance
- **The question becomes**: "Can a machine play the part of A so well that C cannot reliably distinguish it from a human?"

**Why this formulation is brilliant:**
- It's *operationally testable* — no philosophical hand-waving needed
- It separates intellectual capacity from physical form (we don't penalize the machine for not being beautiful or running fast)
- It focuses on *behavior* rather than *inner experience*

**Turing's prediction (Section 6):**
> "I believe that in about fifty years' time it will be possible to programme computers... to make them play the imitation game so well that an average interrogator will not have more than 70 per cent chance of making the right identification after five minutes of questioning."

(By 2000 — roughly correct in spirit, though debatable in practice.)

### Section 2: Critique of the New Problem (pp. 434–435)

- The test cleanly separates physical from intellectual capabilities
- Sample dialogue shows how diverse the test can be: poetry, arithmetic, chess
- A machine playing the game might deliberately make arithmetic errors to seem more human — an early anticipation of AI alignment/deception issues!

### Section 3: The Machines Concerned in the Game (pp. 435–436)

- "Machine" is restricted to **digital computers** (not biological organisms grown in a lab)
- Turing explicitly wants to exclude humans born naturally — otherwise the answer is trivially "yes"
- He acknowledges this is a restriction but argues it's not a severe one, given the universality property

### Section 4: Digital Computers (pp. 436–438)

**The three components of a digital computer:**
1. **Store** — memory (corresponds to human computer's paper)
2. **Executive unit** — performs individual operations (arithmetic, etc.)
3. **Control** — ensures instructions are obeyed in correct order

**Key concepts introduced:**
- **Instruction table** = program (the "book of rules")
- **Programming** = constructing instruction tables
- **Conditional branching** — instructions like "if position 4505 contains 0, obey the instruction in 6707" — enables loops and decision-making
- **Digital computer with a random element** — a machine that can "throw dice," sometimes described as having "free will"

**Historical note:** Turing mentions Babbage's Analytical Engine (1828–1839) as the first digital computer concept, and notes that his own machine at Manchester had a storage capacity of about 165,000 bits.

### Section 5: Universality of Digital Computers (pp. 438–439)

**The universality argument — the paper's logical backbone:**
- Digital computers are **discrete-state machines** — they move between distinct states
- A universal digital computer can mimic *any* discrete-state machine, given enough storage and speed
- Therefore, the question "Can machines think?" reduces to: **"Can a single digital computer, with adequate storage and an appropriate program, play the imitation game satisfactorily?"**

This is a profound simplification: we don't need to build a special "thinking machine" — we just need the right program on a general-purpose computer.

### Section 6: Contrary Views on the Main Question (pp. 439–453)

**The heart of the paper.** Turing presents 9 objections and refutes each:

#### (1) The Theological Objection
- **Claim**: Thinking requires an immortal soul; God gave souls only to humans.
- **Turing's reply**: Theological arguments have historically been wrong about science (e.g., Galileo). Also, could God not give a soul to a machine if He chose to?

#### (2) The "Heads in the Sand" Objection
- **Claim**: The consequences of machines thinking are too dreadful; let's just believe they can't.
- **Turing's reply**: This is not worth refuting seriously. "Consolation would be more appropriate: perhaps this should be sought in the transmigration of souls."

#### (3) The Mathematical Objection
- **Claim**: Gödel's incompleteness theorem shows there are questions machines cannot answer.
- **Turing's reply**: Yes, any particular machine has limitations. But so does any particular human! "There might be men cleverer than any given machine, but then again there might be other machines cleverer again, and so on." The limitation applies to specific machines, not to machines in general.

#### (4) The Argument from Consciousness
- **Claim**: A machine cannot truly *feel* — write a sonnet and *know* it wrote it.
- **Turing's reply**: Taken to its logical extreme, this becomes solipsism (only I can know I think). In practice, we accept that other people think based on behavioral evidence — why not machines? He includes a witty imagined *viva voce* dialogue about Shakespeare.

#### (5) Arguments from Various Disabilities
- **Claim**: Machines can never "be kind, be resourceful, fall in love, enjoy strawberries and cream, make mistakes, learn from experience, do something really new..."
- **Turing's reply**: These objections are based on induction from limited experience with very primitive machines. A machine with enormous storage capacity would behave very differently. Also, "machines cannot make mistakes" confuses *errors of functioning* (hardware bugs) with *errors of conclusion* (wrong answers), which machines certainly can produce.

#### (6) Lady Lovelace's Objection
- **Claim**: "The Analytical Engine has no pretensions to *originate* anything. It can do whatever *we know how to order it* to perform." (Ada Lovelace, 1842)
- **Turing's reply**: A variant says machines can never "surprise" us. But they surprise Turing regularly! "I do not do sufficient calculation to decide what to expect them to do." More importantly, a universal machine *could* in principle simulate a machine that has the property of originality, even if we don't realize it.

#### (7) Argument from Continuity in the Nervous System
- **Claim**: The nervous system is continuous, not discrete; a discrete-state machine cannot mimic it.
- **Turing's reply**: In the imitation game, the interrogator cannot exploit this difference. A digital computer can approximate continuous behavior to any desired precision (e.g., approximating π ≈ 3.1416 with a probability distribution over nearby values).

#### (8) The Argument from Informality of Behaviour
- **Claim**: Human behavior cannot be captured by rules; therefore humans are not machines.
- **Turing's reply**: There's a confusion between "rules of conduct" (which we consciously follow, like traffic rules) and "laws of behaviour" (physical laws governing us). We can't easily convince ourselves that there are no laws of behaviour — the only way to find them is scientific observation. Turing set up a program on the Manchester computer that was unpredictable from its outputs alone, despite being deterministic.

#### (9) The Argument from Extrasensory Perception
- **Claim**: Telepathy and clairvoyance exist; machines don't have ESP, so they'd fail the test.
- **Turing's reply**: Surprisingly, Turing takes this somewhat seriously ("the statistical evidence, at least for telepathy, is overwhelming"). His solution: put all competitors in a "telepathy-proof room."

### Section 7: Learning Machines (pp. 453–460)

**The most visionary section of the paper.** Turing outlines a concrete research program for AI:

**The child machine idea:**
- Instead of programming a complete adult mind, build a simple "child machine" and *educate* it
- Three components of a mind: (a) initial state at birth, (b) education received, (c) other experience
- The child brain is like "a notebook... rather little mechanism, and lots of blank sheets"
- Analogy to evolution:
  - Structure of child machine = hereditary material
  - Changes to child machine = mutation
  - Natural selection = judgment of the experimenter

**Punishment and reward (= Reinforcement Learning!):**
- "The machine has to be so constructed that events which shortly preceded a punishment signal are unlikely to be repeated, whereas a reward signal increased the probability of repetition"
- This is exactly the principle of reinforcement learning — described 42 years before Sutton & Barto's textbook!

**Unemotional channels:**
- Beyond punishment/reward, the machine needs symbolic communication — orders given in language
- The machine should store propositions with different levels of certainty: "well-established facts," "conjectures," "mathematically proved theorems," "statements given by authority"
- Certain propositions are "imperatives" — the machine should act on them automatically

**Randomness in learning:**
- "It is probably wise to include a random element in a learning machine"
- Random search can be more efficient than systematic search when the solution space is large
- This anticipates stochastic gradient descent, random initialization, and evolutionary algorithms

**The final sentence:**
> "We can only see a short distance ahead, but we can see plenty there that needs to be done."

---

## Key Takeaways

1. **The Turing Test reframed AI**: By replacing "Can machines think?" with an operational test, Turing made the question scientifically tractable. The debate is no longer about metaphysics but about engineering and capability.

2. **Universality is the key insight**: Because a universal computer can simulate any discrete-state machine, we don't need specialized hardware for intelligence — just the right software. This justifies the entire field of AI as a *programming* challenge.

3. **The nine objections remain relevant today**: Almost every contemporary debate about AI (consciousness, creativity, Gödel limitations, alignment) was anticipated by Turing in 1950. His responses are still among the best available.

4. **Learning > Programming**: The most prescient part of the paper. Turing realized that hand-coding intelligence is impractical and that machines should *learn* — from punishment/reward, from examples, and from experience. This is the conceptual foundation for all of modern machine learning.

5. **The child machine = tabula rasa + education**: Turing's vision of starting with a simple learnable structure and training it on data is essentially the paradigm of modern deep learning: random initialization → training on data → emergent capabilities.

6. **Turing anticipated deception and alignment**: A machine playing the imitation game might deliberately introduce arithmetic errors to seem more human. This early observation about strategic deception in AI systems is remarkably relevant to modern AI safety research.

7. **Humility and boldness coexist**: Turing makes extraordinary claims (machines will think) while being scrupulously honest about what he doesn't know. His arguments are conjectures, not proofs — and he's clear about this distinction.

---

## Turing's Predictions: Scorecard (from 2026 perspective)

| Prediction | Status |
|---|---|
| Machines will play imitation game well by ~2000 | Partially true (chatbots pass weak versions, but not robustly) |
| "Can machines think?" will become a normal question | True — it's the central question of our era |
| Learning machines are the path forward | Completely correct — ML dominates AI |
| Punishment/reward training works | True — reinforcement learning is a major paradigm |
| Child machine + education > direct programming of adult mind | True — pre-training + fine-tuning is exactly this |
| Random elements aid learning | True — stochastic methods are essential |
| Machines will surprise their creators | Overwhelmingly true (emergent capabilities in LLMs) |

---

## Connections to Modern AI

| Turing (1950) | Modern Equivalent |
|---|---|
| Imitation Game | Turing Test, chatbot evaluations, LMSYS Arena |
| Child machine | Neural network with random initialization |
| Education process | Training on data (pre-training) |
| Punishment and reward | Reinforcement Learning, RLHF |
| Propositions with varying certainty | Probabilistic reasoning, Bayesian approaches |
| Imperatives from teacher | Instruction tuning, system prompts |
| Random element in learning | Stochastic gradient descent, dropout |
| Deliberate errors to seem human | AI alignment/deception concerns |
| "Skin of an onion" analogy for mind | Emergent properties from simple components |

---

## Note: No Code Implementation

This paper is philosophical — there are no algorithms to implement. Its value lies in the conceptual framework it established. The ideas proposed here (especially learning machines) will be realized concretely in the papers that follow, starting with Rosenblatt's Perceptron (Paper #3).

---

## References
- Turing, A.M. (1950). "Computing Machinery and Intelligence." *Mind*, 59(236), 433–460. [DOI: 10.1093/mind/LIX.236.433]
- Lovelace, A. (1842). Notes on the Analytical Engine.
- Gödel, K. (1931). "Über formal unentscheidbare Sätze." *Monatshefte für Mathematik und Physik*, 38, 173–198.
