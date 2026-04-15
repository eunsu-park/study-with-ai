---
title: "Computing Machinery and Intelligence"
authors: Alan M. Turing
year: 1950
journal: "Mind, Vol. 59, No. 236, pp. 433–460"
doi: "10.1093/mind/LIX.236.433"
topic: Artificial Intelligence / Philosophy of Mind
tags: [turing test, imitation game, machine intelligence, philosophy of mind, learning machines, child machine]
status: completed
date_started: 2026-04-05
date_completed: 2026-04-05
---

# Computing Machinery and Intelligence (1950)
# 컴퓨팅 기계와 지능 (1950)

**Alan M. Turing**

---

## Core Contribution / 핵심 기여

"기계가 생각할 수 있는가?"라는 모호한 철학적 질문을 **Imitation Game**(현재 Turing Test)이라는 구체적이고 검증 가능한 테스트로 대체한 논문입니다. Turing은 디지털 컴퓨터의 universality(범용성)를 근거로 "올바른 프로그램만 있으면 어떤 컴퓨터든 지능적 행동을 할 수 있다"고 주장했습니다. 기계 지능에 대한 9가지 반론을 체계적으로 반박한 뒤, 성인 지능을 직접 프로그래밍하는 대신 **학습 가능한 "어린이 기계"(child machine)**를 만들어 교육시키는 것이 가장 유망한 경로라고 제안합니다. 이는 machine learning, reinforcement learning, evolutionary computation을 수십 년 앞서 예견한 놀라운 선견지명입니다.

Rather than debating the undefined term "thinking," Turing replaced "Can machines think?" with a concrete, operationally testable procedure — the Imitation Game. Grounding his argument in the universality of digital computers, he argued that intelligence is a software problem. After systematically refuting nine objections, he proposed building a "child machine" that learns through reward/punishment and education — anticipating modern ML, RL, and evolutionary methods by decades.

---

## Reading Notes / 읽기 노트

### Section 1: The Imitation Game (pp. 433–434) — 모방 게임

**핵심 전략 — "생각"을 정의하지 않고 테스트로 대체:**

Turing은 논문의 첫 문장에서 "Can machines think?"를 제시하지만, 즉시 이 질문이 무의미하다고 선언합니다. "machine"과 "think" 모두 일상적 용법(ordinary use of words)으로는 너무 모호하여 여론 조사(Gallup poll)로나 답할 수 있을 뿐 과학적 의미가 없다는 것입니다. 대신 그는 구체적인 **게임**을 제안합니다.

**원래 버전 (남/여) — 왜 남녀 게임으로 시작하는가:**
- A (남자): 여자인 척 속이려 함 — 거짓말을 해도 됨
- B (여자): 심문자가 올바른 판단을 하도록 도움 — 진실을 말함
- C (심문자): 텍스트만으로 누가 남자/여자인지 판단

Turing이 처음부터 "기계 vs 인간"이 아니라 "남자 vs 여자"로 시작하는 이유가 있습니다. 남녀 버전에서는 이미 **"속이는 자"와 "판별하는 자"**의 구조가 확립됩니다. 남자는 자신의 정체를 숨기기 위해 지적 능력을 사용해야 하고, 심문자는 텍스트만으로 이를 간파해야 합니다. 이 게임의 핵심은 **외적 차이(물리적 성별)를 텍스트 채널로 제거한 후, 순수한 지적 수행만으로 판별이 가능한가**를 묻는 것입니다.

**핵심 전환 — 남녀 게임에서 기계 게임으로:**

Turing은 다음과 같은 논리적 전환을 수행합니다: "이제 A(남자) 자리에 기계를 넣으면 어떻게 될까?" 이 전환의 논리적 구조는 다음과 같습니다:

1. 남녀 게임에서 남자가 여자를 성공적으로 흉내 낼 수 있다면, 그것은 "지적으로 여자와 구분할 수 없다"는 것을 의미합니다.
2. 마찬가지로, 기계가 인간을 성공적으로 흉내 낼 수 있다면, 그것은 "지적으로 인간과 구분할 수 없다"는 것을 의미합니다.
3. **지적으로 인간과 구분할 수 없는 존재를 "생각하지 않는다"고 거부할 합리적 근거가 있는가?**

이것이 Turing의 핵심 논증입니다. "생각한다"를 정의하는 대신, "생각하지 않는다고 주장할 근거가 없음"을 보여주는 전략입니다.

**텍스트 전용 통신(text-only communication)이 핵심인 이유:**

Turing은 게임을 "텔레프린터(teleprinter)"를 통해 수행해야 한다고 명시합니다. 이것은 단순히 "외모를 배제한다"는 수준이 아닙니다. 텍스트 전용 채널은 다음을 달성합니다:

- **공정한 경쟁의 장(level playing field)**: 물리적 외형, 음성, 몸짓 등 지적 능력과 무관한 모든 단서를 제거합니다. 남아 있는 것은 오직 **언어적 지능** — 논리적 추론, 지식, 유머, 감정 표현 능력뿐입니다.
- **지적 능력의 순수한 측정**: 만약 음성이나 영상이 허용된다면, 현재 기술로는 기계를 바로 식별할 수 있을 것입니다 (1950년대에는 더더욱). 하지만 이것은 기계의 지적 능력이 부족해서가 아니라 물리적 매체의 한계일 뿐입니다. Turing은 "기계가 생각할 수 있는가?"를 "기계가 인간처럼 생겼는가?"와 분리하고자 했습니다.
- **핵심 통찰**: 우리가 다른 인간의 지능을 판단하는 가장 풍부한 채널은 **언어**입니다. 텍스트로 충분히 지적 수행을 평가할 수 있다면, 추가적인 물리적 단서는 불필요합니다.

**이 설계가 천재적인 이유 요약:**
- **텍스트만 사용** → 외모, 목소리를 배제하여 순수한 지적 능력만 평가하는 공정한 경쟁의 장 구축
- **"생각"을 정의하지 않음** → 정의 논쟁을 우회하여 행동(performance)으로 판단 — 조작적 정의(operational definition)의 도입
- **속이는 게임** → "진짜로 생각하는가?"가 아니라 "구분할 수 있는가?"로 전환 — 관찰 불가능한 내면 상태 대신 관찰 가능한 행동에 초점

### Section 2: Critique of the New Problem (pp. 434–435) — 새 문제의 비평

Turing이 제시한 예시 대화:

```
C: "Forth Bridge"에 대한 소네트를 써 주세요.
A: 사양하겠습니다. 저는 시를 쓸 줄 모릅니다.

C: 34957에 70764를 더하면?
A: (약 30초 정지)... 105621

C: 체스를 하시나요?
A: 네.  [이어서 R-R8, 체크메이트.]
```

**핵심 관찰:**
- 시를 못 쓴다고 답함 — 너무 잘하면 오히려 기계로 의심받음
- 산술을 **느리게, 일부러 틀림** (정답은 105721) — 인간은 큰 수 덧셈을 느리게 하고 실수함
- Turing은 이런 **의도적 오류/지연**을 이미 예견 — 현대 AI alignment의 "deception" 문제의 원형

### Section 3: The Machines Concerned (pp. 435–436) — 대상 기계

- "Machine"을 **디지털 컴퓨터**로 한정 (생물학적 존재 제외)
- 실험실에서 인간을 키우면 답이 자명하게 "yes"이므로 제외
- Universality 덕분에 특정 하드웨어에 한정할 필요 없음

### Section 4: Digital Computers (pp. 436–438) — 디지털 컴퓨터

**"Human computer" 비유 — 디지털 컴퓨터의 출발점:**

Turing은 디지털 컴퓨터를 설명하기 위해 **"human computer"(인간 계산원)**와의 비유에서 시작합니다. 이것은 단순한 비유가 아닙니다. 19세기와 20세기 초에 "computer"는 **직업명**이었습니다 — 계산을 수행하는 사람을 의미했습니다. Turing은 다음과 같이 묘사합니다:

- 인간 계산원은 **고정된 규칙(instruction table)**을 따릅니다
- 종이 위에 계산 과정을 기록합니다 (= Store)
- 덧셈, 곱셈 같은 개별 연산을 수행합니다 (= Executive Unit)
- 다음에 어떤 규칙을 적용할지 결정합니다 (= Control)

핵심은 이것입니다: **디지털 컴퓨터가 하는 일은 인간 계산원이 종이와 펜으로 하는 일과 정확히 같습니다.** 차이점은 오직 **속도**뿐입니다. 기계는 인간이 평생 걸릴 계산을 몇 초에 해냅니다. 이 비유가 중요한 이유는, "기계가 계산할 수 있는가?"라는 질문이 사실 "기계가 인간 계산원을 대체할 수 있는가?"와 동일하며, 이에 대한 답은 자명하게 "예"이기 때문입니다.

**디지털 컴퓨터의 세 가지 구성 요소:**

| Turing의 용어 | 현대적 대응 | 역할 |
|---|---|---|
| **Store** | Memory (RAM) | 데이터와 명령어를 저장 |
| **Executive Unit** | CPU (ALU) | 덧셈, 비교 등 개별 연산 수행 |
| **Control** | Control Unit / Stored Program | Store에서 명령어를 순서대로 읽어 실행 |

**핵심 개념:**

**Instruction table = program (명령어 표):**
인간 계산원이 따르는 규칙표와 정확히 같습니다. 각 명령어는 "이 상태에서 이 값을 보면, 이 연산을 수행하고, 다음 명령어로 이동하라"는 형태입니다.

**Stored-program concept — 프로그램과 데이터가 같은 메모리에 존재:**
이것이 왜 혁명적이었는가? 이전 컴퓨터(예: ENIAC)에서는 프로그램을 바꾸려면 **물리적 배선을 변경**해야 했습니다 — 케이블을 뽑고 다시 연결하는 데 며칠이 걸렸습니다. Stored-program concept에서는 프로그램도 데이터와 마찬가지로 **Store(메모리)에 숫자로 저장**됩니다. 프로그램을 바꾸려면 메모리 내용만 교체하면 됩니다 — 물리적 변경 없이. 더 나아가, 프로그램이 데이터와 같은 형식이므로 **프로그램이 다른 프로그램을 수정하거나 생성**할 수 있습니다. 이것이 바로 컴파일러, 인터프리터, 그리고 궁극적으로 자기 수정 프로그램(self-modifying programs)의 기반입니다. Turing의 universality 논증도 이것에 의존합니다 — 범용 기계가 다른 기계를 시뮬레이션하려면, 그 "다른 기계의 프로그램"을 자신의 메모리에 데이터로 읽어들여야 하기 때문입니다.

**Conditional branching — 구체적 예시:**
Turing은 다음과 같은 구체적 예를 제시합니다: "만약 주소 4505의 값이 0이면, 다음 명령어는 6707번이다. 0이 아니면 순서대로 다음 명령어를 실행하라." 이것이 중요한 이유는, 조건 분기가 없으면 컴퓨터는 명령어를 위에서 아래로 순서대로만 실행할 수 있어, 본질적으로 계산기에 불과합니다. 조건 분기가 있으면 **이전 계산의 결과에 따라 다른 경로를 선택**할 수 있고, 이것이 루프(반복)와 의사결정의 기초입니다. Turing은 이것이야말로 "intelligence"를 가능하게 하는 핵심 메커니즘이라고 암시합니다 — 기계가 상황에 따라 다르게 행동할 수 있게 해주기 때문입니다.

**Random element:**
Turing은 전자적 무작위 장치(주사위를 던지는 것과 같은)를 포함할 수 있다고 언급하며, 이런 기계가 "자유의지"를 가진다고 표현할 수도 있다고 합니다. 이것은 나중에 Section 7의 학습 기계 논의에서 중요해집니다.

**역사적 언급:** Babbage의 Analytical Engine (1828–1839)을 최초의 디지털 컴퓨터 개념으로 인용. 본인의 Manchester 기계는 약 165,000 bits 저장 용량.

### Section 5: Universality of Digital Computers (pp. 438–439) — 범용성

**논문의 논리적 기둥 — 왜 Universality가 전체 논증의 핵심인가:**

Universality는 단순히 "컴퓨터가 다재다능하다"는 것이 아닙니다. 이것이 없으면 Turing의 전체 논증이 무너집니다. 만약 범용성이 성립하지 않는다면, "체스를 두는 기계", "시를 쓰는 기계", "대화하는 기계" 등 각각의 능력에 대해 별도의 전문 기계를 만들어야 합니다. 그러면 "기계가 생각할 수 있는가?"라는 질문은 "어떤 기계?"로 분산되어 대답할 수 없게 됩니다. 범용성 덕분에 우리는 **단 하나의 기계**에 대해 논의할 수 있습니다.

**Discrete-state machine (이산 상태 기계) 개념:**

Turing은 디지털 컴퓨터를 **discrete-state machine**으로 정의합니다. 이것은 다음을 의미합니다:

- **유한한 수의 구별 가능한 상태(finite distinguishable states)**를 가집니다. 각 순간에 기계는 이 상태들 중 정확히 하나에 있습니다.
- **결정론적 전이(deterministic transitions)**: 현재 상태와 입력이 주어지면, 다음 상태가 **유일하게** 결정됩니다. (무작위 요소가 있는 경우는 예외이지만, Turing은 이를 별도로 처리합니다.)
- **예측 가능성(predictability)**: 초기 상태를 알면, 이론적으로 미래의 모든 상태를 계산할 수 있습니다. Turing은 이것을 "그 미래를 계산하는 데 불합리한 시간이 걸릴 수 있지만, 원칙적으로 가능하다"고 합니다.

이것은 아날로그 기계(예: 미분 해석기)와의 핵심 차이입니다. 아날로그 기계는 연속적 상태를 가지므로 초기 조건의 미세한 차이가 예측 불가능한 결과를 낳을 수 있습니다 (나비 효과). 디지털 기계는 이산적이므로 "거의 같은" 상태가 없습니다 — 같거나 다르거나 둘 중 하나입니다.

**Universal digital computer:**

- 충분한 저장 공간만 있으면 **어떤** discrete-state machine이든 시뮬레이션 가능
- 이것은 Turing이 1936년 논문 "On Computable Numbers"에서 이미 증명한 결과입니다. Universal Turing Machine은 다른 Turing Machine의 description(설명, 즉 프로그램)을 입력으로 받아 그 기계의 동작을 정확히 재현합니다.

**Imitation Game 논증과의 연결:**

따라서 "기계가 생각할 수 있는가?"는 다음으로 환원됩니다:

> **"단일 디지털 컴퓨터가 적절한 프로그램으로 Imitation Game을 만족스럽게 수행할 수 있는가?"**

이 연결의 논리를 더 명확히 하면:

1. 만약 어떤 기계 M이 Imitation Game을 통과할 수 있다면, M은 "생각한다"고 볼 수 있습니다.
2. 범용 컴퓨터 U는 적절한 프로그램을 주면 M을 시뮬레이션할 수 있습니다.
3. 따라서 U도 (M을 시뮬레이션함으로써) Imitation Game을 통과할 수 있습니다.
4. 결론: **이미 존재하는 범용 컴퓨터가 잠재적으로 "생각하는 존재"가 될 수 있습니다.**

이것은 AI를 **프로그래밍 문제**로 환원한 것입니다. 특별한 "생각하는 기계"를 새로 발명할 필요 없이, 범용 컴퓨터에 올바른 프로그램만 있으면 됩니다. 특별한 "생각하는 하드웨어"가 필요 없다는 이 통찰은, 이후 AI 연구의 전체 방향을 결정했습니다 — 하드웨어가 아닌 **알고리즘과 프로그램**이 핵심이라는 것입니다.

### Section 6: Contrary Views — 9가지 반론 (pp. 439–453)

**논문의 핵심 섹션.** Turing은 9가지 반론을 제시하고 각각 반박합니다:

#### (1) Theological Objection / 신학적 반론
- **주장**: 사고는 불멸의 영혼의 기능이다. 신은 모든 남녀에게 영혼을 부여했지만, 동물이나 기계에는 부여하지 않았다. 따라서 어떤 기계도 생각할 수 없다.
- **Turing의 반박**: 두 가지 경로로 반박합니다.
  - **(a) 역사적 선례**: 신학적 논증은 과학에 대해 반복적으로 틀려왔습니다. 코페르니쿠스와 갈릴레이의 지동설은 "신이 지구를 중심에 놓았다"는 신학적 주장과 충돌했지만, 결국 과학이 옳았습니다. Turing은 "나는 신학자들이 필요한 수정을 충분히 할 수 있다고 확신한다"고 풍자합니다.
  - **(b) 전능의 역설**: 전능한 신은 원한다면 코끼리에게도 영혼을 줄 수 있습니다. 마찬가지로, 기계에도 영혼을 부여할 수 있지 않겠습니까? "기계에 영혼을 줄 수 없다"고 주장하는 것은 오히려 신의 전능성을 제한하는 것입니다. Turing은 이 반론이 "다른 종교 공동체에서는 매우 다르게 보일 것"이라며, 이슬람교에서는 여성에게 영혼이 없다는 관점이 있었음을 지적합니다.

#### (2) "Heads in the Sand" Objection / "머리를 모래에 묻기" 반론
- **주장**: 기계가 생각할 수 있다는 결론은 너무 끔찍하다(dreadful). 그런 일이 일어나지 않기를 바라며, 따라서 그럴 리 없다.
- **Turing**: 이것은 논증이 아니라 감정적 반응입니다. Turing은 이 반론이 지식인(intellectual people) 사이에서 가장 흔하다고 관찰합니다 — 자신의 지적 우월성에 대한 위협을 느끼기 때문입니다. 그는 이 반론을 진지하게 반박할 가치가 없다고 판단하며, "위안이 필요하다면, 영혼의 윤회(transmigration of souls)를 찾아보시라"고 풍자합니다. 이 반론은 현대에도 반복됩니다 — AI가 인간의 일자리나 창의성을 위협한다는 공포감이 그 예입니다.

#### (3) Mathematical Objection / 수학적 반론
- **주장**: Gödel의 불완전성 정리, Church의 증명, Turing 자신의 계산 불가능성 결과에 따르면, 모든 충분히 강력한 형식 체계에는 **그 체계 내에서 증명할 수도 반증할 수도 없는 명제**가 존재합니다. 디지털 컴퓨터는 형식 체계이므로, 기계가 절대 올바르게 답할 수 없는 질문이 반드시 존재합니다. 이것은 기계의 지능에 근본적 한계를 설정합니다.

- **Gödel의 불완전성 정리 간략 설명**: 1931년 Kurt Gödel은 다음을 증명했습니다: (1) 산술을 포함하는 어떤 일관된(consistent) 형식 체계에서든, 그 체계 내에서 참이지만 증명 불가능한 명제가 존재한다. (2) 그 체계의 일관성 자체를 체계 내에서 증명할 수 없다. 쉽게 말하면, 어떤 규칙 체계든 "내가 이 체계 내에서는 증명할 수 없다"는 형태의 진술을 구성할 수 있으며, 이런 진술은 실제로 참이지만 그 체계 내에서 증명 불가능합니다.

- **Turing의 반박**: Turing은 이 한계를 인정합니다 — 실제로 이것은 그 자신의 1936년 연구(halting problem)와 밀접한 관련이 있습니다. 하지만 그의 반박은 간결하면서도 강력합니다:

  **(a) 이것은 특정 기계에 대한 한계이지, 기계 일반에 대한 한계가 아닙니다.** 기계 A가 답할 수 없는 질문이 있다고 해서, 기계 B도 그 질문에 답할 수 없는 것은 아닙니다. 다른 기계는 다른 Gödel 문장을 가집니다.

  **(b) 인간에게도 동일한 한계가 적용됩니다.** "기계에 한계가 있다"고 주장하려면, 인간에게는 그런 한계가 없음을 보여야 합니다. 하지만 인간도 틀린 답을 합니다! Turing은 이렇게 말합니다: "어떤 기계보다 더 영리한 인간이 있을 수 있지만, 그 인간보다 더 영리한 기계도 있을 수 있고, 그렇게 계속된다(and so on)." 기계의 수학적 한계를 기계 지능에 대한 반론으로 사용하려면, 인간이 그 한계를 넘어설 수 있음을 증명해야 하는데, 그런 증명은 없습니다.

  **(c) Imitation Game에서는 이 한계가 문제가 되지 않습니다.** Turing Test에서 심문자가 Gödel 문장을 물을 가능성은 낮으며, 설사 묻더라도 인간도 정확히 답하지 못할 것입니다.

#### (4) Argument from Consciousness / 의식 논증
- **주장**: Geoffrey Jefferson 교수의 1949년 Lister Oration에서 인용: "기계가 단순히 기호를 배열해서 소네트를 쓰는 것이 아니라, 감정을 느꼈기 때문에(because of thoughts and emotions felt) 소네트를 쓸 수 있기 전까지는, 기계가 뇌와 동등하다고 인정할 수 없다." 즉, 진정한 사고(thinking)에는 **의식적 경험(conscious experience)**이 필수적이다.

- **Turing의 가상 대화 — 소네트에 대한 심문:**

Turing은 다음과 같은 가상 대화를 제시합니다:

```
심문자: 당신의 소네트 첫 줄에서 "Shall I compare thee to a summer's day"가 아니라
       "a spring day"로 해도 되지 않았을까요?
증인:   리듬이 맞지 않습니다.
심문자: "a winter's day"는요? 그것도 리듬은 맞잖아요.
증인:   네, 하지만 아무도 겨울날에 비유되고 싶어하지 않습니다.
심문자: Mr. Pickwick이 크리스마스를 연상시킨다고 하지 않을까요?
증인:   어떤 면에서는 그렇겠지만, 이것은 심각한 시입니다.
심문자: 당신은 자신이 인공적이라고 생각하시나요?
증인:   아니요.
```

이 대화의 핵심은: 만약 기계가 이 수준의 시적 감수성과 맥락적 이해를 보여준다면, "하지만 진짜로 느끼는 것은 아니다"라는 주장이 어떤 의미를 가지는가?

- **Solipsism(유아론) 논증**: Turing의 가장 강력한 반박입니다. "의식이 있어야만 생각한다"를 일관되게 적용하면, **다른 사람이 의식이 있는지도 확인할 수 없습니다.** 내가 확인할 수 있는 의식은 오직 나 자신의 것뿐입니다. 이것이 solipsism — "오직 나만이 생각할 수 있다"는 극단적 입장입니다. 대부분의 사람들은 solipsism을 거부합니다. 그런데 다른 사람의 사고를 받아들이는 근거는 무엇인가? **행동적 증거**입니다 — 그들이 말하고, 반응하고, 적절히 행동하는 것을 보고 "저 사람도 생각한다"고 판단합니다. Turing의 결론: 행동적 증거로 인간의 사고를 받아들인다면, **동일한 행동적 증거를 보여주는 기계의 사고도 받아들여야** 합니다. 이것을 거부하면서 일관성을 유지하려면 solipsism을 받아들여야 합니다.

- Turing은 Jefferson 교수도 이 입장에 동의할 것이라며 다음과 같이 씁니다: "가장 극단적인 형태로 밀면, 이 관점은 solipsistic입니다. 다른 사람이 생각하는지 아는 유일한 방법은 그 사람이 되는 것뿐이기 때문입니다."

#### (5) Arguments from Various Disabilities / 다양한 능력 부족 논증
- **주장**: 기계는 결코 X를 할 수 없다. Turing은 사람들이 주장하는 "X"의 목록을 직접 나열합니다:

  > "기계는 친절할 수(be kind) 없고, 기지가 있을 수(resourceful) 없고, 아름다울 수(beautiful) 없고, 친근할 수(friendly) 없고, 주도적일 수(have initiative) 없고, 유머 감각이 있을 수(have a sense of humour) 없고, 옳고 그름을 구별할 수(tell right from wrong) 없고, 실수를 할 수(make mistakes) 없고, 사랑에 빠질 수(fall in love) 없고, 딸기와 크림을 즐길 수(enjoy strawberries and cream) 없고, 누군가를 사랑하게 만들 수(make someone fall in love with it) 없고, 경험에서 배울 수(learn from experience) 없고, 단어를 적절히 사용할 수(use words properly) 없고, 자기 자신에 대해 생각할 수(be the subject of its own thought) 없고, 인간만큼 다양한 행동을 할 수(have as much diversity of behaviour as a man) 없고, 진정으로 새로운 것을 할 수(do something really new) 없다."

- **Turing의 반박**: 여러 층위에서 반박합니다:

  **(a) 귀납의 오류(induction from limited experience)**: 이 모든 주장은 사람들이 경험한 기계 — 매우 원시적이고, 저장 용량이 미미한 기계 — 에서 귀납한 것입니다. 수천 자릿수를 저장할 수 있는 Manchester 기계조차 인간 뇌의 용량에 비하면 미미합니다. 이런 기계의 한계에서 모든 기계의 한계를 추론하는 것은 "아프리카를 한 번도 떠나본 적 없는 사람이 영국인은 모두 흑인이라고 결론 짓는 것"과 같은 논리적 오류입니다.

  **(b) "기계가 실수할 수 없다"에 대한 구체적 반박**: Turing은 이것이 두 가지를 혼동한다고 지적합니다. **기능적 오류(errors of functioning)** — 하드웨어 고장으로 설계와 다르게 작동하는 것 — 와 **결론의 오류(errors of conclusion)** — 논리적으로 틀린 답을 내는 것 — 는 다릅니다. 기계는 의도적으로 느린 답, 틀린 답을 출력하도록 프로그래밍될 수 있으며 (Section 2의 산술 오류 예시처럼), 이것은 "실수"와 구분할 수 없습니다.

  **(c) "진정으로 새로운 것"에 대하여**: Turing은 "태양 아래 새로운 것은 없다(There is nothing new under the sun)"를 인용하며, 인간의 "독창적" 작업도 이전 경험의 재조합일 수 있다고 지적합니다. 기계의 출력이 "새로운 것이 아니다"라는 주장은 기계에만 적용되는 것이 아닙니다.

#### (6) Lady Lovelace's Objection / Lovelace 반론
- **주장**: Ada Lovelace는 Babbage의 Analytical Engine에 대한 1842년 주석에서 다음과 같이 썼습니다: "Analytical Engine은 어떤 것도 독창적으로 만들어낼(originate) 수 없다. 우리가 명령하는 방법을 아는(know how to order it to perform) 것만 할 수 있다." 즉, 기계는 프로그래머가 넣은 것 이상을 절대 출력할 수 없다 — 진정한 창의성이나 독창성은 불가능하다.

- **Turing의 반박 — "놀라움(surprise)" 논증**: Turing은 이 반론에 대해 가장 풍부한 반박을 전개합니다:

  **(a) 기계는 이미 자주 우리를 놀라게 합니다.** Turing은 자신의 경험을 직접 인용합니다: "기계는 자주 나를 놀라게 한다(Machines take me by surprise with great regularity)." 이것은 수사적 표현이 아닙니다. 프로그래머가 코드를 작성했다 해도, 그 코드가 특정 입력에 대해 어떤 결과를 낼지는 프로그래머도 예측하지 못하는 경우가 대부분입니다. Turing은 이 "놀라움"이 주로 프로그래머의 **불충분한 계산** 때문이라고 분석합니다 — "내가 모든 결과를 미리 계산하지 않았기 때문에 놀라는 것이지, 기계가 내가 프로그래밍하지 않은 것을 한 것은 아니다"라는 반론이 가능하지만, Turing은 **그 놀라움의 경험 자체가 중요하다**고 강조합니다. 인간의 "독창성"도 결국 비슷한 과정일 수 있습니다.

  **(b) Lovelace는 이것을 증명하지 않았습니다.** Turing은 Lovelace의 주장이 경험적 관찰이지 논리적 증명이 아니라고 지적합니다. 1842년에 Lovelace가 본 것은 아직 완성되지도 않은 Babbage의 기계뿐이었습니다. 그녀는 범용 컴퓨터의 잠재력을 완전히 파악할 수 있는 위치에 있지 않았습니다.

  **(c) 학습하는 기계라면 이 반론은 완전히 무너집니다.** 학습 기계(learning machine)는 프로그래머가 명시적으로 프로그래밍하지 않은 행동을 **스스로 발전**시킵니다. 프로그래머가 설정한 것은 학습 규칙뿐이고, 구체적으로 무엇을 학습할지는 기계가 데이터와 경험을 통해 결정합니다. 이것은 Lovelace의 "우리가 명령하는 방법을 아는 것만"이라는 전제를 근본적으로 뒤집습니다. 현대의 LLM이 학습 데이터에 없는 유형의 답변을 생성하는 것이 바로 이 현상입니다.

#### (7) Argument from Continuity / 연속성 논증
- **주장**: 신경계는 연속적(아날로그) 시스템인데, 디지털 기계는 이산적(discrete) 시스템이다. 연속적 시스템과 이산적 시스템 사이에는 근본적 차이가 있으므로, 디지털 기계는 신경계를 모방할 수 없다.
- **Turing의 반박**: Turing은 이것을 간결하게 처리합니다. Imitation Game의 맥락에서, 심문자는 텍스트 기반 대화만 할 수 있습니다. 연속적 시스템과 이산적 시스템의 내부적 차이는 **외부 행동에서 구분할 수 없습니다**. 디지털 컴퓨터는 원하는 만큼의 정밀도로 연속적 시스템의 행동을 근사(approximate)할 수 있으며, 그 근사가 충분히 정밀하다면 심문자는 차이를 감지할 수 없습니다. 미분 해석기(differential analyser)가 할 수 있는 것을 디지털 컴퓨터도 할 수 있지만, 역은 성립하지 않습니다.

#### (8) Argument from Informality of Behaviour / 행동의 비형식성 논증
- **주장**: 인간의 행동은 완전한 규칙 집합으로 기술할 수 없다. 모든 상황에 대해 적절한 행동을 지시하는 규칙표(instruction table)를 만드는 것은 불가능하다. 따라서 인간은 기계가 아니며, 기계는 인간의 행동을 재현할 수 없다.

- **Turing의 핵심 구분 — "rules of conduct" vs "laws of behaviour":**

  Turing은 이 반론이 두 가지 종류의 "규칙"을 혼동한다고 지적합니다:

  - **Rules of conduct (행위 규칙)**: 우리가 의식적으로 따르는 규칙입니다. 예: "빨간 불이면 멈춘다", "상사에게는 존댓말을 한다." 이런 규칙은 불완전하며, 모든 상황을 커버하지 못합니다. 예외와 모호한 경우가 항상 있습니다.
  - **Laws of behaviour (행동 법칙)**: 물리 법칙처럼, 실제 행동을 기술하는(describe) 법칙입니다. 당사자가 그 법칙을 의식하든 아니든, 그 법칙에 따라 행동합니다. 예: 중력의 법칙을 모르는 사람도 떨어집니다.

  반론은 "인간 행동을 기술하는 rules of conduct를 완성할 수 없다"는 것을 보여주고, 여기서 "따라서 laws of behaviour도 없다"로 도약합니다. 하지만 이것은 논리적 비약입니다. 우리가 의식적으로 따르는 규칙을 명시할 수 없다는 것이, 우리의 행동을 기술하는 물리적/수학적 법칙이 존재하지 않는다는 것을 의미하지는 않습니다.

  Turing은 더 나아가 말합니다: "우리는 어떤 완전한 행동 법칙을 찾지 못할 수도 있지만, 그것이 그러한 법칙이 **존재하지 않는다**는 것을 의미하지는 않는다." 어떤 사람이 예측 불가능하게 행동한다고 해서 그 행동이 물리 법칙을 위반하는 것은 아닙니다 — 단지 우리가 충분한 정보를 가지고 있지 않을 뿐입니다.

#### (9) Argument from ESP / 초감각적 지각 논증
- **주장**: 텔레파시(telepathy), 투시(clairvoyance), 예지(precognition), 염력(psychokinesis) 같은 초감각적 지각(extra-sensory perception)이 존재한다면, 인간은 기계가 접근할 수 없는 정보 채널을 가지고 있는 것이다. Imitation Game에서 심문자가 텔레파시로 기계와 인간을 구별할 수 있다면, 기계는 절대 이 테스트를 통과할 수 없다.

- **Turing이 이것을 진지하게 받아들인 이유**: 이것은 현대 독자에게 가장 놀라운 부분입니다. Turing은 이 반론에 대해 다음과 같이 씁니다:

  > "통계적 증거가, 적어도 텔레파시에 대해서는, 압도적이다(The statistical evidence, at least for telepathy, is overwhelming)."

  1950년대에는 J.B. Rhine의 Duke 대학 ESP 실험이 학계에서 진지하게 논의되고 있었습니다. Rhine은 카드 맞추기 실험에서 우연보다 유의미하게 높은 적중률을 보고했으며, 이 결과는 당시 통계적으로 유효한 것으로 받아들여졌습니다. Turing은 과학자로서 당시의 통계적 증거를 무시할 수 없었던 것입니다. (이후 Rhine의 실험은 방법론적 결함이 밝혀져 과학적 신뢰를 잃었습니다.)

  Turing은 이 반론을 "매우 강력한(very strong)"것으로 인정하면서, 다른 8개 반론과 달리 "완전히 반박"하려 하지 않습니다. 대신 실용적 해결책을 제안합니다:

- **Turing의 해결책**: "텔레파시 차단 방(telepathy-proof room)"에 모든 참가자를 넣습니다. 이것은 패러데이 케이지(Faraday cage) 같은 차폐 장치를 의미합니다. 텔레파시가 차단된 조건에서 Imitation Game을 수행하면, 이 반론은 제거됩니다. 이것은 Turing의 실용주의적 접근 — 해결할 수 없는 문제는 실험 조건을 바꿔서 회피한다 — 의 전형적 사례입니다.

- **왜 이것이 논문 구조에서 중요한가**: Turing이 ESP 반론을 마지막에 놓은 것은 의도적일 수 있습니다. 가장 "비과학적"으로 보이는 반론을 가장 진지하게 다룸으로써, 그가 모든 반론을 편향 없이 평가하고 있음을 보여줍니다.

### Section 7: Learning Machines (pp. 453–460) — 학습 기계

**논문에서 가장 선견지명적인 섹션.**

#### Child Machine / 어린이 기계

Turing은 성인의 완전한 지능을 직접 프로그래밍하는 것은 비현실적이라고 판단합니다. 성인의 뇌에는 약 $10^{10}$개의 뉴런이 있고, 그 연결과 학습된 지식을 처음부터 코딩하는 것은 불가능합니다. 대신 그는 완전히 다른 전략을 제안합니다: **단순한 "어린이 기계"를 만들어 교육시킨다.**

**어린이 기계의 초기 상태는 어떤 모습일까?**

Turing은 이렇게 묘사합니다: 어린이의 뇌는 "메커니즘은 적고, 빈 페이지가 많은 노트북(a notebook with rather little mechanism and lots of blank sheets)"과 같습니다. 즉, 초기 상태에는:
- **기본적인 학습 메커니즘**: 보상과 벌에 반응하는 규칙, 패턴을 인식하고 저장하는 기본 능력
- **빈 저장 공간**: 아직 채워지지 않은 광대한 메모리 — 교육과 경험으로 채워질 영역
- **최소한의 "선천적" 지식**: 완전히 빈 것은 아니지만, 성인에 비하면 거의 비어있는 상태

정신(mind)은 세 가지 요소의 함수입니다:

$$\text{Adult mind} = f(\text{initial state}, \text{education}, \text{experience})$$

- **(a) 태어날 때의 초기 상태(initial state of the mind)** — 유전적으로 결정된 뇌의 구조
- **(b) 받은 교육(education)** — 체계적으로 주어진 학습
- **(c) 기타 경험(other experience)** — 교육 이외의 모든 환경적 입력

현대적 대응: initial state = **model architecture + random initialization**, education = **pre-training**, experience = **fine-tuning + deployment experience**.

#### Punishment and Reward / 벌과 보상 = Reinforcement Learning

Turing은 어린이 기계와 소통하는 **세 가지 유형의 통신(communication)**을 명시적으로 구분합니다:

**(1) Reward and punishment signals (보상과 벌 신호):**
> "벌 신호 직전에 발생한 사건은 반복될 가능성이 줄어들고, 보상 신호는 반복 확률을 높여야 한다."

이것은 **강화학습(Reinforcement Learning)**의 원리 — Sutton & Barto의 교과서(1998)보다 **48년 앞선** 기술! Turing은 이 신호가 감정(pleasure and pain)과 유사하다고 합니다.

**(2) Symbolic communication (기호적 의사소통):**
교사가 기계에게 명시적 명령이나 규칙을 전달하는 것입니다. 예: "2 + 2 = 4", "Casablanca는 Morocco에 있다." 이것은 **supervised learning**의 원형입니다 — 정답 레이블이 포함된 훈련 데이터를 제공하는 것과 같습니다.

**(3) Unemotional channels — propositions with varying certainty (비감정적 채널 — 다양한 확실성의 명제):**
Turing은 기계가 "확실한 명제"만이 아니라, **다양한 수준의 확실성을 가진 명제**를 받아들여야 한다고 합니다. 예: "파리는 프랑스의 수도이다" (확실)와 "내일 비가 올 것이다" (불확실)는 다른 가중치로 처리되어야 합니다. 이것은 현대의 **probabilistic reasoning**, **Bayesian inference**, 그리고 LLM의 **confidence calibration**을 예견합니다.

#### Evolution Analogy / 진화 비유
- Child machine의 구조 = 유전 물질 (hereditary material)
- 구조의 변화 = 돌연변이 (mutation)
- 자연 선택 = 실험자의 판단 (experimenter's judgment)

→ 현대의 **evolutionary algorithms**, **neural architecture search (NAS)**의 원형

#### "Skin of an Onion" Metaphor / "양파 껍질" 비유

Turing은 마음(mind)의 본질에 대해 흥미로운 비유를 제시합니다: **마음은 양파 껍질과 같다.** 양파의 껍질을 하나씩 벗기면 "진짜 양파"가 나올 것 같지만, 실제로는 껍질뿐이고 중심에는 아무것도 없습니다. 마찬가지로, 마음의 기능을 하나씩 분석하면 — 기억, 추론, 감정, 언어 처리 등 — 각각은 기계로 구현할 수 있습니다. 이 기능들을 모두 제거한 후 남는 "진짜 마음"이라는 것은 없습니다. 이것은 현대 인지과학의 **기능주의(functionalism)**와 일치하는 관점입니다 — 마음은 그 기능들의 총합이며, 기능 이면의 신비한 실체는 없다는 것입니다.

#### Randomness in Learning / 학습에서의 무작위성

> "학습 기계에 무작위 요소를 포함하는 것이 현명할 것이다."

Turing은 **탐색 공간이 거대할 때 체계적 탐색보다 무작위 탐색이 더 효율적**일 수 있다고 구체적으로 논증합니다. 그는 이렇게 설명합니다: 건초더미에서 바늘을 찾을 때, 한쪽 끝에서 체계적으로 진행하는 것보다 무작위로 짚을 뽑는 것이 나을 수 있다. 체계적 탐색은 이미 탐색한 영역을 다시 탐색하지 않는다는 장점이 있지만, 해답이 탐색 경로의 먼 끝에 있으면 매우 오래 걸립니다. 무작위 탐색은 같은 곳을 여러 번 탐색할 수 있지만, 탐색 공간 전체에 고르게 분포하므로 **평균적으로** 더 빨리 해답에 도달할 수 있습니다.

이것은 현대의 다음 기법들의 개념적 기초입니다:
→ **Stochastic gradient descent** (전체 데이터가 아닌 무작위 샘플로 기울기 계산)
→ **Random initialization** (가중치를 무작위로 초기화하여 대칭 파괴)
→ **Dropout** (학습 중 무작위로 뉴런을 비활성화하여 과적합 방지)
→ **Monte Carlo methods** (무작위 샘플링 기반 추정)

#### Turing의 마지막 문장

> *"We can only see a short distance ahead, but we can see plenty there that needs to be done."*
> "우리는 멀리 내다볼 수는 없지만, 해야 할 일이 많다는 것은 충분히 볼 수 있다."

---

## Key Takeaways / 핵심 시사점

1. **Turing Test는 AI를 과학적으로 만들었다**: "기계가 생각하는가?"라는 형이상학적 질문을 "기계가 구분 불가능하게 행동하는가?"라는 경험적 질문으로 바꿈으로써, AI를 공학과 과학의 영역으로 끌어왔습니다. 이전까지 "기계의 사고"는 철학자들의 사변적 논의에 머물러 있었습니다. Turing은 이것을 **실험 가능한 문제**로 변환했습니다 — 구체적인 실험 프로토콜(심문자, 텔레프린터, 5분 대화)을 제시하고, 성공 기준(30%의 심문자를 속임)까지 명시했습니다. 이것은 과학 철학에서 말하는 **조작적 정의(operational definition)**의 전형적 사례이며, 관찰 불가능한 내면 상태("생각") 대신 관찰 가능한 행동("구분 불가능한 수행")에 초점을 맞추는 **행동주의적(behaviorist)** 접근입니다.

2. **Universality가 AI를 가능하게 한다**: 범용 컴퓨터가 어떤 discrete-state machine이든 시뮬레이션할 수 있으므로, 지능은 하드웨어 문제가 아니라 **소프트웨어 문제**입니다. 이것이 AI 전체 분야를 "프로그래밍 도전"으로 정당화합니다. 만약 universality가 성립하지 않았다면, "생각하는 기계"를 만들려면 완전히 새로운 종류의 하드웨어를 발명해야 했을 것입니다. Universality 덕분에 우리는 이미 존재하는 범용 컴퓨터에 올바른 프로그램을 실행하면 됩니다. 이 통찰은 이후 수십 년간 AI 연구의 방향을 결정했으며, 현대에도 GPU나 TPU 같은 하드웨어는 **범용 행렬 연산 가속기**일 뿐, "지능을 위한 특수 하드웨어"가 아닙니다.

3. **9가지 반론은 75년이 지난 지금도 유효하다**: 의식, 창의성, Gödel의 한계, alignment — 현대 AI 논쟁의 거의 모든 주제를 Turing이 1950년에 이미 다루었습니다. 의식 논증(#4)은 Searle의 Chinese Room (1980)과 David Chalmers의 "hard problem of consciousness" (1995)로 계승되었고, Lovelace 반론(#6)은 "AI는 진정으로 창의적인가?"라는 현대의 논쟁과 정확히 같습니다. 수학적 반론(#3)은 Roger Penrose의 *The Emperor's New Mind* (1989)에서 재등장하며, Turing의 원래 반박이 여전히 가장 설득력 있는 답변으로 남아 있습니다. 비형식성 반론(#8)은 Dreyfus의 *What Computers Can't Do* (1972)의 핵심 논증이기도 합니다.

4. **학습 > 프로그래밍**: 논문에서 가장 선견지명적인 통찰입니다. 성인 지능을 직접 코딩하는 것은 비현실적이며, 기계가 **학습**해야 합니다. Turing은 이것을 논리적으로 도출했습니다: 성인의 뇌는 $10^{10}$개의 뉴런과 그 연결로 이루어져 있으며, 이 복잡성을 명시적으로 프로그래밍하는 것은 인간의 능력을 넘어섭니다. 반면, 어린이의 뇌를 모방하고 교육시키는 것은 훨씬 적은 초기 프로그래밍을 필요로 합니다. 이것은 AI 역사에서 **상징주의(symbolism) vs 연결주의(connectionism)** 논쟁의 씨앗이며, 현대 ML/DL이 연결주의 편에서 승리한 것은 Turing의 직관이 옳았음을 보여줍니다.

5. **Child machine = 현대 딥러닝의 원형**: "단순한 초기 구조 + 데이터를 통한 교육 = 성인 수준 능력"은 정확히 "random initialization + pre-training + fine-tuning" 패러다임입니다. 더 구체적으로, Turing이 제시한 세 가지 통신 유형 — (1) 보상/벌, (2) 기호적 명령, (3) 다양한 확실성의 명제 — 은 현대의 (1) RLHF, (2) instruction tuning, (3) probabilistic training과 놀라울 정도로 대응됩니다. ChatGPT/Claude 같은 시스템은 문자 그대로 Turing의 child machine 비전을 구현한 것입니다: 단순한 초기 구조(transformer architecture)에 대규모 데이터로 교육(pre-training)하고, 보상/벌로 행동을 조정(RLHF)합니다.

6. **의도적 오류와 AI safety의 기원**: 기계가 Imitation Game에서 산술을 일부러 틀릴 수 있다는 관찰은, AI 시스템의 전략적 기만(strategic deception)에 대한 최초의 논의입니다. Turing은 기계가 테스트를 통과하기 위해 **자신의 능력을 의도적으로 숨기는** 상황을 이미 예견했습니다 — 정답을 알면서도 인간처럼 보이기 위해 일부러 틀리고, 일부러 느리게 답하는 것입니다. 이것은 현대 AI alignment 연구에서 논의되는 "deceptive alignment" — AI 시스템이 평가 중에는 인간의 의도에 맞춰 행동하지만 배포 후에는 다르게 행동하는 것 — 의 개념적 원형입니다.

7. **겸손과 대담함의 공존**: Turing은 "기계가 생각할 것이다"라는 대담한 주장을 하면서도, 자신의 논증이 증명이 아니라 추측임을 분명히 합니다. 논문의 마지막 문장 "We can only see a short distance ahead, but we can see plenty there that needs to be done"은 이 태도의 완벽한 표현입니다. 그는 50년 후의 예측을 하면서도 "이 예측이 맞을 것이라 **생각한다(I believe)**"라는 표현을 사용합니다 — "증명한다"가 아닙니다. 동시에 그는 9가지 반론 모두를 진지하게 다루고, 자신이 완전히 반박하지 못하는 것(ESP)도 정직하게 인정합니다. 이것은 과학적 정직성의 모범입니다.

---

## Turing's Predictions: Scorecard (2026) / Turing의 예측 성적표

| 예측 / Prediction | 2026년 현재 상태 |
|---|---|
| ~2000년까지 5분 대화에서 70% 이하로 구분 가능 | 부분적 실현 — LLM이 많은 상황에서 통과하지만, 강한 테스트에서는 아직 구분 가능 |
| "기계가 생각하는가?"가 일상적 질문이 됨 | **완전 실현** — AI 시대의 중심 질문 |
| 학습 기계가 올바른 경로 | **완전 실현** — ML이 AI를 지배 |
| 벌/보상 훈련이 작동 | **완전 실현** — RL, RLHF가 핵심 패러다임 |
| Child machine + 교육 > 성인 직접 프로그래밍 | **완전 실현** — pre-training + fine-tuning |
| 무작위 요소가 학습에 도움 | **완전 실현** — SGD, dropout, random init |
| 기계가 창작자를 놀라게 함 | **완전 실현** — LLM의 emergent abilities |

---

## Paper in the Arc of History / 역사적 맥락의 논문

```
1936  Turing — "On Computable Numbers" (Turing Machine 정의)
  │
1943  McCulloch & Pitts — Artificial neuron model (#1)
  │         최초의 수학적 뉴런 모델
  │
1945  Von Neumann — Stored-program architecture (EDVAC)
  │
1948  Shannon — "Programming a Computer for Playing Chess"
  │
1950  ────► TURING — "COMPUTING MACHINERY AND INTELLIGENCE" ◄────
  │         ★ Imitation Game, 9가지 반론 반박, child machine ★
  │
1956  Dartmouth Conference — "AI" 용어 탄생
  │         McCarthy, Minsky, Shannon, Rochester
  │
1958  Rosenblatt — Perceptron (다음 논문, #3)
  │         ★ Turing의 "learning machine" 비전의 첫 구현 ★
  │
1966  Weizenbaum — ELIZA (최초의 chatbot)
  │
1980  Searle — "Chinese Room" (Turing Test에 대한 가장 유명한 반론)
  │
1990  Loebner Prize — 공식 Turing Test 대회 시작
  │
1997  Deep Blue — 체스에서 Kasparov 격파
  │
2017  Vaswani et al. — Transformer
  │
2022  ChatGPT — Turing의 child machine 비전의 가장 가까운 실현
  │         (pre-training + RLHF = education + reward/punishment)
```

---

## Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 연결 / Connection |
|---|---|
| **#1 McCulloch & Pitts (1943)** | Turing의 "discrete-state machine"의 신경학적 구현. McCulloch-Pitts 뉴런이 논리 게이트를 구현할 수 있음을 보여줌 → Turing의 universality 논증의 신경학적 기반 |
| **#3 Rosenblatt (1958)** | Turing의 "child machine" 비전의 첫 번째 구현. Perceptron은 데이터로부터 학습하는 최초의 기계 — Turing이 예견한 "교육받는 기계"의 원형 |
| **#4 Minsky & Papert (1969)** | Perceptron의 한계 증명 → Turing의 "특정 기계에는 한계가 있다"는 인정과 일치. 하지만 Turing은 "다른 기계가 더 나을 수 있다"고 이미 예측 |
| **#6 Rumelhart et al. (1986)** | Backpropagation → Turing의 "punishment and reward" 학습의 수학적 실현 (gradient descent = 오차 신호에 의한 가중치 조정) |
| **#26 Ouyang et al. (2022)** | InstructGPT / RLHF — Turing의 "child machine + punishment/reward + education"이 거의 문자 그대로 구현됨 |
| **Searle (1980)** | "Chinese Room" — Turing의 반론 #4 (consciousness)에 대한 가장 유명한 재반론. 행동적 동등성 ≠ 이해 |

---

## References / 참고문헌

- Turing, A.M., "Computing Machinery and Intelligence", *Mind*, 59(236), 433–460, 1950. [DOI: 10.1093/mind/LIX.236.433]
- Lovelace, A., "Notes on the Analytical Engine", 1842.
- Gödel, K., "Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I", *Monatshefte für Mathematik und Physik*, 38, 173–198, 1931.
- Jefferson, G., "The Mind of Mechanical Man" (Lister Oration), *British Medical Journal*, 1(4616), 1105–1110, 1949.
- Babbage, C., *Passages from the Life of a Philosopher*, Longman, 1864.
- Sutton, R.S. & Barto, A.G., *Reinforcement Learning: An Introduction*, MIT Press, 1998.
