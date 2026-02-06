# 08. 프롬프트 엔지니어링

## 학습 목표

- 효과적인 프롬프트 작성
- Zero-shot, Few-shot 기법
- Chain-of-Thought (CoT)
- 고급 프롬프팅 기법

---

## 1. 프롬프트 기초

### 프롬프트 구성 요소

```
┌─────────────────────────────────────────┐
│ [시스템 지시]                            │
│ 당신은 도움이 되는 AI 어시스턴트입니다.    │
├─────────────────────────────────────────┤
│ [컨텍스트]                               │
│ 다음 텍스트를 참고하세요: ...             │
├─────────────────────────────────────────┤
│ [태스크 지시]                            │
│ 위 텍스트를 요약해주세요.                 │
├─────────────────────────────────────────┤
│ [출력 형식]                              │
│ JSON 형식으로 응답해주세요.               │
└─────────────────────────────────────────┘
```

### 기본 원칙

```
1. 명확성: 모호하지 않게 작성
2. 구체성: 원하는 것을 정확히 명시
3. 예시: 가능하면 예시 제공
4. 제약: 출력 형식, 길이 등 제약 명시
```

---

## 2. Zero-shot vs Few-shot

### Zero-shot

```
예시 없이 태스크만 설명

프롬프트:
"""
다음 리뷰의 감성을 분석해주세요.
리뷰: "이 영화는 정말 지루했어요."
감성:
"""

응답: 부정적
```

### Few-shot

```
몇 개의 예시 제공

프롬프트:
"""
다음 리뷰의 감성을 분석해주세요.

리뷰: "정말 재미있는 영화였어요!"
감성: 긍정

리뷰: "최악의 영화, 시간 낭비"
감성: 부정

리뷰: "그냥 그랬어요"
감성: 중립

리뷰: "이 영화는 정말 지루했어요."
감성:
"""

응답: 부정
```

### Few-shot 팁

```python
# 예시 선택 기준
1. 다양성: 모든 클래스의 예시 포함
2. 대표성: 전형적인 예시 사용
3. 유사성: 실제 입력과 유사한 예시
4. 최신성: 관련성 높은 예시

# 예시 개수
- 일반적으로 3-5개
- 복잡한 태스크: 5-10개
- 토큰 제한 고려
```

---

## 3. Chain-of-Thought (CoT)

### 기본 CoT

```
단계별 추론 유도

프롬프트:
"""
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many balls does he have now?

A: Let's think step by step.
1. Roger started with 5 balls.
2. He bought 2 cans × 3 balls = 6 balls.
3. Total: 5 + 6 = 11 balls.
The answer is 11.
"""
```

### Zero-shot CoT

```
간단하게 추론 유도

프롬프트:
"""
Q: 5 + 7 × 3 = ?

Let's think step by step.
"""

응답:
1. First, we need to follow order of operations (PEMDAS).
2. Multiplication comes before addition.
3. 7 × 3 = 21
4. 5 + 21 = 26
The answer is 26.
```

### Self-Consistency

```python
# 여러 추론 경로 생성 후 다수결

responses = []
for _ in range(5):
    response = model.generate(prompt, temperature=0.7)
    responses.append(extract_answer(response))

# 가장 많이 나온 답 선택
final_answer = max(set(responses), key=responses.count)
```

---

## 4. 역할 부여 (Role Playing)

### 전문가 역할

```
시스템 프롬프트:
"""
당신은 10년 경력의 파이썬 개발자입니다.
코드 리뷰를 할 때 다음을 확인합니다:
- 코드 가독성
- 버그 가능성
- 성능 최적화
- 보안 취약점
"""

사용자:
"""
다음 코드를 리뷰해주세요:
def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
```

### 페르소나

```
"""
당신은 친절하고 인내심 있는 초등학교 선생님입니다.
복잡한 개념을 쉬운 비유로 설명합니다.
항상 격려하는 어조를 사용합니다.

질문: 중력이 뭐예요?
"""
```

---

## 5. 출력 형식 지정

### JSON 출력

```
프롬프트:
"""
다음 텍스트에서 인물과 장소를 추출해주세요.

텍스트: "철수는 서울에서 영희를 만났다."

JSON 형식으로 응답:
{
  "persons": [...],
  "locations": [...]
}
"""
```

### 구조화된 출력

```
프롬프트:
"""
다음 기사를 분석해주세요.

## 요약
(2-3문장)

## 핵심 포인트
- 포인트 1
- 포인트 2

## 감성
(긍정/부정/중립)
"""
```

### XML 태그

```
프롬프트:
"""
다음 텍스트를 번역하고 설명해주세요.

<text>Hello, how are you?</text>

<translation>번역 결과</translation>
<explanation>번역 설명</explanation>
"""
```

---

## 6. 고급 기법

### Self-Ask

```
모델이 스스로 질문하고 답변

"""
질문: 바이든 대통령의 고향은 어디인가요?

후속 질문 필요: 네
후속 질문: 바이든 대통령은 누구인가요?
중간 답변: 조 바이든은 미국의 46대 대통령입니다.

후속 질문 필요: 네
후속 질문: 조 바이든은 어디서 태어났나요?
중간 답변: 펜실베이니아 주 스크랜턴에서 태어났습니다.

후속 질문 필요: 아니오
최종 답변: 바이든 대통령의 고향은 펜실베이니아 주 스크랜턴입니다.
"""
```

### ReAct (Reason + Act)

```
추론과 행동을 번갈아 수행

"""
질문: 2023년 노벨 물리학상 수상자는 누구인가요?

Thought: 2023년 노벨 물리학상 수상자를 찾아야 합니다.
Action: Search[2023 노벨 물리학상]
Observation: 피에르 아고스티니, 페렌츠 크라우스, 앤 륄리에가 수상했습니다.

Thought: 검색 결과를 확인했습니다.
Action: Finish[피에르 아고스티니, 페렌츠 크라우스, 앤 륄리에]
"""
```

### Tree of Thoughts

```python
# 여러 사고 경로를 트리로 탐색

def tree_of_thoughts(problem, depth=3, branches=3):
    thoughts = []

    for _ in range(branches):
        # 첫 번째 생각 생성
        thought = generate_thought(problem)
        score = evaluate_thought(thought)
        thoughts.append((thought, score))

    # 상위 생각 선택
    best_thoughts = sorted(thoughts, key=lambda x: x[1], reverse=True)[:2]

    # 재귀적으로 확장
    for thought, _ in best_thoughts:
        if depth > 0:
            extended = tree_of_thoughts(thought, depth-1, branches)
            thoughts.extend(extended)

    return thoughts
```

---

## 7. 프롬프트 최적화

### 반복적 개선

```python
# 1. 기본 프롬프트로 시작
prompt_v1 = "Summarize this text: {text}"

# 2. 결과 분석 후 개선
prompt_v2 = """
Summarize the following text in 2-3 sentences.
Focus on the main points.
Text: {text}
Summary:
"""

# 3. 예시 추가
prompt_v3 = """
Summarize the following text in 2-3 sentences.

Example:
Text: [긴 기사]
Summary: [간단한 요약]

Text: {text}
Summary:
"""
```

### A/B 테스트

```python
import random

def ab_test_prompts(test_cases, prompt_a, prompt_b):
    results = {'A': 0, 'B': 0}

    for case in test_cases:
        response_a = model.generate(prompt_a.format(**case))
        response_b = model.generate(prompt_b.format(**case))

        # 평가 (자동 또는 수동)
        score_a = evaluate(response_a, case['expected'])
        score_b = evaluate(response_b, case['expected'])

        if score_a > score_b:
            results['A'] += 1
        else:
            results['B'] += 1

    return results
```

---

## 8. 프롬프트 템플릿

### 분류

```python
CLASSIFICATION_PROMPT = """
Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
```

### 요약

```python
SUMMARIZATION_PROMPT = """
Summarize the following text in {num_sentences} sentences.
Focus on the key points and main arguments.

Text:
{text}

Summary:"""
```

### 질의응답

```python
QA_PROMPT = """
Answer the question based on the context below.
If the answer cannot be found, say "I don't know."

Context: {context}

Question: {question}

Answer:"""
```

### 코드 생성

```python
CODE_GENERATION_PROMPT = """
Write a {language} function that {task_description}.

Requirements:
{requirements}

Function:
```{language}
"""
```

---

## 9. Python에서 프롬프트 관리

### 템플릿 클래스

```python
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'r') as f:
            return cls(f.read())

# 사용
template = PromptTemplate("""
You are a {role}.
Task: {task}
Input: {input}
Output:
""")

prompt = template.format(
    role="helpful assistant",
    task="translate to Korean",
    input="Hello, world!"
)
```

### LangChain 프롬프트

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# 기본 템플릿
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize: {text}"
)

# Few-shot 템플릿
examples = [
    {"input": "긴 텍스트 1", "output": "요약 1"},
    {"input": "긴 텍스트 2", "output": "요약 2"},
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    ),
    prefix="Summarize the following texts:",
    suffix="Input: {text}\nOutput:",
    input_variables=["text"]
)
```

---

## 정리

### 프롬프트 체크리스트

```
□ 명확한 지시 제공
□ 필요시 예시 포함 (Few-shot)
□ 출력 형식 지정
□ 역할/페르소나 설정
□ 단계별 추론 유도 (필요시)
□ 제약 조건 명시
```

### 기법 선택 가이드

| 상황 | 추천 기법 |
|------|----------|
| 간단한 태스크 | Zero-shot |
| 특정 형식 필요 | Few-shot + 형식 지정 |
| 추론 필요 | Chain-of-Thought |
| 복잡한 추론 | Tree of Thoughts |
| 도구 사용 필요 | ReAct |

---

## 다음 단계

[09_RAG_Basics.md](./09_RAG_Basics.md)에서 검색 증강 생성(RAG) 시스템을 학습합니다.
