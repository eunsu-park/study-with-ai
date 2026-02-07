# 08. Prompt Engineering

## Learning Objectives

- Effective prompt writing
- Zero-shot, Few-shot techniques
- Chain-of-Thought (CoT)
- Advanced prompting techniques

---

## 1. Prompt Basics

### Prompt Components

```
┌─────────────────────────────────────────┐
│ [System Instruction]                     │
│ You are a helpful AI assistant.          │
├─────────────────────────────────────────┤
│ [Context]                                │
│ Please refer to the following text: ...  │
├─────────────────────────────────────────┤
│ [Task Instruction]                       │
│ Please summarize the text above.         │
├─────────────────────────────────────────┤
│ [Output Format]                          │
│ Please respond in JSON format.           │
└─────────────────────────────────────────┘
```

### Basic Principles

```
1. Clarity: Write unambiguously
2. Specificity: Specify exactly what you want
3. Examples: Provide examples when possible
4. Constraints: Specify output format, length, etc.
```

---

## 2. Zero-shot vs Few-shot

### Zero-shot

```
Explain task without examples

Prompt:
"""
Analyze the sentiment of the following review.
Review: "This movie was really boring."
Sentiment:
"""

Response: Negative
```

### Few-shot

```
Provide several examples

Prompt:
"""
Analyze the sentiment of the following reviews.

Review: "Really fun movie!"
Sentiment: Positive

Review: "Worst movie, waste of time"
Sentiment: Negative

Review: "It was okay"
Sentiment: Neutral

Review: "This movie was really boring."
Sentiment:
"""

Response: Negative
```

### Few-shot Tips

```python
# Example selection criteria
1. Diversity: Include examples from all classes
2. Representativeness: Use typical examples
3. Similarity: Examples similar to actual input
4. Relevance: Highly relevant examples

# Number of examples
- Generally 3-5
- Complex tasks: 5-10
- Consider token limits
```

---

## 3. Chain-of-Thought (CoT)

### Basic CoT

```
Guide step-by-step reasoning

Prompt:
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
Simply guide reasoning

Prompt:
"""
Q: 5 + 7 × 3 = ?

Let's think step by step.
"""

Response:
1. First, we need to follow order of operations (PEMDAS).
2. Multiplication comes before addition.
3. 7 × 3 = 21
4. 5 + 21 = 26
The answer is 26.
```

### Self-Consistency

```python
# Generate multiple reasoning paths and take majority vote

responses = []
for _ in range(5):
    response = model.generate(prompt, temperature=0.7)
    responses.append(extract_answer(response))

# Select most common answer
final_answer = max(set(responses), key=responses.count)
```

---

## 4. Role Playing

### Expert Role

```
System prompt:
"""
You are a Python developer with 10 years of experience.
When reviewing code, you check for:
- Code readability
- Potential bugs
- Performance optimization
- Security vulnerabilities
"""

User:
"""
Please review the following code:
def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
```

### Persona

```
"""
You are a kind and patient elementary school teacher.
You explain complex concepts using easy analogies.
You always use an encouraging tone.

Question: What is gravity?
"""
```

---

## 5. Specifying Output Format

### JSON Output

```
Prompt:
"""
Extract persons and locations from the following text.

Text: "Cheolsu met Younghee in Seoul."

Respond in JSON format:
{
  "persons": [...],
  "locations": [...]
}
"""
```

### Structured Output

```
Prompt:
"""
Analyze the following article.

## Summary
(2-3 sentences)

## Key Points
- Point 1
- Point 2

## Sentiment
(Positive/Negative/Neutral)
"""
```

### XML Tags

```
Prompt:
"""
Translate and explain the following text.

<text>Hello, how are you?</text>

<translation>Translation result</translation>
<explanation>Translation explanation</explanation>
"""
```

---

## 6. Advanced Techniques

### Self-Ask

```
Model asks and answers its own questions

"""
Question: Where is President Biden's hometown?

Follow-up needed: Yes
Follow-up question: Who is President Biden?
Intermediate answer: Joe Biden is the 46th president of the United States.

Follow-up needed: Yes
Follow-up question: Where was Joe Biden born?
Intermediate answer: He was born in Scranton, Pennsylvania.

Follow-up needed: No
Final answer: President Biden's hometown is Scranton, Pennsylvania.
"""
```

### ReAct (Reason + Act)

```
Alternate between reasoning and actions

"""
Question: Who won the 2023 Nobel Prize in Physics?

Thought: I need to find who won the 2023 Nobel Prize in Physics.
Action: Search[2023 Nobel Prize in Physics]
Observation: Pierre Agostini, Ferenc Krausz, and Anne L'Huillier won.

Thought: I have confirmed the search results.
Action: Finish[Pierre Agostini, Ferenc Krausz, Anne L'Huillier]
"""
```

### Tree of Thoughts

```python
# Explore multiple thought paths as a tree

def tree_of_thoughts(problem, depth=3, branches=3):
    thoughts = []

    for _ in range(branches):
        # Generate first thought
        thought = generate_thought(problem)
        score = evaluate_thought(thought)
        thoughts.append((thought, score))

    # Select top thoughts
    best_thoughts = sorted(thoughts, key=lambda x: x[1], reverse=True)[:2]

    # Recursively expand
    for thought, _ in best_thoughts:
        if depth > 0:
            extended = tree_of_thoughts(thought, depth-1, branches)
            thoughts.extend(extended)

    return thoughts
```

---

## 7. Prompt Optimization

### Iterative Improvement

```python
# 1. Start with basic prompt
prompt_v1 = "Summarize this text: {text}"

# 2. Improve after analyzing results
prompt_v2 = """
Summarize the following text in 2-3 sentences.
Focus on the main points.
Text: {text}
Summary:
"""

# 3. Add examples
prompt_v3 = """
Summarize the following text in 2-3 sentences.

Example:
Text: [Long article]
Summary: [Brief summary]

Text: {text}
Summary:
"""
```

### A/B Testing

```python
import random

def ab_test_prompts(test_cases, prompt_a, prompt_b):
    results = {'A': 0, 'B': 0}

    for case in test_cases:
        response_a = model.generate(prompt_a.format(**case))
        response_b = model.generate(prompt_b.format(**case))

        # Evaluation (automatic or manual)
        score_a = evaluate(response_a, case['expected'])
        score_b = evaluate(response_b, case['expected'])

        if score_a > score_b:
            results['A'] += 1
        else:
            results['B'] += 1

    return results
```

---

## 8. Prompt Templates

### Classification

```python
CLASSIFICATION_PROMPT = """
Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
```

### Summarization

```python
SUMMARIZATION_PROMPT = """
Summarize the following text in {num_sentences} sentences.
Focus on the key points and main arguments.

Text:
{text}

Summary:"""
```

### Question Answering

```python
QA_PROMPT = """
Answer the question based on the context below.
If the answer cannot be found, say "I don't know."

Context: {context}

Question: {question}

Answer:"""
```

### Code Generation

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

## 9. Managing Prompts in Python

### Template Class

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

# Usage
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

### LangChain Prompts

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Basic template
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize: {text}"
)

# Few-shot template
examples = [
    {"input": "Long text 1", "output": "Summary 1"},
    {"input": "Long text 2", "output": "Summary 2"},
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

## Summary

### Prompt Checklist

```
□ Provide clear instructions
□ Include examples if needed (Few-shot)
□ Specify output format
□ Set role/persona
□ Guide step-by-step reasoning (if needed)
□ Specify constraints
```

### Technique Selection Guide

| Situation | Recommended Technique |
|-----------|----------------------|
| Simple task | Zero-shot |
| Specific format needed | Few-shot + format specification |
| Reasoning required | Chain-of-Thought |
| Complex reasoning | Tree of Thoughts |
| Tool usage needed | ReAct |

---

## Next Steps

Learn about Retrieval-Augmented Generation (RAG) systems in [09_RAG_Basics.md](./09_RAG_Basics.md).
