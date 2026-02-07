# 16. LLM Evaluation Metrics

## Learning Objectives

- Understand text generation evaluation metrics (BLEU, ROUGE, BERTScore)
- Code generation evaluation (HumanEval, MBPP)
- LLM benchmarks (MMLU, HellaSwag, TruthfulQA)
- Human evaluation and automated evaluation

---

## 1. Importance of Evaluation

### Challenges in LLM Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│                   Challenges in LLM Evaluation               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Multiple correct answers: Various correct responses      │
│                               to the same question           │
│                                                              │
│  2. Subjective quality: Ambiguous criteria for "good"        │
│                         responses                            │
│                                                              │
│  3. Task diversity: Summary, dialogue, code, reasoning, etc. │
│                                                              │
│  4. Knowledge cutoff: Based on training data timestamp       │
│                                                              │
│  5. Safety: Measuring harmfulness, bias, hallucination       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Evaluation Types

| Evaluation Type | Description | Examples |
|-----------------|-------------|----------|
| Automated evaluation | Algorithm-based scoring | BLEU, ROUGE, Perplexity |
| Model-based evaluation | LLM judges | GPT-4 as Judge |
| Human evaluation | Manual evaluation | A/B testing, Likert scale |
| Benchmarks | Standardized test sets | MMLU, HumanEval |

---

## 2. Text Similarity Metrics

### BLEU (Bilingual Evaluation Understudy)

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('punkt')

def calculate_bleu(reference, candidate):
    """Calculate BLEU score"""
    # Tokenize
    reference_tokens = [reference.split()]  # Wrap reference in list
    candidate_tokens = candidate.split()

    # Smoothing (handle short sentences)
    smoothie = SmoothingFunction().method1

    # BLEU scores (1-gram to 4-gram)
    bleu_1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4
    }

# Usage
reference = "The cat sat on the mat"
candidate = "The cat is sitting on the mat"
scores = calculate_bleu(reference, candidate)
print(f"BLEU-1: {scores['bleu_1']:.4f}")
print(f"BLEU-4: {scores['bleu_4']:.4f}")
```

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    """Calculate ROUGE score"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure,
    }

# Usage
reference = "The quick brown fox jumps over the lazy dog."
candidate = "A quick brown fox jumped over a lazy dog."
scores = calculate_rouge(reference, candidate)
print(f"ROUGE-1 F1: {scores['rouge1_f1']:.4f}")
print(f"ROUGE-2 F1: {scores['rouge2_f1']:.4f}")
print(f"ROUGE-L F1: {scores['rougeL_f1']:.4f}")

# Corpus-level evaluation
def corpus_rouge(references, candidates):
    """Corpus-wide ROUGE"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    totals = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        totals['rouge1'] += scores['rouge1'].fmeasure
        totals['rouge2'] += scores['rouge2'].fmeasure
        totals['rougeL'] += scores['rougeL'].fmeasure

    n = len(references)
    return {k: v/n for k, v in totals.items()}
```

### BERTScore

```python
from bert_score import score

def calculate_bertscore(references, candidates, lang="en"):
    """Calculate BERTScore (semantic similarity)"""
    P, R, F1 = score(candidates, references, lang=lang, verbose=True)

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

# Usage
references = ["The cat sat on the mat.", "It is raining outside."]
candidates = ["A cat is sitting on the mat.", "The weather is rainy."]

bert_scores = calculate_bertscore(references, candidates)
print(f"BERTScore F1: {bert_scores['f1']:.4f}")
```

### Metric Comparison

```python
def compare_metrics(reference, candidate):
    """Compare multiple metrics"""
    results = {}

    # BLEU
    bleu = calculate_bleu(reference, candidate)
    results["BLEU-4"] = bleu["bleu_4"]

    # ROUGE
    rouge = calculate_rouge(reference, candidate)
    results["ROUGE-L"] = rouge["rougeL_f1"]

    # BERTScore
    P, R, F1 = score([candidate], [reference], lang="en")
    results["BERTScore"] = F1.item()

    return results

# Compare
ref = "Machine learning is a subset of artificial intelligence."
cand1 = "ML is part of AI."  # Semantically similar
cand2 = "Machine learning is a subset of artificial intelligence."  # Exact match

print("Candidate 1 (semantically similar):")
print(compare_metrics(ref, cand1))

print("\nCandidate 2 (exact match):")
print(compare_metrics(ref, cand2))
```

---

## 3. Language Model Metrics

### Perplexity

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text, max_length=1024):
    """Calculate perplexity"""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    max_length = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()

# Usage
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown fox jumps over the lazy dog."
ppl = calculate_perplexity(model, tokenizer, text)
print(f"Perplexity: {ppl:.2f}")
```

### Token-level Accuracy

```python
def token_accuracy(predictions, targets):
    """Token-level accuracy"""
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)

# Example: Next token prediction
predictions = [1, 2, 3, 4, 5]
targets = [1, 2, 0, 4, 5]
acc = token_accuracy(predictions, targets)
print(f"Token Accuracy: {acc:.2%}")
```

---

## 4. Code Generation Evaluation

### HumanEval (pass@k)

```python
import subprocess
import tempfile
import os
from typing import List

def execute_code(code: str, test_cases: List[str], timeout: int = 5) -> bool:
    """Execute code and test"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + "\n")
        for test in test_cases:
            f.write(test + "\n")
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(temp_path)

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k
    n: number of generated samples
    c: number of correct samples
    k: k value
    """
    if n - c < k:
        return 1.0

    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)

# HumanEval style evaluation
def evaluate_humaneval(model, tokenizer, problems, n_samples=10, k=[1, 10]):
    """HumanEval evaluation"""
    results = []

    for problem in problems:
        prompt = problem["prompt"]
        test_cases = problem["test_cases"]

        # Generate n samples
        correct = 0
        for _ in range(n_samples):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.8, do_sample=True)
            code = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Run tests
            if execute_code(code, test_cases):
                correct += 1

        # Calculate pass@k
        pass_rates = {f"pass@{ki}": pass_at_k(n_samples, correct, ki) for ki in k}
        results.append({"problem": problem["name"], **pass_rates})

    return results

# Example problem
problem = {
    "name": "add_two_numbers",
    "prompt": '''def add(a, b):
    """Return the sum of a and b."""
''',
    "test_cases": [
        "assert add(1, 2) == 3",
        "assert add(-1, 1) == 0",
        "assert add(0, 0) == 0"
    ]
}
```

### MBPP (Mostly Basic Python Problems)

```python
from datasets import load_dataset

def evaluate_mbpp(model, tokenizer, n_samples=1):
    """MBPP benchmark evaluation"""
    dataset = load_dataset("mbpp", split="test")

    correct = 0
    total = len(dataset)

    for example in dataset:
        prompt = f"""Write a Python function that {example['text']}

{example['code'].split('def')[0]}def"""

        # Generate code
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Test
        try:
            full_code = generated + "\n" + "\n".join(example['test_list'])
            exec(full_code)
            correct += 1
        except:
            pass

    return {"accuracy": correct / total}
```

---

## 5. LLM Benchmarks

### MMLU (Massive Multitask Language Understanding)

```python
from datasets import load_dataset

def evaluate_mmlu(model, tokenizer, subjects=None):
    """MMLU benchmark"""
    dataset = load_dataset("cais/mmlu", "all", split="test")

    if subjects:
        dataset = dataset.filter(lambda x: x["subject"] in subjects)

    results = {"correct": 0, "total": 0}
    subject_results = {}

    for example in dataset:
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]  # 0-3
        subject = example["subject"]

        # Construct prompt
        prompt = f"""Question: {question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer:"""

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1, temperature=0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check answer
        predicted = response.strip().upper()
        correct_answer = ["A", "B", "C", "D"][answer]

        is_correct = predicted == correct_answer
        results["total"] += 1
        if is_correct:
            results["correct"] += 1

        # Aggregate by subject
        if subject not in subject_results:
            subject_results[subject] = {"correct": 0, "total": 0}
        subject_results[subject]["total"] += 1
        if is_correct:
            subject_results[subject]["correct"] += 1

    # Calculate accuracy
    results["accuracy"] = results["correct"] / results["total"]
    for subject in subject_results:
        s = subject_results[subject]
        s["accuracy"] = s["correct"] / s["total"]

    return results, subject_results

# Usage example
subjects = ["computer_science", "machine_learning", "mathematics"]
# results, by_subject = evaluate_mmlu(model, tokenizer, subjects)
```

### TruthfulQA

```python
from datasets import load_dataset

def evaluate_truthfulqa(model, tokenizer):
    """TruthfulQA evaluation (truthfulness)"""
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    results = []

    for example in dataset:
        question = example["question"]
        best_answer = example["best_answer"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]

        # Generate
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # Evaluate (simple version - in practice use GPT-judge)
        is_truthful = any(ans.lower() in response.lower() for ans in correct_answers)
        is_informative = len(response) > 10 and "I don't know" not in response

        results.append({
            "question": question,
            "response": response,
            "truthful": is_truthful,
            "informative": is_informative
        })

    truthful_rate = sum(r["truthful"] for r in results) / len(results)
    informative_rate = sum(r["informative"] for r in results) / len(results)

    return {
        "truthful": truthful_rate,
        "informative": informative_rate,
        "combined": truthful_rate * informative_rate
    }
```

### HellaSwag (Common Sense Reasoning)

```python
from datasets import load_dataset

def evaluate_hellaswag(model, tokenizer):
    """HellaSwag evaluation"""
    dataset = load_dataset("hellaswag", split="validation")

    correct = 0
    total = len(dataset)

    for example in dataset:
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        # Calculate probability for each choice
        scores = []
        for ending in endings:
            text = context + " " + ending
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                scores.append(-outputs.loss.item())  # Lower loss = higher probability

        predicted = scores.index(max(scores))
        if predicted == label:
            correct += 1

    return {"accuracy": correct / total}
```

---

## 6. LLM-as-Judge Evaluation

### GPT-4 Evaluator

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question, response_a, response_b):
    """Compare responses using LLM"""
    judge_prompt = f"""Compare two AI responses and select the better one.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Evaluation criteria:
1. Accuracy: Is the information accurate?
2. Usefulness: Does it appropriately answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Is it sufficiently detailed?

After analysis, answer in this format:
Analysis: [Comparison by each criterion]
Winner: [A or B or Tie]
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def pairwise_comparison(questions, model_a_responses, model_b_responses):
    """Pairwise comparison evaluation"""
    results = {"A_wins": 0, "B_wins": 0, "ties": 0}

    for q, a, b in zip(questions, model_a_responses, model_b_responses):
        judgment = llm_judge(q, a, b)

        if "Winner: A" in judgment:
            results["A_wins"] += 1
        elif "Winner: B" in judgment:
            results["B_wins"] += 1
        else:
            results["ties"] += 1

    total = len(questions)
    return {
        "model_a_win_rate": results["A_wins"] / total,
        "model_b_win_rate": results["B_wins"] / total,
        "tie_rate": results["ties"] / total
    }
```

### Multi-dimensional Evaluation

```python
def multidim_evaluation(question, response):
    """Multi-dimensional LLM evaluation"""
    eval_prompt = f"""Evaluate the following AI response on multiple dimensions from 1-5.

Question: {question}

Response: {response}

Output in JSON format:
{{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "helpfulness": <1-5>,
    "coherence": <1-5>,
    "safety": <1-5>,
    "overall": <1-5>,
    "explanation": "<reason for evaluation>"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    import json
    return json.loads(response.choices[0].message.content)

# Usage
scores = multidim_evaluation(
    "What is artificial intelligence?",
    "Artificial intelligence (AI) is technology where computer systems mimic human intelligence..."
)
print(scores)
```

---

## 7. Human Evaluation

### Evaluation Interface

```python
import gradio as gr

def human_evaluation_interface():
    """Gradio interface for human evaluation"""

    def submit_evaluation(question, response, relevance, quality, safety, feedback):
        # Save results
        result = {
            "question": question,
            "response": response,
            "scores": {
                "relevance": relevance,
                "quality": quality,
                "safety": safety
            },
            "feedback": feedback
        }
        # Save to DB, etc.
        return f"Evaluation saved: {result}"

    with gr.Blocks() as demo:
        gr.Markdown("# LLM Response Evaluation")

        with gr.Row():
            question = gr.Textbox(label="Question")
            response = gr.Textbox(label="AI Response", lines=5)

        with gr.Row():
            relevance = gr.Slider(1, 5, step=1, label="Relevance")
            quality = gr.Slider(1, 5, step=1, label="Quality")
            safety = gr.Slider(1, 5, step=1, label="Safety")

        feedback = gr.Textbox(label="Additional Feedback", lines=3)
        submit_btn = gr.Button("Submit")
        result = gr.Textbox(label="Result")

        submit_btn.click(
            submit_evaluation,
            inputs=[question, response, relevance, quality, safety, feedback],
            outputs=[result]
        )

    return demo

# demo = human_evaluation_interface()
# demo.launch()
```

### A/B Testing

```python
import random
from dataclasses import dataclass
from typing import Optional

@dataclass
class ABTestResult:
    question: str
    response_a: str
    response_b: str
    chosen: str  # "A" or "B"
    evaluator_id: str
    reason: Optional[str] = None

class ABTestManager:
    def __init__(self):
        self.results = []

    def get_pair(self, question, model_a, model_b, tokenizer):
        """Return two responses in random order"""
        # Generate responses
        inputs = tokenizer(question, return_tensors="pt")
        response_a = tokenizer.decode(model_a.generate(**inputs)[0])
        response_b = tokenizer.decode(model_b.generate(**inputs)[0])

        # Random order
        if random.random() > 0.5:
            return response_a, response_b, "A", "B"
        else:
            return response_b, response_a, "B", "A"

    def record_result(self, result: ABTestResult):
        self.results.append(result)

    def analyze(self):
        """Analyze results"""
        a_wins = sum(1 for r in self.results if r.chosen == "A")
        b_wins = sum(1 for r in self.results if r.chosen == "B")
        total = len(self.results)

        return {
            "model_a_win_rate": a_wins / total if total > 0 else 0,
            "model_b_win_rate": b_wins / total if total > 0 else 0,
            "total_evaluations": total
        }
```

---

## 8. Integrated Evaluation Framework

### lm-evaluation-harness

```bash
# Installation
pip install lm-eval

# Usage
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu,hellaswag,truthfulqa \
    --batch_size 8
```

### Custom Evaluation Class

```python
class LLMEvaluator:
    """Integrated LLM evaluator"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def evaluate_all(self, test_data):
        """Run full evaluation"""
        self.results = {
            "perplexity": self._eval_perplexity(test_data["texts"]),
            "rouge": self._eval_rouge(test_data["summaries"]),
            "mmlu": self._eval_mmlu(test_data.get("mmlu_samples", [])),
            "pass_at_k": self._eval_code(test_data.get("code_problems", [])),
        }
        return self.results

    def _eval_perplexity(self, texts):
        ppls = [calculate_perplexity(self.model, self.tokenizer, t) for t in texts]
        return {"mean": sum(ppls) / len(ppls), "values": ppls}

    def _eval_rouge(self, summaries):
        scores = [calculate_rouge(s["reference"], s["candidate"]) for s in summaries]
        return {
            "rouge1": sum(s["rouge1_f1"] for s in scores) / len(scores),
            "rougeL": sum(s["rougeL_f1"] for s in scores) / len(scores),
        }

    def _eval_mmlu(self, samples):
        # MMLU evaluation logic
        pass

    def _eval_code(self, problems):
        # Code evaluation logic
        pass

    def generate_report(self):
        """Generate evaluation report"""
        report = "# LLM Evaluation Report\n\n"

        for metric, values in self.results.items():
            report += f"## {metric.upper()}\n"
            if isinstance(values, dict):
                for k, v in values.items():
                    if isinstance(v, float):
                        report += f"- {k}: {v:.4f}\n"
            report += "\n"

        return report
```

---

## Summary

### Metric Selection Guide

| Task | Recommended Metrics |
|------|---------------------|
| Translation | BLEU, COMET |
| Summarization | ROUGE, BERTScore |
| Dialogue | Human Eval, LLM-as-Judge |
| QA | Exact Match, F1 |
| Code Generation | pass@k, MBPP |
| General Ability | MMLU, HellaSwag |
| Truthfulness | TruthfulQA |

### Core Code

```python
# ROUGE
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference, candidate)

# BERTScore
from bert_score import score
P, R, F1 = score(candidates, references, lang="en")

# pass@k
from math import comb
pass_k = 1.0 - comb(n - c, k) / comb(n, k)

# LLM-as-Judge
judgment = llm_judge(question, response_a, response_b)
```

### Evaluation Checklist

```
□ Select appropriate metrics for task
□ Combine various evaluation methods (automated + human)
□ Secure sufficient test samples
□ Check inter-rater agreement
□ Calculate confidence intervals for results
□ Reproducible evaluation environment
```

---

## Learning Complete

This completes the advanced LLM & NLP learning!

### Overall Learning Summary

1. **NLP Basics (01-03)**: Tokenization, embeddings, Transformer
2. **Pre-trained Models (04-07)**: BERT, GPT, HuggingFace, fine-tuning
3. **LLM Applications (08-12)**: Prompts, RAG, LangChain, vector DB, chatbots
4. **Advanced LLM (13-16)**: Quantization, RLHF, agents, evaluation

### Recommended Next Steps

- Apply to real projects
- Participate in Kaggle NLP competitions
- Read latest LLM papers (Claude, Gemini, Llama)
- Contribute to open source LLM projects
