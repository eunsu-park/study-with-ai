# 24. API & Evaluation

## Overview

This lesson covers how to use commercial LLM APIs, cost optimization, and benchmarks and methodologies for evaluating LLM performance.

---

## 1. Commercial LLM APIs

### 1.1 Major Provider Comparison

```
API Provider Comparison (2024):
┌────────────────────────────────────────────────────────────────┐
│  Provider    │ Model          │ Input/1M  │ Output/1M │ Context │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  OpenAI      │ GPT-4 Turbo    │ $10       │ $30       │ 128K    │
│              │ GPT-4o         │ $5        │ $15       │ 128K    │
│              │ GPT-3.5 Turbo  │ $0.50     │ $1.50     │ 16K     │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  Anthropic   │ Claude 3 Opus  │ $15       │ $75       │ 200K    │
│              │ Claude 3 Sonnet│ $3        │ $15       │ 200K    │
│              │ Claude 3 Haiku │ $0.25     │ $1.25     │ 200K    │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  Google      │ Gemini 1.5 Pro │ $3.50     │ $10.50    │ 1M      │
│              │ Gemini 1.5 Flash│ $0.35    │ $1.05     │ 1M      │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 OpenAI API

```python
from openai import OpenAI
import tiktoken

class OpenAIClient:
    """OpenAI API client"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

    def chat(
        self,
        messages: list,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> dict:
        """Chat completion"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }

    def stream_chat(self, messages: list, model: str = "gpt-4o", **kwargs):
        """Streaming chat"""
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        return len(self.token_encoder.encode(text))

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o"
    ) -> float:
        """Estimate cost"""
        pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
        }

        if model not in pricing:
            return 0.0

        cost = (
            prompt_tokens * pricing[model]["input"] / 1_000_000 +
            completion_tokens * pricing[model]["output"] / 1_000_000
        )

        return cost


# Function calling
def function_calling_example():
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
        tools=tools,
        tool_choice="auto"
    )

    # Handle tool call
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### 1.3 Anthropic API

```python
from anthropic import Anthropic

class AnthropicClient:
    """Anthropic Claude API client"""

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key)

    def chat(
        self,
        messages: list,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        system: str = None,
        **kwargs
    ) -> dict:
        """Chat"""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=messages,
            **kwargs
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            "model": response.model,
            "stop_reason": response.stop_reason
        }

    def stream_chat(self, messages: list, **kwargs):
        """Streaming"""
        with self.client.messages.stream(
            messages=messages,
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text

    def vision(
        self,
        image_url: str,
        prompt: str,
        model: str = "claude-3-sonnet-20240229"
    ) -> str:
        """Vision API"""
        import base64
        import httpx

        # Load image
        if image_url.startswith("http"):
            image_data = base64.standard_b64encode(
                httpx.get(image_url).content
            ).decode("utf-8")
        else:
            with open(image_url, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        return response.content[0].text
```

### 1.4 Google Gemini API

```python
import google.generativeai as genai

class GeminiClient:
    """Google Gemini API client"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> dict:
        """Chat"""
        # Convert OpenAI format to Gemini format
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = self.model.start_chat(history=history)
        response = chat.send_message(
            messages[-1]["content"],
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )

        return {
            "content": response.text,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count
            }
        }

    def multimodal(
        self,
        image_path: str,
        prompt: str
    ) -> str:
        """Multimodal input"""
        import PIL.Image

        img = PIL.Image.open(image_path)
        response = self.model.generate_content([prompt, img])

        return response.text
```

---

## 2. Cost Optimization

### 2.1 Cost Monitoring

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import json

@dataclass
class UsageRecord:
    """API usage record"""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    request_type: str = "chat"

class CostTracker:
    """Cost tracker"""

    def __init__(self):
        self.records: List[UsageRecord] = []
        self.pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25}
        }

    def log_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_type: str = "chat"
    ):
        """Log request"""
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            request_type=request_type
        )

        self.records.append(record)
        return cost

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost"""
        if model not in self.pricing:
            return 0.0

        pricing = self.pricing[model]
        return (
            prompt_tokens * pricing["input"] / 1_000_000 +
            completion_tokens * pricing["output"] / 1_000_000
        )

    def get_summary(self, period: str = "day") -> Dict:
        """Usage summary"""
        from collections import defaultdict

        summary = defaultdict(lambda: {"tokens": 0, "cost": 0, "requests": 0})

        for record in self.records:
            model = record.model
            summary[model]["tokens"] += record.prompt_tokens + record.completion_tokens
            summary[model]["cost"] += record.cost
            summary[model]["requests"] += 1

        return dict(summary)

    def set_budget_alert(self, daily_limit: float):
        """Set daily budget alert"""
        today_cost = sum(
            r.cost for r in self.records
            if r.timestamp.date() == datetime.now().date()
        )

        if today_cost > daily_limit:
            return f"Warning: Daily budget exceeded: ${today_cost:.2f} / ${daily_limit:.2f}"

        return None
```

### 2.2 Optimization Strategies

```python
class CostOptimizer:
    """Cost optimization strategies"""

    def __init__(self):
        self.cache = {}

    def semantic_cache(self, query: str, threshold: float = 0.95):
        """Semantic caching"""
        # Find similar previous query
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if not hasattr(self, 'encoder'):
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        query_emb = self.encoder.encode(query)

        for cached_query, (cached_emb, response) in self.cache.items():
            similarity = np.dot(query_emb, cached_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
            )
            if similarity > threshold:
                return response

        return None

    def add_to_cache(self, query: str, response: str):
        """Add to cache"""
        if hasattr(self, 'encoder'):
            emb = self.encoder.encode(query)
            self.cache[query] = (emb, response)

    def select_model(
        self,
        task_complexity: str,
        latency_requirement: str = "normal"
    ) -> str:
        """Select model appropriate for task"""
        model_map = {
            # (complexity, latency) -> model
            ("simple", "fast"): "gpt-3.5-turbo",
            ("simple", "normal"): "gpt-3.5-turbo",
            ("medium", "fast"): "claude-3-haiku",
            ("medium", "normal"): "gpt-4o",
            ("complex", "fast"): "gpt-4o",
            ("complex", "normal"): "claude-3-opus",
        }

        return model_map.get(
            (task_complexity, latency_requirement),
            "gpt-4o"
        )

    def prompt_compression(self, text: str, target_ratio: float = 0.5) -> str:
        """Prompt compression"""
        # Can use LLMLingua etc.
        # Here using simple summarization approach
        words = text.split()
        target_len = int(len(words) * target_ratio)

        # Select important sentences (needs more sophisticated method in practice)
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text

        # Keep first and last sentences
        compressed = sentences[0] + '.' + sentences[-1]
        return compressed
```

---

## 3. LLM Evaluation

### 3.1 Benchmarks

```
Major Benchmarks:
┌────────────────────────────────────────────────────────────────┐
│  General                                                        │
│  - MMLU: 57 subjects, multiple choice                          │
│  - HellaSwag: Commonsense reasoning                            │
│  - WinoGrande: Coreference resolution                          │
│                                                                │
│  Reasoning                                                      │
│  - GSM8K: Grade school math                                    │
│  - MATH: Competition math                                       │
│  - ARC: Science questions                                       │
│                                                                │
│  Coding                                                         │
│  - HumanEval: Python code generation                           │
│  - MBPP: Python problems                                       │
│  - CodeContests: Competitive programming                       │
│                                                                │
│  Chat/Instruction                                               │
│  - MT-Bench: Multi-turn conversation                           │
│  - AlpacaEval: Instruction following                           │
│  - Chatbot Arena: Human preference                             │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Automated Evaluation

```python
import re
from typing import List, Dict

class LLMEvaluator:
    """LLM automated evaluation"""

    def __init__(self, model_client):
        self.client = model_client

    def evaluate_factuality(
        self,
        question: str,
        answer: str,
        reference: str
    ) -> Dict:
        """Factuality evaluation"""
        prompt = f"""Evaluate if the answer is factually consistent with the reference.

Question: {question}
Answer: {answer}
Reference: {reference}

Score from 1-5 where:
1 = Completely incorrect
3 = Partially correct
5 = Completely correct

Provide your score and brief explanation.
Format: Score: X
Explanation: ..."""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"]

        # Extract score
        score_match = re.search(r'Score:\s*(\d)', text)
        score = int(score_match.group(1)) if score_match else 3

        return {
            "score": score,
            "explanation": text
        }

    def evaluate_helpfulness(
        self,
        instruction: str,
        response: str
    ) -> Dict:
        """Helpfulness evaluation"""
        prompt = f"""Evaluate how helpful and complete the response is.

Instruction: {instruction}
Response: {response}

Rate on these criteria (1-5 each):
1. Relevance: Does it address the instruction?
2. Completeness: Does it fully answer?
3. Clarity: Is it well-written and clear?
4. Accuracy: Is the information correct?

Format:
Relevance: X
Completeness: X
Clarity: X
Accuracy: X
Overall: X"""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"]

        # Parse scores
        scores = {}
        for criterion in ["Relevance", "Completeness", "Clarity", "Accuracy", "Overall"]:
            match = re.search(rf'{criterion}:\s*(\d)', text)
            scores[criterion.lower()] = int(match.group(1)) if match else 3

        return scores

    def pairwise_comparison(
        self,
        instruction: str,
        response_a: str,
        response_b: str
    ) -> str:
        """Pairwise comparison"""
        prompt = f"""Compare these two responses to the instruction.

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider helpfulness, accuracy, and clarity.
Answer with:
- "A" if Response A is better
- "B" if Response B is better
- "TIE" if they are equally good

Your choice:"""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"].strip().upper()

        if "A" in text and "B" not in text:
            return "A"
        elif "B" in text and "A" not in text:
            return "B"
        else:
            return "TIE"


# MT-Bench style evaluation
class MTBenchEvaluator:
    """MT-Bench style multi-turn evaluation"""

    def __init__(self, judge_model):
        self.judge = judge_model

    def evaluate_conversation(
        self,
        conversation: List[Dict]
    ) -> Dict:
        """Evaluate conversation"""
        # Evaluate each turn
        turn_scores = []

        for i, turn in enumerate(conversation):
            if turn["role"] == "assistant":
                context = conversation[:i+1]
                score = self._evaluate_turn(context)
                turn_scores.append(score)

        return {
            "turn_scores": turn_scores,
            "average": sum(turn_scores) / len(turn_scores) if turn_scores else 0
        }

    def _evaluate_turn(self, context: List[Dict]) -> float:
        """Evaluate individual turn"""
        # Compose evaluation prompt
        context_str = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context
        ])

        prompt = f"""Rate the assistant's last response on a scale of 1-10.

Conversation:
{context_str}

Consider:
- Helpfulness
- Relevance
- Accuracy
- Depth

Score (1-10):"""

        response = self.judge.chat([{"role": "user", "content": prompt}])
        score_match = re.search(r'\d+', response["content"])

        return float(score_match.group()) if score_match else 5.0
```

### 3.3 Human Evaluation

```python
from dataclasses import dataclass
from typing import Optional
import random

@dataclass
class EvaluationItem:
    """Evaluation item"""
    id: str
    instruction: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    winner: Optional[str] = None
    annotator: Optional[str] = None

class HumanEvaluation:
    """Human evaluation management"""

    def __init__(self):
        self.items: List[EvaluationItem] = []
        self.results: Dict[str, int] = {}

    def add_comparison(
        self,
        instruction: str,
        responses: Dict[str, str]  # {model_name: response}
    ):
        """Add comparison item"""
        models = list(responses.keys())
        if len(models) != 2:
            raise ValueError("Exactly 2 models required")

        # Randomize order (prevent bias)
        if random.random() > 0.5:
            models = models[::-1]

        item = EvaluationItem(
            id=str(len(self.items)),
            instruction=instruction,
            response_a=responses[models[0]],
            response_b=responses[models[1]],
            model_a=models[0],
            model_b=models[1]
        )

        self.items.append(item)

    def record_judgment(
        self,
        item_id: str,
        winner: str,  # "A", "B", or "TIE"
        annotator: str
    ):
        """Record evaluation result"""
        for item in self.items:
            if item.id == item_id:
                item.winner = winner
                item.annotator = annotator

                # Record winning model
                if winner == "A":
                    winning_model = item.model_a
                elif winner == "B":
                    winning_model = item.model_b
                else:
                    winning_model = "TIE"

                self.results[winning_model] = self.results.get(winning_model, 0) + 1
                break

    def get_elo_ratings(self) -> Dict[str, float]:
        """Calculate Elo ratings"""
        # Initial ratings
        ratings = {}
        for item in self.items:
            ratings[item.model_a] = 1500
            ratings[item.model_b] = 1500

        K = 32  # K-factor

        for item in self.items:
            if item.winner is None:
                continue

            ra = ratings[item.model_a]
            rb = ratings[item.model_b]

            # Expected scores
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            eb = 1 / (1 + 10 ** ((ra - rb) / 400))

            # Actual scores
            if item.winner == "A":
                sa, sb = 1, 0
            elif item.winner == "B":
                sa, sb = 0, 1
            else:
                sa, sb = 0.5, 0.5

            # Update ratings
            ratings[item.model_a] += K * (sa - ea)
            ratings[item.model_b] += K * (sb - eb)

        return ratings
```

---

## Key Summary

### API Usage Checklist
```
□ Manage API keys as environment variables
□ Pre-calculate token counts
□ Set up cost monitoring
□ Handle rate limits
□ Error handling and retries
□ Implement caching strategy
```

### Evaluation Method Selection
```
- Multiple choice questions → Accuracy
- Generation tasks → LLM-as-Judge
- Chat/conversation → MT-Bench/Chatbot Arena
- Coding → pass@k, HumanEval
- Production → A/B testing
```

---

## References

1. [OpenAI API Documentation](https://platform.openai.com/docs)
2. [Anthropic Claude Documentation](https://docs.anthropic.com/)
3. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
4. Chen et al. (2021). "Evaluating Large Language Models Trained on Code" (HumanEval)
