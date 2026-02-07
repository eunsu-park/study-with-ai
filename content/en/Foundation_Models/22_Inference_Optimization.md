# 22. Inference Optimization

## Overview

LLM inference optimization is a key technology for reducing costs and latency in production environments. This lesson covers vLLM, TGI, Speculative Decoding, and more.

---

## 1. LLM Inference Bottlenecks

### 1.1 Memory Bottleneck

```
LLM Inference Characteristics:
┌─────────────────────────────────────────────────────────┐
│  KV Cache Size Calculation:                             │
│                                                         │
│  Memory = 2 × n_layers × n_heads × head_dim × seq_len  │
│                       × batch_size × dtype_size        │
│                                                         │
│  Example: LLaMA-7B, batch=1, seq=2048, FP16            │
│  = 2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes             │
│  = 1.07 GB per sequence                                │
│                                                         │
│  With batch=32: ~34 GB (KV cache only)                 │
└─────────────────────────────────────────────────────────┘

Problems:
1. GPU memory limitations
2. Variable length sequences → Memory fragmentation
3. Batch size limitations → Low throughput
```

### 1.2 Compute Bottleneck

```
Autoregressive Generation Inefficiency:
┌────────────────────────────────────────────────────────┐
│  Step 1: [prompt] → token_1                            │
│  Step 2: [prompt, token_1] → token_2                   │
│  Step 3: [prompt, token_1, token_2] → token_3          │
│  ...                                                   │
│                                                        │
│  At each step:                                         │
│  - Load entire KV cache                                │
│  - Generate only 1 token                               │
│  - Low GPU utilization (memory-bound)                  │
└────────────────────────────────────────────────────────┘
```

---

## 2. vLLM

### 2.1 PagedAttention

```
PagedAttention Core Idea:
┌────────────────────────────────────────────────────────────┐
│  Traditional: Contiguous memory allocation                  │
│                                                            │
│  Sequence A: [████████████████████░░░░░░]  (padding waste) │
│  Sequence B: [██████████░░░░░░░░░░░░░░░░]  (more waste)    │
│                                                            │
│  PagedAttention: Non-contiguous block allocation           │
│                                                            │
│  Block Pool: [B1][B2][B3][B4][B5][B6][B7][B8]...           │
│                                                            │
│  Sequence A → [B1, B3, B5, B7] (only what's needed)        │
│  Sequence B → [B2, B4] (efficient)                         │
│                                                            │
│  Advantages:                                               │
│  - Minimize memory waste                                   │
│  - Dynamic allocation/deallocation                         │
│  - Copy-on-Write support (beam search efficiency)          │
└────────────────────────────────────────────────────────────┘
```

### 2.2 Using vLLM

```python
from vllm import LLM, SamplingParams

class VLLMInference:
    """vLLM inference engine"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )

    def generate(
        self,
        prompts: list,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Batch generation"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append({
                "prompt": output.prompt,
                "generated": generated_text,
                "tokens": len(output.outputs[0].token_ids)
            })

        return results

    def streaming_generate(self, prompt: str, **kwargs):
        """Streaming generation"""
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        # Async engine needed
        engine_args = AsyncEngineArgs(model=self.model_name)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Streaming implementation requires separate async code


# Running vLLM server
"""
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tensor-parallel-size 2 \
    --port 8000
"""

# Using OpenAI-compatible API
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## 3. Text Generation Inference (TGI)

### 3.1 TGI Features

```
TGI (HuggingFace):
┌────────────────────────────────────────────────────────────┐
│  Core Features:                                            │
│  - Continuous batching                                     │
│  - Flash Attention 2                                       │
│  - Tensor parallelism                                      │
│  - Token streaming                                         │
│  - Quantization (GPTQ, AWQ, EETQ)                         │
│  - Watermarking                                            │
│                                                            │
│  Supported Models:                                         │
│  - LLaMA, Mistral, Falcon                                 │
│  - GPT-2, BLOOM, StarCoder                                │
│  - T5, BART                                               │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Using TGI

```python
# Running TGI with Docker
"""
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --num-shard 2 \
    --quantize awq
"""

from huggingface_hub import InferenceClient

class TGIClient:
    """TGI client"""

    def __init__(self, endpoint: str = "http://localhost:8080"):
        self.client = InferenceClient(endpoint)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ):
        """Generate"""
        if stream:
            return self._stream_generate(prompt, max_new_tokens, temperature)

        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            details=True
        )

        return response

    def _stream_generate(self, prompt: str, max_new_tokens: int, temperature: float):
        """Streaming generation"""
        for token in self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=True
        ):
            yield token

    def get_model_info(self):
        """Model info"""
        return self.client.get_model_info()


# Usage example
def tgi_example():
    client = TGIClient()

    # Normal generation
    response = client.generate(
        "Write a short poem about AI:",
        max_new_tokens=100
    )
    print(response.generated_text)

    # Streaming
    print("\nStreaming:")
    for token in client.generate("Once upon a time,", stream=True):
        print(token, end="", flush=True)
```

---

## 4. Speculative Decoding

### 4.1 Concept

```
Speculative Decoding:
┌────────────────────────────────────────────────────────────┐
│  Idea: Generate draft with small model → Verify with large │
│                                                            │
│  Normal decoding:                                          │
│  Large Model: t1 → t2 → t3 → t4 → t5  (5 forward passes)  │
│                                                            │
│  Speculative decoding:                                     │
│  Draft Model: [t1, t2, t3, t4, t5]  (fast speculation)    │
│  Large Model: verify all at once    (1 forward pass)      │
│                                                            │
│  Result: t1 ✓, t2 ✓, t3 ✗ → regenerate                    │
│                                                            │
│  Speedup: 2-3x (depends on acceptance rate)               │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
import torch
from typing import Tuple

class SpeculativeDecoder:
    """Speculative Decoding implementation"""

    def __init__(
        self,
        target_model,  # Large model
        draft_model,   # Small model
        tokenizer,
        num_speculative_tokens: int = 5
    ):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.k = num_speculative_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate with speculative decoding"""
        generated = input_ids.clone()

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # 1. Speculate k tokens with draft model
            draft_tokens, draft_probs = self._draft_tokens(
                generated, self.k, temperature
            )

            # 2. Verify with target model
            accepted, target_probs = self._verify_tokens(
                generated, draft_tokens, temperature
            )

            # 3. Add accepted tokens
            generated = torch.cat([generated, accepted], dim=1)

            # 4. Sample from target at last rejection position
            if accepted.shape[1] < self.k:
                # Some rejected → Sample next token from target
                next_token = self._sample_from_target(
                    generated, target_probs, temperature
                )
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _draft_tokens(
        self,
        context: torch.Tensor,
        k: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate k tokens with draft model"""
        draft_tokens = []
        draft_probs = []
        current = context

        for _ in range(k):
            outputs = self.draft(current)
            logits = outputs.logits[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)

            # Sampling
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token)
            draft_probs.append(probs)

            current = torch.cat([current, token], dim=1)

        return torch.cat(draft_tokens, dim=1), torch.stack(draft_probs, dim=1)

    def _verify_tokens(
        self,
        context: torch.Tensor,
        draft_tokens: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify with target model"""
        # Forward on entire sequence at once
        full_seq = torch.cat([context, draft_tokens], dim=1)
        outputs = self.target(full_seq)

        # Target probabilities
        target_logits = outputs.logits[:, context.shape[1]-1:-1] / temperature
        target_probs = torch.softmax(target_logits, dim=-1)

        # Draft probabilities (already computed)
        draft_probs = self._get_draft_probs(context, draft_tokens, temperature)

        # Acceptance probability: min(1, p_target / p_draft)
        accepted = []
        for i in range(draft_tokens.shape[1]):
            token = draft_tokens[:, i]
            p_target = target_probs[:, i].gather(1, token.unsqueeze(1))
            p_draft = draft_probs[:, i].gather(1, token.unsqueeze(1))

            accept_prob = torch.clamp(p_target / p_draft, max=1.0)

            if torch.rand(1) < accept_prob:
                accepted.append(token)
            else:
                break

        if accepted:
            return torch.stack(accepted, dim=1), target_probs
        else:
            return torch.tensor([]).reshape(1, 0), target_probs

    def _get_draft_probs(self, context, draft_tokens, temperature):
        """Recompute draft probs"""
        full_seq = torch.cat([context, draft_tokens], dim=1)
        outputs = self.draft(full_seq)
        logits = outputs.logits[:, context.shape[1]-1:-1] / temperature
        return torch.softmax(logits, dim=-1)

    def _sample_from_target(self, context, target_probs, temperature):
        """Sample from target model"""
        # Next token at rejection position
        probs = target_probs[:, -1]
        return torch.multinomial(probs, num_samples=1)
```

---

## 5. Quantization

### 5.1 Quantization Method Comparison

| Method | Precision | Speed | Quality | Memory |
|--------|-----------|-------|---------|--------|
| FP16 | 16-bit | 1x | 100% | 1x |
| GPTQ | 4-bit | ~1.5x | 98-99% | 0.25x |
| AWQ | 4-bit | ~2x | 98-99% | 0.25x |
| GGUF | 2-8bit | ~2x | 95-99% | 0.15-0.5x |
| bitsandbytes | 4/8-bit | ~1.2x | 97-99% | 0.25-0.5x |

### 5.2 Using Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# bitsandbytes 4-bit
def load_4bit_model(model_name: str):
    """Load 4-bit quantized model"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return model


# GPTQ
def load_gptq_model(model_name: str):
    """Load GPTQ quantized model"""
    from auto_gptq import AutoGPTQForCausalLM

    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        device_map="auto",
        use_safetensors=True
    )

    return model


# AWQ
def load_awq_model(model_name: str):
    """Load AWQ quantized model"""
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_quantized(
        model_name,
        fuse_layers=True,
        device_map="auto"
    )

    return model
```

---

## 6. Batch Processing Optimization

### 6.1 Continuous Batching

```python
class ContinuousBatcher:
    """Continuous Batching implementation"""

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        max_seq_len: int = 2048
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Active requests
        self.active_requests = {}
        self.request_queue = []

    def add_request(self, request_id: str, prompt: str, max_tokens: int):
        """Add new request"""
        tokens = self.tokenizer.encode(prompt)
        self.request_queue.append({
            "id": request_id,
            "tokens": tokens,
            "generated": [],
            "max_tokens": max_tokens
        })

    def step(self) -> dict:
        """Process one step"""
        # 1. Add new requests to batch
        while (len(self.active_requests) < self.max_batch_size and
               self.request_queue):
            req = self.request_queue.pop(0)
            self.active_requests[req["id"]] = req

        if not self.active_requests:
            return {}

        # 2. Compose batch
        batch_ids, batch_tokens = self._prepare_batch()

        # 3. Forward pass
        with torch.no_grad():
            outputs = self.model(batch_tokens)
            next_tokens = outputs.logits[:, -1].argmax(dim=-1)

        # 4. Update results
        results = {}
        completed = []

        for i, req_id in enumerate(batch_ids):
            req = self.active_requests[req_id]
            token = next_tokens[i].item()
            req["generated"].append(token)

            # Check completion
            if (len(req["generated"]) >= req["max_tokens"] or
                token == self.tokenizer.eos_token_id):
                results[req_id] = self.tokenizer.decode(req["generated"])
                completed.append(req_id)

        # 5. Remove completed requests
        for req_id in completed:
            del self.active_requests[req_id]

        return results

    def _prepare_batch(self):
        """Prepare batch (with padding)"""
        batch_ids = list(self.active_requests.keys())
        sequences = []

        for req_id in batch_ids:
            req = self.active_requests[req_id]
            seq = req["tokens"] + req["generated"]
            sequences.append(seq)

        # Padding
        max_len = max(len(s) for s in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)))

        return batch_ids, torch.tensor(padded)
```

---

## 7. Performance Benchmarking

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    throughput: float  # tokens/second
    latency_p50: float  # ms
    latency_p99: float  # ms
    memory_gb: float

def benchmark_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    num_runs: int = 10
) -> BenchmarkResult:
    """Inference benchmark"""
    latencies = []
    total_tokens = 0

    # Warmup
    for prompt in prompts[:2]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    for _ in range(num_runs):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # ms
            total_tokens += outputs.shape[1] - inputs["input_ids"].shape[1]

    # Memory
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        memory_gb = 0

    latencies.sort()

    return BenchmarkResult(
        throughput=total_tokens / (sum(latencies) / 1000),
        latency_p50=latencies[len(latencies) // 2],
        latency_p99=latencies[int(len(latencies) * 0.99)],
        memory_gb=memory_gb
    )
```

---

## Key Summary

### Inference Optimization Techniques
```
1. PagedAttention (vLLM): KV cache efficiency
2. Continuous Batching: Dynamic batch processing
3. Speculative Decoding: Draft+Verify
4. Quantization: 4-bit/8-bit compression
5. Flash Attention: Memory-efficient attention
6. Tensor Parallelism: Multi-GPU distribution
```

### Tool Selection Guide
```
- High throughput serving: vLLM
- HuggingFace integration: TGI
- Edge devices: llama.cpp + GGUF
- Development/experimentation: Transformers + bitsandbytes
```

---

## References

1. Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. Leviathan et al. (2023). "Fast Inference from Transformers via Speculative Decoding"
3. [vLLM Documentation](https://docs.vllm.ai/)
4. [TGI Documentation](https://huggingface.co/docs/text-generation-inference/)
