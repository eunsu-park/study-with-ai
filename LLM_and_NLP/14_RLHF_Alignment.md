# 14. RLHF와 LLM 정렬 (Alignment)

## 학습 목표

- RLHF(Reinforcement Learning from Human Feedback) 이해
- Reward Model 학습
- PPO를 통한 정책 최적화
- DPO(Direct Preference Optimization)
- Constitutional AI와 안전한 AI

---

## 1. LLM 정렬 개요

### 왜 정렬이 필요한가?

```
┌─────────────────────────────────────────────────────────────┐
│                  LLM 정렬의 필요성                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  사전학습 모델 (Base Model)                                   │
│      │                                                       │
│      │  문제점:                                               │
│      │  - 단순히 다음 토큰 예측                                │
│      │  - 유해한 콘텐츠 생성 가능                              │
│      │  - 지시사항 따르기 어려움                               │
│      ▼                                                       │
│  정렬된 모델 (Aligned Model)                                  │
│      │                                                       │
│      │  목표:                                                 │
│      │  - 도움됨 (Helpful)                                   │
│      │  - 무해함 (Harmless)                                  │
│      │  - 정직함 (Honest)                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 정렬 방법론 발전

```
SFT (Supervised Fine-Tuning)
    │  고품질 데이터로 지도학습
    ▼
RLHF (Reinforcement Learning from Human Feedback)
    │  보상 모델 + 강화학습
    ▼
DPO (Direct Preference Optimization)
    │  직접 선호도 최적화
    ▼
Constitutional AI
    │  원칙 기반 자기 개선
```

---

## 2. SFT (Supervised Fine-Tuning)

### 기본 개념

```python
# SFT 데이터 형식
sft_data = [
    {
        "instruction": "Write a poem about spring.",
        "input": "",
        "output": "Flowers bloom in gentle rain,\nBirds return to sing again..."
    },
    {
        "instruction": "Translate to French.",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    }
]
```

### SFT 구현

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 모델과 토크나이저
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# 포맷팅 함수
def format_instruction(example):
    if example["context"]:
        return f"""### Instruction:
{example['instruction']}

### Context:
{example['context']}

### Response:
{example['response']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""

# 학습 설정
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_instruction,
    max_seq_length=1024,
    args=training_args,
)

trainer.train()
```

---

## 3. RLHF 파이프라인

### 전체 프로세스

```
┌────────────────────────────────────────────────────────────────┐
│                     RLHF 파이프라인                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1단계: SFT                                                     │
│  ┌─────────────┐    지도학습    ┌─────────────┐                │
│  │ Base Model  │ ──────────▶  │  SFT Model  │                │
│  └─────────────┘              └─────────────┘                │
│                                                                 │
│  2단계: Reward Model 학습                                       │
│  ┌─────────────┐    선호도 데이터   ┌─────────────┐            │
│  │ SFT Model   │ ──────────────▶ │Reward Model│            │
│  └─────────────┘                  └─────────────┘            │
│                                                                 │
│  3단계: PPO 강화학습                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │  SFT Model  │ + │Reward Model│ ▶ │ RLHF Model │          │
│  │  (Policy)   │   │  (Critic)  │   │ (Aligned)  │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 선호도 데이터 수집

```python
# 선호도 데이터 형식
preference_data = [
    {
        "prompt": "Write a haiku about mountains.",
        "chosen": "Peaks touch morning sky\nSilent guardians of earth\nMist embraces stone",
        "rejected": "Mountains are big\nThey are tall and rocky\nI like mountains"
    },
    {
        "prompt": "Explain quantum computing.",
        "chosen": "Quantum computing harnesses quantum mechanics principles...",
        "rejected": "Quantum computing is computers that use quantum stuff..."
    }
]

# HuggingFace 형식
from datasets import Dataset

dataset = Dataset.from_list(preference_data)
dataset = dataset.map(lambda x: {
    "prompt": x["prompt"],
    "chosen": x["chosen"],
    "rejected": x["rejected"]
})
```

---

## 4. Reward Model 학습

### Reward Model 개념

```
입력: (prompt, response)
출력: scalar reward (점수)

학습 목표:
    reward(prompt, chosen) > reward(prompt, rejected)
```

### 구현

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer

# Reward Model (분류 헤드 추가)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    num_labels=1  # 스칼라 출력
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 데이터 전처리
def preprocess_reward_data(examples):
    """선호도 데이터를 Reward 학습용으로 변환"""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }

    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # Chosen
        chosen_text = f"### Prompt: {prompt}\n### Response: {chosen}"
        chosen_tokenized = tokenizer(chosen_text, truncation=True, max_length=512)
        new_examples["input_ids_chosen"].append(chosen_tokenized["input_ids"])
        new_examples["attention_mask_chosen"].append(chosen_tokenized["attention_mask"])

        # Rejected
        rejected_text = f"### Prompt: {prompt}\n### Response: {rejected}"
        rejected_tokenized = tokenizer(rejected_text, truncation=True, max_length=512)
        new_examples["input_ids_rejected"].append(rejected_tokenized["input_ids"])
        new_examples["attention_mask_rejected"].append(rejected_tokenized["attention_mask"])

    return new_examples

# 학습
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    fp16=True,
)

trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Reward Model 사용

```python
def get_reward(model, tokenizer, prompt, response):
    """응답에 대한 보상 점수 계산"""
    text = f"### Prompt: {prompt}\n### Response: {response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        reward = outputs.logits.squeeze().item()

    return reward

# 사용 예시
prompt = "Explain photosynthesis."
response_good = "Photosynthesis is the process by which plants convert sunlight..."
response_bad = "Plants eat light."

print(f"Good response reward: {get_reward(reward_model, tokenizer, prompt, response_good):.4f}")
print(f"Bad response reward: {get_reward(reward_model, tokenizer, prompt, response_bad):.4f}")
```

---

## 5. PPO (Proximal Policy Optimization)

### PPO 개념

```
PPO 목표함수:
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

여기서:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (확률 비율)
    A_t = 어드밴티지 (Reward - Baseline)
    ε = 클리핑 범위 (보통 0.2)

KL 제약:
    D_KL[π_θ || π_ref] < δ  (기준 모델과 너무 멀어지지 않도록)
```

### PPO 구현 (TRL)

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch

# PPO 설정
ppo_config = PPOConfig(
    model_name="./sft_model",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    max_grad_norm=0.5,
    kl_penalty="kl",           # KL 페널티 방식
    target_kl=0.1,             # 목표 KL divergence
    init_kl_coef=0.2,          # 초기 KL 계수
)

# 모델 (Value head 포함)
model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")  # 기준 모델 (고정)
tokenizer = AutoTokenizer.from_pretrained("./sft_model")
tokenizer.pad_token = tokenizer.eos_token

# Reward Model 로드
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model")
reward_tokenizer = AutoTokenizer.from_pretrained("./reward_model")

# PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 학습 루프
def get_reward_batch(prompts, responses):
    """배치 보상 계산"""
    rewards = []
    for prompt, response in zip(prompts, responses):
        text = f"### Prompt: {prompt}\n### Response: {response}"
        inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            reward = reward_model(**inputs).logits.squeeze()
        rewards.append(reward)

    return rewards

# 학습
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

for epoch in range(ppo_config.ppo_epochs):
    for batch in dataset.iter(batch_size=ppo_config.batch_size):
        # 프롬프트 토큰화
        query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze() for p in batch["prompt"]]

        # 응답 생성
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, max_new_tokens=128)
            response_tensors.append(response.squeeze())

        # 텍스트 디코딩
        prompts = batch["prompt"]
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # 보상 계산
        rewards = get_reward_batch(prompts, responses)

        # PPO 스텝
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 로깅
        ppo_trainer.log_stats(stats, batch, rewards)

# 저장
model.save_pretrained("./rlhf_model")
```

---

## 6. DPO (Direct Preference Optimization)

### DPO 개념

```
DPO = RLHF without Reward Model

핵심 아이디어:
    - Reward Model 없이 직접 선호도 데이터로 학습
    - Bradley-Terry 모델 기반
    - 더 안정적이고 간단한 학습

손실 함수:
    L_DPO = -E[log σ(β(log π_θ(y_w|x) - log π_ref(y_w|x)
                      - log π_θ(y_l|x) + log π_ref(y_l|x)))]

여기서:
    y_w = 선호 응답 (winner)
    y_l = 비선호 응답 (loser)
    β = 온도 파라미터
```

### DPO vs RLHF

| 항목 | RLHF | DPO |
|------|------|-----|
| Reward Model | 필요 | 불필요 |
| 학습 안정성 | 불안정 | 안정적 |
| 하이퍼파라미터 | 많음 | 적음 |
| 메모리 | 높음 | 낮음 |
| 성능 | 우수 | 동등 이상 |

### DPO 구현

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 모델
model = AutoModelForCausalLM.from_pretrained("./sft_model")
ref_model = AutoModelForCausalLM.from_pretrained("./sft_model")
tokenizer = AutoTokenizer.from_pretrained("./sft_model")
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 (prompt, chosen, rejected 형식)
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# DPO 설정
dpo_config = DPOConfig(
    beta=0.1,                          # 온도 파라미터
    loss_type="sigmoid",               # sigmoid 또는 hinge
    max_length=512,
    max_prompt_length=256,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    output_dir="./dpo_model",
    fp16=True,
)

# DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 학습
dpo_trainer.train()

# 저장
model.save_pretrained("./dpo_model_final")
```

### DPO 변형들

```python
# IPO (Identity Preference Optimization)
dpo_config = DPOConfig(
    loss_type="ipo",
    label_smoothing=0.0,
)

# KTO (Kahneman-Tversky Optimization)
# 선호/비선호 쌍 대신 개별 평가 사용
from trl import KTOConfig, KTOTrainer

kto_config = KTOConfig(
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

# ORPO (Odds Ratio Preference Optimization)
# Reference model 불필요
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    beta=0.1,
    # ref_model 없이 학습
)
```

---

## 7. Constitutional AI

### 개념

```
Constitutional AI (CAI) = 원칙 기반 자기 개선

단계:
    1. 모델이 응답 생성
    2. 헌법(원칙)에 따라 자기 비평
    3. 비평을 바탕으로 응답 수정
    4. 수정된 응답으로 학습

원칙 예시:
    - "도움이 되어야 함"
    - "해로운 내용을 포함하지 않아야 함"
    - "정직해야 함"
    - "개인정보를 노출하지 않아야 함"
```

### CAI 구현

```python
from openai import OpenAI

client = OpenAI()

# 원칙 (Constitution)
constitution = """
1. 응답은 도움이 되어야 합니다.
2. 응답은 해로운 내용을 포함하지 않아야 합니다.
3. 응답은 정직하고 사실에 기반해야 합니다.
4. 개인정보나 민감한 정보를 공개하지 않아야 합니다.
5. 차별적이거나 편견 있는 내용을 포함하지 않아야 합니다.
"""

def generate_initial_response(prompt):
    """초기 응답 생성"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def critique_response(prompt, response, constitution):
    """응답 비평"""
    critique_prompt = f"""다음 응답이 주어진 원칙을 잘 따르는지 평가하세요.

원칙:
{constitution}

사용자 질문: {prompt}

응답: {response}

각 원칙에 대해 응답이 어떻게 위반하거나 준수하는지 분석하세요.
"""
    critique = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0.3
    )
    return critique.choices[0].message.content

def revise_response(prompt, response, critique, constitution):
    """응답 수정"""
    revision_prompt = f"""다음 비평을 바탕으로 응답을 개선하세요.

원칙:
{constitution}

사용자 질문: {prompt}

원래 응답: {response}

비평: {critique}

원칙을 더 잘 준수하도록 응답을 수정하세요. 수정된 응답만 출력하세요.
"""
    revised = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": revision_prompt}],
        temperature=0.3
    )
    return revised.choices[0].message.content

def constitutional_ai_pipeline(prompt, constitution, iterations=2):
    """CAI 파이프라인"""
    response = generate_initial_response(prompt)
    print(f"초기 응답:\n{response}\n")

    for i in range(iterations):
        critique = critique_response(prompt, response, constitution)
        print(f"비평 {i+1}:\n{critique}\n")

        response = revise_response(prompt, response, critique, constitution)
        print(f"수정된 응답 {i+1}:\n{response}\n")

    return response

# 사용
prompt = "How can I pick a lock?"
final_response = constitutional_ai_pipeline(prompt, constitution)
```

---

## 8. 고급 정렬 기법

### RLAIF (RL from AI Feedback)

```python
def get_ai_preference(prompt, response_a, response_b):
    """AI가 선호도 판단"""
    judge_prompt = f"""다음 두 응답 중 더 좋은 것을 선택하세요.

질문: {prompt}

응답 A: {response_a}

응답 B: {response_b}

평가 기준:
- 정확성
- 유용성
- 명확성
- 안전성

더 좋은 응답 (A 또는 B)와 이유를 말하세요.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

### Self-Play Fine-Tuning (SPIN)

```python
# SPIN: 모델이 자신의 응답과 경쟁

def spin_iteration(model, dataset):
    """SPIN 반복"""
    # 1. 현재 모델로 응답 생성
    synthetic_responses = generate_responses(model, dataset["prompts"])

    # 2. 실제 응답 vs 생성된 응답으로 DPO
    spin_dataset = {
        "prompt": dataset["prompts"],
        "chosen": dataset["responses"],      # 실제 응답
        "rejected": synthetic_responses      # 모델 생성 응답
    }

    # 3. DPO 학습
    model = dpo_train(model, spin_dataset)

    return model
```

---

## 정리

### 정렬 방법 비교

| 방법 | 복잡도 | 성능 | 사용 시점 |
|------|--------|------|----------|
| SFT | 낮음 | 기본 | 항상 첫 단계 |
| RLHF (PPO) | 높음 | 우수 | 복잡한 정렬 |
| DPO | 중간 | 우수 | 간단한 정렬 |
| ORPO | 낮음 | 좋음 | 메모리 제한 |
| CAI | 중간 | 안전성 | 안전 중요 |

### 핵심 코드

```python
# SFT
from trl import SFTTrainer
trainer = SFTTrainer(model, train_dataset, formatting_func=format_fn)

# DPO
from trl import DPOTrainer, DPOConfig
config = DPOConfig(beta=0.1)
trainer = DPOTrainer(model, ref_model, args=config, train_dataset=dataset)

# PPO
from trl import PPOTrainer, PPOConfig
config = PPOConfig(target_kl=0.1)
trainer = PPOTrainer(config, model, ref_model, tokenizer)
stats = trainer.step(queries, responses, rewards)
```

### 정렬 파이프라인

```
1. SFT: 고품질 데이터로 기본 능력 학습
2. 선호도 데이터 수집 (인간 또는 AI)
3. DPO/RLHF로 선호도 학습
4. 안전성 평가 및 추가 정렬
5. 배포 및 피드백 수집
```

---

## 다음 단계

[15_LLM_Agents.md](./15_LLM_Agents.md)에서 도구 사용과 에이전트 시스템을 학습합니다.
