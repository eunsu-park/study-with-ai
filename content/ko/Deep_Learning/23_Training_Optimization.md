[이전: Vision Transformer (ViT)](./22_Impl_ViT.md) | [다음: 손실 함수](./24_Loss_Functions.md)

---

# 23. 학습 최적화

## 학습 목표

- 하이퍼파라미터 튜닝 전략
- 학습률 스케줄링 심화
- Mixed Precision Training
- Gradient Accumulation

---

## 1. 하이퍼파라미터 튜닝

### 주요 하이퍼파라미터

| 파라미터 | 영향 | 일반적 범위 |
|----------|------|------------|
| Learning Rate | 수렴 속도/안정성 | 1e-5 ~ 1e-2 |
| Batch Size | 메모리/일반화 | 16 ~ 512 |
| Weight Decay | 과적합 방지 | 1e-5 ~ 1e-2 |
| Dropout | 과적합 방지 | 0.1 ~ 0.5 |
| Epochs | 학습량 | 데이터 의존적 |

### 탐색 전략

```python
# Grid Search
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        train_and_evaluate(lr, bs)

# Random Search (더 효율적)
import random
for _ in range(20):
    lr = 10 ** random.uniform(-5, -2)  # 로그 스케일
    bs = random.choice([16, 32, 64, 128])
    train_and_evaluate(lr, bs)
```

### Optuna 사용

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

    model = create_model(dropout)
    accuracy = train_and_evaluate(model, lr, batch_size)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best accuracy: {study.best_value}")
```

---

## 2. 학습률 스케줄링 심화

### Warmup

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.base_lr * min(1.0, self.step_num / self.warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### Warmup + Cosine Decay

```python
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### OneCycleLR

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,      # 10% warmup
    anneal_strategy='cos'
)

# 매 배치마다 호출
for batch in train_loader:
    train_step(batch)
    scheduler.step()
```

---

## 3. Mixed Precision Training

### 개념

```
FP32 (32비트) → FP16 (16비트)
- 메모리 절약 (약 50%)
- 속도 향상 (약 2-3배)
- 정확도 유지
```

### PyTorch AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    # 자동 Mixed Precision
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # 스케일링된 역전파
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 전체 학습 루프

```python
def train_with_amp(model, train_loader, optimizer, epochs):
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            with autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

---

## 4. Gradient Accumulation

### 개념

```
작은 배치를 여러 번 → 큰 배치 효과
GPU 메모리 부족 시 유용
```

### 구현

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # 스케일링
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### AMP와 함께 사용

```python
accumulation_steps = 4
scaler = GradScaler()
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## 5. Gradient Clipping

### 기울기 폭발 방지

```python
# Norm 클리핑 (권장)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Value 클리핑
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 학습 루프에서

```python
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # 클리핑 후 업데이트
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## 6. 조기 종료 심화

### Patience와 Delta

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
```

---

## 7. 학습 모니터링

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
```

### Weights & Biases

```python
import wandb

wandb.init(project="my-project", config={
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": epochs
})

for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

wandb.finish()
```

---

## 8. 재현성 (Reproducibility)

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 정리

### 체크리스트

- [ ] 학습률 적절히 설정 (1e-4 시작 권장)
- [ ] Warmup 사용 (Transformer 필수)
- [ ] Mixed Precision 적용 (GPU 효율)
- [ ] Gradient Clipping (RNN/Transformer)
- [ ] 조기 종료 설정
- [ ] 재현성 시드 설정
- [ ] 로깅/모니터링 설정

### 권장 설정

```python
# 기본 최적화 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(loader))
scaler = GradScaler()  # AMP
early_stopping = EarlyStopping(patience=10)
```

---

## 다음 단계

[41_Model_Saving_Deployment.md](./41_Model_Saving_Deployment.md)에서 모델 저장과 배포를 학습합니다.
