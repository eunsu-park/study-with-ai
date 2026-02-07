# 11. Training Optimization

## Learning Objectives

- Hyperparameter tuning strategies
- Advanced learning rate scheduling
- Mixed Precision Training
- Gradient Accumulation

---

## 1. Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Impact | Typical Range |
|-----------|--------|--------------|
| Learning Rate | Convergence speed/stability | 1e-5 ~ 1e-2 |
| Batch Size | Memory/generalization | 16 ~ 512 |
| Weight Decay | Overfitting prevention | 1e-5 ~ 1e-2 |
| Dropout | Overfitting prevention | 0.1 ~ 0.5 |
| Epochs | Training amount | Data-dependent |

### Search Strategies

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

### Using Optuna

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

## 2. Advanced Learning Rate Scheduling

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

### Concept

```
FP32 (32-bit) → FP16 (16-bit)
- Memory savings (approximately 50%)
- Speed improvement (approximately 2-3x)
- Accuracy preservation
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

### Complete Training Loop

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

### Concept

```
Multiple small batches → Large batch effect
Useful when GPU memory is limited
```

### Implementation

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

### Combined with AMP

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

### Preventing Gradient Explosion

```python
# Norm 클리핑 (권장)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Value 클리핑
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### In Training Loop

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

## 6. Advanced Early Stopping

### Patience and Delta

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

## 7. Training Monitoring

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

## 8. Reproducibility

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

## Summary

### Checklist

- [ ] Set learning rate appropriately (recommend starting with 1e-4)
- [ ] Use warmup (essential for Transformers)
- [ ] Apply Mixed Precision (GPU efficiency)
- [ ] Gradient Clipping (RNN/Transformer)
- [ ] Configure early stopping
- [ ] Set reproducibility seed
- [ ] Set up logging/monitoring

### Recommended Configuration

```python
# 기본 최적화 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(loader))
scaler = GradScaler()  # AMP
early_stopping = EarlyStopping(patience=10)
```

---

## Next Steps

Learn about model saving and deployment in [12_Model_Saving_Deployment.md](./12_Model_Saving_Deployment.md).
