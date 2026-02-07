# Scaling Laws

## Learning Objectives
- Understand the concept and mathematical form of Scaling Laws
- Compare Kaplan et al. vs Chinchilla laws
- Learn compute-optimal training strategies
- Grasp how to apply Scaling Laws in practice

---

## 1. What are Scaling Laws?

### 1.1 Definition

**Scaling Laws** are empirical laws that describe the relationship between **number of parameters (N)**, **amount of data (D)**, **compute (C)**, and **performance (Loss)** of models.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Core Relationships in Scaling Laws            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss ≈ A/N^α + B/D^β + E                                       │
│                                                                 │
│  N = Number of model parameters                                 │
│  D = Number of training data tokens                             │
│  C = Compute (FLOPs) ≈ 6 × N × D                                │
│  E = Irreducible minimum loss (entropy of data)                 │
│                                                                 │
│  Key findings:                                                  │
│  • Loss decreases according to Power Law with respect to N, D   │
│  • When C is fixed, there exists an optimal ratio of N and D    │
│  • Larger models utilize data more efficiently                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Important?

```python
"""
Practical value of Scaling Laws:

1. Cost Prediction
   - Estimate required resources before training
   - "How much is needed to train a 10B model?"

2. Optimal Allocation
   - Decide model size vs data amount with fixed budget
   - "What's the best setup with $100M budget?"

3. Performance Prediction
   - Estimate large model performance from small models
   - "With current 7B model, how much better will 70B be?"

4. Research Planning
   - Determine research directions with high ROI
   - "Should we increase data or scale up model?"
"""
```

---

## 2. Kaplan Scaling Laws (2020)

### 2.1 OpenAI's Initial Research

Laws discovered in Kaplan et al.'s 2020 paper "Scaling Laws for Neural Language Models":

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kaplan Scaling Laws                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Loss vs Parameters                                          │
│     L(N) = (N_c / N)^α_N, where α_N ≈ 0.076                     │
│                                                                 │
│  2. Loss vs Data                                                │
│     L(D) = (D_c / D)^α_D, where α_D ≈ 0.095                     │
│                                                                 │
│  3. Loss vs Compute                                             │
│     L(C) = (C_c / C)^α_C, where α_C ≈ 0.050                     │
│                                                                 │
│  Key claims:                                                    │
│  • Parameter count is most important (α_N < α_D)                │
│  • For same compute, larger model + less data is better         │
│  • N ∝ C^0.73, D ∝ C^0.27 (Compute-optimal allocation)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Visualization

```
   Loss (Log)
       │
   3.5 ├─●───────────────────────────── 100M params
       │   ╲
   3.0 ├─────●─────────────────────────  1B params
       │       ╲
   2.5 ├─────────●───────────────────── 10B params
       │           ╲
   2.0 ├─────────────●─────────────────100B params
       │               ╲
   1.5 ├─────────────────●────────────  1T params (predicted)
       │
       └───┬───┬───┬───┬───┬───┬───┬──▶
          10^18  19   20   21   22   23   Compute (FLOPs)

   • Straight line = Power Law (linear in log scale)
   • Slope = α_C ≈ 0.05
```

### 2.3 Model Design Following Kaplan's Law

```python
"""
Example application of Kaplan's law:

Compute budget: 10^21 FLOPs

Kaplan optimal allocation:
- N ∝ C^0.73 → N ≈ 10^15 (about 1 trillion parameters?!)
- D ∝ C^0.27 → D ≈ 10^9 (about 1 billion tokens)

Problem:
- Model becomes too large with insufficient data
- GPT-3 (175B) followed this law but...
- Chinchilla refuted this
"""
```

---

## 3. Chinchilla Scaling Laws (2022)

### 3.1 DeepMind's Rediscovery

Hoffmann et al.'s "Training Compute-Optimal Large Language Models" revised Kaplan's law:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chinchilla Scaling Laws                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Key finding: Existing models are Under-trained!                │
│                                                                 │
│  Compute-optimal scaling:                                        │
│  • N ∝ C^0.5  (number of parameters)                            │
│  • D ∝ C^0.5  (number of data tokens)                           │
│  • i.e., N and D should increase at the same rate for optimality│
│                                                                 │
│  Practical rule:                                                │
│  D ≈ 20 × N  (tokens ≈ 20 × parameters)                         │
│                                                                 │
│  Examples:                                                      │
│  • 1B model → 20B tokens needed                                 │
│  • 7B model → 140B tokens needed                                │
│  • 70B model → 1.4T tokens needed                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Chinchilla vs Gopher Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│               Chinchilla (70B) vs Gopher (280B)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model      │ Parameters│ Train Tokens│ Compute   │ Performance│
│  ───────────│──────────│─────────────│───────────│────────────│
│  Gopher     │ 280B     │ 300B        │ 5.0×10^23 │ Baseline   │
│  Chinchilla │ 70B      │ 1.4T        │ 5.0×10^23 │ +10% better│
│                                                                 │
│  Conclusion:                                                    │
│  • 4x smaller model performs better with same compute!          │
│  • Gopher is Under-trained (insufficient data)                  │
│  • Simply increasing model size is inefficient                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Status of Existing Models

```
             Tokens (D)
                │
          10T   ├                               ● LLaMA 2 (2023)
                │                           ●
           1T   ├                       ● Chinchilla (Optimal)
                │                   ╱
         100B   ├               ╱       ● GPT-3 (Under-trained)
                │           ╱
          10B   ├       ╱
                │   ╱                   ● Gopher (Very Under-trained)
           1B   ├─
                └───┬───┬───┬───┬───┬───┬───┬───▶
                   1B  10B 100B  1T  10T      Parameters (N)

             ╱ = Compute-optimal frontier (D ≈ 20N)

             Points below the line are Under-trained
```

---

## 4. Mathematical Formulation

### 4.1 Loss Function

```python
"""
Mathematical form of Scaling Law:

1. Single Variable Scaling
   L(N) = (N_c / N)^α + L_∞     # Consider parameters only
   L(D) = (D_c / D)^β + L_∞     # Consider data only

2. Combined Scaling (Chinchilla)
   L(N, D) = E + A/N^α + B/D^β

   where:
   - E ≈ 1.69 (irreducible loss, data entropy)
   - A ≈ 406.4
   - B ≈ 410.7
   - α ≈ 0.34
   - β ≈ 0.28

3. Compute Perspective
   C ≈ 6 × N × D  (FLOPs for training)

   Optimization: min L(N, D) subject to C = 6ND

   Result: N* ∝ C^0.5, D* ∝ C^0.5
"""
```

### 4.2 Scaling Law Simulation with Python

```python
import numpy as np
import matplotlib.pyplot as plt

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """
    Calculate Loss according to Chinchilla Scaling Law

    Args:
        N: Number of parameters (billions)
        D: Number of tokens (billions)

    Returns:
        Expected Loss (log of perplexity)
    """
    return E + A / (N ** alpha) + B / (D ** beta)

def optimal_allocation(compute_budget, flops_per_token=6):
    """
    Calculate optimal N, D for given compute budget

    Args:
        compute_budget: Total FLOPs (e.g., 10^23)
        flops_per_token: FLOPs per token (approximately 6N)

    Returns:
        optimal_N, optimal_D (in billions)
    """
    # Chinchilla optimal ratio: D ≈ 20N
    # C = 6 * N * D = 6 * N * 20N = 120 * N^2
    # N = sqrt(C / 120)

    optimal_N = np.sqrt(compute_budget / 120) / 1e9  # billions
    optimal_D = 20 * optimal_N                        # billions

    return optimal_N, optimal_D

# Example: 10^23 FLOPs budget
compute = 1e23
N_opt, D_opt = optimal_allocation(compute)
print(f"Compute budget: 10^23 FLOPs")
print(f"Optimal parameters: {N_opt:.1f}B")
print(f"Optimal tokens: {D_opt:.1f}B")
print(f"Expected loss: {chinchilla_loss(N_opt, D_opt):.3f}")

# Visualization: Loss according to N vs D
N_range = np.logspace(0, 3, 50)  # 1B to 1000B
D_range = np.logspace(0, 4, 50)  # 1B to 10000B

N_grid, D_grid = np.meshgrid(N_range, D_range)
Loss_grid = chinchilla_loss(N_grid, D_grid)

plt.figure(figsize=(10, 8))
plt.contour(N_grid, D_grid, Loss_grid, levels=20)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Parameters N (Billions)')
plt.ylabel('Tokens D (Billions)')
plt.title('Chinchilla Scaling Law: Loss Contours')
plt.colorbar(label='Loss')
plt.plot(N_range, 20*N_range, 'r--', label='Optimal ratio (D=20N)')
plt.legend()
plt.show()
```

---

## 5. Application in Real Models

### 5.1 Scaling Comparison of Major Models

| Model | Parameters (N) | Tokens (D) | D/N Ratio | Status |
|------|-------------|----------|----------|------|
| GPT-3 | 175B | 300B | 1.7 | Under-trained |
| Gopher | 280B | 300B | 1.1 | Very Under-trained |
| Chinchilla | 70B | 1.4T | 20 | Optimal |
| LLaMA 1 | 65B | 1.4T | 21.5 | Near-optimal |
| LLaMA 2 | 70B | 2T | 28.6 | Slight Over-trained |
| Mistral | 7B | 8T (est.) | ~1000 | Over-trained |

### 5.2 Benefits of Over-training

```
┌─────────────────────────────────────────────────────────────────┐
│                Over-training Strategy (LLaMA 2, Mistral)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Chinchilla is "training" optimal but not "deployment" optimal! │
│                                                                 │
│  From deployment perspective:                                   │
│  • Inference cost ∝ N (model size)                              │
│  • Training once, inference trillions of times                  │
│                                                                 │
│  Therefore:                                                     │
│  • Smaller model + more data = inference efficient              │
│  • "Inference-optimal" ≠ "Compute-optimal"                      │
│                                                                 │
│  LLaMA 2 strategy:                                              │
│  • 70B model with 2T tokens (D/N ≈ 29)                          │
│  • Train longer than Chinchilla                                 │
│  • Result: Better performance with smaller model                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Practical Guidelines

```python
"""
Scaling strategies in practice:

1. Research/Experimentation Phase (Compute-limited)
   - Follow Chinchilla rule: D ≈ 20N
   - Iterate quickly with smaller models

2. Production Deployment (Inference-limited)
   - Consider over-training: D > 20N
   - Smaller model + more data
   - Example: Mistral 7B > LLaMA 2 13B (on some tasks)

3. Budget Planning
   - C = 6 * N * D (FLOPs)
   - GPU hours ≈ C / (GPU_FLOPS * utilization)
   - Example: A100 80GB = ~300 TFLOPS (effective)

4. Scale-up Strategy
   - Tune hyperparameters with small models
   - Predict large model performance with Scaling Law
   - Execute large-scale training after validation
"""

def estimate_training_cost(N_billions, D_billions, gpu_price_per_hour=2.0):
    """
    Estimate training cost

    Args:
        N_billions: Number of parameters (B)
        D_billions: Number of tokens (B)
        gpu_price_per_hour: GPU cost per hour (USD)

    Returns:
        dict: Expected cost information
    """
    N = N_billions * 1e9
    D = D_billions * 1e9

    # 6ND FLOPs for training
    total_flops = 6 * N * D

    # A100 80GB: ~300 TFLOPS effective
    gpu_tflops = 300
    gpu_flops = gpu_tflops * 1e12

    # Total GPU time
    total_gpu_seconds = total_flops / gpu_flops
    total_gpu_hours = total_gpu_seconds / 3600

    # Cost
    total_cost = total_gpu_hours * gpu_price_per_hour

    return {
        "total_flops": f"{total_flops:.2e}",
        "gpu_hours": f"{total_gpu_hours:,.0f}",
        "cost_usd": f"${total_cost:,.0f}",
        "cost_with_8gpus": f"${total_cost/8:,.0f} ({total_gpu_hours/8:,.0f} hours)"
    }

# Example: LLaMA 2 7B training cost
cost_7b = estimate_training_cost(7, 2000)
print("LLaMA 2 7B (2T tokens):")
for k, v in cost_7b.items():
    print(f"  {k}: {v}")
```

---

## 6. Extensions of Scaling Laws

### 6.1 Scaling in Other Domains

```
┌─────────────────────────────────────────────────────────────────┐
│                    Scaling Laws by Domain                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Vision (ViT):                                                  │
│  • Similar power law observed                                   │
│  • α ≈ 0.05 (smaller than Language)                            │
│  • Data quality is more important                               │
│                                                                 │
│  Multimodal (CLIP):                                             │
│  • Separate optimization needed for image and text scaling      │
│  • Quality of data pairs is critical                            │
│                                                                 │
│  Code:                                                          │
│  • Steeper scaling (larger α)                                   │
│  • High-quality code data is scarce                             │
│                                                                 │
│  Reasoning:                                                     │
│  • Not smooth due to emergent behavior                          │
│  • Sudden performance improvements at specific thresholds       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Fine-tuning Scaling Laws

```python
"""
Scaling Law also applies to fine-tuning:

Research findings:
- Larger base model = less fine-tuning data needed
- Fine-tuning data also scales according to power law
- PEFT like LoRA follows similar patterns

Practical rules:
- Base model size × 10 = Fine-tuning data amount (approximately)
- 7B model: ~1K-10K examples
- 70B model: ~100-1K examples (to achieve same performance)

However, quality > quantity:
- 100 high-quality examples > 10,000 low-quality examples
"""
```

### 6.3 Inference Scaling (Test-time Compute)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Scaling (o1-style)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Traditional Scaling: Increase compute during training          │
│  Inference Scaling: Increase compute during inference           │
│                                                                 │
│  Methods:                                                       │
│  • Generate longer Chain-of-Thought                             │
│  • Generate multiple answers and vote (Self-consistency)        │
│  • Tree of Thoughts / Beam Search                               │
│  • Iterative Verification/Refinement                            │
│                                                                 │
│  Effects:                                                       │
│  • Significantly improved accuracy on difficult problems        │
│  • Performance improvement possible without training            │
│  • Paradigm shift from GPT-4 → o1 (inference-time scaling)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Limitations of Scaling

### 7.1 Physical Limits

```python
"""
Real limitations of Scaling:

1. Data Limits
   - Total internet text: ~10-50T tokens
   - High-quality data is much less
   - As of 2024, data exhaustion discussion beginning

2. Compute Limits
   - Power consumption (MW scale)
   - Semiconductor supply
   - Cost (billions of dollars)

3. Architecture Limits
   - Attention's O(n²) complexity
   - Memory bandwidth bottleneck
   - Communication overhead in distributed training

4. Diminishing Returns
   - α ≈ 0.05 means 10x compute → ~12% loss reduction
   - Increasingly larger investments needed
"""
```

### 7.2 Improvement Directions Beyond Scaling

| Direction | Description | Examples |
|------|------|------|
| **Architecture** | More efficient structures | Mamba, RWKV, Hyena |
| **Data Quality** | High-quality data curation | Phi, LIMA |
| **Synthetic Data** | Generate training data with AI | Self-Instruct |
| **Efficient Training** | Improve training efficiency | Flash Attention, ZeRO |
| **Test-time Compute** | Increase compute during inference | CoT, Self-consistency, o1 |

---

## Summary

### Key Concepts
- **Scaling Laws**: Power law relationship between parameters, data, compute, and performance
- **Kaplan**: Prioritize N (large model + less data)
- **Chinchilla**: Balance N and D (D ≈ 20N)
- **Over-training**: Train smaller models longer for inference efficiency

### Practical Formulas
```
Compute-optimal: D ≈ 20 × N (tokens)
Training FLOPs: C ≈ 6 × N × D
Inference-optimal: Smaller N, larger D
```

### Next Steps
- [03_Emergent_Abilities.md](03_Emergent_Abilities.md): Emergent abilities at scale
- [08_LLaMA_Family.md](08_LLaMA_Family.md): Scaling application case (LLaMA)

---

## References

### Key Papers
- Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
- Touvron et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"

### Additional Resources
- [Epoch AI Compute Trends](https://epochai.org/trends)
- [AI Scaling Calculator](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/ai-scaling-calculator)
