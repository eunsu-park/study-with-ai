# Python Study Guide

## Introduction

This folder contains materials for systematically learning Python from basics to advanced syntax.

**Target Audience**:
- **Need basics review**: Start with lessons 15, 16 (basic syntax, OOP)
- **Intermediate and above**: Start with lesson 01 (type hints)

---

## Learning Roadmap

```
[Intermediate]          [Intermediate+]         [Advanced]
  │                         │                       │
  ▼                         ▼                       ▼
Type Hints ──────▶ Iterators ───────▶ Descriptors
  │                         │                       │
  ▼                         ▼                       │
Decorators ───────▶ Closures ────────▶ Async
  │                         │                       │
  ▼                         ▼                       ▼
Context Managers ──▶ Metaclasses ────▶ Functional
                                                    │
                                                    ▼
                                              Performance
```

---

## Prerequisites

- Python basic syntax (variables, data types, control flow, functions)
- Object-oriented programming basics (classes, inheritance, methods)
- Module and package usage

---

## File List

| File | Difficulty | Key Content |
|------|-----------|-------------|
| [01_Type_Hints.md](./01_Type_Hints.md) | ⭐⭐ | Type Hints, typing module, mypy |
| [02_Decorators.md](./02_Decorators.md) | ⭐⭐ | Function/class decorators, @wraps |
| [03_Context_Managers.md](./03_Context_Managers.md) | ⭐⭐ | with statement, contextlib |
| [04_Iterators_and_Generators.md](./04_Iterators_and_Generators.md) | ⭐⭐⭐ | __iter__, yield, itertools |
| [05_Closures_and_Scope.md](./05_Closures_and_Scope.md) | ⭐⭐⭐ | LEGB, nonlocal, closure patterns |
| [06_Metaclasses.md](./06_Metaclasses.md) | ⭐⭐⭐ | type, __new__, __init_subclass__ |
| [07_Descriptors.md](./07_Descriptors.md) | ⭐⭐⭐⭐ | __get__, __set__, property implementation |
| [08_Async_Programming.md](./08_Async_Programming.md) | ⭐⭐⭐⭐ | async/await, asyncio |
| [09_Functional_Programming.md](./09_Functional_Programming.md) | ⭐⭐⭐⭐ | map, filter, functools |
| [10_Performance_Optimization.md](./10_Performance_Optimization.md) | ⭐⭐⭐⭐ | Profiling, optimization techniques |
| [11_Testing_and_Quality.md](./11_Testing_and_Quality.md) | ⭐⭐⭐ | pytest, fixtures, mocking, coverage |
| [12_Packaging_and_Distribution.md](./12_Packaging_and_Distribution.md) | ⭐⭐⭐ | pyproject.toml, Poetry, PyPI |
| [13_Dataclasses.md](./13_Dataclasses.md) | ⭐⭐ | @dataclass, field(), frozen |
| [14_Pattern_Matching.md](./14_Pattern_Matching.md) | ⭐⭐⭐ | match/case, structural patterns, guards |
| [15_Python_Basics.md](./15_Python_Basics.md) | ⭐ | Variables, data types, control flow, functions, data structures (prerequisite review) |
| [16_OOP_Basics.md](./16_OOP_Basics.md) | ⭐⭐ | Classes, inheritance, encapsulation, polymorphism (prerequisite review) |

---

## Recommended Learning Order

### Basics Review (if needed)
0. Basic syntax → OOP basics (15 → 16)

### Intermediate (Basic Advanced Syntax)
1. Type Hints → Decorators → Context Managers

### Intermediate+ (Advanced Syntax)
2. Iterators/Generators → Closures → Metaclasses

### Advanced (Expert Level)
3. Descriptors → Async → Functional → Performance Optimization

### Practical (Development Tools)
4. Testing & Quality → Packaging & Distribution → Dataclasses → Pattern Matching

---

## Practice Environment

```bash
# Check Python version (3.10+ recommended)
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install type checker (optional)
pip install mypy
```

---

## Related Materials

- [C_Programming/](../C_Programming/00_Overview.md) - System programming basics
- [Linux/](../Linux/00_Overview.md) - Development in Linux environment
- [PostgreSQL/](../PostgreSQL/00_Overview.md) - Database integration
