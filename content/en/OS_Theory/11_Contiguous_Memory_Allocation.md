# Contiguous Memory Allocation ⭐⭐

## Overview

Contiguous memory allocation is a memory management technique where each process occupies a single contiguous region in memory. We'll learn about fixed and variable partitioning, and efficient memory placement strategies.

---

## Table of Contents

1. [Memory Partitioning Overview](#1-memory-partitioning-overview)
2. [Fixed Partitioning](#2-fixed-partitioning)
3. [Variable Partitioning](#3-variable-partitioning)
4. [Memory Placement Strategies](#4-memory-placement-strategies)
5. [Fragmentation](#5-fragmentation)
6. [Compaction](#6-compaction)
7. [Practice Problems](#7-practice-problems)

---

## 1. Memory Partitioning Overview

### Memory Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Overall Memory Structure                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Low Address                                                │
│   ┌──────────────────────────────────────────┐              │
│   │      Operating System (Kernel)            │  0x0000      │
│   │ Interrupt vectors, drivers, kernel code   │              │
│   ├──────────────────────────────────────────┤              │
│   │                                          │              │
│   │                                          │              │
│   │          User Area                       │              │
│   │      (Used by processes)                 │              │
│   │                                          │              │
│   │                                          │              │
│   └──────────────────────────────────────────┘  0xFFFF      │
│   High Address                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Fixed Partitioning

### 2.1 Concept

Divide memory into fixed-size partitions in advance.

```
┌─────────────────────────────────────────────────────────────┐
│                  Fixed Partitioning (Equal Size)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │                  OS                       │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 1                  │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 2                  │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 3                  │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 4                  │ 64KB         │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  Feature: All partitions same size (64KB)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Fixed Partitioning Problems

```
┌─────────────────────────────────────────────────────────────┐
│                 Internal Fragmentation Occurs                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Allocating 45KB process to 64KB partition:                 │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │          Process (45KB)                   │ Used         │
│  ├──────────────────────────────────────────┤               │
│  │          Waste (19KB)                    │ Internal Frag │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  → 19KB wasted (cannot be used by other processes)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Variable Partitioning

### 3.1 Concept

Dynamically create partitions to match process sizes.

```
┌─────────────────────────────────────────────────────────────┐
│                    Variable Partitioning Example             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Initial state:                                              │
│  ┌──────────────────────────────────────────┐ 0             │
│  │                  OS                       │ 64KB         │
│  ├──────────────────────────────────────────┤ 64KB         │
│  │                                          │               │
│  │             Free Space                    │               │
│  │              (448KB)                     │               │
│  │                                          │               │
│  └──────────────────────────────────────────┘ 512KB        │
│                                                              │
│  After P1(100KB), P2(50KB), P3(200KB) loaded:               │
│  ┌──────────────────────────────────────────┐ 0             │
│  │                  OS                       │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │             P1 (100KB)                   │               │
│  ├──────────────────────────────────────────┤ 164KB        │
│  │          P2 (50KB)                       │               │
│  ├──────────────────────────────────────────┤ 214KB        │
│  │                                          │               │
│  │             P3 (200KB)                   │               │
│  │                                          │               │
│  ├──────────────────────────────────────────┤ 414KB        │
│  │         Free Space (98KB)                │               │
│  └──────────────────────────────────────────┘ 512KB        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Memory Placement Strategies

Strategies for deciding which hole to place a new process in.

### 4.1 First-Fit

```
┌─────────────────────────────────────────────────────────────┐
│                    First-Fit Strategy                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Free space list:                                            │
│  [100KB] -> [500KB] -> [200KB] -> [300KB]                   │
│                                                              │
│  150KB process allocation request:                           │
│                                                              │
│  1. Check [100KB] → 150KB > 100KB → Cannot                  │
│  2. Check [500KB] → 150KB <= 500KB → Allocate here!         │
│                                                              │
│  Result:                                                     │
│  [100KB] -> [350KB] -> [200KB] -> [300KB]                   │
│              ↑ (500-150=350KB remaining)                     │
│                                                              │
│  Advantage: Fast search                                      │
│  Disadvantage: Small holes accumulate at front               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Best-Fit

```
┌─────────────────────────────────────────────────────────────┐
│                    Best-Fit Strategy                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Free space list:                                            │
│  [100KB] -> [500KB] -> [200KB] -> [300KB]                   │
│                                                              │
│  150KB process allocation request:                           │
│                                                              │
│  Full search:                                                │
│  - [100KB]: Cannot (100 < 150)                               │
│  - [500KB]: Possible, remaining = 350KB                      │
│  - [200KB]: Possible, remaining = 50KB  ← Minimum waste!    │
│  - [300KB]: Possible, remaining = 150KB                      │
│                                                              │
│  Allocate to 200KB block!                                    │
│                                                              │
│  Result:                                                     │
│  [100KB] -> [500KB] -> [50KB] -> [300KB]                    │
│                                                              │
│  Advantage: Minimize memory waste                            │
│  Disadvantage: Full search needed, creates tiny holes        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Worst-Fit

```
┌─────────────────────────────────────────────────────────────┐
│                   Worst-Fit Strategy                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Free space list:                                            │
│  [100KB] -> [500KB] -> [200KB] -> [300KB]                   │
│                                                              │
│  150KB process allocation request:                           │
│                                                              │
│  Find largest block:                                         │
│  - [100KB], [500KB], [200KB], [300KB]                       │
│  - Maximum: 500KB ← Allocate here!                          │
│                                                              │
│  Result:                                                     │
│  [100KB] -> [350KB] -> [200KB] -> [300KB]                   │
│              ↑ (Large remaining space = usable later)       │
│                                                              │
│  Advantage: Large remaining space can fit other processes    │
│  Disadvantage: Difficulty placing large processes            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Fragmentation

### 5.1 Internal Fragmentation

```
┌─────────────────────────────────────────────────────────────┐
│                     Internal Fragmentation                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Occurs in fixed partitioning or paging:                     │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │                                          │               │
│  │   Allocated block (e.g., 4KB page)       │               │
│  │                                          │               │
│  │  ┌────────────────────────┐              │               │
│  │  │ Process data (3KB)     │              │               │
│  │  ├────────────────────────┤              │               │
│  │  │ Internal frag (1KB)    │ ← Waste!     │               │
│  │  └────────────────────────┘              │               │
│  │                                          │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  Features:                                                   │
│  - Waste inside allocated block                             │
│  - Cannot be used by other processes                         │
│  - Occurs in fixed partitioning, paging                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 External Fragmentation

```
┌─────────────────────────────────────────────────────────────┐
│                     External Fragmentation                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Occurs in variable partitioning:                            │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │  OS                                      │               │
│  ├──────────────────────────────────────────┤               │
│  │  P1 (100KB)                              │               │
│  ├──────────────────────────────────────────┤               │
│  │  *** Hole (30KB) ***                     │ ← Small hole  │
│  ├──────────────────────────────────────────┤               │
│  │  P2 (200KB)                              │               │
│  ├──────────────────────────────────────────┤               │
│  │  *** Hole (25KB) ***                     │ ← Small hole  │
│  ├──────────────────────────────────────────┤               │
│  │  P3 (150KB)                              │               │
│  ├──────────────────────────────────────────┤               │
│  │  *** Hole (45KB) ***                     │ ← Small hole  │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  Total free space: 30 + 25 + 45 = 100KB                     │
│  But cannot load 50KB process! (No contiguous space)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Compaction

### 6.1 Concept

Move all processes to one end to create large contiguous free space, solving external fragmentation.

```
┌─────────────────────────────────────────────────────────────┐
│                      Compaction Process                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Before compaction:                After compaction:         │
│                                                              │
│  ┌──────────────┐                 ┌──────────────┐          │
│  │  OS          │                 │  OS          │          │
│  ├──────────────┤                 ├──────────────┤          │
│  │  P1 (100KB)  │                 │  P1 (100KB)  │          │
│  ├──────────────┤                 ├──────────────┤          │
│  │  Hole (30KB) │    ────▶        │  P2 (200KB)  │          │
│  ├──────────────┤                 │              │          │
│  │  P2 (200KB)  │                 ├──────────────┤          │
│  │              │                 │  P3 (150KB)  │          │
│  ├──────────────┤                 │              │          │
│  │  Hole (25KB) │                 ├──────────────┤          │
│  ├──────────────┤                 │              │          │
│  │  P3 (150KB)  │                 │  Hole (100KB)│          │
│  │              │                 │  (contiguous)│          │
│  ├──────────────┤                 │              │          │
│  │  Hole (45KB) │                 │              │          │
│  └──────────────┘                 └──────────────┘          │
│                                                              │
│  Total hole: 100KB                Total hole: 100KB (contig)│
│  (Scattered)                      New process can load!     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Problem 1: Placement Strategy
Given free blocks: [200KB, 80KB, 300KB, 150KB]

For a 120KB process, which block is selected by each strategy?
1. First-Fit
2. Best-Fit
3. Worst-Fit

<details>
<summary>Show Answer</summary>

1. First-Fit: 200KB (first block that fits)
2. Best-Fit: 150KB (smallest waste: 150-120=30KB)
3. Worst-Fit: 300KB (largest block)

</details>

---

## Next Steps

Next topic: Paging

---

## References

- Silberschatz, "Operating System Concepts" Chapter 8
- Tanenbaum, "Modern Operating Systems" Chapter 3
