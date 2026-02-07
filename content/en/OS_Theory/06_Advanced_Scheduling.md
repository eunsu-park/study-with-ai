# Advanced Scheduling

## Overview

In this lesson, we learn about advanced scheduling techniques used in real operating systems. We explore MLFQ (Multi-Level Feedback Queue), multiprocessor scheduling, and real-time scheduling.

---

## Table of Contents

1. [MLFQ (Multi-Level Feedback Queue)](#1-mlfq-multi-level-feedback-queue)
2. [Multiprocessor Scheduling](#2-multiprocessor-scheduling)
3. [Processor Affinity](#3-processor-affinity)
4. [Real-time Scheduling](#4-real-time-scheduling)
5. [Linux CFS](#5-linux-cfs)
6. [Practice Problems](#6-practice-problems)

---

## 1. MLFQ (Multi-Level Feedback Queue)

### Concept

```
MLFQ = Scheduling using multiple ready queues
     = Move between queues based on process behavior
     = Advantages of SJF + Adaptive scheduling

┌─────────────────────────────────────────────────────────┐
│                      MLFQ Structure                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  High Priority (Short Time Quantum)                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Queue 0 (TQ=8ms): P1 → P5 → ...                │ ←── New Process
│  └─────────────────────────────────────────────────┘    │
│                          │                              │
│                          ▼ Demote on TQ expiration      │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Queue 1 (TQ=16ms): P2 → P7 → ...               │    │
│  └─────────────────────────────────────────────────┘    │
│                          │                              │
│                          ▼ Demote on TQ expiration      │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Queue 2 (TQ=32ms): P3 → P8 → ...               │    │
│  └─────────────────────────────────────────────────┘    │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Queue N (FCFS): P4 → P6 → ...                  │    │
│  └─────────────────────────────────────────────────┘    │
│  Low Priority (Long Time Quantum or FCFS)              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### MLFQ Rules

```
┌─────────────────────────────────────────────────────────┐
│                    MLFQ Basic Rules                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Rule 1: Execute higher priority process first         │
│         Priority(A) > Priority(B) → Execute A          │
│                                                         │
│  Rule 2: If same priority, use RR                      │
│         Priority(A) = Priority(B) → RR                 │
│                                                         │
│  Rule 3: Place new process in highest queue            │
│         New process → Queue 0                          │
│                                                         │
│  Rule 4: If time quantum is used up, demote            │
│         TQ exhausted → Priority decreased              │
│                                                         │
│  Rule 5: If CPU is yielded before TQ, maintain queue   │
│         I/O request etc → Maintain same priority       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### MLFQ Operation Example

```
Scenario: Mix of long job (CPU-bound) and short job (I/O-bound)

┌─────────────────────────────────────────────────────────┐
│  Time 0: Long job A arrives                            │
│                                                         │
│  Q0: [A] ───▶ Execute A (8ms)                          │
│  Q1: []                                                 │
│  Q2: []                                                 │
├─────────────────────────────────────────────────────────┤
│  Time 8: A exhausts TQ → Demoted                       │
│                                                         │
│  Q0: []                                                 │
│  Q1: [A] ───▶ Execute A (16ms)                         │
│  Q2: []                                                 │
├─────────────────────────────────────────────────────────┤
│  Time 10: Short job B arrives (5ms job)                │
│                                                         │
│  Q0: [B] ───▶ B has higher priority, preempt A         │
│  Q1: [A]     Start executing B                         │
│  Q2: []                                                 │
├─────────────────────────────────────────────────────────┤
│  Time 15: B completes (finished before TQ → I/O-heavy) │
│                                                         │
│  Q0: []                                                 │
│  Q1: [A] ───▶ Resume A                                 │
│  Q2: []                                                 │
└─────────────────────────────────────────────────────────┘

Result: Short job completes first (similar effect to SJF)
        Can prioritize short jobs without knowing burst time
```

### MLFQ Problems and Solutions

```
┌─────────────────────────────────────────────────────────┐
│                  MLFQ Problems and Solutions            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Problem 1: Starvation                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ If high priority jobs keep arriving,              │  │
│  │ low queue jobs never execute                      │  │
│  └───────────────────────────────────────────────────┘  │
│  Solution: Priority Boost                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Periodically move all processes to top queue     │  │
│  │ E.g., every S time units, all jobs to Q0         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Problem 2: Gaming the Scheduler                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Malicious process requests I/O just before TQ     │  │
│  │ → Prevents demotion, monopolizes CPU             │  │
│  └───────────────────────────────────────────────────┘  │
│  Solution: Track cumulative time                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Track total time used at each level              │  │
│  │ Demote when allotment exhausted (even if split)  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### MLFQ Parameters

```
┌─────────────────────────────────────────────────────────┐
│                   MLFQ Key Parameters                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Number of queues                                   │
│     • Typically 3~5                                     │
│     • Too many: overhead, too few: insufficient        │
│                                                         │
│  2. Time Quantum per queue                             │
│     • Usually doubles (8, 16, 32, 64ms...)            │
│     • High queue: Short TQ (fast response)             │
│     • Low queue: Long TQ (reduce context switches)     │
│                                                         │
│  3. Priority Boost period (S)                          │
│     • Too short: CPU-bound jobs advantaged             │
│     • Too long: starvation occurs                      │
│     • Typically 1 second ~ 100ms                       │
│                                                         │
│  Solaris Time-sharing class example:                   │
│  ┌─────────┬──────┬──────────┬─────────────┐           │
│  │ Priority│  TQ  │  Demote  │   Promote   │           │
│  ├─────────┼──────┼──────────┼─────────────┤           │
│  │   59    │ 20ms │   54     │    -        │           │
│  │   40    │ 40ms │   35     │    45       │           │
│  │   20    │ 80ms │   15     │    25       │           │
│  │   0     │ 200ms│   0      │    5        │           │
│  └─────────┴──────┴──────────┴─────────────┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Multiprocessor Scheduling

### Multiprocessor System Types

```
┌─────────────────────────────────────────────────────────┐
│               Multiprocessor System Types               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. SMP (Symmetric Multi-Processing)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                 │  │
│  │  │CPU 0│ │CPU 1│ │CPU 2│ │CPU 3│                 │  │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                 │  │
│  │     └───────┴───────┴───────┘                     │  │
│  │                 │                                 │  │
│  │          ┌──────┴──────┐                         │  │
│  │          │Shared Memory│                         │  │
│  │          └─────────────┘                         │  │
│  │  All processors equal, share memory              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. NUMA (Non-Uniform Memory Access)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ┌─────────────┐     ┌─────────────┐             │  │
│  │  │ CPU 0, 1    │     │ CPU 2, 3    │             │  │
│  │  │Local Memory │◀───▶│Local Memory │             │  │
│  │  └─────────────┘     └─────────────┘             │  │
│  │                                                   │  │
│  │  Local memory access: Fast                       │  │
│  │  Remote memory access: Slow                      │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Scheduling Approaches

```
┌─────────────────────────────────────────────────────────┐
│             Multiprocessor Scheduling Approaches        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Asymmetric Multiprocessing                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ┌─────────────────┐                             │  │
│  │  │ Master Processor│ ← All scheduling decisions  │  │
│  │  │   (CPU 0)       │                             │  │
│  │  └────────┬────────┘                             │  │
│  │           │ Assign tasks                         │  │
│  │     ┌─────┼─────┬─────┐                          │  │
│  │     ▼     ▼     ▼     ▼                          │  │
│  │  ┌─────┐┌─────┐┌─────┐┌─────┐                   │  │
│  │  │CPU 1││CPU 2││CPU 3││CPU 4│ ← Slaves          │  │
│  │  └─────┘└─────┘└─────┘└─────┘                   │  │
│  │                                                   │  │
│  │  Pros: Simple, no data sharing issues            │  │
│  │  Cons: Master bottleneck                         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. Symmetric Multiprocessing (SMP)                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                 │  │
│  │  │CPU 0│ │CPU 1│ │CPU 2│ │CPU 3│                 │  │
│  │  │Sched││Sched││Sched││Sched│                 │  │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                 │  │
│  │     └───────┴───┬───┴───────┘                     │  │
│  │                 ▼                                 │  │
│  │         ┌──────────────┐                         │  │
│  │         │ Shared Ready │ (sync required)         │  │
│  │         │    Queue     │                         │  │
│  │         └──────────────┘                         │  │
│  │  Or:    Each CPU maintains own Ready Queue       │  │
│  │                                                   │  │
│  │  Pros: Scalability, no bottleneck                │  │
│  │  Cons: Complex sync, possible load imbalance     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Load Balancing

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancing Techniques            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Problem: Load imbalance when each CPU has own queue   │
│                                                         │
│  CPU 0: ████████   CPU 1: ██   CPU 2: ████████████     │
│         (overload)     (idle)         (overload)        │
│                                                         │
│  Solution 1: Push Migration                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Periodically check each queue's load             │  │
│  │  Move processes from overloaded → idle queue      │  │
│  │                                                   │  │
│  │  CPU 0: ████████ ──push──▶ CPU 1: ██              │  │
│  │  Result:  █████               █████               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Solution 2: Pull Migration                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Idle CPU pulls tasks from other CPUs            │  │
│  │                                                   │  │
│  │  CPU 0: ████████ ◀──pull── CPU 1: (idle)         │  │
│  │  Result:  █████               ███                 │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Linux: Uses both                                      │
│  • Periodic Push (rebalance task)                      │
│  • Pull on idle (idle_balance)                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Processor Affinity

### What is Processor Affinity?

```
┌─────────────────────────────────────────────────────────┐
│              Processor Affinity                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Definition: Keeping process running on same CPU       │
│                                                         │
│  Reason: Cache efficiency                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Process P runs on CPU 0                         │  │
│  │  → CPU 0's cache loaded with P's data            │  │
│  │                                                   │  │
│  │  If P moves to different CPU:                    │  │
│  │  • CPU 0 cache: P data invalidated               │  │
│  │  • CPU 1 cache: P data needs reloading           │  │
│  │  → Cache misses increase, performance degrades   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Cache state visualization:                            │
│                                                         │
│  CPU 0         CPU 1         CPU 0         CPU 1       │
│  ┌─────┐      ┌─────┐       ┌─────┐       ┌─────┐      │
│  │Cache│      │Cache│       │Cache│       │Cache│      │
│  │[P's │      │[   ]│  Move │[   ]│       │[P's │      │
│  │data]│      │     │ ───▶  │     │       │data]│      │
│  └─────┘      └─────┘       └─────┘       └─────┘      │
│    ↑ Warm      Cold          Cold          ↑ Reload    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Affinity Types

```
┌─────────────────────────────────────────────────────────┐
│                   Affinity Types                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Soft Affinity                                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Default: Try to run on same CPU                │  │
│  │  • Can move if load balancing needed              │  │
│  │  • Linux default behavior                         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. Hard Affinity                                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Pin process to specific CPU set                │  │
│  │  • Won't move even for load balancing             │  │
│  │  • Set via system call                            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Linux configuration:                                   │
│  ```c                                                  │
│  #define _GNU_SOURCE                                   │
│  #include <sched.h>                                    │
│                                                         │
│  cpu_set_t mask;                                       │
│  CPU_ZERO(&mask);                                      │
│  CPU_SET(0, &mask);  // Run only on CPU 0             │
│  CPU_SET(2, &mask);  // Also can run on CPU 2         │
│                                                         │
│  // Set affinity for current process                  │
│  sched_setaffinity(0, sizeof(mask), &mask);            │
│  ```                                                   │
│                                                         │
│  Commands:                                              │
│  ```bash                                               │
│  # Run only on CPU 0,1                                 │
│  taskset -c 0,1 ./program                              │
│                                                         │
│  # Change affinity of running process                  │
│  taskset -c 2,3 -p PID                                 │
│  ```                                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Real-time Scheduling

### Real-time System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Real-time System Overview             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Real-time System = System with time constraints       │
│                     (Deadlines)                         │
│                                                         │
│  Classification:                                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Hard Real-Time                                    │  │
│  │ • Deadline miss = System failure                  │  │
│  │ • Ex: Aviation control, medical devices, ABS brake│ │
│  │ • Deadline guarantee essential                    │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Soft Real-Time                                    │  │
│  │ • Deadline miss = Performance degradation         │  │
│  │ • Ex: Video streaming, games, VoIP               │  │
│  │ • Goal: Meet most deadlines                      │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Real-time Task Characteristics

```
┌─────────────────────────────────────────────────────────┐
│                  Real-time Task Characteristics         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Periodic Task:                                         │
│                                                         │
│  Time ────────────────────────────────────────────▶     │
│  │←── Period P ──→│←── Period P ──→│                   │
│  ┌───┐            ┌───┐            ┌───┐               │
│  │ C │            │ C │            │ C │               │
│  └───┘            └───┘            └───┘               │
│  ↑   ↑            ↑   ↑                                │
│  Arrive Deadline(D) Arrive Deadline(D)                 │
│                                                         │
│  Parameters:                                            │
│  • P (Period): Task repetition period                  │
│  • C (Computation time): Execution time                │
│  • D (Deadline): Completion deadline                   │
│  • Utilization U = C/P                                 │
│                                                         │
│  Example: Video frame                                   │
│  • P = 33ms (30fps)                                    │
│  • C = 10ms (frame processing time)                    │
│  • D = 33ms                                            │
│  • U = 10/33 ≈ 0.3 (30%)                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Rate Monotonic Scheduling (RMS)

```
┌─────────────────────────────────────────────────────────┐
│           Rate Monotonic Scheduling (RMS)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Principle: Shorter period = Higher priority           │
│       (Rate = 1/Period, higher Rate = higher priority) │
│                                                         │
│  Characteristics:                                       │
│  • Static priority (fixed)                             │
│  • Preemptive                                          │
│  • Optimal (among static priority)                     │
│                                                         │
│  Example:                                               │
│  ┌─────────┬───────┬────────┬──────────┐               │
│  │  Task   │Period │Comp. C │ Priority │               │
│  ├─────────┼───────┼────────┼──────────┤               │
│  │   T1    │  50   │   20   │   High   │               │
│  │   T2    │  100  │   35   │   Low    │               │
│  └─────────┴───────┴────────┴──────────┘               │
│                                                         │
│  Gantt Chart:                                           │
│  ┌──────────┬──────────────┬──────────┬─────────────┐  │
│  │    T1    │      T2      │    T1    │  T2(cont.)  │  │
│  └──────────┴──────────────┴──────────┴─────────────┘  │
│  0         20            55        75             100  │
│       T2 starts      T1 arrives(preempt)  T2 resumes   │
│                                                         │
│  Schedulability condition:                              │
│  Total utilization ≤ n(2^(1/n) - 1)                    │
│  • n=1: 100%                                           │
│  • n=2: ~82.8%                                         │
│  • n→∞: ~69.3%                                         │
│                                                         │
│  Example: U = 20/50 + 35/100 = 0.4 + 0.35 = 0.75      │
│          0.75 < 0.828 → Schedulable                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Earliest Deadline First (EDF)

```
┌─────────────────────────────────────────────────────────┐
│            Earliest Deadline First (EDF)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Principle: Nearest deadline = Highest priority        │
│                                                         │
│  Characteristics:                                       │
│  • Dynamic priority (changes with deadline)            │
│  • Preemptive                                          │
│  • Theoretically optimal (up to 100% utilization)      │
│                                                         │
│  Example (same data as RMS):                           │
│  ┌─────────┬───────┬────────┬────────────────┐         │
│  │  Task   │Period │Comp. C │  Deadline D    │         │
│  ├─────────┼───────┼────────┼────────────────┤         │
│  │   T1    │  50   │   20   │ 50, 100, 150...│         │
│  │   T2    │  100  │   35   │ 100, 200...    │         │
│  └─────────┴───────┴────────┴────────────────┘         │
│                                                         │
│  Time 0:                                                │
│  • T1 deadline: 50, T2 deadline: 100                   │
│  • Execute T1 first                                    │
│                                                         │
│  Time 50 (T1 second instance):                         │
│  • T1 deadline: 100, T2 deadline: 100 (tied)          │
│  • Tie → Arbitrary selection or other criteria         │
│                                                         │
│  RMS vs EDF:                                           │
│  ┌──────────────┬────────────────┬─────────────────┐   │
│  │  Feature     │      RMS       │      EDF        │   │
│  ├──────────────┼────────────────┼─────────────────┤   │
│  │ Priority     │ Static         │ Dynamic         │   │
│  │ Max Util.    │ ~69% (n→∞)     │ 100%            │   │
│  │ Complexity   │ Low            │ High            │   │
│  │ Overhead     │ Low            │ High            │   │
│  │ Overload     │ Predictable    │ Domino effect   │   │
│  └──────────────┴────────────────┴─────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Linux CFS

### CFS Overview

```
┌─────────────────────────────────────────────────────────┐
│         CFS (Completely Fair Scheduler)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Linux default scheduler since 2.6.23 (2007~)          │
│                                                         │
│  Core idea:                                             │
│  • Fairly distribute CPU time to all processes         │
│  • Track fairness via virtual runtime                  │
│  • Efficiently manage with Red-Black tree              │
│                                                         │
│  Virtual runtime (vruntime):                            │
│  • Weighted time process has used CPU                  │
│  • High priority → increases slowly                    │
│  • Low priority → increases quickly                    │
│                                                         │
│  Scheduling decision:                                   │
│  • Always select process with smallest vruntime        │
│  • = Process that has run least                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### CFS Operation

```
┌─────────────────────────────────────────────────────────┐
│                    CFS Operation                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Red-Black tree (sorted by vruntime):                  │
│                                                         │
│                    ┌───────┐                            │
│                    │ P2:50 │                            │
│                    └───┬───┘                            │
│              ┌─────────┴─────────┐                      │
│          ┌───────┐           ┌───────┐                  │
│          │ P1:30 │           │ P3:80 │                  │
│          └───┬───┘           └───┬───┘                  │
│        ┌─────┴─────┐       ┌─────┴─────┐                │
│    ┌───────┐   ┌───────┐ ┌───────┐  ┌───────┐          │
│    │ P4:10 │   │ P5:40 │ │ P6:70 │  │ P7:100│          │
│    └───────┘   └───────┘ └───────┘  └───────┘          │
│    ↑                                                    │
│    Leftmost = Smallest vruntime = Next to execute      │
│                                                         │
│  Over time:                                             │
│  1. Execute P4, vruntime increases (10 → 25)           │
│  2. If P4 no longer minimum, switch to another process │
│  3. Restructure tree, select new minimum               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### CFS Timeslice Calculation

```
┌─────────────────────────────────────────────────────────┐
│                CFS Timeslice Calculation                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  CFS uses target latency instead of fixed timeslice    │
│                                                         │
│  Target Latency: Default 6ms (adjustable)              │
│                                                         │
│  Calculation:                                           │
│  Process timeslice = Target latency × (weight/total)   │
│                                                         │
│  Example (n=3, same priority):                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Target latency = 6ms                              │  │
│  │ Each process weight = 1024 (nice 0)              │  │
│  │ Total weight = 1024 × 3 = 3072                    │  │
│  │                                                   │  │
│  │ Each timeslice = 6ms × (1024/3072) = 2ms         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Example (different priorities):                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │ P1: nice -5 (weight 3121)                        │  │
│  │ P2: nice 0 (weight 1024)                         │  │
│  │ P3: nice 5 (weight 335)                          │  │
│  │ Total weight = 4480                              │  │
│  │                                                   │  │
│  │ P1 timeslice = 6ms × (3121/4480) ≈ 4.2ms         │  │
│  │ P2 timeslice = 6ms × (1024/4480) ≈ 1.4ms         │  │
│  │ P3 timeslice = 6ms × (335/4480) ≈ 0.4ms          │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Nice value and weight:                                 │
│  • nice -20 (highest): weight 88761                    │
│  • nice 0 (default): weight 1024                       │
│  • nice 19 (lowest): weight 15                         │
│  • ~10% difference per nice 1 difference               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Practice Problems

### Problem 1: MLFQ

In an MLFQ system with 3 queues:
- Q0: TQ=4ms
- Q1: TQ=8ms
- Q2: FCFS

Explain the execution for the first 20ms when process A (CPU-bound, 30ms job) and process B (I/O-bound, 3ms CPU + I/O repeat) arrive simultaneously.

<details>
<summary>View Answer</summary>

**Initial state:**
- Q0: [A, B]
- Q1: []
- Q2: []

**Time 0-3ms:** B executes (3ms), I/O request → B stays in Q0
**Time 3-7ms:** A executes (4ms), TQ exhausted → A demoted to Q1
**Time 7-10ms:** B completes I/O and arrives, B executes (3ms)
**Time 10-18ms:** A executes (8ms), TQ exhausted → A demoted to Q2
**Time 18-20ms:** If B returned, B executes...

**Result:**
- B, being a short job, gets quick service in high queue
- A gradually moves to lower queues using long timeslice

</details>

### Problem 2: RMS Schedulability

Check if the following task set is schedulable with RMS.

| Task | Period (P) | Execution Time (C) |
|------|-----------|-------------------|
| T1 | 100 | 20 |
| T2 | 150 | 30 |
| T3 | 350 | 80 |

<details>
<summary>View Answer</summary>

**Total Utilization Calculation:**
- U1 = 20/100 = 0.20
- U2 = 30/150 = 0.20
- U3 = 80/350 = 0.23

**Total U = 0.20 + 0.20 + 0.23 = 0.63**

**RMS Schedulability Condition (n=3):**
- Bound = 3 × (2^(1/3) - 1) = 3 × 0.26 = 0.78

**Conclusion:**
- 0.63 < 0.78
- **Schedulable**

</details>

### Problem 3: Processor Affinity

Explain the difference between soft affinity and hard affinity, and provide examples of situations suitable for each.

<details>
<summary>View Answer</summary>

**Soft Affinity:**
- Definition: OS tries to run process on same CPU, but can move if load balancing needed
- Suitable situations:
  - General applications
  - When balance between cache efficiency and load balancing needed
  - Ex: Web servers, databases

**Hard Affinity:**
- Definition: Force process to run only on specific CPU set
- Suitable situations:
  - Real-time systems (need predictable performance)
  - NUMA systems optimizing local memory access
  - Dedicated CPU core allocation
  - Ex: Game server physics engine thread, high-frequency trading systems

</details>

### Problem 4: EDF vs RMS

Compare RMS and EDF for the following task set.

| Task | Period | Execution Time |
|------|-------|---------------|
| T1 | 4 | 1 |
| T2 | 5 | 2 |
| T3 | 10 | 3 |

<details>
<summary>View Answer</summary>

**Utilization:**
- U = 1/4 + 2/5 + 3/10 = 0.25 + 0.4 + 0.3 = 0.95

**RMS Analysis:**
- Bound = 3 × (2^(1/3) - 1) ≈ 0.78
- 0.95 > 0.78 → Not guaranteed by RMS
- (May actually be schedulable but not guaranteed)

**EDF Analysis:**
- Bound = 1.0 (100%)
- 0.95 < 1.0 → Schedulable with EDF!

**Conclusion:**
- This task set is not guaranteed schedulable by RMS
- But schedulable with EDF

</details>

### Problem 5: Linux CFS

When process A has nice value 0 and process B has nice value 5, calculate how much more CPU time A receives compared to B.

<details>
<summary>View Answer</summary>

**Weights (approximate):**
- nice 0: weight 1024
- nice 5: weight ~335 (~1.25x per nice 1 difference)

**CPU Time Ratio:**
- A's ratio = 1024 / (1024 + 335) = 1024 / 1359 ≈ 0.753 (75.3%)
- B's ratio = 335 / 1359 ≈ 0.247 (24.7%)

**A receives about 3 times more CPU time than B**

(Exact ratio: 1024/335 ≈ 3.06x)

</details>

---

## Next Steps

- [07_Synchronization_Basics.md](./07_Synchronization_Basics.md) - Race Conditions and Critical Sections

---

## References

- [OSTEP - MLFQ](https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched-mlfq.pdf)
- [Linux CFS Documentation](https://www.kernel.org/doc/html/latest/scheduler/sched-design-CFS.html)
- [Real-Time Systems (Jane Liu)](https://www.pearson.com/us/higher-education/program/Liu-Real-Time-Systems/PGM293020.html)
