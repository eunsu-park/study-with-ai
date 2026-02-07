# Deadlock

## Overview

Deadlock is a state where two or more processes wait indefinitely for each other to release resources. In this lesson, we'll learn about the four necessary conditions for deadlock, resource allocation graphs, and methods for prevention, avoidance, detection, and recovery.

---

## Table of Contents

1. [What is Deadlock?](#1-what-is-deadlock)
2. [Deadlock Necessary Conditions](#2-deadlock-necessary-conditions)
3. [Resource Allocation Graph](#3-resource-allocation-graph)
4. [Deadlock Prevention](#4-deadlock-prevention)
5. [Deadlock Avoidance](#5-deadlock-avoidance)
6. [Deadlock Detection and Recovery](#6-deadlock-detection-and-recovery)
7. [Practice Problems](#7-practice-problems)

---

## 1. What is Deadlock?

### Definition

```
┌─────────────────────────────────────────────────────────┐
│                    Deadlock                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Deadlock = Circular waiting state                      │
│           = Two or more processes wait indefinitely for │
│             each other's resources                      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                                                 │    │
│  │    Process A                 Process B         │    │
│  │    ┌───────┐                 ┌───────┐         │    │
│  │    │       │ ──R2 request──▶│       │         │    │
│  │    │  R1   │                │  R2   │         │    │
│  │    │ holds │ ◀──R1 request──│ holds │         │    │
│  │    │       │                │       │         │    │
│  │    └───────┘                └───────┘         │    │
│  │                                                 │    │
│  │    A waits for R2, B waits for R1               │    │
│  │    → Both wait forever (deadlock)               │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Real-World Examples

```
┌─────────────────────────────────────────────────────────┐
│                   Real-World Deadlock Examples           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Intersection deadlock:                              │
│                                                         │
│              │     │                                    │
│              │ ▲   │                                    │
│              │ │   │                                    │
│       ───────┼─────┼───────                             │
│          ◀───│     │                                    │
│       ───────┼─────┼───────                             │
│              │     │───▶                                │
│       ───────┼─────┼───────                             │
│              │   │ │                                    │
│              │   ▼ │                                    │
│                                                         │
│     Four cars all waiting for the car in front          │
│                                                         │
│  2. Bridge deadlock:                                    │
│                                                         │
│     ┌───────────────────────────┐                       │
│     │     ◀───   ───▶          │                       │
│     │  Car A    Car B           │                       │
│     └───────────────────────────┘                       │
│     Both entered from opposite ends of narrow bridge    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Code Example

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

void* thread_A(void* arg) {
    pthread_mutex_lock(&lock1);
    printf("Thread A: acquired lock1\n");

    sleep(1);  // Increase deadlock probability

    printf("Thread A: waiting for lock2\n");
    pthread_mutex_lock(&lock2);  // Deadlock!

    printf("Thread A: acquired both\n");
    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);
    return NULL;
}

void* thread_B(void* arg) {
    pthread_mutex_lock(&lock2);
    printf("Thread B: acquired lock2\n");

    sleep(1);  // Increase deadlock probability

    printf("Thread B: waiting for lock1\n");
    pthread_mutex_lock(&lock1);  // Deadlock!

    printf("Thread B: acquired both\n");
    pthread_mutex_unlock(&lock1);
    pthread_mutex_unlock(&lock2);
    return NULL;
}

int main() {
    pthread_t tA, tB;

    pthread_create(&tA, NULL, thread_A, NULL);
    pthread_create(&tB, NULL, thread_B, NULL);

    pthread_join(tA, NULL);
    pthread_join(tB, NULL);

    printf("Done\n");  // Never reached if deadlocked
    return 0;
}

/*
Output (when deadlocked):
Thread A: acquired lock1
Thread B: acquired lock2
Thread A: waiting for lock2
Thread B: waiting for lock1
... (wait forever)
*/
```

---

## 2. Deadlock Necessary Conditions

### Four Necessary Conditions

```
┌─────────────────────────────────────────────────────────┐
│              Four Necessary Conditions for Deadlock      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  All four conditions must hold simultaneously for       │
│  deadlock to occur (break one → no deadlock)            │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Mutual Exclusion                               │  │
│  │    - Resource can be used by only one process     │  │
│  │      at a time                                    │  │
│  │    - Other processes wait until resource is released│ │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 2. Hold and Wait                                  │  │
│  │    - Process holds resources while waiting for    │  │
│  │      additional resources                         │  │
│  │    - Requests more while keeping what it has      │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 3. No Preemption                                  │  │
│  │    - Resources cannot be forcibly taken away      │  │
│  │    - Process must voluntarily release them        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 4. Circular Wait                                  │  │
│  │    - In process set {P0, P1, ..., Pn}            │  │
│  │    - Circular chain exists: P0→P1→P2→...→Pn→P0   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Circular Wait Visualization

```
┌─────────────────────────────────────────────────────────┐
│                    Circular Wait Example                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Three processes, three resources:                      │
│                                                         │
│       ┌───────┐                                         │
│       │  P0   │                                         │
│       └───┬───┘                                         │
│      holds: R0                                          │
│      waits: R1 ───────────┐                             │
│           │                │                             │
│           ▼                ▼                             │
│       ┌───────┐       ┌───────┐                         │
│       │  R1   │       │  P1   │                         │
│       └───────┘       └───┬───┘                         │
│                      holds: R1                          │
│                      waits: R2 ─────┐                   │
│                           │         │                   │
│                           ▼         ▼                   │
│                       ┌───────┐┌───────┐                │
│                       │  R2   ││  P2   │                │
│                       └───────┘└───┬───┘                │
│                              holds: R2                  │
│                              waits: R0 ────▶ R0         │
│                                    │       (P0 holds)   │
│                                    └───────────────────┐│
│                                                        ││
│  P0 → R1 → P1 → R2 → P2 → R0 → P0 (circular!)         ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Resource Allocation Graph

### Graph Notation

```
┌─────────────────────────────────────────────────────────┐
│               Resource Allocation Graph (RAG)            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Nodes:                                                 │
│  • Process: ○ (circle)                                  │
│  • Resource type: □ (rectangle)                         │
│    - Internal dots (●): number of resource instances    │
│                                                         │
│  Edges:                                                 │
│  • Request edge: Pi → Rj (Pi requests Rj)               │
│  • Assignment edge: Rj → Pi (Rj is assigned to Pi)      │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │        Example                                    │  │
│  │                                                   │  │
│  │    ○ P1 ────request───▶ □ R1                     │  │
│  │                         │ ● ●                     │  │
│  │                         │                         │  │
│  │                         ▼ assignment              │  │
│  │                         ○ P2                      │  │
│  │                                                   │  │
│  │    P1 requests R1                                │  │
│  │    One instance of R1 is assigned to P2          │  │
│  │    R1 has 2 instances                            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Graph with Deadlock

```
┌─────────────────────────────────────────────────────────┐
│              Resource Allocation Graph with Deadlock     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│     ┌─────────────────────────────────────────────┐     │
│     │                                             │     │
│     │         ○ P1                                │     │
│     │         │↖                                 │     │
│     │  request│  ╲ assignment                    │     │
│     │         ▼   ╲                              │     │
│     │      ┌─────┐ ╲                             │     │
│     │      │ R1  │  ╲                            │     │
│     │      │  ●  │   ╲                           │     │
│     │      └──┬──┘    ╲                          │     │
│     │         │        ╲                         │     │
│     │  assignment│      ╲                        │     │
│     │         ▼          ╲                       │     │
│     │         ○ P2 ───request───▶ ┌─────┐       │     │
│     │         ↖                   │ R2  │       │     │
│     │          ╲ assignment       │  ●  │       │     │
│     │           ╲                 └──┬──┘       │     │
│     │            ╲                   │assignment│     │
│     │             ╲                  ▼          │     │
│     │              ╲──────────○ P3             │     │
│     │                           │              │     │
│     │                           │ request      │     │
│     │                           ▼              │     │
│     │               ┌─────┐                    │     │
│     │               │ R3  │ ◀───────┐         │     │
│     │               │  ●  │         │assignment│     │
│     │               └─────┘         │         │     │
│     │                    ○ P1 ◀─────┘         │     │
│     │                                             │     │
│     └─────────────────────────────────────────────┘     │
│                                                         │
│     Cycle: P1 → R1 → P2 → R3 → P3 → R2 → P1           │
│            (or P2 → R3 → P3 → R2 → P2)                 │
│                                                         │
│     → Deadlock exists!                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Cycles and Deadlock

```
┌─────────────────────────────────────────────────────────┐
│                Relationship Between Cycles and Deadlock  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Rules:                                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • No cycle → No deadlock                          │  │
│  │ • Cycle exists:                                   │  │
│  │   - Single instance per resource → Deadlock       │  │
│  │   - Multiple instances per resource → Possible    │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Cycle exists but no deadlock case:                     │
│                                                         │
│     ○ P1 ─request─▶ ┌─────┐ ◀─request─ ○ P3          │
│     ↑              │ R1  │             ↑              │
│     │              │ ●●  │             │              │
│     │              └──┬──┘             │              │
│     │                 │                │              │
│     │ assign   assign ▼  assign        │ assign       │
│     │                 │                │              │
│     └────── ○ P2  ○ P4 ──────────────┘              │
│                                                         │
│     R1 has 2 instances → If P2 & P4 release, P1 & P3   │
│     can proceed → Not deadlock                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Deadlock Prevention

### Breaking One of Four Conditions

```
┌─────────────────────────────────────────────────────────┐
│                    Deadlock Prevention                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Break Mutual Exclusion                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Use sharable resources when possible           │  │
│  │  • Example: read-only files                       │  │
│  │  • Limitation: inherently exclusive resources     │  │
│  │    (printers, mutexes) → generally hard to break  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. Break Hold and Wait                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Method 1: Request all resources at start         │  │
│  │    • Request all needed resources at once         │  │
│  │    • Downside: low resource utilization, starvation│ │
│  │                                                   │  │
│  │  Method 2: Release all before requesting new     │  │
│  │    • Return held resources before new request    │  │
│  │    • Downside: difficult to implement            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. Break No Preemption                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Force release of held resources if request fails│ │
│  │  • Or preempt resources from other processes     │  │
│  │  • Applicable: CPU registers, memory              │  │
│  │  • Difficult: mutexes, printers                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  4. Break Circular Wait ★ Most practical               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Assign ordering number to resources           │  │
│  │  • Request resources only in ascending order      │  │
│  │  • Circular wait structurally impossible          │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Circular Wait Prevention Code

```c
#include <stdio.h>
#include <pthread.h>

// Assign ordering numbers to resources
// lock1 = resource 1 (order 1)
// lock2 = resource 2 (order 2)
pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

// Always acquire in ascending order!
void* thread_A(void* arg) {
    // Acquire in order: lock1(1) → lock2(2)
    pthread_mutex_lock(&lock1);  // Order 1
    printf("Thread A: acquired lock1\n");

    pthread_mutex_lock(&lock2);  // Order 2
    printf("Thread A: acquired lock2\n");

    // Critical section
    printf("Thread A: performing work\n");

    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);
    return NULL;
}

void* thread_B(void* arg) {
    // Same order: lock1(1) → lock2(2)
    pthread_mutex_lock(&lock1);  // Order 1
    printf("Thread B: acquired lock1\n");

    pthread_mutex_lock(&lock2);  // Order 2
    printf("Thread B: acquired lock2\n");

    // Critical section
    printf("Thread B: performing work\n");

    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);
    return NULL;
}

int main() {
    pthread_t tA, tB;

    pthread_create(&tA, NULL, thread_A, NULL);
    pthread_create(&tB, NULL, thread_B, NULL);

    pthread_join(tA, NULL);
    pthread_join(tB, NULL);

    printf("Done (no deadlock!)\n");
    return 0;
}
```

---

## 5. Deadlock Avoidance

### Safe State vs Unsafe State

```
┌─────────────────────────────────────────────────────────┐
│               Safe State vs Unsafe State                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Safe State:                                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • All processes can complete without deadlock    │  │
│  │  • Safe sequence exists                           │  │
│  │                                                   │  │
│  │  Safe sequence: <P1, P3, P4, P2, P0>             │  │
│  │  = All can complete if allocated in this order   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Unsafe State:                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • No safe sequence exists                        │  │
│  │  • Deadlock possible (not necessarily occurring)  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │                                                    │ │
│  │  ┌─────────────────────────────────────────────┐  │ │
│  │  │                 Safe State                  │  │ │
│  │  │     (Deadlock impossible)                   │  │ │
│  │  │                                             │  │ │
│  │  └─────────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  │  ┌─────────────────────────────────────────────┐  │ │
│  │  │            Unsafe State                     │  │ │
│  │  │   ┌─────────────────────────────────────┐  │  │ │
│  │  │   │                                     │  │  │ │
│  │  │   │         Deadlock State              │  │  │ │
│  │  │   │                                     │  │  │ │
│  │  │   └─────────────────────────────────────┘  │  │ │
│  │  └─────────────────────────────────────────────┘  │ │
│  │                                                    │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  Avoidance strategy: Maintain safe state only           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Banker's Algorithm

```
┌─────────────────────────────────────────────────────────┐
│                 Banker's Algorithm                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Developed by Dijkstra (1965)                           │
│  Similar to how a banker lends to customers             │
│                                                         │
│  Data structures:                                       │
│  • n = number of processes                              │
│  • m = number of resource types                         │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Available[m]                                      │  │
│  │   Number of available instances of each resource  │  │
│  │                                                   │  │
│  │ Max[n][m]                                         │  │
│  │   Maximum resources each process can request     │  │
│  │                                                   │  │
│  │ Allocation[n][m]                                  │  │
│  │   Resources currently allocated to each process  │  │
│  │                                                   │  │
│  │ Need[n][m] = Max[n][m] - Allocation[n][m]        │  │
│  │   Additional resources needed by each process    │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Banker's Algorithm Example

```
┌─────────────────────────────────────────────────────────┐
│              Banker's Algorithm Example                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Resource types: A, B, C (total: 10, 5, 7)              │
│  Processes: P0, P1, P2, P3, P4                          │
│                                                         │
│  Current state:                                         │
│  ┌─────────┬────────────────┬─────────────┬────────────┐│
│  │ Process │   Allocation   │     Max     │    Need    ││
│  │         │   A   B   C    │  A   B   C  │  A   B   C ││
│  ├─────────┼────────────────┼─────────────┼────────────┤│
│  │   P0    │   0   1   0    │  7   5   3  │  7   4   3 ││
│  │   P1    │   2   0   0    │  3   2   2  │  1   2   2 ││
│  │   P2    │   3   0   2    │  9   0   2  │  6   0   0 ││
│  │   P3    │   2   1   1    │  2   2   2  │  0   1   1 ││
│  │   P4    │   0   0   2    │  4   3   3  │  4   3   1 ││
│  └─────────┴────────────────┴─────────────┴────────────┘│
│                                                         │
│  Available = [3, 3, 2]  (A:3, B:3, C:2 available)       │
│                                                         │
│  Safety check:                                          │
│  1. Need[P1] = [1,2,2] ≤ Available[3,3,2] → P1 can run │
│     After P1: Available = [3,3,2] + [2,0,0] = [5,3,2]   │
│                                                         │
│  2. Need[P3] = [0,1,1] ≤ Available[5,3,2] → P3 can run │
│     After P3: Available = [5,3,2] + [2,1,1] = [7,4,3]   │
│                                                         │
│  3. Need[P4] = [4,3,1] ≤ Available[7,4,3] → P4 can run │
│     After P4: Available = [7,4,3] + [0,0,2] = [7,4,5]   │
│                                                         │
│  4. Need[P0] = [7,4,3] ≤ Available[7,4,5] → P0 can run │
│     After P0: Available = [7,4,5] + [0,1,0] = [7,5,5]   │
│                                                         │
│  5. Need[P2] = [6,0,0] ≤ Available[7,5,5] → P2 can run │
│                                                         │
│  Safe sequence: <P1, P3, P4, P0, P2>                    │
│  → System is in safe state                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Banker's Algorithm Code

```c
#include <stdio.h>
#include <stdbool.h>

#define P 5  // Number of processes
#define R 3  // Number of resource types

int available[R] = {3, 3, 2};
int maximum[P][R] = {
    {7, 5, 3},
    {3, 2, 2},
    {9, 0, 2},
    {2, 2, 2},
    {4, 3, 3}
};
int allocation[P][R] = {
    {0, 1, 0},
    {2, 0, 0},
    {3, 0, 2},
    {2, 1, 1},
    {0, 0, 2}
};

// Calculate Need
int need[P][R];

void calculate_need() {
    for (int i = 0; i < P; i++)
        for (int j = 0; j < R; j++)
            need[i][j] = maximum[i][j] - allocation[i][j];
}

bool is_safe() {
    int work[R];
    bool finish[P] = {false};
    int safe_sequence[P];
    int count = 0;

    // work = copy of available
    for (int i = 0; i < R; i++)
        work[i] = available[i];

    while (count < P) {
        bool found = false;
        for (int i = 0; i < P; i++) {
            if (!finish[i]) {
                // Check if need[i] <= work
                bool can_allocate = true;
                for (int j = 0; j < R; j++) {
                    if (need[i][j] > work[j]) {
                        can_allocate = false;
                        break;
                    }
                }

                if (can_allocate) {
                    // Simulate resource reclamation
                    for (int j = 0; j < R; j++)
                        work[j] += allocation[i][j];
                    finish[i] = true;
                    safe_sequence[count++] = i;
                    found = true;
                }
            }
        }
        if (!found)
            return false;  // No safe sequence
    }

    printf("Safe sequence: ");
    for (int i = 0; i < P; i++)
        printf("P%d ", safe_sequence[i]);
    printf("\n");
    return true;
}

int main() {
    calculate_need();

    if (is_safe())
        printf("System is in safe state.\n");
    else
        printf("System is in unsafe state!\n");

    return 0;
}
```

---

## 6. Deadlock Detection and Recovery

### Detection Algorithm

```
┌─────────────────────────────────────────────────────────┐
│                  Deadlock Detection Algorithm            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Single instance resources:                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Use Wait-for Graph                               │  │
│  │                                                   │  │
│  │  Remove resource nodes from RAG:                 │  │
│  │                                                   │  │
│  │  Pi → Rq → Pj  transforms to→  Pi → Pj           │  │
│  │  (Pi requests Rq, Rq assigned to Pj)             │  │
│  │                                                   │  │
│  │  Cycle in wait-for graph → Deadlock              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Multiple instance resources:                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Use similar approach to Banker's algorithm       │  │
│  │  Safety check based on current requests          │  │
│  │  Identify set of deadlocked processes             │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Detection frequency:                                   │
│  • Every resource request → high overhead               │
│  • Periodically (e.g., every 5 minutes)                 │
│  • When CPU utilization drops                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Wait-for Graph Example

```
┌─────────────────────────────────────────────────────────┐
│                   Wait-for Graph Example                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Resource Allocation Graph:    Wait-for Graph:         │
│                                                         │
│     ○P1 ─request─▶ □R1         ○P1 ─────────┐          │
│         ↑         ↓ assign          ↓        │          │
│     assign│       ○P2             ○P2 ◀───────┘         │
│         │          ↓ request          ↓                 │
│       □R2 ◀────┘                  ○P3 ────────┐        │
│         ↓ assign                      ↓        │        │
│        ○P3 ─request─▶ □R3 ◀assign─ ○P4    ○P4 ◀────┘  │
│                                                         │
│  R1 → P2 → R2 → P3 → R3 ← P4                           │
│                                                         │
│  Wait-for Graph:                                        │
│  P1 → P2 → P3 → P4 → P3 (cycle!)                       │
│                                                         │
│  → Deadlock exists (P3, P4 are deadlocked)             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Deadlock Recovery Methods

```
┌─────────────────────────────────────────────────────────┐
│                  Deadlock Recovery Methods               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Process Termination                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Method 1: Abort all deadlocked processes         │  │
│  │    • Certain but expensive                        │  │
│  │    • Long-running processes also terminated       │  │
│  │                                                   │  │
│  │  Method 2: Abort one at a time                    │  │
│  │    • Recheck for deadlock after each termination  │  │
│  │    • Selection criteria: priority, run time, resources│ │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  2. Resource Preemption                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Forcibly take resources from processes and     │  │
│  │    assign to others                               │  │
│  │                                                   │  │
│  │  Considerations:                                  │  │
│  │  • Victim selection: which process to preempt     │  │
│  │  • Rollback: restore preempted process to safe state│ │
│  │  • Starvation prevention: same process not repeatedly│ │
│  │    victimized                                     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  3. Checkpoint/Recovery                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Periodically save process state (checkpoint)   │  │
│  │  • Rollback to previous checkpoint when deadlock  │  │
│  │  • Commonly used in database systems              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Deadlock Handling Method Comparison

```
┌──────────────┬──────────────┬──────────────┬───────────────┐
│     Method    │   Overhead   │  Resource    │ Implementation│
│              │              │ Utilization  │  Complexity   │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ Prevention   │ High         │ Low          │ Low           │
│              │              │              │               │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ Avoidance    │ High         │ Medium       │ High          │
│              │              │              │               │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ Detection/   │ Medium       │ High         │ Medium        │
│ Recovery     │              │              │               │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ Ignore       │ None         │ Highest      │ None          │
│ (Ostrich)    │              │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘

Ignore (Ostrich Algorithm):
• Assumes deadlock occurs rarely
• Manually reboot if it occurs
• Adopted by most OSes: Unix, Linux, Windows
• Practical approach
```

---

## 7. Practice Problems

### Problem 1: Deadlock Necessary Conditions

Which of the following is NOT one of the four necessary conditions for deadlock?

A. Mutual Exclusion
B. Circular Wait
C. Priority Inversion
D. No Preemption
E. Hold and Wait

<details>
<summary>Show Answer</summary>

**Answer: C. Priority Inversion**

Four necessary conditions for deadlock:
1. Mutual Exclusion
2. Hold and Wait
3. No Preemption
4. Circular Wait

Priority inversion is a different concurrency problem from deadlock.

</details>

### Problem 2: Resource Allocation Graph

Determine if the following resource allocation graph represents a deadlock state.

```
P1 → R1, R1 → P2
P2 → R2, R2 → P3
P3 → R1
```

<details>
<summary>Show Answer</summary>

**Analysis:**
- P1 requests R1
- R1 is assigned to P2
- P2 requests R2
- R2 is assigned to P3
- P3 requests R1

**Wait relationships:**
- P1 → P2 (due to R1)
- P2 → P3 (due to R2)
- P3 → P2 (due to R1, which P2 holds)

**Cycle:** P2 → P3 → P2

**Conclusion: Deadlock** (assuming single instance per resource)

</details>

### Problem 3: Banker's Algorithm

Determine if the system is in a safe state given:

Available = [1, 1, 2]

| Process | Allocation | Max |
|---------|-----------|-----|
| P0 | (0,1,0) | (2,2,2) |
| P1 | (1,0,0) | (1,1,2) |
| P2 | (0,0,1) | (1,2,3) |

<details>
<summary>Show Answer</summary>

**Calculate Need:**
- P0: (2,2,2) - (0,1,0) = (2,1,2)
- P1: (1,1,2) - (1,0,0) = (0,1,2)
- P2: (1,2,3) - (0,0,1) = (1,2,2)

**Safety check:**
1. Available = [1,1,2]
2. Need[P1] = [0,1,2] <= [1,1,2]? Yes!
   - After P1: Available = [1,1,2] + [1,0,0] = [2,1,2]
3. Need[P0] = [2,1,2] <= [2,1,2]? Yes!
   - After P0: Available = [2,1,2] + [0,1,0] = [2,2,2]
4. Need[P2] = [1,2,2] <= [2,2,2]? Yes!
   - After P2: Done

**Safe sequence: <P1, P0, P2>**
**Conclusion: Safe state**

</details>

### Problem 4: Circular Wait Prevention

Explain the method of assigning ordering to resources to prevent circular wait, and prove why this method prevents deadlock.

<details>
<summary>Show Answer</summary>

**Method:**
- Assign unique ordering number to all resource types
- Processes must request resources only in ascending order

**Proof:**
For circular wait to occur:
P0 → R(i0) → P1 → R(i1) → ... → Pn → R(in) → P0

Each arrow means "holds then requests":
- P0 holds R(i0) and P1 waits for R(i0)
- P1 holds R(i1) and P2 waits for R(i1)
- ...
- Pn holds R(in) and P0 waits for R(in)

By ascending order rule:
- For P0 to request R(in), must have i0 < in
- But if circular: in < i0 < i1 < ... < in (contradiction!)

Therefore circular wait impossible → deadlock impossible

</details>

### Problem 5: Deadlock Recovery

When using the "abort one process at a time" recovery method after deadlock occurs, explain the criteria for selecting victims.

<details>
<summary>Show Answer</summary>

**Victim selection criteria:**

1. **Process priority**
   - Terminate lower priority processes first

2. **Execution time**
   - Terminate processes that ran for shorter time (minimize loss)
   - Or protect processes near completion

3. **Resource usage**
   - Terminate processes holding more resources (free more resources)

4. **Resources needed to complete**
   - Terminate processes needing many more resources

5. **Process type**
   - Protect interactive processes over batch jobs

6. **Starvation prevention**
   - Prevent same process from being repeatedly victimized
   - Count and limit number of terminations

**Optimal selection:** Define cost function as weighted sum of above criteria, select minimum cost process

</details>

---

## Next Steps

This completes the process synchronization section. Next learning topic:
- [10_Memory_Management_Basics.md](./10_Memory_Management_Basics.md) - Memory management

---

## References

- [OSTEP - Deadlock](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-bugs.pdf)
- [Operating System Concepts - Chapter 7](https://www.os-book.com/)
- [Banker's Algorithm Visualization](https://www.cs.uic.edu/~jbell/CourseNotes/OperatingSystems/7_Deadlocks.html)
