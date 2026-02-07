# CPU Scheduling Basics

## Overview

CPU scheduling is the process of determining which of the ready processes should be allocated the CPU. This lesson covers CPU burst and I/O burst, scheduling goals, preemptive/non-preemptive scheduling, and the roles of various schedulers.

---

## Table of Contents

1. [What is CPU Scheduling?](#1-what-is-cpu-scheduling)
2. [CPU Burst and I/O Burst](#2-cpu-burst-and-io-burst)
3. [Scheduling Goals](#3-scheduling-goals)
4. [Preemptive vs Non-preemptive Scheduling](#4-preemptive-vs-non-preemptive-scheduling)
5. [Types of Schedulers](#5-types-of-schedulers)
6. [Dispatcher](#6-dispatcher)
7. [Practice Problems](#7-practice-problems)

---

## 1. What is CPU Scheduling?

### Definition

```
CPU Scheduling = Selecting which process in the Ready queue
                should be allocated the CPU

┌─────────────────────────────────────────────────────────┐
│                   CPU Scheduling Location                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────┐    ┌─────────────────────────────────────┐  │
│  │ New    │───▶│          Ready Queue               │  │
│  │Process │    │  P1 → P2 → P3 → P4 → ...          │  │
│  └────────┘    └─────────────┬───────────────────────┘  │
│                              │                          │
│                              │ CPU Scheduler selects    │
│                              ▼                          │
│                       ┌───────────┐                     │
│                       │    CPU    │                     │
│                       │ (Running) │                     │
│                       └─────┬─────┘                     │
│                             │                           │
│            ┌────────────────┼────────────────┐          │
│            ▼                ▼                ▼          │
│       ┌─────────┐    ┌───────────┐    ┌─────────┐      │
│       │Terminate│    │ I/O Wait  │    │ Return  │      │
│       │         │    │  (Wait)   │    │to Ready │      │
│       └─────────┘    └───────────┘    └─────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Why Scheduling is Needed

```
1. Maximize CPU utilization
   - Always have processes ready to execute so CPU doesn't idle

2. Fair resource distribution
   - Provide appropriate CPU time to all processes

3. Optimize system performance
   - Increase throughput, decrease response time

4. Support multitasking
   - Make it appear that multiple processes run simultaneously

┌──────────────────────────────────────────────────────────┐
│             Without Scheduling vs With Scheduling         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Without Scheduling (Sequential):                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ P1 (I/O wait)   │ P2 (I/O wait)   │ P3 (I/O wait)  │ │
│  └─────────────────────────────────────────────────────┘ │
│  Time: ████░░░░░░░████░░░░░░░████░░░░                   │
│               Lots of CPU idle time                      │
│                                                          │
│  With Scheduling:                                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ P1 │ P2 │ P3 │ P1 │ P2 │ P3 │ ...                  │ │
│  └─────────────────────────────────────────────────────┘ │
│  Time: ██████████████████████████                       │
│               Minimal CPU idle time                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 2. CPU Burst and I/O Burst

### What is a Burst?

```
Process Execution = Alternation of CPU Burst and I/O Burst

┌─────────────────────────────────────────────────────────┐
│                    Process Execution Cycle               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Start ──▶ CPU Burst ──▶ I/O Burst ──▶ CPU Burst ──▶ ...│
│             │            │            │                 │
│             ▼            ▼            ▼                 │
│          Computation    I/O Wait   Computation         │
│          (Running)     (Waiting)   (Running)           │
│                                                         │
└─────────────────────────────────────────────────────────┘

Timeline Example:

Process P1:
┌──────┬──────────┬───────┬──────────┬──────┬──────┐
│ CPU  │  I/O     │  CPU  │   I/O    │ CPU  │ Exit │
│ 10ms │  50ms    │  5ms  │  30ms    │ 8ms  │      │
└──────┴──────────┴───────┴──────────┴──────┴──────┘
```

### CPU-bound vs I/O-bound Processes

```
┌──────────────────────────────────────────────────────────┐
│              Burst Patterns by Process Type              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  CPU-bound Process (Computation intensive):              │
│  ┌────────────────────┬──┬────────────────────┬──┐       │
│  │   Long CPU burst   │IO│   Long CPU burst   │IO│       │
│  └────────────────────┴──┴────────────────────┴──┘       │
│  Examples: Scientific computation, video encoding,        │
│            compilers                                     │
│                                                          │
│  I/O-bound Process (I/O intensive):                      │
│  ┌──┬─────┬──┬─────┬──┬─────┬──┬─────┐                   │
│  │CP│ I/O │CP│ I/O │CP│ I/O │CP│ I/O │                   │
│  └──┴─────┴──┴─────┴──┴─────┴──┴─────┘                   │
│  Examples: Text editors, web browsers,                   │
│            interactive programs                          │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────┬────────────────┬────────────────────────┐
│      Feature     │  CPU-bound    │      I/O-bound        │
├─────────────────┼────────────────┼────────────────────────┤
│ CPU burst length│ Long           │ Short                  │
│ I/O burst freq  │ Low            │ High                   │
│ Scheduling need │ Long time slice│ Fast response time     │
│ Examples        │ Numerical comp │ Web server, database   │
└─────────────────┴────────────────┴────────────────────────┘
```

### CPU Burst Distribution

```
Frequency
 ↑
 │  ██
 │  ██
 │  ████
 │  ██████
 │  ████████
 │  ██████████
 │  ████████████
 │  ██████████████
 │  ████████████████
 │  ██████████████████
 └────────────────────────────────▶ CPU Burst Length

Most processes have short CPU bursts
→ Many I/O-bound processes
→ Processing short processes first reduces average wait time
```

---

## 3. Scheduling Goals

### Key Performance Metrics

```
┌─────────────────────────────────────────────────────────┐
│                    Scheduling Metrics                    │
├───────────────────┬─────────────────────────────────────┤
│ CPU Utilization   │ Percentage of time CPU is working   │
│                   │ Goal: Maximize (40%~90%)            │
├───────────────────┼─────────────────────────────────────┤
│ Throughput        │ Number of processes completed per   │
│                   │ unit time. Goal: Maximize           │
├───────────────────┼─────────────────────────────────────┤
│ Turnaround Time   │ Time from submission to completion  │
│                   │ = Wait + Execution + I/O time       │
│                   │ Goal: Minimize                      │
├───────────────────┼─────────────────────────────────────┤
│ Waiting Time      │ Total time spent in Ready queue     │
│                   │ Goal: Minimize                      │
├───────────────────┼─────────────────────────────────────┤
│ Response Time     │ Time from submission to first       │
│                   │ response. Important for interactive │
│                   │ systems. Goal: Minimize             │
└───────────────────┴─────────────────────────────────────┘
```

### Time Concept Visualization

```
Process P Timeline:

Arrival                 Completion
  │                   │
  ▼                   ▼
──┬─────────────────────┬─────────────────────┬──▶ Time
  │                     │                     │
  │◀── Turnaround Time ────────────────────▶│
  │                                           │
  │  Wait  │ Execute │ Wait │ Execute        │
  │◀Wait ▶│         │◀Wait▶│                │
  │                                           │
  │◀─▶First Response                         │
  │Response                                   │
  │Time                                       │


Calculation Example:
┌─────────────────────────────────────────────────────────┐
│ Process P:                                              │
│   Arrival time = 0                                      │
│   Execution time = 10ms (total CPU usage time)          │
│   Completion time = 25ms                                │
│                                                         │
│ Turnaround Time = Completion - Arrival = 25 - 0 = 25ms │
│ Waiting Time = Turnaround - Execution = 25 - 10 = 15ms │
└─────────────────────────────────────────────────────────┘
```

### Trade-offs Between Goals

```
┌─────────────────────────────────────────────────────────┐
│                   Scheduling Goal Trade-offs             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  CPU Utilization ←─────────────────────▶ Response Time  │
│  (Maximize)          Conflicting         (Minimize)     │
│                                                         │
│  Throughput ←───────────────────────▶ Fairness         │
│  (Maximize)        Conflicting        (Equal dist)      │
│                                                         │
│  Priorities differ by system type:                      │
│                                                         │
│  ┌────────────────┬───────────────────────────────────┐ │
│  │ Batch System   │ Throughput, CPU utilization first │ │
│  ├────────────────┼───────────────────────────────────┤ │
│  │ Interactive    │ Response time first               │ │
│  ├────────────────┼───────────────────────────────────┤ │
│  │ Real-time      │ Meeting deadlines first           │ │
│  └────────────────┴───────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Preemptive vs Non-preemptive Scheduling

### Non-preemptive Scheduling

```
┌─────────────────────────────────────────────────────────┐
│              Non-preemptive Scheduling                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  • Process runs until it voluntarily releases CPU       │
│  • CPU cannot be forcibly taken away                    │
│                                                         │
│  CPU Release Points:                                    │
│  1. Process terminates                                  │
│  2. Transitions to waiting state (I/O request)          │
│  3. Voluntary yield (yield())                           │
│                                                         │
│  Timeline:                                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │ P1 (10ms)              │ P2 (5ms)    │ P3 (3ms) │   │
│  └──────────────────────────────────────────────────┘   │
│  P2, P3 wait even if they arrive while P1 runs          │
│                                                         │
│  Advantages: Simple implementation, low context         │
│              switch overhead                            │
│  Disadvantages: Long response time, long process can    │
│                 monopolize CPU                          │
│                                                         │
│  Example Algorithms: FCFS, SJF (non-preemptive)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Preemptive Scheduling

```
┌─────────────────────────────────────────────────────────┐
│                Preemptive Scheduling                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  • Scheduler can forcibly take away CPU from process    │
│  • Can immediately allocate CPU to more important       │
│    process when it arrives                              │
│                                                         │
│  CPU Preemption Points:                                 │
│  1. Time slice (Time Quantum) expires                   │
│  2. Higher priority process arrives                     │
│  3. Process transitions to Ready after I/O completion   │
│                                                         │
│  Timeline (Round Robin example):                        │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐           │
│  │  P1  │  P2  │  P3  │  P1  │  P2  │  P1  │           │
│  └──────┴──────┴──────┴──────┴──────┴──────┘           │
│  Process switches every time slice                      │
│                                                         │
│  Advantages: Better response time, prevents CPU         │
│              monopolization                             │
│  Disadvantages: Context switch overhead,                │
│                 complex synchronization issues          │
│                                                         │
│  Example Algorithms: RR, SRTF, Priority (preemptive)   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Comparison

```
┌─────────────────┬─────────────────┬─────────────────────┐
│      Feature     │  Non-preemptive │     Preemptive      │
├─────────────────┼─────────────────┼─────────────────────┤
│ CPU preemption  │ Not possible    │ Possible            │
├─────────────────┼─────────────────┼─────────────────────┤
│ Response time   │ Can be long     │ Short               │
├─────────────────┼─────────────────┼─────────────────────┤
│ Context switches│ Few             │ Many                │
├─────────────────┼─────────────────┼─────────────────────┤
│ Implementation  │ Low complexity  │ High complexity     │
├─────────────────┼─────────────────┼─────────────────────┤
│ Sync issues     │ Few             │ Need race condition │
│                 │                 │ handling            │
├─────────────────┼─────────────────┼─────────────────────┤
│ Suitable for    │ Batch systems   │ Interactive/RT      │
└─────────────────┴─────────────────┴─────────────────────┘
```

### Scheduling Decision Points

```
┌─────────────────────────────────────────────────────────┐
│                CPU Scheduling Decision Points            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Running → Waiting (I/O request)      [Non-preemptive]│
│     • Process voluntarily releases CPU                  │
│                                                         │
│  2. Running → Ready (interrupt, timeout) [Preemptive]   │
│     • Time slice expired                                │
│     • Higher priority process arrived                   │
│                                                         │
│  3. Waiting → Ready (I/O complete)       [Can preempt]  │
│     • Can preempt if completed I/O process has          │
│       higher priority than current                      │
│                                                         │
│  4. Running → Terminated (exit)          [Non-preemptive]│
│     • Process terminates, need to select next           │
│                                                         │
│                                                         │
│  Only 1, 4: Non-preemptive scheduling                   │
│  All 1~4: Preemptive scheduling                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Types of Schedulers

### Three-Level Scheduler Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     3-Level Scheduler Structure              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Job Queue                         │    │
│  │     All processes on disk                           │    │
│  │     ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                  │    │
│  │     │P1 │ │P2 │ │P3 │ │P4 │ │...│                  │    │
│  │     └───┘ └───┘ └───┘ └───┘ └───┘                  │    │
│  └───────────────────────┬─────────────────────────────┘    │
│                          │                                  │
│                    Long-term Scheduler                      │
│                   (Long-term Scheduler)                     │
│                    • Job admission                          │
│                    • Control degree of multiprogramming     │
│                    • Execution frequency: seconds~minutes   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Ready Queue                       │    │
│  │     Processes in memory waiting to execute          │    │
│  │     ┌───┐ ┌───┐ ┌───┐                              │    │
│  │     │P1 │→│P2 │→│P3 │                              │    │
│  │     └───┘ └───┘ └───┘                              │    │
│  └──────────────┬────────────────────────┬─────────────┘    │
│                 │                        │                  │
│           Short-term            Medium-term                 │
│           Scheduler              Scheduler                  │
│         (Short-term Scheduler) (Medium-term Scheduler)      │
│          • CPU allocation         • Swapping               │
│          • Runs every ms          • Memory management      │
│          • Most frequent          • Adjust multiprogramming │
│                 │                        │                  │
│                 ▼                        ▼                  │
│            ┌─────────┐           ┌───────────────┐          │
│            │   CPU   │           │   Suspended   │          │
│            │(Running)│           │    Queue      │          │
│            └─────────┘           └───────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Long-term Scheduler (Job Scheduler)

```
┌─────────────────────────────────────────────────────────┐
│               Long-term Scheduler (Job Scheduler)        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Role:                                                  │
│  • Disk jobs → Memory (Ready Queue)                    │
│  • Decides which processes to admit to system           │
│  • Controls degree of multiprogramming                  │
│                                                         │
│  Decision Criteria:                                     │
│  • System resource (memory, CPU) status                 │
│  • Appropriate mix of CPU-bound and I/O-bound           │
│                                                         │
│  Execution Frequency: Rarely (seconds~minutes)          │
│                                                         │
│  Modern Systems:                                        │
│  • Rarely used in time-sharing systems                  │
│  • All processes go directly to Ready Queue             │
│  • Virtual memory handles memory management             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Short-term Scheduler (CPU Scheduler)

```
┌─────────────────────────────────────────────────────────┐
│               Short-term Scheduler (CPU Scheduler)       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Role:                                                  │
│  • Ready Queue → CPU (Running)                         │
│  • Decides which process gets CPU allocation            │
│                                                         │
│  Execution Frequency: Very frequently (milliseconds)    │
│  → Fast algorithm essential                             │
│                                                         │
│  Invocation Points:                                     │
│  • Time slice expiration                                │
│  • Process blocking (I/O request)                       │
│  • Process termination                                  │
│  • Interrupt occurrence                                 │
│                                                         │
│  Algorithms Used:                                       │
│  • FCFS, SJF, Priority, Round Robin, MLFQ, etc.        │
│                                                         │
│  Note: Scheduler itself uses CPU time                   │
│        → Need to minimize scheduler overhead            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Medium-term Scheduler (Swapper)

```
┌─────────────────────────────────────────────────────────┐
│              Medium-term Scheduler (Swapper)             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Role:                                                  │
│  • Manage swapping                                      │
│  • Move processes between memory ↔ disk                 │
│  • Dynamically adjust degree of multiprogramming        │
│                                                         │
│  Operation:                                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │    Memory (Ready/Blocked)                       │    │
│  │    ┌───┐ ┌───┐ ┌───┐                           │    │
│  │    │P1 │ │P2 │ │P3 │  ← When memory low,       │    │
│  │    └───┘ └───┘ └───┘       swap out P3         │    │
│  └─────────────────────────────────────────────────┘    │
│              │ swap out          ▲ swap in             │
│              ▼                   │                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │    Disk (Suspended)                             │    │
│  │    ┌───┐ ┌───┐                                 │    │
│  │    │P4 │ │P5 │  → Swap in when memory available│    │
│  │    └───┘ └───┘                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Swap Out Criteria:                                     │
│  • Long-waiting processes                               │
│  • Low priority processes                               │
│  • Memory-intensive processes                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Scheduler Comparison

```
┌────────────────────┬────────────────┬────────────────┬────────────────┐
│       Feature       │   Long-term    │  Short-term    │  Medium-term   │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Other name         │ Job Scheduler  │ CPU Scheduler  │ Swapper       │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Transition         │ Disk→Memory    │ Ready→Running  │ Memory↔Disk   │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Execution freq     │ Rarely         │ Very often     │ Sometimes     │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Speed requirement  │ Can be slow    │ Must be fast   │ Medium        │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Multiprogramming   │ Control total  │ No effect      │ Dynamic adjust│
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Modern systems     │ Rarely used    │ Core usage     │ Replaced by VM│
└────────────────────┴────────────────┴────────────────┴────────────────┘
```

---

## 6. Dispatcher

### What is a Dispatcher?

```
┌─────────────────────────────────────────────────────────┐
│                    Dispatcher                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Dispatcher = Module that actually gives CPU control    │
│              to the process selected by scheduler       │
│                                                         │
│  Scheduler and Dispatcher Relationship:                 │
│  ┌───────────────┐          ┌───────────────┐           │
│  │   Scheduler    │ ─decide─▶│   Dispatcher  │ ─execute─▶│
│  │ "Run P3"      │          │ Perform CPU   │           │
│  │               │          │ allocation    │           │
│  └───────────────┘          └───────────────┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Dispatcher Tasks

```
┌─────────────────────────────────────────────────────────┐
│                  Dispatcher Tasks                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Context Switch                                      │
│     • Save current process state to PCB                 │
│     • Restore new process state from PCB                │
│                                                         │
│  2. Switch to User Mode                                 │
│     • Kernel mode → User mode                           │
│                                                         │
│  3. Restart Program                                     │
│     • Jump to appropriate location in new process       │
│     • Set PC (Program Counter)                          │
│                                                         │
│  Timeline:                                              │
│  ┌────────┬──────────────────┬────────────────────┐     │
│  │ P1 exec│  Dispatcher exec  │      P2 exec       │     │
│  │        │(Dispatch latency) │                   │     │
│  └────────┴──────────────────┴────────────────────┘     │
│           ↑                                             │
│      Dispatch Latency (should be as short as possible)  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Dispatch Latency

```
Dispatch Latency = Time from stopping current process
                  to starting new process execution

Components:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. Interrupt handling time                             │
│  2. Save current process state                          │
│  3. Execute scheduling algorithm                        │
│  4. Restore new process state                           │
│  5. Cache/TLB related costs                             │
│                                                         │
│  Typically: 1~10 microseconds                           │
│                                                         │
│  Minimization Methods:                                  │
│  • Efficient scheduling algorithm                       │
│  • Optimized context switch code                        │
│  • Use hardware support                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Problem 1: Basic Concepts

Define the following terms.

1. CPU Burst
2. Throughput
3. Turnaround Time
4. Preemptive Scheduling

<details>
<summary>Show Answer</summary>

1. **CPU Burst**: The time period during which a process continuously uses the CPU. The interval of performing only computation without I/O requests.

2. **Throughput**: The number of processes completed per unit time. An indicator of system efficiency.

3. **Turnaround Time**: Total time from when a process is submitted to the system until it completes. Waiting time + Execution time + I/O time.

4. **Preemptive Scheduling**: A scheduling method where the scheduler can forcibly take away CPU from a running process. Occurs at points such as time slice expiration or higher priority process arrival.

</details>

### Problem 2: Time Calculation

Calculate the average waiting time and average turnaround time for the following processes.
(FCFS scheduling, all processes arrive at time 0)

| Process | Execution Time |
|---------|---------------|
| P1 | 24 |
| P2 | 3 |
| P3 | 3 |

<details>
<summary>Show Answer</summary>

**Gantt Chart:**
```
┌────────────────────────┬───┬───┐
│          P1            │P2 │P3 │
└────────────────────────┴───┴───┘
0                       24  27  30
```

**Waiting Time:**
- P1: 0
- P2: 24
- P3: 27
- Average Waiting Time = (0 + 24 + 27) / 3 = 17

**Turnaround Time:**
- P1: 24 - 0 = 24
- P2: 27 - 0 = 27
- P3: 30 - 0 = 30
- Average Turnaround Time = (24 + 27 + 30) / 3 = 27

</details>

### Problem 3: Scheduler Identification

Choose the matching scheduler for each description.

A. Long-term Scheduler
B. Short-term Scheduler
C. Medium-term Scheduler

1. ( ) Must execute very frequently, every millisecond.
2. ( ) Selects an appropriate mix of CPU-bound and I/O-bound processes.
3. ( ) Swaps processes to disk when memory is low.
4. ( ) Decides which process gets CPU allocation.

<details>
<summary>Show Answer</summary>

1. (B) Short-term Scheduler - Executes very frequently, needs fast algorithm
2. (A) Long-term Scheduler - Considers process mix for system balance
3. (C) Medium-term Scheduler - Responsible for swapping
4. (B) Short-term Scheduler - Responsible for CPU scheduling

</details>

### Problem 4: Preemptive vs Non-preemptive

Identify whether each situation is preemptive or non-preemptive.

1. Process calls exit() and terminates
2. Time slice expires and switches to another process
3. Process requests I/O and transitions to waiting state
4. High priority process arrives in Ready queue and interrupts current process

<details>
<summary>Show Answer</summary>

1. **Non-preemptive** - Process voluntarily terminates
2. **Preemptive** - Scheduler forcibly takes away CPU
3. **Non-preemptive** - Process voluntarily releases CPU
4. **Preemptive** - Scheduler forcibly takes away CPU

</details>

### Problem 5: Dispatcher

Describe three main tasks performed by the dispatcher, and explain the impact on the system if dispatch latency is long.

<details>
<summary>Show Answer</summary>

**Main Dispatcher Tasks:**
1. **Context Switch**: Save current process's state (registers, PC, etc.) to PCB and restore new process's state from PCB
2. **Mode Switch**: Transition from kernel mode to user mode
3. **Jump**: Transfer control to the appropriate location (address pointed to by PC) in new process

**Impact of Long Dispatch Latency:**
- Decreased CPU utilization (reduced percentage of time used for actual work)
- Increased response time (slower reaction to user requests)
- Decreased throughput (fewer processes completed per unit time)
- Increased system overhead

</details>

---

## Next Steps

- [05_Scheduling_Algorithms.md](./05_Scheduling_Algorithms.md) - FCFS, SJF, Priority, RR algorithms

---

## References

- [OSTEP - CPU Scheduling](https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched.pdf)
- [Operating System Concepts - Chapter 5](https://www.os-book.com/)
- [Linux Scheduler Documentation](https://www.kernel.org/doc/html/latest/scheduler/)
