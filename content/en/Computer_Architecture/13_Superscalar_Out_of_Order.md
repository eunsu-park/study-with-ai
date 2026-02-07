# Superscalar and Out-of-Order Execution

## Overview

Modern high-performance processors cannot achieve significant performance gains by simply increasing clock speed. Superscalar and Out-of-Order Execution are techniques that exploit Instruction-Level Parallelism (ILP) to execute multiple instructions simultaneously in a single cycle. This lesson covers ILP concepts, superscalar architecture, and the principles and implementation of out-of-order execution.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: Pipelining, Branch Prediction, CPU Architecture Basics

---

## Table of Contents

1. [Instruction-Level Parallelism (ILP)](#1-instruction-level-parallelism-ilp)
2. [Superscalar Processors](#2-superscalar-processors)
3. [Need for Out-of-Order Execution](#3-need-for-out-of-order-execution)
4. [Register Renaming](#4-register-renaming)
5. [Tomasulo Algorithm](#5-tomasulo-algorithm)
6. [Reorder Buffer (ROB)](#6-reorder-buffer-rob)
7. [Modern Processor Implementations](#7-modern-processor-implementations)
8. [Practice Problems](#8-practice-problems)

---

## 1. Instruction-Level Parallelism (ILP)

### 1.1 ILP Concept

Instruction-Level Parallelism (ILP) refers to the potential parallelism of instructions within a program that can be executed simultaneously.

```
Sequential vs Parallel Execution:

Sequential Execution:
  Time →
  t1    t2    t3    t4    t5    t6
  ├─────┼─────┼─────┼─────┼─────┼─────┤
  │ I1  │ I2  │ I3  │ I4  │ I5  │ I6  │
  └─────┴─────┴─────┴─────┴─────┴─────┘
  Total 6 cycles

Parallel Execution (ILP = 2):
  Time →
  t1    t2    t3
  ├─────┼─────┼─────┤
  │ I1  │ I3  │ I5  │
  │ I2  │ I4  │ I6  │
  └─────┴─────┴─────┘
  Total 3 cycles
```

### 1.2 Data Dependence

The most important factor limiting ILP is data dependence.

#### RAW (Read After Write) - True Dependence

```assembly
I1: ADD R1, R2, R3    ; R1 = R2 + R3
I2: SUB R4, R1, R5    ; R4 = R1 - R5  (uses R1, needs I1's result)
```

```
I1 ────────→ I2
   R1 dependence

I2 can only execute after I1 writes to R1
```

#### WAR (Write After Read) - Anti-dependence

```assembly
I1: ADD R1, R2, R3    ; Read R2
I2: SUB R2, R4, R5    ; Write R2 (must write after I1 reads R2)
```

```
I2 must not overwrite R2 before I1 reads it
→ Can be resolved by register renaming
```

#### WAW (Write After Write) - Output Dependence

```assembly
I1: ADD R1, R2, R3    ; Write R1
I2: SUB R1, R4, R5    ; Write R1 (writing to same register)
```

```
If I2 completes before I1, final R1 value is wrong
→ Can be resolved by register renaming
```

### 1.3 Dependence Graph

```
Program:
I1: LD   R1, 0(R10)     ; Load from memory
I2: ADD  R2, R1, R3     ; RAW on R1
I3: LD   R4, 8(R10)     ; Independent
I4: MUL  R5, R4, R6     ; RAW on R4
I5: ADD  R7, R2, R5     ; RAW on R2, R5
I6: ST   R7, 16(R10)    ; RAW on R7

Dependence graph:
        I1          I3
         │           │
         ▼           ▼
        I2          I4
         │           │
         └─────┬─────┘
               │
               ▼
              I5
               │
               ▼
              I6

Parallelizable groups:
- Level 1: I1, I3 (can execute simultaneously)
- Level 2: I2, I4 (can execute simultaneously)
- Level 3: I5
- Level 4: I6

Minimum execution time: 4 levels (4 cycles) vs sequential: 6 cycles
ILP = 6/4 = 1.5
```

### 1.4 Control Dependence

```assembly
      BEQ  R1, R2, LABEL    ; Branch instruction
      ADD  R3, R4, R5       ; Execute if branch not taken
      ...
LABEL:
      SUB  R6, R7, R8       ; Execute if branch taken
```

```
Don't know which instruction to execute before branch result
→ Resolved by branch prediction
```

---

## 2. Superscalar Processors

### 2.1 Superscalar Concept

A superscalar processor can fetch, decode, and execute multiple instructions per cycle.

```
Scalar vs Superscalar:

Scalar Pipeline (IPC ≤ 1):
┌─────┬─────┬─────┬─────┬─────┐
│ IF  │ ID  │ EX  │ MEM │ WB  │ I1
└─────┴─────┴─────┴─────┴─────┘
      ┌─────┬─────┬─────┬─────┬─────┐
      │ IF  │ ID  │ EX  │ MEM │ WB  │ I2
      └─────┴─────┴─────┴─────┴─────┘

2-way Superscalar (IPC ≤ 2):
┌─────┬─────┬─────┬─────┬─────┐
│ IF  │ ID  │ EX  │ MEM │ WB  │ I1
├─────┼─────┼─────┼─────┼─────┤
│ IF  │ ID  │ EX  │ MEM │ WB  │ I2
└─────┴─────┴─────┴─────┴─────┘
      ┌─────┬─────┬─────┬─────┬─────┐
      │ IF  │ ID  │ EX  │ MEM │ WB  │ I3
      ├─────┼─────┼─────┼─────┼─────┤
      │ IF  │ ID  │ EX  │ MEM │ WB  │ I4
      └─────┴─────┴─────┴─────┴─────┘
```

### 2.2 Superscalar Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     4-way Superscalar Processor                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Instruction Fetch Unit                       │   │
│  │  ┌──────────┐    ┌──────────────────────────────────┐    │   │
│  │  │    PC    │───→│  Instruction Cache (I-Cache)      │    │   │
│  │  └──────────┘    └──────────────────────────────────┘    │   │
│  │                              │                            │   │
│  │                    4 instructions/cycle                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Instruction Decode Unit                      │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │   │
│  │  │Decoder1│  │Decoder2│  │Decoder3│  │Decoder4│         │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Issue Unit                             │   │
│  │          (Dependence checking and execution unit alloc)   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                │                                 │
│        ┌───────────┬──────────┼──────────┬───────────┐         │
│        ▼           ▼          ▼          ▼           ▼         │
│  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐  │
│  │  ALU 1   ││  ALU 2   ││  FPU     ││  Load    ││  Store   │  │
│  │          ││          ││          ││  Unit    ││  Unit    │  │
│  └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘  │
│        │           │          │          │           │         │
│        └───────────┴──────────┴──────────┴───────────┘         │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Write-back / Commit                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Issue Policies

#### In-Order Issue

```
Instructions issued to execution units in program order

Program:
I1: ADD R1, R2, R3
I2: MUL R4, R1, R5    ; Depends on I1 (stall)
I3: SUB R6, R7, R8    ; Independent but waits behind I2

Timeline:
Cycle 1: Issue I1
Cycle 2: I1 executing, I2 waiting (RAW hazard)
Cycle 3: Issue I2 (after I1 completes)
Cycle 4: Issue I3

Problem: I3 is independent but delayed by I2
```

#### Out-of-Order Issue

```
Independent instructions issued regardless of order

Program:
I1: ADD R1, R2, R3
I2: MUL R4, R1, R5    ; Depends on I1
I3: SUB R6, R7, R8    ; Independent

Timeline:
Cycle 1: Issue I1, I3 (simultaneous)
Cycle 2: I1 completes, Issue I2
Cycle 3: I2 executing

Performance gain: I3 doesn't wait for I2
```

### 2.4 Diverse Execution Units

```
Modern processor execution units example:

┌─────────────────────────────────────────────────┐
│           Execution Units (Intel Core)           │
├────────────────┬────────────────────────────────┤
│ Port 0         │ ALU, FP MUL, FP DIV, Branch    │
├────────────────┼────────────────────────────────┤
│ Port 1         │ ALU, FP ADD, LEA               │
├────────────────┼────────────────────────────────┤
│ Port 2         │ Load (Address Gen)             │
├────────────────┼────────────────────────────────┤
│ Port 3         │ Load (Address Gen)             │
├────────────────┼────────────────────────────────┤
│ Port 4         │ Store Data                     │
├────────────────┼────────────────────────────────┤
│ Port 5         │ ALU, Vector Shuffle, Branch    │
├────────────────┼────────────────────────────────┤
│ Port 6         │ ALU, Branch                    │
├────────────────┼────────────────────────────────┤
│ Port 7         │ Store (Address Gen)            │
└────────────────┴────────────────────────────────┘

Total: 8 ports, max 8 micro-ops per cycle
```

---

## 3. Need for Out-of-Order Execution

### 3.1 Limitations of In-Order Execution

```
Program:
I1: LD   R1, 0(R10)     ; Cache miss - 100 cycles
I2: ADD  R2, R1, R3     ; Depends on I1
I3: LD   R4, 8(R10)     ; Independent - cache hit 4 cycles
I4: MUL  R5, R4, R6     ; Depends on I3
I5: ADD  R7, R8, R9     ; Completely independent

In-Order Execution:
Cycle 1-100: I1 executing (cache miss wait)
Cycle 101:   I2 executes
Cycle 102-105: I3 executes
Cycle 106:   I4 executes
Cycle 107:   I5 executes

Total: 107 cycles

Out-of-Order Execution:
Cycle 1:     I1 starts (cache miss)
Cycle 2-5:   I3 executes (parallel with I1)
Cycle 6:     I4 executes
Cycle 7:     I5 executes
...
Cycle 100:   I1 completes
Cycle 101:   I2 executes

Total: 101 cycles

Speedup: 107/101 = 1.06x (simple example)
Actual difference is much larger
```

### 3.2 Three Stages of OoO Execution

```
┌─────────────────────────────────────────────────────────────┐
│              Out-of-Order Execution Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   In-Order       Out-of-Order       In-Order                │
│   Front-end      Execution          Back-end                │
│                                                              │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Fetch   │    │ Issue/      │    │ Commit/     │         │
│  │ Decode  │───→│ Execute     │───→│ Retire      │         │
│  │ Rename  │    │ (OoO)       │    │ (In-Order)  │         │
│  └─────────┘    └─────────────┘    └─────────────┘         │
│                                                              │
│  Program order   Data flow order    Program order           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Stage 1: Front-end (In-Order)
- Instruction fetch, decode
- Register renaming
- Insert instructions into Issue Queue

Stage 2: Execution (Out-of-Order)
- Execute instructions when operands ready
- Execution order determined by data dependencies
- Parallel execution across multiple units

Stage 3: Back-end (In-Order)
- Commit results in program order
- Ensures precise exception handling
- Update architectural registers
```

---

## 4. Register Renaming

### 4.1 Need for Renaming

```
Program:
I1: ADD R1, R2, R3    ; Write R1
I2: MUL R4, R1, R5    ; Read R1 (RAW)
I3: ADD R1, R6, R7    ; Write R1 (WAW with I1)
I4: SUB R8, R1, R9    ; Read R1 (RAW with I3)

Problem:
- I3 should be able to execute independently of I1, I2
- But they share R1, causing WAW dependence
- If I3 completes before I1, I2 reads wrong value
```

### 4.2 Renaming Operation

```
Architectural Registers: R1-R8 (programmer-visible)
Physical Registers: P1-P64 (actual hardware registers)

Before renaming:
I1: ADD R1, R2, R3
I2: MUL R4, R1, R5
I3: ADD R1, R6, R7
I4: SUB R8, R1, R9

After renaming:
I1: ADD P10, P2, P3    ; R1 → P10
I2: MUL P11, P10, P5   ; R4 → P11, R1 → P10
I3: ADD P12, P6, P7    ; R1 → P12 (new physical register!)
I4: SUB P13, P12, P9   ; R8 → P13, R1 → P12

Result:
- WAW dependence between I1 and I3 eliminated (different physical regs)
- I2 reads P10 (I1's result)
- I4 reads P12 (I3's result)
- I1 and I3 can now execute in parallel!
```

### 4.3 Register Alias Table (RAT)

```
┌─────────────────────────────────────────────────────────┐
│              Register Alias Table (RAT)                  │
├─────────────────────────────────────────────────────────┤
│  Architectural Reg  │  Physical Reg  │  Valid           │
├─────────────────────┼────────────────┼──────────────────┤
│        R0           │      P0        │    1             │
│        R1           │      P12       │    0 (pending)   │
│        R2           │      P2        │    1             │
│        R3           │      P3        │    1             │
│        R4           │      P11       │    0 (pending)   │
│        R5           │      P5        │    1             │
│        R6           │      P6        │    1             │
│        R7           │      P7        │    1             │
│        R8           │      P13       │    0 (pending)   │
│        ...          │      ...       │    ...           │
└─────────────────────┴────────────────┴──────────────────┘

Free List: P14, P15, P16, ...  (available physical registers)
```

### 4.4 Renaming Algorithm

```
Renaming process (instruction: ADD Rd, Rs1, Rs2):

1. Source register renaming:
   - Look up physical registers for Rs1, Rs2 in RAT

2. Destination register renaming:
   - Allocate new physical register from Free List
   - Update RAT mapping for Rd to new physical register

3. Record dependence info:
   - Check if source physical registers are still being produced
   - Establish links to producer instructions

Example:
Instruction: ADD R1, R2, R3

Before:
  RAT[R1] = P5, RAT[R2] = P2, RAT[R3] = P3
  Free List: P10, P11, P12, ...

Renaming:
  1. Rs1(R2) → P2, Rs2(R3) → P3
  2. Rd(R1) → P10 (new allocation)
  3. RAT[R1] = P10

After:
  RAT[R1] = P10, RAT[R2] = P2, RAT[R3] = P3
  Free List: P11, P12, ...

Renamed instruction: ADD P10, P2, P3
```

---

## 5. Tomasulo Algorithm

### 5.1 Background

Algorithm developed by Robert Tomasulo in 1967 for the IBM 360/91 floating-point unit. Forms the basis of modern out-of-order execution processors.

### 5.2 Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                  Tomasulo Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Instruction Queue                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Issue Logic                             │   │
│  │            (Reservation Station allocation)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│            ┌─────────────────┴─────────────────┐                │
│            ▼                                   ▼                │
│  ┌──────────────────────┐          ┌──────────────────────┐    │
│  │  Reservation Stations │          │  Reservation Stations│    │
│  │      (Add/Sub)        │          │      (Mul/Div)       │    │
│  ├──────────────────────┤          ├──────────────────────┤    │
│  │ RS1: Op Vj Vk Qj Qk  │          │ RS4: Op Vj Vk Qj Qk  │    │
│  │ RS2: Op Vj Vk Qj Qk  │          │ RS5: Op Vj Vk Qj Qk  │    │
│  │ RS3: Op Vj Vk Qj Qk  │          │ RS6: Op Vj Vk Qj Qk  │    │
│  └──────────┬───────────┘          └──────────┬───────────┘    │
│             │                                  │                │
│             ▼                                  ▼                │
│  ┌──────────────────────┐          ┌──────────────────────┐    │
│  │      FP Adder        │          │     FP Multiplier    │    │
│  │    (2 cycles)        │          │     (10 cycles)      │    │
│  └──────────┬───────────┘          └──────────┬───────────┘    │
│             │                                  │                │
│             └─────────────────┬────────────────┘                │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Common Data Bus (CDB)                   │   │
│  │               (Result broadcast)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Register File                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Reservation Station Structure

```
┌─────────────────────────────────────────────────────────────┐
│              Reservation Station Entry                       │
├──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────────┤
│ Busy │  Op  │  Vj  │  Vk  │  Qj  │  Qk  │  A   │   Dest    │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼───────────┤
│  1   │ ADD  │ 3.5  │  -   │  -   │ RS4  │  -   │    F2     │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴───────────┘

Field descriptions:
- Busy: Entry in use
- Op: Operation to perform
- Vj, Vk: Source operand values (if already available)
- Qj, Qk: RS that will produce value (if not ready yet)
- A: Memory address (for Load/Store)
- Dest: Destination register for result
```

### 5.4 Operation Process

```
Three-stage processing:

1. Issue
   - Get instruction from Instruction Queue
   - Allocate appropriate Reservation Station
   - Record operand values or producer RS tags

2. Execute
   - Start execution when all operands ready
   - Execute when Qj = 0 AND Qk = 0
   - Perform operation in execution unit

3. Write Result
   - Broadcast result via CDB
   - Waiting RSs receive result
   - Update Register File
```

### 5.5 Example: Tomasulo Execution Trace

```
Program:
I1: LD   F6, 34(R2)
I2: LD   F2, 45(R3)
I3: MUL  F0, F2, F4
I4: SUB  F8, F6, F2
I5: DIV  F10, F0, F6
I6: ADD  F6, F8, F2

Initial state:
  F4 = 2.5 (available)
  Load: 2 cycles, Mul: 10 cycles, Add/Sub: 2 cycles, Div: 40 cycles

=== Cycle 1 ===
Issue I1: LD F6, 34(R2)
  Load1: Busy=1, A=34+R2, Dest=F6
  Register[F6]: Qi=Load1

=== Cycle 2 ===
Issue I2: LD F2, 45(R3)
  Load2: Busy=1, A=45+R3, Dest=F2
  Register[F2]: Qi=Load2
Execute I1: Memory access starts

=== Cycle 3 ===
Issue I3: MUL F0, F2, F4
  Mult1: Busy=1, Op=MUL, Vk=2.5, Qj=Load2, Dest=F0
  Register[F0]: Qi=Mult1
Execute I2: Memory access starts
Write I1: Broadcast F6 value on CDB
  Load1: Busy=0
  Register[F6]: Qi=0, Value=M[34+R2]

=== Cycle 4 ===
Issue I4: SUB F8, F6, F2
  Add1: Busy=1, Op=SUB, Vj=M[34+R2], Qk=Load2, Dest=F8
Write I2: Broadcast F2 value on CDB
  Mult1: Vj=M[45+R3], Qj=0  (value received)
  Add1: Vk=M[45+R3], Qk=0   (value received)

=== Cycle 5 ===
Issue I5: DIV F10, F0, F6
  Mult2: Busy=1, Op=DIV, Vk=M[34+R2], Qj=Mult1, Dest=F10
Execute I3: MUL starts (Vj, Vk ready)
Execute I4: SUB starts (Vj, Vk ready)

=== Cycle 6 ===
Issue I6: ADD F6, F8, F2
  Add2: Busy=1, Op=ADD, Vk=M[45+R3], Qj=Add1, Dest=F6
  Register[F6]: Qi=Add2

=== Cycle 7 ===
Write I4: Broadcast F8 value on CDB
  Add2: Vj=(F6-F2), Qj=0  (value received)

=== Cycle 8 ===
Execute I6: ADD starts

... (continues)

=== Cycle 15 ===
Write I3: MUL complete (cycle 5+10)
  Mult2: Vj=(F2*F4), Qj=0  (value received)

=== Cycle 16 ===
Execute I5: DIV starts

=== Cycle 56 ===
Write I5: DIV complete (cycle 16+40)
```

---

## 6. Reorder Buffer (ROB)

### 6.1 Need for ROB

```
Problem: Precise exception handling impossible with OoO execution

Example:
I1: LD   R1, 0(R2)     ; May cause page fault
I2: ADD  R3, R4, R5    ; Completes before I1

If I2 completes first and I1 page faults:
- I2's result already written to R3
- State inconsistent after exception handling and restart

Solution: Reorder Buffer
- Temporarily store results
- Commit in program order
- Discard uncommitted results on exception
```

### 6.2 ROB Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Reorder Buffer                            │
├─────┬─────────┬──────────┬─────────┬────────┬──────────────┤
│Entry│  Busy   │  State   │  Dest   │ Value  │ Instruction  │
├─────┼─────────┼──────────┼─────────┼────────┼──────────────┤
│  1  │    1    │ Commit   │   F6    │  10.5  │ LD F6,34(R2) │
│  2  │    1    │ Commit   │   F2    │   5.0  │ LD F2,45(R3) │
│  3  │    1    │ Execute  │   F0    │   -    │ MUL F0,F2,F4 │
│  4  │    1    │ Write    │   F8    │   5.5  │ SUB F8,F6,F2 │
│  5  │    1    │ Issue    │  F10    │   -    │ DIV F10,F0,F6│
│  6  │    1    │ Issue    │   F6    │   -    │ ADD F6,F8,F2 │
│  7  │    0    │   -      │   -     │   -    │      -       │
│  8  │    0    │   -      │   -     │   -    │      -       │
└─────┴─────────┴──────────┴─────────┴────────┴──────────────┘
      ↑                                                ↑
    Head                                             Tail
  (Commit)                                         (Issue)

State:
- Issue: Issued, waiting to execute
- Execute: Executing
- Write: Execution complete, result recorded
- Commit: Ready to commit
```

### 6.3 Integrating ROB with Tomasulo

```
┌─────────────────────────────────────────────────────────────────┐
│            Modern Out-of-Order Processor                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Instruction Fetch/Decode                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Register Rename                          │ │
│  │              (RAT + Physical Register File)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Reorder Buffer (ROB) Allocation               │ │
│  │                  (Entry allocation, order tracking)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Issue Queue / Reservation Stations             │ │
│  │                  (Dependence wait, issue control)           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐        │
│  │ Execution  │      │ Execution  │      │   Memory   │        │
│  │  Unit 1    │      │  Unit 2    │      │   Unit     │        │
│  └─────┬──────┘      └─────┬──────┘      └─────┬──────┘        │
│        │                   │                   │                │
│        └───────────────────┼───────────────────┘                │
│                            │                                    │
│                            ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Write Back to ROB                              │ │
│  │              (Record results in ROB)                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                    │
│                            ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Commit (In-Order)                              │ │
│  │     (Update architectural registers from ROB Head)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Commit Process

```
Commit rules:
1. Only ROB Head instruction can commit
2. Instruction must be in complete state
3. All previous instructions must be committed
4. No exception present

On exception:
┌──────────────────────────────────────────────────┐
│  I1  │  I2  │  I3  │  I4  │  I5  │  I6  │       │
│ Done │ Done │ Done │ Exc! │ Done │ Done │       │
└──────────────────────────────────────────────────┘
   ↑
 Head

Exception at I4:
1. Commit I1, I2, I3 complete
2. Exception detected at I4
3. Discard I5, I6 results (not committed yet)
4. Jump to exception handler at I4's address
5. Precise exception state maintained
```

### 6.5 Branch Misprediction Recovery

```
Recovery on branch misprediction:

ROB state:
┌────────────────────────────────────────────────────┐
│  I1  │  I2  │  BR  │  I4  │  I5  │  I6  │         │
│ Done │ Done │Mis-P│ Done │ Done │ Done │         │
└────────────────────────────────────────────────────┘
   ↑           ↑
 Head        branch

Misprediction confirmed at BR:
1. Commit I1, I2
2. Detect misprediction at BR
3. Flush I4, I5, I6 (speculative execution results)
4. Restore RAT to BR checkpoint (checkpoint or ROB reverse walk)
5. Restart fetch at correct branch target

Recovery mechanisms:
- Checkpoint: Save RAT snapshot at each branch
- Gradual recovery: Walk ROB backwards to restore RAT
```

---

## 7. Modern Processor Implementations

### 7.1 Intel Core Architecture (Skylake)

```
┌─────────────────────────────────────────────────────────────────┐
│                Intel Skylake Microarchitecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Front-end (In-Order)                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Branch prediction: 32KB direction predictor, ~4K BTB    │ │
│  │  - L1 I-Cache: 32KB, 8-way                                 │ │
│  │  - Decode: 4-wide (complex instr → micro-ops)              │ │
│  │  - Micro-op Cache: ~1.5K micro-ops                         │ │
│  │  - Allocation: 4 micro-ops/cycle                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Out-of-Order Engine                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - ROB: 224 entries                                        │ │
│  │  - Scheduler: 97 entries                                   │ │
│  │  - Physical Registers: 180 integer + 168 vector            │ │
│  │  - Load Buffer: 72 entries                                 │ │
│  │  - Store Buffer: 56 entries                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Execution Units (8 Ports)                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Port 0: ALU, FMA, FP Div/Sqrt, Branch                     │ │
│  │  Port 1: ALU, FMA, AES                                     │ │
│  │  Port 2: Load AGU                                          │ │
│  │  Port 3: Load AGU                                          │ │
│  │  Port 4: Store Data                                        │ │
│  │  Port 5: ALU, Shuffle, Branch                              │ │
│  │  Port 6: ALU, Branch                                       │ │
│  │  Port 7: Store AGU                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Memory Subsystem                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - L1 D-Cache: 32KB, 8-way, 4 cycles                       │ │
│  │  - L2 Cache: 256KB, 4-way, 12 cycles                       │ │
│  │  - L3 Cache: 2MB/core, 16-way, ~40 cycles                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Performance Metrics                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Theoretical max: 6 micro-ops execution/cycle            │ │
│  │  - Actual IPC: 2-4 depending on workload                   │ │
│  │  - Pipeline depth: 14-19 stages                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 ARM Cortex-A77 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 ARM Cortex-A77 Microarchitecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Front-end                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Fetch: 4 instructions/cycle                             │ │
│  │  - Decode: 4-wide                                          │ │
│  │  - Macro-op Cache: 1.5K entries                            │ │
│  │  - Branch prediction: TAGE-based                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Out-of-Order Engine                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - ROB: 160 entries                                        │ │
│  │  - Dispatch: 6 micro-ops/cycle                             │ │
│  │  - Issue Queue: 120 entries                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Execution Units (10 pipelines)                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - 2x Branch                                               │ │
│  │  - 3x Integer ALU                                          │ │
│  │  - 2x Integer Multi-Cycle                                  │ │
│  │  - 2x FP/NEON                                              │ │
│  │  - 2x Load + 1x Store                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Performance Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│         Modern High-Performance Processor Comparison (2024)       │
├───────────────┬────────────────┬────────────────┬────────────────┤
│     Feature   │  Intel Golden  │   AMD Zen 4   │  Apple M2 P-core│
│               │    Cove        │               │               │
├───────────────┼────────────────┼────────────────┼────────────────┤
│  Decode Width │     6-wide     │    4-wide     │     8-wide    │
├───────────────┼────────────────┼────────────────┼────────────────┤
│  ROB Size     │    512 entry   │   320 entry   │   ~600 entry  │
├───────────────┼────────────────┼────────────────┼────────────────┤
│  Issue Width  │    12 ports    │    10 ports   │   ~13 ports   │
├───────────────┼────────────────┼────────────────┼────────────────┤
│  L1 I-Cache   │     32KB       │     32KB      │    192KB      │
├───────────────┼────────────────┼────────────────┼────────────────┤
│  L1 D-Cache   │     48KB       │     32KB      │    128KB      │
├───────────────┼────────────────┼────────────────┼────────────────┤
│  L2 Cache     │    1.25MB      │     1MB       │     16MB      │
├───────────────┼────────────────┼────────────────┼────────────────┤
│ Actual IPC    │     ~3-4       │     ~3-4      │     ~4-5      │
└───────────────┴────────────────┴────────────────┴────────────────┘
```

### 7.4 Limitations of ILP

```
Limiting factors for ILP exploitation:

1. True Data Dependence (RAW)
   - Cannot be resolved by renaming
   - Long dependency chains determine performance

2. Control Dependence
   - Branch prediction accuracy limits (~95-97%)
   - Pipeline flush on misprediction

3. Memory Dependence
   - Load-Store dependence detection difficulty
   - Constraints on memory instruction reordering

4. Window Size Limits
   - ROB, Issue Queue size constraints
   - Cannot exploit ILP far apart

Actual program ILP:

┌────────────────────────────────────────────────────┐
│  Program Type        │  Avg ILP    │  Limiting Factor │
├────────────────────────────────────────────────────┤
│  Integer (SPEC INT)  │   1.5-2.5  │  Dependency chains│
│  FP (SPEC FP)        │   2.0-4.0  │  Memory bandwidth │
│  Media processing    │   3.0-6.0  │  SIMD utilization │
│  Database            │   1.0-2.0  │  Branch prediction│
│  Web browser         │   1.5-2.5  │  Control flow     │
└────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

1. Classify the data dependence types:
   ```assembly
   I1: ADD R1, R2, R3
   I2: SUB R4, R1, R5
   I3: MUL R1, R6, R7
   I4: DIV R8, R1, R9
   ```

2. What is the theoretical maximum IPC in a 3-way superscalar processor?

3. Which dependence types can register renaming resolve?
   - (a) RAW
   - (b) WAR
   - (c) WAW
   - (d) Control

### Intermediate Problems

4. Apply register renaming to this program:
   ```assembly
   I1: ADD R1, R2, R3
   I2: MUL R4, R1, R5
   I3: ADD R1, R6, R7
   I4: SUB R8, R1, R4
   ```
   (Use physical registers starting from P10)

5. List 3 reasons why ROB is needed.

6. What is the role of CDB (Common Data Bus) in Tomasulo algorithm?

### Advanced Problems

7. Trace Reservation Station states for this Tomasulo execution:
   ```assembly
   I1: LD   F2, 0(R1)     ; 3 cycles
   I2: MUL  F4, F2, F0    ; 5 cycles
   I3: ADD  F6, F4, F2    ; 2 cycles
   ```
   (Initial: F0 = 2.0, RS: Load1, Mult1, Add1, trace cycles 1-10)

8. Explain ROB-based recovery process on branch misprediction.

9. Why does Intel Skylake have 224 ROB entries? What are the effects of making it larger or smaller?

<details>
<summary>Answers</summary>

1. Dependence classification:
   - I1→I2: RAW (R1)
   - I1→I3: WAW (R1)
   - I2→I3: WAR (R1 - I2 reads, I3 writes)
   - I3→I4: RAW (R1)

2. Theoretical max IPC of 3-way superscalar = 3

3. (b) WAR, (c) WAW
   - RAW is true dependence, cannot be resolved
   - Control resolved by branch prediction

4. Register renaming:
   ```assembly
   I1: ADD P10, P2, P3    ; R1 → P10
   I2: MUL P11, P10, P5   ; R4 → P11
   I3: ADD P12, P6, P7    ; R1 → P12 (new register)
   I4: SUB P13, P12, P11  ; R8 → P13
   ```
   I1 and I3 can now execute in parallel

5. ROB needed for:
   - Precise exception handling (in-order commit)
   - Branch misprediction recovery
   - Precise interrupt handling

6. CDB role:
   - Broadcast completed results to all RSs
   - Waiting instructions receive operands
   - Update register file

7. Tomasulo execution trace:
   ```
   Cycle 1: Issue I1 → Load1
   Cycle 2: Execute I1, Issue I2 → Mult1 (Qj=Load1)
   Cycle 3: Execute I1 continues, Issue I3 → Add1 (Qj=Mult1, Qk=Load1)
   Cycle 4: Write I1, Mult1: Vj update, Add1: Vk update
   Cycle 5: Execute I2 starts
   ...
   Cycle 9: Write I2, Add1: Vj update
   Cycle 10: Execute I3 starts
   Cycle 11-12: Execute I3 completes
   ```

8. Branch misprediction recovery:
   - Invalidate results after branch in ROB
   - Restore RAT to branch checkpoint state
   - Restart fetch at correct branch target
   - Pipeline flush

9. ROB size tradeoffs:
   - Larger: More ILP exploitation, tolerates longer memory latency
   - Smaller: Reduced area/power, faster recovery
   - 224 is balance point for memory latency (~100 cycles) and 6-wide issue

</details>

---

## Next Steps

- [14_Memory_Hierarchy.md](./14_Memory_Hierarchy.md) - Memory System Hierarchy and Locality Principles

---

## References

- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- [Intel Optimization Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [WikiChip - Microarchitectures](https://en.wikichip.org/wiki/WikiChip)
- [Agner Fog's microarchitecture](https://www.agner.org/optimize/)
- Tomasulo, R.M. "An Efficient Algorithm for Exploiting Multiple Arithmetic Units" (1967)
