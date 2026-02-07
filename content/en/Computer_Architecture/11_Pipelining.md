# Pipelining

## Overview

Pipelining is a technique that increases CPU throughput by executing multiple instructions simultaneously. Like the washer-dryer analogy, the next task starts before the current one finishes, improving overall efficiency.

---

## Table of Contents

1. [Pipelining Concept](#1-pipelining-concept)
2. [5-Stage Pipeline](#2-5-stage-pipeline)
3. [Pipeline Performance](#3-pipeline-performance)
4. [Pipeline Hazards](#4-pipeline-hazards)
5. [Hazard Resolution Techniques](#5-hazard-resolution-techniques)
6. [Practice Problems](#6-practice-problems)

---

## 1. Pipelining Concept

### Basic Idea

```
Non-pipelined (Sequential Execution):
┌─────┐     ┌─────┐     ┌─────┐
│ I1  │────▶│ I2  │────▶│ I3  │
└─────┘     └─────┘     └─────┘
  5ns         5ns         5ns      Total 15ns

Pipelined (Parallel Execution):
Time:  1ns   2ns   3ns   4ns   5ns   6ns   7ns
I1:   [IF]─[ID]─[EX]─[MEM]─[WB]
I2:        [IF]─[ID]─[EX]─[MEM]─[WB]
I3:             [IF]─[ID]─[EX]─[MEM]─[WB]
                                     Total 7ns
```

### Laundry Analogy

```
Non-pipelined:
Wash1 ──▶ Dry1 ──▶ Wash2 ──▶ Dry2 ──▶ Wash3 ──▶ Dry3

Pipelined:
Time:   1    2    3    4    5    6
Wash1   ■
Dry1        ■
Wash2       ■
Dry2             ■
Wash3            ■
Dry3                  ■

3x faster!
```

---

## 2. 5-Stage Pipeline

### MIPS 5-Stage Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    5-Stage Pipeline                         │
├───────┬───────┬───────┬───────┬───────┐                    │
│  IF   │  ID   │  EX   │  MEM  │  WB   │                    │
│(Fetch)│(Decode│(Exec) │(Memory│(Write │                    │
│       │       │       │Access)│ Back) │                    │
└───────┴───────┴───────┴───────┴───────┘                    │
└────────────────────────────────────────────────────────────┘
```

### Stage Descriptions

| Stage | Name | Operation |
|-------|------|-----------|
| IF | Instruction Fetch | Fetch instruction from memory at PC |
| ID | Instruction Decode | Decode instruction, read registers |
| EX | Execute | Perform ALU operation, address calculation |
| MEM | Memory Access | Read/write memory (load/store) |
| WB | Write Back | Store result to register |

### Pipeline Registers

```
┌─────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐
│ IF  │──▶│ IF/ID   │──▶│ ID/EX   │──▶│ EX/MEM   │──▶│ MEM/WB  │
└─────┘   │Register │   │Register │   │Register  │   │Register │
          └─────────┘   └─────────┘   └──────────┘   └─────────┘

Pipeline Registers: Temporarily store data between stages
- Data is passed to the next stage every clock cycle
```

### Pipeline Usage by Instruction Type

```
Stage usage by instruction type:

R-type (add, sub):
IF ─▶ ID ─▶ EX ─▶ --- ─▶ WB

Load (lw):
IF ─▶ ID ─▶ EX ─▶ MEM ─▶ WB

Store (sw):
IF ─▶ ID ─▶ EX ─▶ MEM ─▶ ---

Branch (beq):
IF ─▶ ID ─▶ EX ─▶ --- ─▶ ---
```

---

## 3. Pipeline Performance

### Ideal Speedup

```
Speedup = Number of pipeline stages (ideal)

5-stage pipeline → Maximum 5x speedup

In practice:
- Pipeline fill/drain time
- Stalls due to hazards
- Stage imbalance
```

### Throughput Calculation

```
Throughput = Number of instructions / Time

Non-pipelined:
- 1 instruction / 5 cycles

Pipelined (ideal):
- 1 instruction / 1 cycle (full pipeline state)

CPI (Cycles Per Instruction):
- Ideal: CPI = 1
- Actual: CPI = 1 + stall cycles
```

### Example: Executing 100 Instructions

```
Non-pipelined:
Time = 100 × 5 = 500 cycles

5-stage pipeline:
Time = 5 + (100 - 1) = 104 cycles
      ↑first instr.  ↑remaining instructions

Speedup = 500 / 104 ≈ 4.8x
```

---

## 4. Pipeline Hazards

### Hazard Types

```
┌─────────────────────────────────────────────────────────┐
│                   Pipeline Hazards                       │
├─────────────────┬─────────────────┬─────────────────────┤
│   Structural    │      Data       │      Control        │
│    Hazard       │     Hazard      │      Hazard         │
├─────────────────┼─────────────────┼─────────────────────┤
│ Hardware        │ Problems due    │ Problems due        │
│ resource        │ to data         │ to branch           │
│ conflicts       │ dependencies    │ instructions        │
└─────────────────┴─────────────────┴─────────────────────┘
```

### 4.1 Structural Hazards

```
Problem: Trying to use the same hardware resource simultaneously

Example: Single memory usage
Cycle 4:
- Instruction 1: MEM stage (data memory access)
- Instruction 4: IF stage (instruction memory access)

    I1: IF─ID─EX─MEM─WB
    I4:          IF ← Conflict!

Solution: Harvard architecture (separate instruction/data memory)
```

### 4.2 Data Hazards

```
Three types:

1. RAW (Read After Write) - Most common
   add $s0, $t0, $t1    # Write to $s0
   sub $t2, $s0, $t3    # Read $s0 ← Not written yet!

2. WAR (Write After Read)
   sub $t2, $s0, $t3    # Read $s0
   add $s0, $t0, $t1    # Write to $s0

3. WAW (Write After Write)
   add $s0, $t0, $t1    # Write to $s0
   sub $s0, $t2, $t3    # Write to $s0
```

### RAW Hazard Example

```
add $s0, $t0, $t1
sub $t2, $s0, $t3

Time:    1    2    3    4    5    6    7
add:    IF   ID   EX  MEM  [WB] ← $s0 written
sub:         IF   ID  [EX] ← $s0 needed!
                       ↑
                   Problem occurs!

$s0 is written in cycle 5,
but sub needs $s0 in cycle 4
```

### 4.3 Control Hazards

```
beq $t0, $t1, target    # Branch decision
add $t2, $t3, $t4       # Should this execute?
sub $t5, $t6, $t7       # Should this execute?

Time:    1    2    3    4    5
beq:    IF   ID  [EX] ← Branch decided
add:         IF   ID   ← Wrong instruction?
sub:              IF   ← Wrong instruction?

Next instructions already entered pipeline before branch decision
```

---

## 5. Hazard Resolution Techniques

### 5.1 Stalling

```
Pause pipeline by inserting bubbles (NOPs)

add $s0, $t0, $t1
sub $t2, $s0, $t3

Time:    1    2    3    4    5    6    7    8    9
add:    IF   ID   EX  MEM  WB
        --- stall ---
        --- stall ---
sub:              IF   ID   EX  MEM  WB

2 cycle stall occurs → Performance degradation
```

### 5.2 Forwarding/Bypassing

```
Pass ALU result directly to next instruction

add $s0, $t0, $t1
sub $t2, $s0, $t3

Time:    1    2    3    4    5    6
add:    IF   ID  [EX] MEM  WB
                  │
                  └─────▶ Forwarding
sub:         IF   ID  [EX] MEM  WB
                       ↑
                    Use $s0 value

Execution possible without stalls!
```

### Forwarding Paths

```
┌─────────────────────────────────────────────────────────┐
│                    Forwarding Unit                       │
│                                                         │
│   EX/MEM.ALUResult ───────────────┐                     │
│                                   ▼                     │
│   MEM/WB.ALUResult ─────────────▶ MUX ──▶ ALU input     │
│                                   ▲                     │
│   ID/EX.RegisterRs ───────────────┘                     │
└─────────────────────────────────────────────────────────┘

Forwarding conditions:
1. EX/MEM.RegisterRd == ID/EX.RegisterRs
2. MEM/WB.RegisterRd == ID/EX.RegisterRs
```

### 5.3 Load-Use Hazard

```
Cases that cannot be resolved by forwarding:

lw  $s0, 0($t0)     # Load from memory
add $t2, $s0, $t3   # Use immediately

Time:    1    2    3    4    5    6    7
lw:     IF   ID   EX  [MEM] WB
                       │
                       └───▶ Data available
add:         IF   ID  stall [EX] MEM  WB
                   ↑
              Need data but not available yet

1 cycle stall mandatory (Load-Use Stall)
```

### 5.4 Branch Prediction

```
Static Prediction:
- Always Not Taken: Predict no branch
- Always Taken: Predict branch taken
- BTFN: Backward Taken, Forward Not Taken

Dynamic Prediction:
- Predict based on branch history
- Covered in detail in next lesson
```

### 5.5 Delayed Branch

```
Place always-executed instruction in slot after branch instruction

beq $t0, $t1, target
add $t2, $t3, $t4    # Delay slot (always executed)
...
target:
sub $t5, $t6, $t7

Compiler places branch-independent instruction in delay slot
```

---

## 6. Practice Problems

### Basic Problems

1. What are the names and roles of each stage in a 5-stage pipeline?

2. How many cycles are needed to execute 100 instructions in a 5-stage pipeline? (Assume no hazards)

3. Which of the following is NOT a type of data hazard?
   - (a) RAW
   - (b) RAR
   - (c) WAR
   - (d) WAW

### Hazard Analysis

4. Find the data hazards in the following code:
```assembly
add $s0, $t0, $t1
sub $s1, $s0, $t2
and $s2, $s0, $s1
```

5. Which hazard cannot be resolved by forwarding?
```assembly
lw  $s0, 0($t0)
add $t1, $s0, $t2
```

### Performance Calculation

6. If 30% of 1000 instructions are branches, branch misprediction rate is 20%, and misprediction penalty is 3 cycles, what is the CPI?

<details>
<summary>Answers</summary>

1. IF (Fetch), ID (Decode), EX (Execute), MEM (Memory Access), WB (Write Back)

2. 5 + (100 - 1) = 104 cycles

3. (b) RAR - Read After Read is not a hazard

4.
- add → sub: RAW on $s0
- add → and: RAW on $s0
- sub → and: RAW on $s1

5. Load-Use hazard. Data is only available after MEM stage of lw, so 1 cycle stall is required

6.
- Number of branch instructions: 1000 × 0.3 = 300
- Mispredictions: 300 × 0.2 = 60
- Penalty cycles: 60 × 3 = 180
- CPI = 1 + 180/1000 = 1.18

</details>

---

## Next Steps

- [12_Branch_Prediction.md](./12_Branch_Prediction.md) - Dynamic Branch Prediction Techniques

---

## References

- Computer Organization and Design, Chapter 4 (Patterson & Hennessy)
- [Pipeline Visualization](https://www.youtube.com/watch?v=eVRdfl4zxfI)
- [MIPS Pipeline Simulator](http://www.cs.umd.edu/~meesh/411/mips-pipe/)
