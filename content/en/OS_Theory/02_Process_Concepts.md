# Process Concepts

## Overview

A process is a program in execution. This lesson covers process memory structure, Process Control Block (PCB), process state transitions, and context switching.

---

## Table of Contents

1. [What is a Process?](#1-what-is-a-process)
2. [Process Memory Structure](#2-process-memory-structure)
3. [Process Control Block (PCB)](#3-process-control-block-pcb)
4. [Process State Transitions](#4-process-state-transitions)
5. [Context Switch](#5-context-switch)
6. [Process Creation and Termination](#6-process-creation-and-termination)
7. [Practice Problems](#7-practice-problems)

---

## 1. What is a Process?

### Program vs Process

```
┌────────────────────────────────────────────────────────┐
│                  Program vs Process                     │
├────────────────────┬───────────────────────────────────┤
│      Program       │              Process               │
├────────────────────┼───────────────────────────────────┤
│ Static entity      │ Dynamic entity                    │
│ Stored on disk     │ Loaded in memory                  │
│ Executable file    │ Executing file                    │
│ Passive            │ Active                            │
│ Doesn't change     │ State constantly changes          │
└────────────────────┴───────────────────────────────────┘

Program ──(load)──▶ Process
          Load into
          memory
```

### Components of a Process

```
Process = Code + Data + Stack + Heap + PCB

┌─────────────────────────────────────┐
│              Process                 │
├─────────────────────────────────────┤
│  ┌───────────────────────────────┐  │
│  │    Text (Code) Section         │  │  Instructions to execute
│  ├───────────────────────────────┤  │
│  │    Data Section                │  │  Global/static variables
│  ├───────────────────────────────┤  │
│  │    Heap                        │  │  Dynamic allocation
│  ├───────────────────────────────┤  │
│  │    Stack                       │  │  Local vars, function calls
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │    PCB (stored in kernel)      │  │  Process metadata
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 2. Process Memory Structure

### Memory Layout

```
High address (0xFFFFFFFF)
┌─────────────────────────────────────┐
│             Kernel Space             │  OS only (no user access)
├─────────────────────────────────────┤ ← 0xC0000000 (Linux 32-bit)
│                                     │
│              Stack                   │  Local variables, parameters
│              ↓ Grows downward        │  Return addresses
│                                     │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│                                     │
│              ↑ Grows upward          │
│              Heap                    │  malloc, new
│                                     │
├─────────────────────────────────────┤
│              BSS                     │  Uninitialized global/static
├─────────────────────────────────────┤
│              Data                    │  Initialized global/static
├─────────────────────────────────────┤
│              Text (Code)             │  Program code (read-only)
└─────────────────────────────────────┘
Low address (0x00000000)
```

### Section Details

```c
#include <stdio.h>
#include <stdlib.h>

// BSS section: uninitialized global
int uninit_global;

// Data section: initialized global
int init_global = 42;

// Data section: static variable
static int static_var = 100;

void example_function(int param) {    // param: stack
    int local_var = 10;               // stack
    static int func_static = 0;       // Data section
    int *heap_ptr;

    heap_ptr = malloc(sizeof(int));   // Allocate on heap
    *heap_ptr = 20;

    printf("local: %d, heap: %d\n", local_var, *heap_ptr);

    free(heap_ptr);                   // Free heap memory
}

// Text section: this code itself
int main() {
    example_function(5);
    return 0;
}
```

### Memory Region Characteristics

```
┌──────────┬──────────┬──────────┬────────────────────────┐
│  Section  │ Read     │ Write    │         Purpose         │
├──────────┼──────────┼──────────┼────────────────────────┤
│ Text     │   O      │   X      │ Program code            │
│ Data     │   O      │   O      │ Initialized global/static│
│ BSS      │   O      │   O      │ Uninitialized global/static│
│ Heap     │   O      │   O      │ Dynamic allocation      │
│ Stack    │   O      │   O      │ Local vars, function calls│
└──────────┴──────────┴──────────┴────────────────────────┘
```

### Stack Frame

```
Function call stack structure:

int add(int a, int b) {
    int result = a + b;
    return result;
}

int main() {
    int x = add(3, 5);
    return 0;
}

┌─────────────────────────────┐ ← High address
│        ...                  │
├─────────────────────────────┤
│    main() stack frame       │
│  ┌───────────────────────┐  │
│  │ x (local variable)    │  │
│  │ Previous frame pointer│  │
│  │ Return address        │  │
│  └───────────────────────┘  │
├─────────────────────────────┤
│    add() stack frame        │
│  ┌───────────────────────┐  │
│  │ result (local var)    │  │
│  │ Previous frame pointer│  │
│  │ Return address        │  │
│  │ b = 5 (parameter)     │  │
│  │ a = 3 (parameter)     │  │
│  └───────────────────────┘  │
├─────────────────────────────┤ ← Stack Pointer (SP)
│        ...                  │
└─────────────────────────────┘ ← Low address
```

---

## 3. Process Control Block (PCB)

### What is PCB?

```
PCB (Process Control Block) = Data structure containing all info to manage a process
                            = Maintained by kernel
                            = Stored in process table

┌───────────────────────────────────────────────────────┐
│                    PCB Structure                       │
├───────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Process Identifier (PID)                         │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Process State (Ready, Running, Waiting...)       │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Program Counter (PC) - next instruction address │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ CPU Registers (general purpose, SP, flags...)    │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ CPU Scheduling Info (priority, scheduling queue) │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Memory Management Info (page table, segment table)│  │
│  ├─────────────────────────────────────────────────┤  │
│  │ Accounting Info (CPU time used, start time...)   │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ I/O Status Info (open files, I/O devices...)     │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Linux task_struct (Simplified)

```c
// Linux kernel process structure (simplified)
struct task_struct {
    // Process identification
    pid_t pid;                    // Process ID
    pid_t tgid;                   // Thread group ID

    // Process state
    volatile long state;          // TASK_RUNNING, TASK_INTERRUPTIBLE...

    // Scheduling info
    int prio;                     // Dynamic priority
    int static_prio;              // Static priority
    struct sched_entity se;       // Scheduling entity

    // CPU context
    struct thread_struct thread;  // CPU register state

    // Memory management
    struct mm_struct *mm;         // Memory descriptor

    // File system
    struct files_struct *files;   // Open file table
    struct fs_struct *fs;         // File system info

    // Process relationships
    struct task_struct *parent;   // Parent process
    struct list_head children;    // Children processes list
    struct list_head sibling;     // Sibling processes list

    // Signals
    struct signal_struct *signal;

    // Timing info
    u64 utime, stime;            // User/system CPU time
    u64 start_time;              // Start time
};
```

### Process Table

```
┌─────────────────────────────────────────────────────────┐
│                   Process Table                          │
├─────┬──────────────────────────────────────────────────┤
│ PID │                    PCB                            │
├─────┼──────────────────────────────────────────────────┤
│  1  │ init: state=Running, priority=20, mem=4MB...     │
├─────┼──────────────────────────────────────────────────┤
│  2  │ kthreadd: state=Sleeping, priority=10...         │
├─────┼──────────────────────────────────────────────────┤
│ 100 │ bash: state=Ready, priority=20, mem=8MB...       │
├─────┼──────────────────────────────────────────────────┤
│ 101 │ vim: state=Waiting, priority=20, mem=12MB...     │
├─────┼──────────────────────────────────────────────────┤
│ ... │ ...                                              │
└─────┴──────────────────────────────────────────────────┘
```

---

## 4. Process State Transitions

### 5-State Model

```
                        New
                            │
                            │ Admitted
                            ▼
         ┌──────────────┐ Dispatch ┌──────────────┐
         │              │─────────▶│              │
         │   Ready      │          │   Running    │──────┐ Exit
         │              │◀─────────│              │      │
         │              │ Interrupt │              │      │
         └──────────────┘(Timeout)  └──────────────┘      │
                ▲                         │              ▼
                │                         │         ┌──────────┐
                │    I/O or               │         │Terminated│
                │   Event Complete        │         │          │
                │                         │         └──────────┘
                │                         │ I/O or
                │                         │ Event Wait
                │                         ▼
                │              ┌──────────────┐
                └──────────────│   Waiting    │
                               │              │
                               └──────────────┘
```

### State Descriptions

```
┌────────────────┬────────────────────────────────────────┐
│      State      │                Description             │
├────────────────┼────────────────────────────────────────┤
│ New            │ Process being created                  │
├────────────────┼────────────────────────────────────────┤
│ Ready          │ Waiting for CPU assignment             │
│                │ Ready to execute                       │
├────────────────┼────────────────────────────────────────┤
│ Running        │ Executing instructions on CPU          │
│                │ Only one process at a time (single CPU)│
├────────────────┼────────────────────────────────────────┤
│ Waiting        │ Waiting for I/O or event completion    │
│                │ Transitions to Ready when I/O completes│
├────────────────┼────────────────────────────────────────┤
│ Terminated     │ Execution completed, releasing resources│
│                │                                        │
└────────────────┴────────────────────────────────────────┘
```

### State Transition Conditions

```
┌─────────────────┬─────────────────────────────────────────┐
│   Transition     │                Condition                │
├─────────────────┼─────────────────────────────────────────┤
│ New → Ready     │ OS admits process                       │
├─────────────────┼─────────────────────────────────────────┤
│ Ready → Running │ Scheduler assigns CPU (dispatch)        │
├─────────────────┼─────────────────────────────────────────┤
│ Running → Ready │ Time slice expired (timeout)            │
│                 │ Higher priority process arrives (preempt)│
├─────────────────┼─────────────────────────────────────────┤
│ Running → Wait  │ I/O request, event wait                 │
├─────────────────┼─────────────────────────────────────────┤
│ Wait → Ready    │ I/O complete, event occurs              │
├─────────────────┼─────────────────────────────────────────┤
│ Running → Term  │ exit() call, normal/abnormal termination│
└─────────────────┴─────────────────────────────────────────┘
```

### 7-State Model (Including Swapping)

```
                           ┌──────────────────────────────────┐
                           │                                  │
                           │                 Swap out          │
                           ▼                   │              │
┌─────────┐           ┌─────────┐          ┌───┴─────┐        │
│  New    │──────────▶│  Ready  │◀────────▶│ Ready   │        │
│         │           │         │  Swap in  │ Suspend │        │
└─────────┘           └─────────┘          └─────────┘        │
                           │                                  │
                      Dispatch                                │
                           │                                  │
                           ▼                                  │
┌─────────┐           ┌─────────┐                             │
│  Term   │◀──────────│ Running │                             │
│         │           │         │                             │
└─────────┘           └─────────┘                             │
                           │                                  │
                      I/O Request                             │
                           │                                  │
                           ▼                   Swap out       │
                      ┌─────────┐          ┌─────────┐        │
                      │ Waiting │◀────────▶│ Waiting │────────┘
                      │         │  Swap in  │ Suspend │
                      └─────────┘          └─────────┘

Suspend state: Process swapped out from memory to disk
```

---

## 5. Context Switch

### What is a Context Switch?

```
Context Switch = Process of switching CPU to another process
               = Save current process state, restore new process state

┌────────────────────────────────────────────────────────────┐
│                    Context Switch Process                   │
└────────────────────────────────────────────────────────────┘

Process P0              Operating System              Process P1
    │                      │                      │
    │  Executing            │                      │  Waiting
    │                      │                      │
    │──Interrupt/Syscall──▶│                      │
    │                      │                      │
    │                  Save state to PCB0         │
    │                      │                      │
    │                  Restore state from PCB1    │
    │                      │                      │
    │                      │──────────────────────▶│
    │  Waiting             │                      │  Executing
    │                      │                      │
    │                      │◀──Interrupt/Syscall──│
    │                      │                      │
    │                  Save state to PCB1         │
    │                      │                      │
    │                  Restore state from PCB0    │
    │                      │                      │
    │◀─────────────────────│                      │
    │  Executing           │                      │  Waiting
```

### Information Saved/Restored in Context Switch

```
┌────────────────────────────────────────────────────────┐
│               Context                                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  CPU Registers:                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • Program Counter (PC)                            │  │
│  │ • Stack Pointer (SP)                             │  │
│  │ • Base Pointer (BP)                              │  │
│  │ • General Purpose Registers (RAX, RBX, RCX, RDX...)│  │
│  │ • Status Register (FLAGS)                        │  │
│  │ • Floating Point Registers                       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Memory Management Info:                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • Page Table Base Register                       │  │
│  │ • Segment Registers                              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Context Switch Cost

```
┌────────────────────────────────────────────────────────┐
│               Context Switch Cost                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Direct cost:                                          │
│  • Register save/restore: ~hundreds of nanoseconds     │
│  • Kernel mode transition: ~hundreds of nanoseconds    │
│                                                        │
│  Indirect cost (larger):                               │
│  • TLB flush: hundreds~thousands of cycles             │
│  • Cache miss increase (cache pollution)               │
│  • Pipeline flush                                      │
│                                                        │
│  Typical total cost: 1~10 microseconds                 │
│                                                        │
└────────────────────────────────────────────────────────┘

Timeline view of context switch:

P0 exec  │ Context Switch │ P1 exec  │ Context Switch │ P0 exec
━━━━━━━━│     Overhead    │━━━━━━━━│     Overhead    │━━━━━━━
        │← ~1-10 μs →│        │← ~1-10 μs →│
                 ↑ No useful work during this time
```

---

## 6. Process Creation and Termination

### Process Creation with fork()

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid;
    int x = 10;

    printf("Parent process starting, PID: %d\n", getpid());

    pid = fork();  // Fork point

    if (pid < 0) {
        // fork failed
        perror("fork failed");
        return 1;
    }
    else if (pid == 0) {
        // Child process
        printf("Child: PID=%d, Parent PID=%d\n", getpid(), getppid());
        x = x + 10;
        printf("Child: x = %d\n", x);
    }
    else {
        // Parent process
        printf("Parent: PID=%d, Child PID=%d\n", getpid(), pid);
        wait(NULL);  // Wait for child termination
        printf("Parent: x = %d\n", x);  // Still 10
    }

    return 0;
}

/*
Output:
Parent process starting, PID: 1234
Parent: PID=1234, Child PID=1235
Child: PID=1235, Parent PID=1234
Child: x = 20
Parent: x = 10
*/
```

### How fork() Works

```
Before fork():
┌─────────────────────────────────┐
│        Parent Process (PID: 100) │
│  ┌─────────────────────────┐    │
│  │ x = 10                  │    │
│  │ Code/Data/Stack/Heap    │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘

After fork():
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│        Parent Process (PID: 100) │    │        Child Process (PID: 101)  │
│  ┌─────────────────────────┐    │    │  ┌─────────────────────────┐    │
│  │ x = 10                  │    │    │  │ x = 10 (copied)         │    │
│  │ Code/Data/Stack/Heap    │    │    │  │ Code/Data/Stack/Heap    │    │
│  │ fork() returns: 101     │    │    │  │ fork() returns: 0       │    │
│  └─────────────────────────┘    │    │  └─────────────────────────┘    │
└─────────────────────────────────┘    └─────────────────────────────────┘
          │                                       │
          │  Two processes are independent        │
          │  (separate memory spaces)             │
          ▼                                       ▼
```

### Program Execution with exec()

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: execute ls command
        printf("Child: executing ls\n");

        // execl: exec with list of arguments
        execl("/bin/ls", "ls", "-l", NULL);

        // If exec succeeds, this code won't execute
        perror("exec failed");
        return 1;
    }
    else {
        // Parent process
        wait(NULL);
        printf("Parent: child terminated\n");
    }

    return 0;
}
```

### How exec() Works

```
Before exec():
┌─────────────────────────────────┐
│        Child Process             │
│  ┌─────────────────────────┐    │
│  │ Original program code    │    │
│  │ Original data            │    │
│  │ Original stack/heap      │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘

After exec("/bin/ls"):
┌─────────────────────────────────┐
│        Child Process             │
│  ┌─────────────────────────┐    │
│  │ ls program code          │    │  ← Completely replaced
│  │ ls data                  │    │    with new program
│  │ New stack/heap           │    │
│  └─────────────────────────┘    │
│                                 │
│  PID remains the same            │
└─────────────────────────────────┘
```

### Process Termination

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) {
        printf("Child process terminating\n");
        exit(42);  // Exit with status code 42
    }
    else {
        int status;
        wait(&status);  // Collect child's exit status

        if (WIFEXITED(status)) {
            printf("Child exit code: %d\n", WEXITSTATUS(status));
        }
    }

    return 0;
}

/*
Output:
Child process terminating
Child exit code: 42
*/
```

### Zombie and Orphan Processes

```
Zombie Process:
- Child has terminated but parent hasn't called wait()
- PCB remains (stores exit status)
- Resources released but process table entry maintained

┌──────────────────────────────────────────────┐
│  Parent Process (PID: 100)                   │
│  - Hasn't called wait()                      │
└──────────────────────────────────────────────┘
          │
          │ (Relationship maintained)
          ▼
┌──────────────────────────────────────────────┐
│  Zombie Process (PID: 101)                   │
│  - State: Z (Zombie)                         │
│  - Code/Data/Stack released                  │
│  - Only PCB remains                          │
└──────────────────────────────────────────────┘


Orphan Process:
- Parent terminated before child
- init (PID 1) or systemd becomes new parent

┌──────────────────────────────────────────────┐
│  init (PID: 1)                               │
│  - New parent of orphan processes            │
│  - Periodically calls wait()                 │
└──────────────────────────────────────────────┘
          │
          │ (Adoption)
          ▼
┌──────────────────────────────────────────────┐
│  Orphan Process (PID: 102)                   │
│  - Original parent (PID: 100) terminated     │
│  - PPID changed to 1                         │
└──────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Problem 1: Memory Region Identification

Identify the memory region where each variable is stored.

```c
int global_var = 100;        // (   )
int uninitialized;           // (   )
const char* str = "hello";   // (   )

void func() {
    int local = 10;          // (   )
    static int stat = 20;    // (   )
    int* ptr = malloc(4);    // ptr: (   ), *ptr: (   )
}
```

Options: Text, Data, BSS, Stack, Heap

<details>
<summary>Show Answer</summary>

```
int global_var = 100;        // (Data)
int uninitialized;           // (BSS)
const char* str = "hello";   // str: Data, "hello": Text (read-only)

void func() {
    int local = 10;          // (Stack)
    static int stat = 20;    // (Data)
    int* ptr = malloc(4);    // ptr: (Stack), *ptr: (Heap)
}
```

</details>

### Problem 2: Process State Transitions

Explain the process state transitions in the following situations.

1. Process A is using CPU, time slice expires
2. Process B requests file read
3. Process C's file read completes
4. Scheduler selects process D

<details>
<summary>Show Answer</summary>

1. A: Running → Ready (timeout/interrupt)
2. B: Running → Waiting (I/O request)
3. C: Waiting → Ready (I/O complete)
4. D: Ready → Running (dispatch)

</details>

### Problem 3: fork() Output Prediction

Predict the output of the following code.

```c
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("A\n");
    fork();
    printf("B\n");
    fork();
    printf("C\n");
    return 0;
}
```

<details>
<summary>Show Answer</summary>

```
A       (1 output - original process)
B       (2 outputs - 2 processes after first fork)
B
C       (4 outputs - 4 processes after second fork)
C
C
C

Total: A 1 time, B 2 times, C 4 times

Process branching:
        main
          │
    ┌─────┴─────┐
    │  fork()   │
    │           │
  main        child1
    │           │
 ┌──┴──┐     ┌──┴──┐
 │fork()│    │fork()│
 │     │    │     │
main  c2   c1    c3

4 processes each print "C"
```

</details>

### Problem 4: PCB Information

Explain the changes to PCB information in the following situations.

1. When a process transitions from Running to Waiting
2. When a context switch occurs

<details>
<summary>Show Answer</summary>

1. Running → Waiting transition:
   - PCB's state field changes from Running to Waiting
   - I/O status info records the waiting I/O operation
   - Process moves to waiting queue

2. Context switch:
   - Save current process's PC, registers, stack pointer to PCB
   - Restore PC, registers, stack pointer from new process's PCB
   - Update memory management info (page table)
   - Change new process's state to Running

</details>

### Problem 5: Context Switch Cost

Describe two direct costs and two indirect costs of context switching.

<details>
<summary>Show Answer</summary>

**Direct costs:**
1. Register save/restore: Time to save current process's registers to PCB and restore new process's registers
2. Kernel mode transition: Overhead of transitioning from user mode to kernel mode and back

**Indirect costs:**
1. Cache pollution: New process's data not in cache, increasing cache misses
2. TLB flush: New process uses different virtual address space, invalidating TLB entries

</details>

---

## Next Steps

- [03_Threads_and_Multithreading.md](./03_Threads_and_Multithreading.md) - Thread concepts and multithreading models

---

## References

- [OSTEP - Processes](https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf)
- [Linux man pages - fork](https://man7.org/linux/man-pages/man2/fork.2.html)
- [Linux Kernel - task_struct](https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h)
