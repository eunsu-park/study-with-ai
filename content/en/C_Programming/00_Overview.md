# C Programming Learning Guide

## Introduction

This folder contains materials for systematically learning C programming. From basic syntax to embedded systems, you can learn step-by-step through hands-on projects.

**Target Audience**: Programming beginners ~ intermediate learners

---

## Learning Roadmap

```
[Basics]         [Intermediate]      [Advanced]       [Embedded]
  │                │                   │                  │
  ▼                ▼                   ▼                  ▼
Setup ──────▶ Dynamic Array ───▶ Snake Game ────▶ Embedded Basics
  │           │                   │              │
  ▼           ▼                   ▼              ▼
Review ─────▶ Linked List ───▶ Mini Shell ───▶ Bit Operations
  │           │                   │              │
  ▼           ▼                   ▼              ▼
Calculator ─▶ File Encrypt ──▶ Multithreading ▶ GPIO Control
  │           │                                  │
  ▼           ▼                                  ▼
Guessing ───▶ Stack & Queue                  Serial Comm
  │           │
  ▼           ▼
Address Book ▶ Hash Table
```

---

## Prerequisites

- Basic computer usage skills
- Terminal/command-line experience (recommended)
- Text editor or IDE usage

---

## File List

| Filename | Difficulty | Key Content |
|--------|--------|----------|
| [01_Environment_Setup.md](./01_Environment_Setup.md) | ⭐ | Development environment setup, compiler installation |
| [02_C_Basics_Review.md](./02_C_Basics_Review.md) | ⭐ | Variables, data types, operators, control structures, functions |
| [03_Project_Calculator.md](./03_Project_Calculator.md) | ⭐ | Functions, switch-case, scanf |
| [04_Project_Number_Guessing.md](./04_Project_Number_Guessing.md) | ⭐ | Loops, random numbers, conditionals |
| [05_Project_Address_Book.md](./05_Project_Address_Book.md) | ⭐⭐ | Structures, arrays, file I/O |
| [06_Project_Dynamic_Array.md](./06_Project_Dynamic_Array.md) | ⭐⭐ | malloc, realloc, free |
| [07_Project_Linked_List.md](./07_Project_Linked_List.md) | ⭐⭐⭐ | Pointers, dynamic data structures |
| [08_Project_File_Encryption.md](./08_Project_File_Encryption.md) | ⭐⭐ | File processing, bit operations |
| [09_Project_Stack_Queue.md](./09_Project_Stack_Queue.md) | ⭐⭐ | Data structures, LIFO/FIFO |
| [10_Project_Hash_Table.md](./10_Project_Hash_Table.md) | ⭐⭐⭐ | Hashing, collision handling |
| [11_Project_Snake_Game.md](./11_Project_Snake_Game.md) | ⭐⭐⭐ | Terminal control, game loop |
| [12_Project_Mini_Shell.md](./12_Project_Mini_Shell.md) | ⭐⭐⭐⭐ | fork, exec, pipes |
| [13_Project_Multithreading.md](./13_Project_Multithreading.md) | ⭐⭐⭐⭐ | pthread, synchronization |
| [14_Embedded_Basics.md](./14_Embedded_Basics.md) | ⭐ | Arduino, GPIO basics |
| [15_Bit_Operations.md](./15_Bit_Operations.md) | ⭐⭐ | Bit masking, registers |
| [16_Project_GPIO_Control.md](./16_Project_GPIO_Control.md) | ⭐⭐ | LED, button, debouncing |
| [17_Project_Serial_Communication.md](./17_Project_Serial_Communication.md) | ⭐⭐ | UART, command parsing |
| [18_Debugging_Memory_Analysis.md](./18_Debugging_Memory_Analysis.md) | ⭐⭐⭐ | GDB, Valgrind, AddressSanitizer |
| [19_Advanced_Embedded_Protocols.md](./19_Advanced_Embedded_Protocols.md) | ⭐⭐⭐ | PWM, I2C, SPI, ADC |
| [20_Advanced_Pointers.md](./20_Advanced_Pointers.md) | ⭐⭐⭐ | Pointer arithmetic, multi-level pointers, function pointers, dynamic memory |
| [21_Network_Programming.md](./21_Network_Programming.md) | ⭐⭐⭐⭐ | TCP/UDP sockets, client-server, I/O multiplexing (select/poll) |
| [22_IPC_and_Signals.md](./22_IPC_and_Signals.md) | ⭐⭐⭐⭐ | Pipes, shared memory, message queues, signal handling |

---

## Recommended Learning Path

### Beginner (C Introduction)
1. Environment Setup → Basics Review → Calculator → Number Guessing → Address Book

### Intermediate (Data Structures & Pointers)
2. Advanced Pointers → Dynamic Array → Linked List → File Encryption → Stack & Queue → Hash Table

### Advanced (Systems Programming)
3. Snake Game → Mini Shell → Multithreading → Network Programming → IPC & Signals

### Embedded (Arduino)
4. Embedded Basics → Bit Operations → GPIO Control → Serial Communication → Advanced Embedded Protocols

### Debugging (Optional)
5. Debugging and Memory Analysis (recommended after completing all courses)

---

## Related Materials

- [Docker Learning](../Docker/00_Overview.md) - Development environment containerization
- [Git Learning](../Git/00_Overview.md) - Version control
