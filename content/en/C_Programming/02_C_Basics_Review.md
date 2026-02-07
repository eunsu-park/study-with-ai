# C Language Basics Quick Review

> A summary of core C syntax for those with experience in other programming languages

## 1. Characteristics of C

### Comparison with Other Languages

| Feature | Python/JS | C |
|------|-----------|---|
| **Memory Management** | Automatic (GC) | Manual (malloc/free) |
| **Type System** | Dynamic typing | Static typing |
| **Execution** | Interpreter | Compiled |
| **Abstraction Level** | High | Low (close to hardware) |

### Why Learn C

- Systems programming (OS, drivers)
- Embedded systems
- Performance-critical applications
- Understanding foundations of other languages (Python, Ruby are written in C)

---

## 2. Basic Structure

```c
#include <stdio.h>    // Include header file (preprocessor directive)

// main function: Program entry point
int main(void) {
    printf("Hello, C!\n");
    return 0;         // 0 = normal exit
}
```

### Comparison with Python

```python
# Python
print("Hello, Python!")
```

```c
// C
#include <stdio.h>
int main(void) {
    printf("Hello, C!\n");
    return 0;
}
```

**C Characteristics:**
- Semicolon `;` required
- Curly braces `{}` for block delimiting
- Explicit main function
- Header file include required

---

## 3. Data Types

### Basic Data Types

```c
#include <stdio.h>

int main(void) {
    // Integer types
    char c = 'A';           // 1 byte (-128 ~ 127)
    short s = 100;          // 2 bytes
    int i = 1000;           // 4 bytes (typically)
    long l = 100000L;       // 4 or 8 bytes
    long long ll = 100000000000LL;  // 8 bytes

    // Unsigned integers
    unsigned int ui = 4000000000U;

    // Floating-point types
    float f = 3.14f;        // 4 bytes
    double d = 3.14159265;  // 8 bytes

    // Output
    printf("char: %c (%d)\n", c, c);  // A (65)
    printf("int: %d\n", i);
    printf("float: %f\n", f);
    printf("double: %.8f\n", d);

    return 0;
}
```

### Format Specifiers (printf)

| Specifier | Type | Example |
|--------|------|------|
| `%d` | int | `printf("%d", 42)` |
| `%u` | unsigned int | `printf("%u", 42)` |
| `%ld` | long | `printf("%ld", 42L)` |
| `%f` | float/double | `printf("%f", 3.14)` |
| `%c` | char | `printf("%c", 'A')` |
| `%s` | string | `printf("%s", "hello")` |
| `%p` | pointer address | `printf("%p", &x)` |
| `%x` | hexadecimal | `printf("%x", 255)` → ff |

### sizeof Operator

```c
printf("int size: %zu bytes\n", sizeof(int));
printf("double size: %zu bytes\n", sizeof(double));
printf("pointer size: %zu bytes\n", sizeof(int*));
```

---

## 4. Pointers (Core of C!)

### What is a Pointer?

**A variable that stores a memory address.**

```
Memory:
Address    Value
0x1000     42      ← int x = 42;
0x1004     0x1000  ← int *p = &x;  (stores address of x)
```

### Basic Syntax

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;      // p stores the address of x

    printf("Value of x: %d\n", x);        // 42
    printf("Address of x: %p\n", &x);     // 0x7fff...
    printf("Value of p (address): %p\n", p); // 0x7fff... (same address)
    printf("Value pointed by p: %d\n", *p);  // 42 (dereferencing)

    // Modify value through pointer
    *p = 100;
    printf("New value of x: %d\n", x);     // 100

    return 0;
}
```

### Pointer Operators

| Operator | Meaning | Example |
|--------|------|------|
| `&` | Address operator | `&x` → address of x |
| `*` | Dereference operator | `*p` → value pointed by p |

### Why Do We Need Pointers?

```c
// Problem: C passes values by copy (call by value)
void wrong_swap(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    // Original values unchanged!
}

// Solution: Pass addresses using pointers
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    // Original values changed!
}

int main(void) {
    int x = 10, y = 20;

    wrong_swap(x, y);
    printf("After wrong_swap: x=%d, y=%d\n", x, y);  // 10, 20 (no change)

    swap(&x, &y);
    printf("After swap: x=%d, y=%d\n", x, y);  // 20, 10

    return 0;
}
```

---

## 5. Arrays

### Basic Arrays

```c
#include <stdio.h>

int main(void) {
    // Array declaration and initialization
    int numbers[5] = {10, 20, 30, 40, 50};

    // Access
    printf("%d\n", numbers[0]);  // 10
    printf("%d\n", numbers[4]);  // 50

    // Size
    int size = sizeof(numbers) / sizeof(numbers[0]);
    printf("Array size: %d\n", size);  // 5

    // Iteration
    for (int i = 0; i < size; i++) {
        printf("numbers[%d] = %d\n", i, numbers[i]);
    }

    return 0;
}
```

### Relationship Between Arrays and Pointers

```c
int arr[5] = {1, 2, 3, 4, 5};

// Array name is the address of the first element
printf("%p\n", arr);      // Address of first element
printf("%p\n", &arr[0]);  // Same address

// Pointer arithmetic
int *p = arr;
printf("%d\n", *p);       // 1 (arr[0])
printf("%d\n", *(p + 1)); // 2 (arr[1])
printf("%d\n", *(p + 2)); // 3 (arr[2])

// arr[i] == *(arr + i)
```

### Strings (char arrays)

```c
#include <stdio.h>
#include <string.h>  // String functions

int main(void) {
    // String is char array + null terminator '\0'
    char str1[] = "Hello";        // Automatically adds '\0'
    char str2[10] = "World";
    char str3[] = {'H', 'i', '\0'};

    printf("%s\n", str1);         // Hello
    printf("Length: %zu\n", strlen(str1));  // 5

    // String copy
    char dest[20];
    strcpy(dest, str1);           // dest = "Hello"

    // String concatenation
    strcat(dest, " ");
    strcat(dest, str2);           // dest = "Hello World"
    printf("%s\n", dest);

    // String comparison
    if (strcmp(str1, "Hello") == 0) {
        printf("Equal!\n");
    }

    return 0;
}
```

---

## 6. Functions

### Basic Functions

```c
#include <stdio.h>

// Function declaration (prototype)
int add(int a, int b);
void greet(const char *name);

int main(void) {
    int result = add(3, 5);
    printf("3 + 5 = %d\n", result);

    greet("Alice");
    return 0;
}

// Function definition
int add(int a, int b) {
    return a + b;
}

void greet(const char *name) {
    printf("Hello, %s!\n", name);
}
```

### Passing Arrays to Functions

```c
// Arrays are passed as pointers (no size information)
void print_array(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Or use this notation (same meaning)
void print_array2(int arr[], int size) {
    // ...
}

int main(void) {
    int nums[] = {1, 2, 3, 4, 5};
    print_array(nums, 5);
    return 0;
}
```

---

## 7. Structures

### Basic Structure

```c
#include <stdio.h>
#include <string.h>

// Structure definition
struct Person {
    char name[50];
    int age;
    float height;
};

int main(void) {
    // Structure variable declaration and initialization
    struct Person p1 = {"John Doe", 25, 175.5};

    // Member access (. operator)
    printf("Name: %s\n", p1.name);
    printf("Age: %d\n", p1.age);

    // Modify member
    p1.age = 26;
    strcpy(p1.name, "Jane Smith");

    return 0;
}
```

### Simplify with typedef

```c
typedef struct {
    char name[50];
    int age;
} Person;  // Now use without 'struct' keyword

int main(void) {
    Person p1 = {"John Doe", 25};
    printf("%s\n", p1.name);
    return 0;
}
```

### Pointers and Structures

```c
typedef struct {
    char name[50];
    int age;
} Person;

void birthday(Person *p) {
    p->age++;  // Use -> operator for pointers
    // Same as (*p).age++;
}

int main(void) {
    Person p1 = {"John Doe", 25};

    birthday(&p1);
    printf("Age: %d\n", p1.age);  // 26

    // Access via pointer
    Person *ptr = &p1;
    printf("Name: %s\n", ptr->name);

    return 0;
}
```

---

## 8. Dynamic Memory Allocation

### malloc / free

```c
#include <stdio.h>
#include <stdlib.h>  // malloc, free

int main(void) {
    // Dynamically allocate one integer
    int *p = (int *)malloc(sizeof(int));
    if (p == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    *p = 42;
    printf("%d\n", *p);
    free(p);  // Free memory (required!)

    // Dynamically allocate array
    int n = 5;
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        return 1;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = i * 10;
    }

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);  // Must free array too!

    return 0;
}
```

### Beware of Memory Leaks

```c
// Bad example: Memory leak
void bad(void) {
    int *p = malloc(sizeof(int));
    *p = 42;
    // No free(p); → Memory leak!
}

// Good example
void good(void) {
    int *p = malloc(sizeof(int));
    if (p == NULL) return;
    *p = 42;
    // After use...
    free(p);
    p = NULL;  // Prevent dangling pointer
}
```

---

## 9. Header Files

### Header File Structure

```c
// utils.h
#ifndef UTILS_H      // include guard
#define UTILS_H

// Function declarations
int add(int a, int b);
int subtract(int a, int b);

// Structure definition
typedef struct {
    int x, y;
} Point;

#endif
```

```c
// utils.c
#include "utils.h"

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}
```

```c
// main.c
#include <stdio.h>
#include "utils.h"

int main(void) {
    printf("%d\n", add(3, 5));
    Point p = {10, 20};
    printf("(%d, %d)\n", p.x, p.y);
    return 0;
}
```

### Compilation

```bash
gcc main.c utils.c -o program
```

---

## 10. Key Differences Summary (Python → C)

| Python | C |
|--------|---|
| `print("Hello")` | `printf("Hello\n");` |
| `x = 10` | `int x = 10;` |
| `if x > 5:` | `if (x > 5) {` |
| `for i in range(5):` | `for (int i = 0; i < 5; i++) {` |
| `def func(x):` | `int func(int x) {` |
| `class Person:` | `struct Person {` |
| Automatic memory | `malloc()` / `free()` |
| `len(arr)` | `sizeof(arr)/sizeof(arr[0])` |

---

## Next Steps

Now let's build actual projects!

[03_Project_Calculator.md](./03_Project_Calculator.md) → Start the first project!
