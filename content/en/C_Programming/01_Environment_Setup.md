# C Language Environment Setup

## 1. What You Need for C Development

| Component | Description |
|-----------|------|
| **Compiler** | Converts C code to executable (GCC, Clang) |
| **Text Editor/IDE** | For writing code (VS Code, Vim, etc.) |
| **Terminal** | For compiling and running |

---

## 2. Compiler Installation

### macOS

Xcode Command Line Tools includes Clang.

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
clang --version
gcc --version  # On macOS, gcc is an alias for clang
```

### Windows

**Method 1: MinGW-w64 (Recommended)**

1. Download and install [MSYS2](https://www.msys2.org/)
2. In MSYS2 terminal:
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```
3. Add to PATH environment variable: `C:\msys64\ucrt64\bin`

**Method 2: WSL (Windows Subsystem for Linux)**

```bash
# After installing WSL, in Ubuntu
sudo apt update
sudo apt install build-essential
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential

# Verify installation
gcc --version
```

---

## 3. VS Code Setup

### Install Extensions

1. **C/C++** (Microsoft) - Required
   - Syntax highlighting, IntelliSense, debugging

2. **Code Runner** (Optional)
   - Quick execution with keyboard shortcuts

### Configuration (settings.json)

```json
{
    "C_Cpp.default.compilerPath": "/usr/bin/gcc",
    "code-runner.executorMap": {
        "c": "cd $dir && gcc $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt"
    },
    "code-runner.runInTerminal": true
}
```

### tasks.json (Build Task)

`.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build C",
            "type": "shell",
            "command": "gcc",
            "args": [
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

Build with `Cmd+Shift+B` (macOS) or `Ctrl+Shift+B` (Windows)

---

## 4. Hello World

### Write Code

`hello.c`:
```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

### Compile and Run

```bash
# Compile
gcc hello.c -o hello

# Run
./hello          # macOS/Linux
hello.exe        # Windows

# Output: Hello, World!
```

### Compiler Options Explained

```bash
gcc hello.c -o hello
#   ↑        ↑   ↑
#   source   output  output filename

# Useful options
gcc -Wall hello.c -o hello      # Show all warnings
gcc -g hello.c -o hello         # Include debug info
gcc -O2 hello.c -o hello        # Optimization level 2
gcc -std=c11 hello.c -o hello   # Use C11 standard
```

### Recommended Compile Command

```bash
gcc -Wall -Wextra -std=c11 -g hello.c -o hello
```

---

## 5. Makefile Basics

As projects grow, use Makefile to automate builds.

### Basic Makefile

```makefile
# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

# Default target
all: hello

# Build hello executable
hello: hello.c
	$(CC) $(CFLAGS) hello.c -o hello

# Clean up
clean:
	rm -f hello

# .PHONY: Specify targets that aren't files
.PHONY: all clean
```

### Usage

```bash
make          # Build
make clean    # Clean up
```

### Multi-File Projects

```
project/
├── Makefile
├── main.c
├── utils.c
└── utils.h
```

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

SRCS = main.c utils.c
OBJS = $(SRCS:.c=.o)
TARGET = myprogram

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

## 6. Debugging Basics

### printf Debugging

```c
#include <stdio.h>

int main(void) {
    int x = 10;
    printf("DEBUG: x = %d\n", x);  // Check value

    x = x * 2;
    printf("DEBUG: x after *2 = %d\n", x);

    return 0;
}
```

### GDB (GNU Debugger)

```bash
# Compile with debug info
gcc -g hello.c -o hello

# Start GDB
gdb ./hello

# GDB commands
(gdb) break main      # Set breakpoint at main function
(gdb) run             # Run
(gdb) next            # Next line (n)
(gdb) step            # Step into function (s)
(gdb) print x         # Print variable x
(gdb) continue        # Continue execution (c)
(gdb) quit            # Quit (q)
```

### VS Code Debugging

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug C",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "cwd": "${workspaceFolder}",
            "preLaunchTask": "Build C",
            "MIMode": "lldb"
        }
    ]
}
```

---

## 7. Example Project Structure

```
my_c_project/
├── Makefile
├── src/
│   ├── main.c
│   └── utils.c
├── include/
│   └── utils.h
├── build/           # Compiled output
└── tests/
    └── test_utils.c
```

---

## Environment Verification Checklist

```bash
# 1. Check compiler
gcc --version

# 2. Create test file
echo '#include <stdio.h>
int main(void) { printf("OK\\n"); return 0; }' > test.c

# 3. Compile
gcc test.c -o test

# 4. Run
./test

# 5. Clean up
rm test test.c
```

If all steps succeed, your environment is ready!

---

## Next Steps

Let's quickly review C language core syntax in [02_C_Basics_Review.md](./02_C_Basics_Review.md)!
