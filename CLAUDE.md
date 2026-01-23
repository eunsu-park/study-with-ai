# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose & Roadmap

### 목적 (Purpose)
개인 학습을 위한 체계적인 기술 자료 모음. 각 주제별로 입문부터 심화까지 단계별 학습이 가능하도록 구성.

### 현재 상태 (Current Status)
- **C_Programming**: 완료 (17개 레슨 + Overview)
- **Docker**: 완료 (6개 레슨 + Overview)
- **Git**: 완료 (7개 레슨 + Overview)
- **Linux**: 완료 (12개 레슨 + Overview) - 서버 관리 포함
- **PostgreSQL**: 완료 (13개 레슨 + Overview)
- **Spanish**: 완료 (5개 파일 + Overview)

### 향후 계획 (Future Plans)
- [ ] Python 기초~고급 자료 추가
- [ ] 알고리즘/자료구조 심화 자료
- [ ] 클라우드 (AWS/GCP) 학습 자료

### 업데이트 이력 (Change Log)
- **2024-01**: C Programming, Docker, Git, Spanish 자료 생성
- **2024-01-23**: PostgreSQL 자료 추가, 각 폴더에 00_Overview.md 추가, C_Programming 파일 넘버링 수정 (00→01 시작)
- **2026-01-23**: Linux 자료 추가 (12개 레슨, 서버 관리 포함, Ubuntu/CentOS 병렬 안내)
- **2026-01-23**: Spanish 파일명 정리 (01_문법~05_기타_품사), 어휘 170개+ 추가 (비즈니스/여행/관용어)

---

## Repository Overview

This is a personal study repository containing educational materials in Korean and English. Content is self-authored, AI-reviewed, or AI-assisted. The repository covers:

- **C_Programming/**: Progressive C programming tutorials with hands-on projects (beginner to advanced, including embedded/Arduino)
- **Docker/**: Docker and Kubernetes learning materials
- **Git/**: Git and GitHub tutorials including GitHub Actions
- **Linux/**: Linux fundamentals to server administration (Ubuntu/Debian and CentOS/RHEL)
- **PostgreSQL/**: PostgreSQL database tutorials from basics to administration
- **Spanish/**: Spanish language grammar and vocabulary reference materials

## PostgreSQL Learning Materials

### Content Structure

The PostgreSQL folder contains 14 lesson files covering:

**Beginner (01-05)**: Installation, DB management, tables, CRUD, conditions/sorting
**Intermediate (06-09)**: JOIN, aggregation, subqueries/CTE, views/indexes
**Advanced (10-13)**: Functions/procedures, transactions, triggers, backup/operations

### Quick Start

```bash
# Docker (recommended)
docker run --name postgres-study \
  -e POSTGRES_PASSWORD=mypassword \
  -p 5432:5432 \
  -d postgres:16

# Connect
docker exec -it postgres-study psql -U postgres
```

### psql Commands

| Command | Description |
|---------|-------------|
| `\l` | List databases |
| `\c dbname` | Connect to database |
| `\dt` | List tables |
| `\d tablename` | Describe table |
| `\q` | Quit |

### SQL Code Blocks

SQL code in the tutorials uses ` ```sql ` for syntax highlighting.

## Linux Learning Materials

### Content Structure

The Linux folder contains 13 files covering:

**Beginner (01-03)**: Linux basics, filesystem, file management
**Intermediate (04-08)**: Text processing, permissions, user management, processes, packages
**Advanced (09-12)**: Shell scripting, networking, monitoring, security/firewall

### Distro Support

Materials cover both distribution families in parallel:
- **Ubuntu/Debian**: apt, ufw, AppArmor
- **CentOS/RHEL**: dnf/yum, firewalld, SELinux

### Quick Start

```bash
# Docker (Ubuntu)
docker run -it ubuntu:22.04 bash

# Docker (Rocky Linux)
docker run -it rockylinux:9 bash
```

### Essential Commands

| Category | Commands |
|----------|----------|
| Navigation | `cd`, `ls`, `pwd`, `find` |
| Files | `cp`, `mv`, `rm`, `tar` |
| Text | `grep`, `sed`, `awk`, `cat` |
| Users | `useradd`, `usermod`, `sudo` |
| Processes | `ps`, `top`, `kill`, `systemctl` |
| Network | `ip`, `ss`, `ssh`, `scp` |

## C Programming Projects

### Build System

The C examples use Makefiles for building projects. There are two primary locations:

1. **Root Makefile** at [C_Programming/examples/Makefile](C_Programming/examples/Makefile) - builds all C programs
2. **Individual Makefiles** in specific project folders (e.g., [C_Programming/practices/Makefile](C_Programming/practices/Makefile))

### Common Commands

From the `C_Programming/examples/` directory:

```bash
# Build all programs
make

# Build only C programs (non-threaded)
make c-programs

# Build only multithreaded programs
make thread-programs

# Clean all compiled binaries
make clean

# Show available commands
make help

# Run specific programs
make run-calculator
make run-guess
make run-array
make run-list
make run-bit
make run-thread
```

### Compilation Patterns

**Standard C programs:**
```bash
gcc -Wall -Wextra -std=c11 program.c -o program
```

**Multithreaded programs:**
```bash
# Linux
gcc -Wall -Wextra -std=c11 -pthread program.c -o program

# macOS
gcc -Wall -Wextra -std=c11 -lpthread program.c -o program
```

**Debug builds:**
```bash
gcc -g -Wall -Wextra program.c -o program
```

### Arduino Programs

Arduino projects (.ino files) in the C_Programming section are designed for:
- **Arduino IDE**: Direct upload to hardware
- **Wokwi Simulator** (https://wokwi.com): Recommended for testing without hardware
- **PlatformIO**: `pio run` and `pio run --target upload`

Arduino projects are found in:
- `13_embedded_basic/`
- `15_gpio_control/`
- `16_serial_comm/`

### Project Structure

C programming examples follow a progressive difficulty structure:

**Beginner (⭐)**: Calculator, Number guessing game, Address book
**Intermediate (⭐⭐)**: Dynamic arrays, File encryption, Stack/Queue, Hash tables
**Advanced (⭐⭐⭐⭐)**: Snake game, Mini shell, Multithreading
**Embedded (Arduino)**: GPIO control, Serial communication, Bit manipulation

Each numbered lesson file (e.g., `02_프로젝트_계산기.md`) contains theory and exercises, with corresponding implementation in `examples/02_calculator/`.

### Debugging Common Issues

**Compilation errors:**
- `undefined reference to 'pthread_create'`: Missing `-pthread` flag
- `implicit declaration of function`: Missing header includes
- `permission denied`: Run `chmod +x program`

**Runtime errors:**
- Segmentation fault: Check pointers, use valgrind
- Memory leaks: Use valgrind for detection

## Repository Architecture

### File Naming Convention

Files use Korean naming with numerical prefixes for ordering:
- `00_`, `01_`, `02_`, etc. for sequential lessons
- Example: `01_C_기초_빠른복습.md` (01_C_Basics_Quick_Review)

### Content Organization

1. **Tutorial markdown files**: Root of each topic directory (e.g., `C_Programming/02_프로젝트_계산기.md`)
2. **Example code**: `examples/` subdirectory with matching numerical prefixes
3. **Practice code**: `practices/` for experimental/scratch work

### Language

- Documentation: Primarily Korean (한국어)
- Code comments: Mix of Korean and English
- File names: Korean with underscores replacing spaces

## Key Points for Development

- When modifying C code, maintain the existing Korean comment style
- Examples are educational and include detailed comments explaining concepts
- Arduino code should be testable on Wokwi simulator
- Makefiles use standard gcc with strict warnings (-Wall -Wextra)
- All C code targets C11 standard
