# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose & Roadmap

### 목적 (Purpose)
개인 학습을 위한 체계적인 기술 자료 모음. 각 주제별로 입문부터 심화까지 단계별 학습이 가능하도록 구성.

### 현재 상태 (Current Status)
- **Algorithm**: 완료 (30개 레슨 + Overview) - 복잡도분석, 정렬/탐색, 그래프, DP, 백트래킹, 실전 문제풀이, 해시테이블, 문자열알고리즘, 정수론, 위상정렬, 비트마스크DP, 세그먼트트리, 트라이, 펜윅트리, SCC, 네트워크플로우, LCA, 기하, 게임이론, 고급DP최적화
- **C_Programming**: 완료 (21개 레슨 + Overview) - 디버깅, 고급 임베디드, 포인터 심화 추가
- **Data_Analysis**: 완료 (12개 레슨 + Overview) - NumPy, Pandas, 데이터전처리, 기술통계/EDA, Matplotlib/Seaborn 시각화, 통계분석
- **Computer_Architecture**: 완료 (19개 레슨 + Overview) - 데이터표현, 논리회로, CPU구조, 파이프라이닝, 캐시, 가상메모리, 병렬처리
- **CPP**: 완료 (19개 레슨 + Overview) - 멀티스레딩, C++20, 디자인패턴 추가
- **Docker**: 완료 (11개 레슨 + Overview) - K8s 보안/심화, Helm, CI/CD 추가
- **Git**: 완료 (11개 레슨 + Overview) - 워크플로우, 고급 기법, 모노레포 추가
- **Linux**: 완료 (27개 레슨 + Overview) - SELinux/AppArmor, 로그관리, 백업, 커널, KVM, Ansible, 고급네트워킹, 클라우드, HA클러스터, 트러블슈팅 추가
- **Machine_Learning**: 완료 (15개 레슨 + Overview) - 선형/로지스틱 회귀, 결정트리, 앙상블, SVM, 클러스터링, 차원축소, sklearn 파이프라인
- **Network**: 완료 (18개 레슨 + Overview) - OSI/TCP-IP, IP/서브네팅, 라우팅, TCP/UDP, HTTP/DNS, 보안, 실무도구
- **Numerical_Simulation**: 완료 (21개 레슨 + Overview) - 수치해석, ODE(Euler/RK4/stiff), PDE, 열/파동방정식, CFD, FDTD, MHD, 플라즈마(PIC), 몬테카를로
- **OpenCV_ComputerVision**: 완료 (21개 레슨 + Overview) - 이미지처리, 필터링, 엣지검출, 특징점, 얼굴인식, DNN
- **OS_Theory**: 완료 (19개 레슨 + Overview) - 프로세스, 스케줄링, 동기화, 메모리, 가상메모리, 파일시스템
- **PostgreSQL**: 완료 (19개 레슨 + Overview) - JSON/JSONB, 쿼리최적화, 복제, 윈도우함수, 파티셔닝 추가
- **Python**: 완료 (15개 레슨 + Overview) - 테스트, 패키징, 데이터클래스, 패턴매칭 추가
- **Spanish**: 완료 (10개 파일 + Overview) - 접속법, 동사활용, 지역변형, 일상회화 추가
- **Statistics_Advanced**: 완료 (15개 레슨 + Overview) - 확률론, 추정/신뢰구간, ANOVA, GLM, 베이지안(PyMC), 시계열(ARIMA), 다변량, 실험설계
- **System_Design**: 완료 (19개 레슨 + Overview) - 확장성, 캐싱, 샤딩, 메시지큐, 마이크로서비스, 분산시스템
- **WebDev**: 완료 (14개 레슨 + Overview) - TypeScript, 웹접근성, SEO, 빌드도구 추가

### 향후 계획 (Future Plans)
- [x] Python 고급 문법 자료 추가
- [x] C++ 학습 자료 추가 (입문~모던C++)
- [x] WebDev 학습 자료 추가 (HTML/CSS/JavaScript)
- [x] 알고리즘/자료구조 심화 자료
- [x] 컴퓨터 구조 이론 자료 추가
- [x] 네트워크 이론 자료 추가
- [x] OpenCV/컴퓨터 비전 자료 추가
- [x] 운영체제 이론 자료 추가
- [x] 시스템 디자인 자료 추가
- [x] 데이터 분석 (NumPy/Pandas/시각화) 자료 추가
- [x] 머신러닝 기초 (sklearn) 자료 추가
- [ ] 클라우드 (AWS/GCP) 학습 자료
- [ ] 딥러닝/PyTorch 학습 자료 (CNN, RNN, Transformer)
- [ ] LLM/자연어처리 학습 자료 (BERT, GPT, 프롬프트 엔지니어링)

### 업데이트 이력 (Change Log)
- **2024-01**: C Programming, Docker, Git, Spanish 자료 생성
- **2024-01-23**: PostgreSQL 자료 추가, 각 폴더에 00_Overview.md 추가, C_Programming 파일 넘버링 수정 (00→01 시작)
- **2026-01-23**: Linux 자료 추가 (12개 레슨, 서버 관리 포함, Ubuntu/CentOS 병렬 안내)
- **2026-01-23**: Spanish 파일명 정리 (01_문법~05_기타_품사), 어휘 170개+ 추가 (비즈니스/여행/관용어)
- **2026-01-23**: Python 고급 문법 자료 추가 (10개 레슨: 타입힌팅, 데코레이터, 메타클래스, 비동기 등)
- **2026-01-23**: C++ 학습 자료 추가 (15개 레슨: 입문, OOP, STL, 템플릿, 스마트포인터, 모던C++)
- **2026-01-23**: WebDev 학습 자료 추가 (9개 레슨: HTML, CSS, JavaScript, 실전 프로젝트)
- **2026-01-23**: 전체 학습 자료 확장 (총 33개 파일 추가)
  - C_Programming: 디버깅/메모리분석, 고급임베디드 프로토콜 추가
  - CPP: 멀티스레딩, C++20 심화, 디자인패턴 추가
  - Docker: K8s 보안/심화, Helm, CI/CD 파이프라인 추가
  - Git: 워크플로우 전략, 고급 Git 기법, 모노레포 관리 추가
  - Linux: systemd 심화, 성능 튜닝, 컨테이너 내부, 저장소 관리 추가
  - PostgreSQL: JSON/JSONB, 쿼리 최적화 심화, 복제/HA, 윈도우 함수, 파티셔닝 추가
  - Python: 테스트(pytest), 패키징(Poetry), 데이터클래스, 패턴매칭 추가
  - Spanish: 접속법 심화, 동사 활용 체계, 지역 변형, 일상 회화/관용어 추가
  - WebDev: TypeScript, 웹 접근성, SEO, 빌드 도구 추가
- **2026-01-23**: Linux 자료 확장 (16개 → 26개 레슨)
  - SELinux/AppArmor, 로그 관리, 백업 및 복구, 커널 관리, KVM 가상화
  - Ansible 기초, 고급 네트워킹, 클라우드 통합, 고가용성 클러스터, 트러블슈팅 가이드
- **2026-01-23**: 전체 Overview 파일 동기화 (8개 폴더)
- **2026-01-24**: Algorithm 폴더 추가 (16개 레슨)
  - 복잡도 분석, 배열/문자열, 스택/큐 활용
  - 정렬/탐색 알고리즘, 그래프 기초, 최단경로, MST
  - 동적 프로그래밍, 탐욕 알고리즘, 분할정복, 백트래킹
  - 트리/BST, 힙/우선순위큐, 실전 문제풀이 (코딩테스트 전략)
- **2026-01-26**: Algorithm 폴더 확장 (16개 → 30개 레슨)
  - Tier 1 (코딩 인터뷰 필수): 해시테이블, 문자열 알고리즘, 수학/정수론, 위상정렬, 비트마스크 DP
  - Tier 2 (고급 자료구조): 세그먼트 트리, 트라이, 펜윅 트리
  - Tier 3 (고급 그래프): 강한연결요소(SCC), 네트워크 플로우, LCA와 트리쿼리
  - Tier 4 (특수 주제): 기하 알고리즘, 게임 이론, 고급 DP 최적화(CHT, D&C, Knuth)
- **2026-01-26**: Algorithm 파일 순서 재배치 (논리적 학습 흐름 최적화)
  - 기초 자료구조 (01-05): 복잡도, 배열/문자열, 스택/큐, 해시테이블, 정렬
  - 탐색과 분할정복 (06-08): 탐색, 분할정복, 백트래킹
  - 트리 자료구조 (09-11): 트리/BST, 힙, 트라이
  - 그래프 (12-17): 그래프 기초, 위상정렬, 최단경로, MST, LCA, SCC
  - DP와 수학 (18-22): DP, 탐욕, 비트마스크DP, 정수론, 문자열
  - 고급 자료구조 (23-24): 세그먼트 트리, 펜윅 트리
  - 고급 그래프/특수 (25-28): 네트워크 플로우, 기하, 게임이론, 고급DP
  - 마무리 (29): 실전 문제풀이
- **2026-01-27**: Computer_Architecture 폴더 추가 (18개 레슨)
  - 기초 (01-05): 컴퓨터 개요, 데이터 표현, 정수/실수 표현, 논리 게이트, 조합 논리
  - CPU (06-10): 순차 논리, CPU 구조, 제어장치, ISA, 어셈블리
  - 성능 (11-13): 파이프라이닝, 분기 예측, 슈퍼스칼라/비순차 실행
  - 메모리 (14-16): 메모리 계층, 캐시, 가상 메모리
  - 병렬 (17-18): 입출력 시스템, 병렬 처리와 멀티코어
- **2026-01-27**: Network 폴더 추가 (17개 레슨)
  - 기초 (01-04): 네트워크 개념, OSI 7계층, TCP/IP 모델, 물리 계층
  - 네트워크 계층 (05-09): 데이터링크, IP/서브네팅, 서브네팅 실습, 라우팅 기초, 라우팅 프로토콜
  - 전송 계층 (10-11): TCP, UDP와 포트
  - 애플리케이션 (12-14): DNS, HTTP/HTTPS, 기타 프로토콜
  - 보안/실무 (15-17): 보안 기초, 위협과 대응, 실무 도구
- **2026-01-27**: OpenCV_ComputerVision 폴더 추가 (20개 레슨)
  - 기초 (01-04): 환경설정, 이미지 기초, 색상 공간, 기하학적 변환
  - 이미지 처리 (05-08): 필터링, 모폴로지, 이진화, 엣지 검출
  - 객체 분석 (09-12): 윤곽선, 도형 분석, 허프 변환, 히스토그램
  - 특징/검출 (13-15): 특징점 검출, 특징점 매칭, 객체 검출 기초
  - 고급 (16-18): 얼굴 검출/인식, 비디오 처리, 카메라 캘리브레이션
  - DNN/실전 (19-20): 딥러닝 DNN 모듈, 실전 프로젝트
- **2026-01-27**: OS_Theory 폴더 추가 (18개 레슨)
  - 기초 (01-03): 운영체제 개요, 프로세스, 스레드
  - 스케줄링 (04-06): 스케줄링 기초/알고리즘/고급
  - 동기화 (07-09): 동기화 기초/도구, 데드락
  - 메모리 (10-13): 메모리 기초, 연속할당, 페이징, 세그멘테이션
  - 가상메모리 (14-15): 가상 메모리, 페이지 교체
  - 파일/IO (16-18): 파일시스템 기초/구현, IO와 IPC
- **2026-01-27**: System_Design 폴더 추가 (18개 레슨)
  - 기초 (01-03): 설계 개요, 확장성, 네트워크 복습
  - 트래픽 (04-05): 로드밸런싱, API 게이트웨이
  - 캐싱 (06-07): 캐싱 전략, 분산 캐시
  - 데이터베이스 (08-10): DB 확장, 복제, 일관성 패턴
  - 메시지큐 (11-12): 메시지큐 기초, 시스템 비교
  - 마이크로서비스 (13-14): 기초, 패턴
  - 분산시스템 (15-16): 개념, 합의 알고리즘
  - 실전 (17-18): 설계 예제
- **2026-01-29**: Data_Analysis 폴더 추가 (12개 레슨)
  - NumPy (01-02): 배열 기초, 선형대수/통계/난수
  - Pandas (03-05): 기초, 데이터 조작, 멀티인덱스/시계열
  - 전처리 (06): 결측치, 이상치, 정규화, 인코딩
  - 분석 (07): 기술통계, EDA
  - 시각화 (08-09): Matplotlib, Seaborn
  - 통계/실전 (10-11): 확률분포, 가설검정, Kaggle 실습
- **2026-01-29**: Machine_Learning 폴더 추가 (15개 레슨)
  - 기초 (01-05): ML 개요, 선형/로지스틱 회귀, 모델 평가, 교차검증
  - 트리 모델 (06-08): 결정트리, Random Forest, XGBoost/LightGBM
  - 기타 알고리즘 (09-10): SVM, k-NN/나이브베이즈
  - 비지도학습 (11-12): 클러스터링(K-Means/DBSCAN), 차원축소(PCA/t-SNE)
  - 실무 (13-14): sklearn 파이프라인, 실전 프로젝트
- **2026-01-29**: Numerical_Simulation 폴더 추가 (20개 레슨)
  - 기초 (01-02): 수치해석 기초, 선형대수 복습
  - ODE (03-06): 상미분방정식 기초, Euler/RK4, stiff 문제, 연립 ODE
  - PDE 기초 (07-12): 편미분방정식 개요, 유한차분법, 열/파동방정식, 라플라스/포아송, 이류방정식
  - CFD (13-14): 유체역학 기초, 비압축성 유동 (SIMPLE)
  - 전자기 (15-16): Maxwell 방정식, FDTD 구현
  - MHD/플라즈마 (17-19): MHD 기초/수치해법, 플라즈마 PIC 시뮬레이션
- **2026-01-29**: Statistics_Advanced 폴더 추가 (15개 레슨)
  - 기초 (01-04): 확률론 복습, 표본/추정, 신뢰구간, 가설검정 심화
  - 회귀 (05-07): ANOVA, 회귀분석 심화, GLM
  - 베이지안 (08-09): 베이지안 기초, PyMC 추론
  - 시계열 (10-11): 시계열 분석 기초, ARIMA 모형
  - 응용 (12-14): 다변량 분석(PCA/요인분석), 비모수 통계, 실험 설계(A/B 테스트)
- **2026-01-30**: Algorithm 예제 대폭 확장 (87개 파일)
  - examples/ 폴더 구조 변경: python/, c/, cpp/ 하위 폴더로 분리
  - Python 예제 29개 (기존 10개 → 전체 29개 레슨 커버)
  - C 예제 29개 + Makefile (전체 신규 작성)
  - C++ 예제 29개 + Makefile (모던 C++17, STL 활용)
  - 100% 커버리지: 모든 레슨에 Python + C + C++ 세 버전 예제 제공

---

## Repository Overview

This is a personal study repository containing educational materials in Korean and English. Content is self-authored, AI-reviewed, or AI-assisted. The repository covers:

- **Algorithm/**: Algorithm and data structure tutorials for coding tests (complexity analysis, sorting, searching, graphs, DP, backtracking, practical problem solving)
- **C_Programming/**: Progressive C programming tutorials with hands-on projects (beginner to advanced, including embedded/Arduino, debugging with GDB/Valgrind)
- **CPP/**: C++ programming from basics to modern C++ (OOP, STL, templates, smart pointers, C++11/14/17/20, multithreading, design patterns)
- **Docker/**: Docker and Kubernetes learning materials (K8s security, Ingress, Helm, CI/CD with GitHub Actions, GitOps)
- **Git/**: Git and GitHub tutorials (GitHub Actions, workflow strategies, advanced techniques, monorepo management)
- **Linux/**: Linux fundamentals to server administration (Ubuntu/Debian and CentOS/RHEL, systemd, performance tuning, container internals, storage management)
- **PostgreSQL/**: PostgreSQL database tutorials (SQL basics to admin, JSON/JSONB, query optimization, replication/HA, window functions, partitioning)
- **Python/**: Advanced Python programming (type hints, decorators, metaclasses, async, testing with pytest, packaging with Poetry, dataclasses, pattern matching)
- **Spanish/**: Spanish language grammar and vocabulary (subjunctive, verb conjugation systems, regional variations, idioms and daily conversation)
- **WebDev/**: Web development (HTML, CSS, JavaScript, TypeScript, web accessibility, SEO, build tools like Vite/webpack)

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

### 파일 추가 후 필수 작업

새로운 레슨 파일을 추가한 후에는 반드시 다음 파일들을 동기화해야 합니다:

1. **`{폴더}/00_Overview.md`**: 파일 목록 테이블에 새 파일 추가, 학습 순서 업데이트
2. **`README.md`**: 레슨 수 업데이트
3. **`CLAUDE.md`**: 현재 상태(Current Status) 업데이트, 변경 이력(Change Log) 추가

예시:
```
# 파일 추가 후
1. PostgreSQL/00_Overview.md 파일 목록 테이블에 새 파일 추가
2. README.md의 PostgreSQL 레슨 수 업데이트 (13개 → 18개)
3. CLAUDE.md 현재 상태 및 변경 이력 업데이트
```

## C++ Learning Materials

### Content Structure

The CPP folder contains 16 files covering C++ programming from basics to modern C++:

**Beginner (01-03)**: Environment setup, variables/types, control flow
**Basics (04-06)**: Functions, arrays/strings, pointers/references
**Intermediate (07-09)**: Classes basics, classes advanced, inheritance/polymorphism
**Intermediate+ (10-12)**: STL containers, STL algorithms, templates
**Advanced (13-15)**: Exception handling/file I/O, smart pointers, modern C++

### Quick Start

```bash
# C++17 compilation
g++ -std=c++17 -Wall -Wextra main.cpp -o main

# C++20 compilation
g++ -std=c++20 -Wall -Wextra main.cpp -o main

# Run
./main
```

### Key Topics

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| 01 | Setup | g++, IDE, Hello World |
| 02 | Variables | int, double, auto, const |
| 03 | Control Flow | if, switch, for, while |
| 04 | Functions | overloading, default params |
| 05 | Arrays/Strings | std::array, std::string |
| 06 | Pointers | references, new/delete |
| 07 | Classes Basic | constructors, destructors |
| 08 | Classes Advanced | operators, copy/move |
| 09 | Inheritance | virtual, abstract classes |
| 10 | STL Containers | vector, map, set |
| 11 | STL Algorithms | sort, find, transform |
| 12 | Templates | function/class templates |
| 13 | Exceptions/Files | try/catch, fstream |
| 14 | Smart Pointers | unique_ptr, shared_ptr |
| 15 | Modern C++ | C++11/14/17/20 features |

---

## Python Learning Materials

### Content Structure

The Python folder contains 11 files covering advanced Python programming:

**Intermediate (01-03)**: Type hints, decorators, context managers
**Intermediate+ (04-06)**: Iterators/generators, closures/scope, metaclasses
**Advanced (07-10)**: Descriptors, async programming, functional programming, performance optimization

### Quick Start

```bash
# Python version (3.10+ recommended)
python --version

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Type checker
pip install mypy
mypy your_script.py
```

### Key Topics

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| 01 | Type Hints | typing module, TypeVar, Protocol |
| 02 | Decorators | @wraps, function/class decorators |
| 03 | Context Managers | with statement, contextlib |
| 04 | Iterators/Generators | yield, itertools |
| 05 | Closures | LEGB, nonlocal, factory patterns |
| 06 | Metaclasses | type, __new__, __init_subclass__ |
| 07 | Descriptors | __get__, __set__, property |
| 08 | Async | async/await, asyncio, Tasks |
| 09 | Functional | map, filter, reduce, functools |
| 10 | Performance | profiling, optimization techniques |

---

## WebDev Learning Materials

### Content Structure

The WebDev folder contains 14 files covering web front-end development:

**HTML (01-02)**: HTML basics, forms and tables
**CSS (03-05)**: CSS basics, layouts (Flexbox/Grid), responsive design
**JavaScript (06-08)**: JS basics, DOM/events, async programming
**Projects (09)**: Todo app, weather app, image gallery
**Advanced (10-13)**: TypeScript, web accessibility (A11y), SEO, build tools (Vite/webpack)

### Quick Start

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Page</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Hello World</h1>
    <script src="main.js" defer></script>
</body>
</html>
```

### Key Topics

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| 01 | HTML Basics | tags, structure, semantic HTML |
| 02 | HTML Forms | input, form, table |
| 03 | CSS Basics | selectors, box model, colors |
| 04 | CSS Layout | Flexbox, Grid, position |
| 05 | CSS Responsive | media queries, mobile-first |
| 06 | JS Basics | variables, functions, arrays, objects |
| 07 | DOM/Events | querySelector, addEventListener, event delegation |
| 08 | Async JS | Promise, async/await, fetch |
| 09 | Projects | Todo app, weather app, gallery |
| 10 | TypeScript | type system, interfaces, generics, utility types |
| 11 | Web Accessibility | WCAG, ARIA, keyboard navigation, screen readers |
| 12 | SEO | meta tags, structured data (JSON-LD), Core Web Vitals |
| 13 | Build Tools | npm/yarn/pnpm, Vite, webpack, ESLint, Prettier |

### Project Templates

```
project/
├── index.html
├── css/
│   └── style.css
└── js/
    └── app.js
```

---

## Algorithm Learning Materials

### Content Structure

The Algorithm folder contains 30 files covering algorithms and data structures for coding tests:

**Basics (01-05)**: Complexity analysis, arrays/strings, stacks/queues, hash tables, sorting
**Search & Divide-Conquer (06-08)**: Binary search, divide and conquer, backtracking
**Trees (09-11)**: Trees/BST, heaps/priority queues, trie
**Graphs (12-17)**: Graph basics, topological sort, shortest paths, MST, LCA, SCC
**DP & Math (18-22)**: Dynamic programming, greedy algorithms, bitmask DP, number theory, string algorithms
**Advanced Data Structures (23-24)**: Segment tree, Fenwick tree (BIT)
**Advanced Graphs (25)**: Network flow
**Special Topics (26-28)**: Computational geometry, game theory, advanced DP optimization (CHT, D&C, Knuth)
**Practice (29)**: Problem-solving strategies for coding tests

### Quick Start

```python
# Python (most common for coding tests)
import heapq
from collections import deque, defaultdict

# BFS template
def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Key Topics

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| 01 | Complexity | Big O, time/space complexity |
| 02 | Arrays/Strings | 2-pointer, sliding window, prefix sum |
| 03 | Stacks/Queues | Bracket validation, monotonic stack |
| 04 | Hash Tables | Hash functions, collision handling |
| 05 | Sorting | Bubble, selection, merge, quick, heap |
| 06 | Searching | Binary search, parametric search |
| 07 | Divide and Conquer | Merge sort, quick sort, matrix exponent |
| 08 | Backtracking | Permutations, combinations, N-Queens |
| 09 | Trees/BST | Traversals, BST operations |
| 10 | Heaps | Heap sort, priority queue, k-th element |
| 11 | Trie | Prefix tree, autocomplete, XOR trie |
| 12 | Graph Basics | DFS, BFS, representations |
| 13 | Topological Sort | Kahn's algorithm, cycle detection |
| 14 | Shortest Path | Dijkstra, Bellman-Ford, Floyd-Warshall |
| 15 | MST | Kruskal, Prim, Union-Find |
| 16 | LCA | Binary lifting, sparse table, HLD |
| 17 | SCC | Tarjan, Kosaraju, 2-SAT |
| 18 | Dynamic Programming | Memoization, tabulation, knapsack, LCS |
| 19 | Greedy | Activity selection, fractional knapsack |
| 20 | Bitmask DP | TSP, subset enumeration |
| 21 | Number Theory | Modular arithmetic, primes, combinatorics |
| 22 | String Algorithms | KMP, Rabin-Karp, Z-algorithm |
| 23 | Segment Tree | Range queries, lazy propagation |
| 24 | Fenwick Tree | BIT, range sum, 2D BIT |
| 25 | Network Flow | Ford-Fulkerson, bipartite matching |
| 26 | Geometry | CCW, convex hull, line intersection |
| 27 | Game Theory | Nim, Sprague-Grundy, minimax |
| 28 | Advanced DP | CHT, D&C optimization, Knuth |
| 29 | Practice | Problem-solving strategies, interview tips |

### Time Complexity Reference

| Input Size (N) | Max Complexity | Suitable Algorithms |
|----------------|----------------|---------------------|
| N ≤ 10 | O(N!) | Brute force, backtracking |
| N ≤ 20 | O(2^N) | Bitmask, backtracking |
| N ≤ 500 | O(N³) | Floyd-Warshall |
| N ≤ 5,000 | O(N²) | DP, brute force |
| N ≤ 100,000 | O(N log N) | Sorting, binary search |
| N ≤ 10^7 | O(N) | Two-pointer, hash |
| N ≤ 10^18 | O(log N) | Binary search, math |
