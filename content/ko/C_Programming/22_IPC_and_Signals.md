# 프로세스 간 통신과 시그널

## 목표

- IPC 메커니즘 이해하기: 파이프(Pipe), FIFO, 공유 메모리(Shared Memory), 메시지 큐(Message Queue)
- 강력한 프로세스 제어를 위한 sigaction을 사용한 시그널(Signal) 처리 마스터하기
- 생산자-소비자(Producer-Consumer) 및 부모-자식 간 협력을 위한 IPC 패턴 적용하기

**난이도**: ⭐⭐⭐⭐ (고급)

---

## 목차

1. [파이프](#1-파이프)
2. [명명된 파이프 (FIFO)](#2-명명된-파이프-fifos)
3. [공유 메모리](#3-공유-메모리)
4. [POSIX 메시지 큐](#4-posix-메시지-큐)
5. [시그널](#5-시그널)
6. [연습 문제](#6-연습-문제)
7. [참고 자료](#7-참고-자료)

---

## 1. 파이프

### 1.1 익명 파이프

파이프는 관련된 프로세스(부모-자식) 간에 단방향 데이터 흐름을 제공합니다.

```
┌────────────────────────────────────────────────────────┐
│                  Pipe Communication                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌───────────┐    pipe    ┌───────────┐               │
│  │  Parent   │───────────▶│  Child    │               │
│  │  (Writer) │   fd[1]    │  (Reader) │               │
│  │           │  ────────▶ │           │               │
│  └───────────┘   fd[0]    └───────────┘               │
│                                                        │
│  pipe(fd) creates:                                     │
│    fd[0] = read end                                    │
│    fd[1] = write end                                   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // Child: read from pipe
        close(pipefd[1]);  // Close unused write end

        char buffer[256];
        ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);
        if (n > 0) {
            buffer[n] = '\0';
            printf("Child received: %s\n", buffer);
        }

        close(pipefd[0]);
        exit(EXIT_SUCCESS);
    } else {
        // Parent: write to pipe
        close(pipefd[0]);  // Close unused read end

        const char *msg = "Hello from parent!";
        write(pipefd[1], msg, strlen(msg));

        close(pipefd[1]);
        wait(NULL);  // Wait for child
    }

    return 0;
}
```

### 1.2 두 개의 파이프를 사용한 양방향 통신

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int parent_to_child[2], child_to_parent[2];
    pipe(parent_to_child);
    pipe(child_to_parent);

    pid_t pid = fork();
    if (pid == 0) {
        // Child
        close(parent_to_child[1]);
        close(child_to_parent[0]);

        char buf[256];
        ssize_t n = read(parent_to_child[0], buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Child got: %s\n", buf);

        const char *reply = "Got it, thanks!";
        write(child_to_parent[1], reply, strlen(reply));

        close(parent_to_child[0]);
        close(child_to_parent[1]);
        exit(0);
    }

    // Parent
    close(parent_to_child[0]);
    close(child_to_parent[1]);

    const char *msg = "Task: process data";
    write(parent_to_child[1], msg, strlen(msg));
    close(parent_to_child[1]);

    char buf[256];
    ssize_t n = read(child_to_parent[0], buf, sizeof(buf) - 1);
    buf[n] = '\0';
    printf("Parent got reply: %s\n", buf);

    close(child_to_parent[0]);
    wait(NULL);
    return 0;
}
```

### 1.3 exec와 함께 사용하는 파이프 (셸 파이핑)

```c
// Simulate: ls -la | grep ".c"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    int pipefd[2];
    pipe(pipefd);

    pid_t pid1 = fork();
    if (pid1 == 0) {
        // First child: ls -la
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);  // stdout → pipe write
        close(pipefd[1]);
        execlp("ls", "ls", "-la", NULL);
        perror("execlp ls");
        exit(1);
    }

    pid_t pid2 = fork();
    if (pid2 == 0) {
        // Second child: grep ".c"
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);  // stdin ← pipe read
        close(pipefd[0]);
        execlp("grep", "grep", ".c", NULL);
        perror("execlp grep");
        exit(1);
    }

    // Parent: close both ends and wait
    close(pipefd[0]);
    close(pipefd[1]);
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);

    return 0;
}
```

---

## 2. 명명된 파이프 (FIFOs)

FIFO는 파일 시스템 엔트리를 통해 관련 없는 프로세스 간 통신을 가능하게 합니다.

### 2.1 FIFO 생성 및 사용

```c
// --- Writer process ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define FIFO_PATH "/tmp/myfifo"

int main(void) {
    // Create FIFO (ignore error if it already exists)
    mkfifo(FIFO_PATH, 0666);

    int fd = open(FIFO_PATH, O_WRONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    const char *messages[] = {"Hello", "World", "Done"};
    for (int i = 0; i < 3; i++) {
        write(fd, messages[i], strlen(messages[i]) + 1);
        printf("Sent: %s\n", messages[i]);
        sleep(1);
    }

    close(fd);
    return 0;
}
```

```c
// --- Reader process ---
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define FIFO_PATH "/tmp/myfifo"

int main(void) {
    int fd = open(FIFO_PATH, O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    char buffer[256];
    ssize_t n;
    while ((n = read(fd, buffer, sizeof(buffer))) > 0) {
        printf("Received: %s\n", buffer);
    }

    close(fd);
    unlink(FIFO_PATH);  // Clean up
    return 0;
}
```

---

## 3. 공유 메모리

공유 메모리는 프로세스 간에 데이터를 복사할 필요가 없기 때문에 가장 빠른 IPC 메커니즘입니다.

### 3.1 POSIX 공유 메모리

```
┌──────────────────────────────────────────────────────────┐
│              Shared Memory Architecture                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐       Physical Memory       ┌─────────┐│
│  │  Process A  │       ┌──────────────┐      │Process B ││
│  │             │       │              │      │          ││
│  │  Virtual    │──────▶│  Shared      │◀─────│ Virtual  ││
│  │  Address    │  mmap │  Region      │ mmap │ Address  ││
│  │  0x7f...    │       │              │      │ 0x7f...  ││
│  │             │       └──────────────┘      │          ││
│  └─────────────┘                             └─────────┘│
│                                                          │
│  ⚠ Requires synchronization (semaphore/mutex)           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

```c
// --- Producer ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME "/my_shm"
#define SEM_NAME "/my_sem"
#define SHM_SIZE 4096

typedef struct {
    int count;
    char data[256];
} shared_data_t;

int main(void) {
    // Create shared memory
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, sizeof(shared_data_t));

    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, shm_fd, 0);

    // Create semaphore for synchronization
    sem_t *sem = sem_open(SEM_NAME, O_CREAT, 0666, 0);

    // Write data
    shm->count = 42;
    snprintf(shm->data, sizeof(shm->data),
             "Hello from producer (PID=%d)", getpid());

    printf("Producer wrote: count=%d, data=%s\n",
           shm->count, shm->data);

    // Signal consumer
    sem_post(sem);

    // Cleanup
    sem_close(sem);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);

    return 0;
}
```

```c
// --- Consumer ---
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME "/my_shm"
#define SEM_NAME "/my_sem"

typedef struct {
    int count;
    char data[256];
} shared_data_t;

int main(void) {
    // Open shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ, MAP_SHARED, shm_fd, 0);

    // Wait for producer
    sem_t *sem = sem_open(SEM_NAME, 0);
    sem_wait(sem);

    // Read data
    printf("Consumer read: count=%d, data=%s\n",
           shm->count, shm->data);

    // Cleanup
    sem_close(sem);
    sem_unlink(SEM_NAME);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);
    shm_unlink(SHM_NAME);

    return 0;
}
```

---

## 4. POSIX 메시지 큐

메시지 큐는 우선순위를 지원하는 구조화된 메시지 전달을 제공합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mqueue.h>
#include <fcntl.h>

#define QUEUE_NAME "/my_queue"
#define MAX_MSG_SIZE 256
#define MAX_MSGS 10

// Sender
void sender(void) {
    struct mq_attr attr = {
        .mq_flags = 0,
        .mq_maxmsg = MAX_MSGS,
        .mq_msgsize = MAX_MSG_SIZE,
        .mq_curmsgs = 0
    };

    mqd_t mq = mq_open(QUEUE_NAME, O_CREAT | O_WRONLY, 0666, &attr);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        exit(1);
    }

    const char *msgs[] = {"High priority!", "Normal message", "Low priority"};
    unsigned int priorities[] = {10, 5, 1};

    for (int i = 0; i < 3; i++) {
        mq_send(mq, msgs[i], strlen(msgs[i]) + 1, priorities[i]);
        printf("Sent (prio=%u): %s\n", priorities[i], msgs[i]);
    }

    mq_close(mq);
}

// Receiver
void receiver(void) {
    mqd_t mq = mq_open(QUEUE_NAME, O_RDONLY);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        exit(1);
    }

    char buffer[MAX_MSG_SIZE];
    unsigned int priority;

    // Messages arrive highest priority first
    for (int i = 0; i < 3; i++) {
        ssize_t bytes = mq_receive(mq, buffer, MAX_MSG_SIZE, &priority);
        if (bytes >= 0) {
            printf("Received (prio=%u): %s\n", priority, buffer);
        }
    }

    mq_close(mq);
    mq_unlink(QUEUE_NAME);
}
```

---

## 5. 시그널

### 5.1 시그널 개요

시그널은 프로세스에 이벤트를 알리기 위해 전달되는 소프트웨어 인터럽트입니다.

```
┌──────────────────────────────────────────────────────────┐
│  Common Signals                                          │
├─────────┬──────────────────────────────────────────────┤
│ Signal  │ Description                                    │
├─────────┼──────────────────────────────────────────────┤
│ SIGINT  │ Interrupt (Ctrl+C)                             │
│ SIGTERM │ Termination request                            │
│ SIGKILL │ Forced kill (cannot be caught)                 │
│ SIGCHLD │ Child process stopped or terminated            │
│ SIGUSR1 │ User-defined signal 1                          │
│ SIGUSR2 │ User-defined signal 2                          │
│ SIGALRM │ Timer alarm                                    │
│ SIGPIPE │ Broken pipe (write to closed socket)           │
│ SIGSEGV │ Segmentation fault                             │
│ SIGSTOP │ Stop process (cannot be caught)                │
│ SIGCONT │ Continue stopped process                       │
└─────────┴──────────────────────────────────────────────┘
```

### 5.2 sigaction을 사용한 시그널 처리

이식성과 신뢰성 있는 동작을 위해 `signal()` 대신 항상 `sigaction()`을 사용하세요.

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

volatile sig_atomic_t running = 1;

void handle_sigint(int sig) {
    (void)sig;  // Suppress unused warning
    running = 0;
    // Only async-signal-safe functions here!
    write(STDOUT_FILENO, "\nCaught SIGINT, shutting down...\n", 33);
}

void handle_sigusr1(int sig, siginfo_t *info, void *context) {
    (void)sig;
    (void)context;
    // siginfo_t gives us sender information
    printf("SIGUSR1 from PID %d\n", info->si_pid);
}

int main(void) {
    // Setup SIGINT handler
    struct sigaction sa_int = {0};
    sa_int.sa_handler = handle_sigint;
    sigemptyset(&sa_int.sa_mask);
    sa_int.sa_flags = 0;
    sigaction(SIGINT, &sa_int, NULL);

    // Setup SIGUSR1 handler with siginfo
    struct sigaction sa_usr = {0};
    sa_usr.sa_sigaction = handle_sigusr1;
    sigemptyset(&sa_usr.sa_mask);
    sa_usr.sa_flags = SA_SIGINFO;
    sigaction(SIGUSR1, &sa_usr, NULL);

    // Ignore SIGPIPE (common in network programs)
    signal(SIGPIPE, SIG_IGN);

    printf("PID: %d - Press Ctrl+C or send SIGUSR1\n", getpid());

    while (running) {
        printf("Working...\n");
        sleep(2);
    }

    printf("Clean shutdown complete\n");
    return 0;
}
```

### 5.3 시그널 마스킹

```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    sigset_t block_set, old_set;

    // Block SIGINT during critical section
    sigemptyset(&block_set);
    sigaddset(&block_set, SIGINT);

    sigprocmask(SIG_BLOCK, &block_set, &old_set);

    // ---- Critical section ----
    printf("SIGINT blocked. Ctrl+C won't interrupt.\n");
    sleep(5);
    printf("Critical section done.\n");
    // ---- End critical section ----

    // Restore original mask
    sigprocmask(SIG_SETMASK, &old_set, NULL);
    printf("SIGINT unblocked. Ctrl+C works again.\n");

    sleep(5);
    return 0;
}
```

### 5.4 SIGCHLD를 사용한 자식 프로세스 정리

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

void handle_sigchld(int sig) {
    (void)sig;
    // Reap all terminated children (non-blocking)
    int status;
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        if (WIFEXITED(status)) {
            // Child exited normally
        }
    }
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handle_sigchld;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    // Fork multiple children
    for (int i = 0; i < 5; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            printf("Child %d (PID=%d) working...\n", i, getpid());
            sleep(i + 1);
            printf("Child %d done\n", i);
            exit(i);
        }
    }

    // Parent continues working
    printf("Parent (PID=%d) waiting...\n", getpid());
    sleep(10);
    printf("Parent done\n");

    return 0;
}
```

---

## 6. 연습 문제

### 문제 1: 공유 메모리를 사용한 생산자-소비자

POSIX 공유 메모리와 세마포어를 사용하여 생산자-소비자 시스템을 구현하세요:
- 생산자는 공유 메모리의 순환 버퍼에 정수 1-100을 씁니다
- 소비자는 그것을 읽고 출력합니다
- 동기화를 위해 세마포어를 사용합니다

### 문제 2: 다중 프로세스 파이프라인

파이프를 사용하여 3단계 파이프라인을 만드세요:
- 1단계: 파일에서 줄을 읽습니다
- 2단계: 대문자로 변환합니다
- 3단계: 단어 빈도를 계산하고 출력합니다

### 문제 3: 워치독 프로세스

다음 기능을 가진 워치독을 작성하세요:
- 자식 워커 프로세스를 포크합니다
- SIGCHLD로 모니터링합니다
- 충돌 시 자동으로 재시작합니다
- 두 프로세스 모두의 우아한 종료를 위해 SIGTERM을 처리합니다

---

## 7. 참고 자료

- W. Richard Stevens, *Advanced Programming in the UNIX Environment* (3rd ed.)
- `man 7 pipe`, `man 7 fifo`, `man 7 shm_overview`, `man 7 mq_overview`
- `man 7 signal`, `man 2 sigaction`, `man 2 sigprocmask`

---

[Previous: 21_Network_Programming](./21_Network_Programming.md) | [Next: 00_Overview](./00_Overview.md)
