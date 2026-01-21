# GitHub Actions

## 1. GitHub Actions란?

GitHub Actions는 **CI/CD 자동화 플랫폼**입니다. 코드 푸시, PR 생성 등의 이벤트에 따라 자동으로 워크플로우를 실행합니다.

### 주요 용도

| 용도 | 예시 |
|------|------|
| **CI (Continuous Integration)** | 테스트 자동 실행, 린트 검사 |
| **CD (Continuous Deployment)** | 자동 배포, Docker 이미지 빌드 |
| **자동화** | 이슈 라벨링, 릴리스 노트 생성 |

### 핵심 개념

```
┌─────────────────────────────────────────────────────┐
│                   Workflow                          │
│   (.github/workflows/ci.yml)                        │
│                                                     │
│   ┌─────────────────────────────────────────────┐  │
│   │                  Job: build                  │  │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐      │  │
│   │   │ Step 1  │→│ Step 2  │→│ Step 3  │      │  │
│   │   │Checkout │ │ Install │ │  Test   │      │  │
│   │   └─────────┘ └─────────┘ └─────────┘      │  │
│   └─────────────────────────────────────────────┘  │
│                        ↓                            │
│   ┌─────────────────────────────────────────────┐  │
│   │                  Job: deploy                 │  │
│   │   ┌─────────┐ ┌─────────┐                   │  │
│   │   │ Build   │→│ Deploy  │                   │  │
│   │   └─────────┘ └─────────┘                   │  │
│   └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

| 개념 | 설명 |
|------|------|
| **Workflow** | 자동화 프로세스 전체 (YAML 파일) |
| **Event** | 워크플로우를 트리거하는 이벤트 |
| **Job** | 같은 러너에서 실행되는 단계 묶음 |
| **Step** | 개별 작업 단위 |
| **Action** | 재사용 가능한 작업 단위 |
| **Runner** | 워크플로우를 실행하는 서버 |

---

## 2. 워크플로우 파일 구조

워크플로우는 `.github/workflows/` 디렉토리에 YAML 파일로 저장합니다.

### 기본 구조

```yaml
# .github/workflows/ci.yml

name: CI Pipeline           # 워크플로우 이름

on:                         # 트리거 이벤트
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:                       # 작업 정의
  build:                    # Job 이름
    runs-on: ubuntu-latest  # 실행 환경

    steps:                  # 단계들
      - name: Checkout      # Step 이름
        uses: actions/checkout@v4

      - name: Run tests
        run: npm test
```

---

## 3. 트리거 이벤트 (on)

### push / pull_request

```yaml
on:
  push:
    branches:
      - main
      - 'release/**'
    paths:
      - 'src/**'          # src 폴더 변경 시만
    paths-ignore:
      - '**.md'           # md 파일 변경 무시

  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]
```

### 수동 실행 (workflow_dispatch)

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: '배포 환경'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
```

### 스케줄 (cron)

```yaml
on:
  schedule:
    - cron: '0 9 * * 1-5'  # 평일 오전 9시 (UTC)

# cron 형식: 분 시 일 월 요일
# 0 9 * * 1-5 = 매주 월-금 09:00
```

### 다른 워크플로우 완료 시

```yaml
on:
  workflow_run:
    workflows: ["Build"]
    types: [completed]
```

---

## 4. Jobs 설정

### 기본 Job

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm test
```

### 실행 환경 (runs-on)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest      # Ubuntu 최신
    # runs-on: ubuntu-22.04     # 특정 버전
    # runs-on: macos-latest     # macOS
    # runs-on: windows-latest   # Windows
```

### Job 의존성 (needs)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: npm run build

  test:
    needs: build              # build 완료 후 실행
    runs-on: ubuntu-latest
    steps:
      - run: npm test

  deploy:
    needs: [build, test]      # 둘 다 완료 후 실행
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

### 병렬 실행

```yaml
jobs:
  test-node-16:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '16'
      - run: npm test

  test-node-18:               # 병렬 실행
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - run: npm test
```

### 매트릭스 전략

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
        os: [ubuntu-latest, macos-latest]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm test
```

### 조건부 실행 (if)

```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh

  notify:
    if: failure()             # 실패 시에만
    runs-on: ubuntu-latest
    steps:
      - run: echo "Build failed!"
```

---

## 5. Steps 설정

### Action 사용 (uses)

```yaml
steps:
  # 공식 Action
  - uses: actions/checkout@v4

  # 특정 버전
  - uses: actions/setup-node@v4
    with:
      node-version: '18'

  # 마켓플레이스 Action
  - uses: docker/build-push-action@v5
```

### 명령어 실행 (run)

```yaml
steps:
  # 단일 명령
  - run: npm install

  # 여러 명령
  - run: |
      npm install
      npm run build
      npm test

  # 작업 디렉토리 지정
  - run: npm install
    working-directory: ./frontend

  # 쉘 지정
  - run: echo "Hello"
    shell: bash
```

### 환경 변수

```yaml
steps:
  - run: echo $MY_VAR
    env:
      MY_VAR: "Hello"

  - run: echo ${{ env.MY_VAR }}
```

### Secrets 사용

```yaml
steps:
  - run: echo ${{ secrets.API_KEY }}
    env:
      API_KEY: ${{ secrets.API_KEY }}
```

> **Secrets 설정**: Repository → Settings → Secrets and variables → Actions

---

## 6. 실습 예제

### 예제 1: Node.js 테스트 자동화

```yaml
# .github/workflows/node-ci.yml

name: Node.js CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18, 20]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run tests
        run: npm test

      - name: Build
        run: npm run build
```

### 예제 2: Docker 이미지 빌드 & 푸시

```yaml
# .github/workflows/docker.yml

name: Docker Build & Push

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

### 예제 3: PR 자동 라벨링

```yaml
# .github/workflows/labeler.yml

name: PR Labeler

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  label:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - uses: actions/labeler@v5
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
```

```yaml
# .github/labeler.yml (라벨 규칙)

frontend:
  - 'src/frontend/**'
  - '*.css'
  - '*.html'

backend:
  - 'src/backend/**'
  - 'api/**'

documentation:
  - '**/*.md'
  - 'docs/**'
```

### 예제 4: 자동 배포 (Vercel)

```yaml
# .github/workflows/deploy.yml

name: Deploy to Vercel

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install Vercel CLI
        run: npm install -g vercel

      - name: Deploy to Vercel
        run: vercel --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

### 예제 5: 릴리스 자동화

```yaml
# .github/workflows/release.yml

name: Release

on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: orhun/git-cliff-action@v3
        with:
          args: --latest --strip header

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          body: ${{ steps.changelog.outputs.content }}
          draft: false
          prerelease: ${{ contains(github.ref, 'beta') }}
```

---

## 7. 유용한 Actions

| Action | 용도 |
|--------|------|
| `actions/checkout@v4` | 코드 체크아웃 |
| `actions/setup-node@v4` | Node.js 설정 |
| `actions/setup-python@v5` | Python 설정 |
| `actions/cache@v4` | 의존성 캐싱 |
| `docker/build-push-action@v5` | Docker 빌드/푸시 |
| `aws-actions/configure-aws-credentials@v4` | AWS 인증 |

### 캐싱으로 속도 향상

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-node@v4
    with:
      node-version: '18'
      cache: 'npm'           # npm 캐시 자동 처리

  - run: npm ci
```

---

## 8. 디버깅

### 로그 확인

- Actions 탭에서 워크플로우 실행 기록 확인
- 각 Step의 로그 펼쳐보기

### 디버그 모드

```yaml
steps:
  - run: echo "Debug info"
    env:
      ACTIONS_RUNNER_DEBUG: true
```

### 로컬 테스트 (act)

```bash
# act 설치 (macOS)
brew install act

# 워크플로우 실행
act push

# 특정 Job만 실행
act -j build
```

---

## 명령어/문법 요약

| 키워드 | 설명 |
|--------|------|
| `name` | 워크플로우/스텝 이름 |
| `on` | 트리거 이벤트 |
| `jobs` | 작업 정의 |
| `runs-on` | 실행 환경 |
| `steps` | 단계 정의 |
| `uses` | Action 사용 |
| `run` | 명령어 실행 |
| `with` | Action 파라미터 |
| `env` | 환경 변수 |
| `if` | 조건부 실행 |
| `needs` | Job 의존성 |
| `strategy.matrix` | 매트릭스 빌드 |

---

## 다음 단계

Kubernetes를 배워서 컨테이너 오케스트레이션을 이해해봅시다!
→ [Docker/06_Kubernetes_입문.md](../Docker/06_Kubernetes_입문.md)
