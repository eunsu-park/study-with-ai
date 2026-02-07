# GitHub Actions

## 1. What is GitHub Actions?

GitHub Actions is a **CI/CD automation platform**. It automatically runs workflows based on events like code pushes and PR creation.

### Main Uses

| Use | Examples |
|------|------|
| **CI (Continuous Integration)** | Run tests automatically, linting |
| **CD (Continuous Deployment)** | Auto deployment, Docker image builds |
| **Automation** | Issue labeling, release note generation |

### Core Concepts

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

| Concept | Description |
|------|------|
| **Workflow** | Entire automation process (YAML file) |
| **Event** | Event that triggers workflow |
| **Job** | Group of steps running on same runner |
| **Step** | Individual work unit |
| **Action** | Reusable work unit |
| **Runner** | Server that executes workflow |

---

## 2. Workflow File Structure

Workflows are stored as YAML files in `.github/workflows/` directory.

### Basic Structure

```yaml
# .github/workflows/ci.yml

name: CI Pipeline           # Workflow name

on:                         # Trigger events
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:                       # Job definitions
  build:                    # Job name
    runs-on: ubuntu-latest  # Execution environment

    steps:                  # Steps
      - name: Checkout      # Step name
        uses: actions/checkout@v4

      - name: Run tests
        run: npm test
```

---

## 3. Trigger Events (on)

### push / pull_request

```yaml
on:
  push:
    branches:
      - main
      - 'release/**'
    paths:
      - 'src/**'          # Only when src folder changes
    paths-ignore:
      - '**.md'           # Ignore md file changes

  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]
```

### Manual Execution (workflow_dispatch)

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
```

### Schedule (cron)

```yaml
on:
  schedule:
    - cron: '0 9 * * 1-5'  # Weekdays at 9 AM (UTC)

# cron format: minute hour day month weekday
# 0 9 * * 1-5 = Monday-Friday 09:00
```

### After Another Workflow Completes

```yaml
on:
  workflow_run:
    workflows: ["Build"]
    types: [completed]
```

---

## 4. Jobs Configuration

### Basic Job

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm test
```

### Execution Environment (runs-on)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest      # Ubuntu latest
    # runs-on: ubuntu-22.04     # Specific version
    # runs-on: macos-latest     # macOS
    # runs-on: windows-latest   # Windows
```

### Job Dependencies (needs)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: npm run build

  test:
    needs: build              # Run after build completes
    runs-on: ubuntu-latest
    steps:
      - run: npm test

  deploy:
    needs: [build, test]      # Run after both complete
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

### Parallel Execution

```yaml
jobs:
  test-node-16:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '16'
      - run: npm test

  test-node-18:               # Runs in parallel
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - run: npm test
```

### Matrix Strategy

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

### Conditional Execution (if)

```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh

  notify:
    if: failure()             # Only on failure
    runs-on: ubuntu-latest
    steps:
      - run: echo "Build failed!"
```

---

## 5. Steps Configuration

### Using Actions (uses)

```yaml
steps:
  # Official Action
  - uses: actions/checkout@v4

  # Specific version
  - uses: actions/setup-node@v4
    with:
      node-version: '18'

  # Marketplace Action
  - uses: docker/build-push-action@v5
```

### Running Commands (run)

```yaml
steps:
  # Single command
  - run: npm install

  # Multiple commands
  - run: |
      npm install
      npm run build
      npm test

  # Specify working directory
  - run: npm install
    working-directory: ./frontend

  # Specify shell
  - run: echo "Hello"
    shell: bash
```

### Environment Variables

```yaml
steps:
  - run: echo $MY_VAR
    env:
      MY_VAR: "Hello"

  - run: echo ${{ env.MY_VAR }}
```

### Using Secrets

```yaml
steps:
  - run: echo ${{ secrets.API_KEY }}
    env:
      API_KEY: ${{ secrets.API_KEY }}
```

> **Secrets Setup**: Repository → Settings → Secrets and variables → Actions

---

## 6. Practice Examples

### Example 1: Node.js Test Automation

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

### Example 2: Docker Image Build & Push

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

### Example 3: PR Auto-Labeling

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
# .github/labeler.yml (label rules)

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

### Example 4: Auto Deployment (Vercel)

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

### Example 5: Release Automation

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

## 7. Useful Actions

| Action | Purpose |
|--------|------|
| `actions/checkout@v4` | Code checkout |
| `actions/setup-node@v4` | Node.js setup |
| `actions/setup-python@v5` | Python setup |
| `actions/cache@v4` | Dependency caching |
| `docker/build-push-action@v5` | Docker build/push |
| `aws-actions/configure-aws-credentials@v4` | AWS authentication |

### Speed Up with Caching

```yaml
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-node@v4
    with:
      node-version: '18'
      cache: 'npm'           # npm cache auto-handled

  - run: npm ci
```

---

## 8. Debugging

### View Logs

- Check workflow execution history in Actions tab
- Expand each Step's logs

### Debug Mode

```yaml
steps:
  - run: echo "Debug info"
    env:
      ACTIONS_RUNNER_DEBUG: true
```

### Local Testing (act)

```bash
# Install act (macOS)
brew install act

# Run workflow
act push

# Run specific job only
act -j build
```

---

## Command/Syntax Summary

| Keyword | Description |
|--------|------|
| `name` | Workflow/step name |
| `on` | Trigger events |
| `jobs` | Job definitions |
| `runs-on` | Execution environment |
| `steps` | Step definitions |
| `uses` | Use Action |
| `run` | Run command |
| `with` | Action parameters |
| `env` | Environment variables |
| `if` | Conditional execution |
| `needs` | Job dependencies |
| `strategy.matrix` | Matrix builds |

---

## Next Steps

Let's learn about container orchestration with Kubernetes!
→ [Docker/06_Kubernetes_Introduction.md](../Docker/06_Kubernetes_입문.md)
