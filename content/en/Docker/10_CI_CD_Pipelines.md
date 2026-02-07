# 10. CI/CD Pipelines

## Learning Objectives
- Understanding CI/CD concepts and workflows
- Building automation with GitHub Actions
- Docker image build and registry push
- Implementing Kubernetes automated deployment
- Understanding GitOps patterns

## Table of Contents
1. [CI/CD Overview](#1-cicd-overview)
2. [GitHub Actions Basics](#2-github-actions-basics)
3. [Docker Build Automation](#3-docker-build-automation)
4. [Kubernetes Deployment Automation](#4-kubernetes-deployment-automation)
5. [Advanced Pipelines](#5-advanced-pipelines)
6. [GitOps](#6-gitops)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. CI/CD Overview

### 1.1 CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Continuous Integration (CI)            │   │
│  ├─────────┬─────────┬─────────┬─────────┬─────────┐   │   │
│  │  Code   │  Build  │  Test   │ Analyze │ Artifact│   │   │
│  │  Push   │         │         │         │  Save   │   │   │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘   │   │
│       │         │         │         │         │         │   │
│       ▼         ▼         ▼         ▼         ▼         │   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Continuous Delivery (CD)               │   │
│  ├─────────┬─────────┬─────────┬─────────┐             │   │
│  │ Staging │  E2E    │ Approval│Production              │   │
│  │ Deploy  │  Test   │         │ Deploy  │             │   │
│  └─────────┴─────────┴─────────┴─────────┘             │   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Pipeline Stages

```
┌────────────────────────────────────────────────────────────┐
│                        CI Stage                             │
├────────────────────────────────────────────────────────────┤
│  1. Source Checkout                                        │
│     └─ Fetch code, cache dependencies                     │
│                                                            │
│  2. Build                                                  │
│     └─ Compile, bundle, Docker image build                │
│                                                            │
│  3. Test                                                   │
│     ├─ Unit Test                                          │
│     ├─ Integration Test                                   │
│     └─ Code Coverage                                       │
│                                                            │
│  4. Code Analysis                                          │
│     ├─ Lint (ESLint, pylint, etc.)                       │
│     ├─ Static Analysis (SonarQube)                        │
│     └─ Security Scan (Snyk, Trivy)                        │
│                                                            │
│  5. Artifact Storage                                       │
│     └─ Docker images, binaries, packages                  │
├────────────────────────────────────────────────────────────┤
│                        CD Stage                             │
├────────────────────────────────────────────────────────────┤
│  6. Staging Deployment                                     │
│     └─ Auto deploy to test environment                    │
│                                                            │
│  7. E2E Test                                               │
│     └─ Full system integration test                       │
│                                                            │
│  8. Approval (Optional)                                    │
│     └─ Manual or automatic approval                       │
│                                                            │
│  9. Production Deployment                                  │
│     └─ Rolling update, Blue-Green, Canary                 │
└────────────────────────────────────────────────────────────┘
```

---

## 2. GitHub Actions Basics

### 2.1 Workflow Structure

```yaml
# .github/workflows/ci.yaml
name: CI Pipeline                    # Workflow name

on:                                  # Triggers
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:                 # Manual execution

env:                                 # Global environment variables
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:                                # Job definitions
  build:
    runs-on: ubuntu-latest           # Runner

    steps:                           # Steps
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run tests
      run: npm test
```

### 2.2 Common Actions

```yaml
# .github/workflows/common-actions.yaml
name: Common Actions Example

on: push

jobs:
  example:
    runs-on: ubuntu-latest

    steps:
    # 1. Code checkout
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history (for tags, etc.)

    # 2. Language setup
    - uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - uses: actions/setup-go@v5
      with:
        go-version: '1.21'

    # 3. Caching
    - uses: actions/cache@v4
      with:
        path: ~/.npm
        key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
        restore-keys: |
          ${{ runner.os }}-node-

    # 4. Upload artifact
    - uses: actions/upload-artifact@v4
      with:
        name: build-output
        path: dist/
        retention-days: 7

    # 5. Download artifact
    - uses: actions/download-artifact@v4
      with:
        name: build-output
        path: ./dist

    # 6. Docker setup
    - uses: docker/setup-buildx-action@v3

    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    # 7. Kubernetes setup
    - uses: azure/setup-kubectl@v4
      with:
        version: 'v1.28.0'

    - uses: azure/setup-helm@v4
      with:
        version: 'v3.13.0'
```

### 2.3 Job Dependencies and Matrix

```yaml
# .github/workflows/matrix.yaml
name: Matrix Build

on: push

jobs:
  # Build matrix
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node-version: [18, 20, 22]
        exclude:
          - os: windows-latest
            node-version: 18
        include:
          - os: ubuntu-latest
            node-version: 20
            coverage: true

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}

    - run: npm ci
    - run: npm test

    - name: Upload coverage
      if: matrix.coverage
      uses: codecov/codecov-action@v4

  # Job dependencies
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - run: npm ci
    - run: npm run build

    - uses: actions/upload-artifact@v4
      with:
        name: dist
        path: dist/

  # Conditional execution
  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
    - run: echo "Deploying to production"
```

### 2.4 Secrets and Environment Variables

```yaml
# .github/workflows/secrets.yaml
name: Secrets Example

on: push

jobs:
  deploy:
    runs-on: ubuntu-latest

    # Environment selection (applies GitHub environment protection rules)
    environment:
      name: production
      url: https://myapp.example.com

    env:
      # Normal environment variables
      NODE_ENV: production
      # Secret reference
      DATABASE_URL: ${{ secrets.DATABASE_URL }}

    steps:
    - uses: actions/checkout@v4

    - name: Deploy
      env:
        # Step-level environment variables
        API_KEY: ${{ secrets.API_KEY }}
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        echo "Deploying with secret..."
        # Secrets are masked in logs

    - name: Use GITHUB_TOKEN
      # GITHUB_TOKEN is automatically provided
      env:
        GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        gh release create v1.0.0 --notes "Release notes"
```

---

## 3. Docker Build Automation

### 3.1 Basic Docker Build

```yaml
# .github/workflows/docker-build.yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
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
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 3.2 Multi-Stage Dockerfile

```dockerfile
# Dockerfile
# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies first (caching optimization)
COPY package*.json ./
RUN npm ci --only=production

# Copy source and build
COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine AS production

WORKDIR /app

# Non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Copy build results only
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./

USER nextjs

EXPOSE 3000

ENV NODE_ENV=production

CMD ["node", "dist/main.js"]
```

### 3.3 Security Scanning

```yaml
# .github/workflows/security-scan.yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  # Image vulnerability scan
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Build image
      run: docker build -t myapp:${{ github.sha }} .

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'myapp:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'CRITICAL,HIGH'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'

  # Code vulnerability scan
  codeql:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
    - uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: javascript

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3

  # Dependency scan
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/node@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
```

---

## 4. Kubernetes Deployment Automation

### 4.1 kubectl Deployment

```yaml
# .github/workflows/k8s-deploy.yaml
name: Kubernetes Deploy

on:
  push:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.version }}

    steps:
    - uses: actions/checkout@v4

    - name: Build and push
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

    - name: Extract metadata
      id: meta
      run: echo "version=${{ github.sha }}" >> $GITHUB_OUTPUT

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Update image tag
      run: |
        sed -i "s|IMAGE_TAG|${{ needs.build.outputs.image-tag }}|g" k8s/deployment.yaml

    - name: Deploy to staging
      run: |
        kubectl apply -f k8s/ -n staging
        kubectl rollout status deployment/myapp -n staging --timeout=300s

  deploy-production:
    needs: [build, deploy-staging]
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://myapp.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > ~/.kube/config

    - name: Deploy to production
      run: |
        sed -i "s|IMAGE_TAG|${{ needs.build.outputs.image-tag }}|g" k8s/deployment.yaml
        kubectl apply -f k8s/ -n production
        kubectl rollout status deployment/myapp -n production --timeout=300s
```

### 4.2 Helm Deployment

```yaml
# .github/workflows/helm-deploy.yaml
name: Helm Deploy

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.vars.outputs.tag }}

    steps:
    - uses: actions/checkout@v4

    - name: Set variables
      id: vars
      run: |
        if [[ $GITHUB_REF == refs/tags/* ]]; then
          echo "tag=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT
        else
          echo "tag=${{ github.sha }}" >> $GITHUB_OUTPUT
        fi

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ghcr.io/${{ github.repository }}:${{ steps.vars.outputs.tag }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Setup Helm
      uses: azure/setup-helm@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Deploy with Helm
      run: |
        helm upgrade --install myapp ./charts/myapp \
          --namespace production \
          --create-namespace \
          --set image.tag=${{ needs.build.outputs.image-tag }} \
          --set image.repository=ghcr.io/${{ github.repository }} \
          -f ./charts/myapp/values-prod.yaml \
          --wait \
          --timeout 5m

    - name: Verify deployment
      run: |
        kubectl get pods -n production -l app=myapp
        kubectl rollout status deployment/myapp -n production
```

### 4.3 Kustomize Deployment

```yaml
# .github/workflows/kustomize-deploy.yaml
name: Kustomize Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [staging, production]

    environment: ${{ matrix.environment }}

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Update image tag
      working-directory: k8s/overlays/${{ matrix.environment }}
      run: |
        kustomize edit set image myapp=ghcr.io/${{ github.repository }}:${{ github.sha }}

    - name: Deploy with Kustomize
      run: |
        kubectl apply -k k8s/overlays/${{ matrix.environment }}
        kubectl rollout status deployment/myapp -n ${{ matrix.environment }} --timeout=300s
```

### 4.4 Kustomize Directory Structure

```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── overlays/
    ├── staging/
    │   ├── kustomization.yaml
    │   ├── replica-patch.yaml
    │   └── configmap-patch.yaml
    └── production/
        ├── kustomization.yaml
        ├── replica-patch.yaml
        ├── hpa.yaml
        └── configmap-patch.yaml
```

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml

commonLabels:
  app: myapp

---
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: production

resources:
  - ../../base
  - hpa.yaml

patches:
  - replica-patch.yaml
  - configmap-patch.yaml

images:
  - name: myapp
    newName: ghcr.io/myorg/myapp
    newTag: latest
```

---

## 5. Advanced Pipelines

### 5.1 Complete CI/CD Pipeline

```yaml
# .github/workflows/complete-pipeline.yaml
name: Complete CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # 1. Lint and static analysis
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - run: npm ci
    - run: npm run lint
    - run: npm run type-check

  # 2. Unit tests
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - run: npm ci
    - run: npm test -- --coverage

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        files: ./coverage/lcov.info

  # 3. Integration tests
  integration-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'

    - run: npm ci
    - name: Run integration tests
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost:5432/test
        REDIS_URL: redis://localhost:6379
      run: npm run test:integration

  # 4. Build
  build:
    needs: [lint, test, integration-test]
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.version }}
      image-digest: ${{ steps.build.outputs.digest }}

    permissions:
      contents: read
      packages: write

    steps:
    - uses: actions/checkout@v4

    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Container Registry
      if: github.event_name != 'pull_request'
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
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=sha

    - name: Build and push
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # 5. Security scan
  security-scan:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request'

    steps:
    - name: Run Trivy
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build.outputs.image-digest }}
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'CRITICAL,HIGH'

    - name: Upload scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'

  # 6. Staging deployment
  deploy-staging:
    needs: [build, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.myapp.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Setup Helm
      uses: azure/setup-helm@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Deploy to staging
      run: |
        helm upgrade --install myapp ./charts/myapp \
          --namespace staging \
          --create-namespace \
          --set image.tag=${{ needs.build.outputs.image-tag }} \
          -f ./charts/myapp/values-staging.yaml \
          --wait

  # 7. E2E tests
  e2e-test:
    needs: deploy-staging
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'

    - name: Install Playwright
      run: |
        npm ci
        npx playwright install --with-deps

    - name: Run E2E tests
      env:
        BASE_URL: https://staging.myapp.example.com
      run: npm run test:e2e

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: playwright-report
        path: playwright-report/

  # 8. Production deployment
  deploy-production:
    needs: [build, e2e-test]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment:
      name: production
      url: https://myapp.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Setup Helm
      uses: azure/setup-helm@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > ~/.kube/config

    - name: Deploy to production
      run: |
        helm upgrade --install myapp ./charts/myapp \
          --namespace production \
          --create-namespace \
          --set image.tag=${{ needs.build.outputs.image-tag }} \
          -f ./charts/myapp/values-prod.yaml \
          --wait \
          --timeout 10m

  # 9. Release notes
  release:
    needs: deploy-production
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')

    permissions:
      contents: write

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Generate changelog
      id: changelog
      uses: mikepenz/release-changelog-builder-action@v4
      with:
        configuration: ".github/changelog-config.json"
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

    - name: Create Release
      uses: softprops/action-gh-release@v1
      with:
        body: ${{ steps.changelog.outputs.changelog }}
        draft: false
        prerelease: ${{ contains(github.ref, 'beta') || contains(github.ref, 'alpha') }}
```

### 5.2 Canary Deployment

```yaml
# .github/workflows/canary-deploy.yaml
name: Canary Deployment

on:
  workflow_dispatch:
    inputs:
      canary-weight:
        description: 'Canary traffic percentage (0-100)'
        required: true
        default: '10'
      promote:
        description: 'Promote canary to stable'
        type: boolean
        default: false

jobs:
  canary:
    runs-on: ubuntu-latest
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Setup kubectl
      uses: azure/setup-kubectl@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config

    - name: Deploy Canary
      if: ${{ !inputs.promote }}
      run: |
        # Create Canary Deployment
        helm upgrade --install myapp-canary ./charts/myapp \
          --namespace production \
          --set image.tag=${{ github.sha }} \
          --set replicaCount=1 \
          --set canary.enabled=true \
          --set canary.weight=${{ inputs.canary-weight }} \
          -f ./charts/myapp/values-canary.yaml

    - name: Monitor Canary
      if: ${{ !inputs.promote }}
      run: |
        # Monitor error rate for 5 minutes
        for i in {1..30}; do
          error_rate=$(kubectl exec -n production deploy/prometheus -- \
            promtool query instant 'sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m])) * 100' | jq -r '.data.result[0].value[1]')

          if (( $(echo "$error_rate > 5" | bc -l) )); then
            echo "Error rate too high: $error_rate%. Rolling back."
            helm rollback myapp-canary -n production
            exit 1
          fi

          sleep 10
        done

    - name: Promote Canary
      if: ${{ inputs.promote }}
      run: |
        # Promote Canary to Stable
        helm upgrade --install myapp ./charts/myapp \
          --namespace production \
          --set image.tag=${{ github.sha }} \
          -f ./charts/myapp/values-prod.yaml \
          --wait

        # Delete Canary
        helm uninstall myapp-canary -n production || true
```

---

## 6. GitOps

### 6.1 GitOps Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GitOps Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │  App Repo    │                  │ Config Repo  │        │
│  │ (Source Code)│                  │(K8s Manifests)│       │
│  └──────┬───────┘                  └──────┬───────┘        │
│         │                                  │                │
│         │ 1. Push                          │ 3. Push        │
│         ▼                                  ▼                │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │    CI        │  2. Update image │   GitOps     │        │
│  │  Pipeline    │──────tag────────▶│  Controller  │        │
│  └──────────────┘                  │  (ArgoCD)    │        │
│         │                          └──────┬───────┘        │
│         │ Build                           │ 4. Sync        │
│         ▼                                  ▼                │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │  Container   │                  │  Kubernetes  │        │
│  │  Registry    │◀────Pull─────────│   Cluster    │        │
│  └──────────────┘                  └──────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 ArgoCD Application

```yaml
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default

  source:
    repoURL: https://github.com/myorg/myapp-config
    targetRevision: HEAD
    path: overlays/production

  destination:
    server: https://kubernetes.default.svc
    namespace: production

  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - Validate=true
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

---
# Using Helm chart
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-helm
  namespace: argocd
spec:
  project: default

  source:
    repoURL: https://github.com/myorg/myapp-config
    targetRevision: HEAD
    path: charts/myapp
    helm:
      valueFiles:
        - values-prod.yaml
      parameters:
        - name: image.tag
          value: "v1.0.0"

  destination:
    server: https://kubernetes.default.svc
    namespace: production
```

### 6.3 Update Config Repo from CI

```yaml
# .github/workflows/update-config.yaml
name: Update GitOps Config

on:
  workflow_run:
    workflows: ["Docker Build"]
    types: [completed]
    branches: [main]

jobs:
  update-config:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest

    steps:
    - name: Checkout config repo
      uses: actions/checkout@v4
      with:
        repository: myorg/myapp-config
        token: ${{ secrets.CONFIG_REPO_TOKEN }}
        path: config

    - name: Get image tag
      id: tag
      run: |
        echo "tag=${{ github.event.workflow_run.head_sha }}" >> $GITHUB_OUTPUT

    - name: Update image tag
      working-directory: config
      run: |
        # Using Kustomize
        cd overlays/production
        kustomize edit set image myapp=ghcr.io/myorg/myapp:${{ steps.tag.outputs.tag }}

        # Or using yq
        # yq e '.spec.template.spec.containers[0].image = "ghcr.io/myorg/myapp:${{ steps.tag.outputs.tag }}"' -i deployment.yaml

    - name: Commit and push
      working-directory: config
      run: |
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add .
        git commit -m "Update myapp image to ${{ steps.tag.outputs.tag }}"
        git push
```

---

## 7. Practice Exercises

### Exercise 1: Basic CI Pipeline
```yaml
# Requirements:
# 1. CI pipeline for Node.js project
# 2. Lint, test, build stages
# 3. Run on PR and main branch
# 4. Upload test coverage report

# Write workflow
```

### Exercise 2: Docker Multi-Architecture Build
```yaml
# Requirements:
# 1. Build AMD64, ARM64 images
# 2. Tags: latest, git sha, semver
# 3. Caching configuration
# 4. Security scan

# Write workflow
```

### Exercise 3: Blue-Green Deployment
```yaml
# Requirements:
# 1. Blue/Green environment switching
# 2. Health check before traffic switch
# 3. Rollback capability
# 4. Manual approval stage

# Write workflow
```

### Exercise 4: GitOps Setup
```yaml
# Requirements:
# 1. ArgoCD Application setup
# 2. Auto-update Config Repo from CI
# 3. Automatic sync and self-healing
# 4. Slack notifications

# Write ArgoCD Application and CI workflow
```

---

## Next Steps

- [07_Kubernetes_Security](07_Kubernetes_Security.md) - Review security
- [08_Kubernetes_Advanced](08_Kubernetes_Advanced.md) - Advanced K8s features
- [09_Helm_Package_Management](09_Helm_Package_Management.md) - Helm charts

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [GitOps Principles](https://opengitops.dev/)

---

[← Previous: Helm Package Management](09_Helm_Package_Management.md) | [Table of Contents](00_Overview.md)
