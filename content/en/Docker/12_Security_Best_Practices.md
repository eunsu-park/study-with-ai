# Security Best Practices

## Learning Objectives
- Understand the container security threat model
- Apply image security best practices and vulnerability scanning
- Implement runtime security controls and least privilege
- Manage secrets securely in containerized applications
- Configure network security and isolation
- Secure container registries with signing and content trust
- Apply Kubernetes security contexts and Pod Security Standards
- Monitor and audit container runtime behavior

## Table of Contents
1. [Container Security Overview](#1-container-security-overview)
2. [Image Security](#2-image-security)
3. [Dockerfile Best Practices](#3-dockerfile-best-practices)
4. [Runtime Security](#4-runtime-security)
5. [Secrets Management](#5-secrets-management)
6. [Network Security](#6-network-security)
7. [Container Registry Security](#7-container-registry-security)
8. [Kubernetes Security Context](#8-kubernetes-security-context)
9. [Monitoring and Auditing](#9-monitoring-and-auditing)
10. [Practice Exercises](#10-practice-exercises)

**Difficulty**: ⭐⭐⭐⭐

---

## 1. Container Security Overview

### Container Threat Model

```
┌─────────────────────────────────────────────────────────────┐
│                 Container Attack Surface                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Supply Chain Attacks                           │    │
│  │     - Malicious base images                        │    │
│  │     - Compromised dependencies                     │    │
│  │     - Vulnerable packages                          │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  2. Image Vulnerabilities                          │    │
│  │     - Known CVEs in OS/libraries                   │    │
│  │     - Outdated software                            │    │
│  │     - Exposed secrets in layers                    │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  3. Runtime Threats                                │    │
│  │     - Container breakout                           │    │
│  │     - Privilege escalation                         │    │
│  │     - Resource abuse                               │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  4. Host/Orchestrator Attacks                      │    │
│  │     - Compromised Docker daemon                    │    │
│  │     - Kubernetes API abuse                         │    │
│  │     - Node compromise                              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Defense in Depth Strategy

```
┌─────────────────────────────────────────────────────────────┐
│               Container Security Layers                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 7: Monitoring & Incident Response                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Runtime detection, Audit logs, Alerting           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 6: Network Security                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Network policies, TLS, Firewalls                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 5: Secrets Management                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Vault, Docker secrets, Encrypted storage          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 4: Runtime Security                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Capabilities, Seccomp, AppArmor, SELinux          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 3: Image Security                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Scanning, Signing, Minimal base, No root          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 2: Host Security                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  OS hardening, CIS benchmarks, Updates             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 1: Infrastructure Security                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  IAM, VPC isolation, Physical security             │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Security Principles

1. **Least Privilege**: Run with minimum necessary permissions
2. **Defense in Depth**: Multiple layers of security controls
3. **Immutability**: Treat containers as immutable artifacts
4. **Minimal Attack Surface**: Reduce exposed components
5. **Shift Left**: Security early in development lifecycle
6. **Zero Trust**: Verify everything, trust nothing

---

## 2. Image Security

### Choosing Secure Base Images

```dockerfile
# ❌ BAD: Large attack surface, many vulnerabilities
FROM ubuntu:latest

# ❌ BAD: "latest" tag is not reproducible
FROM node:latest

# ✅ GOOD: Minimal base image
FROM alpine:3.19

# ✅ GOOD: Distroless (no shell, package manager)
FROM gcr.io/distroless/base-debian12

# ✅ GOOD: Specific version tag for reproducibility
FROM node:18.19-alpine3.19

# ✅ BEST: Digest pinning for immutability
FROM node:18.19-alpine3.19@sha256:abc123...
```

### Multi-Stage Builds for Minimal Production Images

```dockerfile
# ❌ BAD: Build tools in production image
FROM golang:1.21
WORKDIR /app
COPY . .
RUN go build -o myapp
CMD ["./myapp"]

# ✅ GOOD: Multi-stage build
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN go build -o myapp

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/myapp /myapp
USER 1000
CMD ["/myapp"]

# ✅ BEST: Distroless final image
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o myapp

FROM gcr.io/distroless/static-debian12
COPY --from=builder /app/myapp /myapp
USER nonroot:nonroot
CMD ["/myapp"]
```

### Image Scanning with Trivy

```bash
# Install Trivy
# macOS
brew install trivy

# Linux
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy

# Scan Docker image
trivy image nginx:latest

# Output (truncated):
# nginx:latest (debian 12.4)
# Total: 85 (CRITICAL: 12, HIGH: 23, MEDIUM: 35, LOW: 15)
#
# ├── libssl3 (CVE-2023-12345) CRITICAL
# │   Installed Version: 3.0.11-1~deb12u1
# │   Fixed Version: 3.0.11-1~deb12u2
# └── ...

# Scan with severity filter
trivy image --severity CRITICAL,HIGH nginx:latest

# Scan and exit with error if vulnerabilities found
trivy image --exit-code 1 --severity CRITICAL myapp:latest

# Scan for secrets in image
trivy image --scanners secret nginx:latest

# Generate JSON report
trivy image -f json -o report.json nginx:latest
```

### Scanning with Snyk

```bash
# Install Snyk
npm install -g snyk

# Authenticate
snyk auth

# Scan image
snyk container test nginx:latest

# Scan and monitor
snyk container monitor nginx:latest

# Scan Dockerfile
snyk container test nginx:latest --file=Dockerfile
```

### Automated Scanning in CI/CD

```yaml
# .github/workflows/scan.yml
name: Security Scan

on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run Trivy scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
```

---

## 3. Dockerfile Best Practices

### Running as Non-Root User

```dockerfile
# ❌ BAD: Running as root (default)
FROM nginx:alpine
COPY app /usr/share/nginx/html

# ✅ GOOD: Create and use non-root user
FROM nginx:alpine
RUN addgroup -g 1000 appgroup && \
    adduser -D -u 1000 -G appgroup appuser
USER appuser
COPY app /usr/share/nginx/html

# ✅ GOOD: Use numeric UID (works better in Kubernetes)
FROM node:18-alpine
RUN addgroup -g 1001 nodegroup && \
    adduser -D -u 1001 -G nodegroup nodeuser
USER 1001
COPY --chown=1001:1001 . /app
WORKDIR /app
CMD ["node", "server.js"]

# ✅ BEST: Distroless with non-root
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM gcr.io/distroless/nodejs18-debian12
COPY --from=builder /app/node_modules /app/node_modules
COPY --chown=nonroot:nonroot . /app
WORKDIR /app
USER nonroot
CMD ["server.js"]
```

### COPY vs ADD

```dockerfile
# ❌ BAD: ADD has unexpected behavior (auto-extraction)
ADD archive.tar.gz /app/
ADD https://example.com/file.txt /app/

# ✅ GOOD: Use COPY for local files
COPY archive.tar.gz /app/
COPY src/ /app/src/

# ✅ GOOD: Explicit tar extraction if needed
RUN wget https://example.com/archive.tar.gz && \
    tar xzf archive.tar.gz && \
    rm archive.tar.gz
```

### Secret Handling

```dockerfile
# ❌ BAD: Secrets in environment variables
FROM alpine
ENV API_KEY=sk-1234567890abcdef
CMD ["./app"]

# ❌ BAD: Secrets in build args (exposed in history)
FROM alpine
ARG API_KEY
RUN curl -H "Authorization: Bearer $API_KEY" https://api.example.com
CMD ["./app"]

# ✅ GOOD: Use Docker BuildKit secrets
# syntax=docker/dockerfile:1.4
FROM alpine
RUN --mount=type=secret,id=api_key \
    API_KEY=$(cat /run/secrets/api_key) && \
    curl -H "Authorization: Bearer $API_KEY" https://api.example.com
CMD ["./app"]

# Build with:
# docker buildx build --secret id=api_key,src=./api_key.txt -t myapp .

# ✅ GOOD: Runtime secrets with Docker Swarm
# Secrets mounted at /run/secrets/<secret_name>
FROM alpine
CMD ["sh", "-c", "API_KEY=$(cat /run/secrets/api_key) ./app"]
```

### Using .dockerignore

```bash
# .dockerignore
# Prevent sensitive files from being copied into image

# Git
.git
.gitignore

# Secrets
.env
.env.*
*.key
*.pem
secrets/
credentials.json

# Build artifacts
node_modules/
dist/
build/
*.log

# Documentation
README.md
docs/
*.md

# CI/CD
.github/
.gitlab-ci.yml
Jenkinsfile

# IDE
.vscode/
.idea/
*.swp
```

### Layer Optimization

```dockerfile
# ❌ BAD: Many layers, cache invalidation
FROM node:18-alpine
COPY package.json .
RUN npm install
COPY src/ ./src/
RUN npm run build
RUN npm prune --production

# ✅ GOOD: Optimized layer caching
FROM node:18-alpine AS builder
WORKDIR /app
# Cache dependencies separately
COPY package*.json ./
RUN npm ci
# Copy source and build
COPY . .
RUN npm run build && npm prune --production

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
USER node
CMD ["node", "dist/server.js"]
```

### Hardened Dockerfile Example

```dockerfile
# syntax=docker/dockerfile:1.4
FROM golang:1.21-alpine3.19 AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /build

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build with security flags
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build \
    -ldflags="-w -s -extldflags '-static'" \
    -a \
    -o app \
    ./cmd/server

# Production image
FROM scratch

# Copy necessary files from builder
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /build/app /app

# Use non-root user (UID 65534 = nobody)
USER 65534:65534

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app", "healthcheck"]

# Run application
ENTRYPOINT ["/app"]
```

---

## 4. Runtime Security

### Dropping Capabilities

Linux capabilities provide fine-grained privilege control.

```bash
# ❌ BAD: Running with all capabilities
docker run --privileged myapp

# ✅ GOOD: Drop all capabilities, add only needed ones
docker run \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  myapp

# List of common capabilities:
# NET_BIND_SERVICE - Bind to ports < 1024
# CHOWN - Change file ownership
# DAC_OVERRIDE - Bypass file permission checks
# SETUID/SETGID - Change UID/GID
# NET_ADMIN - Network configuration
# SYS_TIME - Set system clock
```

### Docker Compose with Capabilities

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: nginx:alpine
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
      - CHOWN
      - SETUID
      - SETGID
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /var/run
      - /var/cache/nginx
      - /tmp
```

### Read-Only Filesystem

```bash
# ❌ BAD: Writable filesystem (default)
docker run myapp

# ✅ GOOD: Read-only root filesystem
docker run --read-only myapp

# ✅ GOOD: Read-only with tmpfs for writable dirs
docker run \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  myapp
```

```yaml
# docker-compose.yml
services:
  app:
    image: myapp:latest
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=64m
      - /var/run:noexec,nosuid,size=64m
```

### Seccomp Profiles

Seccomp (Secure Computing Mode) restricts system calls.

```json
// seccomp-profile.json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_AARCH64"
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "bind",
        "connect",
        "socket",
        "read",
        "write",
        "open",
        "close",
        "stat",
        "fstat",
        "mmap",
        "mprotect",
        "rt_sigaction",
        "rt_sigreturn",
        "futex",
        "exit_group"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

```bash
# Run with custom seccomp profile
docker run \
  --security-opt seccomp=seccomp-profile.json \
  myapp

# Disable seccomp (not recommended in production)
docker run --security-opt seccomp=unconfined myapp
```

### AppArmor Profiles

```bash
# Check AppArmor status
sudo aa-status

# Docker's default AppArmor profile: docker-default
# Located at: /etc/apparmor.d/docker

# Run with custom AppArmor profile
docker run \
  --security-opt apparmor=docker-custom \
  myapp

# Disable AppArmor (not recommended)
docker run --security-opt apparmor=unconfined myapp
```

### Security Options Example

```yaml
# docker-compose.yml
version: '3.8'

services:
  secure-app:
    image: myapp:latest

    # Drop all capabilities
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

    # Read-only filesystem
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,nodev,size=64m

    # Security options
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
      - seccomp:seccomp-profile.json

    # User
    user: "1000:1000"

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

    # Prevent privilege escalation
    privileged: false
```

---

## 5. Secrets Management

### Docker Secrets (Swarm Mode)

```bash
# Create secret from file
echo "my-db-password" | docker secret create db_password -

# Create secret from stdin
docker secret create api_key api_key.txt

# List secrets
docker secret ls

# Inspect secret (content not shown)
docker secret inspect db_password

# Use secret in service
docker service create \
  --name web \
  --secret db_password \
  --secret api_key \
  myapp:latest

# Secret available at /run/secrets/<secret_name>
```

```yaml
# docker-compose.yml (Swarm stack)
version: '3.8'

services:
  app:
    image: myapp:latest
    secrets:
      - db_password
      - api_key
    environment:
      # Reference secret file in env var
      - DB_PASSWORD_FILE=/run/secrets/db_password
    deploy:
      replicas: 3

secrets:
  db_password:
    external: true
  api_key:
    file: ./api_key.txt
```

```python
# app.py - Reading secrets
import os

def get_secret(secret_name):
    """Read secret from file."""
    secret_path = f'/run/secrets/{secret_name}'
    try:
        with open(secret_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to environment variable (dev only)
        return os.getenv(secret_name.upper())

db_password = get_secret('db_password')
api_key = get_secret('api_key')
```

### Environment Variable Pitfalls

```bash
# ❌ BAD: Secrets in environment variables (visible in docker inspect)
docker run -e DB_PASSWORD=secret123 myapp

# ❌ BAD: Secrets in Dockerfile
ENV API_KEY=sk-1234567890

# ❌ BAD: Secrets visible in process list
docker run myapp --api-key=sk-1234567890

# ✅ GOOD: Secrets in files
docker run -v /path/to/secrets:/secrets:ro myapp

# ✅ GOOD: Docker secrets (Swarm)
docker service create --secret db_password myapp

# ✅ BETTER: External secret manager
docker run -e VAULT_ADDR=https://vault.example.com myapp
```

### External Secret Managers

#### HashiCorp Vault

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    image: myapp:latest
    environment:
      - VAULT_ADDR=https://vault.example.com
      - VAULT_TOKEN_FILE=/run/secrets/vault-token
    secrets:
      - vault-token
    command: sh -c "
      export VAULT_TOKEN=$(cat /run/secrets/vault-token) &&
      export DB_PASSWORD=$(vault kv get -field=password secret/db) &&
      ./app
      "

secrets:
  vault-token:
    file: ./vault-token.txt
```

#### AWS Secrets Manager

```python
# app.py
import boto3
import json

def get_secret(secret_name):
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)

    if 'SecretString' in response:
        return json.loads(response['SecretString'])
    else:
        return base64.b64decode(response['SecretBinary'])

# Retrieve database credentials
db_creds = get_secret('prod/db/credentials')
db_password = db_creds['password']
```

```dockerfile
# Dockerfile
FROM python:3.11-slim
RUN pip install boto3
COPY app.py /app/
WORKDIR /app

# IAM role attached to ECS task or EC2 instance
# No hardcoded credentials needed
CMD ["python", "app.py"]
```

### Kubernetes Secrets

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  # Base64 encoded
  password: cGFzc3dvcmQxMjM=

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        # Or mount as file
        volumeMounts:
        - name: secrets
          mountPath: /secrets
          readOnly: true
      volumes:
      - name: secrets
        secret:
          secretName: db-secret
```

---

## 6. Network Security

### Network Isolation

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    image: nginx
    networks:
      - public
    ports:
      - "80:80"

  backend:
    image: api:latest
    networks:
      - public
      - private

  database:
    image: postgres
    networks:
      - private
    # Database not exposed to public network

networks:
  public:
    driver: bridge
  private:
    driver: bridge
    internal: true  # No external access
```

### TLS Encryption

```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    networks:
      - frontend

  app:
    image: myapp:latest
    environment:
      - TLS_CERT=/certs/server.crt
      - TLS_KEY=/certs/server.key
    volumes:
      - ./certs:/certs:ro
    networks:
      - frontend
```

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name example.com;

    # TLS configuration
    ssl_certificate /etc/nginx/certs/server.crt;
    ssl_certificate_key /etc/nginx/certs/server.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://app:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Egress Control

```bash
# Block all egress except specific destinations
docker network create \
  --internal \
  isolated-net

docker run -d \
  --name proxy \
  --network isolated-net \
  squid

# Configure Squid to allow only specific domains
# /etc/squid/squid.conf
# acl allowed_sites dstdomain .example.com .api.trusted.com
# http_access allow allowed_sites
# http_access deny all
```

### Kubernetes Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-db
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

---

## 7. Container Registry Security

### Image Signing with Docker Content Trust

```bash
# Enable Docker Content Trust
export DOCKER_CONTENT_TRUST=1

# Generate root and repository keys
docker trust key generate mykey

# Add signer to repository
docker trust signer add --key mykey.pub myuser myrepo/myimage

# Push signed image
docker push myrepo/myimage:v1.0
# Signing and pushing trust metadata...

# Pull signed image (verification automatic with DCT enabled)
docker pull myrepo/myimage:v1.0
# Pull (1 of 1): myrepo/myimage:v1.0@sha256:abc...
# Tagging myrepo/myimage:v1.0@sha256:abc... as myrepo/myimage:v1.0
```

### Notary for Image Signing

```bash
# Install Notary
go install github.com/notaryproject/notation/cmd/notation@latest

# Initialize repository
notary init myrepo/myimage

# Sign image
notary addhash myrepo/myimage v1.0 sha256:abc123...

# List trusted tags
notary list myrepo/myimage

# Verify signature
notary verify myrepo/myimage v1.0
```

### Private Registry with Authentication

```yaml
# docker-compose.yml
version: '3.8'

services:
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    environment:
      REGISTRY_AUTH: htpasswd
      REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
      REGISTRY_AUTH_HTPASSWD_REALM: Registry Realm
      REGISTRY_HTTP_TLS_CERTIFICATE: /certs/domain.crt
      REGISTRY_HTTP_TLS_KEY: /certs/domain.key
    volumes:
      - ./auth:/auth
      - ./certs:/certs
      - registry-data:/var/lib/registry

volumes:
  registry-data:
```

```bash
# Create htpasswd file
docker run --rm --entrypoint htpasswd \
  httpd:2 -Bbn myuser mypassword > auth/htpasswd

# Login to private registry
docker login localhost:5000
# Username: myuser
# Password: mypassword

# Tag and push image
docker tag myapp:latest localhost:5000/myapp:latest
docker push localhost:5000/myapp:latest
```

### Harbor Registry

Harbor provides enterprise-grade registry with security features:

```yaml
# docker-compose.yml (simplified Harbor setup)
version: '3.8'

services:
  harbor-core:
    image: goharbor/harbor-core:v2.9.0
    environment:
      - CORE_SECRET=not-a-secure-secret
      - JOBSERVICE_SECRET=not-a-secure-secret
    depends_on:
      - harbor-db
      - redis

  harbor-db:
    image: goharbor/harbor-db:v2.9.0
    environment:
      - POSTGRES_PASSWORD=root123

  redis:
    image: goharbor/redis-photon:v2.9.0

  harbor-portal:
    image: goharbor/harbor-portal:v2.9.0
    ports:
      - "80:8080"
```

**Harbor Features**:
- Vulnerability scanning (Trivy, Clair)
- Image signing (Notary)
- RBAC and multi-tenancy
- Replication across registries
- Audit logging
- Quota management

---

## 8. Kubernetes Security Context

### Pod Security Context

```yaml
# secure-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    # Pod-level security context
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault

  containers:
  - name: app
    image: myapp:latest
    securityContext:
      # Container-level security context (overrides pod-level)
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 1000
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE

    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache

  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir: {}
```

### Pod Security Standards

Kubernetes defines three security levels:

1. **Privileged**: Unrestricted (not recommended)
2. **Baseline**: Minimally restrictive
3. **Restricted**: Heavily restricted (best practice)

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### Restricted Pod Example

```yaml
# restricted-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: app
        image: myapp:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL

        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
          requests:
            cpu: "100m"
            memory: "128Mi"

        volumeMounts:
        - name: tmp
          mountPath: /tmp

        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 100Mi
```

### RBAC for Pods

```yaml
# service-account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: myapp-sa
  namespace: production

---
# role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: myapp-role
  namespace: production
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]

---
# role-binding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: myapp-rolebinding
  namespace: production
subjects:
- kind: ServiceAccount
  name: myapp-sa
  namespace: production
roleRef:
  kind: Role
  name: myapp-role
  apiGroup: rbac.authorization.k8s.io

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  template:
    spec:
      serviceAccountName: myapp-sa
      automountServiceAccountToken: false  # Disable if not needed
      containers:
      - name: app
        image: myapp:latest
```

---

## 9. Monitoring and Auditing

### Falco for Runtime Anomaly Detection

```yaml
# falco-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: falco
  namespace: falco
spec:
  selector:
    matchLabels:
      app: falco
  template:
    metadata:
      labels:
        app: falco
    spec:
      serviceAccountName: falco
      hostNetwork: true
      hostPID: true
      containers:
      - name: falco
        image: falcosecurity/falco:0.36.0
        securityContext:
          privileged: true
        volumeMounts:
        - name: docker-socket
          mountPath: /var/run/docker.sock
        - name: dev
          mountPath: /host/dev
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: boot
          mountPath: /host/boot
          readOnly: true
        - name: lib-modules
          mountPath: /host/lib/modules
          readOnly: true
        - name: usr
          mountPath: /host/usr
          readOnly: true
        - name: etc
          mountPath: /host/etc
          readOnly: true
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
      - name: dev
        hostPath:
          path: /dev
      - name: proc
        hostPath:
          path: /proc
      - name: boot
        hostPath:
          path: /boot
      - name: lib-modules
        hostPath:
          path: /lib/modules
      - name: usr
        hostPath:
          path: /usr
      - name: etc
        hostPath:
          path: /etc
```

**Falco Rules** (`/etc/falco/falco_rules.yaml`):

```yaml
# Detect shell in container
- rule: Terminal shell in container
  desc: A shell was spawned in a container
  condition: >
    spawned_process and
    container and
    shell_procs and
    proc.tty != 0
  output: >
    Shell spawned in container
    (user=%user.name container=%container.name
    shell=%proc.name parent=%proc.pname
    cmdline=%proc.cmdline)
  priority: WARNING

# Detect file modification in /etc
- rule: Write to /etc directory
  desc: File was written to /etc directory
  condition: >
    evt.type in (write, open) and
    evt.dir = < and
    fd.name startswith /etc
  output: >
    File written in /etc
    (user=%user.name file=%fd.name
    command=%proc.cmdline container=%container.name)
  priority: ERROR

# Detect privilege escalation
- rule: Set Setuid or Setgid bit
  desc: An attempt to set setuid or setgid bit
  condition: >
    evt.type = chmod or evt.type = fchmod and
    ((evt.arg.mode contains S_ISUID) or
     (evt.arg.mode contains S_ISGID))
  output: >
    Setuid or setgid bit set
    (user=%user.name file=%evt.arg.filename
    mode=%evt.arg.mode container=%container.name)
  priority: CRITICAL
```

### Docker Bench Security

```bash
# Run Docker Bench Security
docker run -it --rm \
  --net host \
  --pid host \
  --userns host \
  --cap-add audit_control \
  -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
  -v /etc:/etc:ro \
  -v /usr/bin/containerd:/usr/bin/containerd:ro \
  -v /usr/bin/runc:/usr/bin/runc:ro \
  -v /usr/lib/systemd:/usr/lib/systemd:ro \
  -v /var/lib:/var/lib:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  --label docker_bench_security \
  docker/docker-bench-security

# Output:
# [INFO] 1 - Host Configuration
# [PASS] 1.1.1 - Ensure a separate partition for containers...
# [WARN] 1.1.2 - Ensure only trusted users are allowed...
# ...
# [INFO] 4 - Container Images and Build File
# [WARN] 4.1 - Ensure a user for the container has been created
# [PASS] 4.5 - Ensure Content trust for Docker is Enabled
```

### Audit Logging

#### Docker Daemon Audit

```json
// /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "audit-log-enabled": true,
  "audit-log-path": "/var/log/docker-audit.log",
  "audit-log-max-size": "100m",
  "audit-log-max-backups": "5"
}
```

#### Kubernetes Audit Policy

```yaml
# audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
# Log pod changes at RequestResponse level
- level: RequestResponse
  resources:
  - group: ""
    resources: ["pods"]

# Log secret access at Metadata level
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets"]

# Don't log read-only requests
- level: None
  verbs: ["get", "list", "watch"]
```

### Monitoring with Prometheus

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

# Alerting rules
rule_files:
  - '/etc/prometheus/alerts.yml'
```

```yaml
# alerts.yml
groups:
- name: container_security
  rules:
  - alert: ContainerRunningAsRoot
    expr: container_running_as_root == 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Container running as root user"

  - alert: HighCPUUsage
    expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Container CPU usage above 80%"
```

---

## 10. Practice Exercises

### Exercise 1: Secure Dockerfile

Transform an insecure Dockerfile into a secure one.

**Insecure Dockerfile**:
```dockerfile
FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
ENV SECRET_KEY=mysecretkey123
CMD ["python3", "app.py"]
```

**Tasks**:
1. Use minimal base image
2. Implement multi-stage build
3. Run as non-root user
4. Remove hardcoded secret
5. Add security scanning
6. Implement read-only filesystem

### Exercise 2: Runtime Security Configuration

Create a secure Docker Compose configuration.

```yaml
# TODO: Harden this configuration
version: '3.8'

services:
  web:
    image: nginx:latest
    ports:
      - "80:80"

  app:
    build: .
    environment:
      - DB_PASSWORD=password123

  db:
    image: postgres:latest
```

**Tasks**:
1. Add capability dropping
2. Implement read-only filesystem
3. Configure proper user
4. Remove environment variable secrets
5. Add network isolation
6. Implement resource limits

### Exercise 3: Kubernetes Security Context

Create a deployment with restricted Pod Security Standard.

```yaml
# TODO: Add security context
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:latest
        ports:
        - containerPort: 8080
```

**Tasks**:
1. Add pod-level security context
2. Add container-level security context
3. Drop all capabilities
4. Implement read-only root filesystem
5. Add resource limits
6. Configure proper probes

### Exercise 4: Image Scanning Pipeline

Create a CI/CD pipeline that scans images for vulnerabilities.

**GitHub Actions Workflow**:
```yaml
# TODO: Complete this workflow
name: Security Scan

on: [push]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      # TODO: Add Trivy scanning
      # TODO: Fail on CRITICAL vulnerabilities
      # TODO: Upload results to GitHub Security
```

### Exercise 5: Secret Management

Implement proper secret management for a multi-container application.

**Scenario**: A web application needs:
- Database password
- API key for external service
- TLS certificates

**Tasks**:
1. Implement Docker Swarm secrets
2. Create application code to read secrets from files
3. Configure proper file permissions
4. Implement secret rotation strategy
5. Document secret lifecycle

### Exercise 6: Network Security

Implement network isolation for a three-tier application.

```yaml
# TODO: Add network security
version: '3.8'

services:
  frontend:
    image: nginx
    ports:
      - "443:443"

  backend:
    image: api:latest

  database:
    image: postgres

  cache:
    image: redis
```

**Tasks**:
1. Create isolated networks
2. Ensure frontend cannot reach database directly
3. Implement internal network for backend services
4. Configure TLS for all communications
5. Add network policies (if using Kubernetes)

---

## Summary

In this lesson, you learned:

- Container security threat model and defense in depth strategy
- Image security: minimal base images, multi-stage builds, vulnerability scanning
- Dockerfile best practices: non-root users, secret handling, layer optimization
- Runtime security: capabilities, seccomp, AppArmor, read-only filesystems
- Secrets management: Docker secrets, external vaults, Kubernetes secrets
- Network security: isolation, TLS, egress control, network policies
- Registry security: image signing, content trust, private registries
- Kubernetes security context: Pod Security Standards, RBAC, restricted pods
- Monitoring and auditing: Falco, Docker Bench, audit logging, Prometheus

**Key Takeaways**:
- Security is a continuous process, not a one-time configuration
- Apply defense in depth: multiple layers of security controls
- Shift left: integrate security early in development lifecycle
- Least privilege: run with minimum necessary permissions
- Monitor and audit: detect and respond to security incidents
- Automate security scanning in CI/CD pipelines

**Production Checklist**:
- [ ] Use minimal base images (Alpine, distroless)
- [ ] Scan images for vulnerabilities (Trivy, Snyk)
- [ ] Run containers as non-root users
- [ ] Drop unnecessary capabilities
- [ ] Implement read-only filesystems
- [ ] Use secrets management (Vault, Docker secrets)
- [ ] Enable network isolation and encryption
- [ ] Sign and verify images (Content Trust)
- [ ] Apply Pod Security Standards (Kubernetes)
- [ ] Monitor runtime behavior (Falco)
- [ ] Implement audit logging
- [ ] Regular security reviews and updates

**Further Reading**:
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [NIST Application Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---

[Previous: 11_Container_Networking](./11_Container_Networking.md) | [Next: 00_Overview](./00_Overview.md)
