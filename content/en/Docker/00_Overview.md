# Docker & Kubernetes Learning Guide

## Introduction

This folder contains learning materials for Docker and Kubernetes. Learn step-by-step from basic container concepts to orchestration.

**Target Audience**: Developers, DevOps beginners

---

## Learning Roadmap

```
[Docker Basics]           [Docker Advanced]        [Orchestration]
     │                          │                       │
     ▼                          ▼                       ▼
Docker Basics ──────▶ Dockerfile ──────▶ Kubernetes Intro
     │                   │
     ▼                   ▼
Images/Containers ──▶ Docker Compose
                         │
                         ▼
                    Practical Examples
```

---

## Prerequisites

- Basic Linux commands
- Terminal/shell usage experience
- Basic understanding of web applications (recommended)

---

## File List

| File | Difficulty | Main Topics |
|------|------------|-------------|
| [01_Docker_Basics.md](./01_Docker_Basics.md) | ⭐ | Docker concepts, installation, basic commands |
| [02_Images_and_Containers.md](./02_Images_and_Containers.md) | ⭐ | Image management, container execution/management |
| [03_Dockerfile.md](./03_Dockerfile.md) | ⭐⭐ | Dockerfile creation, image building |
| [04_Docker_Compose.md](./04_Docker_Compose.md) | ⭐⭐ | Multi-container, docker-compose.yml |
| [05_Practical_Examples.md](./05_Practical_Examples.md) | ⭐⭐⭐ | Web app containerization, DB integration |
| [06_Kubernetes_Intro.md](./06_Kubernetes_Intro.md) | ⭐⭐⭐ | K8s concepts, Pod, Deployment, Service |
| [07_Kubernetes_Security.md](./07_Kubernetes_Security.md) | ⭐⭐⭐⭐ | RBAC, NetworkPolicy, Secrets, PodSecurity |
| [08_Kubernetes_Advanced.md](./08_Kubernetes_Advanced.md) | ⭐⭐⭐⭐ | Ingress, StatefulSet, DaemonSet, PV/PVC |
| [09_Helm_Package_Management.md](./09_Helm_Package_Management.md) | ⭐⭐⭐ | Helm charts, values.yaml, release management |
| [10_CI_CD_Pipelines.md](./10_CI_CD_Pipelines.md) | ⭐⭐⭐⭐ | GitHub Actions, image building, K8s deployment |
| [11_Container_Networking.md](./11_Container_Networking.md) | ⭐⭐⭐ | Bridge, host, overlay, macvlan, DNS, port mapping |
| [12_Security_Best_Practices.md](./12_Security_Best_Practices.md) | ⭐⭐⭐⭐ | Image scanning, rootless, seccomp, secrets, Falco |

---

## Recommended Learning Order

### Step 1: Docker Basics
1. Docker Basics → Images and Containers

### Step 2: Docker Application
2. Dockerfile → Docker Compose → Practical Examples

### Step 3: Orchestration
3. Kubernetes Intro → Kubernetes Security → Kubernetes Advanced

### Step 4: Deployment Automation
4. Helm Package Management → CI/CD Pipelines

---

## Practice Environment

### Docker Installation

```bash
# macOS
brew install --cask docker

# Ubuntu
sudo apt-get install docker.io

# Verify installation
docker --version
```

### Practice Examples

```bash
# Hello World
docker run hello-world

# Nginx web server
docker run -d -p 8080:80 nginx
```

---

## Related Materials

- [Git Learning](../Git/00_Overview.md) - Code version control
- [PostgreSQL Learning](../PostgreSQL/00_Overview.md) - Database (used with Docker)
