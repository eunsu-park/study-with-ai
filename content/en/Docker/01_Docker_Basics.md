# Docker Basics

## 1. What is Docker?

Docker is a **container-based virtualization platform**. It packages applications and their execution environments so they can run identically anywhere.

### Why use Docker?

**Problem scenario:**
```
Developer A: "It works on my computer?"
Developer B: "I have Node 18 but the server has Node 16..."
Operations team: "Different library versions cause errors"
```

**Docker solution:**
```
Package entire environment in a container → Runs identically everywhere
```

### Advantages of Docker

| Advantage | Description |
|-----------|-------------|
| **Consistency** | Identical dev/test/production environments |
| **Isolation** | Applications run independently |
| **Portability** | Runs identically anywhere |
| **Lightweight** | Faster and lighter than VMs |
| **Version control** | Manage environment versions with images |

---

## 2. Containers vs Virtual Machines (VM)

```
┌────────────────────────────────────────────────────────────┐
│         Virtual Machine (VM)            Container           │
├────────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐ ┌─────┐ ┌─────┐     │
│  │App A│ │App B│ │App C│     │App A│ │App B│ │App C│     │
│  ├─────┤ ├─────┤ ├─────┤     ├─────┴─┴─────┴─┴─────┤     │
│  │Guest│ │Guest│ │Guest│     │     Docker Engine    │     │
│  │ OS  │ │ OS  │ │ OS  │     ├──────────────────────┤     │
│  ├─────┴─┴─────┴─┴─────┤     │       Host OS        │     │
│  │     Hypervisor      │     ├──────────────────────┤     │
│  ├──────────────────────┤     │      Hardware        │     │
│  │       Host OS        │     └──────────────────────┘     │
│  ├──────────────────────┤                                  │
│  │      Hardware        │     ✓ Shares OS → Light & fast  │
│  └──────────────────────┘     ✓ Starts in seconds         │
│  ✗ Each VM needs OS          ✓ Low resource usage         │
│  ✗ Starts in minutes                                       │
│  ✗ High resource usage                                     │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Docker Core Concepts

### Image

- **Template** for creating containers
- Read-only
- Composed of layers

```
┌─────────────────────┐
│   Application       │  ← My application
├─────────────────────┤
│   Node.js 18        │  ← Runtime
├─────────────────────┤
│   Ubuntu 22.04      │  ← Base OS
└─────────────────────┘
       Image layers
```

### Container

- Running **instance** of an image
- Read/write capable
- Runs in isolated environment

```
Image ────▶ Container
(Blueprint)  (Actual building)

One image → Can create multiple containers
```

### Docker Hub

- Docker image repository (like GitHub)
- Provides official images: nginx, node, python, mysql, etc.
- https://hub.docker.com

---

## 4. Installing Docker

### macOS

**Docker Desktop installation (recommended):**
1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Run DMG file
3. Drag to Applications folder
4. Run Docker Desktop

**Install via Homebrew:**
```bash
brew install --cask docker
```

### Windows

1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Run installer
3. Enable WSL 2 backend (recommended)
4. Run Docker Desktop after restart

### Linux (Ubuntu)

```bash
# 1. Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# 2. Install required packages
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release

# 3. Add Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 6. Add user to docker group (to use without sudo)
sudo usermod -aG docker $USER
# Log out and log back in
```

---

## 5. Verify Installation

```bash
# Check Docker version
docker --version
# Output example: Docker version 24.0.7, build afdd53b

# Docker detailed information
docker info

# Run test container
docker run hello-world
```

### hello-world execution result

```
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
 3. The Docker daemon created a new container from that image.
 4. The Docker daemon streamed that output to the Docker client.
...
```

---

## 6. Docker Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  docker run nginx                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Docker    │───▶│   Docker    │───▶│  Docker     │         │
│  │   Client    │    │   Daemon    │    │  Hub        │         │
│  │  (CLI)      │    │  (Server)   │    │ (Image repo)│         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                  │                 │
│                            │   Download image │                 │
│                            │◀─────────────────┘                 │
│                            │                                    │
│                            ▼                                    │
│                     ┌─────────────┐                             │
│                     │  Container  │                             │
│                     │   (nginx)   │                             │
│                     └─────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. Execute **docker run** command
2. Docker Client requests Docker Daemon
3. If image doesn't exist locally, download from Docker Hub
4. Create and run container from image

---

## Practice Examples

### Example 1: Run First Container

```bash
# Run hello-world image
docker run hello-world

# Check running containers
docker ps

# Check all containers (including stopped)
docker ps -a
```

### Example 2: Run Nginx Web Server

```bash
# Run Nginx container (background)
docker run -d -p 8080:80 nginx

# Access in browser at http://localhost:8080

# Check running containers
docker ps

# Stop container
docker stop <container-ID>
```

---

## Command Summary

| Command | Description |
|---------|-------------|
| `docker --version` | Check version |
| `docker info` | Docker detailed information |
| `docker run image` | Run container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |

---

## Next Steps

Let's dive deeper into images and containers in [02_Images_and_Containers.md](./02_Images_and_Containers.md)!
