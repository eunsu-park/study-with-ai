# Docker Images and Containers

## 1. Docker Images

### What is an Image?

- **Read-only template** for creating containers
- Includes application + execution environment
- Efficiently stored in layer structure

### Image Name Structure

```
[registry/]repository:tag

Examples:
nginx                    → nginx:latest (default)
nginx:1.25              → specific version
node:18-alpine          → Node 18, Alpine Linux based
myname/myapp:v1.0       → user image
gcr.io/project/app:tag  → Google Container Registry
```

| Component | Description | Example |
|-----------|-------------|---------|
| Registry | Image repository | docker.io, gcr.io |
| Repository | Image name | nginx, node |
| Tag | Version | latest, 1.25, alpine |

---

## 2. Image Management Commands

### Search Images

```bash
# Search on Docker Hub
docker search nginx

# Output example:
# NAME          DESCRIPTION                 STARS   OFFICIAL
# nginx         Official build of Nginx     18000   [OK]
# bitnami/nginx Bitnami nginx Docker Image  150
```

### Download Images (Pull)

```bash
# Download latest version
docker pull nginx

# Download specific version
docker pull nginx:1.25

# Download specific tag
docker pull node:18-alpine
```

### List Images

```bash
# List local images
docker images

# Output example:
# REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
# nginx        latest    a6bd71f48f68   2 days ago     187MB
# node         18-alpine 5d5f5d5f5d5f   1 week ago     175MB
```

### Delete Images

```bash
# Delete image
docker rmi nginx

# Delete by image ID
docker rmi a6bd71f48f68

# Force delete (image in use)
docker rmi -f nginx

# Delete all unused images
docker image prune

# Delete all images (caution!)
docker rmi $(docker images -q)
```

### Image Details

```bash
# Image detailed information
docker inspect nginx

# Image history (check layers)
docker history nginx
```

---

## 3. Running Containers

### Basic Execution

```bash
# Basic run
docker run nginx

# Background execution (-d: detached)
docker run -d nginx

# Specify name
docker run -d --name my-nginx nginx

# Auto-remove after exit (--rm)
docker run --rm nginx
```

### Port Mapping (-p)

```bash
# Map host:container ports
docker run -d -p 8080:80 nginx

# Multiple port mappings
docker run -d -p 8080:80 -p 8443:443 nginx

# Random port mapping
docker run -d -P nginx
```

```
┌─────────────────────────────────────────────────────┐
│  Host (my computer)                                  │
│                                                     │
│  localhost:8080 ──────────────┐                     │
│                               │                     │
│  ┌────────────────────────────▼────────────────┐   │
│  │           Container (nginx)                  │   │
│  │                                             │   │
│  │           :80 (nginx default port)          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Environment Variables (-e)

```bash
# Set environment variable
docker run -d -e MYSQL_ROOT_PASSWORD=secret mysql

# Multiple environment variables
docker run -d \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=mydb \
  mysql
```

### Volume Mounting (-v)

```bash
# Mount host directory
docker run -d -v /host/path:/container/path nginx

# Mount current directory
docker run -d -v $(pwd):/app node

# Read-only mount
docker run -d -v /host/path:/container/path:ro nginx

# Named Volume
docker run -d -v mydata:/var/lib/mysql mysql
```

### Interactive Mode (-it)

```bash
# Access container shell
docker run -it ubuntu bash

# Inside container:
# root@container:/# ls
# root@container:/# exit
```

---

## 4. Container Management

### List Containers

```bash
# Running containers
docker ps

# All containers (including stopped)
docker ps -a

# Container IDs only
docker ps -q

# Output example:
# CONTAINER ID   IMAGE   COMMAND                  STATUS          PORTS                  NAMES
# abc123def456   nginx   "/docker-entrypoint.…"   Up 2 hours      0.0.0.0:8080->80/tcp   my-nginx
```

### Start/Stop/Restart Containers

```bash
# Stop
docker stop my-nginx

# Start (stopped container)
docker start my-nginx

# Restart
docker restart my-nginx

# Force kill
docker kill my-nginx
```

### Delete Containers

```bash
# Delete container (stopped only)
docker rm my-nginx

# Force delete (even if running)
docker rm -f my-nginx

# Delete all stopped containers
docker container prune

# Delete all containers (caution!)
docker rm -f $(docker ps -aq)
```

### Container Logs

```bash
# View logs
docker logs my-nginx

# Real-time logs (-f: follow)
docker logs -f my-nginx

# Last 100 lines
docker logs --tail 100 my-nginx

# Include timestamps
docker logs -t my-nginx
```

### Access Running Container

```bash
# Access container shell
docker exec -it my-nginx bash

# Execute specific command
docker exec my-nginx cat /etc/nginx/nginx.conf

# Access with root privileges
docker exec -it -u root my-nginx bash
```

### Container Information

```bash
# Detailed information
docker inspect my-nginx

# Resource usage
docker stats

# Real-time resource monitoring
docker stats my-nginx
```

---

## 5. Practice Examples

### Example 1: Nginx Web Server

```bash
# 1. Run Nginx container
docker run -d --name web -p 8080:80 nginx

# 2. Check in browser
# http://localhost:8080

# 3. Check logs
docker logs web

# 4. Access container
docker exec -it web bash

# 5. Check Nginx configuration
cat /etc/nginx/nginx.conf

# 6. Cleanup
exit
docker stop web
docker rm web
```

### Example 2: Serve Custom HTML

```bash
# 1. Create HTML file
mkdir -p ~/docker-test
echo "<h1>Hello Docker!</h1>" > ~/docker-test/index.html

# 2. Run with volume mount
docker run -d \
  --name my-web \
  -p 8080:80 \
  -v ~/docker-test:/usr/share/nginx/html:ro \
  nginx

# 3. Check in browser
# http://localhost:8080

# 4. Edit HTML (reflected in real-time)
echo "<h1>Updated!</h1>" > ~/docker-test/index.html

# 5. Cleanup
docker rm -f my-web
```

### Example 3: MySQL Database

```bash
# 1. Run MySQL container
docker run -d \
  --name mydb \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=testdb \
  -p 3306:3306 \
  mysql:8

# 2. Check startup with logs
docker logs -f mydb

# 3. Connect to MySQL client
docker exec -it mydb mysql -uroot -psecret

# 4. Inside MySQL:
# mysql> SHOW DATABASES;
# mysql> USE testdb;
# mysql> CREATE TABLE users (id INT, name VARCHAR(50));
# mysql> exit

# 5. Cleanup
docker rm -f mydb
```

### Example 4: Node.js Application

```bash
# 1. Create project directory
mkdir -p ~/node-docker
cd ~/node-docker

# 2. Create package.json
cat > package.json << 'EOF'
{
  "name": "docker-test",
  "version": "1.0.0",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  }
}
EOF

# 3. Create app.js
cat > app.js << 'EOF'
const http = require('http');
const server = http.createServer((req, res) => {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Hello from Node.js in Docker!\n');
});
server.listen(3000, () => {
  console.log('Server running on port 3000');
});
EOF

# 4. Run container
docker run -d \
  --name node-app \
  -p 3000:3000 \
  -v $(pwd):/app \
  -w /app \
  node:18-alpine \
  node app.js

# 5. Test
curl http://localhost:3000

# 6. Cleanup
docker rm -f node-app
```

---

## 6. Useful Option Combinations

### Development Environment

```bash
docker run -d \
  --name dev-server \
  -p 3000:3000 \
  -v $(pwd):/app \
  -w /app \
  --restart unless-stopped \
  node:18-alpine \
  npm run dev
```

### Data Persistence

```bash
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15
```

---

## Command Summary

### Image Commands

| Command | Description |
|---------|-------------|
| `docker pull image` | Download image |
| `docker images` | List images |
| `docker rmi image` | Delete image |
| `docker image prune` | Delete unused images |

### Container Commands

| Command | Description |
|---------|-------------|
| `docker run` | Create and run container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |
| `docker stop` | Stop container |
| `docker start` | Start container |
| `docker rm` | Delete container |
| `docker logs` | View logs |
| `docker exec -it` | Access container |

### Key Options

| Option | Description |
|--------|-------------|
| `-d` | Run in background |
| `-p host:container` | Port mapping |
| `-v host:container` | Volume mount |
| `-e KEY=VALUE` | Environment variable |
| `--name` | Container name |
| `--rm` | Auto-remove on exit |
| `-it` | Interactive mode |

---

## Next Steps

Let's create our own Docker images in [03_Dockerfile.md](./03_Dockerfile.md)!
