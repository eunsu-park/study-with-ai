# Dockerfile

## 1. What is a Dockerfile?

A Dockerfile is a **configuration file** for creating Docker images. When you write commands in a text file, Docker executes them in order to create an image.

```
Dockerfile → docker build → Docker Image → docker run → Container
(Blueprint)    (Build)       (Template)      (Run)      (Instance)
```

### Why use a Dockerfile?

| Advantage | Description |
|-----------|-------------|
| **Reproducibility** | Create identical images repeatedly |
| **Automation** | No manual setup needed |
| **Version control** | Track history with Git |
| **Documentation** | Environment setup recorded as code |

---

## 2. Dockerfile Basic Syntax

### Basic Structure

```dockerfile
# Comment
INSTRUCTION argument
```

### Main Instructions

| Instruction | Description | Example |
|-------------|-------------|---------|
| `FROM` | Base image | `FROM node:18` |
| `WORKDIR` | Working directory | `WORKDIR /app` |
| `COPY` | Copy files | `COPY . .` |
| `RUN` | Execute command during build | `RUN npm install` |
| `CMD` | Container startup command | `CMD ["npm", "start"]` |
| `EXPOSE` | Expose port | `EXPOSE 3000` |
| `ENV` | Environment variable | `ENV NODE_ENV=production` |

---

## 3. Instruction Details

### FROM - Base Image

Every Dockerfile starts with `FROM`.

```dockerfile
# Basic
FROM ubuntu:22.04

# Node.js image
FROM node:18

# Lightweight Alpine image (recommended)
FROM node:18-alpine

# Multi-stage build
FROM node:18 AS builder
FROM nginx:alpine AS production
```

### WORKDIR - Working Directory

Sets the directory where subsequent commands will execute.

```dockerfile
WORKDIR /app

# Subsequent commands execute in /app
COPY . .          # Copy to /app
RUN npm install   # Execute in /app
```

### COPY - Copy Files

Copies files from host to image.

```dockerfile
# Copy file
COPY package.json .

# Copy directory
COPY src/ ./src/

# Copy all files
COPY . .

# Copy multiple files
COPY package.json package-lock.json ./
```

### ADD vs COPY

```dockerfile
# COPY: Simple copy (recommended)
COPY local-file.txt /app/

# ADD: URL download, archive extraction
ADD https://example.com/file.tar.gz /app/
ADD archive.tar.gz /app/  # Auto-extracts
```

### RUN - Execute Build Command

Executes during image build.

```dockerfile
# Basic
RUN npm install

# Multiple commands (layer optimization)
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Separate for cache utilization
COPY package*.json ./
RUN npm install
COPY . .
```

### CMD - Container Startup Command

Executes when container starts.

```dockerfile
# exec form (recommended)
CMD ["npm", "start"]
CMD ["node", "app.js"]

# shell form
CMD npm start
```

### ENTRYPOINT vs CMD

```dockerfile
# ENTRYPOINT: Always executes (hard to change)
ENTRYPOINT ["node"]
CMD ["app.js"]
# Executes: node app.js

# docker run myimage other.js
# Executes: node other.js (only CMD changed)
```

### ENV - Environment Variables

```dockerfile
# Single variable
ENV NODE_ENV=production

# Multiple variables
ENV NODE_ENV=production \
    PORT=3000 \
    DB_HOST=localhost
```

### EXPOSE - Document Port

```dockerfile
# Expose port (documentation purpose, actual mapping with -p option)
EXPOSE 3000
EXPOSE 80 443
```

### ARG - Build-time Variables

```dockerfile
# Variables passed during build
ARG NODE_VERSION=18
FROM node:${NODE_VERSION}

ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}
```

```bash
# Pass value during build
docker build --build-arg NODE_VERSION=20 .
```

---

## 4. Practice Examples

### Example 1: Node.js Application

**Project structure:**
```
my-node-app/
├── Dockerfile
├── package.json
└── app.js
```

**package.json:**
```json
{
  "name": "my-node-app",
  "version": "1.0.0",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
```

**app.js:**
```javascript
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.json({ message: 'Hello from Docker!', version: '1.0.0' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**Dockerfile:**
```dockerfile
# Base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy dependency files (utilize cache)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Run command
CMD ["npm", "start"]
```

**Build and run:**
```bash
# Build image
docker build -t my-node-app .

# Run container
docker run -d -p 3000:3000 --name node-app my-node-app

# Test
curl http://localhost:3000

# Cleanup
docker rm -f node-app
```

### Example 2: Python Flask Application

**Project structure:**
```
my-flask-app/
├── Dockerfile
├── requirements.txt
└── app.py
```

**requirements.txt:**
```
flask==3.0.0
gunicorn==21.2.0
```

**app.py:**
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify(message='Hello from Flask in Docker!')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 5000

# Run with Gunicorn (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**Build and run:**
```bash
docker build -t my-flask-app .
docker run -d -p 5000:5000 my-flask-app
curl http://localhost:5000
```

### Example 3: Static Website (Nginx)

**Project structure:**
```
my-website/
├── Dockerfile
├── nginx.conf
└── public/
    └── index.html
```

**public/index.html:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>My Docker Website</title>
</head>
<body>
    <h1>Hello from Nginx in Docker!</h1>
</body>
</html>
```

**Dockerfile:**
```dockerfile
FROM nginx:alpine

# Copy custom config (optional)
# COPY nginx.conf /etc/nginx/nginx.conf

# Copy static files
COPY public/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

---

## 5. Multi-stage Build

Separate build and runtime environments to reduce image size.

### React App Example

```dockerfile
# Stage 1: Build
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Runtime (serve with nginx)
FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Go App Example

```dockerfile
# Stage 1: Build
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
RUN go build -o main .

# Stage 2: Runtime (minimal image)
FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/main .

EXPOSE 8080
CMD ["./main"]
```

**Size comparison:**
```
golang:1.21-alpine  →  ~300MB (build environment)
Final image         →  ~15MB (runtime environment)
```

---

## 6. Best Practices

### .dockerignore File

Exclude unnecessary files from build.

```
# .dockerignore
node_modules
npm-debug.log
.git
.gitignore
.env
*.md
Dockerfile
.dockerignore
```

### Layer Optimization

```dockerfile
# Bad: Full reinstall every time
COPY . .
RUN npm install

# Good: Reinstall only when package.json changes
COPY package*.json ./
RUN npm install
COPY . .
```

### Use Small Images

```dockerfile
# Large
FROM node:18           # ~1GB

# Recommended
FROM node:18-alpine    # ~175MB

# Minimal
FROM node:18-slim      # ~200MB
```

### Security

```dockerfile
# Run as non-root user
FROM node:18-alpine

RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

WORKDIR /app
COPY --chown=appuser:appgroup . .
```

---

## 7. Image Build Commands

```bash
# Basic build
docker build -t imagename .

# Specify tag
docker build -t myapp:1.0 .

# Use different Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# Pass build arguments
docker build --build-arg NODE_ENV=production -t myapp .

# Build without cache
docker build --no-cache -t myapp .

# Verbose build output
docker build --progress=plain -t myapp .
```

---

## Command Summary

| Dockerfile Instruction | Description |
|------------------------|-------------|
| `FROM` | Specify base image |
| `WORKDIR` | Set working directory |
| `COPY` | Copy files/directories |
| `RUN` | Execute command during build |
| `CMD` | Container startup command |
| `EXPOSE` | Document port |
| `ENV` | Set environment variable |
| `ARG` | Build-time variable |
| `ENTRYPOINT` | Fixed execution command |

---

## Next Steps

Learn how to manage multiple containers together in [04_Docker_Compose.md](./04_Docker_Compose.md)!
