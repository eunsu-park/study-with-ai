# Docker Compose

## 1. What is Docker Compose?

Docker Compose is a tool for **defining and running multiple containers**. Manage entire application stacks with a single YAML file.

### Why use Docker Compose?

**Regular Docker commands:**
```bash
# Create network
docker network create myapp-network

# Run database
docker run -d \
  --name db \
  --network myapp-network \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  postgres:15

# Run backend
docker run -d \
  --name backend \
  --network myapp-network \
  -e DATABASE_URL=postgres://... \
  -p 3000:3000 \
  my-backend

# Run frontend
docker run -d \
  --name frontend \
  --network myapp-network \
  -p 80:80 \
  my-frontend
```

**Docker Compose:**
```bash
docker compose up -d
```

| Advantage | Description |
|-----------|-------------|
| **Simplicity** | Run everything with one command |
| **Declarative** | Clearly defined in YAML |
| **Version control** | Manage config files with Git |
| **Reproducibility** | Reproduce identical environments |

---

## 2. Installation Check

Docker Compose is included with Docker Desktop.

```bash
# Check version
docker compose version
# Docker Compose version v2.23.0

# Or (old version)
docker-compose --version
```

> **Note:** `docker-compose` (with hyphen) is the old version, `docker compose` (with space) is the new version.

---

## 3. docker-compose.yml Basic Structure

```yaml
# docker-compose.yml

services:
  service-name1:
    image: image-name
    ports:
      - "host:container"
    environment:
      - variable=value
    volumes:
      - volume:path
    depends_on:
      - other-service

  service-name2:
    build: ./path
    ...

volumes:
  volume-name:

networks:
  network-name:
```

---

## 4. Main Configuration Options

### services - Define Services

```yaml
services:
  web:
    image: nginx:alpine
```

### image - Specify Image

```yaml
services:
  db:
    image: postgres:15

  redis:
    image: redis:7-alpine
```

### build - Build with Dockerfile

```yaml
services:
  app:
    build: .                    # Dockerfile in current directory

  api:
    build:
      context: ./backend        # Build context
      dockerfile: Dockerfile    # Dockerfile path
      args:                     # Build arguments
        - NODE_ENV=production
```

### ports - Port Mapping

```yaml
services:
  web:
    ports:
      - "8080:80"              # host:container
      - "443:443"

  api:
    ports:
      - "3000:3000"
```

### environment - Environment Variables

```yaml
services:
  db:
    environment:
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=myapp

  # Or key: value format
  api:
    environment:
      NODE_ENV: production
      DB_HOST: db
```

### env_file - Environment Variable File

```yaml
services:
  api:
    env_file:
      - .env
      - .env.local
```

**.env file:**
```
DB_HOST=localhost
DB_PASSWORD=secret
API_KEY=abc123
```

### volumes - Volume Mounts

```yaml
services:
  db:
    volumes:
      - pgdata:/var/lib/postgresql/data    # Named volume
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql  # Bind mount

  app:
    volumes:
      - ./src:/app/src                      # Source code mount (dev)
      - /app/node_modules                   # Anonymous volume (prevent overwrite)

volumes:
  pgdata:                                   # Named volume declaration
```

### depends_on - Dependencies

```yaml
services:
  api:
    depends_on:
      - db
      - redis

  db:
    image: postgres:15

  redis:
    image: redis:7
```

> **Note:** `depends_on` only ensures startup order. It doesn't wait for the service to be "ready".

### networks - Networks

```yaml
services:
  frontend:
    networks:
      - frontend-net

  backend:
    networks:
      - frontend-net
      - backend-net

  db:
    networks:
      - backend-net

networks:
  frontend-net:
  backend-net:
```

### restart - Restart Policy

```yaml
services:
  web:
    restart: always              # Always restart

  api:
    restart: unless-stopped      # Restart until manually stopped

  worker:
    restart: on-failure          # Restart on failure
```

### healthcheck - Health Check

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## 5. Docker Compose Commands

### Run

```bash
# Run (foreground)
docker compose up

# Run in background
docker compose up -d

# Rebuild images then run
docker compose up --build

# Run specific services only
docker compose up -d web api
```

### Stop/Remove

```bash
# Stop
docker compose stop

# Stop and remove containers
docker compose down

# Also remove volumes
docker compose down -v

# Also remove images
docker compose down --rmi all
```

### Check Status

```bash
# List services
docker compose ps

# View logs
docker compose logs

# View specific service logs
docker compose logs api

# Real-time logs
docker compose logs -f
```

### Service Management

```bash
# Restart
docker compose restart

# Restart specific service
docker compose restart api

# Scale services
docker compose up -d --scale api=3

# Execute command in service
docker compose exec api bash
docker compose exec db psql -U postgres
```

---

## 6. Practice Examples

### Example 1: Web + Database

**Project structure:**
```
my-webapp/
├── docker-compose.yml
├── .env
└── app/
    ├── Dockerfile
    └── index.js
```

**docker-compose.yml:**
```yaml
services:
  app:
    build: ./app
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

**app/Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "index.js"]
```

**app/index.js:**
```javascript
const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.json({
    message: 'Hello from Docker Compose!',
    db_url: process.env.DATABASE_URL ? 'Connected' : 'Not set'
  });
});

app.listen(3000, () => console.log('Server on port 3000'));
```

**Run:**
```bash
cd my-webapp
docker compose up -d
curl http://localhost:3000
docker compose logs -f
docker compose down
```

### Example 2: Full Stack Application

```yaml
# docker-compose.yml

services:
  # Frontend (React)
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

  # Backend (Node.js)
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db
      - DB_NAME=myapp
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis

  # Database (PostgreSQL)
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql

  # Cache (Redis)
  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

  # Admin tool (pgAdmin)
  pgadmin:
    image: dpage/pgadmin4
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@example.com
      - PGADMIN_DEFAULT_PASSWORD=admin
    ports:
      - "5050:80"
    depends_on:
      - db

volumes:
  pgdata:
  redisdata:
```

**.env:**
```
DB_PASSWORD=supersecret123
```

### Example 3: Development Environment

```yaml
# docker-compose.dev.yml

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app                    # Mount source code
      - /app/node_modules         # Protect node_modules
    environment:
      - NODE_ENV=development
    command: npm run dev          # Run dev server

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"              # Allow local access
```

**Run:**
```bash
# Development environment
docker compose -f docker-compose.dev.yml up

# Production environment
docker compose -f docker-compose.yml up -d
```

---

## 7. Useful Patterns

### Environment-specific Configuration

```yaml
# docker-compose.yml (base)
services:
  app:
    image: myapp

# docker-compose.override.yml (dev, auto-merged)
services:
  app:
    build: .
    volumes:
      - .:/app

# docker-compose.prod.yml (production)
services:
  app:
    restart: always
```

```bash
# Development: auto-merges docker-compose.yml + docker-compose.override.yml
docker compose up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Service Wait (wait-for-it)

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:15
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
```

---

## Command Summary

| Command | Description |
|---------|-------------|
| `docker compose up` | Start services |
| `docker compose up -d` | Start in background |
| `docker compose up --build` | Rebuild then start |
| `docker compose down` | Stop and remove services |
| `docker compose down -v` | Also remove volumes |
| `docker compose ps` | Service status |
| `docker compose logs` | View logs |
| `docker compose logs -f` | Real-time logs |
| `docker compose exec service command` | Execute command |
| `docker compose restart` | Restart |

---

## Next Steps

Apply Docker to real projects in [05_Practical_Examples.md](./05_Practical_Examples.md)!
