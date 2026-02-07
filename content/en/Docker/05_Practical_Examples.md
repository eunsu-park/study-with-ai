# Docker Practical Examples

This document provides step-by-step practice on applying Docker to real projects.

---

## Example 1: Node.js + Express + PostgreSQL

### Project Structure

```
nodejs-postgres-app/
├── docker-compose.yml
├── .env
├── .dockerignore
├── backend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       └── index.js
└── db/
    └── init.sql
```

### Creating Files

**backend/package.json:**
```json
{
  "name": "express-postgres-app",
  "version": "1.0.0",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "node --watch src/index.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.3"
  }
}
```

**backend/src/index.js:**
```javascript
const express = require('express');
const { Pool } = require('pg');

const app = express();
app.use(express.json());

// PostgreSQL connection
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'myapp',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'password'
});

// Routes
app.get('/', (req, res) => {
  res.json({ message: 'Hello Docker!', status: 'running' });
});

app.get('/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({ status: 'healthy', database: 'connected' });
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});

app.get('/users', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM users ORDER BY id');
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/users', async (req, res) => {
  const { name, email } = req.body;
  try {
    const result = await pool.query(
      'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *',
      [name, email]
    );
    res.status(201).json(result.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**backend/Dockerfile:**
```dockerfile
FROM node:18-alpine

WORKDIR /app

# Install dependencies (utilize cache)
COPY package*.json ./
RUN npm install --production

# Copy source code
COPY . .

# Run as non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

EXPOSE 3000

CMD ["npm", "start"]
```

**backend/.dockerignore:**
```
node_modules
npm-debug.log
.git
.env
*.md
```

**db/init.sql:**
```sql
-- Create initial table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data
INSERT INTO users (name, email) VALUES
    ('John Doe', 'john@example.com'),
    ('Jane Smith', 'jane@example.com'),
    ('Bob Johnson', 'bob@example.com');
```

**.env:**
```
DB_PASSWORD=secretpassword123
DB_USER=appuser
DB_NAME=myapp
```

**docker-compose.yml:**
```yaml
services:
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      timeout: 5s
      retries: 5
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

### Run and Test

```bash
# Create directories and navigate
mkdir -p nodejs-postgres-app/backend/src nodejs-postgres-app/db
cd nodejs-postgres-app

# (After creating above files)

# Run
docker compose up -d

# Check status
docker compose ps

# Check logs
docker compose logs -f backend

# API tests
curl http://localhost:3000/
curl http://localhost:3000/health
curl http://localhost:3000/users

# Add user
curl -X POST http://localhost:3000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Park", "email": "alice@example.com"}'

# Cleanup
docker compose down -v
```

---

## Example 2: React + Nginx (Production Build)

### Project Structure

```
react-nginx-app/
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
├── package.json
├── public/
│   └── index.html
└── src/
    ├── App.js
    └── index.js
```

### Creating Files

**package.json:**
```json
{
  "name": "react-docker-app",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version"]
  }
}
```

**public/index.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>React Docker App</title>
</head>
<body>
  <div id="root"></div>
</body>
</html>
```

**src/index.js:**
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

**src/App.js:**
```javascript
import React, { useState, useEffect } from 'react';

function App() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    setMessage('Hello from React in Docker!');
  }, []);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{ textAlign: 'center' }}>
        <h1>{message}</h1>
        <p>This app is deployed with Docker.</p>
        <p>Build time: {new Date().toLocaleString()}</p>
      </div>
    </div>
  );
}

export default App;
```

**Dockerfile (Multi-stage build):**
```dockerfile
# Stage 1: Build
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy build output
COPY --from=builder /app/build /usr/share/nginx/html

# Copy Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # React Router support (SPA)
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Static file caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
}
```

**docker-compose.yml:**
```yaml
services:
  frontend:
    build: .
    ports:
      - "80:80"
    restart: unless-stopped
```

### Run

```bash
# Build and run
docker compose up -d --build

# Check in browser
# http://localhost

# Cleanup
docker compose down
```

---

## Example 3: Full Stack (React + Node.js + PostgreSQL + Redis)

### Project Structure

```
fullstack-app/
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── (React project)
├── backend/
│   ├── Dockerfile
│   └── (Express project)
└── db/
    └── init.sql
```

**docker-compose.yml:**
```yaml
services:
  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped

  # Backend API
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped

  # PostgreSQL database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redisdata:/data
    restart: unless-stopped

volumes:
  pgdata:
  redisdata:
```

**docker-compose.dev.yml (Development override):**
```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3001:3000"
    volumes:
      - ./frontend/src:/app/src
    environment:
      - REACT_APP_API_URL=http://localhost:3000

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend/src:/app/src
    environment:
      - NODE_ENV=development
    command: npm run dev

  db:
    ports:
      - "5432:5432"

  redis:
    ports:
      - "6379:6379"
```

### Run Commands

```bash
# Production
docker compose up -d

# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Specific service logs
docker compose logs -f backend

# Database access
docker compose exec db psql -U ${DB_USER} -d ${DB_NAME}

# Redis CLI
docker compose exec redis redis-cli

# Full cleanup
docker compose down -v
```

---

## Example 4: WordPress + MySQL

### docker-compose.yml

```yaml
services:
  wordpress:
    image: wordpress:latest
    ports:
      - "8080:80"
    environment:
      - WORDPRESS_DB_HOST=db
      - WORDPRESS_DB_USER=wordpress
      - WORDPRESS_DB_PASSWORD=${DB_PASSWORD}
      - WORDPRESS_DB_NAME=wordpress
    volumes:
      - wordpress_data:/var/www/html
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: mysql:8
    environment:
      - MYSQL_DATABASE=wordpress
      - MYSQL_USER=wordpress
      - MYSQL_PASSWORD=${DB_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${DB_ROOT_PASSWORD}
    volumes:
      - db_data:/var/lib/mysql
    restart: unless-stopped

  # phpMyAdmin (optional)
  phpmyadmin:
    image: phpmyadmin:latest
    ports:
      - "8081:80"
    environment:
      - PMA_HOST=db
      - PMA_USER=wordpress
      - PMA_PASSWORD=${DB_PASSWORD}
    depends_on:
      - db

volumes:
  wordpress_data:
  db_data:
```

**.env:**
```
DB_PASSWORD=wordpresspass123
DB_ROOT_PASSWORD=rootpass123
```

### Run

```bash
docker compose up -d

# WordPress: http://localhost:8080
# phpMyAdmin: http://localhost:8081
```

---

## Useful Command Collection

### Debugging

```bash
# Access container
docker compose exec backend sh

# Real-time log monitoring
docker compose logs -f

# Check resource usage
docker stats

# Check network
docker network ls
docker network inspect <network_name>
```

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup (caution!)
docker system prune -a --volumes
```

### Backup

```bash
# Backup volume
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/pgdata-backup.tar.gz -C /data .

# Restore volume
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/pgdata-backup.tar.gz -C /data
```

---

## Learning Complete!

You've completed Docker learning. Next steps:

1. **Practice**: Dockerize your own projects
2. **CI/CD**: Integrate Docker with GitHub Actions
3. **Orchestration**: Learn Kubernetes basics
4. **Security**: Docker security best practices

### Additional Learning Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Play with Docker](https://labs.play-with-docker.com/) - Docker practice in browser
