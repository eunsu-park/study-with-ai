# Dockerfile

## 1. Dockerfile이란?

Dockerfile은 Docker 이미지를 만들기 위한 **설정 파일**입니다. 텍스트 파일에 명령어를 작성하면 Docker가 순서대로 실행하여 이미지를 생성합니다.

```
Dockerfile → docker build → Docker Image → docker run → Container
(설계도)       (빌드)        (템플릿)        (실행)      (인스턴스)
```

### 왜 Dockerfile을 사용할까요?

| 장점 | 설명 |
|------|------|
| **재현성** | 동일한 이미지를 반복 생성 |
| **자동화** | 수동 설정 불필요 |
| **버전 관리** | Git으로 이력 추적 |
| **문서화** | 환경 설정이 코드로 기록 |

---

## 2. Dockerfile 기본 문법

### 기본 구조

```dockerfile
# 주석
명령어 인자
```

### 주요 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `FROM` | 베이스 이미지 | `FROM node:18` |
| `WORKDIR` | 작업 디렉토리 | `WORKDIR /app` |
| `COPY` | 파일 복사 | `COPY . .` |
| `RUN` | 빌드 시 명령 실행 | `RUN npm install` |
| `CMD` | 컨테이너 시작 명령 | `CMD ["npm", "start"]` |
| `EXPOSE` | 포트 노출 | `EXPOSE 3000` |
| `ENV` | 환경 변수 | `ENV NODE_ENV=production` |

---

## 3. 명령어 상세 설명

### FROM - 베이스 이미지

모든 Dockerfile은 `FROM`으로 시작합니다.

```dockerfile
# 기본
FROM ubuntu:22.04

# Node.js 이미지
FROM node:18

# 경량 Alpine 이미지 (권장)
FROM node:18-alpine

# 멀티 스테이지 빌드
FROM node:18 AS builder
FROM nginx:alpine AS production
```

### WORKDIR - 작업 디렉토리

이후 명령어가 실행될 디렉토리를 설정합니다.

```dockerfile
WORKDIR /app

# 이후 명령어는 /app에서 실행
COPY . .          # /app으로 복사
RUN npm install   # /app에서 실행
```

### COPY - 파일 복사

호스트의 파일을 이미지로 복사합니다.

```dockerfile
# 파일 복사
COPY package.json .

# 디렉토리 복사
COPY src/ ./src/

# 모든 파일 복사
COPY . .

# 여러 파일 복사
COPY package.json package-lock.json ./
```

### ADD vs COPY

```dockerfile
# COPY: 단순 복사 (권장)
COPY local-file.txt /app/

# ADD: URL 다운로드, 압축 해제 가능
ADD https://example.com/file.tar.gz /app/
ADD archive.tar.gz /app/  # 자동 압축 해제
```

### RUN - 빌드 시 명령 실행

이미지 빌드 중에 실행됩니다.

```dockerfile
# 기본
RUN npm install

# 여러 명령어 (레이어 최적화)
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# 캐시 활용을 위한 분리
COPY package*.json ./
RUN npm install
COPY . .
```

### CMD - 컨테이너 시작 명령

컨테이너가 시작될 때 실행됩니다.

```dockerfile
# exec 형식 (권장)
CMD ["npm", "start"]
CMD ["node", "app.js"]

# shell 형식
CMD npm start
```

### ENTRYPOINT vs CMD

```dockerfile
# ENTRYPOINT: 항상 실행 (변경 어려움)
ENTRYPOINT ["node"]
CMD ["app.js"]
# 실행: node app.js

# docker run myimage other.js
# 실행: node other.js (CMD만 변경됨)
```

### ENV - 환경 변수

```dockerfile
# 단일 변수
ENV NODE_ENV=production

# 여러 변수
ENV NODE_ENV=production \
    PORT=3000 \
    DB_HOST=localhost
```

### EXPOSE - 포트 문서화

```dockerfile
# 포트 노출 (문서 목적, 실제 매핑은 -p 옵션)
EXPOSE 3000
EXPOSE 80 443
```

### ARG - 빌드 시 변수

```dockerfile
# 빌드 시 전달받는 변수
ARG NODE_VERSION=18
FROM node:${NODE_VERSION}

ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}
```

```bash
# 빌드 시 값 전달
docker build --build-arg NODE_VERSION=20 .
```

---

## 4. 실습 예제

### 예제 1: Node.js 애플리케이션

**프로젝트 구조:**
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
# 베이스 이미지
FROM node:18-alpine

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 (캐시 활용)
COPY package*.json ./

# 의존성 설치
RUN npm install

# 소스 코드 복사
COPY . .

# 포트 노출
EXPOSE 3000

# 실행 명령
CMD ["npm", "start"]
```

**빌드 및 실행:**
```bash
# 이미지 빌드
docker build -t my-node-app .

# 컨테이너 실행
docker run -d -p 3000:3000 --name node-app my-node-app

# 테스트
curl http://localhost:3000

# 정리
docker rm -f node-app
```

### 예제 2: Python Flask 애플리케이션

**프로젝트 구조:**
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

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

EXPOSE 5000

# Gunicorn으로 실행 (프로덕션)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**빌드 및 실행:**
```bash
docker build -t my-flask-app .
docker run -d -p 5000:5000 my-flask-app
curl http://localhost:5000
```

### 예제 3: 정적 웹사이트 (Nginx)

**프로젝트 구조:**
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

# 커스텀 설정 복사 (선택사항)
# COPY nginx.conf /etc/nginx/nginx.conf

# 정적 파일 복사
COPY public/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

---

## 5. 멀티 스테이지 빌드

빌드 환경과 실행 환경을 분리하여 이미지 크기를 줄입니다.

### React 앱 예시

```dockerfile
# Stage 1: 빌드
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: 실행 (nginx로 서빙)
FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Go 앱 예시

```dockerfile
# Stage 1: 빌드
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
RUN go build -o main .

# Stage 2: 실행 (최소 이미지)
FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/main .

EXPOSE 8080
CMD ["./main"]
```

**크기 비교:**
```
golang:1.21-alpine  →  약 300MB (빌드 환경)
최종 이미지         →  약 15MB (실행 환경)
```

---

## 6. 베스트 프랙티스

### .dockerignore 파일

불필요한 파일을 빌드에서 제외합니다.

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

### 레이어 최적화

```dockerfile
# 나쁜 예: 매번 전체 재설치
COPY . .
RUN npm install

# 좋은 예: package.json 변경 시만 재설치
COPY package*.json ./
RUN npm install
COPY . .
```

### 작은 이미지 사용

```dockerfile
# 크기 큼
FROM node:18           # ~1GB

# 권장
FROM node:18-alpine    # ~175MB

# 최소
FROM node:18-slim      # ~200MB
```

### 보안

```dockerfile
# 루트가 아닌 사용자로 실행
FROM node:18-alpine

RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

WORKDIR /app
COPY --chown=appuser:appgroup . .
```

---

## 7. 이미지 빌드 명령어

```bash
# 기본 빌드
docker build -t 이미지명 .

# 태그 지정
docker build -t myapp:1.0 .

# 다른 Dockerfile 사용
docker build -f Dockerfile.prod -t myapp:prod .

# 빌드 인자 전달
docker build --build-arg NODE_ENV=production -t myapp .

# 캐시 없이 빌드
docker build --no-cache -t myapp .

# 빌드 과정 상세 출력
docker build --progress=plain -t myapp .
```

---

## 명령어 요약

| Dockerfile 명령어 | 설명 |
|------------------|------|
| `FROM` | 베이스 이미지 지정 |
| `WORKDIR` | 작업 디렉토리 설정 |
| `COPY` | 파일/디렉토리 복사 |
| `RUN` | 빌드 시 명령 실행 |
| `CMD` | 컨테이너 시작 명령 |
| `EXPOSE` | 포트 문서화 |
| `ENV` | 환경 변수 설정 |
| `ARG` | 빌드 시 변수 |
| `ENTRYPOINT` | 고정 실행 명령 |

---

## 다음 단계

[04_Docker_Compose.md](./04_Docker_Compose.md)에서 여러 컨테이너를 함께 관리하는 방법을 배워봅시다!
