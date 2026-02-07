# Container Services (ECS/EKS/Fargate vs GKE/Cloud Run)

## 1. Container Overview

### 1.1 Container vs VM

```
┌─────────────────────────────────────────────────────────────┐
│                   Virtual Machine (VM)                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │Guest OS │ │Guest OS │ │Guest OS │  ← OS per VM          │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Hypervisor                           ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Container                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │  Libs   │ │  Libs   │ │  Libs   │  ← Libraries only     │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Container Runtime                      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Service Comparison

| Category | AWS | GCP |
|------|-----|-----|
| **Container Registry** | ECR | Artifact Registry |
| **Container Orchestration** | ECS | - |
| **Managed Kubernetes** | EKS | GKE |
| **Serverless Containers** | Fargate | Cloud Run |
| **App Platform** | App Runner | Cloud Run |

---

## 2. Container Registry

### 2.1 AWS ECR (Elastic Container Registry)

```bash
# 1. ECR 레포지토리 생성
aws ecr create-repository \
    --repository-name my-app \
    --region ap-northeast-2

# 2. Docker 로그인
aws ecr get-login-password --region ap-northeast-2 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com

# 3. 이미지 빌드 및 태그
docker build -t my-app .
docker tag my-app:latest \
    123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# 4. 이미지 푸시
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest

# 5. 이미지 목록 확인
aws ecr list-images --repository-name my-app
```

### 2.2 GCP Artifact Registry

```bash
# 1. Artifact Registry API 활성화
gcloud services enable artifactregistry.googleapis.com

# 2. 레포지토리 생성
gcloud artifacts repositories create my-repo \
    --repository-format=docker \
    --location=asia-northeast3 \
    --description="My Docker repository"

# 3. Docker 인증 설정
gcloud auth configure-docker asia-northeast3-docker.pkg.dev

# 4. 이미지 빌드 및 태그
docker build -t my-app .
docker tag my-app:latest \
    asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest

# 5. 이미지 푸시
docker push asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest

# 6. 이미지 목록 확인
gcloud artifacts docker images list \
    asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo
```

---

## 3. AWS ECS (Elastic Container Service)

### 3.1 ECS Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                        ECS Cluster                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                      Service                            ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    Task       │  │    Task       │  ← Container group││
│  │  │ ┌───────────┐ │  │ ┌───────────┐ │                   ││
│  │  │ │ Container │ │  │ │ Container │ │                   ││
│  │  │ └───────────┘ │  │ └───────────┘ │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌───────────────────┐  ┌───────────────────┐               │
│  │ EC2 Instance      │  │ Fargate           │               │
│  │ (self-managed)    │  │ (serverless)      │               │
│  └───────────────────┘  └───────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Creating an ECS Cluster

```bash
# 1. 클러스터 생성 (Fargate)
aws ecs create-cluster \
    --cluster-name my-cluster \
    --capacity-providers FARGATE FARGATE_SPOT

# 2. Task Definition 생성
# task-definition.json
{
    "family": "my-task",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "my-container",
            "image": "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 80,
                    "protocol": "tcp"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/my-task",
                    "awslogs-region": "ap-northeast-2",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}

aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. 서비스 생성
aws ecs create-service \
    --cluster my-cluster \
    --service-name my-service \
    --task-definition my-task:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

---

## 4. AWS EKS (Elastic Kubernetes Service)

### 4.1 Creating an EKS Cluster

```bash
# 1. eksctl 설치 (macOS)
brew tap weaveworks/tap
brew install weaveworks/tap/eksctl

# 2. 클러스터 생성
eksctl create cluster \
    --name my-cluster \
    --region ap-northeast-2 \
    --nodegroup-name my-nodes \
    --node-type t3.medium \
    --nodes 2 \
    --nodes-min 1 \
    --nodes-max 4

# 3. kubeconfig 업데이트
aws eks update-kubeconfig --name my-cluster --region ap-northeast-2

# 4. 클러스터 확인
kubectl get nodes
```

### 4.2 Deploying Applications

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 80
```

```bash
# 배포
kubectl apply -f deployment.yaml

# 상태 확인
kubectl get pods
kubectl get services
```

---

## 5. GCP GKE (Google Kubernetes Engine)

### 5.1 Creating a GKE Cluster

```bash
# 1. GKE API 활성화
gcloud services enable container.googleapis.com

# 2. 클러스터 생성 (Autopilot - 권장)
gcloud container clusters create-auto my-cluster \
    --region=asia-northeast3

# 또는 Standard 클러스터
gcloud container clusters create my-cluster \
    --region=asia-northeast3 \
    --num-nodes=2 \
    --machine-type=e2-medium

# 3. 클러스터 인증 정보 가져오기
gcloud container clusters get-credentials my-cluster \
    --region=asia-northeast3

# 4. 클러스터 확인
kubectl get nodes
```

### 5.2 GKE Autopilot vs Standard

| Category | Autopilot | Standard |
|------|-----------|----------|
| **Node Management** | Google auto-managed | User-managed |
| **Billing** | Pod resource-based | Node-based |
| **Security** | Enhanced security defaults | Manual configuration |
| **Scalability** | Auto-scaling | Manual/auto configuration |
| **Best For** | Most workloads | Fine-grained control needed |

### 5.3 Deploying Applications

```yaml
# deployment.yaml (GKE)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 80
```

```bash
kubectl apply -f deployment.yaml
kubectl get services
```

---

## 6. Serverless Containers

### 6.1 AWS Fargate

Fargate runs containers without server provisioning.

**Features:**
- No EC2 instance management needed
- Define resources at task level
- Use with ECS or EKS

```bash
# ECS + Fargate로 서비스 생성
aws ecs create-service \
    --cluster my-cluster \
    --service-name my-fargate-service \
    --task-definition my-task:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 6.2 GCP Cloud Run

Cloud Run runs containers in a serverless manner.

**Features:**
- Fully managed
- Request-based auto-scaling (to zero)
- Pay only for what you use
- HTTP traffic or event-driven

```bash
# 1. 이미지 배포
gcloud run deploy my-service \
    --image=asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/my-app:latest \
    --region=asia-northeast3 \
    --platform=managed \
    --allow-unauthenticated

# 2. 서비스 URL 확인
gcloud run services describe my-service \
    --region=asia-northeast3 \
    --format='value(status.url)'

# 3. 트래픽 분할 (Blue/Green)
gcloud run services update-traffic my-service \
    --region=asia-northeast3 \
    --to-revisions=my-service-00002-abc=50,my-service-00001-xyz=50
```

### 6.3 Cloud Run vs App Runner Comparison

| Category | GCP Cloud Run | AWS App Runner |
|------|--------------|----------------|
| **Source** | Container image, source code | Container image, source code |
| **Max Memory** | 32GB | 12GB |
| **Max Timeout** | 60 minutes | 30 minutes |
| **Scale to Zero** | Supported | Supported (optional) |
| **VPC Connection** | Supported | Supported |
| **GPU** | Supported | Not supported |

---

## 7. Service Selection Guide

### 7.1 Decision Tree

```
Do you need serverless containers?
├── Yes → Cloud Run / Fargate / App Runner
│         └── Need Kubernetes features?
│             ├── Yes → Fargate on EKS
│             └── No → Cloud Run (GCP) / Fargate on ECS (AWS)
└── No → Do you need Kubernetes?
          ├── Yes → GKE (Autopilot/Standard) / EKS
          └── No → ECS on EC2 / Compute Engine + Docker
```

### 7.2 Recommendations by Use Case

| Use Case | AWS Recommended | GCP Recommended |
|----------|---------|---------|
| **Simple Web App** | App Runner | Cloud Run |
| **Microservices** | ECS Fargate | Cloud Run |
| **Complex K8s Workloads** | EKS | GKE Standard |
| **ML Deployment** | EKS + GPU | GKE + GPU |
| **Batch Jobs** | ECS Task | Cloud Run Jobs |
| **Event Processing** | Fargate + EventBridge | Cloud Run + Eventarc |

---

## 8. Pricing Comparison

### 8.1 ECS/EKS vs GKE

**AWS ECS (Fargate):**
```
vCPU: $0.04048/hour (Seoul)
Memory: $0.004445/GB/hour (Seoul)

Example: 0.5 vCPU, 1GB, 24 hours
= (0.5 × $0.04048 × 24) + (1 × $0.004445 × 24)
= $0.49 + $0.11 = $0.60/day
```

**AWS EKS:**
```
Cluster: $0.10/hour ($72/month)
+ Node costs (EC2) or Fargate costs
```

**GCP GKE:**
```
Autopilot: vCPU $0.0445/hour, Memory $0.0049/GB/hour
Standard: Management fee $0.10/hour/cluster + node costs

Example: Autopilot 0.5 vCPU, 1GB, 24 hours
= (0.5 × $0.0445 × 24) + (1 × $0.0049 × 24)
= $0.53 + $0.12 = $0.65/day
```

### 8.2 Cloud Run Pricing

```
CPU: $0.00002400/vCPU-second (during request processing)
Memory: $0.00000250/GB-second
Requests: $0.40/million requests

Free Tier:
- 2 million requests/month
- 360,000 GB-seconds
- 180,000 vCPU-seconds
```

---

## 9. Hands-on: Deploy a Simple Web App

### 9.1 Prepare Dockerfile

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
```

```python
# app.py
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return f"Hello from {os.environ.get('CLOUD_PROVIDER', 'Container')}!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

```
# requirements.txt
flask==3.0.0
gunicorn==21.2.0
```

### 9.2 Deploy to GCP Cloud Run

```bash
# 빌드 및 배포 (소스에서 직접)
gcloud run deploy my-app \
    --source=. \
    --region=asia-northeast3 \
    --allow-unauthenticated \
    --set-env-vars=CLOUD_PROVIDER=GCP
```

### 9.3 Deploy to AWS App Runner

```bash
# 1. ECR에 이미지 푸시 (앞서 설명한 방법)

# 2. App Runner 서비스 생성
aws apprunner create-service \
    --service-name my-app \
    --source-configuration '{
        "ImageRepository": {
            "ImageIdentifier": "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8080",
                "RuntimeEnvironmentVariables": {
                    "CLOUD_PROVIDER": "AWS"
                }
            }
        },
        "AuthenticationConfiguration": {
            "AccessRoleArn": "arn:aws:iam::123456789012:role/AppRunnerECRAccessRole"
        }
    }'
```

---

## 10. Next Steps

- [07_Object_Storage.md](./07_Object_Storage.md) - Object Storage
- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC Networking

---

## References

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [GCP GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Docker/](../Docker/) - Docker Basics
