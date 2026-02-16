# 컨테이너 서비스 (ECS/EKS/Fargate vs GKE/Cloud Run)

## 1. 컨테이너 개요

### 1.1 컨테이너 vs VM

```
┌─────────────────────────────────────────────────────────────┐
│                     가상 머신 (VM)                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │Guest OS │ │Guest OS │ │Guest OS │  ← 각 VM마다 OS        │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Hypervisor                           ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       컨테이너                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  App A  │ │  App B  │ │  App C  │                        │
│  ├─────────┤ ├─────────┤ ├─────────┤                        │
│  │  Libs   │ │  Libs   │ │  Libs   │  ← 라이브러리만        │
│  └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Container Runtime                      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Host OS                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 서비스 비교

| 항목 | AWS | GCP |
|------|-----|-----|
| **컨테이너 레지스트리** | ECR | Artifact Registry |
| **컨테이너 오케스트레이션** | ECS | - |
| **Kubernetes 관리형** | EKS | GKE |
| **서버리스 컨테이너** | Fargate | Cloud Run |
| **App Platform** | App Runner | Cloud Run |

---

## 2. 컨테이너 레지스트리

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

### 3.1 ECS 개념

```
┌─────────────────────────────────────────────────────────────┐
│                        ECS Cluster                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                      Service                            ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    Task       │  │    Task       │  ← 컨테이너 그룹   ││
│  │  │ ┌───────────┐ │  │ ┌───────────┐ │                   ││
│  │  │ │ Container │ │  │ │ Container │ │                   ││
│  │  │ └───────────┘ │  │ └───────────┘ │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌───────────────────┐  ┌───────────────────┐               │
│  │ EC2 Instance      │  │ Fargate           │               │
│  │ (자체 관리)       │  │ (서버리스)        │               │
│  └───────────────────┘  └───────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 ECS 클러스터 생성

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

### 3.3 ECS Service Connect

ECS Service Connect는 별도의 프록시나 서비스 메시 설정 없이 서비스 간 통신을 위한 내장 서비스 메시 기능을 제공합니다.

```json
// Service Connect가 포함된 서비스 정의
{
    "cluster": "my-cluster",
    "serviceName": "backend-service",
    "taskDefinition": "backend-task:1",
    "serviceConnectConfiguration": {
        "enabled": true,
        "namespace": "my-app-namespace",
        "services": [
            {
                "portName": "http",
                "discoveryName": "backend",
                "clientAliases": [
                    {
                        "port": 80,
                        "dnsName": "backend.local"
                    }
                ]
            }
        ]
    },
    "desiredCount": 2,
    "launchType": "FARGATE",
    "networkConfiguration": {
        "awsvpcConfiguration": {
            "subnets": ["subnet-xxx"],
            "securityGroups": ["sg-xxx"]
        }
    }
}
```

**주요 장점:**
- 내장 서비스 디스커버리 (AWS Cloud Map 통합)
- 서비스 간 자동 로드 밸런싱
- 추가 에이전트 없이 트래픽 메트릭 및 관측성(Observability) 제공
- 외부 서비스 메시(Istio, Consul) 불필요

### 3.4 ECS Exec (컨테이너 디버깅)

ECS Exec을 사용하면 실행 중인 컨테이너에 대화형 셸로 접근하여 디버깅할 수 있습니다.

```bash
# 서비스에서 ECS Exec 활성화
aws ecs update-service \
    --cluster my-cluster \
    --service my-service \
    --enable-execute-command

# 대화형 셸 세션 시작
aws ecs execute-command \
    --cluster my-cluster \
    --task TASK_ID \
    --container my-container \
    --interactive \
    --command "/bin/sh"

# 일회성 명령 실행
aws ecs execute-command \
    --cluster my-cluster \
    --task TASK_ID \
    --container my-container \
    --command "cat /app/config.json"
```

> **참고:** ECS Exec을 사용하려면 태스크 역할에 `ssmmessages` 권한이 필요하며, 태스크 정의에 `initProcessEnabled: true`가 포함되어야 합니다.

---

## 4. AWS EKS (Elastic Kubernetes Service)

### 4.1 EKS 클러스터 생성

```bash
# 1. eksctl 설치 (macOS)
brew install eksctl

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

### 4.2 EKS Auto Mode

EKS Auto Mode(2024년 말 출시)는 GKE Autopilot과 유사하게 노드 관리를 자동화하여 EKS 운영을 단순화합니다.

```bash
# Auto Mode로 EKS 클러스터 생성
eksctl create cluster \
    --name my-auto-cluster \
    --region ap-northeast-2 \
    --auto-mode

# 또는 기존 클러스터에 Auto Mode 활성화
aws eks update-cluster-config \
    --name my-cluster \
    --compute-config enabled=true \
    --kubernetes-network-config '{"elasticLoadBalancing":{"enabled":true}}' \
    --storage-config '{"blockStorage":{"enabled":true}}'
```

| 기능 | EKS Standard | EKS Auto Mode |
|------|-------------|---------------|
| **노드 프로비저닝** | 수동 (관리형 노드 그룹 또는 Karpenter) | 자동 |
| **노드 OS 업데이트** | 사용자 관리 | AWS 관리 |
| **로드 밸런서** | AWS LB Controller 설치 필요 | 내장 |
| **스토리지(EBS CSI)** | EBS CSI 드라이버 설치 필요 | 내장 |
| **과금** | EC2 인스턴스 기반 | Pod 리소스 기반 (오버헤드 포함) |
| **적합 대상** | 세밀한 제어 필요 시 | 간소화된 운영 |

### 4.3 애플리케이션 배포

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

### 5.1 GKE 클러스터 생성

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

### 5.2 GKE Autopilot 심화

GKE Autopilot은 Google이 노드, 스케일링, 보안을 포함한 전체 클러스터 인프라를 관리하는 완전 관리형 Kubernetes 모드입니다.

**Autopilot vs Standard:**

| 항목 | Autopilot | Standard |
|------|-----------|----------|
| **노드 관리** | Google 자동 관리 | 사용자 관리 |
| **과금** | Pod 리소스 기반 | 노드 기반 |
| **보안** | 강화된 기본값 (경량화 OS, Workload Identity, Shielded GKE 노드) | 수동 구성 |
| **확장성** | 자동 HPA/VPA | 수동/자동 구성 |
| **GPU 지원** | 지원 (L4, A100, H100, TPU) | 지원 |
| **Spot Pod** | 지원 | 지원 (선점형 노드) |
| **DaemonSet** | 허용 (과금 포함) | 허용 |
| **특권 Pod** | 불허 | 허용 |
| **적합 대상** | 대부분의 워크로드, 비용 최적화 | 세밀한 제어, 특수 커널 요구 시 |

**Autopilot 보안 기능 (기본 활성화):**
- `containerd` 기반 Container-Optimized OS
- Workload Identity (노드 서비스 계정 키 불필요)
- Shielded GKE 노드 (보안 부팅, 무결성 모니터링)
- 네트워크 정책 적용
- Pod 보안 표준 (기본 Baseline)
- Binary Authorization 지원

```bash
# Autopilot에서 Spot Pod 배포 (최대 60-91% 비용 절감)
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-processor
spec:
  replicas: 5
  selector:
    matchLabels:
      app: batch-processor
  template:
    metadata:
      labels:
        app: batch-processor
    spec:
      nodeSelector:
        cloud.google.com/gke-spot: "true"
      terminationGracePeriodSeconds: 25
      containers:
      - name: worker
        image: asia-northeast3-docker.pkg.dev/PROJECT_ID/my-repo/worker:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
            # Autopilot에서 GPU 요청
            # nvidia.com/gpu: "1"
          limits:
            cpu: "500m"
            memory: "1Gi"
      tolerations:
      - key: cloud.google.com/gke-spot
        operator: Equal
        value: "true"
        effect: NoSchedule
EOF
```

### 5.3 애플리케이션 배포

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

## 6. 서버리스 컨테이너

### 6.1 AWS Fargate

Fargate는 서버 프로비저닝 없이 컨테이너를 실행합니다.

**특징:**
- EC2 인스턴스 관리 불필요
- 태스크 수준에서 리소스 정의
- ECS 또는 EKS와 함께 사용

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

Cloud Run은 컨테이너를 서버리스로 실행합니다.

**특징:**
- 완전 관리형
- 요청 기반 자동 확장 (0까지)
- 사용한 만큼만 과금
- HTTP 트래픽 또는 이벤트 기반

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

### 6.3 Cloud Run vs App Runner 비교

| 항목 | GCP Cloud Run | AWS App Runner |
|------|--------------|----------------|
| **소스** | 컨테이너 이미지, 소스 코드 | 컨테이너 이미지, 소스 코드 |
| **최대 메모리** | 32GB | 12GB |
| **최대 타임아웃** | 60분 | 30분 |
| **0으로 스케일** | 지원 | 지원 (옵션) |
| **VPC 연결** | 지원 | 지원 |
| **GPU** | 지원 | 미지원 |

---

## 7. 서비스 선택 가이드

### 7.1 결정 트리

```
서버리스 컨테이너가 필요한가?
├── Yes → Cloud Run / Fargate / App Runner
│         └── Kubernetes 기능 필요?
│             ├── Yes → Fargate on EKS
│             └── No → Cloud Run (GCP) / Fargate on ECS (AWS)
└── No → Kubernetes가 필요한가?
          ├── Yes → GKE (Autopilot/Standard) / EKS
          └── No → ECS on EC2 / Compute Engine + Docker
```

### 7.2 사용 사례별 권장

| 사용 사례 | AWS 권장 | GCP 권장 |
|----------|---------|---------|
| **단순 웹앱** | App Runner | Cloud Run |
| **마이크로서비스** | ECS Fargate + Service Connect | Cloud Run |
| **K8s (간소화)** | EKS Auto Mode | GKE Autopilot |
| **K8s (전체 제어)** | EKS Standard | GKE Standard |
| **ML/GPU 워크로드** | EKS + GPU | GKE Autopilot + GPU |
| **배치 작업** | ECS Task | Cloud Run Jobs |
| **이벤트 처리** | Fargate + EventBridge | Cloud Run + Eventarc |
| **비용 민감 배치** | Fargate Spot | Autopilot Spot Pod |

---

## 8. 과금 비교

### 8.1 ECS/EKS vs GKE

**AWS ECS (Fargate):**
```
vCPU: $0.04048/시간 (서울)
메모리: $0.004445/GB/시간 (서울)

예: 0.5 vCPU, 1GB, 24시간
= (0.5 × $0.04048 × 24) + (1 × $0.004445 × 24)
= $0.49 + $0.11 = $0.60/일
```

**AWS EKS:**
```
클러스터: $0.10/시간 ($72/월)
+ 노드 비용 (EC2) 또는 Fargate 비용
```

**GCP GKE:**
```
Autopilot: vCPU $0.0445/시간, 메모리 $0.0049/GB/시간
Standard: 관리 수수료 $0.10/시간/클러스터 + 노드 비용

예: Autopilot 0.5 vCPU, 1GB, 24시간
= (0.5 × $0.0445 × 24) + (1 × $0.0049 × 24)
= $0.53 + $0.12 = $0.65/일
```

### 8.2 Cloud Run 과금

```
CPU: $0.00002400/vCPU-초 (요청 처리 중)
메모리: $0.00000250/GB-초
요청: $0.40/100만 요청

무료 티어:
- 200만 요청/월
- 360,000 GB-초
- 180,000 vCPU-초
```

---

## 9. 실습: 간단한 웹앱 배포

### 9.1 Dockerfile 준비

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

### 9.2 GCP Cloud Run 배포

```bash
# 빌드 및 배포 (소스에서 직접)
gcloud run deploy my-app \
    --source=. \
    --region=asia-northeast3 \
    --allow-unauthenticated \
    --set-env-vars=CLOUD_PROVIDER=GCP
```

### 9.3 AWS App Runner 배포

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

## 10. 다음 단계

- [07_Object_Storage.md](./07_Object_Storage.md) - 객체 스토리지
- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC 네트워킹

---

## 참고 자료

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [GCP GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Docker/](../Docker/) - Docker 기초
