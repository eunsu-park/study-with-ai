# 가상 머신 (EC2 / Compute Engine)

## 1. 가상 머신 개요

가상 머신(VM)은 클라우드에서 가장 기본적인 컴퓨팅 리소스입니다.

### 1.1 서비스 비교

| 항목 | AWS EC2 | GCP Compute Engine |
|------|---------|-------------------|
| 서비스명 | Elastic Compute Cloud | Compute Engine |
| 인스턴스 단위 | Instance | Instance |
| 이미지 | AMI | Image |
| 인스턴스 유형 | Instance Types | Machine Types |
| 시작 스크립트 | User Data | Startup Script |
| 메타데이터 | Instance Metadata | Metadata Server |

---

## 2. 인스턴스 유형

### 2.1 AWS EC2 인스턴스 유형

**명명 규칙:** `{패밀리}{세대}{추가속성}.{크기}`

예: `t3.medium`, `m5.xlarge`, `c6i.2xlarge`

| 패밀리 | 용도 | 예시 |
|--------|------|------|
| **t** | 범용 (버스터블) | t3.micro, t3.small |
| **m** | 범용 (균형) | m5.large, m6i.xlarge |
| **c** | 컴퓨팅 최적화 | c5.xlarge, c6i.2xlarge |
| **r** | 메모리 최적화 | r5.large, r6i.xlarge |
| **i** | 스토리지 최적화 | i3.large, i3en.xlarge |
| **g/p** | GPU | g4dn.xlarge, p4d.24xlarge |

**주요 인스턴스 스펙:**

| 유형 | vCPU | 메모리 | 네트워크 | 용도 |
|------|------|--------|----------|------|
| t3.micro | 2 | 1 GB | Low | 무료 티어, 개발 |
| t3.medium | 2 | 4 GB | Low-Mod | 소규모 앱 |
| m5.large | 2 | 8 GB | Up to 10 Gbps | 범용 |
| c5.xlarge | 4 | 8 GB | Up to 10 Gbps | CPU 집약 |
| r5.large | 2 | 16 GB | Up to 10 Gbps | 메모리 집약 |

### 2.2 GCP Machine Types

**명명 규칙:** `{시리즈}-{유형}-{vCPU수}` 또는 커스텀

예: `e2-medium`, `n2-standard-4`, `c2-standard-8`

| 시리즈 | 용도 | 예시 |
|--------|------|------|
| **e2** | 비용 효율 범용 | e2-micro, e2-medium |
| **n2/n2d** | 범용 (균형) | n2-standard-2, n2-highmem-4 |
| **c2/c2d** | 컴퓨팅 최적화 | c2-standard-4 |
| **m1/m2** | 메모리 최적화 | m1-megamem-96 |
| **a2** | GPU (A100) | a2-highgpu-1g |

**주요 머신 타입 스펙:**

| 유형 | vCPU | 메모리 | 네트워크 | 용도 |
|------|------|--------|----------|------|
| e2-micro | 0.25-2 | 1 GB | 1 Gbps | 무료 티어 |
| e2-medium | 1-2 | 4 GB | 2 Gbps | 소규모 앱 |
| n2-standard-2 | 2 | 8 GB | 10 Gbps | 범용 |
| c2-standard-4 | 4 | 16 GB | 10 Gbps | CPU 집약 |
| n2-highmem-2 | 2 | 16 GB | 10 Gbps | 메모리 집약 |

### 2.3 커스텀 머신 타입 (GCP)

GCP에서는 vCPU와 메모리를 개별 지정할 수 있습니다.

```bash
# 커스텀 머신 타입 생성
gcloud compute instances create my-instance \
    --custom-cpu=6 \
    --custom-memory=24GB \
    --zone=asia-northeast3-a
```

---

## 3. 이미지 (AMI / Image)

### 3.1 AWS AMI

**AMI (Amazon Machine Image)** 구성요소:
- 루트 볼륨 템플릿 (OS, 애플리케이션)
- 인스턴스 유형, 보안 그룹 기본값
- 블록 디바이스 매핑

```bash
# 사용 가능한 AMI 검색 (Amazon Linux 2023)
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-x86_64" \
    --query 'Images | sort_by(@, &CreationDate) | [-1]'

# 주요 AMI 유형
# Amazon Linux 2023: al2023-ami-*
# Ubuntu 22.04: ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-*
# Windows Server: Windows_Server-2022-*
```

### 3.2 GCP Images

```bash
# 사용 가능한 이미지 목록
gcloud compute images list

# 특정 프로젝트의 이미지
gcloud compute images list \
    --filter="family:ubuntu-2204-lts"

# 주요 이미지 패밀리
# debian-11, debian-12
# ubuntu-2204-lts, ubuntu-2404-lts
# centos-stream-9, rocky-linux-9
# windows-2022
```

---

## 4. 인스턴스 생성

### 4.1 AWS EC2 인스턴스 생성

**Console:**
1. EC2 대시보드 → "Launch instance"
2. 이름 입력
3. AMI 선택 (예: Amazon Linux 2023)
4. 인스턴스 유형 선택 (예: t3.micro)
5. 키 페어 생성/선택
6. 네트워크 설정 (VPC, 서브넷, 보안 그룹)
7. 스토리지 설정
8. "Launch instance"

**AWS CLI:**
```bash
# 키 페어 생성
aws ec2 create-key-pair \
    --key-name my-key \
    --query 'KeyMaterial' \
    --output text > my-key.pem
chmod 400 my-key.pem

# 인스턴스 생성
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MyServer}]'
```

### 4.2 GCP Compute Engine 인스턴스 생성

**Console:**
1. Compute Engine → VM 인스턴스 → "만들기"
2. 이름 입력
3. 리전/Zone 선택
4. 머신 구성 선택 (예: e2-medium)
5. 부팅 디스크 (OS 이미지 선택)
6. 방화벽 설정 (HTTP/HTTPS 허용)
7. "만들기"

**gcloud CLI:**
```bash
# 인스턴스 생성
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --tags=http-server,https-server

# SSH 키는 자동 관리 (OS Login 또는 프로젝트 메타데이터)
```

---

## 5. SSH 접속

### 5.1 AWS EC2 SSH 접속

```bash
# 퍼블릭 IP 확인
aws ec2 describe-instances \
    --instance-ids i-1234567890abcdef0 \
    --query 'Reservations[0].Instances[0].PublicIpAddress'

# SSH 접속
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

# Amazon Linux: ec2-user
# Ubuntu: ubuntu
# CentOS: centos
# Debian: admin
```

**EC2 Instance Connect (브라우저):**
1. EC2 Console → 인스턴스 선택
2. "연결" 버튼 클릭
3. "EC2 Instance Connect" 탭
4. "연결" 클릭

### 5.2 GCP SSH 접속

```bash
# gcloud로 SSH (키 자동 관리)
gcloud compute ssh my-instance --zone=asia-northeast3-a

# 외부 IP 확인
gcloud compute instances describe my-instance \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# 직접 SSH (키를 수동 등록한 경우)
ssh -i ~/.ssh/google_compute_engine username@<EXTERNAL_IP>
```

**브라우저 SSH:**
1. Compute Engine → VM 인스턴스
2. 인스턴스 행의 "SSH" 버튼 클릭
3. 새 창에서 브라우저 터미널 열림

---

## 6. User Data / Startup Script

인스턴스 시작 시 자동으로 실행되는 스크립트입니다.

### 6.1 AWS User Data

```bash
#!/bin/bash
# User Data 예시 (Amazon Linux 2023)

# 패키지 업데이트
dnf update -y

# Nginx 설치
dnf install -y nginx
systemctl start nginx
systemctl enable nginx

# 커스텀 페이지
echo "<h1>Hello from $(hostname)</h1>" > /usr/share/nginx/html/index.html
```

**CLI에서 User Data 지정:**
```bash
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t3.micro \
    --user-data file://startup.sh \
    ...
```

**User Data 로그 확인:**
```bash
# 인스턴스 내부에서
cat /var/log/cloud-init-output.log
```

### 6.2 GCP Startup Script

```bash
#!/bin/bash
# Startup Script 예시 (Ubuntu)

# 패키지 업데이트
apt-get update

# Nginx 설치
apt-get install -y nginx
systemctl start nginx
systemctl enable nginx

# 커스텀 페이지
echo "<h1>Hello from $(hostname)</h1>" > /var/www/html/index.html
```

**CLI에서 Startup Script 지정:**
```bash
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --metadata-from-file=startup-script=startup.sh \
    ...

# 또는 인라인으로
gcloud compute instances create my-instance \
    --metadata=startup-script='#!/bin/bash
    apt-get update
    apt-get install -y nginx'
```

**Startup Script 로그 확인:**
```bash
# 인스턴스 내부에서
sudo journalctl -u google-startup-scripts.service
# 또는
cat /var/log/syslog | grep startup-script
```

---

## 7. 인스턴스 메타데이터

인스턴스 내부에서 자신의 정보를 조회할 수 있습니다.

### 7.1 AWS Instance Metadata Service (IMDS)

```bash
# 인스턴스 ID
curl http://169.254.169.254/latest/meta-data/instance-id

# 퍼블릭 IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# 가용 영역
curl http://169.254.169.254/latest/meta-data/placement/availability-zone

# IAM 역할 자격 증명
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/<role-name>

# IMDSv2 (권장 - 토큰 필요)
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id
```

### 7.2 GCP Metadata Server

```bash
# 인스턴스 이름
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name

# 외부 IP
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip

# Zone
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone

# 서비스 계정 토큰
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token

# 프로젝트 ID
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id
```

---

## 8. 인스턴스 관리

### 8.1 인스턴스 상태 관리

**AWS:**
```bash
# 인스턴스 중지
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# 인스턴스 시작
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# 인스턴스 재부팅
aws ec2 reboot-instances --instance-ids i-1234567890abcdef0

# 인스턴스 종료 (삭제)
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# 인스턴스 상태 확인
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# 인스턴스 중지
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# 인스턴스 시작
gcloud compute instances start my-instance --zone=asia-northeast3-a

# 인스턴스 재시작 (reset)
gcloud compute instances reset my-instance --zone=asia-northeast3-a

# 인스턴스 삭제
gcloud compute instances delete my-instance --zone=asia-northeast3-a

# 인스턴스 상태 확인
gcloud compute instances describe my-instance --zone=asia-northeast3-a
```

### 8.2 인스턴스 유형 변경

**AWS:**
```bash
# 1. 인스턴스 중지
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# 2. 인스턴스 유형 변경
aws ec2 modify-instance-attribute \
    --instance-id i-1234567890abcdef0 \
    --instance-type t3.large

# 3. 인스턴스 시작
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# 1. 인스턴스 중지
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# 2. 머신 타입 변경
gcloud compute instances set-machine-type my-instance \
    --zone=asia-northeast3-a \
    --machine-type=n2-standard-4

# 3. 인스턴스 시작
gcloud compute instances start my-instance --zone=asia-northeast3-a
```

---

## 9. 과금 옵션

### 9.1 온디맨드 vs 예약 vs 스팟

| 옵션 | AWS | GCP | 할인율 | 특징 |
|------|-----|-----|--------|------|
| **온디맨드** | On-Demand | On-demand | 0% | 약정 없음, 유연함 |
| **예약** | Reserved/Savings Plans | Committed Use | 최대 72% | 1-3년 약정 |
| **스팟/선점형** | Spot Instances | Spot/Preemptible | 최대 90% | 중단 가능 |
| **자동 할인** | - | Sustained Use | 최대 30% | 월 사용량 자동 |

### 9.2 AWS Spot Instance

```bash
# 스팟 인스턴스 요청
aws ec2 request-spot-instances \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification '{
        "ImageId": "ami-12345678",
        "InstanceType": "t3.large",
        "KeyName": "my-key"
    }'

# 스팟 가격 확인
aws ec2 describe-spot-price-history \
    --instance-types t3.large \
    --product-descriptions "Linux/UNIX"
```

### 9.3 GCP Preemptible/Spot VM

```bash
# Spot VM 생성 (Preemptible 후속)
gcloud compute instances create spot-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

# Preemptible VM 생성 (레거시)
gcloud compute instances create preemptible-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --preemptible
```

---

## 10. 실습: 웹 서버 배포

### 10.1 AWS EC2 웹 서버

```bash
# 1. 보안 그룹 생성
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server security group"

# 2. 인바운드 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

# 3. EC2 인스턴스 생성 (User Data 포함)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-groups web-sg \
    --user-data '#!/bin/bash
dnf update -y
dnf install -y nginx
systemctl start nginx
echo "<h1>AWS EC2 Web Server</h1>" > /usr/share/nginx/html/index.html' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'
```

### 10.2 GCP Compute Engine 웹 서버

```bash
# 1. 방화벽 규칙 생성
gcloud compute firewall-rules create allow-http \
    --allow tcp:80 \
    --target-tags http-server

# 2. Compute Engine 인스턴스 생성
gcloud compute instances create web-server \
    --zone=asia-northeast3-a \
    --machine-type=e2-micro \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y nginx
echo "<h1>GCP Compute Engine Web Server</h1>" > /var/www/html/index.html'

# 3. 외부 IP 확인
gcloud compute instances describe web-server \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

---

## 11. 다음 단계

- [05_Serverless_Functions.md](./05_Serverless_Functions.md) - 서버리스 함수
- [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) - 블록 스토리지 (EBS/PD)

---

## 참고 자료

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [GCP Compute Engine Documentation](https://cloud.google.com/compute/docs)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
- [GCP Machine Types](https://cloud.google.com/compute/docs/machine-types)
