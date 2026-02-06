# VPC (Virtual Private Cloud)

## 1. VPC 개요

### 1.1 VPC란?

VPC는 클라우드 내에서 논리적으로 격리된 가상 네트워크입니다.

**핵심 개념:**
- 자체 IP 주소 범위 정의
- 서브넷으로 분할
- 라우팅 테이블로 트래픽 제어
- 보안 그룹/방화벽으로 접근 통제

### 1.2 AWS vs GCP VPC 차이

| 항목 | AWS VPC | GCP VPC |
|------|---------|---------|
| **범위** | 리전 단위 | **글로벌** |
| **서브넷 범위** | 가용 영역 (AZ) | 리전 |
| **기본 VPC** | 리전당 1개 | 프로젝트당 1개 (default) |
| **피어링** | 리전 간 가능 | 글로벌 자동 |
| **IP 범위** | 생성 시 고정 | 서브넷 추가 가능 |

```
AWS VPC 구조:
┌──────────────────────────────────────────────────────────────┐
│  VPC (리전: ap-northeast-2)                                  │
│  CIDR: 10.0.0.0/16                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-a (AZ-a)     │  │ Subnet-b (AZ-b)     │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘

GCP VPC 구조:
┌──────────────────────────────────────────────────────────────┐
│  VPC (글로벌)                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-asia         │  │ Subnet-us           │            │
│  │ (asia-northeast3)   │  │ (us-central1)       │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 서브넷

### 2.1 퍼블릭 vs 프라이빗 서브넷

| 유형 | 인터넷 접근 | 용도 |
|------|-----------|------|
| **퍼블릭** | 직접 가능 | 웹 서버, Bastion |
| **프라이빗** | NAT 통해서만 | 애플리케이션, DB |

### 2.2 AWS 서브넷 생성

```bash
# 1. VPC 생성
aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=MyVPC}]'

# 2. 퍼블릭 서브넷 생성
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.1.0/24 \
    --availability-zone ap-northeast-2a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Public-Subnet-1}]'

# 3. 프라이빗 서브넷 생성
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.10.0/24 \
    --availability-zone ap-northeast-2a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Private-Subnet-1}]'

# 4. 퍼블릭 IP 자동 할당 (퍼블릭 서브넷)
aws ec2 modify-subnet-attribute \
    --subnet-id subnet-public \
    --map-public-ip-on-launch
```

### 2.3 GCP 서브넷 생성

```bash
# 1. 커스텀 모드 VPC 생성
gcloud compute networks create my-vpc \
    --subnet-mode=custom

# 2. 서브넷 생성 (서울)
gcloud compute networks subnets create subnet-asia \
    --network=my-vpc \
    --region=asia-northeast3 \
    --range=10.0.1.0/24

# 3. 서브넷 생성 (미국)
gcloud compute networks subnets create subnet-us \
    --network=my-vpc \
    --region=us-central1 \
    --range=10.0.2.0/24

# 4. 프라이빗 Google 액세스 활성화
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-private-ip-google-access
```

---

## 3. 인터넷 게이트웨이

### 3.1 AWS Internet Gateway

```bash
# 1. IGW 생성
aws ec2 create-internet-gateway \
    --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=MyIGW}]'

# 2. VPC에 연결
aws ec2 attach-internet-gateway \
    --internet-gateway-id igw-12345678 \
    --vpc-id vpc-12345678

# 3. 라우팅 테이블에 경로 추가
aws ec2 create-route \
    --route-table-id rtb-12345678 \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id igw-12345678

# 4. 퍼블릭 서브넷에 라우팅 테이블 연결
aws ec2 associate-route-table \
    --route-table-id rtb-12345678 \
    --subnet-id subnet-public
```

### 3.2 GCP 인터넷 접근

GCP는 별도의 인터넷 게이트웨이 없이 외부 IP가 있으면 인터넷 접근이 가능합니다.

```bash
# 외부 IP 할당 (인스턴스 생성 시)
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --network=my-vpc \
    --subnet=subnet-asia \
    --address=EXTERNAL_IP  # 또는 생략하면 임시 IP 할당

# 정적 IP 예약
gcloud compute addresses create my-static-ip \
    --region=asia-northeast3
```

---

## 4. NAT Gateway

프라이빗 서브넷의 인스턴스가 인터넷에 접근할 수 있도록 합니다.

### 4.1 AWS NAT Gateway

```bash
# 1. Elastic IP 할당
aws ec2 allocate-address --domain vpc

# 2. NAT Gateway 생성 (퍼블릭 서브넷에)
aws ec2 create-nat-gateway \
    --subnet-id subnet-public \
    --allocation-id eipalloc-12345678 \
    --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=MyNAT}]'

# 3. 프라이빗 라우팅 테이블에 경로 추가
aws ec2 create-route \
    --route-table-id rtb-private \
    --destination-cidr-block 0.0.0.0/0 \
    --nat-gateway-id nat-12345678

# 4. 프라이빗 서브넷에 라우팅 테이블 연결
aws ec2 associate-route-table \
    --route-table-id rtb-private \
    --subnet-id subnet-private
```

### 4.2 GCP Cloud NAT

```bash
# 1. Cloud Router 생성
gcloud compute routers create my-router \
    --network=my-vpc \
    --region=asia-northeast3

# 2. Cloud NAT 생성
gcloud compute routers nats create my-nat \
    --router=my-router \
    --region=asia-northeast3 \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges
```

---

## 5. 보안 그룹 / 방화벽

### 5.1 AWS Security Groups

보안 그룹은 인스턴스 레벨의 **상태 저장(stateful)** 방화벽입니다.

```bash
# 보안 그룹 생성
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server security group" \
    --vpc-id vpc-12345678

# 인바운드 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr 203.0.113.0/24  # 특정 IP만 허용

# 다른 보안 그룹에서 오는 트래픽 허용
aws ec2 authorize-security-group-ingress \
    --group-id sg-db \
    --protocol tcp \
    --port 3306 \
    --source-group sg-web

# 규칙 조회
aws ec2 describe-security-groups --group-ids sg-12345678
```

### 5.2 GCP Firewall Rules

GCP 방화벽 규칙은 VPC 레벨에서 작동하며 **태그** 또는 서비스 계정으로 대상을 지정합니다.

```bash
# HTTP 트래픽 허용 (태그 기반)
gcloud compute firewall-rules create allow-http \
    --network=my-vpc \
    --allow=tcp:80,tcp:443 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0

# SSH 허용 (특정 IP)
gcloud compute firewall-rules create allow-ssh \
    --network=my-vpc \
    --allow=tcp:22 \
    --target-tags=ssh-server \
    --source-ranges=203.0.113.0/24

# 내부 통신 허용
gcloud compute firewall-rules create allow-internal \
    --network=my-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/8

# 규칙 목록 조회
gcloud compute firewall-rules list --filter="network:my-vpc"

# 규칙 삭제
gcloud compute firewall-rules delete allow-http
```

### 5.3 AWS NACL (Network ACL)

NACL은 서브넷 레벨의 **상태 비저장(stateless)** 방화벽입니다.

```bash
# NACL 생성
aws ec2 create-network-acl \
    --vpc-id vpc-12345678 \
    --tag-specifications 'ResourceType=network-acl,Tags=[{Key=Name,Value=MyNACL}]'

# 인바운드 규칙 추가 (규칙 번호로 우선순위)
aws ec2 create-network-acl-entry \
    --network-acl-id acl-12345678 \
    --ingress \
    --rule-number 100 \
    --protocol tcp \
    --port-range From=80,To=80 \
    --cidr-block 0.0.0.0/0 \
    --rule-action allow

# 아웃바운드 규칙도 필요 (stateless)
aws ec2 create-network-acl-entry \
    --network-acl-id acl-12345678 \
    --egress \
    --rule-number 100 \
    --protocol tcp \
    --port-range From=1024,To=65535 \
    --cidr-block 0.0.0.0/0 \
    --rule-action allow
```

---

## 6. VPC 피어링

### 6.1 AWS VPC Peering

```bash
# 1. 피어링 연결 요청
aws ec2 create-vpc-peering-connection \
    --vpc-id vpc-requester \
    --peer-vpc-id vpc-accepter \
    --peer-region ap-northeast-1  # 다른 리전인 경우

# 2. 피어링 연결 수락
aws ec2 accept-vpc-peering-connection \
    --vpc-peering-connection-id pcx-12345678

# 3. 양쪽 VPC의 라우팅 테이블에 경로 추가
# Requester VPC
aws ec2 create-route \
    --route-table-id rtb-requester \
    --destination-cidr-block 10.1.0.0/16 \
    --vpc-peering-connection-id pcx-12345678

# Accepter VPC
aws ec2 create-route \
    --route-table-id rtb-accepter \
    --destination-cidr-block 10.0.0.0/16 \
    --vpc-peering-connection-id pcx-12345678
```

### 6.2 GCP VPC Peering

```bash
# 1. 첫 번째 VPC에서 피어링 생성
gcloud compute networks peerings create peer-vpc1-to-vpc2 \
    --network=vpc1 \
    --peer-network=vpc2

# 2. 두 번째 VPC에서 피어링 생성 (양쪽 필요)
gcloud compute networks peerings create peer-vpc2-to-vpc1 \
    --network=vpc2 \
    --peer-network=vpc1

# 라우팅은 자동으로 추가됨
```

---

## 7. 프라이빗 엔드포인트

인터넷을 거치지 않고 AWS/GCP 서비스에 접근합니다.

### 7.1 AWS VPC Endpoints

**Gateway Endpoint (S3, DynamoDB):**
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-12345678 \
    --service-name com.amazonaws.ap-northeast-2.s3 \
    --route-table-ids rtb-12345678
```

**Interface Endpoint (다른 서비스):**
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-12345678 \
    --service-name com.amazonaws.ap-northeast-2.secretsmanager \
    --vpc-endpoint-type Interface \
    --subnet-ids subnet-12345678 \
    --security-group-ids sg-12345678
```

### 7.2 GCP Private Service Connect

```bash
# Private Google Access 활성화
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-private-ip-google-access

# Private Service Connect 엔드포인트
gcloud compute addresses create psc-endpoint \
    --region=asia-northeast3 \
    --subnet=subnet-asia \
    --purpose=PRIVATE_SERVICE_CONNECT
```

---

## 8. 일반적인 VPC 아키텍처

### 8.1 3티어 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  VPC (10.0.0.0/16)                                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Public Subnets (10.0.1.0/24, 10.0.2.0/24)               ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │    ALB      │  │   Bastion   │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Private Subnets - App (10.0.10.0/24, 10.0.11.0/24)      ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │   App-1     │  │   App-2     │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Private Subnets - DB (10.0.20.0/24, 10.0.21.0/24)       ││
│  │  ┌─────────────┐  ┌─────────────┐                        ││
│  │  │  DB Primary │  │  DB Standby │                        ││
│  │  └─────────────┘  └─────────────┘                        ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │     IGW      │  │   NAT GW     │                         │
│  └──────────────┘  └──────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 AWS VPC 예시 (Terraform)

```hcl
# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "main-vpc" }
}

# Public Subnets
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "public-${count.index + 1}" }
}

# Private Subnets
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "private-${count.index + 1}" }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

# NAT Gateway
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
}
```

---

## 9. 문제 해결

### 9.1 연결 문제 체크리스트

```
□ 보안 그룹 인바운드 규칙 확인
□ NACL 규칙 확인 (AWS)
□ 방화벽 규칙 확인 (GCP)
□ 라우팅 테이블 확인
□ 인터넷 게이트웨이 연결 확인
□ NAT 게이트웨이 상태 확인
□ 인스턴스에 퍼블릭 IP 있는지 확인
□ VPC 피어링 라우팅 확인
```

### 9.2 디버깅 명령어

**AWS:**
```bash
# VPC Flow Logs 활성화
aws ec2 create-flow-logs \
    --resource-type VPC \
    --resource-ids vpc-12345678 \
    --traffic-type ALL \
    --log-destination-type cloud-watch-logs \
    --log-group-name vpc-flow-logs

# Reachability Analyzer
aws ec2 create-network-insights-path \
    --source i-source \
    --destination i-destination \
    --destination-port 80 \
    --protocol tcp
```

**GCP:**
```bash
# VPC Flow Logs 활성화
gcloud compute networks subnets update subnet-asia \
    --region=asia-northeast3 \
    --enable-flow-logs

# Connectivity Tests
gcloud network-management connectivity-tests create my-test \
    --source-instance=projects/PROJECT/zones/ZONE/instances/source \
    --destination-instance=projects/PROJECT/zones/ZONE/instances/dest \
    --destination-port=80 \
    --protocol=TCP
```

---

## 10. 다음 단계

- [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) - 로드밸런싱
- [14_Security_Services.md](./14_Security_Services.md) - 보안 상세

---

## 참고 자료

- [AWS VPC Documentation](https://docs.aws.amazon.com/vpc/)
- [GCP VPC Documentation](https://cloud.google.com/vpc/docs)
- [AWS VPC Best Practices](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-best-practices.html)
- [Networking/](../Networking/) - 네트워크 기초 이론
