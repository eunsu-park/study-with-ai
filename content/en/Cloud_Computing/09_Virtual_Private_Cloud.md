# VPC (Virtual Private Cloud)

## 1. VPC Overview

### 1.1 What is VPC?

VPC is a logically isolated virtual network within the cloud.

**Core Concepts:**
- Define your own IP address range
- Divide into subnets
- Control traffic with routing tables
- Access control with security groups/firewalls

### 1.2 AWS vs GCP VPC Differences

| Category | AWS VPC | GCP VPC |
|------|---------|---------|
| **Scope** | Regional | **Global** |
| **Subnet Scope** | Availability Zone (AZ) | Regional |
| **Default VPC** | 1 per region | 1 per project (default) |
| **Peering** | Cross-region possible | Global automatic |
| **IP Range** | Fixed at creation | Subnets can be added |

```
AWS VPC Structure:
┌──────────────────────────────────────────────────────────────┐
│  VPC (Region: ap-northeast-2)                                │
│  CIDR: 10.0.0.0/16                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-a (AZ-a)     │  │ Subnet-b (AZ-b)     │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘

GCP VPC Structure:
┌──────────────────────────────────────────────────────────────┐
│  VPC (Global)                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │ Subnet-asia         │  │ Subnet-us           │            │
│  │ (asia-northeast3)   │  │ (us-central1)       │            │
│  │ 10.0.1.0/24         │  │ 10.0.2.0/24         │            │
│  └─────────────────────┘  └─────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Subnets

### 2.1 Public vs Private Subnets

| Type | Internet Access | Use Case |
|------|-----------|------|
| **Public** | Direct access | Web servers, Bastion |
| **Private** | Only through NAT | Applications, Databases |

### 2.2 AWS Subnet Creation

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

### 2.3 GCP Subnet Creation

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

## 3. Internet Gateway

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

### 3.2 GCP Internet Access

GCP allows internet access without a separate internet gateway if an external IP is present.

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

Allows instances in private subnets to access the internet.

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

## 5. Security Groups / Firewalls

### 5.1 AWS Security Groups

Security groups are instance-level **stateful** firewalls.

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

GCP firewall rules operate at the VPC level and target resources using **tags** or service accounts.

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

NACL is a subnet-level **stateless** firewall.

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

## 6. VPC Peering

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

## 7. Private Endpoints

Access AWS/GCP services without going through the internet.

### 7.1 AWS VPC Endpoints

**Gateway Endpoint (S3, DynamoDB):**
```bash
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-12345678 \
    --service-name com.amazonaws.ap-northeast-2.s3 \
    --route-table-ids rtb-12345678
```

**Interface Endpoint (Other Services):**
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

## 8. Common VPC Architectures

### 8.1 3-Tier Architecture

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

### 8.2 AWS VPC Example (Terraform)

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

## 9. Troubleshooting

### 9.1 Connection Issue Checklist

```
□ Check security group inbound rules
□ Check NACL rules (AWS)
□ Check firewall rules (GCP)
□ Check routing tables
□ Verify internet gateway attachment
□ Check NAT gateway status
□ Verify instance has public IP
□ Check VPC peering routing
```

### 9.2 Debugging Commands

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

## 10. Next Steps

- [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) - Load Balancing
- [14_Security_Services.md](./14_Security_Services.md) - Security Details

---

## References

- [AWS VPC Documentation](https://docs.aws.amazon.com/vpc/)
- [GCP VPC Documentation](https://cloud.google.com/vpc/docs)
- [AWS VPC Best Practices](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-best-practices.html)
- [Networking/](../Networking/) - Networking Fundamentals
