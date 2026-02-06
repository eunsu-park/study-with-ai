# 리전과 가용 영역

## 1. 글로벌 인프라 개요

클라우드 제공자는 전 세계에 분산된 데이터센터를 통해 서비스를 제공합니다.

### 1.1 인프라 계층 구조

```
글로벌 네트워크
├── 리전 (Region)
│   ├── 가용 영역 (Availability Zone / Zone)
│   │   └── 데이터센터
│   ├── 가용 영역
│   │   └── 데이터센터
│   └── 가용 영역
│       └── 데이터센터
├── 리전
│   └── ...
└── 엣지 로케이션 (CDN, DNS)
```

### 1.2 AWS vs GCP 용어 비교

| 개념 | AWS | GCP |
|------|-----|-----|
| 지리적 영역 | Region | Region |
| 독립 데이터센터 | Availability Zone (AZ) | Zone |
| 로컬 서비스 | Local Zones, Wavelength | - |
| CDN 엣지 | Edge Locations | Edge PoPs |
| 프라이빗 연결 | Direct Connect | Cloud Interconnect |

---

## 2. 리전 (Region)

### 2.1 정의

리전은 지리적으로 분리된 클라우드 인프라 영역입니다.

**특징:**
- 각 리전은 독립적으로 운영
- 리전 간 데이터 복제는 명시적 설정 필요
- 대부분의 서비스는 리전 단위로 제공

### 2.2 AWS 주요 리전

| 리전 코드 | 위치 | 한국에서 권장 |
|----------|------|--------------|
| ap-northeast-2 | 서울 | ✅ 가장 권장 |
| ap-northeast-1 | 도쿄 | ✅ 차선책 |
| ap-northeast-3 | 오사카 | 선택적 |
| ap-southeast-1 | 싱가포르 | 선택적 |
| us-east-1 | 버지니아 북부 | 글로벌 서비스 |
| us-west-2 | 오레곤 | 비용 최적화 |
| eu-west-1 | 아일랜드 | 유럽 서비스 |

```bash
# 현재 리전 확인
aws configure get region

# 리전 설정
aws configure set region ap-northeast-2

# 사용 가능한 리전 목록
aws ec2 describe-regions --output table
```

### 2.3 GCP 주요 리전

| 리전 코드 | 위치 | 한국에서 권장 |
|----------|------|--------------|
| asia-northeast3 | 서울 | ✅ 가장 권장 |
| asia-northeast1 | 도쿄 | ✅ 차선책 |
| asia-northeast2 | 오사카 | 선택적 |
| asia-southeast1 | 싱가포르 | 선택적 |
| us-central1 | 아이오와 | 무료 티어 |
| us-east1 | 사우스캐롤라이나 | 무료 티어 |
| europe-west1 | 벨기에 | 유럽 서비스 |

```bash
# 현재 리전 확인
gcloud config get-value compute/region

# 리전 설정
gcloud config set compute/region asia-northeast3

# 사용 가능한 리전 목록
gcloud compute regions list
```

---

## 3. 가용 영역 (Availability Zone / Zone)

### 3.1 정의

가용 영역은 리전 내의 독립적인 데이터센터 그룹입니다.

**특징:**
- 물리적으로 분리된 위치
- 독립적인 전력, 냉각, 네트워크
- 저지연 고속 네트워크로 연결
- 한 AZ 장애가 다른 AZ에 영향 없음

### 3.2 AWS 가용 영역

```
서울 리전 (ap-northeast-2)
├── ap-northeast-2a
├── ap-northeast-2b
├── ap-northeast-2c
└── ap-northeast-2d
```

**AZ 명명 규칙:**
- `{리전코드}{영역문자}` 형식
- 예: `ap-northeast-2a`, `us-east-1b`

```bash
# 가용 영역 목록 확인
aws ec2 describe-availability-zones --region ap-northeast-2

# 출력 예시
{
    "AvailabilityZones": [
        {
            "ZoneName": "ap-northeast-2a",
            "State": "available",
            "ZoneType": "availability-zone"
        },
        ...
    ]
}
```

### 3.3 GCP Zone

```
서울 리전 (asia-northeast3)
├── asia-northeast3-a
├── asia-northeast3-b
└── asia-northeast3-c
```

**Zone 명명 규칙:**
- `{리전코드}-{영역문자}` 형식
- 예: `asia-northeast3-a`, `us-central1-f`

```bash
# Zone 목록 확인
gcloud compute zones list --filter="region:asia-northeast3"

# 출력 예시
NAME                 REGION           STATUS
asia-northeast3-a    asia-northeast3  UP
asia-northeast3-b    asia-northeast3  UP
asia-northeast3-c    asia-northeast3  UP
```

---

## 4. 멀티 AZ 아키텍처

### 4.1 고가용성을 위한 설계

```
┌────────────────────────────────────────────────────────────┐
│                     리전 (Region)                          │
│  ┌──────────────────┐    ┌──────────────────┐             │
│  │    AZ-a          │    │    AZ-b          │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │   Web-1    │  │    │  │   Web-2    │  │             │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │   App-1    │  │    │  │   App-2    │  │             │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │  DB-Primary │ │───▶│  │ DB-Standby │  │  (동기 복제) │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  └──────────────────┘    └──────────────────┘             │
│                                                            │
│  ┌──────────────────────────────────────────┐             │
│  │           Load Balancer (리전 범위)       │             │
│  └──────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────┘
```

### 4.2 서비스별 Multi-AZ 옵션

**AWS:**

| 서비스 | Multi-AZ 방식 |
|--------|--------------|
| EC2 | Auto Scaling Group으로 분산 |
| RDS | Multi-AZ 옵션 활성화 |
| ElastiCache | 복제본 다른 AZ 배치 |
| ELB | 자동 Multi-AZ |
| S3 | 자동 Multi-AZ 복제 |

**GCP:**

| 서비스 | Multi-Zone 방식 |
|--------|----------------|
| Compute Engine | Instance Group으로 분산 |
| Cloud SQL | 고가용성 옵션 활성화 |
| Memorystore | 복제본 다른 Zone 배치 |
| Cloud Load Balancing | 자동 Multi-Zone |
| Cloud Storage | Regional 클래스 사용 |

---

## 5. 리전 선택 기준

### 5.1 주요 고려 사항

| 기준 | 설명 | 권장 |
|------|------|------|
| **지연 시간** | 사용자와의 물리적 거리 | 사용자 근처 리전 |
| **규정 준수** | 데이터 거주 요구사항 | 법적 요구사항 확인 |
| **서비스 가용성** | 모든 서비스가 모든 리전에 없음 | 필요 서비스 확인 |
| **비용** | 리전별 가격 차이 | 비용 비교 |
| **재해 복구** | DR 사이트 거리 | 충분히 먼 리전 |

### 5.2 지연 시간 테스트

**AWS 지연 시간 측정:**
```bash
# CloudPing 사이트 활용
# https://www.cloudping.info/

# 또는 직접 ping 테스트
ping ec2.ap-northeast-2.amazonaws.com
ping ec2.ap-northeast-1.amazonaws.com
```

**GCP 지연 시간 측정:**
```bash
# GCP Ping 테스트 사이트
# https://gcping.com/

# 또는 직접 측정
ping asia-northeast3-run.googleapis.com
```

### 5.3 서비스 가용성 확인

**AWS:**
- https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/

**GCP:**
- https://cloud.google.com/about/locations

### 5.4 비용 비교 (EC2/Compute Engine 예시)

| 인스턴스 타입 | 서울 (AWS) | 버지니아 (AWS) | 서울 (GCP) | 아이오와 (GCP) |
|--------------|-----------|---------------|-----------|---------------|
| 범용 2vCPU/8GB | ~$0.10/시간 | ~$0.08/시간 | ~$0.09/시간 | ~$0.07/시간 |

*가격은 변동될 수 있으므로 공식 가격표 확인 필요*

---

## 6. 글로벌/리전/존 서비스

### 6.1 AWS 서비스 범위

| 범위 | 서비스 예시 |
|------|-----------|
| **글로벌** | IAM, Route 53, CloudFront, WAF |
| **리전** | VPC, S3, Lambda, RDS, EC2 (AMI) |
| **가용 영역** | EC2 인스턴스, EBS 볼륨, 서브넷 |

### 6.2 GCP 서비스 범위

| 범위 | 서비스 예시 |
|------|-----------|
| **글로벌** | Cloud IAM, Cloud DNS, Cloud CDN, VPC (네트워크) |
| **리전** | Cloud Storage (Regional), Cloud SQL, Cloud Run |
| **존** | Compute Engine, Persistent Disk |

**GCP VPC 특이점:**
- GCP의 VPC는 **글로벌** 리소스 (AWS VPC는 리전 단위)
- 서브넷은 리전 범위

```
AWS VPC vs GCP VPC

AWS:
├── VPC (리전 범위) ─── us-east-1
│   ├── Subnet-a (AZ 범위) ─── us-east-1a
│   └── Subnet-b (AZ 범위) ─── us-east-1b
└── VPC (별도 리전) ─── ap-northeast-2
    └── Subnet-a ─── ap-northeast-2a

GCP:
└── VPC (글로벌)
    ├── Subnet-us (리전 범위) ─── us-central1
    ├── Subnet-asia (리전 범위) ─── asia-northeast3
    └── Subnet-eu (리전 범위) ─── europe-west1
```

---

## 7. 엣지 로케이션

### 7.1 CDN 엣지

**AWS CloudFront:**
- 400+ 엣지 로케이션
- 정적 콘텐츠 캐싱
- DDoS 보호 (AWS Shield)

**GCP Cloud CDN:**
- Google의 글로벌 엣지 네트워크 활용
- 자동 SSL/TLS
- Cloud Armor 통합

### 7.2 DNS 엣지

**AWS Route 53:**
- 글로벌 Anycast DNS
- 지연 시간 기반 라우팅
- 지리적 라우팅

**GCP Cloud DNS:**
- 글로벌 Anycast
- 100% 가용성 SLA
- DNSSEC 지원

---

## 8. 재해 복구 전략

### 8.1 DR 패턴

| 패턴 | RTO | RPO | 비용 | 설명 |
|------|-----|-----|------|------|
| **Backup & Restore** | 시간~일 | 시간~일 | 낮음 | 백업만 다른 리전에 저장 |
| **Pilot Light** | 분~시간 | 분~시간 | 중간 | 핵심 시스템만 대기 |
| **Warm Standby** | 분 | 분 | 높음 | 축소된 환경 상시 운영 |
| **Active-Active** | 초 | 거의 0 | 매우 높음 | 모든 리전 동시 운영 |

### 8.2 크로스 리전 복제

**AWS S3 크로스 리전 복제:**
```bash
# S3 버킷 복제 설정
aws s3api put-bucket-replication \
    --bucket source-bucket \
    --replication-configuration '{
        "Role": "arn:aws:iam::account-id:role/replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Destination": {
                "Bucket": "arn:aws:s3:::destination-bucket"
            }
        }]
    }'
```

**GCP Cloud Storage 복제:**
```bash
# Dual-region 또는 Multi-region 버킷 사용
gsutil mb -l asia gs://my-multi-region-bucket

# 또는 Storage Transfer Service로 복제
gcloud transfer jobs create \
    gs://source-bucket gs://destination-bucket
```

---

## 9. 실습: 리전/AZ 정보 조회

### 9.1 AWS CLI 실습

```bash
# 1. 모든 리전 목록
aws ec2 describe-regions --query 'Regions[*].RegionName' --output text

# 2. 서울 리전의 AZ 목록
aws ec2 describe-availability-zones \
    --region ap-northeast-2 \
    --query 'AvailabilityZones[*].[ZoneName,State]' \
    --output table

# 3. 특정 서비스의 리전별 가용성 확인 (SSM 파라미터)
aws ssm get-parameters-by-path \
    --path /aws/service/global-infrastructure/regions \
    --query 'Parameters[*].Name'
```

### 9.2 GCP gcloud 실습

```bash
# 1. 모든 리전 목록
gcloud compute regions list --format="value(name)"

# 2. 서울 리전의 Zone 목록
gcloud compute zones list \
    --filter="region:asia-northeast3" \
    --format="table(name,status)"

# 3. 특정 리전의 머신 타입 확인
gcloud compute machine-types list \
    --filter="zone:asia-northeast3-a" \
    --limit=10
```

---

## 10. 다음 단계

- [04_Virtual_Machines.md](./04_Virtual_Machines.md) - 가상 머신 생성 및 관리
- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC 네트워킹

---

## 참고 자료

- [AWS Global Infrastructure](https://aws.amazon.com/about-aws/global-infrastructure/)
- [GCP Locations](https://cloud.google.com/about/locations)
- [AWS Regions and Availability Zones](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)
- [GCP Regions and Zones](https://cloud.google.com/compute/docs/regions-zones)
