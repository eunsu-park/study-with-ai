# 관리형 관계형 데이터베이스 (RDS / Cloud SQL)

## 1. 관리형 DB 개요

### 1.1 관리형 vs 자체 관리

| 작업 | 자체 관리 (EC2) | 관리형 (RDS/Cloud SQL) |
|------|----------------|----------------------|
| 하드웨어 프로비저닝 | 사용자 | 제공자 |
| OS 패치 | 사용자 | 제공자 |
| DB 설치/설정 | 사용자 | 제공자 |
| 백업 | 사용자 | 자동 |
| 고가용성 | 사용자 | 옵션 제공 |
| 스케일링 | 수동 | 버튼 클릭 |
| 모니터링 | 설정 필요 | 기본 제공 |

### 1.2 서비스 비교

| 항목 | AWS | GCP |
|------|-----|-----|
| 관리형 RDB | RDS | Cloud SQL |
| 고성능 DB | Aurora | Cloud Spanner, AlloyDB |
| 지원 엔진 | MySQL, PostgreSQL, MariaDB, Oracle, SQL Server | MySQL, PostgreSQL, SQL Server |

---

## 2. AWS RDS

### 2.1 RDS 인스턴스 생성

```bash
# DB 서브넷 그룹 생성
aws rds create-db-subnet-group \
    --db-subnet-group-name my-subnet-group \
    --db-subnet-group-description "My DB subnets" \
    --subnet-ids subnet-1 subnet-2

# 파라미터 그룹 생성 (선택)
aws rds create-db-parameter-group \
    --db-parameter-group-name my-params \
    --db-parameter-group-family mysql8.0 \
    --description "Custom parameters"

# RDS 인스턴스 생성
aws rds create-db-instance \
    --db-instance-identifier my-database \
    --db-instance-class db.t3.micro \
    --engine mysql \
    --engine-version 8.0 \
    --master-username admin \
    --master-user-password MyPassword123! \
    --allocated-storage 20 \
    --storage-type gp3 \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-12345678 \
    --backup-retention-period 7 \
    --multi-az \
    --publicly-accessible false

# 생성 상태 확인
aws rds describe-db-instances \
    --db-instance-identifier my-database \
    --query 'DBInstances[0].DBInstanceStatus'
```

### 2.2 Multi-AZ 배포

```
┌─────────────────────────────────────────────────────────────┐
│  VPC                                                        │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │     AZ-a            │  │     AZ-b            │          │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │          │
│  │  │  Primary DB   │──┼──┼──│  Standby DB   │  │          │
│  │  │  (읽기/쓰기)  │  │  │  │  (동기 복제)  │  │          │
│  │  └───────────────┘  │  │  └───────────────┘  │          │
│  └─────────────────────┘  └─────────────────────┘          │
│              ↑ 자동 장애 조치                               │
└─────────────────────────────────────────────────────────────┘
```

```bash
# 기존 인스턴스를 Multi-AZ로 변경
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --multi-az \
    --apply-immediately
```

### 2.3 읽기 복제본

```bash
# 읽기 복제본 생성 (같은 리전)
aws rds create-db-instance-read-replica \
    --db-instance-identifier my-read-replica \
    --source-db-instance-identifier my-database

# 다른 리전에 읽기 복제본 (크로스 리전)
aws rds create-db-instance-read-replica \
    --db-instance-identifier my-replica-us \
    --source-db-instance-identifier arn:aws:rds:ap-northeast-2:123456789012:db:my-database \
    --region us-east-1

# 복제본 승격 (마스터로 변환)
aws rds promote-read-replica \
    --db-instance-identifier my-read-replica
```

### 2.4 백업 및 복원

```bash
# 수동 스냅샷 생성
aws rds create-db-snapshot \
    --db-instance-identifier my-database \
    --db-snapshot-identifier my-snapshot-2024

# 스냅샷에서 복원
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier my-restored-db \
    --db-snapshot-identifier my-snapshot-2024

# 특정 시점 복원 (Point-in-Time Recovery)
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier my-database \
    --target-db-instance-identifier my-pitr-db \
    --restore-time 2024-01-15T10:00:00Z

# 자동 백업 설정 확인/변경
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --backup-retention-period 14 \
    --preferred-backup-window "03:00-04:00"
```

---

## 3. GCP Cloud SQL

### 3.1 Cloud SQL 인스턴스 생성

```bash
# Cloud SQL API 활성화
gcloud services enable sqladmin.googleapis.com

# MySQL 인스턴스 생성
gcloud sql instances create my-database \
    --database-version=MYSQL_8_0 \
    --tier=db-f1-micro \
    --region=asia-northeast3 \
    --root-password=MyPassword123! \
    --storage-size=10GB \
    --storage-type=SSD \
    --backup-start-time=03:00 \
    --availability-type=REGIONAL

# PostgreSQL 인스턴스 생성
gcloud sql instances create my-postgres \
    --database-version=POSTGRES_15 \
    --tier=db-g1-small \
    --region=asia-northeast3

# 인스턴스 정보 확인
gcloud sql instances describe my-database
```

### 3.2 고가용성 (HA)

```bash
# 고가용성 인스턴스 생성
gcloud sql instances create my-ha-db \
    --database-version=MYSQL_8_0 \
    --tier=db-n1-standard-2 \
    --region=asia-northeast3 \
    --availability-type=REGIONAL \
    --root-password=MyPassword123!

# 기존 인스턴스를 HA로 변경
gcloud sql instances patch my-database \
    --availability-type=REGIONAL
```

### 3.3 읽기 복제본

```bash
# 읽기 복제본 생성
gcloud sql instances create my-read-replica \
    --master-instance-name=my-database \
    --region=asia-northeast3

# 복제본 승격
gcloud sql instances promote-replica my-read-replica

# 복제본 목록 확인
gcloud sql instances list --filter="masterInstanceName:my-database"
```

### 3.4 백업 및 복원

```bash
# 온디맨드 백업 생성
gcloud sql backups create \
    --instance=my-database \
    --description="Manual backup"

# 백업 목록 확인
gcloud sql backups list --instance=my-database

# 백업에서 복원 (새 인스턴스)
gcloud sql instances restore-backup my-restored-db \
    --backup-instance=my-database \
    --backup-id=1234567890

# Point-in-Time Recovery
gcloud sql instances clone my-database my-pitr-db \
    --point-in-time="2024-01-15T10:00:00Z"
```

---

## 4. 연결 설정

### 4.1 AWS RDS 연결

**보안 그룹 설정:**
```bash
# RDS 보안 그룹에 애플리케이션 접근 허용
aws ec2 authorize-security-group-ingress \
    --group-id sg-rds \
    --protocol tcp \
    --port 3306 \
    --source-group sg-app

# 엔드포인트 확인
aws rds describe-db-instances \
    --db-instance-identifier my-database \
    --query 'DBInstances[0].Endpoint'
```

**애플리케이션에서 연결:**
```python
import pymysql

connection = pymysql.connect(
    host='my-database.xxxx.ap-northeast-2.rds.amazonaws.com',
    user='admin',
    password='MyPassword123!',
    database='mydb',
    port=3306
)
```

### 4.2 GCP Cloud SQL 연결

**연결 방법:**

1. **퍼블릭 IP (권장하지 않음)**
```bash
# 퍼블릭 IP 허용
gcloud sql instances patch my-database \
    --authorized-networks=203.0.113.0/24

# 연결
mysql -h <PUBLIC_IP> -u root -p
```

2. **Cloud SQL Proxy (권장)**
```bash
# Cloud SQL Proxy 다운로드
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy

# Proxy 실행
./cloud-sql-proxy PROJECT_ID:asia-northeast3:my-database

# 다른 터미널에서 연결
mysql -h 127.0.0.1 -u root -p
```

3. **Private IP (VPC 내부)**
```bash
# Private IP 활성화
gcloud sql instances patch my-database \
    --network=projects/PROJECT_ID/global/networks/my-vpc

# VPC 내 인스턴스에서 연결
mysql -h <PRIVATE_IP> -u root -p
```

**Python 연결 (Cloud SQL Connector):**
```python
from google.cloud.sql.connector import Connector
import pymysql

connector = Connector()

def get_conn():
    return connector.connect(
        "project:region:instance",
        "pymysql",
        user="root",
        password="password",
        db="mydb"
    )

connection = get_conn()
```

---

## 5. 성능 최적화

### 5.1 인스턴스 크기 조정

**AWS RDS:**
```bash
# 인스턴스 클래스 변경
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --db-instance-class db.m5.large \
    --apply-immediately

# 스토리지 확장 (축소 불가)
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --allocated-storage 100
```

**GCP Cloud SQL:**
```bash
# 머신 타입 변경
gcloud sql instances patch my-database \
    --tier=db-n1-standard-4

# 스토리지 확장
gcloud sql instances patch my-database \
    --storage-size=100GB
```

### 5.2 파라미터 튜닝

**AWS RDS 파라미터 그룹:**
```bash
# 파라미터 변경
aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=max_connections,ParameterValue=500,ApplyMethod=pending-reboot"

aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=innodb_buffer_pool_size,ParameterValue={DBInstanceClassMemory*3/4},ApplyMethod=pending-reboot"
```

**GCP Cloud SQL 플래그:**
```bash
# 플래그 설정
gcloud sql instances patch my-database \
    --database-flags=max_connections=500,innodb_buffer_pool_size=1073741824
```

---

## 6. Aurora / AlloyDB / Spanner

### 6.1 AWS Aurora

Aurora는 클라우드 네이티브 관계형 데이터베이스입니다.

**특징:**
- MySQL/PostgreSQL 호환
- 최대 128TB 자동 확장
- 6개 복제본 (3개 AZ)
- 읽기 복제본 최대 15개
- 서버리스 옵션 (Aurora Serverless)

```bash
# Aurora 클러스터 생성
aws rds create-db-cluster \
    --db-cluster-identifier my-aurora \
    --engine aurora-mysql \
    --engine-version 8.0.mysql_aurora.3.04.0 \
    --master-username admin \
    --master-user-password MyPassword123! \
    --db-subnet-group-name my-subnet-group \
    --vpc-security-group-ids sg-12345678

# Aurora 인스턴스 추가
aws rds create-db-instance \
    --db-instance-identifier my-aurora-instance-1 \
    --db-cluster-identifier my-aurora \
    --db-instance-class db.r5.large \
    --engine aurora-mysql
```

### 6.2 GCP Cloud Spanner

Spanner는 글로벌 분산 관계형 데이터베이스입니다.

**특징:**
- 글로벌 트랜잭션
- 무제한 확장
- 99.999% SLA
- PostgreSQL 호환 인터페이스

```bash
# Spanner 인스턴스 생성
gcloud spanner instances create my-spanner \
    --config=regional-asia-northeast3 \
    --nodes=1 \
    --description="My Spanner instance"

# 데이터베이스 생성
gcloud spanner databases create mydb \
    --instance=my-spanner
```

### 6.3 GCP AlloyDB

AlloyDB는 PostgreSQL 호환 고성능 데이터베이스입니다.

```bash
# AlloyDB 클러스터 생성
gcloud alloydb clusters create my-cluster \
    --region=asia-northeast3 \
    --password=MyPassword123!

# 기본 인스턴스 생성
gcloud alloydb instances create primary \
    --cluster=my-cluster \
    --region=asia-northeast3 \
    --instance-type=PRIMARY \
    --cpu-count=2
```

---

## 7. 비용 비교

### 7.1 AWS RDS 비용 (서울)

| 인스턴스 | vCPU | 메모리 | 시간당 비용 |
|----------|------|--------|------------|
| db.t3.micro | 2 | 1 GB | ~$0.02 |
| db.t3.small | 2 | 2 GB | ~$0.04 |
| db.m5.large | 2 | 8 GB | ~$0.18 |
| db.r5.large | 2 | 16 GB | ~$0.26 |

**추가 비용:**
- 스토리지: gp3 $0.114/GB/월
- 백업: 보관량 × $0.095/GB/월
- Multi-AZ: 인스턴스 비용 2배

### 7.2 GCP Cloud SQL 비용 (서울)

| 티어 | vCPU | 메모리 | 시간당 비용 |
|------|------|--------|------------|
| db-f1-micro | 공유 | 0.6 GB | ~$0.01 |
| db-g1-small | 공유 | 1.7 GB | ~$0.03 |
| db-n1-standard-2 | 2 | 7.5 GB | ~$0.13 |
| db-n1-highmem-2 | 2 | 13 GB | ~$0.16 |

**추가 비용:**
- 스토리지: SSD $0.180/GB/월
- 고가용성: 인스턴스 비용 2배
- 백업: $0.08/GB/월

---

## 8. 보안

### 8.1 암호화

**AWS RDS:**
```bash
# 저장 시 암호화 (생성 시)
aws rds create-db-instance \
    --storage-encrypted \
    --kms-key-id arn:aws:kms:...:key/xxx \
    ...

# SSL 강제
aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=require_secure_transport,ParameterValue=1"
```

**GCP Cloud SQL:**
```bash
# SSL 인증서 생성
gcloud sql ssl client-certs create my-client \
    --instance=my-database \
    --common-name=my-client

# SSL 필수 설정
gcloud sql instances patch my-database \
    --require-ssl
```

### 8.2 IAM 인증

**AWS RDS IAM 인증:**
```bash
# IAM 인증 활성화
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --enable-iam-database-authentication

# 임시 토큰 생성
aws rds generate-db-auth-token \
    --hostname my-database.xxxx.rds.amazonaws.com \
    --port 3306 \
    --username iam_user
```

**GCP Cloud SQL IAM:**
```bash
# IAM 인증 활성화
gcloud sql instances patch my-database \
    --enable-database-flags \
    --database-flags=cloudsql_iam_authentication=on

# IAM 사용자 추가
gcloud sql users create user@example.com \
    --instance=my-database \
    --type=CLOUD_IAM_USER
```

---

## 9. 다음 단계

- [12_NoSQL_Databases.md](./12_NoSQL_Databases.md) - NoSQL 데이터베이스
- [PostgreSQL/](../PostgreSQL/) - PostgreSQL 상세

---

## 참고 자료

- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)
- [AWS Aurora Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/)
- [GCP Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [GCP Cloud Spanner](https://cloud.google.com/spanner/docs)
