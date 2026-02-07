# Managed Relational Databases (RDS / Cloud SQL)

## 1. Managed DB Overview

### 1.1 Managed vs Self-Managed

| Task | Self-Managed (EC2) | Managed (RDS/Cloud SQL) |
|------|----------------|----------------------|
| Hardware Provisioning | User | Provider |
| OS Patching | User | Provider |
| DB Installation/Setup | User | Provider |
| Backup | User | Automatic |
| High Availability | User | Option provided |
| Scaling | Manual | Button click |
| Monitoring | Setup required | Built-in |

### 1.2 Service Comparison

| Category | AWS | GCP |
|------|-----|-----|
| Managed RDB | RDS | Cloud SQL |
| High-Performance DB | Aurora | Cloud Spanner, AlloyDB |
| Supported Engines | MySQL, PostgreSQL, MariaDB, Oracle, SQL Server | MySQL, PostgreSQL, SQL Server |

---

## 2. AWS RDS

### 2.1 Creating an RDS Instance

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

### 2.2 Multi-AZ Deployment

```
┌─────────────────────────────────────────────────────────────┐
│  VPC                                                        │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │     AZ-a            │  │     AZ-b            │          │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │          │
│  │  │  Primary DB   │──┼──┼──│  Standby DB   │  │          │
│  │  │  (Read/Write) │  │  │  │  (Sync repl)  │  │          │
│  │  └───────────────┘  │  │  └───────────────┘  │          │
│  └─────────────────────┘  └─────────────────────┘          │
│              ↑ Automatic failover                           │
└─────────────────────────────────────────────────────────────┘
```

```bash
# 기존 인스턴스를 Multi-AZ로 변경
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --multi-az \
    --apply-immediately
```

### 2.3 Read Replicas

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

### 2.4 Backup and Restore

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

### 3.1 Creating a Cloud SQL Instance

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

### 3.2 High Availability (HA)

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

### 3.3 Read Replicas

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

### 3.4 Backup and Restore

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

## 4. Connection Setup

### 4.1 AWS RDS Connection

**Security Group Setup:**
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

**Application Connection:**
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

### 4.2 GCP Cloud SQL Connection

**Connection Methods:**

1. **Public IP (Not Recommended)**
```bash
# 퍼블릭 IP 허용
gcloud sql instances patch my-database \
    --authorized-networks=203.0.113.0/24

# 연결
mysql -h <PUBLIC_IP> -u root -p
```

2. **Cloud SQL Proxy (Recommended)**
```bash
# Cloud SQL Proxy 다운로드
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy

# Proxy 실행
./cloud-sql-proxy PROJECT_ID:asia-northeast3:my-database

# 다른 터미널에서 연결
mysql -h 127.0.0.1 -u root -p
```

3. **Private IP (Within VPC)**
```bash
# Private IP 활성화
gcloud sql instances patch my-database \
    --network=projects/PROJECT_ID/global/networks/my-vpc

# VPC 내 인스턴스에서 연결
mysql -h <PRIVATE_IP> -u root -p
```

**Python Connection (Cloud SQL Connector):**
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

## 5. Performance Optimization

### 5.1 Instance Resizing

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

### 5.2 Parameter Tuning

**AWS RDS Parameter Group:**
```bash
# 파라미터 변경
aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=max_connections,ParameterValue=500,ApplyMethod=pending-reboot"

aws rds modify-db-parameter-group \
    --db-parameter-group-name my-params \
    --parameters "ParameterName=innodb_buffer_pool_size,ParameterValue={DBInstanceClassMemory*3/4},ApplyMethod=pending-reboot"
```

**GCP Cloud SQL Flags:**
```bash
# 플래그 설정
gcloud sql instances patch my-database \
    --database-flags=max_connections=500,innodb_buffer_pool_size=1073741824
```

---

## 6. Aurora / AlloyDB / Spanner

### 6.1 AWS Aurora

Aurora is a cloud-native relational database.

**Features:**
- MySQL/PostgreSQL compatible
- Auto-scaling up to 128TB
- 6 replicas (3 AZs)
- Up to 15 read replicas
- Serverless option (Aurora Serverless)

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

Spanner is a globally distributed relational database.

**Features:**
- Global transactions
- Unlimited scaling
- 99.999% SLA
- PostgreSQL-compatible interface

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

AlloyDB is a PostgreSQL-compatible high-performance database.

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

## 7. Cost Comparison

### 7.1 AWS RDS Cost (Seoul)

| Instance | vCPU | Memory | Hourly Cost |
|----------|------|--------|------------|
| db.t3.micro | 2 | 1 GB | ~$0.02 |
| db.t3.small | 2 | 2 GB | ~$0.04 |
| db.m5.large | 2 | 8 GB | ~$0.18 |
| db.r5.large | 2 | 16 GB | ~$0.26 |

**Additional Costs:**
- Storage: gp3 $0.114/GB/month
- Backup: retention × $0.095/GB/month
- Multi-AZ: Instance cost × 2

### 7.2 GCP Cloud SQL Cost (Seoul)

| Tier | vCPU | Memory | Hourly Cost |
|------|------|--------|------------|
| db-f1-micro | Shared | 0.6 GB | ~$0.01 |
| db-g1-small | Shared | 1.7 GB | ~$0.03 |
| db-n1-standard-2 | 2 | 7.5 GB | ~$0.13 |
| db-n1-highmem-2 | 2 | 13 GB | ~$0.16 |

**Additional Costs:**
- Storage: SSD $0.180/GB/month
- High Availability: Instance cost × 2
- Backup: $0.08/GB/month

---

## 8. Security

### 8.1 Encryption

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

### 8.2 IAM Authentication

**AWS RDS IAM Authentication:**
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

## 9. Next Steps

- [12_NoSQL_Databases.md](./12_NoSQL_Databases.md) - NoSQL Databases
- [PostgreSQL/](../PostgreSQL/) - PostgreSQL Details

---

## References

- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)
- [AWS Aurora Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/)
- [GCP Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [GCP Cloud Spanner](https://cloud.google.com/spanner/docs)
