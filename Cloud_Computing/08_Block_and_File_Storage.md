# 블록 및 파일 스토리지 (EBS/EFS vs Persistent Disk/Filestore)

## 1. 스토리지 유형 비교

### 1.1 블록 vs 파일 vs 객체 스토리지

| 유형 | 특징 | 사용 사례 | AWS | GCP |
|------|------|----------|-----|-----|
| **블록** | 저수준 디스크 접근 | DB, OS 부팅 디스크 | EBS | Persistent Disk |
| **파일** | 공유 파일시스템 | 공유 스토리지, CMS | EFS | Filestore |
| **객체** | HTTP 기반, 무제한 | 백업, 미디어, 로그 | S3 | Cloud Storage |

### 1.2 서비스 매핑

| 기능 | AWS | GCP |
|------|-----|-----|
| 블록 스토리지 | EBS (Elastic Block Store) | Persistent Disk (PD) |
| 공유 파일 스토리지 | EFS (Elastic File System) | Filestore |
| 로컬 SSD | Instance Store | Local SSD |

---

## 2. 블록 스토리지

### 2.1 AWS EBS (Elastic Block Store)

**EBS 볼륨 유형:**

| 유형 | 용도 | IOPS | 처리량 | 비용 |
|------|------|------|--------|------|
| **gp3** | 범용 SSD | 최대 16,000 | 최대 1,000 MB/s | 낮음 |
| **gp2** | 범용 SSD (이전) | 최대 16,000 | 최대 250 MB/s | 중간 |
| **io2** | 프로비저닝 IOPS | 최대 64,000 | 최대 1,000 MB/s | 높음 |
| **st1** | 처리량 최적화 HDD | 최대 500 | 최대 500 MB/s | 낮음 |
| **sc1** | 콜드 HDD | 최대 250 | 최대 250 MB/s | 매우 낮음 |

```bash
# EBS 볼륨 생성
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --size 100 \
    --volume-type gp3 \
    --iops 3000 \
    --throughput 125 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=MyVolume}]'

# EC2에 볼륨 연결
aws ec2 attach-volume \
    --volume-id vol-1234567890abcdef0 \
    --instance-id i-1234567890abcdef0 \
    --device /dev/sdf

# 인스턴스 내에서 마운트
sudo mkfs -t xfs /dev/xvdf
sudo mkdir /data
sudo mount /dev/xvdf /data

# fstab에 추가 (영구 마운트)
echo '/dev/xvdf /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

### 2.2 GCP Persistent Disk

**Persistent Disk 유형:**

| 유형 | 용도 | IOPS (읽기) | 처리량 (읽기) | 비용 |
|------|------|------------|--------------|------|
| **pd-standard** | HDD | 최대 7,500 | 최대 180 MB/s | 낮음 |
| **pd-balanced** | SSD (균형) | 최대 80,000 | 최대 1,200 MB/s | 중간 |
| **pd-ssd** | SSD (고성능) | 최대 100,000 | 최대 1,200 MB/s | 높음 |
| **pd-extreme** | 고IOPS SSD | 최대 120,000 | 최대 2,400 MB/s | 매우 높음 |

```bash
# Persistent Disk 생성
gcloud compute disks create my-disk \
    --zone=asia-northeast3-a \
    --size=100GB \
    --type=pd-ssd

# VM에 디스크 연결
gcloud compute instances attach-disk my-instance \
    --disk=my-disk \
    --zone=asia-northeast3-a

# 인스턴스 내에서 마운트
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir /data
sudo mount -o discard,defaults /dev/sdb /data

# fstab에 추가
echo UUID=$(sudo blkid -s UUID -o value /dev/sdb) /data ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
```

---

## 3. 스냅샷

### 3.1 AWS EBS 스냅샷

```bash
# 스냅샷 생성
aws ec2 create-snapshot \
    --volume-id vol-1234567890abcdef0 \
    --description "My snapshot" \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=MySnapshot}]'

# 스냅샷 목록 조회
aws ec2 describe-snapshots \
    --owner-ids self \
    --query 'Snapshots[*].[SnapshotId,VolumeId,StartTime,State]'

# 스냅샷에서 볼륨 복원
aws ec2 create-volume \
    --availability-zone ap-northeast-2a \
    --snapshot-id snap-1234567890abcdef0 \
    --volume-type gp3

# 스냅샷 복사 (다른 리전)
aws ec2 copy-snapshot \
    --source-region ap-northeast-2 \
    --source-snapshot-id snap-1234567890abcdef0 \
    --destination-region us-east-1

# 스냅샷 삭제
aws ec2 delete-snapshot --snapshot-id snap-1234567890abcdef0
```

**자동 스냅샷 (Data Lifecycle Manager):**
```bash
# DLM 정책 생성 (매일 스냅샷, 7일 보관)
aws dlm create-lifecycle-policy \
    --description "Daily snapshots" \
    --state ENABLED \
    --execution-role-arn arn:aws:iam::123456789012:role/AWSDataLifecycleManagerDefaultRole \
    --policy-details '{
        "ResourceTypes": ["VOLUME"],
        "TargetTags": [{"Key": "Backup", "Value": "true"}],
        "Schedules": [{
            "Name": "DailySnapshots",
            "CreateRule": {"Interval": 24, "IntervalUnit": "HOURS", "Times": ["03:00"]},
            "RetainRule": {"Count": 7}
        }]
    }'
```

### 3.2 GCP 스냅샷

```bash
# 스냅샷 생성
gcloud compute snapshots create my-snapshot \
    --source-disk=my-disk \
    --source-disk-zone=asia-northeast3-a

# 스냅샷 목록 조회
gcloud compute snapshots list

# 스냅샷에서 디스크 복원
gcloud compute disks create restored-disk \
    --source-snapshot=my-snapshot \
    --zone=asia-northeast3-a

# 스냅샷 삭제
gcloud compute snapshots delete my-snapshot
```

**스냅샷 스케줄:**
```bash
# 스케줄 정책 생성 (매일, 7일 보관)
gcloud compute resource-policies create snapshot-schedule daily-snapshot \
    --region=asia-northeast3 \
    --max-retention-days=7 \
    --start-time=04:00 \
    --daily-schedule

# 디스크에 스케줄 연결
gcloud compute disks add-resource-policies my-disk \
    --resource-policies=daily-snapshot \
    --zone=asia-northeast3-a
```

---

## 4. 볼륨 확장

### 4.1 AWS EBS 볼륨 확장

```bash
# 1. 볼륨 크기 수정 (온라인 가능)
aws ec2 modify-volume \
    --volume-id vol-1234567890abcdef0 \
    --size 200

# 2. 수정 상태 확인
aws ec2 describe-volumes-modifications \
    --volume-id vol-1234567890abcdef0

# 3. 인스턴스 내에서 파일시스템 확장
# XFS
sudo xfs_growfs -d /data

# ext4
sudo resize2fs /dev/xvdf
```

### 4.2 GCP Persistent Disk 확장

```bash
# 1. 디스크 크기 확장 (온라인 가능)
gcloud compute disks resize my-disk \
    --size=200GB \
    --zone=asia-northeast3-a

# 2. 인스턴스 내에서 파일시스템 확장
# ext4
sudo resize2fs /dev/sdb

# XFS
sudo xfs_growfs /data
```

---

## 5. 파일 스토리지

### 5.1 AWS EFS (Elastic File System)

**특징:**
- NFS v4.1 프로토콜
- 자동 확장/축소
- 멀티 AZ 지원
- 최대 수천 개 EC2 동시 연결

```bash
# 1. EFS 파일 시스템 생성
aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode bursting \
    --encrypted \
    --tags Key=Name,Value=my-efs

# 2. 마운트 타겟 생성 (각 서브넷)
aws efs create-mount-target \
    --file-system-id fs-12345678 \
    --subnet-id subnet-12345678 \
    --security-groups sg-12345678

# 3. EC2에서 마운트
sudo yum install -y amazon-efs-utils
sudo mkdir /efs
sudo mount -t efs fs-12345678:/ /efs

# 또는 NFS로 마운트
sudo mount -t nfs4 -o nfsvers=4.1 \
    fs-12345678.efs.ap-northeast-2.amazonaws.com:/ /efs

# fstab에 추가
echo 'fs-12345678:/ /efs efs defaults,_netdev 0 0' | sudo tee -a /etc/fstab
```

**EFS 스토리지 클래스:**
| 클래스 | 용도 | 비용 |
|--------|------|------|
| Standard | 자주 액세스 | 높음 |
| Infrequent Access (IA) | 드문 액세스 | 낮음 |
| Archive | 장기 보관 | 매우 낮음 |

```bash
# 수명 주기 정책 설정 (30일 후 IA로 이동)
aws efs put-lifecycle-configuration \
    --file-system-id fs-12345678 \
    --lifecycle-policies '[{"TransitionToIA":"AFTER_30_DAYS"}]'
```

### 5.2 GCP Filestore

**특징:**
- NFS v3 프로토콜
- 사전 프로비저닝된 용량
- 고성능 옵션 제공

**Filestore 티어:**
| 티어 | 용량 | 성능 | 용도 |
|------|------|------|------|
| Basic HDD | 1TB-63.9TB | 100 MB/s | 파일 공유 |
| Basic SSD | 2.5TB-63.9TB | 1,200 MB/s | 고성능 |
| Zonal | 1TB-100TB | 최대 2,560 MB/s | 고성능 워크로드 |
| Enterprise | 1TB-10TB | 최대 1,200 MB/s | 미션 크리티컬 |

```bash
# 1. Filestore 인스턴스 생성
gcloud filestore instances create my-filestore \
    --zone=asia-northeast3-a \
    --tier=BASIC_SSD \
    --file-share=name=vol1,capacity=2.5TB \
    --network=name=default

# 2. Filestore 정보 조회
gcloud filestore instances describe my-filestore \
    --zone=asia-northeast3-a

# 3. VM에서 마운트
sudo apt-get install -y nfs-common
sudo mkdir /filestore
sudo mount 10.0.0.2:/vol1 /filestore

# fstab에 추가
echo '10.0.0.2:/vol1 /filestore nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab
```

---

## 6. 로컬 SSD

### 6.1 AWS Instance Store

Instance Store는 EC2 인스턴스에 물리적으로 연결된 임시 스토리지입니다.

**특징:**
- 인스턴스 중지/종료 시 데이터 손실
- 매우 높은 IOPS
- 추가 비용 없음 (인스턴스 가격에 포함)

```bash
# Instance Store가 있는 인스턴스 유형 확인
aws ec2 describe-instance-types \
    --filters "Name=instance-storage-supported,Values=true" \
    --query 'InstanceTypes[*].[InstanceType,InstanceStorageInfo.TotalSizeInGB]'

# 예: i3.large, d2.xlarge 등

# 인스턴스 내에서 마운트
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir /local-ssd
sudo mount /dev/nvme1n1 /local-ssd
```

### 6.2 GCP Local SSD

```bash
# Local SSD가 있는 인스턴스 생성
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=n2-standard-4 \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME

# 인스턴스 내에서 마운트
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir /local-ssd
sudo mount /dev/nvme0n1 /local-ssd
```

**Local SSD 특징:**
- 375GB 단위로 추가
- 최대 24개 (9TB)
- 인스턴스 중지 시 데이터 손실
- Live Migration 불가 (일부)

---

## 7. 성능 최적화

### 7.1 IOPS vs 처리량

```
IOPS (Input/Output Per Second):
- 초당 읽기/쓰기 작업 수
- 작은 랜덤 I/O에 중요
- 데이터베이스, 트랜잭션 처리

처리량 (Throughput):
- 초당 전송 데이터량 (MB/s)
- 큰 순차 I/O에 중요
- 비디오 스트리밍, 빅데이터
```

### 7.2 최적화 팁

**AWS EBS:**
```bash
# gp3 IOPS/처리량 조정
aws ec2 modify-volume \
    --volume-id vol-xxx \
    --iops 10000 \
    --throughput 500

# EBS 최적화 인스턴스 사용
aws ec2 run-instances \
    --instance-type m5.large \
    --ebs-optimized \
    ...
```

**GCP Persistent Disk:**
```bash
# 더 큰 디스크 = 더 높은 성능
# pd-ssd 100GB: 최대 3,000 IOPS
# pd-ssd 500GB: 최대 15,000 IOPS
# pd-ssd 1TB: 최대 30,000 IOPS

# 성능을 위해 디스크 크기 증가
gcloud compute disks resize my-disk --size=500GB
```

---

## 8. 비용 비교

### 8.1 블록 스토리지 비용 (서울 리전)

| 유형 | AWS EBS | GCP PD |
|------|---------|--------|
| 범용 SSD | $0.114/GB (gp3) | $0.102/GB (pd-balanced) |
| 고성능 SSD | $0.138/GB (io1) | $0.180/GB (pd-ssd) |
| HDD | $0.054/GB (st1) | $0.044/GB (pd-standard) |

### 8.2 파일 스토리지 비용

| 서비스 | 비용 |
|--------|------|
| AWS EFS Standard | ~$0.33/GB/월 |
| AWS EFS IA | ~$0.025/GB/월 |
| GCP Filestore Basic SSD | ~$0.24/GB/월 |
| GCP Filestore Basic HDD | ~$0.12/GB/월 |

---

## 9. 실습: 공유 스토리지 설정

### 9.1 AWS EFS 멀티 인스턴스 마운트

```bash
# 1. 두 개의 서브넷에 마운트 타겟 생성
aws efs create-mount-target --file-system-id fs-xxx --subnet-id subnet-1 --security-groups sg-xxx
aws efs create-mount-target --file-system-id fs-xxx --subnet-id subnet-2 --security-groups sg-xxx

# 2. 보안 그룹에 NFS 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxx \
    --protocol tcp \
    --port 2049 \
    --source-group sg-instance

# 3. 각 인스턴스에서 마운트
# Instance 1
sudo mkdir /shared && sudo mount -t efs fs-xxx:/ /shared
echo "Hello from Instance 1" | sudo tee /shared/test.txt

# Instance 2
sudo mkdir /shared && sudo mount -t efs fs-xxx:/ /shared
cat /shared/test.txt  # "Hello from Instance 1" 출력
```

### 9.2 GCP Filestore 멀티 인스턴스 마운트

```bash
# 1. 방화벽 규칙 추가
gcloud compute firewall-rules create allow-nfs \
    --allow tcp:2049,tcp:111,udp:2049,udp:111 \
    --source-ranges 10.0.0.0/8

# 2. 각 인스턴스에서 마운트
# Instance 1
sudo mkdir /shared && sudo mount 10.0.0.2:/vol1 /shared
echo "Hello from Instance 1" | sudo tee /shared/test.txt

# Instance 2
sudo mkdir /shared && sudo mount 10.0.0.2:/vol1 /shared
cat /shared/test.txt  # "Hello from Instance 1" 출력
```

---

## 10. 다음 단계

- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC 네트워킹
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - 데이터베이스 스토리지

---

## 참고 자료

- [AWS EBS Documentation](https://docs.aws.amazon.com/ebs/)
- [AWS EFS Documentation](https://docs.aws.amazon.com/efs/)
- [GCP Persistent Disk](https://cloud.google.com/compute/docs/disks)
- [GCP Filestore](https://cloud.google.com/filestore/docs)
