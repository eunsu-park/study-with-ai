# Block and File Storage (EBS/EFS vs Persistent Disk/Filestore)

## 1. Storage Type Comparison

### 1.1 Block vs File vs Object Storage

| Type | Characteristics | Use Cases | AWS | GCP |
|------|------|----------|-----|-----|
| **Block** | Low-level disk access | Databases, OS boot disk | EBS | Persistent Disk |
| **File** | Shared filesystem | Shared storage, CMS | EFS | Filestore |
| **Object** | HTTP-based, unlimited | Backup, media, logs | S3 | Cloud Storage |

### 1.2 Service Mapping

| Feature | AWS | GCP |
|------|-----|-----|
| Block Storage | EBS (Elastic Block Store) | Persistent Disk (PD) |
| Shared File Storage | EFS (Elastic File System) | Filestore |
| Local SSD | Instance Store | Local SSD |

---

## 2. Block Storage

### 2.1 AWS EBS (Elastic Block Store)

**EBS Volume Types:**

| Type | Use Case | IOPS | Throughput | Cost |
|------|------|------|--------|------|
| **gp3** | General purpose SSD | Up to 16,000 | Up to 1,000 MB/s | Low |
| **gp2** | General purpose SSD (legacy) | Up to 16,000 | Up to 250 MB/s | Medium |
| **io2** | Provisioned IOPS | Up to 64,000 | Up to 1,000 MB/s | High |
| **st1** | Throughput-optimized HDD | Up to 500 | Up to 500 MB/s | Low |
| **sc1** | Cold HDD | Up to 250 | Up to 250 MB/s | Very low |

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

**Persistent Disk Types:**

| Type | Use Case | IOPS (read) | Throughput (read) | Cost |
|------|------|------------|--------------|------|
| **pd-standard** | HDD | Up to 7,500 | Up to 180 MB/s | Low |
| **pd-balanced** | SSD (balanced) | Up to 80,000 | Up to 1,200 MB/s | Medium |
| **pd-ssd** | SSD (high performance) | Up to 100,000 | Up to 1,200 MB/s | High |
| **pd-extreme** | High IOPS SSD | Up to 120,000 | Up to 2,400 MB/s | Very high |

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

## 3. Snapshots

### 3.1 AWS EBS Snapshots

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

**Automated Snapshots (Data Lifecycle Manager):**
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

### 3.2 GCP Snapshots

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

**Snapshot Schedule:**
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

## 4. Volume Expansion

### 4.1 AWS EBS Volume Expansion

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

### 4.2 GCP Persistent Disk Expansion

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

## 5. File Storage

### 5.1 AWS EFS (Elastic File System)

**Features:**
- NFS v4.1 protocol
- Auto-scaling (expand/shrink)
- Multi-AZ support
- Supports thousands of concurrent EC2 connections

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

**EFS Storage Classes:**
| Class | Use Case | Cost |
|--------|------|------|
| Standard | Frequent access | High |
| Infrequent Access (IA) | Infrequent access | Low |
| Archive | Long-term retention | Very low |

```bash
# 수명 주기 정책 설정 (30일 후 IA로 이동)
aws efs put-lifecycle-configuration \
    --file-system-id fs-12345678 \
    --lifecycle-policies '[{"TransitionToIA":"AFTER_30_DAYS"}]'
```

### 5.2 GCP Filestore

**Features:**
- NFS v3 protocol
- Pre-provisioned capacity
- High-performance options available

**Filestore Tiers:**
| Tier | Capacity | Performance | Use Case |
|------|------|------|------|
| Basic HDD | 1TB-63.9TB | 100 MB/s | File sharing |
| Basic SSD | 2.5TB-63.9TB | 1,200 MB/s | High performance |
| Zonal | 1TB-100TB | Up to 2,560 MB/s | High-performance workloads |
| Enterprise | 1TB-10TB | Up to 1,200 MB/s | Mission-critical |

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

## 6. Local SSD

### 6.1 AWS Instance Store

Instance Store is temporary storage physically attached to EC2 instances.

**Features:**
- Data loss on instance stop/termination
- Very high IOPS
- No additional cost (included in instance price)

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

**Local SSD Features:**
- Added in 375GB units
- Up to 24 units (9TB)
- Data loss on instance stop
- Live Migration not available (for some)

---

## 7. Performance Optimization

### 7.1 IOPS vs Throughput

```
IOPS (Input/Output Per Second):
- Number of read/write operations per second
- Important for small random I/O
- Databases, transaction processing

Throughput:
- Amount of data transferred per second (MB/s)
- Important for large sequential I/O
- Video streaming, big data
```

### 7.2 Optimization Tips

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

## 8. Cost Comparison

### 8.1 Block Storage Cost (Seoul Region)

| Type | AWS EBS | GCP PD |
|------|---------|--------|
| General Purpose SSD | $0.114/GB (gp3) | $0.102/GB (pd-balanced) |
| High Performance SSD | $0.138/GB (io1) | $0.180/GB (pd-ssd) |
| HDD | $0.054/GB (st1) | $0.044/GB (pd-standard) |

### 8.2 File Storage Cost

| Service | Cost |
|--------|------|
| AWS EFS Standard | ~$0.33/GB/month |
| AWS EFS IA | ~$0.025/GB/month |
| GCP Filestore Basic SSD | ~$0.24/GB/month |
| GCP Filestore Basic HDD | ~$0.12/GB/month |

---

## 9. Hands-on: Setting Up Shared Storage

### 9.1 AWS EFS Multi-Instance Mount

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

### 9.2 GCP Filestore Multi-Instance Mount

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

## 10. Next Steps

- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC Networking
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - Database Storage

---

## References

- [AWS EBS Documentation](https://docs.aws.amazon.com/ebs/)
- [AWS EFS Documentation](https://docs.aws.amazon.com/efs/)
- [GCP Persistent Disk](https://cloud.google.com/compute/docs/disks)
- [GCP Filestore](https://cloud.google.com/filestore/docs)
