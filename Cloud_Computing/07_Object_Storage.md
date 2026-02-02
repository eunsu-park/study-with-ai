# 객체 스토리지 (S3 / Cloud Storage)

## 1. 객체 스토리지 개요

### 1.1 객체 스토리지란?

객체 스토리지는 데이터를 객체 단위로 저장하는 스토리지 아키텍처입니다.

**객체 구성요소:**
- **데이터**: 실제 파일 내용
- **메타데이터**: 파일 정보 (생성일, 크기, 커스텀 속성)
- **고유 식별자**: 객체를 찾기 위한 키

### 1.2 서비스 비교

| 항목 | AWS S3 | GCP Cloud Storage |
|------|--------|------------------|
| 서비스명 | Simple Storage Service | Cloud Storage |
| 컨테이너 단위 | Bucket | Bucket |
| 최대 객체 크기 | 5TB | 5TB |
| 멀티파트 업로드 | 지원 (5MB-5GB 파트) | 지원 (복합 업로드) |
| 버전 관리 | Versioning | Object Versioning |
| 수명 주기 | Lifecycle Rules | Lifecycle Management |
| 암호화 | SSE-S3, SSE-KMS, SSE-C | Google-managed, CMEK, CSEK |

---

## 2. 스토리지 클래스

### 2.1 AWS S3 스토리지 클래스

| 클래스 | 용도 | 가용성 | 최소 저장 기간 |
|--------|------|--------|---------------|
| **S3 Standard** | 자주 액세스 | 99.99% | - |
| **S3 Intelligent-Tiering** | 액세스 패턴 불명 | 99.9% | - |
| **S3 Standard-IA** | 드문 액세스 | 99.9% | 30일 |
| **S3 One Zone-IA** | 드문 액세스 (단일 AZ) | 99.5% | 30일 |
| **S3 Glacier Instant** | 아카이브 (즉시 액세스) | 99.9% | 90일 |
| **S3 Glacier Flexible** | 아카이브 (분~시간) | 99.99% | 90일 |
| **S3 Glacier Deep Archive** | 장기 아카이브 | 99.99% | 180일 |

### 2.2 GCP Cloud Storage 클래스

| 클래스 | 용도 | 가용성 SLA | 최소 저장 기간 |
|--------|------|-----------|---------------|
| **Standard** | 자주 액세스 | 99.95% (리전) | - |
| **Nearline** | 월 1회 미만 액세스 | 99.9% | 30일 |
| **Coldline** | 분기 1회 미만 액세스 | 99.9% | 90일 |
| **Archive** | 연 1회 미만 액세스 | 99.9% | 365일 |

### 2.3 비용 비교 (서울 리전 기준)

| 클래스 | S3 ($/GB/월) | GCS ($/GB/월) |
|--------|-------------|---------------|
| 표준 | $0.025 | $0.023 |
| 드문 액세스 | $0.0138 | $0.016 (Nearline) |
| 아카이브 | $0.005 (Glacier) | $0.0025 (Archive) |

*가격은 변동될 수 있음*

---

## 3. 버킷 생성 및 관리

### 3.1 AWS S3 버킷

```bash
# 버킷 생성
aws s3 mb s3://my-unique-bucket-name-2024 --region ap-northeast-2

# 버킷 목록 조회
aws s3 ls

# 버킷 내용 조회
aws s3 ls s3://my-bucket/

# 버킷 삭제 (비어있어야 함)
aws s3 rb s3://my-bucket

# 버킷 삭제 (내용 포함)
aws s3 rb s3://my-bucket --force
```

**버킷 이름 규칙:**
- 전역적으로 고유해야 함
- 3-63자
- 소문자, 숫자, 하이픈만 사용
- 문자 또는 숫자로 시작/끝

### 3.2 GCP Cloud Storage 버킷

```bash
# 버킷 생성
gsutil mb -l asia-northeast3 gs://my-unique-bucket-name-2024

# 또는 gcloud 사용
gcloud storage buckets create gs://my-bucket \
    --location=asia-northeast3

# 버킷 목록 조회
gsutil ls
# 또는
gcloud storage buckets list

# 버킷 내용 조회
gsutil ls gs://my-bucket/

# 버킷 삭제
gsutil rb gs://my-bucket

# 버킷 삭제 (내용 포함)
gsutil rm -r gs://my-bucket
```

---

## 4. 객체 업로드/다운로드

### 4.1 AWS S3 객체 작업

```bash
# 단일 파일 업로드
aws s3 cp myfile.txt s3://my-bucket/

# 폴더 업로드 (재귀)
aws s3 cp ./local-folder s3://my-bucket/remote-folder --recursive

# 파일 다운로드
aws s3 cp s3://my-bucket/myfile.txt ./

# 폴더 다운로드
aws s3 cp s3://my-bucket/folder ./local-folder --recursive

# 동기화 (변경된 파일만)
aws s3 sync ./local-folder s3://my-bucket/folder
aws s3 sync s3://my-bucket/folder ./local-folder

# 파일 삭제
aws s3 rm s3://my-bucket/myfile.txt

# 폴더 삭제
aws s3 rm s3://my-bucket/folder --recursive

# 파일 이동
aws s3 mv s3://my-bucket/file1.txt s3://my-bucket/folder/file1.txt

# 파일 복사
aws s3 cp s3://source-bucket/file.txt s3://dest-bucket/file.txt
```

### 4.2 GCP Cloud Storage 객체 작업

```bash
# 단일 파일 업로드
gsutil cp myfile.txt gs://my-bucket/

# 또는 gcloud 사용
gcloud storage cp myfile.txt gs://my-bucket/

# 폴더 업로드 (재귀)
gsutil cp -r ./local-folder gs://my-bucket/

# 파일 다운로드
gsutil cp gs://my-bucket/myfile.txt ./

# 폴더 다운로드
gsutil cp -r gs://my-bucket/folder ./

# 동기화
gsutil rsync -r ./local-folder gs://my-bucket/folder

# 파일 삭제
gsutil rm gs://my-bucket/myfile.txt

# 폴더 삭제
gsutil rm -r gs://my-bucket/folder

# 파일 이동
gsutil mv gs://my-bucket/file1.txt gs://my-bucket/folder/

# 파일 복사
gsutil cp gs://source-bucket/file.txt gs://dest-bucket/
```

### 4.3 대용량 파일 업로드

**AWS S3 멀티파트 업로드:**
```bash
# AWS CLI는 자동으로 멀티파트 업로드 사용 (8MB 이상)
aws s3 cp large-file.zip s3://my-bucket/ \
    --expected-size 10737418240  # 10GB

# 멀티파트 설정 조정
aws configure set s3.multipart_threshold 64MB
aws configure set s3.multipart_chunksize 16MB
```

**GCP 복합 업로드:**
```bash
# gsutil은 자동으로 복합 업로드 사용 (150MB 이상)
gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
    cp large-file.zip gs://my-bucket/
```

---

## 5. 수명 주기 관리

### 5.1 AWS S3 Lifecycle

```json
{
    "Rules": [
        {
            "ID": "Move to IA after 30 days",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "logs/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ],
            "Expiration": {
                "Days": 365
            }
        },
        {
            "ID": "Delete old versions",
            "Status": "Enabled",
            "Filter": {},
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 30
            }
        }
    ]
}
```

```bash
# 수명 주기 정책 적용
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json

# 수명 주기 정책 조회
aws s3api get-bucket-lifecycle-configuration --bucket my-bucket
```

### 5.2 GCP Lifecycle Management

```json
{
    "lifecycle": {
        "rule": [
            {
                "action": {
                    "type": "SetStorageClass",
                    "storageClass": "NEARLINE"
                },
                "condition": {
                    "age": 30,
                    "matchesPrefix": ["logs/"]
                }
            },
            {
                "action": {
                    "type": "SetStorageClass",
                    "storageClass": "COLDLINE"
                },
                "condition": {
                    "age": 90
                }
            },
            {
                "action": {
                    "type": "Delete"
                },
                "condition": {
                    "age": 365
                }
            }
        ]
    }
}
```

```bash
# 수명 주기 정책 적용
gsutil lifecycle set lifecycle.json gs://my-bucket

# 수명 주기 정책 조회
gsutil lifecycle get gs://my-bucket
```

---

## 6. 접근 제어

### 6.1 AWS S3 접근 제어

**버킷 정책:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicRead",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*"
        }
    ]
}
```

```bash
# 버킷 정책 적용
aws s3api put-bucket-policy \
    --bucket my-bucket \
    --policy file://bucket-policy.json

# 퍼블릭 액세스 차단 (권장)
aws s3api put-public-access-block \
    --bucket my-bucket \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

**미리 서명된 URL (Presigned URL):**
```bash
# 다운로드 URL 생성 (1시간 유효)
aws s3 presign s3://my-bucket/private-file.pdf --expires-in 3600

# 업로드 URL 생성
aws s3 presign s3://my-bucket/uploads/file.txt --expires-in 3600
```

### 6.2 GCP Cloud Storage 접근 제어

**IAM 정책:**
```bash
# 사용자에게 버킷 접근 권한 부여
gsutil iam ch user:user@example.com:objectViewer gs://my-bucket

# 모든 사용자에게 읽기 권한 (퍼블릭)
gsutil iam ch allUsers:objectViewer gs://my-bucket
```

**균일 버킷 수준 액세스 (권장):**
```bash
# 균일 액세스 활성화
gsutil uniformbucketlevelaccess set on gs://my-bucket
```

**서명된 URL:**
```bash
# 다운로드 URL 생성 (1시간 유효)
gsutil signurl -d 1h service-account.json gs://my-bucket/private-file.pdf

# gcloud 사용
gcloud storage sign-url gs://my-bucket/file.pdf \
    --private-key-file=key.json \
    --duration=1h
```

---

## 7. 정적 웹사이트 호스팅

### 7.1 AWS S3 정적 호스팅

```bash
# 1. 정적 웹사이트 호스팅 활성화
aws s3 website s3://my-bucket/ \
    --index-document index.html \
    --error-document error.html

# 2. 퍼블릭 액세스 허용 (블록 해제)
aws s3api put-public-access-block \
    --bucket my-bucket \
    --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# 3. 버킷 정책 (퍼블릭 읽기)
aws s3api put-bucket-policy --bucket my-bucket --policy '{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-bucket/*"
    }]
}'

# 4. 파일 업로드
aws s3 sync ./website s3://my-bucket/

# 웹사이트 URL: http://my-bucket.s3-website.ap-northeast-2.amazonaws.com
```

### 7.2 GCP Cloud Storage 정적 호스팅

```bash
# 1. 버킷 생성 (도메인 이름과 일치하면 커스텀 도메인 가능)
gsutil mb -l asia-northeast3 gs://www.example.com

# 2. 웹사이트 설정
gsutil web set -m index.html -e 404.html gs://my-bucket

# 3. 퍼블릭 액세스 허용
gsutil iam ch allUsers:objectViewer gs://my-bucket

# 4. 파일 업로드
gsutil cp -r ./website/* gs://my-bucket/

# 웹사이트 URL: https://storage.googleapis.com/my-bucket/index.html
# 로드 밸런서를 통해 커스텀 도메인 설정 가능
```

---

## 8. 버전 관리

### 8.1 AWS S3 Versioning

```bash
# 버전 관리 활성화
aws s3api put-bucket-versioning \
    --bucket my-bucket \
    --versioning-configuration Status=Enabled

# 버전 관리 상태 확인
aws s3api get-bucket-versioning --bucket my-bucket

# 모든 버전 조회
aws s3api list-object-versions --bucket my-bucket

# 특정 버전 다운로드
aws s3api get-object \
    --bucket my-bucket \
    --key myfile.txt \
    --version-id "abc123" \
    myfile-old.txt

# 특정 버전 삭제
aws s3api delete-object \
    --bucket my-bucket \
    --key myfile.txt \
    --version-id "abc123"
```

### 8.2 GCP Object Versioning

```bash
# 버전 관리 활성화
gsutil versioning set on gs://my-bucket

# 버전 관리 상태 확인
gsutil versioning get gs://my-bucket

# 모든 버전 조회
gsutil ls -a gs://my-bucket/

# 특정 버전 다운로드
gsutil cp gs://my-bucket/myfile.txt#1234567890123456 ./

# 특정 버전 삭제
gsutil rm gs://my-bucket/myfile.txt#1234567890123456
```

---

## 9. 크로스 리전 복제

### 9.1 AWS S3 Cross-Region Replication

```bash
# 1. 소스 버킷 버전 관리 활성화
aws s3api put-bucket-versioning \
    --bucket source-bucket \
    --versioning-configuration Status=Enabled

# 2. 대상 버킷 생성 및 버전 관리 활성화
aws s3 mb s3://dest-bucket --region eu-west-1
aws s3api put-bucket-versioning \
    --bucket dest-bucket \
    --versioning-configuration Status=Enabled

# 3. 복제 규칙 설정
aws s3api put-bucket-replication \
    --bucket source-bucket \
    --replication-configuration '{
        "Role": "arn:aws:iam::123456789012:role/s3-replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Priority": 1,
            "DeleteMarkerReplication": {"Status": "Disabled"},
            "Filter": {},
            "Destination": {
                "Bucket": "arn:aws:s3:::dest-bucket"
            }
        }]
    }'
```

### 9.2 GCP Dual/Multi-Region Buckets

```bash
# 듀얼 리전 버킷 생성
gsutil mb -l asia1 gs://my-dual-region-bucket

# 또는 멀티 리전 버킷
gsutil mb -l asia gs://my-multi-region-bucket

# 리전 간 복사 (수동)
gsutil cp -r gs://source-bucket/* gs://dest-bucket/
```

---

## 10. SDK 사용 예시

### 10.1 Python (boto3 / google-cloud-storage)

**AWS S3 (boto3):**
```python
import boto3

s3 = boto3.client('s3')

# 업로드
s3.upload_file('local_file.txt', 'my-bucket', 'remote_file.txt')

# 다운로드
s3.download_file('my-bucket', 'remote_file.txt', 'local_file.txt')

# 객체 목록
response = s3.list_objects_v2(Bucket='my-bucket', Prefix='folder/')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Presigned URL 생성
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'file.txt'},
    ExpiresIn=3600
)
```

**GCP Cloud Storage:**
```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-bucket')

# 업로드
blob = bucket.blob('remote_file.txt')
blob.upload_from_filename('local_file.txt')

# 다운로드
blob = bucket.blob('remote_file.txt')
blob.download_to_filename('local_file.txt')

# 객체 목록
blobs = client.list_blobs('my-bucket', prefix='folder/')
for blob in blobs:
    print(blob.name)

# Signed URL 생성
from datetime import timedelta
url = blob.generate_signed_url(expiration=timedelta(hours=1))
```

---

## 11. 다음 단계

- [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) - 블록 스토리지
- [10_Load_Balancing_CDN.md](./10_Load_Balancing_CDN.md) - CDN과 함께 사용

---

## 참고 자료

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [GCP Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [S3 Pricing](https://aws.amazon.com/s3/pricing/)
- [Cloud Storage Pricing](https://cloud.google.com/storage/pricing)
