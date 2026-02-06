# IAM (Identity and Access Management)

## 1. IAM 개요

### 1.1 IAM이란?

IAM은 클라우드 리소스에 대한 접근을 안전하게 제어하는 서비스입니다.

**핵심 질문:**
- **누가 (Who)**: 사용자, 그룹, 역할
- **무엇을 (What)**: 리소스
- **어떻게 (How)**: 권한 (허용/거부)

### 1.2 AWS vs GCP IAM 비교

| 항목 | AWS IAM | GCP IAM |
|------|---------|---------|
| 범위 | 계정 수준 | 조직/프로젝트 수준 |
| 정책 부착 | 사용자/그룹/역할에 | 리소스에 |
| 역할 | 역할을 맡음 (AssumeRole) | 역할 바인딩 |
| 서비스 계정 | 역할 + 인스턴스 프로파일 | 서비스 계정 |

---

## 2. AWS IAM

### 2.1 핵심 개념

```
┌─────────────────────────────────────────────────────────────┐
│  AWS 계정                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  IAM                                                    ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    사용자     │  │     그룹      │                   ││
│  │  │  (Users)      │  │  (Groups)     │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  │         ↓                  ↓                            ││
│  │  ┌─────────────────────────────────────────────┐        ││
│  │  │              정책 (Policies)                │        ││
│  │  │  { "Effect": "Allow",                       │        ││
│  │  │    "Action": "s3:*",                        │        ││
│  │  │    "Resource": "*" }                        │        ││
│  │  └─────────────────────────────────────────────┘        ││
│  │                     ↓                                   ││
│  │  ┌───────────────────────────────────────────────────┐  ││
│  │  │              역할 (Roles)                         │  ││
│  │  │  - EC2 인스턴스 역할                              │  ││
│  │  │  - Lambda 실행 역할                               │  ││
│  │  │  - 교차 계정 역할                                 │  ││
│  │  └───────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 사용자 및 그룹

```bash
# 사용자 생성
aws iam create-user --user-name john

# 로그인 비밀번호 설정
aws iam create-login-profile \
    --user-name john \
    --password 'TempPassword123!' \
    --password-reset-required

# 액세스 키 생성 (프로그래밍 접근)
aws iam create-access-key --user-name john

# 그룹 생성
aws iam create-group --group-name Developers

# 그룹에 사용자 추가
aws iam add-user-to-group --group-name Developers --user-name john

# 그룹 멤버 확인
aws iam get-group --group-name Developers
```

### 2.3 정책 (Policies)

**정책 구조:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3Read",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ],
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                }
            }
        }
    ]
}
```

```bash
# 관리형 정책 연결
aws iam attach-user-policy \
    --user-name john \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# 커스텀 정책 생성
aws iam create-policy \
    --policy-name MyS3Policy \
    --policy-document file://policy.json

# 그룹에 정책 연결
aws iam attach-group-policy \
    --group-name Developers \
    --policy-arn arn:aws:iam::123456789012:policy/MyS3Policy

# 인라인 정책 추가
aws iam put-user-policy \
    --user-name john \
    --policy-name InlinePolicy \
    --policy-document file://inline-policy.json
```

### 2.4 역할 (Roles)

**EC2 인스턴스 역할:**
```bash
# 신뢰 정책 (누가 역할을 맡을 수 있는지)
cat > trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# 역할 생성
aws iam create-role \
    --role-name EC2-S3-Access \
    --assume-role-policy-document file://trust-policy.json

# 정책 연결
aws iam attach-role-policy \
    --role-name EC2-S3-Access \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# 인스턴스 프로파일 생성 및 역할 추가
aws iam create-instance-profile --instance-profile-name EC2-S3-Profile
aws iam add-role-to-instance-profile \
    --instance-profile-name EC2-S3-Profile \
    --role-name EC2-S3-Access

# EC2에 인스턴스 프로파일 연결
aws ec2 associate-iam-instance-profile \
    --instance-id i-1234567890abcdef0 \
    --iam-instance-profile Name=EC2-S3-Profile
```

**교차 계정 역할:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::OTHER_ACCOUNT_ID:root"},
            "Action": "sts:AssumeRole"
        }
    ]
}
```

```bash
# 다른 계정에서 역할 맡기
aws sts assume-role \
    --role-arn arn:aws:iam::TARGET_ACCOUNT:role/CrossAccountRole \
    --role-session-name MySession
```

---

## 3. GCP IAM

### 3.1 핵심 개념

```
┌─────────────────────────────────────────────────────────────┐
│  조직 (Organization)                                        │
│  ├── 폴더 (Folder)                                          │
│  │   └── 프로젝트 (Project)                                 │
│  │       └── 리소스 (Resource)                              │
│  └─────────────────────────────────────────────────────────│
│                                                             │
│  IAM 바인딩:                                                │
│  주체 (Member) + 역할 (Role) = 리소스에 대한 권한           │
│                                                             │
│  예: user:john@example.com + roles/storage.admin            │
│      → gs://my-bucket에 대한 관리자 권한                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 역할 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **기본 역할** | 넓은 권한 | Owner, Editor, Viewer |
| **사전정의 역할** | 서비스별 세분화 | roles/storage.admin |
| **커스텀 역할** | 사용자 정의 | my-custom-role |

### 3.3 역할 바인딩

```bash
# 프로젝트 수준 역할 부여
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"

# 버킷 수준 역할 부여
gsutil iam ch user:john@example.com:objectViewer gs://my-bucket

# 역할 바인딩 조회
gcloud projects get-iam-policy PROJECT_ID

# 역할 제거
gcloud projects remove-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"
```

### 3.4 서비스 계정

```bash
# 서비스 계정 생성
gcloud iam service-accounts create my-service-account \
    --display-name="My Service Account"

# 역할 부여
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:my-service-account@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# 키 파일 생성 (프로그래밍 접근)
gcloud iam service-accounts keys create key.json \
    --iam-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com

# Compute Engine에 서비스 계정 연결
gcloud compute instances create my-instance \
    --service-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com \
    --scopes=cloud-platform
```

### 3.5 워크로드 아이덴티티 (GKE)

```bash
# 워크로드 아이덴티티 풀 활성화
gcloud container clusters update my-cluster \
    --region=asia-northeast3 \
    --workload-pool=PROJECT_ID.svc.id.goog

# Kubernetes 서비스 계정과 GCP 서비스 계정 연결
gcloud iam service-accounts add-iam-policy-binding \
    my-gcp-sa@PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/iam.workloadIdentityUser \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/K8S_SA]"
```

---

## 4. 최소 권한 원칙

### 4.1 원칙

```
최소 권한 = 작업 수행에 필요한 최소한의 권한만 부여

잘못된 예:
- Admin 권한을 모든 사용자에게
- * (모든 리소스)에 대한 권한

올바른 예:
- 필요한 Action만 명시
- 특정 리소스에 대한 권한
- 조건부 접근
```

### 4.2 AWS 정책 예시

**나쁜 예:**
```json
{
    "Effect": "Allow",
    "Action": "*",
    "Resource": "*"
}
```

**좋은 예:**
```json
{
    "Effect": "Allow",
    "Action": [
        "s3:GetObject",
        "s3:PutObject"
    ],
    "Resource": "arn:aws:s3:::my-bucket/uploads/*",
    "Condition": {
        "StringEquals": {
            "s3:x-amz-acl": "private"
        }
    }
}
```

### 4.3 GCP 역할 선택

```bash
# 권한이 너무 넓은 역할 (피할 것)
roles/owner
roles/editor

# 적절한 역할
roles/storage.objectViewer  # 객체 읽기만
roles/compute.instanceAdmin.v1  # 인스턴스 관리만
roles/cloudsql.client  # SQL 연결만
```

---

## 5. 조건부 접근

### 5.1 AWS 조건

```json
{
    "Effect": "Allow",
    "Action": "s3:*",
    "Resource": "*",
    "Condition": {
        "IpAddress": {
            "aws:SourceIp": "203.0.113.0/24"
        },
        "Bool": {
            "aws:MultiFactorAuthPresent": "true"
        },
        "DateGreaterThan": {
            "aws:CurrentTime": "2024-01-01T00:00:00Z"
        },
        "StringEquals": {
            "aws:RequestedRegion": "ap-northeast-2"
        }
    }
}
```

### 5.2 GCP 조건

```bash
# 조건부 역할 바인딩
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin" \
    --condition='expression=request.time < timestamp("2024-12-31T23:59:59Z"),title=Temporary Access,description=Access until end of year'

# IP 기반 조건 (VPC Service Controls와 함께)
expression: 'resource.name.startsWith("projects/PROJECT_ID/zones/asia-northeast3")'
```

---

## 6. 권한 분석

### 6.1 AWS IAM Access Analyzer

```bash
# Access Analyzer 생성
aws accessanalyzer create-analyzer \
    --analyzer-name my-analyzer \
    --type ACCOUNT

# 분석 결과 조회
aws accessanalyzer list-findings --analyzer-arn arn:aws:access-analyzer:...:analyzer/my-analyzer

# 정책 검증
aws accessanalyzer validate-policy \
    --policy-document file://policy.json \
    --policy-type IDENTITY_POLICY
```

### 6.2 GCP Policy Analyzer

```bash
# IAM 정책 분석
gcloud asset analyze-iam-policy \
    --organization=ORG_ID \
    --identity="user:john@example.com"

# 권한 확인
gcloud projects get-iam-policy PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:john@example.com" \
    --format="table(bindings.role)"
```

---

## 7. MFA (다중 인증)

### 7.1 AWS MFA

```bash
# 가상 MFA 활성화
aws iam create-virtual-mfa-device \
    --virtual-mfa-device-name john-mfa \
    --outfile qrcode.png \
    --bootstrap-method QRCodePNG

# MFA 디바이스 연결
aws iam enable-mfa-device \
    --user-name john \
    --serial-number arn:aws:iam::123456789012:mfa/john-mfa \
    --authentication-code1 123456 \
    --authentication-code2 789012

# MFA 필수 정책
{
    "Effect": "Deny",
    "Action": "*",
    "Resource": "*",
    "Condition": {
        "BoolIfExists": {
            "aws:MultiFactorAuthPresent": "false"
        }
    }
}
```

### 7.2 GCP 2단계 인증

```bash
# 조직 수준에서 2FA 강제 (Admin Console에서)
# Google Workspace Admin → Security → 2-Step Verification

# 서비스 계정은 MFA 불가 → 대신:
# - 키 파일 안전 관리
# - 워크로드 아이덴티티 사용
# - 단기 토큰 사용
```

---

## 8. 일반적인 역할 패턴

### 8.1 AWS 일반 역할

| 역할 | 권한 | 용도 |
|------|------|------|
| AdministratorAccess | 전체 | 관리자 |
| PowerUserAccess | IAM 제외 전체 | 개발자 |
| ReadOnlyAccess | 읽기 전용 | 감사/뷰어 |
| AmazonS3FullAccess | S3 전체 | 스토리지 관리 |
| AmazonEC2FullAccess | EC2 전체 | 컴퓨팅 관리 |

### 8.2 GCP 일반 역할

| 역할 | 권한 | 용도 |
|------|------|------|
| roles/owner | 전체 | 관리자 |
| roles/editor | IAM 제외 편집 | 개발자 |
| roles/viewer | 읽기 전용 | 뷰어 |
| roles/compute.admin | Compute 전체 | 인프라 관리 |
| roles/storage.admin | Storage 전체 | 스토리지 관리 |

---

## 9. 보안 모범 사례

```
□ Root/Owner 계정은 일상 업무에 사용하지 않음
□ Root/Owner 계정에 MFA 활성화
□ 최소 권한 원칙 적용
□ 그룹/역할을 통한 권한 관리 (개별 사용자 X)
□ 정기적인 권한 검토 (미사용 권한 제거)
□ 서비스 계정 키 파일 안전 관리
□ 임시 자격 증명 사용 (STS, 워크로드 아이덴티티)
□ 조건부 접근 활용 (IP, 시간, MFA)
□ 감사 로그 활성화 (CloudTrail, Cloud Audit Logs)
□ 정책 변경 알림 설정
```

---

## 10. 다음 단계

- [14_Security_Services.md](./14_Security_Services.md) - 보안 서비스
- [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) - 계정 초기 설정

---

## 참고 자료

- [AWS IAM Documentation](https://docs.aws.amazon.com/iam/)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [GCP IAM Documentation](https://cloud.google.com/iam/docs)
- [GCP IAM Best Practices](https://cloud.google.com/iam/docs/using-iam-securely)
