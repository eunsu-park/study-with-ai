# AWS & GCP 계정 설정

## 1. AWS 계정 생성

### 1.1 계정 생성 절차

1. **AWS 가입 페이지 접속**
   - https://aws.amazon.com/ 에서 "Create an AWS Account" 클릭

2. **계정 정보 입력**
   - 이메일 주소 (Root 계정용)
   - AWS 계정 이름
   - 비밀번호

3. **연락처 정보**
   - 계정 유형: 개인(Personal) 또는 비즈니스(Business)
   - 이름, 주소, 전화번호

4. **결제 정보**
   - 신용카드 등록 (무료 티어 사용 시에도 필수)
   - $1 인증 결제 후 환불됨

5. **본인 확인**
   - SMS 또는 음성 통화로 PIN 확인

6. **지원 플랜 선택**
   - Basic Support (무료) 선택 권장

### 1.2 Root 계정 보안

Root 계정은 모든 권한을 가지므로 반드시 보안 강화가 필요합니다.

```
⚠️ Root 계정 보안 체크리스트
□ 강력한 비밀번호 설정 (16자 이상, 특수문자 포함)
□ MFA(다중 인증) 활성화
□ 액세스 키 생성 금지
□ 일상 업무에 Root 계정 사용 금지
□ IAM 사용자 생성하여 사용
```

### 1.3 MFA 설정 (AWS)

**Console에서 MFA 활성화:**

1. AWS Console → 우측 상단 계정명 → "Security credentials"
2. "Multi-factor authentication (MFA)" 섹션
3. "Activate MFA" 클릭
4. MFA 디바이스 유형 선택:
   - **Virtual MFA device**: Google Authenticator, Authy 앱 사용
   - **Hardware TOTP token**: 물리적 토큰
   - **Security key**: FIDO 보안 키

**Virtual MFA 설정:**
```
1. 앱 설치: Google Authenticator 또는 Authy
2. 앱에서 QR 코드 스캔
3. 연속된 두 개의 MFA 코드 입력
4. "Assign MFA" 클릭
```

---

## 2. GCP 계정 생성

### 2.1 계정 생성 절차

1. **GCP Console 접속**
   - https://console.cloud.google.com/

2. **Google 계정 로그인**
   - 기존 Google 계정 사용 또는 새로 생성

3. **GCP 이용약관 동의**
   - 국가 선택
   - 서비스 약관 동의

4. **결제 계정 설정** (무료 체험용)
   - 신용카드 정보 입력
   - $300 무료 크레딧 활성화 (90일)

5. **첫 번째 프로젝트 생성**
   - 프로젝트명 지정
   - 조직 선택 (개인은 "조직 없음")

### 2.2 GCP 보안 설정

**Google 계정 보안 강화:**

```
⚠️ GCP 계정 보안 체크리스트
□ Google 계정에 2단계 인증 활성화
□ 비밀번호 보안 강화
□ 복구 이메일/전화번호 설정
□ 조직 정책 검토 (비즈니스)
□ 서비스 계정 사용 권장
```

### 2.3 2단계 인증 설정 (GCP)

1. Google 계정 설정 → "보안"
2. "2단계 인증" 활성화
3. 인증 방법 선택:
   - Google 메시지
   - 인증 앱 (Google Authenticator)
   - 보안 키
   - 백업 코드

---

## 3. 콘솔 탐색

### 3.1 AWS Management Console

**주요 UI 구성:**

```
┌─────────────────────────────────────────────────────────────┐
│  [AWS 로고]  서비스 검색창              리전 ▼  계정 ▼     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [서비스 메뉴]                                              │
│   ├── Compute (EC2, Lambda, ECS...)                         │
│   ├── Storage (S3, EBS, EFS...)                             │
│   ├── Database (RDS, DynamoDB...)                           │
│   ├── Networking (VPC, Route 53...)                         │
│   ├── Security (IAM, KMS...)                                │
│   └── Management (CloudWatch, CloudFormation...)            │
│                                                             │
│  [최근 방문 서비스]                                         │
│  [즐겨찾기 서비스]                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**유용한 기능:**
- **서비스 검색**: 상단 검색창에 서비스명 입력
- **리전 선택**: 우측 상단에서 작업할 리전 선택
- **CloudShell**: 브라우저 내 터미널 (AWS CLI 사전 설치)
- **Resource Groups**: 리소스 그룹화 및 태그 관리

### 3.2 GCP Console

**주요 UI 구성:**

```
┌─────────────────────────────────────────────────────────────┐
│  [GCP 로고]  [프로젝트 선택 ▼]  검색창         [계정 아이콘]│
├──────────────┬──────────────────────────────────────────────┤
│  [네비게이션]│                                              │
│   │          │  [대시보드]                                   │
│   ├─ 컴퓨팅   │   ├── 프로젝트 정보                          │
│   ├─ 스토리지 │   ├── 리소스 요약                            │
│   ├─ 네트워킹 │   ├── API 활동                               │
│   ├─ 데이터베이스│ └── 빌링 요약                             │
│   ├─ 보안     │                                              │
│   ├─ 도구     │                                              │
│   └─ 빌링     │                                              │
│              │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

**유용한 기능:**
- **프로젝트 선택**: 좌측 상단에서 프로젝트 전환
- **Cloud Shell**: 우측 상단 터미널 아이콘 (gcloud 사전 설치)
- **핀 고정**: 자주 사용하는 서비스를 메뉴에 고정
- **API 및 서비스**: API 활성화 관리

---

## 4. 첫 번째 프로젝트/리소스 그룹

### 4.1 AWS: 태그를 통한 리소스 관리

AWS는 프로젝트 개념 대신 **태그**로 리소스를 관리합니다.

```bash
# 리소스 생성 시 태그 지정 예시
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t2.micro \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=MyApp},{Key=Environment,Value=dev}]'
```

**태그 모범 사례:**

| 태그 키 | 예시 값 | 용도 |
|--------|--------|------|
| Project | MyApp | 프로젝트별 비용 추적 |
| Environment | dev, staging, prod | 환경 구분 |
| Owner | john@example.com | 담당자 식별 |
| CostCenter | IT-001 | 비용 센터 할당 |

### 4.2 GCP: 프로젝트 생성

GCP는 **프로젝트** 단위로 리소스와 빌링을 격리합니다.

**프로젝트 생성:**
1. Console 상단 → 프로젝트 선택 드롭다운
2. "새 프로젝트" 클릭
3. 프로젝트 이름 입력 (고유한 ID 자동 생성)
4. 빌링 계정 연결
5. "만들기" 클릭

```bash
# gcloud로 프로젝트 생성
gcloud projects create my-project-id \
    --name="My Project" \
    --labels=env=dev

# 프로젝트 전환
gcloud config set project my-project-id
```

**프로젝트 구조 권장:**

```
Organization (선택)
├── Folder: Development
│   ├── Project: dev-frontend
│   └── Project: dev-backend
├── Folder: Production
│   ├── Project: prod-frontend
│   └── Project: prod-backend
└── Folder: Shared
    └── Project: shared-services
```

---

## 5. 비용 알림 설정

### 5.1 AWS 예산 알림

**AWS Budgets 설정:**

1. AWS Console → "Billing and Cost Management" → "Budgets"
2. "Create budget" 클릭
3. 예산 유형 선택: "Cost budget"
4. 예산 설정:
   - 이름: "Monthly Budget"
   - 금액: 원하는 한도 (예: $50)
   - 기간: Monthly

5. 알림 조건:
   - 실제 비용이 예산의 80% 도달 시 알림
   - 예측 비용이 100% 초과 시 알림

6. 알림 수신:
   - 이메일 주소 입력
   - SNS 토픽 연결 (선택)

```bash
# AWS CLI로 예산 생성
aws budgets create-budget \
    --account-id 123456789012 \
    --budget '{
        "BudgetName": "Monthly-50USD",
        "BudgetLimit": {"Amount": "50", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[{
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80
        },
        "Subscribers": [{
            "SubscriptionType": "EMAIL",
            "Address": "your@email.com"
        }]
    }]'
```

### 5.2 GCP 예산 알림

**GCP Billing 예산 설정:**

1. Console → "결제" → "예산 및 알림"
2. "예산 만들기" 클릭
3. 예산 설정:
   - 이름: "Monthly Budget"
   - 프로젝트: 전체 또는 특정 프로젝트
   - 금액: 지정 금액 (예: $50)

4. 알림 임계값:
   - 50%, 90%, 100% 알림 설정

5. 알림 채널:
   - 이메일 수신자
   - Cloud Monitoring (선택)
   - Pub/Sub 토픽 (자동화용)

```bash
# gcloud로 예산 생성
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Monthly Budget" \
    --budget-amount=50USD \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0
```

---

## 6. 무료 티어 활용

### 6.1 AWS 무료 티어

| 유형 | 서비스 | 무료 한도 |
|------|--------|----------|
| **12개월 무료** | EC2 | t2.micro 750시간/월 |
| | S3 | 5GB 스토리지 |
| | RDS | db.t2.micro 750시간/월 |
| | CloudFront | 50GB 데이터 전송 |
| **항상 무료** | Lambda | 100만 요청/월 |
| | DynamoDB | 25GB 스토리지, 25 WCU/RCU |
| | SNS | 100만 요청/월 |
| | CloudWatch | 기본 모니터링 |

**무료 티어 모니터링:**
- Console → "Billing" → "Free Tier" 탭에서 사용량 확인

### 6.2 GCP 무료 티어

| 유형 | 서비스 | 무료 한도 |
|------|--------|----------|
| **$300 크레딧** | 모든 서비스 | 90일간 (신규 계정) |
| **Always Free** | Compute Engine | e2-micro 1개 (특정 리전) |
| | Cloud Storage | 5GB (US 리전) |
| | Cloud Functions | 200만 호출/월 |
| | BigQuery | 1TB 쿼리/월, 10GB 스토리지 |
| | Cloud Run | 200만 요청/월 |
| | Firestore | 1GB 스토리지, 50K 읽기/일 |

**Always Free 리전 제한:**
- Compute Engine e2-micro: us-west1, us-central1, us-east1만 해당

---

## 7. 초기 보안 설정 요약

### 7.1 AWS 초기 보안 체크리스트

```
□ Root 계정 MFA 활성화
□ Root 액세스 키 삭제 확인
□ IAM 사용자 생성 및 MFA 활성화
□ IAM 비밀번호 정책 강화
□ CloudTrail 활성화 (감사 로그)
□ 예산 알림 설정
□ S3 퍼블릭 액세스 차단 설정 확인
```

### 7.2 GCP 초기 보안 체크리스트

```
□ Google 계정 2단계 인증 활성화
□ 조직 정책 검토 (해당 시)
□ 서비스 계정 생성 (애플리케이션용)
□ 최소 권한 IAM 역할 부여
□ Cloud Audit Logs 활성화
□ 예산 알림 설정
□ VPC 방화벽 규칙 검토
```

---

## 8. 다음 단계

- [03_Regions_Availability_Zones.md](./03_Regions_Availability_Zones.md) - 리전과 가용 영역 이해
- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM 상세 설정

---

## 참고 자료

- [AWS 계정 생성 가이드](https://docs.aws.amazon.com/accounts/latest/reference/manage-acct-creating.html)
- [GCP 시작하기](https://cloud.google.com/docs/get-started)
- [AWS 무료 티어](https://aws.amazon.com/free/)
- [GCP 무료 티어](https://cloud.google.com/free)
