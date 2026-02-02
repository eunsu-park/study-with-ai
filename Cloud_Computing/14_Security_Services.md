# 보안 서비스

## 1. 보안 개요

### 1.1 클라우드 보안 계층

```
┌─────────────────────────────────────────────────────────────┐
│  애플리케이션 보안                                          │
│  - 입력 검증, 인증/인가, 세션 관리                          │
├─────────────────────────────────────────────────────────────┤
│  데이터 보안                                                │
│  - 암호화 (저장 시, 전송 중), 키 관리                       │
├─────────────────────────────────────────────────────────────┤
│  네트워크 보안                                              │
│  - 방화벽, VPC, WAF, DDoS 보호                              │
├─────────────────────────────────────────────────────────────┤
│  인프라 보안                                                │
│  - 패치 관리, 취약점 스캐닝                                 │
├─────────────────────────────────────────────────────────────┤
│  ID/접근 관리                                               │
│  - IAM, MFA, 최소 권한                                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 서비스 매핑

| 기능 | AWS | GCP |
|------|-----|-----|
| 네트워크 방화벽 | Security Groups, NACL | Firewall Rules |
| WAF | AWS WAF | Cloud Armor |
| DDoS | AWS Shield | Cloud Armor |
| 키 관리 | KMS | Cloud KMS |
| 비밀 관리 | Secrets Manager | Secret Manager |
| 취약점 스캐닝 | Inspector | Security Command Center |
| 위협 탐지 | GuardDuty | Security Command Center |

---

## 2. 네트워크 보안

### 2.1 AWS Security Groups

```bash
# 보안 그룹 생성
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server SG" \
    --vpc-id vpc-12345678

# 인바운드 규칙 추가
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --ip-permissions '[
        {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "203.0.113.0/24", "Description": "Office IP"}]}
    ]'

# 다른 보안 그룹에서 오는 트래픽 허용
aws ec2 authorize-security-group-ingress \
    --group-id sg-db \
    --source-group sg-app \
    --protocol tcp \
    --port 3306

# 규칙 삭제
aws ec2 revoke-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

### 2.2 GCP Firewall Rules

```bash
# 방화벽 규칙 생성
gcloud compute firewall-rules create allow-http \
    --network=my-vpc \
    --allow=tcp:80,tcp:443 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=http-server \
    --priority=1000

# SSH 허용 (특정 IP)
gcloud compute firewall-rules create allow-ssh-office \
    --network=my-vpc \
    --allow=tcp:22 \
    --source-ranges=203.0.113.0/24 \
    --target-tags=ssh-server

# 내부 통신 허용
gcloud compute firewall-rules create allow-internal \
    --network=my-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/8

# 거부 규칙 (낮은 우선순위)
gcloud compute firewall-rules create deny-all-ingress \
    --network=my-vpc \
    --action=DENY \
    --rules=all \
    --source-ranges=0.0.0.0/0 \
    --priority=65534

# 규칙 삭제
gcloud compute firewall-rules delete allow-http
```

---

## 3. WAF (Web Application Firewall)

### 3.1 AWS WAF

```bash
# 웹 ACL 생성
aws wafv2 create-web-acl \
    --name my-web-acl \
    --scope REGIONAL \
    --default-action Allow={} \
    --visibility-config SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=my-web-acl \
    --rules '[
        {
            "Name": "AWSManagedRulesCommonRuleSet",
            "Priority": 1,
            "OverrideAction": {"None": {}},
            "Statement": {
                "ManagedRuleGroupStatement": {
                    "VendorName": "AWS",
                    "Name": "AWSManagedRulesCommonRuleSet"
                }
            },
            "VisibilityConfig": {
                "SampledRequestsEnabled": true,
                "CloudWatchMetricsEnabled": true,
                "MetricName": "CommonRules"
            }
        }
    ]'

# ALB에 연결
aws wafv2 associate-web-acl \
    --web-acl-arn arn:aws:wafv2:...:webacl/my-web-acl/xxx \
    --resource-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx
```

**일반 규칙:**
- AWSManagedRulesCommonRuleSet: OWASP Top 10
- AWSManagedRulesSQLiRuleSet: SQL 인젝션
- AWSManagedRulesKnownBadInputsRuleSet: 악성 입력
- AWSManagedRulesAmazonIpReputationList: IP 평판

### 3.2 GCP Cloud Armor

```bash
# 보안 정책 생성
gcloud compute security-policies create my-policy \
    --description="My security policy"

# 규칙 추가 (SQL 인젝션 차단)
gcloud compute security-policies rules create 1000 \
    --security-policy=my-policy \
    --expression="evaluatePreconfiguredWaf('sqli-v33-stable')" \
    --action=deny-403

# 규칙 추가 (XSS 차단)
gcloud compute security-policies rules create 2000 \
    --security-policy=my-policy \
    --expression="evaluatePreconfiguredWaf('xss-v33-stable')" \
    --action=deny-403

# 규칙 추가 (IP 차단)
gcloud compute security-policies rules create 3000 \
    --security-policy=my-policy \
    --src-ip-ranges="203.0.113.0/24" \
    --action=deny-403

# 속도 제한
gcloud compute security-policies rules create 4000 \
    --security-policy=my-policy \
    --expression="true" \
    --action=rate-based-ban \
    --rate-limit-threshold-count=1000 \
    --rate-limit-threshold-interval-sec=60

# 백엔드 서비스에 연결
gcloud compute backend-services update my-backend \
    --security-policy=my-policy \
    --global
```

---

## 4. 키 관리 (KMS)

### 4.1 AWS KMS

```bash
# 고객 관리 키 생성
aws kms create-key \
    --description "My encryption key" \
    --key-usage ENCRYPT_DECRYPT \
    --origin AWS_KMS

# 별칭 생성
aws kms create-alias \
    --alias-name alias/my-key \
    --target-key-id 12345678-1234-1234-1234-123456789012

# 데이터 암호화
aws kms encrypt \
    --key-id alias/my-key \
    --plaintext fileb://plaintext.txt \
    --output text \
    --query CiphertextBlob | base64 --decode > encrypted.bin

# 데이터 복호화
aws kms decrypt \
    --ciphertext-blob fileb://encrypted.bin \
    --output text \
    --query Plaintext | base64 --decode > decrypted.txt

# 키 정책 업데이트
aws kms put-key-policy \
    --key-id 12345678-1234-1234-1234-123456789012 \
    --policy-name default \
    --policy file://key-policy.json
```

**S3 서버 측 암호화:**
```bash
# 버킷 암호화 설정
aws s3api put-bucket-encryption \
    --bucket my-bucket \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "aws:kms",
                "KMSMasterKeyID": "alias/my-key"
            }
        }]
    }'
```

### 4.2 GCP Cloud KMS

```bash
# 키 링 생성
gcloud kms keyrings create my-keyring \
    --location=asia-northeast3

# 암호화 키 생성
gcloud kms keys create my-key \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --purpose=encryption

# 데이터 암호화
gcloud kms encrypt \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --key=my-key \
    --plaintext-file=plaintext.txt \
    --ciphertext-file=encrypted.bin

# 데이터 복호화
gcloud kms decrypt \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --key=my-key \
    --ciphertext-file=encrypted.bin \
    --plaintext-file=decrypted.txt

# 서비스 계정에 암호화 권한 부여
gcloud kms keys add-iam-policy-binding my-key \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --member="serviceAccount:my-sa@PROJECT.iam.gserviceaccount.com" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

**Cloud Storage CMEK:**
```bash
# 버킷 생성 시 CMEK 지정
gsutil mb -l asia-northeast3 \
    -k projects/PROJECT/locations/asia-northeast3/keyRings/my-keyring/cryptoKeys/my-key \
    gs://my-encrypted-bucket
```

---

## 5. 비밀 관리

### 5.1 AWS Secrets Manager

```bash
# 비밀 생성
aws secretsmanager create-secret \
    --name my-database-credentials \
    --secret-string '{"username":"admin","password":"MySecretPassword123!"}'

# 비밀 조회
aws secretsmanager get-secret-value \
    --secret-id my-database-credentials \
    --query SecretString \
    --output text

# 비밀 업데이트
aws secretsmanager update-secret \
    --secret-id my-database-credentials \
    --secret-string '{"username":"admin","password":"NewPassword456!"}'

# 자동 로테이션 설정
aws secretsmanager rotate-secret \
    --secret-id my-database-credentials \
    --rotation-lambda-arn arn:aws:lambda:...:function:RotateSecret \
    --rotation-rules AutomaticallyAfterDays=30
```

**애플리케이션에서 사용:**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

credentials = get_secret('my-database-credentials')
db_password = credentials['password']
```

### 5.2 GCP Secret Manager

```bash
# 비밀 생성
echo -n "MySecretPassword123!" | gcloud secrets create my-secret \
    --data-file=-

# 또는 파일에서
gcloud secrets create my-secret --data-file=secret.txt

# 비밀 조회
gcloud secrets versions access latest --secret=my-secret

# 새 버전 추가
echo -n "NewPassword456!" | gcloud secrets versions add my-secret \
    --data-file=-

# 서비스 계정에 접근 권한 부여
gcloud secrets add-iam-policy-binding my-secret \
    --member="serviceAccount:my-sa@PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**애플리케이션에서 사용:**
```python
from google.cloud import secretmanager

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/PROJECT_ID/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

password = get_secret('my-secret')
```

---

## 6. 암호화

### 6.1 저장 시 암호화 (Encryption at Rest)

| 서비스 | AWS | GCP |
|--------|-----|-----|
| 객체 스토리지 | S3 SSE-S3, SSE-KMS | Cloud Storage CMEK |
| 블록 스토리지 | EBS 암호화 | PD 암호화 |
| 데이터베이스 | RDS 암호화 | Cloud SQL 암호화 |
| 기본 암호화 | 일부 서비스 기본 | 모든 서비스 기본 |

### 6.2 전송 중 암호화 (Encryption in Transit)

```bash
# AWS ALB HTTPS 강제
aws elbv2 modify-listener \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --protocol HTTPS \
    --certificates CertificateArn=arn:aws:acm:...:certificate/xxx

# GCP HTTPS 로드밸런서
gcloud compute target-https-proxies create my-https-proxy \
    --url-map=my-url-map \
    --ssl-certificates=my-cert

# RDS SSL 강제
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --ca-certificate-identifier rds-ca-2019

# Cloud SQL SSL 강제
gcloud sql instances patch my-database --require-ssl
```

---

## 7. 취약점 탐지

### 7.1 AWS Inspector

```bash
# Inspector v2 활성화 (계정 수준)
aws inspector2 enable \
    --resource-types EC2 ECR

# 스캔 결과 조회
aws inspector2 list-findings \
    --filter-criteria '{
        "findingStatus": [{"comparison": "EQUALS", "value": "ACTIVE"}],
        "severity": [{"comparison": "EQUALS", "value": "HIGH"}]
    }'
```

### 7.2 GCP Security Command Center

```bash
# 조직 수준 활성화 필요 (Console에서)

# 발견 항목 조회
gcloud scc findings list ORGANIZATION_ID \
    --source=SOURCE_ID \
    --filter="state=\"ACTIVE\""
```

---

## 8. 위협 탐지

### 8.1 AWS GuardDuty

```bash
# GuardDuty 활성화
aws guardduty create-detector --enable

# 결과 조회
aws guardduty list-findings --detector-id DETECTOR_ID

aws guardduty get-findings \
    --detector-id DETECTOR_ID \
    --finding-ids FINDING_ID

# 신뢰할 수 있는 IP 목록 추가
aws guardduty create-ip-set \
    --detector-id DETECTOR_ID \
    --name "Trusted IPs" \
    --format TXT \
    --location s3://my-bucket/trusted-ips.txt \
    --activate
```

### 8.2 GCP Security Command Center

```bash
# 위협 탐지 (Premium 필요)
# Event Threat Detection
# Container Threat Detection
# Virtual Machine Threat Detection

# 조직 정책 위반 확인
gcloud scc findings list ORGANIZATION_ID \
    --source=SECURITY_HEALTH_ANALYTICS \
    --filter="category=\"PUBLIC_BUCKET_ACL\""
```

---

## 9. 감사 로깅

### 9.1 AWS CloudTrail

```bash
# 트레일 생성
aws cloudtrail create-trail \
    --name my-trail \
    --s3-bucket-name my-log-bucket \
    --is-multi-region-trail \
    --enable-log-file-validation

# 로깅 시작
aws cloudtrail start-logging --name my-trail

# 이벤트 조회
aws cloudtrail lookup-events \
    --lookup-attributes AttributeKey=EventName,AttributeValue=ConsoleLogin \
    --start-time 2024-01-01T00:00:00Z
```

### 9.2 GCP Cloud Audit Logs

```bash
# 감사 로그는 기본 활성화

# 로그 조회
gcloud logging read 'logName:"cloudaudit.googleapis.com"' \
    --project=PROJECT_ID \
    --limit=10

# Data Access 로그 활성화 (추가 설정 필요)
gcloud projects get-iam-policy PROJECT_ID --format=json > policy.json
# 수정 후
gcloud projects set-iam-policy PROJECT_ID policy.json
```

---

## 10. 보안 체크리스트

### 10.1 계정/IAM
```
□ Root/Owner MFA 활성화
□ 최소 권한 원칙 적용
□ 정기적인 권한 검토
□ 미사용 자격 증명 비활성화
□ 강력한 비밀번호 정책
```

### 10.2 네트워크
```
□ 기본 보안 그룹 규칙 제거
□ 필요한 포트만 개방
□ 프라이빗 서브넷 활용
□ VPC Flow Logs 활성화
□ WAF 적용 (웹 앱)
```

### 10.3 데이터
```
□ 저장 시 암호화 활성화
□ 전송 중 암호화 (HTTPS/TLS)
□ 퍼블릭 액세스 차단
□ 백업 암호화
□ 키 로테이션
```

### 10.4 모니터링
```
□ CloudTrail/Audit Logs 활성화
□ GuardDuty/SCC 활성화
□ 보안 알림 설정
□ 정기적인 취약점 스캔
□ 인시던트 대응 계획
```

---

## 11. 다음 단계

- [15_CLI_and_SDK.md](./15_CLI_and_SDK.md) - CLI/SDK 자동화
- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM 상세

---

## 참고 자료

- [AWS Security Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices)
- [AWS WAF](https://docs.aws.amazon.com/waf/)
- [GCP Cloud Armor](https://cloud.google.com/armor/docs)
