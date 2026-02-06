# 모니터링, 로깅 & 비용 관리

## 1. 모니터링 개요

### 1.1 모니터링이 필요한 이유

- 시스템 가용성 확보
- 성능 문제 조기 발견
- 용량 계획
- 비용 최적화
- 보안 이상 탐지

### 1.2 서비스 매핑

| 기능 | AWS | GCP |
|------|-----|-----|
| 메트릭 모니터링 | CloudWatch | Cloud Monitoring |
| 로그 수집 | CloudWatch Logs | Cloud Logging |
| 추적 | X-Ray | Cloud Trace |
| 대시보드 | CloudWatch Dashboards | Cloud Monitoring Dashboards |
| 알림 | CloudWatch Alarms + SNS | Alerting Policies |
| 비용 관리 | Cost Explorer, Budgets | Billing, Budgets |

---

## 2. AWS CloudWatch

### 2.1 메트릭

```bash
# EC2 메트릭 조회
aws cloudwatch list-metrics --namespace AWS/EC2

# 메트릭 데이터 조회
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-01T23:59:59Z \
    --period 300 \
    --statistics Average

# 커스텀 메트릭 발행
aws cloudwatch put-metric-data \
    --namespace MyApp \
    --metric-name RequestCount \
    --value 100 \
    --unit Count \
    --dimensions Environment=Production
```

**주요 메트릭:**

| 서비스 | 메트릭 | 설명 |
|--------|--------|------|
| EC2 | CPUUtilization | CPU 사용률 |
| EC2 | NetworkIn/Out | 네트워크 트래픽 |
| RDS | DatabaseConnections | DB 연결 수 |
| RDS | FreeStorageSpace | 남은 스토리지 |
| ALB | RequestCount | 요청 수 |
| ALB | TargetResponseTime | 응답 시간 |
| Lambda | Invocations | 호출 수 |
| Lambda | Duration | 실행 시간 |

### 2.2 알람

```bash
# CPU 알람 생성
aws cloudwatch put-metric-alarm \
    --alarm-name high-cpu \
    --alarm-description "CPU over 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:ap-northeast-2:123456789012:alerts

# 알람 목록
aws cloudwatch describe-alarms

# 알람 상태 확인
aws cloudwatch describe-alarm-history \
    --alarm-name high-cpu
```

### 2.3 대시보드

```bash
# 대시보드 생성
aws cloudwatch put-dashboard \
    --dashboard-name MyDashboard \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/EC2", "CPUUtilization", "InstanceId", "i-xxx"]
                    ],
                    "title": "EC2 CPU",
                    "period": 300
                }
            }
        ]
    }'
```

---

## 3. AWS CloudWatch Logs

### 3.1 로그 그룹 관리

```bash
# 로그 그룹 생성
aws logs create-log-group --log-group-name /myapp/production

# 보존 기간 설정
aws logs put-retention-policy \
    --log-group-name /myapp/production \
    --retention-in-days 30

# 로그 스트림 조회
aws logs describe-log-streams --log-group-name /myapp/production

# 로그 조회
aws logs filter-log-events \
    --log-group-name /myapp/production \
    --filter-pattern "ERROR" \
    --start-time 1704067200000 \
    --end-time 1704153600000
```

### 3.2 로그 인사이트

```bash
# 로그 쿼리 실행
aws logs start-query \
    --log-group-name /myapp/production \
    --start-time 1704067200 \
    --end-time 1704153600 \
    --query-string 'fields @timestamp, @message
        | filter @message like /ERROR/
        | sort @timestamp desc
        | limit 20'

# 쿼리 결과 조회
aws logs get-query-results --query-id QUERY_ID
```

### 3.3 EC2에서 로그 전송

```bash
# CloudWatch Agent 설치 (Amazon Linux)
sudo yum install -y amazon-cloudwatch-agent

# 설정 파일
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/myapp/*.log",
                        "log_group_name": "/myapp/production",
                        "log_stream_name": "{instance_id}"
                    }
                ]
            }
        }
    }
}
EOF

# Agent 시작
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s
```

---

## 4. GCP Cloud Monitoring

### 4.1 메트릭

```bash
# 메트릭 목록 조회
gcloud monitoring metrics list --filter="metric.type:compute.googleapis.com"

# 메트릭 데이터 조회 (gcloud에서는 제한적, API/콘솔 권장)
gcloud monitoring metrics read \
    "compute.googleapis.com/instance/cpu/utilization" \
    --project=PROJECT_ID
```

**주요 메트릭:**

| 서비스 | 메트릭 | 설명 |
|--------|--------|------|
| Compute | cpu/utilization | CPU 사용률 |
| Compute | network/received_bytes | 수신 트래픽 |
| Cloud SQL | database/disk/utilization | 디스크 사용률 |
| Cloud Run | request_count | 요청 수 |
| GKE | node/cpu/utilization | 노드 CPU |

### 4.2 알림 정책

```bash
# 알림 채널 생성 (이메일)
gcloud alpha monitoring channels create \
    --display-name="Email Alerts" \
    --type=email \
    --channel-labels=email_address=admin@example.com

# 알림 정책 생성
gcloud alpha monitoring policies create \
    --display-name="High CPU Alert" \
    --condition-display-name="CPU > 80%" \
    --condition-filter='metric.type="compute.googleapis.com/instance/cpu/utilization"' \
    --condition-threshold-value=0.8 \
    --condition-threshold-comparison=COMPARISON_GT \
    --condition-threshold-duration=300s \
    --notification-channels=projects/PROJECT/notificationChannels/CHANNEL_ID
```

---

## 5. GCP Cloud Logging

### 5.1 로그 조회

```bash
# 로그 조회
gcloud logging read 'resource.type="gce_instance"' \
    --limit=10 \
    --format=json

# 에러 로그만
gcloud logging read 'severity>=ERROR' \
    --limit=20

# 특정 시간대
gcloud logging read 'timestamp>="2024-01-01T00:00:00Z"' \
    --limit=100

# 로그 싱크 생성 (Cloud Storage로 내보내기)
gcloud logging sinks create my-sink \
    storage.googleapis.com/my-log-bucket \
    --log-filter='resource.type="gce_instance"'
```

### 5.2 로그 기반 메트릭

```bash
# 에러 수 메트릭 생성
gcloud logging metrics create error-count \
    --description="Count of errors" \
    --log-filter='severity>=ERROR'

# 메트릭 목록
gcloud logging metrics list
```

---

## 6. 비용 관리

### 6.1 AWS Cost Explorer

```bash
# 월별 비용 조회
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE

# 서비스별 비용
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --output table
```

### 6.2 AWS Budgets

```bash
# 월 예산 생성
aws budgets create-budget \
    --account-id 123456789012 \
    --budget '{
        "BudgetName": "Monthly-100USD",
        "BudgetLimit": {"Amount": "100", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80,
                "ThresholdType": "PERCENTAGE"
            },
            "Subscribers": [
                {"SubscriptionType": "EMAIL", "Address": "admin@example.com"}
            ]
        }
    ]'

# 예산 목록
aws budgets describe-budgets --account-id 123456789012
```

### 6.3 GCP Billing

```bash
# 빌링 계정 조회
gcloud billing accounts list

# 프로젝트 빌링 연결
gcloud billing projects link PROJECT_ID \
    --billing-account=BILLING_ACCOUNT_ID

# 예산 생성
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Monthly Budget" \
    --budget-amount=100USD \
    --threshold-rule=percent=0.8,basis=CURRENT_SPEND \
    --all-updates-rule-pubsub-topic=projects/PROJECT/topics/budget-alerts
```

---

## 7. 비용 최적화 전략

### 7.1 컴퓨팅 최적화

| 전략 | AWS | GCP |
|------|-----|-----|
| 예약 인스턴스 | Reserved Instances | Committed Use |
| 스팟/선점형 | Spot Instances | Spot/Preemptible VMs |
| 오토스케일링 | Auto Scaling | Managed Instance Groups |
| 적정 사이징 | AWS Compute Optimizer | Recommender |

```bash
# AWS 권장 사항 조회
aws compute-optimizer get-ec2-instance-recommendations

# GCP 권장 사항 조회
gcloud recommender recommendations list \
    --project=PROJECT_ID \
    --location=global \
    --recommender=google.compute.instance.MachineTypeRecommender
```

### 7.2 스토리지 최적화

```bash
# S3 스토리지 클래스 전환
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "Archive old data",
            "Status": "Enabled",
            "Transitions": [
                {"Days": 30, "StorageClass": "STANDARD_IA"},
                {"Days": 90, "StorageClass": "GLACIER"}
            ]
        }]
    }'

# GCP 수명 주기 정책
gsutil lifecycle set lifecycle.json gs://my-bucket
```

### 7.3 비용 절감 체크리스트

```
□ 미사용 리소스 정리
  - 중지된 인스턴스 (스토리지 비용 계속 발생)
  - 연결되지 않은 EBS/PD 볼륨
  - 오래된 스냅샷
  - 미사용 Elastic IP / 정적 IP

□ 적정 사이징
  - 인스턴스 사용률 분석
  - 오버프로비저닝 확인
  - Rightsizing 권장사항 적용

□ 예약 용량
  - 안정적 워크로드에 예약 인스턴스
  - 1년/3년 약정 검토

□ 스팟/선점형 활용
  - 배치 작업, 개발 환경
  - 중단 허용 워크로드

□ 스토리지 최적화
  - 수명 주기 정책 적용
  - 적절한 스토리지 클래스
  - 불필요한 데이터 정리

□ 네트워크 비용
  - 같은 AZ/리전 내 통신
  - CDN 활용
  - NAT Gateway 트래픽 최적화
```

---

## 8. 태그 기반 비용 추적

### 8.1 태그 전략

```hcl
# Terraform 예시
locals {
  common_tags = {
    Environment = "production"
    Project     = "myapp"
    CostCenter  = "engineering"
    Owner       = "team-a"
    ManagedBy   = "terraform"
  }
}

resource "aws_instance" "web" {
  # ...
  tags = local.common_tags
}
```

### 8.2 비용 할당 태그

```bash
# AWS 비용 할당 태그 활성화 (Billing Console에서)

# 태그별 비용 조회
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=TAG,Key=Project

# GCP 라벨별 비용 (BigQuery 내보내기 필요)
SELECT
  labels.key,
  labels.value,
  SUM(cost) as total_cost
FROM `billing_export.gcp_billing_export_v1_*`
CROSS JOIN UNNEST(labels) as labels
GROUP BY 1, 2
ORDER BY total_cost DESC
```

---

## 9. 대시보드 예시

### 9.1 운영 대시보드 구성

```
┌──────────────────────────────────────────────────────────────┐
│  운영 대시보드                                               │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   CPU 사용률    │  │  메모리 사용률  │  │  요청 수     │ │
│  │   [그래프]      │  │   [그래프]      │  │  [그래프]    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   응답 시간     │  │   에러율        │  │  활성 연결   │ │
│  │   [그래프]      │  │   [그래프]      │  │  [그래프]    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   최근 알람 / 인시던트                                  │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   비용 요약 (이번 달)                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. 알림 설정 권장사항

### 10.1 필수 알림

| 카테고리 | 조건 | 긴급도 |
|----------|------|--------|
| CPU | > 80% (5분) | 중 |
| CPU | > 95% (2분) | 높 |
| 메모리 | > 85% | 중 |
| 디스크 | > 80% | 중 |
| 디스크 | > 90% | 높 |
| 헬스체크 | 실패 | 높 |
| 에러율 | > 1% | 중 |
| 에러율 | > 5% | 높 |
| 응답 시간 | > 2초 | 중 |
| 비용 | > 80% 예산 | 중 |

### 10.2 알림 채널

```bash
# AWS SNS 토픽 생성
aws sns create-topic --name alerts

# 이메일 구독
aws sns subscribe \
    --topic-arn arn:aws:sns:...:alerts \
    --protocol email \
    --notification-endpoint admin@example.com

# Slack 웹훅 (Lambda 통해)
# PagerDuty, Opsgenie 등 연동
```

---

## 11. 다음 단계

- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC Flow Logs
- [14_Security_Services.md](./14_Security_Services.md) - 보안 모니터링

---

## 참고 자료

- [AWS CloudWatch Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [AWS Cost Management](https://docs.aws.amazon.com/cost-management/)
- [GCP Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [GCP Billing](https://cloud.google.com/billing/docs)
