# 로드밸런싱 & CDN

## 1. 로드밸런싱 개요

### 1.1 로드밸런서란?

로드밸런서는 들어오는 트래픽을 여러 서버에 분산시키는 서비스입니다.

**장점:**
- 고가용성 (장애 서버 자동 제외)
- 확장성 (서버 추가/제거 용이)
- 성능 향상 (부하 분산)
- 보안 (DDoS 완화, SSL 오프로딩)

### 1.2 서비스 비교

| 항목 | AWS | GCP |
|------|-----|-----|
| L7 (HTTP/HTTPS) | ALB | HTTP(S) Load Balancing |
| L4 (TCP/UDP) | NLB | TCP/UDP Load Balancing |
| 클래식 | CLB (레거시) | - |
| 내부 | Internal ALB/NLB | Internal Load Balancing |
| 글로벌 | Global Accelerator | Global Load Balancing |

---

## 2. AWS Elastic Load Balancing

### 2.1 로드밸런서 유형

| 유형 | 계층 | 사용 사례 | 특징 |
|------|------|----------|------|
| **ALB** | L7 | 웹 앱, 마이크로서비스 | 경로/호스트 라우팅, 웹소켓 |
| **NLB** | L4 | 고성능, 정적 IP 필요 | 수백만 RPS, 초저지연 |
| **GWLB** | L3 | 방화벽, IDS/IPS | 투명 게이트웨이 |

### 2.2 ALB (Application Load Balancer)

```bash
# 1. 대상 그룹 생성
aws elbv2 create-target-group \
    --name my-targets \
    --protocol HTTP \
    --port 80 \
    --vpc-id vpc-12345678 \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --target-type instance

# 2. 인스턴스 등록
aws elbv2 register-targets \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --targets Id=i-12345678 Id=i-87654321

# 3. ALB 생성
aws elbv2 create-load-balancer \
    --name my-alb \
    --subnets subnet-1 subnet-2 \
    --security-groups sg-12345678 \
    --scheme internet-facing \
    --type application

# 4. 리스너 생성
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx
```

**경로 기반 라우팅:**
```bash
# 규칙 추가 (/api/* → API 대상 그룹)
aws elbv2 create-rule \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --priority 10 \
    --conditions Field=path-pattern,Values='/api/*' \
    --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/api-targets/xxx
```

### 2.3 NLB (Network Load Balancer)

```bash
# NLB 생성 (정적 IP)
aws elbv2 create-load-balancer \
    --name my-nlb \
    --subnets subnet-1 subnet-2 \
    --type network \
    --scheme internet-facing

# TCP 리스너
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/net/my-nlb/xxx \
    --protocol TCP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/tcp-targets/xxx
```

### 2.4 SSL/TLS 설정

```bash
# ACM 인증서 요청
aws acm request-certificate \
    --domain-name example.com \
    --subject-alternative-names "*.example.com" \
    --validation-method DNS

# HTTPS 리스너 추가
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=arn:aws:acm:...:certificate/xxx \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx

# HTTP → HTTPS 리다이렉트
aws elbv2 modify-listener \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --default-actions Type=redirect,RedirectConfig='{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}'
```

---

## 3. GCP Cloud Load Balancing

### 3.1 로드밸런서 유형

| 유형 | 범위 | 계층 | 사용 사례 |
|------|------|------|----------|
| **Global HTTP(S)** | 글로벌 | L7 | 웹 앱, CDN 통합 |
| **Regional HTTP(S)** | 리전 | L7 | 단일 리전 앱 |
| **Global TCP/SSL** | 글로벌 | L4 | TCP 프록시 |
| **Regional TCP/UDP** | 리전 | L4 | 네트워크 LB |
| **Internal HTTP(S)** | 리전 | L7 | 내부 마이크로서비스 |
| **Internal TCP/UDP** | 리전 | L4 | 내부 TCP/UDP |

### 3.2 HTTP(S) Load Balancer

```bash
# 1. 인스턴스 그룹 생성 (비관리형)
gcloud compute instance-groups unmanaged create my-group \
    --zone=asia-northeast3-a

gcloud compute instance-groups unmanaged add-instances my-group \
    --zone=asia-northeast3-a \
    --instances=instance-1,instance-2

# 2. 헬스 체크 생성
gcloud compute health-checks create http my-health-check \
    --port=80 \
    --request-path=/health

# 3. 백엔드 서비스 생성
gcloud compute backend-services create my-backend \
    --protocol=HTTP \
    --health-checks=my-health-check \
    --global

# 4. 인스턴스 그룹을 백엔드에 추가
gcloud compute backend-services add-backend my-backend \
    --instance-group=my-group \
    --instance-group-zone=asia-northeast3-a \
    --global

# 5. URL 맵 생성
gcloud compute url-maps create my-url-map \
    --default-service=my-backend

# 6. 대상 HTTP 프록시 생성
gcloud compute target-http-proxies create my-proxy \
    --url-map=my-url-map

# 7. 전역 전달 규칙 생성
gcloud compute forwarding-rules create my-lb \
    --global \
    --target-http-proxy=my-proxy \
    --ports=80
```

### 3.3 SSL/TLS 설정

```bash
# 1. 관리형 SSL 인증서
gcloud compute ssl-certificates create my-cert \
    --domains=example.com,www.example.com \
    --global

# 2. HTTPS 대상 프록시
gcloud compute target-https-proxies create my-https-proxy \
    --url-map=my-url-map \
    --ssl-certificates=my-cert

# 3. HTTPS 전달 규칙
gcloud compute forwarding-rules create my-https-lb \
    --global \
    --target-https-proxy=my-https-proxy \
    --ports=443

# 4. HTTP → HTTPS 리다이렉트
gcloud compute url-maps import my-url-map --source=- <<EOF
name: my-url-map
defaultUrlRedirect:
  httpsRedirect: true
  redirectResponseCode: MOVED_PERMANENTLY_DEFAULT
EOF
```

### 3.4 경로 기반 라우팅

```bash
# URL 맵에 경로 규칙 추가
gcloud compute url-maps add-path-matcher my-url-map \
    --path-matcher-name=api-matcher \
    --default-service=default-backend \
    --path-rules="/api/*=api-backend,/static/*=static-backend"
```

---

## 4. 헬스 체크

### 4.1 AWS 헬스 체크

```bash
# 대상 그룹 헬스 체크 설정
aws elbv2 modify-target-group \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx \
    --health-check-protocol HTTP \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3

# 대상 헬스 상태 확인
aws elbv2 describe-target-health \
    --target-group-arn arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx
```

### 4.2 GCP 헬스 체크

```bash
# HTTP 헬스 체크
gcloud compute health-checks create http my-http-check \
    --port=80 \
    --request-path=/health \
    --check-interval=30s \
    --timeout=5s \
    --healthy-threshold=2 \
    --unhealthy-threshold=3

# TCP 헬스 체크
gcloud compute health-checks create tcp my-tcp-check \
    --port=3306

# 헬스 체크 상태 확인
gcloud compute backend-services get-health my-backend --global
```

---

## 5. Auto Scaling 연동

### 5.1 AWS Auto Scaling Group + ALB

```bash
# 시작 템플릿 생성
aws ec2 create-launch-template \
    --launch-template-name my-template \
    --launch-template-data '{
        "ImageId": "ami-12345678",
        "InstanceType": "t3.micro",
        "SecurityGroupIds": ["sg-12345678"]
    }'

# Auto Scaling Group 생성 (대상 그룹 연결)
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name my-asg \
    --launch-template LaunchTemplateName=my-template,Version='$Latest' \
    --min-size 2 \
    --max-size 10 \
    --desired-capacity 2 \
    --vpc-zone-identifier "subnet-1,subnet-2" \
    --target-group-arns "arn:aws:elasticloadbalancing:...:targetgroup/my-targets/xxx"

# 스케일링 정책
aws autoscaling put-scaling-policy \
    --auto-scaling-group-name my-asg \
    --policy-name cpu-scaling \
    --policy-type TargetTrackingScaling \
    --target-tracking-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ASGAverageCPUUtilization"
        }
    }'
```

### 5.2 GCP Managed Instance Group + LB

```bash
# 인스턴스 템플릿 생성
gcloud compute instance-templates create my-template \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server

# 관리형 인스턴스 그룹 생성
gcloud compute instance-groups managed create my-mig \
    --template=my-template \
    --size=2 \
    --zone=asia-northeast3-a

# 오토스케일링 설정
gcloud compute instance-groups managed set-autoscaling my-mig \
    --zone=asia-northeast3-a \
    --min-num-replicas=2 \
    --max-num-replicas=10 \
    --target-cpu-utilization=0.7

# 로드밸런서에 연결
gcloud compute backend-services add-backend my-backend \
    --instance-group=my-mig \
    --instance-group-zone=asia-northeast3-a \
    --global
```

---

## 6. CDN (Content Delivery Network)

### 6.1 AWS CloudFront

```bash
# CloudFront 배포 생성 (S3 오리진)
aws cloudfront create-distribution \
    --distribution-config '{
        "CallerReference": "my-distribution-2024",
        "Origins": {
            "Quantity": 1,
            "Items": [{
                "Id": "S3-my-bucket",
                "DomainName": "my-bucket.s3.amazonaws.com",
                "S3OriginConfig": {
                    "OriginAccessIdentity": ""
                }
            }]
        },
        "DefaultCacheBehavior": {
            "TargetOriginId": "S3-my-bucket",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {
                "Quantity": 2,
                "Items": ["GET", "HEAD"]
            },
            "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
            "Compress": true
        },
        "Enabled": true,
        "DefaultRootObject": "index.html"
    }'

# 캐시 무효화
aws cloudfront create-invalidation \
    --distribution-id EDFDVBD632BHDS5 \
    --paths "/*"
```

**CloudFront + ALB:**
```bash
# ALB를 오리진으로 하는 CloudFront
{
    "Origins": {
        "Items": [{
            "Id": "ALB-origin",
            "DomainName": "my-alb-12345.ap-northeast-2.elb.amazonaws.com",
            "CustomOriginConfig": {
                "HTTPPort": 80,
                "HTTPSPort": 443,
                "OriginProtocolPolicy": "https-only"
            }
        }]
    }
}
```

### 6.2 GCP Cloud CDN

```bash
# 1. 백엔드 서비스에 CDN 활성화
gcloud compute backend-services update my-backend \
    --enable-cdn \
    --global

# 2. Cloud Storage 버킷을 CDN 오리진으로
gcloud compute backend-buckets create my-cdn-bucket \
    --gcs-bucket-name=my-static-bucket \
    --enable-cdn

# 3. URL 맵에 버킷 추가
gcloud compute url-maps add-path-matcher my-url-map \
    --path-matcher-name=static-matcher \
    --default-backend-bucket=my-cdn-bucket \
    --path-rules="/static/*=my-cdn-bucket"

# 4. 캐시 무효화
gcloud compute url-maps invalidate-cdn-cache my-url-map \
    --path="/*"
```

### 6.3 CDN 캐시 정책

**AWS CloudFront 캐시 정책:**
```bash
# 캐시 정책 생성
aws cloudfront create-cache-policy \
    --cache-policy-config '{
        "Name": "MyPolicy",
        "DefaultTTL": 86400,
        "MaxTTL": 31536000,
        "MinTTL": 0,
        "ParametersInCacheKeyAndForwardedToOrigin": {
            "EnableAcceptEncodingGzip": true,
            "HeadersConfig": {"HeaderBehavior": "none"},
            "CookiesConfig": {"CookieBehavior": "none"},
            "QueryStringsConfig": {"QueryStringBehavior": "none"}
        }
    }'
```

**GCP Cloud CDN 캐시 모드:**
```bash
# 캐시 모드 설정
gcloud compute backend-services update my-backend \
    --cache-mode=CACHE_ALL_STATIC \
    --default-ttl=3600 \
    --max-ttl=86400 \
    --global
```

---

## 7. 비용 비교

### 7.1 로드밸런서 비용

| 서비스 | 고정 비용 | 처리 비용 |
|--------|----------|----------|
| AWS ALB | ~$18/월 | $0.008/LCU-시간 |
| AWS NLB | ~$18/월 | $0.006/NLCU-시간 |
| GCP HTTP(S) LB | ~$18/월 | $0.008/GB 처리량 |
| GCP TCP/UDP LB | 리전당 $18/월 | 규칙당 추가 |

### 7.2 CDN 비용

| 서비스 | 데이터 전송 (첫 10TB) |
|--------|---------------------|
| AWS CloudFront | ~$0.085/GB (미국/유럽) |
| GCP Cloud CDN | ~$0.08/GB (미국/유럽) |

---

## 8. 모니터링

### 8.1 AWS CloudWatch 메트릭

```bash
# ALB 메트릭 조회
aws cloudwatch get-metric-statistics \
    --namespace AWS/ApplicationELB \
    --metric-name RequestCount \
    --dimensions Name=LoadBalancer,Value=app/my-alb/xxx \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-01T23:59:59Z \
    --period 300 \
    --statistics Sum

# 주요 메트릭:
# - RequestCount
# - HTTPCode_Target_2XX_Count
# - TargetResponseTime
# - HealthyHostCount
# - UnHealthyHostCount
```

### 8.2 GCP Cloud Monitoring

```bash
# 메트릭 조회
gcloud monitoring metrics list \
    --filter="metric.type:loadbalancing"

# 알림 정책 생성
gcloud alpha monitoring policies create \
    --display-name="High Latency Alert" \
    --condition-display-name="Latency > 1s" \
    --condition-filter='metric.type="loadbalancing.googleapis.com/https/backend_latencies"' \
    --condition-threshold-value=1000 \
    --notification-channels=projects/PROJECT/notificationChannels/xxx
```

---

## 9. 다음 단계

- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - 데이터베이스
- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - 모니터링 상세

---

## 참고 자료

- [AWS ELB Documentation](https://docs.aws.amazon.com/elasticloadbalancing/)
- [AWS CloudFront Documentation](https://docs.aws.amazon.com/cloudfront/)
- [GCP Load Balancing](https://cloud.google.com/load-balancing/docs)
- [GCP Cloud CDN](https://cloud.google.com/cdn/docs)
