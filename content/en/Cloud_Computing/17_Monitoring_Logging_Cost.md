# Monitoring, Logging & Cost Management

## 1. Monitoring Overview

### 1.1 Why Monitoring is Needed

- Ensure system availability
- Early detection of performance issues
- Capacity planning
- Cost optimization
- Security anomaly detection

### 1.2 Service Mapping

| Function | AWS | GCP |
|------|-----|-----|
| Metric Monitoring | CloudWatch | Cloud Monitoring |
| Log Collection | CloudWatch Logs | Cloud Logging |
| Tracing | X-Ray | Cloud Trace |
| Dashboards | CloudWatch Dashboards | Cloud Monitoring Dashboards |
| Alerting | CloudWatch Alarms + SNS | Alerting Policies |
| Cost Management | Cost Explorer, Budgets | Billing, Budgets |

---

## 2. AWS CloudWatch

### 2.1 Metrics

```bash
# List EC2 metrics
aws cloudwatch list-metrics --namespace AWS/EC2

# Get metric data
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-01T23:59:59Z \
    --period 300 \
    --statistics Average

# Publish custom metric
aws cloudwatch put-metric-data \
    --namespace MyApp \
    --metric-name RequestCount \
    --value 100 \
    --unit Count \
    --dimensions Environment=Production
```

**Key Metrics:**

| Service | Metric | Description |
|--------|--------|------|
| EC2 | CPUUtilization | CPU usage |
| EC2 | NetworkIn/Out | Network traffic |
| RDS | DatabaseConnections | DB connections |
| RDS | FreeStorageSpace | Remaining storage |
| ALB | RequestCount | Request count |
| ALB | TargetResponseTime | Response time |
| Lambda | Invocations | Invocation count |
| Lambda | Duration | Execution time |

### 2.2 Alarms

```bash
# Create CPU alarm
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

# List alarms
aws cloudwatch describe-alarms

# Check alarm history
aws cloudwatch describe-alarm-history \
    --alarm-name high-cpu
```

### 2.3 Dashboards

```bash
# Create dashboard
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

### 3.1 Log Group Management

```bash
# Create log group
aws logs create-log-group --log-group-name /myapp/production

# Set retention policy
aws logs put-retention-policy \
    --log-group-name /myapp/production \
    --retention-in-days 30

# List log streams
aws logs describe-log-streams --log-group-name /myapp/production

# Query logs
aws logs filter-log-events \
    --log-group-name /myapp/production \
    --filter-pattern "ERROR" \
    --start-time 1704067200000 \
    --end-time 1704153600000
```

### 3.2 Log Insights

```bash
# Start log query
aws logs start-query \
    --log-group-name /myapp/production \
    --start-time 1704067200 \
    --end-time 1704153600 \
    --query-string 'fields @timestamp, @message
        | filter @message like /ERROR/
        | sort @timestamp desc
        | limit 20'

# Get query results
aws logs get-query-results --query-id QUERY_ID
```

### 3.3 Send Logs from EC2

```bash
# Install CloudWatch Agent (Amazon Linux)
sudo yum install -y amazon-cloudwatch-agent

# Configuration file
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

# Start agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s
```

---

## 4. GCP Cloud Monitoring

### 4.1 Metrics

```bash
# List metrics
gcloud monitoring metrics list --filter="metric.type:compute.googleapis.com"

# Read metric data (limited in gcloud, API/console recommended)
gcloud monitoring metrics read \
    "compute.googleapis.com/instance/cpu/utilization" \
    --project=PROJECT_ID
```

**Key Metrics:**

| Service | Metric | Description |
|--------|--------|------|
| Compute | cpu/utilization | CPU usage |
| Compute | network/received_bytes | Received traffic |
| Cloud SQL | database/disk/utilization | Disk usage |
| Cloud Run | request_count | Request count |
| GKE | node/cpu/utilization | Node CPU |

### 4.2 Alerting Policies

```bash
# Create notification channel (email)
gcloud alpha monitoring channels create \
    --display-name="Email Alerts" \
    --type=email \
    --channel-labels=email_address=admin@example.com

# Create alerting policy
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

### 5.1 Log Queries

```bash
# Query logs
gcloud logging read 'resource.type="gce_instance"' \
    --limit=10 \
    --format=json

# Error logs only
gcloud logging read 'severity>=ERROR' \
    --limit=20

# Specific time range
gcloud logging read 'timestamp>="2024-01-01T00:00:00Z"' \
    --limit=100

# Create log sink (export to Cloud Storage)
gcloud logging sinks create my-sink \
    storage.googleapis.com/my-log-bucket \
    --log-filter='resource.type="gce_instance"'
```

### 5.2 Log-based Metrics

```bash
# Create error count metric
gcloud logging metrics create error-count \
    --description="Count of errors" \
    --log-filter='severity>=ERROR'

# List metrics
gcloud logging metrics list
```

---

## 6. Cost Management

### 6.1 AWS Cost Explorer

```bash
# Query monthly cost
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE

# Cost by service
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --output table
```

### 6.2 AWS Budgets

```bash
# Create monthly budget
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

# List budgets
aws budgets describe-budgets --account-id 123456789012
```

### 6.3 GCP Billing

```bash
# List billing accounts
gcloud billing accounts list

# Link project to billing
gcloud billing projects link PROJECT_ID \
    --billing-account=BILLING_ACCOUNT_ID

# Create budget
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Monthly Budget" \
    --budget-amount=100USD \
    --threshold-rule=percent=0.8,basis=CURRENT_SPEND \
    --all-updates-rule-pubsub-topic=projects/PROJECT/topics/budget-alerts
```

---

## 7. Cost Optimization Strategies

### 7.1 Compute Optimization

| Strategy | AWS | GCP |
|------|-----|-----|
| Reserved Instances | Reserved Instances | Committed Use |
| Spot/Preemptible | Spot Instances | Spot/Preemptible VMs |
| Auto Scaling | Auto Scaling | Managed Instance Groups |
| Right Sizing | AWS Compute Optimizer | Recommender |

```bash
# AWS recommendations
aws compute-optimizer get-ec2-instance-recommendations

# GCP recommendations
gcloud recommender recommendations list \
    --project=PROJECT_ID \
    --location=global \
    --recommender=google.compute.instance.MachineTypeRecommender
```

### 7.2 Storage Optimization

```bash
# S3 storage class transitions
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

# GCP lifecycle policy
gsutil lifecycle set lifecycle.json gs://my-bucket
```

### 7.3 Cost Savings Checklist

```
□ Clean up unused resources
  - Stopped instances (storage costs continue)
  - Unattached EBS/PD volumes
  - Old snapshots
  - Unused Elastic IP / static IP

□ Right sizing
  - Analyze instance utilization
  - Check over-provisioning
  - Apply rightsizing recommendations

□ Reserved capacity
  - Reserved instances for stable workloads
  - Review 1-year/3-year commitments

□ Use spot/preemptible
  - Batch jobs, dev environments
  - Interrupt-tolerant workloads

□ Storage optimization
  - Apply lifecycle policies
  - Use appropriate storage class
  - Clean up unnecessary data

□ Network costs
  - Communicate within same AZ/region
  - Use CDN
  - Optimize NAT Gateway traffic
```

---

## 8. Tag-based Cost Tracking

### 8.1 Tag Strategy

```hcl
# Terraform example
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

### 8.2 Cost Allocation Tags

```bash
# Enable AWS cost allocation tags (in Billing Console)

# Query cost by tag
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=TAG,Key=Project

# GCP cost by label (requires BigQuery export)
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

## 9. Dashboard Example

### 9.1 Operations Dashboard Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Operations Dashboard                                        │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   CPU Usage     │  │  Memory Usage   │  │  Requests    │ │
│  │   [Graph]       │  │   [Graph]       │  │  [Graph]     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Response Time  │  │   Error Rate    │  │  Active Conn │ │
│  │   [Graph]       │  │   [Graph]       │  │  [Graph]     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Recent Alarms / Incidents                            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Cost Summary (This Month)                            │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Alert Configuration Recommendations

### 10.1 Essential Alerts

| Category | Condition | Urgency |
|----------|------|--------|
| CPU | > 80% (5min) | Medium |
| CPU | > 95% (2min) | High |
| Memory | > 85% | Medium |
| Disk | > 80% | Medium |
| Disk | > 90% | High |
| Health Check | Failed | High |
| Error Rate | > 1% | Medium |
| Error Rate | > 5% | High |
| Response Time | > 2s | Medium |
| Cost | > 80% budget | Medium |

### 10.2 Notification Channels

```bash
# Create AWS SNS topic
aws sns create-topic --name alerts

# Subscribe email
aws sns subscribe \
    --topic-arn arn:aws:sns:...:alerts \
    --protocol email \
    --notification-endpoint admin@example.com

# Slack webhook (via Lambda)
# PagerDuty, Opsgenie, etc. integration
```

---

## 11. Next Steps

- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC Flow Logs
- [14_Security_Services.md](./14_Security_Services.md) - Security Monitoring

---

## References

- [AWS CloudWatch Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [AWS Cost Management](https://docs.aws.amazon.com/cost-management/)
- [GCP Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [GCP Billing](https://cloud.google.com/billing/docs)
