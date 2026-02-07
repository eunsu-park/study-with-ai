# Security Services

## 1. Security Overview

### 1.1 Cloud Security Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Application Security                                       │
│  - Input validation, authentication/authorization, session  │
├─────────────────────────────────────────────────────────────┤
│  Data Security                                              │
│  - Encryption (at rest, in transit), key management         │
├─────────────────────────────────────────────────────────────┤
│  Network Security                                           │
│  - Firewall, VPC, WAF, DDoS protection                      │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Security                                    │
│  - Patch management, vulnerability scanning                 │
├─────────────────────────────────────────────────────────────┤
│  Identity/Access Management                                 │
│  - IAM, MFA, least privilege                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Service Mapping

| Function | AWS | GCP |
|------|-----|-----|
| Network Firewall | Security Groups, NACL | Firewall Rules |
| WAF | AWS WAF | Cloud Armor |
| DDoS | AWS Shield | Cloud Armor |
| Key Management | KMS | Cloud KMS |
| Secret Management | Secrets Manager | Secret Manager |
| Vulnerability Scanning | Inspector | Security Command Center |
| Threat Detection | GuardDuty | Security Command Center |

---

## 2. Network Security

### 2.1 AWS Security Groups

```bash
# Create security group
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server SG" \
    --vpc-id vpc-12345678

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --ip-permissions '[
        {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "203.0.113.0/24", "Description": "Office IP"}]}
    ]'

# Allow traffic from another security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-db \
    --source-group sg-app \
    --protocol tcp \
    --port 3306

# Remove rule
aws ec2 revoke-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

### 2.2 GCP Firewall Rules

```bash
# Create firewall rule
gcloud compute firewall-rules create allow-http \
    --network=my-vpc \
    --allow=tcp:80,tcp:443 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=http-server \
    --priority=1000

# Allow SSH (specific IP)
gcloud compute firewall-rules create allow-ssh-office \
    --network=my-vpc \
    --allow=tcp:22 \
    --source-ranges=203.0.113.0/24 \
    --target-tags=ssh-server

# Allow internal communication
gcloud compute firewall-rules create allow-internal \
    --network=my-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/8

# Deny rule (lower priority)
gcloud compute firewall-rules create deny-all-ingress \
    --network=my-vpc \
    --action=DENY \
    --rules=all \
    --source-ranges=0.0.0.0/0 \
    --priority=65534

# Delete rule
gcloud compute firewall-rules delete allow-http
```

---

## 3. WAF (Web Application Firewall)

### 3.1 AWS WAF

```bash
# Create web ACL
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

# Associate with ALB
aws wafv2 associate-web-acl \
    --web-acl-arn arn:aws:wafv2:...:webacl/my-web-acl/xxx \
    --resource-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-alb/xxx
```

**Common Rules:**
- AWSManagedRulesCommonRuleSet: OWASP Top 10
- AWSManagedRulesSQLiRuleSet: SQL injection
- AWSManagedRulesKnownBadInputsRuleSet: Malicious input
- AWSManagedRulesAmazonIpReputationList: IP reputation

### 3.2 GCP Cloud Armor

```bash
# Create security policy
gcloud compute security-policies create my-policy \
    --description="My security policy"

# Add rule (block SQL injection)
gcloud compute security-policies rules create 1000 \
    --security-policy=my-policy \
    --expression="evaluatePreconfiguredWaf('sqli-v33-stable')" \
    --action=deny-403

# Add rule (block XSS)
gcloud compute security-policies rules create 2000 \
    --security-policy=my-policy \
    --expression="evaluatePreconfiguredWaf('xss-v33-stable')" \
    --action=deny-403

# Add rule (block IP)
gcloud compute security-policies rules create 3000 \
    --security-policy=my-policy \
    --src-ip-ranges="203.0.113.0/24" \
    --action=deny-403

# Rate limiting
gcloud compute security-policies rules create 4000 \
    --security-policy=my-policy \
    --expression="true" \
    --action=rate-based-ban \
    --rate-limit-threshold-count=1000 \
    --rate-limit-threshold-interval-sec=60

# Attach to backend service
gcloud compute backend-services update my-backend \
    --security-policy=my-policy \
    --global
```

---

## 4. Key Management (KMS)

### 4.1 AWS KMS

```bash
# Create customer managed key
aws kms create-key \
    --description "My encryption key" \
    --key-usage ENCRYPT_DECRYPT \
    --origin AWS_KMS

# Create alias
aws kms create-alias \
    --alias-name alias/my-key \
    --target-key-id 12345678-1234-1234-1234-123456789012

# Encrypt data
aws kms encrypt \
    --key-id alias/my-key \
    --plaintext fileb://plaintext.txt \
    --output text \
    --query CiphertextBlob | base64 --decode > encrypted.bin

# Decrypt data
aws kms decrypt \
    --ciphertext-blob fileb://encrypted.bin \
    --output text \
    --query Plaintext | base64 --decode > decrypted.txt

# Update key policy
aws kms put-key-policy \
    --key-id 12345678-1234-1234-1234-123456789012 \
    --policy-name default \
    --policy file://key-policy.json
```

**S3 Server-Side Encryption:**
```bash
# Set bucket encryption
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
# Create key ring
gcloud kms keyrings create my-keyring \
    --location=asia-northeast3

# Create encryption key
gcloud kms keys create my-key \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --purpose=encryption

# Encrypt data
gcloud kms encrypt \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --key=my-key \
    --plaintext-file=plaintext.txt \
    --ciphertext-file=encrypted.bin

# Decrypt data
gcloud kms decrypt \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --key=my-key \
    --ciphertext-file=encrypted.bin \
    --plaintext-file=decrypted.txt

# Grant encryption permission to service account
gcloud kms keys add-iam-policy-binding my-key \
    --location=asia-northeast3 \
    --keyring=my-keyring \
    --member="serviceAccount:my-sa@PROJECT.iam.gserviceaccount.com" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

**Cloud Storage CMEK:**
```bash
# Create bucket with CMEK
gsutil mb -l asia-northeast3 \
    -k projects/PROJECT/locations/asia-northeast3/keyRings/my-keyring/cryptoKeys/my-key \
    gs://my-encrypted-bucket
```

---

## 5. Secret Management

### 5.1 AWS Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
    --name my-database-credentials \
    --secret-string '{"username":"admin","password":"MySecretPassword123!"}'

# Retrieve secret
aws secretsmanager get-secret-value \
    --secret-id my-database-credentials \
    --query SecretString \
    --output text

# Update secret
aws secretsmanager update-secret \
    --secret-id my-database-credentials \
    --secret-string '{"username":"admin","password":"NewPassword456!"}'

# Enable automatic rotation
aws secretsmanager rotate-secret \
    --secret-id my-database-credentials \
    --rotation-lambda-arn arn:aws:lambda:...:function:RotateSecret \
    --rotation-rules AutomaticallyAfterDays=30
```

**Use in Application:**
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
# Create secret
echo -n "MySecretPassword123!" | gcloud secrets create my-secret \
    --data-file=-

# Or from file
gcloud secrets create my-secret --data-file=secret.txt

# Retrieve secret
gcloud secrets versions access latest --secret=my-secret

# Add new version
echo -n "NewPassword456!" | gcloud secrets versions add my-secret \
    --data-file=-

# Grant access permission to service account
gcloud secrets add-iam-policy-binding my-secret \
    --member="serviceAccount:my-sa@PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**Use in Application:**
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

## 6. Encryption

### 6.1 Encryption at Rest

| Service | AWS | GCP |
|--------|-----|-----|
| Object Storage | S3 SSE-S3, SSE-KMS | Cloud Storage CMEK |
| Block Storage | EBS encryption | PD encryption |
| Database | RDS encryption | Cloud SQL encryption |
| Default Encryption | Some services | All services |

### 6.2 Encryption in Transit

```bash
# AWS ALB enforce HTTPS
aws elbv2 modify-listener \
    --listener-arn arn:aws:elasticloadbalancing:...:listener/xxx \
    --protocol HTTPS \
    --certificates CertificateArn=arn:aws:acm:...:certificate/xxx

# GCP HTTPS load balancer
gcloud compute target-https-proxies create my-https-proxy \
    --url-map=my-url-map \
    --ssl-certificates=my-cert

# RDS enforce SSL
aws rds modify-db-instance \
    --db-instance-identifier my-database \
    --ca-certificate-identifier rds-ca-2019

# Cloud SQL enforce SSL
gcloud sql instances patch my-database --require-ssl
```

---

## 7. Vulnerability Detection

### 7.1 AWS Inspector

```bash
# Enable Inspector v2 (account level)
aws inspector2 enable \
    --resource-types EC2 ECR

# View scan results
aws inspector2 list-findings \
    --filter-criteria '{
        "findingStatus": [{"comparison": "EQUALS", "value": "ACTIVE"}],
        "severity": [{"comparison": "EQUALS", "value": "HIGH"}]
    }'
```

### 7.2 GCP Security Command Center

```bash
# Organization-level activation required (in Console)

# View findings
gcloud scc findings list ORGANIZATION_ID \
    --source=SOURCE_ID \
    --filter="state=\"ACTIVE\""
```

---

## 8. Threat Detection

### 8.1 AWS GuardDuty

```bash
# Enable GuardDuty
aws guardduty create-detector --enable

# View findings
aws guardduty list-findings --detector-id DETECTOR_ID

aws guardduty get-findings \
    --detector-id DETECTOR_ID \
    --finding-ids FINDING_ID

# Add trusted IP list
aws guardduty create-ip-set \
    --detector-id DETECTOR_ID \
    --name "Trusted IPs" \
    --format TXT \
    --location s3://my-bucket/trusted-ips.txt \
    --activate
```

### 8.2 GCP Security Command Center

```bash
# Threat detection (Premium required)
# Event Threat Detection
# Container Threat Detection
# Virtual Machine Threat Detection

# Check organization policy violations
gcloud scc findings list ORGANIZATION_ID \
    --source=SECURITY_HEALTH_ANALYTICS \
    --filter="category=\"PUBLIC_BUCKET_ACL\""
```

---

## 9. Audit Logging

### 9.1 AWS CloudTrail

```bash
# Create trail
aws cloudtrail create-trail \
    --name my-trail \
    --s3-bucket-name my-log-bucket \
    --is-multi-region-trail \
    --enable-log-file-validation

# Start logging
aws cloudtrail start-logging --name my-trail

# View events
aws cloudtrail lookup-events \
    --lookup-attributes AttributeKey=EventName,AttributeValue=ConsoleLogin \
    --start-time 2024-01-01T00:00:00Z
```

### 9.2 GCP Cloud Audit Logs

```bash
# Audit logs are enabled by default

# View logs
gcloud logging read 'logName:"cloudaudit.googleapis.com"' \
    --project=PROJECT_ID \
    --limit=10

# Enable Data Access logs (additional setup required)
gcloud projects get-iam-policy PROJECT_ID --format=json > policy.json
# Edit and apply
gcloud projects set-iam-policy PROJECT_ID policy.json
```

---

## 10. Security Checklist

### 10.1 Account/IAM
```
□ Enable Root/Owner MFA
□ Apply least privilege principle
□ Regular permission review
□ Disable unused credentials
□ Strong password policy
```

### 10.2 Network
```
□ Remove default security group rules
□ Open only necessary ports
□ Use private subnets
□ Enable VPC Flow Logs
□ Apply WAF (web apps)
```

### 10.3 Data
```
□ Enable encryption at rest
□ Encryption in transit (HTTPS/TLS)
□ Block public access
□ Encrypt backups
□ Key rotation
```

### 10.4 Monitoring
```
□ Enable CloudTrail/Audit Logs
□ Enable GuardDuty/SCC
□ Set up security alerts
□ Regular vulnerability scans
□ Incident response plan
```

---

## 11. Next Steps

- [15_CLI_and_SDK.md](./15_CLI_and_SDK.md) - CLI/SDK Automation
- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM Details

---

## References

- [AWS Security Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices)
- [AWS WAF](https://docs.aws.amazon.com/waf/)
- [GCP Cloud Armor](https://cloud.google.com/armor/docs)
