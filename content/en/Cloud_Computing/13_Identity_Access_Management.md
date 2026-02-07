# IAM (Identity and Access Management)

## 1. IAM Overview

### 1.1 What is IAM?

IAM is a service that securely controls access to cloud resources.

**Core Questions:**
- **Who**: Users, groups, roles
- **What**: Resources
- **How**: Permissions (allow/deny)

### 1.2 AWS vs GCP IAM Comparison

| Item | AWS IAM | GCP IAM |
|------|---------|---------|
| Scope | Account level | Organization/project level |
| Policy Attachment | To users/groups/roles | To resources |
| Roles | Assume role (AssumeRole) | Role binding |
| Service Account | Role + instance profile | Service account |

---

## 2. AWS IAM

### 2.1 Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│  AWS Account                                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  IAM                                                    ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    Users      │  │     Groups    │                   ││
│  │  │               │  │               │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  │         ↓                  ↓                            ││
│  │  ┌─────────────────────────────────────────────┐        ││
│  │  │              Policies                       │        ││
│  │  │  { "Effect": "Allow",                       │        ││
│  │  │    "Action": "s3:*",                        │        ││
│  │  │    "Resource": "*" }                        │        ││
│  │  └─────────────────────────────────────────────┘        ││
│  │                     ↓                                   ││
│  │  ┌───────────────────────────────────────────────────┐  ││
│  │  │              Roles                                │  ││
│  │  │  - EC2 instance role                             │  ││
│  │  │  - Lambda execution role                         │  ││
│  │  │  - Cross-account role                            │  ││
│  │  └───────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Users and Groups

```bash
# Create user
aws iam create-user --user-name john

# Set login password
aws iam create-login-profile \
    --user-name john \
    --password 'TempPassword123!' \
    --password-reset-required

# Create access key (programmatic access)
aws iam create-access-key --user-name john

# Create group
aws iam create-group --group-name Developers

# Add user to group
aws iam add-user-to-group --group-name Developers --user-name john

# List group members
aws iam get-group --group-name Developers
```

### 2.3 Policies

**Policy Structure:**
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
# Attach managed policy
aws iam attach-user-policy \
    --user-name john \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create custom policy
aws iam create-policy \
    --policy-name MyS3Policy \
    --policy-document file://policy.json

# Attach policy to group
aws iam attach-group-policy \
    --group-name Developers \
    --policy-arn arn:aws:iam::123456789012:policy/MyS3Policy

# Add inline policy
aws iam put-user-policy \
    --user-name john \
    --policy-name InlinePolicy \
    --policy-document file://inline-policy.json
```

### 2.4 Roles

**EC2 Instance Role:**
```bash
# Trust policy (who can assume the role)
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

# Create role
aws iam create-role \
    --role-name EC2-S3-Access \
    --assume-role-policy-document file://trust-policy.json

# Attach policy
aws iam attach-role-policy \
    --role-name EC2-S3-Access \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile and add role
aws iam create-instance-profile --instance-profile-name EC2-S3-Profile
aws iam add-role-to-instance-profile \
    --instance-profile-name EC2-S3-Profile \
    --role-name EC2-S3-Access

# Attach instance profile to EC2
aws ec2 associate-iam-instance-profile \
    --instance-id i-1234567890abcdef0 \
    --iam-instance-profile Name=EC2-S3-Profile
```

**Cross-Account Role:**
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
# Assume role from another account
aws sts assume-role \
    --role-arn arn:aws:iam::TARGET_ACCOUNT:role/CrossAccountRole \
    --role-session-name MySession
```

---

## 3. GCP IAM

### 3.1 Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│  Organization                                               │
│  ├── Folder                                                 │
│  │   └── Project                                            │
│  │       └── Resource                                       │
│  └─────────────────────────────────────────────────────────│
│                                                             │
│  IAM Binding:                                               │
│  Member + Role = Permission on Resource                     │
│                                                             │
│  Example: user:john@example.com + roles/storage.admin      │
│      → Admin permission on gs://my-bucket                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Role Types

| Type | Description | Example |
|------|------|------|
| **Basic Roles** | Broad permissions | Owner, Editor, Viewer |
| **Predefined Roles** | Service-specific granular | roles/storage.admin |
| **Custom Roles** | User-defined | my-custom-role |

### 3.3 Role Bindings

```bash
# Grant project-level role
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"

# Grant bucket-level role
gsutil iam ch user:john@example.com:objectViewer gs://my-bucket

# View role bindings
gcloud projects get-iam-policy PROJECT_ID

# Remove role
gcloud projects remove-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"
```

### 3.4 Service Accounts

```bash
# Create service account
gcloud iam service-accounts create my-service-account \
    --display-name="My Service Account"

# Grant role
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:my-service-account@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Create key file (programmatic access)
gcloud iam service-accounts keys create key.json \
    --iam-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com

# Attach service account to Compute Engine
gcloud compute instances create my-instance \
    --service-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com \
    --scopes=cloud-platform
```

### 3.5 Workload Identity (GKE)

```bash
# Enable workload identity pool
gcloud container clusters update my-cluster \
    --region=asia-northeast3 \
    --workload-pool=PROJECT_ID.svc.id.goog

# Bind Kubernetes service account to GCP service account
gcloud iam service-accounts add-iam-policy-binding \
    my-gcp-sa@PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/iam.workloadIdentityUser \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/K8S_SA]"
```

---

## 4. Principle of Least Privilege

### 4.1 Principle

```
Least Privilege = Grant only minimum permissions needed for the task

Bad Examples:
- Admin permissions to all users
- Permissions on * (all resources)

Good Examples:
- Specify only required Actions
- Permissions on specific resources
- Conditional access
```

### 4.2 AWS Policy Examples

**Bad Example:**
```json
{
    "Effect": "Allow",
    "Action": "*",
    "Resource": "*"
}
```

**Good Example:**
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

### 4.3 GCP Role Selection

```bash
# Too broad roles (avoid)
roles/owner
roles/editor

# Appropriate roles
roles/storage.objectViewer  # Read objects only
roles/compute.instanceAdmin.v1  # Manage instances only
roles/cloudsql.client  # SQL connection only
```

---

## 5. Conditional Access

### 5.1 AWS Conditions

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

### 5.2 GCP Conditions

```bash
# Conditional role binding
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin" \
    --condition='expression=request.time < timestamp("2024-12-31T23:59:59Z"),title=Temporary Access,description=Access until end of year'

# IP-based condition (with VPC Service Controls)
expression: 'resource.name.startsWith("projects/PROJECT_ID/zones/asia-northeast3")'
```

---

## 6. Permission Analysis

### 6.1 AWS IAM Access Analyzer

```bash
# Create Access Analyzer
aws accessanalyzer create-analyzer \
    --analyzer-name my-analyzer \
    --type ACCOUNT

# View findings
aws accessanalyzer list-findings --analyzer-arn arn:aws:access-analyzer:...:analyzer/my-analyzer

# Validate policy
aws accessanalyzer validate-policy \
    --policy-document file://policy.json \
    --policy-type IDENTITY_POLICY
```

### 6.2 GCP Policy Analyzer

```bash
# Analyze IAM policy
gcloud asset analyze-iam-policy \
    --organization=ORG_ID \
    --identity="user:john@example.com"

# Check permissions
gcloud projects get-iam-policy PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:john@example.com" \
    --format="table(bindings.role)"
```

---

## 7. MFA (Multi-Factor Authentication)

### 7.1 AWS MFA

```bash
# Enable virtual MFA
aws iam create-virtual-mfa-device \
    --virtual-mfa-device-name john-mfa \
    --outfile qrcode.png \
    --bootstrap-method QRCodePNG

# Attach MFA device
aws iam enable-mfa-device \
    --user-name john \
    --serial-number arn:aws:iam::123456789012:mfa/john-mfa \
    --authentication-code1 123456 \
    --authentication-code2 789012

# MFA required policy
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

### 7.2 GCP 2-Step Verification

```bash
# Enforce 2FA at organization level (in Admin Console)
# Google Workspace Admin → Security → 2-Step Verification

# Service accounts don't support MFA → Instead:
# - Secure key file management
# - Use workload identity
# - Use short-lived tokens
```

---

## 8. Common Role Patterns

### 8.1 AWS Common Roles

| Role | Permission | Use Case |
|------|------|------|
| AdministratorAccess | Full | Administrator |
| PowerUserAccess | All except IAM | Developer |
| ReadOnlyAccess | Read-only | Auditor/Viewer |
| AmazonS3FullAccess | S3 full | Storage management |
| AmazonEC2FullAccess | EC2 full | Compute management |

### 8.2 GCP Common Roles

| Role | Permission | Use Case |
|------|------|------|
| roles/owner | Full | Administrator |
| roles/editor | Edit except IAM | Developer |
| roles/viewer | Read-only | Viewer |
| roles/compute.admin | Compute full | Infrastructure management |
| roles/storage.admin | Storage full | Storage management |

---

## 9. Security Best Practices

```
□ Don't use Root/Owner account for daily tasks
□ Enable MFA on Root/Owner account
□ Apply principle of least privilege
□ Manage permissions via groups/roles (not individual users)
□ Regular permission review (remove unused permissions)
□ Secure service account key files
□ Use temporary credentials (STS, workload identity)
□ Use conditional access (IP, time, MFA)
□ Enable audit logs (CloudTrail, Cloud Audit Logs)
□ Set up policy change notifications
```

---

## 10. Next Steps

- [14_Security_Services.md](./14_Security_Services.md) - Security Services
- [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) - Initial Account Setup

---

## References

- [AWS IAM Documentation](https://docs.aws.amazon.com/iam/)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [GCP IAM Documentation](https://cloud.google.com/iam/docs)
- [GCP IAM Best Practices](https://cloud.google.com/iam/docs/using-iam-securely)
