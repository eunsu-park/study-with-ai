# Virtual Machines (EC2 / Compute Engine)

## 1. Virtual Machine Overview

Virtual machines (VMs) are the most fundamental computing resources in the cloud.

### 1.1 Service Comparison

| Item | AWS EC2 | GCP Compute Engine |
|------|---------|-------------------|
| Service Name | Elastic Compute Cloud | Compute Engine |
| Instance Unit | Instance | Instance |
| Image | AMI | Image |
| Instance Type | Instance Types | Machine Types |
| Launch Script | User Data | Startup Script |
| Metadata | Instance Metadata | Metadata Server |

---

## 2. Instance Types

### 2.1 AWS EC2 Instance Types

**Naming Convention:** `{family}{generation}{attributes}.{size}`

Examples: `t3.medium`, `m5.xlarge`, `c6i.2xlarge`

| Family | Purpose | Examples |
|--------|------|------|
| **t** | General Purpose (Burstable) | t3.micro, t3.small |
| **m** | General Purpose (Balanced) | m5.large, m6i.xlarge |
| **c** | Compute Optimized | c5.xlarge, c6i.2xlarge |
| **r** | Memory Optimized | r5.large, r6i.xlarge |
| **i** | Storage Optimized | i3.large, i3en.xlarge |
| **g/p** | GPU | g4dn.xlarge, p4d.24xlarge |

**Key Instance Specifications:**

| Type | vCPU | Memory | Network | Use Case |
|------|------|--------|----------|------|
| t3.micro | 2 | 1 GB | Low | Free tier, development |
| t3.medium | 2 | 4 GB | Low-Mod | Small apps |
| m5.large | 2 | 8 GB | Up to 10 Gbps | General purpose |
| c5.xlarge | 4 | 8 GB | Up to 10 Gbps | CPU intensive |
| r5.large | 2 | 16 GB | Up to 10 Gbps | Memory intensive |

### 2.2 GCP Machine Types

**Naming Convention:** `{series}-{type}-{vCPU-count}` or custom

Examples: `e2-medium`, `n2-standard-4`, `c2-standard-8`

| Series | Purpose | Examples |
|--------|------|------|
| **e2** | Cost-effective General Purpose | e2-micro, e2-medium |
| **n2/n2d** | General Purpose (Balanced) | n2-standard-2, n2-highmem-4 |
| **c2/c2d** | Compute Optimized | c2-standard-4 |
| **m1/m2** | Memory Optimized | m1-megamem-96 |
| **a2** | GPU (A100) | a2-highgpu-1g |

**Key Machine Type Specifications:**

| Type | vCPU | Memory | Network | Use Case |
|------|------|--------|----------|------|
| e2-micro | 0.25-2 | 1 GB | 1 Gbps | Free tier |
| e2-medium | 1-2 | 4 GB | 2 Gbps | Small apps |
| n2-standard-2 | 2 | 8 GB | 10 Gbps | General purpose |
| c2-standard-4 | 4 | 16 GB | 10 Gbps | CPU intensive |
| n2-highmem-2 | 2 | 16 GB | 10 Gbps | Memory intensive |

### 2.3 Custom Machine Types (GCP)

GCP allows you to specify vCPU and memory individually.

```bash
# Create custom machine type
gcloud compute instances create my-instance \
    --custom-cpu=6 \
    --custom-memory=24GB \
    --zone=asia-northeast3-a
```

---

## 3. Images (AMI / Image)

### 3.1 AWS AMI

**AMI (Amazon Machine Image)** Components:
- Root volume template (OS, applications)
- Instance type, security group defaults
- Block device mapping

```bash
# Search available AMI (Amazon Linux 2023)
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-x86_64" \
    --query 'Images | sort_by(@, &CreationDate) | [-1]'

# Major AMI types
# Amazon Linux 2023: al2023-ami-*
# Ubuntu 22.04: ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-*
# Windows Server: Windows_Server-2022-*
```

### 3.2 GCP Images

```bash
# List available images
gcloud compute images list

# Images from specific project
gcloud compute images list \
    --filter="family:ubuntu-2204-lts"

# Major image families
# debian-11, debian-12
# ubuntu-2204-lts, ubuntu-2404-lts
# centos-stream-9, rocky-linux-9
# windows-2022
```

---

## 4. Creating Instances

### 4.1 AWS EC2 Instance Creation

**Console:**
1. EC2 Dashboard → "Launch instance"
2. Enter name
3. Select AMI (e.g., Amazon Linux 2023)
4. Select instance type (e.g., t3.micro)
5. Create/select key pair
6. Network settings (VPC, subnet, security group)
7. Storage configuration
8. "Launch instance"

**AWS CLI:**
```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name my-key \
    --query 'KeyMaterial' \
    --output text > my-key.pem
chmod 400 my-key.pem

# Create instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MyServer}]'
```

### 4.2 GCP Compute Engine Instance Creation

**Console:**
1. Compute Engine → VM instances → "Create"
2. Enter name
3. Select region/zone
4. Select machine configuration (e.g., e2-medium)
5. Boot disk (select OS image)
6. Firewall settings (allow HTTP/HTTPS)
7. "Create"

**gcloud CLI:**
```bash
# Create instance
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --tags=http-server,https-server

# SSH keys are automatically managed (OS Login or project metadata)
```

---

## 5. SSH Connection

### 5.1 AWS EC2 SSH Connection

```bash
# Check public IP
aws ec2 describe-instances \
    --instance-ids i-1234567890abcdef0 \
    --query 'Reservations[0].Instances[0].PublicIpAddress'

# SSH connection
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

# Amazon Linux: ec2-user
# Ubuntu: ubuntu
# CentOS: centos
# Debian: admin
```

**EC2 Instance Connect (Browser):**
1. EC2 Console → Select instance
2. Click "Connect" button
3. "EC2 Instance Connect" tab
4. Click "Connect"

### 5.2 GCP SSH Connection

```bash
# SSH with gcloud (automatic key management)
gcloud compute ssh my-instance --zone=asia-northeast3-a

# Check external IP
gcloud compute instances describe my-instance \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Direct SSH (if key manually registered)
ssh -i ~/.ssh/google_compute_engine username@<EXTERNAL_IP>
```

**Browser SSH:**
1. Compute Engine → VM instances
2. Click "SSH" button in instance row
3. Browser terminal opens in new window

---

## 6. User Data / Startup Script

Scripts that run automatically when instances start.

### 6.1 AWS User Data

```bash
#!/bin/bash
# User Data example (Amazon Linux 2023)

# Update packages
dnf update -y

# Install Nginx
dnf install -y nginx
systemctl start nginx
systemctl enable nginx

# Custom page
echo "<h1>Hello from $(hostname)</h1>" > /usr/share/nginx/html/index.html
```

**Specify User Data in CLI:**
```bash
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t3.micro \
    --user-data file://startup.sh \
    ...
```

**Check User Data logs:**
```bash
# Inside instance
cat /var/log/cloud-init-output.log
```

### 6.2 GCP Startup Script

```bash
#!/bin/bash
# Startup Script example (Ubuntu)

# Update packages
apt-get update

# Install Nginx
apt-get install -y nginx
systemctl start nginx
systemctl enable nginx

# Custom page
echo "<h1>Hello from $(hostname)</h1>" > /var/www/html/index.html
```

**Specify Startup Script in CLI:**
```bash
gcloud compute instances create my-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --metadata-from-file=startup-script=startup.sh \
    ...

# Or inline
gcloud compute instances create my-instance \
    --metadata=startup-script='#!/bin/bash
    apt-get update
    apt-get install -y nginx'
```

**Check Startup Script logs:**
```bash
# Inside instance
sudo journalctl -u google-startup-scripts.service
# Or
cat /var/log/syslog | grep startup-script
```

---

## 7. Instance Metadata

Query instance information from inside the instance.

### 7.1 AWS Instance Metadata Service (IMDS)

```bash
# Instance ID
curl http://169.254.169.254/latest/meta-data/instance-id

# Public IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Availability Zone
curl http://169.254.169.254/latest/meta-data/placement/availability-zone

# IAM role credentials
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/<role-name>

# IMDSv2 (recommended - token required)
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
curl -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id
```

### 7.2 GCP Metadata Server

```bash
# Instance name
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name

# External IP
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip

# Zone
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone

# Service account token
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token

# Project ID
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id
```

---

## 8. Instance Management

### 8.1 Instance State Management

**AWS:**
```bash
# Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Reboot instance
aws ec2 reboot-instances --instance-ids i-1234567890abcdef0

# Terminate instance (delete)
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# Check instance status
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# Stop instance
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# Start instance
gcloud compute instances start my-instance --zone=asia-northeast3-a

# Restart instance (reset)
gcloud compute instances reset my-instance --zone=asia-northeast3-a

# Delete instance
gcloud compute instances delete my-instance --zone=asia-northeast3-a

# Check instance status
gcloud compute instances describe my-instance --zone=asia-northeast3-a
```

### 8.2 Change Instance Type

**AWS:**
```bash
# 1. Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# 2. Change instance type
aws ec2 modify-instance-attribute \
    --instance-id i-1234567890abcdef0 \
    --instance-type t3.large

# 3. Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

**GCP:**
```bash
# 1. Stop instance
gcloud compute instances stop my-instance --zone=asia-northeast3-a

# 2. Change machine type
gcloud compute instances set-machine-type my-instance \
    --zone=asia-northeast3-a \
    --machine-type=n2-standard-4

# 3. Start instance
gcloud compute instances start my-instance --zone=asia-northeast3-a
```

---

## 9. Pricing Options

### 9.1 On-Demand vs Reserved vs Spot

| Option | AWS | GCP | Discount | Characteristics |
|------|-----|-----|--------|------|
| **On-Demand** | On-Demand | On-demand | 0% | No commitment, flexible |
| **Reserved** | Reserved/Savings Plans | Committed Use | Up to 72% | 1-3 year commitment |
| **Spot/Preemptible** | Spot Instances | Spot/Preemptible | Up to 90% | Can be interrupted |
| **Auto Discount** | - | Sustained Use | Up to 30% | Automatic monthly usage |

### 9.2 AWS Spot Instance

```bash
# Request spot instance
aws ec2 request-spot-instances \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification '{
        "ImageId": "ami-12345678",
        "InstanceType": "t3.large",
        "KeyName": "my-key"
    }'

# Check spot pricing
aws ec2 describe-spot-price-history \
    --instance-types t3.large \
    --product-descriptions "Linux/UNIX"
```

### 9.3 GCP Preemptible/Spot VM

```bash
# Create Spot VM (successor to Preemptible)
gcloud compute instances create spot-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

# Create Preemptible VM (legacy)
gcloud compute instances create preemptible-instance \
    --zone=asia-northeast3-a \
    --machine-type=e2-medium \
    --preemptible
```

---

## 10. Practice: Deploy Web Server

### 10.1 AWS EC2 Web Server

```bash
# 1. Create security group
aws ec2 create-security-group \
    --group-name web-sg \
    --description "Web server security group"

# 2. Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
    --group-name web-sg \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

# 3. Create EC2 instance (with User Data)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-groups web-sg \
    --user-data '#!/bin/bash
dnf update -y
dnf install -y nginx
systemctl start nginx
echo "<h1>AWS EC2 Web Server</h1>" > /usr/share/nginx/html/index.html' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'
```

### 10.2 GCP Compute Engine Web Server

```bash
# 1. Create firewall rule
gcloud compute firewall-rules create allow-http \
    --allow tcp:80 \
    --target-tags http-server

# 2. Create Compute Engine instance
gcloud compute instances create web-server \
    --zone=asia-northeast3-a \
    --machine-type=e2-micro \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y nginx
echo "<h1>GCP Compute Engine Web Server</h1>" > /var/www/html/index.html'

# 3. Check external IP
gcloud compute instances describe web-server \
    --zone=asia-northeast3-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

---

## 11. Next Steps

- [05_Serverless_Functions.md](./05_Serverless_Functions.md) - Serverless functions
- [08_Block_and_File_Storage.md](./08_Block_and_File_Storage.md) - Block storage (EBS/PD)

---

## References

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [GCP Compute Engine Documentation](https://cloud.google.com/compute/docs)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
- [GCP Machine Types](https://cloud.google.com/compute/docs/machine-types)
