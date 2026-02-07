# Regions and Availability Zones

## 1. Global Infrastructure Overview

Cloud providers deliver services through data centers distributed worldwide.

### 1.1 Infrastructure Hierarchy

```
Global Network
├── Region
│   ├── Availability Zone / Zone
│   │   └── Data Center
│   ├── Availability Zone
│   │   └── Data Center
│   └── Availability Zone
│       └── Data Center
├── Region
│   └── ...
└── Edge Locations (CDN, DNS)
```

### 1.2 AWS vs GCP Terminology Comparison

| Concept | AWS | GCP |
|------|-----|-----|
| Geographic Area | Region | Region |
| Independent Data Center | Availability Zone (AZ) | Zone |
| Local Services | Local Zones, Wavelength | - |
| CDN Edge | Edge Locations | Edge PoPs |
| Private Connection | Direct Connect | Cloud Interconnect |

---

## 2. Regions

### 2.1 Definition

A region is a geographically separated cloud infrastructure area.

**Characteristics:**
- Each region operates independently
- Data replication between regions requires explicit configuration
- Most services are provided on a per-region basis

### 2.2 Major AWS Regions

| Region Code | Location | Recommended from Korea |
|----------|------|--------------|
| ap-northeast-2 | Seoul | ✅ Most recommended |
| ap-northeast-1 | Tokyo | ✅ Alternative |
| ap-northeast-3 | Osaka | Optional |
| ap-southeast-1 | Singapore | Optional |
| us-east-1 | N. Virginia | Global services |
| us-west-2 | Oregon | Cost optimization |
| eu-west-1 | Ireland | European services |

```bash
# Check current region
aws configure get region

# Set region
aws configure set region ap-northeast-2

# List available regions
aws ec2 describe-regions --output table
```

### 2.3 Major GCP Regions

| Region Code | Location | Recommended from Korea |
|----------|------|--------------|
| asia-northeast3 | Seoul | ✅ Most recommended |
| asia-northeast1 | Tokyo | ✅ Alternative |
| asia-northeast2 | Osaka | Optional |
| asia-southeast1 | Singapore | Optional |
| us-central1 | Iowa | Free tier |
| us-east1 | South Carolina | Free tier |
| europe-west1 | Belgium | European services |

```bash
# Check current region
gcloud config get-value compute/region

# Set region
gcloud config set compute/region asia-northeast3

# List available regions
gcloud compute regions list
```

---

## 3. Availability Zones / Zones

### 3.1 Definition

An availability zone is an independent data center group within a region.

**Characteristics:**
- Physically separated locations
- Independent power, cooling, and networking
- Connected by low-latency, high-speed networks
- Failure in one AZ does not affect other AZs

### 3.2 AWS Availability Zones

```
Seoul Region (ap-northeast-2)
├── ap-northeast-2a
├── ap-northeast-2b
├── ap-northeast-2c
└── ap-northeast-2d
```

**AZ Naming Convention:**
- Format: `{region-code}{zone-letter}`
- Examples: `ap-northeast-2a`, `us-east-1b`

```bash
# List availability zones
aws ec2 describe-availability-zones --region ap-northeast-2

# Example output
{
    "AvailabilityZones": [
        {
            "ZoneName": "ap-northeast-2a",
            "State": "available",
            "ZoneType": "availability-zone"
        },
        ...
    ]
}
```

### 3.3 GCP Zones

```
Seoul Region (asia-northeast3)
├── asia-northeast3-a
├── asia-northeast3-b
└── asia-northeast3-c
```

**Zone Naming Convention:**
- Format: `{region-code}-{zone-letter}`
- Examples: `asia-northeast3-a`, `us-central1-f`

```bash
# List zones
gcloud compute zones list --filter="region:asia-northeast3"

# Example output
NAME                 REGION           STATUS
asia-northeast3-a    asia-northeast3  UP
asia-northeast3-b    asia-northeast3  UP
asia-northeast3-c    asia-northeast3  UP
```

---

## 4. Multi-AZ Architecture

### 4.1 High Availability Design

```
┌────────────────────────────────────────────────────────────┐
│                     Region                                  │
│  ┌──────────────────┐    ┌──────────────────┐             │
│  │    AZ-a          │    │    AZ-b          │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │   Web-1    │  │    │  │   Web-2    │  │             │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │   App-1    │  │    │  │   App-2    │  │             │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  │  ┌────────────┐  │    │  ┌────────────┐  │             │
│  │  │ DB-Primary │ │───▶│  │ DB-Standby │  │  (Sync Repl) │
│  │  └────────────┘  │    │  └────────────┘  │             │
│  └──────────────────┘    └──────────────────┘             │
│                                                            │
│  ┌──────────────────────────────────────────┐             │
│  │        Load Balancer (Region-scope)       │             │
│  └──────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Multi-AZ Options by Service

**AWS:**

| Service | Multi-AZ Method |
|--------|--------------|
| EC2 | Distribute with Auto Scaling Group |
| RDS | Enable Multi-AZ option |
| ElastiCache | Place replica in different AZ |
| ELB | Automatic Multi-AZ |
| S3 | Automatic Multi-AZ replication |

**GCP:**

| Service | Multi-Zone Method |
|--------|----------------|
| Compute Engine | Distribute with Instance Group |
| Cloud SQL | Enable high availability option |
| Memorystore | Place replica in different Zone |
| Cloud Load Balancing | Automatic Multi-Zone |
| Cloud Storage | Use Regional class |

---

## 5. Region Selection Criteria

### 5.1 Key Considerations

| Criterion | Description | Recommendation |
|------|------|------|
| **Latency** | Physical distance to users | Region near users |
| **Compliance** | Data residency requirements | Check legal requirements |
| **Service Availability** | Not all services in all regions | Check required services |
| **Cost** | Price differences by region | Compare costs |
| **Disaster Recovery** | DR site distance | Sufficiently distant region |

### 5.2 Latency Testing

**AWS Latency Measurement:**
```bash
# Use CloudPing site
# https://www.cloudping.info/

# Or direct ping test
ping ec2.ap-northeast-2.amazonaws.com
ping ec2.ap-northeast-1.amazonaws.com
```

**GCP Latency Measurement:**
```bash
# GCP Ping test site
# https://gcping.com/

# Or direct measurement
ping asia-northeast3-run.googleapis.com
```

### 5.3 Check Service Availability

**AWS:**
- https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/

**GCP:**
- https://cloud.google.com/about/locations

### 5.4 Cost Comparison (EC2/Compute Engine Example)

| Instance Type | Seoul (AWS) | Virginia (AWS) | Seoul (GCP) | Iowa (GCP) |
|--------------|-----------|---------------|-----------|---------------|
| General 2vCPU/8GB | ~$0.10/hour | ~$0.08/hour | ~$0.09/hour | ~$0.07/hour |

*Prices may vary, check official pricing*

---

## 6. Global/Regional/Zonal Services

### 6.1 AWS Service Scope

| Scope | Service Examples |
|------|-----------|
| **Global** | IAM, Route 53, CloudFront, WAF |
| **Regional** | VPC, S3, Lambda, RDS, EC2 (AMI) |
| **Availability Zone** | EC2 instances, EBS volumes, Subnets |

### 6.2 GCP Service Scope

| Scope | Service Examples |
|------|-----------|
| **Global** | Cloud IAM, Cloud DNS, Cloud CDN, VPC (network) |
| **Regional** | Cloud Storage (Regional), Cloud SQL, Cloud Run |
| **Zonal** | Compute Engine, Persistent Disk |

**GCP VPC Distinction:**
- GCP VPC is a **global** resource (AWS VPC is regional)
- Subnets are regional scope

```
AWS VPC vs GCP VPC

AWS:
├── VPC (Regional) ─── us-east-1
│   ├── Subnet-a (AZ scope) ─── us-east-1a
│   └── Subnet-b (AZ scope) ─── us-east-1b
└── VPC (Separate region) ─── ap-northeast-2
    └── Subnet-a ─── ap-northeast-2a

GCP:
└── VPC (Global)
    ├── Subnet-us (Regional) ─── us-central1
    ├── Subnet-asia (Regional) ─── asia-northeast3
    └── Subnet-eu (Regional) ─── europe-west1
```

---

## 7. Edge Locations

### 7.1 CDN Edge

**AWS CloudFront:**
- 400+ edge locations
- Static content caching
- DDoS protection (AWS Shield)

**GCP Cloud CDN:**
- Leverages Google's global edge network
- Automatic SSL/TLS
- Cloud Armor integration

### 7.2 DNS Edge

**AWS Route 53:**
- Global Anycast DNS
- Latency-based routing
- Geolocation routing

**GCP Cloud DNS:**
- Global Anycast
- 100% availability SLA
- DNSSEC support

---

## 8. Disaster Recovery Strategy

### 8.1 DR Patterns

| Pattern | RTO | RPO | Cost | Description |
|------|-----|-----|------|------|
| **Backup & Restore** | Hours~Days | Hours~Days | Low | Only backups stored in different region |
| **Pilot Light** | Minutes~Hours | Minutes~Hours | Medium | Only core systems on standby |
| **Warm Standby** | Minutes | Minutes | High | Scaled-down environment always running |
| **Active-Active** | Seconds | Near 0 | Very High | All regions running simultaneously |

### 8.2 Cross-Region Replication

**AWS S3 Cross-Region Replication:**
```bash
# Configure S3 bucket replication
aws s3api put-bucket-replication \
    --bucket source-bucket \
    --replication-configuration '{
        "Role": "arn:aws:iam::account-id:role/replication-role",
        "Rules": [{
            "Status": "Enabled",
            "Destination": {
                "Bucket": "arn:aws:s3:::destination-bucket"
            }
        }]
    }'
```

**GCP Cloud Storage Replication:**
```bash
# Use Dual-region or Multi-region bucket
gsutil mb -l asia gs://my-multi-region-bucket

# Or replicate with Storage Transfer Service
gcloud transfer jobs create \
    gs://source-bucket gs://destination-bucket
```

---

## 9. Practice: Query Region/AZ Information

### 9.1 AWS CLI Practice

```bash
# 1. List all regions
aws ec2 describe-regions --query 'Regions[*].RegionName' --output text

# 2. List AZs in Seoul region
aws ec2 describe-availability-zones \
    --region ap-northeast-2 \
    --query 'AvailabilityZones[*].[ZoneName,State]' \
    --output table

# 3. Check service availability by region (SSM parameter)
aws ssm get-parameters-by-path \
    --path /aws/service/global-infrastructure/regions \
    --query 'Parameters[*].Name'
```

### 9.2 GCP gcloud Practice

```bash
# 1. List all regions
gcloud compute regions list --format="value(name)"

# 2. List Zones in Seoul region
gcloud compute zones list \
    --filter="region:asia-northeast3" \
    --format="table(name,status)"

# 3. Check machine types in specific region
gcloud compute machine-types list \
    --filter="zone:asia-northeast3-a" \
    --limit=10
```

---

## 10. Next Steps

- [04_Virtual_Machines.md](./04_Virtual_Machines.md) - Virtual machine creation and management
- [09_Virtual_Private_Cloud.md](./09_Virtual_Private_Cloud.md) - VPC networking

---

## References

- [AWS Global Infrastructure](https://aws.amazon.com/about-aws/global-infrastructure/)
- [GCP Locations](https://cloud.google.com/about/locations)
- [AWS Regions and Availability Zones](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)
- [GCP Regions and Zones](https://cloud.google.com/compute/docs/regions-zones)
