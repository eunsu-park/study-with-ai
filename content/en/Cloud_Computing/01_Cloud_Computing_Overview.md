# Cloud Computing Overview

## 1. What is Cloud Computing?

Cloud computing is a service model that provides IT resources (servers, storage, networks, databases, etc.) on-demand through the internet.

### 1.1 Traditional Infrastructure vs Cloud

| Category | On-Premises (Traditional) | Cloud |
|------|------------------|----------|
| **Initial Cost** | High (hardware purchase) | Low (usage-based) |
| **Scalability** | Weeks to months | Minutes to hours |
| **Maintenance** | Self-managed | Provider-managed |
| **Risk** | Wasted unused resources | Pay only for what you use |
| **Responsibility** | Manage all layers directly | Shared responsibility model |

### 1.2 NIST's 5 Essential Characteristics of Cloud

Core characteristics of cloud computing defined by the National Institute of Standards and Technology (NIST):

1. **On-demand Self-service**
   - Users provision resources directly
   - Automated deployment without human intervention

2. **Broad Network Access**
   - Access through standard mechanisms over the network
   - Support for various client platforms

3. **Resource Pooling**
   - Resources shared through multi-tenant model
   - Physical location abstraction

4. **Rapid Elasticity**
   - Automatic scaling up/down based on demand
   - Resources appear unlimited to users

5. **Measured Service**
   - Usage monitoring and reporting
   - Transparent billing

---

## 2. Service Models: IaaS, PaaS, SaaS

### 2.1 Concept Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                        SaaS                                 │
│  (Software as a Service)                                    │
│  Examples: Gmail, Salesforce, Slack                         │
├─────────────────────────────────────────────────────────────┤
│                        PaaS                                 │
│  (Platform as a Service)                                    │
│  Examples: Heroku, App Engine, Elastic Beanstalk            │
├─────────────────────────────────────────────────────────────┤
│                        IaaS                                 │
│  (Infrastructure as a Service)                              │
│  Examples: EC2, Compute Engine, Azure VMs                   │
├─────────────────────────────────────────────────────────────┤
│                   Physical Infrastructure                    │
│  Data centers, servers, network equipment                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Responsibility Scope Comparison

| Layer | On-Premises | IaaS | PaaS | SaaS |
|------|-----------|------|------|------|
| Application | Customer | Customer | Customer | **Provider** |
| Data | Customer | Customer | Customer | Customer* |
| Runtime | Customer | Customer | **Provider** | **Provider** |
| Middleware | Customer | Customer | **Provider** | **Provider** |
| OS | Customer | Customer | **Provider** | **Provider** |
| Virtualization | Customer | **Provider** | **Provider** | **Provider** |
| Servers | Customer | **Provider** | **Provider** | **Provider** |
| Storage | Customer | **Provider** | **Provider** | **Provider** |
| Networking | Customer | **Provider** | **Provider** | **Provider** |

*Data management responsibility remains with the customer even in SaaS

### 2.3 Use Cases

**IaaS Best For:**
- When complete infrastructure control is needed
- Legacy application migration
- Development/test environments
- High-performance computing (HPC)

**PaaS Best For:**
- Rapid application development
- Microservices architecture
- API development
- Minimizing infrastructure management burden

**SaaS Best For:**
- Email and collaboration tools
- CRM, ERP systems
- Need for ready-to-use solutions

---

## 3. AWS vs GCP Comparison

### 3.1 Market Positioning

| Item | AWS | GCP |
|------|-----|-----|
| **Launch** | 2006 | 2008 |
| **Market Share** | ~32% (1st) | ~10% (3rd) |
| **Strengths** | Service diversity, ecosystem | Data analytics, ML/AI, pricing |
| **Service Count** | 200+ | 100+ |
| **Global Regions** | 30+ | 35+ |

### 3.2 Core Service Mapping

| Category | AWS | GCP |
|----------|-----|-----|
| **Virtual Machines** | EC2 | Compute Engine |
| **Serverless Functions** | Lambda | Cloud Functions |
| **Container Orchestration** | EKS | GKE |
| **Serverless Containers** | Fargate | Cloud Run |
| **Object Storage** | S3 | Cloud Storage |
| **Block Storage** | EBS | Persistent Disk |
| **Managed RDB** | RDS, Aurora | Cloud SQL, Spanner |
| **NoSQL (Key-Value)** | DynamoDB | Firestore |
| **Cache** | ElastiCache | Memorystore |
| **DNS** | Route 53 | Cloud DNS |
| **CDN** | CloudFront | Cloud CDN |
| **Load Balancer** | ELB (ALB/NLB) | Cloud Load Balancing |
| **VPC** | VPC | VPC |
| **IAM** | IAM | IAM |
| **Key Management** | KMS | Cloud KMS |
| **Secret Management** | Secrets Manager | Secret Manager |
| **Monitoring** | CloudWatch | Cloud Monitoring |
| **Logging** | CloudWatch Logs | Cloud Logging |
| **IaC** | CloudFormation | Deployment Manager |
| **CLI** | AWS CLI | gcloud CLI |

---

## 4. Pricing Model

### 4.1 Pricing Principles

Both platforms follow the **Pay-as-you-go** principle.

```
Total Cost = Computing + Storage + Network + Additional Services
```

### 4.2 Computing Pricing Options

| Option | AWS | GCP | Features |
|------|-----|-----|------|
| **On-Demand** | On-Demand | On-demand | Hourly/per-second, no commitment |
| **Reserved** | Reserved Instances | Committed Use | 1-3 year commitment, up to 72% discount |
| **Spot/Preemptible** | Spot Instances | Preemptible/Spot VMs | Up to 90% discount, can be interrupted |
| **Auto Discount** | - | Sustained Use | Automatic discount based on monthly usage |

### 4.3 Data Transfer Costs

```
┌─────────────────────────────────────────────────────────┐
│                        Cloud                            │
│                                                         │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐  │
│   │ Inbound │   Free   │  Same   │   Free   │Outbound│  │
│   │(Out→In) │ ────→   │ Region  │ ────→   │(In→Out)│  │
│   └─────────┘         └─────────┘         └─────────┘  │
│       Free             Free/Cheap           Charged      │
└─────────────────────────────────────────────────────────┘
```

- **Inbound**: Generally free
- **Within Same Region**: Free or inexpensive
- **Outbound**: Charged per GB (after monthly free quota)

### 4.4 Free Tier

**AWS Free Tier:**
- 12 months free: t2.micro EC2 (750 hours/month), 5GB S3, 750 hours RDS
- Always free: Lambda 1M requests/month, DynamoDB 25GB

**GCP Free Tier:**
- $300 credit for 90 days (new accounts)
- Always Free: e2-micro VM, 5GB Cloud Storage, Cloud Functions 2M invocations/month

---

## 5. Shared Responsibility Model

Cloud security responsibilities are shared between provider and customer.

### 5.1 Responsibility Distribution

```
┌────────────────────────────────────────────────────────────┐
│                Customer Responsibility (IN the cloud)       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Customer data                                      │  │
│  │  • Platform, applications, IAM                        │  │
│  │  • Operating system, network, firewall configuration  │  │
│  │  • Client-side data encryption                        │  │
│  │  • Server-side encryption (file system/data)          │  │
│  │  • Network traffic protection (encryption, integrity, │  │
│  │    authentication)                                    │  │
│  └──────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────┤
│                Provider Responsibility (OF the cloud)       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Global infrastructure (regions, AZs, edge          │  │
│  │    locations)                                         │  │
│  │  • Hardware (compute, storage, networking)            │  │
│  │  • Software (host OS, virtualization)                 │  │
│  │  • Physical security (data centers)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Responsibility by Service Type

| Service Type | Customer Responsibility | Provider Responsibility |
|------------|----------|------------|
| **IaaS (EC2)** | OS to application | Hardware, virtualization |
| **PaaS (Lambda)** | Code, data | Runtime, OS, infrastructure |
| **SaaS** | Data, access management | Almost everything |

---

## 6. Cloud Architecture Principles

### 6.1 Well-Architected Framework

Both AWS and GCP present similar design principles:

| Principle | Description |
|------|------|
| **Operational Excellence** | System execution and monitoring, continuous improvement |
| **Security** | Protecting data, systems, and assets |
| **Reliability** | Failure recovery, responding to demand changes |
| **Performance Efficiency** | Efficient resource usage, technology selection |
| **Cost Optimization** | Eliminating unnecessary costs, efficient spending |
| **Sustainability** | Minimizing environmental impact (emphasized by GCP) |

### 6.2 Design Best Practices

```
1. Design for Failure
   - Eliminate single points of failure
   - Multi-AZ/region deployment
   - Automatic recovery

2. Loose Coupling
   - Microservices architecture
   - Utilize message queues
   - API-based communication

3. Elasticity
   - Leverage auto-scaling
   - Consider serverless
   - Prepare for unpredictable loads

4. Security by Design
   - Principle of least privilege
   - Apply encryption by default
   - Network isolation
```

---

## 7. Learning Roadmap

### 7.1 Beginner Path (1-2 weeks)

```
[Cloud Concepts] → [Account Creation] → [Console Navigation] → [First VM] → [S3/GCS Practice]
```

### 7.2 Basic Practical Path (1-2 months)

```
[VPC Networking] → [Load Balancer] → [RDS/Cloud SQL] → [IAM Policies] → [Monitoring]
```

### 7.3 Advanced Path (3-6 months)

```
[Container/K8s] → [Serverless] → [Terraform IaC] → [CI/CD] → [Cost Optimization]
```

---

## 8. Next Steps

- [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) - Account creation and initial setup
- [03_Regions_Availability_Zones.md](./03_Regions_Availability_Zones.md) - Understanding global infrastructure

---

## References

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework)
- [NIST Cloud Computing Definition](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-145.pdf)
