# Infrastructure as Code (Terraform)

## 1. IaC 개요

### 1.1 Infrastructure as Code란?

IaC는 인프라를 코드로 정의하고 관리하는 방식입니다.

**장점:**
- 버전 관리 (Git)
- 재현 가능성
- 자동화
- 문서화
- 협업

### 1.2 IaC 도구 비교

| 도구 | 유형 | 언어 | 멀티 클라우드 |
|------|------|------|-------------|
| **Terraform** | 선언적 | HCL | ✅ |
| CloudFormation | 선언적 | JSON/YAML | AWS만 |
| Deployment Manager | 선언적 | YAML/Jinja | GCP만 |
| Pulumi | 선언적 | Python/TS 등 | ✅ |
| Ansible | 절차적 | YAML | ✅ |

---

## 2. Terraform 기초

### 2.1 설치

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# 버전 확인
terraform version
```

### 2.2 기본 개념

```
┌─────────────────────────────────────────────────────────────┐
│  Terraform 워크플로우                                        │
│                                                             │
│  1. Write    → .tf 파일 작성                                │
│  2. Init     → terraform init (프로바이더 다운로드)         │
│  3. Plan     → terraform plan (변경 사항 미리보기)          │
│  4. Apply    → terraform apply (인프라 적용)                │
│  5. Destroy  → terraform destroy (인프라 삭제)              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 HCL 문법

```hcl
# 프로바이더 설정
provider "aws" {
  region = "ap-northeast-2"
}

# 리소스 정의
resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t3.micro"

  tags = {
    Name = "WebServer"
  }
}

# 변수
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

# 출력
output "public_ip" {
  value = aws_instance.web.public_ip
}

# 로컬 값
locals {
  environment = "production"
  common_tags = {
    Environment = local.environment
    ManagedBy   = "Terraform"
  }
}

# 데이터 소스 (기존 리소스 참조)
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}
```

---

## 3. AWS 인프라 구성

### 3.1 VPC + EC2 예제

```hcl
# main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project}-vpc"
  }
}

# 퍼블릭 서브넷
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project}-public-${count.index + 1}"
  }
}

# 인터넷 게이트웨이
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project}-igw"
  }
}

# 라우팅 테이블
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# 보안 그룹
resource "aws_security_group" "web" {
  name        = "${var.project}-web-sg"
  description = "Web server security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 인스턴스
resource "aws_instance" "web" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.web.id]
  key_name               = var.key_name

  user_data = <<-EOF
    #!/bin/bash
    dnf update -y
    dnf install -y nginx
    systemctl start nginx
    systemctl enable nginx
    echo "<h1>Hello from Terraform!</h1>" > /usr/share/nginx/html/index.html
  EOF

  tags = {
    Name = "${var.project}-web"
  }
}

# 데이터 소스
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}
```

```hcl
# variables.tf

variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-2"
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "myapp"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed for SSH"
  type        = string
  default     = "0.0.0.0/0"
}
```

```hcl
# outputs.tf

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_ip" {
  description = "Web server public IP"
  value       = aws_instance.web.public_ip
}

output "website_url" {
  description = "Website URL"
  value       = "http://${aws_instance.web.public_ip}"
}
```

---

## 4. GCP 인프라 구성

### 4.1 VPC + Compute Engine 예제

```hcl
# main.tf

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC
resource "google_compute_network" "main" {
  name                    = "${var.name_prefix}-vpc"
  auto_create_subnetworks = false
}

# 서브넷
resource "google_compute_subnetwork" "public" {
  name          = "${var.name_prefix}-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.main.id
}

# 방화벽 규칙 - HTTP
resource "google_compute_firewall" "http" {
  name    = "${var.name_prefix}-allow-http"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}

# 방화벽 규칙 - SSH
resource "google_compute_firewall" "ssh" {
  name    = "${var.name_prefix}-allow-ssh"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = [var.ssh_allowed_cidr]
  target_tags   = ["ssh-server"]
}

# Compute Engine 인스턴스
resource "google_compute_instance" "web" {
  name         = "${var.name_prefix}-web"
  machine_type = var.machine_type
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 20
    }
  }

  network_interface {
    network    = google_compute_network.main.name
    subnetwork = google_compute_subnetwork.public.name

    access_config {
      // 외부 IP 할당
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y nginx
    echo "<h1>Hello from Terraform on GCP!</h1>" > /var/www/html/index.html
  EOF

  tags = ["http-server", "ssh-server"]

  labels = {
    environment = var.environment
  }
}
```

```hcl
# variables.tf

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-northeast3"
}

variable "name_prefix" {
  description = "Resource name prefix"
  type        = string
  default     = "myapp"
}

variable "machine_type" {
  description = "Compute Engine machine type"
  type        = string
  default     = "e2-micro"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "ssh_allowed_cidr" {
  description = "CIDR allowed for SSH"
  type        = string
  default     = "0.0.0.0/0"
}
```

```hcl
# outputs.tf

output "instance_ip" {
  description = "Instance external IP"
  value       = google_compute_instance.web.network_interface[0].access_config[0].nat_ip
}

output "website_url" {
  description = "Website URL"
  value       = "http://${google_compute_instance.web.network_interface[0].access_config[0].nat_ip}"
}
```

---

## 5. 상태 관리

### 5.1 원격 상태 저장소

**AWS S3 백엔드:**
```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "ap-northeast-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"  # 상태 잠금
  }
}
```

**GCP Cloud Storage 백엔드:**
```hcl
terraform {
  backend "gcs" {
    bucket = "my-terraform-state"
    prefix = "prod/terraform.tfstate"
  }
}
```

### 5.2 상태 명령어

```bash
# 상태 목록
terraform state list

# 상태 조회
terraform state show aws_instance.web

# 상태에서 리소스 제거 (실제 리소스는 유지)
terraform state rm aws_instance.web

# 상태 이동 (리팩토링)
terraform state mv aws_instance.old aws_instance.new

# 상태 가져오기 (기존 리소스)
terraform import aws_instance.web i-1234567890abcdef0
```

---

## 6. 모듈

### 6.1 모듈 구조

```
modules/
├── vpc/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
└── ec2/
    ├── main.tf
    ├── variables.tf
    └── outputs.tf
```

### 6.2 모듈 정의

```hcl
# modules/vpc/main.tf

resource "aws_vpc" "this" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true

  tags = merge(var.tags, {
    Name = var.name
  })
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnets)
  vpc_id                  = aws_vpc.this.id
  cidr_block              = var.public_subnets[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name = "${var.name}-public-${count.index + 1}"
  })
}
```

```hcl
# modules/vpc/variables.tf

variable "name" {
  type = string
}

variable "cidr_block" {
  type    = string
  default = "10.0.0.0/16"
}

variable "public_subnets" {
  type = list(string)
}

variable "availability_zones" {
  type = list(string)
}

variable "tags" {
  type    = map(string)
  default = {}
}
```

```hcl
# modules/vpc/outputs.tf

output "vpc_id" {
  value = aws_vpc.this.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}
```

### 6.3 모듈 사용

```hcl
# main.tf

module "vpc" {
  source = "./modules/vpc"

  name               = "myapp"
  cidr_block         = "10.0.0.0/16"
  public_subnets     = ["10.0.1.0/24", "10.0.2.0/24"]
  availability_zones = ["ap-northeast-2a", "ap-northeast-2c"]

  tags = {
    Environment = "production"
  }
}

module "ec2" {
  source = "./modules/ec2"

  name          = "myapp-web"
  subnet_id     = module.vpc.public_subnet_ids[0]
  instance_type = "t3.micro"
}
```

---

## 7. 워크스페이스

```bash
# 워크스페이스 목록
terraform workspace list

# 새 워크스페이스 생성
terraform workspace new dev
terraform workspace new prod

# 워크스페이스 전환
terraform workspace select prod

# 현재 워크스페이스
terraform workspace show
```

```hcl
# 워크스페이스별 설정
locals {
  environment = terraform.workspace

  instance_type = {
    dev  = "t3.micro"
    prod = "t3.large"
  }
}

resource "aws_instance" "web" {
  instance_type = local.instance_type[local.environment]
  # ...
}
```

---

## 8. 모범 사례

### 8.1 디렉토리 구조

```
terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   └── prod/
│       ├── main.tf
│       ├── variables.tf
│       └── terraform.tfvars
├── modules/
│   ├── vpc/
│   ├── ec2/
│   └── rds/
└── global/
    └── iam/
```

### 8.2 코드 스타일

```hcl
# 리소스 이름 규칙
resource "aws_instance" "web" { }  # 단수
resource "aws_subnet" "public" { } # 복수 사용 시 count/for_each

# 변수 기본값
variable "instance_type" {
  description = "EC2 instance type"  # 항상 설명 포함
  type        = string
  default     = "t3.micro"
}

# 태그 일관성
locals {
  common_tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
```

### 8.3 보안

```hcl
# 민감한 변수
variable "db_password" {
  type      = string
  sensitive = true
}

# 민감한 출력
output "db_password" {
  value     = var.db_password
  sensitive = true
}
```

---

## 9. CI/CD 통합

### 9.1 GitHub Actions

```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0

      - name: Terraform Init
        run: terraform init

      - name: Terraform Plan
        run: terraform plan -no-color
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main'
        run: terraform apply -auto-approve
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

## 10. 다음 단계

- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - 모니터링
- [Docker/](../Docker/) - Kubernetes IaC

---

## 참고 자료

- [Terraform Documentation](https://www.terraform.io/docs)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
