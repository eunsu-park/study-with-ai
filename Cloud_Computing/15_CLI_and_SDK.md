# CLI & SDK

## 1. CLI 개요

### 1.1 AWS CLI vs gcloud CLI

| 항목 | AWS CLI | gcloud CLI |
|------|---------|------------|
| 설치 패키지 | awscli | google-cloud-sdk |
| 구성 명령 | aws configure | gcloud init |
| 프로필 | --profile | --configuration |
| 출력 형식 | json, text, table, yaml | json, text, yaml, csv |

---

## 2. AWS CLI

### 2.1 설치

```bash
# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# pip
pip install awscli

# 버전 확인
aws --version
```

### 2.2 구성

```bash
# 기본 구성
aws configure
# AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region name: ap-northeast-2
# Default output format: json

# 프로필 추가
aws configure --profile production
aws configure --profile development

# 프로필 목록
aws configure list-profiles

# 환경 변수로 설정
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI...
export AWS_DEFAULT_REGION=ap-northeast-2
export AWS_PROFILE=production
```

**~/.aws/credentials:**
```ini
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI...

[production]
aws_access_key_id = AKIAI44QH8DHBEXAMPLE
aws_secret_access_key = je7MtGbClwBF/2Zp9Utk...
```

**~/.aws/config:**
```ini
[default]
region = ap-northeast-2
output = json

[profile production]
region = ap-northeast-1
output = table
```

### 2.3 주요 명령어

```bash
# EC2
aws ec2 describe-instances
aws ec2 run-instances --image-id ami-xxx --instance-type t3.micro
aws ec2 start-instances --instance-ids i-xxx
aws ec2 stop-instances --instance-ids i-xxx
aws ec2 terminate-instances --instance-ids i-xxx

# S3
aws s3 ls
aws s3 cp file.txt s3://bucket/
aws s3 sync ./folder s3://bucket/folder
aws s3 rm s3://bucket/file.txt

# IAM
aws iam list-users
aws iam create-user --user-name john
aws iam attach-user-policy --user-name john --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess

# Lambda
aws lambda list-functions
aws lambda invoke --function-name my-func output.json

# RDS
aws rds describe-db-instances
aws rds create-db-snapshot --db-instance-identifier mydb --db-snapshot-identifier mysnap
```

### 2.4 출력 필터링 (--query)

```bash
# JMESPath 쿼리 사용
aws ec2 describe-instances \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
    --output table

# 특정 태그로 필터
aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text

# 정렬
aws ec2 describe-instances \
    --query 'sort_by(Reservations[].Instances[], &LaunchTime)[*].[InstanceId,LaunchTime]'

# 조건 필터
aws ec2 describe-instances \
    --query 'Reservations[].Instances[?State.Name==`running`].InstanceId'
```

---

## 3. gcloud CLI

### 3.1 설치

```bash
# macOS
brew install --cask google-cloud-sdk

# 또는 직접 설치
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Linux
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# 버전 확인
gcloud --version
```

### 3.2 구성

```bash
# 초기화 (브라우저 인증)
gcloud init

# 로그인
gcloud auth login

# 서비스 계정 인증
gcloud auth activate-service-account --key-file=key.json

# 프로젝트 설정
gcloud config set project PROJECT_ID

# 리전/존 설정
gcloud config set compute/region asia-northeast3
gcloud config set compute/zone asia-northeast3-a

# 현재 구성 확인
gcloud config list

# 구성 프로필
gcloud config configurations create production
gcloud config configurations activate production
gcloud config configurations list
```

### 3.3 주요 명령어

```bash
# Compute Engine
gcloud compute instances list
gcloud compute instances create my-vm --machine-type=e2-medium
gcloud compute instances start my-vm
gcloud compute instances stop my-vm
gcloud compute instances delete my-vm
gcloud compute ssh my-vm

# Cloud Storage
gsutil ls
gsutil cp file.txt gs://bucket/
gsutil rsync -r ./folder gs://bucket/folder
gsutil rm gs://bucket/file.txt

# IAM
gcloud iam service-accounts list
gcloud iam service-accounts create my-sa
gcloud projects add-iam-policy-binding PROJECT --member=user:john@example.com --role=roles/viewer

# Cloud Functions
gcloud functions list
gcloud functions deploy my-func --runtime=python312 --trigger-http

# Cloud SQL
gcloud sql instances list
gcloud sql instances create mydb --database-version=MYSQL_8_0
```

### 3.4 출력 필터링 (--filter, --format)

```bash
# 필터링
gcloud compute instances list \
    --filter="status=RUNNING AND zone:asia-northeast3"

# 출력 형식
gcloud compute instances list \
    --format="table(name,zone.basename(),status,networkInterfaces[0].accessConfigs[0].natIP)"

# JSON 출력
gcloud compute instances describe my-vm --format=json

# 특정 필드만
gcloud compute instances list \
    --format="value(name,networkInterfaces[0].accessConfigs[0].natIP)"

# CSV
gcloud compute instances list \
    --format="csv(name,zone,status)"
```

---

## 4. Python SDK

### 4.1 AWS SDK (boto3)

**설치:**
```bash
pip install boto3
```

**기본 사용:**
```python
import boto3

# 클라이언트 방식
ec2_client = boto3.client('ec2')
response = ec2_client.describe_instances()

# 리소스 방식 (고수준)
ec2 = boto3.resource('ec2')
instances = ec2.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
)
for instance in instances:
    print(instance.id, instance.public_ip_address)
```

**서비스별 예시:**
```python
import boto3

# S3
s3 = boto3.client('s3')
s3.upload_file('file.txt', 'my-bucket', 'file.txt')
s3.download_file('my-bucket', 'file.txt', 'downloaded.txt')

# DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')
table.put_item(Item={'userId': '001', 'name': 'John'})
response = table.get_item(Key={'userId': '001'})

# Lambda
lambda_client = boto3.client('lambda')
response = lambda_client.invoke(
    FunctionName='my-function',
    Payload=json.dumps({'key': 'value'})
)

# SQS
sqs = boto3.client('sqs')
sqs.send_message(
    QueueUrl='https://sqs.../my-queue',
    MessageBody='Hello'
)

# Secrets Manager
secrets = boto3.client('secretsmanager')
secret = secrets.get_secret_value(SecretId='my-secret')
```

### 4.2 GCP SDK (google-cloud)

**설치:**
```bash
pip install google-cloud-storage
pip install google-cloud-compute
pip install google-cloud-firestore
# 필요한 라이브러리별 설치
```

**기본 사용:**
```python
from google.cloud import storage
from google.cloud import compute_v1

# 인증 (서비스 계정)
# export GOOGLE_APPLICATION_CREDENTIALS="key.json"

# Cloud Storage
client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('file.txt')
blob.upload_from_filename('file.txt')
blob.download_to_filename('downloaded.txt')

# Compute Engine
instance_client = compute_v1.InstancesClient()
instances = instance_client.list(project='my-project', zone='asia-northeast3-a')
for instance in instances:
    print(instance.name, instance.status)
```

**서비스별 예시:**
```python
# Firestore
from google.cloud import firestore

db = firestore.Client()
doc_ref = db.collection('users').document('001')
doc_ref.set({'name': 'John', 'age': 30})
doc = doc_ref.get()
print(doc.to_dict())

# Pub/Sub
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('project', 'topic')
publisher.publish(topic_path, b'Hello')

# Secret Manager
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = f"projects/PROJECT/secrets/my-secret/versions/latest"
response = client.access_secret_version(request={"name": name})
secret = response.payload.data.decode("UTF-8")

# BigQuery
from google.cloud import bigquery

client = bigquery.Client()
query = "SELECT * FROM `project.dataset.table` LIMIT 10"
results = client.query(query)
for row in results:
    print(row)
```

---

## 5. 자동화 스크립트 예시

### 5.1 리소스 정리 스크립트

**AWS - 미사용 EBS 볼륨 삭제:**
```python
import boto3

ec2 = boto3.client('ec2')

# 미사용 볼륨 찾기
volumes = ec2.describe_volumes(
    Filters=[{'Name': 'status', 'Values': ['available']}]
)

for vol in volumes['Volumes']:
    vol_id = vol['VolumeId']
    print(f"Deleting unused volume: {vol_id}")
    # ec2.delete_volume(VolumeId=vol_id)  # 주석 해제하여 실제 삭제
```

**GCP - 오래된 스냅샷 삭제:**
```python
from google.cloud import compute_v1
from datetime import datetime, timedelta

client = compute_v1.SnapshotsClient()
project = 'my-project'

snapshots = client.list(project=project)
cutoff = datetime.now() - timedelta(days=30)

for snapshot in snapshots:
    created = datetime.fromisoformat(snapshot.creation_timestamp.replace('Z', '+00:00'))
    if created < cutoff.replace(tzinfo=created.tzinfo):
        print(f"Deleting old snapshot: {snapshot.name}")
        # client.delete(project=project, snapshot=snapshot.name)
```

### 5.2 배포 스크립트

**AWS Lambda 배포:**
```python
import boto3
import zipfile
import os

def deploy_lambda(function_name, source_dir):
    # 코드 압축
    with zipfile.ZipFile('function.zip', 'w') as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                zf.write(os.path.join(root, file))

    # Lambda 업데이트
    lambda_client = boto3.client('lambda')
    with open('function.zip', 'rb') as f:
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=f.read()
        )

    print(f"Deployed {function_name}")

deploy_lambda('my-function', './src')
```

### 5.3 모니터링 스크립트

**AWS 인스턴스 상태 확인:**
```bash
#!/bin/bash
# check-instances.sh

instances=$(aws ec2 describe-instances \
    --filters "Name=tag:Environment,Values=Production" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
    --output text)

while read -r instance_id state; do
    if [ "$state" != "running" ]; then
        echo "WARNING: $instance_id is $state"
        # 알림 전송
        aws sns publish \
            --topic-arn arn:aws:sns:...:alerts \
            --message "Instance $instance_id is $state"
    fi
done <<< "$instances"
```

---

## 6. 페이지네이션 처리

### 6.1 AWS CLI 페이지네이션

```bash
# 자동 페이지네이션
aws s3api list-objects-v2 --bucket my-bucket

# 수동 페이지네이션
aws s3api list-objects-v2 --bucket my-bucket --max-items 100

# 다음 페이지
aws s3api list-objects-v2 --bucket my-bucket --starting-token TOKEN
```

**boto3 Paginator:**
```python
import boto3

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket='my-bucket'):
    for obj in page.get('Contents', []):
        print(obj['Key'])
```

### 6.2 gcloud 페이지네이션

```bash
# 자동 페이지네이션
gcloud compute instances list

# 수동
gcloud compute instances list --limit=100 --page-token=TOKEN
```

**Python:**
```python
from google.cloud import storage

client = storage.Client()
blobs = client.list_blobs('my-bucket')  # 자동 페이지네이션

for blob in blobs:
    print(blob.name)
```

---

## 7. 에러 처리

### 7.1 boto3 에러 처리

```python
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

try:
    s3 = boto3.client('s3')
    s3.head_object(Bucket='my-bucket', Key='file.txt')
except NoCredentialsError:
    print("Credentials not found")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        print("Object not found")
    elif error_code == '403':
        print("Access denied")
    else:
        raise
```

### 7.2 GCP 에러 처리

```python
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden

try:
    client = storage.Client()
    bucket = client.get_bucket('my-bucket')
    blob = bucket.blob('file.txt')
    blob.download_to_filename('file.txt')
except NotFound:
    print("Bucket or object not found")
except Forbidden:
    print("Access denied")
```

---

## 8. 다음 단계

- [16_Infrastructure_as_Code.md](./16_Infrastructure_as_Code.md) - Terraform
- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - 모니터링

---

## 참고 자료

- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/)
- [AWS CLI Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/index.html)
- [gcloud CLI Documentation](https://cloud.google.com/sdk/gcloud/reference)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Google Cloud Python Client](https://googleapis.dev/python/google-api-core/latest/)
