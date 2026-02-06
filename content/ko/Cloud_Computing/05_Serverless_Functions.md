# 서버리스 함수 (Lambda / Cloud Functions)

## 1. 서버리스 개요

### 1.1 서버리스란?

서버리스는 서버 관리 없이 코드를 실행하는 컴퓨팅 모델입니다.

**특징:**
- 서버 프로비저닝/관리 불필요
- 자동 확장
- 사용한 만큼만 과금 (실행 시간 + 요청 수)
- 이벤트 기반 실행

### 1.2 서비스 비교

| 항목 | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| 런타임 | Node.js, Python, Java, Go, Ruby, .NET, Custom | Node.js, Python, Go, Java, Ruby, PHP, .NET |
| 메모리 | 128MB ~ 10GB | 128MB ~ 32GB |
| 최대 실행 시간 | 15분 | 9분 (1세대) / 60분 (2세대) |
| 동시 실행 | 1000 (기본, 증가 가능) | 무제한 (기본) |
| 트리거 | API Gateway, S3, DynamoDB, SNS 등 | HTTP, Pub/Sub, Cloud Storage 등 |
| 컨테이너 지원 | 지원 (Container Image) | 2세대만 지원 |

---

## 2. Cold Start

### 2.1 Cold Start란?

함수가 처음 호출되거나 유휴 상태에서 깨어날 때 발생하는 지연입니다.

```
요청 → [Cold Start] → 컨테이너 생성 → 런타임 초기화 → 코드 로드 → 핸들러 실행
        ~100ms-수초                                                    ~수ms-수초

요청 → [Warm Start] → 핸들러 실행
        ~수ms
```

### 2.2 Cold Start 완화 전략

| 전략 | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| **Provisioned Concurrency** | 지원 (유료) | - |
| **최소 인스턴스** | - | 2세대에서 min-instances |
| **경량 런타임** | Python, Node.js 권장 | Python, Node.js 권장 |
| **패키지 최소화** | 불필요한 의존성 제거 | 불필요한 의존성 제거 |
| **지속적 호출** | CloudWatch Events로 warm-up | Cloud Scheduler로 warm-up |

---

## 3. AWS Lambda

### 3.1 함수 생성 (Console)

1. Lambda 콘솔 → "함수 생성"
2. "새로 작성" 선택
3. 함수 이름 입력
4. 런타임 선택 (예: Python 3.12)
5. 아키텍처 선택 (x86_64 또는 arm64)
6. "함수 생성"

### 3.2 함수 코드 (Python)

```python
import json

def lambda_handler(event, context):
    """
    event: 트리거에서 전달된 데이터
    context: 런타임 정보 (함수명, 메모리, 남은 시간 등)
    """
    # 이벤트 로깅
    print(f"Event: {json.dumps(event)}")

    # 비즈니스 로직
    name = event.get('name', 'World')
    message = f"Hello, {name}!"

    # 응답 반환
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            'message': message
        })
    }
```

### 3.3 함수 생성 (AWS CLI)

```bash
# 1. 코드 패키징
zip function.zip lambda_function.py

# 2. IAM 역할 생성 (Lambda 실행 역할)
aws iam create-role \
    --role-name lambda-execution-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# 3. 기본 정책 연결
aws iam attach-role-policy \
    --role-name lambda-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# 4. Lambda 함수 생성
aws lambda create-function \
    --function-name my-function \
    --runtime python3.12 \
    --handler lambda_function.lambda_handler \
    --role arn:aws:iam::123456789012:role/lambda-execution-role \
    --zip-file fileb://function.zip

# 5. 함수 테스트
aws lambda invoke \
    --function-name my-function \
    --payload '{"name": "Claude"}' \
    --cli-binary-format raw-in-base64-out \
    output.json

cat output.json
```

### 3.4 API Gateway 연동

```bash
# 1. REST API 생성
aws apigateway create-rest-api \
    --name my-api \
    --endpoint-configuration types=REGIONAL

# 2. Lambda와 통합 (Console에서 하는 것이 더 쉬움)
# Lambda 콘솔 → 함수 → 트리거 추가 → API Gateway 선택
```

---

## 4. GCP Cloud Functions

### 4.1 함수 생성 (Console)

1. Cloud Functions 콘솔 → "함수 만들기"
2. 환경 선택 (1세대 또는 2세대)
3. 함수 이름 입력
4. 리전 선택
5. 트리거 유형 선택 (HTTP, Pub/Sub 등)
6. 런타임 선택 (예: Python 3.12)
7. 코드 작성 후 "배포"

### 4.2 함수 코드 (Python)

**HTTP 트리거:**
```python
import functions_framework
from flask import jsonify

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request: Flask request object
    Returns:
        Response object
    """
    request_json = request.get_json(silent=True)
    name = 'World'

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request.args and 'name' in request.args:
        name = request.args.get('name')

    return jsonify({
        'message': f'Hello, {name}!'
    })
```

**Pub/Sub 트리거:**
```python
import base64
import functions_framework

@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    """Pub/Sub Cloud Function.
    Args:
        cloud_event: CloudEvent object
    """
    data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    print(f"Received message: {data}")
```

### 4.3 함수 배포 (gcloud CLI)

```bash
# 1. 프로젝트 구조
# my-function/
# ├── main.py
# └── requirements.txt

# requirements.txt
# functions-framework==3.*

# 2. HTTP 함수 배포
gcloud functions deploy hello-http \
    --gen2 \
    --region=asia-northeast3 \
    --runtime=python312 \
    --trigger-http \
    --allow-unauthenticated \
    --entry-point=hello_http \
    --source=.

# 3. 함수 URL 확인
gcloud functions describe hello-http \
    --region=asia-northeast3 \
    --format='value(url)'

# 4. 함수 테스트
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"name": "Claude"}' \
    https://asia-northeast3-PROJECT_ID.cloudfunctions.net/hello-http
```

### 4.4 Cloud Storage 트리거

```python
import functions_framework

@functions_framework.cloud_event
def hello_gcs(cloud_event):
    """Cloud Storage trigger function.
    Args:
        cloud_event: CloudEvent object
    """
    data = cloud_event.data

    bucket = data["bucket"]
    name = data["name"]

    print(f"File uploaded: gs://{bucket}/{name}")
```

```bash
# Cloud Storage 트리거 배포
gcloud functions deploy process-upload \
    --gen2 \
    --region=asia-northeast3 \
    --runtime=python312 \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=my-bucket" \
    --entry-point=hello_gcs \
    --source=.
```

---

## 5. 트리거 유형 비교

### 5.1 AWS Lambda 트리거

| 트리거 | 설명 | 예시 |
|--------|------|------|
| **API Gateway** | HTTP 요청 | REST API, WebSocket |
| **S3** | 객체 이벤트 | 업로드, 삭제 |
| **DynamoDB Streams** | 테이블 변경 | Insert, Modify, Remove |
| **SNS** | 알림 메시지 | 푸시 알림 |
| **SQS** | 큐 메시지 | 비동기 처리 |
| **CloudWatch Events** | 스케줄, 이벤트 | 크론 작업 |
| **Kinesis** | 스트림 데이터 | 실시간 분석 |
| **Cognito** | 인증 이벤트 | 회원가입 후처리 |

### 5.2 GCP Cloud Functions 트리거

| 트리거 | 설명 | 예시 |
|--------|------|------|
| **HTTP** | HTTP 요청 | REST API |
| **Cloud Storage** | 객체 이벤트 | 업로드, 삭제 |
| **Pub/Sub** | 메시지 | 비동기 처리 |
| **Firestore** | 문서 변경 | Insert, Update, Delete |
| **Cloud Scheduler** | 스케줄 | 크론 작업 |
| **Eventarc** | 다양한 GCP 이벤트 | 2세대 통합 트리거 |

---

## 6. 환경 변수 및 비밀 관리

### 6.1 AWS Lambda 환경 변수

```bash
# 환경 변수 설정
aws lambda update-function-configuration \
    --function-name my-function \
    --environment "Variables={DB_HOST=mydb.example.com,DB_PORT=5432}"
```

**코드에서 사용:**
```python
import os

def lambda_handler(event, context):
    db_host = os.environ.get('DB_HOST')
    db_port = os.environ.get('DB_PORT')
    # ...
```

**Secrets Manager 연동:**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

def lambda_handler(event, context):
    secrets = get_secret('my-database-credentials')
    db_password = secrets['password']
    # ...
```

### 6.2 GCP Cloud Functions 환경 변수

```bash
# 환경 변수 설정
gcloud functions deploy my-function \
    --set-env-vars DB_HOST=mydb.example.com,DB_PORT=5432 \
    ...
```

**Secret Manager 연동:**
```bash
# Secret 참조
gcloud functions deploy my-function \
    --set-secrets 'DB_PASSWORD=projects/PROJECT_ID/secrets/db-password:latest' \
    ...
```

**코드에서 사용:**
```python
import os

def hello_http(request):
    db_host = os.environ.get('DB_HOST')
    db_password = os.environ.get('DB_PASSWORD')  # Secret Manager에서 자동 주입
    # ...
```

---

## 7. 과금 비교

### 7.1 AWS Lambda 과금

```
월 비용 = 요청 비용 + 실행 시간 비용

요청 비용: $0.20 / 100만 요청
실행 시간: $0.0000166667 / GB-초 (x86)
         $0.0000133334 / GB-초 (ARM)

무료 티어 (항상 무료):
- 100만 요청/월
- 40만 GB-초/월
```

**예시 계산:**
```
조건: 512MB 메모리, 200ms 실행, 100만 요청/월

요청 비용: (1,000,000 - 1,000,000) × $0.20/1M = $0
실행 시간:
  - 0.5GB × 0.2초 × 1,000,000 = 100,000 GB-초
  - 무료: 400,000 GB-초
  - 비용: $0 (무료 티어 내)

총 비용: $0/월 (무료 티어 활용)
```

### 7.2 GCP Cloud Functions 과금

```
월 비용 = 호출 비용 + 컴퓨팅 시간 + 네트워크 비용

호출 비용: $0.40 / 100만 호출
컴퓨팅 시간:
  - CPU: $0.0000100 / GHz-초
  - 메모리: $0.0000025 / GB-초

무료 티어 (항상 무료):
- 200만 호출/월
- 40만 GB-초, 20만 GHz-초
- 5GB 네트워크 이그레스
```

### 7.3 비용 최적화 팁

1. **적절한 메모리 할당**: 메모리 ↔ 성능 트레이드오프 테스트
2. **ARM 아키텍처 사용** (AWS): 20% 저렴
3. **Provisioned Concurrency 최소화**: 필요한 만큼만
4. **비동기 호출 활용**: API Gateway보다 직접 호출이 저렴
5. **코드 최적화**: 실행 시간 단축

---

## 8. 로컬 개발 및 테스트

### 8.1 AWS SAM (Serverless Application Model)

```bash
# SAM CLI 설치
pip install aws-sam-cli

# 프로젝트 초기화
sam init

# 로컬 테스트
sam local invoke MyFunction --event events/event.json

# 로컬 API 실행
sam local start-api

# 배포
sam build
sam deploy --guided
```

**template.yaml 예시:**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  HelloFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.lambda_handler
      Runtime: python3.12
      Events:
        Api:
          Type: Api
          Properties:
            Path: /hello
            Method: get
```

### 8.2 GCP Functions Framework

```bash
# Functions Framework 설치
pip install functions-framework

# 로컬 실행
functions-framework --target=hello_http --debug

# 다른 터미널에서 테스트
curl http://localhost:8080
```

---

## 9. 실습: 이미지 리사이즈 함수

### 9.1 AWS Lambda (S3 트리거)

```python
import boto3
from PIL import Image
import io

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # S3 이벤트에서 버킷과 키 추출
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # 원본 이미지 다운로드
    response = s3.get_object(Bucket=bucket, Key=key)
    image = Image.open(io.BytesIO(response['Body'].read()))

    # 리사이즈
    image.thumbnail((200, 200))

    # 썸네일 업로드
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_key = f"thumbnails/{key}"
    s3.put_object(Bucket=bucket, Key=thumb_key, Body=buffer)

    return {'statusCode': 200, 'body': f'Thumbnail created: {thumb_key}'}
```

### 9.2 GCP Cloud Functions (Cloud Storage 트리거)

```python
from google.cloud import storage
from PIL import Image
import io
import functions_framework

@functions_framework.cloud_event
def resize_image(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]

    # 썸네일 폴더의 이미지는 무시
    if file_name.startswith("thumbnails/"):
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # 원본 이미지 다운로드
    blob = bucket.blob(file_name)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data))

    # 리사이즈
    image.thumbnail((200, 200))

    # 썸네일 업로드
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_blob = bucket.blob(f"thumbnails/{file_name}")
    thumb_blob.upload_from_file(buffer, content_type='image/jpeg')

    print(f"Thumbnail created: thumbnails/{file_name}")
```

---

## 10. 다음 단계

- [06_Container_Services.md](./06_Container_Services.md) - 컨테이너 서비스
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - 데이터베이스 연동

---

## 참고 자료

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [GCP Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [AWS SAM](https://aws.amazon.com/serverless/sam/)
- [Functions Framework](https://github.com/GoogleCloudPlatform/functions-framework)
