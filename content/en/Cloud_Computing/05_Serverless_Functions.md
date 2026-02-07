# Serverless Functions (Lambda / Cloud Functions)

## 1. Serverless Overview

### 1.1 What is Serverless?

Serverless is a computing model that executes code without server management.

**Characteristics:**
- No server provisioning/management required
- Automatic scaling
- Pay only for what you use (execution time + requests)
- Event-driven execution

### 1.2 Service Comparison

| Item | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| Runtime | Node.js, Python, Java, Go, Ruby, .NET, Custom | Node.js, Python, Go, Java, Ruby, PHP, .NET |
| Memory | 128MB ~ 10GB | 128MB ~ 32GB |
| Max Execution Time | 15 minutes | 9 minutes (1st gen) / 60 minutes (2nd gen) |
| Concurrent Executions | 1000 (default, increasable) | Unlimited (default) |
| Triggers | API Gateway, S3, DynamoDB, SNS, etc. | HTTP, Pub/Sub, Cloud Storage, etc. |
| Container Support | Supported (Container Image) | 2nd gen only |

---

## 2. Cold Start

### 2.1 What is Cold Start?

Latency that occurs when a function is first invoked or wakes from idle state.

```
Request → [Cold Start] → Create Container → Initialize Runtime → Load Code → Execute Handler
          ~100ms-seconds                                                      ~ms-seconds

Request → [Warm Start] → Execute Handler
          ~ms
```

### 2.2 Cold Start Mitigation Strategies

| Strategy | AWS Lambda | GCP Cloud Functions |
|------|-----------|-------------------|
| **Provisioned Concurrency** | Supported (paid) | - |
| **Minimum Instances** | - | min-instances in 2nd gen |
| **Lightweight Runtime** | Python, Node.js recommended | Python, Node.js recommended |
| **Minimize Packages** | Remove unnecessary dependencies | Remove unnecessary dependencies |
| **Continuous Invocation** | Warm-up with CloudWatch Events | Warm-up with Cloud Scheduler |

---

## 3. AWS Lambda

### 3.1 Function Creation (Console)

1. Lambda console → "Create function"
2. Select "Author from scratch"
3. Enter function name
4. Select runtime (e.g., Python 3.12)
5. Select architecture (x86_64 or arm64)
6. "Create function"

### 3.2 Function Code (Python)

```python
import json

def lambda_handler(event, context):
    """
    event: Data passed from trigger
    context: Runtime information (function name, memory, remaining time, etc.)
    """
    # Log event
    print(f"Event: {json.dumps(event)}")

    # Business logic
    name = event.get('name', 'World')
    message = f"Hello, {name}!"

    # Return response
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

### 3.3 Function Creation (AWS CLI)

```bash
# 1. Package code
zip function.zip lambda_function.py

# 2. Create IAM role (Lambda execution role)
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

# 3. Attach basic policy
aws iam attach-role-policy \
    --role-name lambda-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# 4. Create Lambda function
aws lambda create-function \
    --function-name my-function \
    --runtime python3.12 \
    --handler lambda_function.lambda_handler \
    --role arn:aws:iam::123456789012:role/lambda-execution-role \
    --zip-file fileb://function.zip

# 5. Test function
aws lambda invoke \
    --function-name my-function \
    --payload '{"name": "Claude"}' \
    --cli-binary-format raw-in-base64-out \
    output.json

cat output.json
```

### 3.4 API Gateway Integration

```bash
# 1. Create REST API
aws apigateway create-rest-api \
    --name my-api \
    --endpoint-configuration types=REGIONAL

# 2. Integrate with Lambda (easier through Console)
# Lambda console → Function → Add trigger → Select API Gateway
```

---

## 4. GCP Cloud Functions

### 4.1 Function Creation (Console)

1. Cloud Functions console → "Create Function"
2. Select environment (1st gen or 2nd gen)
3. Enter function name
4. Select region
5. Select trigger type (HTTP, Pub/Sub, etc.)
6. Select runtime (e.g., Python 3.12)
7. Write code and "Deploy"

### 4.2 Function Code (Python)

**HTTP Trigger:**
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

**Pub/Sub Trigger:**
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

### 4.3 Function Deployment (gcloud CLI)

```bash
# 1. Project structure
# my-function/
# ├── main.py
# └── requirements.txt

# requirements.txt
# functions-framework==3.*

# 2. Deploy HTTP function
gcloud functions deploy hello-http \
    --gen2 \
    --region=asia-northeast3 \
    --runtime=python312 \
    --trigger-http \
    --allow-unauthenticated \
    --entry-point=hello_http \
    --source=.

# 3. Check function URL
gcloud functions describe hello-http \
    --region=asia-northeast3 \
    --format='value(url)'

# 4. Test function
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"name": "Claude"}' \
    https://asia-northeast3-PROJECT_ID.cloudfunctions.net/hello-http
```

### 4.4 Cloud Storage Trigger

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
# Deploy Cloud Storage trigger
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

## 5. Trigger Type Comparison

### 5.1 AWS Lambda Triggers

| Trigger | Description | Examples |
|--------|------|------|
| **API Gateway** | HTTP requests | REST API, WebSocket |
| **S3** | Object events | Upload, Delete |
| **DynamoDB Streams** | Table changes | Insert, Modify, Remove |
| **SNS** | Notification messages | Push notifications |
| **SQS** | Queue messages | Async processing |
| **CloudWatch Events** | Schedule, Events | Cron jobs |
| **Kinesis** | Stream data | Real-time analytics |
| **Cognito** | Auth events | Post-registration processing |

### 5.2 GCP Cloud Functions Triggers

| Trigger | Description | Examples |
|--------|------|------|
| **HTTP** | HTTP requests | REST API |
| **Cloud Storage** | Object events | Upload, Delete |
| **Pub/Sub** | Messages | Async processing |
| **Firestore** | Document changes | Insert, Update, Delete |
| **Cloud Scheduler** | Schedule | Cron jobs |
| **Eventarc** | Various GCP events | 2nd gen unified trigger |

---

## 6. Environment Variables and Secret Management

### 6.1 AWS Lambda Environment Variables

```bash
# Set environment variables
aws lambda update-function-configuration \
    --function-name my-function \
    --environment "Variables={DB_HOST=mydb.example.com,DB_PORT=5432}"
```

**Use in code:**
```python
import os

def lambda_handler(event, context):
    db_host = os.environ.get('DB_HOST')
    db_port = os.environ.get('DB_PORT')
    # ...
```

**Secrets Manager Integration:**
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

### 6.2 GCP Cloud Functions Environment Variables

```bash
# Set environment variables
gcloud functions deploy my-function \
    --set-env-vars DB_HOST=mydb.example.com,DB_PORT=5432 \
    ...
```

**Secret Manager Integration:**
```bash
# Reference secret
gcloud functions deploy my-function \
    --set-secrets 'DB_PASSWORD=projects/PROJECT_ID/secrets/db-password:latest' \
    ...
```

**Use in code:**
```python
import os

def hello_http(request):
    db_host = os.environ.get('DB_HOST')
    db_password = os.environ.get('DB_PASSWORD')  # Auto-injected from Secret Manager
    # ...
```

---

## 7. Pricing Comparison

### 7.1 AWS Lambda Pricing

```
Monthly cost = Request cost + Execution time cost

Request cost: $0.20 / 1M requests
Execution time: $0.0000166667 / GB-second (x86)
               $0.0000133334 / GB-second (ARM)

Free tier (always free):
- 1M requests/month
- 400K GB-seconds/month
```

**Example Calculation:**
```
Conditions: 512MB memory, 200ms execution, 1M requests/month

Request cost: (1,000,000 - 1,000,000) × $0.20/1M = $0
Execution time:
  - 0.5GB × 0.2s × 1,000,000 = 100,000 GB-seconds
  - Free: 400,000 GB-seconds
  - Cost: $0 (within free tier)

Total cost: $0/month (using free tier)
```

### 7.2 GCP Cloud Functions Pricing

```
Monthly cost = Invocation cost + Compute time + Network cost

Invocation cost: $0.40 / 1M invocations
Compute time:
  - CPU: $0.0000100 / GHz-second
  - Memory: $0.0000025 / GB-second

Free tier (always free):
- 2M invocations/month
- 400K GB-seconds, 200K GHz-seconds
- 5GB network egress
```

### 7.3 Cost Optimization Tips

1. **Appropriate Memory Allocation**: Test memory ↔ performance trade-off
2. **Use ARM Architecture** (AWS): 20% cheaper
3. **Minimize Provisioned Concurrency**: Only what's needed
4. **Use Async Invocation**: Direct invocation cheaper than API Gateway
5. **Code Optimization**: Reduce execution time

---

## 8. Local Development and Testing

### 8.1 AWS SAM (Serverless Application Model)

```bash
# Install SAM CLI
pip install aws-sam-cli

# Initialize project
sam init

# Local testing
sam local invoke MyFunction --event events/event.json

# Run local API
sam local start-api

# Deploy
sam build
sam deploy --guided
```

**template.yaml example:**
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
# Install Functions Framework
pip install functions-framework

# Run locally
functions-framework --target=hello_http --debug

# Test from another terminal
curl http://localhost:8080
```

---

## 9. Practice: Image Resize Function

### 9.1 AWS Lambda (S3 Trigger)

```python
import boto3
from PIL import Image
import io

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Extract bucket and key from S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download original image
    response = s3.get_object(Bucket=bucket, Key=key)
    image = Image.open(io.BytesIO(response['Body'].read()))

    # Resize
    image.thumbnail((200, 200))

    # Upload thumbnail
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_key = f"thumbnails/{key}"
    s3.put_object(Bucket=bucket, Key=thumb_key, Body=buffer)

    return {'statusCode': 200, 'body': f'Thumbnail created: {thumb_key}'}
```

### 9.2 GCP Cloud Functions (Cloud Storage Trigger)

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

    # Ignore thumbnails folder images
    if file_name.startswith("thumbnails/"):
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download original image
    blob = bucket.blob(file_name)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data))

    # Resize
    image.thumbnail((200, 200))

    # Upload thumbnail
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    thumb_blob = bucket.blob(f"thumbnails/{file_name}")
    thumb_blob.upload_from_file(buffer, content_type='image/jpeg')

    print(f"Thumbnail created: thumbnails/{file_name}")
```

---

## 10. Next Steps

- [06_Container_Services.md](./06_Container_Services.md) - Container services
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - Database integration

---

## References

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [GCP Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [AWS SAM](https://aws.amazon.com/serverless/sam/)
- [Functions Framework](https://github.com/GoogleCloudPlatform/functions-framework)
