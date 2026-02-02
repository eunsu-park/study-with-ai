# NoSQL 데이터베이스

## 1. NoSQL 개요

### 1.1 NoSQL vs RDBMS

| 항목 | RDBMS | NoSQL |
|------|-------|-------|
| 스키마 | 엄격한 스키마 | 유연한 스키마 |
| 확장성 | 수직 확장 | 수평 확장 |
| 트랜잭션 | ACID | BASE (일부 ACID) |
| 쿼리 | SQL | 다양한 API |
| 사용 사례 | 트랜잭션, 복잡한 관계 | 대용량, 유연한 데이터 |

### 1.2 서비스 비교

| 유형 | AWS | GCP |
|------|-----|-----|
| Key-Value / Document | DynamoDB | Firestore |
| Wide Column | - | Bigtable |
| In-Memory Cache | ElastiCache | Memorystore |
| Document (MongoDB) | DocumentDB | MongoDB Atlas (마켓플레이스) |

---

## 2. AWS DynamoDB

### 2.1 DynamoDB 개요

**특징:**
- 완전 관리형 Key-Value / Document DB
- 밀리초 지연 시간
- 무한 확장
- 서버리스 (온디맨드 용량)

**핵심 개념:**
- **테이블**: 데이터 컨테이너
- **항목 (Item)**: 레코드
- **속성 (Attribute)**: 필드
- **Primary Key**: 파티션 키 + (선택적) 정렬 키

### 2.2 테이블 생성

```bash
# 테이블 생성 (파티션 키만)
aws dynamodb create-table \
    --table-name Users \
    --attribute-definitions \
        AttributeName=userId,AttributeType=S \
    --key-schema \
        AttributeName=userId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# 테이블 생성 (파티션 키 + 정렬 키)
aws dynamodb create-table \
    --table-name Orders \
    --attribute-definitions \
        AttributeName=customerId,AttributeType=S \
        AttributeName=orderId,AttributeType=S \
    --key-schema \
        AttributeName=customerId,KeyType=HASH \
        AttributeName=orderId,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST

# 테이블 목록
aws dynamodb list-tables

# 테이블 정보
aws dynamodb describe-table --table-name Users
```

### 2.3 CRUD 작업

```bash
# 항목 추가 (PutItem)
aws dynamodb put-item \
    --table-name Users \
    --item '{
        "userId": {"S": "user-001"},
        "name": {"S": "John Doe"},
        "email": {"S": "john@example.com"},
        "age": {"N": "30"}
    }'

# 항목 조회 (GetItem)
aws dynamodb get-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# 항목 업데이트 (UpdateItem)
aws dynamodb update-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}' \
    --update-expression "SET age = :newAge" \
    --expression-attribute-values '{":newAge": {"N": "31"}}'

# 항목 삭제 (DeleteItem)
aws dynamodb delete-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# 스캔 (전체 테이블)
aws dynamodb scan --table-name Users

# 쿼리 (파티션 키 기반)
aws dynamodb query \
    --table-name Orders \
    --key-condition-expression "customerId = :cid" \
    --expression-attribute-values '{":cid": {"S": "customer-001"}}'
```

### 2.4 Python SDK (boto3)

```python
import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')

# 항목 추가
table.put_item(Item={
    'userId': 'user-002',
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# 항목 조회
response = table.get_item(Key={'userId': 'user-002'})
item = response.get('Item')

# 쿼리 (GSI 사용 시)
response = table.query(
    IndexName='email-index',
    KeyConditionExpression='email = :email',
    ExpressionAttributeValues={':email': 'jane@example.com'}
)

# 배치 쓰기
with table.batch_writer() as batch:
    for i in range(100):
        batch.put_item(Item={'userId': f'user-{i}', 'name': f'User {i}'})
```

### 2.5 글로벌 보조 인덱스 (GSI)

```bash
# GSI 추가
aws dynamodb update-table \
    --table-name Users \
    --attribute-definitions \
        AttributeName=email,AttributeType=S \
    --global-secondary-index-updates '[
        {
            "Create": {
                "IndexName": "email-index",
                "KeySchema": [{"AttributeName": "email", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"}
            }
        }
    ]'
```

### 2.6 DynamoDB Streams

변경 데이터 캡처 (CDC)를 위한 스트림입니다.

```bash
# 스트림 활성화
aws dynamodb update-table \
    --table-name Users \
    --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES

# Lambda 트리거 연결
aws lambda create-event-source-mapping \
    --function-name process-dynamodb \
    --event-source-arn arn:aws:dynamodb:...:table/Users/stream/xxx \
    --starting-position LATEST
```

---

## 3. GCP Firestore

### 3.1 Firestore 개요

**특징:**
- 문서 기반 NoSQL DB
- 실시간 동기화
- 오프라인 지원
- 자동 확장

**핵심 개념:**
- **컬렉션**: 문서 그룹
- **문서**: JSON과 유사한 데이터
- **하위 컬렉션**: 계층 구조

### 3.2 Firestore 설정

```bash
# Firestore API 활성화
gcloud services enable firestore.googleapis.com

# 데이터베이스 생성 (Native 모드)
gcloud firestore databases create \
    --location=asia-northeast3 \
    --type=firestore-native
```

### 3.3 Python SDK

```python
from google.cloud import firestore

db = firestore.Client()

# 문서 추가 (자동 ID)
doc_ref = db.collection('users').add({
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
})

# 문서 추가/업데이트 (지정 ID)
db.collection('users').document('user-001').set({
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# 문서 조회
doc = db.collection('users').document('user-001').get()
if doc.exists:
    print(doc.to_dict())

# 부분 업데이트
db.collection('users').document('user-001').update({
    'age': 26
})

# 문서 삭제
db.collection('users').document('user-001').delete()

# 쿼리
users = db.collection('users').where('age', '>=', 25).stream()
for user in users:
    print(f'{user.id} => {user.to_dict()}')

# 복합 쿼리 (인덱스 필요)
users = db.collection('users') \
    .where('age', '>=', 25) \
    .order_by('age') \
    .limit(10) \
    .stream()
```

### 3.4 실시간 리스너

```python
# 문서 변경 감지
def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        print(f'Received document snapshot: {doc.id}')

doc_ref = db.collection('users').document('user-001')
doc_watch = doc_ref.on_snapshot(on_snapshot)

# 컬렉션 변경 감지
col_ref = db.collection('users')
col_watch = col_ref.on_snapshot(on_snapshot)
```

### 3.5 보안 규칙

```javascript
// firestore.rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // 인증된 사용자만 자신의 문서 접근
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }

    // 공개 읽기
    match /public/{document=**} {
      allow read: if true;
      allow write: if request.auth != null;
    }
  }
}
```

```bash
# 보안 규칙 배포
firebase deploy --only firestore:rules
```

---

## 4. 인메모리 캐시

### 4.1 AWS ElastiCache

**지원 엔진:**
- Redis
- Memcached

```bash
# Redis 클러스터 생성
aws elasticache create-cache-cluster \
    --cache-cluster-id my-redis \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 1 \
    --cache-subnet-group-name my-subnet-group \
    --security-group-ids sg-12345678

# 복제 그룹 생성 (고가용성)
aws elasticache create-replication-group \
    --replication-group-id my-redis-cluster \
    --replication-group-description "Redis cluster" \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-node-groups 1 \
    --replicas-per-node-group 1 \
    --automatic-failover-enabled \
    --cache-subnet-group-name my-subnet-group

# 엔드포인트 확인
aws elasticache describe-cache-clusters \
    --cache-cluster-id my-redis \
    --show-cache-node-info
```

**Python 연결:**
```python
import redis

# 단일 노드
r = redis.Redis(
    host='my-redis.xxx.cache.amazonaws.com',
    port=6379,
    decode_responses=True
)

# SET/GET
r.set('key', 'value')
value = r.get('key')

# 해시
r.hset('user:1000', mapping={'name': 'John', 'email': 'john@example.com'})
user = r.hgetall('user:1000')

# TTL
r.setex('session:abc', 3600, 'user_data')
```

### 4.2 GCP Memorystore

**지원 엔진:**
- Redis
- Memcached

```bash
# Redis 인스턴스 생성
gcloud redis instances create my-redis \
    --region=asia-northeast3 \
    --tier=BASIC \
    --size=1 \
    --redis-version=redis_6_x

# 인스턴스 정보 확인
gcloud redis instances describe my-redis \
    --region=asia-northeast3

# 연결 정보 (호스트/포트)
gcloud redis instances describe my-redis \
    --region=asia-northeast3 \
    --format='value(host,port)'
```

**연결:**
```python
import redis

# Memorystore Redis (Private IP)
r = redis.Redis(
    host='10.0.0.3',  # Private IP
    port=6379,
    decode_responses=True
)

r.set('hello', 'world')
print(r.get('hello'))
```

---

## 5. 용량 모드

### 5.1 DynamoDB 용량 모드

| 모드 | 특징 | 적합한 경우 |
|------|------|-----------|
| **온디맨드** | 자동 확장, 요청당 과금 | 트래픽 예측 불가 |
| **프로비저닝** | 용량 사전 지정 | 안정적 트래픽 |

```bash
# 온디맨드 모드
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PAY_PER_REQUEST

# 프로비저닝 모드
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PROVISIONED \
    --provisioned-throughput ReadCapacityUnits=100,WriteCapacityUnits=100

# Auto Scaling 설정
aws application-autoscaling register-scalable-target \
    --service-namespace dynamodb \
    --resource-id "table/Users" \
    --scalable-dimension "dynamodb:table:ReadCapacityUnits" \
    --min-capacity 5 \
    --max-capacity 1000
```

### 5.2 Firestore 용량

Firestore는 완전 서버리스로 자동 확장됩니다.

**과금:**
- 문서 읽기: $0.06 / 100,000
- 문서 쓰기: $0.18 / 100,000
- 문서 삭제: $0.02 / 100,000
- 스토리지: $0.18 / GB / 월

---

## 6. 비용 비교

### 6.1 DynamoDB

| 항목 | 온디맨드 | 프로비저닝 |
|------|---------|-----------|
| 읽기 | $0.25 / 100만 RRU | $0.00013 / RCU / 시간 |
| 쓰기 | $1.25 / 100만 WRU | $0.00065 / WCU / 시간 |
| 스토리지 | $0.25 / GB / 월 | $0.25 / GB / 월 |

### 6.2 Firestore

| 항목 | 비용 |
|------|------|
| 문서 읽기 | $0.06 / 100,000 |
| 문서 쓰기 | $0.18 / 100,000 |
| 스토리지 | $0.18 / GB / 월 |

### 6.3 ElastiCache / Memorystore

| 서비스 | 노드 타입 | 시간당 비용 |
|--------|----------|------------|
| ElastiCache | cache.t3.micro | ~$0.02 |
| ElastiCache | cache.r5.large | ~$0.20 |
| Memorystore | 1GB Basic | ~$0.05 |
| Memorystore | 1GB Standard (HA) | ~$0.10 |

---

## 7. 사용 사례별 선택

| 사용 사례 | 권장 서비스 |
|----------|-----------|
| 세션 관리 | ElastiCache / Memorystore |
| 사용자 프로필 | DynamoDB / Firestore |
| 실시간 채팅 | Firestore (실시간 동기화) |
| 게임 리더보드 | ElastiCache Redis |
| IoT 데이터 | DynamoDB / Bigtable |
| 장바구니 | DynamoDB / Firestore |
| 캐싱 | ElastiCache / Memorystore |

---

## 8. 설계 패턴

### 8.1 DynamoDB 단일 테이블 설계

```
PK              | SK              | 속성
----------------|-----------------|------------------
USER#123        | USER#123        | name, email
USER#123        | ORDER#001       | product, quantity
USER#123        | ORDER#002       | product, quantity
PRODUCT#A       | PRODUCT#A       | name, price
PRODUCT#A       | REVIEW#001      | rating, comment
```

### 8.2 캐시 패턴

**Cache-Aside (Lazy Loading):**
```python
def get_user(user_id):
    # 캐시 확인
    cached = cache.get(f'user:{user_id}')
    if cached:
        return cached

    # DB에서 조회
    user = db.get_user(user_id)

    # 캐시 저장
    cache.setex(f'user:{user_id}', 3600, user)
    return user
```

**Write-Through:**
```python
def update_user(user_id, data):
    # DB 업데이트
    db.update_user(user_id, data)

    # 캐시 업데이트
    cache.set(f'user:{user_id}', data)
```

---

## 9. 다음 단계

- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - RDB

---

## 참고 자료

- [AWS DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [AWS ElastiCache Documentation](https://docs.aws.amazon.com/elasticache/)
- [GCP Firestore Documentation](https://cloud.google.com/firestore/docs)
- [GCP Memorystore Documentation](https://cloud.google.com/memorystore/docs)
