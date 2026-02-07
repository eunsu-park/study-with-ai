# NoSQL Databases

## 1. NoSQL Overview

### 1.1 NoSQL vs RDBMS

| Item | RDBMS | NoSQL |
|------|-------|-------|
| Schema | Strict schema | Flexible schema |
| Scalability | Vertical scaling | Horizontal scaling |
| Transaction | ACID | BASE (some ACID) |
| Query | SQL | Various APIs |
| Use Cases | Transactions, complex relationships | Large volume, flexible data |

### 1.2 Service Comparison

| Type | AWS | GCP |
|------|-----|-----|
| Key-Value / Document | DynamoDB | Firestore |
| Wide Column | - | Bigtable |
| In-Memory Cache | ElastiCache | Memorystore |
| Document (MongoDB) | DocumentDB | MongoDB Atlas (Marketplace) |

---

## 2. AWS DynamoDB

### 2.1 DynamoDB Overview

**Features:**
- Fully managed Key-Value / Document DB
- Millisecond latency
- Unlimited scaling
- Serverless (on-demand capacity)

**Core Concepts:**
- **Table**: Data container
- **Item**: Record
- **Attribute**: Field
- **Primary Key**: Partition key + (optional) sort key

### 2.2 Table Creation

```bash
# Create table (partition key only)
aws dynamodb create-table \
    --table-name Users \
    --attribute-definitions \
        AttributeName=userId,AttributeType=S \
    --key-schema \
        AttributeName=userId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# Create table (partition key + sort key)
aws dynamodb create-table \
    --table-name Orders \
    --attribute-definitions \
        AttributeName=customerId,AttributeType=S \
        AttributeName=orderId,AttributeType=S \
    --key-schema \
        AttributeName=customerId,KeyType=HASH \
        AttributeName=orderId,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST

# List tables
aws dynamodb list-tables

# Table information
aws dynamodb describe-table --table-name Users
```

### 2.3 CRUD Operations

```bash
# Add item (PutItem)
aws dynamodb put-item \
    --table-name Users \
    --item '{
        "userId": {"S": "user-001"},
        "name": {"S": "John Doe"},
        "email": {"S": "john@example.com"},
        "age": {"N": "30"}
    }'

# Get item (GetItem)
aws dynamodb get-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# Update item (UpdateItem)
aws dynamodb update-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}' \
    --update-expression "SET age = :newAge" \
    --expression-attribute-values '{":newAge": {"N": "31"}}'

# Delete item (DeleteItem)
aws dynamodb delete-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# Scan (entire table)
aws dynamodb scan --table-name Users

# Query (partition key based)
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

# Add item
table.put_item(Item={
    'userId': 'user-002',
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# Get item
response = table.get_item(Key={'userId': 'user-002'})
item = response.get('Item')

# Query (with GSI)
response = table.query(
    IndexName='email-index',
    KeyConditionExpression='email = :email',
    ExpressionAttributeValues={':email': 'jane@example.com'}
)

# Batch write
with table.batch_writer() as batch:
    for i in range(100):
        batch.put_item(Item={'userId': f'user-{i}', 'name': f'User {i}'})
```

### 2.5 Global Secondary Index (GSI)

```bash
# Add GSI
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

Stream for Change Data Capture (CDC).

```bash
# Enable stream
aws dynamodb update-table \
    --table-name Users \
    --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES

# Connect Lambda trigger
aws lambda create-event-source-mapping \
    --function-name process-dynamodb \
    --event-source-arn arn:aws:dynamodb:...:table/Users/stream/xxx \
    --starting-position LATEST
```

---

## 3. GCP Firestore

### 3.1 Firestore Overview

**Features:**
- Document-based NoSQL DB
- Real-time synchronization
- Offline support
- Automatic scaling

**Core Concepts:**
- **Collection**: Group of documents
- **Document**: JSON-like data
- **Subcollection**: Hierarchical structure

### 3.2 Firestore Setup

```bash
# Enable Firestore API
gcloud services enable firestore.googleapis.com

# Create database (Native mode)
gcloud firestore databases create \
    --location=asia-northeast3 \
    --type=firestore-native
```

### 3.3 Python SDK

```python
from google.cloud import firestore

db = firestore.Client()

# Add document (auto ID)
doc_ref = db.collection('users').add({
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
})

# Add/update document (specified ID)
db.collection('users').document('user-001').set({
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# Get document
doc = db.collection('users').document('user-001').get()
if doc.exists:
    print(doc.to_dict())

# Partial update
db.collection('users').document('user-001').update({
    'age': 26
})

# Delete document
db.collection('users').document('user-001').delete()

# Query
users = db.collection('users').where('age', '>=', 25).stream()
for user in users:
    print(f'{user.id} => {user.to_dict()}')

# Complex query (index required)
users = db.collection('users') \
    .where('age', '>=', 25) \
    .order_by('age') \
    .limit(10) \
    .stream()
```

### 3.4 Real-time Listeners

```python
# Document change detection
def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        print(f'Received document snapshot: {doc.id}')

doc_ref = db.collection('users').document('user-001')
doc_watch = doc_ref.on_snapshot(on_snapshot)

# Collection change detection
col_ref = db.collection('users')
col_watch = col_ref.on_snapshot(on_snapshot)
```

### 3.5 Security Rules

```javascript
// firestore.rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Authenticated users can access only their own documents
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }

    // Public read
    match /public/{document=**} {
      allow read: if true;
      allow write: if request.auth != null;
    }
  }
}
```

```bash
# Deploy security rules
firebase deploy --only firestore:rules
```

---

## 4. In-Memory Cache

### 4.1 AWS ElastiCache

**Supported Engines:**
- Redis
- Memcached

```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id my-redis \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 1 \
    --cache-subnet-group-name my-subnet-group \
    --security-group-ids sg-12345678

# Create replication group (high availability)
aws elasticache create-replication-group \
    --replication-group-id my-redis-cluster \
    --replication-group-description "Redis cluster" \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-node-groups 1 \
    --replicas-per-node-group 1 \
    --automatic-failover-enabled \
    --cache-subnet-group-name my-subnet-group

# Check endpoint
aws elasticache describe-cache-clusters \
    --cache-cluster-id my-redis \
    --show-cache-node-info
```

**Python Connection:**
```python
import redis

# Single node
r = redis.Redis(
    host='my-redis.xxx.cache.amazonaws.com',
    port=6379,
    decode_responses=True
)

# SET/GET
r.set('key', 'value')
value = r.get('key')

# Hash
r.hset('user:1000', mapping={'name': 'John', 'email': 'john@example.com'})
user = r.hgetall('user:1000')

# TTL
r.setex('session:abc', 3600, 'user_data')
```

### 4.2 GCP Memorystore

**Supported Engines:**
- Redis
- Memcached

```bash
# Create Redis instance
gcloud redis instances create my-redis \
    --region=asia-northeast3 \
    --tier=BASIC \
    --size=1 \
    --redis-version=redis_6_x

# Instance information
gcloud redis instances describe my-redis \
    --region=asia-northeast3

# Connection information (host/port)
gcloud redis instances describe my-redis \
    --region=asia-northeast3 \
    --format='value(host,port)'
```

**Connection:**
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

## 5. Capacity Modes

### 5.1 DynamoDB Capacity Modes

| Mode | Features | Best For |
|------|------|-----------|
| **On-Demand** | Auto scaling, pay per request | Unpredictable traffic |
| **Provisioned** | Pre-specified capacity | Stable traffic |

```bash
# On-demand mode
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PAY_PER_REQUEST

# Provisioned mode
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PROVISIONED \
    --provisioned-throughput ReadCapacityUnits=100,WriteCapacityUnits=100

# Auto Scaling setup
aws application-autoscaling register-scalable-target \
    --service-namespace dynamodb \
    --resource-id "table/Users" \
    --scalable-dimension "dynamodb:table:ReadCapacityUnits" \
    --min-capacity 5 \
    --max-capacity 1000
```

### 5.2 Firestore Capacity

Firestore is fully serverless with automatic scaling.

**Pricing:**
- Document reads: $0.06 / 100,000
- Document writes: $0.18 / 100,000
- Document deletes: $0.02 / 100,000
- Storage: $0.18 / GB / month

---

## 6. Cost Comparison

### 6.1 DynamoDB

| Item | On-Demand | Provisioned |
|------|---------|-----------|
| Reads | $0.25 / 1M RRU | $0.00013 / RCU / hour |
| Writes | $1.25 / 1M WRU | $0.00065 / WCU / hour |
| Storage | $0.25 / GB / month | $0.25 / GB / month |

### 6.2 Firestore

| Item | Cost |
|------|------|
| Document reads | $0.06 / 100,000 |
| Document writes | $0.18 / 100,000 |
| Storage | $0.18 / GB / month |

### 6.3 ElastiCache / Memorystore

| Service | Node Type | Hourly Cost |
|--------|----------|------------|
| ElastiCache | cache.t3.micro | ~$0.02 |
| ElastiCache | cache.r5.large | ~$0.20 |
| Memorystore | 1GB Basic | ~$0.05 |
| Memorystore | 1GB Standard (HA) | ~$0.10 |

---

## 7. Use Case Selection

| Use Case | Recommended Service |
|----------|-----------|
| Session management | ElastiCache / Memorystore |
| User profiles | DynamoDB / Firestore |
| Real-time chat | Firestore (real-time sync) |
| Game leaderboard | ElastiCache Redis |
| IoT data | DynamoDB / Bigtable |
| Shopping cart | DynamoDB / Firestore |
| Caching | ElastiCache / Memorystore |

---

## 8. Design Patterns

### 8.1 DynamoDB Single Table Design

```
PK              | SK              | Attributes
----------------|-----------------|------------------
USER#123        | USER#123        | name, email
USER#123        | ORDER#001       | product, quantity
USER#123        | ORDER#002       | product, quantity
PRODUCT#A       | PRODUCT#A       | name, price
PRODUCT#A       | REVIEW#001      | rating, comment
```

### 8.2 Cache Patterns

**Cache-Aside (Lazy Loading):**
```python
def get_user(user_id):
    # Check cache
    cached = cache.get(f'user:{user_id}')
    if cached:
        return cached

    # Query from DB
    user = db.get_user(user_id)

    # Save to cache
    cache.setex(f'user:{user_id}', 3600, user)
    return user
```

**Write-Through:**
```python
def update_user(user_id, data):
    # Update DB
    db.update_user(user_id, data)

    # Update cache
    cache.set(f'user:{user_id}', data)
```

---

## 9. Next Steps

- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - RDB

---

## References

- [AWS DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [AWS ElastiCache Documentation](https://docs.aws.amazon.com/elasticache/)
- [GCP Firestore Documentation](https://cloud.google.com/firestore/docs)
- [GCP Memorystore Documentation](https://cloud.google.com/memorystore/docs)
