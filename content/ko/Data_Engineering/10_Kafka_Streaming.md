# Kafka 스트리밍

## 개요

Apache Kafka는 분산 이벤트 스트리밍 플랫폼으로, 실시간 데이터 파이프라인과 스트리밍 애플리케이션 구축에 사용됩니다. 높은 처리량과 내결함성을 제공합니다.

---

## 1. Kafka 개요

### 1.1 Kafka 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                      Kafka Architecture                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Producers                         Consumers                    │
│   ┌─────────┐ ┌─────────┐          ┌─────────┐ ┌─────────┐      │
│   │Producer1│ │Producer2│          │Consumer1│ │Consumer2│      │
│   └────┬────┘ └────┬────┘          └────┬────┘ └────┬────┘      │
│        │           │                    │           │            │
│        └─────┬─────┘                    └─────┬─────┘            │
│              ↓                                ↑                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    Kafka Cluster                          │  │
│   │  ┌──────────────────────────────────────────────────────┐│  │
│   │  │                    Topic: orders                      ││  │
│   │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐       ││  │
│   │  │  │Partition 0 │ │Partition 1 │ │Partition 2 │       ││  │
│   │  │  │ [0,1,2,3]  │ │ [0,1,2]    │ │ [0,1,2,3,4]│       ││  │
│   │  │  └────────────┘ └────────────┘ └────────────┘       ││  │
│   │  └──────────────────────────────────────────────────────┘│  │
│   │                                                          │  │
│   │  Broker 1         Broker 2         Broker 3              │  │
│   │  ┌──────────┐    ┌──────────┐    ┌──────────┐           │  │
│   │  │ P0(L)    │    │ P1(L)    │    │ P2(L)    │           │  │
│   │  │ P1(R)    │    │ P2(R)    │    │ P0(R)    │           │  │
│   │  └──────────┘    └──────────┘    └──────────┘           │  │
│   │                   L=Leader, R=Replica                    │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              ↑                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    ZooKeeper / KRaft                      │  │
│   │             (클러스터 메타데이터 관리)                      │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 개념

| 개념 | 설명 |
|------|------|
| **Broker** | Kafka 서버, 메시지 저장/전달 |
| **Topic** | 메시지 카테고리 (논리적 채널) |
| **Partition** | Topic의 물리적 분할, 병렬 처리 |
| **Producer** | 메시지 발행자 |
| **Consumer** | 메시지 소비자 |
| **Consumer Group** | 협력하여 소비하는 Consumer 그룹 |
| **Offset** | 파티션 내 메시지 위치 |
| **Replication** | 파티션 복제로 내결함성 확보 |

---

## 2. 설치 및 설정

### 2.1 Docker Compose 설정

```yaml
# docker-compose.yaml
version: '3'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"

  # Kafka UI (선택사항)
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
```

```bash
# 실행
docker-compose up -d

# 토픽 생성 (컨테이너 내부에서)
docker exec -it kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic my-topic \
    --partitions 3 \
    --replication-factor 1
```

### 2.2 Python 클라이언트 설치

```bash
# confluent-kafka (권장)
pip install confluent-kafka

# kafka-python (대안)
pip install kafka-python
```

---

## 3. Topic과 Partition

### 3.1 Topic 관리

```bash
# 토픽 생성
kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 6 \
    --replication-factor 3

# 토픽 목록
kafka-topics --list --bootstrap-server localhost:9092

# 토픽 상세 정보
kafka-topics --describe \
    --bootstrap-server localhost:9092 \
    --topic orders

# 토픽 삭제
kafka-topics --delete \
    --bootstrap-server localhost:9092 \
    --topic orders

# 파티션 수 증가 (축소 불가)
kafka-topics --alter \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 12
```

### 3.2 Partition 전략

```python
"""
파티션 선택 전략:
1. Key가 있으면: hash(key) % partitions
2. Key가 없으면: Round-robin

파티션 수 결정 요소:
- 예상 처리량 / 단일 파티션 처리량
- Consumer 수 (파티션 >= Consumer)
- 디스크 I/O 고려
"""

# 파티션 수 권장
"""
- 파티션 당 100MB/s 처리 가정
- 1GB/s 처리 필요 → 최소 10개 파티션
- Consumer 확장성 고려 → 예상 Consumer 수의 2-3배

주의:
- 너무 많은 파티션 → 리더 선출 지연, 메모리 사용 증가
- 너무 적은 파티션 → 병렬성 제한
"""
```

---

## 4. Producer

### 4.1 기본 Producer

```python
from confluent_kafka import Producer
import json

# Producer 설정
config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'my-producer',
    'acks': 'all',  # 모든 replica 확인
}

producer = Producer(config)

# 배달 확인 콜백
def delivery_callback(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}')

# 메시지 전송
def send_message(topic: str, key: str, value: dict):
    producer.produce(
        topic=topic,
        key=key.encode('utf-8'),
        value=json.dumps(value).encode('utf-8'),
        callback=delivery_callback
    )
    # 버퍼 플러시 (비동기 전송 완료 대기)
    producer.flush()

# 사용 예시
send_message(
    topic='orders',
    key='order-123',
    value={
        'order_id': 'order-123',
        'customer_id': 'cust-456',
        'amount': 99.99,
        'timestamp': '2024-01-15T10:30:00Z'
    }
)
```

### 4.2 고성능 Producer

```python
from confluent_kafka import Producer
import json
import time

class HighThroughputProducer:
    """고처리량 Producer"""

    def __init__(self, bootstrap_servers: str):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'high-throughput-producer',

            # 성능 설정
            'acks': '1',                    # 리더만 확인 (빠름)
            'linger.ms': 5,                 # 배치 대기 시간
            'batch.size': 16384,            # 배치 크기 (16KB)
            'buffer.memory': 33554432,      # 버퍼 메모리 (32MB)
            'compression.type': 'snappy',   # 압축

            # 재시도 설정
            'retries': 3,
            'retry.backoff.ms': 100,
        }
        self.producer = Producer(self.config)
        self.message_count = 0

    def send(self, topic: str, key: str, value: dict):
        """비동기 전송"""
        self.producer.produce(
            topic=topic,
            key=key.encode('utf-8') if key else None,
            value=json.dumps(value).encode('utf-8'),
            callback=self._on_delivery
        )
        self.message_count += 1

        # 주기적 폴링 (이벤트 처리)
        if self.message_count % 1000 == 0:
            self.producer.poll(0)

    def _on_delivery(self, err, msg):
        if err:
            print(f'Delivery failed: {err}')

    def flush(self):
        """모든 메시지 전송 완료 대기"""
        self.producer.flush()

    def close(self):
        self.flush()


# 대량 전송 예시
producer = HighThroughputProducer('localhost:9092')

start = time.time()
for i in range(100000):
    producer.send(
        topic='events',
        key=f'key-{i % 100}',
        value={'event_id': i, 'data': 'test'}
    )

producer.flush()
print(f'Sent 100,000 messages in {time.time() - start:.2f} seconds')
```

---

## 5. Consumer

### 5.1 기본 Consumer

```python
from confluent_kafka import Consumer
import json

# Consumer 설정
config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-consumer-group',
    'auto.offset.reset': 'earliest',  # 처음부터 읽기
    'enable.auto.commit': True,
    'auto.commit.interval.ms': 5000,
}

consumer = Consumer(config)

# 토픽 구독
consumer.subscribe(['orders'])

# 메시지 소비
try:
    while True:
        msg = consumer.poll(timeout=1.0)  # 1초 대기

        if msg is None:
            continue

        if msg.error():
            print(f'Consumer error: {msg.error()}')
            continue

        # 메시지 처리
        key = msg.key().decode('utf-8') if msg.key() else None
        value = json.loads(msg.value().decode('utf-8'))

        print(f'Received: topic={msg.topic()}, partition={msg.partition()}, '
              f'offset={msg.offset()}, key={key}, value={value}')

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
```

### 5.2 수동 커밋

```python
from confluent_kafka import Consumer
import json

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'manual-commit-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # 자동 커밋 비활성화
}

consumer = Consumer(config)
consumer.subscribe(['orders'])

def process_message(value: dict) -> bool:
    """메시지 처리 로직"""
    try:
        # 실제 비즈니스 로직
        print(f"Processing: {value}")
        return True
    except Exception as e:
        print(f"Processing failed: {e}")
        return False

try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue
        if msg.error():
            continue

        value = json.loads(msg.value().decode('utf-8'))

        # 처리 성공 시에만 커밋
        if process_message(value):
            consumer.commit(msg)  # 특정 메시지 커밋
            # 또는 consumer.commit() # 현재 오프셋까지 커밋
        else:
            print("Message processing failed, not committing")

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
```

---

## 6. Consumer Group

### 6.1 Consumer Group 개념

```
┌────────────────────────────────────────────────────────────────┐
│                    Consumer Group 동작                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Topic: orders (6 partitions)                                 │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│   │ P0   │ │ P1   │ │ P2   │ │ P3   │ │ P4   │ │ P5   │      │
│   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      │
│      │        │        │        │        │        │           │
│   Consumer Group A (3 consumers)                               │
│      │        │        │        │        │        │           │
│      ↓        ↓        ↓        ↓        ↓        ↓           │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │ Consumer 1  │  │ Consumer 2  │  │ Consumer 3  │          │
│   │  P0, P1     │  │  P2, P3     │  │  P4, P5     │          │
│   └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                                │
│   각 파티션은 그룹 내 하나의 Consumer에만 할당                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 리밸런싱

```python
from confluent_kafka import Consumer

def on_assign(consumer, partitions):
    """파티션 할당 콜백"""
    print(f"Partitions assigned: {[p.partition for p in partitions]}")

def on_revoke(consumer, partitions):
    """파티션 해제 콜백"""
    print(f"Partitions revoked: {[p.partition for p in partitions]}")
    # 처리 중인 메시지 커밋
    consumer.commit()

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest',
    'partition.assignment.strategy': 'cooperative-sticky',  # 점진적 리밸런싱
}

consumer = Consumer(config)
consumer.subscribe(
    ['orders'],
    on_assign=on_assign,
    on_revoke=on_revoke
)
```

### 6.3 Consumer Group 모니터링

```bash
# Consumer Group 목록
kafka-consumer-groups --list --bootstrap-server localhost:9092

# Consumer Group 상세
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group

# 출력 예시:
# GROUP           TOPIC    PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# my-group        orders   0          1500            1550            50
# my-group        orders   1          1200            1200            0

# Lag 모니터링 (처리 지연)
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group \
    --members
```

---

## 7. 실시간 데이터 처리 패턴

### 7.1 이벤트 기반 처리

```python
from confluent_kafka import Consumer, Producer
import json

class EventProcessor:
    """이벤트 기반 처리 파이프라인"""

    def __init__(self, bootstrap_servers: str, group_id: str):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        })
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
        })

    def process_and_forward(
        self,
        source_topic: str,
        target_topic: str,
        transform_func
    ):
        """메시지 처리 후 다른 토픽으로 전달"""
        self.consumer.subscribe([source_topic])

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue

                # 변환
                value = json.loads(msg.value().decode('utf-8'))
                transformed = transform_func(value)

                if transformed:
                    # 다음 토픽으로 전달
                    self.producer.produce(
                        topic=target_topic,
                        key=msg.key(),
                        value=json.dumps(transformed).encode('utf-8')
                    )
                    self.producer.poll(0)

                # 커밋
                self.consumer.commit(msg)

        except KeyboardInterrupt:
            pass
        finally:
            self.producer.flush()
            self.consumer.close()


# 사용 예시: 주문 → 배송 이벤트 변환
def order_to_shipment(order: dict) -> dict:
    """주문 이벤트를 배송 이벤트로 변환"""
    return {
        'shipment_id': f"ship-{order['order_id']}",
        'order_id': order['order_id'],
        'customer_id': order['customer_id'],
        'status': 'pending',
        'created_at': order['timestamp']
    }

processor = EventProcessor('localhost:9092', 'order-processor')
processor.process_and_forward('orders', 'shipments', order_to_shipment)
```

### 7.2 집계 처리 (Windowing)

```python
from confluent_kafka import Consumer
from collections import defaultdict
from datetime import datetime, timedelta
import json
import threading
import time

class WindowedAggregator:
    """시간 윈도우 기반 집계"""

    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        self.windows = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()

    def add(self, key: str, value: int, timestamp: datetime):
        """값 추가"""
        window_start = self._get_window_start(timestamp)
        with self.lock:
            self.windows[window_start][key] += value

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """윈도우 시작 시간 계산"""
        seconds = int(timestamp.timestamp())
        window_start_seconds = (seconds // self.window_size) * self.window_size
        return datetime.fromtimestamp(window_start_seconds)

    def get_and_clear_completed_windows(self) -> dict:
        """완료된 윈도우 결과 반환"""
        current_window = self._get_window_start(datetime.now())
        completed = {}

        with self.lock:
            for window_start, data in list(self.windows.items()):
                if window_start < current_window:
                    completed[window_start] = dict(data)
                    del self.windows[window_start]

        return completed


# 사용 예시: 분당 카테고리별 판매 수 집계
aggregator = WindowedAggregator(window_size_seconds=60)

def process_sales():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'sales-aggregator',
        'auto.offset.reset': 'earliest',
    })
    consumer.subscribe(['sales'])

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg and not msg.error():
            value = json.loads(msg.value().decode('utf-8'))
            aggregator.add(
                key=value['category'],
                value=1,
                timestamp=datetime.fromisoformat(value['timestamp'])
            )

        # 완료된 윈도우 출력
        completed = aggregator.get_and_clear_completed_windows()
        for window, data in completed.items():
            print(f"Window {window}: {data}")
```

---

## 8. Kafka Streams와 대안

### 8.1 Faust (Python Kafka Streams)

```python
import faust

# Faust 앱 생성
app = faust.App(
    'myapp',
    broker='kafka://localhost:9092',
    value_serializer='json',
)

# 토픽 정의
orders_topic = app.topic('orders', value_type=dict)
processed_topic = app.topic('processed_orders', value_type=dict)

# 스트림 처리 에이전트
@app.agent(orders_topic)
async def process_orders(orders):
    async for order in orders:
        # 처리 로직
        processed = {
            **order,
            'processed': True,
            'processed_at': str(datetime.now())
        }
        # 다른 토픽으로 전송
        await processed_topic.send(value=processed)

# 테이블 (상태 저장)
order_counts = app.Table('order_counts', default=int)

@app.agent(orders_topic)
async def count_orders(orders):
    async for order in orders:
        customer_id = order['customer_id']
        order_counts[customer_id] += 1

# 실행: faust -A myapp worker
```

---

## 연습 문제

### 문제 1: Producer/Consumer
주문 이벤트를 생성하는 Producer와 소비하는 Consumer를 작성하세요.

### 문제 2: Consumer Group
3개의 Consumer로 구성된 Consumer Group을 만들고 파티션 할당을 확인하세요.

### 문제 3: 실시간 집계
실시간 판매 이벤트에서 분당 총 매출을 계산하는 스트리밍 애플리케이션을 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **Topic** | 메시지의 논리적 카테고리 |
| **Partition** | Topic의 물리적 분할, 병렬 처리 단위 |
| **Producer** | 메시지 발행자 |
| **Consumer** | 메시지 소비자 |
| **Consumer Group** | 협력적으로 소비하는 Consumer 집합 |
| **Offset** | 파티션 내 메시지 위치 |

---

## 참고 자료

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Kafka Python](https://docs.confluent.io/kafka-clients/python/current/overview.html)
- [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781491936153/)
