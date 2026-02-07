# Kafka Streaming

## Overview

Apache Kafka is a distributed event streaming platform used for building real-time data pipelines and streaming applications. It provides high throughput and fault tolerance.

---

## 1. Kafka Overview

### 1.1 Kafka Architecture

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
│   │             (Cluster metadata management)                 │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Broker** | Kafka server, stores/delivers messages |
| **Topic** | Message category (logical channel) |
| **Partition** | Physical division of topic, parallel processing |
| **Producer** | Message publisher |
| **Consumer** | Message consumer |
| **Consumer Group** | Group of consumers working cooperatively |
| **Offset** | Message position within partition |
| **Replication** | Partition replication for fault tolerance |

---

## 2. Installation and Configuration

### 2.1 Docker Compose Configuration

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

  # Kafka UI (optional)
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
```

```bash
# Run
docker-compose up -d

# Create topic (inside container)
docker exec -it kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic my-topic \
    --partitions 3 \
    --replication-factor 1
```

### 2.2 Python Client Installation

```bash
# confluent-kafka (recommended)
pip install confluent-kafka

# kafka-python (alternative)
pip install kafka-python
```

---

## 3. Topics and Partitions

### 3.1 Topic Management

```bash
# Create topic
kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 6 \
    --replication-factor 3

# List topics
kafka-topics --list --bootstrap-server localhost:9092

# Describe topic
kafka-topics --describe \
    --bootstrap-server localhost:9092 \
    --topic orders

# Delete topic
kafka-topics --delete \
    --bootstrap-server localhost:9092 \
    --topic orders

# Increase partitions (cannot decrease)
kafka-topics --alter \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 12
```

### 3.2 Partition Strategy

```python
"""
Partition selection strategy:
1. With key: hash(key) % partitions
2. Without key: Round-robin

Factors for determining partition count:
- Expected throughput / single partition throughput
- Number of consumers (partitions >= consumers)
- Disk I/O considerations
"""

# Recommended partition count
"""
- Assuming 100MB/s per partition
- Need 1GB/s throughput → minimum 10 partitions
- Consider consumer scalability → 2-3x expected consumer count

Cautions:
- Too many partitions → leader election delay, increased memory usage
- Too few partitions → limited parallelism
"""
```

---

## 4. Producer

### 4.1 Basic Producer

```python
from confluent_kafka import Producer
import json

# Producer configuration
config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'my-producer',
    'acks': 'all',  # Verify all replicas
}

producer = Producer(config)

# Delivery confirmation callback
def delivery_callback(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}')

# Send message
def send_message(topic: str, key: str, value: dict):
    producer.produce(
        topic=topic,
        key=key.encode('utf-8'),
        value=json.dumps(value).encode('utf-8'),
        callback=delivery_callback
    )
    # Flush buffer (wait for async send completion)
    producer.flush()

# Usage example
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

### 4.2 High-Performance Producer

```python
from confluent_kafka import Producer
import json
import time

class HighThroughputProducer:
    """High-throughput Producer"""

    def __init__(self, bootstrap_servers: str):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'high-throughput-producer',

            # Performance settings
            'acks': '1',                    # Leader only (faster)
            'linger.ms': 5,                 # Batch wait time
            'batch.size': 16384,            # Batch size (16KB)
            'buffer.memory': 33554432,      # Buffer memory (32MB)
            'compression.type': 'snappy',   # Compression

            # Retry settings
            'retries': 3,
            'retry.backoff.ms': 100,
        }
        self.producer = Producer(self.config)
        self.message_count = 0

    def send(self, topic: str, key: str, value: dict):
        """Async send"""
        self.producer.produce(
            topic=topic,
            key=key.encode('utf-8') if key else None,
            value=json.dumps(value).encode('utf-8'),
            callback=self._on_delivery
        )
        self.message_count += 1

        # Periodic polling (event processing)
        if self.message_count % 1000 == 0:
            self.producer.poll(0)

    def _on_delivery(self, err, msg):
        if err:
            print(f'Delivery failed: {err}')

    def flush(self):
        """Wait for all message delivery"""
        self.producer.flush()

    def close(self):
        self.flush()


# Bulk send example
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

### 5.1 Basic Consumer

```python
from confluent_kafka import Consumer
import json

# Consumer configuration
config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-consumer-group',
    'auto.offset.reset': 'earliest',  # Read from beginning
    'enable.auto.commit': True,
    'auto.commit.interval.ms': 5000,
}

consumer = Consumer(config)

# Subscribe to topic
consumer.subscribe(['orders'])

# Consume messages
try:
    while True:
        msg = consumer.poll(timeout=1.0)  # Wait 1 second

        if msg is None:
            continue

        if msg.error():
            print(f'Consumer error: {msg.error()}')
            continue

        # Process message
        key = msg.key().decode('utf-8') if msg.key() else None
        value = json.loads(msg.value().decode('utf-8'))

        print(f'Received: topic={msg.topic()}, partition={msg.partition()}, '
              f'offset={msg.offset()}, key={key}, value={value}')

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
```

### 5.2 Manual Commit

```python
from confluent_kafka import Consumer
import json

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'manual-commit-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # Disable auto commit
}

consumer = Consumer(config)
consumer.subscribe(['orders'])

def process_message(value: dict) -> bool:
    """Message processing logic"""
    try:
        # Actual business logic
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

        # Commit only on successful processing
        if process_message(value):
            consumer.commit(msg)  # Commit specific message
            # Or consumer.commit() # Commit up to current offset
        else:
            print("Message processing failed, not committing")

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
```

---

## 6. Consumer Groups

### 6.1 Consumer Group Concept

```
┌────────────────────────────────────────────────────────────────┐
│                    Consumer Group Operation                     │
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
│   Each partition assigned to only one consumer in the group    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 Rebalancing

```python
from confluent_kafka import Consumer

def on_assign(consumer, partitions):
    """Partition assignment callback"""
    print(f"Partitions assigned: {[p.partition for p in partitions]}")

def on_revoke(consumer, partitions):
    """Partition revocation callback"""
    print(f"Partitions revoked: {[p.partition for p in partitions]}")
    # Commit messages being processed
    consumer.commit()

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest',
    'partition.assignment.strategy': 'cooperative-sticky',  # Incremental rebalancing
}

consumer = Consumer(config)
consumer.subscribe(
    ['orders'],
    on_assign=on_assign,
    on_revoke=on_revoke
)
```

### 6.3 Consumer Group Monitoring

```bash
# List consumer groups
kafka-consumer-groups --list --bootstrap-server localhost:9092

# Describe consumer group
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group

# Example output:
# GROUP           TOPIC    PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# my-group        orders   0          1500            1550            50
# my-group        orders   1          1200            1200            0

# Monitor lag (processing delay)
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group \
    --members
```

---

## 7. Real-Time Data Processing Patterns

### 7.1 Event-Based Processing

```python
from confluent_kafka import Consumer, Producer
import json

class EventProcessor:
    """Event-based processing pipeline"""

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
        """Process message and forward to another topic"""
        self.consumer.subscribe([source_topic])

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue

                # Transform
                value = json.loads(msg.value().decode('utf-8'))
                transformed = transform_func(value)

                if transformed:
                    # Forward to next topic
                    self.producer.produce(
                        topic=target_topic,
                        key=msg.key(),
                        value=json.dumps(transformed).encode('utf-8')
                    )
                    self.producer.poll(0)

                # Commit
                self.consumer.commit(msg)

        except KeyboardInterrupt:
            pass
        finally:
            self.producer.flush()
            self.consumer.close()


# Usage example: Order → Shipment event transformation
def order_to_shipment(order: dict) -> dict:
    """Transform order event to shipment event"""
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

### 7.2 Aggregation Processing (Windowing)

```python
from confluent_kafka import Consumer
from collections import defaultdict
from datetime import datetime, timedelta
import json
import threading
import time

class WindowedAggregator:
    """Time window-based aggregation"""

    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        self.windows = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()

    def add(self, key: str, value: int, timestamp: datetime):
        """Add value"""
        window_start = self._get_window_start(timestamp)
        with self.lock:
            self.windows[window_start][key] += value

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """Calculate window start time"""
        seconds = int(timestamp.timestamp())
        window_start_seconds = (seconds // self.window_size) * self.window_size
        return datetime.fromtimestamp(window_start_seconds)

    def get_and_clear_completed_windows(self) -> dict:
        """Return completed window results"""
        current_window = self._get_window_start(datetime.now())
        completed = {}

        with self.lock:
            for window_start, data in list(self.windows.items()):
                if window_start < current_window:
                    completed[window_start] = dict(data)
                    del self.windows[window_start]

        return completed


# Usage example: Aggregate sales per category per minute
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

        # Output completed windows
        completed = aggregator.get_and_clear_completed_windows()
        for window, data in completed.items():
            print(f"Window {window}: {data}")
```

---

## 8. Kafka Streams and Alternatives

### 8.1 Faust (Python Kafka Streams)

```python
import faust

# Create Faust app
app = faust.App(
    'myapp',
    broker='kafka://localhost:9092',
    value_serializer='json',
)

# Define topics
orders_topic = app.topic('orders', value_type=dict)
processed_topic = app.topic('processed_orders', value_type=dict)

# Stream processing agent
@app.agent(orders_topic)
async def process_orders(orders):
    async for order in orders:
        # Processing logic
        processed = {
            **order,
            'processed': True,
            'processed_at': str(datetime.now())
        }
        # Send to another topic
        await processed_topic.send(value=processed)

# Table (stateful)
order_counts = app.Table('order_counts', default=int)

@app.agent(orders_topic)
async def count_orders(orders):
    async for order in orders:
        customer_id = order['customer_id']
        order_counts[customer_id] += 1

# Run: faust -A myapp worker
```

---

## Practice Problems

### Problem 1: Producer/Consumer
Write a Producer that generates order events and a Consumer that consumes them.

### Problem 2: Consumer Group
Create a Consumer Group with 3 consumers and verify partition assignment.

### Problem 3: Real-Time Aggregation
Write a streaming application that calculates total sales revenue per minute from real-time sales events.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Topic** | Logical category of messages |
| **Partition** | Physical division of topic, unit of parallel processing |
| **Producer** | Message publisher |
| **Consumer** | Message consumer |
| **Consumer Group** | Set of consumers working cooperatively |
| **Offset** | Message position within partition |

---

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Kafka Python](https://docs.confluent.io/kafka-clients/python/current/overview.html)
- [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781491936153/)
