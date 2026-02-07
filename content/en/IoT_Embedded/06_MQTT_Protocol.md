# 06. MQTT Protocol

This lesson covers MQTT (Message Queuing Telemetry Transport), a lightweight messaging protocol designed for IoT devices. We'll learn MQTT broker setup, publish/subscribe patterns, QoS levels, and Python implementation using paho-mqtt.

---

## 1. MQTT Basics

### 1.1 What is MQTT?

**MQTT (Message Queuing Telemetry Transport)**
- Lightweight publish/subscribe messaging protocol
- Designed for constrained devices and low-bandwidth networks
- Widely used in IoT, home automation, industrial monitoring

**Key Features:**
- Minimal protocol overhead (2-byte header)
- Asynchronous bidirectional communication
- Quality of Service (QoS) levels
- Persistent sessions and retained messages
- Last Will and Testament (LWT) for disconnection handling

### 1.2 MQTT Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Publisher  │────────>│  MQTT Broker │<────────│ Subscriber  │
│  (Sensor)   │ Publish │              │Subscribe│  (Client)   │
└─────────────┘         └──────────────┘         └─────────────┘
                             │      │
                    ┌────────┘      └────────┐
                    ▼                        ▼
              ┌─────────────┐          ┌─────────────┐
              │ Subscriber  │          │ Subscriber  │
              │  (Client)   │          │  (Client)   │
              └─────────────┘          └─────────────┘
```

**Components:**
- **Broker**: Central server that receives messages and routes them to subscribers
- **Publisher**: Client that sends messages to topics
- **Subscriber**: Client that receives messages from topics
- **Topic**: Message routing path (e.g., `home/livingroom/temperature`)

### 1.3 MQTT vs HTTP

| Feature | MQTT | HTTP |
|---------|------|------|
| **Pattern** | Pub/Sub (asynchronous) | Request/Response (synchronous) |
| **Connection** | Persistent | Per-request |
| **Overhead** | Low (2-byte header) | High (HTTP headers) |
| **Bi-directional** | Native | Requires polling/WebSocket |
| **QoS** | 3 levels | None (TCP reliability only) |
| **Best For** | Real-time updates, constrained devices | Web APIs, file transfer |

---

## 2. Mosquitto Broker Setup

### 2.1 Mosquitto Installation

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
```

**Check status:**

```bash
sudo systemctl status mosquitto
```

**Start broker:**

```bash
sudo systemctl start mosquitto
sudo systemctl enable mosquitto  # Auto-start on boot
```

### 2.2 Mosquitto Configuration

**Configuration file:** `/etc/mosquitto/mosquitto.conf`

```conf
# Basic configuration
listener 1883
protocol mqtt

# Allow anonymous connections (disable for production)
allow_anonymous true

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_type all

# Persistence
persistence true
persistence_location /var/lib/mosquitto/
```

**Apply configuration:**

```bash
sudo systemctl restart mosquitto
```

### 2.3 Mosquitto with Authentication

**Create password file:**

```bash
# Create user
sudo mosquitto_passwd -c /etc/mosquitto/passwd username

# Add more users (omit -c flag)
sudo mosquitto_passwd /etc/mosquitto/passwd username2
```

**Update configuration:**

```conf
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd
```

**Restart broker:**

```bash
sudo systemctl restart mosquitto
```

### 2.4 Command Line Testing

**Terminal 1 - Subscribe:**

```bash
mosquitto_sub -h localhost -t test/topic

# With authentication
mosquitto_sub -h localhost -t test/topic -u username -P password
```

**Terminal 2 - Publish:**

```bash
mosquitto_pub -h localhost -t test/topic -m "Hello MQTT!"

# With authentication
mosquitto_pub -h localhost -t test/topic -m "Hello" -u username -P password
```

---

## 3. MQTT Topics

### 3.1 Topic Structure

Topics use hierarchical structure with `/` separator:

```
home/livingroom/temperature
home/livingroom/humidity
home/bedroom/temperature
home/kitchen/light/status
sensors/outdoor/weather/wind
```

**Best Practices:**
- Use descriptive names
- Start with general → specific
- Use lowercase
- Avoid leading `/`
- Keep depth reasonable (3-5 levels)

### 3.2 Wildcards

**Single-level wildcard (`+`)**: Matches one level

```
home/+/temperature
Matches:
  ✓ home/livingroom/temperature
  ✓ home/bedroom/temperature
  ✗ home/livingroom/sensor/temperature  (too deep)
```

**Multi-level wildcard (`#`)**: Matches all sub-levels

```
home/livingroom/#
Matches:
  ✓ home/livingroom/temperature
  ✓ home/livingroom/humidity
  ✓ home/livingroom/sensor/temp
  ✓ home/livingroom/sensor/data/raw
```

**Combined wildcards:**

```
home/+/sensor/#
Matches:
  ✓ home/livingroom/sensor/temp
  ✓ home/bedroom/sensor/data/raw
```

### 3.3 Reserved Topics

Topics starting with `$` are reserved:

- `$SYS/broker/clients/connected`: Number of connected clients
- `$SYS/broker/messages/received`: Total messages received
- `$SYS/broker/uptime`: Broker uptime

```bash
# Monitor broker statistics
mosquitto_sub -h localhost -t '$SYS/#' -v
```

---

## 4. Quality of Service (QoS)

### 4.1 QoS Levels

| Level | Name | Guarantee | Use Case |
|-------|------|-----------|----------|
| **QoS 0** | At most once | No guarantee (fire and forget) | Non-critical sensor data, high-frequency updates |
| **QoS 1** | At least once | Message delivered, duplicates possible | Most IoT applications, general telemetry |
| **QoS 2** | Exactly once | Message delivered exactly once, no duplicates | Critical commands, billing, safety systems |

### 4.2 QoS Flow Diagrams

**QoS 0:**
```
Publisher ──PUBLISH──> Broker ──PUBLISH──> Subscriber
```

**QoS 1:**
```
Publisher ──PUBLISH──> Broker ──PUBLISH──> Subscriber
         <──PUBACK───         <──PUBACK───
```

**QoS 2:**
```
Publisher ──PUBLISH──> Broker ──PUBLISH──> Subscriber
         <──PUBREC───         <──PUBREC───
          ──PUBREL──>          ──PUBREL──>
         <──PUBCOMP──         <──PUBCOMP──
```

**Performance Comparison:**
- QoS 0: Fastest, lowest bandwidth
- QoS 1: Good balance (recommended)
- QoS 2: Highest overhead, use only when necessary

---

## 5. Python MQTT with paho-mqtt

### 5.1 paho-mqtt Installation

```bash
pip3 install paho-mqtt
```

### 5.2 MQTT Publisher

```python
import paho.mqtt.client as mqtt
import time
import random

# Broker configuration
BROKER = "localhost"
PORT = 1883
TOPIC = "sensors/temperature"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

def on_publish(client, userdata, mid):
    print(f"Message {mid} published")

# Create client
client = mqtt.Client()
client.on_connect = on_connect
client.on_publish = on_publish

# Connect to broker
client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

try:
    while True:
        # Simulate sensor reading
        temperature = round(random.uniform(20.0, 30.0), 2)

        # Publish message
        result = client.publish(TOPIC, str(temperature), qos=1)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published: {temperature}°C")
        else:
            print(f"Publish failed: {result.rc}")

        time.sleep(5)

except KeyboardInterrupt:
    print("\nStopping publisher...")
finally:
    client.loop_stop()
    client.disconnect()
```

### 5.3 MQTT Subscriber

```python
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "sensors/#"  # Subscribe to all sensor topics

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe on connection
        client.subscribe(TOPIC, qos=1)
        print(f"Subscribed to: {TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode()}")

# Create client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect and start loop
client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()
```

### 5.4 MQTT Client with Authentication

```python
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
USERNAME = "iot_user"
PASSWORD = "secure_password"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully")
    elif rc == 5:
        print("Authentication failed")
    else:
        print(f"Connection failed, code {rc}")

client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect

client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()
```

### 5.5 Retained Messages

Retained messages are stored by the broker and immediately delivered to new subscribers.

```python
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("localhost", 1883)

# Publish retained message
client.publish(
    "home/livingroom/light/status",
    "ON",
    qos=1,
    retain=True  # Retain message
)

print("Retained message published")
client.disconnect()
```

**When a new subscriber connects, it immediately receives the last retained message.**

### 5.6 Last Will and Testament (LWT)

LWT is a message automatically published by the broker when a client disconnects unexpectedly.

```python
import paho.mqtt.client as mqtt
import time

client = mqtt.Client()

# Set Last Will (before connecting)
client.will_set(
    "devices/rpi001/status",
    payload="offline",
    qos=1,
    retain=True
)

client.connect("localhost", 1883)
client.loop_start()

# Publish online status
client.publish("devices/rpi001/status", "online", qos=1, retain=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Graceful shutdown: publish offline status manually
    client.publish("devices/rpi001/status", "offline", qos=1, retain=True)
    client.disconnect()
    client.loop_stop()
```

---

## 6. Practical Project: IoT Sensor System with MQTT

Complete sensor monitoring system using MQTT.

### 6.1 Sensor Publisher (Raspberry Pi)

```python
import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime
import random

class SensorPublisher:
    def __init__(self, broker, port=1883, client_id="sensor_rpi001"):
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.client = mqtt.Client(client_id)

        # Set callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish

        # Set Last Will
        self.client.will_set(
            f"devices/{client_id}/status",
            payload="offline",
            qos=1,
            retain=True
        )

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[✓] Connected to broker: {self.broker}")
            # Publish online status
            self.client.publish(
                f"devices/{self.client_id}/status",
                "online",
                qos=1,
                retain=True
            )
        else:
            print(f"[✗] Connection failed, code {rc}")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"[!] Unexpected disconnect, code {rc}")
        else:
            print("[✓] Disconnected")

    def on_publish(self, client, userdata, mid):
        print(f"  → Message {mid} published")

    def read_sensors(self):
        """Simulate sensor readings"""
        return {
            "timestamp": datetime.now().isoformat(),
            "temperature": round(random.uniform(20.0, 30.0), 2),
            "humidity": round(random.uniform(40.0, 70.0), 2),
            "pressure": round(random.uniform(1000, 1020), 1)
        }

    def publish_sensor_data(self):
        """Read and publish sensor data"""
        data = self.read_sensors()

        # Publish to separate topics
        self.client.publish(
            f"sensors/{self.client_id}/temperature",
            data['temperature'],
            qos=1
        )
        self.client.publish(
            f"sensors/{self.client_id}/humidity",
            data['humidity'],
            qos=1
        )

        # Publish aggregated JSON
        self.client.publish(
            f"sensors/{self.client_id}/data",
            json.dumps(data),
            qos=1
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Published: {data}")

    def run(self, interval=10):
        """Connect and start publishing"""
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()

        try:
            while True:
                self.publish_sensor_data()
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[!] Stopping publisher...")
        finally:
            # Graceful shutdown
            self.client.publish(
                f"devices/{self.client_id}/status",
                "offline",
                qos=1,
                retain=True
            )
            self.client.disconnect()
            self.client.loop_stop()

if __name__ == "__main__":
    publisher = SensorPublisher(broker="192.168.1.100")
    publisher.run(interval=10)
```

### 6.2 Data Subscriber and Logger

```python
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import csv
import os

class SensorLogger:
    def __init__(self, broker, port=1883, log_file="sensor_data.csv"):
        self.broker = broker
        self.port = port
        self.log_file = log_file
        self.client = mqtt.Client("logger_001")

        # Set callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Initialize CSV file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'device_id', 'temperature', 'humidity', 'pressure'])

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[✓] Connected to broker: {self.broker}")

            # Subscribe to all sensor data
            client.subscribe("sensors/+/data", qos=1)
            # Subscribe to device status
            client.subscribe("devices/+/status", qos=1)

            print("[✓] Subscribed to topics")
        else:
            print(f"[✗] Connection failed, code {rc}")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode()

        # Handle device status
        if "/status" in topic:
            device_id = topic.split('/')[1]
            print(f"[STATUS] {device_id}: {payload}")
            return

        # Handle sensor data
        if "/data" in topic:
            try:
                data = json.loads(payload)
                device_id = topic.split('/')[1]

                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{device_id}: Temp={data['temperature']}°C, "
                      f"Humidity={data['humidity']}%, "
                      f"Pressure={data['pressure']}hPa")

                # Log to CSV
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        data['timestamp'],
                        device_id,
                        data['temperature'],
                        data['humidity'],
                        data['pressure']
                    ])

            except json.JSONDecodeError:
                print(f"[✗] Invalid JSON: {payload}")

    def run(self):
        """Connect and start logging"""
        self.client.connect(self.broker, self.port, keepalive=60)

        print(f"[✓] Logging to: {self.log_file}")
        print("[✓] Waiting for messages... (Ctrl+C to stop)")

        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\n[!] Stopping logger...")
            self.client.disconnect()

if __name__ == "__main__":
    logger = SensorLogger(broker="192.168.1.100")
    logger.run()
```

---

## 7. Advanced MQTT Patterns

### 7.1 Request/Response Pattern

MQTT can implement request/response using separate topics:

```python
import paho.mqtt.client as mqtt
import json
import time
import uuid

class MQTTRequestResponse:
    def __init__(self, broker, port=1883):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.pending_requests = {}

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        # Subscribe to response topic
        client.subscribe("response/#", qos=1)

    def on_message(self, client, userdata, msg):
        # Handle response
        request_id = msg.topic.split('/')[-1]
        if request_id in self.pending_requests:
            response = json.loads(msg.payload.decode())
            self.pending_requests[request_id] = response

    def send_request(self, request_data, timeout=5):
        """Send request and wait for response"""
        request_id = str(uuid.uuid4())
        self.pending_requests[request_id] = None

        # Publish request
        request = {
            'request_id': request_id,
            'data': request_data
        }
        self.client.publish("request", json.dumps(request), qos=1)

        # Wait for response
        start_time = time.time()
        while self.pending_requests[request_id] is None:
            if time.time() - start_time > timeout:
                del self.pending_requests[request_id]
                raise TimeoutError("Request timeout")
            time.sleep(0.1)

        response = self.pending_requests[request_id]
        del self.pending_requests[request_id]
        return response

# Server side (request handler)
def on_request(client, userdata, msg):
    request = json.loads(msg.payload.decode())
    request_id = request['request_id']

    # Process request
    result = {"status": "success", "result": "processed"}

    # Send response
    client.publish(f"response/{request_id}", json.dumps(result), qos=1)
```

### 7.2 Message Routing Pattern

```python
import paho.mqtt.client as mqtt
import json

class MessageRouter:
    def __init__(self, broker, port=1883):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Routing rules
        self.routes = {
            'sensors/+/temperature': self.handle_temperature,
            'sensors/+/alert': self.handle_alert,
            'devices/+/command': self.handle_command
        }

    def on_connect(self, client, userdata, flags, rc):
        # Subscribe to all routing topics
        for topic in self.routes.keys():
            client.subscribe(topic, qos=1)
            print(f"Subscribed to: {topic}")

    def on_message(self, client, userdata, msg):
        # Find matching route
        for pattern, handler in self.routes.items():
            if self.topic_matches(msg.topic, pattern):
                handler(msg)
                break

    def topic_matches(self, topic, pattern):
        """Simple topic matching with + wildcard"""
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')

        if len(topic_parts) != len(pattern_parts):
            return False

        for t, p in zip(topic_parts, pattern_parts):
            if p != '+' and t != p:
                return False
        return True

    def handle_temperature(self, msg):
        device_id = msg.topic.split('/')[1]
        temp = float(msg.payload.decode())
        print(f"Temperature from {device_id}: {temp}°C")

        # Route to alerting if high temperature
        if temp > 35.0:
            self.client.publish(
                f"sensors/{device_id}/alert",
                json.dumps({"type": "high_temp", "value": temp}),
                qos=1
            )

    def handle_alert(self, msg):
        device_id = msg.topic.split('/')[1]
        alert = json.loads(msg.payload.decode())
        print(f"[ALERT] {device_id}: {alert}")

    def handle_command(self, msg):
        device_id = msg.topic.split('/')[1]
        command = msg.payload.decode()
        print(f"[COMMAND] {device_id}: {command}")

    def run(self):
        self.client.connect(self.broker, self.port)
        self.client.loop_forever()
```

---

## 8. Summary

### Completed Tasks

- ✅ **MQTT Basics**: Pub/Sub architecture, MQTT vs HTTP comparison
- ✅ **Mosquitto Broker**: Installation, configuration, authentication
- ✅ **Topics**: Hierarchical structure, wildcards, reserved topics
- ✅ **QoS Levels**: QoS 0/1/2 guarantees and use cases
- ✅ **paho-mqtt**: Publisher, subscriber, authentication, retained messages, LWT
- ✅ **Practical Project**: Complete sensor system with MQTT
- ✅ **Advanced Patterns**: Request/response, message routing

### Next Steps

| Next Lesson | Topic | Content |
|-------------|-------|---------|
| **07. HTTP REST for IoT** | RESTful API design | Flask server, CRUD operations, API validation |
| **08. Edge AI with TFLite** | Machine learning on edge | TensorFlow Lite, model optimization, inference |

### Hands-On Exercises

1. **Multi-Sensor System**:
   - Deploy 3 virtual sensors (temp, humidity, motion)
   - Each publishes to separate topics
   - Create unified dashboard subscriber

2. **Alert System**:
   - Monitor sensor values
   - Trigger alerts on threshold violations
   - Implement escalation (email, SMS)

3. **Device Control**:
   - Create command topics for LED control
   - Implement status reporting
   - Add acknowledgment mechanism

4. **MQTT Bridge**:
   - Connect two MQTT brokers
   - Forward messages between them
   - Implement topic filtering

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Connection refused** | Broker not running | Check `systemctl status mosquitto` |
| **Authentication failed** | Wrong credentials | Verify username/password, check password file |
| **Messages not received** | Topic mismatch | Verify topic spelling, check wildcards |
| **High latency** | Network issues | Check QoS level, broker load, network quality |

---

## References

- [MQTT Protocol Specification](https://mqtt.org/mqtt-specification/)
- [Mosquitto Documentation](https://mosquitto.org/documentation/)
- [paho-mqtt Documentation](https://www.eclipse.org/paho/index.php?page=clients/python/docs/index.php)
- [HiveMQ MQTT Essentials](https://www.hivemq.com/mqtt-essentials/)
