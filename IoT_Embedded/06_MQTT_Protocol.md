# 06. MQTT 프로토콜

## 학습 목표

- MQTT 프로토콜의 원리와 특징 이해
- Mosquitto 브로커 설치 및 설정
- Topic 구조와 QoS 레벨 이해
- paho-mqtt 라이브러리 사용법 습득
- 메시지 발행 및 구독 구현

---

## 1. MQTT 프로토콜 개요

### 1.1 MQTT란?

**MQTT (Message Queuing Telemetry Transport)**는 경량 메시징 프로토콜로, IoT 환경에 최적화되어 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    MQTT 아키텍처                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Publisher                    Broker                        │
│   (센서)                       (중계자)                       │
│   ┌─────────┐                 ┌─────────┐                   │
│   │ 온도    │ ──PUBLISH────▶ │         │                   │
│   │ 센서    │    (topic:     │ Mosquitto│                   │
│   └─────────┘   home/temp)   │         │                   │
│                               │         │                   │
│   ┌─────────┐                 │         │    ┌─────────┐    │
│   │ 습도    │ ──PUBLISH────▶ │         │ ──▶│ 모바일  │    │
│   │ 센서    │    (topic:     │         │    │   앱    │    │
│   └─────────┘   home/humid)  │         │    └─────────┘    │
│                               │         │    Subscriber     │
│                               │         │                   │
│                               │         │    ┌─────────┐    │
│                               │         │ ──▶│ 웹      │    │
│                               │         │    │ 대시보드│    │
│                               └─────────┘    └─────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 MQTT 특징

| 특징 | 설명 |
|------|------|
| **경량** | 최소 2바이트 헤더 (HTTP 대비 매우 작음) |
| **Pub/Sub** | 발행/구독 패턴 (느슨한 결합) |
| **QoS** | 3가지 메시지 전달 보장 수준 |
| **Last Will** | 비정상 연결 종료 시 알림 |
| **Retained** | 마지막 메시지 저장 |
| **Keep Alive** | 연결 상태 모니터링 |

### 1.3 MQTT vs HTTP 비교

```python
# 프로토콜 비교
comparison = {
    "Header Size": {
        "MQTT": "2 bytes (최소)",
        "HTTP": "~800 bytes (평균)"
    },
    "Pattern": {
        "MQTT": "Pub/Sub (비동기)",
        "HTTP": "Request/Response (동기)"
    },
    "Connection": {
        "MQTT": "지속 연결",
        "HTTP": "비연결 (HTTP/1.1) 또는 지속 (HTTP/2)"
    },
    "Bidirectional": {
        "MQTT": "지원 (양방향)",
        "HTTP": "서버 푸시 제한적"
    },
    "Use Case": {
        "MQTT": "실시간 센서, 저전력, 저대역폭",
        "HTTP": "웹 API, 대용량 데이터"
    }
}
```

---

## 2. Mosquitto 브로커

### 2.1 설치

```bash
# Ubuntu/Debian (라즈베리파이)
sudo apt update
sudo apt install mosquitto mosquitto-clients

# 서비스 시작 및 활성화
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# 상태 확인
sudo systemctl status mosquitto
```

### 2.2 기본 설정

```bash
# 설정 파일 편집
sudo nano /etc/mosquitto/mosquitto.conf
```

```conf
# /etc/mosquitto/mosquitto.conf

# 기본 설정
pid_file /run/mosquitto/mosquitto.pid

# 리스너 설정
listener 1883
protocol mqtt

# 익명 접속 (테스트용)
allow_anonymous true

# 로그 설정
log_dest file /var/log/mosquitto/mosquitto.log
log_type all

# 지속성 (메시지 저장)
persistence true
persistence_location /var/lib/mosquitto/

# 추가 설정 파일 포함
include_dir /etc/mosquitto/conf.d
```

### 2.3 인증 설정

```bash
# 비밀번호 파일 생성
sudo mosquitto_passwd -c /etc/mosquitto/passwd iotuser

# 추가 사용자
sudo mosquitto_passwd /etc/mosquitto/passwd anotheruser
```

```conf
# /etc/mosquitto/conf.d/auth.conf

# 익명 접속 비활성화
allow_anonymous false

# 비밀번호 파일
password_file /etc/mosquitto/passwd
```

```bash
# 설정 적용
sudo systemctl restart mosquitto
```

### 2.4 TLS 설정 (보안 연결)

```bash
# 인증서 생성 (자체 서명)
mkdir -p ~/mqtt-certs && cd ~/mqtt-certs

# CA 키 및 인증서
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 365 -key ca.key -out ca.crt

# 서버 키 및 인증서
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365
```

```conf
# /etc/mosquitto/conf.d/tls.conf

listener 8883
protocol mqtt

cafile /home/pi/mqtt-certs/ca.crt
certfile /home/pi/mqtt-certs/server.crt
keyfile /home/pi/mqtt-certs/server.key

require_certificate false
```

### 2.5 CLI 테스트

```bash
# 터미널 1: 구독
mosquitto_sub -h localhost -t "test/topic" -v

# 터미널 2: 발행
mosquitto_pub -h localhost -t "test/topic" -m "Hello MQTT!"

# 인증 포함
mosquitto_pub -h localhost -t "test/topic" -m "Hello" -u iotuser -P password

# QoS 지정
mosquitto_pub -h localhost -t "test/topic" -m "QoS 1" -q 1
```

---

## 3. Topic과 QoS

### 3.1 Topic 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    MQTT Topic 구조                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   계층적 구조 (슬래시로 구분)                                 │
│                                                              │
│   home/                                                      │
│   ├── living-room/                                          │
│   │   ├── temperature      → 거실 온도                      │
│   │   ├── humidity         → 거실 습도                      │
│   │   └── light            → 거실 조명                      │
│   ├── bedroom/                                              │
│   │   ├── temperature                                       │
│   │   └── motion           → 침실 모션 센서                 │
│   └── kitchen/                                              │
│       └── smoke            → 주방 연기 감지                 │
│                                                              │
│   와일드카드:                                                │
│   • + (단일 레벨): home/+/temperature                       │
│     → home/living-room/temperature, home/bedroom/temperature│
│                                                              │
│   • # (다중 레벨): home/#                                    │
│     → home 아래 모든 토픽                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Topic 설계 가이드

```python
# 좋은 Topic 설계 예시
topic_examples = {
    # 위치/장치/센서
    "home/living-room/temperature": "거실 온도",
    "office/floor1/room101/hvac/status": "사무실 HVAC 상태",

    # 장치ID/데이터타입
    "sensor/abc123/data": "센서 데이터",
    "sensor/abc123/status": "센서 상태",

    # 명령 및 응답
    "device/led001/command": "LED 명령",
    "device/led001/response": "LED 응답",

    # 클라우드 연동
    "aws/things/sensor001/shadow/update": "AWS IoT 섀도우",
}

# 피해야 할 패턴
bad_patterns = [
    "/leading/slash",     # 선행 슬래시 불필요
    "space in topic",     # 공백 피하기
    "UpperCase/Mixed",    # 소문자 권장
    "too/deep/hierarchy/a/b/c/d/e",  # 과도한 깊이
]
```

### 3.3 QoS (Quality of Service)

```
┌─────────────────────────────────────────────────────────────┐
│                    MQTT QoS 레벨                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  QoS 0: At most once (최대 1회)                             │
│  ┌────────┐         ┌────────┐                              │
│  │Publisher│──PUBLISH──▶│Broker│                             │
│  └────────┘         └────────┘                              │
│  • 전송 확인 없음                                            │
│  • 가장 빠름, 메시지 손실 가능                               │
│  • 용도: 센서 데이터 (손실 허용)                             │
│                                                              │
│  QoS 1: At least once (최소 1회)                            │
│  ┌────────┐         ┌────────┐                              │
│  │Publisher│──PUBLISH──▶│Broker│                             │
│  │        │◀──PUBACK───│      │                             │
│  └────────┘         └────────┘                              │
│  • 확인 응답, 재전송 가능                                    │
│  • 중복 가능, 손실 없음                                      │
│  • 용도: 중요 알림                                          │
│                                                              │
│  QoS 2: Exactly once (정확히 1회)                           │
│  ┌────────┐         ┌────────┐                              │
│  │Publisher│──PUBLISH──▶│Broker│                             │
│  │        │◀──PUBREC───│      │                             │
│  │        │──PUBREL──▶│      │                             │
│  │        │◀──PUBCOMP──│      │                             │
│  └────────┘         └────────┘                              │
│  • 4-way handshake                                          │
│  • 가장 느림, 정확한 전달 보장                               │
│  • 용도: 결제, 중요 명령                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Retained 메시지

```python
# Retained 메시지 개념
"""
Retained Message:
- 브로커가 마지막 메시지를 저장
- 새 구독자가 연결 시 즉시 수신
- 센서 현재 상태 전달에 유용

예:
1. 센서가 온도 25도를 retain=True로 발행
2. 브로커가 메시지 저장
3. 새 구독자 연결 시 즉시 25도 수신
4. 센서 오프라인이어도 마지막 값 유지
"""

# 사용 예
retained_use_cases = {
    "장치 상태": "device/sensor01/status (online/offline)",
    "현재 값": "home/temperature (마지막 측정값)",
    "설정": "device/config (현재 설정)",
}
```

---

## 4. paho-mqtt 라이브러리

### 4.1 설치

```bash
pip install paho-mqtt
```

### 4.2 기본 Publisher

```python
#!/usr/bin/env python3
"""MQTT Publisher 기본 예제"""

import paho.mqtt.client as mqtt
import json
import time

# 브로커 설정
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "sensor/temperature"

def on_connect(client, userdata, flags, rc):
    """연결 콜백"""
    if rc == 0:
        print("브로커 연결 성공")
    else:
        print(f"연결 실패: {rc}")

def on_publish(client, userdata, mid):
    """발행 완료 콜백"""
    print(f"메시지 발행됨: mid={mid}")

def publish_sensor_data():
    """센서 데이터 발행"""
    client = mqtt.Client(client_id="temperature_sensor_01")
    client.on_connect = on_connect
    client.on_publish = on_publish

    # 연결
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    try:
        while True:
            # 센서 데이터 생성
            data = {
                "sensor_id": "temp_01",
                "temperature": round(20 + (time.time() % 10), 1),
                "timestamp": int(time.time())
            }

            payload = json.dumps(data)

            # 발행 (QoS 1, Retained 사용)
            result = client.publish(TOPIC, payload, qos=1, retain=True)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"발행: {payload}")
            else:
                print(f"발행 실패: {result.rc}")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n종료")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    publish_sensor_data()
```

### 4.3 기본 Subscriber

```python
#!/usr/bin/env python3
"""MQTT Subscriber 기본 예제"""

import paho.mqtt.client as mqtt
import json

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPICS = [
    ("sensor/temperature", 1),
    ("sensor/humidity", 1),
]

def on_connect(client, userdata, flags, rc):
    """연결 콜백"""
    if rc == 0:
        print("브로커 연결 성공")
        # 토픽 구독
        for topic, qos in TOPICS:
            client.subscribe(topic, qos)
            print(f"구독: {topic} (QoS {qos})")
    else:
        print(f"연결 실패: {rc}")

def on_message(client, userdata, msg):
    """메시지 수신 콜백"""
    try:
        payload = json.loads(msg.payload.decode())
        print(f"[{msg.topic}] {payload}")
    except json.JSONDecodeError:
        print(f"[{msg.topic}] {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    """연결 해제 콜백"""
    print(f"연결 해제: {rc}")

def subscribe_sensors():
    """센서 데이터 구독"""
    client = mqtt.Client(client_id="sensor_monitor_01")
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n종료")
        client.disconnect()

if __name__ == "__main__":
    subscribe_sensors()
```

### 4.4 인증 사용

```python
#!/usr/bin/env python3
"""MQTT 인증 연결"""

import paho.mqtt.client as mqtt
import ssl

BROKER_HOST = "mqtt.example.com"
BROKER_PORT = 8883  # TLS

def create_secure_client(username: str, password: str) -> mqtt.Client:
    """보안 MQTT 클라이언트 생성"""
    client = mqtt.Client(client_id="secure_client_01")

    # 인증 설정
    client.username_pw_set(username, password)

    # TLS 설정
    client.tls_set(
        ca_certs="/path/to/ca.crt",
        certfile="/path/to/client.crt",  # 클라이언트 인증서 (옵션)
        keyfile="/path/to/client.key",   # 클라이언트 키 (옵션)
        tls_version=ssl.PROTOCOL_TLS
    )

    # 호스트명 검증 비활성화 (자체 서명 인증서용)
    # client.tls_insecure_set(True)

    return client

def connect_secure():
    """보안 연결"""
    client = create_secure_client("iotuser", "password123")

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("보안 연결 성공")
        else:
            print(f"연결 실패: {rc}")

    client.on_connect = on_connect
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()
```

### 4.5 Last Will and Testament (LWT)

```python
#!/usr/bin/env python3
"""MQTT Last Will (비정상 종료 알림)"""

import paho.mqtt.client as mqtt
import time

def create_client_with_lwt(client_id: str) -> mqtt.Client:
    """LWT가 설정된 클라이언트 생성"""
    client = mqtt.Client(client_id=client_id)

    # Last Will 설정
    # 비정상 연결 종료 시 이 메시지가 발행됨
    client.will_set(
        topic=f"device/{client_id}/status",
        payload="offline",
        qos=1,
        retain=True
    )

    return client

def run_sensor_with_lwt():
    """LWT가 있는 센서 실행"""
    client_id = "sensor_with_lwt"
    client = create_client_with_lwt(client_id)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("연결됨")
            # 온라인 상태 발행
            client.publish(
                f"device/{client_id}/status",
                "online",
                qos=1,
                retain=True
            )

    client.on_connect = on_connect
    client.connect("localhost", 1883, keepalive=60)
    client.loop_start()

    try:
        while True:
            client.publish(f"device/{client_id}/data", "sensor data")
            time.sleep(5)
    except KeyboardInterrupt:
        # 정상 종료 시 오프라인 상태 발행
        client.publish(f"device/{client_id}/status", "offline", qos=1, retain=True)
        client.disconnect()
```

---

## 5. 고급 패턴

### 5.1 메시지 라우팅

```python
#!/usr/bin/env python3
"""토픽 기반 메시지 라우팅"""

import paho.mqtt.client as mqtt
import json
from typing import Callable

class MQTTRouter:
    """MQTT 메시지 라우터"""

    def __init__(self, broker_host: str, broker_port: int = 1883):
        self.client = mqtt.Client()
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.routes: dict[str, Callable] = {}

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def route(self, topic_pattern: str):
        """라우트 데코레이터"""
        def decorator(func: Callable):
            self.routes[topic_pattern] = func
            return func
        return decorator

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("라우터 연결됨")
            for topic in self.routes.keys():
                client.subscribe(topic)
                print(f"라우트 등록: {topic}")

    def _on_message(self, client, userdata, msg):
        # 매칭되는 핸들러 찾기
        for pattern, handler in self.routes.items():
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    payload = json.loads(msg.payload.decode())
                except json.JSONDecodeError:
                    payload = msg.payload.decode()

                handler(msg.topic, payload)
                break

    def run(self):
        """라우터 실행"""
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_forever()

# 사용 예
router = MQTTRouter("localhost")

@router.route("sensor/+/temperature")
def handle_temperature(topic: str, payload: dict):
    sensor_id = topic.split('/')[1]
    print(f"온도 [{sensor_id}]: {payload}")

@router.route("sensor/+/humidity")
def handle_humidity(topic: str, payload: dict):
    sensor_id = topic.split('/')[1]
    print(f"습도 [{sensor_id}]: {payload}")

@router.route("device/+/command")
def handle_command(topic: str, payload: dict):
    device_id = topic.split('/')[1]
    print(f"명령 [{device_id}]: {payload}")

if __name__ == "__main__":
    router.run()
```

### 5.2 비동기 MQTT (asyncio)

```python
#!/usr/bin/env python3
"""비동기 MQTT 클라이언트 (asyncio-mqtt)"""

import asyncio
import aiomqtt  # pip install aiomqtt
import json

async def publish_sensor_data():
    """비동기 발행"""
    async with aiomqtt.Client("localhost") as client:
        while True:
            data = {
                "temperature": 25.5,
                "timestamp": asyncio.get_event_loop().time()
            }
            await client.publish("sensor/temp", json.dumps(data))
            print(f"발행: {data}")
            await asyncio.sleep(5)

async def subscribe_sensor_data():
    """비동기 구독"""
    async with aiomqtt.Client("localhost") as client:
        async with client.messages() as messages:
            await client.subscribe("sensor/#")

            async for message in messages:
                print(f"[{message.topic}] {message.payload.decode()}")

async def main():
    """발행과 구독 동시 실행"""
    await asyncio.gather(
        publish_sensor_data(),
        subscribe_sensor_data()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.3 재연결 로직

```python
#!/usr/bin/env python3
"""자동 재연결 MQTT 클라이언트"""

import paho.mqtt.client as mqtt
import time

class RobustMQTTClient:
    """자동 재연결을 지원하는 MQTT 클라이언트"""

    def __init__(self, broker_host: str, broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client()
        self.connected = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("연결됨")
            self.connected = True
            self.reconnect_delay = 1  # 리셋
        else:
            print(f"연결 실패: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"연결 해제: {rc}")
        self.connected = False

        if rc != 0:
            self._reconnect()

    def _reconnect(self):
        """재연결 시도"""
        while not self.connected:
            try:
                print(f"재연결 시도... ({self.reconnect_delay}초 후)")
                time.sleep(self.reconnect_delay)
                self.client.reconnect()
            except Exception as e:
                print(f"재연결 실패: {e}")
                # 지수 백오프
                self.reconnect_delay = min(
                    self.reconnect_delay * 2,
                    self.max_reconnect_delay
                )

    def connect(self):
        """초기 연결"""
        self.client.connect(self.broker_host, self.broker_port)

    def run(self):
        """클라이언트 실행"""
        self.connect()
        self.client.loop_forever()

if __name__ == "__main__":
    client = RobustMQTTClient("localhost")
    client.run()
```

---

## 연습 문제

### 문제 1: 온습도 모니터
1. 온도와 습도 데이터를 5초마다 발행하는 Publisher를 작성하세요.
2. 해당 데이터를 구독하여 콘솔에 출력하는 Subscriber를 작성하세요.

### 문제 2: 장치 상태 관리
1. LWT를 사용하여 장치 온라인/오프라인 상태를 관리하세요.
2. 와일드카드를 사용하여 모든 장치 상태를 모니터링하세요.

### 문제 3: 명령-응답 시스템
1. 명령 토픽으로 LED 제어 명령을 수신하세요.
2. 응답 토픽으로 실행 결과를 발행하세요.

---

## 다음 단계

- [07_HTTP_REST_for_IoT.md](07_HTTP_REST_for_IoT.md): REST API와 MQTT 통합
- [10_Home_Automation_Project.md](10_Home_Automation_Project.md): MQTT 기반 스마트홈

---

*최종 업데이트: 2026-02-01*
