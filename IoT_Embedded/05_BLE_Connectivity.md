# 05. BLE 연결

## 학습 목표

- BLE(Bluetooth Low Energy) 프로토콜 개요 이해
- GATT 구조 (서비스, 특성) 파악
- Python bleak 라이브러리 사용법 습득
- BLE 장치 스캔 및 연결
- 센서 데이터 수신

---

## 1. BLE 프로토콜 개요

### 1.1 BLE vs 클래식 Bluetooth

| 특성 | BLE (Bluetooth Low Energy) | 클래식 Bluetooth |
|------|---------------------------|------------------|
| **전력 소비** | 매우 낮음 | 높음 |
| **데이터 전송률** | 1-2 Mbps | 1-3 Mbps |
| **범위** | ~100m | ~100m |
| **지연 시간** | ~6ms | ~100ms |
| **페어링** | 간단/자동 | 복잡 |
| **용도** | IoT 센서, 웨어러블 | 오디오, 파일 전송 |

### 1.2 BLE 프로토콜 스택

```
┌─────────────────────────────────────────────────────────────┐
│                    BLE 프로토콜 스택                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Application                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                GAP (Generic Access Profile)          │    │
│  │           디바이스 검색, 연결, 보안                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │             GATT (Generic Attribute Profile)         │    │
│  │              서비스, 특성, 데이터 교환                │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                 ATT (Attribute Protocol)             │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    L2CAP                             │    │
│  │           논리 링크 제어 및 적응 프로토콜             │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Link Layer + Physical Layer             │    │
│  │                  무선 통신 처리                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 BLE 역할

```python
# BLE 역할 정의
ble_roles = {
    "Central (Master)": {
        "description": "다른 장치를 스캔하고 연결을 시작",
        "example": "스마트폰, 라즈베리파이",
        "behavior": ["스캔", "연결 요청", "데이터 요청"]
    },
    "Peripheral (Slave)": {
        "description": "광고하고 연결을 기다림",
        "example": "센서, 비콘, 웨어러블",
        "behavior": ["광고", "연결 대기", "데이터 제공"]
    },
    "Observer": {
        "description": "광고 패킷만 수신 (연결 없음)",
        "example": "비콘 리더",
        "behavior": ["스캔만"]
    },
    "Broadcaster": {
        "description": "광고 패킷만 송신 (연결 없음)",
        "example": "비콘",
        "behavior": ["광고만"]
    }
}
```

---

## 2. GATT 구조

### 2.1 GATT 계층 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    GATT 계층 구조                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   GATT Server (Peripheral)                                   │
│   │                                                          │
│   ├── Profile                                                │
│   │   │                                                      │
│   │   ├── Service (UUID: 0x180F - Battery Service)          │
│   │   │   │                                                  │
│   │   │   └── Characteristic (UUID: 0x2A19 - Battery Level)│
│   │   │       ├── Value: 85 (0-100%)                        │
│   │   │       ├── Properties: Read, Notify                  │
│   │   │       └── Descriptors                               │
│   │   │           └── CCCD (Client Config Descriptor)       │
│   │   │                                                      │
│   │   └── Service (UUID: 0x181A - Environmental Sensing)    │
│   │       │                                                  │
│   │       ├── Characteristic: Temperature (0x2A6E)          │
│   │       │   └── Value: 25.5°C                             │
│   │       │                                                  │
│   │       └── Characteristic: Humidity (0x2A6F)             │
│   │           └── Value: 60%                                 │
│   │                                                          │
│   └── ...                                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 표준 UUID

```python
# 표준 BLE 서비스 UUID (16-bit)
standard_services = {
    "0x1800": "Generic Access",
    "0x1801": "Generic Attribute",
    "0x180A": "Device Information",
    "0x180F": "Battery Service",
    "0x181A": "Environmental Sensing",
    "0x180D": "Heart Rate",
}

# 표준 특성 UUID
standard_characteristics = {
    "0x2A00": "Device Name",
    "0x2A19": "Battery Level",
    "0x2A6E": "Temperature",
    "0x2A6F": "Humidity",
    "0x2A37": "Heart Rate Measurement",
}

# 16-bit UUID를 128-bit로 변환
def uuid_16_to_128(uuid_16: str) -> str:
    """16-bit UUID를 128-bit BLE 기본 UUID로 변환"""
    base_uuid = "00000000-0000-1000-8000-00805f9b34fb"
    uuid_16_clean = uuid_16.replace("0x", "").lower()
    return f"0000{uuid_16_clean}{base_uuid[8:]}"

# 예: 0x180F -> 0000180f-0000-1000-8000-00805f9b34fb
```

### 2.3 특성 속성

```python
# 특성 속성 (Properties)
characteristic_properties = {
    "Broadcast": 0x01,       # 광고에 포함 가능
    "Read": 0x02,            # 읽기 가능
    "Write No Response": 0x04,  # 응답 없이 쓰기
    "Write": 0x08,           # 응답 있는 쓰기
    "Notify": 0x10,          # 알림 (응답 없음)
    "Indicate": 0x20,        # 표시 (응답 있음)
}

def parse_properties(props: int) -> list:
    """속성 비트마스크를 리스트로 변환"""
    result = []
    for name, value in characteristic_properties.items():
        if props & value:
            result.append(name)
    return result

# 예: parse_properties(0x12) -> ['Read', 'Notify']
```

---

## 3. bleak 라이브러리

### 3.1 설치 및 설정

```bash
# bleak 설치
pip install bleak

# Linux에서 추가 설정 (bluetoothctl 접근 권한)
sudo usermod -a -G bluetooth $USER

# D-Bus 서비스 확인
sudo systemctl status bluetooth
```

### 3.2 BLE 장치 스캔

```python
#!/usr/bin/env python3
"""BLE 장치 스캔 (bleak)"""

import asyncio
from bleak import BleakScanner

async def scan_devices(timeout: float = 10.0):
    """주변 BLE 장치 스캔"""
    print(f"BLE 장치 스캔 중... ({timeout}초)")

    devices = await BleakScanner.discover(timeout=timeout)

    print(f"\n발견된 장치: {len(devices)}개\n")

    for device in devices:
        rssi = device.rssi if hasattr(device, 'rssi') else 'N/A'
        print(f"  이름: {device.name or 'Unknown'}")
        print(f"  주소: {device.address}")
        print(f"  RSSI: {rssi} dBm")
        print()

    return devices

async def scan_with_filter(name_filter: str = None):
    """이름 필터로 스캔"""
    devices = await BleakScanner.discover()

    if name_filter:
        devices = [d for d in devices if d.name and name_filter.lower() in d.name.lower()]

    return devices

async def continuous_scan(callback=None, duration: float = 30.0):
    """연속 스캔 (장치 발견 시 콜백)"""
    def detection_callback(device, advertisement_data):
        print(f"발견: {device.name} ({device.address})")
        if callback:
            callback(device, advertisement_data)

    scanner = BleakScanner(detection_callback=detection_callback)

    print(f"연속 스캔 시작 ({duration}초)")
    await scanner.start()
    await asyncio.sleep(duration)
    await scanner.stop()

if __name__ == "__main__":
    asyncio.run(scan_devices(10))
```

### 3.3 BLE 장치 연결

```python
#!/usr/bin/env python3
"""BLE 장치 연결 및 서비스 탐색"""

import asyncio
from bleak import BleakClient, BleakScanner

async def connect_and_explore(address: str):
    """장치 연결 후 서비스/특성 탐색"""
    print(f"연결 중: {address}")

    async with BleakClient(address) as client:
        print(f"연결됨! MTU: {client.mtu_size}")

        # 서비스 탐색
        for service in client.services:
            print(f"\n서비스: {service.uuid}")
            print(f"  설명: {service.description}")

            # 특성 탐색
            for char in service.characteristics:
                print(f"    특성: {char.uuid}")
                print(f"      속성: {char.properties}")

                # 읽기 가능하면 값 읽기
                if "read" in char.properties:
                    try:
                        value = await client.read_gatt_char(char.uuid)
                        print(f"      값: {value}")
                    except Exception as e:
                        print(f"      읽기 실패: {e}")

async def find_and_connect(name_filter: str):
    """이름으로 장치 찾아 연결"""
    print(f"장치 검색: '{name_filter}'")

    device = await BleakScanner.find_device_by_name(name_filter)

    if device:
        print(f"장치 발견: {device.address}")
        await connect_and_explore(device.address)
    else:
        print("장치를 찾을 수 없습니다.")

if __name__ == "__main__":
    # MAC 주소로 직접 연결
    # asyncio.run(connect_and_explore("AA:BB:CC:DD:EE:FF"))

    # 이름으로 검색 후 연결
    asyncio.run(find_and_connect("Temperature"))
```

---

## 4. 센서 데이터 수신

### 4.1 특성 값 읽기

```python
#!/usr/bin/env python3
"""BLE 특성 값 읽기"""

import asyncio
from bleak import BleakClient
import struct

# 표준 UUID
BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
TEMPERATURE_UUID = "00002a6e-0000-1000-8000-00805f9b34fb"

async def read_battery_level(address: str) -> int | None:
    """배터리 레벨 읽기"""
    async with BleakClient(address) as client:
        try:
            data = await client.read_gatt_char(BATTERY_LEVEL_UUID)
            # 배터리 레벨은 1바이트 (0-100%)
            return data[0]
        except Exception as e:
            print(f"읽기 실패: {e}")
            return None

async def read_temperature(address: str) -> float | None:
    """온도 읽기 (IEEE 11073 형식)"""
    async with BleakClient(address) as client:
        try:
            data = await client.read_gatt_char(TEMPERATURE_UUID)
            # 온도는 16-bit 부호있는 정수 (0.01도 단위)
            temp_raw = struct.unpack('<h', data[:2])[0]
            return temp_raw * 0.01
        except Exception as e:
            print(f"읽기 실패: {e}")
            return None

async def read_multiple_chars(address: str, char_uuids: list) -> dict:
    """여러 특성 한 번에 읽기"""
    results = {}

    async with BleakClient(address) as client:
        for uuid in char_uuids:
            try:
                data = await client.read_gatt_char(uuid)
                results[uuid] = data
            except Exception as e:
                results[uuid] = None
                print(f"UUID {uuid} 읽기 실패: {e}")

    return results

if __name__ == "__main__":
    address = "AA:BB:CC:DD:EE:FF"

    # 배터리 레벨
    level = asyncio.run(read_battery_level(address))
    if level is not None:
        print(f"배터리: {level}%")

    # 온도
    temp = asyncio.run(read_temperature(address))
    if temp is not None:
        print(f"온도: {temp}°C")
```

### 4.2 알림(Notification) 수신

```python
#!/usr/bin/env python3
"""BLE 알림 수신 (실시간 센서 데이터)"""

import asyncio
from bleak import BleakClient
from datetime import datetime

# 예시 UUID (장치에 따라 다름)
HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

def notification_handler(sender, data):
    """알림 수신 콜백"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 수신 ({sender}): {data.hex()}")

    # 데이터 파싱 (예: Heart Rate Measurement)
    flags = data[0]
    if flags & 0x01:  # 16-bit heart rate
        hr = int.from_bytes(data[1:3], 'little')
    else:  # 8-bit heart rate
        hr = data[1]

    print(f"  심박수: {hr} bpm")

async def subscribe_notifications(address: str, char_uuid: str, duration: float = 60):
    """알림 구독"""
    async with BleakClient(address) as client:
        print(f"연결됨: {address}")

        # 알림 시작
        await client.start_notify(char_uuid, notification_handler)
        print(f"알림 구독 시작: {char_uuid}")

        # 지정된 시간 동안 수신
        await asyncio.sleep(duration)

        # 알림 중지
        await client.stop_notify(char_uuid)
        print("알림 구독 종료")

async def subscribe_multiple(address: str, char_uuids: list, duration: float = 60):
    """여러 특성 알림 구독"""
    async with BleakClient(address) as client:
        print(f"연결됨: {address}")

        for uuid in char_uuids:
            await client.start_notify(uuid, notification_handler)
            print(f"구독: {uuid}")

        await asyncio.sleep(duration)

        for uuid in char_uuids:
            await client.stop_notify(uuid)

if __name__ == "__main__":
    asyncio.run(subscribe_notifications("AA:BB:CC:DD:EE:FF", HEART_RATE_UUID, 30))
```

### 4.3 특성 값 쓰기

```python
#!/usr/bin/env python3
"""BLE 특성에 값 쓰기"""

import asyncio
from bleak import BleakClient

async def write_characteristic(address: str, char_uuid: str, data: bytes):
    """특성에 값 쓰기 (응답 있음)"""
    async with BleakClient(address) as client:
        await client.write_gatt_char(char_uuid, data, response=True)
        print(f"쓰기 완료: {data.hex()}")

async def write_without_response(address: str, char_uuid: str, data: bytes):
    """특성에 값 쓰기 (응답 없음 - 빠름)"""
    async with BleakClient(address) as client:
        await client.write_gatt_char(char_uuid, data, response=False)
        print(f"쓰기 전송: {data.hex()}")

async def toggle_led(address: str, led_uuid: str, state: bool):
    """LED 제어 예제"""
    data = bytes([0x01 if state else 0x00])
    await write_characteristic(address, led_uuid, data)

async def set_sensor_interval(address: str, config_uuid: str, interval_ms: int):
    """센서 측정 주기 설정"""
    # 2바이트 리틀엔디안
    data = interval_ms.to_bytes(2, 'little')
    await write_characteristic(address, config_uuid, data)
    print(f"측정 주기 설정: {interval_ms}ms")

if __name__ == "__main__":
    # LED 토글
    asyncio.run(toggle_led("AA:BB:CC:DD:EE:FF", "custom-led-uuid", True))
```

---

## 5. 종합 예제: BLE 센서 모니터

```python
#!/usr/bin/env python3
"""BLE 환경 센서 모니터"""

import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime
import struct

class BLESensorMonitor:
    """BLE 환경 센서 모니터링 클래스"""

    # 표준 Environmental Sensing 서비스
    ENV_SENSING_SERVICE = "0000181a-0000-1000-8000-00805f9b34fb"
    TEMPERATURE_CHAR = "00002a6e-0000-1000-8000-00805f9b34fb"
    HUMIDITY_CHAR = "00002a6f-0000-1000-8000-00805f9b34fb"

    def __init__(self, device_address: str = None, device_name: str = None):
        self.device_address = device_address
        self.device_name = device_name
        self.client = None
        self.data_buffer = []

    async def find_device(self) -> str | None:
        """장치 검색"""
        if self.device_address:
            return self.device_address

        if self.device_name:
            print(f"장치 검색: {self.device_name}")
            device = await BleakScanner.find_device_by_name(self.device_name)
            if device:
                self.device_address = device.address
                return device.address

        return None

    def _handle_temperature(self, sender, data):
        """온도 데이터 핸들러"""
        # 0.01도 단위 16-bit 정수
        temp = struct.unpack('<h', data[:2])[0] * 0.01
        timestamp = datetime.now()

        self.data_buffer.append({
            'type': 'temperature',
            'value': temp,
            'unit': '°C',
            'timestamp': timestamp
        })

        print(f"[{timestamp.strftime('%H:%M:%S')}] 온도: {temp:.2f}°C")

    def _handle_humidity(self, sender, data):
        """습도 데이터 핸들러"""
        # 0.01% 단위 16-bit 정수
        humidity = struct.unpack('<H', data[:2])[0] * 0.01
        timestamp = datetime.now()

        self.data_buffer.append({
            'type': 'humidity',
            'value': humidity,
            'unit': '%',
            'timestamp': timestamp
        })

        print(f"[{timestamp.strftime('%H:%M:%S')}] 습도: {humidity:.2f}%")

    async def start_monitoring(self, duration: float = 60):
        """모니터링 시작"""
        address = await self.find_device()
        if not address:
            print("장치를 찾을 수 없습니다.")
            return

        print(f"연결 중: {address}")

        async with BleakClient(address) as client:
            self.client = client
            print("연결됨!")

            # 서비스 확인
            services = client.services
            has_env_sensing = any(
                self.ENV_SENSING_SERVICE in str(s.uuid)
                for s in services
            )

            if not has_env_sensing:
                print("Environmental Sensing 서비스를 찾을 수 없습니다.")
                print("사용 가능한 서비스:")
                for s in services:
                    print(f"  - {s.uuid}")
                return

            # 알림 구독
            try:
                await client.start_notify(self.TEMPERATURE_CHAR, self._handle_temperature)
                print("온도 알림 구독 시작")
            except Exception as e:
                print(f"온도 구독 실패: {e}")

            try:
                await client.start_notify(self.HUMIDITY_CHAR, self._handle_humidity)
                print("습도 알림 구독 시작")
            except Exception as e:
                print(f"습도 구독 실패: {e}")

            print(f"\n모니터링 중... ({duration}초)")
            await asyncio.sleep(duration)

            # 정리
            await client.stop_notify(self.TEMPERATURE_CHAR)
            await client.stop_notify(self.HUMIDITY_CHAR)

        print("\n=== 모니터링 종료 ===")
        print(f"수집된 데이터: {len(self.data_buffer)}개")

    async def read_once(self) -> dict:
        """한 번 읽기"""
        address = await self.find_device()
        if not address:
            return {}

        async with BleakClient(address) as client:
            result = {}

            try:
                data = await client.read_gatt_char(self.TEMPERATURE_CHAR)
                result['temperature'] = struct.unpack('<h', data[:2])[0] * 0.01
            except:
                pass

            try:
                data = await client.read_gatt_char(self.HUMIDITY_CHAR)
                result['humidity'] = struct.unpack('<H', data[:2])[0] * 0.01
            except:
                pass

            return result

    def get_summary(self) -> dict:
        """수집된 데이터 요약"""
        if not self.data_buffer:
            return {}

        temps = [d['value'] for d in self.data_buffer if d['type'] == 'temperature']
        humids = [d['value'] for d in self.data_buffer if d['type'] == 'humidity']

        summary = {}

        if temps:
            summary['temperature'] = {
                'min': min(temps),
                'max': max(temps),
                'avg': sum(temps) / len(temps),
                'count': len(temps)
            }

        if humids:
            summary['humidity'] = {
                'min': min(humids),
                'max': max(humids),
                'avg': sum(humids) / len(humids),
                'count': len(humids)
            }

        return summary

if __name__ == "__main__":
    # 장치 이름으로 검색
    monitor = BLESensorMonitor(device_name="EnvSensor")

    # 또는 MAC 주소로 직접 지정
    # monitor = BLESensorMonitor(device_address="AA:BB:CC:DD:EE:FF")

    try:
        asyncio.run(monitor.start_monitoring(duration=30))
    except KeyboardInterrupt:
        print("\n사용자 중단")

    # 요약 출력
    summary = monitor.get_summary()
    if summary:
        print("\n=== 데이터 요약 ===")
        for sensor, stats in summary.items():
            print(f"{sensor}:")
            print(f"  최소: {stats['min']:.2f}")
            print(f"  최대: {stats['max']:.2f}")
            print(f"  평균: {stats['avg']:.2f}")
```

---

## 연습 문제

### 문제 1: BLE 스캐너
1. 주변 BLE 장치를 스캔하는 프로그램을 작성하세요.
2. RSSI 값 기준으로 정렬하여 출력하세요.

### 문제 2: 심박수 모니터
1. 심박수 센서(Heart Rate Service: 0x180D)에 연결하세요.
2. 실시간 심박수를 콘솔에 출력하세요.

### 문제 3: 데이터 로깅
1. BLE 온도 센서 데이터를 수신하세요.
2. 데이터를 CSV 파일에 저장하세요.

---

## 다음 단계

- [06_MQTT_Protocol.md](06_MQTT_Protocol.md): BLE 데이터를 MQTT로 전송
- [10_Home_Automation_Project.md](10_Home_Automation_Project.md): BLE 스마트홈 프로젝트

---

*최종 업데이트: 2026-02-01*
