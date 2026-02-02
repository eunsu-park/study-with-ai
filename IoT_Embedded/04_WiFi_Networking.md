# 04. WiFi 네트워킹

## 학습 목표

- 라즈베리파이 WiFi 설정 방법 습득
- Python 소켓 프로그래밍 기초 이해
- ESP32 WiFi 개요 파악
- 네트워크 스캔 및 모니터링
- HTTP 클라이언트로 데이터 전송

---

## 1. 라즈베리파이 WiFi 설정

### 1.1 명령줄 WiFi 설정

```bash
# 현재 WiFi 상태 확인
iwconfig wlan0

# 사용 가능한 네트워크 스캔
sudo iwlist wlan0 scan | grep -E "ESSID|Quality"

# WiFi 연결 (nmcli 사용)
sudo nmcli dev wifi connect "SSID이름" password "비밀번호"

# 연결 상태 확인
nmcli connection show

# IP 주소 확인
ip addr show wlan0
```

### 1.2 wpa_supplicant 설정

```bash
# /etc/wpa_supplicant/wpa_supplicant.conf 편집
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

```conf
# /etc/wpa_supplicant/wpa_supplicant.conf
country=KR
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

# 기본 WPA2 네트워크
network={
    ssid="MyNetwork"
    psk="MyPassword"
    key_mgmt=WPA-PSK
}

# 숨겨진 네트워크
network={
    ssid="HiddenNetwork"
    scan_ssid=1
    psk="Password"
}

# 우선순위 설정 (높은 값 = 우선)
network={
    ssid="PreferredNetwork"
    psk="Password"
    priority=10
}
```

### 1.3 Python으로 WiFi 정보 조회

```python
#!/usr/bin/env python3
"""WiFi 연결 정보 조회"""

import subprocess
import re

def get_wifi_info() -> dict:
    """현재 WiFi 연결 정보 반환"""
    info = {}

    try:
        # SSID 조회
        result = subprocess.run(
            ['iwgetid', '-r'],
            capture_output=True,
            text=True
        )
        info['ssid'] = result.stdout.strip()

        # IP 주소 조회
        result = subprocess.run(
            ['hostname', '-I'],
            capture_output=True,
            text=True
        )
        ips = result.stdout.strip().split()
        info['ip_addresses'] = ips

        # 신호 강도 조회
        result = subprocess.run(
            ['iwconfig', 'wlan0'],
            capture_output=True,
            text=True
        )
        match = re.search(r'Signal level=(-?\d+)', result.stdout)
        if match:
            info['signal_dbm'] = int(match.group(1))

        # MAC 주소
        result = subprocess.run(
            ['cat', '/sys/class/net/wlan0/address'],
            capture_output=True,
            text=True
        )
        info['mac_address'] = result.stdout.strip()

    except Exception as e:
        info['error'] = str(e)

    return info

def get_wifi_networks() -> list:
    """주변 WiFi 네트워크 스캔"""
    networks = []

    try:
        result = subprocess.run(
            ['sudo', 'iwlist', 'wlan0', 'scan'],
            capture_output=True,
            text=True
        )

        current_network = {}
        for line in result.stdout.split('\n'):
            if 'ESSID:' in line:
                ssid = re.search(r'ESSID:"(.+)"', line)
                if ssid and current_network:
                    networks.append(current_network)
                current_network = {'ssid': ssid.group(1) if ssid else ''}

            elif 'Quality=' in line:
                quality = re.search(r'Quality=(\d+)/(\d+)', line)
                if quality:
                    current_network['quality'] = f"{quality.group(1)}/{quality.group(2)}"

                signal = re.search(r'Signal level=(-?\d+)', line)
                if signal:
                    current_network['signal_dbm'] = int(signal.group(1))

        if current_network:
            networks.append(current_network)

    except Exception as e:
        print(f"스캔 실패: {e}")

    return networks

if __name__ == "__main__":
    print("=== WiFi 연결 정보 ===")
    info = get_wifi_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n=== 주변 WiFi 네트워크 ===")
    networks = get_wifi_networks()
    for net in networks[:10]:  # 상위 10개만
        print(f"  {net.get('ssid', 'Unknown')}: {net.get('signal_dbm', 'N/A')} dBm")
```

---

## 2. Python 소켓 프로그래밍

### 2.1 소켓 기초

```
┌─────────────────────────────────────────────────────────────┐
│                    소켓 통신 흐름                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   클라이언트                              서버                │
│   ┌─────────┐                        ┌─────────┐            │
│   │ socket()│                        │ socket()│            │
│   └────┬────┘                        └────┬────┘            │
│        │                                  │                 │
│        │                             ┌────┴────┐            │
│        │                             │  bind() │            │
│        │                             └────┬────┘            │
│        │                             ┌────┴────┐            │
│        │                             │ listen()│            │
│        │                             └────┬────┘            │
│   ┌────┴────┐      연결 요청         ┌────┴────┐            │
│   │connect()│ ──────────────────────▶│ accept()│            │
│   └────┬────┘                        └────┬────┘            │
│        │                                  │                 │
│   ┌────┴────┐      데이터 송수신     ┌────┴────┐            │
│   │  send() │ ◀────────────────────▶│  recv() │            │
│   │  recv() │                        │  send() │            │
│   └────┬────┘                        └────┬────┘            │
│        │                                  │                 │
│   ┌────┴────┐                        ┌────┴────┐            │
│   │ close() │                        │ close() │            │
│   └─────────┘                        └─────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 TCP 서버

```python
#!/usr/bin/env python3
"""TCP 서버 - 센서 데이터 수신"""

import socket
import json
from datetime import datetime

HOST = '0.0.0.0'  # 모든 인터페이스
PORT = 9999

def start_tcp_server():
    """TCP 서버 시작"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # 주소 재사용 허용
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server.bind((HOST, PORT))
        server.listen(5)

        print(f"TCP 서버 시작: {HOST}:{PORT}")

        while True:
            client, address = server.accept()
            print(f"클라이언트 연결: {address}")

            with client:
                while True:
                    data = client.recv(1024)
                    if not data:
                        break

                    try:
                        # JSON 데이터 파싱
                        message = json.loads(data.decode('utf-8'))
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] 수신: {message}")

                        # 응답 전송
                        response = {
                            "status": "ok",
                            "received": message.get("sensor_id")
                        }
                        client.sendall(json.dumps(response).encode('utf-8'))

                    except json.JSONDecodeError:
                        print(f"잘못된 JSON: {data}")

            print(f"클라이언트 연결 종료: {address}")

if __name__ == "__main__":
    start_tcp_server()
```

### 2.3 TCP 클라이언트

```python
#!/usr/bin/env python3
"""TCP 클라이언트 - 센서 데이터 전송"""

import socket
import json
import time
import random

SERVER_HOST = '192.168.1.100'  # 서버 IP
SERVER_PORT = 9999

def send_sensor_data():
    """센서 데이터를 서버로 전송"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((SERVER_HOST, SERVER_PORT))
        print(f"서버 연결: {SERVER_HOST}:{SERVER_PORT}")

        sensor_id = "temp_sensor_01"

        try:
            while True:
                # 센서 데이터 생성
                data = {
                    "sensor_id": sensor_id,
                    "temperature": round(random.uniform(20, 30), 1),
                    "humidity": round(random.uniform(40, 70), 1),
                    "timestamp": time.time()
                }

                # 전송
                message = json.dumps(data).encode('utf-8')
                client.sendall(message)
                print(f"전송: {data}")

                # 응답 수신
                response = client.recv(1024)
                if response:
                    print(f"응답: {response.decode('utf-8')}")

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n연결 종료")

if __name__ == "__main__":
    send_sensor_data()
```

### 2.4 UDP 소켓

```python
#!/usr/bin/env python3
"""UDP 소켓 통신 (빠른 센서 데이터 전송)"""

import socket
import json
import time

# === UDP 서버 ===
def udp_server(port: int = 9998):
    """UDP 서버"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))

    print(f"UDP 서버 시작: 포트 {port}")

    while True:
        data, addr = sock.recvfrom(1024)
        message = json.loads(data.decode('utf-8'))
        print(f"[{addr}] {message}")

# === UDP 클라이언트 ===
def udp_client(server_ip: str, port: int = 9998):
    """UDP 클라이언트"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sensor_data = {
        "sensor_id": "motion_01",
        "motion_detected": True,
        "timestamp": time.time()
    }

    message = json.dumps(sensor_data).encode('utf-8')
    sock.sendto(message, (server_ip, port))
    print(f"전송 완료: {sensor_data}")
    sock.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        udp_server()
    else:
        udp_client('192.168.1.100')
```

---

## 3. ESP32 WiFi 개요

### 3.1 ESP32와 라즈베리파이 비교

| 특성 | ESP32 | Raspberry Pi |
|------|-------|--------------|
| **프로세서** | Xtensa 240MHz | ARM 1.5GHz |
| **RAM** | 520KB | 1-8GB |
| **OS** | FreeRTOS/없음 | Linux |
| **언어** | C/C++, MicroPython | Python, 모든 언어 |
| **WiFi** | 내장 | 내장 (Pi 3+) |
| **전력** | 낮음 (80mA) | 높음 (700mA+) |
| **용도** | 센서 노드 | 게이트웨이, 엣지 |

### 3.2 ESP32 MicroPython WiFi 예제

```python
# ESP32용 MicroPython 코드
# (참고용 - 라즈베리파이에서는 실행 불가)

import network
import time

def connect_wifi(ssid: str, password: str) -> str:
    """ESP32 WiFi 연결"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if not wlan.isconnected():
        print(f'WiFi 연결 중: {ssid}')
        wlan.connect(ssid, password)

        # 연결 대기
        timeout = 10
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1

    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f'연결됨! IP: {ip}')
        return ip
    else:
        print('연결 실패')
        return None

# 사용
# ip = connect_wifi("MySSID", "MyPassword")
```

### 3.3 라즈베리파이 - ESP32 통신 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│              라즈베리파이 - ESP32 통신                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐         WiFi         ┌──────────────┐    │
│   │              │◀────────────────────▶│              │    │
│   │  Raspberry   │                      │    ESP32     │    │
│   │     Pi       │                      │   센서 노드  │    │
│   │              │                      │              │    │
│   │  - MQTT      │       TCP/UDP        │  - 온도 센서 │    │
│   │    Broker    │◀────────────────────▶│  - 습도 센서 │    │
│   │  - 데이터    │                      │  - 모션 센서 │    │
│   │    수집      │       HTTP           │              │    │
│   │  - 분석      │◀────────────────────▶│  저전력      │    │
│   │              │                      │  동작        │    │
│   └──────────────┘                      └──────────────┘    │
│        │                                      │             │
│        │                                      │             │
│        ▼                                      ▼             │
│   ┌──────────────┐                      ┌──────────────┐    │
│   │   클라우드   │                      │   배터리     │    │
│   │   AWS/GCP    │                      │   동작 가능  │    │
│   └──────────────┘                      └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 네트워크 스캔 및 모니터링

### 4.1 네트워크 장치 스캔

```python
#!/usr/bin/env python3
"""네트워크 장치 스캔"""

import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
import socket

def get_local_network() -> str:
    """로컬 네트워크 주소 반환"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    # 네트워크 주소 추출 (예: 192.168.1.0/24)
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def ping_host(ip: str) -> dict | None:
    """단일 호스트 핑"""
    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', ip],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return {'ip': ip, 'status': 'up'}
    except:
        pass
    return None

def scan_network(network: str = None) -> list:
    """네트워크 전체 스캔"""
    if network is None:
        network = get_local_network()

    # IP 범위 생성
    base = '.'.join(network.split('.')[:-1])
    ips = [f"{base}.{i}" for i in range(1, 255)]

    print(f"스캔 중: {network}")

    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for result in executor.map(ping_host, ips):
            if result:
                results.append(result)
                print(f"  발견: {result['ip']}")

    return results

def get_hostname(ip: str) -> str:
    """IP 주소에서 호스트명 조회"""
    try:
        return socket.gethostbyaddr(ip)[0]
    except:
        return "Unknown"

if __name__ == "__main__":
    devices = scan_network()
    print(f"\n=== 발견된 장치: {len(devices)}개 ===")

    for device in devices:
        hostname = get_hostname(device['ip'])
        print(f"  {device['ip']:15} - {hostname}")
```

### 4.2 포트 스캔

```python
#!/usr/bin/env python3
"""간단한 포트 스캔"""

import socket
from concurrent.futures import ThreadPoolExecutor

COMMON_PORTS = {
    22: 'SSH',
    80: 'HTTP',
    443: 'HTTPS',
    1883: 'MQTT',
    3306: 'MySQL',
    5432: 'PostgreSQL',
    8080: 'HTTP-Alt',
    8883: 'MQTT-TLS'
}

def check_port(target: str, port: int) -> dict | None:
    """포트 열림 확인"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)

    try:
        result = sock.connect_ex((target, port))
        if result == 0:
            return {
                'port': port,
                'status': 'open',
                'service': COMMON_PORTS.get(port, 'unknown')
            }
    except:
        pass
    finally:
        sock.close()

    return None

def scan_ports(target: str, ports: list = None) -> list:
    """여러 포트 스캔"""
    if ports is None:
        ports = list(COMMON_PORTS.keys())

    print(f"포트 스캔: {target}")

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_port, target, port): port for port in ports}
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
                print(f"  포트 {result['port']} ({result['service']}): OPEN")

    return results

if __name__ == "__main__":
    target = input("스캔할 IP 주소: ")
    scan_ports(target)
```

---

## 5. HTTP 클라이언트

### 5.1 requests 라이브러리

```python
#!/usr/bin/env python3
"""HTTP 클라이언트 - 센서 데이터 전송"""

import requests
import time
import json

API_BASE = "http://192.168.1.100:5000/api"

def send_sensor_data(sensor_id: str, data: dict) -> bool:
    """센서 데이터 POST 전송"""
    url = f"{API_BASE}/sensors/{sensor_id}/data"

    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )

        if response.status_code == 201:
            print(f"데이터 전송 성공: {data}")
            return True
        else:
            print(f"전송 실패: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"네트워크 오류: {e}")
        return False

def get_sensor_config(sensor_id: str) -> dict | None:
    """센서 설정 조회"""
    url = f"{API_BASE}/sensors/{sensor_id}/config"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"조회 실패: {e}")

    return None

def periodic_reporting(sensor_id: str, interval: int = 10):
    """주기적 데이터 리포팅"""
    import random

    print(f"센서 {sensor_id} 리포팅 시작 (간격: {interval}초)")

    while True:
        data = {
            "temperature": round(random.uniform(20, 30), 1),
            "humidity": round(random.uniform(40, 70), 1),
            "timestamp": int(time.time())
        }

        send_sensor_data(sensor_id, data)
        time.sleep(interval)

if __name__ == "__main__":
    periodic_reporting("sensor_001", 10)
```

### 5.2 비동기 HTTP 클라이언트

```python
#!/usr/bin/env python3
"""비동기 HTTP 클라이언트 (aiohttp)"""

import asyncio
import aiohttp
import time

API_BASE = "http://192.168.1.100:5000/api"

async def send_data_async(session: aiohttp.ClientSession,
                          sensor_id: str,
                          data: dict) -> bool:
    """비동기 데이터 전송"""
    url = f"{API_BASE}/sensors/{sensor_id}/data"

    try:
        async with session.post(url, json=data) as response:
            if response.status == 201:
                return True
    except aiohttp.ClientError as e:
        print(f"오류: {e}")

    return False

async def batch_send(sensors: list, data_list: list):
    """여러 센서 데이터 동시 전송"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_data_async(session, sensor, data)
            for sensor, data in zip(sensors, data_list)
        ]

        results = await asyncio.gather(*tasks)
        success = sum(results)
        print(f"전송 완료: {success}/{len(results)}")

if __name__ == "__main__":
    sensors = ["sensor_001", "sensor_002", "sensor_003"]
    data_list = [
        {"temperature": 25.5, "timestamp": time.time()},
        {"temperature": 26.0, "timestamp": time.time()},
        {"temperature": 24.8, "timestamp": time.time()}
    ]

    asyncio.run(batch_send(sensors, data_list))
```

### 5.3 HTTP 클라이언트 with 재시도

```python
#!/usr/bin/env python3
"""재시도 로직이 있는 HTTP 클라이언트"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

def create_session_with_retry(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (500, 502, 503, 504)
) -> requests.Session:
    """재시도 설정이 된 세션 생성"""
    session = requests.Session()

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session

class IoTHttpClient:
    """IoT용 HTTP 클라이언트"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = create_session_with_retry()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'IoT-Sensor/1.0'
        })

    def send_data(self, endpoint: str, data: dict) -> dict:
        """데이터 전송"""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.post(url, json=data, timeout=10)
            response.raise_for_status()
            return {"success": True, "data": response.json()}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_config(self, endpoint: str) -> dict:
        """설정 조회"""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return {"success": True, "data": response.json()}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def close(self):
        """세션 종료"""
        self.session.close()

# 사용 예
if __name__ == "__main__":
    client = IoTHttpClient("http://192.168.1.100:5000/api")

    result = client.send_data("sensors/001/data", {
        "temperature": 25.5,
        "timestamp": time.time()
    })

    print(result)
    client.close()
```

---

## 연습 문제

### 문제 1: WiFi 모니터링
1. 현재 WiFi 연결 상태를 모니터링하는 스크립트를 작성하세요.
2. 신호 강도가 -70dBm 이하로 떨어지면 경고를 출력하세요.

### 문제 2: 로컬 서버
1. TCP 서버를 작성하여 센서 데이터를 수신하세요.
2. 수신된 데이터를 파일에 저장하세요.

### 문제 3: HTTP 리포터
1. 온도 센서 데이터를 주기적으로 HTTP POST하는 클라이언트를 작성하세요.
2. 네트워크 오류 시 재시도 로직을 구현하세요.

---

## 다음 단계

- [05_BLE_Connectivity.md](05_BLE_Connectivity.md): BLE 통신으로 저전력 센서 연결
- [06_MQTT_Protocol.md](06_MQTT_Protocol.md): MQTT로 효율적인 IoT 메시징

---

*최종 업데이트: 2026-02-01*
