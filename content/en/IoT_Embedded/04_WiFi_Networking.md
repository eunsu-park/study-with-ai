# 04. WiFi Networking

This lesson covers WiFi network configuration on Raspberry Pi and Python socket programming for IoT device communication. We'll learn network setup, TCP/UDP communication, and HTTP client implementation.

---

## 1. WiFi Network Setup

### 1.1 WiFi Configuration with wpa_supplicant

**Edit wpa_supplicant.conf:**

```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

**Add network configuration:**

```
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={
    ssid="YourNetworkName"
    psk="YourPassword"
    key_mgmt=WPA-PSK
}
```

**Apply configuration:**

```bash
sudo wpa_cli -i wlan0 reconfigure
```

**Check connection status:**

```bash
iwconfig wlan0
ping -c 4 8.8.8.8  # Test internet connectivity
```

### 1.2 Multiple Network Configuration

Configure multiple WiFi networks with priority:

```
network={
    ssid="HomeNetwork"
    psk="HomePassword"
    priority=10  # Higher priority
    id_str="home"
}

network={
    ssid="OfficeNetwork"
    psk="OfficePassword"
    priority=5
    id_str="office"
}

network={
    ssid="PublicWiFi"
    key_mgmt=NONE  # Open network (no password)
    priority=1
    id_str="public"
}
```

### 1.3 WiFi Network Scanning

```bash
# Scan available networks
sudo iwlist wlan0 scan

# Show SSIDs only
sudo iwlist wlan0 scan | grep ESSID
```

**Python network scanner:**

```python
import subprocess
import re

def scan_wifi():
    try:
        result = subprocess.check_output(['sudo', 'iwlist', 'wlan0', 'scan'],
                                        universal_newlines=True)

        # Parse SSIDs
        ssids = re.findall(r'ESSID:"(.+?)"', result)
        # Parse signal strength
        signals = re.findall(r'Signal level=(-\d+) dBm', result)

        networks = []
        for ssid, signal in zip(ssids, signals):
            networks.append({'ssid': ssid, 'signal': int(signal)})

        # Sort by signal strength
        networks.sort(key=lambda x: x['signal'], reverse=True)

        return networks

    except subprocess.CalledProcessError as e:
        print(f"Scan error: {e}")
        return []

if __name__ == "__main__":
    print("Scanning WiFi networks...")
    networks = scan_wifi()

    for idx, net in enumerate(networks, 1):
        print(f"{idx}. {net['ssid']:30s} | Signal: {net['signal']} dBm")
```

### 1.4 Network Information

```python
import socket
import subprocess

def get_ip_address(interface='wlan0'):
    """Get IP address for specified interface"""
    try:
        result = subprocess.check_output(['hostname', '-I'],
                                        universal_newlines=True)
        return result.strip().split()[0]
    except:
        return None

def get_mac_address(interface='wlan0'):
    """Get MAC address"""
    try:
        with open(f'/sys/class/net/{interface}/address', 'r') as f:
            return f.read().strip()
    except:
        return None

def get_network_info():
    hostname = socket.gethostname()
    ip = get_ip_address()
    mac = get_mac_address()

    print(f"Hostname: {hostname}")
    print(f"IP Address: {ip}")
    print(f"MAC Address: {mac}")

if __name__ == "__main__":
    get_network_info()
```

---

## 2. Python Socket Programming

### 2.1 TCP Server

TCP provides reliable, connection-oriented communication.

```python
import socket
import threading

class TCPServer:
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.server_socket = None

    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        print(f"[+] Connection from {address}")

        try:
            while True:
                # Receive data (max 1024 bytes)
                data = client_socket.recv(1024)
                if not data:
                    break

                message = data.decode('utf-8')
                print(f"[{address}] Received: {message}")

                # Echo back
                response = f"Echo: {message}"
                client_socket.send(response.encode('utf-8'))

        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"[-] Connection closed: {address}")

    def start(self):
        """Start TCP server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        print(f"[*] TCP Server listening on {self.host}:{self.port}")

        try:
            while True:
                client_socket, address = self.server_socket.accept()
                # Handle each client in separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()

        except KeyboardInterrupt:
            print("\n[*] Server shutting down...")
        finally:
            self.server_socket.close()

if __name__ == "__main__":
    server = TCPServer(host='0.0.0.0', port=8888)
    server.start()
```

### 2.2 TCP Client

```python
import socket

class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """Connect to server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            print(f"[+] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False

    def send_message(self, message):
        """Send message to server"""
        try:
            self.socket.send(message.encode('utf-8'))
            response = self.socket.recv(1024).decode('utf-8')
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            print("[-] Connection closed")

if __name__ == "__main__":
    client = TCPClient('192.168.1.100', 8888)

    if client.connect():
        try:
            while True:
                message = input("Enter message (or 'quit' to exit): ")
                if message.lower() == 'quit':
                    break

                response = client.send_message(message)
                if response:
                    print(f"Server response: {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            client.close()
```

### 2.3 UDP Communication

UDP is connectionless and suitable for real-time data streaming.

**UDP Server:**

```python
import socket

class UDPServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))

    def start(self):
        print(f"[*] UDP Server listening on {self.host}:{self.port}")

        try:
            while True:
                data, address = self.socket.recvfrom(1024)
                message = data.decode('utf-8')
                print(f"[{address}] Received: {message}")

                # Send response
                response = f"Received: {message}"
                self.socket.sendto(response.encode('utf-8'), address)

        except KeyboardInterrupt:
            print("\n[*] Server shutting down...")
        finally:
            self.socket.close()

if __name__ == "__main__":
    server = UDPServer()
    server.start()
```

**UDP Client:**

```python
import socket

class UDPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_message(self, message):
        try:
            self.socket.sendto(message.encode('utf-8'), (self.host, self.port))
            # Set timeout for response
            self.socket.settimeout(2.0)

            data, server = self.socket.recvfrom(1024)
            return data.decode('utf-8')

        except socket.timeout:
            return "No response (timeout)"
        except Exception as e:
            return f"Error: {e}"

    def close(self):
        self.socket.close()

if __name__ == "__main__":
    client = UDPClient('192.168.1.100', 9999)

    try:
        for i in range(5):
            message = f"Message {i+1}"
            response = client.send_message(message)
            print(f"Sent: {message} | Response: {response}")

    finally:
        client.close()
```

---

## 3. HTTP Client

### 3.1 requests Library

The `requests` library provides simple HTTP operations.

**Installation:**

```bash
pip3 install requests
```

**Basic GET request:**

```python
import requests

# GET request
response = requests.get('https://api.github.com')

print(f"Status Code: {response.status_code}")
print(f"Headers: {response.headers['Content-Type']}")
print(f"Response: {response.json()}")  # Parse JSON
```

**POST request:**

```python
import requests

# POST request with JSON data
url = 'https://httpbin.org/post'
data = {
    'sensor_id': 'temp001',
    'temperature': 24.5,
    'humidity': 60.2
}

response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

### 3.2 IoT Sensor Data Transmission

```python
import requests
import time
import random

class SensorDataSender:
    def __init__(self, api_url, device_id):
        self.api_url = api_url
        self.device_id = device_id

    def read_sensor(self):
        """Simulate sensor reading"""
        return {
            'device_id': self.device_id,
            'temperature': round(random.uniform(20.0, 30.0), 1),
            'humidity': round(random.uniform(40.0, 70.0), 1),
            'timestamp': time.time()
        }

    def send_data(self, data):
        """Send data to API"""
        try:
            response = requests.post(
                self.api_url,
                json=data,
                timeout=5
            )

            if response.status_code == 200:
                print(f"[✓] Data sent successfully: {data}")
                return True
            else:
                print(f"[✗] Send failed (Status {response.status_code})")
                return False

        except requests.exceptions.Timeout:
            print("[✗] Request timeout")
            return False
        except requests.exceptions.ConnectionError:
            print("[✗] Connection error")
            return False
        except Exception as e:
            print(f"[✗] Error: {e}")
            return False

    def run(self, interval=10):
        """Continuous data transmission"""
        print(f"Starting data transmission (interval: {interval}s)")

        try:
            while True:
                data = self.read_sensor()
                self.send_data(data)
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopped by user")

if __name__ == "__main__":
    sender = SensorDataSender(
        api_url='https://your-api-endpoint.com/sensor/data',
        device_id='RPi_001'
    )
    sender.run(interval=10)
```

### 3.3 HTTP Request with Retry Logic

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retry():
    """Create session with retry logic"""
    session = requests.Session()

    # Retry configuration
    retry = Retry(
        total=3,                      # Max retry count
        backoff_factor=1,             # Wait time: {backoff factor} * (2 ** (retry count - 1))
        status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session

# Usage example
session = create_session_with_retry()

try:
    response = session.post(
        'https://api.example.com/data',
        json={'sensor': 'temp', 'value': 25.3},
        timeout=5
    )
    print(f"Success: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request failed after retries: {e}")
```

### 3.4 Batch Data Transmission

```python
import requests
import time
from collections import deque

class BatchDataSender:
    def __init__(self, api_url, batch_size=10, send_interval=60):
        self.api_url = api_url
        self.batch_size = batch_size
        self.send_interval = send_interval
        self.buffer = deque(maxlen=100)  # Buffer with max 100 items

    def add_data(self, data):
        """Add data to buffer"""
        self.buffer.append(data)
        print(f"Data buffered (buffer size: {len(self.buffer)})")

        # Send when buffer reaches batch_size
        if len(self.buffer) >= self.batch_size:
            self.send_batch()

    def send_batch(self):
        """Send buffered data in batch"""
        if not self.buffer:
            return

        batch_data = list(self.buffer)
        self.buffer.clear()

        try:
            response = requests.post(
                self.api_url,
                json={'data': batch_data},
                timeout=10
            )

            if response.status_code == 200:
                print(f"[✓] Batch sent successfully ({len(batch_data)} items)")
            else:
                print(f"[✗] Batch send failed (Status {response.status_code})")
                # Re-buffer failed data
                self.buffer.extend(batch_data)

        except Exception as e:
            print(f"[✗] Send error: {e}")
            # Re-buffer failed data
            self.buffer.extend(batch_data)

    def run(self):
        """Periodic batch transmission"""
        print(f"Starting batch sender (interval: {self.send_interval}s)")

        try:
            while True:
                time.sleep(self.send_interval)
                self.send_batch()

        except KeyboardInterrupt:
            print("\nSending remaining data...")
            self.send_batch()

# Usage example
sender = BatchDataSender(
    api_url='https://api.example.com/batch',
    batch_size=10,
    send_interval=60
)

# Simulate data generation
import threading

def generate_data():
    import random
    while True:
        data = {
            'timestamp': time.time(),
            'temperature': round(random.uniform(20, 30), 1)
        }
        sender.add_data(data)
        time.sleep(5)

# Start data generation thread
data_thread = threading.Thread(target=generate_data, daemon=True)
data_thread.start()

# Start batch sender
sender.run()
```

---

## 4. Practical Example: Network Device Scanner

Scan IoT devices on local network:

```python
import socket
import concurrent.futures
import ipaddress

class NetworkScanner:
    def __init__(self, network='192.168.1.0/24'):
        self.network = network

    def check_port(self, ip, port, timeout=1):
        """Check if port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((str(ip), port))
            sock.close()
            return result == 0
        except:
            return False

    def scan_host(self, ip):
        """Scan common IoT ports on host"""
        common_ports = {
            80: 'HTTP',
            443: 'HTTPS',
            1883: 'MQTT',
            8883: 'MQTT-TLS',
            8080: 'HTTP-Alt',
            22: 'SSH',
            23: 'Telnet'
        }

        open_ports = []
        for port, service in common_ports.items():
            if self.check_port(ip, port, timeout=0.5):
                open_ports.append((port, service))

        return ip, open_ports

    def scan_network(self):
        """Scan entire network"""
        print(f"Scanning network: {self.network}")
        print("-" * 50)

        network = ipaddress.ip_network(self.network)
        devices = []

        # Parallel scanning with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(self.scan_host, ip): ip
                      for ip in network.hosts()}

            for future in concurrent.futures.as_completed(futures):
                ip, open_ports = future.result()
                if open_ports:
                    devices.append((ip, open_ports))
                    print(f"\n[+] Device found: {ip}")
                    for port, service in open_ports:
                        print(f"    Port {port:5d} ({service})")

        print("\n" + "=" * 50)
        print(f"Scan complete. Found {len(devices)} device(s)")

        return devices

if __name__ == "__main__":
    scanner = NetworkScanner('192.168.1.0/24')
    scanner.scan_network()
```

---

## 5. ESP32 Comparison

ESP32 is another popular IoT platform. Here's a comparison with Raspberry Pi:

| Feature | Raspberry Pi | ESP32 |
|---------|--------------|-------|
| **Type** | Single-board computer (Linux) | Microcontroller |
| **CPU** | ARM Cortex-A (multi-core GHz) | Xtensa dual-core (240MHz) |
| **Memory** | 1-8GB RAM | 520KB SRAM |
| **Connectivity** | WiFi, Ethernet, USB | WiFi, Bluetooth |
| **Power** | 5V, 2-3A (10-15W) | 3.3V, 0.5A (1.5W) |
| **GPIO** | 40 pins | 30+ pins |
| **Programming** | Python, C/C++, etc. | C/C++ (Arduino), MicroPython |
| **Use Cases** | Complex processing, AI, multi-task | Low power, real-time control, battery operation |
| **Price** | $35-75 | $5-15 |

**Selection Guidelines:**
- **Complex Processing/AI**: Raspberry Pi
- **Low Power/Battery**: ESP32
- **Real-Time Control**: ESP32
- **Networked Applications**: Both suitable

---

## 6. Summary

### Completed Tasks

- ✅ **WiFi Configuration**: wpa_supplicant setup, multiple networks
- ✅ **Socket Programming**: TCP/UDP server and client implementation
- ✅ **HTTP Communication**: requests library, retry logic, batch transmission
- ✅ **Network Scanning**: WiFi network scanner, IoT device scanner
- ✅ **Platform Comparison**: Raspberry Pi vs ESP32

### Next Steps

| Next Lesson | Topic | Content |
|-------------|-------|---------|
| **05. BLE Connectivity** | Bluetooth Low Energy | BLE protocol, GATT structure, sensor communication |
| **06. MQTT Protocol** | IoT messaging protocol | Mosquitto broker, pub/sub patterns, QoS levels |
| **07. HTTP REST for IoT** | RESTful API design | Flask server, API design, request validation |

### Hands-On Exercises

1. **Temperature Monitoring Server**:
   - Create TCP server on Raspberry Pi
   - Send simulated temperature data from client
   - Log data to file with timestamps

2. **Remote LED Control**:
   - Build HTTP server with `/led/on` and `/led/off` endpoints
   - Control GPIO LED via HTTP requests
   - Add status endpoint to check LED state

3. **Network Monitor**:
   - Scan local network every 5 minutes
   - Detect new devices
   - Send alert when unknown device connects

4. **Data Logger**:
   - Read sensor data every 10 seconds
   - Buffer data and send in batches of 20 items
   - Implement retry logic for failed transmissions

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Connection timeout** | Server unreachable or firewall | Check IP/port, verify firewall rules |
| **Address already in use** | Port not released | Use SO_REUSEADDR, wait or change port |
| **Broken pipe** | Client disconnected during send | Add exception handling, check connection |
| **DNS resolution failure** | Network not configured | Check /etc/resolv.conf, test with IP |

---

## References

- [Python socket Documentation](https://docs.python.org/3/library/socket.html)
- [requests Library](https://requests.readthedocs.io/)
- [Raspberry Pi Network Configuration](https://www.raspberrypi.org/documentation/configuration/wireless/)
- [wpa_supplicant Configuration](https://w1.fi/wpa_supplicant/)
