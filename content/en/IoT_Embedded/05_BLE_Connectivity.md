# 05. BLE Connectivity

This lesson covers Bluetooth Low Energy (BLE) communication for IoT devices. We'll learn BLE protocol basics, GATT structure, Python BLE programming with the bleak library, and sensor data collection via BLE.

---

## 1. BLE Basics

### 1.1 BLE vs Classic Bluetooth

| Feature | Classic Bluetooth | BLE (Bluetooth Low Energy) |
|---------|------------------|---------------------------|
| **Purpose** | Audio streaming, file transfer | IoT sensors, wearables |
| **Power Consumption** | High (100mA) | Very low (10-50μA) |
| **Data Rate** | 1-3 Mbps | 125-1000 kbps |
| **Range** | 10-100m | 10-50m |
| **Connection Time** | ~6 seconds | ~6ms |
| **Use Cases** | Audio devices, smartphones | Fitness trackers, smart home sensors |

**BLE Advantages:**
- Ultra-low power → suitable for battery operation
- Fast connection establishment
- Simple device implementation
- Widely supported (smartphones, tablets, PCs)

### 1.2 BLE Protocol Stack

```
┌──────────────────────────────────┐
│     Application Layer            │
├──────────────────────────────────┤
│     GATT (Generic Attribute)     │  ← Service/Characteristic definitions
├──────────────────────────────────┤
│     ATT (Attribute Protocol)     │  ← Read/Write/Notify operations
├──────────────────────────────────┤
│     L2CAP (Logical Link Control) │  ← Data segmentation/reassembly
├──────────────────────────────────┤
│     Link Layer                   │  ← Advertising, connection management
├──────────────────────────────────┤
│     Physical Layer               │  ← 2.4GHz radio transmission
└──────────────────────────────────┘
```

### 1.3 GATT (Generic Attribute Profile) Structure

GATT defines how BLE devices exchange data using a hierarchical structure:

```
Device
  └── Service (UUID: 0x180F - Battery Service)
       ├── Characteristic (UUID: 0x2A19 - Battery Level)
       │    ├── Properties: Read, Notify
       │    ├── Value: 0x64 (100%)
       │    └── Descriptor (Client Characteristic Configuration)
       └── Characteristic (UUID: 0x2A1A - Battery Power State)
            └── Value: ...
```

**Key Concepts:**
- **Service**: Logical grouping of related characteristics (e.g., Heart Rate Service, Temperature Service)
- **Characteristic**: Individual data point (e.g., Heart Rate Measurement, Temperature Value)
- **Properties**: Read, Write, Notify, Indicate
- **UUID**: Unique identifier (16-bit for standard services, 128-bit for custom)

---

## 2. Python BLE Programming with bleak

### 2.1 bleak Library Installation

`bleak` is a cross-platform Python BLE library.

```bash
pip3 install bleak
```

**Dependencies (Linux):**

```bash
sudo apt install bluez
```

### 2.2 BLE Device Scanning

```python
import asyncio
from bleak import BleakScanner

async def scan_devices():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    print(f"\nFound {len(devices)} device(s):\n")
    for idx, device in enumerate(devices, 1):
        print(f"{idx}. {device.name or 'Unknown'}")
        print(f"   Address: {device.address}")
        print(f"   RSSI: {device.rssi} dBm")
        print()

if __name__ == "__main__":
    asyncio.run(scan_devices())
```

**Output Example:**

```
Scanning for BLE devices...

Found 3 device(s):

1. Mi Smart Band 5
   Address: A4:C1:38:XX:XX:XX
   RSSI: -56 dBm

2. Arduino Nano 33 BLE
   Address: 00:11:22:XX:XX:XX
   RSSI: -72 dBm

3. Unknown
   Address: F0:98:9D:XX:XX:XX
   RSSI: -85 dBm
```

### 2.3 Connect to BLE Device

```python
import asyncio
from bleak import BleakClient

async def connect_device(address):
    print(f"Connecting to {address}...")

    async with BleakClient(address) as client:
        print(f"Connected: {client.is_connected}")

        # List all services
        print("\nServices:")
        for service in client.services:
            print(f"\n[Service] {service.uuid}")
            for char in service.characteristics:
                print(f"  [Char] {char.uuid}")
                print(f"    Properties: {char.properties}")

if __name__ == "__main__":
    # Replace with your device address
    ADDRESS = "A4:C1:38:XX:XX:XX"
    asyncio.run(connect_device(ADDRESS))
```

### 2.4 Read Characteristic

```python
import asyncio
from bleak import BleakClient

# Standard UUIDs
BATTERY_SERVICE_UUID = "0000180f-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_CHAR_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

async def read_battery_level(address):
    async with BleakClient(address) as client:
        # Read battery level
        value = await client.read_gatt_char(BATTERY_LEVEL_CHAR_UUID)
        battery_level = int.from_bytes(value, byteorder='little')

        print(f"Battery Level: {battery_level}%")

if __name__ == "__main__":
    ADDRESS = "A4:C1:38:XX:XX:XX"
    asyncio.run(read_battery_level(ADDRESS))
```

### 2.5 Write Characteristic

```python
import asyncio
from bleak import BleakClient

async def write_data(address, char_uuid, data):
    async with BleakClient(address) as client:
        # Write data to characteristic
        await client.write_gatt_char(char_uuid, data)
        print(f"Data written: {data.hex()}")

if __name__ == "__main__":
    ADDRESS = "00:11:22:XX:XX:XX"
    CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

    # Send command (example: LED ON)
    command = bytes([0x01, 0xFF])
    asyncio.run(write_data(ADDRESS, CHAR_UUID, command))
```

### 2.6 Notification Handling

Notifications allow the device to push data without polling.

```python
import asyncio
from bleak import BleakClient

async def notification_handler(sender, data):
    """Callback for notifications"""
    print(f"Notification from {sender}: {data.hex()}")

    # Parse data (example: temperature sensor)
    if len(data) >= 2:
        temp = int.from_bytes(data[:2], byteorder='little') / 100.0
        print(f"Temperature: {temp:.2f}°C")

async def subscribe_notifications(address, char_uuid):
    async with BleakClient(address) as client:
        print(f"Connected to {address}")

        # Start notifications
        await client.start_notify(char_uuid, notification_handler)
        print(f"Subscribed to notifications on {char_uuid}")

        # Keep connection alive
        await asyncio.sleep(30)

        # Stop notifications
        await client.stop_notify(char_uuid)
        print("Unsubscribed from notifications")

if __name__ == "__main__":
    ADDRESS = "A4:C1:38:XX:XX:XX"
    CHAR_UUID = "00002a1c-0000-1000-8000-00805f9b34fb"  # Temperature Measurement

    asyncio.run(subscribe_notifications(ADDRESS, CHAR_UUID))
```

---

## 3. Standard BLE Services and Characteristics

### 3.1 Common Standard UUIDs

| Service | UUID | Description |
|---------|------|-------------|
| **Battery Service** | 0x180F | Battery information |
| **Device Information** | 0x180A | Manufacturer, model, firmware |
| **Heart Rate** | 0x180D | Heart rate monitoring |
| **Environmental Sensing** | 0x181A | Temperature, humidity, pressure |
| **Health Thermometer** | 0x1809 | Body temperature |

| Characteristic | UUID | Type | Description |
|----------------|------|------|-------------|
| **Battery Level** | 0x2A19 | uint8 | Battery percentage (0-100) |
| **Temperature** | 0x2A1C | sint16 | Temperature (×0.01°C) |
| **Humidity** | 0x2A6F | uint16 | Humidity (×0.01%) |
| **Manufacturer Name** | 0x2A29 | string | Manufacturer name |
| **Firmware Revision** | 0x2A26 | string | Firmware version |

### 3.2 Read Device Information

```python
import asyncio
from bleak import BleakClient

DEVICE_INFO_SERVICE = "0000180a-0000-1000-8000-00805f9b34fb"
MANUFACTURER_CHAR = "00002a29-0000-1000-8000-00805f9b34fb"
MODEL_CHAR = "00002a24-0000-1000-8000-00805f9b34fb"
FIRMWARE_CHAR = "00002a26-0000-1000-8000-00805f9b34fb"

async def read_device_info(address):
    async with BleakClient(address) as client:
        try:
            manufacturer = await client.read_gatt_char(MANUFACTURER_CHAR)
            print(f"Manufacturer: {manufacturer.decode('utf-8')}")
        except:
            print("Manufacturer: N/A")

        try:
            model = await client.read_gatt_char(MODEL_CHAR)
            print(f"Model: {model.decode('utf-8')}")
        except:
            print("Model: N/A")

        try:
            firmware = await client.read_gatt_char(FIRMWARE_CHAR)
            print(f"Firmware: {firmware.decode('utf-8')}")
        except:
            print("Firmware: N/A")

if __name__ == "__main__":
    ADDRESS = "A4:C1:38:XX:XX:XX"
    asyncio.run(read_device_info(ADDRESS))
```

---

## 4. Practical Project: BLE Sensor Monitor

Comprehensive BLE sensor monitoring system with reconnection logic.

```python
import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BLESensorMonitor:
    def __init__(self, device_name=None, device_address=None):
        self.device_name = device_name
        self.device_address = device_address
        self.client = None
        self.running = False

        # Characteristic UUIDs (customize for your device)
        self.BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
        self.TEMP_UUID = "00002a1c-0000-1000-8000-00805f9b34fb"
        self.HUMIDITY_UUID = "00002a6f-0000-1000-8000-00805f9b34fb"

    async def find_device(self):
        """Find device by name or address"""
        if self.device_address:
            return self.device_address

        logger.info(f"Scanning for device: {self.device_name}")
        devices = await BleakScanner.discover(timeout=10.0)

        for device in devices:
            if device.name == self.device_name:
                logger.info(f"Found device: {device.address}")
                return device.address

        logger.error(f"Device '{self.device_name}' not found")
        return None

    async def notification_handler(self, sender, data):
        """Handle notifications from device"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if sender == self.TEMP_UUID:
            # Temperature (×0.01°C)
            temp = int.from_bytes(data[:2], byteorder='little', signed=True) / 100.0
            logger.info(f"[{timestamp}] Temperature: {temp:.2f}°C")

        elif sender == self.HUMIDITY_UUID:
            # Humidity (×0.01%)
            humidity = int.from_bytes(data[:2], byteorder='little') / 100.0
            logger.info(f"[{timestamp}] Humidity: {humidity:.1f}%")

        else:
            logger.info(f"[{timestamp}] Notification from {sender}: {data.hex()}")

    async def read_battery(self):
        """Read battery level"""
        try:
            value = await self.client.read_gatt_char(self.BATTERY_UUID)
            battery = int.from_bytes(value, byteorder='little')
            logger.info(f"Battery Level: {battery}%")
            return battery
        except Exception as e:
            logger.warning(f"Failed to read battery: {e}")
            return None

    async def connect(self):
        """Connect to device and subscribe to notifications"""
        address = await self.find_device()
        if not address:
            return False

        try:
            self.client = BleakClient(address)
            await self.client.connect()
            logger.info(f"Connected to {address}")

            # Read battery level
            await self.read_battery()

            # Subscribe to notifications
            try:
                await self.client.start_notify(self.TEMP_UUID, self.notification_handler)
                logger.info("Subscribed to temperature notifications")
            except:
                logger.warning("Temperature notifications not available")

            try:
                await self.client.start_notify(self.HUMIDITY_UUID, self.notification_handler)
                logger.info("Subscribed to humidity notifications")
            except:
                logger.warning("Humidity notifications not available")

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from device"""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            logger.info("Disconnected")

    async def run(self):
        """Main monitoring loop with auto-reconnect"""
        self.running = True

        while self.running:
            if not await self.connect():
                logger.info("Retrying in 10 seconds...")
                await asyncio.sleep(10)
                continue

            try:
                # Stay connected and handle notifications
                while self.running and self.client.is_connected:
                    await asyncio.sleep(1)

                    # Periodic battery check (every 60 seconds)
                    if int(asyncio.get_event_loop().time()) % 60 == 0:
                        await self.read_battery()

            except Exception as e:
                logger.error(f"Error during monitoring: {e}")

            finally:
                await self.disconnect()
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    def stop(self):
        """Stop monitoring"""
        self.running = False

async def main():
    # Method 1: Search by device name
    monitor = BLESensorMonitor(device_name="Arduino Nano 33 BLE")

    # Method 2: Direct connection with address
    # monitor = BLESensorMonitor(device_address="A4:C1:38:XX:XX:XX")

    try:
        await monitor.run()
    except KeyboardInterrupt:
        logger.info("\nStopping monitor...")
        monitor.stop()
        await monitor.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

**Features:**
- ✅ Device scanning by name or direct address connection
- ✅ Automatic reconnection on disconnect
- ✅ Notification handling for temperature/humidity
- ✅ Periodic battery level checks
- ✅ Comprehensive error handling and logging

---

## 5. BLE Security

### 5.1 Security Features

| Feature | Description |
|---------|-------------|
| **Pairing** | Authentication process between devices |
| **Bonding** | Storing encryption keys for future connections |
| **Encryption** | AES-128 CCM encryption for data transmission |
| **Privacy** | MAC address randomization to prevent tracking |

### 5.2 Pairing in Python

```python
import asyncio
from bleak import BleakClient

async def pair_device(address):
    async with BleakClient(address) as client:
        # Pairing is usually handled automatically by the OS
        # On Linux, you may need to use bluetoothctl:
        # $ bluetoothctl
        # [bluetooth]# pair A4:C1:38:XX:XX:XX

        paired = await client.pair()
        print(f"Pairing successful: {paired}")

        # Now you can access protected characteristics
        # ...

if __name__ == "__main__":
    ADDRESS = "A4:C1:38:XX:XX:XX"
    asyncio.run(pair_device(ADDRESS))
```

**Manual Pairing (Linux):**

```bash
# Start bluetoothctl
bluetoothctl

# Scan for devices
scan on

# Pair with device
pair A4:C1:38:XX:XX:XX

# Trust device (auto-connect in future)
trust A4:C1:38:XX:XX:XX

# Connect
connect A4:C1:38:XX:XX:XX
```

---

## 6. Troubleshooting

### 6.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **No devices found** | Bluetooth disabled, device out of range | Check `hciconfig`, move closer to device |
| **Connection timeout** | Device not advertising, already connected | Reset device, disconnect other connections |
| **Permission denied** | Insufficient privileges | Run with sudo or add user to bluetooth group |
| **Characteristic not found** | Wrong UUID, service not available | Verify UUID, check device documentation |
| **Notifications not working** | Client Characteristic Configuration Descriptor (CCCD) not enabled | Ensure start_notify is called correctly |

### 6.2 Bluetooth Service Management

```bash
# Check Bluetooth status
sudo systemctl status bluetooth

# Restart Bluetooth service
sudo systemctl restart bluetooth

# Check Bluetooth adapter
hciconfig

# Enable adapter
sudo hciconfig hci0 up

# Scan for devices (command line)
sudo hcitool lescan

# Reset Bluetooth adapter
sudo hciconfig hci0 reset
```

### 6.3 Debug Logging

```python
import asyncio
from bleak import BleakClient
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def debug_connection(address):
    async with BleakClient(address) as client:
        print(f"Connected: {client.is_connected}")

        # List all services and characteristics
        for service in client.services:
            print(f"\n[Service] {service.uuid}: {service.description}")
            for char in service.characteristics:
                print(f"  [Char] {char.uuid}")
                print(f"    Properties: {char.properties}")
                print(f"    Handle: {char.handle}")

                # Try to read if readable
                if "read" in char.properties:
                    try:
                        value = await client.read_gatt_char(char.uuid)
                        print(f"    Value: {value.hex()}")
                    except Exception as e:
                        print(f"    Read failed: {e}")

asyncio.run(debug_connection("A4:C1:38:XX:XX:XX"))
```

---

## 7. Summary

### Completed Tasks

- ✅ **BLE Basics**: BLE vs Classic Bluetooth, protocol stack, GATT structure
- ✅ **bleak Library**: Device scanning, connection, read/write/notify operations
- ✅ **Standard Services**: Battery Service, Environmental Sensing, Device Information
- ✅ **Practical Project**: Complete BLE sensor monitor with auto-reconnect
- ✅ **Security**: Pairing, encryption, privacy features
- ✅ **Troubleshooting**: Common issues and solutions

### Next Steps

| Next Lesson | Topic | Content |
|-------------|-------|---------|
| **06. MQTT Protocol** | IoT messaging protocol | Mosquitto broker, pub/sub patterns, QoS levels |
| **07. HTTP REST for IoT** | RESTful API design | Flask server, CRUD operations, request validation |
| **08. Edge AI with TFLite** | Machine learning on edge | TensorFlow Lite, model optimization, inference |

### Hands-On Exercises

1. **Heart Rate Monitor**:
   - Connect to BLE heart rate sensor
   - Display real-time heart rate
   - Log data to CSV file with timestamps

2. **Multi-Device Monitor**:
   - Connect to multiple BLE sensors simultaneously
   - Aggregate data from all sensors
   - Display unified dashboard

3. **BLE Proximity Alert**:
   - Monitor RSSI (signal strength)
   - Alert when device moves out of range
   - Implement geofencing logic

4. **Custom BLE Server** (Advanced):
   - Use Raspberry Pi as BLE peripheral
   - Expose GPIO state as BLE characteristics
   - Allow remote LED control via BLE

---

## References

- [bleak Documentation](https://bleak.readthedocs.io/)
- [Bluetooth SIG Specifications](https://www.bluetooth.com/specifications/gatt/)
- [GATT Services and Characteristics](https://www.bluetooth.com/specifications/assigned-numbers/)
- [BLE Security Guide](https://www.bluetooth.com/blog/bluetooth-pairing-part-1-pairing-feature-exchange/)
