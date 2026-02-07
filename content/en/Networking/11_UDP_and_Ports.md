# UDP and Ports

## Overview

This document covers UDP (User Datagram Protocol) and the concept of port numbers. You will understand UDP's characteristics in contrast to TCP and learn how ports are used in the transport layer.

**Difficulty**: ⭐⭐
**Estimated Learning Time**: 2 hours
**Prerequisites**: [10_TCP_Protocol.md](./10_TCP_Protocol.md)

---

## Table of Contents

1. [UDP Characteristics](#1-udp-characteristics)
2. [UDP Header Structure](#2-udp-header-structure)
3. [TCP vs UDP Comparison](#3-tcp-vs-udp-comparison)
4. [Port Number Concept](#4-port-number-concept)
5. [Port Number Ranges](#5-port-number-ranges)
6. [Sockets](#6-sockets)
7. [Practice Problems](#7-practice-problems)
8. [Next Steps](#8-next-steps)
9. [References](#9-references)

---

## 1. UDP Characteristics

### 1.1 UDP Basics

UDP (User Datagram Protocol) is a connectionless protocol for simple and fast transmission.

```
┌─────────────────────────────────────────────────────────────────┐
│                       UDP Characteristics                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Connectionless                                               │
│     - No connection setup/teardown                               │
│     - Sends data immediately without handshake                   │
│                                                                  │
│  2. Unreliable                                                   │
│     - No delivery guarantee                                      │
│     - No ordering guarantee                                      │
│     - No retransmission                                          │
│                                                                  │
│  3. Fast Transmission                                            │
│     - Minimal overhead                                           │
│     - No connection setup delay                                  │
│                                                                  │
│  4. Simple                                                       │
│     - Small header (8 bytes)                                     │
│     - No state maintenance required                              │
│                                                                  │
│  5. Broadcast/Multicast Support                                  │
│     - Can send to multiple recipients simultaneously             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 UDP Operation

```
UDP Data Transmission

┌─────────────────┐                    ┌─────────────────┐
│     Sender      │                    │    Receiver     │
│                 │                    │                 │
│  Application    │                    │  Application    │
│      │          │                    │      ▲          │
│      ▼          │                    │      │          │
│  ┌───────────┐  │                    │  ┌───────────┐  │
│  │   UDP     │  │     Datagram 1     │  │   UDP     │  │
│  │           │──┼────────────────────┼─►│           │  │
│  │ No state  │  │     Datagram 2     │  │ No state  │  │
│  │ No ACK    │──┼────────────────────┼─►│ No ACK    │  │
│  │           │  │     Datagram 3     │  │           │  │
│  │           │──┼─────────X (lost)   │  │           │  │
│  └───────────┘  │                    │  └───────────┘  │
│                 │                    │                 │
└─────────────────┘                    └─────────────────┘

Characteristics:
- Each datagram is independent
- No retransmission if lost
- Application handles reliability
```

### 1.3 UDP Use Cases

```
When UDP is Appropriate:

┌─────────────────────────────────────────────────────────────────┐
│ 1. Real-time Streaming                                          │
│    - Video, voice calls (VoIP)                                  │
│    - Delay is more problematic than some packet loss            │
│                                                                  │
│ 2. Gaming                                                       │
│    - Fast response is critical                                  │
│    - Old position data is meaningless                           │
│                                                                  │
│ 3. DNS Queries                                                  │
│    - Single request/response                                    │
│    - Connection setup overhead unnecessary                      │
│                                                                  │
│ 4. DHCP                                                         │
│    - Broadcast required                                         │
│                                                                  │
│ 5. IoT / Sensor Data                                            │
│    - Large volume of small messages                             │
│    - Some loss acceptable                                       │
│                                                                  │
│ 6. SNMP (Network Management)                                    │
│    - Simple request/response                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 UDP Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Fast transmission speed | No delivery guarantee |
| Low overhead | No ordering guarantee |
| No connection setup needed | No congestion control |
| Multicast support | No flow control |
| Low server load | Security vulnerable (spoofing) |

---

## 2. UDP Header Structure

### 2.1 UDP Header Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source Port          |       Destination Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            Length             |           Checksum            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Data                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Total Header Size: 8 bytes (64 bits)
```

### 2.2 Header Field Descriptions

| Field | Size | Description |
|-------|------|-------------|
| Source Port | 16 bits | Sender port number (optional, can be 0) |
| Destination Port | 16 bits | Receiver port number |
| Length | 16 bits | Total length of UDP header + data (minimum 8) |
| Checksum | 16 bits | Error detection (optional in IPv4, mandatory in IPv6) |

### 2.3 UDP Checksum Calculation

```
UDP checksum is calculated including Pseudo Header

Pseudo Header (IPv4):
┌─────────────────────────────────────────────────────────────────┐
│                       Source IP Address                         │
├─────────────────────────────────────────────────────────────────┤
│                    Destination IP Address                       │
├────────────────┬─────────────────┬──────────────────────────────┤
│    Zero (8)    │  Protocol (17)  │        UDP Length            │
└────────────────┴─────────────────┴──────────────────────────────┘

Checksum calculation range:
1. Pseudo Header
2. UDP Header
3. UDP Data

Purpose:
- Verify IP header address information hasn't been modified
- Data integrity verification
```

### 2.4 UDP vs TCP Header Comparison

```
TCP Header (20-60 bytes):
┌────────────────────────────────────────────────────────────────┐
│ Src Port│Dst Port│  Seq Number  │  Ack Number  │Offset│Flags  │
│ Window  │Checksum│Urgent Pointer│    Options   │      │       │
└────────────────────────────────────────────────────────────────┘

UDP Header (8 bytes):
┌────────────────────────────────────────────────────────────────┐
│ Src Port│Dst Port│   Length    │   Checksum   │               │
└────────────────────────────────────────────────────────────────┘

Differences:
- TCP: Includes sequence number, ACK, flags, window, options, etc.
- UDP: Contains only minimal information (port, length, checksum)
```

---

## 3. TCP vs UDP Comparison

### 3.1 Detailed Comparison Table

| Characteristic | TCP | UDP |
|----------------|-----|-----|
| Connection Type | Connection-oriented (3-way handshake) | Connectionless |
| Reliability | Reliable (retransmission) | Unreliable (Best Effort) |
| Ordering | Ordering guaranteed (sequence number) | No ordering |
| Flow Control | Sliding window | None |
| Congestion Control | Slow Start, AIMD, etc. | None |
| Header Size | 20-60 bytes | 8 bytes |
| Transmission Unit | Segment | Datagram |
| Communication Pattern | 1:1 | 1:1, 1:N, N:N |
| Speed | Relatively slow | Fast |
| Overhead | High | Low |

### 3.2 Usage Scenario Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                      Protocol Selection Criteria                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Choose TCP:                                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Data integrity is critical (file transfer, email)      │  │
│  │ • Ordering is important (web pages, databases)           │  │
│  │ • Connection state management needed                     │  │
│  │ • Retransmission is essential                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Choose UDP:                                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Real-time performance is critical (streaming, gaming)  │  │
│  │ • Some loss is acceptable                                │  │
│  │ • Simple request/response (DNS)                          │  │
│  │ • Broadcast/multicast required                           │  │
│  │ • Custom reliability mechanism implemented               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Main Usage Examples by Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                          TCP Usage                               │
├───────────────────────────┬─────────────────────────────────────┤
│ HTTP/HTTPS (80/443)       │ Web browsing                        │
│ FTP (20/21)               │ File transfer                       │
│ SMTP (25)                 │ Email sending                       │
│ POP3 (110) / IMAP (143)   │ Email receiving                     │
│ SSH (22)                  │ Secure remote access                │
│ Telnet (23)               │ Remote access                       │
│ MySQL (3306)              │ Database                            │
│ PostgreSQL (5432)         │ Database                            │
└───────────────────────────┴─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          UDP Usage                               │
├───────────────────────────┬─────────────────────────────────────┤
│ DNS (53)                  │ Domain lookup                       │
│ DHCP (67/68)              │ IP automatic assignment             │
│ SNMP (161/162)            │ Network management                  │
│ NTP (123)                 │ Time synchronization                │
│ TFTP (69)                 │ Simple file transfer                │
│ RTP                       │ Real-time media streaming           │
│ VoIP (SIP)                │ Internet telephony                  │
│ Online Gaming             │ Real-time game data                 │
└───────────────────────────┴─────────────────────────────────────┘
```

### 3.4 Hybrid Approach

```
Using TCP and UDP Together:

1. Gaming
   ┌─────────────────────────────────────────────────────────────┐
   │  TCP: Login, chat, inventory (reliability needed)           │
   │  UDP: Character movement, real-time combat (speed needed)   │
   └─────────────────────────────────────────────────────────────┘

2. Streaming
   ┌─────────────────────────────────────────────────────────────┐
   │  TCP: Control channel (play/pause/volume)                   │
   │  UDP: Media data transmission (RTP)                         │
   └─────────────────────────────────────────────────────────────┘

3. QUIC (HTTP/3)
   ┌─────────────────────────────────────────────────────────────┐
   │  Reliability layer implemented on top of UDP                │
   │  Advantages: Fast connection, solves HOL Blocking           │
   └─────────────────────────────────────────────────────────────┘
```

---

## 4. Port Number Concept

### 4.1 Role of Ports

A port is a number that identifies a process running within a host.

```
Role of Ports

┌─────────────────────────────────────────────────────────────────┐
│                          Host                                    │
│                      192.168.1.100                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │  Web    │  │  SSH    │  │  FTP    │  │  MySQL  │    │    │
│  │  │ Server  │  │ Server  │  │ Server  │  │ Server  │    │    │
│  │  │         │  │         │  │         │  │         │    │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │    │
│  │       │            │            │            │          │    │
│  │    Port 80     Port 22     Port 21     Port 3306       │    │
│  │       │            │            │            │          │    │
│  │       └────────────┴────────────┴────────────┘          │    │
│  │                         │                                │    │
│  │              ┌──────────┴──────────┐                    │    │
│  │              │    TCP/IP Stack     │                    │    │
│  │              └──────────┬──────────┘                    │    │
│  │                         │                                │    │
│  └─────────────────────────┼────────────────────────────────┘    │
│                            │                                      │
│                     Network Interface                             │
│                      192.168.1.100                                │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                          Network

Checks destination port of each packet and forwards to corresponding process
```

### 4.2 Socket Address

In network communication, an endpoint is identified by a combination of IP address and port.

```
Socket Address

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Socket Address = IP Address + Port Number                      │
│                                                                  │
│  Examples:                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  192.168.1.100:80    (Web server)                        │   │
│  │  10.0.0.5:443        (HTTPS server)                      │   │
│  │  192.168.1.50:50000  (Client ephemeral port)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Unique identification of TCP connection:                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  (Source IP, Source Port, Dest IP, Dest Port, Protocol)  │   │
│  │  = 5-tuple                                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Port Number Notation

```
IPv4:
  IP:Port format
  Example: 192.168.1.100:80

IPv6:
  [IP]:Port format (wrap IP address in brackets)
  Example: [2001:db8::1]:80
           [::1]:8080

In URLs:
  http://example.com:8080/path
  https://[2001:db8::1]:443/
```

---

## 5. Port Number Ranges

### 5.1 Port Range Classification

```
Port Number Range (0 - 65535)

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Well-known Ports                                        │    │
│  │  0 - 1023                                                │    │
│  │  • System services and standard protocols                │    │
│  │  • Requires root/admin privileges                        │    │
│  │  • Managed by IANA                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Registered Ports                                        │    │
│  │  1024 - 49151                                            │    │
│  │  • For specific applications/services                    │    │
│  │  • Registered with IANA (not mandatory)                  │    │
│  │  • Can be used by regular users                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Dynamic/Private Ports                                   │    │
│  │  49152 - 65535                                           │    │
│  │  • Ephemeral ports                                       │    │
│  │  • Automatically assigned for client connections         │    │
│  │  • Cannot be registered                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Major Well-known Ports

| Port | Protocol | Service | Description |
|------|----------|---------|-------------|
| 20 | TCP | FTP-Data | FTP data transfer |
| 21 | TCP | FTP-Control | FTP control |
| 22 | TCP | SSH | Secure shell |
| 23 | TCP | Telnet | Remote access (unencrypted) |
| 25 | TCP | SMTP | Email sending |
| 53 | TCP/UDP | DNS | Domain Name Service |
| 67 | UDP | DHCP Server | IP automatic assignment (server) |
| 68 | UDP | DHCP Client | IP automatic assignment (client) |
| 69 | UDP | TFTP | Trivial file transfer |
| 80 | TCP | HTTP | Web (unencrypted) |
| 110 | TCP | POP3 | Email receiving |
| 123 | UDP | NTP | Time synchronization |
| 143 | TCP | IMAP | Email receiving |
| 161 | UDP | SNMP | Network management |
| 443 | TCP | HTTPS | Web (encrypted) |
| 445 | TCP | SMB | File sharing (Windows) |
| 465 | TCP | SMTPS | SMTP over SSL |
| 514 | UDP | Syslog | System logging |
| 993 | TCP | IMAPS | IMAP over SSL |
| 995 | TCP | POP3S | POP3 over SSL |

### 5.3 Major Registered Ports

| Port | Protocol | Service | Description |
|------|----------|---------|-------------|
| 1433 | TCP | MSSQL | Microsoft SQL Server |
| 1521 | TCP | Oracle | Oracle Database |
| 3306 | TCP | MySQL | MySQL Database |
| 3389 | TCP | RDP | Remote Desktop |
| 5432 | TCP | PostgreSQL | PostgreSQL Database |
| 5900 | TCP | VNC | Remote desktop |
| 6379 | TCP | Redis | Redis cache |
| 8080 | TCP | HTTP-Alt | Alternative HTTP port |
| 8443 | TCP | HTTPS-Alt | Alternative HTTPS port |
| 9000 | TCP | Various | PHP-FPM, etc. |
| 27017 | TCP | MongoDB | MongoDB Database |

### 5.4 Ephemeral Ports

```
Ephemeral port assignment during client connection

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Client                            Server                        │
│  (192.168.1.10)                   (10.0.0.5)                    │
│                                                                  │
│  ┌─────────────┐                  ┌─────────────┐               │
│  │ Web Browser │                  │  Web Server │               │
│  │  Port: ?    │─────────────────►│  Port: 80   │               │
│  └─────────────┘                  └─────────────┘               │
│                                                                  │
│  OS automatically assigns ephemeral port:                        │
│  Example: 192.168.1.10:52431 → 10.0.0.5:80                     │
│                                                                  │
│  Ephemeral port range by OS:                                    │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Linux:   32768 - 60999 (net.ipv4.ip_local_port_range) │     │
│  │ Windows: 49152 - 65535                                 │     │
│  │ macOS:   49152 - 65535                                 │     │
│  │ BSD:     1024 - 5000 (older versions)                  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Sockets

### 6.1 Socket Concept

A socket is an abstraction of the endpoint for network communication.

```
Socket Communication Model

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│     Application                         Application             │
│         │                                    │                  │
│    ┌────┴────┐                          ┌────┴────┐             │
│    │ Socket  │                          │ Socket  │             │
│    │ API     │                          │ API     │             │
│    └────┬────┘                          └────┬────┘             │
│         │                                    │                  │
│    ┌────┴────┐                          ┌────┴────┐             │
│    │ Socket  │◄═════════════════════════│ Socket  │             │
│    │192.168. │    TCP/UDP Connection    │10.0.0.5 │             │
│    │1.10:5000│                          │:80      │             │
│    └─────────┘                          └─────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Socket = (Protocol, IP Address, Port Number)
```

### 6.2 Socket Types

```
┌─────────────────────────────────────────────────────────────────┐
│                        Socket Types                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SOCK_STREAM (Stream Socket)                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Uses TCP                                                │  │
│  │ • Connection-oriented                                     │  │
│  │ • Reliable bidirectional byte stream                     │  │
│  │ • Ordering guaranteed                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  SOCK_DGRAM (Datagram Socket)                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Uses UDP                                                │  │
│  │ • Connectionless                                          │  │
│  │ • Fixed-size messages                                     │  │
│  │ • No ordering/delivery guarantee                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  SOCK_RAW (Raw Socket)                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Direct IP layer access                                  │  │
│  │ • Custom protocol implementation                          │  │
│  │ • Requires root privileges                                │  │
│  │ • Used for ping, traceroute, etc.                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 TCP Socket Programming Flow

```
TCP Server/Client Flow

       Server                             Client
         │                                    │
    socket()                             socket()
         │                                    │
      bind()                                  │
         │                                    │
     listen()                                 │
         │                                    │
     accept() ◄─────── connect() ─────────────┤
         │         (3-way handshake)          │
         │                                    │
      read() ◄──────── write() ──────────────┤
         │                                    │
     write() ────────► read()                 │
         │                                    │
     close() ◄──────── close() ──────────────┤
                   (4-way handshake)

Python Example (Server):
```

```python
import socket

# Create socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind address
server.bind(('0.0.0.0', 8080))

# Listen for connections
server.listen(5)

# Accept client connection
client, addr = server.accept()
print(f"Connected: {addr}")

# Send/receive data
data = client.recv(1024)
client.send(b"Hello, Client!")

# Close connection
client.close()
server.close()
```

### 6.4 UDP Socket Programming Flow

```
UDP Server/Client Flow

       Server                             Client
         │                                    │
    socket()                             socket()
         │                                    │
      bind()                                  │
         │                                    │
   recvfrom() ◄────── sendto() ──────────────┤
         │                                    │
    sendto() ─────────► recvfrom()            │
         │                                    │
     close()                              close()

Characteristics:
- No connect() needed (connectionless)
- Each message includes destination address
```

```python
import socket

# UDP Server
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('0.0.0.0', 9999))

data, addr = server.recvfrom(1024)
print(f"From {addr}: {data}")
server.sendto(b"ACK", addr)
```

### 6.5 Checking Socket State

```bash
# Linux - Check socket state
ss -tuln                    # TCP/UDP listening sockets
ss -tan                     # All TCP connections
ss -tan state established   # Established TCP only

# netstat (older version)
netstat -an | grep LISTEN
netstat -tunlp

# macOS
netstat -an | grep LISTEN
lsof -i -P | grep LISTEN

# Windows
netstat -an | findstr LISTEN
netstat -ano
```

---

## 7. Practice Problems

### Problem 1: TCP vs UDP Selection

Choose the appropriate protocol for the following scenarios.

a) Banking transaction system
b) Live streaming broadcast
c) Email transmission
d) Multiplayer game character position synchronization
e) Large file download
f) IoT sensor data collection (every second)

### Problem 2: Port Number Matching

Match the following services with their default port numbers.

```
Service:              Port:
a) HTTPS              1) 22
b) MySQL              2) 25
c) SMTP               3) 53
d) SSH                4) 443
e) DNS                5) 3306
```

### Problem 3: UDP Header Analysis

Analyze the following UDP header (hexadecimal).

```
01 BB 00 35 00 1C 8A 7E
```

a) What is the Source Port?
b) What is the Destination Port? (Which service?)
c) What is the UDP Length? (Data size?)
d) What is the Checksum?

### Problem 4: Socket Identification

A server is handling the following requests simultaneously.

```
Client A: 192.168.1.10:50001 → Server: 10.0.0.5:80
Client B: 192.168.1.10:50002 → Server: 10.0.0.5:80
Client C: 192.168.1.20:50001 → Server: 10.0.0.5:80
```

a) How does the server distinguish these three connections?
b) Express each connection using a 5-tuple.

---

## Answers

### Problem 1 Answers

a) Banking transaction → **TCP** (reliability required)
b) Live streaming → **UDP** (real-time important, some loss acceptable)
c) Email transmission → **TCP** (data integrity required)
d) Game character position → **UDP** (real-time, only latest data meaningful)
e) File download → **TCP** (complete data required)
f) IoT sensor data → **UDP** (frequent small messages, some loss acceptable)

### Problem 2 Answers

- a) HTTPS → 4) 443
- b) MySQL → 5) 3306
- c) SMTP → 2) 25
- d) SSH → 1) 22
- e) DNS → 3) 53

### Problem 3 Answers

```
01 BB 00 35 00 1C 8A 7E

a) Source Port: 0x01BB = 443 (HTTPS)
b) Destination Port: 0x0035 = 53 (DNS)
c) UDP Length: 0x001C = 28 bytes
   Data size: 28 - 8 = 20 bytes
d) Checksum: 0x8A7E
```

### Problem 4 Answers

a) Server distinguishes each connection using **5-tuple**:
   (Protocol, Src IP, Src Port, Dst IP, Dst Port)

b) 5-tuple representation:
   - Client A: (TCP, 192.168.1.10, 50001, 10.0.0.5, 80)
   - Client B: (TCP, 192.168.1.10, 50002, 10.0.0.5, 80)
   - Client C: (TCP, 192.168.1.20, 50001, 10.0.0.5, 80)

   All three connections are uniquely identified because either Src IP or Src Port differs.

---

## 8. Next Steps

After understanding UDP and ports, learn about DNS.

### Next Lesson
- [12_DNS.md](./12_DNS.md) - DNS operation principles, record types

### Related Lessons
- [10_TCP_Protocol.md](./10_TCP_Protocol.md) - TCP details
- [14_Other_Application_Protocols.md](./14_Other_Application_Protocols.md) - DHCP, SNMP

### Recommended Exercises
1. Check open ports on your system using `ss` or `netstat`
2. Capture and analyze UDP packets with Wireshark
3. Implement a simple TCP/UDP echo server

---

## 9. References

### RFC Documents

- RFC 768 - User Datagram Protocol
- RFC 793 - Transmission Control Protocol
- RFC 6335 - Internet Assigned Numbers Authority (IANA) Procedures

### Command Reference

```bash
# Port checking (Linux)
ss -tuln                     # Listening ports
ss -tan state established    # Established sockets
lsof -i :80                  # Process using specific port

# Port checking (macOS)
netstat -an | grep LISTEN
lsof -iTCP -sTCP:LISTEN

# Port checking (Windows)
netstat -an | findstr LISTENING
netstat -ano | findstr :80

# Port scanning
nmap -p 1-1000 target_ip     # TCP port scan
nmap -sU -p 53,67,123 target # UDP port scan

# UDP testing
nc -u target_ip 53           # UDP connection test
```

### Learning Resources

- [IANA Port Number Registry](https://www.iana.org/assignments/service-names-port-numbers/)
- [RFC 768 - UDP](https://tools.ietf.org/html/rfc768)
- Unix Network Programming - W. Richard Stevens

---

**Document Information**
- Last Modified: 2024
- Difficulty: ⭐⭐
- Estimated Learning Time: 2 hours
