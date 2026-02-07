# Other Application Protocols

## Overview

Besides HTTP, networks use various application layer protocols. This chapter covers DHCP, FTP, email protocols (SMTP, POP3, IMAP), SSH, Telnet, WebSocket, and more.

**Difficulty**: ⭐⭐

**Learning Objectives**:
- Understand each protocol's role and operating principles
- Familiarize with protocol-specific port numbers
- Identify security considerations
- Learn practical usage methods

---

## Table of Contents

1. [DHCP](#1-dhcp)
2. [FTP](#2-ftp)
3. [Email Protocols](#3-email-protocols)
4. [SSH](#4-ssh)
5. [Telnet](#5-telnet)
6. [WebSocket](#6-websocket)
7. [Practice Problems](#7-practice-problems)
8. [Next Steps](#8-next-steps)
9. [References](#9-references)

---

## 1. DHCP

### DHCP Overview

DHCP (Dynamic Host Configuration Protocol) is a protocol that automatically assigns IP addresses and network configurations to devices connected to a network.

```
┌─────────────────────────────────────────────────────────────────┐
│                      DHCP Basics                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ports: UDP 67 (server), UDP 68 (client)                        │
│  RFC: 2131                                                      │
│                                                                 │
│  Assigned Information:                                          │
│  - IP address                                                   │
│  - Subnet mask                                                  │
│  - Default gateway                                              │
│  - DNS server                                                   │
│  - Lease time                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DHCP DORA Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    DHCP DORA Process                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                            [DHCP Server]              │
│   (No IP)                            (192.168.1.1)              │
│      │                                        │                 │
│      │──(1) DHCP Discover ─────────────────▶ │                 │
│      │     Source: 0.0.0.0                   │                 │
│      │     Destination: 255.255.255.255 (broadcast)            │
│      │     "I need an IP address"            │                 │
│      │                                        │                 │
│      │◀─(2) DHCP Offer ─────────────────────│                 │
│      │     "I offer 192.168.1.100"           │                 │
│      │     Subnet: 255.255.255.0             │                 │
│      │     Gateway: 192.168.1.1              │                 │
│      │     DNS: 8.8.8.8                      │                 │
│      │     Lease time: 24 hours              │                 │
│      │                                        │                 │
│      │──(3) DHCP Request ──────────────────▶ │                 │
│      │     "I request 192.168.1.100"         │                 │
│      │                                        │                 │
│      │◀─(4) DHCP Ack ───────────────────────│                 │
│      │     "192.168.1.100 approved"          │                 │
│      │                                        │                 │
│  [IP Configuration Complete]                  │                 │
│  (192.168.1.100)                              │                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

DORA: Discover → Offer → Request → Acknowledge
```

### DHCP Message Types

| Message | Direction | Description |
|--------|------|------|
| DISCOVER | Client → Server | Request IP address (broadcast) |
| OFFER | Server → Client | Offer IP address |
| REQUEST | Client → Server | Accept offer or renew request |
| ACK | Server → Client | Approve request |
| NAK | Server → Client | Deny request |
| RELEASE | Client → Server | Release IP |
| INFORM | Client → Server | Request additional configuration |

### DHCP Lease Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    DHCP Lease Time                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Lease start                                                    │
│      │                                                          │
│      ├────────────────────────────────────── Lease time (T)     │
│      │                                                          │
│      ├──────────────── T/2 (50%)                                │
│      │                 │                                        │
│      │                 └─ Renewal attempt (Request to server)   │
│      │                    If successful: lease time extended    │
│      │                                                          │
│      ├──────────────────────────── T * 7/8 (87.5%)              │
│      │                             │                            │
│      │                             └─ Rebinding attempt         │
│      │                                (broadcast)               │
│      │                                                          │
│      └────────────────────────────────────── T (100%)           │
│                                              │                  │
│                                              └─ Lease expired   │
│                                                 IP released     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DHCP-Related Commands

```bash
# Linux - DHCP client
# Release IP
sudo dhclient -r eth0

# Renew IP
sudo dhclient eth0

# Verbose output
sudo dhclient -v eth0

# Windows
ipconfig /release     # Release IP
ipconfig /renew       # Renew IP

# macOS
sudo ipconfig set en0 DHCP    # Reset DHCP
```

---

## 2. FTP

### FTP Overview

FTP (File Transfer Protocol) is a protocol for transferring files between client and server.

```
┌─────────────────────────────────────────────────────────────────┐
│                      FTP Basic Info                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ports:                                                         │
│  - Control connection: TCP 21                                   │
│  - Data connection: TCP 20 (Active) or random port (Passive)    │
│                                                                 │
│  Features:                                                      │
│  - Uses two channels (control + data)                           │
│  - Plaintext transmission (security vulnerable)                 │
│  - ASCII/Binary transfer modes                                  │
│  - Active/Passive modes                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### FTP Active vs Passive Mode

```
┌─────────────────────────────────────────────────────────────────┐
│                    Active Mode                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                              [FTP Server]             │
│  (Random port N)                        (Port 21, 20)           │
│      │                                        │                 │
│      │──(1) Control connection (N → 21) ───▶ │                 │
│      │      "PORT N+1"                       │                 │
│      │                                        │                 │
│      │◀─(2) Data connection (20 → N+1) ──────│                 │
│      │      Server connects to client        │                 │
│      │                                        │                 │
│  ※ Problems occur if client firewall blocks external connections│
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Passive Mode                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                              [FTP Server]             │
│  (Random port)                         (Port 21, random port)   │
│      │                                        │                 │
│      │──(1) Control connection (N → 21) ───▶ │                 │
│      │      "PASV"                           │                 │
│      │                                        │                 │
│      │◀──── Response: Use port P ────────────│                 │
│      │                                        │                 │
│      │──(2) Data connection (N+1 → P) ───────▶ │                 │
│      │      Client connects to server        │                 │
│      │                                        │                 │
│  ※ Client always initiates connection → fewer firewall issues   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### FTP Main Commands

| Command | Description | Example |
|------|------|------|
| USER | Send username | `USER username` |
| PASS | Send password | `PASS password` |
| LIST | Directory listing | `LIST` |
| CWD | Change directory | `CWD /home/user` |
| PWD | Print working directory | `PWD` |
| RETR | Download file | `RETR file.txt` |
| STOR | Upload file | `STOR file.txt` |
| DELE | Delete file | `DELE file.txt` |
| MKD | Create directory | `MKD newdir` |
| RMD | Remove directory | `RMD olddir` |
| PASV | Passive mode | `PASV` |
| PORT | Active mode | `PORT 192,168,1,100,4,1` |
| QUIT | Close connection | `QUIT` |

### FTP Response Codes

| Code Range | Meaning | Examples |
|-----------|------|------|
| 1xx | Positive preliminary | 150 File status OK |
| 2xx | Positive completion | 200 Command OK, 226 Transfer complete |
| 3xx | Positive intermediate | 331 Username OK, password needed |
| 4xx | Transient negative | 421 Service unavailable |
| 5xx | Permanent negative | 530 Login failed, 550 No permission |

### FTP Client Usage

```bash
# Basic FTP client
ftp ftp.example.com

# Commands after connection
ftp> user username
ftp> pass password
ftp> ls              # List files
ftp> cd directory    # Change directory
ftp> get file.txt    # Download
ftp> put file.txt    # Upload
ftp> binary          # Binary mode
ftp> ascii           # ASCII mode
ftp> passive         # Toggle passive mode
ftp> bye             # Close connection
```

### SFTP and FTPS

```
┌─────────────────────────────────────────────────────────────────┐
│                FTP vs SFTP vs FTPS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FTP          │  SFTP              │  FTPS                      │
│  (Plaintext)  │  (SSH-based)       │  (TLS/SSL-based)           │
│               │                    │                            │
│  ┌─────────┐  │  ┌─────────┐       │  ┌─────────┐               │
│  │   FTP   │  │  │  SFTP   │       │  │   FTP   │               │
│  ├─────────┤  │  ├─────────┤       │  ├─────────┤               │
│  │   TCP   │  │  │   SSH   │       │  │ TLS/SSL │               │
│  └─────────┘  │  ├─────────┤       │  ├─────────┤               │
│               │  │   TCP   │       │  │   TCP   │               │
│  Port: 21     │  └─────────┘       │  └─────────┘               │
│               │                    │                            │
│  Security: None│ Port: 22          │  Port: 990 (implicit)      │
│               │  Security: SSH     │        21 (explicit)       │
│               │  encryption        │  Security: TLS encryption  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Recommendation: Use SFTP (SSH-based, single port)
```

---

## 3. Email Protocols

### Email Transmission Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Email Transmission Process                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Sender]     [Sending Server]   [Receiving Server] [Recipient]│
│  alice@a.com   mail.a.com        mail.b.com    bob@b.com       │
│      │            │                  │            │            │
│      │──(1) SMTP─▶│                  │            │            │
│      │   (compose)│                  │            │            │
│      │            │                  │            │            │
│      │            │──(2) SMTP───────▶│            │            │
│      │            │   (server-to-server)          │            │
│      │            │                  │            │            │
│      │            │                  │──(3) Store─▶│            │
│      │            │                  │   (mailbox)│            │
│      │            │                  │            │            │
│      │            │                  │◀─(4) POP3/IMAP──│       │
│      │            │                  │   (retrieve) │          │
│                                                                 │
│  Protocols:                                                     │
│  - SMTP: Mail transmission (sending)                            │
│  - POP3: Mail retrieval (download and delete)                   │
│  - IMAP: Mail retrieval (keep on server)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SMTP (Simple Mail Transfer Protocol)

```
┌─────────────────────────────────────────────────────────────────┐
│                      SMTP Information                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ports: TCP 25 (default), 587 (submission), 465 (SSL)          │
│  RFC: 5321                                                      │
│  Purpose: Email transmission                                    │
│                                                                 │
│  SMTP Session Example:                                          │
│                                                                 │
│  C: Connect (TCP 25)                                            │
│  S: 220 mail.example.com ESMTP ready                            │
│  C: EHLO client.example.com                                     │
│  S: 250-mail.example.com Hello                                  │
│  S: 250-AUTH LOGIN PLAIN                                        │
│  S: 250 STARTTLS                                                │
│  C: MAIL FROM:<alice@example.com>                               │
│  S: 250 OK                                                      │
│  C: RCPT TO:<bob@example.com>                                   │
│  S: 250 OK                                                      │
│  C: DATA                                                        │
│  S: 354 Start mail input                                        │
│  C: Subject: Test                                               │
│  C:                                                             │
│  C: Hello, this is a test.                                      │
│  C: .                                                           │
│  S: 250 OK Message queued                                       │
│  C: QUIT                                                        │
│  S: 221 Bye                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SMTP Main Commands

| Command | Description |
|------|------|
| HELO/EHLO | Client identification (EHLO is extended SMTP) |
| MAIL FROM | Specify sender |
| RCPT TO | Specify recipient |
| DATA | Start mail content |
| QUIT | Close connection |
| AUTH | Authentication |
| STARTTLS | Start TLS encryption |
| RSET | Reset transaction |

### POP3 (Post Office Protocol v3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      POP3 Information                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ports: TCP 110 (default), 995 (SSL/TLS)                        │
│  RFC: 1939                                                      │
│  Purpose: Email retrieval (download)                            │
│                                                                 │
│  Features:                                                      │
│  - Downloads mail locally                                       │
│  - Deletes from server (by default)                             │
│  - Suitable for single device                                   │
│  - Offline reading possible                                     │
│                                                                 │
│  POP3 Session Example:                                          │
│                                                                 │
│  S: +OK POP3 server ready                                       │
│  C: USER alice                                                  │
│  S: +OK                                                         │
│  C: PASS password123                                            │
│  S: +OK Logged in                                               │
│  C: STAT                                                        │
│  S: +OK 3 1024                                                  │
│  C: LIST                                                        │
│  S: +OK 3 messages                                              │
│  S: 1 512                                                       │
│  S: 2 256                                                       │
│  S: 3 256                                                       │
│  S: .                                                           │
│  C: RETR 1                                                      │
│  S: +OK 512 octets                                              │
│  S: (mail content)                                              │
│  S: .                                                           │
│  C: DELE 1                                                      │
│  S: +OK Deleted                                                 │
│  C: QUIT                                                        │
│  S: +OK Bye                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### IMAP (Internet Message Access Protocol)

```
┌─────────────────────────────────────────────────────────────────┐
│                      IMAP Information                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ports: TCP 143 (default), 993 (SSL/TLS)                        │
│  RFC: 3501 (IMAP4)                                              │
│  Purpose: Email retrieval (server synchronization)              │
│                                                                 │
│  Features:                                                      │
│  - Stores mail on server                                        │
│  - Synchronizes across multiple devices                         │
│  - Folder management available                                  │
│  - Supports partial download                                    │
│  - Requires online connection                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### POP3 vs IMAP Comparison

| Feature | POP3 | IMAP |
|------|------|------|
| Mail storage location | Local (download) | Server |
| Multi-device sync | Difficult | Supported |
| Offline reading | Easy | Limited |
| Server storage space | Small | Large needed |
| Folder management | Limited | Supported |
| Partial download | Not possible | Possible |
| Best use case | Single device | Multiple devices |

### Email Port Summary

| Protocol | Port | Security |
|----------|------|------|
| SMTP | 25 | Plaintext |
| SMTP Submission | 587 | STARTTLS |
| SMTPS | 465 | SSL/TLS |
| POP3 | 110 | Plaintext |
| POP3S | 995 | SSL/TLS |
| IMAP | 143 | Plaintext |
| IMAPS | 993 | SSL/TLS |

---

## 4. SSH

### SSH Overview

SSH (Secure Shell) is a protocol that provides encrypted remote access over networks.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SSH Information                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Port: TCP 22                                                   │
│  RFC: 4251-4256                                                 │
│  Version: SSH-2 (recommended), SSH-1 (deprecated)               │
│                                                                 │
│  Features:                                                      │
│  - Encrypted remote shell access                                │
│  - File transfer (SCP, SFTP)                                    │
│  - Port forwarding/tunneling                                    │
│  - X11 forwarding                                               │
│                                                                 │
│  Authentication methods:                                        │
│  - Password authentication                                      │
│  - Public key authentication (recommended)                      │
│  - Host-based authentication                                    │
│  - Keyboard-interactive                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SSH Connection Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSH Connection Process                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                              [Server]                 │
│      │                                        │                 │
│      │──(1) TCP connection (port 22) ───────▶ │                 │
│      │                                        │                 │
│      │◀─(2) Protocol version exchange ───────│                 │
│      │     SSH-2.0-OpenSSH_8.9               │                 │
│      │                                        │                 │
│      │◀──(3) Key exchange algorithm negotiation─▶│              │
│      │     Encryption, MAC, compression      │                 │
│      │                                        │                 │
│      │◀──(4) Key exchange (DH/ECDH) ─────────▶│                 │
│      │     Generate session key              │                 │
│      │                                        │                 │
│      │◀════(5) Start encrypted communication══▶│                 │
│      │                                        │                 │
│      │──(6) User authentication ──────────▶  │                 │
│      │     (password or public key)          │                 │
│      │                                        │                 │
│      │◀─(7) Authentication success ──────────│                 │
│      │                                        │                 │
│      │◀════(8) Start shell session ═══════════▶│                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SSH Public Key Authentication

```
┌─────────────────────────────────────────────────────────────────┐
│                SSH Public Key Authentication                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                              [Server]                 │
│                                                                 │
│  Private key (id_ed25519)        Public key (authorized_keys)   │
│  ┌──────────────────┐               ┌──────────────────┐        │
│  │ Never disclose   │               │ Client public    │        │
│  │ Keep local only  │               │ key stored       │        │
│  └──────────────────┘               └──────────────────┘        │
│           │                                   │                 │
│           │                                   │                 │
│  Authentication process:                                        │
│  1. Server sends random challenge                               │
│  2. Client signs challenge with private key                     │
│  3. Server verifies signature with public key                   │
│  4. Authentication succeeds if verification passes              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SSH Main Commands

```bash
# Basic connection
ssh user@hostname
ssh -p 2222 user@hostname    # Specify port

# Generate keys
ssh-keygen -t ed25519 -C "email@example.com"
ssh-keygen -t rsa -b 4096

# Copy public key
ssh-copy-id user@hostname
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@hostname

# File transfer (SCP)
scp file.txt user@host:/path/
scp user@host:/path/file.txt ./
scp -r directory/ user@host:/path/

# SFTP
sftp user@hostname
```

### SSH Tunneling

```bash
# Local port forwarding
# Local 8080 → via remote server → target server 80
ssh -L 8080:target.example.com:80 user@jump.example.com

# Remote port forwarding
# Remote server 8080 → local machine 80
ssh -R 8080:localhost:80 user@remote.example.com

# Dynamic port forwarding (SOCKS proxy)
ssh -D 1080 user@proxy.example.com
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSH Local Port Forwarding                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ssh -L 8080:db.internal:3306 user@jump.example.com             │
│                                                                 │
│  [Local PC]          [Jump Server]          [DB Server]         │
│  localhost:8080      jump.example.com    db.internal:3306       │
│      │                    │                   │                 │
│      │════ SSH Tunnel ════▶│                   │                 │
│      │                    │──────────────────▶│                 │
│      │                    │                   │                 │
│      │◀═══════════════════│◀──────────────────│                 │
│                                                                 │
│  Usage: mysql -h 127.0.0.1 -P 8080 (connect to DB locally)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Telnet

### Telnet Overview

Telnet provides terminal access to remote hosts. **SSH is recommended for security**.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Telnet Information                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Port: TCP 23                                                   │
│  RFC: 854                                                       │
│                                                                 │
│  Features:                                                      │
│  - Plaintext transmission (no encryption)                       │
│  - Password exposure risk                                       │
│  - Vulnerable to sniffing                                       │
│  - Legacy systems only                                          │
│                                                                 │
│  ⚠️  Security Warning:                                          │
│  - Do NOT use Telnet on the Internet                            │
│  - Replace with SSH                                             │
│  - Avoid even on internal networks                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Telnet Usage (Testing Only)

Telnet is unsuitable for remote access due to security, but useful for port connectivity testing.

```bash
# Port connectivity test
telnet example.com 80
telnet example.com 443

# HTTP test
telnet example.com 80
GET / HTTP/1.1
Host: example.com
(blank line)

# SMTP test
telnet mail.example.com 25
EHLO test
QUIT

# Connection check only (nc recommended)
nc -zv example.com 80
nc -zv example.com 22
```

### Telnet vs SSH

| Feature | Telnet | SSH |
|------|--------|-----|
| Port | 23 | 22 |
| Encryption | None (plaintext) | Yes |
| Authentication | Plaintext password | Various methods |
| Security | Very vulnerable | Strong |
| File transfer | None | SCP, SFTP |
| Tunneling | None | Supported |
| Recommendation | Testing only | Always recommended |

---

## 6. WebSocket

### WebSocket Overview

WebSocket is a protocol that provides bidirectional real-time communication between client and server.

```
┌─────────────────────────────────────────────────────────────────┐
│                    WebSocket Information                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ports: TCP 80 (ws://), TCP 443 (wss://)                        │
│  RFC: 6455                                                      │
│                                                                 │
│  Features:                                                      │
│  - Starts with HTTP upgrade                                     │
│  - Full-duplex bidirectional communication                      │
│  - Persistent connection                                        │
│  - Low overhead (minimal headers)                               │
│  - Real-time data transmission                                  │
│                                                                 │
│  Use cases:                                                     │
│  - Real-time chat                                               │
│  - Online gaming                                                │
│  - Stock/cryptocurrency prices                                  │
│  - Live streaming                                               │
│  - Collaboration tools                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HTTP vs WebSocket

```
┌─────────────────────────────────────────────────────────────────┐
│                  HTTP vs WebSocket                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HTTP (Request-Response)          WebSocket (Bidirectional)     │
│                                                                 │
│  Client        Server             Client        Server          │
│      │            │                  │            │             │
│      │──Request1─▶│                  │══Connected═│             │
│      │◀─Response1─│                  │            │             │
│      │            │                  │◀─ Message ─│             │
│      │──Request2─▶│                  │── Message─▶│             │
│      │◀─Response2─│                  │◀─ Message ─│             │
│      │            │                  │── Message─▶│             │
│      │──Request3─▶│                  │            │             │
│      │◀─Response3─│                  │◀─ Message ─│             │
│                                     │══Closed════│              │
│                                                                 │
│  Create connection each time        Once connected, persistent  │
│  Only client can request            Both sides can send         │
│  Request header overhead            Minimal frame header        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### WebSocket Handshake

```
┌─────────────────────────────────────────────────────────────────┐
│                  WebSocket Handshake                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client Request - HTTP Upgrade]                                │
│  ─────────────────────────────────                              │
│  GET /chat HTTP/1.1                                             │
│  Host: server.example.com                                       │
│  Upgrade: websocket                                             │
│  Connection: Upgrade                                            │
│  Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==                   │
│  Sec-WebSocket-Version: 13                                      │
│                                                                 │
│  [Server Response - 101 Switching Protocols]                    │
│  ──────────────────────────────────────                         │
│  HTTP/1.1 101 Switching Protocols                               │
│  Upgrade: websocket                                             │
│  Connection: Upgrade                                            │
│  Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=             │
│                                                                 │
│  [Then communicate via WebSocket frames]                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### WebSocket Client (JavaScript)

```javascript
// WebSocket connection
const socket = new WebSocket('wss://example.com/chat');

// Connection opened
socket.onopen = function(event) {
    console.log('WebSocket connected');
    socket.send('Hello Server!');
};

// Receive message
socket.onmessage = function(event) {
    console.log('Message received:', event.data);
};

// Connection closed
socket.onclose = function(event) {
    console.log('Connection closed:', event.code, event.reason);
};

// Error handling
socket.onerror = function(error) {
    console.error('WebSocket error:', error);
};

// Send message
socket.send(JSON.stringify({ type: 'chat', message: 'Hello!' }));

// Close connection
socket.close();
```

### WebSocket Server (Node.js)

```javascript
const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', function(ws) {
    console.log('Client connected');

    // Receive message
    ws.on('message', function(message) {
        console.log('Received message:', message);

        // Echo response
        ws.send('Server response: ' + message);

        // Broadcast to all clients
        wss.clients.forEach(function(client) {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        });
    });

    ws.on('close', function() {
        console.log('Client connection closed');
    });
});
```

---

## 7. Practice Problems

### Basic Problems

1. **DHCP**
   - Explain each step of DORA.
   - List all information assigned by DHCP server.

2. **FTP**
   - What's the difference between Active and Passive modes?
   - Explain FTP's security issues and alternatives.

3. **Email**
   - Explain the roles of SMTP, POP3, and IMAP respectively.
   - What's the difference between POP3 and IMAP?

### Intermediate Problems

4. **SSH**
   - What are the advantages of SSH public key authentication?
   - Provide an example scenario for using SSH local port forwarding.

5. **Port Numbers**
   - Write the port numbers for these protocols:
     - DHCP (client/server)
     - FTP (control/data)
     - SMTP (default/SSL)
     - SSH
     - POP3S
     - IMAPS

6. **Practical Problems**

```bash
# Predict the results of these commands

# 1. Connect to remote DB via SSH tunnel
ssh -L 3306:db.internal:3306 user@bastion.example.com
mysql -h 127.0.0.1 -P 3306 -u dbuser -p

# 2. Upload file via SFTP
sftp user@server.example.com
put localfile.txt /home/user/

# 3. Test HTTP with Telnet
telnet example.com 80
GET / HTTP/1.1
Host: example.com
```

### Advanced Problems

7. **WebSocket**
   - What's the difference between HTTP Polling and WebSocket?
   - Why does WebSocket handshake through HTTP?

8. **Security Comparison**
   - Explain the security issues of Telnet, FTP, and SMTP.
   - What are the secure alternatives for each?

---

## 8. Next Steps

In [15_Network_Security_Basics.md](./15_Network_Security_Basics.md), let's learn about network security basics including firewalls, NAT, and VPNs!

---

## 9. References

### RFC Documents

- [RFC 2131](https://tools.ietf.org/html/rfc2131) - DHCP
- [RFC 959](https://tools.ietf.org/html/rfc959) - FTP
- [RFC 5321](https://tools.ietf.org/html/rfc5321) - SMTP
- [RFC 1939](https://tools.ietf.org/html/rfc1939) - POP3
- [RFC 3501](https://tools.ietf.org/html/rfc3501) - IMAP
- [RFC 4251-4256](https://tools.ietf.org/html/rfc4251) - SSH
- [RFC 6455](https://tools.ietf.org/html/rfc6455) - WebSocket

### Port Number Summary

| Protocol | Plaintext Port | Secure Port |
|----------|-----------|-----------|
| FTP | 21 (control), 20 (data) | 990 (FTPS) |
| SSH/SFTP | 22 | - |
| Telnet | 23 | - |
| SMTP | 25, 587 | 465 |
| DNS | 53 | 853 (DoT) |
| DHCP | 67/68 | - |
| HTTP | 80 | 443 (HTTPS) |
| POP3 | 110 | 995 |
| IMAP | 143 | 993 |
