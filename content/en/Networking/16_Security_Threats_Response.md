# Security Threats and Response

## Overview

Network security threats continuously evolve. This chapter covers major network attack types, operating principles, and effective response strategies.

**Difficulty**: ⭐⭐⭐⭐

**Learning Objectives**:
- Understand major network attack types and principles
- Identify sniffing, spoofing, and DoS/DDoS attacks
- Learn web security threat concepts (SQL Injection, XSS)
- Understand Intrusion Detection/Prevention Systems (IDS/IPS)
- Establish effective security response strategies

---

## Table of Contents

1. [Network Security Threat Types](#1-network-security-threat-types)
2. [Sniffing](#2-sniffing)
3. [Spoofing](#3-spoofing)
4. [DoS/DDoS Attacks](#4-dosddos-attacks)
5. [MITM Attacks](#5-mitm-attacks)
6. [Web Security Threats](#6-web-security-threats)
7. [Intrusion Detection Systems](#7-intrusion-detection-systems)
8. [Security Response Strategies](#8-security-response-strategies)
9. [Practice Problems](#9-practice-problems)
10. [Next Steps](#10-next-steps)
11. [References](#11-references)

---

## 1. Network Security Threat Types

### Threat Classification System

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Threat Classification                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Attack Types                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Passive Attacks                                        │   │
│  │  └─ Eavesdropping, sniffing, traffic analysis           │   │
│  │     (Gathering information without data modification)   │   │
│  │                                                         │   │
│  │  Active Attacks                                         │   │
│  │  └─ Modification, forgery, denial of service            │   │
│  │     (Directly affecting data or systems)                │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Classification by Attack Target               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Network Layer Attacks                                  │   │
│  │  └─ IP spoofing, ICMP flooding, routing attacks         │   │
│  │                                                         │   │
│  │  Transport Layer Attacks                                │   │
│  │  └─ TCP SYN Flood, UDP Flood, session hijacking         │   │
│  │                                                         │   │
│  │  Application Layer Attacks                              │   │
│  │  └─ SQL Injection, XSS, CSRF                            │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Major Attack Types Summary

| Attack Type | Target | CIA Threat | Layer |
|----------|------|----------|------|
| Sniffing | Information gathering | Confidentiality | L2-L7 |
| Spoofing | Trust exploitation | Integrity, Authentication | L2-L4 |
| DoS/DDoS | Service disruption | Availability | L3-L7 |
| MITM | Interception | Confidentiality, Integrity | L2-L7 |
| SQL Injection | Data theft | Confidentiality, Integrity | L7 |
| XSS | User attack | Confidentiality | L7 |

---

## 2. Sniffing

### Sniffing Overview

Sniffing is a passive attack that intercepts network traffic to gather information.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sniffing Attack                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Normal communication:                                          │
│                                                                 │
│  [Client] ────────────────────────────────▶ [Server]           │
│               "User: admin, Password: 1234"                     │
│                                                                 │
│  Sniffing attack:                                               │
│                                                                 │
│  [Client] ────────────────────────────────▶ [Server]           │
│               "User: admin, Password: 1234"                     │
│                           │                                     │
│                           │ Eavesdropping                       │
│                           ▼                                     │
│                      [Attacker]                                 │
│                  "Credentials obtained!"                        │
│                                                                 │
│  Obtainable information:                                        │
│  - User account credentials                                     │
│  - Email content                                                │
│  - Financial information                                        │
│  - Session tokens                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sniffing Types

#### 1. Passive Sniffing (Hub Environment)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hub Environment Sniffing                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Hub broadcasts packets to all ports                            │
│                                                                 │
│        ┌────────────────────────────────┐                      │
│        │             Hub                │                      │
│        │  (Sends packets to all ports)  │                      │
│        └─┬──────┬──────┬──────┬────────┘                      │
│          │      │      │      │                                │
│          ▼      ▼      ▼      ▼                                │
│        ┌──┐   ┌──┐   ┌──┐   ┌───────┐                         │
│        │PC│   │PC│   │PC│   │Attacker│                        │
│        │A │   │B │   │C │   │   D   │                         │
│        └──┘   └──┘   └──┘   └───────┘                         │
│                              ↑                                 │
│                         Can receive                             │
│                         all traffic                             │
│                                                                 │
│  * Hubs are rarely used today                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Active Sniffing (Switch Environment)

```
┌─────────────────────────────────────────────────────────────────┐
│                Switch Environment Sniffing Techniques            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Switches use MAC address table to send to destination port only│
│  → Additional techniques required                               │
│                                                                 │
│  1. ARP Spoofing/Poisoning                                      │
│     - Manipulate MAC table with fake ARP responses              │
│     - Induce traffic to pass through attacker                   │
│                                                                 │
│  2. MAC Flooding                                                │
│     - Overflow switch table with fake MAC addresses             │
│     - Switch acts like a hub                                    │
│                                                                 │
│  3. SPAN/Mirror Port                                            │
│     - Abuse switch monitoring port (insider threat)             │
│                                                                 │
│  4. DHCP Spoofing                                               │
│     - Manipulate gateway with fake DHCP server                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ARP Spoofing Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARP Spoofing Attack                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Normal state:                                                  │
│                                                                 │
│  [Victim PC]              [Gateway]                             │
│  IP: 192.168.1.10        IP: 192.168.1.1                        │
│  MAC: AA:AA:AA           MAC: BB:BB:BB                          │
│       │                        │                                │
│       │◀─────── Normal communication ──────▶│                   │
│                                                                 │
│  After ARP spoofing:                                            │
│                                                                 │
│  [Victim PC]        [Attacker]         [Gateway]                │
│  192.168.1.10      192.168.1.100    192.168.1.1                 │
│  AA:AA:AA          CC:CC:CC          BB:BB:BB                   │
│       │                │                 │                      │
│       │                │                 │                      │
│  ARP Table:       Send fake ARP:      ARP Table:                │
│  ┌───────────┐    "192.168.1.1's      ┌───────────┐            │
│  │192.168.1.1│     MAC is CC:CC:CC"   │192.168.1.10│            │
│  │→ CC:CC:CC │    "192.168.1.10's     │→ CC:CC:CC │            │
│  └───────────┘     MAC is CC:CC:CC"   └───────────┘            │
│       │                │                 │                      │
│       │                │                 │                      │
│       └───────▶ Attacker ◀────────────────┘                     │
│                 (Relay and eavesdrop)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sniffing Countermeasures

| Countermeasure | Description |
|----------|------|
| Use encryption | HTTPS, SSH, VPN encrypted communication |
| Dynamic ARP Inspection (DAI) | Verify ARP packets on switch |
| Static ARP table | Fix ARP entries for critical servers |
| 802.1X | Port-based network access control |
| Network segregation | Isolate sensitive traffic with VLANs |
| IDS/IPS | Detect abnormal ARP traffic |

---

## 3. Spoofing

### Spoofing Overview

Spoofing is an attack that forges identity to exploit trust.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Spoofing Types                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                       Spoofing Attacks                     │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │                                                           │ │
│  │  IP Spoofing                                              │ │
│  │  └─ Forge source IP address                               │ │
│  │                                                           │ │
│  │  MAC Spoofing                                             │ │
│  │  └─ Forge source MAC address                              │ │
│  │                                                           │ │
│  │  ARP Spoofing                                             │ │
│  │  └─ Manipulate IP-MAC mapping with fake ARP responses     │ │
│  │                                                           │ │
│  │  DNS Spoofing                                             │ │
│  │  └─ Redirect to malicious server with fake DNS responses  │ │
│  │                                                           │ │
│  │  Email Spoofing                                           │ │
│  │  └─ Forge sender address                                  │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### IP Spoofing

```
┌─────────────────────────────────────────────────────────────────┐
│                    IP Spoofing                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Attacker]                                      [Server]       │
│  Real IP: 10.0.0.100                          192.168.1.1       │
│       │                                           │             │
│       │  Forged packet                            │             │
│       │  ┌─────────────────────────┐             │             │
│       │  │ Source: 192.168.1.50    │             │             │
│       │  │ (Forged, trusted IP)    │             │             │
│       │  │ Dest: 192.168.1.1       │             │             │
│       │  └─────────────────────────┘             │             │
│       │                                           │             │
│       └───────────────────────────────────────────▶             │
│                                                   │             │
│                    Response goes to forged IP      │            │
│  [Victim]◀────────────────────────────────────────┘             │
│  192.168.1.50                                                   │
│                                                                 │
│  Use cases:                                                     │
│  - DoS attacks (reflection attacks)                             │
│  - Bypass access control                                        │
│  - Evade logging                                                │
│                                                                 │
│  Limitations:                                                   │
│  - Difficult to establish TCP connection (3-way handshake)      │
│  - Cannot receive responses                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DNS Spoofing

```
┌─────────────────────────────────────────────────────────────────┐
│                    DNS Spoofing                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Normal DNS lookup:                                             │
│                                                                 │
│  [User]          [DNS Server]          [bank.com]               │
│     │                 │                  IP: 1.2.3.4            │
│     │──"bank.com?"───▶│                     │                   │
│     │                 │──── Query ────────▶│                   │
│     │◀─"1.2.3.4"──────│                     │                   │
│     │                                        │                   │
│     │───────── Normal access ───────────────▶│                   │
│                                                                 │
│  DNS spoofing attack:                                           │
│                                                                 │
│  [User]     [Attacker]     [DNS Server]    [bank.com]  [Malicious]│
│     │           │            │            │         IP:9.9.9.9  │
│     │──"bank.com?"─────────▶│            │            │        │
│     │           │            │            │            │        │
│     │◀─"9.9.9.9"─┘           │            │            │        │
│     │  (Fake response)       │            │            │        │
│     │     (Faster response)                             │        │
│     │                                                   │        │
│     │───────────────── Access malicious server ─────────▶│        │
│                        (Phishing site)                           │
│                                                                 │
│  Attack methods:                                                │
│  1. Manipulate DNS responses after ARP spoofing                 │
│  2. DNS cache poisoning                                         │
│  3. Modify local hosts file                                     │
│  4. Rogue DNS server (combined with DHCP spoofing)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Spoofing Countermeasures

| Spoofing Type | Countermeasure |
|------------|----------|
| IP Spoofing | Ingress/egress filtering, BCP38 |
| MAC Spoofing | 802.1X, port security |
| ARP Spoofing | DAI, static ARP, ARP watch |
| DNS Spoofing | DNSSEC, DoH/DoT, DNS monitoring |
| Email Spoofing | SPF, DKIM, DMARC |

---

## 4. DoS/DDoS Attacks

### DoS Overview

DoS (Denial of Service) attacks disrupt normal services by exhausting system or network resources.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DoS vs DDoS                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DoS (Single source)                DDoS (Distributed sources)  │
│                                                                 │
│     [Attacker]                          [Attacker]              │
│        │                                 │                      │
│        │                          ┌──────┼──────┐               │
│        │                          │      │      │               │
│        ▼                         ▼      ▼      ▼              │
│     [Target]                  [Bot] [Bot] [Bot] [Bot]          │
│                                   │      │      │               │
│                                   └──────┼──────┘               │
│                                          │                      │
│                                          ▼                      │
│                                       [Target]                  │
│                                                                 │
│  Features:                          Features:                   │
│  - Can defend by blocking single IP - Multiple sources, difficult to block│
│  - Bandwidth limitation              - Uses botnets             │
│                                     - Generates large traffic   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DoS Attack Types

#### 1. TCP SYN Flood

```
┌─────────────────────────────────────────────────────────────────┐
│                    TCP SYN Flood                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Normal TCP 3-way Handshake:                                    │
│                                                                 │
│  [Client]                              [Server]                 │
│      │                                        │                 │
│      │──── SYN ────────────────────────────▶ │                 │
│      │                                        │ Wait for        │
│      │◀─── SYN-ACK ──────────────────────── │ connection      │
│      │                                        │ (allocate       │
│      │──── ACK ────────────────────────────▶ │ resources)      │
│      │                                        │ Connection      │
│                                                                 │
│  SYN Flood attack:                                              │
│                                                                 │
│  [Attacker]                                  [Server]           │
│      │                                        │                 │
│      │──── SYN (forged IP) ──────────────────▶│                 │
│      │──── SYN (forged IP) ──────────────────▶│ Half-open       │
│      │──── SYN (forged IP) ──────────────────▶│ connections     │
│      │──── SYN (forged IP) ──────────────────▶│ accumulate      │
│      :           ×1000                        │ Resources       │
│                                              │ exhausted       │
│                                        ┌─────┴─────┐           │
│                                        │ Connection│           │
│                                        │ table full│           │
│                                        │           │           │
│                                        │ Normal    │           │
│                                        │ connections│          │
│                                        │ impossible│           │
│                                        └───────────┘           │
│                                                                 │
│  Countermeasures: SYN Cookies, connection limits, firewall filtering│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. UDP Flood

```
┌─────────────────────────────────────────────────────────────────┐
│                    UDP Flood                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Attacker/Botnet]                                              │
│       │                                                         │
│       │  Large volume of UDP packets                            │
│       │  (random ports)                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                        Target Server                     │   │
│  │                                                         │   │
│  │   Check UDP port → No service → Generate ICMP response  │   │
│  │                                                         │   │
│  │   Repeated processing exhausts CPU/bandwidth            │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Features:                                                      │
│  - Exploits connectionless protocol                             │
│  - Bandwidth saturation                                         │
│  - Source IP spoofing easy                                      │
│                                                                 │
│  Countermeasures: Rate limiting, blackhole routing, minimize UDP services│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3. Amplification Attacks

```
┌─────────────────────────────────────────────────────────────────┐
│                Amplification Attacks (DNS, NTP)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DNS Amplification:                                             │
│                                                                 │
│  [Attacker]                                                     │
│      │                                                          │
│      │  Small request (60 bytes)                                │
│      │  Source: Victim IP (spoofed)                             │
│      ▼                                                          │
│  ┌─────────┐                                                   │
│  │ Open    │  Large response (3000 bytes)                       │
│  │ DNS     │──────────────────────────────▶ [Victim]           │
│  │ Server  │  Amplification ratio: 50x                          │
│  └─────────┘                                                   │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  Protocol   │ Amplification │  Used Port                    ││
│  ├────────────────────────────────────────────────────────────┤│
│  │  DNS        │  28-54x      │  UDP 53                        ││
│  │  NTP        │  556x        │  UDP 123                       ││
│  │  SSDP       │  30x         │  UDP 1900                      ││
│  │  Memcached  │  51,000x     │  UDP 11211                     ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Countermeasures:                                               │
│  - Block open resolvers                                         │
│  - BCP38 (ingress filtering)                                    │
│  - Response Rate Limiting (RRL)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DDoS Attack Response

```
┌─────────────────────────────────────────────────────────────────┐
│                    DDoS Defense Strategy                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layered defense:                                               │
│                                                                 │
│  [Internet]                                                     │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. CDN/Cloud-based protection                           │   │
│  │    - Cloudflare, AWS Shield, Akamai                     │   │
│  │    - Distributed processing, bandwidth absorption       │   │
│  └─────────────────────────────────────────────────────────┘   │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. ISP Level filtering                                  │   │
│  │    - Blackhole routing                                  │   │
│  │    - Upstream filtering                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. On-premise equipment                                 │   │
│  │    - DDoS mitigation appliances                         │   │
│  │    - Firewall/IPS                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. Application level                                    │   │
│  │    - Rate limiting                                      │   │
│  │    - CAPTCHA                                            │   │
│  │    - WAF                                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. MITM Attacks

### MITM Overview

MITM (Man-in-the-Middle) attacks intercept communication between two parties to eavesdrop or modify.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MITM Attack                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Normal communication:                                          │
│                                                                 │
│  [Client] ◀═══════════════════════════════▶ [Server]           │
│                      Direct communication                        │
│                                                                 │
│  MITM attack:                                                   │
│                                                                 │
│  [Client]        [Attacker]         [Server]                    │
│       │                │                │                       │
│       │════════════════│════════════════│                       │
│       │  Fake connection│ Fake connection│                       │
│       │                │                │                       │
│       │──"Transfer"───▶│                │                       │
│       │                │──"Transfer"───▶│                       │
│       │                │(Content can be modified)               │
│       │                │                │                       │
│       │                │◀──"Done"──────│                       │
│       │◀──"Done"──────│                │                       │
│                                                                 │
│  Attacker capabilities:                                         │
│  - Eavesdrop on all communication                               │
│  - Modify data                                                  │
│  - Session hijacking                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MITM Attack Techniques

#### 1. SSL Stripping

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSL Stripping                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Downgrade HTTPS connection to HTTP                             │
│                                                                 │
│  [Client]        [Attacker]         [Server]                    │
│       │                │                │                       │
│       │──HTTP request─▶│                │                       │
│       │                │══HTTPS═══════▶│                       │
│       │                │                │                       │
│       │◀──HTTP response│◀══HTTPS════════│                       │
│       │    (Plaintext) │   (Encrypted)  │                       │
│       │                │                │                       │
│       │ Unencrypted    │ Maintain both  │                       │
│       │ communication  │ connections    │                       │
│       │                │ Relay and eavesdrop                    │
│                                                                 │
│  Countermeasures:                                               │
│  - HSTS (HTTP Strict Transport Security)                        │
│  - HTTPS Everywhere                                             │
│  - Check for padlock icon in address bar                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Wi-Fi MITM (Evil Twin)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Evil Twin Attack                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Real AP]              [Evil Twin]            [Victim]         │
│  SSID: CafeWiFi        SSID: CafeWiFi         Smartphone       │
│  Signal: Weak          Signal: Strong             │            │
│     │                      │                    │              │
│     │                      │◀══════════════════│              │
│     │                      │  Connect to strong│              │
│     │                      │  signal           │              │
│     │           ┌──────────┼──────────┐        │              │
│     │           │ Attacker laptop      │        │              │
│     │           │ - Packet capture     │        │              │
│     │           │ - DNS spoofing       │        │              │
│     │           │ - Phishing pages     │        │              │
│     │           └──────────┼──────────┘        │              │
│     │                      │                    │              │
│     │◀═════════════════════│                    │              │
│            Internet connection                                  │
│                                                                 │
│  Countermeasures:                                               │
│  - Use VPN                                                     │
│  - Be cautious with public Wi-Fi                               │
│  - 802.1X authentication                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MITM Countermeasures

| Countermeasure | Description |
|----------|------|
| Use TLS/SSL | End-to-end encryption prevents eavesdropping |
| Certificate validation | Verify server certificate validity |
| HSTS | Force HTTPS usage |
| Certificate Pinning | Allow only specific certificates |
| VPN | Use tunnels on public networks |
| 2FA | Prevent credential theft |

---

## 6. Web Security Threats

### SQL Injection

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQL Injection                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Vulnerable code example:                                       │
│                                                                 │
│  query = "SELECT * FROM users WHERE id = '" + user_input + "'"  │
│                                                                 │
│  Normal input:                                                  │
│  user_input = "123"                                             │
│  query = "SELECT * FROM users WHERE id = '123'"                 │
│                                                                 │
│  Malicious input:                                               │
│  user_input = "' OR '1'='1"                                     │
│  query = "SELECT * FROM users WHERE id = '' OR '1'='1'"         │
│                → Expose all user information                    │
│                                                                 │
│  More dangerous attack:                                         │
│  user_input = "'; DROP TABLE users;--"                          │
│  query = "SELECT * FROM users WHERE id = ''; DROP TABLE users;--"│
│                → Delete table                                   │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Attack types                                                ││
│  ├────────────────────────────────────────────────────────────┤│
│  │ In-band SQLi    : Results directly displayed on screen     ││
│  │ Blind SQLi      : Extract information via true/false responses││
│  │ Out-of-band SQLi: Send results via different channel       ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Countermeasures:                                               │
│  - Use Prepared Statements (parameterized queries)              │
│  - Input validation and escaping                                │
│  - Least privilege DB accounts                                  │
│  - Use WAF                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Safe Code Examples

```python
# Vulnerable code (Python)
cursor.execute("SELECT * FROM users WHERE id = '%s'" % user_id)

# Safe code (Prepared Statement)
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

```java
// Vulnerable code (Java)
String query = "SELECT * FROM users WHERE id = '" + userId + "'";
Statement stmt = conn.createStatement();
ResultSet rs = stmt.executeQuery(query);

// Safe code (PreparedStatement)
String query = "SELECT * FROM users WHERE id = ?";
PreparedStatement pstmt = conn.prepareStatement(query);
pstmt.setString(1, userId);
ResultSet rs = pstmt.executeQuery();
```

### XSS (Cross-Site Scripting)

```
┌─────────────────────────────────────────────────────────────────┐
│                    XSS Attack                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Stored XSS                                               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  1. [Attacker] ──Post malicious script──▶ [Web Server DB]│   │
│  │     Post: <script>malicious code</script>               │   │
│  │                                                         │   │
│  │  2. [Victim] ──Request page──▶ [Web Server]             │   │
│  │                                                         │   │
│  │  3. [Victim] ◀──Response with malicious script──        │   │
│  │     Script executes in browser                          │   │
│  │     → Cookie theft, session hijacking                   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Reflected XSS                                            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Attack URL:                                            │   │
│  │  http://example.com/search?q=<script>malicious</script>│   │
│  │                                                         │   │
│  │  1. [Attacker] ──Send malicious URL──▶ [Victim]         │   │
│  │  2. [Victim] ──Click URL──▶ [Web Server]                │   │
│  │  3. [Web Server] ──Return search term as-is──▶ [Victim] │   │
│  │     Script executes in browser                          │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Countermeasures:                                               │
│  - Output encoding (HTML Entity)                                │
│  - Input validation                                             │
│  - CSP (Content Security Policy)                                │
│  - HttpOnly cookies                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### CSRF (Cross-Site Request Forgery)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSRF Attack                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Exploit user's logged-in state to execute unwanted requests    │
│                                                                 │
│  [Victim]                [Malicious Site]         [Bank]        │
│  (Logged into bank)          │               bank.com           │
│       │                     │                  │               │
│       │──Visit malicious──▶│                  │               │
│       │   page             │                  │               │
│       │                     │                  │               │
│       │   Hidden request triggered:            │               │
│       │   <img src="http://bank.com/transfer?to=attacker&     │
│       │                     amount=10000">     │               │
│       │                     │                  │               │
│       │──────────────────────────────────────▶│               │
│       │  Request with session cookie           │               │
│       │  (Bank sees as legitimate request)     │               │
│       │                     │                  │               │
│       │                     │       ┌──────────┴───────────┐  │
│       │                     │       │ Transfer to attacker │  │
│       │                     │       │ account complete     │  │
│       │                     │       └──────────────────────┘  │
│                                                                 │
│  Countermeasures:                                               │
│  - CSRF tokens                                                  │
│  - SameSite cookies                                             │
│  - Referer validation                                           │
│  - Re-authentication for critical actions                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Intrusion Detection Systems

### IDS/IPS Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    IDS vs IPS                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IDS (Intrusion Detection System) - Detection                  │
│  ─────────────────────────────────────                          │
│                                                                 │
│  [Internet] ──────────────▶ [Firewall] ──────────────▶ [Internal]│
│                              │                                  │
│                              │ Mirroring                        │
│                              ▼                                  │
│                          [IDS]                                  │
│                       Detection and alerting                    │
│                                                                 │
│  IPS (Intrusion Prevention System) - Detection + Blocking      │
│  ───────────────────────────────────────────                    │
│                                                                 │
│  [Internet] ────▶ [Firewall] ────▶ [IPS] ────▶ [Internal]      │
│                                 │                               │
│                            Inline deployment                    │
│                          Detection and blocking                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### IDS Types

| Type | Description | Pros | Cons |
|------|------|------|------|
| NIDS | Network traffic analysis | Monitor all traffic | Difficult to analyze encrypted traffic |
| HIDS | Host activity analysis | Detailed analysis possible | Must install on each host |
| Signature-based | Match known patterns | Accurately detect known attacks | Cannot detect zero-day |
| Anomaly detection | Detect deviations from normal | Can detect new attacks | High false positives |

### IDS/IPS Signature Examples (Snort)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Snort Rule Examples                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  # SQL Injection detection                                      │
│  alert tcp any any -> any 80 (                                  │
│    msg:"SQL Injection Attempt";                                 │
│    content:"SELECT"; nocase;                                    │
│    content:"FROM"; nocase;                                      │
│    content:"WHERE"; nocase;                                     │
│    sid:1000001;                                                 │
│  )                                                              │
│                                                                 │
│  # XSS attack detection                                         │
│  alert tcp any any -> any 80 (                                  │
│    msg:"XSS Attack Attempt";                                    │
│    content:"<script>"; nocase;                                  │
│    sid:1000002;                                                 │
│  )                                                              │
│                                                                 │
│  # Port scan detection                                          │
│  alert tcp any any -> any any (                                 │
│    msg:"Possible Port Scan";                                    │
│    flags:S;                                                     │
│    threshold:type threshold, track by_src, count 5, seconds 60; │
│    sid:1000003;                                                 │
│  )                                                              │
│                                                                 │
│  Rule structure:                                                │
│  [Action] [Protocol] [Source] [Port] -> [Dest] [Port] (Options)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SIEM (Security Information and Event Management)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIEM System                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Log Collection                       │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │   │
│  │  │ FW  │ │IDS │ │Srv │ │ DB │ │App │ │ AD │           │   │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘       │   │
│  │     └───────┴───────┴───────┴───────┴───────┘           │   │
│  └─────────────────────────────┬───────────────────────────┘   │
│                                │                               │
│                                ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SIEM Engine                           │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │ Normalize → Correlate → Anomaly detect → Alert/Dashboard│  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                │                               │
│                                ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Security Response                    │   │
│  │  - Real-time alerts                                     │   │
│  │  - Incident investigation                                │   │
│  │  - Compliance reports                                    │   │
│  │  - Forensic analysis                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Major SIEM products:                                           │
│  - Splunk                                                      │
│  - IBM QRadar                                                  │
│  - Elastic SIEM                                                │
│  - Microsoft Sentinel                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Security Response Strategies

### Security Operations Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Operations Cycle                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────┐                              │
│                    │   Identify  │                              │
│                    │             │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                  │
│         ▼                 │                 ▼                  │
│  ┌─────────────┐          │          ┌─────────────┐           │
│  │   Protect   │          │          │   Recover   │           │
│  │             │          │          │             │           │
│  └──────┬──────┘          │          └──────┬──────┘           │
│         │                 │                 │                  │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   Detect    │   │   Analyze   │   │   Respond   │          │
│  │             │──▶│             │──▶│             │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│  Based on NIST Cybersecurity Framework                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Prevention

| Area | Countermeasures |
|------|----------|
| Network | Firewall, VLAN, network segregation |
| System | Patch management, security configuration, least privilege |
| Application | Secure coding, input validation, WAF |
| User | Security training, phishing drills, MFA |

### Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                    Detection Framework                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Network Monitoring                                       │   │
│  │ - IDS/IPS                                               │   │
│  │ - Network Traffic Analysis (NTA)                        │   │
│  │ - NetFlow analysis                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Endpoint Monitoring                                      │   │
│  │ - EDR (Endpoint Detection and Response)                  │   │
│  │ - Antivirus                                             │   │
│  │ - Host-based IDS (HIDS)                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Log Analysis                                             │   │
│  │ - SIEM                                                  │   │
│  │ - Centralized log management                            │   │
│  │ - Anomaly detection                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Threat Intelligence                                      │   │
│  │ - IOC (Indicators of Compromise)                         │   │
│  │ - Threat feeds                                          │   │
│  │ - Vulnerability information                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Response

```
┌─────────────────────────────────────────────────────────────────┐
│                    Incident Response Process                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Preparation                                                 │
│     └─ Develop response plan, form team, prepare tools          │
│                                                                 │
│  2. Identification                                              │
│     └─ Detect incident, assess scope, evaluate severity         │
│                                                                 │
│  3. Containment                                                 │
│     └─ Short-term: Immediate isolation                          │
│     └─ Long-term: Apply temporary fixes                         │
│                                                                 │
│  4. Eradication                                                 │
│     └─ Remove malware, patch vulnerabilities                    │
│                                                                 │
│  5. Recovery                                                    │
│     └─ Restore systems, resume normal operations                │
│                                                                 │
│  6. Lessons Learned                                             │
│     └─ Post-incident analysis, derive improvements              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Security Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│                    Basic Security Checklist                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Network Security                                               │
│  ☐ Review and minimize firewall rules                           │
│  ☐ Network segmentation (VLAN)                                  │
│  ☐ Block unnecessary ports/services                             │
│  ☐ VPN encrypted communication                                  │
│  ☐ Operate IDS/IPS                                              │
│                                                                 │
│  System Security                                                │
│  ☐ Apply security patches regularly                             │
│  ☐ Disable unnecessary services                                 │
│  ☐ Strong password policy                                       │
│  ☐ SSH key-based authentication                                 │
│  ☐ Log collection and monitoring                                │
│                                                                 │
│  Application Security                                           │
│  ☐ Input validation                                             │
│  ☐ Use Prepared Statements                                      │
│  ☐ Output encoding                                              │
│  ☐ Apply HTTPS                                                  │
│  ☐ Configure security headers                                   │
│                                                                 │
│  Data Security                                                  │
│  ☐ Encrypt critical data                                        │
│  ☐ Regular backups                                              │
│  ☐ Minimize access permissions                                  │
│  ☐ Log retention                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Practice Problems

### Basic Problems

1. **Sniffing/Spoofing**
   - Explain the difference between sniffing and spoofing.
   - Explain the operating principle of ARP spoofing.

2. **DoS/DDoS**
   - What's the difference between DoS and DDoS?
   - Explain the principle and countermeasures of SYN Flood attacks.

3. **Web Security**
   - What's the most effective method to prevent SQL Injection?
   - What's the difference between Stored XSS and Reflected XSS?

### Intermediate Problems

4. **MITM**
   - Explain SSL Stripping attacks.
   - How does HSTS prevent this attack?

5. **IDS/IPS**
   - Compare pros and cons of signature-based and anomaly detection.
   - Why are IDS and IPS deployed at different locations?

6. **Scenario Analysis**
   Suggest possible attacks and countermeasures for these situations:
   ```
   - ARP table abnormally modified in company network
   - No padlock icon when accessing web server
   - Large volume of SELECT queries executed on database
   ```

### Advanced Problems

7. **Comprehensive Security**
   - Identify security vulnerabilities in this architecture:
   ```
   Internet ─── Web Server ─── DB Server
                         (same network)
   ```

8. **Incident Response**
   - List the response procedures for ransomware infection in order.

---

## 10. Next Steps

In [17_Practical_Network_Tools.md](./17_Practical_Network_Tools.md), let's learn about practical network tools like ping, traceroute, and Wireshark!

---

## 11. References

### Security Frameworks

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [MITRE ATT&CK](https://attack.mitre.org/)
- [OWASP Top 10](https://owasp.org/Top10/)

### Tools

- Snort/Suricata - Open source IDS/IPS
- Wireshark - Packet analysis
- Burp Suite - Web security testing
- Nmap - Network scanning

### Learning Resources

- [SANS Reading Room](https://www.sans.org/reading-room/)
- [Krebs on Security](https://krebsonsecurity.com/)
- [The Hacker News](https://thehackernews.com/)
