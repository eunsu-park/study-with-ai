# HTTP and HTTPS

## Overview

HTTP (HyperText Transfer Protocol) is an application layer protocol for exchanging data between clients and servers on the web. HTTPS is a protocol that enhances security by adding TLS/SSL encryption to HTTP.

**Difficulty**: ⭐⭐⭐

**Learning Objectives**:
- Understand HTTP request/response structure
- Master HTTP methods and status codes
- Identify differences between HTTP versions
- Understand HTTPS and TLS/SSL operation principles

---

## Table of Contents

1. [HTTP Basics](#1-http-basics)
2. [HTTP Methods](#2-http-methods)
3. [HTTP Status Codes](#3-http-status-codes)
4. [HTTP Headers](#4-http-headers)
5. [HTTP Version Comparison](#5-http-version-comparison)
6. [HTTPS and TLS/SSL](#6-https-and-tlsssl)
7. [Certificates](#7-certificates)
8. [Practice Problems](#8-practice-problems)
9. [Next Steps](#9-next-steps)
10. [References](#10-references)

---

## 1. HTTP Basics

### HTTP Communication Structure

```
┌──────────────┐                              ┌──────────────┐
│   Client     │                              │    Server    │
│  (Browser)   │                              │ (Web Server) │
├──────────────┤                              ├──────────────┤
│              │  ──── HTTP Request ────────▶ │              │
│   GET /index │  (Method, URL, Headers, Body)│   Nginx      │
│              │                              │   Apache     │
│              │  ◀──── HTTP Response ──────  │              │
│   HTML Page  │  (Status Code, Headers, Body)│              │
└──────────────┘                              └──────────────┘
        │                                            │
        │        TCP Connection (Default Port 80)    │
        └────────────────────────────────────────────┘
```

### HTTP Characteristics

| Characteristic | Description |
|------|------|
| Connectionless | Connection closes after request-response (HTTP/1.0) |
| Stateless | Each request is independent, no previous state stored |
| Text-based | Human-readable format |
| Request-Response | Client requests, server responds |

### HTTP Request Structure

```
┌─────────────────────────────────────────────────────────┐
│ Request Line                                            │
├─────────────────────────────────────────────────────────┤
│ GET /api/users HTTP/1.1                                 │
│ └─┘ └────────┘ └───────┘                                │
│ Method  URI    Version                                  │
├─────────────────────────────────────────────────────────┤
│ Headers                                                 │
├─────────────────────────────────────────────────────────┤
│ Host: api.example.com                                   │
│ User-Agent: Mozilla/5.0                                 │
│ Accept: application/json                                │
│ Content-Type: application/json                          │
│ Authorization: Bearer eyJhbGciOiJ...                    │
├─────────────────────────────────────────────────────────┤
│ Blank Line (CRLF)                                       │
├─────────────────────────────────────────────────────────┤
│ Body (Optional)                                         │
├─────────────────────────────────────────────────────────┤
│ {"name": "John", "email": "john@example.com"}           │
└─────────────────────────────────────────────────────────┘
```

### HTTP Response Structure

```
┌─────────────────────────────────────────────────────────┐
│ Status Line                                             │
├─────────────────────────────────────────────────────────┤
│ HTTP/1.1 200 OK                                         │
│ └───────┘ └─┘ └┘                                        │
│  Version  Status Reason                                 │
├─────────────────────────────────────────────────────────┤
│ Headers                                                 │
├─────────────────────────────────────────────────────────┤
│ Content-Type: application/json                          │
│ Content-Length: 128                                     │
│ Date: Mon, 27 Jan 2026 10:30:00 GMT                     │
│ Server: nginx/1.24.0                                    │
│ Cache-Control: no-cache                                 │
├─────────────────────────────────────────────────────────┤
│ Blank Line (CRLF)                                       │
├─────────────────────────────────────────────────────────┤
│ Body                                                    │
├─────────────────────────────────────────────────────────┤
│ {"id": 1, "name": "John", "status": "active"}           │
└─────────────────────────────────────────────────────────┘
```

### Checking HTTP Requests with curl

```bash
# Basic GET request
curl http://example.com

# Output with headers
curl -i http://example.com

# Verbose request/response details
curl -v http://example.com

# Headers only
curl -I http://example.com

# JSON POST request
curl -X POST http://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John"}'
```

---

## 2. HTTP Methods

### Main HTTP Methods

```
┌────────────┬──────────────────────────────────────────────┐
│   Method   │                Description                    │
├────────────┼──────────────────────────────────────────────┤
│    GET     │ Retrieve resource (read)                      │
│    POST    │ Create resource (write)                       │
│    PUT     │ Replace entire resource                       │
│   PATCH    │ Partially modify resource                     │
│   DELETE   │ Delete resource                               │
│    HEAD    │ Retrieve headers only (no body)               │
│  OPTIONS   │ Check supported methods                       │
│   TRACE    │ Loopback test (debugging)                     │
│  CONNECT   │ Establish proxy tunnel                        │
└────────────┴──────────────────────────────────────────────┘
```

### Method Properties

```
┌────────────┬────────────┬──────────────┬──────────────┐
│   Method   │   Safety   │ Idempotency  │  Cacheable   │
│            │  (Safe)    │ (Idempotent) │ (Cacheable)  │
├────────────┼────────────┼──────────────┼──────────────┤
│    GET     │     O      │      O       │      O       │
│    HEAD    │     O      │      O       │      O       │
│   OPTIONS  │     O      │      O       │      X       │
│    POST    │     X      │      X       │  Conditional │
│    PUT     │     X      │      O       │      X       │
│   DELETE   │     X      │      O       │      X       │
│   PATCH    │     X      │      X       │      X       │
└────────────┴────────────┴──────────────┴──────────────┘

* Safety: Does not change server state
* Idempotency: Multiple executions produce same result
* Cacheable: Response can be cached
```

### GET vs POST Comparison

| Characteristic | GET | POST |
|------|-----|------|
| Purpose | Data retrieval | Data transmission/creation |
| Data Location | URL query string | Request body |
| Data Size | URL length limit (~2KB) | No limit |
| Caching | Possible | Not by default |
| Security | Exposed in URL | Relatively safe |
| Bookmarkable | Yes | No |

### Method Usage in RESTful APIs

```
┌─────────────────────────────────────────────────────────────────┐
│                   RESTful API Examples                           │
├─────────────────────────────────────────────────────────────────┤
│  Operation    │ Method  │ Endpoint          │ Description       │
├─────────────────────────────────────────────────────────────────┤
│  List all     │ GET     │ /api/users        │ Retrieve all users│
│  Get one      │ GET     │ /api/users/1      │ Retrieve user ID=1│
│  Create       │ POST    │ /api/users        │ Create new user   │
│  Full update  │ PUT     │ /api/users/1      │ Full user update  │
│  Partial      │ PATCH   │ /api/users/1      │ Partial update    │
│  Delete       │ DELETE  │ /api/users/1      │ Delete user       │
└─────────────────────────────────────────────────────────────────┘
```

### Method Examples

```bash
# GET - Retrieve resource
curl -X GET "http://api.example.com/users?page=1&limit=10"

# POST - Create resource
curl -X POST http://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com"
  }'

# PUT - Full update
curl -X PUT http://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john.new@example.com",
    "status": "active"
  }'

# PATCH - Partial update
curl -X PATCH http://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "inactive"}'

# DELETE - Delete
curl -X DELETE http://api.example.com/users/1

# HEAD - Headers only
curl -I http://api.example.com/users/1

# OPTIONS - Check supported methods
curl -X OPTIONS http://api.example.com/users
```

---

## 3. HTTP Status Codes

### Status Code Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                 HTTP Status Code Categories                      │
├─────────────────────────────────────────────────────────────────┤
│  Category │ Range     │ Meaning                                  │
├─────────────────────────────────────────────────────────────────┤
│  1xx     │ 100-199  │ Informational - Processing                │
│  2xx     │ 200-299  │ Success - Request succeeded               │
│  3xx     │ 300-399  │ Redirection - Further action needed       │
│  4xx     │ 400-499  │ Client Error                              │
│  5xx     │ 500-599  │ Server Error                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1xx - Informational Responses

| Code | Name | Description |
|------|------|------|
| 100 | Continue | Request can continue |
| 101 | Switching Protocols | Protocol switch (WebSocket, etc.) |
| 102 | Processing | Processing (WebDAV) |

### 2xx - Success Responses

| Code | Name | Description | Use Case |
|------|------|------|----------|
| 200 | OK | Request successful | GET success |
| 201 | Created | Resource created | POST success |
| 202 | Accepted | Request accepted (async processing) | Async task |
| 204 | No Content | Success, no response body | DELETE success |
| 206 | Partial Content | Partial content | Range request |

### 3xx - Redirection

| Code | Name | Description | Cached |
|------|------|------|------|
| 301 | Moved Permanently | Permanent move | Cached |
| 302 | Found | Temporary move | Not cached |
| 303 | See Other | Different location (change to GET) | Not cached |
| 304 | Not Modified | No change (use cache) | - |
| 307 | Temporary Redirect | Temporary move (keep method) | Not cached |
| 308 | Permanent Redirect | Permanent move (keep method) | Cached |

### 4xx - Client Errors

| Code | Name | Description |
|------|------|------|
| 400 | Bad Request | Malformed request (syntax error) |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Access denied (no permission) |
| 404 | Not Found | Resource not found |
| 405 | Method Not Allowed | Method not allowed |
| 408 | Request Timeout | Request timeout |
| 409 | Conflict | Conflict (concurrent modification) |
| 413 | Payload Too Large | Request body too large |
| 414 | URI Too Long | URI too long |
| 415 | Unsupported Media Type | Unsupported media type |
| 422 | Unprocessable Entity | Unprocessable entity |
| 429 | Too Many Requests | Rate limit exceeded |

### 5xx - Server Errors

| Code | Name | Description |
|------|------|------|
| 500 | Internal Server Error | Server internal error |
| 501 | Not Implemented | Not implemented |
| 502 | Bad Gateway | Gateway error |
| 503 | Service Unavailable | Service unavailable |
| 504 | Gateway Timeout | Gateway timeout |

### Status Code Flow

```
                        ┌─────────────┐
                        │ HTTP Request│
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  Validation │
                        └──────┬──────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │ Syntax Error │     │ Auth Check  │     │ Permission  │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │    400      │     │    401      │     │    403      │
    │ Bad Request │     │Unauthorized │     │  Forbidden  │
    └─────────────┘     └─────────────┘     └─────────────┘

                        ┌──────────────┐
                        │ Find Resource│
                        └──────┬───────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
          ┌──────▼──────┐  ┌───▼───┐  ┌──────▼──────┐
          │  Not Found  │  │Success│  │Server Error │
          └──────┬──────┘  └───┬───┘  └──────┬──────┘
                 │             │             │
          ┌──────▼──────┐  ┌───▼───┐  ┌──────▼──────┐
          │    404      │  │  200  │  │    500      │
          │  Not Found  │  │  OK   │  │   Internal  │
          └─────────────┘  └───────┘  └─────────────┘
```

---

## 4. HTTP Headers

### Header Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                   HTTP Header Categories                         │
├─────────────────────────────────────────────────────────────────┤
│  Category          │ Description                                 │
├─────────────────────────────────────────────────────────────────┤
│  General Headers   │ Used in both requests/responses             │
│  Request Headers   │ Client information (Host, User-Agent, etc.) │
│  Response Headers  │ Server information (Server, Set-Cookie, etc)│
│  Entity Headers    │ Body information (Content-Type, Length, etc)│
└─────────────────────────────────────────────────────────────────┘
```

### Main Request Headers

| Header | Description | Example |
|------|------|------|
| Host | Request host | `Host: api.example.com` |
| User-Agent | Client information | `User-Agent: Mozilla/5.0` |
| Accept | Desired response type | `Accept: application/json` |
| Accept-Language | Preferred language | `Accept-Language: en-US,en;q=0.9` |
| Accept-Encoding | Supported encoding | `Accept-Encoding: gzip, deflate` |
| Authorization | Authentication info | `Authorization: Bearer token123` |
| Cookie | Send cookies | `Cookie: session_id=abc123` |
| Content-Type | Request body type | `Content-Type: application/json` |
| Content-Length | Request body size | `Content-Length: 256` |
| Referer | Previous page URL | `Referer: https://google.com` |
| Origin | Request origin | `Origin: https://example.com` |

### Main Response Headers

| Header | Description | Example |
|------|------|------|
| Content-Type | Response body type | `Content-Type: text/html; charset=utf-8` |
| Content-Length | Response body size | `Content-Length: 1024` |
| Content-Encoding | Compression method | `Content-Encoding: gzip` |
| Cache-Control | Cache control | `Cache-Control: max-age=3600` |
| Expires | Expiration time | `Expires: Wed, 27 Jan 2027 10:00:00 GMT` |
| ETag | Resource version identifier | `ETag: "abc123"` |
| Last-Modified | Last modified time | `Last-Modified: Mon, 01 Jan 2026 00:00:00 GMT` |
| Set-Cookie | Set cookie | `Set-Cookie: id=abc; HttpOnly; Secure` |
| Location | Redirect location | `Location: https://example.com/new` |
| Server | Server information | `Server: nginx/1.24.0` |

### Security-Related Headers

| Header | Description |
|------|------|
| Strict-Transport-Security (HSTS) | Force HTTPS |
| X-Content-Type-Options | Prevent MIME sniffing |
| X-Frame-Options | Prevent clickjacking |
| X-XSS-Protection | Enable XSS filter |
| Content-Security-Policy (CSP) | Content security policy |
| Access-Control-Allow-Origin | CORS allowed origins |

### Caching-Related Headers

```
┌─────────────────────────────────────────────────────────────────┐
│                      HTTP Caching Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                              [Server]                 │
│      │                                        │                 │
│      │──── GET /image.png ────────────────▶  │                 │
│      │                                        │                 │
│      │◀─── 200 OK ──────────────────────────│                 │
│      │     Cache-Control: max-age=3600       │                 │
│      │     ETag: "abc123"                    │                 │
│      │     Last-Modified: Mon, 01 Jan...     │                 │
│      │                                        │                 │
│  [Cache Stored]                               │                 │
│      │                                        │                 │
│      │──── GET /image.png ────────────────▶  │                 │
│      │     If-None-Match: "abc123"           │                 │
│      │                                        │                 │
│      │◀─── 304 Not Modified ────────────────│                 │
│      │     (No body, use cache)              │                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cache-Control Directives

| Directive | Description |
|--------|------|
| `no-store` | Prohibit cache storage |
| `no-cache` | Validation required before cache use |
| `max-age=N` | Valid for N seconds |
| `s-maxage=N` | Valid for N seconds in shared cache |
| `private` | Private cache only |
| `public` | Shared cache allowed |
| `must-revalidate` | Must revalidate after expiration |

---

## 5. HTTP Version Comparison

### HTTP Version Evolution

```
┌─────────────────────────────────────────────────────────────────┐
│                 HTTP Version Evolution                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HTTP/0.9 (1991)                                                │
│  └── GET only, no headers                                       │
│       │                                                         │
│       ▼                                                         │
│  HTTP/1.0 (1996)                                                │
│  └── Headers added, status codes, POST/HEAD                     │
│       │                                                         │
│       ▼                                                         │
│  HTTP/1.1 (1997)                                                │
│  └── Persistent connections, pipelining, Host header required   │
│       │                                                         │
│       ▼                                                         │
│  HTTP/2 (2015)                                                  │
│  └── Binary protocol, multiplexing, header compression          │
│       │                                                         │
│       ▼                                                         │
│  HTTP/3 (2022)                                                  │
│  └── QUIC (UDP-based), improved connection setup                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HTTP/1.1 vs HTTP/2 vs HTTP/3

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|------|----------|--------|--------|
| Protocol | Text | Binary | Binary |
| Transport Layer | TCP | TCP | QUIC (UDP) |
| Multiplexing | X | O | O |
| Header Compression | X | HPACK | QPACK |
| Server Push | X | O | O |
| Requests per Connection | Sequential | Concurrent multiple | Concurrent multiple |
| HOL Blocking | Present | Present at TCP level | None |

### HTTP/1.1 Connection Methods

```
HTTP/1.0 (Non-persistent)    HTTP/1.1 (Persistent)

Request 1 ──────▶              Request 1 ──────▶
      ◀────── Response 1             ◀────── Response 1
[Connection closed]             Request 2 ──────▶
Request 2 ──────▶                    ◀────── Response 2
      ◀────── Response 2       Request 3 ──────▶
[Connection closed]                    ◀────── Response 3
Request 3 ──────▶              [Connection maintained then closed]
      ◀────── Response 3
[Connection closed]

※ 3 TCP connections            ※ 1 TCP connection
```

### HTTP/2 Multiplexing

```
┌─────────────────────────────────────────────────────────────────┐
│                      HTTP/2 Multiplexing                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HTTP/1.1 (Sequential)     │  HTTP/2 (Concurrent)               │
│                           │                                     │
│  Request1 ──────────────▶ │  Request1 ────▶                    │
│  Response1 ◀────────────  │  Request2 ────▶ (simultaneous)     │
│  Request2 ──────────────▶ │  Request3 ────▶ (simultaneous)     │
│  Response2 ◀────────────  │  Response2 ◀───                    │
│  Request3 ──────────────▶ │  Response1 ◀───                    │
│  Response3 ◀────────────  │  Response3 ◀───                    │
│                           │                                     │
│  ├────────────────────┤   │  ├──────────────┤                   │
│       Long time            │     Short time                      │
│                           │                                     │
└─────────────────────────────────────────────────────────────────┘
```

### HTTP/3 and QUIC

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP/3 (QUIC-based)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐                           │
│  │  HTTP/2     │      │  HTTP/3     │                           │
│  ├─────────────┤      ├─────────────┤                           │
│  │    TLS      │      │    QUIC     │◀─ TLS 1.3 built-in        │
│  ├─────────────┤      │ (encryption)│                           │
│  │    TCP      │      ├─────────────┤                           │
│  ├─────────────┤      │    UDP      │                           │
│  │    IP       │      ├─────────────┤                           │
│  └─────────────┘      │    IP       │                           │
│                       └─────────────┘                           │
│                                                                 │
│  QUIC Advantages:                                               │
│  - 0-RTT connection (when reconnecting)                         │
│  - Packet loss doesn't affect other streams                     │
│  - Connection migration (connection maintained on IP change)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. HTTPS and TLS/SSL

### HTTPS Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP vs HTTPS                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HTTP (Port 80)           │  HTTPS (Port 443)                   │
│                           │                                     │
│  ┌─────────────┐          │  ┌─────────────┐                    │
│  │    HTTP     │          │  │    HTTP     │                    │
│  ├─────────────┤          │  ├─────────────┤                    │
│  │    TCP      │          │  │  TLS/SSL    │◀─ Encryption layer │
│  ├─────────────┤          │  ├─────────────┤                    │
│  │    IP       │          │  │    TCP      │                    │
│  └─────────────┘          │  ├─────────────┤                    │
│                           │  │    IP       │                    │
│  Plaintext transmission   │  └─────────────┘                    │
│  Data exposure risk       │                                     │
│                           │  Encrypted transmission             │
│                           │  Data protection                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### TLS/SSL History

| Version | Year | Status |
|------|------|------|
| SSL 2.0 | 1995 | Deprecated (security vulnerabilities) |
| SSL 3.0 | 1996 | Deprecated (POODLE vulnerability) |
| TLS 1.0 | 1999 | Deprecation recommended |
| TLS 1.1 | 2006 | Deprecation recommended |
| TLS 1.2 | 2008 | In use |
| TLS 1.3 | 2018 | Recommended (current latest) |

### TLS Handshake (TLS 1.2)

```
┌─────────────────────────────────────────────────────────────────┐
│                  TLS 1.2 Handshake                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                              [Server]                 │
│      │                                        │                 │
│      │──(1) ClientHello ─────────────────▶  │                 │
│      │    - Supported TLS version            │                 │
│      │    - Supported cipher suites          │                 │
│      │    - Client random                    │                 │
│      │                                        │                 │
│      │◀─(2) ServerHello ─────────────────── │                 │
│      │    - Selected TLS version             │                 │
│      │    - Selected cipher suite            │                 │
│      │    - Server random                    │                 │
│      │                                        │                 │
│      │◀─(3) Certificate ─────────────────── │                 │
│      │    - Server certificate (public key)  │                 │
│      │                                        │                 │
│      │◀─(4) ServerHelloDone ──────────────  │                 │
│      │                                        │                 │
│      │──(5) ClientKeyExchange ───────────▶  │                 │
│      │    - Pre-Master Secret (encrypted)    │                 │
│      │                                        │                 │
│      │──(6) ChangeCipherSpec ────────────▶  │                 │
│      │──(7) Finished ────────────────────▶  │                 │
│      │                                        │                 │
│      │◀─(8) ChangeCipherSpec ───────────────│                 │
│      │◀─(9) Finished ───────────────────────│                 │
│      │                                        │                 │
│      │◀════════ Encrypted Communication ═══▶│                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### TLS 1.3 Handshake (Simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│               TLS 1.3 Handshake (1-RTT)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Client]                              [Server]                 │
│      │                                        │                 │
│      │──(1) ClientHello + KeyShare ──────▶  │                 │
│      │                                        │                 │
│      │◀─(2) ServerHello + KeyShare ─────────│                 │
│      │      Certificate                      │                 │
│      │      Finished                         │                 │
│      │                                        │                 │
│      │──(3) Finished ────────────────────▶  │                 │
│      │                                        │                 │
│      │◀════════ Encrypted Communication ═══▶│                 │
│                                                                 │
│  ※ Handshake complete in just 1 RTT (round trip)               │
│  ※ 0-RTT: Can send data from first request when resuming       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Encryption Types

```
┌─────────────────────────────────────────────────────────────────┐
│                      Encryption Methods                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Symmetric Encryption                                           │
│  ─────────────────────                                          │
│  - Encrypt/decrypt with same key                                │
│  - Fast speed                                                   │
│  - Examples: AES, ChaCha20                                      │
│                                                                 │
│      Plaintext ──[Key]──▶ Ciphertext ──[Key]──▶ Plaintext      │
│                                                                 │
│  Asymmetric Encryption                                          │
│  ──────────────────────                                         │
│  - Uses public/private key pair                                 │
│  - Slow speed, used for key exchange                            │
│  - Examples: RSA, ECDSA                                         │
│                                                                 │
│      Plaintext ──[Public Key]──▶ Ciphertext ──[Private Key]──▶ Plaintext│
│                                                                 │
│  Usage in TLS                                                   │
│  ─────────────                                                  │
│  1. Asymmetric key for session key exchange                     │
│  2. Symmetric key (session key) for actual data encryption      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Certificates

### Certificate Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    X.509 Certificate Structure                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Version: V3                                             │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Serial Number: 0x1234...                                │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Signature Algorithm: SHA256withRSA                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Issuer: CN=Let's Encrypt Authority X3                  │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Validity                                                │   │
│  │    Not Before: 2026-01-01 00:00:00                       │   │
│  │    Not After:  2026-04-01 00:00:00                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Subject: CN=www.example.com                             │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Public Key Info                                         │   │
│  │    Algorithm: RSA                                        │   │
│  │    Key Size: 2048 bits                                   │   │
│  │    Public Key: 30 82 01 0a 02 82 01 01 00...            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Extensions                                              │   │
│  │    Subject Alternative Names: www.example.com,           │   │
│  │                               example.com                │   │
│  │    Key Usage: Digital Signature, Key Encipherment        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Signature: 48 46 2b 88 2d...                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Certificate Chain

```
┌─────────────────────────────────────────────────────────────────┐
│                    Certificate Chain (Chain of Trust)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Root Certificate (Root CA)                  │   │
│  │         - Self-signed                                    │   │
│  │         - Built into browser/OS                          │   │
│  │         - Examples: DigiCert, GlobalSign                 │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │ Signs                               │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Intermediate Certificate (Intermediate CA)     │   │
│  │         - Signed by Root CA                              │   │
│  │         - Example: Let's Encrypt R3                      │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │ Signs                               │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │             Server Certificate (End-Entity)              │   │
│  │         - Signed by Intermediate CA                      │   │
│  │         - Domain: www.example.com                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Validation order: Server cert → Intermediate cert → Root cert │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Certificate Types

| Type | Validation Level | Issuance Time | Use Case |
|------|----------|----------|------|
| DV (Domain Validation) | Domain ownership only | Minutes | Personal, blogs |
| OV (Organization Validation) | Organization verification | 1-3 days | Companies, institutions |
| EV (Extended Validation) | Strict verification | 1-2 weeks | Financial, large corporations |
| Wildcard | Includes subdomains | Varies | *.example.com |
| Multi-Domain (SAN) | Multiple domains | Varies | Multiple domains |

### Certificate Issuance Process (Let's Encrypt)

```bash
# Install Certbot (Ubuntu)
sudo apt install certbot python3-certbot-nginx

# Issue certificate (Nginx)
sudo certbot --nginx -d example.com -d www.example.com

# Issue certificate (Apache)
sudo certbot --apache -d example.com

# Renew certificate
sudo certbot renew

# Check certificate
sudo certbot certificates

# Auto-renewal (cron)
0 12 * * * /usr/bin/certbot renew --quiet
```

### Certificate Verification Commands

```bash
# Check domain certificate
openssl s_client -connect example.com:443 -servername example.com

# Certificate details
echo | openssl s_client -connect example.com:443 2>/dev/null | \
  openssl x509 -noout -text

# Check expiration date
echo | openssl s_client -connect example.com:443 2>/dev/null | \
  openssl x509 -noout -enddate

# Check local certificate file
openssl x509 -in certificate.crt -text -noout
```

---

## 8. Practice Problems

### Basic Problems

1. **HTTP Methods**
   - Explain 3 differences between GET and POST.
   - What is idempotency, and list all idempotent methods.

2. **Status Codes**
   - Choose appropriate status codes for these situations:
     - User login failure (authentication failure)
     - Page not found
     - Internal server error occurs
     - Resource created successfully via POST request

3. **Headers**
   - What's the difference between Cache-Control: no-cache and no-store?
   - What is the purpose of the ETag header?

### Intermediate Problems

4. **HTTP Versions**
   - Explain HTTP/1.1's HOL (Head-of-Line) Blocking problem.
   - How does HTTP/2 solve this problem?

5. **HTTPS/TLS**
   - What are 3 security benefits of using HTTPS?
   - What is the handshake RTT difference between TLS 1.2 and TLS 1.3?

6. **Practical Problems**

```bash
# Analyze the results of the following curl commands

# 1. What is included in the request headers?
curl -v http://example.com

# 2. What status code do you receive if this request succeeds?
curl -I -X DELETE http://api.example.com/users/1

# 3. What is the Content-Type in this request?
curl -X POST http://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "test"}'
```

### Advanced Problems

7. **Certificate Chain**
   - Why doesn't the root CA sign server certificates directly, using intermediate CAs instead?

8. **Security Headers**
   - Suggest HTTP headers to prevent these security vulnerabilities:
     - Clickjacking
     - XSS (Cross-Site Scripting)
     - MIME sniffing

---

## 9. Next Steps

In [14_Other_Application_Protocols.md](./14_Other_Application_Protocols.md), let's learn about other application layer protocols such as DHCP, FTP, SMTP, and SSH!

---

## 10. References

### RFC Documents

- [RFC 7230-7235](https://tools.ietf.org/html/rfc7230) - HTTP/1.1
- [RFC 7540](https://tools.ietf.org/html/rfc7540) - HTTP/2
- [RFC 9110-9114](https://tools.ietf.org/html/rfc9110) - HTTP Semantics
- [RFC 8446](https://tools.ietf.org/html/rfc8446) - TLS 1.3

### Online Resources

- [MDN HTTP Documentation](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [HTTP Status Codes](https://httpstatuses.com/)
- [SSL Labs Server Test](https://www.ssllabs.com/ssltest/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)

### Tools

- curl - Command-line HTTP client
- Postman - API testing tool
- Charles Proxy - HTTP proxy/monitoring
- Wireshark - Packet analysis
