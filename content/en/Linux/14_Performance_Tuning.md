# 14. Linux Performance Tuning

## Learning Objectives
- System performance monitoring and analysis
- Kernel parameter optimization via sysctl
- CPU, memory, and I/O performance tuning
- Profiling with perf and flamegraphs

## Table of Contents
1. [Performance Analysis Fundamentals](#1-performance-analysis-fundamentals)
2. [CPU Tuning](#2-cpu-tuning)
3. [Memory Tuning](#3-memory-tuning)
4. [I/O Tuning](#4-io-tuning)
5. [Network Tuning](#5-network-tuning)
6. [Profiling Tools](#6-profiling-tools)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. Performance Analysis Fundamentals

### 1.1 USE Methodology

```
┌─────────────────────────────────────────────────────────────┐
│                USE Methodology (Brendan Gregg)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Check for each resource:                                   │
│                                                             │
│  U - Utilization                                            │
│      How much is the resource being used?                   │
│      Example: CPU at 80% usage                              │
│                                                             │
│  S - Saturation                                             │
│      Are tasks waiting?                                     │
│      Example: 10 processes in run queue                     │
│                                                             │
│  E - Errors                                                 │
│      Are errors occurring?                                  │
│      Example: Network packet drops                          │
│                                                             │
│  Key resources:                                             │
│  • CPU: mpstat, vmstat, top                                │
│  • Memory: free, vmstat, /proc/meminfo                     │
│  • Disk I/O: iostat, iotop                                 │
│  • Network: netstat, ss, sar                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Basic Monitoring Tools

```bash
# top - real-time process monitoring
top
# Shortcuts: 1=per CPU, M=sort by memory, P=sort by CPU, k=kill

# htop - enhanced top
htop

# vmstat - virtual memory statistics
vmstat 1 5  # 1 second interval, 5 times
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
#  2  0      0 1234567 12345 234567    0    0     1     2  100  200  5  2 93  0  0
# r: processes waiting to run
# b: processes waiting for I/O
# si/so: swap in/out
# bi/bo: block in/out
# us/sy/id/wa: user/system/idle/wait

# mpstat - CPU statistics
mpstat -P ALL 1  # All CPUs, 1 second interval

# iostat - I/O statistics
iostat -x 1      # Extended info, 1 second interval

# sar - system activity report
sar -u 1 5       # CPU
sar -r 1 5       # Memory
sar -d 1 5       # Disk
sar -n DEV 1 5   # Network

# free - memory usage
free -h

# uptime - load average
uptime
# load average: 1.50, 1.20, 0.80  (1min, 5min, 15min)
```

### 1.3 sysctl Basics

```bash
# View current settings
sysctl -a                    # All settings
sysctl vm.swappiness         # Specific setting
cat /proc/sys/vm/swappiness  # Direct read

# Temporary change
sysctl -w vm.swappiness=10
# Or
echo 10 > /proc/sys/vm/swappiness

# Persistent configuration
# /etc/sysctl.conf or /etc/sysctl.d/*.conf
echo "vm.swappiness = 10" >> /etc/sysctl.d/99-custom.conf
sysctl -p /etc/sysctl.d/99-custom.conf  # Apply
sysctl --system  # Load all configuration files
```

---

## 2. CPU Tuning

### 2.1 CPU Information

```bash
# CPU information
lscpu
cat /proc/cpuinfo

# CPU frequency
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
cpupower frequency-info

# NUMA information
numactl --hardware
lscpu | grep NUMA
```

### 2.2 CPU Governor

```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
# performance, powersave, userspace, ondemand, conservative, schedutil

# Change governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Using cpupower
cpupower frequency-set -g performance

# Persistent configuration (Ubuntu)
# /etc/default/cpufrequtils
GOVERNOR="performance"
```

### 2.3 Process Priority

```bash
# nice value (-20 to 19, lower is higher priority)
nice -n -10 ./high-priority-task
renice -n -10 -p <PID>

# Real-time scheduling
chrt -f 50 ./realtime-task  # FIFO, priority 50
chrt -r 50 ./realtime-task  # Round Robin

# CPU affinity
taskset -c 0,1 ./my-program  # Run on CPU 0, 1 only
taskset -cp 0-3 <PID>        # Change running process

# CPU limit with cgroups
# /sys/fs/cgroup/cpu/mygroup/
mkdir /sys/fs/cgroup/cpu/mygroup
echo 50000 > /sys/fs/cgroup/cpu/mygroup/cpu.cfs_quota_us  # 50% limit
echo <PID> > /sys/fs/cgroup/cpu/mygroup/cgroup.procs
```

### 2.4 CPU-related sysctl

```bash
# /etc/sysctl.d/99-cpu.conf

# Scheduler tuning
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000
kernel.sched_migration_cost_ns = 5000000

# Workload-specific optimization
# Server workload (throughput-oriented)
kernel.sched_autogroup_enabled = 0

# Desktop workload (responsiveness-oriented)
kernel.sched_autogroup_enabled = 1
```

---

## 3. Memory Tuning

### 3.1 Memory Information

```bash
# Memory usage
free -h
cat /proc/meminfo

# Per-process memory
ps aux --sort=-%mem | head
pmap -x <PID>

# Page cache status
cat /proc/meminfo | grep -E "Cached|Buffers|Dirty"

# NUMA memory
numastat
```

### 3.2 Swap Tuning

```bash
# swappiness (0-100, lower uses less swap)
sysctl -w vm.swappiness=10  # Server: 10, Desktop: 60

# Create swap file
dd if=/dev/zero of=/swapfile bs=1G count=4
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Add to /etc/fstab
# /swapfile none swap sw 0 0

# Swap status
swapon --show
cat /proc/swaps
```

### 3.3 Memory-related sysctl

```bash
# /etc/sysctl.d/99-memory.conf

# Reduce swap usage
vm.swappiness = 10

# Dirty page ratio (write delay)
vm.dirty_ratio = 20              # Allow up to 20% of total memory dirty
vm.dirty_background_ratio = 5    # Start background flush at 5%

# Or absolute values
vm.dirty_bytes = 1073741824      # 1GB
vm.dirty_background_bytes = 268435456  # 256MB

# Cache pressure
vm.vfs_cache_pressure = 50       # Default 100, lower keeps cache

# OOM Killer tuning
vm.overcommit_memory = 0         # 0=heuristic, 1=always allow, 2=limit
vm.overcommit_ratio = 50         # Used when overcommit_memory=2

# Memory compaction
vm.compaction_proactiveness = 20

# Transparent Huge Pages
# /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never
```

### 3.4 Cache Management

```bash
# Clear page cache (use with caution in production!)
sync
echo 1 > /proc/sys/vm/drop_caches  # Page cache
echo 2 > /proc/sys/vm/drop_caches  # dentries, inodes
echo 3 > /proc/sys/vm/drop_caches  # All

# Check file cache
vmtouch -v /path/to/file
fincore /path/to/file

# Per-process cache usage
cat /proc/<PID>/smaps | grep -E "^(Rss|Shared|Private)"
```

---

## 4. I/O Tuning

### 4.1 I/O Scheduler

```bash
# Check current scheduler
cat /sys/block/sda/queue/scheduler
# [mq-deadline] kyber bfq none

# Scheduler types
# - none: For NVMe SSD (NOOP)
# - mq-deadline: Deadline-based, server default
# - bfq: Budget Fair Queueing, desktop
# - kyber: For fast devices

# Change scheduler
echo mq-deadline > /sys/block/sda/queue/scheduler

# Persistent configuration (GRUB)
# /etc/default/grub
# GRUB_CMDLINE_LINUX="elevator=mq-deadline"
# update-grub

# Set via udev rules
# /etc/udev/rules.d/60-scheduler.rules
# ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/scheduler}="mq-deadline"
# ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="none"
```

### 4.2 Disk I/O Tuning

```bash
# Readahead
cat /sys/block/sda/queue/read_ahead_kb  # Default 128
echo 256 > /sys/block/sda/queue/read_ahead_kb

# Queue depth
cat /sys/block/sda/queue/nr_requests
echo 256 > /sys/block/sda/queue/nr_requests

# Maximum sectors
cat /sys/block/sda/queue/max_sectors_kb

# Enable SSD TRIM
fstrim -v /
# Or automatic TRIM (mount option: discard)
# /dev/sda1 / ext4 defaults,discard 0 1

# Periodic TRIM (recommended)
systemctl enable fstrim.timer
```

### 4.3 Filesystem Tuning

```bash
# ext4 mount options
# /etc/fstab
# noatime    - Don't update access time (performance gain)
# nodiratime - Don't update directory access time
# data=writeback - Journaling mode (risky but fast)
# barrier=0  - Disable write barrier (risky)
# commit=60  - Commit interval (seconds)

# XFS tuning
# logbufs=8 - Number of log buffers
# logbsize=256k - Log buffer size

# Filesystem information
tune2fs -l /dev/sda1  # ext4
xfs_info /dev/sda1    # XFS
```

### 4.4 I/O Priority

```bash
# ionice - I/O priority
ionice -c 3 command        # Idle
ionice -c 2 -n 0 command   # Best-effort, high priority
ionice -c 1 command        # Realtime (root only)

# Change running process
ionice -c 2 -n 7 -p <PID>  # Lower priority

# Check current I/O priority
ionice -p <PID>
```

---

## 5. Network Tuning

### 5.1 Network Information

```bash
# Interface information
ip link show
ethtool eth0

# Network statistics
ss -s
netstat -s
cat /proc/net/netstat

# Connection status
ss -tuln   # Listening ports
ss -tupn   # All connections
conntrack -L  # Connection tracking table
```

### 5.2 TCP Tuning

```bash
# /etc/sysctl.d/99-network.conf

# TCP buffer sizes
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576

# TCP socket buffer (min, default, max)
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216

# TCP backlog
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# TIME_WAIT optimization
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1

# TCP Keepalive
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 60
net.ipv4.tcp_keepalive_probes = 3

# TCP congestion control
net.ipv4.tcp_congestion_control = bbr  # Or cubic
net.core.default_qdisc = fq

# Port range
net.ipv4.ip_local_port_range = 1024 65535

# SYN cookies (SYN flood defense)
net.ipv4.tcp_syncookies = 1
```

### 5.3 High-Performance Web Server Configuration

```bash
# /etc/sysctl.d/99-webserver.conf

# File handle limits
fs.file-max = 2097152
fs.nr_open = 2097152

# Network stack
net.core.somaxconn = 65535
net.ipv4.tcp_max_tw_buckets = 2000000
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# Buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 12582912 16777216
net.ipv4.tcp_wmem = 4096 12582912 16777216

# TCP optimization
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_mtu_probing = 1

# BBR
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
```

### 5.4 Connection Limits

```bash
# System limits
ulimit -n        # Current limit
ulimit -n 65535  # Change

# /etc/security/limits.conf
# * soft nofile 65535
# * hard nofile 65535

# systemd service limits
# [Service]
# LimitNOFILE=65535
```

---

## 6. Profiling Tools

### 6.1 perf Basics

```bash
# Install perf
apt install linux-tools-common linux-tools-$(uname -r)

# CPU profiling
perf stat ./my-program
perf stat -d ./my-program  # Detailed

# Sampling
perf record -g ./my-program
perf record -g -p <PID> -- sleep 30

# Analyze results
perf report
perf report --stdio

# Real-time monitoring
perf top
perf top -p <PID>

# System-wide
perf record -a -g -- sleep 10
```

### 6.2 Flamegraph

```bash
# Install FlameGraph tools
git clone https://github.com/brendangregg/FlameGraph

# Collect data with perf
perf record -g -p <PID> -- sleep 60

# Generate flamegraph
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flame.svg

# Or all at once
perf record -F 99 -a -g -- sleep 60
perf script | \
  ./FlameGraph/stackcollapse-perf.pl | \
  ./FlameGraph/flamegraph.pl > flame.svg
```

### 6.3 strace/ltrace

```bash
# System call tracing
strace ./my-program
strace -p <PID>

# Specific system calls only
strace -e open,read,write ./my-program

# Time measurement
strace -T ./my-program    # Time per syscall
strace -c ./my-program    # Summary statistics

# Library call tracing
ltrace ./my-program
```

### 6.4 Other Tools

```bash
# bpftrace - eBPF-based tracing
bpftrace -e 'tracepoint:syscalls:sys_enter_open { printf("%s %s\n", comm, str(args->filename)); }'

# Memory profiling (Valgrind)
valgrind --tool=massif ./my-program
ms_print massif.out.*

# CPU profiling (Valgrind)
valgrind --tool=callgrind ./my-program
kcachegrind callgrind.out.*

# Benchmarking
stress-ng --cpu 4 --timeout 60s
fio --name=random-write --ioengine=libaio --iodepth=32 --rw=randwrite --bs=4k --direct=1 --size=1G --numjobs=4 --runtime=60
```

### 6.5 Performance Checklist

```bash
#!/bin/bash
# performance-check.sh

echo "=== System Information ==="
uname -a
uptime

echo -e "\n=== CPU ==="
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Model name)"
mpstat 1 1

echo -e "\n=== Memory ==="
free -h
cat /proc/meminfo | grep -E "^(MemTotal|MemFree|Buffers|Cached|SwapTotal|SwapFree)"

echo -e "\n=== Disk I/O ==="
iostat -x 1 1

echo -e "\n=== Network ==="
ss -s
cat /proc/net/netstat | grep -E "^(Tcp|Udp)"

echo -e "\n=== Load Average ==="
cat /proc/loadavg

echo -e "\n=== Top Processes (CPU) ==="
ps aux --sort=-%cpu | head -5

echo -e "\n=== Top Processes (Memory) ==="
ps aux --sort=-%mem | head -5

echo -e "\n=== Open Files ==="
cat /proc/sys/fs/file-nr

echo -e "\n=== Network Connections ==="
ss -s
```

---

## 7. Practice Exercises

### Exercise 1: Web Server Tuning
```bash
# Requirements:
# 1. Support 100,000 concurrent connections
# 2. TCP optimization (BBR, keepalive)
# 3. Increase file handle limits
# 4. Choose appropriate I/O scheduler

# Write sysctl configuration:
```

### Exercise 2: Database Server Tuning
```bash
# Requirements:
# 1. Memory optimization (low swappiness)
# 2. Disk I/O optimization
# 3. Dirty page management
# 4. CPU affinity configuration

# Write configuration and commands:
```

### Exercise 3: Performance Problem Diagnosis
```bash
# Scenario:
# List items to check sequentially when server becomes slow

# Diagnostic command list:
```

### Exercise 4: Flamegraph Analysis
```bash
# Requirements:
# 1. Write or select CPU-intensive program
# 2. Profile with perf
# 3. Generate flamegraph
# 4. Analyze bottlenecks

# Commands and analysis approach:
```

---

## Next Steps

- [15_Container_Internals](15_Container_Internals.md) - cgroups, namespaces
- [16_Storage_Management](16_Storage_Management.md) - LVM, RAID
- [Brendan Gregg's Blog](https://www.brendangregg.com/)

## References

- [Linux Performance](https://www.brendangregg.com/linuxperf.html)
- [Red Hat Performance Tuning Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/index)
- [kernel.org sysctl Documentation](https://www.kernel.org/doc/Documentation/sysctl/)
- [perf Examples](https://www.brendangregg.com/perf.html)

---

[← Previous: Advanced systemd](13_Systemd_Advanced.md) | [Next: Container Internals →](15_Container_Internals.md) | [Table of Contents](00_Overview.md)
