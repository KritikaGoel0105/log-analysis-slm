"""
synthetic_generator.py

Generates synthetic instruction-following examples using
parameterized templates, as specified in the internship
document (Section 5.3):

    "Sources: open-source log datasets (LogHub, Loghub-2.0),
     synthetic generation using templates, and manually
     curated examples."

Each template defines:
  - "logs"     : normalized log window (format string)
  - "severity" : CRITICAL | HIGH | MEDIUM | LOW | INFO
  - "incident_type", "root_cause", "summary" (format strings)
  - "recommended_actions" : list of format strings
  - "variants" : list of dicts; each dict is substituted into
                 the format strings, producing one example per
                 variant. This keeps input logs and output
                 labels consistent with each other.

All placeholders (<TIMESTAMP>, <IP_ADDR>, <PORT>, <FILE_PATH>,
<USER_ID>, <UUID>, <MEM_ADDR>) match the Week 2 normalization
tokens so synthetic inputs look like real preprocessed windows.
"""

from .templates import SYSTEM_PROMPT, format_output


# ---------------------------------------------------------------------
# Synthetic Templates
# ---------------------------------------------------------------------

SYNTHETIC_TEMPLATES = [

    # ============================================================
    # DATABASE INCIDENTS
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> ERROR [db-pool] Connection timeout: pool exhausted (0/{pool} available)\n"
            "<TIMESTAMP> ERROR [api-gw] Upstream <IP_ADDR>:<PORT> returned 503\n"
            "<TIMESTAMP> ERROR [api-gw] Request queue: {queue} pending (limit: {limit})\n"
            "<TIMESTAMP> CRIT [health-chk] Database health check FAILED for {polls} consecutive polls"
        ),
        "severity": "CRITICAL",
        "incident_type": "Database Connection Pool Exhaustion",
        "root_cause": (
            "Database connection pool fully exhausted (0/{pool} connections available), "
            "likely due to slow queries or a connection leak causing upstream API "
            "timeouts and queue overflow."
        ),
        "summary": (
            "The database connection pool is fully depleted, causing all API calls to "
            "fail with 503 errors. The request queue has reached {queue} pending requests "
            "against a limit of {limit}, indicating system saturation."
        ),
        "recommended_actions": [
            "Immediately restart the database connection pool manager",
            "Identify and kill long-running queries (> 30s)",
            "Increase pool size temporarily: max_connections={newpool}",
            "Review application code for connection leak patterns",
        ],
        "variants": [
            {"pool": 50, "queue": 1247, "limit": 500, "polls": 3, "newpool": 100},
            {"pool": 100, "queue": 3892, "limit": 1000, "polls": 5, "newpool": 200},
            {"pool": 25, "queue": 641, "limit": 250, "polls": 4, "newpool": 50},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [mysql] Deadlock found when trying to get lock; try restarting transaction\n"
            "<TIMESTAMP> ERROR [order-svc] Transaction rollback: deadlock on table '{table}'\n"
            "<TIMESTAMP> WARN [order-svc] Retry {retry}/3 for transaction <UUID>\n"
            "<TIMESTAMP> ERROR [mysql] Deadlock found when trying to get lock; try restarting transaction"
        ),
        "severity": "HIGH",
        "incident_type": "Database Deadlock",
        "root_cause": (
            "Concurrent transactions are acquiring row locks on table '{table}' in "
            "conflicting order, causing repeated deadlocks and transaction rollbacks."
        ),
        "summary": (
            "The database is detecting deadlocks on table '{table}', forcing transaction "
            "rollbacks in the order service. Retries are occurring but repeated deadlocks "
            "indicate a lock-ordering problem in application code."
        ),
        "recommended_actions": [
            "Review transactions touching '{table}' for consistent lock ordering",
            "Shorten transaction scope to reduce lock hold time",
            "Examine SHOW ENGINE INNODB STATUS for the deadlock cycle",
            "Add retry-with-backoff for deadlock-prone transactions",
        ],
        "variants": [
            {"table": "orders", "retry": 2},
            {"table": "inventory", "retry": 1},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [db-replica] Replication lag: {lag}s behind primary\n"
            "<TIMESTAMP> WARN [db-replica] Relay log read position falling behind\n"
            "<TIMESTAMP> WARN [report-svc] Stale read detected: data age {lag}s exceeds threshold {thresh}s"
        ),
        "severity": "MEDIUM",
        "incident_type": "Database Replication Lag",
        "root_cause": (
            "The database replica has fallen {lag} seconds behind the primary, likely due "
            "to heavy write load on the primary or insufficient replica I/O capacity."
        ),
        "summary": (
            "Replication lag of {lag}s has been detected on the database replica, exceeding "
            "the {thresh}s staleness threshold. Services reading from the replica are "
            "receiving stale data until the replica catches up."
        ),
        "recommended_actions": [
            "Check primary write throughput for unusual spikes",
            "Verify replica disk I/O is not saturated",
            "Route consistency-sensitive reads to the primary temporarily",
            "Monitor Seconds_Behind_Master until lag returns below {thresh}s",
        ],
        "variants": [
            {"lag": 187, "thresh": 60},
            {"lag": 452, "thresh": 120},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [mysql-slow] Query took {secs}s: SELECT * FROM {table} WHERE status = 'pending'\n"
            "<TIMESTAMP> WARN [mysql-slow] Query took {secs2}s: SELECT * FROM {table} WHERE status = 'pending'\n"
            "<TIMESTAMP> WARN [api] Endpoint /v1/{table} p95 latency degraded to {lat}ms"
        ),
        "severity": "MEDIUM",
        "incident_type": "Slow Database Query",
        "root_cause": (
            "Unindexed full-table scans on '{table}' filtering by status column are taking "
            "{secs}+ seconds, degrading API latency for dependent endpoints."
        ),
        "summary": (
            "The slow query log shows repeated {secs}s+ scans on table '{table}'. API p95 "
            "latency for the /v1/{table} endpoint has degraded to {lat}ms as a result. An "
            "index on the status column is likely missing."
        ),
        "recommended_actions": [
            "Run EXPLAIN on the slow query to confirm full table scan",
            "Add an index on {table}(status)",
            "Avoid SELECT *; fetch only required columns",
            "Monitor endpoint latency after index deployment",
        ],
        "variants": [
            {"table": "orders", "secs": 14, "secs2": 17, "lat": 8200},
            {"table": "sessions", "secs": 9, "secs2": 11, "lat": 5400},
        ],
    },

    # ============================================================
    # MEMORY INCIDENTS
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> WARN [kernel] Memory pressure: {pct}% used, swapping heavily\n"
            "<TIMESTAMP> ERROR [kernel] Out of memory: Kill process {pid} ({proc}) score {score} or sacrifice child\n"
            "<TIMESTAMP> ERROR [kernel] Killed process {pid} ({proc}) total-vm:{vm}kB\n"
            "<TIMESTAMP> ERROR [systemd] {proc}.service: Main process exited, code=killed, status=9/KILL"
        ),
        "severity": "CRITICAL",
        "incident_type": "Out of Memory (OOM) Kill",
        "root_cause": (
            "System memory was exhausted ({pct}% used with heavy swapping), causing the "
            "kernel OOM killer to terminate the '{proc}' process (PID {pid}) which had "
            "the highest OOM score."
        ),
        "summary": (
            "The kernel OOM killer terminated '{proc}' after memory usage reached {pct}%. "
            "The service is now down until restarted. This indicates either a memory leak "
            "in '{proc}' or insufficient memory provisioning for the workload."
        ),
        "recommended_actions": [
            "Restart the '{proc}' service and confirm it is healthy",
            "Analyze '{proc}' heap/RSS growth over time for leaks",
            "Set memory limits (cgroups/systemd MemoryMax) to protect other services",
            "Increase instance memory if the workload legitimately requires it",
        ],
        "variants": [
            {"pct": 98, "pid": 21437, "proc": "java", "score": 912, "vm": 18874368},
            {"pct": 97, "pid": 8823, "proc": "python3", "score": 887, "vm": 9437184},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [jvm] GC overhead: {gcpct}% of last 60s spent in garbage collection\n"
            "<TIMESTAMP> WARN [jvm] Old generation at {oldpct}% after full GC\n"
            "<TIMESTAMP> ERROR [app] java.lang.OutOfMemoryError: GC overhead limit exceeded"
        ),
        "severity": "HIGH",
        "incident_type": "JVM Heap Exhaustion",
        "root_cause": (
            "The JVM old generation remains at {oldpct}% even after full GC, and {gcpct}% "
            "of CPU time is spent collecting garbage — the heap is too small for the live "
            "object set or a memory leak is retaining objects."
        ),
        "summary": (
            "The application's JVM is thrashing in garbage collection ({gcpct}% GC time) "
            "and has thrown OutOfMemoryError. Throughput has collapsed and the process "
            "needs a heap dump analysis and restart."
        ),
        "recommended_actions": [
            "Capture a heap dump (jmap) before restarting for leak analysis",
            "Restart the application to restore service",
            "Analyze dominator tree in the heap dump for retained objects",
            "Increase -Xmx only if analysis shows legitimate live-set growth",
        ],
        "variants": [
            {"gcpct": 91, "oldpct": 99},
            {"gcpct": 84, "oldpct": 97},
        ],
    },

    # ============================================================
    # DISK / STORAGE INCIDENTS
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> CRIT [disk-mon] Filesystem {mount} usage at {pct}% ({free}MB free)\n"
            "<TIMESTAMP> ERROR [app] Failed to write <FILE_PATH>: No space left on device\n"
            "<TIMESTAMP> ERROR [db] Cannot extend data file: disk full\n"
            "<TIMESTAMP> ERROR [logger] Log rotation failed: insufficient space"
        ),
        "severity": "CRITICAL",
        "incident_type": "Disk Space Exhaustion",
        "root_cause": (
            "The {mount} filesystem is {pct}% full with only {free}MB remaining, causing "
            "write failures across the application, database, and logging subsystems."
        ),
        "summary": (
            "Disk space on {mount} is exhausted. Application writes, database file "
            "extension, and log rotation are all failing. Data loss is possible if "
            "space is not freed immediately."
        ),
        "recommended_actions": [
            "Free space immediately: remove old logs, temp files, and archives on {mount}",
            "Identify the largest recent growth with du -sh {mount}/*",
            "Extend the volume or add storage if growth is legitimate",
            "Configure disk usage alerts at 80% to catch this earlier",
        ],
        "variants": [
            {"mount": "/var", "pct": 100, "free": 12},
            {"mount": "/data", "pct": 99, "free": 87},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [disk-mon] Filesystem {mount} usage at {pct}% \n"
            "<TIMESTAMP> WARN [disk-mon] Growth rate {rate}GB/day, projected full in {days} days\n"
            "<TIMESTAMP> INFO [cleanup] Scheduled cleanup job queued"
        ),
        "severity": "MEDIUM",
        "incident_type": "Disk Space Warning",
        "root_cause": (
            "The {mount} filesystem has reached {pct}% utilization and is growing at "
            "{rate}GB/day, projecting exhaustion within {days} days."
        ),
        "summary": (
            "Disk usage on {mount} has crossed the warning threshold at {pct}%. At the "
            "current growth rate of {rate}GB/day the volume will be full in roughly "
            "{days} days. Cleanup or expansion is needed before that."
        ),
        "recommended_actions": [
            "Run the cleanup job and verify space is reclaimed",
            "Archive or compress old data on {mount}",
            "Plan volume expansion before the projected exhaustion date",
            "Review retention policies for logs and temporary data",
        ],
        "variants": [
            {"mount": "/var/log", "pct": 87, "rate": 2.4, "days": 5},
            {"mount": "/opt/data", "pct": 91, "rate": 1.1, "days": 8},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [kernel] I/O error, dev {dev}, sector {sector}\n"
            "<TIMESTAMP> ERROR [kernel] Buffer I/O error on device {dev}, logical block {block}\n"
            "<TIMESTAMP> WARN [smartd] Device {dev}: {realloc} reallocated sectors (was {prev})"
        ),
        "severity": "HIGH",
        "incident_type": "Disk I/O Failure",
        "root_cause": (
            "Device {dev} is reporting sector-level I/O errors and its reallocated sector "
            "count has grown from {prev} to {realloc}, indicating progressive physical "
            "disk failure."
        ),
        "summary": (
            "Physical disk {dev} is failing: the kernel is logging I/O errors and SMART "
            "shows a rising reallocated sector count. Data on this device is at risk and "
            "the disk should be replaced proactively."
        ),
        "recommended_actions": [
            "Verify current backups of data on {dev} immediately",
            "Schedule disk replacement as soon as possible",
            "Run smartctl -a on {dev} and preserve the report",
            "If in RAID, check array health and start rebuild planning",
        ],
        "variants": [
            {"dev": "sda", "sector": 48291712, "block": 6036464, "realloc": 137, "prev": 24},
            {"dev": "nvme0n1", "sector": 90118244, "block": 11264780, "realloc": 52, "prev": 3},
        ],
    },

    # ============================================================
    # NETWORK INCIDENTS
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> ERROR [resolver] DNS resolution failed for {host}: SERVFAIL\n"
            "<TIMESTAMP> ERROR [payment-svc] Cannot connect to {host}: Name or service not known\n"
            "<TIMESTAMP> ERROR [resolver] DNS resolution failed for {host}: SERVFAIL\n"
            "<TIMESTAMP> WARN [payment-svc] Circuit breaker OPEN for {host} after {fails} failures"
        ),
        "severity": "HIGH",
        "incident_type": "DNS Resolution Failure",
        "root_cause": (
            "DNS lookups for {host} are returning SERVFAIL, preventing the payment "
            "service from connecting. The upstream DNS server is failing or the zone "
            "for this domain is broken."
        ),
        "summary": (
            "The payment service cannot resolve {host} due to repeated DNS SERVFAIL "
            "responses. After {fails} consecutive failures the circuit breaker opened, "
            "so all calls to this dependency are now failing fast."
        ),
        "recommended_actions": [
            "Test resolution directly: dig {host} against each configured resolver",
            "Check upstream DNS server health and zone configuration",
            "Add a fallback resolver or temporary hosts-file entry if critical",
            "Reset the circuit breaker once resolution is confirmed working",
        ],
        "variants": [
            {"host": "api.payments.internal", "fails": 12},
            {"host": "auth.corp.local", "fails": 8},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [svc-mesh] Connection refused to <IP_ADDR>:<PORT> ({svc})\n"
            "<TIMESTAMP> ERROR [svc-mesh] Connection refused to <IP_ADDR>:<PORT> ({svc})\n"
            "<TIMESTAMP> WARN [lb] Backend {svc} marked unhealthy: {down}/{total} instances down"
        ),
        "severity": "HIGH",
        "incident_type": "Service Connection Refused",
        "root_cause": (
            "Connections to the {svc} service are being refused, meaning the process is "
            "not listening on the expected port. {down} of {total} instances are down — "
            "the service crashed or failed to bind its port after deployment."
        ),
        "summary": (
            "The load balancer has marked {svc} unhealthy with {down}/{total} instances "
            "refusing connections. Traffic to this service is failing. The instances are "
            "down rather than slow, pointing to crash or misconfiguration."
        ),
        "recommended_actions": [
            "Check {svc} process status on the affected instances",
            "Review {svc} startup logs for bind/crash errors",
            "Roll back the latest {svc} deployment if it coincides with the failures",
            "Restore at least one healthy instance, then investigate the rest",
        ],
        "variants": [
            {"svc": "inventory-svc", "down": 3, "total": 4},
            {"svc": "user-profile-svc", "down": 2, "total": 2},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [ingress] TLS handshake error from <IP_ADDR>:<PORT>: certificate has expired\n"
            "<TIMESTAMP> ERROR [ingress] TLS handshake error from <IP_ADDR>:<PORT>: certificate has expired\n"
            "<TIMESTAMP> WARN [cert-mon] Certificate for {domain} expired {hrs} hours ago"
        ),
        "severity": "CRITICAL",
        "incident_type": "TLS Certificate Expiry",
        "root_cause": (
            "The TLS certificate for {domain} expired {hrs} hours ago and was not renewed, "
            "causing all client TLS handshakes to fail at the ingress."
        ),
        "summary": (
            "All HTTPS connections to {domain} are failing because its certificate expired "
            "{hrs} hours ago. Every client is affected. Certificate renewal automation "
            "either failed or was never configured for this domain."
        ),
        "recommended_actions": [
            "Renew and deploy a valid certificate for {domain} immediately",
            "Reload the ingress/proxy to pick up the new certificate",
            "Investigate why automated renewal did not run",
            "Add expiry monitoring alerts at 30/14/7 days before expiration",
        ],
        "variants": [
            {"domain": "api.example.com", "hrs": 6},
            {"domain": "portal.internal.net", "hrs": 14},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [net-mon] Packet loss to <IP_ADDR>: {loss}% over last 5m\n"
            "<TIMESTAMP> WARN [rpc] Retransmission rate elevated: {retrans}%\n"
            "<TIMESTAMP> WARN [app] Upstream call latency p99 {lat}ms (baseline {base}ms)"
        ),
        "severity": "MEDIUM",
        "incident_type": "Network Packet Loss",
        "root_cause": (
            "{loss}% packet loss on the path to the upstream host is forcing TCP "
            "retransmissions ({retrans}%), inflating p99 latency from {base}ms to {lat}ms."
        ),
        "summary": (
            "Network degradation is causing {loss}% packet loss and elevated "
            "retransmissions to an upstream dependency. Application latency has "
            "degraded roughly {lat}ms at p99 but requests are still succeeding."
        ),
        "recommended_actions": [
            "Run mtr/traceroute to locate the lossy hop",
            "Check switch/NIC error counters on both endpoints",
            "Engage the network team with the loss evidence",
            "Consider rerouting traffic if an alternate path exists",
        ],
        "variants": [
            {"loss": 7, "retrans": 4.2, "lat": 2100, "base": 180},
            {"loss": 12, "retrans": 8.9, "lat": 4800, "base": 220},
        ],
    },

    # ============================================================
    # AUTHENTICATION / SECURITY INCIDENTS
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> WARN [auth] Failed login for user <USER_ID> from <IP_ADDR> (attempt {a1})\n"
            "<TIMESTAMP> WARN [auth] Failed login for user <USER_ID> from <IP_ADDR> (attempt {a2})\n"
            "<TIMESTAMP> WARN [auth] Failed login for user <USER_ID> from <IP_ADDR> (attempt {a3})\n"
            "<TIMESTAMP> ERROR [auth] {count} failed logins from <IP_ADDR> across {users} accounts in {mins} minutes"
        ),
        "severity": "HIGH",
        "incident_type": "Brute Force Attack",
        "root_cause": (
            "A single source IP generated {count} failed login attempts across {users} "
            "different accounts within {mins} minutes — an automated credential "
            "brute-force or password-spraying attack."
        ),
        "summary": (
            "An active brute-force attack is under way: {count} failed logins across "
            "{users} accounts from one IP in {mins} minutes. No successful login from "
            "this source appears in the window, but the attack is ongoing."
        ),
        "recommended_actions": [
            "Block the source IP at the firewall or WAF immediately",
            "Enable or verify rate limiting on the login endpoint",
            "Check for any successful logins from this IP and force password resets",
            "Review lockout policy effectiveness against distributed attempts",
        ],
        "variants": [
            {"a1": 4, "a2": 5, "a3": 6, "count": 312, "users": 47, "mins": 10},
            {"a1": 2, "a2": 3, "a3": 4, "count": 128, "users": 19, "mins": 5},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [auth] Account <USER_ID> locked after {n} failed attempts\n"
            "<TIMESTAMP> INFO [auth] Lockout duration: {mins} minutes\n"
            "<TIMESTAMP> INFO [auth] Notification sent to account owner"
        ),
        "severity": "MEDIUM",
        "incident_type": "Account Lockout",
        "root_cause": (
            "The account exceeded {n} failed login attempts and was automatically locked "
            "for {mins} minutes by policy — either a forgotten password or a targeted "
            "guessing attempt."
        ),
        "summary": (
            "A user account was locked out after {n} consecutive failed logins. The "
            "lockout policy engaged correctly and the owner was notified. Verify whether "
            "the attempts were the legitimate user or an attacker."
        ),
        "recommended_actions": [
            "Contact the account owner to confirm whether the attempts were theirs",
            "Review source IPs of the failed attempts for anomalies",
            "If suspicious, force a password reset before unlocking",
            "Monitor the account for further attempts after unlock",
        ],
        "variants": [
            {"n": 5, "mins": 30},
            {"n": 10, "mins": 60},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> CRIT [audit] User <USER_ID> executed privileged command outside change window\n"
            "<TIMESTAMP> CRIT [audit] sudo escalation from unrecognized session on {host}\n"
            "<TIMESTAMP> WARN [audit] New SSH key added to authorized_keys for root on {host}"
        ),
        "severity": "CRITICAL",
        "incident_type": "Suspected Privilege Escalation",
        "root_cause": (
            "An unrecognized session on {host} performed sudo escalation and added a new "
            "root SSH key outside any approved change window — consistent with an "
            "attacker establishing persistence after account compromise."
        ),
        "summary": (
            "Audit logs show unauthorized privilege escalation on {host}: a privileged "
            "command from an unrecognized session followed by a new root SSH key. This "
            "is a likely active compromise requiring immediate incident response."
        ),
        "recommended_actions": [
            "Isolate {host} from the network immediately",
            "Remove the unauthorized SSH key and terminate the suspect session",
            "Preserve logs and disk image for forensic analysis",
            "Rotate credentials for the affected user and escalate to the security team",
        ],
        "variants": [
            {"host": "prod-app-03"},
            {"host": "build-server-01"},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [ids] Port scan detected from <IP_ADDR>: {ports} ports probed in {secs}s\n"
            "<TIMESTAMP> WARN [firewall] Dropped {drops} connection attempts from <IP_ADDR>\n"
            "<TIMESTAMP> INFO [ids] Source added to watch list"
        ),
        "severity": "MEDIUM",
        "incident_type": "Port Scan Detected",
        "root_cause": (
            "An external host probed {ports} ports in {secs} seconds — automated "
            "reconnaissance scanning for exposed services. The firewall dropped all "
            "{drops} attempts."
        ),
        "summary": (
            "The IDS detected a port scan ({ports} ports in {secs}s) from a single "
            "external source. The firewall blocked the attempts and no successful "
            "connections resulted. This is reconnaissance, not yet an intrusion."
        ),
        "recommended_actions": [
            "Confirm no services responded to the scanning source",
            "Add the source IP to the block list if scanning repeats",
            "Verify exposed ports match the approved service inventory",
            "Keep the source on the IDS watch list for follow-up activity",
        ],
        "variants": [
            {"ports": 1024, "secs": 42, "drops": 1019},
            {"ports": 400, "secs": 15, "drops": 398},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> INFO [auth] Successful login for user <USER_ID> from <IP_ADDR>\n"
            "<TIMESTAMP> INFO [auth] MFA challenge completed\n"
            "<TIMESTAMP> INFO [auth] Session <UUID> created, expires in {hrs}h"
        ),
        "severity": "INFO",
        "incident_type": "Successful Authentication",
        "root_cause": (
            "A user completed password and MFA authentication successfully and a "
            "{hrs}-hour session was issued — routine authentication activity."
        ),
        "summary": (
            "Normal successful login with MFA verification. A session was created with a "
            "{hrs}-hour expiry. No anomalies present in this window."
        ),
        "recommended_actions": [
            "No immediate action required",
            "Retain the audit record per compliance policy",
        ],
        "variants": [
            {"hrs": 8},
            {"hrs": 12},
        ],
    },

    # ============================================================
    # APPLICATION FAILURES
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> ERROR [{svc}] Unhandled exception: NullPointerException at {cls}.{method}({cls}.java:{line})\n"
            "<TIMESTAMP> ERROR [{svc}] Request <UUID> failed with HTTP 500\n"
            "<TIMESTAMP> ERROR [{svc}] Unhandled exception: NullPointerException at {cls}.{method}({cls}.java:{line})\n"
            "<TIMESTAMP> WARN [{svc}] Error rate {rate}% over last 5m (threshold {thresh}%)"
        ),
        "severity": "HIGH",
        "incident_type": "Application Unhandled Exception",
        "root_cause": (
            "A NullPointerException in {cls}.{method} (line {line}) is thrown on a "
            "recurring code path, returning HTTP 500 to clients and pushing the error "
            "rate to {rate}%."
        ),
        "summary": (
            "The {svc} service is repeatedly failing with a NullPointerException in "
            "{cls}.{method}. The error rate has reached {rate}%, exceeding the {thresh}% "
            "threshold. A recent input pattern or deployment is triggering an unguarded "
            "null dereference."
        ),
        "recommended_actions": [
            "Inspect {cls}.java line {line} for the null dereference",
            "Correlate failing requests to identify the triggering input",
            "Deploy a null-check hotfix or roll back the recent release",
            "Add a regression test covering the failing input",
        ],
        "variants": [
            {"svc": "checkout-svc", "cls": "PriceCalculator", "method": "applyDiscount", "line": 214, "rate": 18, "thresh": 5},
            {"svc": "profile-svc", "cls": "AvatarService", "method": "resize", "line": 88, "rate": 9, "thresh": 5},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [kernel] {proc}[{pid}]: segfault at <MEM_ADDR> ip <MEM_ADDR> sp <MEM_ADDR> error 4 in {lib}\n"
            "<TIMESTAMP> ERROR [systemd] {proc}.service: Main process exited, code=dumped, status=11/SEGV\n"
            "<TIMESTAMP> INFO [systemd] {proc}.service: Scheduled restart job, restart counter is at {n}"
        ),
        "severity": "HIGH",
        "incident_type": "Process Segmentation Fault",
        "root_cause": (
            "The {proc} process crashed with a segmentation fault inside {lib}, indicating "
            "an invalid memory access — a native-code bug or an incompatible library "
            "version. This is restart number {n}."
        ),
        "summary": (
            "{proc} is crash-looping with SIGSEGV in {lib} ({n} restarts so far). systemd "
            "keeps restarting it but the underlying memory bug persists, so the crash "
            "will recur until the faulty code path or library is fixed."
        ),
        "recommended_actions": [
            "Collect the core dump and get a backtrace with gdb",
            "Check whether {lib} was recently updated; pin the previous version if so",
            "Report the backtrace upstream if the fault is in third-party code",
            "Add a restart limit to avoid infinite crash-looping",
        ],
        "variants": [
            {"proc": "nginx", "pid": 3341, "lib": "libssl.so.3", "n": 4},
            {"proc": "redis-server", "pid": 9902, "lib": "libc-2.31.so", "n": 7},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [{svc}] Configuration error: required key '{key}' missing in <FILE_PATH>\n"
            "<TIMESTAMP> ERROR [{svc}] Startup aborted\n"
            "<TIMESTAMP> ERROR [systemd] {svc}.service: Failed with result 'exit-code'"
        ),
        "severity": "HIGH",
        "incident_type": "Service Configuration Error",
        "root_cause": (
            "The {svc} service refuses to start because required configuration key "
            "'{key}' is missing from its config file — most likely an incomplete "
            "deployment or a bad config change."
        ),
        "summary": (
            "{svc} fails at startup with a missing '{key}' configuration key and exits "
            "immediately. The service is fully down. The most recent config change or "
            "deploy did not include the required key."
        ),
        "recommended_actions": [
            "Add the missing '{key}' key to the configuration file",
            "Diff the current config against the last known-good version",
            "Restart {svc} and verify it reaches healthy state",
            "Add config validation to the deployment pipeline",
        ],
        "variants": [
            {"svc": "notification-svc", "key": "smtp.host"},
            {"svc": "search-svc", "key": "index.path"},
        ],
    },

    # ============================================================
    # SERVICE / INFRASTRUCTURE INCIDENTS
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> ERROR [health] {svc} health check failed: connect timeout after {t}s\n"
            "<TIMESTAMP> ERROR [health] {svc} health check failed: connect timeout after {t}s\n"
            "<TIMESTAMP> ERROR [health] {svc} health check failed: connect timeout after {t}s\n"
            "<TIMESTAMP> CRIT [orchestrator] {svc} declared DOWN after {n} consecutive failures"
        ),
        "severity": "CRITICAL",
        "incident_type": "Service Health Check Failure",
        "root_cause": (
            "The {svc} service failed {n} consecutive health checks with {t}s connect "
            "timeouts and has been declared down — the process is hung, overloaded, or "
            "the host is unreachable."
        ),
        "summary": (
            "{svc} is down: {n} consecutive health checks timed out and the orchestrator "
            "has removed it from service. Connect-level timeouts (not HTTP errors) point "
            "to a hung process or host-level problem rather than application errors."
        ),
        "recommended_actions": [
            "Check whether the {svc} host is reachable (ping/SSH)",
            "Inspect the process state — hung, deadlocked, or CPU-saturated",
            "Restart {svc} or fail over to a standby instance",
            "Capture thread/stack dumps before restart if possible for diagnosis",
        ],
        "variants": [
            {"svc": "payment-gateway", "t": 5, "n": 3},
            {"svc": "session-store", "t": 3, "n": 5},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [k8s] Back-off restarting failed container {ctr} in pod {pod}\n"
            "<TIMESTAMP> WARN [k8s] Pod {pod} restart count: {n}\n"
            "<TIMESTAMP> ERROR [k8s] Container {ctr} last state: OOMKilled (exit code 137)"
        ),
        "severity": "HIGH",
        "incident_type": "Container Crash Loop",
        "root_cause": (
            "Container {ctr} in pod {pod} is in CrashLoopBackOff with {n} restarts; the "
            "last termination was OOMKilled (exit 137), so its memory limit is below "
            "actual usage."
        ),
        "summary": (
            "Pod {pod} is crash-looping because container {ctr} keeps exceeding its "
            "memory limit and getting OOMKilled ({n} restarts). The workload cannot "
            "stay up until the limit is raised or memory usage is reduced."
        ),
        "recommended_actions": [
            "Compare the container memory limit to its actual working set",
            "Raise the memory limit or optimize application memory usage",
            "Check for a recent change that increased memory consumption",
            "Monitor restart count after the fix to confirm stability",
        ],
        "variants": [
            {"ctr": "api", "pod": "api-7d9f8b-x2k4l", "n": 17},
            {"ctr": "worker", "pod": "worker-59c6d-p8s2m", "n": 9},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [lb] Backend pool '{pool}': {down}/{total} instances unhealthy\n"
            "<TIMESTAMP> WARN [lb] Serving degraded: capacity at {cap}%\n"
            "<TIMESTAMP> ERROR [lb] HTTP 502 rate: {rate}% of requests"
        ),
        "severity": "HIGH",
        "incident_type": "Load Balancer Backend Degradation",
        "root_cause": (
            "{down} of {total} instances in backend pool '{pool}' are unhealthy, leaving "
            "only {cap}% serving capacity and causing {rate}% of requests to fail with "
            "HTTP 502."
        ),
        "summary": (
            "The load balancer's '{pool}' backend pool has lost {down} of {total} "
            "instances. Remaining capacity ({cap}%) cannot absorb the load, so {rate}% "
            "of requests are returning 502 errors to clients."
        ),
        "recommended_actions": [
            "Investigate why the {down} instances went unhealthy (crash, deploy, host)",
            "Scale up healthy instances to restore capacity",
            "Enable request shedding or a maintenance page if 502s worsen",
            "Review deployment strategy if a rollout caused simultaneous failures",
        ],
        "variants": [
            {"pool": "web-frontend", "down": 5, "total": 8, "cap": 37, "rate": 22},
            {"pool": "api-backend", "down": 2, "total": 6, "cap": 66, "rate": 8},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> ERROR [{svc}] Timeout calling {dep} after {t}ms\n"
            "<TIMESTAMP> ERROR [{svc}] Timeout calling {dep} after {t}ms\n"
            "<TIMESTAMP> WARN [{svc}] Thread pool saturated: {busy}/{size} workers busy\n"
            "<TIMESTAMP> ERROR [upstream] {svc} response time degraded, timeouts propagating"
        ),
        "severity": "HIGH",
        "incident_type": "Cascading Dependency Timeout",
        "root_cause": (
            "Slow responses from {dep} are exceeding the {t}ms timeout in {svc}, tying "
            "up all {size} worker threads and propagating the slowdown to upstream "
            "callers — a classic cascading failure."
        ),
        "summary": (
            "{svc} is timing out on its dependency {dep}, its thread pool is saturated "
            "({busy}/{size}), and the latency is now cascading to upstream services. "
            "The root problem is in {dep}; everything above it is collateral."
        ),
        "recommended_actions": [
            "Investigate {dep} directly — it is the origin of the slowdown",
            "Apply a circuit breaker in {svc} to fail fast instead of queuing",
            "Reduce the {t}ms timeout if callers cannot wait that long",
            "Shed non-critical load until {dep} recovers",
        ],
        "variants": [
            {"svc": "order-svc", "dep": "inventory-svc", "t": 5000, "busy": 200, "size": 200},
            {"svc": "feed-svc", "dep": "ranking-svc", "t": 3000, "busy": 128, "size": 128},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [queue] {q} depth {depth} exceeds threshold {thresh}\n"
            "<TIMESTAMP> WARN [consumer] Processing lag {lag}s and growing\n"
            "<TIMESTAMP> WARN [queue] Oldest message age: {age}s"
        ),
        "severity": "MEDIUM",
        "incident_type": "Message Queue Backlog",
        "root_cause": (
            "Queue {q} has grown to {depth} messages (threshold {thresh}) because "
            "consumers are processing slower than producers are publishing, creating "
            "{lag}s of processing lag."
        ),
        "summary": (
            "A backlog is building on queue {q}: depth {depth} with the oldest message "
            "{age}s old. Consumers are falling behind producers. Data is not lost but "
            "downstream processing is delayed and the backlog is still growing."
        ),
        "recommended_actions": [
            "Scale out consumers for queue {q}",
            "Check consumers for errors or slow external calls",
            "Verify producers are not publishing abnormally high volume",
            "Monitor queue depth trend after scaling",
        ],
        "variants": [
            {"q": "email-jobs", "depth": 48210, "thresh": 10000, "lag": 340, "age": 612},
            {"q": "image-processing", "depth": 9100, "thresh": 5000, "lag": 95, "age": 180},
        ],
    },

    # ============================================================
    # PERFORMANCE INCIDENTS
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> WARN [apm] {svc} p95 latency {lat}ms (baseline {base}ms)\n"
            "<TIMESTAMP> WARN [apm] Latency regression started at <TIMESTAMP>\n"
            "<TIMESTAMP> INFO [deploy] Release {rel} rolled out {mins} minutes before regression"
        ),
        "severity": "MEDIUM",
        "incident_type": "Latency Regression After Deployment",
        "root_cause": (
            "{svc} p95 latency jumped from {base}ms to {lat}ms beginning shortly after "
            "release {rel} was deployed — the release introduced a performance "
            "regression."
        ),
        "summary": (
            "A latency regression in {svc} (p95 {base}ms → {lat}ms) correlates directly "
            "with release {rel}, deployed {mins} minutes before the regression began. "
            "Requests are succeeding but significantly slower."
        ),
        "recommended_actions": [
            "Roll back release {rel} to confirm it is the cause",
            "Profile the new code path for the added latency",
            "Compare downstream call patterns before and after the release",
            "Add a latency budget check to the deployment pipeline",
        ],
        "variants": [
            {"svc": "search-api", "lat": 1900, "base": 240, "rel": "v2.14.0", "mins": 12},
            {"svc": "cart-svc", "lat": 850, "base": 110, "rel": "v5.3.1", "mins": 7},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [sys] CPU utilization {cpu}% sustained for {mins}m on {host}\n"
            "<TIMESTAMP> WARN [sys] Load average {load} on {cores}-core host\n"
            "<TIMESTAMP> WARN [app] Request processing delayed, worker queue growing"
        ),
        "severity": "MEDIUM",
        "incident_type": "CPU Saturation",
        "root_cause": (
            "Host {host} has sustained {cpu}% CPU for {mins} minutes with load average "
            "{load} on {cores} cores — the workload exceeds available compute, delaying "
            "request processing."
        ),
        "summary": (
            "CPU on {host} is saturated ({cpu}% for {mins}m, load {load} on {cores} "
            "cores). Request processing is queuing behind the CPU bottleneck. Either an "
            "abnormal workload spike or a runaway process is consuming compute."
        ),
        "recommended_actions": [
            "Identify top CPU consumers on {host} (top/pidstat)",
            "Check for runaway processes or busy loops",
            "Scale out or migrate load if the demand is legitimate",
            "Set CPU alerts to catch saturation before user impact",
        ],
        "variants": [
            {"cpu": 97, "mins": 25, "host": "app-node-02", "load": 31.4, "cores": 16},
            {"cpu": 94, "mins": 12, "host": "worker-05", "load": 18.2, "cores": 8},
        ],
    },

    # ============================================================
    # LOW / INFO — ROUTINE AND MINOR EVENTS
    # (needed so the model learns non-incident classification;
    #  supports the < 10% false positive rate target, Sec. 6.1)
    # ============================================================
    {
        "logs": (
            "<TIMESTAMP> INFO [deploy] Release {rel} deployment started\n"
            "<TIMESTAMP> INFO [deploy] Rolling update: {done}/{total} instances updated\n"
            "<TIMESTAMP> INFO [deploy] Release {rel} deployment completed successfully\n"
            "<TIMESTAMP> INFO [health] All health checks passing post-deploy"
        ),
        "severity": "INFO",
        "incident_type": "Successful Deployment",
        "root_cause": (
            "Release {rel} was rolled out across all {total} instances and post-deploy "
            "health checks pass — a routine, successful deployment."
        ),
        "summary": (
            "Release {rel} deployed successfully to {total} instances with no errors and "
            "all health checks green. No action needed."
        ),
        "recommended_actions": [
            "No immediate action required",
            "Monitor error rates and latency during the post-deploy window",
        ],
        "variants": [
            {"rel": "v3.2.0", "done": 6, "total": 6},
            {"rel": "v1.9.4", "done": 12, "total": 12},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> INFO [backup] Nightly backup started for {db}\n"
            "<TIMESTAMP> INFO [backup] {size}GB written to <FILE_PATH>\n"
            "<TIMESTAMP> INFO [backup] Backup completed in {mins}m, checksum verified"
        ),
        "severity": "INFO",
        "incident_type": "Scheduled Backup Completion",
        "root_cause": (
            "The nightly backup job for {db} ran to completion, wrote {size}GB, and "
            "passed checksum verification — routine scheduled maintenance."
        ),
        "summary": (
            "Nightly backup of {db} completed normally in {mins} minutes with verified "
            "integrity. No anomalies."
        ),
        "recommended_actions": [
            "No immediate action required",
            "Periodically test restore from backups to validate recoverability",
        ],
        "variants": [
            {"db": "orders-db", "size": 42, "mins": 18},
            {"db": "analytics-db", "size": 210, "mins": 74},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [cache] Cache hit rate {rate}% (baseline {base}%)\n"
            "<TIMESTAMP> INFO [cache] Eviction count elevated after key-space growth\n"
            "<TIMESTAMP> INFO [db] Read load slightly elevated, within capacity"
        ),
        "severity": "LOW",
        "incident_type": "Cache Hit Rate Degradation",
        "root_cause": (
            "The cache hit rate dropped from {base}% to {rate}% due to key-space growth "
            "triggering evictions; the database is absorbing the extra reads within "
            "capacity."
        ),
        "summary": (
            "Cache efficiency has degraded ({base}% → {rate}% hit rate) from increased "
            "evictions. There is no user impact yet since the database absorbs the "
            "extra load, but cache sizing should be reviewed."
        ),
        "recommended_actions": [
            "Review cache memory sizing against the grown key space",
            "Check TTL settings for unnecessarily long-lived entries",
            "Monitor database read load for capacity headroom",
        ],
        "variants": [
            {"rate": 71, "base": 93},
            {"rate": 78, "base": 91},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [ntp] Clock drift {ms}ms detected on {host}\n"
            "<TIMESTAMP> INFO [ntp] Resynchronizing with time server <IP_ADDR>\n"
            "<TIMESTAMP> INFO [ntp] Synchronization complete, offset corrected"
        ),
        "severity": "LOW",
        "incident_type": "Clock Drift Correction",
        "root_cause": (
            "Host {host} drifted {ms}ms from reference time; NTP detected the drift and "
            "resynchronized automatically."
        ),
        "summary": (
            "A {ms}ms clock drift on {host} was automatically corrected by NTP. Minor "
            "and self-healed; only recurring large drift would need investigation."
        ),
        "recommended_actions": [
            "No immediate action required",
            "Investigate hardware clock health if drift recurs frequently",
        ],
        "variants": [
            {"ms": 340, "host": "db-node-01"},
            {"ms": 780, "host": "app-node-04"},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> INFO [cron] Job '{job}' started\n"
            "<TIMESTAMP> INFO [{job}] Processed {n} records\n"
            "<TIMESTAMP> INFO [cron] Job '{job}' completed with exit code 0 in {secs}s"
        ),
        "severity": "INFO",
        "incident_type": "Scheduled Job Completion",
        "root_cause": (
            "The scheduled job '{job}' ran on time, processed {n} records, and exited "
            "cleanly — routine batch activity."
        ),
        "summary": (
            "Scheduled job '{job}' completed normally in {secs}s after processing {n} "
            "records. No anomalies detected."
        ),
        "recommended_actions": [
            "No immediate action required",
            "Track job duration trend for gradual slowdowns",
        ],
        "variants": [
            {"job": "invoice-rollup", "n": 18240, "secs": 96},
            {"job": "session-cleanup", "n": 5211, "secs": 12},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [pool] Connection pool usage {used}/{size} ({pct}%)\n"
            "<TIMESTAMP> INFO [pool] Usage elevated during peak traffic window\n"
            "<TIMESTAMP> INFO [pool] No connection wait events recorded"
        ),
        "severity": "LOW",
        "incident_type": "Elevated Connection Pool Usage",
        "root_cause": (
            "Connection pool usage reached {pct}% ({used}/{size}) during peak traffic, "
            "but no requests waited for a connection — utilization is high yet healthy."
        ),
        "summary": (
            "Pool utilization peaked at {pct}% with zero wait events, so there is no "
            "current impact. Headroom is shrinking and pool sizing should be reviewed "
            "before the next traffic peak."
        ),
        "recommended_actions": [
            "Review pool sizing against peak traffic growth trends",
            "Set an alert on connection wait events, not just usage percentage",
            "No immediate operational action required",
        ],
        "variants": [
            {"used": 42, "size": 50, "pct": 84},
            {"used": 86, "size": 100, "pct": 86},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [tls-mon] Certificate for {domain} expires in {days} days\n"
            "<TIMESTAMP> INFO [tls-mon] Auto-renewal scheduled\n"
            "<TIMESTAMP> INFO [tls-mon] Renewal dry-run succeeded"
        ),
        "severity": "LOW",
        "incident_type": "Upcoming Certificate Expiry",
        "root_cause": (
            "The certificate for {domain} is {days} days from expiry; auto-renewal is "
            "scheduled and its dry-run succeeded, so renewal should proceed normally."
        ),
        "summary": (
            "Advance notice: {domain}'s certificate expires in {days} days. Automation "
            "is in place and validated by dry-run. Only confirmation after renewal is "
            "needed."
        ),
        "recommended_actions": [
            "Verify the certificate renews successfully on the scheduled date",
            "Confirm the renewed certificate deploys to all endpoints",
        ],
        "variants": [
            {"domain": "api.example.com", "days": 14},
            {"domain": "www.example.org", "days": 21},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> INFO [autoscaler] Scale-up triggered: CPU {cpu}% > target {target}%\n"
            "<TIMESTAMP> INFO [autoscaler] Instances: {old} -> {new}\n"
            "<TIMESTAMP> INFO [autoscaler] New instances healthy, load normalized"
        ),
        "severity": "INFO",
        "incident_type": "Autoscaling Event",
        "root_cause": (
            "CPU utilization ({cpu}%) exceeded the {target}% autoscaling target, so the "
            "autoscaler added instances ({old} → {new}) and load normalized — the "
            "system responded to demand as designed."
        ),
        "summary": (
            "Routine autoscaling: capacity grew from {old} to {new} instances in "
            "response to CPU load and the new instances are healthy. No intervention "
            "needed."
        ),
        "recommended_actions": [
            "No immediate action required",
            "Review scaling frequency for cost optimization if events are frequent",
        ],
        "variants": [
            {"cpu": 82, "target": 70, "old": 4, "new": 6},
            {"cpu": 88, "target": 75, "old": 8, "new": 11},
        ],
    },
    {
        "logs": (
            "<TIMESTAMP> WARN [api] Rate limit applied to client <UUID>: {n} requests in {secs}s\n"
            "<TIMESTAMP> INFO [api] HTTP 429 returned with Retry-After header\n"
            "<TIMESTAMP> INFO [api] Client backed off, request rate normalized"
        ),
        "severity": "LOW",
        "incident_type": "API Rate Limiting",
        "root_cause": (
            "A client exceeded the API rate limit ({n} requests in {secs}s), received "
            "429 responses, and backed off — the rate limiter worked as intended."
        ),
        "summary": (
            "Rate limiting engaged against a client sending {n} requests in {secs}s. "
            "The client honored Retry-After and normalized. Protective mechanism "
            "functioning correctly; no service impact."
        ),
        "recommended_actions": [
            "No immediate action required",
            "Contact the client owner if the burst pattern repeats regularly",
        ],
        "variants": [
            {"n": 4800, "secs": 60},
            {"n": 1250, "secs": 10},
        ],
    },
]


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------

def generate_synthetic_examples() -> list[dict]:
    """
    Expand every template x variant into an instruction-following
    example: {"instruction", "input", "output"}.

    Both the input logs and the output fields are rendered from the
    same variant values, guaranteeing label/input consistency.
    """
    dataset = []

    for template in SYNTHETIC_TEMPLATES:
        for variant in template["variants"]:

            logs = template["logs"].format(**variant)

            output = format_output(
                severity=template["severity"],
                incident_type=template["incident_type"],
                root_cause=template["root_cause"].format(**variant),
                summary=template["summary"].format(**variant),
                recommended_actions=[
                    action.format(**variant)
                    for action in template["recommended_actions"]
                ],
            )

            dataset.append(
                {
                    "instruction": SYSTEM_PROMPT,
                    "input": logs,
                    "output": output,
                }
            )

    return dataset


if __name__ == "__main__":
    examples = generate_synthetic_examples()

    severities = {}
    for ex in examples:
        sev = ex["output"].split("\n")[0].replace("SEVERITY:", "").strip()
        severities[sev] = severities.get(sev, 0) + 1

    print("=" * 60)
    print("Synthetic Dataset Generator")
    print("=" * 60)
    print(f"Templates : {len(SYNTHETIC_TEMPLATES)}")
    print(f"Examples  : {len(examples)}")
    print(f"Severity  : {severities}")
