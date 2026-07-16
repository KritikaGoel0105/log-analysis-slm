"""
label_generator.py

Rule-based label generator for processed log windows.
Uses source-aware pattern matching to assign severity,
incident type, root cause, summary, and recommended actions
that are specific to the actual log content.

Each LogHub source has known log patterns. This module maps
those patterns to structured labels following the 5-field
output format defined in the internship document (Section 5.3).
"""

import re
from .templates import format_output


# ================================================================
# Source-specific rule sets
# ================================================================
# Each rule is a dict with:
#   "pattern"  : compiled regex to match against the joined log text
#   "severity" : CRITICAL | HIGH | MEDIUM | LOW | INFO
#   "incident_type" : brief category
#   "root_cause_fn" : callable(text) -> str  (returns root cause)
#   "summary_fn"    : callable(text) -> str  (returns 2-3 sentence summary)
#   "actions"       : list[str]
#
# Rules are evaluated in order; first match wins.
# ================================================================


def _count_occurrences(text, pattern):
    """Count how many times a regex pattern occurs in text."""
    return len(re.findall(pattern, text, re.IGNORECASE))


# ----- Apache -----

APACHE_RULES = [
    {
        "pattern": re.compile(
            r"\[error\].*mod_jk.*error state", re.IGNORECASE
        ),
        "severity": "HIGH",
        "incident_type": "Application Server Connector Failure",
        "root_cause_fn": lambda t: (
            "The mod_jk connector between Apache and the backend "
            "application server (Tomcat/JBoss) has entered an error "
            "state, indicating the backend worker is unresponsive or "
            "has crashed."
        ),
        "summary_fn": lambda t: (
            "Apache's mod_jk module reports worker environment errors, "
            "meaning the connection to the backend Java application "
            "server is broken. Requests routed through this worker "
            "will fail until the backend is restarted."
        ),
        "actions": [
            "Check the backend application server (Tomcat/JBoss) status and restart if down",
            "Review mod_jk worker configuration for connectivity issues",
            "Check backend application logs for crash or OOM errors",
            "Monitor worker recovery after restart",
        ],
    },
    {
        "pattern": re.compile(
            r"\[error\].*jk2_init.*child.*scoreboard", re.IGNORECASE
        ),
        "severity": "MEDIUM",
        "incident_type": "Worker Process Initialization Issue",
        "root_cause_fn": lambda t: (
            "Apache child worker processes are being re-initialized "
            "in the scoreboard, which indicates worker recycling or "
            "recovery from a previous error state."
        ),
        "summary_fn": lambda t: (
            "Apache mod_jk is re-initializing child workers in the "
            "scoreboard. This typically follows a backend connectivity "
            "issue and indicates the system is attempting recovery."
        ),
        "actions": [
            "Verify backend application server is running and healthy",
            "Monitor for repeated worker re-initialization cycles",
            "Check Apache error logs for the initial failure cause",
            "Review mod_jk timeout and retry settings",
        ],
    },
    {
        "pattern": re.compile(
            r"\[notice\].*workerEnv\.init\(\) ok", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "Service Startup Event",
        "root_cause_fn": lambda t: (
            "Apache mod_jk worker environment initialized successfully. "
            "This is a normal startup or reload event."
        ),
        "summary_fn": lambda t: (
            "The Apache mod_jk worker environment has been initialized "
            "and the connection to the backend application server is "
            "operational. No issues detected."
        ),
        "actions": [
            "No immediate action required",
            "Verify backend application is serving requests correctly",
            "Confirm end-to-end connectivity through a test request",
        ],
    },
    {
        "pattern": re.compile(
            r"\[error\]", re.IGNORECASE
        ),
        "severity": "HIGH",
        "incident_type": "Web Server Error",
        "root_cause_fn": lambda t: (
            "Apache web server has logged error-level events indicating "
            "request processing failures or module-level problems."
        ),
        "summary_fn": lambda t: (
            "Apache error logs indicate one or more request processing "
            "failures. These errors may affect client requests and "
            "should be investigated for their specific cause."
        ),
        "actions": [
            "Review Apache error log for specific failure details",
            "Check backend service health and connectivity",
            "Verify Apache configuration is correct",
            "Monitor error rate for trend analysis",
        ],
    },
]


# ----- OpenSSH -----

OPENSSH_RULES = [
    {
        "pattern": re.compile(
            r"(Failed password|authentication failure|invalid user).*"
            r"(Failed password|authentication failure|invalid user)",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "HIGH",
        "incident_type": "SSH Brute Force Attack",
        "root_cause_fn": lambda t: (
            "Multiple SSH authentication failures detected from "
            + (
                "the same IP address"
                if _count_occurrences(t, r"<IP_ADDR>") <= 2
                else "multiple IP addresses"
            )
            + ", indicating a brute-force login attempt against "
            "the SSH service."
        ),
        "summary_fn": lambda t: (
            "Repeated SSH login failures are occurring, characteristic "
            "of a brute-force attack. "
            + (
                "The attacker is targeting the root account. "
                if "root" in t.lower()
                else "The attacker is targeting multiple user accounts. "
            )
            + "The SSH service is under active attack and access "
            "controls should be reviewed immediately."
        ),
        "actions": [
            "Block the offending IP addresses using firewall rules (iptables/ufw)",
            "Enable fail2ban or similar intrusion prevention for SSH",
            "Disable SSH root login (PermitRootLogin no) if not already disabled",
            "Review authorized_keys and ensure key-based authentication is preferred",
        ],
    },
    {
        "pattern": re.compile(
            r"reverse mapping.*POSSIBLE BREAK-IN", re.IGNORECASE
        ),
        "severity": "MEDIUM",
        "incident_type": "DNS Reverse Lookup Anomaly",
        "root_cause_fn": lambda t: (
            "SSH reverse DNS lookup for the connecting IP address does "
            "not match the forward DNS resolution, which can indicate "
            "a spoofed origin or misconfigured DNS."
        ),
        "summary_fn": lambda t: (
            "SSH detected a reverse DNS mapping inconsistency for an "
            "incoming connection. While this can be a false alarm from "
            "misconfigured DNS, it may also indicate IP spoofing or "
            "a compromised host."
        ),
        "actions": [
            "Verify DNS records for the connecting host",
            "Check if the source IP is in known threat intelligence feeds",
            "Review SSH access logs for additional suspicious activity",
            "Consider enabling UseDNS no in sshd_config if false alarms are frequent",
        ],
    },
    {
        "pattern": re.compile(
            r"(Failed password|authentication failure|invalid user)",
            re.IGNORECASE,
        ),
        "severity": "MEDIUM",
        "incident_type": "SSH Authentication Failure",
        "root_cause_fn": lambda t: (
            "An SSH authentication attempt failed due to "
            + (
                "an invalid username that does not exist on the system."
                if "invalid user" in t.lower()
                else "incorrect credentials for an existing account."
            )
        ),
        "summary_fn": lambda t: (
            "SSH login failure detected. This could be a legitimate "
            "user mistyping their password, or an unauthorized access "
            "attempt. Single failures are low risk but should be "
            "monitored for patterns."
        ),
        "actions": [
            "Check if the login attempt was from a known user or IP",
            "Monitor for repeated failures from the same source",
            "Review SSH access policy and account credentials",
        ],
    },
    {
        "pattern": re.compile(
            r"Accepted (password|publickey)", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "Successful SSH Login",
        "root_cause_fn": lambda t: (
            "A user successfully authenticated to the SSH service "
            "using valid credentials."
        ),
        "summary_fn": lambda t: (
            "Successful SSH login recorded. This is a routine "
            "authentication event and no issues are indicated."
        ),
        "actions": [
            "No immediate action required",
            "Verify the login was from an expected user and source IP",
            "Maintain audit trail of SSH access for compliance",
        ],
    },
]


# ----- HDFS -----

HDFS_RULES = [
    {
        "pattern": re.compile(
            r"(Exception|error).*block.*replicate|BLOCK.*NameSystem.*"
            r"(delete|remove|invalid)",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "HIGH",
        "incident_type": "HDFS Block Replication Failure",
        "root_cause_fn": lambda t: (
            "HDFS block replication encountered errors, indicating "
            "that some data blocks could not be replicated to the "
            "required number of DataNodes. This may be caused by "
            "DataNode failures or network issues between nodes."
        ),
        "summary_fn": lambda t: (
            "HDFS is experiencing block replication failures. Data "
            "redundancy is compromised for affected blocks, which "
            "increases the risk of data loss if additional DataNodes "
            "fail before replication recovers."
        ),
        "actions": [
            "Check DataNode health status using hdfs dfsadmin -report",
            "Review HDFS balancer status for uneven block distribution",
            "Verify network connectivity between NameNode and DataNodes",
            "Monitor under-replicated block count until resolved",
        ],
    },
    {
        "pattern": re.compile(
            r"PacketResponder.*for block", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "HDFS Block Transfer Operation",
        "root_cause_fn": lambda t: (
            "HDFS DataNode packet responders are processing block "
            "transfer operations. This is normal data pipeline "
            "activity during writes and replication."
        ),
        "summary_fn": lambda t: (
            "HDFS DataNodes are actively handling block write and "
            "replication traffic. Packet responders are processing "
            "data pipeline operations normally. No issues indicated."
        ),
        "actions": [
            "No immediate action required",
            "Monitor DataNode throughput if performance concerns arise",
            "Check for slow DataNodes that may be bottlenecking writes",
        ],
    },
    {
        "pattern": re.compile(
            r"BLOCK.*NameSystem.*addStoredBlock|blockMap updated",
            re.IGNORECASE,
        ),
        "severity": "INFO",
        "incident_type": "HDFS Block Registration",
        "root_cause_fn": lambda t: (
            "The NameNode is registering newly stored blocks in its "
            "block map. This is a normal part of the HDFS write path "
            "and block replication process."
        ),
        "summary_fn": lambda t: (
            "HDFS NameNode is updating its block map with newly "
            "stored blocks. This indicates normal cluster operation "
            "with active data writes or replication."
        ),
        "actions": [
            "No immediate action required",
            "Monitor NameNode heap usage during heavy write operations",
            "Verify block counts are consistent across the cluster",
        ],
    },
    {
        "pattern": re.compile(
            r"Receiving block|Received block", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "HDFS Data Ingestion",
        "root_cause_fn": lambda t: (
            "DataNodes are receiving new data blocks as part of "
            "normal HDFS write or replication operations."
        ),
        "summary_fn": lambda t: (
            "HDFS DataNodes are receiving data blocks. This is "
            "routine cluster activity indicating active data writes "
            "or block replication. No anomalies detected."
        ),
        "actions": [
            "No immediate action required",
            "Monitor disk space on DataNodes during heavy ingestion",
            "Check write throughput if application performance degrades",
        ],
    },
]


# ----- BGL (Blue Gene/L) -----

BGL_RULES = [
    {
        "pattern": re.compile(
            r"FATAL.*data (TLB|storage|address)", re.IGNORECASE
        ),
        "severity": "CRITICAL",
        "incident_type": "Hardware Memory Fault",
        "root_cause_fn": lambda t: (
            "A fatal hardware memory error occurred in the Blue Gene/L "
            "compute node. The data TLB (Translation Lookaside Buffer) "
            "or data storage interrupt indicates a memory subsystem "
            "failure, possibly due to faulty DIMM or ECC uncorrectable "
            "error."
        ),
        "summary_fn": lambda t: (
            "A Blue Gene/L compute node has experienced a fatal "
            "hardware memory error (data TLB/storage interrupt). "
            "The affected node's computation is corrupted and the "
            "job running on this node has likely failed or produced "
            "incorrect results."
        ),
        "actions": [
            "Isolate the affected compute node from the job scheduler",
            "Run hardware diagnostics on the node's memory subsystem",
            "Replace faulty DIMM if ECC uncorrectable errors are confirmed",
            "Resubmit affected jobs to healthy nodes",
        ],
    },
    {
        "pattern": re.compile(
            r"FATAL.*machine check|FATAL.*instruction",
            re.IGNORECASE,
        ),
        "severity": "CRITICAL",
        "incident_type": "Hardware Machine Check Exception",
        "root_cause_fn": lambda t: (
            "A fatal machine check exception occurred on a Blue Gene/L "
            "compute node, indicating an unrecoverable hardware error "
            "in the CPU, cache, or memory controller."
        ),
        "summary_fn": lambda t: (
            "A compute node has suffered an unrecoverable machine check "
            "exception. This is a critical hardware failure that has "
            "terminated the running computation. The node requires "
            "hardware inspection and repair."
        ),
        "actions": [
            "Mark the affected node as unavailable in the job scheduler",
            "Run full hardware diagnostic suite on the node",
            "Check for related errors on neighboring nodes in the same midplane",
            "Reschedule affected jobs on healthy hardware",
        ],
    },
    {
        "pattern": re.compile(
            r"SEVERE.*ciod.*Error|SEVERE.*signal", re.IGNORECASE
        ),
        "severity": "HIGH",
        "incident_type": "Compute Node Process Failure",
        "root_cause_fn": lambda t: (
            "The CIOD (Compute I/O Daemon) on a Blue Gene/L compute "
            "node reported an error or signal, indicating that a user "
            "process terminated abnormally due to segfault, bus error, "
            "or other signal."
        ),
        "summary_fn": lambda t: (
            "A compute node process failed with a CIOD error or "
            "unexpected signal. This typically indicates an application "
            "crash rather than hardware failure, possibly due to memory "
            "access violations or resource exhaustion."
        ),
        "actions": [
            "Review the application code for memory access bugs",
            "Check resource limits (memory, file descriptors) on the node",
            "Examine core dump if available for root cause analysis",
            "Resubmit the job with additional debugging enabled",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*CE sym|INFO.*correctable", re.IGNORECASE
        ),
        "severity": "LOW",
        "incident_type": "Correctable Memory Error",
        "root_cause_fn": lambda t: (
            "A correctable ECC (Error Correcting Code) memory error "
            "was detected and automatically corrected by the hardware. "
            "This indicates minor memory cell degradation."
        ),
        "summary_fn": lambda t: (
            "The hardware detected and corrected a single-bit memory "
            "error using ECC. While individual correctable errors are "
            "harmless, a high rate of such errors on the same DIMM "
            "may predict an impending uncorrectable failure."
        ),
        "actions": [
            "Log the error location (DIMM, rank, bank) for trending",
            "Monitor correctable error rate on this node over the next 24 hours",
            "Schedule preventive DIMM replacement if error rate exceeds threshold",
            "No immediate impact on running jobs",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*generating core", re.IGNORECASE
        ),
        "severity": "MEDIUM",
        "incident_type": "Application Core Dump",
        "root_cause_fn": lambda t: (
            "A compute node is generating a core dump file, indicating "
            "that a running process crashed and the system is preserving "
            "its memory state for post-mortem debugging."
        ),
        "summary_fn": lambda t: (
            "A core dump is being generated on a Blue Gene/L compute "
            "node following a process crash. The running job on this "
            "node has terminated abnormally and its output may be "
            "incomplete or corrupted."
        ),
        "actions": [
            "Retrieve and analyze the core dump for crash root cause",
            "Check if the crash correlates with hardware errors on the node",
            "Review job logs for error messages preceding the crash",
            "Resubmit the job once root cause is understood",
        ],
    },
]


# ----- Linux -----

LINUX_RULES = [
    {
        "pattern": re.compile(
            r"authentication failure.*authentication failure",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "HIGH",
        "incident_type": "Repeated Authentication Failure",
        "root_cause_fn": lambda t: (
            "Multiple PAM authentication failures detected on the "
            "Linux system. The repeated failures from "
            + (
                "an unknown user account suggest an unauthorized "
                "access attempt."
                if "user unknown" in t.lower()
                else "a known account may indicate a compromised "
                "or brute-forced password."
            )
        ),
        "summary_fn": lambda t: (
            "The system's PAM authentication module is logging "
            "repeated login failures. This pattern is consistent "
            "with brute-force access attempts against the SSH or "
            "login service. Account lockout policies should be "
            "verified."
        ),
        "actions": [
            "Review /var/log/auth.log for source IPs and targeted accounts",
            "Block offending IPs with iptables or fail2ban",
            "Verify that password complexity policies are enforced",
            "Consider enabling account lockout after repeated failures",
        ],
    },
    {
        "pattern": re.compile(
            r"authentication failure", re.IGNORECASE
        ),
        "severity": "MEDIUM",
        "incident_type": "Authentication Failure",
        "root_cause_fn": lambda t: (
            "A PAM authentication failure occurred on the Linux "
            "system, indicating a login attempt with invalid "
            "credentials."
        ),
        "summary_fn": lambda t: (
            "A single authentication failure was detected via PAM. "
            "This may be a legitimate user mistyping credentials or "
            "a probing attempt. Monitor for recurrence."
        ),
        "actions": [
            "Check if the failed login was from a known user",
            "Monitor auth logs for additional failures from the same source",
            "Verify account status and credential validity",
        ],
    },
    {
        "pattern": re.compile(
            r"session (opened|closed).*cron", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "Scheduled Task Execution",
        "root_cause_fn": lambda t: (
            "The PAM session manager recorded cron job session "
            "open/close events, indicating normal scheduled task "
            "execution."
        ),
        "summary_fn": lambda t: (
            "Cron job sessions are being opened and closed as part "
            "of normal scheduled task execution. No issues detected."
        ),
        "actions": [
            "No immediate action required",
            "Verify cron jobs are executing as expected",
            "Review crontab entries if unexpected tasks appear",
        ],
    },
    {
        "pattern": re.compile(
            r"segfault|Oops|kernel panic|BUG:", re.IGNORECASE
        ),
        "severity": "CRITICAL",
        "incident_type": "Kernel/Process Crash",
        "root_cause_fn": lambda t: (
            "A kernel-level error or process segmentation fault "
            "was detected, indicating a critical software or "
            "hardware failure."
        ),
        "summary_fn": lambda t: (
            "The Linux kernel or a user-space process has crashed "
            "with a segfault or kernel panic. This is a critical "
            "event that may require system restart and investigation."
        ),
        "actions": [
            "Check dmesg and /var/log/kern.log for full stack trace",
            "Identify the crashing process or module",
            "Update kernel and drivers if a known bug is identified",
            "Monitor system stability after recovery",
        ],
    },
]


# ----- Android -----

ANDROID_RULES = [
    {
        "pattern": re.compile(
            r"PowerManagerService.*acquire lock.*WindowManager",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "INFO",
        "incident_type": "Power Management Activity",
        "root_cause_fn": lambda t: (
            "The Android PowerManagerService and WindowManager are "
            "handling wake lock acquisitions and window state changes "
            "as part of normal system operation."
        ),
        "summary_fn": lambda t: (
            "Android system services (PowerManagerService, "
            "WindowManager) are performing routine operations "
            "including wake lock management and display state "
            "transitions. This is normal system activity."
        ),
        "actions": [
            "No immediate action required",
            "Monitor battery drain if excessive wake locks are suspected",
            "Review wake lock holding duration for optimization",
        ],
    },
    {
        "pattern": re.compile(
            r"ANR|Application Not Responding", re.IGNORECASE
        ),
        "severity": "HIGH",
        "incident_type": "Application Not Responding",
        "root_cause_fn": lambda t: (
            "An Android application failed to respond to user input "
            "within the timeout period, triggering an ANR (Application "
            "Not Responding) event. This is typically caused by "
            "blocking operations on the main/UI thread."
        ),
        "summary_fn": lambda t: (
            "An ANR event was detected, meaning an application "
            "became unresponsive to user interaction. The main "
            "thread was likely blocked by a long-running operation "
            "such as network I/O, database queries, or heavy "
            "computation."
        ),
        "actions": [
            "Review the ANR trace file for the blocking call stack",
            "Move long-running operations to background threads",
            "Optimize database queries and I/O operations",
            "Profile the application for main thread bottlenecks",
        ],
    },
    {
        "pattern": re.compile(
            r"FATAL EXCEPTION|java\.lang\.\w+Exception",
            re.IGNORECASE,
        ),
        "severity": "HIGH",
        "incident_type": "Application Crash",
        "root_cause_fn": lambda t: (
            "An Android application threw an uncaught exception, "
            "causing a fatal crash. The exception type indicates "
            "a programming error in the application code."
        ),
        "summary_fn": lambda t: (
            "An Android application has crashed due to an unhandled "
            "Java/Kotlin exception. The crash will have been reported "
            "to the system's crash handler and the application "
            "process was terminated."
        ),
        "actions": [
            "Collect the crash stack trace from logcat",
            "Identify the exception type and root cause in the code",
            "Fix the code path that produced the uncaught exception",
            "Add proper exception handling for the affected component",
        ],
    },
    {
        "pattern": re.compile(
            r"GC_|garbage collect|dalvik.*gc", re.IGNORECASE
        ),
        "severity": "LOW",
        "incident_type": "Garbage Collection Event",
        "root_cause_fn": lambda t: (
            "The Android runtime's garbage collector has run to "
            "reclaim unused memory. Frequent GC events may indicate "
            "memory pressure from the application."
        ),
        "summary_fn": lambda t: (
            "Garbage collection events are being logged by the "
            "Android runtime. Occasional GC is normal, but frequent "
            "or long-pause GC may indicate memory pressure."
        ),
        "actions": [
            "Monitor GC frequency and pause duration",
            "Profile application heap usage for memory leaks",
            "Optimize object allocation patterns if GC is excessive",
        ],
    },
]


# ----- Hadoop -----

HADOOP_RULES = [
    {
        "pattern": re.compile(
            r"(Exception|error).*Communication|Connection refused",
            re.IGNORECASE,
        ),
        "severity": "HIGH",
        "incident_type": "Hadoop Node Communication Failure",
        "root_cause_fn": lambda t: (
            "A Hadoop cluster node failed to communicate with another "
            "node, indicating either a network issue, a down service, "
            "or firewall blocking the RPC port."
        ),
        "summary_fn": lambda t: (
            "Communication failure detected between Hadoop cluster "
            "nodes. This may disrupt HDFS operations, MapReduce jobs, "
            "or YARN resource allocation depending on which services "
            "are affected."
        ),
        "actions": [
            "Check network connectivity between the affected nodes",
            "Verify the target service (NameNode/DataNode/ResourceManager) is running",
            "Review firewall rules for Hadoop RPC ports",
            "Check cluster node health via the YARN ResourceManager UI",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*Attempt.*task_|mapreduce.*task", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "MapReduce Task Execution",
        "root_cause_fn": lambda t: (
            "Hadoop MapReduce task attempts are being logged as part "
            "of normal job execution on the cluster."
        ),
        "summary_fn": lambda t: (
            "MapReduce tasks are executing on the Hadoop cluster. "
            "This is normal job processing activity with task attempts "
            "being scheduled, executed, and reported."
        ),
        "actions": [
            "No immediate action required",
            "Monitor job completion time for performance baseline",
            "Check task failure rate if job performance degrades",
        ],
    },
    {
        "pattern": re.compile(
            r"WARN.*lease|WARN.*heartbeat|WARN.*timeout",
            re.IGNORECASE,
        ),
        "severity": "MEDIUM",
        "incident_type": "Hadoop Cluster Health Warning",
        "root_cause_fn": lambda t: (
            "Hadoop cluster management has issued warnings about "
            "lease renewals, heartbeat timeouts, or node responsiveness, "
            "suggesting intermittent connectivity or overloaded nodes."
        ),
        "summary_fn": lambda t: (
            "Warning-level events from Hadoop cluster management "
            "indicate potential health issues such as missed heartbeats "
            "or lease timeouts. These may resolve on their own but "
            "should be monitored for escalation."
        ),
        "actions": [
            "Check DataNode and NodeManager heartbeat status",
            "Review network latency between cluster nodes",
            "Verify cluster nodes have sufficient CPU and memory resources",
            "Monitor for escalation to node failures",
        ],
    },
]


# ----- HPC -----

HPC_RULES = [
    {
        "pattern": re.compile(
            r"error.*node|node.*down|node.*fail", re.IGNORECASE
        ),
        "severity": "HIGH",
        "incident_type": "HPC Compute Node Failure",
        "root_cause_fn": lambda t: (
            "An HPC compute node has experienced an error or gone "
            "offline, which will affect running jobs scheduled on "
            "that node."
        ),
        "summary_fn": lambda t: (
            "An HPC cluster node failure has been detected. Jobs "
            "running on the affected node may have failed or will "
            "need to be rescheduled to healthy nodes."
        ),
        "actions": [
            "Check node hardware status via IPMI/BMC",
            "Drain the failed node from the job scheduler",
            "Requeue affected jobs to available healthy nodes",
            "Run hardware diagnostics before returning the node to service",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*PBS|INFO.*job|starting|completed", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "HPC Job Scheduling Event",
        "root_cause_fn": lambda t: (
            "The HPC job scheduler is processing job submissions, "
            "starting executions, or recording completions as part "
            "of normal cluster operation."
        ),
        "summary_fn": lambda t: (
            "HPC job scheduler events indicate normal cluster "
            "operation with jobs being queued, started, and "
            "completed. No anomalies detected."
        ),
        "actions": [
            "No immediate action required",
            "Monitor queue wait times for scheduling efficiency",
            "Check cluster utilization for capacity planning",
        ],
    },
]


# ----- HealthApp -----

HEALTHAPP_RULES = [
    {
        "pattern": re.compile(
            r"(error|fail|exception|crash)", re.IGNORECASE
        ),
        "severity": "MEDIUM",
        "incident_type": "Health Application Error",
        "root_cause_fn": lambda t: (
            "The health monitoring application encountered an error "
            "during data collection or processing, which may affect "
            "the accuracy of health metrics being reported."
        ),
        "summary_fn": lambda t: (
            "Errors were detected in the health application logs. "
            "These may affect health data collection and should be "
            "investigated to ensure monitoring accuracy."
        ),
        "actions": [
            "Review application logs for specific error details",
            "Verify health data collection is still functioning",
            "Restart the health application if errors persist",
            "Check sensor connectivity and data source availability",
        ],
    },
    {
        "pattern": re.compile(r".*", re.DOTALL),
        "severity": "INFO",
        "incident_type": "Health Data Collection Event",
        "root_cause_fn": lambda t: (
            "The health monitoring application is recording routine "
            "data collection and processing events."
        ),
        "summary_fn": lambda t: (
            "Normal health application activity recorded, including "
            "data collection, step counting, or sensor readings. "
            "No anomalies detected."
        ),
        "actions": [
            "No immediate action required",
            "Verify data collection intervals are as configured",
            "Monitor for gaps in health data recording",
        ],
    },
]


# ----- Mac -----

MAC_RULES = [
    {
        "pattern": re.compile(
            r"kernel.*panic|kernel.*error|NVDA.*error", re.IGNORECASE
        ),
        "severity": "CRITICAL",
        "incident_type": "macOS Kernel Error",
        "root_cause_fn": lambda t: (
            "A macOS kernel-level error or panic has occurred, "
            "indicating a critical system failure possibly related "
            "to driver incompatibility or hardware issues."
        ),
        "summary_fn": lambda t: (
            "The macOS kernel has reported critical errors. This "
            "may cause system instability or require a reboot to "
            "recover normal operation."
        ),
        "actions": [
            "Check Console.app and /var/log/system.log for full crash details",
            "Update macOS and all drivers to the latest versions",
            "Run Apple Diagnostics to check for hardware issues",
            "Monitor for recurrence after restart",
        ],
    },
    {
        "pattern": re.compile(
            r"error|WARNING|failed", re.IGNORECASE
        ),
        "severity": "LOW",
        "incident_type": "macOS System Warning",
        "root_cause_fn": lambda t: (
            "macOS system services have logged warning or error "
            "events that are typically non-critical and related "
            "to normal system operation edge cases."
        ),
        "summary_fn": lambda t: (
            "Warning-level events from macOS system services. "
            "These are common during normal operation and typically "
            "do not indicate serious issues unless they recur "
            "frequently."
        ),
        "actions": [
            "Review Console.app for specific warning details",
            "Update relevant applications if warnings are app-specific",
            "Monitor for escalation to error-level events",
        ],
    },
    {
        "pattern": re.compile(r".*", re.DOTALL),
        "severity": "INFO",
        "incident_type": "macOS System Activity",
        "root_cause_fn": lambda t: (
            "Normal macOS system service activity including process "
            "management, service lifecycle events, and system "
            "housekeeping operations."
        ),
        "summary_fn": lambda t: (
            "Routine macOS system activity recorded. System services "
            "are operating normally with standard lifecycle events."
        ),
        "actions": [
            "No immediate action required",
            "Review logs periodically for emerging patterns",
        ],
    },
]


# ----- OpenStack -----

OPENSTACK_RULES = [
    {
        "pattern": re.compile(
            r"(ERROR|CRITICAL).*nova|nova.*(error|fail|exception)",
            re.IGNORECASE,
        ),
        "severity": "HIGH",
        "incident_type": "OpenStack Compute Service Error",
        "root_cause_fn": lambda t: (
            "The OpenStack Nova compute service has encountered "
            "errors that may affect VM instance management, "
            "scheduling, or API operations."
        ),
        "summary_fn": lambda t: (
            "OpenStack Nova (compute) service errors detected. "
            "These may impact virtual machine provisioning, "
            "migration, or lifecycle management operations "
            "on the cloud platform."
        ),
        "actions": [
            "Check Nova service status with 'openstack compute service list'",
            "Review Nova logs on affected compute/controller nodes",
            "Verify RabbitMQ message queue connectivity",
            "Check database (MariaDB/MySQL) connectivity for Nova",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*nova.*wsgi|INFO.*nova.*api", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "OpenStack API Request Processing",
        "root_cause_fn": lambda t: (
            "The OpenStack Nova API (WSGI) is processing incoming "
            "HTTP requests as part of normal cloud platform "
            "operations."
        ),
        "summary_fn": lambda t: (
            "Nova API request processing activity detected. The "
            "OpenStack compute API is handling incoming requests "
            "for VM management operations. This is normal cloud "
            "platform activity."
        ),
        "actions": [
            "No immediate action required",
            "Monitor API response times for performance degradation",
            "Check request rates against API rate limits",
        ],
    },
]


# ----- Proxifier -----

PROXIFIER_RULES = [
    {
        "pattern": re.compile(
            r"(error|fail|close.*error|refused)", re.IGNORECASE
        ),
        "severity": "MEDIUM",
        "incident_type": "Proxy Connection Failure",
        "root_cause_fn": lambda t: (
            "Proxifier encountered connection failures while routing "
            "traffic through the proxy server, indicating either "
            "proxy server unavailability or target host unreachability."
        ),
        "summary_fn": lambda t: (
            "Proxy connection failures detected. Network traffic "
            "routing through the proxy is experiencing issues that "
            "may affect application connectivity."
        ),
        "actions": [
            "Verify proxy server is running and accessible",
            "Check proxy server capacity and connection limits",
            "Test direct connectivity to target hosts",
            "Review Proxifier rules for misconfigured routes",
        ],
    },
    {
        "pattern": re.compile(r".*", re.DOTALL),
        "severity": "INFO",
        "incident_type": "Proxy Traffic Routing",
        "root_cause_fn": lambda t: (
            "Proxifier is routing application network traffic "
            "through configured proxy rules as part of normal "
            "operation."
        ),
        "summary_fn": lambda t: (
            "Normal proxy traffic routing activity. Proxifier is "
            "handling application connections according to its "
            "configured rules. No anomalies detected."
        ),
        "actions": [
            "No immediate action required",
            "Monitor proxy throughput and latency",
            "Review connection statistics for unusual patterns",
        ],
    },
]


# ----- Spark -----

SPARK_RULES = [
    {
        "pattern": re.compile(
            r"(ERROR|WARN).*Lost.*executor|ERROR.*task.*fail",
            re.IGNORECASE,
        ),
        "severity": "HIGH",
        "incident_type": "Spark Executor Failure",
        "root_cause_fn": lambda t: (
            "A Spark executor was lost or a task failed, which "
            "typically indicates OOM (Out of Memory) on the executor "
            "JVM, network disconnection from the driver, or the "
            "executor being killed by YARN for exceeding memory limits."
        ),
        "summary_fn": lambda t: (
            "Spark executor failure detected. Tasks scheduled on "
            "the lost executor will be retried on other executors. "
            "Repeated executor losses indicate resource configuration "
            "issues that need attention."
        ),
        "actions": [
            "Review executor memory configuration (spark.executor.memory)",
            "Check YARN container logs for OOM or killed-by-YARN messages",
            "Increase executor memory or reduce partition sizes",
            "Monitor executor count during job execution for stability",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*executor\.Executor.*Running task|"
            r"INFO.*CoarseGrainedExecutorBackend.*assigned task",
            re.IGNORECASE,
        ),
        "severity": "INFO",
        "incident_type": "Spark Task Execution",
        "root_cause_fn": lambda t: (
            "Spark executors are running tasks as part of normal "
            "distributed computation. Tasks are being assigned by "
            "the driver and executed on cluster workers."
        ),
        "summary_fn": lambda t: (
            "Spark tasks are being scheduled and executed across "
            "cluster executors. This is normal Spark job processing "
            "activity with no anomalies detected."
        ),
        "actions": [
            "No immediate action required",
            "Monitor task completion rate and stage progress",
            "Check for straggler tasks that may be slowing the job",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*MemoryStore.*Block.*stored|INFO.*storage",
            re.IGNORECASE,
        ),
        "severity": "INFO",
        "incident_type": "Spark Data Caching Event",
        "root_cause_fn": lambda t: (
            "Spark is storing broadcast variables or cached RDD "
            "partitions in its in-memory MemoryStore as part of "
            "normal distributed data processing."
        ),
        "summary_fn": lambda t: (
            "Spark MemoryStore is caching data blocks (broadcast "
            "variables or RDD partitions) in memory for efficient "
            "access. This is normal Spark runtime behavior."
        ),
        "actions": [
            "No immediate action required",
            "Monitor storage memory usage if jobs fail with OOM",
            "Adjust spark.memory.storageFraction if cache eviction is frequent",
        ],
    },
]


# ----- Thunderbird -----

THUNDERBIRD_RULES = [
    {
        "pattern": re.compile(
            r"unable to qualify.*domain name|sendmail.*error",
            re.IGNORECASE,
        ),
        "severity": "MEDIUM",
        "incident_type": "Mail Server Configuration Error",
        "root_cause_fn": lambda t: (
            "The sendmail service on the Thunderbird supercomputer "
            "node cannot resolve its own domain name, indicating "
            "DNS misconfiguration on the affected compute or "
            "admin node."
        ),
        "summary_fn": lambda t: (
            "Multiple Thunderbird cluster nodes report that "
            "sendmail cannot qualify its own domain name. This "
            "prevents email-based alerting from these nodes. The "
            "issue is a DNS configuration problem on the cluster."
        ),
        "actions": [
            "Verify DNS configuration on affected cluster nodes",
            "Check /etc/resolv.conf for correct nameserver entries",
            "Update sendmail configuration with the correct domain name",
            "Test DNS resolution from the affected nodes",
        ],
    },
    {
        "pattern": re.compile(
            r"ntpd.*synchronized|ntp.*stratum", re.IGNORECASE
        ),
        "severity": "INFO",
        "incident_type": "NTP Time Synchronization",
        "root_cause_fn": lambda t: (
            "NTP daemon on the cluster node has synchronized its "
            "clock with the configured time server, maintaining "
            "accurate timestamps across the cluster."
        ),
        "summary_fn": lambda t: (
            "Cluster nodes are synchronizing time via NTP. Time "
            "synchronization is critical for log correlation and "
            "distributed job coordination."
        ),
        "actions": [
            "No immediate action required",
            "Verify all cluster nodes are synced to the same stratum",
            "Monitor for clock drift warnings",
        ],
    },
    {
        "pattern": re.compile(
            r"error|FATAL|SEVERE|failure", re.IGNORECASE
        ),
        "severity": "MEDIUM",
        "incident_type": "Supercomputer Node Error",
        "root_cause_fn": lambda t: (
            "Error-level events detected on Thunderbird supercomputer "
            "nodes, which may affect cluster jobs or system services."
        ),
        "summary_fn": lambda t: (
            "Error events on Thunderbird supercomputer nodes have "
            "been detected. These should be investigated to determine "
            "impact on running jobs and cluster health."
        ),
        "actions": [
            "Review full error details in node-specific logs",
            "Check if affected nodes have running jobs",
            "Verify node health via cluster management tools",
            "Escalate to system administrators if hardware is suspected",
        ],
    },
]


# ----- Windows -----

WINDOWS_RULES = [
    {
        "pattern": re.compile(
            r"Error|Warning.*source.*Service Control Manager",
            re.IGNORECASE,
        ),
        "severity": "MEDIUM",
        "incident_type": "Windows Service Error",
        "root_cause_fn": lambda t: (
            "Windows Event Log recorded service-related errors from "
            "the Service Control Manager, indicating a Windows service "
            "failed to start, stopped unexpectedly, or encountered "
            "a dependency failure."
        ),
        "summary_fn": lambda t: (
            "Windows service errors detected via the Event Log. "
            "One or more services may have failed to start or "
            "crashed, which could affect system functionality "
            "depending on the affected service."
        ),
        "actions": [
            "Check Windows Event Viewer for detailed error descriptions",
            "Verify the affected service dependencies are running",
            "Restart the failed service and monitor for stability",
            "Review service account permissions if access-related errors",
        ],
    },
    {
        "pattern": re.compile(r".*", re.DOTALL),
        "severity": "INFO",
        "incident_type": "Windows System Event",
        "root_cause_fn": lambda t: (
            "Standard Windows Event Log entries recording system "
            "operations, service lifecycle events, and configuration "
            "changes."
        ),
        "summary_fn": lambda t: (
            "Routine Windows system events recorded in the Event "
            "Log. No critical issues detected."
        ),
        "actions": [
            "No immediate action required",
            "Review Event Viewer periodically for emerging patterns",
            "Maintain Windows Update schedule for security patches",
        ],
    },
]


# ----- Zookeeper -----

ZOOKEEPER_RULES = [
    {
        "pattern": re.compile(
            r"(WARN|ERROR).*Connection broken|Connection.*refused",
            re.IGNORECASE,
        ),
        "severity": "HIGH",
        "incident_type": "ZooKeeper Quorum Connection Failure",
        "root_cause_fn": lambda t: (
            "A ZooKeeper ensemble member lost connection to another "
            "member, breaking the quorum communication channel. "
            "This may be caused by network issues, a down ZooKeeper "
            "instance, or firewall rules blocking the quorum port."
        ),
        "summary_fn": lambda t: (
            "ZooKeeper quorum connectivity issues detected. Broken "
            "connections between ensemble members can lead to leader "
            "election instability and affect all services that depend "
            "on ZooKeeper for coordination (Kafka, HBase, etc.)."
        ),
        "actions": [
            "Verify all ZooKeeper ensemble members are running",
            "Check network connectivity between ZooKeeper nodes on quorum ports",
            "Review ZooKeeper logs on all ensemble members for the full picture",
            "Monitor ZooKeeper leader election stability",
        ],
    },
    {
        "pattern": re.compile(
            r"WARN.*SendWorker.*Interrupted|WARN.*RecvWorker",
            re.IGNORECASE,
        ),
        "severity": "MEDIUM",
        "incident_type": "ZooKeeper Worker Thread Interruption",
        "root_cause_fn": lambda t: (
            "ZooKeeper's internal send/receive worker threads have "
            "been interrupted, typically as a result of a peer "
            "connection being broken or a leadership change in "
            "the ensemble."
        ),
        "summary_fn": lambda t: (
            "ZooKeeper worker threads are being interrupted, which "
            "usually follows a peer connection loss. The ensemble "
            "may be in the process of re-establishing connections "
            "or electing a new leader."
        ),
        "actions": [
            "Check if a ZooKeeper leader election is in progress",
            "Verify peer connectivity across the ensemble",
            "Monitor for repeated interruption cycles indicating instability",
            "Review tickTime and syncLimit settings if elections are frequent",
        ],
    },
    {
        "pattern": re.compile(
            r"INFO.*Received connection request|INFO.*QuorumCnx",
            re.IGNORECASE,
        ),
        "severity": "INFO",
        "incident_type": "ZooKeeper Quorum Communication",
        "root_cause_fn": lambda t: (
            "ZooKeeper ensemble members are exchanging connection "
            "requests as part of normal quorum communication and "
            "leader election protocol."
        ),
        "summary_fn": lambda t: (
            "ZooKeeper quorum communication events observed. "
            "Ensemble members are exchanging connection requests "
            "as part of the normal ZooKeeper protocol."
        ),
        "actions": [
            "No immediate action required",
            "Verify leader has been elected and the ensemble is stable",
            "Monitor connection request rate for unusual spikes",
        ],
    },
]


# ================================================================
# Source detection and rule dispatch
# ================================================================

# Map LogHub source names (from processed JSONL filenames) to rule sets
SOURCE_RULES = {
    "apache":      APACHE_RULES,
    "openssh":     OPENSSH_RULES,
    "hdfs":        HDFS_RULES,
    "bgl":         BGL_RULES,
    "linux":       LINUX_RULES,
    "android":     ANDROID_RULES,
    "hadoop":      HADOOP_RULES,
    "hpc":         HPC_RULES,
    "healthapp":   HEALTHAPP_RULES,
    "mac":         MAC_RULES,
    "openstack":   OPENSTACK_RULES,
    "proxifier":   PROXIFIER_RULES,
    "spark":       SPARK_RULES,
    "thunderbird": THUNDERBIRD_RULES,
    "windows":     WINDOWS_RULES,
    "zookeeper":   ZOOKEEPER_RULES,
}

# Heuristic patterns for source detection when source is unknown
SOURCE_SIGNATURES = {
    "openssh":     re.compile(r"sshd\[|ssh2|preauth", re.IGNORECASE),
    "apache":      re.compile(r"mod_jk|workerEnv|jk2_init", re.IGNORECASE),
    "hdfs":        re.compile(r"dfs\.(DataNode|FSNamesystem|DataBlockScanner)|BLOCK\*|PacketResponder", re.IGNORECASE),
    "bgl":         re.compile(r"RAS KERNEL|Blue Gene|KERNDTLB|ciod:", re.IGNORECASE),
    "linux":       re.compile(r"pam_unix|sshd\(pam|CRON\[|combo ", re.IGNORECASE),
    "android":     re.compile(r"PowerManagerService|WindowManager|ActivityManager|dalvikvm", re.IGNORECASE),
    "hadoop":      re.compile(r"mapreduce\.|mapred\.|org\.apache\.hadoop", re.IGNORECASE),
    "hpc":         re.compile(r"PBS_|pbsserver|pbs_mom", re.IGNORECASE),
    "healthapp":   re.compile(r"Step.*Count|HealthApp|calculateCalories", re.IGNORECASE),
    "mac":         re.compile(r"com\.apple\.|CoreServicesUIAgent|backupd\[", re.IGNORECASE),
    "openstack":   re.compile(r"nova[\.\-]|neutron[\.\-]|cinder[\.\-]|osapi_compute", re.IGNORECASE),
    "proxifier":   re.compile(r"Proxifier|proxy.*rule|HTTPS.*close", re.IGNORECASE),
    "spark":       re.compile(r"spark\.|executor\.Executor|CoarseGrainedExecutorBackend|MemoryStore", re.IGNORECASE),
    "thunderbird": re.compile(r"tbird-|badmin|sendmail\[|sbin/gmetad", re.IGNORECASE),
    "windows":     re.compile(r"Microsoft-Windows-|Event.*Log|Service Control Manager", re.IGNORECASE),
    "zookeeper":   re.compile(r"QuorumCnxManager|ZooKeeper|FastLeader|SendWorker|RecvWorker", re.IGNORECASE),
}


def detect_source(text: str) -> str:
    """
    Attempt to detect the LogHub source from the log content.
    Returns the source key in lowercase or "unknown".
    """
    for source, sig in SOURCE_SIGNATURES.items():
        if sig.search(text):
            return source
    return "unknown"


# ================================================================
# Generic fallback rules (when source is unknown or no rule matches)
# ================================================================

GENERIC_RULES = [
    {
        "pattern": re.compile(
            r"(CRIT|CRITICAL|FATAL|EMERG|panic|kernel panic)",
            re.IGNORECASE,
        ),
        "severity": "CRITICAL",
        "incident_type": "Critical System Failure",
        "root_cause_fn": lambda t: (
            "Critical or fatal-level events detected in the logs, "
            "indicating a severe system failure requiring immediate "
            "attention."
        ),
        "summary_fn": lambda t: (
            "Critical-level log events indicate a severe system "
            "failure. The affected service or system component "
            "may be down or producing incorrect results."
        ),
        "actions": [
            "Identify and isolate the failing component",
            "Check system resources (CPU, memory, disk, network)",
            "Restart the affected service and monitor recovery",
            "Escalate to system administrators if root cause is unclear",
        ],
    },
    {
        "pattern": re.compile(
            r"(error|fail|exception|denied|refused).*"
            r"(error|fail|exception|denied|refused)",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "HIGH",
        "incident_type": "Multiple Error Events",
        "root_cause_fn": lambda t: (
            "Multiple error-level events are occurring in the log "
            "window, indicating an ongoing issue that is generating "
            "repeated failures."
        ),
        "summary_fn": lambda t: (
            "Multiple error events detected within a short time "
            "window. The repeated nature of these errors suggests "
            "a systemic issue rather than a transient glitch."
        ),
        "actions": [
            "Identify the common thread across the error messages",
            "Check upstream dependencies for the failing component",
            "Review recent configuration or deployment changes",
            "Monitor error rate for stabilization or escalation",
        ],
    },
    {
        "pattern": re.compile(
            r"(error|fail|exception|denied|refused)",
            re.IGNORECASE,
        ),
        "severity": "MEDIUM",
        "incident_type": "Service Error Event",
        "root_cause_fn": lambda t: (
            "An error-level event was detected in the logs, "
            "indicating a service operation failed or was rejected."
        ),
        "summary_fn": lambda t: (
            "An error event was logged that may indicate a service "
            "issue. Single errors may be transient, but should be "
            "monitored for recurrence."
        ),
        "actions": [
            "Review the full error message for specific failure details",
            "Check if the error correlates with a known issue",
            "Monitor for recurrence of the same error pattern",
        ],
    },
    {
        "pattern": re.compile(
            r"(warn|warning|WARN)", re.IGNORECASE
        ),
        "severity": "LOW",
        "incident_type": "System Warning",
        "root_cause_fn": lambda t: (
            "Warning-level events in the logs indicate conditions "
            "that are not yet failures but may lead to issues if "
            "left unaddressed."
        ),
        "summary_fn": lambda t: (
            "Warning events detected in the log window. These "
            "are not failures but indicate conditions that should "
            "be monitored to prevent escalation."
        ),
        "actions": [
            "Review warning details for potential action items",
            "Monitor for escalation to error-level events",
            "Address the root condition causing the warning",
        ],
    },
    {
        "pattern": re.compile(r".*", re.DOTALL),
        "severity": "INFO",
        "incident_type": "Routine System Activity",
        "root_cause_fn": lambda t: (
            "The log window contains informational system events "
            "with no error, warning, or critical indicators."
        ),
        "summary_fn": lambda t: (
            "Routine system activity logged. No errors, warnings, "
            "or anomalies detected in this log window."
        ),
        "actions": [
            "No immediate action required",
            "Continue routine log monitoring",
            "Archive logs according to retention policy",
        ],
    },
]


# ================================================================
# Main API
# ================================================================

def generate_label(log_text: str, source: str = None) -> str:
    """
    Generate structured label output for a log window.

    Parameters
    ----------
    log_text : str
        The normalized log window text (lines joined by newlines).
    source : str, optional
        The LogHub source name (e.g. "Apache", "OpenSSH").
        If not provided, source detection is attempted from content.

    Returns
    -------
    str
        Formatted 5-field output matching the document specification.
    """
    # Determine source
    if source:
        source_key = source.lower().strip()
    else:
        source_key = detect_source(log_text)

    # Get source-specific rules (fall back to generic)
    rules = SOURCE_RULES.get(source_key, GENERIC_RULES)

    # Try source-specific rules first
    for rule in rules:
        if rule["pattern"].search(log_text):
            return format_output(
                severity=rule["severity"],
                incident_type=rule["incident_type"],
                root_cause=rule["root_cause_fn"](log_text),
                summary=rule["summary_fn"](log_text),
                recommended_actions=rule["actions"],
            )

    # Fall through to generic rules if no source rule matched
    for rule in GENERIC_RULES:
        if rule["pattern"].search(log_text):
            return format_output(
                severity=rule["severity"],
                incident_type=rule["incident_type"],
                root_cause=rule["root_cause_fn"](log_text),
                summary=rule["summary_fn"](log_text),
                recommended_actions=rule["actions"],
            )

    # Should never reach here (last generic rule matches everything)
    return format_output(
        severity="INFO",
        incident_type="Unclassified Event",
        root_cause="Unable to determine root cause from available log data.",
        summary="Log window could not be classified by any known pattern.",
        recommended_actions=["Review logs manually for context"],
    )
