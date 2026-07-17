# Week 4 — Zero-Shot Baseline Evaluation Report (D3)

- **Model:** Qwen/Qwen2.5-3B-Instruct (unmodified, zero-shot)
- **Test set:** 198 held-out examples (document Section 6: 200+ log scenarios)
- **Generated:** 2026-07-17T08:39:01.044125+00:00
- **Environment:** Python 3.11.9, Windows-10-10.0.26200-SP0
- **Decoding:** greedy (deterministic, reproducible)

All inference and scoring performed **fully offline** (HF_HUB_OFFLINE=1, local model weights, no external APIs).

## Metrics vs. Section 6.1 Targets

| Metric | Baseline | Target | Status |
|---|---|---|---|
| Severity Classification Accuracy | 43.9% | > 85% | FAIL |
| Incident Type F1 (macro) | 0.001 | > 0.80 | FAIL |
| ROUGE-L (summaries) | 0.121 | > 0.55 | FAIL |
| False Positive Rate | 33.3% | < 10% | FAIL |
| Root Cause Accuracy | *human evaluation — see example outputs below and baseline_predictions.jsonl* | > 75% | MANUAL |
| API Response Time (p95) | *measured in Week 9 (/analyze endpoint); raw model p95 = 216.2s/example* | < 5 s | DEFERRED |
| RAG Retrieval Precision@3 | *measured in Week 7* | > 70% | DEFERRED |

A zero-shot baseline is **expected to miss the targets** — these numbers are the reference point that Week 5-6 fine-tuning must beat (document Section 10.1: "Fine-tuned model outperforms baseline on all key metrics").

## Output Format Compliance

- Parse failures (missing/invalid fields): 16 / 198 (8.1%)
- Unparseable fields are scored as incorrect, never dropped.

## Severity Classification Report

```
              precision    recall  f1-score   support

    CRITICAL       0.36      0.80      0.50         5
        HIGH       0.83      0.21      0.34        47
        INFO       0.82      0.63      0.71       100
         LOW       0.11      0.08      0.09        26
      MEDIUM       0.13      0.40      0.19        20
 UNPARSEABLE       0.00      0.00      0.00         0

    accuracy                           0.44       198
   macro avg       0.37      0.35      0.31       198
weighted avg       0.65      0.44      0.48       198
```

## Severity Confusion Matrix (reference rows × prediction columns)

| ref \ pred | CRITICAL | HIGH | INFO | LOW | MEDIUM | UNPARSEABLE |
|---|---|---|---|---|---|---|
| **CRITICAL** | 4 | 1 | 0 | 0 | 0 | 0 |
| **HIGH** | 5 | 10 | 8 | 4 | 15 | 5 |
| **INFO** | 0 | 0 | 63 | 8 | 29 | 0 |
| **LOW** | 2 | 0 | 2 | 2 | 11 | 9 |
| **MEDIUM** | 0 | 1 | 4 | 5 | 8 | 2 |

## Latency (per-example model inference)

| mean | p50 | p95 | max |
|---|---|---|---|
| 92.6s | 81.0s | 216.2s | 326.8s |

## Metric Implementations

- ROUGE-L: `rouge_score`
- Incident F1: `sklearn`

## Example Outputs (for human root-cause review)

### Example 0 (reference severity: INFO)

**Input (truncated):**
```
17/06/09 20:11:10 INFO storage.BlockManager: Found block rdd_42_25 locally
17/06/09 20:11:10 INFO executor.CoarseGrainedExecutorBackend: Got assigned task 1307
17/06/09 20:11:10 INFO executor.Executor: Running task 27.0 in stage 28.0 (TID 1307)
17/06/09 20:11:10 INFO executor.Executor: Finished task 24.0 in stage 28.0 (TID 1304). 2087 bytes result sent to driver
17/06/09 20:11:10 INFO python.PythonRunner: Times: total = 43, boot = 15, init = 27, finish = 1
17/06/09 20:11:10 INFO storage.BlockManager: Found block rdd_42_26 locally
17/06/09 20:11:10 INFO executor.Executor: Finished task 23.0 in 
```
**Reference output:**
```
SEVERITY: INFO
INCIDENT_TYPE: Spark Task Execution
ROOT_CAUSE: Spark executors are running tasks as part of normal distributed computation. Tasks are being assigned by the driver and executed on cluster workers.
SUMMARY: Spark tasks are being scheduled and executed across cluster executors. This is normal Spark job processing activity with no anomalies detected.
RECOMMENDED_ACTIONS:
1. No immediate action required
2. Monitor task completion rate and stage progress
3. Check for straggler tasks that may be slowing the job
```
**Model prediction:**
```
1. SEVERITY: INFO
2. INCIDENT_TYPE: Task Assignment and Execution
3. ROOT_CAUSE: No issues or errors reported; tasks were successfully assigned and executed.
4. SUMMARY: The system successfully assigned and executed multiple tasks for data processing without any errors or failures. Blocks were found locally, allowing for efficient task completion.
5. RECOMMENDED_ACTIONS:
1. Verify that the workload is balanced across all available executors to ensure optimal resource utilization.
2. Monitor the system for any future anomalies or delays in task execution.
```

### Example 1 (reference severity: HIGH)

**Input (truncated):**
```
Jun 29 03:22:22 combo ftpd[13253]: connection from <IP_ADDR> () at Wed Jun 29 03:22:22 2005
Jun 29 03:22:22 combo ftpd[13247]: connection from <IP_ADDR> () at Wed Jun 29 03:22:22 2005
Jun 29 03:22:22 combo ftpd[13248]: connection from <IP_ADDR> () at Wed Jun 29 03:22:22 2005
Jun 29 03:22:23 combo ftpd[13265]: connection from <IP_ADDR> () at Wed Jun 29 03:22:23 2005
Jun 29 04:03:10 combo su(pam_unix)[13665]: session opened for user cyrus by (uid=0)
Jun 29 04:03:11 combo su(pam_unix)[13665]: session closed for user cyrus
Jun 29 04:03:12 combo logrotate: ALERT exited abnormally with [1]
Jun 29 04
```
**Reference output:**
```
SEVERITY: HIGH
INCIDENT_TYPE: Repeated Authentication Failure
ROOT_CAUSE: Multiple PAM authentication failures detected on the Linux system. The repeated failures from a known account may indicate a compromised or brute-forced password.
SUMMARY: The system's PAM authentication module is logging repeated login failures. This pattern is consistent with brute-force access attempts against the SSH or login service. Account lockout policies should be verified.
RECOMMENDED_ACTIONS:
1. Review /var/log/auth.log for source IPs and targeted accounts
2. Block offending IPs with iptables or fail2ban
3. Verify that password complexity policies are enforced
4. Consider enabling account lockout after repeated failures
```
**Model prediction:**
```
1. SEVERITY: LOW
2. INCIDENT_TYPE: Authentication Attempt
3. ROOT_CAUSE: Multiple failed SSH login attempts from a single IP address, potentially indicative of brute force attack.
4. SUMMARY: Multiple failed SSH login attempts detected from a single IP address on June 29th, suggesting potential security breach.
5. RECOMMENDED_ACTIONS:
1. Increase SSH login attempts threshold and implement rate limiting.
2. Enable two-factor authentication for all users.
3. Review access logs to identify any unauthorized access patterns.
```

### Example 4 (reference severity: LOW)

**Input (truncated):**
```
<TIMESTAMP>,728 - WARN  [QuorumPeer[myid=1]/0:0:0:0:0:0:0:0:<PORT>:QuorumCnxManager@368] - Cannot open channel to 3 at election address /<IP_ADDR>:<PORT>
```
**Reference output:**
```
SEVERITY: LOW
INCIDENT_TYPE: System Warning
ROOT_CAUSE: Warning-level events in the logs indicate conditions that are not yet failures but may lead to issues if left unaddressed.
SUMMARY: Warning events detected in the log window. These are not failures but indicate conditions that should be monitored to prevent escalation.
RECOMMENDED_ACTIONS:
1. Review warning details for potential action items
2. Monitor for escalation to error-level events
3. Address the root condition causing the warning
```
**Model prediction:**
```
1. SEVERITY: WARNING
2. INCIDENT_TYPE: QUORUM_CHANNEL_FAILURE
3. ROOT_CAUSE: FAILED TO ESTABLISH COMMUNICATION WITH A QUORUM MEMBER
4. SUMMARY: The system failed to establish a communication channel with one of its quorum members, resulting in a warning.
5. RECOMMENDED_ACTIONS:
1. Check network connectivity between the affected node and the quorum member.
2. Verify firewall rules allowing communication on the specified port.
3. Ensure that the quorum member is running and accessible.
```

### Example 13 (reference severity: MEDIUM)

**Input (truncated):**
```
<TIMESTAMP>,935 WARN [LeaseRenewer:msrabi@msra-sa-41:<PORT>] org.apache.hadoop.hdfs.LeaseRenewer: Failed to renew lease for [DFSClient_NONMAPREDUCE_1537864556_1] for 74 seconds.  Will retry shortly ...
<TIMESTAMP>,997 INFO [RMCommunicator Allocator] org.apache.hadoop.ipc.Client: Retrying connect to server: msra-sa-41:<PORT>. Already tried 0 time(s); retry policy is RetryUpToMaximumCountWithFixedSleep(maxRetries=10, sleepTime=1000 MILLISECONDS)
<TIMESTAMP>,997 ERROR [RMCommunicator Allocator] org.apache.hadoop.mapreduce.v2.app.rm.RMContainerAllocator: ERROR IN CONTACTING RM.
<TIMESTAMP>,466 INF
```
**Reference output:**
```
SEVERITY: MEDIUM
INCIDENT_TYPE: Hadoop Cluster Health Warning
ROOT_CAUSE: Hadoop cluster management has issued warnings about lease renewals, heartbeat timeouts, or node responsiveness, suggesting intermittent connectivity or overloaded nodes.
SUMMARY: Warning-level events from Hadoop cluster management indicate potential health issues such as missed heartbeats or lease timeouts. These may resolve on their own but should be monitored for escalation.
RECOMMENDED_ACTIONS:
1. Check DataNode and NodeManager heartbeat status
2. Review network latency between cluster nodes
3. Verify cluster nodes have sufficient CPU and memory resources
4. Monitor for escalation to node failures
```
**Model prediction:**
```
1. **SEVERITY**: MEDIUM
2. **INCIDENT_TYPE**: HDFS Lease Renewal Failure
3. **ROOT_CAUSE**: Network address changes causing intermittent connectivity issues between the client and ResourceManager (RM).
4. **SUMMARY**: The system experiences repeated failures in renewing HDFS leases due to network address changes, leading to intermittent communication issues with the ResourceManager.
5. **RECOMMENDED_ACTIONS**:
   1. Monitor network connectivity and address changes more closely.
   2. Implement a more robust retry mechanism for lease renewal processes.
   3. Investigate potential network configuration or firewall settings that might be causing these address changes.
```

### Example 96 (reference severity: CRITICAL)

**Input (truncated):**
```
<TIMESTAMP> ERROR [ingress] TLS handshake error from <IP_ADDR>:<PORT>: certificate has expired
<TIMESTAMP> ERROR [ingress] TLS handshake error from <IP_ADDR>:<PORT>: certificate has expired
<TIMESTAMP> WARN [cert-mon] Certificate for portal.internal.net expired 14 hours ago
```
**Reference output:**
```
SEVERITY: CRITICAL
INCIDENT_TYPE: TLS Certificate Expiry
ROOT_CAUSE: The TLS certificate for portal.internal.net expired 14 hours ago and was not renewed, causing all client TLS handshakes to fail at the ingress.
SUMMARY: All HTTPS connections to portal.internal.net are failing because its certificate expired 14 hours ago. Every client is affected. Certificate renewal automation either failed or was never configured for this domain.
RECOMMENDED_ACTIONS:
1. Renew and deploy a valid certificate for portal.internal.net immediately
2. Reload the ingress/proxy to pick up the new certificate
3. Investigate why automated renewal did not run
4. Add expiry monitoring alerts at 30/14/7 days before expiration
```
**Model prediction:**
```
1. SEVERITY: CRITICAL
2. INCIDENT_TYPE: TLS Handshake Error
3. ROOT_CAUSE: SSL/TLS certificate has expired
4. SUMMARY: Multiple TLS handshake errors occurred due to an expired SSL/TLS certificate on the ingress service, which is also detected by the cert-mon component.
5. RECOMMENDED_ACTIONS:
1. Replace the expired SSL/TLS certificate immediately.
2. Verify that the new certificate is correctly installed and configured.
3. Schedule a maintenance window to update the certificate if necessary.
```

## Artifacts

- `reports/baseline_predictions.jsonl` — every raw prediction
- `reports/baseline_metrics.json` — machine-readable metrics

---
*Generated by `python -m src.training.evaluate` (Week 4 deliverable D3).*
