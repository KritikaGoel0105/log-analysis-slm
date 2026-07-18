# Week 6 — Fine-Tuned Model Evaluation Report (D4)

- **Model:** Qwen/Qwen2.5-3B-Instruct (QLoRA fine-tuned, Week 5 adapter)
- **Test set:** 198 held-out examples (document Section 6: 200+ log scenarios)
- **Generated:** 2026-07-18T08:48:45.906802+00:00
- **Environment:** Python 3.11.9, Windows-10-10.0.26200-SP0
- **Decoding:** greedy (deterministic, reproducible)

All inference and scoring performed **fully offline** (HF_HUB_OFFLINE=1, local model weights, no external APIs).

## Metrics vs. Section 6.1 Targets

| Metric | Fine-Tuned | Target | Status |
|---|---|---|---|
| Severity Classification Accuracy | 73.7% | > 85% | FAIL |
| Incident Type F1 (macro) | 0.215 | > 0.80 | FAIL |
| ROUGE-L (summaries) | 0.594 | > 0.55 | PASS |
| False Positive Rate | 8.7% | < 10% | PASS |
| Root Cause Accuracy | *human evaluation — see example outputs below and finetuned_predictions.jsonl* | > 75% | MANUAL |
| API Response Time (p95) | *measured in Week 9 (/analyze endpoint); raw model p95 = 191.9s/example* | < 5 s | DEFERRED |
| RAG Retrieval Precision@3 | *measured in Week 7* | > 70% | DEFERRED |

## Output Format Compliance

- Parse failures (missing/invalid fields): 0 / 198 (0.0%)
- Unparseable fields are scored as incorrect, never dropped.

## Severity Classification Report

```
              precision    recall  f1-score   support

    CRITICAL       0.80      0.80      0.80         5
        HIGH       0.70      0.74      0.72        47
        INFO       0.96      0.80      0.87       100
         LOW       0.48      0.77      0.59        26
      MEDIUM       0.39      0.35      0.37        20

    accuracy                           0.74       198
   macro avg       0.67      0.69      0.67       198
weighted avg       0.77      0.74      0.75       198
```

## Severity Confusion Matrix (reference rows × prediction columns)

| ref \ pred | CRITICAL | HIGH | INFO | LOW | MEDIUM |
|---|---|---|---|---|---|
| **CRITICAL** | 4 | 1 | 0 | 0 | 0 |
| **HIGH** | 0 | 35 | 2 | 1 | 9 |
| **INFO** | 0 | 4 | 80 | 15 | 1 |
| **LOW** | 1 | 4 | 0 | 20 | 1 |
| **MEDIUM** | 0 | 6 | 1 | 6 | 7 |

## Latency (per-example model inference)

| mean | p50 | p95 | max |
|---|---|---|---|
| 74.5s | 63.8s | 191.9s | 240.2s |

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
SEVERITY: INFO
INCIDENT_TYPE: Spark Task Execution
ROOT_CAUSE: Spark executors are running tasks as part of normal distributed computation. Tasks are being assigned by the driver and executed on cluster workers.
SUMMARY: Spark is performing routine task execution on its cluster. Executors are processing data partitions across the cluster nodes. This activity is typical for running computations.
RECOMMENDED_ACTIONS:
1. No immediate action required
2. Monitor task completion rate and stage progress
3. Check for straggler tasks that may be slowing the job
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
SEVERITY: HIGH
INCIDENT_TYPE: Repeated Authentication Failure
ROOT_CAUSE: Multiple PAM authentication failures detected on the Linux system, indicating repeated login attempts using invalid credentials.
SUMMARY: The system's PAM authentication module is logging repeated login failures. The repeated nature of these failures suggests a brute-force or scripted attack against the login service.
RECOMMENDED_ACTIONS:
1. Review /var/log/auth.log for details about the failed login attempts (username, IP source).
2. Block the offending IP addresses using firewall rules (iptables/ufw).
3. Enable fail2ban or similar intrusion prevention for login attempts.
4. Verify that account lockout policies are configured and effective.
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
SEVERITY: LOW
INCIDENT_TYPE: System Warning
ROOT_CAUSE: Warning-level events in the logs indicate conditions that are not yet failures but may lead to issues if left unaddressed.
SUMMARY: Warning events detected in the log window. These are not failures but indicate conditions that should be monitored to prevent escalation.
RECOMMENDED_ACTIONS:
1. Review warning details for potential action items
2. Monitor for escalation to error-level events
3. Address the root condition causing the warning
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
SEVERITY: MEDIUM
INCIDENT_TYPE: Hadoop Cluster Health Warning
ROOT_CAUSE: Hadoop cluster management has issued warnings about lease renewals, heartbeat timeouts, or node responsiveness, suggesting intermittent connectivity or overloaded nodes.
SUMMARY: Warning-level events in the Hadoop cluster log indicate potential health issues such as lease timeouts or heartbeat failures. These may resolve on their own but should be monitored for recurrence.
RECOMMENDED_ACTIONS:
1. Check DataNode and NodeManager heartbeat status
2. Review network latency between cluster nodes
3. Verify cluster nodes have sufficient CPU and memory resources
4. Monitor for escalation to node failures
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
SEVERITY: HIGH
INCIDENT_TYPE: Certificate Expired TLS Handshake Failure
ROOT_CAUSE: The TLS certificate for the 'portal.internal.net' host has expired 14 hours ago, causing all connections to fail during the handshake process.
SUMMARY: Multiple TLS handshake failures are occurring on the ingress route for 'portal.internal.net' due to an expired certificate. Connections to this service are currently broken and may require certificate renewal.
RECOMMENDED_ACTIONS:
1. Verify the certificate authority's revocation status for the expired certificate
2. Renew the certificate before the next expiration date
3. Test TLS connectivity to the affected service after certificate renewal
4. Review certificate rotation policy for compliance
```

## Artifacts

- `reports/finetuned_predictions.jsonl` — every raw prediction
- `reports/finetuned_metrics.json` — machine-readable metrics

---
*Generated by `python -m src.training.evaluate --adapter models\checkpoints\final-adapter` (Week 6 deliverable D4).*
