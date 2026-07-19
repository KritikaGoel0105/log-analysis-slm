# Week 8 — Full Evaluation Report (D6)

**Comparison: baseline vs fine-tuned vs RAG pipeline** (document Section 8, D6).

- **Model:** Qwen/Qwen2.5-3B-Instruct (base for all three systems)
- **Test set:** 198 held-out examples (document Section 6: 200+ log scenarios; identical set for all three systems)
- **Generated:** 2026-07-19T17:44:27.307181+00:00
- **Decoding:** greedy (deterministic) — identical generation settings for all three systems; for the RAG column the retrieved context is the only variable versus the fine-tuned column.
- **Retrieval:** top-3 similar incidents from the Week 7 FAISS index (train split only — val/test never indexed, so no evaluation leakage).

All inference and scoring performed **fully offline** (HF_HUB_OFFLINE=1, local weights, local FAISS index, no external APIs).

## Full Metrics Table — Baseline vs Fine-Tuned vs RAG

| Metric | Baseline | Fine-Tuned | Fine-Tuned + RAG | Target (6.1) | RAG Status |
|---|---|---|---|---|---|
| Severity Accuracy | 43.9% | 73.7% | **80.8%** | > 85.0% | FAIL |
| Incident Type F1 (macro) | 0.001 | 0.215 | **0.240** | > 0.80 | FAIL |
| ROUGE-L (summaries) | 0.121 | **0.594** | 0.464 | > 0.55 | FAIL |
| False Positive Rate | 33.3% | **8.7%** | 20.6% | < 10.0% | FAIL |
| Parse Failures | 16/198 | 0/198 | 0/198 | — | — |
| Root Cause Accuracy | *human evaluation* | *human evaluation* | *human evaluation — see rag_predictions.jsonl* | > 75% | MANUAL |
| Model p95 latency (per example) | 216.2s | 191.9s | 276.3s | *API p95 < 5s measured in Week 9* | DEFERRED |
| RAG Retrieval Precision@3 (Week 7, D5) | — | — | 82.0% | > 70% | PASS |

Bold marks the best of the three systems per metric.

## ROUGE Scores

| System | Mean ROUGE-L (summaries) | Implementation |
|---|---|---|
| Baseline (zero-shot) | 0.121 | `rouge_score` |
| Fine-tuned | 0.594 | `rouge_score` |
| Fine-tuned + RAG | 0.464 | `rouge_score` |

## Severity Confusion Matrices (reference rows × prediction columns)

### Baseline (zero-shot)

| ref \ pred | CRITICAL | HIGH | INFO | LOW | MEDIUM | UNPARSEABLE |
|---|---|---|---|---|---|---|
| **CRITICAL** | 4 | 1 | 0 | 0 | 0 | 0 |
| **HIGH** | 5 | 10 | 8 | 4 | 15 | 5 |
| **INFO** | 0 | 0 | 63 | 8 | 29 | 0 |
| **LOW** | 2 | 0 | 2 | 2 | 11 | 9 |
| **MEDIUM** | 0 | 1 | 4 | 5 | 8 | 2 |

### Fine-tuned

| ref \ pred | CRITICAL | HIGH | INFO | LOW | MEDIUM |
|---|---|---|---|---|---|
| **CRITICAL** | 4 | 1 | 0 | 0 | 0 |
| **HIGH** | 0 | 35 | 2 | 1 | 9 |
| **INFO** | 0 | 4 | 80 | 15 | 1 |
| **LOW** | 1 | 4 | 0 | 20 | 1 |
| **MEDIUM** | 0 | 6 | 1 | 6 | 7 |

### Fine-tuned + RAG

| ref \ pred | CRITICAL | HIGH | INFO | LOW | MEDIUM |
|---|---|---|---|---|---|
| **CRITICAL** | 5 | 0 | 0 | 0 | 0 |
| **HIGH** | 0 | 46 | 0 | 1 | 0 |
| **INFO** | 0 | 10 | 74 | 7 | 9 |
| **LOW** | 0 | 4 | 0 | 19 | 3 |
| **MEDIUM** | 0 | 3 | 0 | 1 | 16 |

## RAG Pipeline — Severity Classification Report

```
              precision    recall  f1-score   support

    CRITICAL       1.00      1.00      1.00         5
        HIGH       0.73      0.98      0.84        47
        INFO       1.00      0.74      0.85       100
         LOW       0.68      0.73      0.70        26
      MEDIUM       0.57      0.80      0.67        20

    accuracy                           0.81       198
   macro avg       0.80      0.85      0.81       198
weighted avg       0.85      0.81      0.81       198
```

## Latency (per-example model inference)

| System | mean | p50 | p95 | max |
|---|---|---|---|---|
| Baseline | 92.6s | 81.0s | 216.2s | 326.8s |
| Fine-tuned | 74.5s | 63.8s | 191.9s | 240.2s |
| Fine-tuned + RAG | 120.0s | 110.1s | 276.3s | 471.9s |

*API response time (p95 < 5s, Section 6.1) is a property of the Week 9 `/analyze` endpoint and is measured there.*

## Methodology Notes

- **Identical harness:** all three systems were evaluated with the same frozen prompt template, greedy decoding, output parser and metric implementations (`src/training/evaluate.py`, imported unmodified). The RAG column adds only the Week 7 retrieved-context block to the model input.
- **Incident Type F1 (macro)** is exact string match over 79 fine-grained incident-type categories, macro-averaged; many categories have tiny support, so near-miss labels (e.g. a correct family with different wording) score 0. It understates practical performance relative to severity accuracy.
- **No leakage:** the FAISS index contains only `train.jsonl` incidents (1,550 vectors); the 198 test queries were never indexed.
- **Root Cause Accuracy** (Section 6.1: human evaluation) requires manual review of `reports/rag_predictions.jsonl`; it is not automatable and is reported as MANUAL above.

## Artifacts

- `reports/rag_predictions.jsonl` — per-example raw prediction, augmented input, retrieved incidents
- `reports/rag_metrics.json` — machine-readable RAG metrics
- `reports/baseline_metrics.json`, `reports/finetuned_metrics.json` — frozen Week 4/6 inputs (read-only)
- `reports/rag_retrieval_metrics.json` — Week 7 D5 precision@3 measurement (read-only)
- `notebooks/week8_demo.ipynb` — end-to-end pipeline demo (Week 8 deliverable)

---
*Generated by `python -m src.rag.evaluate_rag` (Week 8 deliverable D6). Baseline and fine-tuned artifacts were read, never regenerated.*
