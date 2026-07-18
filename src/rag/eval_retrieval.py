"""
eval_retrieval.py

Week 7 — D5 acceptance measurement (document Section 8):
    "D5: RAG System — rag_retriever.py; FAISS index file;
     retrieval precision@3 > 70% on test queries"
and Section 6.1:
    "RAG Retrieval Precision@3 — % of retrieved incidents that are
     relevant to the query — > 70%"

Test queries: the `input` field of every example in
`data/dataset/test.jsonl` (the held-out split — "test queries").

Relevance judgment (stated explicitly, since the document defines
precision@3 but not the relevance oracle): a retrieved incident is
RELEVANT to a query iff its INCIDENT_TYPE equals the query's
reference INCIDENT_TYPE. This is the strictest automatable criterion
available in the repository: incident type identifies "the same kind
of incident", and in this dataset it also determines severity
(verified: each of the 79 training incident types maps to exactly one
severity). A same-severity-only criterion would be far looser and is
reported as a secondary number, not the headline.

Precision@3 = mean over test queries of
    (# relevant among the 3 retrieved) / 3.

Outputs:
  * reports/rag_retrieval_metrics.json
  * reports/Week7_RAG_Report.md

Usage (offline, from repo root, after build_index):
    python -m src.rag.eval_retrieval
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .build_index import _field
from .rag_retriever import (
    DEFAULT_INDEX_FILE,
    DEFAULT_SIDECAR_FILE,
    OfflineRAGRetriever,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = REPO_ROOT / "data" / "dataset" / "test.jsonl"
REPORTS_DIR = REPO_ROOT / "reports"
METRICS_FILE = REPORTS_DIR / "rag_retrieval_metrics.json"
REPORT_FILE = REPORTS_DIR / "Week7_RAG_Report.md"

TOP_K = 3          # Section 6.1: Precision@3
TARGET = 0.70      # Section 6.1 / D5: > 70%


def evaluate(retriever: OfflineRAGRetriever, test_file: Path = TEST_FILE):
    """Score precision@3 over all test queries."""
    per_query = []
    with open(test_file, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            query = rec["input"]
            ref_type = _field(rec["output"], "INCIDENT_TYPE")
            ref_sev = _field(rec["output"], "SEVERITY")

            retrieved = retriever.retrieve_with_metadata(query, TOP_K)
            type_hits = sum(
                1 for _, m in retrieved
                if m.get("incident_type") == ref_type)
            sev_hits = sum(
                1 for _, m in retrieved
                if m.get("severity") == ref_sev)

            per_query.append({
                "index": i,
                "reference_incident_type": ref_type,
                "reference_severity": ref_sev,
                "retrieved": [
                    {"incident_type": m.get("incident_type"),
                     "severity": m.get("severity"),
                     "train_index": m.get("train_index")}
                    for _, m in retrieved],
                "precision_at_3": type_hits / TOP_K,
                "severity_precision_at_3": sev_hits / TOP_K,
            })
    return per_query


def summarize(per_query: list[dict]) -> dict:
    n = len(per_query)
    p3 = sum(q["precision_at_3"] for q in per_query) / n
    sev_p3 = sum(q["severity_precision_at_3"] for q in per_query) / n
    perfect = sum(1 for q in per_query if q["precision_at_3"] == 1.0)
    zero = sum(1 for q in per_query if q["precision_at_3"] == 0.0)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_test_queries": n,
        "top_k": TOP_K,
        "relevance_criterion": "retrieved.incident_type == "
                               "reference.incident_type (strict; see "
                               "module docstring)",
        "retrieval_precision_at_3": p3,
        "severity_precision_at_3_secondary": sev_p3,
        "queries_with_perfect_p3": perfect,
        "queries_with_zero_p3": zero,
        "target": TARGET,
        "target_met": p3 > TARGET,
        "index_file": str(DEFAULT_INDEX_FILE.relative_to(REPO_ROOT)),
        "embedding_model": "all-MiniLM-L6-v2 (local, offline)",
    }


def write_report(summary: dict, per_query: list[dict]) -> None:
    lines = []
    add = lines.append
    add("# Week 7 — Offline RAG Retrieval Report (D5)")
    add("")
    add(f"- **Generated:** {summary['generated_at']}")
    add("- **Retriever:** FAISS IndexFlatIP (cosine via L2-normalized "
        "embeddings), all-MiniLM-L6-v2 embedder loaded from "
        "`models/sentence-transformers/` — fully offline "
        "(document Section 5.5).")
    add("- **Index corpus:** 1,550 training incidents "
        "(`train.jsonl` only; val/test excluded to prevent "
        "evaluation leakage).")
    add(f"- **Test queries:** {summary['num_test_queries']} held-out "
        "test-split inputs.")
    add("")
    add("## Result (Section 6.1 / D5 acceptance)")
    add("")
    add("| Metric | Value | Target | Status |")
    add("|---|---|---|---|")
    p3 = summary["retrieval_precision_at_3"]
    status = "PASS" if summary["target_met"] else "MISS"
    add(f"| RAG Retrieval Precision@3 | {p3:.1%} | > 70% | "
        f"**{status}** |")
    add(f"| Severity-match P@3 (secondary, looser) | "
        f"{summary['severity_precision_at_3_secondary']:.1%} | — | — |")
    add("")
    add(f"- Queries with all 3 retrievals relevant: "
        f"{summary['queries_with_perfect_p3']} / "
        f"{summary['num_test_queries']}")
    add(f"- Queries with zero relevant retrievals: "
        f"{summary['queries_with_zero_p3']} / "
        f"{summary['num_test_queries']}")
    add("")
    add("## Relevance criterion")
    add("")
    add("A retrieved incident counts as relevant iff its "
        "`INCIDENT_TYPE` equals the query's reference "
        "`INCIDENT_TYPE`. This is the strictest automatable "
        "criterion in the repository; incident type also uniquely "
        "determines severity in this dataset, so it subsumes "
        "severity relevance.")
    add("")
    add("## Worst-retrieved incident types")
    add("")
    from collections import defaultdict
    by_type = defaultdict(list)
    for q in per_query:
        by_type[q["reference_incident_type"]].append(q["precision_at_3"])
    worst = sorted(((sum(v) / len(v), t, len(v))
                    for t, v in by_type.items()))[:8]
    add("| Reference incident type | Mean P@3 | Test queries |")
    add("|---|---|---|")
    for score, t, n in worst:
        add(f"| {t} | {score:.2f} | {n} |")
    add("")
    add("## Artifacts")
    add("")
    add("- `data/faiss.index` — FAISS index (D5)")
    add("- `data/faiss_incidents.json` — incident texts + metadata "
        "sidecar")
    add("- `reports/rag_retrieval_metrics.json` — this evaluation")
    add("")
    add("---")
    add("*Generated by `python -m src.rag.eval_retrieval` "
        "(Week 7 / D5). Weeks 1-6 artifacts untouched.*")

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("=" * 60)
    print("Week 7 — RAG retrieval precision@3 evaluation (D5)")
    print("=" * 60)

    if not DEFAULT_INDEX_FILE.exists():
        print(f"ERROR: {DEFAULT_INDEX_FILE} not found. "
              "Run: python -m src.rag.build_index")
        return 1
    if not TEST_FILE.exists():
        print(f"ERROR: {TEST_FILE} not found (Week 3 dataset).")
        return 1

    retriever = OfflineRAGRetriever()
    retriever.load(DEFAULT_INDEX_FILE, DEFAULT_SIDECAR_FILE)
    print(f"Loaded index: {retriever.index.ntotal} vectors")

    per_query = evaluate(retriever)
    summary = summarize(per_query)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(
        json.dumps({"summary": summary, "per_query": per_query},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    write_report(summary, per_query)

    p3 = summary["retrieval_precision_at_3"]
    print(f"Precision@3: {p3:.1%}  (target > 70%: "
          f"{'PASS' if summary['target_met'] else 'MISS'})")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Report:  {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
