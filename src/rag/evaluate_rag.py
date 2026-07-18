"""
evaluate_rag.py

Week 8 — End-to-end RAG pipeline evaluation + D6 evaluation report.

Document basis:
  * Week 8 (Section 4): "End-to-end pipeline integration + full
    evaluation on test set" — deliverable "Evaluation report + demo
    notebook".
  * D6 (Section 8): "Evaluation Report — Full metrics table; confusion
    matrices; ROUGE scores; comparison table: baseline vs fine-tuned
    vs RAG — End of Week 8".
  * Section 5.6 pipeline order: normalized logs -> retriever.retrieve
    -> model.generate(context=...) -> parse_model_output. The test
    set inputs are already normalized log windows (Week 2-3
    preprocessing), so this script performs retrieve -> generate ->
    parse per test example.
  * Section 6.1: metric definitions and targets.

Design (all ADDITIVE — nothing in Weeks 1-7 is modified):
  * The frozen `src/training/evaluate.py` is IMPORTED, never edited.
    Retrieved context is injected into a COPY of each test example's
    `input` (via the Week 7 `inject_context`), and the augmented
    examples are passed through the frozen `run_inference` /
    `score_predictions` unchanged. Prompt template, greedy decoding,
    latency measurement, parser and metrics are therefore byte-
    identical to the Week 4 baseline and Week 6 fine-tuned runs —
    the retrieved context is the ONLY variable.
  * Retrieval uses the Week 7 index (train.jsonl only — val/test were
    never indexed, so no evaluation leakage).
  * Outputs go to SEPARATE new files; baseline_* and finetuned_*
    artifacts are never touched.

Offline guarantees: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set before
any HF import; model, adapter, embedder and index all load from local
paths only.

Outputs (all under reports/):
  * rag_predictions.jsonl        — raw + augmented input per example
  * rag_metrics.json             — machine-readable RAG-pipeline metrics
  * Week8_Evaluation_Report.md   — the D6 report (3-way comparison)

Usage (from repository root, inside the offline venv):
    python -m src.rag.evaluate_rag --device cuda      # full test set
    python -m src.rag.evaluate_rag --device cuda --limit 5   # smoke
    python -m src.rag.evaluate_rag --skip-inference   # re-score cached
        # rag_predictions.jsonl without loading the model
    python -m src.rag.evaluate_rag --report-only      # rebuild the D6
        # report from the three existing metrics JSONs (no inference)
"""

import argparse
import json
import os
from pathlib import Path

# Offline enforcement BEFORE any HF-dependent import (Section 2:
# zero internet dependencies at runtime).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Frozen Week 4/6 evaluation machinery — imported, NOT modified.
from ..training.evaluate import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_TEST_FILE,
    GENERATION_KWARGS,
    REPORTS_DIR,
    TARGETS,
    run_inference,
    score_predictions,
)
from ..training.utils import read_jsonl, write_jsonl

# Week 7 retriever + context injection — imported, NOT modified.
from .rag_retriever import (
    DEFAULT_INDEX_FILE,
    DEFAULT_SIDECAR_FILE,
    OfflineRAGRetriever,
    inject_context,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Week 8 (D6) outputs — SEPARATE files; frozen artifacts untouched.
RAG_PREDICTIONS_FILE = REPORTS_DIR / "rag_predictions.jsonl"
RAG_METRICS_FILE = REPORTS_DIR / "rag_metrics.json"
WEEK8_REPORT_FILE = REPORTS_DIR / "Week8_Evaluation_Report.md"

# Frozen inputs to the D6 comparison (read-only).
BASELINE_METRICS_FILE = REPORTS_DIR / "baseline_metrics.json"
FINETUNED_METRICS_FILE = REPORTS_DIR / "finetuned_metrics.json"
RETRIEVAL_METRICS_FILE = REPORTS_DIR / "rag_retrieval_metrics.json"

TOP_K = 3  # Section 6.1: Precision@3 / Section 5.6: top_k=3


# ------------------------------------------------------------------
# Retrieval-augmentation of the test set (Section 5.6 order:
# retrieve -> generate; inputs are already normalized)
# ------------------------------------------------------------------

def augment_examples(examples: list[dict],
                     retriever: OfflineRAGRetriever,
                     top_k: int = TOP_K):
    """
    For each test example, retrieve top-k similar training incidents
    and inject them into a COPY of the example's input via the Week 7
    context-injection pipeline. Returns (augmented_examples,
    retrieval_records); the originals are never mutated.
    """
    augmented, retrievals = [], []
    total = len(examples)
    for i, ex in enumerate(examples):
        retrieved = retriever.retrieve_with_metadata(ex["input"], top_k)
        aug = dict(ex)
        aug["input"] = inject_context(ex["input"], retrieved)
        augmented.append(aug)
        retrievals.append([
            {"train_index": m.get("train_index"),
             "severity": m.get("severity"),
             "incident_type": m.get("incident_type")}
            for _, m in retrieved
        ])
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"  retrieved context for {i + 1}/{total} examples")
    return augmented, retrievals


# ------------------------------------------------------------------
# D6 Markdown report (baseline vs fine-tuned vs RAG)
# ------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _pass_fail(value: float, key: str) -> str:
    spec = TARGETS[key]
    ok = (value > spec["target"] if spec["direction"] == ">"
          else value < spec["target"])
    return "PASS" if ok else "FAIL"


def _best_marker(values: list[float], higher_is_better: bool = True):
    """Index of the best value across the three systems."""
    best = max(values) if higher_is_better else min(values)
    return [v == best for v in values]


def _cm_section(lines: list, title: str, cm: dict) -> None:
    """Render one severity confusion matrix as a Markdown table."""
    add = lines.append
    add(f"### {title}")
    add("")
    all_labels = sorted(
        set(cm.keys()) | {p for row in cm.values() for p in row.keys()}
    )
    add("| ref \\ pred | " + " | ".join(all_labels) + " |")
    add("|---|" + "---|" * len(all_labels))
    for ref_label in sorted(cm.keys()):
        row = cm[ref_label]
        cells = [str(row.get(p, 0)) for p in all_labels]
        add(f"| **{ref_label}** | " + " | ".join(cells) + " |")
    add("")


def write_week8_report(baseline: dict, finetuned: dict, rag: dict,
                       retrieval_summary: dict | None,
                       report_file: Path = WEEK8_REPORT_FILE) -> None:
    """
    Render the D6 evaluation report (Section 8): full metrics table,
    confusion matrices, ROUGE scores, and the 3-way comparison table.
    All numbers come from the metrics JSONs — the frozen baseline and
    fine-tuned artifacts are read, never regenerated.
    """
    lines = []
    add = lines.append

    add("# Week 8 — Full Evaluation Report (D6)")
    add("")
    add("**Comparison: baseline vs fine-tuned vs RAG pipeline** "
        "(document Section 8, D6).")
    add("")
    add(f"- **Model:** {rag['model']} (base for all three systems)")
    add(f"- **Test set:** {rag['num_examples']} held-out examples "
        "(document Section 6: 200+ log scenarios; identical set for "
        "all three systems)")
    add(f"- **Generated:** {rag['generated_at']}")
    add("- **Decoding:** greedy (deterministic) — identical "
        "generation settings for all three systems; for the RAG "
        "column the retrieved context is the only variable versus "
        "the fine-tuned column.")
    add("- **Retrieval:** top-3 similar incidents from the Week 7 "
        "FAISS index (train split only — val/test never indexed, so "
        "no evaluation leakage).")
    add("")
    add("All inference and scoring performed **fully offline** "
        "(HF_HUB_OFFLINE=1, local weights, local FAISS index, no "
        "external APIs).")
    add("")

    # ---------------- 3-way comparison table (D6) ----------------
    add("## Full Metrics Table — Baseline vs Fine-Tuned vs RAG")
    add("")
    add("| Metric | Baseline | Fine-Tuned | Fine-Tuned + RAG | "
        "Target (6.1) | RAG Status |")
    add("|---|---|---|---|---|---|")

    def row(label, key_path, fmt, target_key, higher=True):
        vals = []
        for m in (baseline, finetuned, rag):
            v = m
            for k in key_path:
                v = v[k]
            vals.append(v)
        best = _best_marker(vals, higher)
        cells = [f"**{fmt(v)}**" if b else fmt(v)
                 for v, b in zip(vals, best)]
        spec = TARGETS[target_key]
        tgt = (f"> {_pct(spec['target'])}" if spec["direction"] == ">"
               else f"< {_pct(spec['target'])}")
        if target_key in ("incident_type_f1_macro", "rouge_l_mean"):
            tgt = (f"> {spec['target']:.2f}")
        status = _pass_fail(vals[2], target_key)
        add(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{tgt} | {status} |")

    row("Severity Accuracy", ["severity_accuracy"], _pct,
        "severity_accuracy")
    row("Incident Type F1 (macro)",
        ["incident_type_f1_macro", "value"],
        lambda v: f"{v:.3f}", "incident_type_f1_macro")
    row("ROUGE-L (summaries)", ["rouge_l", "mean"],
        lambda v: f"{v:.3f}", "rouge_l_mean")
    row("False Positive Rate", ["false_positive_rate", "rate"], _pct,
        "false_positive_rate", higher=False)

    # Non-comparable Section 6.1 rows
    pf = [f"{m['parse_failures']}/{m['num_examples']}"
          for m in (baseline, finetuned, rag)]
    add(f"| Parse Failures | {pf[0]} | {pf[1]} | {pf[2]} | — | — |")
    add("| Root Cause Accuracy | *human evaluation* | *human "
        "evaluation* | *human evaluation — see "
        "rag_predictions.jsonl* | > 75% | MANUAL |")
    p95 = [m.get("latency", {}).get("p95") for m in
           (baseline, finetuned, rag)]
    p95c = [f"{v:.1f}s" if isinstance(v, (int, float)) else "—"
            for v in p95]
    add(f"| Model p95 latency (per example) | {p95c[0]} | {p95c[1]} | "
        f"{p95c[2]} | *API p95 < 5s measured in Week 9* | DEFERRED |")
    if retrieval_summary:
        p3 = retrieval_summary["retrieval_precision_at_3"]
        st = "PASS" if retrieval_summary["target_met"] else "MISS"
        add(f"| RAG Retrieval Precision@3 (Week 7, D5) | — | — | "
            f"{_pct(p3)} | > 70% | {st} |")
    add("")
    add("Bold marks the best of the three systems per metric.")
    add("")

    # ---------------- ROUGE scores (D6) ----------------
    add("## ROUGE Scores")
    add("")
    add("| System | Mean ROUGE-L (summaries) | Implementation |")
    add("|---|---|---|")
    for name, m in (("Baseline (zero-shot)", baseline),
                    ("Fine-tuned", finetuned),
                    ("Fine-tuned + RAG", rag)):
        add(f"| {name} | {m['rouge_l']['mean']:.3f} | "
            f"`{m['rouge_l']['implementation']}` |")
    add("")

    # ---------------- Confusion matrices (D6) ----------------
    add("## Severity Confusion Matrices "
        "(reference rows × prediction columns)")
    add("")
    _cm_section(lines, "Baseline (zero-shot)",
                baseline["severity_confusion_matrix"])
    _cm_section(lines, "Fine-tuned",
                finetuned["severity_confusion_matrix"])
    _cm_section(lines, "Fine-tuned + RAG",
                rag["severity_confusion_matrix"])

    # ---------------- Per-class detail for the RAG system --------
    add("## RAG Pipeline — Severity Classification Report")
    add("")
    add("```")
    add(rag["severity_classification_report"].rstrip())
    add("```")
    add("")

    # ---------------- Latency ----------------
    add("## Latency (per-example model inference)")
    add("")
    add("| System | mean | p50 | p95 | max |")
    add("|---|---|---|---|---|")
    for name, m in (("Baseline", baseline), ("Fine-tuned", finetuned),
                    ("Fine-tuned + RAG", rag)):
        lat = m.get("latency") or {}
        if lat:
            add(f"| {name} | {lat['mean']:.1f}s | {lat['p50']:.1f}s | "
                f"{lat['p95']:.1f}s | {lat['max']:.1f}s |")
        else:
            add(f"| {name} | — | — | — | — |")
    add("")
    add("*API response time (p95 < 5s, Section 6.1) is a property of "
        "the Week 9 `/analyze` endpoint and is measured there.*")
    add("")

    # ---------------- Methodology notes ----------------
    add("## Methodology Notes")
    add("")
    add("- **Identical harness:** all three systems were evaluated "
        "with the same frozen prompt template, greedy decoding, "
        "output parser and metric implementations "
        "(`src/training/evaluate.py`, imported unmodified). The RAG "
        "column adds only the Week 7 retrieved-context block to the "
        "model input.")
    add("- **Incident Type F1 (macro)** is exact string match over "
        "79 fine-grained incident-type categories, macro-averaged; "
        "many categories have tiny support, so near-miss labels "
        "(e.g. a correct family with different wording) score 0. "
        "It understates practical performance relative to severity "
        "accuracy.")
    add("- **No leakage:** the FAISS index contains only "
        "`train.jsonl` incidents (1,550 vectors); the 198 test "
        "queries were never indexed.")
    add("- **Root Cause Accuracy** (Section 6.1: human evaluation) "
        "requires manual review of `reports/rag_predictions.jsonl`; "
        "it is not automatable and is reported as MANUAL above.")
    add("")

    # ---------------- Artifacts ----------------
    add("## Artifacts")
    add("")
    add("- `reports/rag_predictions.jsonl` — per-example raw "
        "prediction, augmented input, retrieved incidents")
    add("- `reports/rag_metrics.json` — machine-readable RAG metrics")
    add("- `reports/baseline_metrics.json`, "
        "`reports/finetuned_metrics.json` — frozen Week 4/6 inputs "
        "(read-only)")
    add("- `reports/rag_retrieval_metrics.json` — Week 7 D5 "
        "precision@3 measurement (read-only)")
    add("- `notebooks/week8_demo.ipynb` — end-to-end pipeline demo "
        "(Week 8 deliverable)")
    add("")
    add("---")
    add("*Generated by `python -m src.rag.evaluate_rag` (Week 8 "
        "deliverable D6). Baseline and fine-tuned artifacts were "
        "read, never regenerated.*")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _load_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found: {path} — required for the D6 "
            "comparison table.")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Week 8 end-to-end RAG pipeline evaluation "
                    "(offline, additive).")
    parser.add_argument("--test-file", type=Path,
                        default=DEFAULT_TEST_FILE)
    parser.add_argument("--model-dir", type=Path,
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter", type=Path,
                        default=DEFAULT_ADAPTER_DIR,
                        help="Week 5 LoRA adapter (the Week 8 "
                             "pipeline is fine-tuned model + RAG)")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N examples "
                             "(smoke test)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-new-tokens", type=int,
                        default=GENERATION_KWARGS["max_new_tokens"])
    parser.add_argument("--skip-inference", action="store_true",
                        help="re-score existing "
                             "reports/rag_predictions.jsonl without "
                             "loading the model")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild Week8_Evaluation_Report.md from "
                             "the three existing metrics JSONs")
    args = parser.parse_args()

    print("=" * 60)
    print("Week 8 — End-to-end RAG pipeline evaluation (offline)")
    print("=" * 60)

    if args.report_only:
        rag_metrics = _load_json(RAG_METRICS_FILE, "RAG metrics")
    elif args.skip_inference:
        records = read_jsonl(RAG_PREDICTIONS_FILE)
        if not records:
            print(f"ERROR: {RAG_PREDICTIONS_FILE} is empty or "
                  "missing; run without --skip-inference first.")
            return 1
        print(f"Re-scoring {len(records)} cached predictions "
              f"from {RAG_PREDICTIONS_FILE}")
        rag_metrics = score_predictions(
            records,
            mode=("RAG pipeline (fine-tuned + top-3 retrieved "
                  "context)"),
        )
    else:
        examples = read_jsonl(args.test_file)
        if not examples:
            print(f"ERROR: no examples in {args.test_file}")
            return 1
        if args.limit:
            examples = examples[: args.limit]

        if not DEFAULT_INDEX_FILE.exists():
            print(f"ERROR: {DEFAULT_INDEX_FILE} not found. "
                  "Run: python -m src.rag.build_index")
            return 1

        print(f"Test examples : {len(examples)}")
        print(f"Adapter       : {args.adapter}")
        print(f"Device        : {args.device}")
        print(f"Retrieval     : top-{args.top_k} from "
              f"{DEFAULT_INDEX_FILE.name}")

        print("Loading Week 7 retriever (offline)...")
        retriever = OfflineRAGRetriever()
        retriever.load(DEFAULT_INDEX_FILE, DEFAULT_SIDECAR_FILE)
        print(f"Index loaded: {retriever.index.ntotal} vectors")

        augmented, retrievals = augment_examples(
            examples, retriever, args.top_k)

        # Frozen inference harness, unchanged — the augmented input
        # is the only difference versus the Week 6 fine-tuned run.
        records = run_inference(
            augmented, args.model_dir, args.device,
            args.max_new_tokens, adapter_dir=args.adapter,
        )
        # Attach provenance: original (un-augmented) input + what was
        # retrieved, for the D6 human root-cause review.
        for rec, ex, ret in zip(records, examples, retrievals):
            rec["original_input"] = ex["input"]
            rec["retrieved"] = ret

        write_jsonl(RAG_PREDICTIONS_FILE, records)
        print(f"Predictions saved to {RAG_PREDICTIONS_FILE}")

        rag_metrics = score_predictions(
            records,
            mode=("RAG pipeline (fine-tuned + top-3 retrieved "
                  "context)"),
        )

    if not args.report_only:
        rag_metrics["retrieval"] = {
            "top_k": args.top_k,
            "index_file": str(DEFAULT_INDEX_FILE.relative_to(REPO_ROOT)),
            "corpus": "train.jsonl only (no val/test leakage)",
        }
        RAG_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RAG_METRICS_FILE.write_text(
            json.dumps(rag_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"Metrics saved to {RAG_METRICS_FILE}")

    # ---- D6 report: 3-way comparison from the metrics JSONs ----
    baseline = _load_json(BASELINE_METRICS_FILE,
                          "Frozen Week 4 baseline metrics")
    finetuned = _load_json(FINETUNED_METRICS_FILE,
                           "Frozen Week 6 fine-tuned metrics")
    retrieval_summary = None
    if RETRIEVAL_METRICS_FILE.exists():
        retrieval_summary = json.loads(
            RETRIEVAL_METRICS_FILE.read_text(
                encoding="utf-8"))["summary"]
    else:
        print(f"WARNING: {RETRIEVAL_METRICS_FILE} missing — D5 "
              "precision@3 row omitted from the report.")

    write_week8_report(baseline, finetuned, rag_metrics,
                       retrieval_summary)
    print(f"Report saved to {WEEK8_REPORT_FILE}")

    print()
    print(f"Severity accuracy : {rag_metrics['severity_accuracy']:.1%}")
    print(f"Incident F1 macro : "
          f"{rag_metrics['incident_type_f1_macro']['value']:.3f}")
    print(f"Mean ROUGE-L      : {rag_metrics['rouge_l']['mean']:.3f}")
    print(f"False positive %  : "
          f"{rag_metrics['false_positive_rate']['rate']:.1%}")
    print(f"Parse failures    : {rag_metrics['parse_failures']}"
          f"/{rag_metrics['num_examples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
