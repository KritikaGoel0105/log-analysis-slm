"""
evaluate.py

Week 4 — Baseline (zero-shot) evaluation of the unmodified SLM.

Document basis:
  * Week 4 (Section 4): "Baseline model: run inference with
    unmodified SLM, measure baseline metrics" — deliverable
    "Baseline accuracy report (PDF/Markdown)".
  * Section 6.1: evaluation metrics and targets.
  * Section 6.2: evaluation script skeleton (rouge_scorer,
    classification_report, parse_model_output) — implemented here
    and in metrics.py / output_parser.py.
  * Section 6 intro: "All evaluation is performed offline on a
    held-out test set of 200+ log scenarios."
  * D3 (Section 8): "PDF/Markdown report: zero-shot SLM performance
    on all 6 metrics with example outputs" — end of Week 4.

Offline guarantees:
  * HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set BEFORE importing
    transformers; the model is loaded with local_files_only=True
    from ./models/qwen25-3b. No network access is attempted.

Outputs (all under reports/):
  * baseline_predictions.jsonl  — raw + parsed prediction per example
  * baseline_metrics.json       — machine-readable metrics
  * Week4_Baseline_Report.md    — the D3 Markdown report

Usage (from repository root, inside the offline venv):
    python -m src.training.evaluate                 # full test set
    python -m src.training.evaluate --limit 20      # smoke run
    python -m src.training.evaluate --skip-inference  # re-score
        # existing baseline_predictions.jsonl without re-running
        # the model (useful after parser/metric changes)
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Offline enforcement — MUST happen before transformers is imported
# (document Section 2/6: no internet-dependent runtime components).
# ------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .utils import read_jsonl, write_jsonl
from .output_parser import parse_model_output, parse_reference_output
from . import metrics as metrics_mod

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "qwen25-3b"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # doc Section 5.4 primary
DEFAULT_TEST_FILE = REPO_ROOT / "data" / "dataset" / "test.jsonl"

REPORTS_DIR = REPO_ROOT / "reports"
PREDICTIONS_FILE = REPORTS_DIR / "baseline_predictions.jsonl"
METRICS_FILE = REPORTS_DIR / "baseline_metrics.json"
REPORT_FILE = REPORTS_DIR / "Week4_Baseline_Report.md"

# Deterministic generation: greedy decoding so the baseline is
# reproducible run-to-run (project requirement: reproducible).
GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

# Section 6.1 targets, used for the pass/fail table in the report.
TARGETS = {
    "severity_accuracy": {"target": 0.85, "direction": ">"},
    "incident_type_f1_macro": {"target": 0.80, "direction": ">"},
    "rouge_l_mean": {"target": 0.55, "direction": ">"},
    "false_positive_rate": {"target": 0.10, "direction": "<"},
}


# ------------------------------------------------------------------
# Model loading (local weights only)
# ------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: Path, device: str):
    """
    Load the unmodified SLM strictly from local files.

    Imports of torch/transformers are deferred to here so that
    --skip-inference re-scoring works on machines without the ML
    stack installed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Download weights while online (document Section 5.1, "
            "Step 4) before running offline evaluation."
        )

    # Weights were cached with cache_dir=./models/qwen25-3b
    # (document Section 5.1 Step 4), so load via cache_dir +
    # local_files_only to resolve the snapshot without network.
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_NAME,
        cache_dir=str(model_dir),
        local_files_only=True,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        cache_dir=str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return model, tokenizer


def build_prompt(tokenizer, instruction: str, log_input: str) -> str:
    """
    Build the chat-formatted prompt for one example using the
    model's own chat template (instruction = system role, logs =
    user role — mirrors the dataset's instruction/input fields).
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": log_input},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def run_inference(examples: list[dict], model_dir: Path,
                  device: str, max_new_tokens: int) -> list[dict]:
    """
    Zero-shot inference over the test examples. Returns one record
    per example: {index, input, reference, prediction, latency_s}.
    """
    import torch

    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    generation_kwargs = dict(GENERATION_KWARGS)
    generation_kwargs["max_new_tokens"] = max_new_tokens
    # Remove sampling params entirely for greedy decoding
    generation_kwargs = {
        k: v for k, v in generation_kwargs.items() if v is not None
    }

    records = []
    total = len(examples)

    for i, example in enumerate(examples):
        prompt = build_prompt(
            tokenizer, example["instruction"], example["input"]
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
        latency = time.perf_counter() - start

        # Strip the prompt tokens; keep only the generation
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True)

        records.append(
            {
                "index": i,
                "input": example["input"],
                "reference": example["output"],
                "prediction": prediction,
                "latency_s": round(latency, 3),
            }
        )

        print(f"[{i + 1}/{total}] latency={latency:.1f}s")

    return records


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score_predictions(records: list[dict]) -> dict:
    """
    Compute all Section 6.1 metrics measurable in Week 4 from the
    prediction records. Returns the full metrics dict written to
    baseline_metrics.json.
    """
    severity_refs, severity_preds = [], []
    incident_refs, incident_preds = [], []
    summary_refs, summary_preds = [], []
    latencies = []
    parse_failure_count = 0

    for record in records:
        ref = parse_reference_output(record["reference"])
        pred = parse_model_output(record["prediction"])

        if pred["parse_errors"]:
            parse_failure_count += 1

        severity_refs.append(ref["severity"])
        # unparseable severity -> "UNPARSEABLE" (counts as wrong,
        # never silently dropped — a format failure IS a failure)
        severity_preds.append(pred["severity"] or "UNPARSEABLE")

        incident_refs.append(ref["incident_type"])
        incident_preds.append(pred["incident_type"] or "UNPARSEABLE")

        summary_refs.append(ref["summary"])
        summary_preds.append(pred["summary"])

        if "latency_s" in record:
            latencies.append(record["latency_s"])

    rouge = metrics_mod.rouge_l_scores(summary_refs, summary_preds)
    incident_f1 = metrics_mod.macro_f1(incident_refs, incident_preds)
    severity_f1 = metrics_mod.macro_f1(severity_refs, severity_preds)
    fpr = metrics_mod.false_positive_rate(severity_refs, severity_preds)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": DEFAULT_MODEL_NAME,
        "mode": "zero-shot baseline (unmodified SLM)",
        "num_examples": len(records),
        "parse_failures": parse_failure_count,
        "parse_failure_rate": (
            parse_failure_count / len(records) if records else 0.0
        ),
        "severity_accuracy": metrics_mod.severity_accuracy(
            severity_refs, severity_preds
        ),
        "severity_f1_macro": severity_f1,
        "severity_classification_report": metrics_mod.severity_report(
            severity_refs, severity_preds
        ),
        "severity_confusion_matrix": metrics_mod.confusion_matrix(
            severity_refs, severity_preds
        ),
        "incident_type_f1_macro": incident_f1,
        "rouge_l": {
            "mean": rouge["mean"],
            "implementation": rouge["implementation"],
        },
        "false_positive_rate": fpr,
        "latency": metrics_mod.latency_stats(latencies),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "targets": TARGETS,
    }


def select_example_outputs(records: list[dict], count: int = 5) -> list[dict]:
    """
    Pick representative examples for the D3 report ("with example
    outputs"): mix of parse successes and failures across severities.
    """
    chosen = []
    seen_severities = set()

    for record in records:
        ref = parse_model_output(record["reference"])
        if ref["severity"] not in seen_severities:
            seen_severities.add(ref["severity"])
            chosen.append(record)
        if len(chosen) >= count:
            break

    return chosen


# ------------------------------------------------------------------
# Markdown report (Deliverable D3)
# ------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _pass_fail(value: float, key: str) -> str:
    spec = TARGETS[key]
    ok = value > spec["target"] if spec["direction"] == ">" else value < spec["target"]
    return "PASS" if ok else "FAIL"


def write_markdown_report(metrics: dict, examples: list[dict]) -> None:
    """Render Week4_Baseline_Report.md (deliverable D3)."""
    m = metrics
    lines = []
    add = lines.append

    add("# Week 4 — Zero-Shot Baseline Evaluation Report (D3)")
    add("")
    add(f"- **Model:** {m['model']} (unmodified, zero-shot)")
    add(f"- **Test set:** {m['num_examples']} held-out examples "
        "(document Section 6: 200+ log scenarios)")
    add(f"- **Generated:** {m['generated_at']}")
    add(f"- **Environment:** Python {m['environment']['python']}, "
        f"{m['environment']['platform']}")
    add("- **Decoding:** greedy (deterministic, reproducible)")
    add("")
    add("All inference and scoring performed **fully offline** "
        "(HF_HUB_OFFLINE=1, local model weights, no external APIs).")
    add("")

    add("## Metrics vs. Section 6.1 Targets")
    add("")
    add("| Metric | Baseline | Target | Status |")
    add("|---|---|---|---|")
    add(f"| Severity Classification Accuracy | "
        f"{_pct(m['severity_accuracy'])} | > 85% | "
        f"{_pass_fail(m['severity_accuracy'], 'severity_accuracy')} |")
    add(f"| Incident Type F1 (macro) | "
        f"{m['incident_type_f1_macro']['value']:.3f} | > 0.80 | "
        f"{_pass_fail(m['incident_type_f1_macro']['value'], 'incident_type_f1_macro')} |")
    add(f"| ROUGE-L (summaries) | {m['rouge_l']['mean']:.3f} | > 0.55 | "
        f"{_pass_fail(m['rouge_l']['mean'], 'rouge_l_mean')} |")
    add(f"| False Positive Rate | "
        f"{_pct(m['false_positive_rate']['rate'])} | < 10% | "
        f"{_pass_fail(m['false_positive_rate']['rate'], 'false_positive_rate')} |")
    add("| Root Cause Accuracy | *human evaluation — see example "
        "outputs below and baseline_predictions.jsonl* | > 75% | MANUAL |")
    add("| API Response Time (p95) | *measured in Week 9 (/analyze "
        f"endpoint); raw model p95 = {m['latency']['p95']:.1f}s/example* "
        "| < 5 s | DEFERRED |")
    add("| RAG Retrieval Precision@3 | *measured in Week 7* | > 70% "
        "| DEFERRED |")
    add("")
    add("A zero-shot baseline is **expected to miss the targets** — "
        "these numbers are the reference point that Week 5-6 "
        "fine-tuning must beat (document Section 10.1: \"Fine-tuned "
        "model outperforms baseline on all key metrics\").")
    add("")

    add("## Output Format Compliance")
    add("")
    add(f"- Parse failures (missing/invalid fields): "
        f"{m['parse_failures']} / {m['num_examples']} "
        f"({_pct(m['parse_failure_rate'])})")
    add("- Unparseable fields are scored as incorrect, never dropped.")
    add("")

    add("## Severity Classification Report")
    add("")
    add("```")
    add(m["severity_classification_report"].rstrip())
    add("```")
    add("")

    add("## Severity Confusion Matrix (reference rows × prediction columns)")
    add("")
    cm = m["severity_confusion_matrix"]
    all_labels = sorted(
        set(cm.keys())
        | {p for row in cm.values() for p in row.keys()}
    )
    add("| ref \\ pred | " + " | ".join(all_labels) + " |")
    add("|---|" + "---|" * len(all_labels))
    for ref_label in sorted(cm.keys()):
        row = cm[ref_label]
        cells = [str(row.get(p, 0)) for p in all_labels]
        add(f"| **{ref_label}** | " + " | ".join(cells) + " |")
    add("")

    add("## Latency (per-example model inference)")
    add("")
    lat = m["latency"]
    add("| mean | p50 | p95 | max |")
    add("|---|---|---|---|")
    add(f"| {lat['mean']:.1f}s | {lat['p50']:.1f}s | "
        f"{lat['p95']:.1f}s | {lat['max']:.1f}s |")
    add("")

    add("## Metric Implementations")
    add("")
    add(f"- ROUGE-L: `{m['rouge_l']['implementation']}`")
    add(f"- Incident F1: `{m['incident_type_f1_macro']['implementation']}`")
    add("")
    if (m["rouge_l"]["implementation"] == "fallback"
            or m["incident_type_f1_macro"]["implementation"] == "fallback"):
        add("> **Note:** fallback (standard-library) implementations were "
            "used because `rouge-score`/`scikit-learn` were unavailable in "
            "this environment. Re-run with the document-mandated libraries "
            "installed (Section 7) for the official numbers.")
        add("")

    add("## Example Outputs (for human root-cause review)")
    add("")
    for ex in examples:
        ref = parse_model_output(ex["reference"])
        add(f"### Example {ex['index']} "
            f"(reference severity: {ref['severity']})")
        add("")
        add("**Input (truncated):**")
        add("```")
        add(ex["input"][:600])
        add("```")
        add("**Reference output:**")
        add("```")
        add(ex["reference"].rstrip())
        add("```")
        add("**Model prediction:**")
        add("```")
        add((ex["prediction"] or "<empty>").rstrip()[:1200])
        add("```")
        add("")

    add("## Artifacts")
    add("")
    add("- `reports/baseline_predictions.jsonl` — every raw prediction")
    add("- `reports/baseline_metrics.json` — machine-readable metrics")
    add("")
    add("---")
    add("*Generated by `python -m src.training.evaluate` "
        "(Week 4 deliverable D3).*")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Week 4 zero-shot baseline evaluation (offline)."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="cuda strongly recommended; CPU inference "
                             "on a 3B model takes ~30-60s per example")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N examples "
                             "(smoke test)")
    parser.add_argument("--max-new-tokens", type=int,
                        default=GENERATION_KWARGS["max_new_tokens"])
    parser.add_argument("--skip-inference", action="store_true",
                        help="re-score existing "
                             "reports/baseline_predictions.jsonl "
                             "without loading the model")
    args = parser.parse_args()

    print("=" * 60)
    print("Week 4 — Zero-Shot Baseline Evaluation (offline)")
    print("=" * 60)

    if args.skip_inference:
        records = read_jsonl(PREDICTIONS_FILE)
        if not records:
            print(f"ERROR: {PREDICTIONS_FILE} is empty or missing; "
                  "run without --skip-inference first.")
            return 1
        print(f"Re-scoring {len(records)} cached predictions "
              f"from {PREDICTIONS_FILE}")
    else:
        examples = read_jsonl(args.test_file)
        if not examples:
            print(f"ERROR: no examples in {args.test_file}")
            return 1
        if args.limit:
            examples = examples[: args.limit]
        print(f"Test examples : {len(examples)}")
        print(f"Model         : {DEFAULT_MODEL_NAME} "
              f"(local, {args.model_dir})")
        print(f"Device        : {args.device}")

        records = run_inference(
            examples, args.model_dir, args.device, args.max_new_tokens
        )
        write_jsonl(PREDICTIONS_FILE, records)
        print(f"Predictions saved to {PREDICTIONS_FILE}")

    metrics = score_predictions(records)

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {METRICS_FILE}")

    examples_for_report = select_example_outputs(records)
    write_markdown_report(metrics, examples_for_report)
    print(f"Report saved to {REPORT_FILE}")

    print()
    print(f"Severity accuracy : {metrics['severity_accuracy']:.1%}")
    print(f"Incident F1 macro : "
          f"{metrics['incident_type_f1_macro']['value']:.3f}")
    print(f"Mean ROUGE-L      : {metrics['rouge_l']['mean']:.3f}")
    print(f"False positive %  : "
          f"{metrics['false_positive_rate']['rate']:.1%}")
    print(f"Parse failures    : {metrics['parse_failures']}"
          f"/{metrics['num_examples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
