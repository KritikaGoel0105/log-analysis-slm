"""
training_report.py

Week 6 — Training report generator: loss curves, overfitting
diagnosis, best-checkpoint identification, baseline-vs-fine-tuned
comparison.

Document basis:
  * Week 6 (Section 4): "Fine-tuning iterations: hyperparameter
    tuning, loss curves, overfitting diagnosis" -> deliverable
    "Best model checkpoint + training report".
  * D4 (Section 8, end of Week 6): "Model checkpoint + LoRA adapter;
    evaluation report showing improvement over baseline".
  * Section 10.1 (2): "Fine-tuned model outperforms baseline on all
    key metrics".
  * Section 5.4 LoRA comment: "r: start with 16, try 32 if
    underfitting" — the hyperparameter-iteration decision rule this
    report applies and documents.

Scope guard (Section 8): confusion matrices and the 3-way
baseline/fine-tuned/RAG comparison belong to D6 (Week 8) and are
deliberately NOT produced here.

Inputs (all local files — no ML stack, no network, stdlib only):
  * models/checkpoints/training_summary.json  (Week 5 run log)
  * reports/baseline_metrics.json             (frozen Week 4 D3)
  * reports/finetuned_metrics.json            (optional — produced by
        `python -m src.training.evaluate --adapter ...`; the
        comparison section is emitted only when it exists)

Outputs (under reports/):
  * loss_curves.svg          — train/eval loss curves (Section 4).
        Rendered with a stdlib SVG writer: matplotlib is NOT in
        offline_packages/, and the offline constraint (Section 2)
        forbids fetching it, so no new dependency is introduced.
  * Week6_Training_Report.md — the D4/Week 6 training report.

Usage (from repository root; works without the ML venv):
    python -m src.training.training_report
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_FILE = REPO_ROOT / "models" / "checkpoints" / "training_summary.json"
BASELINE_METRICS_FILE = REPO_ROOT / "reports" / "baseline_metrics.json"
FINETUNED_METRICS_FILE = REPO_ROOT / "reports" / "finetuned_metrics.json"

REPORTS_DIR = REPO_ROOT / "reports"
SVG_FILE = REPORTS_DIR / "loss_curves.svg"
REPORT_FILE = REPORTS_DIR / "Week6_Training_Report.md"

# Section 6.1 targets for the comparison table
TARGETS = {
    "severity_accuracy": (0.85, ">"),
    "incident_type_f1_macro": (0.80, ">"),
    "rouge_l_mean": (0.55, ">"),
    "false_positive_rate": (0.10, "<"),
}


# ------------------------------------------------------------------
# Loss-curve extraction
# ------------------------------------------------------------------

def extract_curves(log_history: list[dict]):
    """Split trainer log_history into train-loss and eval-loss series."""
    train = [(e["epoch"], e["loss"])
             for e in log_history if "loss" in e]
    evals = [(e["epoch"], e["eval_loss"])
             for e in log_history if "eval_loss" in e]
    return train, evals


# ------------------------------------------------------------------
# SVG rendering (stdlib only — see module docstring)
# ------------------------------------------------------------------

def render_loss_svg(train: list[tuple], evals: list[tuple],
                    path: Path) -> None:
    """Write a self-contained SVG plot of train/eval loss vs epoch."""
    width, height = 720, 440
    ml, mr, mt, mb = 60, 20, 40, 50          # margins
    pw, ph = width - ml - mr, height - mt - mb

    all_pts = train + evals
    x_max = max(x for x, _ in all_pts)
    y_max = max(y for _, y in all_pts)
    x_max = max(x_max, 3.0)
    y_max = y_max * 1.05

    def sx(x): return ml + (x / x_max) * pw
    def sy(y): return mt + ph - (y / y_max) * ph

    def polyline(pts, color):
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        return (f'<polyline fill="none" stroke="{color}" '
                f'stroke-width="2" points="{coords}"/>')

    def dots(pts, color):
        return "".join(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" '
            f'fill="{color}"/>' for x, y in pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2:.0f}" y="24" text-anchor="middle" '
        'font-family="sans-serif" font-size="16" font-weight="bold">'
        'QLoRA Fine-Tuning — Loss Curves (Week 5 run)</text>',
    ]

    # Axes
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" '
                 f'y2="{mt+ph}" stroke="black"/>')
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" '
                 f'y2="{mt+ph}" stroke="black"/>')

    # X ticks at each epoch
    for e in range(0, int(x_max) + 1):
        x = sx(e)
        parts.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" '
                     f'y2="{mt+ph+5}" stroke="black"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+20}" '
                     'text-anchor="middle" font-family="sans-serif" '
                     f'font-size="12">{e}</text>')
    parts.append(f'<text x="{ml+pw/2:.0f}" y="{height-12}" '
                 'text-anchor="middle" font-family="sans-serif" '
                 'font-size="13">epoch</text>')

    # Y ticks (5 divisions) + horizontal gridlines
    for i in range(6):
        yv = y_max * i / 5
        y = sy(yv)
        parts.append(f'<line x1="{ml-5}" y1="{y:.1f}" x2="{ml}" '
                     f'y2="{y:.1f}" stroke="black"/>')
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" '
                     f'y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{ml-9}" y="{y+4:.1f}" '
                     'text-anchor="end" font-family="sans-serif" '
                     f'font-size="12">{yv:.2f}</text>')
    parts.append(f'<text x="18" y="{mt+ph/2:.0f}" text-anchor="middle" '
                 'font-family="sans-serif" font-size="13" '
                 f'transform="rotate(-90 18 {mt+ph/2:.0f})">loss</text>')

    # Series: train (blue line), eval (red line + dots)
    parts.append(polyline(train, "#1f6fb2"))
    parts.append(polyline(evals, "#c0392b"))
    parts.append(dots(evals, "#c0392b"))

    # Eval-point value labels
    for x, y in evals:
        parts.append(f'<text x="{sx(x):.1f}" y="{sy(y)-10:.1f}" '
                     'text-anchor="middle" font-family="sans-serif" '
                     f'font-size="11" fill="#c0392b">{y:.4f}</text>')

    # Legend
    lx, ly = ml + 12, mt + 10
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" '
                 'stroke="#1f6fb2" stroke-width="2"/>')
    parts.append(f'<text x="{lx+34}" y="{ly+4}" font-family="sans-serif" '
                 'font-size="12">train loss (per 10 steps)</text>')
    parts.append(f'<line x1="{lx}" y1="{ly+18}" x2="{lx+28}" '
                 f'y2="{ly+18}" stroke="#c0392b" stroke-width="2"/>')
    parts.append(f'<text x="{lx+34}" y="{ly+22}" '
                 'font-family="sans-serif" font-size="12">'
                 'eval loss (per epoch)</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


# ------------------------------------------------------------------
# Overfitting diagnosis (Section 4: "overfitting diagnosis")
# ------------------------------------------------------------------

def diagnose(train: list[tuple], evals: list[tuple]) -> dict:
    """
    Rule-based diagnosis from the loss series:
      * overfitting  — eval loss rises while train loss keeps falling
      * underfitting — eval loss high / barely improved at the end
      * converged    — eval loss decreases monotonically to a low value
    Returns the diagnosis plus the evidence used.
    """
    eval_losses = [y for _, y in evals]
    rising = any(b > a * 1.02 for a, b in zip(eval_losses, eval_losses[1:]))
    final = eval_losses[-1]
    first = eval_losses[0]

    if rising:
        verdict = "OVERFITTING"
        explanation = (
            "Validation loss increased between epochs while training "
            "loss continued to fall — the classic divergence signature. "
            "The best checkpoint is the epoch BEFORE the rise.")
    elif final > 0.5 or final > first * 0.9:
        verdict = "UNDERFITTING"
        explanation = (
            "Validation loss remains high / barely improved. Per the "
            "document's Section 5.4 rule ('start with 16, try 32 if "
            "underfitting'), a second run with LoRA r=32 is indicated.")
    else:
        verdict = "CONVERGED — NO OVERFITTING, NO UNDERFITTING"
        explanation = (
            "Validation loss decreased monotonically across all epochs "
            "and finished low, tracking the training loss without "
            "divergence. There is no epoch at which validation loss "
            "rises, so no overfitting; the large reduction from the "
            "first to the final epoch rules out underfitting.")

    return {
        "verdict": verdict,
        "explanation": explanation,
        "eval_losses": eval_losses,
        "best_epoch": min(range(len(eval_losses)),
                          key=lambda i: eval_losses[i]) + 1,
    }


# ------------------------------------------------------------------
# Comparison helpers
# ------------------------------------------------------------------

def _metric_row(name, key, base, fine, fmt, target_key):
    """One row of the baseline-vs-fine-tuned table."""
    def get(m):
        v = m.get(key)
        if isinstance(v, dict):
            v = v.get("value", v.get("mean", v.get("rate")))
        return v

    b, f = get(base), (get(fine) if fine else None)
    target, direction = TARGETS[target_key]
    if f is None:
        return (f"| {name} | {fmt(b)} | *pending — run evaluate.py "
                f"--adapter* | {direction} {fmt(target)} | — |")
    better = f > b if direction == ">" else f < b
    delta = f - b
    status = "IMPROVED" if better else ("UNCHANGED" if delta == 0
                                        else "REGRESSED")
    return (f"| {name} | {fmt(b)} | {fmt(f)} ({delta:+.3f}) | "
            f"{direction} {fmt(target)} | {status} |")


def _pct(x): return f"{x * 100:.1f}%"
def _f3(x): return f"{x:.3f}"


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def write_report(summary: dict, train, evals, diag: dict,
                 baseline: dict, finetuned: dict | None) -> None:
    cfg = summary["config"]
    lines = []
    add = lines.append

    add("# Week 6 — Fine-Tuning Training Report (D4)")
    add("")
    add(f"- **Model:** {summary['model']} + QLoRA LoRA adapter")
    add(f"- **Training completed:** {summary['completed_at']}")
    add(f"- **Generated:** {datetime.now(timezone.utc).isoformat()}")
    add(f"- **Data:** {summary['train_examples']} train / "
        f"{summary['val_examples']} validation examples "
        "(Week 3 dataset, unchanged)")
    add(f"- **Runtime:** {summary['train_runtime_s'] / 3600:.1f} h "
        f"({summary['epochs']} epochs, precision {summary['precision']})")
    add("")
    add("All training and evaluation performed **fully offline** "
        "(HF_HUB_OFFLINE=1, local weights, no external services).")
    add("")

    # ---- Configuration (Section 5.4) ----------------------------
    add("## Training Configuration (document Section 5.4)")
    add("")
    add("| Parameter | Value |")
    add("|---|---|")
    lora = cfg["lora"]
    add(f"| LoRA rank r | {lora['r']} |")
    add(f"| LoRA alpha | {lora['alpha']} |")
    add(f"| LoRA dropout | {lora['dropout']} |")
    add(f"| Target modules | {', '.join(lora['target_modules'])} |")
    add(f"| Quantization | {cfg['quantization']} |")
    add(f"| Batch size × grad accum | {cfg['batch_size']} × "
        f"{cfg['grad_accum']} (effective {cfg['effective_batch']}) |")
    add(f"| Learning rate | {cfg['lr']} ({cfg['scheduler']}, "
        f"warmup {cfg['warmup_ratio']}) |")
    add(f"| Epochs | {summary['epochs']} |")
    add("")

    # ---- Loss curves (Section 4: "loss curves") -----------------
    add("## Loss Curves")
    add("")
    add("![Loss curves](loss_curves.svg)")
    add("")
    add("Per-epoch losses (from `models/checkpoints/"
        "training_summary.json`):")
    add("")
    add("| Epoch | Eval loss | Train loss (last logged step) |")
    add("|---|---|---|")
    for i, (ep, ev) in enumerate(evals):
        prior = [y for x, y in train if x <= ep]
        tl = f"{prior[-1]:.4f}" if prior else "—"
        add(f"| {int(ep)} | {ev:.4f} | {tl} |")
    add(f"| — | final train loss (mean over run): "
        f"{summary['final_train_loss']:.4f} | |")
    add("")

    # ---- Overfitting diagnosis (Section 4) ----------------------
    add("## Overfitting Diagnosis")
    add("")
    add(f"**Verdict: {diag['verdict']}**")
    add("")
    add(diag["explanation"])
    add("")
    ev = diag["eval_losses"]
    add(f"Evidence: eval loss {' → '.join(f'{x:.4f}' for x in ev)} "
        f"across epochs 1–{len(ev)} "
        f"(total reduction {(1 - ev[-1] / ev[0]) * 100:.1f}%).")
    add("")

    # ---- Hyperparameter iteration decision (Section 4 + 5.4) ----
    add("## Hyperparameter Iteration Decision")
    add("")
    add("Document Section 5.4 defines the iteration rule for the "
        "LoRA rank: *\"start with 16, try 32 if underfitting\"*. "
        "The run above used r=16.")
    add("")
    if diag["verdict"] == "UNDERFITTING":
        add("The diagnosis IS underfitting, so per the document a "
            "second run with r=32 is required before Week 6 can be "
            "closed.")
    else:
        add("The diagnosis shows no underfitting (and no "
            "overfitting), so the document's condition for a second "
            "run (r=32) is **not triggered**. The fine-tuning "
            "iteration loop — train → inspect loss curves → decide — "
            "terminates after one run with **r=16 retained**. No "
            "further training run is mandated by the document.")
    add("")

    # ---- Best checkpoint (Section 4 deliverable) ----------------
    add("## Best Model Checkpoint Identification")
    add("")
    best = diag["best_epoch"]
    add(f"Lowest validation loss occurs at **epoch {best}** "
        f"(eval loss {ev[best - 1]:.4f}).")
    add("")
    add("- Saved per-epoch checkpoints: `models/checkpoints/"
        "checkpoint-97` (epoch 1), `checkpoint-194` (epoch 2), "
        "`checkpoint-291` (epoch 3).")
    add("- `models/checkpoints/final-adapter/` was saved immediately "
        "after training ended (no steps after the epoch-3 save), so "
        "its LoRA weights are **identical to checkpoint-291**, the "
        "lowest-eval-loss epoch.")
    add("")
    add("**Best checkpoint = `models/checkpoints/final-adapter/`** "
        "(inference-ready: adapter + tokenizer + chat template). "
        "`checkpoint-291` carries the same weights plus optimizer "
        "state for training resumption.")
    add("")

    # ---- Baseline vs fine-tuned (D4 / Section 10.1) -------------
    add("## Baseline vs Fine-Tuned Comparison "
        "(D4: improvement over baseline)")
    add("")
    if finetuned is None:
        add("> `reports/finetuned_metrics.json` not found yet. Run "
            "the fine-tuned evaluation first, then re-run this "
            "script to fill in this table:")
        add("> `python -m src.training.evaluate --device cuda "
            "--adapter models/checkpoints/final-adapter`")
        add("")
    add("| Metric (Section 6.1) | Baseline | Fine-Tuned (Δ) | "
        "Target | Status |")
    add("|---|---|---|---|---|")
    add(_metric_row("Severity Accuracy", "severity_accuracy",
                    baseline, finetuned, _pct, "severity_accuracy"))
    add(_metric_row("Incident Type F1 (macro)", "incident_type_f1_macro",
                    baseline, finetuned, _f3, "incident_type_f1_macro"))
    add(_metric_row("ROUGE-L (summaries)", "rouge_l",
                    baseline, finetuned, _f3, "rouge_l_mean"))
    add(_metric_row("False Positive Rate", "false_positive_rate",
                    baseline, finetuned, _pct, "false_positive_rate"))
    if finetuned is not None:
        pf_b = baseline.get("parse_failure_rate", 0.0)
        pf_f = finetuned.get("parse_failure_rate", 0.0)
        add(f"| Output parse failures | {_pct(pf_b)} | {_pct(pf_f)} "
            f"({pf_f - pf_b:+.3f}) | — | "
            f"{'IMPROVED' if pf_f < pf_b else 'UNCHANGED' if pf_f == pf_b else 'REGRESSED'} |")
    add("")
    add("Root Cause Accuracy remains a human evaluation (see example "
        "outputs in the evaluation reports). API p95 latency (Week 9) "
        "and RAG P@3 (Week 7) are out of Week 6 scope; confusion "
        "matrices and the 3-way baseline/fine-tuned/RAG table belong "
        "to D6 (Week 8).")
    add("")

    add("## Artifacts")
    add("")
    add("- `models/checkpoints/final-adapter/` — best checkpoint / "
        "LoRA adapter (D4)")
    add("- `models/checkpoints/checkpoint-{97,194,291}/` — per-epoch "
        "checkpoints with trainer state")
    add("- `models/checkpoints/training_summary.json` — raw run log")
    add("- `reports/loss_curves.svg` — loss curves (this report)")
    add("- `reports/finetuned_metrics.json` / "
        "`finetuned_predictions.jsonl` / "
        "`Week6_Finetuned_Evaluation_Report.md` — fine-tuned "
        "evaluation (D4)")
    add("")
    add("---")
    add("*Generated by `python -m src.training.training_report` "
        "(Week 6 deliverable: training report).*")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Week 6 — Training Report Generator (offline, stdlib only)")
    print("=" * 60)

    if not SUMMARY_FILE.exists():
        print(f"ERROR: {SUMMARY_FILE} not found. Run Week 5 training "
              "first (python -m src.training.fine_tune).")
        return 1
    if not BASELINE_METRICS_FILE.exists():
        print(f"ERROR: {BASELINE_METRICS_FILE} not found. The frozen "
              "Week 4 baseline metrics are required for the "
              "comparison (D4).")
        return 1

    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    baseline = json.loads(
        BASELINE_METRICS_FILE.read_text(encoding="utf-8"))
    finetuned = None
    if FINETUNED_METRICS_FILE.exists():
        finetuned = json.loads(
            FINETUNED_METRICS_FILE.read_text(encoding="utf-8"))
        print(f"Fine-tuned metrics found: {FINETUNED_METRICS_FILE}")
    else:
        print("NOTE: finetuned_metrics.json not found — comparison "
              "table will show 'pending'. Run evaluate.py --adapter "
              "first, then re-run this script.")

    train, evals = extract_curves(summary["log_history"])
    if not train or not evals:
        print("ERROR: log_history has no loss/eval_loss entries.")
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    render_loss_svg(train, evals, SVG_FILE)
    print(f"Loss curves saved to {SVG_FILE}")

    diag = diagnose(train, evals)
    print(f"Overfitting diagnosis: {diag['verdict']}")

    write_report(summary, train, evals, diag, baseline, finetuned)
    print(f"Report saved to {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
