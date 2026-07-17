"""
metrics.py

Week 4 evaluation metrics, implementing document Section 6.1:

    Severity Classification Accuracy   target > 85%
    Incident Type F1 (macro)           target > 0.80
    ROUGE-L (summaries)                target > 0.55
    False Positive Rate                target < 10%
    (Root Cause Accuracy is human evaluation — example outputs are
     exported by evaluate.py for manual review.)
    (API p95 latency and RAG P@3 belong to Weeks 7-9; inference
     latency is still measured per-example as an early indicator.)

Primary implementations use the libraries mandated by document
Section 6.2 / Section 7:

    rouge_score   (ROUGE-L, use_stemmer=True)
    scikit-learn  (classification_report, f1_score)

FALLBACK: the wheels for rouge-score are not yet present in
offline_packages/ (see requirements.txt note). If the mandated
libraries are unavailable at runtime, standard-library
implementations are used instead and every result is tagged
"implementation": "fallback" so the report makes the substitution
explicit. Fallback ROUGE-L does not apply Porter stemming, so scores
may differ slightly from the reference implementation.
"""

from collections import Counter, defaultdict

# ------------------------------------------------------------------
# Optional imports (document-mandated libraries)
# ------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer  # type: ignore

    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    from sklearn.metrics import classification_report, f1_score  # type: ignore

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# Severities that count as "incident" for false-positive purposes.
# Ground-truth INFO windows are non-incidents (routine activity);
# predicting MEDIUM or above on them is a false positive
# (document Section 6.1: "% of non-incident log windows incorrectly
# flagged", target < 10%).
NON_INCIDENT_SEVERITIES = {"INFO", "LOW"}
FLAGGED_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM"}


# ------------------------------------------------------------------
# Severity accuracy
# ------------------------------------------------------------------

def severity_accuracy(refs: list[str], preds: list[str]) -> float:
    """Fraction of examples where predicted severity == reference."""
    if not refs:
        return 0.0
    correct = sum(1 for r, p in zip(refs, preds) if r == p)
    return correct / len(refs)


# ------------------------------------------------------------------
# Macro F1 (severity and incident type)
# ------------------------------------------------------------------

def _fallback_macro_f1(refs: list[str], preds: list[str]) -> float:
    """Standard-library macro-averaged F1."""
    labels = sorted(set(refs) | set(preds))
    f1s = []
    for label in labels:
        tp = sum(1 for r, p in zip(refs, preds) if r == label and p == label)
        fp = sum(1 for r, p in zip(refs, preds) if r != label and p == label)
        fn = sum(1 for r, p in zip(refs, preds) if r == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def macro_f1(refs: list[str], preds: list[str]) -> dict:
    """Macro-averaged F1 with implementation provenance."""
    if _HAS_SKLEARN:
        value = float(
            f1_score(refs, preds, average="macro", zero_division=0)
        )
        impl = "sklearn"
    else:
        value = _fallback_macro_f1(refs, preds)
        impl = "fallback"
    return {"value": value, "implementation": impl}


def severity_report(refs: list[str], preds: list[str]) -> str:
    """
    Per-class precision/recall/F1 table
    (document Section 6.2: classification_report).
    """
    if _HAS_SKLEARN:
        return classification_report(refs, preds, zero_division=0)

    # Fallback: minimal per-class table
    lines = [f"{'class':<12} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>5}"]
    for label in sorted(set(refs) | set(preds)):
        tp = sum(1 for r, p in zip(refs, preds) if r == label and p == label)
        fp = sum(1 for r, p in zip(refs, preds) if r != label and p == label)
        fn = sum(1 for r, p in zip(refs, preds) if r == label and p != label)
        n = sum(1 for r in refs if r == label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        lines.append(
            f"{label:<12} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f} {n:>5}"
        )
    lines.append("(fallback implementation — install scikit-learn "
                 "for the full report)")
    return "\n".join(lines)


# ------------------------------------------------------------------
# ROUGE-L
# ------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (O(len(a)*len(b)) DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _fallback_rouge_l(reference: str, prediction: str) -> float:
    """ROUGE-L F-measure without stemming (standard library only)."""
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    lcs = _lcs_length(ref_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_scores(references: list[str], predictions: list[str]) -> dict:
    """
    Per-example and mean ROUGE-L F-measure between reference and
    predicted summaries (document Section 6.2).
    """
    if _HAS_ROUGE:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(references, predictions)
        ]
        impl = "rouge_score"
    else:
        scores = [
            _fallback_rouge_l(ref, pred)
            for ref, pred in zip(references, predictions)
        ]
        impl = "fallback"

    mean = sum(scores) / len(scores) if scores else 0.0
    return {"mean": mean, "per_example": scores, "implementation": impl}


# ------------------------------------------------------------------
# False positive rate
# ------------------------------------------------------------------

def false_positive_rate(refs: list[str], preds: list[str]) -> dict:
    """
    Document Section 6.1: "% of non-incident log windows incorrectly
    flagged" — ground truth INFO/LOW predicted as MEDIUM/HIGH/CRITICAL.
    """
    non_incident_total = 0
    false_positives = 0

    for ref, pred in zip(refs, preds):
        if ref in NON_INCIDENT_SEVERITIES:
            non_incident_total += 1
            if pred in FLAGGED_SEVERITIES:
                false_positives += 1

    rate = (
        false_positives / non_incident_total if non_incident_total else 0.0
    )
    return {
        "rate": rate,
        "false_positives": false_positives,
        "non_incident_total": non_incident_total,
    }


# ------------------------------------------------------------------
# Confusion matrix (used by the Markdown report)
# ------------------------------------------------------------------

def confusion_matrix(refs: list[str], preds: list[str]) -> dict:
    """Nested dict: matrix[reference][prediction] = count."""
    matrix: dict = defaultdict(Counter)
    for ref, pred in zip(refs, preds):
        matrix[ref][pred] += 1
    return {r: dict(c) for r, c in matrix.items()}


# ------------------------------------------------------------------
# Latency
# ------------------------------------------------------------------

def latency_stats(latencies: list[float]) -> dict:
    """Mean / p50 / p95 / max of per-example inference latency (s)."""
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        index = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[index]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }
