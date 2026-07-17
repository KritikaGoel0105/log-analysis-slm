"""
test_evaluation.py

Unit tests for the Week 4 evaluation components that do not require
the ML stack: output_parser.py and metrics.py.

Run from the repository root:
    python -m pytest src/training/tests/ -v
or without pytest:
    python -m src.training.tests.test_evaluation
"""

import sys
from pathlib import Path

# Allow direct execution (python path fix)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.training.output_parser import (  # noqa: E402
    parse_model_output,
    parse_reference_output,
)
from src.training import metrics  # noqa: E402


# ------------------------------------------------------------------
# output_parser tests
# ------------------------------------------------------------------

WELL_FORMED = """SEVERITY: HIGH
INCIDENT_TYPE: SSH Brute Force Attack
ROOT_CAUSE: Repeated failed password attempts from a single IP.
SUMMARY: Multiple failed SSH logins detected. The pattern indicates a brute-force attack.
RECOMMENDED_ACTIONS:
1. Block the source IP at the firewall.
2. Enforce key-based authentication.
"""


def test_parse_well_formed():
    parsed = parse_model_output(WELL_FORMED)
    assert parsed["severity"] == "HIGH"
    assert parsed["incident_type"] == "SSH Brute Force Attack"
    assert parsed["root_cause"].startswith("Repeated failed")
    assert "brute-force" in parsed["summary"]
    assert len(parsed["recommended_actions"]) == 2
    assert parsed["recommended_actions"][0].startswith("Block")
    assert parsed["parse_errors"] == []


def test_parse_markdown_decorated():
    """Zero-shot models often emit markdown bold and numbering."""
    text = (
        "1. **SEVERITY**: [CRITICAL]\n"
        "2. **INCIDENT_TYPE**: Disk Failure\n"
        "3. **ROOT_CAUSE**: Hardware fault\n"
        "4. **SUMMARY**: The disk failed.\n"
        "5. **RECOMMENDED_ACTIONS**:\n"
        "- Replace the disk\n"
    )
    parsed = parse_model_output(text)
    assert parsed["severity"] == "CRITICAL"
    assert parsed["incident_type"] == "Disk Failure"
    assert parsed["recommended_actions"] == ["Replace the disk"]
    assert parsed["parse_errors"] == []


def test_parse_multiline_summary():
    text = (
        "SEVERITY: LOW\n"
        "INCIDENT_TYPE: Routine\n"
        "ROOT_CAUSE: None\n"
        "SUMMARY: First line.\n"
        "Second line continues the summary.\n"
        "RECOMMENDED_ACTIONS:\n1. Nothing\n"
    )
    parsed = parse_model_output(text)
    assert "Second line" in parsed["summary"]


def test_parse_missing_fields():
    parsed = parse_model_output("The logs look fine to me!")
    assert parsed["severity"] == ""
    assert len(parsed["parse_errors"]) == 5


def test_parse_invalid_severity():
    text = "SEVERITY: BANANAS\nINCIDENT_TYPE: x\nROOT_CAUSE: y\nSUMMARY: z\nRECOMMENDED_ACTIONS:\n1. a\n"
    parsed = parse_model_output(text)
    assert parsed["severity"] == ""
    assert "SEVERITY_INVALID" in parsed["parse_errors"]


def test_parse_empty_input():
    parsed = parse_model_output("")
    assert parsed["severity"] == ""
    assert len(parsed["parse_errors"]) == 5


def test_reference_parser_raises_on_bad_reference():
    try:
        parse_reference_output("garbage")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ------------------------------------------------------------------
# metrics tests
# ------------------------------------------------------------------

def test_severity_accuracy():
    refs = ["HIGH", "INFO", "LOW", "HIGH"]
    preds = ["HIGH", "INFO", "HIGH", "LOW"]
    assert metrics.severity_accuracy(refs, preds) == 0.5


def test_macro_f1_perfect():
    refs = ["A", "B", "A", "B"]
    result = metrics.macro_f1(refs, refs)
    assert abs(result["value"] - 1.0) < 1e-9


def test_macro_f1_all_wrong():
    refs = ["A", "A"]
    preds = ["B", "B"]
    result = metrics.macro_f1(refs, preds)
    assert result["value"] == 0.0


def test_fallback_rouge_l_identical():
    r = metrics._fallback_rouge_l("the disk failed today", "the disk failed today")
    assert abs(r - 1.0) < 1e-9


def test_fallback_rouge_l_disjoint():
    assert metrics._fallback_rouge_l("aaa bbb", "ccc ddd") == 0.0


def test_rouge_l_scores_shape():
    result = metrics.rouge_l_scores(["a b c"], ["a b c"])
    assert result["mean"] > 0.99
    assert len(result["per_example"]) == 1
    assert result["implementation"] in ("rouge_score", "fallback")


def test_false_positive_rate():
    refs = ["INFO", "INFO", "LOW", "HIGH"]
    preds = ["HIGH", "INFO", "MEDIUM", "HIGH"]
    fpr = metrics.false_positive_rate(refs, preds)
    # 3 non-incidents (INFO, INFO, LOW); 2 flagged (HIGH, MEDIUM)
    assert fpr["non_incident_total"] == 3
    assert fpr["false_positives"] == 2
    assert abs(fpr["rate"] - 2 / 3) < 1e-9


def test_confusion_matrix():
    cm = metrics.confusion_matrix(["A", "A", "B"], ["A", "B", "B"])
    assert cm["A"]["A"] == 1
    assert cm["A"]["B"] == 1
    assert cm["B"]["B"] == 1


def test_latency_stats():
    stats = metrics.latency_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["max"] == 4.0
    assert stats["mean"] == 2.5
    assert stats["p95"] in (3.0, 4.0)


def test_latency_stats_empty():
    assert metrics.latency_stats([])["p95"] == 0.0


# ------------------------------------------------------------------
# Standalone runner (no pytest needed — offline-friendly)
# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
