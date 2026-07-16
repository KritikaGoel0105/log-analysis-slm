"""
Read-only audit of logs_dataset.jsonl against the current label pipeline.

Does not modify any dataset files.
"""

import json
import re
from collections import Counter
from pathlib import Path

from src.training.label_generator import generate_label, infer_severity
from src.training.synthetic_generator import SYNTHETIC_TEMPLATES

DATASET = Path("data/dataset/logs_dataset.jsonl")
LEGACY_PHRASE = "could not be determined"
MISAPPLIED_DB_PHRASE = "connection pool appears to be exhausted"
SYNTHETIC_INPUTS = {t["logs"].strip() for t in SYNTHETIC_TEMPLATES}


def parse_output(output: str) -> dict:
    fields = {}
    for key in ("SEVERITY", "INCIDENT_TYPE", "ROOT_CAUSE", "SUMMARY"):
        match = re.search(rf"^{key}:\s*(.+)$", output, re.M)
        fields[key.lower()] = match.group(1).strip() if match else None
    return fields


def load_examples():
    examples = []
    with DATASET.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def audit():
    examples = load_examples()
    stats = Counter()
    severity_mismatch = []
    false_db_incident = []
    false_critical = []
    legacy_samples = []
    exact_match = 0

    for index, example in enumerate(examples, start=1):
        text_input = example["input"].strip()
        text_output = example["output"]
        parsed = parse_output(text_output)
        severity = parsed.get("severity")
        incident = parsed.get("incident_type")

        stats["total"] += 1
        if severity:
            stats[f"severity:{severity}"] += 1
        if incident:
            stats[f"incident:{incident}"] += 1

        if text_input in SYNTHETIC_INPUTS:
            stats["synthetic"] += 1
            continue

        if LEGACY_PHRASE in text_output:
            stats["legacy_labeling"] += 1
            legacy_samples.append(index)

        if MISAPPLIED_DB_PHRASE in text_output:
            stats["misapplied_db_pool"] += 1
            if "database" not in text_input.lower() and "db-" not in text_input.lower():
                false_db_incident.append(index)

        regenerated = generate_label(text_input)
        if regenerated.strip() == text_output.strip():
            exact_match += 1
            stats["exact_match_current_generator"] += 1

        inferred = infer_severity(text_input)
        if severity and severity != inferred:
            severity_mismatch.append((index, severity, inferred))

        lowered = text_input.lower()
        if severity == "CRITICAL" and not any(
            token in lowered for token in ("crit", "critical", "fatal", "error", "[error]")
        ):
            false_critical.append(index)

    return {
        "stats": stats,
        "exact_match": exact_match,
        "severity_mismatch": severity_mismatch,
        "false_db_incident": false_db_incident,
        "false_critical": false_critical,
        "legacy_samples": legacy_samples,
    }


def main():
    result = audit()
    stats = result["stats"]

    print("=" * 60)
    print("Dataset Label Audit (read-only)")
    print("=" * 60)
    print(f"Dataset file     : {DATASET.resolve()}")
    print(f"Total examples   : {stats['total']}")
    print(f"Synthetic        : {stats['synthetic']}")
    print(f"Legacy labels    : {stats['legacy_labeling']}")
    print(f"Misapplied DB    : {stats['misapplied_db_pool']}")
    print(f"False DB labels  : {len(result['false_db_incident'])}")
    print(f"Exact match now  : {result['exact_match']}")
    print(f"Severity mismatch: {len(result['severity_mismatch'])}")
    print(f"False CRITICAL   : {len(result['false_critical'])}")
    print()
    print("Severity distribution:")
    for key, value in sorted(stats.items()):
        if key.startswith("severity:"):
            print(f"  {key.split(':', 1)[1]}: {value}")
    print()
    print("Top incident types:")
    incident_counts = [
        (k.split(":", 1)[1], v)
        for k, v in stats.items()
        if k.startswith("incident:")
    ]
    for name, count in sorted(incident_counts, key=lambda item: -item[1])[:10]:
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
