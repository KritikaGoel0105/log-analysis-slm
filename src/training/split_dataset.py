"""
split_dataset.py

Splits logs_dataset.jsonl into train/val/test sets with the
80/10/10 ratio required by the internship document
(Section 8, Deliverable D2):

    "logs_dataset.jsonl: 1000+ examples;
     train/val/test split 80/10/10;
     README describing sources"

The split is stratified by SEVERITY so that each split has a
representative distribution of all severity classes. A fixed
random seed makes the split reproducible.

Runs fully offline — only the Python standard library is used.

Usage (from repository root):
    python -m src.training.split_dataset
"""

import random
from collections import defaultdict
from pathlib import Path

from .utils import read_jsonl, write_jsonl, validate_example

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATASET_DIR = Path("data/dataset")
INPUT_DATASET = DATASET_DIR / "logs_dataset.jsonl"

TRAIN_FILE = DATASET_DIR / "train.jsonl"
VAL_FILE = DATASET_DIR / "val.jsonl"
TEST_FILE = DATASET_DIR / "test.jsonl"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# test ratio is the remainder (0.10)

RANDOM_SEED = 42


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def extract_severity(example: dict) -> str:
    """
    Extract the SEVERITY value from an example's output field.
    Returns "UNKNOWN" if the output does not start with the
    expected 'SEVERITY: <value>' line.
    """
    output = example.get("output", "")

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("SEVERITY:"):
            return line.split(":", 1)[1].strip()

    return "UNKNOWN"


def stratified_split(examples: list[dict]):
    """
    Split examples into train/val/test (80/10/10), stratified
    by severity class. Within each class, examples are shuffled
    with a fixed seed, then partitioned by ratio.
    """
    by_severity = defaultdict(list)

    for example in examples:
        by_severity[extract_severity(example)].append(example)

    rng = random.Random(RANDOM_SEED)

    train, val, test = [], [], []

    for severity in sorted(by_severity):
        group = by_severity[severity]
        rng.shuffle(group)

        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    # Shuffle each split so classes are interleaved, not grouped
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def severity_distribution(examples: list[dict]) -> dict:
    """Count examples per severity class."""
    counts = defaultdict(int)
    for example in examples:
        counts[extract_severity(example)] += 1
    return dict(sorted(counts.items()))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    examples = read_jsonl(INPUT_DATASET)

    if not examples:
        print(f"ERROR: no examples found in {INPUT_DATASET}")
        print("Run build_dataset.py first.")
        return

    # Validate schema before splitting
    invalid = [
        i for i, ex in enumerate(examples) if not validate_example(ex)
    ]
    if invalid:
        print(f"ERROR: {len(invalid)} examples missing required fields")
        print(f"First invalid indices: {invalid[:10]}")
        return

    train, val, test = stratified_split(examples)

    write_jsonl(TRAIN_FILE, train)
    write_jsonl(VAL_FILE, val)
    write_jsonl(TEST_FILE, test)

    total = len(examples)

    print("=" * 60)
    print("Week 3 Dataset Split (80/10/10, stratified by severity)")
    print("=" * 60)
    print(f"Total examples : {total}")
    print(f"Train          : {len(train)} ({len(train) / total:.1%})")
    print(f"Val            : {len(val)} ({len(val) / total:.1%})")
    print(f"Test           : {len(test)} ({len(test) / total:.1%})")
    print()
    print("Severity distribution per split:")
    print(f"  train : {severity_distribution(train)}")
    print(f"  val   : {severity_distribution(val)}")
    print(f"  test  : {severity_distribution(test)}")
    print()
    print("Files written:")
    print(f"  {TRAIN_FILE.resolve()}")
    print(f"  {VAL_FILE.resolve()}")
    print(f"  {TEST_FILE.resolve()}")


if __name__ == "__main__":
    main()
