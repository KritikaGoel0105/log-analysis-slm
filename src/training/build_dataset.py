"""
build_dataset.py

Builds the Week 3 instruction-following dataset from all
processed LogHub windows and synthetic examples.
"""

from pathlib import Path

from .templates import SYSTEM_PROMPT
from .synthetic_generator import generate_synthetic_examples
from .label_generator import generate_label
from .utils import (
    read_jsonl,
    write_jsonl,
    window_to_text,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUT_DATASET = Path("data/dataset/logs_dataset.jsonl")


# ------------------------------------------------------------------
# Read all processed JSONL files
# ------------------------------------------------------------------

def build_from_processed_windows():
    """
    Build instruction-following examples from all processed
    LogHub JSONL files. The filename (without extension) is
    used as the source identifier for source-aware labeling.
    """

    dataset = []

    processed_files = sorted(PROCESSED_DIR.glob("*.jsonl"))

    if not processed_files:
        print("No processed JSONL files found.")
        return dataset

    for file in processed_files:

        # Use filename as source identifier (e.g. "Apache", "OpenSSH")
        source = file.stem

        windows = read_jsonl(file)

        for window in windows:

            log_text = window_to_text(window)

            dataset.append(
                {
                    "instruction": SYSTEM_PROMPT,
                    "input": log_text,
                    "output": generate_label(log_text, source=source),
                }
            )

    return dataset


def deduplicate(dataset: list[dict]) -> list[dict]:
    """
    Remove examples with duplicate inputs (identical log windows
    occur in some LogHub sources). Duplicates would cause data
    leakage between train/val/test splits during evaluation.
    First occurrence is kept.
    """
    seen = set()
    unique = []

    for example in dataset:
        if example["input"] not in seen:
            seen.add(example["input"])
            unique.append(example)

    return unique


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    processed_examples = build_from_processed_windows()

    synthetic_examples = generate_synthetic_examples()

    dataset = deduplicate(processed_examples + synthetic_examples)

    duplicates_removed = (
        len(processed_examples) + len(synthetic_examples) - len(dataset)
    )

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        OUTPUT_DATASET,
        dataset,
    )

    print("=" * 60)
    print("Week 3 Dataset Creation")
    print("=" * 60)

    print(f"Processed examples : {len(processed_examples)}")
    print(f"Synthetic examples : {len(synthetic_examples)}")
    print(f"Duplicates removed : {duplicates_removed}")
    print(f"Total examples     : {len(dataset)}")

    print()
    print("Dataset saved to:")
    print(OUTPUT_DATASET.resolve())


if __name__ == "__main__":
    main()
