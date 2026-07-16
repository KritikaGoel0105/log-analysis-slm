"""
utils.py

Utility functions for Week 3 dataset creation.
"""

import json
from pathlib import Path


def read_jsonl(file_path: str):
    """
    Read a JSONL file and return a list of dictionaries.
    """
    data = []

    path = Path(file_path)

    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def write_jsonl(file_path: str, records: list):
    """
    Write a list of dictionaries to a JSONL file.
    """

    path = Path(file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def window_to_text(window: dict) -> str:
    """
    Convert one processed log window into
    plain multiline text.
    """

    return "\n".join(window["logs"])


def validate_example(example: dict) -> bool:
    """
    Verify that one training example contains
    all required fields.
    """

    required = {
        "instruction",
        "input",
        "output"
    }

    return required.issubset(example.keys())