import json
import re
from pathlib import Path
from datetime import datetime

# ----------------------------
# Normalization Patterns
# ----------------------------

NORMALIZATION_PATTERNS = [
    # IP addresses
    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP_ADDR>'),

    # UUIDs
    (r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>'),

    # ISO 8601 timestamps
    (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?', '<TIMESTAMP>'),

    # User IDs
    (r'\b(user[_\s]?)(\d+)\b', r'\1<USER_ID>'),

    # File paths
    (r'/[a-zA-Z0-9_/.-]+\.[a-zA-Z]{2,4}', '<FILE_PATH>'),

    # Memory addresses
    (r'0x[0-9a-fA-F]+', '<MEM_ADDR>'),

    # Port numbers
    (r'(?<=:)\d{4,5}\b', '<PORT>'),
]


# ----------------------------
# Normalize a single log line
# ----------------------------

def normalize_log_line(line: str) -> str:
    """
    Replace variable values with placeholders
    to create model-friendly log text.
    """
    for pattern, replacement in NORMALIZATION_PATTERNS:
        line = re.sub(pattern, replacement, line)

    return line.strip()


# ----------------------------
# Timestamp Parser
# ----------------------------

def parse_timestamp(line: str):
    """
    Extract timestamp from a log line.
    Returns datetime object if found,
    otherwise returns None.
    """

    match = re.search(
        r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)',
        line
    )

    if not match:
        return None

    timestamp = match.group(1)

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue

    return None
# ----------------------------
# Log Window Extraction
# ----------------------------

def extract_windows(
    log_lines: list[str],
    max_gap_seconds: int = 60,
    max_lines: int = 20
) -> list[list[str]]:

    windows = []
    current = []
    prev_ts = None

    for line in log_lines:

        ts = parse_timestamp(line)

        if ts and prev_ts:
            if (ts - prev_ts).seconds > max_gap_seconds:
                if current:
                    windows.append(current)
                current = []

        if len(current) >= max_lines:
            windows.append(current)
            current = []

        current.append(normalize_log_line(line))

        prev_ts = ts or prev_ts

    if current:
        windows.append(current)

    return windows

def process_log_file(
    input_file: str,
    max_gap_seconds: int = 60,
    max_lines: int = 20,
) -> list[list[str]]:
    """
    Read a raw log file and convert it into normalized log windows.

    Args:
        input_file: Path to the raw log file.
        max_gap_seconds: Maximum allowed time gap before creating a new window.
        max_lines: Maximum number of lines in one window.

    Returns:
        List of normalized log windows.
    """

    input_path = Path(input_file)

    with input_path.open("r", encoding="utf-8") as file:
        log_lines = file.readlines()

    return extract_windows(
        log_lines,
        max_gap_seconds=max_gap_seconds,
        max_lines=max_lines,
    )

def save_processed_windows(
    windows: list[list[str]],
    output_file: str,
) -> None:
    """
    Save normalized log windows to a local JSONL file.
    One window is stored per line.
    """

    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for index, window in enumerate(windows, start=1):
            record = {
                "window_id": index,
                "logs": window,
            }

            file.write(json.dumps(record))
            file.write("\n")