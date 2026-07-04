from pathlib import Path

from src.preprocessing.preprocessor import (
    process_log_file,
    save_processed_windows,
)

input_file = Path("data/raw/sample.log")
output_file = Path("data/processed/sample.jsonl")

windows = process_log_file(str(input_file))

save_processed_windows(
    windows,
    str(output_file),
)

print(f"Processed {len(windows)} window(s).")
print(f"Output saved to: {output_file}")