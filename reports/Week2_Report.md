# Week 2 Report – Log Preprocessing Pipeline

## Project

**AI-Powered Log Analysis using Small Language Models (SLMs)**

---

# Objective

The objective of Week 2 was to implement an offline preprocessing pipeline capable of converting raw system logs into normalized, model-friendly log windows. The preprocessing pipeline performs regex-based normalization, timestamp parsing, log window extraction, and stores processed log windows in JSONL format for later dataset creation.

---

# Tasks Completed

## 1. Project Structure

Created the following directories:

```text
src/preprocessing/
src/preprocessing/tests/
data/raw/
data/processed/
```

Created the following files:

```text
src/preprocessing/preprocessor.py
src/preprocessing/run_preprocessor.py
src/preprocessing/tests/test_preprocessor.py
data/raw/sample.log
```

---

# 2. Implemented Log Normalization

Implemented regex-based normalization using Python's `re` module.

The preprocessing module replaces variable values with placeholder tokens.

Supported normalization patterns:

* IP Addresses → `<IP_ADDR>`
* UUIDs → `<UUID>`
* ISO 8601 Timestamps → `<TIMESTAMP>`
* User IDs → `<USER_ID>`
* File Paths → `<FILE_PATH>`
* Memory Addresses → `<MEM_ADDR>`
* Port Numbers → `<PORT>`

---

# 3. Timestamp Parsing

Implemented `parse_timestamp()` to extract timestamps from log entries.

Supported formats:

* YYYY-MM-DD HH:MM:SS
* YYYY-MM-DDTHH:MM:SS
* Millisecond variants of both formats

---

# 4. Log Window Extraction

Implemented `extract_windows()`.

Configuration:

* Maximum time gap: 60 seconds
* Maximum window size: 20 log lines

The function groups related log entries into contextual windows suitable for downstream machine learning tasks.

---

# 5. File Processing

Implemented helper functions:

* `process_log_file()`
* `save_processed_windows()`

The preprocessing pipeline now supports:

Raw Log File

↓

Normalization

↓

Window Extraction

↓

JSONL Storage

---

# 6. Sample Dataset

Created:

```text
data/raw/sample.log
```

Processed output generated:

```text
data/processed/sample.jsonl
```

Example processed output:

```json
{
  "window_id": 1,
  "logs": [
    "<TIMESTAMP> INFO User user_<USER_ID> logged in from <IP_ADDR>",
    "<TIMESTAMP> INFO Opened <FILE_PATH>"
  ]
}
```

---

# 7. Unit Testing

Created:

```text
src/preprocessing/tests/test_preprocessor.py
```

Implemented unit tests for:

* Log normalization
* Timestamp parsing
* Window extraction
* Edge cases

Result:

```text
Ran 14 tests

OK
```

All implemented functionality passed successfully.

---

# Commands Executed

## Run Unit Tests

```powershell
python -m unittest src.preprocessing.tests.test_preprocessor
```

Output:

```text
Ran 14 tests in 0.004s

OK
```

---

## Run Preprocessing Pipeline

```powershell
python -m src.preprocessing.run_preprocessor
```

Expected Output

```text
Processed 2 window(s).
Output saved to: data/processed/sample.jsonl
```

---

# Challenges Encountered

## Python Regular Expression Limitation

The original project document used the following regex for User IDs:

```python
(?<=user[_\s]?)\d+
```

Python's built-in `re` module does not support variable-width lookbehind expressions.

To maintain equivalent functionality while ensuring compatibility with Python 3.11, the expression was replaced with an equivalent capture-group based implementation.

---

# Verification

Verified:

* Offline execution
* Regex normalization
* Timestamp parsing
* Window extraction
* JSONL generation
* Unit tests passing

No internet connectivity or cloud services were used during implementation.

---

# Deliverables Completed

# Deliverables Completed

## Core Week 2 Deliverables

- `src/preprocessing/preprocessor.py`
- `src/preprocessing/tests/test_preprocessor.py`

These files implement the offline log preprocessing pipeline specified for Week 2, including normalization, timestamp parsing, log window extraction, and comprehensive unit testing.

---

## Supporting Utilities

The following files were created to validate and demonstrate the preprocessing pipeline. Although they are not explicitly listed as Week 2 deliverables in the internship document, they support testing and future development.

- `src/preprocessing/run_preprocessor.py`
- `data/raw/sample.log`
- `data/processed/sample.jsonl`

---

# Conclusion

# Conclusion

Week 2 preprocessing pipeline has been successfully implemented and validated according to the internship specification. The module now supports offline log normalization, timestamp parsing, contextual log window extraction, and generation of normalized log windows in JSONL format, providing a solid foundation for the dataset creation phase in Week 3.

All implemented functionality was verified through unit testing and end-to-end pipeline execution using a sample log file. The preprocessing module executed successfully and produced the expected normalized output.

The `coverage` package was not available in the prepared offline Python environment or offline package repository. As a result, a numerical code coverage report could not be generated. However, all implemented functionality—including the seven normalization patterns, timestamp parsing, log window extraction, file processing, and pipeline execution—was successfully validated through unit tests and functional verification within the offline environment.