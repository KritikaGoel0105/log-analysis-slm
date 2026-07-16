# Project Audit Report — AI-Powered Log Analysis Using SLMs

**Date:** 2026-07-14  
**Scope:** Full project audit against internship document v2.0  
**Status:** Weeks 1-2 complete, Week 3 partially complete (broken)

---

## 1. Document Compliance Audit

### Week 1 — Environment Setup: FULLY COMPLIANT

| Requirement | Status | Evidence |
|---|---|---|
| Python virtual environment | Compliant | `venv/` and `testenv/` directories exist |
| Offline packages downloaded | Compliant | `offline_packages/` directory populated |
| Model weights downloaded (Qwen2.5-3B-Instruct) | Compliant | `models/qwen25-3b/models--Qwen--Qwen2.5-3B-Instruct/` |
| Sentence Transformer weights (all-MiniLM-L6-v2) | Compliant | `models/sentence-transformers/all-MiniLM-L6-v2/` |
| LogHub dataset cloned | Compliant | `data/loghub/` with 16 log sources |
| 500+ raw log samples | Compliant | 16 sources × 2000 lines each = 32,000+ raw lines |
| requirements.txt | Partially compliant | Exists but versions are NOT pinned (document says "pinned versions") |
| Git initialized | Compliant | 2 commits on `main` branch |

**Issues:**

- `requirements.txt` has no pinned versions (e.g. `torch` instead of `torch==2.3.0`). The document (Section 9) specifies "All Python dependencies with pinned versions."
- `setup_offline.sh` is **missing**. The document requires "One-command offline environment setup script."
- `requirements_verified.txt` exists but is not in the document spec — not a problem, just extra.

### Week 2 — Preprocessing Pipeline: FULLY COMPLIANT

| Requirement | Status | Evidence |
|---|---|---|
| `preprocessor.py` | Compliant | `src/preprocessing/preprocessor.py` |
| All 7 normalization patterns | Compliant | IP, UUID, Timestamp, User ID, File Path, Mem Addr, Port |
| PII/secret masking | Compliant | Handled via normalization patterns |
| Structured field extraction (log windows) | Compliant | `extract_windows()` with 60s gap, 20-line max |
| `parse_timestamp()` | Compliant | Handles ISO 8601 with space and T separator |
| Unit tests | Compliant | 14 tests in `test_preprocessor.py`, all passing |
| Processed output saved as JSONL | Compliant | `save_processed_windows()` writes JSONL |
| > 95% test coverage | **Cannot verify** | `coverage` package not available in offline env |

**Issues:**

- The User ID regex was modified from the document's lookbehind pattern (`(?<=user[_\s]?)\d+`) to a word-boundary approach (`\b(user[_\s]?)(\d+)\b`) due to Python's variable-width lookbehind limitation. This is a valid and justified deviation — the Week 2 report documents this.
- Processed JSONL files exist for all 16 LogHub sources (1,890 windows total), but the timestamp normalization doesn't fire on many LogHub logs because their timestamp formats (e.g. `Dec 10 09:14:30`, `03-17 16:13:38.811`) don't match the ISO 8601 regex. This means many processed windows contain **raw timestamps that were not normalized**. This is a gap in the preprocessing — the document's regex only covers ISO 8601, but real logs use many other formats.
- The `FILE_PATH` normalization truncates some matches oddly (e.g. `<FILE_PATH>erties` from Apache logs where the path ends in `.properties`). The regex captures the path up to a 2-4 character extension, which works for `.log` but truncates longer extensions.

### Week 3 — Dataset Creation: INCORRECT IMPLEMENTATION

| Requirement | Status | Evidence |
|---|---|---|
| `logs_dataset.jsonl` | Exists but **broken** | 1,893 examples with severely flawed labels |
| 1000+ labeled examples | Partially compliant | Count is 1,893, but quality is unacceptable |
| Instruction-tuning format (instruction/input/output) | Compliant | All examples have the 3 required fields |
| Output follows SEVERITY/INCIDENT_TYPE/ROOT_CAUSE/SUMMARY/ACTIONS format | Compliant structurally | Format is correct, content is wrong |
| Train/val/test split 80/10/10 | **NOT DONE** | `train.jsonl`, `val.jsonl`, `test.jsonl` are all EMPTY (0 lines) |
| README describing sources | **NOT DONE** | `data/dataset/README.md` is empty (0 bytes) |
| `split_dataset.py` | **EMPTY** | File exists but contains no code |
| Synthetic examples | **Insufficient** | Only 3 synthetic templates (document suggests many more) |

**This is the critical failure point. See Section 3 below for full diagnosis.**

---

## 2. Repository Structure Audit

### Document-Required Structure vs Actual

```
log-analysis-slm/
├── README.md                    ⚠️  EXISTS but incomplete (59 lines, unclosed code block, only covers Week 1)
├── requirements.txt             ⚠️  EXISTS but no pinned versions
├── setup_offline.sh             ❌  MISSING (required by document Section 9)
├── data/
│   ├── raw/                     ⚠️  Only sample.log (10 lines). LogHub data is in data/loghub/ instead
│   ├── processed/               ✅  16 LogHub sources + sample.jsonl (1,890 windows)
│   ├── dataset/                 ⚠️  logs_dataset.jsonl exists but broken; splits empty; README empty
│   └── faiss.index              —   Not yet needed (Week 7)
├── models/                      ✅  Qwen2.5-3B + sentence-transformers downloaded
│   ├── qwen25-3b/               ✅
│   ├── checkpoints/             ❌  MISSING (needed Week 5-6)
│   └── sentence-transformers/   ✅
├── offline_packages/            ✅  Populated
├── src/
│   ├── preprocessing/
│   │   ├── preprocessor.py      ✅
│   │   ├── run_preprocessor.py  ✅  (extra helper, not in doc spec — fine)
│   │   └── tests/               ✅
│   ├── training/
│   │   ├── fine_tune.py         ❌  MISSING (needed Week 5)
│   │   ├── evaluate.py          ❌  MISSING (needed Week 4)
│   │   ├── build_dataset.py     ✅  (Week 3, but produces bad output)
│   │   ├── label_generator.py   ⚠️  Exists but fundamentally flawed
│   │   ├── templates.py         ✅
│   │   ├── synthetic_generator.py ⚠️  Only 3 templates
│   │   ├── split_dataset.py     ❌  EMPTY
│   │   ├── audit_labels.py      ✅  (diagnostic tool, not in spec — useful)
│   │   └── utils.py             ✅
│   ├── rag/                     —   Not yet needed (Week 7)
│   ├── api/                     —   Not yet needed (Week 9)
│   └── dashboard/               —   Not yet needed (Week 10)
├── notebooks/                   ❌  EMPTY directory
├── reports/                     ✅  Week1_Report.md + Week2_Report.md
└── docker/                      ❌  EMPTY directory (needed Week 11)
```

### Issues Found

**Unnecessary files:**
- `requirements_verified.txt` — duplicate of requirements.txt. Remove or merge.
- `data/dataset/test_utils.jsonl` — contains 1 placeholder line. Not in document spec.
- `src/dataset/` — `__pycache__` folder exists with `incident_rules.cpython-311.pyc` compiled bytecode from a file that no longer exists. This is a ghost artifact from a deleted module.
- Multiple `__pycache__/` directories in src/ — should be in `.gitignore` (they are, but they exist on disk).

**Missing files:**
- No `__init__.py` files anywhere in `src/`. The `build_dataset.py` uses relative imports (e.g. `from .templates import SYSTEM_PROMPT`), which requires either `__init__.py` files or running as `python -m src.training.build_dataset`. This is a practical issue — the code may fail if run directly.
- `setup_offline.sh` is required by the document but missing.
- `models/checkpoints/` directory doesn't exist yet.

**Structural deviation:**
- Raw LogHub logs are stored in `data/loghub/` instead of `data/raw/`. The document says `data/raw/` should contain "Original log files." The `data/raw/` directory only has the 10-line `sample.log` test file. This is a minor naming deviation — the data exists, just in a different location.

---

## 3. Week 3 Diagnosis — Root Cause of Dataset Generation Failure

### The Problem

The generated dataset (`logs_dataset.jsonl`, 1,893 examples) is **unusable for instruction-tuning** because the output labels are either generic boilerplate or factually incorrect.

### Quantitative Evidence

| Metric | Value | Expected |
|---|---|---|
| Total examples | 1,893 | 1,000-2,000 |
| "System Event" incident type | 1,526 (80.6%) | Diverse distribution |
| "Database Connection Pool Exhaustion" (misapplied) | 359 (19.0%) | Only for actual DB incidents |
| Other incident types (Auth, Disk) | 8 (0.4%) | Many categories |
| Generic "could not be determined" root causes | 1,526 (80.6%) | Specific, log-derived causes |
| Generic "should be reviewed" summaries | 1,526 (80.6%) | Specific 2-3 sentence analysis |
| Synthetic examples | 3 | Many more needed |
| Train/val/test split done | No | 80/10/10 required |

### Root Cause Analysis

**There are two distinct bugs, layered on top of a fundamental design problem:**

**Bug 1 — Legacy label generator (affects 1,526 examples, 80.6%):**
An earlier version of the label generator produced outputs with the phrase "could not be determined" for root cause and a generic summary. These examples all have `INCIDENT_TYPE: System Event` regardless of actual log content. Example:

```
INPUT:  SSH authentication failure logs (brute force attack pattern)
OUTPUT: SEVERITY: INFO
        INCIDENT_TYPE: System Event
        ROOT_CAUSE: The logs indicate a system event, but the exact root cause
                    could not be determined from the available information.
```

This is wrong on every field. SSH auth failures should be HIGH severity, Authentication Failure type, with a specific root cause.

**Bug 2 — Misapplied database labels (affects 359 examples, 19.0%):**
Another version of the label generator (or a buggy intermediate state) hard-coded "Database Connection Pool Exhaustion" as the incident type for many non-database logs. Example:

```
INPUT:  Android PowerManagerService/WindowManager logs (normal phone activity)
OUTPUT: SEVERITY: INFO
        INCIDENT_TYPE: Database Connection Pool Exhaustion
        ROOT_CAUSE: The database connection pool appears to be exhausted...
```

There is zero database activity in these Android system logs.

**Fundamental design problem — keyword-only labeling:**
The current `label_generator.py` uses simple keyword matching (e.g. if `"timeout"` in text → "Connection Timeout", if `"error"` in text → severity HIGH). This approach:
- Cannot understand log context or causal relationships
- Produces only 6 possible incident categories
- Generates identical boilerplate summaries for every example with the same severity
- Creates recommended actions that are the same templates regardless of specific log content

**The document specification on page 12 shows what outputs SHOULD look like:**
```
SEVERITY: CRITICAL
INCIDENT_TYPE: Database Connection Pool Exhaustion
ROOT_CAUSE: Database connection pool fully exhausted, likely due to slow queries or
  connection leak causing upstream API timeouts and queue overflow.
SUMMARY: The database connection pool is fully depleted, causing all API calls to fail
  with 503 errors. The request queue has exceeded its limit indicating system saturation.
RECOMMENDED_ACTIONS:
  1. Immediately restart the database connection pool manager
  2. Identify and kill long-running queries (> 30s)
  3. Increase pool size temporarily: max_connections=100
  4. Review application code for connection leak patterns
```

Notice how the reference output is **specific to the log content** — it references actual values from the logs (503, pool count, queue depth). The current generator cannot produce this kind of output.

### Affected Files

| File | Problem |
|---|---|
| `src/training/label_generator.py` | Core problem — keyword matching too simplistic, produces generic/wrong labels |
| `src/training/synthetic_generator.py` | Only 3 templates — needs 50-100+ diverse scenarios |
| `src/training/build_dataset.py` | Logic is sound, but depends on broken label_generator |
| `src/training/split_dataset.py` | Empty — split never implemented |
| `data/dataset/logs_dataset.jsonl` | Output is corrupted — needs complete regeneration |
| `data/dataset/README.md` | Empty — needs source documentation |
| `data/dataset/train.jsonl` | Empty — split not done |
| `data/dataset/val.jsonl` | Empty — split not done |
| `data/dataset/test.jsonl` | Empty — split not done |

---

## 4. Fix Strategy for Week 3

### Step 1: Expand the synthetic template library (PRIMARY fix)

**What changes:** Replace the 3 templates in `synthetic_generator.py` with 50-100+ diverse, manually-crafted instruction-tuning examples covering all major incident categories.

**Why:** The document explicitly says the dataset should come from "synthetic generation using templates, and manually curated examples." The auto-labeling approach was never going to work because generating high-quality instruction-following outputs requires understanding the semantic meaning of log patterns, which simple regex/keyword rules cannot do.

**Categories to cover (at minimum):**
- Database: connection pool exhaustion, query timeout, replication lag, deadlock
- Network: DNS failure, connection refused, TLS handshake failure, packet loss
- Disk: space exhaustion, I/O errors, read-only filesystem
- Memory: OOM killer, heap exhaustion, memory leak patterns
- Authentication: brute force, credential expiry, MFA failure, account lockout
- Application: crash/segfault, stack overflow, null pointer, unhandled exception
- Service: health check failure, cascade failure, dependency timeout
- Infrastructure: node down, container restart loop, load balancer errors
- Performance: latency spike, queue overflow, thread pool exhaustion
- Security: unauthorized access, privilege escalation, suspicious activity

Each template needs: realistic normalized log window (3-10 lines using `<TIMESTAMP>`, `<IP_ADDR>` etc.), plus a specific, detailed output with all 5 fields that references the actual log content.

**Files modified:** `src/training/synthetic_generator.py`  
**Expected outcome:** 100+ high-quality synthetic examples

### Step 2: Fix the label generator for LogHub windows

**What changes:** Replace keyword-matching `label_generator.py` with a rule-based system that uses LogHub's structured CSV data (which includes parsed templates and severity information) to generate more accurate labels.

Each LogHub source has `*_structured.csv` and `*_templates.csv` files that contain parsed log components. We can use these to create better rules specific to each log source.

For log windows where automatic labeling cannot produce specific, high-quality output, we should label them with source-appropriate categories and honest (but useful) analysis rather than pretending to know a root cause.

**Files modified:** `src/training/label_generator.py`, `src/training/build_dataset.py`  
**Expected outcome:** Higher quality labels for LogHub-derived examples

### Step 3: Implement the dataset split

**What changes:** Implement `split_dataset.py` to create 80/10/10 train/val/test splits.

**Files modified:** `src/training/split_dataset.py`  
**New files created:** Populated `train.jsonl`, `val.jsonl`, `test.jsonl`

### Step 4: Write the dataset README

**What changes:** Document all data sources, generation methodology, category distribution, and format.

**Files modified:** `data/dataset/README.md`

### Step 5: Regenerate and validate

**What changes:** Run the full pipeline: `build_dataset.py` → `split_dataset.py` → `audit_labels.py`

**Validation criteria:**
- 1,000+ total examples
- No single incident type > 30% of dataset
- No "could not be determined" or other placeholder text in any output
- All outputs reference specific content from the input logs
- Train/val/test files populated with correct 80/10/10 ratio
- README documents all sources

---

## 5. Implementation Roadmap — Weeks 4-6

### Week 4: Baseline Model Evaluation

**Objective:** Run the unmodified Qwen2.5-3B-Instruct model on the test set and measure zero-shot performance as a baseline.

**Deliverables:** Baseline accuracy report (PDF or Markdown)

**Inputs:**
- `data/dataset/test.jsonl` (from completed Week 3)
- `models/qwen25-3b/` (already downloaded)

**Outputs:**
- `reports/baseline_report.md` — zero-shot performance on all 6 metrics
- `src/training/evaluate.py` — evaluation script

**Code changes:**
- Create `src/training/evaluate.py` following the document's Section 6.2 evaluation script
- Needs `parse_model_output()` function to extract SEVERITY, INCIDENT_TYPE, etc. from model output
- Uses `rouge-score` for ROUGE-L and `sklearn` for classification metrics

**Dependencies:** torch, transformers, rouge-score, scikit-learn (all offline-compatible, already in requirements.txt — but rouge-score and scikit-learn need to be added to requirements.txt)

**Execution flow:**
1. Load Qwen2.5-3B-Instruct from `models/qwen25-3b/`
2. For each test example, format the prompt using the system prompt + input
3. Generate model output (zero-shot, no fine-tuning)
4. Parse the structured output
5. Compute: severity accuracy, incident type F1, ROUGE-L on summaries
6. Generate confusion matrices and example outputs
7. Write report

**Validation:** Report must include all metrics from Section 6.1 with actual values. Expected baseline performance will be low (model hasn't been fine-tuned yet), which is the point — it establishes the floor that fine-tuning must beat.

**Risks:**
- Model may produce unstructured output that doesn't follow the SEVERITY/INCIDENT_TYPE format → `parse_model_output()` needs robust regex fallbacks
- Inference on CPU will be slow (~30-60s per example for 3B model) → consider batching or using 4-bit quantization for inference
- Memory: 3B model needs ~6GB RAM minimum → ensure test machine has sufficient RAM

### Week 5: LoRA Fine-Tuning Setup

**Objective:** Set up the QLoRA fine-tuning pipeline and produce the first fine-tuned checkpoint.

**Deliverables:** First fine-tuned model checkpoint in `models/checkpoints/`

**Inputs:**
- `data/dataset/train.jsonl` + `data/dataset/val.jsonl`
- `models/qwen25-3b/`

**Outputs:**
- `src/training/fine_tune.py` — training script
- `models/checkpoints/checkpoint-epoch-1/` through `checkpoint-epoch-3/`
- Training loss logs

**Code changes:**
- Create `src/training/fine_tune.py` following document Section 5.4 exactly:
  - BitsAndBytesConfig: 4-bit, nf4, bfloat16, double quant
  - LoraConfig: r=16, alpha=32, dropout=0.1, target_modules=[q_proj, v_proj, k_proj, o_proj]
  - TrainingArguments: 3 epochs, batch_size=2, grad_accum=8, lr=2e-4, cosine scheduler
- Format training data into the chat template expected by Qwen2.5-Instruct
- Use HuggingFace `SFTTrainer` or manual training loop

**Dependencies:** torch, transformers, peft, accelerate, bitsandbytes (all already in requirements.txt)

**Execution flow:**
1. Load base model with 4-bit quantization
2. Apply LoRA adapters
3. Load and tokenize training dataset
4. Train for 3 epochs with validation evaluation each epoch
5. Save checkpoints after each epoch

**Validation:**
- Training loss should decrease across epochs
- Validation loss should decrease (or at least not increase significantly after epoch 2)
- Checkpoint files should exist and be loadable

**Risks:**
- bitsandbytes may not compile on all systems (especially Windows without CUDA) → fallback to fp16 without quantization
- GPU memory: QLoRA needs ~8GB VRAM for 3B model. CPU-only training is possible but very slow (~24+ hours)
- Tokenization: must match Qwen's chat template exactly or model will produce garbage

### Week 6: Fine-Tuning Iterations & Training Report

**Objective:** Iterate on hyperparameters, diagnose overfitting, produce the best model checkpoint and a training report.

**Deliverables:** Best model checkpoint + training report

**Inputs:**
- `src/training/fine_tune.py` (from Week 5)
- `src/training/evaluate.py` (from Week 4)
- Training checkpoints from Week 5

**Outputs:**
- `models/checkpoints/best/` — best performing checkpoint
- `reports/training_report.md` — hyperparameter experiments, loss curves, comparison tables
- Updated `src/training/evaluate.py` if needed

**Code changes:**
- Run evaluation on each checkpoint against val set
- Try hyperparameter variations if time permits: r=32, different learning rates, more epochs
- Compare all checkpoints and select best

**Execution flow:**
1. Evaluate each epoch checkpoint on validation set
2. Plot training/validation loss curves
3. Select best checkpoint based on validation metrics
4. Run full evaluation on test set with best checkpoint
5. Compare fine-tuned results to baseline (Week 4)
6. Write training report with all comparisons

**Validation:**
- Fine-tuned model must outperform baseline on all key metrics (document Section 10.1)
- Training report must include: loss curves, hyperparameter table, baseline vs fine-tuned comparison
- No evidence of severe overfitting (val loss should not diverge dramatically from train loss)

**Risks:**
- Overfitting on small dataset → use dropout, early stopping, and ensure diverse training data
- If model doesn't improve over baseline, the training data quality is the likely cause → go back to Week 3 and improve dataset
- Catastrophic forgetting: model may lose general language ability if over-trained on log-specific data

---

## 6. Final Deliverables Checklist (Weeks 4-6)

After completing Weeks 4-6, the following files should exist:

| File | Description | Week |
|---|---|---|
| `src/training/evaluate.py` | Evaluation script with all metrics | 4 |
| `reports/baseline_report.md` | Zero-shot performance report | 4 |
| `src/training/fine_tune.py` | QLoRA training script | 5 |
| `models/checkpoints/checkpoint-epoch-1/` | Epoch 1 checkpoint | 5 |
| `models/checkpoints/checkpoint-epoch-2/` | Epoch 2 checkpoint | 5 |
| `models/checkpoints/checkpoint-epoch-3/` | Epoch 3 checkpoint | 5 |
| `models/checkpoints/best/` | Best performing checkpoint | 6 |
| `reports/training_report.md` | Full training report with comparisons | 6 |

---

## 7. Additional Recommendations (will not change without your approval)

These are improvements I noticed that are NOT required by the document but would strengthen the project. I am listing them for your awareness only — I will not implement any of these unless you explicitly ask.

1. **Add `__init__.py` files** to `src/`, `src/preprocessing/`, `src/training/`, etc. so relative imports work reliably.

2. **Pin versions in `requirements.txt`** — run `pip freeze` in your working environment and pin all versions to prevent compatibility issues in the offline environment.

3. **Create `setup_offline.sh`** — the document requires this. It should be a simple script that creates the venv, installs from offline_packages, and verifies the installation.

4. **Fix the README.md** — it's incomplete and has an unclosed code block. Should be updated as the project progresses.

5. **Add timestamp formats** to the preprocessor regex to handle non-ISO formats (syslog-style `Jun 10 10:15:22`, Android-style `03-17 16:13:38.811`, etc.). This would improve normalization quality for LogHub data.

6. **Clean up ghost artifacts** — remove `src/dataset/__pycache__/incident_rules.cpython-311.pyc` and `data/dataset/test_utils.jsonl`.

---

*End of Audit Report*
