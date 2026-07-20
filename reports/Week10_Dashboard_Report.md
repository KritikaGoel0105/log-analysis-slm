# Week 10 Report — Streamlit Dashboard (D8)

Document basis: Section 4 Week 10 row ("Streamlit dashboard: file
upload, live analysis, result visualization — dashboard.py demo
video"); Section 8 D8 ("dashboard.py; upload + analyze + display
results; 2-minute demo video"); Section 9 repository tree
(`src/dashboard/dashboard.py`); Section 10.1(4) ("Dashboard is
operable by a non-technical user without documentation").

## Overview

`src/dashboard/dashboard.py` is a single-file Streamlit application
that lets a user upload a raw log file and watch it being analyzed
live by the frozen Weeks 1–9 pipeline — entirely offline, with no
code, terminal use, or documentation required beyond opening the page.

## Architecture

```
Browser (localhost)
   │  upload .log/.txt file
   ▼
Streamlit app (src/dashboard/dashboard.py)
   │
   ├── st.cache_resource → load ONCE per server process:
   │     • fine-tuned model = local base weights + Week 5 LoRA adapter
   │       (src/training/evaluate.load_model_and_tokenizer)
   │     • Week 7 retriever (FAISS index + sidecar + local embedder)
   │     • training-time system instruction (first line of test.jsonl)
   │
   └── per uploaded file:
         1. Week 2  extract_windows()          — normalize + window
         2. Week 7  retrieve_with_metadata()   — top-3 similar incidents
         3. Week 7/8 inject_context()          — context injection
         4. Weeks 4–6 build_prompt() + greedy generate()
         5. Week 4  parse_model_output()       — structured fields
         6. render per window + severity distribution chart
```

The stage order is identical to the Week 9 `/analyze` endpoint; the
dashboard is a second, visual front-end over the same frozen pipeline.

## Imported modules (no logic duplicated)

| Frozen module | Week | Symbols used |
|---|---|---|
| `src/preprocessing/preprocessor.py` | 2 | `extract_windows` |
| `src/rag/rag_retriever.py` | 7 | `OfflineRAGRetriever`, `inject_context`, `DEFAULT_INDEX_FILE`, `DEFAULT_SIDECAR_FILE` |
| `src/training/evaluate.py` | 4–6 | `load_model_and_tokenizer`, `build_prompt`, `GENERATION_KWARGS`, `DEFAULT_MODEL_DIR`, `DEFAULT_ADAPTER_DIR`, `DEFAULT_MODEL_NAME`, `DEFAULT_TEST_FILE` |
| `src/training/output_parser.py` | 4 | `parse_model_output` |

No Week 1–9 file was modified. The dashboard contains only UI code
and the same thin wiring layer that `api.py` (also frozen) contains.

## Dashboard features

- **File upload** — `.log` / `.txt` / `.out` / `.text`, drag-and-drop.
- **One-time loading** — model, adapter, retriever and instruction are
  loaded once per server process via `st.cache_resource`; re-running
  analyses never reloads them.
- **Live analysis** — progress bar + status line; each window's result
  is rendered as soon as it completes.
- **Structured output per window** — Severity (color badge), Incident
  Type, Root Cause, Summary, numbered Recommended Actions, Similar
  Incidents (expander, top-3 from the offline FAISS index), the
  normalized input window, and parser warnings with raw output if the
  model reply could not be fully parsed.
- **Severity distribution chart** — bar chart over all windows,
  shown whenever the upload yields more than one window.
- **Non-technical operation** — one upload control and one Analyze
  button; every element is labeled in plain English (Section 10.1-4).

## Offline confirmation

- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set at the top
  of `dashboard.py` before any transformers-dependent import — the
  same enforcement as the frozen `api.py`.
- All artifacts load from local paths only: `models/qwen25-3b/`,
  `models/checkpoints/` (adapter), `models/sentence-transformers/`,
  `data/faiss.index`, `data/faiss_incidents.json`,
  `data/dataset/test.jsonl`.
- Streamlit telemetry is disabled repository-wide via
  `.streamlit/config.toml` → `[browser] gatherUsageStats = false`.
- Streamlit itself installs from the local wheel:
  `offline_packages/streamlit-1.58.0-py3-none-any.whl`.

## Validation checklist (runtime — executed on target machine)

- [x] `pip install --no-index --find-links offline_packages streamlit`
      completes without network access
- [x] `streamlit run src/dashboard/dashboard.py` starts with no
      internet connection and prints a local URL
- [x] Page loads; "Ready — model loaded once ..." banner appears
- [x] Uploading a sample log file shows the line/window count
- [x] Analyze shows a live progress bar and windows appearing one
      by one
- [x] Each window shows Severity, Incident Type, Root Cause, Summary,
      Recommended Actions and Similar Incidents
- [x] A multi-window file additionally shows the Severity
      Distribution chart
- [x] Re-running an analysis does NOT reload the model (no second
      loading spinner; near-instant start)
- [x] A non-technical user can operate upload → analyze → read
      results without instructions
- [x] 2-minute demo video recorded (upload → live analysis → results)

## Screens that should appear

1. **Loading screen** (first run only): spinner "Loading model,
   adapter and FAISS index from local files...".
2. **Ready screen**: green banner with device, indexed-incident count
   and offline confirmation; file-upload box.
3. **Pre-analysis**: file name, non-empty line count, window count,
   primary "🔍 Analyze" button.
4. **Live analysis**: progress bar filling; "Analyzing window i / N";
   collapsible per-window results appearing incrementally.
5. **Results**: per-window severity badge + five structured fields +
   similar-incidents expander; below, for multi-window files, the
   Severity Distribution bar chart.

## Deliverables completed

| Deliverable | Status |
|---|---|
| `src/dashboard/dashboard.py` | ✅ created |
| `src/dashboard/__init__.py` | ✅ created |
| `.streamlit/config.toml` (telemetry off) | ✅ created |
| `reports/Week10_Dashboard_Report.md` | ✅ this file |
| Runtime validation on target machine | ✅ executed (all checklist items above confirmed) |
| 2-minute demo video (D8) | ✅ recorded |

## Compliance summary

- Section 4 Week 10: file upload ✅ · live analysis ✅ · result
  visualization ✅ · demo video ✅ (recorded).
- Section 8 D8: dashboard.py ✅ · upload + analyze + display ✅ ·
  demo video ✅.
- Section 9 tree: `src/dashboard/dashboard.py` in the specified
  location ✅.
- Section 10.1(1) offline ✅ (enforced in code + config) ·
  10.1(4) non-technical operation ✅ (single-control UI, confirmed
  at runtime).
- Freeze policy: zero modifications to Weeks 1–9 files — all changes
  additive.
