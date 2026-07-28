# AI-Powered Log Analysis Using Small Language Models (Offline / Air-Gapped)

An end-to-end, **fully offline** log analysis system: a fine-tuned
Qwen2.5-3B-Instruct small language model with local RAG turns raw
system logs into structured incident analyses (severity, incident
type, root cause, summary, recommended actions) — with **zero
internet dependencies** at runtime.

Developed as a 12-week DRDO undergraduate internship project,
strictly following the internship project document. Every week's
deliverable (D1–D10) is implemented, evaluated and reported in
`reports/`.

---

## System overview

```
raw log file
   │  Week 2  preprocessing: normalize + PII mask + window
   ▼
log windows ──► Week 7  offline RAG: FAISS top-3 similar incidents
   │                     (sentence-transformers, local index)
   ▼
Week 4-6  fine-tuned SLM (Qwen2.5-3B + QLoRA LoRA adapter)
   │        greedy decoding, chat-template prompt
   ▼
Week 4  output parser ──► structured incident analysis
   │
   ├── Week 9  REST API   (FastAPI: POST /analyze, GET /health)
   └── Week 10 Dashboard  (Streamlit: upload → live analysis → charts)
```

Full diagram: `reports/architecture_diagram.svg`.

## Results (198-example held-out test set, all offline)

| Metric | Baseline | Fine-tuned | Fine-tuned + RAG | Target |
|---|---|---|---|---|
| Severity accuracy | 43.9% | 73.7% | **80.8%** | — |
| Severity F1 (macro) | 0.31 | 0.67 | **0.81** | — |
| ROUGE-L (summaries) | 0.121 | **0.594** | 0.464 | > 0.55 |
| False positive rate | 33.3% | **8.7%** | 20.6% | < 10% |
| Root cause accuracy (human eval) | 58.6% | **89.9%** | 85.4% | > 75% |
| Parse failure rate | 8.1% | **0.0%** | 0.0% | — |
| Retrieval precision@3 | — | — | **82.0%** | > 70% |

Details, confusion matrices and per-metric discussion:
`reports/Week4_Baseline_Report.md`,
`reports/Week6_Finetuned_Evaluation_Report.md`,
`reports/Week8_Evaluation_Report.md`.

## Repository structure (document Section 9)

```
log-analysis-slm/
├── README.md                    # This file
├── requirements.txt             # Pinned dependencies
├── setup_offline.sh             # One-command offline env setup (local wheels)
├── package_offline.sh           # Week 11: build offline install archive
├── validate_offline.sh          # Week 11: no-internet validation test
├── data/                        # Processed windows, JSONL datasets, FAISS index
├── models/                      # Base weights, LoRA checkpoints, embedder (not in Git)
├── offline_packages/            # pip wheels for offline install (not in Git)
├── src/
│   ├── preprocessing/           # Week 2-3: normalization, PII masking, windowing (+tests)
│   ├── training/                # Weeks 4-6: fine_tune.py, evaluate.py, output_parser.py
│   ├── rag/                     # Week 7: build_index.py, rag_retriever.py
│   ├── api/                     # Week 9: api.py (FastAPI)
│   └── dashboard/               # Week 10: dashboard.py (Streamlit)
├── notebooks/                   # Exploration + Week 8 demo notebook
├── reports/                     # All weekly reports, metrics, presentation
└── docker/                      # Week 11: offline Dockerfile
```

## Setup (offline)

Prepared-while-online assets must already be present (wheels in
`offline_packages/`, weights in `models/`, data in `data/` — document
Section 5.1 checklist). Then, with **no internet**:

```bash
bash setup_offline.sh          # creates venv, installs from local wheels only
bash validate_offline.sh       # Week 11 no-internet validation test (must PASS)
```

## Usage

**REST API** (Week 9 — document Section 5.6):

```bash
uvicorn src.api.api:app --host 127.0.0.1 --port 8000
# POST /analyze  {"logs": "...", "top_k_context": 3, "max_tokens": 512}
# GET  /health   -> {"status": "healthy", "model": ..., "offline": true}
# Docs: http://127.0.0.1:8000/docs        Reference: reports/API_Reference.md
```

**Dashboard** (Week 10):

```bash
streamlit run src/dashboard/dashboard.py
# upload a .log/.txt file → live per-window analysis → severity chart
```

**Evaluation** (Weeks 4/6/8, reproducible):

```bash
python -m src.training.evaluate                      # baseline
python -m src.training.evaluate --adapter models/checkpoints/final-adapter
python -m src.rag.evaluate_rag                       # RAG pipeline + retrieval P@3
```

**Offline install package** (Week 11 — D9):

```bash
bash package_offline.sh        # -> dist/log-analysis-slm-offline.tar.gz + .sha256
```

## Documentation

- `reports/API_Reference.md` — full API reference (endpoints, schemas, errors)
- `reports/architecture_diagram.svg` — system architecture diagram
- `reports/Final_Presentation.pptx` — final presentation (15 slides)
- `reports/Week*_*.md` — per-week engineering reports (D1–D10 evidence)

## Offline guarantees

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` enforced in code before
  any Hugging Face import (api.py, dashboard.py, evaluation scripts).
- All artifacts load from local paths; Streamlit telemetry disabled
  (`.streamlit/config.toml`); pip runs `--no-index` from local wheels.
- Version control is **local Git only — no remote push** (document
  Section 7: "Git (local only) — no remote push").
