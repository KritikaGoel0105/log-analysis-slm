# API Reference — Log Analysis API v1.0.0

Week 12 (D10) deliverable: "API reference" (document Section 8 D10).
Documents the frozen Week 9 REST API (`src/api/api.py`, document
Section 5.6) exactly as implemented. Interactive OpenAPI docs are
also auto-served offline at `http://127.0.0.1:8000/docs`.

## Running the API (offline)

```bash
# from repository root, inside the offline venv
uvicorn src.api.api:app --host 127.0.0.1 --port 8000
```

Startup loads everything **once** (Section 5.6: "All models and
indices are loaded once at startup from local file paths"): the
fine-tuned model (local Qwen2.5-3B-Instruct base weights + Week 5
LoRA adapter, merged), the Week 7 retriever (local embedder + FAISS
index) and the training-time system instruction. Expect several
minutes of load time on CPU. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` are set by the module itself before any
Hugging Face import — no network is ever contacted.

Device selection: `API_DEVICE` environment variable (`cpu`, `cuda`,
or default `auto` = cuda if available else cpu).

---

## POST /analyze

Runs the full frozen pipeline on raw log text:
Week 2 preprocessing → Week 7 top-k retrieval → context injection →
Weeks 4–6 greedy generation → Week 4 output parsing.

### Request body — `LogAnalysisRequest`

| Field | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `logs` | string | yes | — | non-empty after blank-line removal | Raw log text (one or more lines) |
| `top_k_context` | int | no | 3 | 0 ≤ k ≤ 20 | Number of similar incidents to retrieve; 0 disables RAG context |
| `max_tokens` | int | no | 512 | 1 ≤ n ≤ 2048 | Maximum generated tokens |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "logs": "2025-06-10 10:15:22 ERROR [db-service] Database connection timeout after 30s",
        "top_k_context": 3,
        "max_tokens": 512
      }'
```

### Response `200` — `LogAnalysisResponse`

| Field | Type | Description |
|---|---|---|
| `severity` | string | One of CRITICAL / HIGH / MEDIUM / LOW / INFO |
| `incident_type` | string | Predicted incident category |
| `root_cause` | string | Model's root-cause statement |
| `summary` | string | One-paragraph incident summary |
| `recommended_actions` | list[string] | Ordered remediation steps |
| `similar_incidents` | list[string] | Top-k retrieved incidents (empty if `top_k_context` = 0) |
| `processing_time_ms` | float | Server-side wall time for the request |

```json
{
  "severity": "HIGH",
  "incident_type": "database_outage",
  "root_cause": "Database connection pool exhausted ...",
  "summary": "The db-service experienced connection timeouts ...",
  "recommended_actions": ["Check database server health", "..."],
  "similar_incidents": ["...", "...", "..."],
  "processing_time_ms": 84211.7
}
```

### Error responses

| Status | Condition | Body `detail` |
|---|---|---|
| `400` | `logs` contains no non-blank lines | `"Empty request: 'logs' contains no log lines."` |
| `422` | Pydantic validation failure (missing `logs`, out-of-range `top_k_context` / `max_tokens`, wrong types) | FastAPI validation error object |
| `500` | Model inference raised an exception | `"Model inference failed: <error>"` |
| `500` | Model output could not be parsed into the required structure | object with `error`, `parse_errors`, `raw_output` (first 1000 chars) |
| `503` | Startup incomplete or failed (model not loaded) | `"Model not loaded (startup incomplete or failed)."` |

---

## GET /health

Section 5.6 health check, verbatim response shape.

```bash
curl -s http://127.0.0.1:8000/health
```

Response `200`:

```json
{"status": "healthy", "model": "Qwen/Qwen2.5-3B-Instruct", "offline": true}
```

Note: `/health` responds as soon as the server accepts connections;
`/analyze` returns `503` until the one-time model load completes.

---

## Determinism & performance notes

- Decoding is **greedy** (`do_sample=False`, the frozen
  `GENERATION_KWARGS` from `src/training/evaluate.py`) — identical
  input yields identical output on identical hardware/software.
- Measured CPU latency on the 198-example test set (Week 6, fine-tuned,
  max 256 new tokens): mean ≈ 75 s, p95 ≈ 192 s per window. The
  Section 6.1 "API p95 < 5 s" target is achievable only with GPU
  inference; see `reports/Week8_Evaluation_Report.md` for discussion.
- The API is single-model, in-process; concurrent requests queue on
  the Python GIL/model — deploy one worker (`uvicorn` default).

## Interactive documentation (offline)

FastAPI auto-generates OpenAPI docs served locally with no CDN
dependencies required for the JSON spec: `GET /openapi.json`,
Swagger UI at `/docs`, ReDoc at `/redoc`.
