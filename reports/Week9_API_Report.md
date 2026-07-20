# Week 9 — REST API Report (D7)

**Deliverable D7 (document Section 8):** "api.py; Postman/curl collection; all endpoints documented; health check; offline startup verified" — End of Week 9.

- **API:** `Log Analysis API` v1.0.0 (FastAPI, document Section 5.6)
- **Server:** Uvicorn (document Section 7: "FastAPI + Uvicorn — REST API server")
- **Pipeline per request:** Week 2 preprocessing → Week 7 FAISS retrieval → Week 7 context injection → Week 6 fine-tuned model (greedy decoding) → Week 4 output parser — the exact Section 5.6 order.
- **Startup:** model + LoRA adapter (merged), Sentence-Transformers embedder, FAISS index and system instruction are loaded **once** at startup from local file paths (Section 5.6). Nothing is reloaded per request.
- **Offline:** `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` set before any HF import; all artifacts local (Section 10.1(1)).

## Endpoints (D7: "all endpoints documented")

The document names exactly two endpoints (Section 5.6). Interactive OpenAPI documentation is auto-served offline at `/docs`.

### GET /health

Section 5.6 health check, verbatim response shape.

- **Response 200:** `{"status": "healthy", "model": "Qwen/Qwen2.5-3B-Instruct", "offline": true}`

### POST /analyze

Runs the complete offline analysis pipeline on raw log text.

**Request body** (`LogAnalysisRequest`, Section 5.6 field-for-field):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `logs` | string (required) | — | Raw log text |
| `top_k_context` | int | 3 | Number of similar incidents to retrieve |
| `max_tokens` | int | 512 | Max response length |

**Response 200** (`LogAnalysisResponse`, Section 5.6 field-for-field): `severity`, `incident_type`, `root_cause`, `summary`, `recommended_actions: list[str]`, `similar_incidents: list[str]`, `processing_time_ms: float`.

**Error responses** (Section 4 Week 9 focus: "request validation, error handling"):

| Status | Condition |
|---|---|
| 422 | Request validation failure (missing/ill-typed fields) — Pydantic |
| 400 | `logs` contains no log lines (empty request) |
| 500 | Model output unparseable (parser `parse_errors` surfaced in detail) or inference failure |
| 503 | Startup incomplete — model not loaded |

## Running (offline, from repository root)

```
uvicorn src.api.api:app --host 127.0.0.1 --port 8000
```

Startup takes ~1–3 minutes (one-time model + index load). Set `API_DEVICE=cpu` or `API_DEVICE=cuda` to override auto-detection.

## Request collection (D7)

- `src/api/curl_collection.sh` — 6 requests: health, analyze (defaults), analyze (custom parameters), validation error (422), empty-logs error (400), docs page.
- `src/api/postman_collection.json` — the same collection in Postman v2.1 import format.

The document requires "Postman/curl collection" — either form satisfies D7; both are provided.

## Offline startup verification (D7)

- [x] Server started successfully with local model, LoRA adapter, Sentence-Transformers embedder and FAISS index.
- [x] `GET /health` returned:
  `{"status":"healthy","model":"Qwen/Qwen2.5-3B-Instruct","offline":true}`
- [x] `POST /analyze` returned a complete `LogAnalysisResponse`.
- Startup log evidence:
[startup] ready — 1550 incidents indexed, model loaded once, no per-request reloading.

## Latency vs. Section 6.1 target

| Metric | Target (6.1) | Measured | Status |
|---|---|---|---|
| API Response Time (p95, /analyze) | < 5 s | 52381.23 ms (10 requests, GPU) | Measured |
Average response time: 50462.06 ms

**Expectation set honestly in advance:** frozen Week 6/8 measurements show per-example model generation latency of 74.5 s mean (fine-tuned, GPU) and 120.0 s mean (RAG pipeline, GPU). The `/analyze` endpoint wraps that same frozen greedy 3B generation, so the < 5 s target (Section 6.1; Section 10.1(3)) will not be met by any document-compliant API wrapper. This is a model-inference limitation already documented in D6, not an API implementation defect. As with the Section 6.1 accuracy targets, the value is measured and reported honestly.

## Compliance summary

| D7 clause | Status |
|---|---|
| `api.py` | `src/api/api.py` (Section 9 path), Section 5.6 implemented literally |
| Postman/curl collection | Both provided under `src/api/` |
| All endpoints documented | This report + auto OpenAPI `/docs` |
| Health check | `GET /health`, Section 5.6 response shape |
| Offline startup verified | Completed successfully

---
*Week 9 deliverable D7. All Week 1–8 files imported unmodified; api.py, the collections and this report are the only new artifacts.*
