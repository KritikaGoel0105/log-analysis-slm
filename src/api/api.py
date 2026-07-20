"""
api.py

Week 9 — FastAPI REST API (document Section 5.6: "REST API Design
(Week 9)"; Section 4 Week 9 row: "FastAPI REST API: all endpoints,
request validation, error handling, offline packaging"; Section 8 D7:
"api.py; Postman/curl collection; all endpoints documented; health
check; offline startup verified").

Design (all ADDITIVE — no Week 1-8 file modified):
  * Section 5.6 is implemented literally: `FastAPI(title='Log
    Analysis API', version='1.0.0')`, `LogAnalysisRequest` /
    `LogAnalysisResponse` with the exact fields and defaults from the
    document, `POST /analyze` running preprocess -> retrieve ->
    generate -> parse, and `GET /health` returning
    {'status': 'healthy', 'model': MODEL_NAME, 'offline': True}.
  * Section 5.6: "All models and indices are loaded once at startup
    from local file paths" — the fine-tuned model (base + Week 5
    LoRA adapter, merged), the Week 7 retriever (FAISS index +
    sidecar + local embedder) and the system instruction are loaded
    exactly once in the FastAPI lifespan handler. Nothing is
    reloaded per request.
  * All pipeline stages are IMPORTED from the frozen Weeks 1-8 code:
      - preprocessing:  src/preprocessing/preprocessor.py (Week 2)
      - retrieval:      src/rag/rag_retriever.py (Week 7)
      - context:        inject_context (Week 7, used by Week 8)
      - model loading / prompt / greedy decoding:
                        src/training/evaluate.py (Weeks 4-6)
      - output parsing: src/training/output_parser.py (Week 4)
    The Section 5.6 skeleton's illustrative names
    (`preprocessor.process`, `model.generate(..., context=...)`) are
    realized by these frozen functions; the adaptation lives entirely
    in this file.
  * `parse_model_output` returns a `parse_errors` key that the
    Section 5.6 response model does not include; it is consumed HERE
    (error handling, Section 4 Week 9 focus) and never leaks into
    the response. The parser itself is untouched.

Offline guarantees: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set
before any HF-dependent import; every artifact loads from a local
path (Section 10.1: "System runs completely offline with zero
internet dependencies").

Run (from repository root, inside the offline venv):
    uvicorn src.api.api:app --host 127.0.0.1 --port 8000
Interactive endpoint documentation (D7: "all endpoints documented"):
    http://127.0.0.1:8000/docs   (auto-generated OpenAPI, offline)
"""

import json
import os
import time
from contextlib import asynccontextmanager

# ------------------------------------------------------------------
# Offline enforcement — MUST happen before transformers is imported
# (document Section 2/6: no internet-dependent runtime components).
# ------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Frozen Week 1-8 pipeline — imported, NEVER modified or duplicated.
from ..preprocessing.preprocessor import extract_windows        # Week 2
from ..rag.rag_retriever import (                                # Week 7
    DEFAULT_INDEX_FILE,
    DEFAULT_SIDECAR_FILE,
    OfflineRAGRetriever,
    inject_context,
)
from ..training.evaluate import (                                # Weeks 4-6
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEST_FILE,
    GENERATION_KWARGS,
    build_prompt,
    load_model_and_tokenizer,
)
from ..training.output_parser import parse_model_output          # Week 4

MODEL_NAME = DEFAULT_MODEL_NAME  # Section 5.6 /health uses MODEL_NAME

# Device: cuda if available, else cpu. The document does not specify
# the device (engineering decision, not document requirement);
# Section 10.1(3) references CPU operation, so CPU must work.
API_DEVICE = os.environ.get("API_DEVICE", "auto")

# Loaded ONCE at startup (Section 5.6) — populated by the lifespan
# handler below.
STATE: dict = {"model": None, "tokenizer": None, "retriever": None,
               "instruction": None, "device": None}


def _resolve_device() -> str:
    if API_DEVICE in ("cpu", "cuda"):
        return API_DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_instruction() -> str:
    """The system instruction the model was trained/evaluated with
    (dataset `instruction` field — same source as the Week 8 demo)."""
    with open(DEFAULT_TEST_FILE, encoding="utf-8") as fh:
        return json.loads(fh.readline())["instruction"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Section 5.6: 'All models and indices are loaded once at
    startup from local file paths.'"""
    device = _resolve_device()
    print(f"[startup] loading fine-tuned model on {device} "
          "(offline, local weights + Week 5 adapter)...")
    model, tokenizer = load_model_and_tokenizer(
        DEFAULT_MODEL_DIR, device, adapter_dir=DEFAULT_ADAPTER_DIR
    )
    print("[startup] loading Week 7 retriever "
          "(local embedder + FAISS index)...")
    retriever = OfflineRAGRetriever()
    retriever.load(DEFAULT_INDEX_FILE, DEFAULT_SIDECAR_FILE)
    STATE.update(model=model, tokenizer=tokenizer,
                 retriever=retriever,
                 instruction=_load_instruction(),
                 device=device)
    print(f"[startup] ready — {retriever.index.ntotal} incidents "
          "indexed, model loaded once, no per-request reloading.")
    yield
    STATE.update(model=None, tokenizer=None, retriever=None)


app = FastAPI(title="Log Analysis API", version="1.0.0",
              lifespan=lifespan)


# ------------------------------------------------------------------
# Request / response models — Section 5.6, field-for-field.
# ------------------------------------------------------------------

class LogAnalysisRequest(BaseModel):
    logs: str                       # Raw log text
    top_k_context: int = Field(default=3, ge=0, le=20)  # similar incidents
    max_tokens: int = Field(default=512, ge=1, le=2048)  # max response length


class LogAnalysisResponse(BaseModel):
    severity: str
    incident_type: str
    root_cause: str
    summary: str
    recommended_actions: list[str]
    similar_incidents: list[str]
    processing_time_ms: float


# ------------------------------------------------------------------
# Endpoints — exactly the two named by the document.
# ------------------------------------------------------------------

@app.post("/analyze", response_model=LogAnalysisResponse)
async def analyze_logs(request: LogAnalysisRequest):
    """Section 5.6 pipeline: preprocess -> retrieve -> generate ->
    parse. Section 4 Week 9 focus: request validation (Pydantic,
    above) + error handling (HTTPException paths, below)."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded (startup "
                                   "incomplete or failed).")

    raw_lines = [l for l in request.logs.splitlines() if l.strip()]
    if not raw_lines:
        raise HTTPException(status_code=400,
                            detail="Empty request: 'logs' contains "
                                   "no log lines.")

    start = time.time()

    # 1. Week 2 preprocessing: normalize + window the raw log text.
    windows = extract_windows(raw_lines)
    normalized = "\n".join(line for w in windows for line in w)

    # 2. Week 7 retrieval (Section 5.6:
    #    retriever.retrieve(normalized, top_k=request.top_k_context)).
    if request.top_k_context > 0:
        retrieved = STATE["retriever"].retrieve_with_metadata(
            normalized, request.top_k_context)
    else:
        retrieved = []
    similar_incidents = [text for text, _ in retrieved]

    # 3. Week 7/8 context injection + Weeks 4-6 generation
    #    (Section 5.6: model.generate(normalized, context=...,
    #    max_tokens=request.max_tokens) — same greedy decoding as
    #    every frozen evaluation).
    import torch

    augmented = inject_context(normalized, retrieved)
    prompt = build_prompt(STATE["tokenizer"], STATE["instruction"],
                          augmented)
    inputs = STATE["tokenizer"](prompt, return_tensors="pt").to(
        STATE["device"])
    gen_kwargs = {k: v for k, v in GENERATION_KWARGS.items()
                  if v is not None}
    gen_kwargs["max_new_tokens"] = request.max_tokens
    try:
        with torch.no_grad():
            output_ids = STATE["model"].generate(
                **inputs,
                pad_token_id=STATE["tokenizer"].eos_token_id,
                **gen_kwargs,
            )
    except Exception as exc:  # internal error path (Week 9 focus)
        raise HTTPException(status_code=500,
                            detail=f"Model inference failed: {exc}")
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    result = STATE["tokenizer"].decode(generated,
                                       skip_special_tokens=True)

    # 4. Week 4 parser (Section 5.6: parse_model_output(result)).
    #    `parse_errors` is handled HERE — the parser is untouched and
    #    the Section 5.6 response schema is preserved exactly.
    parsed = parse_model_output(result)
    if parsed["parse_errors"] or not parsed["severity"]:
        raise HTTPException(
            status_code=500,
            detail={"error": "Model output could not be parsed into "
                             "the required analysis format.",
                    "parse_errors": parsed["parse_errors"],
                    "raw_output": result[:1000]})

    return LogAnalysisResponse(
        severity=parsed["severity"],
        incident_type=parsed["incident_type"] or "",
        root_cause=parsed["root_cause"] or "",
        summary=parsed["summary"] or "",
        recommended_actions=parsed["recommended_actions"] or [],
        similar_incidents=similar_incidents,
        processing_time_ms=(time.time() - start) * 1000,
    )


@app.get("/health")
async def health_check():
    """Section 5.6 health check, verbatim response shape."""
    return {"status": "healthy", "model": MODEL_NAME, "offline": True}
