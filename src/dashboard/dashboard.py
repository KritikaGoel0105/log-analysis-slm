"""
dashboard.py

Week 10 — Streamlit dashboard (document Section 4 Week 10 row:
"Streamlit dashboard: file upload, live analysis, result
visualization — dashboard.py demo video"; Section 8 D8: "dashboard.py;
upload + analyze + display results; 2-minute demo video"; Section 9
tree: src/dashboard/dashboard.py; Section 10.1(4): "Dashboard is
operable by a non-technical user without documentation").

Design (all ADDITIVE — no Week 1-9 file modified):
  * Every pipeline stage is IMPORTED from the frozen Weeks 1-9 code —
    nothing is duplicated:
      - preprocessing:  src/preprocessing/preprocessor.py (Week 2,
                        extract_windows)
      - retrieval:      src/rag/rag_retriever.py (Week 7,
                        OfflineRAGRetriever + inject_context)
      - model loading / prompt / greedy decoding:
                        src/training/evaluate.py (Weeks 4-6,
                        load_model_and_tokenizer + build_prompt +
                        GENERATION_KWARGS; fine-tuned = base + Week 5
                        LoRA adapter, same as api.py)
      - output parsing: src/training/output_parser.py (Week 4,
                        parse_model_output)
  * Model, tokenizer, retriever and system instruction are loaded
    exactly ONCE per Streamlit server process via st.cache_resource
    (mirrors the Section 5.6 "loaded once at startup" rule that
    api.py implements with its lifespan handler).
  * "Live analysis" (Section 4 Week 10): the uploaded file is
    windowed by the frozen Week 2 extract_windows and each window is
    analyzed and RENDERED as soon as it completes, with a progress
    bar and status line — the user watches results appear.
  * "Result visualization" (Section 4 Week 10): structured per-window
    display of the five Section 5.3 output fields plus similar
    incidents, and a severity distribution bar chart when the upload
    produces more than one window.

Offline guarantees: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set
before any HF-dependent import; all artifacts load from local paths;
Streamlit telemetry is disabled via .streamlit/config.toml
(browser.gatherUsageStats = false) at the repository root.

Run (from repository root, inside the offline venv):
    streamlit run src/dashboard/dashboard.py
"""

import os
import sys
import time
from collections import Counter
from pathlib import Path

# ------------------------------------------------------------------
# Offline enforcement — MUST happen before transformers is imported
# (same rule and same mechanism as the frozen api.py).
# ------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Streamlit executes this file as a top-level script (no package
# context), so the repository root must be on sys.path for the
# frozen `src.*` imports to resolve. Additive path fix only.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

# Frozen Week 1-9 pipeline — imported, NEVER modified or duplicated.
from src.preprocessing.preprocessor import extract_windows       # Week 2
from src.rag.rag_retriever import (                              # Week 7
    DEFAULT_INDEX_FILE,
    DEFAULT_SIDECAR_FILE,
    OfflineRAGRetriever,
    inject_context,
)
from src.training.evaluate import (                              # Weeks 4-6
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEST_FILE,
    GENERATION_KWARGS,
    build_prompt,
    load_model_and_tokenizer,
)
from src.training.output_parser import parse_model_output        # Week 4

# Same generation config as the frozen Weeks 4-8 evaluations:
# GENERATION_KWARGS (evaluate.py) already fixes max_new_tokens=256,
# do_sample=False. Using the evaluation value (256) — not the API
# request default (512) — keeps dashboard outputs identical to the
# frozen evaluation pipeline and halves worst-case CPU decode time.
TOP_K_CONTEXT = 3
MAX_NEW_TOKENS = GENERATION_KWARGS["max_new_tokens"]

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
SEVERITY_BADGE = {          # engineering decision: visual cue only
    "CRITICAL": "🟥", "HIGH": "🟧", "MEDIUM": "🟨",
    "LOW": "🟩", "INFO": "🟦",
}


# ------------------------------------------------------------------
# One-time loading (st.cache_resource == "loaded once", Section 5.6
# rule applied to the dashboard, per Week 10 requirements).
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load fine-tuned model (base + Week 5 LoRA adapter), Week 7
    retriever and the training-time system instruction — once per
    server process, from local files only.

    NOTE on st.cache_resource: the cache key includes a hash of this
    function's SOURCE CODE (streamlit/runtime/caching/cache_utils.py,
    _make_function_key -> inspect.getsource). Editing this function
    while the server is running therefore invalidates the cache and
    causes exactly one extra reload on the next rerun. That is
    expected Streamlit behavior, not a bug — never edit dashboard.py
    while the server is running; restart it instead.
    """
    import json

    import torch

    # Diagnostic: this line must appear in the terminal EXACTLY ONCE
    # per server start. If it prints again without a server restart
    # (or a source-file edit), the resource cache was invalidated.
    print(f"[{time.strftime('%H:%M:%S')}] load_pipeline called "
          f"(pid {os.getpid()}) — cache miss, loading everything...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Matmul precision hints (final optimization audit). Safe here:
    # on CUDA the frozen loader uses float16 weights, and TF32 only
    # affects float32 matmuls — so generated text cannot change; any
    # residual fp32 op merely runs faster. No-op on CPU-only builds.
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_model_and_tokenizer(
        DEFAULT_MODEL_DIR, device, adapter_dir=DEFAULT_ADAPTER_DIR
    )
    retriever = OfflineRAGRetriever()
    retriever.load(DEFAULT_INDEX_FILE, DEFAULT_SIDECAR_FILE)
    with open(DEFAULT_TEST_FILE, encoding="utf-8") as fh:
        instruction = json.loads(fh.readline())["instruction"]
    return {"model": model, "tokenizer": tokenizer,
            "retriever": retriever, "instruction": instruction,
            "device": device}


def analyze_window(pipeline: dict, window_text: str) -> dict:
    """Frozen pipeline for one window: retrieve (Week 7) -> inject
    context (Week 7/8) -> generate (Weeks 4-6 greedy decoding) ->
    parse (Week 4). Identical stage order to api.py /analyze.

    Every stage is timed individually (profiling requirement); the
    timings are returned under `stage_times` and shown in the UI.
    """
    import torch

    t = {}
    start = time.time()

    t0 = time.time()
    retrieved = pipeline["retriever"].retrieve_with_metadata(
        window_text, TOP_K_CONTEXT)
    similar_incidents = [text for text, _ in retrieved]
    t["retrieval"] = time.time() - t0

    t0 = time.time()
    augmented = inject_context(window_text, retrieved)
    prompt = build_prompt(pipeline["tokenizer"],
                          pipeline["instruction"], augmented)
    t["prompt_construction"] = time.time() - t0

    t0 = time.time()
    # padding=False is the tokenizer default for a single sequence —
    # stated explicitly (final optimization audit): no pad tokens can
    # ever enter the prompt.
    inputs = pipeline["tokenizer"](prompt, return_tensors="pt",
                                   padding=False).to(
        pipeline["device"])
    t["tokenization"] = time.time() - t0

    gen_kwargs = {k: v for k, v in GENERATION_KWARGS.items()
                  if v is not None}
    gen_kwargs["max_new_tokens"] = MAX_NEW_TOKENS
    # Safe speed flags — do NOT change outputs: greedy decoding is
    # already fixed by GENERATION_KWARGS (do_sample=False); num_beams
    # defaults to 1 (no beam search); use_cache=True only reuses KV
    # states (mathematically identical logits). The four output_*
    # flags below are the HF defaults, stated explicitly so no config
    # drift can ever allocate scores/attentions/hidden-states buffers.
    gen_kwargs.setdefault("num_beams", 1)
    gen_kwargs.setdefault("use_cache", True)
    gen_kwargs.setdefault("return_dict_in_generate", False)
    gen_kwargs.setdefault("output_scores", False)
    gen_kwargs.setdefault("output_attentions", False)
    gen_kwargs.setdefault("output_hidden_states", False)
    t0 = time.time()
    with torch.inference_mode():
        output_ids = pipeline["model"].generate(
            **inputs,
            pad_token_id=pipeline["tokenizer"].eos_token_id,
            **gen_kwargs,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = pipeline["tokenizer"].decode(generated,
                                       skip_special_tokens=True)
    t["generation"] = time.time() - t0
    t["generated_tokens"] = int(generated.shape[0])
    t["tokens_per_s"] = (t["generated_tokens"] / t["generation"]
                         if t["generation"] > 0 else 0.0)

    t0 = time.time()
    parsed = parse_model_output(raw)
    t["parsing"] = time.time() - t0

    parsed["similar_incidents"] = similar_incidents
    parsed["raw_output"] = raw
    parsed["processing_time_s"] = time.time() - start
    parsed["stage_times"] = t
    return parsed


def render_result(parsed: dict) -> None:
    """Structured display of the Section 5.3 output fields."""
    severity = (parsed.get("severity") or "UNPARSED").upper()
    badge = SEVERITY_BADGE.get(severity, "⬜")
    st.markdown(f"### {badge} Severity: **{severity}**")
    st.markdown(f"**Incident Type:** "
                f"{parsed.get('incident_type') or '—'}")
    st.markdown(f"**Root Cause:** {parsed.get('root_cause') or '—'}")
    st.markdown(f"**Summary:** {parsed.get('summary') or '—'}")
    actions = parsed.get("recommended_actions") or []
    st.markdown("**Recommended Actions:**")
    if actions:
        for i, action in enumerate(actions, 1):
            st.markdown(f"{i}. {action}")
    else:
        st.markdown("—")
    similar = parsed.get("similar_incidents") or []
    with st.expander(f"Similar Incidents ({len(similar)} retrieved "
                     "from offline FAISS index)"):
        if similar:
            for i, inc in enumerate(similar, 1):
                st.text(f"[{i}] {inc}")
        else:
            st.text("No similar incidents retrieved.")
    if parsed.get("parse_errors"):
        with st.expander("⚠ Parser warnings (raw model output)"):
            st.write(parsed["parse_errors"])
            st.text(parsed.get("raw_output", ""))
    times = parsed.get("stage_times", {})
    st.caption(
        f"Analyzed in {parsed['processing_time_s']:.1f} s (offline) — "
        f"retrieval {times.get('retrieval', 0):.2f}s · "
        f"prompt {times.get('prompt_construction', 0):.3f}s · "
        f"tokenize {times.get('tokenization', 0):.3f}s · "
        f"generation {times.get('generation', 0):.1f}s "
        f"({times.get('generated_tokens', 0)} tokens, "
        f"{times.get('tokens_per_s', 0):.2f} tok/s) · "
        f"parse {times.get('parsing', 0):.3f}s"
    )


@st.cache_data(show_spinner=False)
def compute_windows(raw_text: str) -> list[list[str]]:
    """Week 2 preprocessing, cached on file content: Streamlit reruns
    the whole script on every UI interaction, and without caching the
    upload would be re-windowed each time. Pure function of the text,
    so caching cannot change results."""
    raw_lines = [l for l in raw_text.splitlines() if l.strip()]
    return extract_windows(raw_lines) if raw_lines else []


def main() -> None:
    st.set_page_config(page_title="Log Analysis Dashboard",
                       page_icon="📊", layout="wide")
    st.title("📊 AI-Powered Log Analysis — Offline Dashboard")
    st.caption(
        "Week 10 (D8) · Fine-tuned SLM + offline RAG · "
        "no internet required · model: " + DEFAULT_MODEL_NAME
    )

    with st.spinner("Loading model, adapter and FAISS index from "
                    "local files (first run only)..."):
        pipeline = load_pipeline()
    st.success(
        f"Ready — model loaded once on {pipeline['device'].upper()}, "
        f"{pipeline['retriever'].index.ntotal} incidents indexed, "
        "offline mode enforced."
    )

    uploaded = st.file_uploader(
        "Upload a log file to analyze",
        type=["log", "txt", "out", "text"],
        help="Plain-text log file. It will be normalized, split into "
             "windows and analyzed entirely on this machine.",
    )
    if uploaded is None:
        st.info("⬆ Upload a log file to begin. No data leaves this "
                "machine.")
        return

    raw_text = uploaded.getvalue().decode("utf-8", errors="replace")
    # Week 2 preprocessing — cached; same call chain as api.py.
    windows = compute_windows(raw_text)
    if not windows:
        st.error("The uploaded file contains no log lines.")
        return
    n_lines = sum(1 for l in raw_text.splitlines() if l.strip())
    st.markdown(f"**File:** `{uploaded.name}` — {n_lines} "
                f"non-empty lines → **{len(windows)} window(s)** "
                "(Week 2 preprocessing)")

    # Results survive Streamlit reruns (expander clicks etc.) via
    # session_state — nothing is ever recomputed after completion.
    result_key = f"results::{uploaded.name}::{len(raw_text)}"
    results = st.session_state.get(result_key)

    if results is None:
        if not st.button("🔍 Analyze", type="primary"):
            return
        # Live analysis: progress bar + results appearing per window.
        progress = st.progress(0.0)
        status = st.empty()
        results = []
        for w_idx, window in enumerate(windows, 1):
            status.markdown(f"⏳ Analyzing window **{w_idx} / "
                            f"{len(windows)}** ...")
            window_text = "\n".join(window)
            parsed = analyze_window(pipeline, window_text)
            results.append((window_text, parsed))
            header = (f"Window {w_idx} of {len(windows)} — "
                      f"{SEVERITY_BADGE.get((parsed.get('severity') or 'UNPARSED').upper(), '⬜')} "
                      f"{(parsed.get('severity') or 'UNPARSED').upper()}")
            with st.expander(header, expanded=(len(windows) == 1)):
                with st.expander("Input log window (normalized)"):
                    st.text(window_text)
                render_result(parsed)
            progress.progress(w_idx / len(windows))
        status.markdown(f"✅ Done — {len(windows)} window(s) "
                        "analyzed offline.")
        st.session_state[result_key] = results
    else:
        # Rerun after completion: render stored results, zero compute.
        st.markdown(f"✅ {len(results)} window(s) analyzed offline "
                    "(cached — not recomputed).")
        for w_idx, (window_text, parsed) in enumerate(results, 1):
            sev = (parsed.get("severity") or "UNPARSED").upper()
            header = (f"Window {w_idx} of {len(results)} — "
                      f"{SEVERITY_BADGE.get(sev, '⬜')} {sev}")
            with st.expander(header, expanded=(len(results) == 1)):
                with st.expander("Input log window (normalized)"):
                    st.text(window_text)
                render_result(parsed)

    severities = [(p.get("severity") or "UNPARSED").upper()
                  for _, p in results]

    # Aggregate stage timing across all windows (profiling view).
    agg: Counter = Counter()
    for _, p in results:
        for k, v in p.get("stage_times", {}).items():
            if k not in ("generated_tokens", "tokens_per_s"):
                agg[k] += v
    total = sum(p["processing_time_s"] for _, p in results)
    total_tokens = sum(p.get("stage_times", {})
                       .get("generated_tokens", 0) for _, p in results)
    gen_time = sum(p.get("stage_times", {})
                   .get("generation", 0) for _, p in results)
    with st.expander("⏱ Stage timing profile (all windows)"):
        st.markdown(
            f"- Total analysis time: **{total:.1f} s** for "
            f"{len(results)} window(s) "
            f"({total / len(results):.1f} s/window)\n"
            f"- Generation throughput: **{total_tokens} tokens in "
            f"{gen_time:.1f} s = "
            f"{(total_tokens / gen_time) if gen_time else 0:.2f} "
            "tokens/sec**\n"
            + "\n".join(
                f"- {k.replace('_', ' ')}: {v:.2f} s "
                f"({100 * v / total:.1f}%)"
                for k, v in agg.most_common()))

    # Severity distribution chart for multi-window uploads.
    if len(severities) > 1:
        st.markdown("## Severity Distribution")
        counts = Counter(severities)
        order = SEVERITY_ORDER + sorted(
            set(counts) - set(SEVERITY_ORDER))
        st.bar_chart({"windows": {s: counts.get(s, 0)
                                  for s in order if s in counts}})


main()
