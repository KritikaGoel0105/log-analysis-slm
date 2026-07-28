#!/usr/bin/env bash
# =====================================================================
# validate_offline.sh — Week 11 (D9): no-internet validation test
#
# Document basis:
#   Section 4 Week 11: "...no-internet validation test".
#   Section 8 D9: "clean offline install tested on VM with no
#     internet".
#   Section 10.1(1): "System runs completely offline with zero
#     internet dependencies".
#   Section 5.1 checklist (6): "Test full installation on a clean VM
#     before site deployment".
#
# Run AFTER setup_offline.sh on the target machine, with networking
# DISABLED (or on an air-gapped VM). Every check must PASS.
#
# The script only READS the frozen Weeks 1-10 code — it imports and
# exercises it exactly as the API/dashboard do, changing nothing.
#
# Usage (from repository root, networking disabled):
#   bash validate_offline.sh
# =====================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "$REPO_ROOT/venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/venv/bin/python"          # Linux/macOS venv
    elif [ -x "$REPO_ROOT/venv/Scripts/python.exe" ]; then
        PYTHON="$REPO_ROOT/venv/Scripts/python.exe"  # Windows venv (Git Bash)
    else
        PYTHON="python"
    fi
fi

PASS=0
FAIL=0
check() {  # check <name> <command...>
    local name="$1"; shift
    if "$@" >/tmp/validate_out.log 2>&1; then
        echo "  PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"
        sed 's/^/        /' /tmp/validate_out.log | tail -5
        FAIL=$((FAIL + 1))
    fi
}

echo "======================================================"
echo " Week 11 (D9) — No-internet validation test"
echo "======================================================"

# ---- 1. Confirm we are actually offline ----------------------------
echo "[1/5] Network isolation check..."
if "$PYTHON" - <<'EOF' >/dev/null 2>&1
import socket
socket.setdefaulttimeout(3)
socket.create_connection(("8.8.8.8", 53))
EOF
then
    echo "  WARN  Internet IS reachable. The D9 acceptance test"
    echo "        requires a no-internet environment. Disable"
    echo "        networking and re-run for the official result."
    NET_STATE="ONLINE (warning — rerun offline for D9 evidence)"
else
    echo "  PASS  No internet reachable (air-gapped as required)."
    NET_STATE="OFFLINE (verified)"
fi

# ---- 2. Offline assets present -------------------------------------
echo "[2/5] Offline asset presence..."
check "offline_packages/ wheels present" \
    bash -c 'ls offline_packages/*.whl >/dev/null'
check "base model weights (models/qwen25-3b)" \
    test -d models/qwen25-3b
check "LoRA adapter (models/checkpoints)" \
    test -d models/checkpoints
check "embedder (models/sentence-transformers)" \
    test -d models/sentence-transformers
check "FAISS index (data/faiss.index)" \
    test -f data/faiss.index
check "dataset (data/dataset/test.jsonl)" \
    test -f data/dataset/test.jsonl

# ---- 3. Python environment from local wheels only ------------------
echo "[3/5] Python environment (must import with HF offline mode)..."
check "torch imports" \
    "$PYTHON" -c "import torch"
check "transformers imports" \
    "$PYTHON" -c "import transformers"
check "peft imports" \
    "$PYTHON" -c "import peft"
check "faiss imports" \
    "$PYTHON" -c "import faiss"
check "sentence_transformers imports" \
    "$PYTHON" -c "import sentence_transformers"
check "fastapi imports" \
    "$PYTHON" -c "import fastapi"
check "streamlit imports" \
    "$PYTHON" -c "import streamlit"

# ---- 4. Frozen pipeline loads and runs end-to-end offline ----------
echo "[4/5] End-to-end pipeline (frozen Weeks 1-10 code, offline"
echo "      env vars enforced — this loads the 3B model and may"
echo "      take a few minutes on CPU)..."
check "full pipeline: preprocess -> retrieve -> generate -> parse" \
    "$PYTHON" - <<'EOF'
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json, sys
sys.path.insert(0, os.getcwd())

from src.preprocessing.preprocessor import extract_windows
from src.rag.rag_retriever import (DEFAULT_INDEX_FILE,
                                   DEFAULT_SIDECAR_FILE,
                                   OfflineRAGRetriever, inject_context)
from src.training.evaluate import (DEFAULT_ADAPTER_DIR,
                                   DEFAULT_MODEL_DIR,
                                   DEFAULT_TEST_FILE,
                                   GENERATION_KWARGS, build_prompt,
                                   load_model_and_tokenizer)
from src.training.output_parser import parse_model_output
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer(
    DEFAULT_MODEL_DIR, device, adapter_dir=DEFAULT_ADAPTER_DIR)
retriever = OfflineRAGRetriever()
retriever.load(DEFAULT_INDEX_FILE, DEFAULT_SIDECAR_FILE)
with open(DEFAULT_TEST_FILE, encoding="utf-8") as fh:
    first = json.loads(fh.readline())
instruction = first["instruction"]

sample = ("2025-06-10 10:15:22 ERROR [db-service] Database "
          "connection timeout after 30s")
windows = extract_windows([sample])
text = "\n".join(windows[0])
retrieved = retriever.retrieve_with_metadata(text, 3)
prompt = build_prompt(tokenizer, instruction,
                      inject_context(text, retrieved))
inputs = tokenizer(prompt, return_tensors="pt").to(device)
gen_kwargs = {k: v for k, v in GENERATION_KWARGS.items()
              if v is not None}
with torch.inference_mode():
    out = model.generate(**inputs,
                         pad_token_id=tokenizer.eos_token_id,
                         **gen_kwargs)
raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True)
parsed = parse_model_output(raw)
assert parsed["severity"], f"no severity parsed from: {raw[:200]}"
print("severity:", parsed["severity"],
      "| incidents indexed:", retriever.index.ntotal)
EOF

# ---- 5. API health check (starts uvicorn briefly) ------------------
echo "[5/5] API startup + /health (offline)..."
check "uvicorn starts and /health returns healthy" \
    "$PYTHON" - <<'EOF'
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys, threading, time, json, urllib.request
sys.path.insert(0, os.getcwd())
import uvicorn
from src.api.api import app

config = uvicorn.Config(app, host="127.0.0.1", port=8765,
                        log_level="error")
server = uvicorn.Server(config)
t = threading.Thread(target=server.run, daemon=True)
t.start()
deadline = time.time() + 600   # model load can take minutes on CPU
ok = False
while time.time() < deadline:
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:8765/health", timeout=5) as r:
            data = json.load(r)
            ok = data.get("status") == "healthy" and data.get("offline")
            break
    except Exception:
        time.sleep(5)
server.should_exit = True
assert ok, "health check did not return healthy/offline"
print("health:", data)
EOF

echo "======================================================"
echo " Network state : $NET_STATE"
echo " Result        : $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo " D9 no-internet validation: SUCCESS"
    exit 0
else
    echo " D9 no-internet validation: FAILURE — see FAIL lines above"
    exit 1
fi
