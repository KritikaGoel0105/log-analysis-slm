#!/usr/bin/env bash
# =====================================================================
# setup_offline.sh — Offline environment setup (document Section 9)
#
# Installs all Python dependencies from local wheels ONLY.
# No internet connection is used or required.
#
# Prerequisites (prepared while online, before air-gap transfer):
#   - Python 3.11 available on PATH as `python` (or set $PYTHON)
#   - offline_packages/  : all wheels (pip download -r requirements.txt)
#   - models/            : pre-downloaded model weights
#
# Usage (from repository root):
#   bash setup_offline.sh
# =====================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WHEEL_DIR="$REPO_ROOT/offline_packages"
VENV_DIR="$REPO_ROOT/venv"

echo "======================================================"
echo " Offline Log-Analysis SLM — Environment Setup"
echo "======================================================"

# ---- 1. Sanity checks (fail early, before touching anything) -------
if [ ! -d "$WHEEL_DIR" ]; then
    echo "ERROR: $WHEEL_DIR not found."
    echo "Copy the offline_packages/ directory into the repo root first."
    exit 1
fi

WHEEL_COUNT=$(find "$WHEEL_DIR" -name '*.whl' | wc -l)
echo "Found $WHEEL_COUNT wheels in offline_packages/"
if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo "ERROR: no .whl files in $WHEEL_DIR"
    exit 1
fi

PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python interpreter: $PYTHON ($PYVER)"
if [ "$PYVER" != "3.11" ]; then
    echo "WARNING: wheels in offline_packages/ were built for Python 3.11;"
    echo "         detected $PYVER. Binary wheels may fail to install."
fi

# ---- 2. Create virtual environment (skip if it already exists) -----
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Reusing existing virtual environment at $VENV_DIR"
fi

# Windows (Git Bash) puts python in Scripts/, Linux/macOS in bin/
if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
    VENV_PY="$VENV_DIR/Scripts/python.exe"
else
    VENV_PY="$VENV_DIR/bin/python"
fi

# ---- 3. Install from local wheels only (NO network access) ---------
echo "Installing dependencies (offline, --no-index) ..."
"$VENV_PY" -m pip install \
    --no-index \
    --find-links "$WHEEL_DIR" \
    -r "$REPO_ROOT/requirements.txt"

# ---- 4. Verify critical imports ------------------------------------
echo "Verifying installation ..."
"$VENV_PY" - << 'PYEOF'
import importlib

packages = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
    ("sentence-transformers", "sentence_transformers"),
    ("faiss-cpu", "faiss"),
    ("scikit-learn", "sklearn"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("streamlit", "streamlit"),
    ("sentencepiece", "sentencepiece"),
    ("tiktoken", "tiktoken"),
]

failed = []
for name, module in packages:
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "?")
        print(f"  OK   {name:<24} {version}")
    except Exception as exc:
        print(f"  FAIL {name:<24} {exc}")
        failed.append(name)

if failed:
    raise SystemExit(f"\nInstallation incomplete: {failed}")
print("\nAll core packages import successfully.")
PYEOF

# ---- 5. Check local model weights are present -----------------------
MODEL_DIR="$REPO_ROOT/models"
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Model directory present: $MODEL_DIR"
else
    echo "WARNING: $MODEL_DIR is missing or empty."
    echo "         Copy the pre-downloaded model weights before Week 4+."
fi

echo "======================================================"
echo " Setup complete. Activate with:"
echo "   source venv/bin/activate          (Linux/macOS)"
echo "   venv\\Scripts\\activate            (Windows)"
echo "======================================================"
