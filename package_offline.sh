#!/usr/bin/env bash
# =====================================================================
# package_offline.sh — Week 11 (D9): build the offline install archive
#
# Document basis:
#   Section 4 Week 11: "Offline deployment packaging: Docker image,
#     setup scripts, no-internet validation test" — deliverable
#     "Offline install package (.tar.gz or Docker image)".
#   Section 8 D9: "Docker image OR install archive; clean offline
#     install tested on VM with no internet".
#   Section 10.1(5): "Offline install package deploys successfully on
#     a fresh VM".
#
# Produces: dist/log-analysis-slm-offline.tar.gz
#   A single self-contained archive of everything a fresh, air-gapped
#   machine needs: source code, reports, notebooks, offline wheels,
#   model weights, data (FAISS index + datasets), setup and
#   validation scripts. Nothing in the archive requires internet.
#
# Explicitly EXCLUDED (engineering decision, stated):
#   .git/            — version control history, not needed to run
#   venv/, testenv/  — virtual envs are machine-specific; the target
#                      machine recreates one via setup_offline.sh
#   dist/            — the archive must not contain itself
#
# This script only READS the frozen Weeks 1-10 tree. It modifies
# nothing.
#
# Usage (from repository root):
#   bash package_offline.sh
# =====================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$REPO_ROOT/dist"
PKG_NAME="log-analysis-slm-offline"
ARCHIVE="$DIST_DIR/$PKG_NAME.tar.gz"

echo "======================================================"
echo " Week 11 (D9) — Offline install package builder"
echo "======================================================"

# ---- 1. Pre-flight: every offline asset must exist ------------------
echo "[1/4] Verifying offline assets (Section 5.1 checklist)..."
MISSING=0
for path in \
    README.md requirements.txt setup_offline.sh validate_offline.sh \
    offline_packages models/qwen25-3b models/checkpoints \
    models/sentence-transformers data/faiss.index \
    data/faiss_incidents.json data/dataset \
    src/preprocessing/preprocessor.py src/training/fine_tune.py \
    src/training/evaluate.py src/training/output_parser.py \
    src/rag/rag_retriever.py src/rag/build_index.py \
    src/api/api.py src/dashboard/dashboard.py \
    .streamlit/config.toml docker/Dockerfile
do
    if [ ! -e "$REPO_ROOT/$path" ]; then
        echo "  MISSING: $path"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "ERROR: required offline assets missing — aborting."
    exit 1
fi
WHEELS=$(find "$REPO_ROOT/offline_packages" -name '*.whl' | wc -l)
echo "  All required paths present ($WHEELS wheels)."

# ---- 2. Build the archive ------------------------------------------
echo "[2/4] Building $ARCHIVE (this copies ~7 GB of model weights"
echo "      and wheels; it may take several minutes)..."
mkdir -p "$DIST_DIR"
rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" \
    -C "$REPO_ROOT" \
    --exclude='./.git' \
    --exclude='./venv' \
    --exclude='./testenv' \
    --exclude='./dist' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --transform "s|^\./|$PKG_NAME/|" \
    .

# ---- 3. Integrity manifest -----------------------------------------
echo "[3/4] Writing SHA-256 checksum..."
( cd "$DIST_DIR" && sha256sum "$PKG_NAME.tar.gz" \
    > "$PKG_NAME.tar.gz.sha256" )
cat "$DIST_DIR/$PKG_NAME.tar.gz.sha256"

# ---- 4. Summary -----------------------------------------------------
SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "[4/4] Done."
echo "------------------------------------------------------"
echo " Archive : $ARCHIVE ($SIZE)"
echo " Checksum: $ARCHIVE.sha256"
echo ""
echo " Deploy on the air-gapped target machine:"
echo "   1. Copy the .tar.gz and .sha256 files (USB/secure media)."
echo "   2. sha256sum -c $PKG_NAME.tar.gz.sha256"
echo "   3. tar -xzf $PKG_NAME.tar.gz"
echo "   4. cd $PKG_NAME && bash setup_offline.sh"
echo "   5. bash validate_offline.sh   # no-internet validation test"
echo "======================================================"
