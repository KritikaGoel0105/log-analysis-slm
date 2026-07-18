"""
build_index.py

Week 7 — Build the offline FAISS index (document Section 9 names this
file: `src/rag/build_index.py`; Section 5.5 defines the build logic;
Section 9 places the output at `data/faiss.index`).

Corpus choice (documented, not invented): the index stores
"embeddings of historical incidents" (Section 3.2, Component 3).
The only labeled incident corpus in this project is the Week 3
dataset. Only `data/dataset/train.jsonl` is indexed — indexing val or
test examples would leak evaluation data into retrieval context and
invalidate the D5 precision@3 measurement and Week 8 evaluation.

Each indexed document is the normalized log window (`input` field),
which is exactly what will be embedded as a query at inference time
(Section 5.6: `context = retriever.retrieve(normalized, ...)`), so
index-side and query-side text are the same distribution. Metadata
(severity, incident_type) comes from the reference output and is used
by the D5 relevance judgment.

Usage (offline, from repo root):
    python -m src.rag.build_index
"""

import json
from pathlib import Path

from .rag_retriever import (
    DEFAULT_INDEX_FILE,
    DEFAULT_SIDECAR_FILE,
    OfflineRAGRetriever,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = REPO_ROOT / "data" / "dataset" / "train.jsonl"


def _field(output: str, name: str) -> str:
    """Extract 'NAME: value' from a reference output block."""
    for line in output.splitlines():
        if line.startswith(f"{name}:"):
            return line.split(":", 1)[1].strip()
    return ""


def load_corpus(train_file: Path = TRAIN_FILE):
    """Return (texts, metadata) from the Week 3 training split."""
    texts, metadata = [], []
    with open(train_file, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            texts.append(rec["input"])
            metadata.append({
                "train_index": i,
                "severity": _field(rec["output"], "SEVERITY"),
                "incident_type": _field(rec["output"], "INCIDENT_TYPE"),
            })
    return texts, metadata


def main() -> int:
    print("=" * 60)
    print("Week 7 — FAISS index builder (offline)")
    print("=" * 60)

    if not TRAIN_FILE.exists():
        print(f"ERROR: {TRAIN_FILE} not found (Week 3 dataset required).")
        return 1

    texts, metadata = load_corpus()
    print(f"Corpus: {len(texts)} training incidents "
          f"(train.jsonl only — val/test excluded to avoid leakage)")

    retriever = OfflineRAGRetriever()
    retriever.build_index(texts, metadata)
    retriever.save(DEFAULT_INDEX_FILE, DEFAULT_SIDECAR_FILE)

    print(f"Index:   {DEFAULT_INDEX_FILE} "
          f"({retriever.index.ntotal} vectors, dim "
          f"{retriever.index.d})")
    print(f"Sidecar: {DEFAULT_SIDECAR_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
