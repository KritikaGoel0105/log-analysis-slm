"""
rag_retriever.py

Week 7 — Offline RAG retriever (document Section 5.5).

Implements the `OfflineRAGRetriever` skeleton from the internship
document verbatim in behaviour:

  * Sentence Transformers embedder loaded from a LOCAL path
    (`models/sentence-transformers/all-MiniLM-L6-v2`) — Section 5.5:
    "The entire RAG system runs offline using FAISS for vector
    storage and Sentence Transformers for embedding — no internet
    required."
  * Cosine similarity via `faiss.normalize_L2` + `IndexFlatIP`
    (Section 5.5 code listing).
  * Index persisted to `data/faiss.index` (Sections 5.5 and 9).
  * `retrieve(query, top_k=3)` returns the stored incident texts.

Additions beyond the skeleton (each justified):
  * `(text, metadata)` incident tuples — declared by the document's
    own comment in Section 5.5 ("List of (text, metadata) tuples");
    metadata carries severity/incident_type needed for the D5
    precision@3 measurement and Week 8/9 context display.
  * `save()` / `load()` — Section 5.5 persists the index with
    `faiss.write_index`; Section 5.6 (Week 9 API) requires "All
    models and indices are loaded once at startup from local file
    paths", so a load path is mandatory downstream. FAISS stores
    vectors only, therefore incident texts/metadata are persisted in
    a JSON sidecar next to the index.
  * `format_context()` / `inject_context()` — the "context injection
    pipeline" named in the Section 4 Week 7 focus, and the `context`
    argument consumed by the Week 9 API skeleton (Section 5.6).

faiss / sentence_transformers are imported lazily inside methods so
that this module (and its pure-logic tests) import cleanly on
machines without the ML stack.
"""

import json
import os
from pathlib import Path

# Offline enforcement BEFORE any HF-dependent import (Section 2:
# zero internet dependencies at runtime).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Section 5.5 / Section 9 canonical paths
DEFAULT_EMBEDDING_MODEL = (
    REPO_ROOT / "models" / "sentence-transformers" / "all-MiniLM-L6-v2"
)
DEFAULT_INDEX_FILE = REPO_ROOT / "data" / "faiss.index"
# FAISS persists vectors only; incident texts + metadata live here.
DEFAULT_SIDECAR_FILE = REPO_ROOT / "data" / "faiss_incidents.json"


class OfflineRAGRetriever:
    """Offline FAISS + Sentence Transformers retriever (Section 5.5)."""

    def __init__(self, model_path: str | Path = DEFAULT_EMBEDDING_MODEL):
        # Load embedding model from local path (downloaded offline)
        from sentence_transformers import SentenceTransformer

        self.model_path = str(model_path)
        self.embedder = SentenceTransformer(self.model_path)
        self.index = None
        self.incidents: list[tuple[str, dict]] = []  # (text, metadata)

    # -- build ------------------------------------------------------

    def build_index(self, incident_texts: list[str],
                    metadata: list[dict] | None = None) -> None:
        """Embed incident texts and build a cosine-similarity index."""
        import faiss
        import numpy as np

        if metadata is None:
            metadata = [{} for _ in incident_texts]
        if len(metadata) != len(incident_texts):
            raise ValueError("metadata length must match incident_texts")

        embeddings = self.embedder.encode(incident_texts,
                                          show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)  # For cosine similarity
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.incidents = list(zip(incident_texts, metadata))

    # -- persistence (Sections 5.5 + 5.6) ---------------------------

    def save(self, index_path: Path = DEFAULT_INDEX_FILE,
             sidecar_path: Path = DEFAULT_SIDECAR_FILE) -> None:
        import faiss

        if self.index is None:
            raise RuntimeError("No index built — call build_index() first")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))  # Persist locally
        sidecar = [{"text": t, "metadata": m} for t, m in self.incidents]
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")

    def load(self, index_path: Path = DEFAULT_INDEX_FILE,
             sidecar_path: Path = DEFAULT_SIDECAR_FILE) -> None:
        import faiss

        self.index = faiss.read_index(str(index_path))
        records = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.incidents = [(r["text"], r["metadata"]) for r in records]
        if self.index.ntotal != len(self.incidents):
            raise ValueError(
                f"Index/sidecar mismatch: {self.index.ntotal} vectors "
                f"vs {len(self.incidents)} incident records")

    # -- retrieval --------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Top-k most similar incident texts (Section 5.5 signature)."""
        return [t for t, _ in self.retrieve_with_metadata(query, top_k)]

    def retrieve_with_metadata(self, query: str,
                               top_k: int = 3) -> list[tuple[str, dict]]:
        """Top-k (text, metadata) tuples — used by the D5 precision@3
        evaluation and by Week 8/9 context display."""
        import faiss
        import numpy as np

        if self.index is None:
            raise RuntimeError("No index loaded — build_index() or load()")
        query_emb = np.array(
            self.embedder.encode([query])).astype("float32")
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, top_k)
        return [self.incidents[i] for i in indices[0] if i != -1]


# ------------------------------------------------------------------
# Context injection pipeline (Section 4 Week 7 focus; consumed as the
# `context` argument by the Week 9 API skeleton in Section 5.6).
# Pure functions — no faiss/ST needed, unit-testable anywhere.
# ------------------------------------------------------------------

def format_context(retrieved: list[tuple[str, dict]],
                   max_chars_per_incident: int = 400) -> str:
    """Render retrieved incidents as a numbered context block."""
    if not retrieved:
        return ""
    lines = ["SIMILAR PAST INCIDENTS (retrieved offline via FAISS):"]
    for n, (text, meta) in enumerate(retrieved, 1):
        tag = ""
        if meta.get("incident_type") or meta.get("severity"):
            tag = (f" [{meta.get('severity', '?')} — "
                   f"{meta.get('incident_type', '?')}]")
        snippet = " ".join(text.split())[:max_chars_per_incident]
        lines.append(f"{n}.{tag} {snippet}")
    return "\n".join(lines)


def inject_context(log_input: str,
                   retrieved: list[tuple[str, dict]]) -> str:
    """Prepend retrieved-incident context to the normalized log input.

    Output feeds the existing prompt builder unchanged, so Week 8 can
    pass it where the plain `input` went — no Week 4-6 code modified.
    """
    block = format_context(retrieved)
    if not block:
        return log_input
    return f"{block}\n\nCURRENT LOG WINDOW TO ANALYZE:\n{log_input}"
