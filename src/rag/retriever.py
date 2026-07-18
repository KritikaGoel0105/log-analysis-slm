"""
retriever.py — Section 9 structure alias.

The document names this file in two ways:
  * Section 9 repository structure: `src/rag/retriever.py`
  * Section 4 Week 7 deliverable and Section 8 D5: `rag_retriever.py`

The implementation lives in rag_retriever.py (the deliverable name);
this module re-exports it so both documented paths resolve.
"""

from .rag_retriever import (  # noqa: F401
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_FILE,
    DEFAULT_SIDECAR_FILE,
    OfflineRAGRetriever,
    format_context,
    inject_context,
)
