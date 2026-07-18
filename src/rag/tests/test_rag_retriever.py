"""
Week 7 integration tests (document Section 4: "rag_retriever.py with
integration tests").

Two tiers, so the suite is useful both on the GPU workstation (full
ML venv) and on audit machines without faiss/sentence-transformers:

  * Pure-logic tests (context injection pipeline, corpus loading,
    sidecar round-trip format, module import) — always run.
  * End-to-end integration tests (embed -> build -> persist -> load
    -> retrieve on real repository data) — run when faiss and
    sentence_transformers are importable, else skipped with a clear
    reason.

Run from repo root:
    python -m pytest src/rag/tests/ -v
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.rag.build_index import TRAIN_FILE, _field, load_corpus
from src.rag.rag_retriever import format_context, inject_context

HAS_ML_STACK = (importlib.util.find_spec("faiss") is not None
                and importlib.util.find_spec("sentence_transformers")
                is not None)


class TestContextInjection(unittest.TestCase):
    """The 'context injection pipeline' (Section 4 Week 7 focus)."""

    RETRIEVED = [
        ("disk failure on /dev/sda detected by smartd",
         {"severity": "HIGH", "incident_type": "Disk Failure"}),
        ("raid array degraded after drive dropout",
         {"severity": "HIGH", "incident_type": "RAID Degradation"}),
    ]

    def test_format_context_numbers_and_tags(self):
        block = format_context(self.RETRIEVED)
        self.assertIn("SIMILAR PAST INCIDENTS", block)
        self.assertIn("1. [HIGH — Disk Failure]", block)
        self.assertIn("2. [HIGH — RAID Degradation]", block)

    def test_format_context_empty(self):
        self.assertEqual(format_context([]), "")

    def test_format_context_truncates_long_incidents(self):
        long_text = "x" * 5000
        block = format_context([(long_text, {})],
                               max_chars_per_incident=100)
        self.assertLess(len(block), 300)

    def test_inject_context_prepends_and_preserves_input(self):
        log = "kernel: Out of memory: Kill process 1234"
        merged = inject_context(log, self.RETRIEVED)
        self.assertTrue(merged.index("SIMILAR PAST INCIDENTS")
                        < merged.index(log))
        self.assertIn("CURRENT LOG WINDOW TO ANALYZE:", merged)
        self.assertIn(log, merged)

    def test_inject_context_no_retrievals_is_identity(self):
        log = "systemd: Started Session 12 of user root."
        self.assertEqual(inject_context(log, []), log)


class TestCorpusLoading(unittest.TestCase):
    """build_index corpus rules against the real Week 3 dataset."""

    def test_field_extraction(self):
        out = ("SEVERITY: HIGH\nINCIDENT_TYPE: Disk Failure\n"
               "ROOT_CAUSE: bad sectors\n")
        self.assertEqual(_field(out, "SEVERITY"), "HIGH")
        self.assertEqual(_field(out, "INCIDENT_TYPE"), "Disk Failure")
        self.assertEqual(_field(out, "SUMMARY"), "")

    def test_corpus_is_train_split_only(self):
        texts, metadata = load_corpus()
        with open(TRAIN_FILE, encoding="utf-8") as fh:
            n_train = sum(1 for _ in fh)
        self.assertEqual(len(texts), n_train)
        self.assertEqual(len(metadata), n_train)

    def test_corpus_metadata_complete(self):
        _, metadata = load_corpus()
        missing_sev = sum(1 for m in metadata if not m["severity"])
        missing_type = sum(1 for m in metadata if not m["incident_type"])
        self.assertEqual(missing_sev, 0)
        self.assertEqual(missing_type, 0)


class TestSidecarFormat(unittest.TestCase):
    """Persistence sidecar round-trip (format-level, no faiss)."""

    def test_roundtrip(self):
        incidents = [("text a", {"severity": "INFO"}),
                     ("text b", {"severity": "HIGH"})]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sidecar.json"
            p.write_text(json.dumps(
                [{"text": t, "metadata": m} for t, m in incidents]),
                encoding="utf-8")
            back = [(r["text"], r["metadata"])
                    for r in json.loads(p.read_text(encoding="utf-8"))]
        self.assertEqual(back, incidents)


@unittest.skipUnless(HAS_ML_STACK,
                     "faiss / sentence_transformers not installed — "
                     "run in the ML venv for full integration coverage")
class TestEndToEndRetrieval(unittest.TestCase):
    """Full pipeline on real repository data (ML venv only)."""

    @classmethod
    def setUpClass(cls):
        from src.rag.rag_retriever import OfflineRAGRetriever

        texts, metadata = load_corpus()
        cls.texts, cls.metadata = texts[:200], metadata[:200]
        cls.retriever = OfflineRAGRetriever()
        cls.retriever.build_index(cls.texts, cls.metadata)

    def test_self_retrieval_top1(self):
        # An indexed document must retrieve itself first.
        hits = self.retriever.retrieve_with_metadata(self.texts[0], 3)
        self.assertEqual(hits[0][0], self.texts[0])

    def test_retrieve_returns_top_k_texts(self):
        hits = self.retriever.retrieve(self.texts[5], top_k=3)
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(isinstance(h, str) for h in hits))

    def test_persist_and_reload(self):
        from src.rag.rag_retriever import OfflineRAGRetriever

        with tempfile.TemporaryDirectory() as td:
            idx = Path(td) / "faiss.index"
            sidecar = Path(td) / "incidents.json"
            self.retriever.save(idx, sidecar)
            self.assertTrue(idx.exists())

            fresh = OfflineRAGRetriever()
            fresh.load(idx, sidecar)
            self.assertEqual(fresh.index.ntotal,
                             self.retriever.index.ntotal)
            a = self.retriever.retrieve(self.texts[3], 3)
            b = fresh.retrieve(self.texts[3], 3)
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
