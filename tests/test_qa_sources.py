"""answer() must report chunks without source metadata as "unknown".

A chunk built without a "source" entry used to land as None in the
sources list, so the CLI printed "Sources: [None]". Mirrors the
evaluate() fix: label such chunks "unknown", never None.

Uses a stub store so no model downloads are needed.
"""

from langchain_core.documents import Document

from src.qa import answer


class _StubStore:
    """Minimal stand-in for a FAISS store returning a fixed hit list."""

    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, question, k=3):
        return list(self._docs)[:k]


def test_chunk_without_source_metadata_reports_unknown_not_none():
    """A sourceless chunk must show up as "unknown", never None."""
    store = _StubStore([
        Document(page_content="Some text.", metadata={}),
    ])
    result = answer("q", store, generator=None)

    assert result["sources"] == ["unknown"]
    assert None not in result["sources"]


def test_sourceless_chunk_does_not_hide_a_real_source():
    """Mixed hits: a real source still shows beside an unknown one."""
    store = _StubStore([
        Document(page_content="RAG text.", metadata={"source": "a.txt"}),
        Document(page_content="Other text.", metadata={}),
    ])
    result = answer("q", store, generator=None)

    assert result["sources"] == ["a.txt", "unknown"]
