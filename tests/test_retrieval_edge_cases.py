"""Boundary tests for retrieval: oversized top-k and an empty result set.

Uses a stub store so no model downloads are needed.
"""

from langchain_core.documents import Document

from src.qa import answer


class _StubStore:
    """Minimal stand-in for a FAISS store returning a fixed hit list."""

    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, query, k=3):
        return list(self._docs)[:k]


def _docs():
    return [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "a.txt"}),
        Document(page_content="FAISS was developed by Meta.", metadata={"source": "b.txt"}),
    ]


def test_top_k_larger_than_doc_count_returns_all_docs():
    """k bigger than the collection must not error or invent docs."""
    store = _StubStore(_docs())
    docs = store.similarity_search("What is RAG?", k=99)
    assert len(docs) == 2
    assert [d.metadata["source"] for d in docs] == ["a.txt", "b.txt"]


def test_answer_handles_empty_retrieval():
    """No hits: context is empty, sources empty, and nothing crashes."""
    result = answer("unknown topic", _StubStore([]), generator=None, k=3)
    assert result["answer"] is None
    assert result["context"] == ""
    assert result["sources"] == []


def test_answer_clamps_top_k_through_pipeline():
    """Oversized k flows through answer() without padding the sources."""
    result = answer("RAG", _StubStore(_docs()), generator=None, k=100)
    assert result["sources"] == ["a.txt", "b.txt"]
    assert result["context"] != ""
