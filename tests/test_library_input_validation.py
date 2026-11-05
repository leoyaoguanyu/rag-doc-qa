"""Library-level input validation for answer() and evaluate().

The CLIs already reject empty questions, k < 1, and empty eval lists,
but a direct caller (notebook, another script) could still pass them
and hit the same cryptic failures the CLI guards were added for:
similarity_search("") returns arbitrary chunks, k=0 trips a bare
FAISS assertion, and an empty pairs list yields an empty DataFrame
whose hit-rate is NaN. Both functions now fail fast with ValueError.

Uses a stub store so no model downloads are needed.
"""

import pytest
from langchain_core.documents import Document

from src.evaluate import evaluate
from src.qa import answer


class _StubStore:
    """Minimal stand-in for a FAISS store returning a fixed hit list."""

    def __init__(self, docs):
        self._docs = docs
        self.calls = []

    def similarity_search(self, question, k=3):
        self.calls.append((question, k))
        return list(self._docs)[:k]


def test_answer_rejects_empty_question_before_search():
    """An empty question must not reach the retriever at all."""
    store = _StubStore([
        Document(page_content="RAG text.", metadata={"source": "a.txt"}),
    ])
    with pytest.raises(ValueError, match="non-empty"):
        answer("", store, generator=None)
    with pytest.raises(ValueError, match="non-empty"):
        answer("   ", store, generator=None)
    assert store.calls == []


def test_answer_rejects_k_below_one():
    """k=0 must raise ValueError, not a bare FAISS assertion."""
    store = _StubStore([
        Document(page_content="RAG text.", metadata={"source": "a.txt"}),
    ])
    with pytest.raises(ValueError, match="at least 1"):
        answer("what is RAG?", store, generator=None, k=0)


def test_answer_accepts_k_of_one():
    """The guard must not block the smallest valid k."""
    store = _StubStore([
        Document(page_content="RAG text.", metadata={"source": "a.txt"}),
    ])
    result = answer("what is RAG?", store, generator=None, k=1)

    assert result["sources"] == ["a.txt"]


def test_evaluate_rejects_k_below_one():
    """k=0 must raise ValueError, not a bare FAISS assertion."""
    store = _StubStore([
        Document(page_content="RAG text.", metadata={"source": "a.txt"}),
    ])
    with pytest.raises(ValueError, match="at least 1"):
        evaluate(store, [{"question": "q", "expected_source": "a.txt"}], k=0)


def test_evaluate_rejects_empty_pairs():
    """An empty pairs list must raise instead of a NaN hit-rate."""
    store = _StubStore([
        Document(page_content="RAG text.", metadata={"source": "a.txt"}),
    ])
    with pytest.raises(ValueError, match="non-empty"):
        evaluate(store, [])
    assert store.calls == []
