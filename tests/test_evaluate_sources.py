"""evaluate() must survive chunks that carry no source metadata.

A chunk built without a "source" entry used to crash evaluate() with a
TypeError inside ", ".join(sources); now it is reported as "unknown"
and simply does not count as a hit.

Uses a stub store so no model downloads are needed.
"""

from langchain_core.documents import Document

from src.evaluate import evaluate


class _StubStore:
    """Minimal stand-in for a FAISS store returning a fixed hit list."""

    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, question, k=3):
        return list(self._docs)[:k]


def test_chunk_without_source_metadata_reports_unknown_not_crash():
    """A sourceless chunk must not crash the summary table."""
    store = _StubStore([
        Document(page_content="Some text.", metadata={}),
    ])
    df = evaluate(store, [{"question": "q", "expected_source": "a.txt"}])

    assert len(df) == 1
    assert df.iloc[0]["retrieved"] == "unknown"
    assert df.iloc[0]["hit"] == False  # pandas stores bools as numpy bool_


def test_sourceless_chunk_does_not_block_a_real_hit():
    """Mixed hits: a real source still counts even beside an unknown one."""
    store = _StubStore([
        Document(page_content="RAG text.", metadata={"source": "a.txt"}),
        Document(page_content="Other text.", metadata={}),
    ])
    df = evaluate(store, [{"question": "q", "expected_source": "a.txt"}])

    assert df.iloc[0]["retrieved"] == "a.txt, unknown"
    assert df.iloc[0]["hit"] == True  # pandas stores bools as numpy bool_
