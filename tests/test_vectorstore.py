"""FAISS round-trip tests with a fake embedding model (no downloads)."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), float(len(t) % 5), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), float(len(text) % 5), 1.0]


def _docs():
    return [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "a.txt"}),
        Document(page_content="FAISS was developed by Meta.", metadata={"source": "b.txt"}),
    ]


def test_faiss_save_load_roundtrip(tmp_path):
    store = FAISS.from_documents(_docs(), FakeEmbeddings())
    store.save_local(str(tmp_path))
    loaded = FAISS.load_local(str(tmp_path), FakeEmbeddings(), allow_dangerous_deserialization=True)
    assert loaded.index.ntotal == 2


def test_similarity_search_returns_documents():
    store = FAISS.from_documents(_docs(), FakeEmbeddings())
    results = store.similarity_search("What is RAG?", k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] in {"a.txt", "b.txt"}
