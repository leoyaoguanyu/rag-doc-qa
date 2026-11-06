import pytest
from langchain_core.documents import Document

from src.ingest import chunk_documents, load_documents


@pytest.mark.parametrize("chunk_size", [0, -5])
def test_chunk_documents_rejects_bad_chunk_size(chunk_size):
    docs = [Document(page_content="hello world. " * 20)]
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=10)


def test_chunk_documents_rejects_negative_overlap():
    docs = [Document(page_content="hello world. " * 20)]
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunk_documents(docs, chunk_size=100, chunk_overlap=-1)


def test_chunk_documents_rejects_overlap_not_smaller_than_size():
    docs = [Document(page_content="hello world. " * 20)]
    with pytest.raises(ValueError, match="must be smaller than chunk_size"):
        chunk_documents(docs, chunk_size=100, chunk_overlap=100)


def test_chunk_documents_accepts_valid_params():
    docs = [Document(page_content="hello world. " * 200)]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_chunking_splits_long_text(tmp_path):
    (tmp_path / "a.txt").write_text("hello world. " * 200)
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(c.page_content.strip() for c in chunks)


def test_chunking_keeps_source_metadata(tmp_path):
    (tmp_path / "notes.txt").write_text("some content here. " * 50)
    chunks = chunk_documents(load_documents(tmp_path), chunk_size=100, chunk_overlap=10)
    assert {c.metadata["source"] for c in chunks} == {"notes.txt"}


def test_load_empty_dir(tmp_path):
    assert load_documents(tmp_path) == []


def test_only_txt_files_loaded(tmp_path):
    (tmp_path / "doc.txt").write_text("text")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    assert len(load_documents(tmp_path)) == 1
