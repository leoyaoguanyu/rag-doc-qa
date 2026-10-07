from langchain_core.documents import Document

from src.ingest import chunk_documents, load_documents


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
