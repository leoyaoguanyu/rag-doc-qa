"""FAISS vector store backed by HuggingFace sentence embeddings."""

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# HF model id by default; override with a local path via EMBEDDING_MODEL
# (useful where the HF Hub is unreachable, e.g. sandboxed CI).
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Build HuggingFace sentence embeddings for indexing and retrieval.

    Args:
        model_name: HF model id or local path; defaults to the
            ``EMBEDDING_MODEL`` env override, else all-MiniLM-L6-v2.

    Returns:
        A ``HuggingFaceEmbeddings`` instance ready for FAISS.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(chunks: list[Document], persist_dir: str | Path = "vectorstore") -> FAISS:
    """Embed chunks, index them in FAISS, and persist the index to disk."""
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(persist_dir))
    return store


def load_vectorstore(persist_dir: str | Path = "vectorstore") -> FAISS:
    """Load a previously persisted FAISS index."""
    return FAISS.load_local(
        str(persist_dir), get_embeddings(), allow_dangerous_deserialization=True
    )
