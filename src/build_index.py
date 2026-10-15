"""Build the FAISS index: load docs -> chunk -> embed -> persist."""

import argparse
from pathlib import Path

from .ingest import chunk_documents, load_documents
from .vectorstore import build_vectorstore


def main() -> None:
    """CLI: load docs -> chunk -> embed -> persist the FAISS index to disk."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/sample_docs"))
    ap.add_argument("--out", type=Path, default=Path("vectorstore"))
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    args = ap.parse_args()

    docs = load_documents(args.data)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")
    build_vectorstore(chunks, args.out)
    print(f"FAISS index saved to {args.out}")


if __name__ == "__main__":
    main()
