"""Build the FAISS index: load docs -> chunk -> embed -> persist."""

import argparse
import sys
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

    if args.chunk_size < 1:
        print(
            f"Error: --chunk-size must be at least 1 (got {args.chunk_size}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.chunk_overlap < 0:
        print(
            f"Error: --chunk-overlap cannot be negative "
            f"(got {args.chunk_overlap}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.chunk_overlap >= args.chunk_size:
        print(
            f"Error: --chunk-overlap ({args.chunk_overlap}) must be smaller "
            f"than --chunk-size ({args.chunk_size}), otherwise the splitter "
            "would loop forever re-emitting the same chunk.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.data.is_dir():
        print(
            f"Error: data directory '{args.data}' does not exist. "
            "Point --data at a folder containing .txt documents.",
            file=sys.stderr,
        )
        sys.exit(1)
    docs = load_documents(args.data)
    if not docs:
        print(
            f"Error: no .txt documents found in '{args.data}'. "
            "Add at least one .txt file before building the index.",
            file=sys.stderr,
        )
        sys.exit(1)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")
    build_vectorstore(chunks, args.out)
    print(f"FAISS index saved to {args.out}")


if __name__ == "__main__":
    main()
