"""Retrieval evaluation: hit-rate@k over the Q&A pairs in data/eval.json.

A "hit" means at least one of the top-k retrieved chunks comes from the
expected source document. Reported with a pandas summary table."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS

from .vectorstore import load_vectorstore


def evaluate(store: FAISS, pairs: list[dict[str, str]], k: int = 3) -> pd.DataFrame:
    """Score retrieval with hit-rate@k over question/expected-source pairs.

    A pair counts as a "hit" when at least one of the top-k retrieved
    chunks comes from the expected source document.

    Args:
        store: FAISS vector store supporting ``similarity_search``.
        pairs: Question pairs, each with ``question`` and ``expected_source`` keys.
        k: Number of chunks retrieved per question.

    Returns:
        DataFrame with one row per pair: ``question`` (truncated to 60
        chars), ``expected``, ``retrieved`` (comma-joined sources), and
        ``hit`` (bool).
    """
    rows = []
    for pair in pairs:
        docs = store.similarity_search(pair["question"], k=k)
        sources = [d.metadata.get("source") for d in docs]
        rows.append({
            "question": pair["question"][:60],
            "expected": pair["expected_source"],
            "retrieved": ", ".join(sources),
            "hit": pair["expected_source"] in sources,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """CLI: run retrieval evaluation and print the summary table plus hit-rate@k."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectorstore", type=Path, default=Path("vectorstore"))
    ap.add_argument("--eval", type=Path, default=Path("data/eval.json"))
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    if args.k < 1:
        print(
            f"Error: --k must be at least 1 (got {args.k}): "
            "choose how many chunks to retrieve per question.",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = json.loads(args.eval.read_text(encoding="utf-8"))
    df = evaluate(load_vectorstore(args.vectorstore), pairs, k=args.k)
    print(df.to_string(index=False))
    print(f"\nhit-rate@{args.k}: {df['hit'].mean():.2f} ({int(df['hit'].sum())}/{len(df)})")


if __name__ == "__main__":
    main()
