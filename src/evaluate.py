"""Retrieval evaluation: hit-rate@k over the Q&A pairs in data/eval.json.

A "hit" means at least one of the top-k retrieved chunks comes from the
expected source document. Reported with a pandas summary table."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .vectorstore import load_vectorstore


def evaluate(store, pairs: list[dict], k: int = 3) -> pd.DataFrame:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectorstore", type=Path, default=Path("vectorstore"))
    ap.add_argument("--eval", type=Path, default=Path("data/eval.json"))
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    pairs = json.loads(args.eval.read_text(encoding="utf-8"))
    df = evaluate(load_vectorstore(args.vectorstore), pairs, k=args.k)
    print(df.to_string(index=False))
    print(f"\nhit-rate@{args.k}: {df['hit'].mean():.2f} ({int(df['hit'].sum())}/{len(df)})")


if __name__ == "__main__":
    main()
