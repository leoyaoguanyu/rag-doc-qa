# RAG Document Q&A

[![CI](https://github.com/leoyaoguanyu/rag-doc-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/leoyaoguanyu/rag-doc-qa/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A retrieval-augmented generation (RAG) pipeline that answers questions over
your own documents. Built as a portfolio project for AI Engineer roles:
everything runs locally, no paid API key needed.

## How it works

```
PDF/TXT docs → chunk (LangChain splitter) → embed (sentence-transformers)
→ FAISS vector index → top-k retrieval → prompt template → local LLM answer
```

- **Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` embeddings + FAISS
  cosine-similarity search (top-k).
- **Prompt engineering**: explicit RAG template (`src/prompts.py`) with a
  grounding instruction and an "I don't know" guardrail against hallucination.
- **Generation**: small local instruct LLM via HuggingFace `transformers`
  (no OpenAI key required).
- **Evaluation**: `src/evaluate.py` measures retrieval **hit-rate@k** over a
  labeled Q&A set (`data/eval.json`), reported as a pandas table.

## Results

| Metric | Value |
|---|---|
| Documents → chunks | 3 → 9 |
| Retrieval hit-rate@3 | **1.00 (8/8)** |
| Unit tests | 12 passed (pytest, incl. retriever edge cases) |
| CI | GitHub Actions: tests + index build + eval smoke test |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the FAISS index from data/sample_docs
python -m src.build_index

# 2. Ask a question (retrieval + local LLM generation)
python -m src.qa "Which vector search library was developed by Meta?"

# 2b. Retrieval only, skip the LLM
python -m src.qa "What is RAG?" --no-generate

# 3. Run the retrieval evaluation
python -m src.evaluate
```

Drop your own `.txt` files into `data/sample_docs/` and re-run
`python -m src.build_index` to query them.

## Project structure

```
src/
  ingest.py        # load .txt docs, chunk with LangChain RecursiveCharacterTextSplitter
  vectorstore.py   # embed + FAISS build/save/load (EMBEDDING_MODEL env override)
  prompts.py       # RAG prompt template (grounding instruction + guardrail)
  qa.py            # retrieve → stuff prompt → generate (CLI included)
  evaluate.py      # hit-rate@k over data/eval.json
  build_index.py   # one-shot: ingest → chunk → index → persist
tests/             # 12 pytest tests (FAISS round-trip uses fake embeddings: no downloads)
notebooks/
  rag_demo.ipynb   # end-to-end walkthrough
data/
  sample_docs/     # 3 sample docs (RAG, vector DBs, prompt engineering)
  eval.json        # 8 labeled Q&A pairs for retrieval eval
```

## Notes

- The embedding model (`all-MiniLM-L6-v2`, ~90MB) downloads from the
  HuggingFace Hub on first run. If the Hub is unreachable, set
  `EMBEDDING_MODEL` to a local model directory.
- On CPU-only machines generation with the default 360M model is slow but
  works; use `--no-generate` for instant retrieval-only answers.
