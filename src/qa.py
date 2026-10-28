"""RAG pipeline: retrieve top-k chunks from FAISS, stuff them into the
prompt template, and generate an answer with a local HuggingFace LLM."""

import argparse
import os
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

from .prompts import RAG_PROMPT
from .vectorstore import load_vectorstore

GEN_MODEL = os.environ.get("GEN_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")


def get_generator(model_name: str = GEN_MODEL, max_new_tokens: int = 160) -> HuggingFacePipeline:
    """Build a local text-generation pipeline wrapped for LangChain.

    Uses greedy decoding (``do_sample=False``) so answers are deterministic.

    Args:
        model_name: HF model id or local path; defaults to the
            ``GEN_MODEL`` env override, else SmolLM2-360M-Instruct.
        max_new_tokens: Cap on generated tokens per answer.

    Returns:
        A ``HuggingFacePipeline`` usable as the ``generator`` in ``answer``.
    """
    gen = hf_pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=gen)


def answer(
    question: str,
    store: FAISS,
    generator: HuggingFacePipeline | None = None,
    k: int = 3,
) -> dict[str, object]:
    """Run one RAG turn: retrieve top-k chunks, stuff the prompt, generate.

    Args:
        question: The user's question.
        store: FAISS vector store built by ``build_vectorstore``.
        generator: Text-generation pipeline (see :func:`get_generator`).
            With ``None``, skips generation and returns retrieved context
            only (useful for retrieval-only mode and tests).
        k: Number of chunks retrieved per question.

    Returns:
        Dict with ``answer`` (generated text, or ``None`` when
        ``generator`` is ``None``), ``context`` (the retrieved chunks
        joined as one string), and ``sources`` (source document names).
    """
    docs = store.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source") for d in docs]
    if generator is None:
        return {"answer": None, "context": context, "sources": sources}
    prompt = RAG_PROMPT.format(context=context, question=question)
    text = generator.invoke(prompt).strip()
    return {"answer": text, "context": context, "sources": sources}


def main() -> None:
    """CLI: ask one question over the persisted index.

    With ``--no-generate`` only the retrieved context is printed (no LLM call).
    """
    ap = argparse.ArgumentParser(description="Ask a question over your documents.")
    ap.add_argument("question", nargs="?", help="Question to answer")
    ap.add_argument("--vectorstore", type=Path, default=Path("vectorstore"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-generate", action="store_true",
                    help="Retrieval only: print context without calling the LLM")
    args = ap.parse_args()

    if args.k < 1:
        print(
            f"Error: --k must be at least 1 (got {args.k}): "
            "choose how many chunks to retrieve per question.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        question = args.question or input("Question: ")
    except (EOFError, KeyboardInterrupt):
        print("\nNo question asked, exiting.", file=sys.stderr)
        sys.exit(0)
    if not question.strip():
        print(
            "Error: question is empty. Ask a non-empty question, e.g. "
            "`python -m src.qa 'what is RAG?'`.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.vectorstore.exists():
        print(
            f"Error: no vectorstore found at '{args.vectorstore}'. "
            "Run `python -m src.build_index` first to index your documents.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.vectorstore.is_dir():
        print(
            f"Error: vectorstore path '{args.vectorstore}' is not a directory. "
            "Point --vectorstore at the folder created by "
            "`python -m src.build_index`.",
            file=sys.stderr,
        )
        sys.exit(1)
    missing_files = [
        name
        for name in ("index.faiss", "index.pkl")
        if not (args.vectorstore / name).is_file()
    ]
    if missing_files:
        print(
            f"Error: vectorstore at '{args.vectorstore}' is incomplete "
            f"(missing {', '.join(missing_files)}). "
            "Re-run `python -m src.build_index` to rebuild the index.",
            file=sys.stderr,
        )
        sys.exit(1)
    store = load_vectorstore(args.vectorstore)
    generator = None if args.no_generate else get_generator()
    result = answer(question, store, generator, k=args.k)
    print(f"Sources: {result['sources']}")
    if result["answer"] is None:
        print("\n--- retrieved context ---\n" + result["context"])
    else:
        print("\n" + result["answer"])


if __name__ == "__main__":
    main()
