"""Document loading and chunking with a LangChain text splitter."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_dir: str | Path) -> list[Document]:
    """Load every .txt file in data_dir as a LangChain Document."""
    docs: list[Document] = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )
    return docs


def chunk_documents(
    docs: list[Document], chunk_size: int = 400, chunk_overlap: int = 50
) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Args:
        docs: Documents to split.
        chunk_size: Maximum characters per chunk; must be at least 1.
        chunk_overlap: Characters shared between consecutive chunks; must
            be non-negative and smaller than chunk_size, otherwise the
            splitter would loop forever re-emitting the same chunk.

    Raises:
        ValueError: If chunk_size < 1, chunk_overlap < 0, or
            chunk_overlap >= chunk_size.
    """
    if chunk_size < 1:
        raise ValueError(
            f"chunk_size must be at least 1 (got {chunk_size})."
        )
    if chunk_overlap < 0:
        raise ValueError(
            f"chunk_overlap cannot be negative (got {chunk_overlap})."
        )
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}), otherwise the splitter would "
            "loop forever re-emitting the same chunk."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
