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
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
