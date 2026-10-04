from pathlib import Path
from typing import Protocol

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from app.core.exceptions import UnsupportedFileTypeError

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}
SUPPORTED_EXTENSIONS = set(LOADERS)


class DocumentLoader(Protocol):
    def load(self, path: str) -> list[Document]: ...


class LangChainLoader:
    def load(self, path: str) -> list[Document]:
        ext = Path(path).suffix.lower()
        loader_cls = LOADERS.get(ext)
        if loader_cls is None:
            raise UnsupportedFileTypeError(f"Format {ext} is not supported")
        if ext in {".txt", ".md"}:
            return loader_cls(path, encoding="utf-8").load()
        return loader_cls(path).load()