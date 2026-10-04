from pathlib import Path
from uuid import uuid4

from app.core.exceptions import UnsupportedFileTypeError
from app.db.models import DocumentRecord, DocumentStatus
from app.ml.vector_store import VectorStore
from app.repositories.document_repository import DocumentRepository
from app.services.loaders import SUPPORTED_EXTENSIONS


class DocumentService:
    def __init__(self, repo: DocumentRepository, store: VectorStore, upload_dir: str):
        self._repo = repo
        self._store = store
        self._upload_dir = Path(upload_dir)

    def upload(self, filename: str, content: bytes) -> DocumentRecord:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Format '{ext}' is not supported. Check on: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        path = self._upload_dir / f"{uuid4().hex}{ext}"
        path.write_bytes(content)
        return self._repo.add(DocumentRecord(filename=filename, file_path=str(path)))

    def get(self, doc_id: int) -> DocumentRecord:
        return self._repo.get(doc_id)

    def list(self, status: DocumentStatus | None = None) -> list[DocumentRecord]:
        return self._repo.list(status)

    def archive(self, doc_id: int) -> DocumentRecord:
        doc = self._repo.get(doc_id)
        doc.transition_to(DocumentStatus.ARCHIVED)
        self._store.delete_document(doc.id)
        return self._repo.save(doc)

    def delete(self, doc_id: int) -> None:
        doc = self._repo.get(doc_id)
        self._store.delete_document(doc.id)
        Path(doc.file_path).unlink(missing_ok=True)
        self._repo.delete(doc)