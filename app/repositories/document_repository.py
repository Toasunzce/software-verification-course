from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DocumentNotFoundError
from app.db.models import DocumentRecord, DocumentStatus, QueryLog


class DocumentRepository:
    def __init__(self, session: Session):
        self._session = session

    def add(self, doc: DocumentRecord) -> DocumentRecord:
        self._session.add(doc)
        self._session.commit()
        self._session.refresh(doc)
        return doc

    def get(self, doc_id: int) -> DocumentRecord:
        doc = self._session.get(DocumentRecord, doc_id)
        if doc is None:
            raise DocumentNotFoundError(f"Document {doc_id} wasn't found")
        return doc

    def list(self, status: DocumentStatus | None = None) -> list[DocumentRecord]:
        query = select(DocumentRecord).order_by(DocumentRecord.id)
        if status is not None:
            query = query.where(DocumentRecord.status == status)
        return list(self._session.scalars(query))

    def save(self, doc: DocumentRecord) -> DocumentRecord:
        self._session.commit()
        self._session.refresh(doc)
        return doc

    def delete(self, doc: DocumentRecord) -> None:
        self._session.delete(doc)
        self._session.commit()

    def log_query(self, kind: str, text: str) -> None:
        self._session.add(QueryLog(kind=kind, text=text))
        self._session.commit()