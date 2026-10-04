from app.clients.llm import LLMClient
from app.core.exceptions import DocumentNotReadyError
from app.db.models import DocumentStatus
from app.ml.vector_store import VectorStore
from app.repositories.document_repository import DocumentRepository

MAX_CHARS = 6000
SYSTEM_PROMPT = "You're a helper. Make a small summery in 3-5 sentences on context language."


class SummaryService:
    def __init__(self, repo: DocumentRepository, store: VectorStore, llm: LLMClient):
        self._repo = repo
        self._store = store
        self._llm = llm

    def summarize(self, doc_id: int) -> str:
        doc = self._repo.get(doc_id)
        if doc.status != DocumentStatus.INDEXED:
            raise DocumentNotReadyError("Summary is available only for indeced documents")
        if doc.summary:
            return doc.summary

        text = "\n".join(self._store.get_chunks(doc.id))[:MAX_CHARS]
        doc.summary = self._llm.complete(text, SYSTEM_PROMPT)
        self._repo.save(doc)
        return doc.summary