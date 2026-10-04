from app.ml.embedder import Embedder
from app.ml.vector_store import SearchHit, VectorStore
from app.repositories.document_repository import DocumentRepository


class SearchService:
    def __init__(self, repo: DocumentRepository, embedder: Embedder, store: VectorStore):
        self._repo = repo
        self._embedder = embedder
        self._store = store

    def search(
        self, query: str, top_k: int, document_id: int | None = None
    ) -> list[SearchHit]:
        embedding = self._embedder.embed([query])[0]
        hits = self._store.search(embedding, top_k, document_id)
        self._repo.log_query("search", query)
        return hits