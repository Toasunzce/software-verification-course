import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.models import DocumentRecord, DocumentStatus
from app.ml.embedder import Embedder
from app.ml.vector_store import VectorStore
from app.repositories.document_repository import DocumentRepository
from app.services.classifier import ClassifierService
from app.services.loaders import DocumentLoader


class IndexingService:
    def __init__(
        self,
        repo: DocumentRepository,
        loader: DocumentLoader,
        splitter: RecursiveCharacterTextSplitter,
        embedder: Embedder,
        store: VectorStore,
        classifier: ClassifierService,
    ):
        self._repo = repo
        self._loader = loader
        self._splitter = splitter
        self._embedder = embedder
        self._store = store
        self._classifier = classifier

    def index(self, doc_id: int) -> DocumentRecord:
        doc = self._repo.get(doc_id)
        doc.transition_to(DocumentStatus.PROCESSING)
        doc.error = None
        self._repo.save(doc)

        try:
            pages = self._loader.load(doc.file_path)
            chunks = self._splitter.split_documents(pages)
            if not chunks:
                raise ValueError("Text not found in the document")

            embeddings = self._embedder.embed([c.page_content for c in chunks])
            self._store.delete_document(doc.id)
            self._store.add_chunks(doc.id, chunks, embeddings)

            doc.category, _ = self._classifier.classify_vector(
                np.mean(embeddings, axis=0)
            )
            doc.chunks_count = len(chunks)
        except Exception as e:
            doc.error = str(e)
            doc.transition_to(DocumentStatus.FAILED)
            return self._repo.save(doc)

        doc.transition_to(DocumentStatus.INDEXED)
        return self._repo.save(doc)