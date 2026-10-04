from functools import lru_cache
from typing import Iterator

import chromadb
from fastapi import Depends
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.clients.llm import GroqClient, LLMClient
from app.core.config import Settings
from app.db.database import make_engine
from app.ml.embedder import Embedder, SentenceTransformerEmbedder
from app.ml.vector_store import ChromaVectorStore, VectorStore
from app.repositories.document_repository import DocumentRepository
from app.services.classifier import ClassifierService
from app.services.document_service import DocumentService
from app.services.indexing import IndexingService
from app.services.loaders import DocumentLoader, LangChainLoader
from app.services.qa import QAService
from app.services.search import SearchService
from app.services.summary import SummaryService


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


@lru_cache
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine())


@lru_cache
def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder(get_settings().embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    client = chromadb.PersistentClient(path=get_settings().chroma_path)
    return ChromaVectorStore(client)


@lru_cache
def get_llm_client() -> LLMClient:
    s = get_settings()
    return GroqClient(api_key=s.groq_api_key, model=s.groq_model, url=s.groq_url)


@lru_cache
def get_loader() -> DocumentLoader:
    return LangChainLoader()


@lru_cache
def get_classifier() -> ClassifierService:
    return ClassifierService(get_embedder(), get_settings().categories)



def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_splitter(settings: Settings = Depends(get_settings)) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


def get_repository(session: Session = Depends(get_session)) -> DocumentRepository:
    return DocumentRepository(session)


def get_document_service(
    repo: DocumentRepository = Depends(get_repository),
    store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    return DocumentService(repo, store, settings.upload_dir)


def get_indexing_service(
    repo: DocumentRepository = Depends(get_repository),
    loader: DocumentLoader = Depends(get_loader),
    splitter: RecursiveCharacterTextSplitter = Depends(get_splitter),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
    classifier: ClassifierService = Depends(get_classifier),
) -> IndexingService:
    return IndexingService(repo, loader, splitter, embedder, store, classifier)


def get_search_service(
    repo: DocumentRepository = Depends(get_repository),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> SearchService:
    return SearchService(repo, embedder, store)


def get_qa_service(
    repo: DocumentRepository = Depends(get_repository),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> QAService:
    return QAService(repo, embedder, store, llm, settings.top_k)


def get_summary_service(
    repo: DocumentRepository = Depends(get_repository),
    store: VectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(get_llm_client),
) -> SummaryService:
    return SummaryService(repo, store, llm)