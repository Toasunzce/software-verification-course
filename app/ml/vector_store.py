from dataclasses import dataclass
from typing import Protocol

import chromadb
from langchain_core.documents import Document


@dataclass
class SearchHit:
    text: str
    document_id: int
    chunk_index: int
    score: float


class VectorStore(Protocol):
    def add_chunks(
        self, document_id: int, chunks: list[Document], embeddings: list[list[float]]
    ) -> None: ...

    def search(
        self, embedding: list[float], top_k: int, document_id: int | None = None
    ) -> list[SearchHit]: ...

    def get_chunks(self, document_id: int) -> list[str]: ...

    def delete_document(self, document_id: int) -> None: ...


class ChromaVectorStore:
    def __init__(self, client: chromadb.ClientAPI, collection_name: str = "chunks"):
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, document_id, chunks, embeddings):
        self._collection.add(
            ids=[f"{document_id}-{i}" for i in range(len(chunks))],
            documents=[c.page_content for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"document_id": document_id, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    def search(self, embedding, top_k, document_id=None):
        where = {"document_id": document_id} if document_id is not None else None
        result = self._collection.query(
            query_embeddings=[embedding], n_results=top_k, where=where
        )
        hits = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                SearchHit(
                    text=text,
                    document_id=meta["document_id"],
                    chunk_index=meta["chunk_index"],
                    score=1 - dist,  # косинусное расстояние -> сходство
                )
            )
        return hits

    def get_chunks(self, document_id):
        result = self._collection.get(where={"document_id": document_id})
        pairs = zip(result["metadatas"], result["documents"])
        ordered = sorted(pairs, key=lambda p: p[0]["chunk_index"])
        return [text for _, text in ordered]

    def delete_document(self, document_id):
        self._collection.delete(where={"document_id": document_id})