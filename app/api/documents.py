from fastapi import APIRouter, Depends, Response, UploadFile

from app.api.schemas import DocumentOut, SummaryOut
from app.core.dependencies import (
    get_document_service,
    get_indexing_service,
    get_summary_service,
)
from app.db.models import DocumentStatus
from app.services.document_service import DocumentService
from app.services.indexing import IndexingService
from app.services.summary import SummaryService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=201)
def upload_document(file: UploadFile, service: DocumentService = Depends(get_document_service)):
    return service.upload(file.filename, file.file.read())


@router.get("", response_model=list[DocumentOut])
def list_documents(
    status: DocumentStatus | None = None,
    service: DocumentService = Depends(get_document_service),
):
    return service.list(status)


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, service: DocumentService = Depends(get_document_service)):
    return service.get(doc_id)


@router.post("/{doc_id}/index", response_model=DocumentOut)
def index_document(doc_id: int, service: IndexingService = Depends(get_indexing_service)):
    return service.index(doc_id)


@router.post("/{doc_id}/archive", response_model=DocumentOut)
def archive_document(doc_id: int, service: DocumentService = Depends(get_document_service)):
    return service.archive(doc_id)


@router.get("/{doc_id}/summary", response_model=SummaryOut)
def get_summary(doc_id: int, service: SummaryService = Depends(get_summary_service)):
    return SummaryOut(document_id=doc_id, summary=service.summarize(doc_id))


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: int, service: DocumentService = Depends(get_document_service)):
    service.delete(doc_id)
    return Response(status_code=204)