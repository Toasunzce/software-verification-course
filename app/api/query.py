from fastapi import APIRouter, Depends

from app.api.schemas import (
    AnswerOut,
    AskRequest,
    ClassifyOut,
    ClassifyRequest,
    SearchHitOut,
    SearchRequest,
)
from app.core.dependencies import get_classifier, get_qa_service, get_search_service
from app.services.classifier import ClassifierService
from app.services.qa import QAService
from app.services.search import SearchService

router = APIRouter(tags=["query"])


@router.post("/search", response_model=list[SearchHitOut])
def search(body: SearchRequest, service: SearchService = Depends(get_search_service)):
    return service.search(body.query, body.top_k, body.document_id)


@router.post("/ask", response_model=AnswerOut)
def ask(body: AskRequest, service: QAService = Depends(get_qa_service)):
    return service.answer(body.question, body.document_id)


@router.post("/classify", response_model=ClassifyOut)
def classify(body: ClassifyRequest, service: ClassifierService = Depends(get_classifier)):
    category, score = service.classify_text(body.text)
    return ClassifyOut(category=category, score=score)