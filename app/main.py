import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import documents, query
from app.core.dependencies import get_engine, get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentNotReadyError,
    InvalidTransitionError,
    LLMError,
    UnsupportedFileTypeError,
)
from app.db.database import Base

STATUS_CODES = {
    DocumentNotFoundError: 404,
    InvalidTransitionError: 409,
    DocumentNotReadyError: 409,
    UnsupportedFileTypeError: 415,
    LLMError: 502,
}


async def handle_app_error(request: Request, exc: Exception):
    return JSONResponse(status_code=STATUS_CODES[type(exc)], content={"detail": str(exc)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(get_settings().upload_dir, exist_ok=True)  # заодно создаст ./data
    Base.metadata.create_all(get_engine())
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="DocMind", lifespan=lifespan)
    app.include_router(documents.router)
    app.include_router(query.router)
    for exc_type in STATUS_CODES:
        app.add_exception_handler(exc_type, handle_app_error)
    return app


app = create_app()