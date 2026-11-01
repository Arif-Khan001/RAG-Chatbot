from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.paths import UPLOAD_DIR
from backend.app.models.schemas import UploadResponse
from backend.app.services import rag_service
from backend.app.utils.file_utils import validate_upload_filename

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a supported document type and ingest it into ChromaDB.
    """
    try:
        safe_name = validate_upload_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    save_path = UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        save_path.write_bytes(content)
    except Exception as e:
        logger.exception("Failed saving uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}") from e

    try:
        chunks_added = rag_service.ingest_file(save_path, filename=safe_name)
    except Exception as e:
        logger.exception("Failed ingesting document into Chroma")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}") from e

    return UploadResponse(
        filename=safe_name,
        saved_path=str(save_path),
        chunks_added=chunks_added,
        collection_name=rag_service.COLLECTION_NAME,
    )

