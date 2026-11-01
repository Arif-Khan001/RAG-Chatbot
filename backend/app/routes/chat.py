from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import ChatRequest, ChatResponse
from backend.app.services.chat_service import answer_with_rag

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Chat with conversational RAG over documents previously uploaded via POST /upload.
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    try:
        answer, metadata = answer_with_rag(payload.session_id, payload.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}") from e

    return ChatResponse(answer=answer, session_id=payload.session_id, metadata=metadata)

