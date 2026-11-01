from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    chunks_added: int
    collection_name: str


class ChatRequest(BaseModel):
    """
    Client sends a session_id so backend can keep per-user conversation memory.
    """

    session_id: str
    message: str


class RetrievedChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: Literal["ok"]

