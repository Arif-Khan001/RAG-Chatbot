from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables early (we'll add .env handling later)
from dotenv import load_dotenv
from pathlib import Path

from backend.app.routes.chat import router as chat_router
from backend.app.routes.health import router as health_router
from backend.app.routes.upload import router as upload_router

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="RAG Chatbot API",
    version="0.1.0",
    description="Backend API for a simple RAG-based chatbot with file upload and chat endpoints.",
)

# Allow local frontend and docs access; can be extended later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

