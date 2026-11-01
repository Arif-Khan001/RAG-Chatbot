from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.core.config import get_settings
from backend.app.services.document_loader import extract_text
from backend.app.services.embedding_service import get_embeddings

logger = logging.getLogger(__name__)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_DEFAULT = 8
COLLECTION_NAME = "rag_documents"


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def get_vectorstore() -> Chroma:
    settings = get_settings()
    persist_dir: Path = settings.chroma_db_dir
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Chroma uses the provided embedding function
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def ingest_file(file_path: Path, filename: str, *, overwrite: bool = False) -> int:
    """
    Extract text, chunk it, embed chunks, and store in Chroma.
    Returns number of chunks added.
    """
    extension = file_path.suffix
    raw_text = extract_text(file_path=file_path, extension=extension)

    if not raw_text or not raw_text.strip():
        logger.warning("No text extracted from %s", filename)
        return 0

    splitter = get_text_splitter()
    chunk_texts = splitter.split_text(raw_text)

    docs: List[Document] = []
    for i, chunk in enumerate(chunk_texts):
        if not chunk or not chunk.strip():
            continue
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_index": i,
                },
            )
        )

    if not docs:
        logger.warning("No chunks created for %s", filename)
        return 0

    vectorstore = get_vectorstore()

    # If overwrite is enabled, we create a new collection by "dropping" the existing one.
    # Chroma's Python API doesn't provide an always-stable "drop collection" here,
    # so for MVP we'll just add by default.
    if overwrite:
        logger.info("overwrite=True requested, but MVP keeps simple (no collection drop).")

    vectorstore.add_documents(docs)
    vectorstore.persist()

    return len(docs)


def retrieve(query: str, *, top_k: int = TOP_K_DEFAULT) -> List[Document]:
    """
    Similarity search in Chroma for the given query.
    """
    vectorstore = get_vectorstore()
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return results or []
    except Exception:
        logger.exception("Retrieval failed (possibly empty vector DB).")
        return []


def format_context(docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Formats retrieved documents into a single context string for the LLM.
    """
    if not docs:
        return "", []

    context_lines: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", None)
        context_lines.append(f"[{idx}] (source: {source})\n{doc.page_content}\n")
        sources.append(
            {
                "source": source,
                "chunk_index": chunk_index,
            }
        )

    return "\n".join(context_lines).strip(), sources

