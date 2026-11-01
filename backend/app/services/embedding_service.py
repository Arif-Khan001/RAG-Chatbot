from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import SentenceTransformerEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache()
def get_embeddings() -> SentenceTransformerEmbeddings:
    """
    Cached embedding model instance for fast subsequent calls.
    """
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

