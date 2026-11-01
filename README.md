# RAG Chatbot (FastAPI + LangChain + Chroma + Streamlit)

Lightweight implementation for a RAG-based conversational chatbot:
- FastAPI backend
- Streamlit frontend
- File upload (PDF/DOCX/TXT)
- ChromaDB persistent vector storage
- SentenceTransformers embeddings
- Groq LLM (configurable via `GROQ_MODEL`, default `llama-3.1-8b-instant`)
- Conversational memory (per `session_id`)

## Project Layout

- `backend/` FastAPI API + RAG services
- `frontend/` Streamlit UI
- `data/` uploaded files
- `chroma_db/` persistent Chroma database


