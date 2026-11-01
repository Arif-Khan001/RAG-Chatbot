from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def ensure_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []

    if "doc_ingested" not in st.session_state:
        st.session_state.doc_ingested = False


def upload_document(backend_url: str, file) -> Dict[str, Any]:
    files = {"file": (file.name, file.getvalue())}
    resp = requests.post(f"{backend_url}/upload", files=files, timeout=300)
    resp.raise_for_status()
    return resp.json()


def send_chat(backend_url: str, session_id: str, message: str) -> Dict[str, Any]:
    payload = {"session_id": session_id, "message": message}
    resp = requests.post(f"{backend_url}/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("RAG Chatbot")

    ensure_session_state()

    uploaded = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Ingest document", disabled=uploaded is None):
            try:
                with st.spinner("Uploading & ingesting..."):
                    result = upload_document(BACKEND_URL, uploaded)
                st.success(f"Ingested: {result.get('filename')} (chunks added: {result.get('chunks_added')})")
                st.session_state.doc_ingested = True
            except Exception as e:
                st.error(f"Upload failed: {e}")

    with col2:
        st.info(f"Session ID: `{st.session_state.session_id}`")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_message = st.chat_input("Ask a question about your document...")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        if not st.session_state.doc_ingested:
            assistant_text = "Please upload a document first using the button above."
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
            return

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = send_chat(BACKEND_URL, st.session_state.session_id, user_message)
                    answer = result.get("answer", "")
                    metadata = result.get("metadata") or {}

                    # Optional: show retrieved sources in small text
                    sources = metadata.get("sources") or []
                    if sources and metadata.get("retrieved"):
                        source_lines = []
                        for s in sources[:4]:
                            src = s.get("source", "unknown")
                            idx = s.get("chunk_index")
                            if idx is None:
                                source_lines.append(f"- {src}")
                            else:
                                source_lines.append(f"- {src} (chunk {idx})")
                        answer_with_sources = answer + "\n\n" + "Retrieved from:\n" + "\n".join(source_lines)
                    else:
                        answer_with_sources = answer

                    st.markdown(answer_with_sources)
                    st.session_state.messages.append({"role": "assistant", "content": answer_with_sources})
                except Exception as e:
                    assistant_text = f"Chat failed: {e}"
                    st.markdown(assistant_text)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()

