from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Tuple

from groq import BadRequestError
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.app.core.config import get_settings
from backend.app.services import rag_service

logger = logging.getLogger(__name__)


class ChatMemoryStore:
    """
    Simple in-memory session store.

    Note: This is not persistent across backend restarts (MVP-friendly).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # session_id -> list of turns:
        # [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
        self._memories: Dict[str, List[Dict[str, str]]] = {}

    def get(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            if session_id not in self._memories:
                self._memories[session_id] = []
            return self._memories[session_id]

    def append_turn(
        self,
        session_id: str,
        *,
        user_message: str,
        assistant_message: str,
        max_turns: int = 20,
    ) -> None:
        """
        Append a user/assistant pair and cap the memory size.
        """
        mem = self.get(session_id)
        mem.append({"role": "user", "content": user_message})
        mem.append({"role": "assistant", "content": assistant_message})

        # Keep memory bounded to avoid huge prompts.
        # We store 2 entries per turn (user + assistant).
        max_entries = max_turns * 2
        if len(mem) > max_entries:
            del mem[: len(mem) - max_entries]


_MEMORY_STORE = ChatMemoryStore()


def _history_to_str(history: Any) -> str:
    """
    Converts stored chat history into readable text.
    """
    if history is None:
        return ""

    if isinstance(history, list):
        parts: List[str] = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role or 'Unknown'}: {content}")
        return "\n".join(parts).strip()

    # Sometimes it might be a plain string
    if isinstance(history, str):
        return history.strip()

    return str(history)


def _build_messages(chat_history: str, context: str, question: str) -> List[BaseMessage]:
    system_prompt = (
        "You are a helpful AI assistant that answers using the provided context. "
        "If the context does not contain enough information, say you don't know. "
        "Keep answers concise and grounded in the context."
    )

    user_prompt = (
        f"Chat history:\n{chat_history}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def _rewrite_followup_query(
    *,
    llm: ChatGroq,
    chat_history: str,
    user_question: str,
) -> str:
    """
    Rewrite follow-up questions into standalone search queries.

    This improves retrieval quality for context-dependent questions like:
    "How many sessions are there?" after previous conversation turns.
    """
    # If there's no history, rewriting is unnecessary.
    if not chat_history.strip():
        return user_question

    rewrite_system_prompt = (
        "You rewrite user questions for vector retrieval in a RAG system. "
        "Return ONLY one standalone search query. "
        "Do not answer the question. "
        "Keep it concise and faithful to the conversation context."
    )
    rewrite_user_prompt = (
        f"Chat history:\n{chat_history}\n\n"
        f"Follow-up question:\n{user_question}\n\n"
        "Standalone retrieval query:"
    )

    rewritten = llm.invoke(
        [
            SystemMessage(content=rewrite_system_prompt),
            HumanMessage(content=rewrite_user_prompt),
        ]
    )
    text = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
    text = text.strip()
    return text or user_question


def answer_with_rag(session_id: str, message: str) -> Tuple[str, Dict[str, Any]]:
    """
    Main chat logic:
    - retrieve relevant chunks from Chroma
    - include conversation history for follow-ups
    - call Groq LLM to generate grounded answer
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    # 1) Load conversation history
    memory = _MEMORY_STORE.get(session_id)
    chat_history = _history_to_str(memory)

    # 2) Build LLM client once and reuse for rewrite + answer
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0.2,
        groq_api_key=settings.groq_api_key,
    )

    # 3) Rewrite follow-up to standalone query for retrieval
    try:
        standalone_query = _rewrite_followup_query(
            llm=llm,
            chat_history=chat_history,
            user_question=message,
        )
    except Exception:
        logger.exception("Query rewrite failed, using raw user message for retrieval.")
        standalone_query = message

    # 4) Retrieve relevant chunks with rewritten query
    docs = rag_service.retrieve(standalone_query, top_k=rag_service.TOP_K_DEFAULT)
    context, sources = rag_service.format_context(docs)

    if not context:
        answer = "I couldn't find relevant information in your uploaded documents. Please upload more files or ask a different question."
        metadata: Dict[str, Any] = {
            "retrieved": False,
            "sources": [],
            "standalone_query": standalone_query,
        }
        _MEMORY_STORE.append_turn(session_id, user_message=message, assistant_message=answer)
        return answer, metadata

    # 5) Answer using full chat context + retrieved docs
    messages = _build_messages(chat_history=chat_history, context=context, question=message)
    try:
        response = llm.invoke(messages)
    except BadRequestError as e:
        err_text = str(e)
        if "model_decommissioned" in err_text:
            raise ValueError(
                f"Configured GROQ_MODEL '{settings.groq_model}' is decommissioned. "
                "Set GROQ_MODEL in backend/.env to a currently supported Groq model."
            ) from e
        raise

    answer = response.content if hasattr(response, "content") else str(response)

    # 4) Update memory after response
    _MEMORY_STORE.append_turn(session_id, user_message=message, assistant_message=answer)

    metadata = {
        "retrieved": True,
        "sources": sources,
        "standalone_query": standalone_query,
    }
    return answer, metadata

