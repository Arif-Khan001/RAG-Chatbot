from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


def extract_text_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            texts.append(page_text)
    return "\n".join(texts).strip()


def extract_text_from_docx(file_path: Path) -> str:
    doc = DocxDocument(str(file_path))
    texts: list[str] = []
    for para in doc.paragraphs:
        if para.text and para.text.strip():
            texts.append(para.text)
    return "\n".join(texts).strip()


def extract_text_from_txt(file_path: Path) -> str:
    # Best-effort decoding: UTF-8 first, then fallback.
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1").strip()


def extract_text(file_path: Path, extension: str) -> str:
    """
    Extract plain text from supported document formats.

    extension should be one of: ".pdf", ".docx", ".txt"
    """
    ext = extension.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext == ".txt":
        return extract_text_from_txt(file_path)
    raise ValueError(f"Unsupported extension: {ext}")

