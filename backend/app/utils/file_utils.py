from __future__ import annotations

import os
from typing import Optional


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def get_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext


def validate_upload_filename(filename: Optional[str]) -> str:
    """
    Validates filename presence and supported extension.
    Returns the safe basename.
    """
    if not filename:
        raise ValueError("Uploaded file must have a filename.")

    basename = os.path.basename(filename)
    ext = get_extension(basename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    return basename

