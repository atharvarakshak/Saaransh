from __future__ import annotations

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Split text into chunks using recursive character splitter.

    Args:
        text: Input text.
        chunk_size: Max size per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
