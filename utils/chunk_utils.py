from __future__ import annotations

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import math
from collections import Counter

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


def entropy_based_chunking(text: str, window_size: int = 100, entropy_threshold: float = 0.5) -> List[str]:
    """
    Split text into chunks based on entropy variation.

    Args:
        text: Input text string.
        window_size: Number of words per entropy window.
        entropy_threshold: Entropy change threshold for splitting.

    Returns:
        List of text chunks.
    """
    def entropy(words):
        """Calculate Shannon entropy of a list of words."""
        counts = Counter(words)
        total = len(words)
        return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

    words = text.split()
    n = len(words)
    if n <= window_size:
        return [text]

    chunks = []
    start = 0
    prev_entropy = entropy(words[:window_size])

    for i in range(window_size, n, window_size // 2):  # half-window stride
        current_entropy = entropy(words[i - window_size:i])
        if abs(current_entropy - prev_entropy) > entropy_threshold:
            # Split where entropy changes significantly
            chunk = " ".join(words[start:i])
            chunks.append(chunk)
            start = i
        prev_entropy = current_entropy

    # Add last chunk
    if start < n:
        chunks.append(" ".join(words[start:]))

    return chunks

