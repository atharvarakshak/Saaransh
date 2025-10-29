from __future__ import annotations

from typing import List, Tuple, Union
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image


def load_embedder(model_name: str = "clip-ViT-B-32") -> SentenceTransformer:
    """Load CLIP model for multimodal embeddings (text + images).

    Args:
        model_name: CLIP model name (default: clip-ViT-B-32 for multilingual support).
                   Other options: clip-ViT-B-32, clip-ViT-L-14, etc.
    """
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Encode texts and return normalized float32 embeddings for cosine similarity.

    Args:
        model: Loaded CLIP model.
        texts: List of strings to encode.
    Returns:
        2D numpy array [N, D] normalized.
    """
    emb = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    emb = np.array(emb, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


def embed_images(model: SentenceTransformer, image_paths: List[Union[str, Path]]) -> np.ndarray:
    """Encode images and return normalized float32 embeddings for cosine similarity.

    Args:
        model: Loaded CLIP model.
        image_paths: List of image file paths (str or Path).
    Returns:
        2D numpy array [N, D] normalized.
    """
    # Load images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            # Skip invalid images
            continue
    
    if not images:
        return np.array([], dtype=np.float32).reshape(0, model.get_sentence_embedding_dimension())
    
    emb = model.encode(images, convert_to_tensor=False, show_progress_bar=False)
    emb = np.array(emb, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


def embed_query(model: SentenceTransformer, query: Union[str, Image.Image]) -> np.ndarray:
    """Encode a query (text or image) for search.

    Args:
        model: Loaded CLIP model.
        query: Query string or PIL Image.
    Returns:
        1D numpy array [D] normalized.
    """
    emb = model.encode([query], convert_to_tensor=False, show_progress_bar=False)
    emb = np.array(emb[0], dtype=np.float32)
    # Normalize for cosine similarity
    norm = np.linalg.norm(emb) + 1e-12
    return emb / norm


def build_faiss_index(embeddings: np.ndarray) -> Tuple[faiss.IndexFlatIP, int]:
    """Build FAISS inner-product index for cosine similarity.

    Args:
        embeddings: Normalized embeddings [N, D]
    Returns:
        (index, dim)
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, dim


def search_index(index: faiss.IndexFlatIP, query_emb: np.ndarray, k: int = 5, min_similarity: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index with normalized query embedding and filter by similarity.

    Args:
        index: FAISS index.
        query_emb: [1, D] normalized query embedding.
        k: top-k.
        min_similarity: Minimum cosine similarity threshold.
    Returns:
        (distances, indices) - filtered by similarity
    """
    distances, indices = index.search(query_emb, k)
    # Filter by similarity threshold (distances are cosine similarities for normalized embeddings)
    mask = distances[0] >= min_similarity
    return distances[:, mask], indices[:, mask]
