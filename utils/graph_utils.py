from __future__ import annotations

from typing import List
import numpy as np
from pyvis.network import Network


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for normalized embeddings.

    Args:
        embeddings: [N, D] normalized embeddings.
    Returns:
        [N, N] similarity matrix.
    """
    return embeddings @ embeddings.T


def build_graph(similarity_matrix: np.ndarray, chunk_texts: List[str], threshold: float = 0.75) -> Network:
    """Create an interactive graph based on similarity scores.

    Args:
        similarity_matrix: Pairwise cosine similarities.
        chunk_texts: Node labels/tooltips.
        threshold: Minimum edge weight to draw.
    Returns:
        PyVis Network.
    """
    net = Network(height="600px", width="800px", bgcolor="#222222", font_color="white")
    
    # Disable physics/animation to prevent nodes from moving
    net.set_options("""
    {
      "physics": {
        "enabled": false
      }
    }
    """)

    for i, text in enumerate(chunk_texts):
        label = (text[:30] + "...") if len(text) > 30 else text
        net.add_node(i, label=label, title=text)

    n = len(chunk_texts)
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = float(similarity_matrix[i, j])
            if sim_score > threshold:
                net.add_edge(i, j, value=sim_score)

    return net
