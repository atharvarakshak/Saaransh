from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from pyvis.network import Network


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for normalized embeddings."""
    return embeddings @ embeddings.T


def build_graph(
    similarity_matrix: np.ndarray, 
    chunk_texts: List[str], 
    chunk_metadata: List[Dict[str, Any]],
    threshold: float = 0.75
) -> Network:
    """Create a completely static PyVis graph with fixed positions (no physics, no animation)."""

    # Initialize PyVis network
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#1a1a1a",
        font_color="white",
        directed=False,
        notebook=False,
        cdn_resources='in_line'
    )

    # Absolute disable physics
    net.set_options("""
    {
      "physics": {
        "enabled": false,
        "stabilization": { "enabled": false },
        "solver": "forceAtlas2Based"
      },
      "layout": {
        "improvedLayout": false,
        "randomSeed": 42
      },
       "interaction": {
         "dragNodes": false,
         "zoomView": false,
         "dragView": false,
         "hover": true
      },
      "manipulation": {
        "enabled": false
      }
    }
    """)

    TEXT_COLOR = {"background": "#4A90E2", "border": "#2E5C8A"}
    IMAGE_COLOR = {"background": "#E24A4A", "border": "#A02E2E"}

    n_total = len(chunk_texts)
    radius = 350 if n_total <= 50 else 500

    # Precompute fixed positions in a circle
    for i, (chunk, meta) in enumerate(zip(chunk_texts, chunk_metadata)):
        chunk_type = meta.get("type", "text")
        if chunk_type == "image":
            img_path = Path(chunk)
            label = f"🖼️ {img_path.name[:20]}" + ("..." if len(img_path.name) > 20 else "")
            tooltip = f"Image: {img_path.name}\nSource: {meta.get('source', 'Unknown')}"
            color = IMAGE_COLOR["background"]
        else:
            label = f"📄 {(chunk[:20] + '...') if len(chunk) > 20 else chunk}"
            tooltip = f"Text Chunk:\n{chunk}\n\nSource: {meta.get('source', 'Unknown')}"
            color = TEXT_COLOR["background"]

        # Fixed position (circle)
        theta = (2.0 * np.pi * i) / max(1, n_total)
        x_coord = float(radius * np.cos(theta))
        y_coord = float(radius * np.sin(theta))

        # Each node is fixed — no movement at all
        net.add_node(
            i,
            label=label,
            title=tooltip,
            color=color,
            size=28,
            shape="dot",
            font="14px arial white",
            borderWidth=3,
            borderWidthSelected=5,
            shadow=True,
            physics=False,
            fixed={"x": True, "y": True},
            x=x_coord,
            y=y_coord
        )

    # Add edges above threshold
    n = len(chunk_texts)
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = float(similarity_matrix[i, j])
            if sim_score > threshold:
                edge_width = max(3, sim_score * 8)
                edge_opacity = 0.6 + sim_score * 0.3
                edge_color = f"rgba(255, 255, 255, {edge_opacity})"
                net.add_edge(
                    i, j,
                    value=sim_score,
                    width=edge_width,
                    color=edge_color,
                    title=f"Similarity: {sim_score:.3f}",
                    shadow={"enabled": True},
                    physics=False
                )
    return net
