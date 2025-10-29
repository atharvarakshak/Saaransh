import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

from utils.file_utils import extract_text, extract_images_from_pdf
from utils.chunk_utils import chunk_text
from utils.embed_utils import load_embedder, embed_texts, embed_images, embed_query, build_faiss_index, search_index
from utils.graph_utils import compute_similarity_matrix, build_graph
from utils.llm_utils import generate_response_with_context

# Load environment variables
load_dotenv()

# Data structures to track content types
CONTENT_TYPES = {
    "text": "text",
    "image": "image"
}

# -------- Streamlit UI -------- #
st.set_page_config(page_title="GraphRAG Document Explorer", layout="wide")
st.title("📚 GraphRAG - Multi-Modal Document Visualization")

# Upload section
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
with col2:
    uploaded_images = st.file_uploader(
        "Upload images (JPG, PNG, etc.)",
        type=["jpg", "jpeg", "png", "gif", "bmp"],
        accept_multiple_files=True
    )

if uploaded_files or uploaded_images:
    # Initialize session state for image storage
    if 'image_dir' not in st.session_state:
        st.session_state.image_dir = Path(tempfile.mkdtemp())
        st.session_state.image_paths = []
    
    # Initialize CLIP model for multimodal embeddings
    embedder = load_embedder('clip-ViT-B-32')
    
    # Process text content
    all_chunks = []
    chunk_metadata = []  # Track type and additional info for each chunk
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded successfully!")
        
        # Extract & chunk text from documents
        splitter_chunk_size = 500
        splitter_overlap = 100
        
        for file in uploaded_files:
            # Extract text
            text = extract_text(file)
            if text:
                chunks = chunk_text(text, chunk_size=splitter_chunk_size, chunk_overlap=splitter_overlap)
                all_chunks.extend(chunks)
                chunk_metadata.extend([{"type": "text", "source": file.name}] * len(chunks))
            
            # Extract images from PDFs
            if file.name.lower().endswith('.pdf'):
                file.seek(0)  # Reset file pointer
                image_paths = extract_images_from_pdf(file, st.session_state.image_dir)
                if image_paths:
                    for img_path in image_paths:
                        all_chunks.append(str(img_path))  # Store path
                        chunk_metadata.append({"type": "image", "source": file.name, "path": img_path})
                        st.session_state.image_paths.append(img_path)
    
    # Process uploaded images
    if uploaded_images:
        st.success(f"{len(uploaded_images)} image(s) uploaded successfully!")
        
        for img_file in uploaded_images:
            # Save uploaded image to persistent directory
            img_path = st.session_state.image_dir / img_file.name
            with open(img_path, "wb") as f:
                f.write(img_file.read())
            
            all_chunks.append(str(img_path))
            chunk_metadata.append({"type": "image", "source": img_file.name, "path": img_path})
            st.session_state.image_paths.append(img_path)
    
    if not all_chunks:
        st.warning("No content extracted from uploaded files!")
        st.stop()
    
    st.write(f"Total Items: {len(all_chunks)} (Text chunks + Images)")
    
    # Generate embeddings based on content type
    text_chunks = [all_chunks[i] for i, meta in enumerate(chunk_metadata) if meta["type"] == "text"]
    image_chunks = [all_chunks[i] for i, meta in enumerate(chunk_metadata) if meta["type"] == "image"]
    
    embeddings_list = []
    
    # Embed text chunks
    if text_chunks:
        text_embeddings = embed_texts(embedder, text_chunks)
        embeddings_list.append(text_embeddings)
    
    # Embed image chunks
    if image_chunks:
        with st.spinner("Processing images..."):
            image_paths = [Path(chunk) for chunk in image_chunks]
            image_embeddings = embed_images(embedder, image_paths)
            embeddings_list.append(image_embeddings)
    
    # Concatenate all embeddings
    np_embeddings = np.vstack(embeddings_list) if embeddings_list else np.array([], dtype=np.float32)
    
    st.success("Embeddings generated successfully!")

    # Store in FAISS (inner product for cosine)
    index, dim = build_faiss_index(np_embeddings)

    st.success("Stored in FAISS successfully!")
    
    # Build similarity matrix (only for visualization, can be slow with many items)
    if len(all_chunks) <= 100:  # Limit graph size for performance
        similarity_matrix = compute_similarity_matrix(np_embeddings)
        graph = build_graph(similarity_matrix, all_chunks)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            graph.write_html(tmp_file.name)
            html_path = tmp_file.name

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=600, width=1200, scrolling=True)
        st.success("Graph visualization complete!")
    else:
        st.info("Graph visualization skipped (too many items). Focus on search functionality.")

    # Search interface
    st.divider()
    st.subheader("🔍 Search")
    
    # Sidebar controls for search
    with st.sidebar:
        st.subheader("🔧 Search Settings")
        similarity_threshold = st.slider(
            "Minimum Similarity Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.25, 
            step=0.05,
            help="Higher values = more relevant results, fewer results"
        )
        max_results = st.slider(
            "Maximum Results", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Maximum number of results to retrieve"
        )
    
    # Unified query input - text or image
    query_text = st.text_input("Enter your query (text):", placeholder="Type your question here...")
    query_image = st.file_uploader("Or upload an image query:", type=["jpg", "jpeg", "png"], help="Upload an image to search visually")
    
    # Process query
    query = None
    if query_image:
        query = Image.open(query_image).convert("RGB")
        st.image(query, caption="Query Image", width=300)
    elif query_text:
        query = query_text
    
    if query:
        # Get API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("Please set GOOGLE_API_KEY in your .env file")
            st.stop()
        
        # Embed query
        q_emb = embed_query(embedder, query)
        
        # Search
        q_emb = q_emb.reshape(1, -1)  # Reshape for FAISS
        D, I = search_index(index, q_emb, k=max_results, min_similarity=similarity_threshold)
        
        if len(I[0]) == 0:
            st.warning(f"No results found above similarity threshold {similarity_threshold:.2f}. Try lowering the threshold.")
            st.stop()
        
        # Get relevant results
        relevant_items = [all_chunks[idx] for idx in I[0]]
        relevant_metadata = [chunk_metadata[idx] for idx in I[0]]
        
        # Display results and generate response
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"📄 Relevant Results ({len(relevant_items)} found)")
            # Separate text chunks and image paths
            relevant_text_chunks = []
            relevant_image_paths = []
            
            for i, idx in enumerate(I[0]):
                similarity_score = float(D[0][i])
                meta = chunk_metadata[idx]
                item = all_chunks[idx]
                
                color = "🟢" if similarity_score > 0.7 else "🟡" if similarity_score > 0.5 else "🔴"
                result_type = "📄" if meta["type"] == "text" else "🖼️"
                
                with st.expander(f"{color} {result_type} Result {i+1} (Similarity: {similarity_score:.3f})"):
                    if meta["type"] == "text":
                        st.write(item)
                        relevant_text_chunks.append(item)
                    else:
                        # Display image
                        img_path = meta.get("path", item)
                        if isinstance(img_path, str):
                            img_path = Path(img_path)
                        if img_path.exists():
                            st.image(str(img_path), caption=f"From: {meta['source']}")
                            relevant_image_paths.append(img_path)
                        else:
                            st.warning(f"Image not found: {img_path}")
        
        with col2:
            st.subheader("🤖 AI Response")
            with st.spinner("Generating response (this may include analyzing images)..."):
                response = generate_response_with_context(
                    query, 
                    relevant_text_chunks, 
                    relevant_image_paths, 
                    api_key
                )
                st.write(response)
