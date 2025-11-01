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
    
    text_count = sum(1 for m in chunk_metadata if m['type'] == 'text')
    image_count = sum(1 for m in chunk_metadata if m['type'] == 'image')
    st.write(f"Total Items: {len(all_chunks)} (Text chunks: {text_count}, Images: {image_count})")
    
    # Generate embeddings in the same order as all_chunks to maintain alignment
    text_indices = [i for i, meta in enumerate(chunk_metadata) if meta["type"] == "text"]
    image_indices = [i for i, meta in enumerate(chunk_metadata) if meta["type"] == "image"]
    
    # Build embeddings maintaining the order of all_chunks
    with st.spinner("Generating embeddings for all chunks..."):
        # Process in batches to maintain order
        text_chunks_batch = [all_chunks[i] for i in text_indices]
        image_chunks_batch = [all_chunks[i] for i in image_indices]
        
        # Create embeddings for each type
        text_embeddings_dict = {}
        text_embeddings_array = None
        if text_chunks_batch:
            text_embeddings = embed_texts(embedder, text_chunks_batch)
            text_embeddings_array = text_embeddings  # keep separate text embeddings
            for idx, emb in zip(text_indices, text_embeddings):
                text_embeddings_dict[idx] = emb
        
        image_embeddings_dict = {}
        image_embeddings_array = None
        if image_chunks_batch:
            image_paths = [Path(all_chunks[i]) for i in image_indices]
            image_embeddings = embed_images(embedder, image_paths)
            image_embeddings_array = image_embeddings  # keep separate image embeddings
            for idx, emb in zip(image_indices, image_embeddings):
                image_embeddings_dict[idx] = emb
        
        # Reconstruct embeddings in the same order as all_chunks
        ordered_embeddings = []
        for i in range(len(all_chunks)):
            if i in text_embeddings_dict:
                ordered_embeddings.append(text_embeddings_dict[i])
            elif i in image_embeddings_dict:
                ordered_embeddings.append(image_embeddings_dict[i])
        
        np_embeddings = np.array(ordered_embeddings, dtype=np.float32)
    
    st.success("Embeddings generated successfully! (separate text/image embeddings computed)")

    # Store in FAISS (inner product for cosine)
    index, dim = build_faiss_index(np_embeddings)
    # Build separate FAISS indices for text and images (if available)
    text_index = None
    image_index = None
    if text_embeddings_array is not None and len(text_embeddings_array) > 0:
        text_index, _ = build_faiss_index(text_embeddings_array)
    if image_embeddings_array is not None and len(image_embeddings_array) > 0:
        image_index, _ = build_faiss_index(image_embeddings_array)

    st.success("Stored in FAISS successfully!")
    
    # Graph visualization settings in sidebar
    with st.sidebar:
        st.subheader("📊 Graph Settings")
        graph_threshold = st.slider(
            "Graph Edge Threshold",
            min_value=0.1,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Minimum similarity score to show an edge between nodes"
        )
    
    # Build similarity matrix (only for visualization, can be slow with many items)
    if len(all_chunks) <= 100:  # Limit graph size for performance
        with st.spinner("Building similarity matrix and graph..."):
            if np_embeddings.size == 0 or np_embeddings.shape[0] == 0:
                st.warning("No embeddings to visualize.")
                html_content = None
                edge_count = 0
            else:
                similarity_matrix = compute_similarity_matrix(np_embeddings)
                n_items = similarity_matrix.shape[0]
                if n_items < 2:
                    max_sim = min_sim = mean_sim = 0.0
                    edge_count = 0
                else:
                    triu_indices = np.triu_indices(n_items, k=1)
                    upper_vals = similarity_matrix[triu_indices]
                    if upper_vals.size == 0:
                        max_sim = min_sim = mean_sim = 0.0
                        edge_count = 0
                    else:
                        max_sim = float(np.max(upper_vals))
                        min_sim = float(np.min(upper_vals))
                        mean_sim = float(np.mean(upper_vals))
                        edge_count = int(np.sum(upper_vals > graph_threshold))
                if n_items >= 2:
                    st.info(f"📊 Similarity stats: Min={min_sim:.3f}, Max={max_sim:.3f}, Mean={mean_sim:.3f}")
            
                if edge_count == 0 and n_items > 1:
                    st.warning(f"⚠️ No edges found above threshold {graph_threshold:.2f}. Try lowering it to see connections.")
            
            if np_embeddings.size != 0:
                graph = build_graph(similarity_matrix, all_chunks, chunk_metadata, threshold=graph_threshold)
            
            # Verify nodes and edges were added to the network
            try:
                node_count_check = len(graph.nodes)
                edge_count_check = len(graph.edges)
                st.debug(f"Graph object has {node_count_check} nodes and {edge_count_check} edges")
            except:
                pass
            
            # Generate HTML in-memory to avoid IO/timing issues
            try:
                html_content = graph.generate_html()
            except Exception as e:
                st.error(f"Error generating graph HTML: {str(e)}")
                html_content = None
            else:
                if len(html_content or "") < 100:
                    st.error("⚠️ Generated HTML content seems too short. Check PyVis installation.")
                    html_content = None
                # Do not inject any on-load scripts that could trigger movement

        st.subheader("🕸️ Knowledge Graph Visualization")
        st.caption(f"Showing {len(all_chunks)} nodes ({text_count} text, {image_count} images) with {edge_count} edges (similarity > {graph_threshold:.2f})")
        
        # Display the graph with better error handling
        if html_content and len(html_content) > 100:
            try:
                st.components.v1.html(html_content, height=700, width=None, scrolling=True)
            except Exception as e:
                st.error(f"❌ Error displaying graph: {str(e)}")
                st.warning("💡 Troubleshooting tips:")
                st.markdown("- Lower the threshold in the sidebar (try 0.3-0.5)")
                st.markdown("- Ensure you have at least 2 chunks to visualize")
                st.markdown("- Check browser console for JavaScript errors")
        else:
            st.error("⚠️ Failed to generate graph HTML. Check that nodes were added correctly.")
    else:
        st.info(f"Graph visualization skipped (too many items: {len(all_chunks)}). Focus on search functionality.")

    # Search interface
    st.divider()
    st.subheader("🔍 Search")
    
    # Sidebar controls for search
    with st.sidebar:
        st.subheader("🔧 Search Settings")
        search_modality = st.selectbox(
            "Search Modality",
            options=["All", "Text only", "Image only"],
            index=0,
            help="Choose which embeddings to search over"
        )
        similarity_threshold = st.slider(
            "Minimum Similarity Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.20, 
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
        # Route to the selected modality
        if search_modality == "Text only":
            if text_index is None:
                st.warning("No text embeddings available to search.")
                st.stop()
            D_tmp, I_tmp = search_index(text_index, q_emb, k=max_results, min_similarity=similarity_threshold)
            # Map indices from text batch back to original all_chunks indices
            mapped_indices = [text_indices[i] for i in I_tmp[0]] if I_tmp.size > 0 else []
            D = np.array([D_tmp[0][:len(mapped_indices)]]) if len(mapped_indices) > 0 else np.array([[]])
            I = np.array([mapped_indices], dtype=int) if len(mapped_indices) > 0 else np.array([[]], dtype=int)
        elif search_modality == "Image only":
            if image_index is None:
                st.warning("No image embeddings available to search.")
                st.stop()
            D_tmp, I_tmp = search_index(image_index, q_emb, k=max_results, min_similarity=similarity_threshold)
            # Map indices from image batch back to original all_chunks indices
            mapped_indices = [image_indices[i] for i in I_tmp[0]] if I_tmp.size > 0 else []
            D = np.array([D_tmp[0][:len(mapped_indices)]]) if len(mapped_indices) > 0 else np.array([[]])
            I = np.array([mapped_indices], dtype=int) if len(mapped_indices) > 0 else np.array([[]], dtype=int)
        else:
            # All embeddings (combined)
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
