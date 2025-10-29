from __future__ import annotations

from typing import List, Union
from pathlib import Path
from PIL import Image
import google.generativeai as genai


def generate_response_with_context(
    query: Union[str, Image.Image], 
    relevant_text_chunks: List[str], 
    relevant_image_paths: List[Path],
    api_key: str, 
    model_name: str = "models/gemini-2.5-flash"
) -> str:
    """Generate a response using Gemini with relevant chunks and images as context.

    Args:
        query: User question (text or image).
        relevant_text_chunks: Retrieved text chunks to serve as context.
        relevant_image_paths: Retrieved image paths to serve as context.
        api_key: Google API key.
        model_name: Gemini model to use.
    Returns:
        Model response text or error message.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Prepare content parts: text prompt + images
        content_parts = []
        
        # Add text context
        if relevant_text_chunks:
            context = "\n\n".join([f"Text Chunk {i+1}: {chunk}" for i, chunk in enumerate(relevant_text_chunks)])
            text_prompt = (
                f"Based on the following document chunks and images, please answer: {query if isinstance(query, str) else 'this query'}\n\n"
                f"Document Context:\n{context}\n\n"
            )
            if relevant_image_paths:
                text_prompt += f"I'm also providing {len(relevant_image_paths)} relevant image(s) that match the query.\n\n"
            text_prompt += (
                f"Question/Query: {query if isinstance(query, str) else '[Image query shown in images below]'}\n\n"
                "Please provide a comprehensive answer based on the provided text and images. "
                "If the context doesn't contain enough information to answer, please say so."
            )
            content_parts.append(text_prompt)
        else:
            if isinstance(query, str):
                text_prompt = f"Answer this question based on the provided images: {query}"
            else:
                text_prompt = "Describe what you see in these images and how they relate to the query image."
            content_parts.append(text_prompt)
        
        # Add images (from query if it's an image, then relevant images)
        if isinstance(query, Image.Image):
            content_parts.append(query)
        
        # Add relevant images
        for img_path in relevant_image_paths:
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    content_parts.append(img)
                except Exception:
                    continue
        
        # Generate response
        response = model.generate_content(content_parts)
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"Error generating response: {str(e)}"
