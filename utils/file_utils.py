from __future__ import annotations

import tempfile
import os
from typing import BinaryIO, List, Tuple
from pathlib import Path


def extract_text(uploaded_file: BinaryIO, use_pdfplumber: bool = True) -> str:
    """Extract text from PDF, DOCX, or TXT with improved parsing for complex PDFs.

    Args:
        uploaded_file: File-like object from Streamlit uploader.
        use_pdfplumber: If True, use pdfplumber for better PDF parsing (tables, layouts).

    Returns:
        Extracted UTF-8 text or empty string if unsupported.
    """
    ext = uploaded_file.name.split(".")[-1].lower()
    
    if ext == "pdf":
        if use_pdfplumber:
            try:
                # pdfplumber works better with file paths or BytesIO
                # Save to temp file for better compatibility
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                import pdfplumber  # type: ignore
                
                text_parts = []
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        # Extract text preserving layout
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        
                        # Extract tables and format as text
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                # Format table as markdown-like structure
                                table_text = "\n".join(
                                    " | ".join(str(cell) if cell else "" for cell in row)
                                    for row in table
                                )
                                text_parts.append(f"\n[Table]\n{table_text}\n[/Table]\n")
                
                # Clean up temp file
                import os
                os.unlink(tmp_path)
                
                return "\n\n".join(text_parts)
            except Exception as e:
                # Fallback to PyPDF2 if pdfplumber fails
                uploaded_file.seek(0)  # Reset file pointer
                return _extract_text_pypdf2(uploaded_file)
        else:
            return _extract_text_pypdf2(uploaded_file)
    
    if ext == "docx":
        import docx  # type: ignore
        uploaded_file.seek(0)  # Reset file pointer
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    
    if ext == "txt":
        uploaded_file.seek(0)  # Reset file pointer
        return uploaded_file.read().decode("utf-8")
    
    return ""


def _extract_text_pypdf2(uploaded_file: BinaryIO) -> str:
    """Fallback PDF extraction using PyPDF2."""
    from PyPDF2 import PdfReader
    uploaded_file.seek(0)  # Reset file pointer
    reader = PdfReader(uploaded_file)
    return "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def extract_images_from_pdf(uploaded_file: BinaryIO, output_dir: Path) -> List[Path]:
    """Extract images from PDF and save them to output directory.

    Args:
        uploaded_file: File-like object from Streamlit uploader.
        output_dir: Directory to save extracted images.
    Returns:
        List of paths to extracted images.
    """
    try:
        import fitz  # PyMuPDF
        uploaded_file.seek(0)
        
        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        doc = fitz.open(tmp_path)
        image_paths = []
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save image
                    image_path = output_dir / f"page_{page_num}_img_{img_index}.{image_ext}"
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    image_paths.append(image_path)
                except Exception:
                    continue
        
        doc.close()
        os.unlink(tmp_path)
        return image_paths
    
    except ImportError:
        # PyMuPDF not available, try pdf2image
        try:
            from pdf2image import convert_from_bytes
            uploaded_file.seek(0)
            images = convert_from_bytes(uploaded_file.read())
            image_paths = []
            
            for page_num, img in enumerate(images):
                image_path = output_dir / f"page_{page_num}.png"
                img.save(image_path)
                image_paths.append(image_path)
            
            return image_paths
        except ImportError:
            return []
    except Exception:
        return []
