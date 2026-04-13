"""
Document ingestion and chunking module.

Functions:
- load_text_files(data_dir): Load raw text documents from directory
- chunk_text(text, chunk_size, overlap): Split text into overlapping chunks
- build_chunks(docs, chunk_size, overlap): Build chunk dataset from documents
"""

import os
from pathlib import Path


def load_text_files(data_dir):
    """
    Load raw text files from a directory.
    
    Args:
        data_dir (str): Path to directory containing .txt files
        
    Returns:
        list: List of dicts with keys: {doc_id, source, text}
    """
    docs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Directory {data_dir} does not exist")
        return docs
    
    # Find all .txt files
    txt_files = sorted(data_path.glob("*.txt"))
    
    for idx, file_path in enumerate(txt_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            doc = {
                "doc_id": f"doc_{idx}",
                "source": str(file_path),
                "text": text
            }
            docs.append(doc)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return docs


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between consecutive chunks
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    step = chunk_size - overlap
    
    # Generate chunks with overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        
        # Stop if we've reached the end
        if i + chunk_size >= len(text):
            break
    
    return chunks


def build_chunks(docs, chunk_size=500, overlap=100):
    """
    Build a dataset of chunks from documents.
    
    Args:
        docs (list): List of dicts with keys: {doc_id, source, text}
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        list: List of dicts with keys: {chunk_id, doc_id, source, text}
    """
    chunks = []
    
    for doc in docs:
        doc_id = doc["doc_id"]
        source = doc["source"]
        text = doc["text"]
        
        # Chunk the document
        text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        # Build chunk records
        for chunk_idx, chunk_text_val in enumerate(text_chunks):
            chunk_record = {
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,
                "source": source,
                "text": chunk_text_val
            }
            chunks.append(chunk_record)
    
    return chunks
