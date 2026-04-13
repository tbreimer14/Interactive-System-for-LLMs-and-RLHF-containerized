"""
FAISS indexing module.

Functions:
- build_index_from_chunks(chunks, embeddings): Create HF Dataset with FAISS index
- save_index(dataset, index_dir): Save dataset and FAISS index to disk
- load_index(index_dir): Load dataset and FAISS index from disk
"""

from pathlib import Path
import datasets
import numpy as np


def build_index_from_chunks(chunks, embeddings):
    """
    Build a Hugging Face Dataset with FAISS index from chunks and embeddings.
    
    Args:
        chunks (list): List of dicts with chunk data
        embeddings (np.ndarray): Matrix of embeddings (n_chunks, embedding_dim)
        
    Returns:
        datasets.Dataset: HF Dataset with FAISS index on embeddings column
    """
    # Create dataset from chunks
    dataset = datasets.Dataset.from_dict({
        "chunk_id": [c["chunk_id"] for c in chunks],
        "doc_id": [c["doc_id"] for c in chunks],
        "source": [c["source"] for c in chunks],
        "text": [c["text"] for c in chunks],
        "embedding": embeddings
    })
    
    # Add FAISS index for fast similarity search
    dataset.add_faiss_index(column="embedding")
    
    return dataset


def save_index(dataset, index_dir):
    """
    Save the dataset and FAISS index to disk.
    
    Args:
        dataset (datasets.Dataset): HF Dataset with FAISS index
        index_dir (str): Path to save index files
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Drop FAISS index before saving (will re-attach on load)
    dataset.drop_index("embedding")
    
    # Save the dataset
    dataset.save_to_disk(str(index_path))


def load_index(index_dir):
    """
    Load the dataset and FAISS index from disk.
    
    Args:
        index_dir (str): Path to index directory
        
    Returns:
        datasets.Dataset: HF Dataset with loaded FAISS index
    """
    dataset = datasets.load_from_disk(str(index_dir))
    
    # Ensure embedding column exists
    if "embedding" not in dataset.column_names:
        raise ValueError("Embedding column not found in loaded dataset")
    
    # Re-attach FAISS index (was dropped before saving)
    dataset.add_faiss_index(column="embedding")
    
    return dataset
