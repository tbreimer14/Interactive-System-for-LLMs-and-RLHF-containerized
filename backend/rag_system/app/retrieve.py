"""
Retrieval module for FAISS-backed similarity search.

Functions:
- retrieve_top_k(dataset, query_embedding, k): Retrieve top-k similar chunks
"""

import numpy as np


def retrieve_top_k(dataset, query_embedding, k=5):
    """
    Retrieve top-k chunks most similar to query embedding.
    
    Args:
        dataset (datasets.Dataset): HF Dataset with FAISS index
        query_embedding (np.ndarray): Query embedding vector (embedding_dim,)
        k (int): Number of chunks to retrieve
        
    Returns:
        list: List of dicts with keys:
              {score, chunk_id, doc_id, source, text}
              Sorted by score (highest first)
    """
    # Query the FAISS index
    # get_nearest_examples returns (scores, examples) where examples is a dict of columns
    scores, examples = dataset.get_nearest_examples(
        index_name="embedding",
        query=query_embedding,
        k=k
    )
    
    # Build results with metadata
    results = []
    for i, score in enumerate(scores):
        result = {
            "score": float(score),
            "chunk_id": examples["chunk_id"][i],
            "doc_id": examples["doc_id"][i],
            "source": examples["source"][i],
            "text": examples["text"][i]
        }
        results.append(result)
    
    # Sort by score in descending order (highest first)
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    return results
