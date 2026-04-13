"""
Embedding module using Sentence Transformers.

Classes:
- Embedder: Handles text embedding with a Sentence Transformers model
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Embedder for converting text to normalized embeddings.
    
    Uses sentence-transformers/all-MiniLM-L6-v2 by default.
    
    Methods:
    - encode_texts(texts, batch_size): Encode multiple texts
    - encode_query(query): Encode a single query
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the Embedder.
        
        Args:
            model_name (str): Name of the sentence-transformers model.
                             Defaults to all-MiniLM-L6-v2
        """
        if model_name is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode_texts(self, texts, batch_size=32):
        """
        Encode a batch of texts to normalized embeddings.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for encoding
            
        Returns:
            np.ndarray: Array of normalized embeddings (n_samples, embedding_dim)
        """
        # Encode with show_progress_bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings
    
    def encode_query(self, query):
        """
        Encode a single query string.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Normalized embedding vector
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding
