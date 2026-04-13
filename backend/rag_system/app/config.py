"""
Configuration constants for the RAG system.
"""

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2 embeddings

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval Configuration
DEFAULT_K = 5  # Number of chunks to retrieve

# Generation Configuration
GENERATOR_MODEL = "gpt2"  # Placeholder, can be swapped later
MAX_NEW_TOKENS = 256

# Data Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_INDEX_DIR = "data/index"

# FAISS Index File
FAISS_INDEX_FILE = "faiss_index.bin"
DATASET_FILE = "dataset.huggingface"
