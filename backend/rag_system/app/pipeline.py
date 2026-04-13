"""
Main RAG pipeline orchestrator.

Classes:
- RAGPipeline: Orchestrates the full retrieval-augmented generation pipeline
"""

from pathlib import Path
from app.ingest import load_text_files, build_chunks
from app.embed import Embedder
from app.index import build_index_from_chunks, save_index, load_index as load_index_fn
from app.retrieve import retrieve_top_k
from app.generate import Generator


class RAGPipeline:
    """
    Main pipeline for RAG system.
    
    Orchestrates:
    - Embedding
    - Indexing
    - Retrieval
    - Generation
    
    Methods:
    - build_index(raw_data_dir, out_dir): Build index from raw documents
    - load_index(out_dir): Load pre-built index
    - answer(query, k): Get RAG answer with retrieved context
    """
    
    def __init__(self, embedder_model=None, generator_model=None):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedder_model (str): Embedding model name (uses default if None)
            generator_model (str): Generation model name (uses default if None)
        """
        self.embedder = Embedder(model_name=embedder_model)
        self.generator = Generator(model_name=generator_model)
        self.dataset = None
    
    def build_index(self, raw_data_dir="data/raw", out_dir="data/index"):
        """
        Build index from raw documents.
        
        Steps:
        1. Load text files from raw_data_dir
        2. Chunk documents
        3. Embed chunks
        4. Build FAISS index
        5. Save to out_dir
        
        Args:
            raw_data_dir (str): Path to raw data
            out_dir (str): Path to save index
        """
        print(f"Loading documents from {raw_data_dir}...")
        docs = load_text_files(raw_data_dir)
        print(f"  Loaded {len(docs)} documents")
        
        print(f"Building chunks...")
        chunks = build_chunks(docs)
        print(f"  Created {len(chunks)} chunks")
        
        print(f"Embedding chunks...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode_texts(chunk_texts)
        print(f"  Embedded {len(chunks)} chunks")
        
        print(f"Building FAISS index...")
        dataset = build_index_from_chunks(chunks, embeddings)
        print(f"  Index built with {len(dataset)} entries")
        
        print(f"Saving index to {out_dir}...")
        save_index(dataset, out_dir)
        print(f"  Index saved successfully")
        
        # Re-attach FAISS index after saving (save_index drops it)
        dataset.add_faiss_index(column="embedding")
        self.dataset = dataset
    
    def load_index(self, out_dir="data/index"):
        """
        Load pre-built index from disk.
        
        Args:
            out_dir (str): Path to index directory
        """
        print(f"Loading index from {out_dir}...")
        self.dataset = load_index_fn(out_dir)
        print(f"  Index loaded with {len(self.dataset)} entries")
    
    def answer(self, query, k=5):
        """
        Get RAG answer for a query.
        
        Steps:
        1. Embed query
        2. Retrieve top-k chunks
        3. Build prompt with context
        4. Generate answer
        
        Args:
            query (str): User query
            k (int): Number of chunks to retrieve
            
        Returns:
            dict: Keys: {query, retrieved, answer}
                  - query: original query
                  - retrieved: list of retrieved chunks with scores
                  - answer: generated response
        """
        if self.dataset is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        # Embed query
        query_embedding = self.embedder.encode_query(query)
        
        # Retrieve top-k chunks
        retrieved = retrieve_top_k(self.dataset, query_embedding, k=k)
        
        # Extract chunk texts for prompt
        retrieved_texts = [r["text"] for r in retrieved]
        
        # Build prompt
        prompt = self.generator.build_prompt(query, retrieved_texts)
        
        # Generate answer
        answer = self.generator.generate(prompt)
        
        # Return result
        result = {
            "query": query,
            "retrieved": retrieved,
            "answer": answer
        }
        
        return result
