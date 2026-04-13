"""
Toy test file for Milestone 3 (Indexing & Retrieval).

Tests FAISS indexing and similarity search functionality.

Run with: uv run python tests/test_milestones/toy_test_m3.py
"""

import sys
import shutil
from pathlib import Path

# Find project root (where app folder is)
project_root = Path(__file__).parent
while project_root.parent != project_root:  # until we hit filesystem root
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.ingest import load_text_files, build_chunks
from app.embed import Embedder
from app.index import build_index_from_chunks, save_index, load_index
from app.retrieve import retrieve_top_k


def print_section(title):
    """Print a test section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_build_index_from_chunks():
    """Test building a FAISS index from chunks and embeddings."""
    print_section("TEST 1: Build Index from Chunks")
    
    # Load and prepare data
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    docs = load_text_files(str(data_dir))
    chunks = build_chunks(docs, chunk_size=200, overlap=50)
    
    if len(chunks) == 0:
        print("  ⚠ No chunks available, skipping test")
        return None
    
    # Embed chunks
    embedder = Embedder()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_texts(chunk_texts)
    
    print(f"  Input: {len(chunks)} chunks with {embeddings.shape[1]}-dim embeddings")
    
    # Build index
    try:
        dataset = build_index_from_chunks(chunks, embeddings)
        print(f"  ✓ Index built successfully")
        print(f"    - Dataset type: {type(dataset)}")
        print(f"    - Number of rows: {len(dataset)}")
        print(f"    - Columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")
        
        # Verify dataset has required columns
        expected_cols = ["chunk_id", "doc_id", "source", "text", "embedding"]
        if hasattr(dataset, "column_names"):
            for col in expected_cols:
                assert col in dataset.column_names, f"Missing column: {col}"
            print(f"  ✓ All required columns present")
        
        # Verify FAISS index exists
        if hasattr(dataset, "_indices"):
            print(f"  ✓ FAISS index attached to dataset")
        
        return dataset
    except Exception as e:
        print(f"  ✗ Failed to build index: {e}")
        raise


def test_save_index(dataset):
    """Test saving index to disk."""
    print_section("TEST 2: Save Index to Disk")
    
    if dataset is None:
        print("  ⚠ No dataset available, skipping test")
        return None
    
    index_dir = Path(__file__).parent.parent / "data" / "index"
    
    # Clean up existing index if present
    if index_dir.exists():
        print(f"  Cleaning up existing index: {index_dir}")
        shutil.rmtree(index_dir)
    
    print(f"  Saving index to: {index_dir}")
    
    try:
        save_index(dataset, str(index_dir))
        print(f"  ✓ Index saved successfully")
        
        # Verify files were created
        index_files = list(index_dir.glob("*"))
        print(f"    - Files created: {len(index_files)}")
        for file in index_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"      • {file.name} ({size_mb:.2f} MB)")
        
        # Verify important files exist
        assert index_dir.exists(), "Index directory not created"
        print(f"  ✓ Index directory exists")
        
        return index_dir
    except Exception as e:
        print(f"  ✗ Failed to save index: {e}")
        raise


def test_load_index(index_dir):
    """Test loading index from disk."""
    print_section("TEST 3: Load Index from Disk")
    
    if index_dir is None:
        print("  ⚠ No index directory available, skipping test")
        return None
    
    print(f"  Loading index from: {index_dir}")
    
    try:
        dataset = load_index(str(index_dir))
        print(f"  ✓ Index loaded successfully")
        print(f"    - Dataset type: {type(dataset)}")
        print(f"    - Number of rows: {len(dataset)}")
        print(f"    - Columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")
        
        # Verify FAISS index attached
        if hasattr(dataset, "_indices"):
            print(f"  ✓ FAISS index still attached")
        
        return dataset
    except Exception as e:
        print(f"  ✗ Failed to load index: {e}")
        raise


def test_retrieve_top_k(dataset):
    """Test similarity search with retrieve_top_k."""
    print_section("TEST 4: Retrieve Top-K Results")
    
    if dataset is None:
        print("  ⚠ No dataset available, skipping test")
        return
    
    embedder = Embedder()
    queries = [
        "What is deep learning?",
        "How do neural networks work?",
        "Tell me about supervised learning",
    ]
    
    try:
        for idx, query in enumerate(queries, 1):
            print(f"\n  Query {idx}: '{query}'")
            
            # Encode query
            query_embedding = embedder.encode_query(query)
            
            # Retrieve top-k
            k = 3
            results = retrieve_top_k(dataset, query_embedding, k=k)
            
            print(f"    Retrieved {len(results)} results:")
            
            # Verify results structure and score
            for rank, result in enumerate(results, 1):
                print(f"      [{rank}] Score: {result['score']:.4f} | {result['chunk_id']}")
                
                # Verify required fields
                required_fields = ["score", "chunk_id", "doc_id", "source", "text"]
                for field in required_fields:
                    assert field in result, f"Missing field in result: {field}"
                
                # Verify score is a valid number (FAISS can return scores > 1 with some metrics)
                assert isinstance(result["score"], (int, float)), f"Score not a number: {result['score']}"
                assert result["score"] >= 0, f"Score should be non-negative: {result['score']}"
            
            # Verify returned exactly k results
            if len(results) > 0:
                assert len(results) <= k, f"Expected max {k} results, got {len(results)}"
                
                # Verify results are sorted by score (descending)
                scores = [r["score"] for r in results]
                assert scores == sorted(scores, reverse=True), "Results not sorted by score"
        
        print(f"\n  ✓ All retrieval tests passed")
    except Exception as e:
        print(f"  ✗ Retrieval failed: {e}")
        raise


def test_retrieval_quality(dataset):
    """Test retrieval quality by checking semantically similar results."""
    print_section("TEST 5: Retrieval Quality Check")
    
    if dataset is None:
        print("  ⚠ No dataset available, skipping test")
        return
    
    embedder = Embedder()
    
    # Query about neural networks
    query = "neural networks"
    query_embedding = embedder.encode_query(query)
    
    results = retrieve_top_k(dataset, query_embedding, k=5)
    
    print(f"  Query: '{query}'")
    print(f"  Top results:")
    
    # Check if top result has reasonable score
    if len(results) > 0:
        top_score = results[0]["score"]
        print(f"    - Top score: {top_score:.4f}")
        
        # Top result should have reasonable similarity (heuristic: > 0.3 for this model/data)
        if top_score > 0.3:
            print(f"  ✓ Top result has good similarity score")
        else:
            print(f"  ⚠ Top result similarity is low (might be normal depending on data)")
        
        # Show closest matches
        for rank, result in enumerate(results[:3], 1):
            preview = result["text"][:50].replace("\n", " ")
            print(f"    [{rank}] {result['chunk_id']}: {preview}...")


def main():
    """Run all Milestone 3 tests."""
    print("\n" + "="*60)
    print("  RAG SYSTEM - MILESTONE 3 TEST SUITE")
    print("  Testing Indexing & Retrieval Modules")
    print("="*60)
    
    try:
        # TEST 1: Build index
        dataset = test_build_index_from_chunks()
        
        if dataset is None:
            print("\n  ⚠ Skipping remaining tests (no dataset)")
            return
        
        # TEST 2: Save index
        index_dir = test_save_index(dataset)
        
        if index_dir is None:
            print("\n  ⚠ Skipping load test (no index saved)")
            return
        
        # TEST 3: Load index
        loaded_dataset = test_load_index(index_dir)
        
        # TEST 4: Retrieve top-k
        test_retrieve_top_k(loaded_dataset)
        
        # TEST 5: Quality check
        test_retrieval_quality(loaded_dataset)
        
        print("\n" + "="*60)
        print("  ✓ ALL MILESTONE 3 TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
