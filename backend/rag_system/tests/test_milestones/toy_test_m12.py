"""
Toy test file for iterative testing during development.

Run with: uv run python -m pytest tests/toy_test.py -v
Or:       uv run python tests/toy_test.py
"""

import sys
from pathlib import Path

# Find project root (where app folder is)
project_root = Path(__file__).parent
while project_root.parent != project_root:  # until we hit filesystem root
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.ingest import load_text_files, chunk_text, build_chunks
from app.embed import Embedder


def print_section(title):
    """Print a test section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_chunking_logic():
    """Test the chunk_text function with known input."""
    print_section("TEST 1: Chunking Logic")
    
    # Create a test string we can predict
    text = "A" * 1000  # 1000 A's
    chunk_size = 300
    overlap = 50
    
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    print(f"  Input text length: {len(text)}")
    print(f"  Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"  Number of chunks created: {len(chunks)}")
    
    # Verify each chunk size (except possibly the last)
    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk) == chunk_size, f"Chunk {i} has wrong size: {len(chunk)}"
    
    print(f"  ✓ All chunk sizes correct (except last)")
    print(f"  ✓ Last chunk size: {len(chunks[-1])}")
    
    # Verify overlap
    if len(chunks) > 1:
        overlap_text_1 = chunks[0][-overlap:]
        overlap_text_2 = chunks[1][:overlap:]
        assert overlap_text_1 == overlap_text_2, "Overlap mismatch!"
        print(f"  ✓ Overlap verified between chunks")
    
    return chunks


def test_document_loading():
    """Test load_text_files function."""
    print_section("TEST 2: Document Loading")
    
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"  ⚠ Data directory not found: {data_dir}")
        print(f"  Creating test file...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = data_dir / "toy_test.txt"
        test_content = """Machine Learning Basics

            Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data. 
            There are three main types: supervised learning, unsupervised learning, and reinforcement learning.

            Supervised learning uses labeled data to train models.
            Unsupervised learning finds patterns in unlabeled data.
            Reinforcement learning trains agents through rewards and penalties.

            Deep learning uses neural networks to process data.
            Neural networks are inspired by biological neurons and can learn complex patterns.
            Convolutional neural networks are effective for image processing tasks.
            Recurrent neural networks are good for sequential data like text.

            Training a machine learning model involves several key steps.
            First, you need to prepare your data and split it into training and testing sets.
            Then, you select an appropriate algorithm and model architecture.
            During training, the model learns by minimizing a loss function.
            Finally, you evaluate performance on the test set and iterate to improve results.

            Common machine learning algorithms include decision trees, random forests, support vector machines, and neural networks.
            Deep learning has become increasingly popular due to its effectiveness on large datasets.
            Transfer learning allows us to use pre-trained models and adapt them to new tasks.
            This approach has led to breakthroughs in computer vision and natural language processing."""
        
        test_file.write_text(test_content)
        print(f"  ✓ Created test file: {test_file}")
    
    # Load documents
    docs = load_text_files(str(data_dir))
    print(f"  ✓ Loaded {len(docs)} document(s)")
    
    if len(docs) > 0:
        doc = docs[0]
        print(f"    - doc_id: {doc['doc_id']}")
        print(f"    - source: {doc['source']}")
        print(f"    - text length: {len(doc['text'])} chars")
        print(f"    - preview: {doc['text'][:80]}...")
        
        # Verify structure
        assert "doc_id" in doc, "Missing doc_id"
        assert "source" in doc, "Missing source"
        assert "text" in doc, "Missing text"
        print(f"  ✓ Document structure valid")
    
    return docs


def test_build_chunks():
    """Test the full build_chunks pipeline."""
    print_section("TEST 3: Build Chunks Pipeline")
    
    # Load documents
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    docs = load_text_files(str(data_dir))
    
    if len(docs) == 0:
        print("  ⚠ No documents loaded, skipping test")
        return
    
    # Build chunks
    chunks = build_chunks(docs, chunk_size=200, overlap=50)
    print(f"  ✓ Built {len(chunks)} chunks from {len(docs)} document(s)")
    
    # Verify chunk structure
    if len(chunks) > 0:
        chunk = chunks[0]
        print(f"\n  Sample chunk structure:")
        print(f"    - chunk_id: {chunk['chunk_id']}")
        print(f"    - doc_id: {chunk['doc_id']}")
        print(f"    - source: {chunk['source']}")
        print(f"    - text length: {len(chunk['text'])} chars")
        print(f"    - text preview: {chunk['text'][:60]}...")
        
        # Verify required fields
        required_fields = ["chunk_id", "doc_id", "source", "text"]
        for field in required_fields:
            assert field in chunk, f"Missing field: {field}"
        print(f"\n  ✓ All required fields present")
    
    return chunks


def test_embedder_initialization():
    """Test Embedder class initialization."""
    print_section("TEST 4: Embedder Initialization")
    
    print("  Loading Embedder (first run downloads model ~100MB)...")
    try:
        embedder = Embedder()
        print(f"  ✓ Embedder loaded successfully")
        print(f"    - Model: {embedder.model_name}")
        print(f"    - Model type: {type(embedder.model)}")
    except Exception as e:
        print(f"  ✗ Failed to load Embedder: {e}")
        raise
    
    return embedder


def test_query_embedding(embedder):
    """Test encoding a single query."""
    print_section("TEST 5: Query Embedding")
    
    query = "What is machine learning?"
    print(f"  Query: '{query}'")
    
    try:
        embedding = embedder.encode_query(query)
        print(f"  ✓ Query embedded successfully")
        print(f"    - Shape: {embedding.shape}")
        print(f"    - L2 norm: {(embedding ** 2).sum() ** 0.5:.6f} (should be ~1.0)")
        print(f"    - First 5 values: {embedding[:5]}")
        
        # Verify shape
        assert embedding.shape == (384,), f"Wrong embedding shape: {embedding.shape}"
        print(f"  ✓ Embedding shape correct (384,)")
        
        # Verify normalization (L2 norm should be ~1.0)
        norm = (embedding ** 2).sum() ** 0.5
        assert 0.99 < norm < 1.01, f"Embedding not normalized: {norm}"
        print(f"  ✓ Embedding properly normalized")
    except Exception as e:
        print(f"  ✗ Failed to embed query: {e}")
        raise
    
    return embedding


def test_batch_embedding(embedder):
    """Test encoding multiple texts."""
    print_section("TEST 6: Batch Embedding")
    
    texts = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Supervised learning needs labeled data."
    ]
    
    print(f"  Encoding {len(texts)} texts...")
    try:
        embeddings = embedder.encode_texts(texts, batch_size=2)
        print(f"  ✓ Batch encoded successfully")
        print(f"    - Shape: {embeddings.shape}")
        print(f"    - L2 norm (first): {(embeddings[0] ** 2).sum() ** 0.5:.6f}")
        print(f"    - L2 norm (last): {(embeddings[-1] ** 2).sum() ** 0.5:.6f}")
        
        # Verify shape
        assert embeddings.shape == (len(texts), 384), f"Wrong shape: {embeddings.shape}"
        print(f"  ✓ Embedding shape correct ({len(texts)}, 384)")
        
        # Verify all embeddings are normalized
        for i, emb in enumerate(embeddings):
            norm = (emb ** 2).sum() ** 0.5
            assert 0.99 < norm < 1.01, f"Embedding {i} not normalized: {norm}"
        print(f"  ✓ All embeddings properly normalized")
    except Exception as e:
        print(f"  ✗ Failed to batch embed: {e}")
        raise
    
    return embeddings


def test_full_pipeline():
    """Test the complete ingestion -> embedding pipeline."""
    print_section("TEST 7: Full Pipeline")
    
    try:
        # Load
        data_dir = Path(__file__).parent.parent / "data" / "raw"
        docs = load_text_files(str(data_dir))
        print(f"  ✓ Loaded {len(docs)} documents")
        
        # Chunk
        chunks = build_chunks(docs, chunk_size=200, overlap=50)
        print(f"  ✓ Built {len(chunks)} chunks")
        
        # Embed
        embedder = Embedder()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_texts(chunk_texts)
        print(f"  ✓ Embedded {len(chunks)} chunks")
        print(f"    - Shape: {embeddings.shape}")
        
        # Create records
        records = [
            {
                "chunk": chunks[i],
                "embedding": embeddings[i]
            }
            for i in range(len(chunks))
        ]
        print(f"  ✓ Created {len(records)} records with chunk + embedding")
        print(f"\n  Full pipeline successful! Ready for indexing & retrieval.")
    except Exception as e:
        print(f"  ✗ Pipeline failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  RAG SYSTEM TOY TEST SUITE")
    print("  Testing Ingestion & Embedding Modules")
    print("="*60)
    
    try:
        # Test chunking logic
        test_chunking_logic()
        
        # Test document loading
        test_document_loading()
        
        # Test build_chunks pipeline
        test_build_chunks()
        
        # Test embedder
        embedder = test_embedder_initialization()
        
        # Test query embedding
        test_query_embedding(embedder)
        
        # Test batch embedding
        test_batch_embedding(embedder)
        
        # Test full pipeline
        test_full_pipeline()
        
        print("\n" + "="*60)
        print("  ✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
