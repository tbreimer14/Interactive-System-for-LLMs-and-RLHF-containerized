"""
Toy test file for Milestone 4 (Generation & Pipeline).

Tests text generation and full RAG pipeline functionality.

Run with: uv run python tests/test_milestones/toy_test_m4.py
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

from app.generate import Generator
from app.pipeline import RAGPipeline


def print_section(title):
    """Print a test section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_generator_initialization():
    """Test Generator class initialization."""
    print_section("TEST 1: Generator Initialization")
    
    print("  Loading Generator (downloads GPT-2 ~500MB first time)...")
    try:
        generator = Generator()
        print(f"  ✓ Generator loaded successfully")
        print(f"    - Model: {generator.model_name}")
        print(f"    - Tokenizer: {type(generator.tokenizer).__name__}")
        print(f"    - Model: {type(generator.model).__name__}")
    except Exception as e:
        print(f"  ✗ Failed to load Generator: {e}")
        raise
    
    return generator


def test_build_prompt(generator):
    """Test prompt building."""
    print_section("TEST 2: Build Prompt")
    
    query = "What is machine learning?"
    chunks = [
        "Machine learning is a subset of AI.",
        "It focuses on learning from data.",
        "Algorithms improve through experience."
    ]
    
    try:
        prompt = generator.build_prompt(query, chunks)
        print(f"  ✓ Prompt built successfully")
        print(f"    - Prompt length: {len(prompt)} characters")
        print(f"    - Query in prompt: {'Question:' in prompt}")
        print(f"    - Context in prompt: {'[Document' in prompt}")
        
        # Verify structure
        assert "Question:" in prompt, "Query not in prompt"
        assert query in prompt, "Query text missing"
        assert "[Document 1]" in prompt, "First chunk not in prompt"
        assert "Instructions:" in prompt, "Instructions missing"
        
        print(f"  ✓ Prompt structure correct")
        
        # Show snippet
        print(f"\n  Prompt preview (first 200 chars):")
        print(f"  {prompt[:200]}...")
    except Exception as e:
        print(f"  ✗ Failed to build prompt: {e}")
        raise
    
    return prompt


def test_generate_text(generator):
    """Test text generation."""
    print_section("TEST 3: Generate Text")
    
    prompt = """Machine learning is a branch of artificial intelligence.
Question: What is machine learning?
Answer:"""
    
    print(f"  Generating text from prompt...")
    try:
        answer = generator.generate(prompt, max_new_tokens=50)
        print(f"  ✓ Text generated successfully")
        print(f"    - Answer length: {len(answer)} characters")
        print(f"    - Answer tokens: ~{len(answer) // 4}")  # rough estimate
        
        # Verify output
        assert isinstance(answer, str), "Answer is not a string"
        assert len(answer) > 0, "Answer is empty"
        
        print(f"  ✓ Answer structure valid")
        
        # Show answer
        print(f"\n  Generated answer:")
        print(f"  {answer}")
    except Exception as e:
        print(f"  ✗ Failed to generate text: {e}")
        raise
    
    return answer


def test_pipeline_initialization():
    """Test RAGPipeline initialization."""
    print_section("TEST 4: Pipeline Initialization")
    
    print("  Initializing RAGPipeline...")
    try:
        pipeline = RAGPipeline()
        print(f"  ✓ Pipeline initialized successfully")
        print(f"    - Embedder: {pipeline.embedder.model_name}")
        print(f"    - Generator: {pipeline.generator.model_name}")
        print(f"    - Dataset: {pipeline.dataset}")
        
        # Verify components
        assert pipeline.embedder is not None, "Embedder not initialized"
        assert pipeline.generator is not None, "Generator not initialized"
        assert pipeline.dataset is None, "Dataset should be None until loaded"
        
        print(f"  ✓ All components initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize pipeline: {e}")
        raise
    
    return pipeline


def test_build_and_answer(pipeline):
    """Test building index and getting answers."""
    print_section("TEST 5: Full Pipeline Answer")
    
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    index_dir = Path(__file__).parent.parent / "data" / "index_m4"
    
    try:
        # Build index
        print(f"  Building index from {data_dir}...")
        pipeline.build_index(str(data_dir), str(index_dir))
        print(f"  ✓ Index built")
        
        # Ask questions
        queries = [
            "What is machine learning?",
            "Tell me about neural networks.",
        ]
        
        for query in queries:
            print(f"\n  Query: '{query}'")
            result = pipeline.answer(query, k=3)
            
            # Verify result structure
            assert "query" in result, "Missing 'query' in result"
            assert "retrieved" in result, "Missing 'retrieved' in result"
            assert "answer" in result, "Missing 'answer' in result"
            
            # Verify retrieved
            assert len(result["retrieved"]) > 0, "No chunks retrieved"
            assert len(result["retrieved"]) <= 3, "Too many chunks retrieved"
            
            # Verify answer
            assert isinstance(result["answer"], str), "Answer is not string"
            assert len(result["answer"]) > 0, "Answer is empty"
            
            print(f"    Retrieved {len(result['retrieved'])} chunks")
            print(f"    Answer length: {len(result['answer'])} chars")
            print(f"    Answer preview: {result['answer'][:80]}...")
        
        print(f"\n  ✓ Full pipeline successful")
    except Exception as e:
        print(f"  ✗ Pipeline failed: {e}")
        raise


def main():
    """Run all Milestone 4 tests."""
    print("\n" + "="*60)
    print("  RAG SYSTEM - MILESTONE 4 TEST SUITE")
    print("  Testing Generation & Pipeline Modules")
    print("="*60)
    
    try:
        # TEST 1: Generator init
        generator = test_generator_initialization()
        
        # TEST 2: Build prompt
        prompt = test_build_prompt(generator)
        
        # TEST 3: Generate text
        test_generate_text(generator)
        
        # TEST 4: Pipeline init
        pipeline = test_pipeline_initialization()
        
        # TEST 5: Full pipeline
        test_build_and_answer(pipeline)
        
        print("\n" + "="*60)
        print("  ✓ ALL MILESTONE 4 TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
