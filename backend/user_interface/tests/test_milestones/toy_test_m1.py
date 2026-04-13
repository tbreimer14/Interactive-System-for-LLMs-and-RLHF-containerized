"""
Toy test for Milestone 1 (RAG Stub).

Tests that rag_stub.answer() returns a valid dict with the expected structure.

Run with: uv run python tests/test_milestones/toy_test_m1.py
"""

import sys
from pathlib import Path

# Walk up from this file until we find the app/ folder (the user_interface root)
project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.rag_stub import answer


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_return_structure():
    """Test that answer() returns a dict with the required keys."""
    print_section("TEST 1: Return structure")

    query = "What is reinforcement learning?"

    try:
        result = answer(query, k=3)
        required_keys = {"query", "retrieved", "answer"}
        missing = required_keys - result.keys()
        assert not missing, f"Missing keys in result: {missing}"
        print(f"  [PASS] Result has all required keys: {required_keys}")
        return result
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_query_passthrough(result, query):
    """Test that the query is passed through unchanged."""
    print_section("TEST 2: Query passthrough")

    try:
        assert result["query"] == query, \
            f"Expected query '{query}', got '{result['query']}'"
        print(f"  [PASS] Query passed through correctly: '{result['query']}'")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_retrieved_chunks(result, k=3):
    """Test that retrieved is a list of k dicts with 'text' and 'source'."""
    print_section("TEST 3: Retrieved chunks")

    try:
        chunks = result["retrieved"]
        assert isinstance(chunks, list), f"'retrieved' should be a list, got {type(chunks)}"
        assert len(chunks) == k, f"Expected {k} chunks, got {len(chunks)}"
        print(f"  [PASS] Retrieved {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            assert "text" in chunk, f"Chunk {i} missing 'text'"
            assert "source" in chunk, f"Chunk {i} missing 'source'"
            assert isinstance(chunk["text"], str), f"Chunk {i} 'text' should be str"
            assert isinstance(chunk["source"], str), f"Chunk {i} 'source' should be str"

        print(f"  [PASS] All chunks have 'text' and 'source' (str)")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_answer_is_string(result):
    """Test that the answer field is a non-empty string."""
    print_section("TEST 4: Answer field")

    try:
        ans = result["answer"]
        assert isinstance(ans, str), f"'answer' should be str, got {type(ans)}"
        assert len(ans) > 0, "'answer' should not be empty"
        print(f"  [PASS] Answer is a non-empty string")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_determinism():
    """Test that the same query always returns the same result."""
    print_section("TEST 5: Determinism (same query -> same output)")

    query = "Determinism test query"

    try:
        r1 = answer(query, k=2)
        r2 = answer(query, k=2)
        assert r1 == r2, "Same query should return identical results"
        print(f"  [PASS] Two calls with the same query returned identical results")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_print_result(result):
    """Print the result so the user can visually confirm it looks right."""
    print_section("TEST 6: Visual output")

    print(f"  Query:  {result['query']}")
    print(f"  Answer: {result['answer'][:80]}...")
    print(f"  Retrieved chunks:")
    for chunk in result["retrieved"]:
        print(f"    [{chunk['source']}] {chunk['text'][:60]}...")
    print(f"\n  [PASS] Output looks correct above")


def main():
    print("\n" + "=" * 60)
    print("  USER INTERFACE - MILESTONE 1 TEST SUITE")
    print("  Testing RAG Stub")
    print("=" * 60)

    QUERY = "What is reinforcement learning?"
    K = 3

    try:
        result = test_return_structure()
        test_query_passthrough(result, QUERY)
        test_retrieved_chunks(result, k=K)
        test_answer_is_string(result)
        test_determinism()
        test_print_result(result)

        print("\n" + "=" * 60)
        print("  ALL MILESTONE 1 TESTS PASSED")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  TEST FAILED: {e}")
        print("=" * 60 + "\n")
        raise


if __name__ == "__main__":
    main()
