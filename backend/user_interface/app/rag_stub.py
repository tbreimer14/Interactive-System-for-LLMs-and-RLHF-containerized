"""
rag_stub.py

Stub for the RAG pipeline.

RIGHT NOW: returns mock retrieved chunks and a mock response.
LATER: replace the body of answer() with the real RAG call, e.g.:
    from rag_system.pipeline import answer as rag_answer
    return rag_answer(query, k=k)

The function signature must stay the same so nothing else in the system breaks.
"""

import random


def answer(query: str, k: int = 3) -> dict:
    """
    Run the RAG pipeline for a given query and return the result.

    Args:
        query: the user's question or prompt
        k:     number of chunks to retrieve (default 3)

    Returns:
        {
            "query":     the original query string,
            "retrieved": list of k chunk dicts, each with "text" and "source",
            "answer":    the generated response string
        }

    --- STUB BEHAVIOUR ---
    Seeds the RNG with a hash of the query so the same query always returns
    the same mock output. Swap this body with a real RAG call when ready.
    """
    rng = random.Random(hash(query))

    mock_chunks = [
        {
            "text": (
                f"[Mock chunk {i + 1}] Retrieved passage related to: "
                f"'{query[:50]}'. This is placeholder content — "
                "wire in the real retriever to see actual chunks."
            ),
            "source": f"doc_{rng.randint(1, 20)}.txt",
        }
        for i in range(k)
    ]

    mock_answer = (
        f"[Mock response] Generated answer for: '{query}'. "
        "This is placeholder text — replace rag_stub.answer() "
        "with your real rag_system.pipeline.answer() call."
    )

    return {
        "query": query,
        "retrieved": mock_chunks,
        "answer": mock_answer,
    }
