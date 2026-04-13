"""
Interactive CLI for querying the RAG system.

Usage:
    uv run python scripts/chat_cli.py <index_dir>

Example:
    uv run python scripts/chat_cli.py data/index

Commands:
    - Type a query to retrieve and answer
    - Type 'exit' or 'quit' to quit
"""

import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import RAGPipeline


def main():
    """Run the interactive chat CLI loop."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/chat_cli.py <index_dir>")
        sys.exit(1)
    
    index_dir = sys.argv[1]
    
    print(f"Loading RAG index from {index_dir}...")
    
    try:
        pipeline = RAGPipeline()
        pipeline.load_index(out_dir=index_dir)
    except Exception as e:
        print(f"✗ Failed to load index: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("RAG Chat CLI")
    print("="*60)
    print("Type a query to retrieve and answer")
    print("Type 'exit' or 'quit' to quit\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            print("\nRetrieving context and generating answer...")
            result = pipeline.answer(query, k=3)
            
            print(f"\n--- Query ---")
            print(f"{result['query']}")
            
            print(f"\n--- Retrieved Context ---")
            for i, chunk in enumerate(result["retrieved"], 1):
                print(f"[{i}] (score: {chunk['score']:.4f})")
                print(f"    {chunk['text'][:100]}..." if len(chunk["text"]) > 100 else f"    {chunk['text']}")
            
            print(f"\n--- Answer ---")
            print(f"{result['answer']}")
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()
