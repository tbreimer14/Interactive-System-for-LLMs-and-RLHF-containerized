"""
Build FAISS index from raw documents.

Usage:
    uv run python scripts/build_index.py <raw_data_dir> <out_dir>

Example:
    uv run python scripts/build_index.py data/raw data/index
"""

import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import RAGPipeline


def main():
    """Build index from raw documents."""
    if len(sys.argv) < 3:
        print("Usage: python scripts/build_index.py <raw_data_dir> <out_dir>")
        sys.exit(1)
    
    raw_data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    
    print(f"Building index from {raw_data_dir}...")
    
    try:
        pipeline = RAGPipeline()
        pipeline.build_index(raw_data_dir=raw_data_dir, out_dir=out_dir)
        print(f"[OK] Index built and saved to {out_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to build index: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
