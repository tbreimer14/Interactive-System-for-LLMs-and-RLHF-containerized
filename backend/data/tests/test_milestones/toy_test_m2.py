"""
Milestone 2 test: cleaner.py

Checks:
- clean_text strips divider lines and collapses blank lines
- clean_dataset produces correct dict keys
- filter_short removes documents below the threshold
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.cleaner import clean_text, clean_dataset, filter_short


SAMPLE_DATASET = {
    "data": [
        "Hello world!\n\n----------\n\nThis is a test post.\n\nIt has multiple paragraphs.",
        "Short.",
        "A longer post with enough content to pass the filter. " * 5,
    ],
    "target": [0, 1, 0],
    "target_names": ["sci.space", "sci.med"],
}


def test_clean_text_removes_dividers():
    print("Test 1: clean_text removes divider lines...")
    raw = "Content here.\n\n----------\n\nMore content."
    cleaned = clean_text(raw)
    assert "----------" not in cleaned
    assert "Content here." in cleaned
    assert "More content." in cleaned
    print("  OK")


def test_clean_text_collapses_blank_lines():
    print("Test 2: clean_text collapses excessive blank lines...")
    raw = "Line 1\n\n\n\n\nLine 2"
    cleaned = clean_text(raw)
    assert "\n\n\n" not in cleaned
    print("  OK")


def test_clean_dataset_keys():
    print("Test 3: clean_dataset returns correct keys...")
    docs = clean_dataset(SAMPLE_DATASET)
    assert len(docs) == 3
    for doc in docs:
        assert "text" in doc
        assert "category" in doc
        assert "category_index" in doc
    print(f"  OK — {len(docs)} docs with correct keys.")


def test_filter_short():
    print("Test 4: filter_short removes short posts...")
    docs = clean_dataset(SAMPLE_DATASET)
    filtered = filter_short(docs, min_chars=50)
    assert all(len(d["text"]) >= 50 for d in filtered)
    assert len(filtered) < len(docs)
    print(f"  OK — kept {len(filtered)}/{len(docs)} docs after filtering.")


if __name__ == "__main__":
    print("=" * 50)
    print("Milestone 2: Cleaner Tests")
    print("=" * 50)
    test_clean_text_removes_dividers()
    test_clean_text_collapses_blank_lines()
    test_clean_dataset_keys()
    test_filter_short()
    print()
    print("All Milestone 2 tests passed.")
