"""
Milestone 1 test: loader.py

Checks:
- load_20newsgroups returns expected keys
- Data, target, target_names, filenames are all non-empty
- target_names contains known categories
- category filtering works correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.loader import load_20newsgroups, get_category_names


def test_load_all():
    print("Test 1: load all categories...")
    dataset = load_20newsgroups(subset="train")
    assert "data" in dataset
    assert "target" in dataset
    assert "target_names" in dataset
    assert "filenames" in dataset
    assert len(dataset["data"]) > 0
    assert len(dataset["data"]) == len(dataset["target"])
    assert len(dataset["target_names"]) == 20
    print(f"  OK — loaded {len(dataset['data'])} training posts across 20 categories.")


def test_load_subset_categories():
    print("Test 2: load a subset of categories...")
    cats = ["sci.space", "sci.med"]
    dataset = load_20newsgroups(categories=cats, subset="train")
    assert set(dataset["target_names"]) == set(cats)
    assert len(dataset["data"]) > 0
    print(f"  OK — loaded {len(dataset['data'])} posts for {cats}.")


def test_get_category_names():
    print("Test 3: get_category_names returns all 20...")
    names = get_category_names()
    assert len(names) == 20
    assert "sci.space" in names
    assert "comp.graphics" in names
    print(f"  OK — {len(names)} category names returned.")


if __name__ == "__main__":
    print("=" * 50)
    print("Milestone 1: Loader Tests")
    print("=" * 50)
    test_load_all()
    test_load_subset_categories()
    test_get_category_names()
    print()
    print("All Milestone 1 tests passed.")
