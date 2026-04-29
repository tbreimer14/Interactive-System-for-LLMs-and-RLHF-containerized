"""
Export module for saving preprocessed 20 Newsgroups documents to disk.

Two export modes:
- by_document: one .txt file per post  →  data/raw/sci_space_0042.txt
- by_category: one .txt file per category (all posts concatenated)  →  data/raw/sci_space.txt

Functions:
- export_by_document(docs, out_dir, max_docs_per_category): Save one file per post
- export_by_category(docs, out_dir): Save one file per category
- slugify(name): Convert a category name to a safe filename stem
"""

import os
import re
from collections import defaultdict
from pathlib import Path


def slugify(name):
    """
    Convert a category name like 'sci.space' to a safe filename stem 'sci_space'.

    Args:
        name (str): Category name.

    Returns:
        str: Safe filename stem.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def export_by_document(docs, out_dir, max_docs_per_category=None):
    """
    Save each document as a separate .txt file.

    File naming: <category_slug>_<index>.txt
    Example: sci_space_0042.txt

    Args:
        docs (list[dict]): Output of cleaner.clean_dataset() / filter_short().
        out_dir (str): Directory to write files into.
        max_docs_per_category (int | None): Cap on posts saved per category.

    Returns:
        list[str]: Paths of written files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Track per-category counts to enforce max_docs_per_category
    category_counts = defaultdict(int)
    written = []

    for doc in docs:
        category = doc["category"]
        slug = slugify(category)

        if max_docs_per_category is not None:
            if category_counts[slug] >= max_docs_per_category:
                continue

        idx = category_counts[slug]
        filename = f"{slug}_{idx:04d}.txt"
        file_path = out_path / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["text"])

        category_counts[slug] += 1
        written.append(str(file_path))

    return written


def export_by_category(docs, out_dir, separator="\n\n---\n\n"):
    """
    Concatenate all posts per category and save as one .txt file per category.

    File naming: <category_slug>.txt
    Example: sci_space.txt

    Args:
        docs (list[dict]): Output of cleaner.clean_dataset() / filter_short().
        out_dir (str): Directory to write files into.
        separator (str): Text inserted between posts within a file.

    Returns:
        list[str]: Paths of written files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Group posts by category
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc["category"]].append(doc["text"])

    written = []
    for category, texts in sorted(grouped.items()):
        slug = slugify(category)
        file_path = out_path / f"{slug}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(separator.join(texts))

        written.append(str(file_path))

    return written


def export_summary(docs, out_dir):
    """
    Write a summary JSON file listing category counts and total document stats.

    Args:
        docs (list[dict]): Preprocessed document list.
        out_dir (str): Directory to write summary into.

    Returns:
        str: Path of the written summary file.
    """
    import json

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    total_chars = 0
    for doc in docs:
        counts[doc["category"]] += 1
        total_chars += len(doc["text"])

    summary = {
        "total_documents": len(docs),
        "total_characters": total_chars,
        "average_chars_per_doc": round(total_chars / max(len(docs), 1)),
        "categories": dict(sorted(counts.items())),
    }

    summary_path = out_path / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return str(summary_path)
