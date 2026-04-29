"""
Preprocessing script for the 20 Newsgroups dataset.

Usage:
    uv run python scripts/preprocess.py [OPTIONS]

Options:
    --mode          Export mode: "by_document" or "by_category"  (default: by_category)
    --subset        Dataset split: "train", "test", or "all"     (default: all)
    --categories    Comma-separated list of category names       (default: all 20)
    --out-dir       Output directory for .txt files              (default: data/raw)
    --min-chars     Minimum characters to keep a post           (default: 100)
    --max-per-cat   Max posts per category (by_document only)   (default: no limit)
    --no-summary    Skip writing dataset_summary.json

Examples:
    # Export all categories as one file each (default)
    uv run python scripts/preprocess.py

    # Export individual posts, 200 per category, to data/raw
    uv run python scripts/preprocess.py --mode by_document --max-per-cat 200

    # Export only science categories
    uv run python scripts/preprocess.py --categories sci.space,sci.med,sci.crypt

    # Export to rag_system's data/raw
    uv run python scripts/preprocess.py --out-dir ../rag_system/data/raw
"""

import argparse
import sys
from pathlib import Path

# Allow running from both scripts/ and project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.loader import load_20newsgroups, get_category_names
from app.cleaner import clean_dataset, filter_short
from app.exporter import export_by_document, export_by_category, export_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess the 20 Newsgroups dataset into .txt files."
    )
    parser.add_argument(
        "--mode",
        choices=["by_document", "by_category"],
        default="by_category",
        help="Export mode: one file per post or one file per category.",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "test", "all"],
        default="all",
        help="Which split of the dataset to use.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated category names (default: all 20).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/raw",
        help="Output directory for .txt files.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Minimum characters per document after cleaning.",
    )
    parser.add_argument(
        "--max-per-cat",
        type=int,
        default=None,
        help="Maximum posts per category (by_document mode only).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip writing dataset_summary.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
        valid = set(get_category_names())
        bad = [c for c in categories if c not in valid]
        if bad:
            print(f"Unknown categories: {bad}")
            print(f"Valid options: {sorted(valid)}")
            sys.exit(1)

    print("=" * 62)
    print("  DATA PREPROCESSING - 20 Newsgroups")
    print("=" * 62)
    print(f"  Subset    : {args.subset}")
    print(f"  Mode      : {args.mode}")
    print(f"  Output    : {args.out_dir}")
    print(f"  Min chars : {args.min_chars}")
    if categories:
        print(f"  Categories: {categories}")
    else:
        print(f"  Categories: all 20")
    print()

    # Step 1: Load
    print("[1/4] Loading dataset from sklearn...")
    dataset = load_20newsgroups(categories=categories, subset=args.subset)
    print(f"      Loaded {len(dataset['data'])} posts across {len(dataset['target_names'])} categories.")

    # Step 2: Clean
    print("[2/4] Cleaning and normalizing text...")
    docs = clean_dataset(dataset)

    # Step 3: Filter
    print(f"[3/4] Filtering posts shorter than {args.min_chars} chars...")
    before = len(docs)
    docs = filter_short(docs, min_chars=args.min_chars)
    removed = before - len(docs)
    print(f"      Kept {len(docs)} posts  (removed {removed} too-short posts).")

    # Step 4: Export
    print(f"[4/4] Exporting to '{args.out_dir}' using mode='{args.mode}'...")
    if args.mode == "by_document":
        written = export_by_document(docs, args.out_dir, max_docs_per_category=args.max_per_cat)
    else:
        written = export_by_category(docs, args.out_dir)

    print(f"      Wrote {len(written)} file(s).")

    if not args.no_summary:
        summary_path = export_summary(docs, args.out_dir)
        print(f"      Summary written to {summary_path}.")

    print()
    print("=" * 62)
    print("  Done! Files are ready for the RAG system.")
    print("=" * 62)


if __name__ == "__main__":
    main()
