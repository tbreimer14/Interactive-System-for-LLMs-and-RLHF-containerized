"""
Verification script — checks that preprocessed .txt files are valid and ready
to be fed into the RAG system.

Usage:
    uv run python scripts/verify.py [--data-dir data/raw]

Checks:
    1. At least one .txt file exists
    2. No file is empty
    3. No file is excessively large (>10 MB, likely an accidental dump)
    4. UTF-8 readable
    5. Prints a summary table: filename, size (chars), first 80 chars preview
"""

import argparse
import sys
from pathlib import Path


MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


def parse_args():
    parser = argparse.ArgumentParser(description="Verify preprocessed data files.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing .txt files to verify.",
    )
    return parser.parse_args()


def verify(data_dir):
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        print("       Run scripts/preprocess.py first.")
        sys.exit(1)

    txt_files = sorted(data_path.glob("*.txt"))

    if not txt_files:
        print(f"ERROR: No .txt files found in {data_dir}")
        print("       Run scripts/preprocess.py first.")
        sys.exit(1)

    print("=" * 70)
    print("  DATA VERIFICATION")
    print("=" * 70)
    print(f"  Directory : {data_dir}")
    print(f"  Files     : {len(txt_files)}")
    print()

    errors = []
    total_chars = 0

    print(f"  {'File':<35} {'Chars':>8}  Preview")
    print(f"  {'-'*35} {'-'*8}  {'-'*20}")

    for fpath in txt_files:
        byte_size = fpath.stat().st_size

        # Check: too large
        if byte_size > MAX_FILE_BYTES:
            errors.append(f"  WARN  {fpath.name}: file is {byte_size // 1024} KB (>10 MB)")

        # Check: empty
        if byte_size == 0:
            errors.append(f"  ERROR {fpath.name}: file is empty")
            print(f"  {fpath.name:<35} {'EMPTY':>8}")
            continue

        # Check: UTF-8 readable
        try:
            text = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            errors.append(f"  ERROR {fpath.name}: UTF-8 decode failed — {e}")
            print(f"  {fpath.name:<35} {'DECODE ERR':>8}")
            continue

        char_count = len(text)
        total_chars += char_count
        preview = text[:80].replace("\n", " ").strip()
        if len(text) > 80:
            preview += "..."

        print(f"  {fpath.name:<35} {char_count:>8,}  {preview}")

    print()
    print(f"  Total characters : {total_chars:,}")
    print(f"  Avg chars / file : {total_chars // max(len(txt_files), 1):,}")
    print()

    if errors:
        print("  ISSUES FOUND:")
        for e in errors:
            print(f"    {e}")
        print()
        sys.exit(1)
    else:
        print("  All files passed verification.")
        print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    verify(args.data_dir)
