# Data Preprocessing Documentation

## Overview

This module downloads, cleans, and exports the **20 Newsgroups** dataset from `sklearn` into plain `.txt` files that the RAG system can ingest directly.

### Architecture

```
sklearn.datasets.fetch_20newsgroups
    ↓
app/loader.py  — fetch posts (headers/footers/quotes stripped)
    ↓
app/cleaner.py — normalize whitespace, remove divider lines
    ↓
filter_short() — drop posts < 100 chars
    ↓
app/exporter.py — write .txt files (by_document or by_category mode)
    ↓
data/raw/*.txt  ← ready for RAG system's build_index.py
```

---

## System Components

### 1. **Loader** (`app/loader.py`)

**Purpose:** Fetch the 20 Newsgroups dataset from sklearn with optional filtering.

**Key Functions:**

- `load_20newsgroups(categories, subset, remove)` — returns a dict with keys:
  - `data`: list of raw post strings
  - `target`: list of integer category indices
  - `target_names`: list of category name strings
  - `filenames`: list of original source file paths

- `get_category_names()` — returns the full list of all 20 category strings

**Example:**
```python
from app.loader import load_20newsgroups

dataset = load_20newsgroups(categories=["sci.space", "sci.med"], subset="train")
print(len(dataset["data"]))        # number of posts loaded
print(dataset["target_names"])     # ["sci.med", "sci.space"]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `categories` | list\[str\] \| None | None | Subset of the 20 categories. None = all. |
| `subset` | str | "all" | "train", "test", or "all" |
| `remove` | tuple | ("headers","footers","quotes") | What sklearn strips from each post |

---

### 2. **Cleaner** (`app/cleaner.py`)

**Purpose:** Normalize raw post text before saving to disk.

**Key Functions:**

- `clean_text(text)` — cleans a single post:
  - Removes lines made entirely of `-`, `=`, `*`, `|`, etc. (email dividers)
  - Removes bare attribution lines ("writes:", "wrote:", etc.)
  - Collapses 3+ consecutive blank lines into 2
  - Strips leading/trailing whitespace

- `clean_dataset(dataset)` — applies `clean_text` to every post and returns:
  ```python
  [
      {"text": "...", "category": "sci.space", "category_index": 14},
      ...
  ]
  ```

- `filter_short(docs, min_chars=100)` — drops any doc whose cleaned text is below `min_chars` characters

**Example:**
```python
from app.loader import load_20newsgroups
from app.cleaner import clean_dataset, filter_short

dataset = load_20newsgroups(subset="train")
docs = clean_dataset(dataset)
docs = filter_short(docs, min_chars=100)
print(f"{len(docs)} usable posts")
```

---

### 3. **Exporter** (`app/exporter.py`)

**Purpose:** Write cleaned documents to `.txt` files on disk.

**Key Functions:**

- `export_by_category(docs, out_dir, separator)` *(default mode)*
  - One `.txt` file per category
  - Posts are concatenated with a `\n\n---\n\n` separator
  - File name: `sci_space.txt`, `comp_graphics.txt`, etc.
  - Produces 20 files total (or fewer if `categories` was filtered)

- `export_by_document(docs, out_dir, max_docs_per_category)`
  - One `.txt` file per post
  - File name: `sci_space_0000.txt`, `sci_space_0001.txt`, ...
  - Use `max_docs_per_category` to cap output size

- `export_summary(docs, out_dir)` — writes `dataset_summary.json`:
  ```json
  {
    "total_documents": 18846,
    "total_characters": 12340000,
    "average_chars_per_doc": 655,
    "categories": {
      "alt.atheism": 799,
      "comp.graphics": 1000,
      ...
    }
  }
  ```

- `slugify(name)` — converts `"sci.space"` → `"sci_space"` for safe filenames

**When to use each mode:**

| Mode | Files | Best for |
|---|---|---|
| `by_category` | 20 files (~1 per topic) | Quick RAG test, small index |
| `by_document` | Up to ~18,000 files | Fine-grained retrieval per post |

---

## Configuration

All defaults can be overridden via `scripts/preprocess.py` flags:

| Flag | Default | Description |
|---|---|---|
| `--mode` | `by_category` | `by_document` or `by_category` |
| `--subset` | `all` | `train`, `test`, or `all` |
| `--categories` | all 20 | Comma-separated category names |
| `--out-dir` | `data/raw` | Where to write `.txt` files |
| `--min-chars` | `100` | Minimum chars after cleaning |
| `--max-per-cat` | None | Cap per category (by_document only) |
| `--no-summary` | False | Skip `dataset_summary.json` |

---

## Usage

### Option 1: Default export (20 category files)

```bash
uv run python scripts/preprocess.py
```

Output: `data/raw/` with 20 `.txt` files + `dataset_summary.json`

---

### Option 2: Per-document export with limits

```bash
uv run python scripts/preprocess.py --mode by_document --max-per-cat 200
```

Output: up to 200 files per category (4,000 files total for all 20 categories)

---

### Option 3: Selected categories only

```bash
uv run python scripts/preprocess.py --categories sci.space,sci.med,sci.crypt
```

Output: 3 `.txt` files — one per category

---

### Option 4: Direct output to RAG system

```bash
uv run python scripts/preprocess.py --out-dir ../rag_system/data/raw
```

After this, run the RAG index builder directly:

```bash
cd ../rag_system
uv run python scripts/build_index.py data/raw data/index
```

---

### Option 5: Programmatic use

```python
from app.loader import load_20newsgroups
from app.cleaner import clean_dataset, filter_short
from app.exporter import export_by_category, export_summary

dataset = load_20newsgroups(categories=["sci.space"], subset="train")
docs = clean_dataset(dataset)
docs = filter_short(docs, min_chars=100)
written = export_by_category(docs, out_dir="data/raw")
export_summary(docs, out_dir="data/raw")

print(f"Wrote {len(written)} files")
```

---

### Option 6: Verify output

```bash
uv run python scripts/verify.py
# or specify a different directory:
uv run python scripts/verify.py --data-dir ../rag_system/data/raw
```

Sample output:
```
======================================================================
  DATA VERIFICATION
======================================================================
  Directory : data/raw
  Files     : 20

  File                                Chars  Preview
  --------------------------------- --------  --------------------
  alt_atheism.txt                   412,831  In article <...> the claim is made that...
  comp_graphics.txt                 503,200  Does anyone know of a good JPEG library...
  ...

  Total characters : 9,845,032
  Avg chars / file : 492,251

  All files passed verification.
```

---

## Testing

```bash
# Milestone 1 — Loader (downloads dataset ~15 MB on first run)
uv run python tests/test_milestones/toy_test_m1.py

# Milestone 2 — Cleaner (synthetic data, instant)
uv run python tests/test_milestones/toy_test_m2.py

# Milestone 3 — Exporter (uses temp directories, instant)
uv run python tests/test_milestones/toy_test_m3.py
```

**First run:** ~30 seconds (dataset download ~15 MB, cached afterwards)
**Subsequent runs:** ~5 seconds

---

## Integration with RAG System

The RAG system's `ingest.py` reads all `*.txt` files from a directory:

```python
# rag_system/app/ingest.py
docs = load_text_files("data/raw/")  # reads *.txt
chunks = build_chunks(docs, chunk_size=500, overlap=100)
```

**Recommended workflow:**

```bash
# 1. Preprocess 20newsgroups → data/raw/
cd backend/data
uv run python scripts/preprocess.py --out-dir ../rag_system/data/raw

# 2. Verify output
uv run python scripts/verify.py --data-dir ../rag_system/data/raw

# 3. Build RAG index
cd ../rag_system
uv run python scripts/build_index.py data/raw data/index

# 4. Query the system
uv run python scripts/chat_cli.py data/index
```

---

## Files Overview

```
backend/data/
├── app/
│   ├── __init__.py
│   ├── loader.py        # fetch_20newsgroups wrapper
│   ├── cleaner.py       # text normalization + filtering
│   └── exporter.py      # write .txt files to disk
├── scripts/
│   ├── preprocess.py    # CLI: run the full pipeline
│   └── verify.py        # CLI: validate output files
├── tests/
│   └── test_milestones/
│       ├── toy_test_m1.py   # loader tests (needs dataset download)
│       ├── toy_test_m2.py   # cleaner tests (synthetic data)
│       └── toy_test_m3.py   # exporter tests (temp dirs)
├── data/
│   ├── raw/             # output .txt files (populated by preprocess.py)
│   └── processed/       # reserved for future intermediate outputs
├── DATA_README.md
└── DATA_DOCUMENTATION.md
```

---

## Troubleshooting

### Problem: `ModuleNotFoundError: sklearn`
**Solution:** Install scikit-learn:
```bash
uv add scikit-learn
```

### Problem: Slow first run
**Solution:** sklearn downloads ~15 MB on first call to `fetch_20newsgroups`. It caches to `~/scikit_learn_data/` automatically.

### Problem: Empty `.txt` files
**Solution:** Some posts are completely stripped by sklearn's `remove=("headers","footers","quotes")`. These are filtered out by `filter_short(min_chars=100)` before export.

### Problem: Too many files in `by_document` mode
**Solution:** Use `--max-per-cat` to limit output:
```bash
uv run python scripts/preprocess.py --mode by_document --max-per-cat 100
```

### Problem: Characters outside ASCII in output
**Solution:** All files are written as UTF-8. Downstream systems should open them with `encoding="utf-8"` — the RAG `ingest.py` already does this.

---

## Dataset Attribution

- **20 Newsgroups**: Ken Lang, originally collected for the paper *"Newsweeder: Learning to filter netnews"* (1995)
- Distributed via [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
- License: original posts are public Usenet content
