# Data Preprocessing

# data pipeline overview
    sklearn fetch_20newsgroups
    ↓
    loader.py loads raw posts (strips headers/footers/quotes)
    ↓
    cleaner.py normalizes text
    (remove dividers, collapse whitespace)
    ↓
    filter_short() drops posts < 100 chars
    ↓
    exporter.py saves to .txt files
    (by_document: one file per post  OR  by_category: one file per category)
    ↓
    data/raw/*.txt  →  ready for RAG system

# dataset info
    - Source   : sklearn.datasets.fetch_20newsgroups
    - Topics   : 20 newsgroup categories (see below)
    - Size     : ~18,000 posts (all splits)
    - Cleaning : headers, footers, and quote lines stripped by sklearn

    Categories:
      alt.atheism               sci.crypt
      comp.graphics             sci.electronics
      comp.os.ms-windows.misc   sci.med
      comp.sys.ibm.pc.hardware  sci.space
      comp.sys.mac.hardware     soc.religion.christian
      comp.windows.x            talk.politics.guns
      misc.forsale              talk.politics.mideast
      rec.autos                 talk.politics.misc
      rec.motorcycles           talk.religion.misc
      rec.sport.baseball        (20 total)
      rec.sport.hockey

# output file formats
    by_category (default):
      → data/raw/sci_space.txt          (all sci.space posts concatenated)
      → data/raw/sci_med.txt
      → ...  (20 files total)

    by_document:
      → data/raw/sci_space_0000.txt     (single post)
      → data/raw/sci_space_0001.txt
      → ...  (up to 18,000 files)

# Implementation
## Milestones

- **Milestone 1:** Dataset loader (`app/loader.py`)
- **Milestone 2:** Text cleaner (`app/cleaner.py`)
- **Milestone 3:** File exporter (`app/exporter.py`)
- **Milestone 4:** Preprocessing script + verification (`scripts/preprocess.py`, `scripts/verify.py`)

---

## Milestone 1: Dataset Loader

Load the 20 Newsgroups dataset with optional category and split filtering.

### What to create

- `app/loader.py` — `load_20newsgroups()` and `get_category_names()`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `loader.py` | `load_20newsgroups()` | categories, subset, remove | dict with data / target / target_names / filenames |
| `loader.py` | `get_category_names()` | — | list of all 20 category strings |

### Example workflow after Milestone 1

```python
from app.loader import load_20newsgroups

dataset = load_20newsgroups(categories=["sci.space"], subset="train")
print(f"Loaded {len(dataset['data'])} posts")
print(dataset['data'][0][:200])
```

### How to verify after Milestone 1

```bash
uv run python tests/test_milestones/toy_test_m1.py
```

---

## Milestone 2: Text Cleaner

Clean individual posts and filter out low-quality documents.

### What to create

- `app/cleaner.py` — `clean_text()`, `clean_dataset()`, `filter_short()`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `cleaner.py` | `clean_text()` | raw post string | cleaned string |
| `cleaner.py` | `clean_dataset()` | loader dataset dict | list of `{text, category, category_index}` dicts |
| `cleaner.py` | `filter_short()` | docs list + min_chars | filtered docs list |

### How to verify after Milestone 2

```bash
uv run python tests/test_milestones/toy_test_m2.py
```

---

## Milestone 3: File Exporter

Write cleaned documents to `.txt` files for the RAG system.

### What to create

- `app/exporter.py` — `export_by_document()`, `export_by_category()`, `export_summary()`, `slugify()`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `exporter.py` | `export_by_document()` | docs, out_dir, max_per_cat | list of written file paths |
| `exporter.py` | `export_by_category()` | docs, out_dir | list of written file paths |
| `exporter.py` | `export_summary()` | docs, out_dir | path to `dataset_summary.json` |
| `exporter.py` | `slugify()` | category name | safe filename stem |

### How to verify after Milestone 3

```bash
uv run python tests/test_milestones/toy_test_m3.py
```

---

## Milestone 4: Scripts

CLI tools for running preprocessing and verifying output.

### Files

- `scripts/preprocess.py` — full pipeline CLI with flags
- `scripts/verify.py` — check that .txt files are valid

### How to run

```bash
# Default: export all 20 categories as one file each
uv run python scripts/preprocess.py

# Export individual posts, max 200 per category
uv run python scripts/preprocess.py --mode by_document --max-per-cat 200

# Export only science categories to rag_system's data folder
uv run python scripts/preprocess.py --categories sci.space,sci.med --out-dir ../rag_system/data/raw

# Verify the output
uv run python scripts/verify.py
```

---

# Testing

```bash
uv run python tests/test_milestones/toy_test_m1.py   # Loader
uv run python tests/test_milestones/toy_test_m2.py   # Cleaner
uv run python tests/test_milestones/toy_test_m3.py   # Exporter
```

- Milestone 1 test downloads the dataset (~15 MB, cached after first run)
- Milestones 2–3 tests use synthetic in-memory data — no download needed
