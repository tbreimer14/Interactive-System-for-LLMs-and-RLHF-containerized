RAG System

# rag pipeline overview
    Raw documents
    ↓
    Chunking
    ↓
    Embeddings
    ↓
    Vector index (FAISS)
    ↓
    User query
    ↓
    Query embedding
    ↓
    Retriever gets top-k chunks
    ↓
    Prompt builder combines query + chunks
    ↓
    LLM generates answer
    ↓
    Final response

# chunking documents
    - chunking_size = 500 characters per chunk
    - overlap of 100 characters between adjacent chunks
    -> stored as a dictionaty with the format as

    {
    "chunk_id": "doc_1_chunk_0",      # Unique chunk identifier
    "doc_id": "doc_1",                # Which document it came from
    "source": "data/raw/document1.txt", # Original file path
    "text": "..."                     # The 500 chars of text
    }

# Implementation 
# Local RAG Model
## Milestones

- **Milestone 1:** Project structure + placeholder modules (just folders & basic files)
- **Milestone 2:** Data ingestion & embedding modules (`ingest.py`, `embed.py`)
- **Milestone 3:** Indexing & retrieval (`index.py`, `retrieve.py`)
- **Milestone 4:** Generation module (`generate.py`, `pipeline.py`)
- **Milestone 5:** Scripts & tests (`build_index.py`, `chat_cli.py`, `tests`)
- **Milestone 6:** Documentation & final validation

---

## Milestone 1: Project Structure

Set up the folder structure and create placeholder files with docstrings.

### What to create

- `app/` folder with all module placeholders
- `data/` folder with subdirectories
- `scripts/` folder
- `tests/` folder
- `app/config.py` — configuration constants

---

## Milestone 2: Ingestion & Embedding

### Files to modify

- `app/ingest.py` — implement 3 functions for document loading and chunking
- `app/embed.py` — implement `Embedder` class for text embedding

### What each does (high-level)

| Module | Function | Input | Output |
|---|---|---|---|
| `ingest.py` | `load_text_files()` | Directory path | List of docs with text content |
| `ingest.py` | `chunk_text()` | Text string | List of overlapping text chunks |
| `ingest.py` | `build_chunks()` | List of docs | List of chunks with metadata `id`, `source` |
| `embed.py` | `Embedder` class | - | Encodes text → 384-dim vectors |
| `embed.py` | `.encode_texts()` | List of strings | Matrix of embeddings |
| `embed.py` | `.encode_query()` | Query string | Single embedding vector |

### Example workflow after Milestone 2

```python
from app.ingest import load_text_files, build_chunks
from app.embed import Embedder

# Load documents from data/raw/
docs = load_text_files("data/raw/")
print(f"Loaded {len(docs)} documents")

# Split into chunks
chunks = build_chunks(docs, chunk_size=500, overlap=100)
print(f"Created {len(chunks)} chunks")

# Embed all chunks
embedder = Embedder()
chunk_texts = [c["text"] for c in chunks]
embeddings = embedder.encode_texts(chunk_texts)
print(f"Embeddings shape: {embeddings.shape}")  # Should be (n_chunks, 384)

# Embed a query
query = "What is the meaning of life?"
query_emb = embedder.encode_query(query)
print(f"Query embedding shape: {query_emb.shape}")  # Should be (384,)
```

### How to verify after Milestone 2

```bash
uv run python tests/toy_test.py
```

---

## Milestone 3: Indexing & Retrieval

This stage takes the chunks and embeddings from Milestone 2 and creates a searchable FAISS index for fast similarity search.

### Files to modify

- `app/index.py` — implement 3 functions for building and saving a FAISS index
- `app/retrieve.py` — implement 1 function for similarity search

### Workflow

```text
Chunks + Embeddings (from Milestone 2)
          ↓
   Build HF Dataset
          ↓
   Add FAISS Index
          ↓
   Save to disk
          ↓
      [Index ready!]
          ↓
   Load index from disk
          ↓
   Query embedding → Search FAISS
          ↓
   Return top-k similar chunks
```

### What each module does

| Module | Function | Input | Output |
|---|---|---|---|
| `index.py` | `build_index_from_chunks()` | Chunks + embeddings | HF Dataset with FAISS index |
| `index.py` | `save_index()` | Dataset + directory | Saves 2 files to disk |
| `index.py` | `load_index()` | Directory path | Loads Dataset from disk |
| `retrieve.py` | `retrieve_top_k()` | Dataset + query embedding + `k` | List of top-k chunks with scores |

### Data structures

#### Input to `index.py`

```python
chunks = [
    {"chunk_id": "doc_0_chunk_0", "doc_id": "doc_0", "source": "...", "text": "..."},
    {"chunk_id": "doc_0_chunk_1", "doc_id": "doc_0", "source": "...", "text": "..."},
    ...
]

embeddings = np.array([[...], [...], ...])  # shape (n_chunks, 384)
```

#### Output from `retrieve_top_k()`

```python
[
    {
        "score": 0.85,
        "chunk_id": "doc_0_chunk_0",
        "doc_id": "doc_0",
        "source": "data/raw/toy_test.txt",
        "text": "..."
    },
    {
        "score": 0.78,
        "chunk_id": "doc_0_chunk_1",
        ...
    },
    ...
]
```

### Files saved to disk

After running `save_index()`:

```text
data/index/
├── dataset.huggingface
└── faiss_index.bin
```

- `dataset.huggingface` — HF Dataset with chunks + embeddings
- `faiss_index.bin` — FAISS index for similarity search

---

## Milestone 4: Generation & Pipeline

### Files to modify

- `app/generate.py` — implement `Generator` class for text generation
- `app/pipeline.py` — implement `RAGPipeline` class to orchestrate everything

### What each does (high-level)

| Module | Class/Function | Input | Output |
|---|---|---|---|
| `generate.py` | `Generator.__init__()` | Model name (optional) | Initialized generator with LLM loaded |
| `generate.py` | `Generator.build_prompt()` | Query + retrieved chunks | Formatted prompt with context instruction |
| `generate.py` | `Generator.generate()` | Prompt | Generated answer text |
| `pipeline.py` | `RAGPipeline.__init__()` | - | Initialized with embedder + generator |
| `pipeline.py` | `RAGPipeline.build_index()` | Raw data dir | Index saved to disk |
| `pipeline.py` | `RAGPipeline.load_index()` | Index dir | Index loaded to memory |
| `pipeline.py` | `RAGPipeline.answer()` | Query + `k` | `{query, retrieved, answer}` dict |

### Workflow

```python
from app.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Build index (one-time setup)
pipeline.build_index("data/raw/", "data/index/")

# Or load pre-built index
pipeline.load_index("data/index/")

# Ask a question
result = pipeline.answer("What is deep learning?", k=5)
```

### Output

```python
{query, retrieved, answer}
```

### Data flow

```text
User Query
   ↓
pipeline.answer()
   ├─ Embed query
   ├─ Retrieve top-k chunks (via FAISS)
   ├─ Build prompt with context
   ├─ Generate answer with LLM
   └─ Return {query, retrieved, answer}
```

### Prompt structure

- Answer using only the retrieved context
- Say **"I don't know"** if evidence is insufficient
- Keep the answer concise and relevant

---

## Milestone 5: Build Scripts & CLI

This milestone creates two practical command-line tools.

### Files to modify

#### `scripts/build_index.py`

**Purpose:** build and save a FAISS index from raw documents.

- Takes raw data directory as input
- Uses `RAGPipeline` to chunk, embed, and index
- Saves index to specified output directory
- Prints success confirmation

**Run:**

```bash
uv run python scripts/build_index.py data/raw data/index
```

#### `scripts/chat_cli.py`

**Purpose:** interactive CLI for querying the RAG system.

- Loads pre-built index at startup
- Enters a loop:
  - user types query
  - RAG answers
  - shows results with retrieved chunks
- Commands to terminate: `exit` or `quit`
- Shows query, top-k retrieved chunks with scores, and generated answer

**Run:**

```bash
uv run python scripts/chat_cli.py data/index
```

### Tests

- Verify index building from script works
- Verify CLI output parsing with simulated input

This milestone uses `RAGPipeline` and all previously implemented modules.

---

## Milestone 6: Testing and Documentation

### Integration tests

- End-to-end RAG workflow tests
- Test index building → querying → answer generation
- Test with different query types:
  - factual
  - comparative
  - open-ended
- Validate retrieved chunks are semantically relevant
- Validate generated answers use context appropriately

### Documentation
Include:

- System architecture overview
- Component descriptions:
  - `ingest`
  - `embed`
  - `index`
  - `retrieve`
  - `generate`
- Configuration constants and how to customize them
- Usage examples:
  - building an index from custom documents
  - using `chat_cli.py` for interactive queries
  - using `build_index.py` for batch indexing
  - programmatic API usage via `RAGPipeline`
- Troubleshooting section
- Performance notes and optimization tips

# Testing
- running tests/test_milestones/toy_test_mX.py: should take around 2-4 minutes in the first run
    - when model is cached then it would be quicker