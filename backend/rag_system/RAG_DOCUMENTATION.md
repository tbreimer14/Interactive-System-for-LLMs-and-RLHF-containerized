# RAG System Documentation

## Overview

This is a **local, modular Retrieval-Augmented Generation (RAG) pipeline** that combines document search with LLM-based answer generation. The system processes raw text documents through a pipeline of stages: ingestion, embedding, indexing, retrieval, and generation.

### Architecture

```
Raw documents
    ↓
Chunking (500 chars, 100 char overlap)
    ↓
Sentence Transformer embeddings (384-dim vectors)
    ↓
FAISS vector index (fast similarity search)
    ↓
User query
    ↓
Query embedding
    ↓
Retriever gets top-k (default k=3) similar chunks
    ↓
Prompt builder combines query + retrieved context
    ↓
GPT-2 generates answer
    ↓
Final response {query, retrieved_chunks, answer}
```

---

## System Components

### 1. **Ingestion** (`app/ingest.py`)
**Purpose:** Load text documents from disk and chunk them for processing.

**Key Functions:**
- `load_text_files(data_dir)` - Reads all `.txt` files from a directory
- `build_chunks(docs, chunk_size=500, overlap=100)` - Splits documents into overlapping chunks
- `_chunk_text(text, chunk_id, doc_id, source)` - Chunks a single document with step-based iteration

**Chunk Format:**
```python
{
    "chunk_id": "doc_1_chunk_0",      # Unique identifier
    "doc_id": "doc_1",                # Source document
    "source": "data/raw/doc1.txt",    # File path
    "text": "..."                     # 500 chars of text
}
```

**Configuration:**
```python
CHUNK_SIZE = 500      # Characters per chunk
OVERLAP = 100         # Overlap between adjacent chunks
```

---

### 2. **Embedding** (`app/embed.py`)
**Purpose:** Convert text into dense 384-dimensional vector embeddings.

**Key Class:** `Embedder`
- `encode_texts(texts)` - Batch encode multiple text strings → numpy array of embeddings
- `encode_query(query)` - Encode a single query → normalized embedding vector

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional vectors
- Optimized for semantic similarity
- Normalized embeddings (L2 norm = 1.0) for cosine similarity

---

### 3. **Indexing** (`app/index.py`)
**Purpose:** Build and persist FAISS index for fast similarity search.

**Key Functions:**
- `build_index_from_chunks(chunks, embeddings)` - Create FAISS index from chunks + embeddings
- `save_index(dataset, out_dir)` - Persist index to disk
- `load_index(out_dir)` - Load pre-built index from disk

**Technical Details:**
- Uses HuggingFace `datasets` library + FAISS integration
- Index attached to dataset via `add_faiss_index("embedding")`
- Before saving: `drop_index()` → save → reload with `add_faiss_index()` again
- Supports fast k-NN search via `get_nearest_examples()`

---

### 4. **Retrieval** (`app/retrieve.py`)
**Purpose:** Query FAISS index and return top-k similar chunks.

**Key Function:** `retrieve_top_k(dataset, query_embedding, k=3)`
- Returns list of chunks sorted by similarity score (highest first)
- Each result: `{chunk_id, doc_id, source, text, score}`
- Scores: cosine similarity [0, 2] (can exceed 1.0 with FAISS)

---

### 5. **Generation** (`app/generate.py`)
**Purpose:** Generate answers using context from retrieved chunks.

**Key Class:** `Generator`
- `build_prompt(query, retrieved_texts)` - Format context + query into prompt
- `generate(prompt)` - Generate answer using GPT-2

**Model:** `gpt2`
- Text generation with temperature sampling
- Settings: `temperature=0.7`, `top_p=0.9` (diversity + quality)
- Max tokens: 100 (concise answers)

**Prompt Format:**
```
Context:
[Retrieved chunk 1]
---
[Retrieved chunk 2]
...

Question: [User query]

Answer:
```

---

### 6. **Pipeline** (`app/pipeline.py`)
**Purpose:** Orchestrate the complete RAG workflow.

**Key Class:** `RAGPipeline`

**Methods:**
- `build_index(raw_data_dir, out_dir)` - Build and save index
- `load_index(out_dir)` - Load pre-built index
- `answer(query, k=5)` - Get RAG answer with retrieved context

**Returns from `answer()`:**
```python
{
    "query": "original query",
    "retrieved": [
        {"chunk_id": "...", "text": "...", "score": 0.85},
        ...
    ],
    "answer": "Generated answer text..."
}
```

---

## Configuration

**File:** `app/config.py`

Key constants:
```python
CHUNK_SIZE = 500
OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "gpt2"
```

To customize:
1. Modify `app/config.py`
2. Update chunk size/overlap for different granularity
3. Swap embedding model for different vector spaces
4. Swap generator model (e.g., `distilgpt2`, `gpt2-medium`)

---

## Usage

### Option 1: Build Index Script

Build and save a FAISS index from raw documents:

```bash
uv run python scripts/build_index.py <raw_data_dir> <out_dir>
```

**Example:**
```bash
uv run python scripts/build_index.py data/raw data/index
```

**Output:** Index files saved to `data/index/`

---

### Option 2: Interactive CLI

Query the RAG system interactively:

```bash
uv run python scripts/chat_cli.py <index_dir>
```

**Example:**
```bash
uv run python scripts/chat_cli.py data/index
```

**Usage:**
```
Loading RAG index from data/index...
==============================================================
RAG Chat CLI
==============================================================
Type a query to retrieve and answer
Type 'exit' or 'quit' to quit

Query: What is machine learning?

Retrieving context and generating answer...

--- Query ---
What is machine learning?

--- Retrieved Context ---
[1] (score: 0.8532)
    Machine learning is a branch of AI where systems learn...
[2] (score: 0.7891)
    Learning algorithms can be supervised or unsupervised...
[3] (score: 0.6234)
    Neural networks are inspired by biological neurons...

--- Answer ---
Machine learning is a field of artificial intelligence where systems learn patterns from data. Retrieved context shows it can be categorized into supervised and unsupervised learning approaches...

Query: exit

Goodbye!
```

---

### Option 3: Programmatic API

Use RAGPipeline in Python code:

```python
from app.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Build index from raw documents
pipeline.build_index(raw_data_dir="data/raw", out_dir="data/index")

# OR load pre-built index
# pipeline.load_index(out_dir="data/index")

# Query the system
result = pipeline.answer("What is machine learning?", k=3)

# Process result
print(f"Query: {result['query']}")
print(f"Retrieved chunks: {len(result['retrieved'])}")
for chunk in result['retrieved']:
    print(f"  - Score: {chunk['score']:.4f}, Text: {chunk['text'][:60]}...")
print(f"Answer: {result['answer']}")
```

---

## Testing

### Run Milestone Tests

Each milestone has comprehensive tests:

```bash
uv run python tests/test_milestones/toy_test_m12.py  # Ingestion & Embedding
uv run python tests/test_milestones/toy_test_m3.py   # Indexing & Retrieval
uv run python tests/test_milestones/toy_test_m4.py   # Generation & Pipeline
uv run python tests/test_milestones/toy_test_m5.py   # Scripts & CLI
uv run python tests/test_milestones/toy_test_m6.py   # Integration & Validation
```

**First Run:** ~3-5 minutes (downloading models)
**Subsequent Runs:** ~30-60 seconds (models cached)

### Run All Tests

```bash
for test in tests/test_milestones/toy_test_m*.py; do
    uv run python $test
done
```

---

## Performance Notes

### Memory Usage
- **Embedder model:** ~50 MB (Sentence Transformers)
- **Generator model:** ~120 MB (GPT-2)
- **FAISS index:** ~10 MB per 1000 chunks (384-dim vectors)
- **Total for small corpus:** ~300 MB

### Speed
- **Embedding:** ~100 texts/sec on CPU
- **Retrieval:** ~1 ms per query (FAISS)
- **Generation:** ~2-5 sec per answer (CPU)

### Optimization Tips
1. **Increase chunk size** for broader context (trade-off: precision)
2. **Decrease overlap** for smaller index size (trade-off: context loss)
3. **Use smaller model** (e.g., `distilgpt2`) for faster generation
4. **Increase k** for more retrieved chunks (trade-off: prompt size)
5. **Cache embeddings** if working with fixed documents

---

## Troubleshooting

### Problem: "Index not initialized" error
**Solution:** Call `build_index()` or `load_index()` before calling `answer()`

### Problem: FAISS index not found after save
**Solution:** Index is dropped before saving (by design). After loading with `load_index()`, it's automatically re-attached.

### Problem: Very slow embeddings on first run
**Solution:** First run downloads model (~50 MB). Subsequent runs use cache. Allow 2-3 minutes.

### Problem: Unicode encoding errors on Windows
**Solution:** Fixed in latest version. Output uses ASCII-safe characters.

### Problem: Out of memory with large corpus
**Solution:** 
- Process documents in batches
- Use smaller chunk size
- Use `distilgpt2` instead of `gpt2`
- Consider GPU acceleration (modify `app/embed.py`)

---

## Architecture Decisions

1. **Separate Components:** Each stage (ingest, embed, index, retrieve, generate) is independent → easy to debug and modify

2. **HuggingFace Datasets:** Provides FAISS integration + efficient data handling

3. **Normalized Embeddings:** L2 normalization ensures cosine similarity = dot product

4. **Step-based Chunking:** Controlled overlap ensures smooth context transitions

5. **GPT-2 for Generation:** Local model that doesn't require API keys; fast enough for CPU

---

## Next Steps for Enhancement

1. **GPU Support:** Modify `Embedder` and `Generator` to use CUDA
2. **Larger Models:** Swap to `all-MiniLM-L12-v2` (embedder) or `gpt2-large` (generator)
3. **Reranking:** Add BM25 or cross-encoder reranking after retrieval
4. **Multi-file Index:** Support building index from nested directories
5. **Query Expansion:** Add query refinement before retrieval
6. **Custom Prompts:** Make prompt templates configurable
7. **Vector DB:** Replace FAISS with Pinecone/Weaviate for production scale

---

## Files Overview

```
backend/rag_system/
├── app/
│   ├── ingest.py          # Document loading & chunking
│   ├── embed.py           # Text embedding (Sentence Transformers)
│   ├── index.py           # FAISS index building & persistence
│   ├── retrieve.py        # Similarity search
│   ├── generate.py        # Answer generation (GPT-2)
│   ├── pipeline.py        # Main orchestrator
│   └── config.py          # Configuration constants
├── scripts/
│   ├── build_index.py     # CLI script to build index
│   └── chat_cli.py        # Interactive query interface
├── tests/
│   ├── data/
│   │   └── raw/           # Test documents
│   └── test_milestones/
│       ├── toy_test_m12.py   # M1-2 tests
│       ├── toy_test_m3.py    # M3 tests
│       ├── toy_test_m4.py    # M4 tests
│       ├── toy_test_m5.py    # M5 tests
│       └── toy_test_m6.py    # M6 integration tests
├── RAG_README.md          # Original overview
└── RAG_DOCUMENTATION.md   # Complete documentation (this file)
```

---

## License & Attribution

- **Sentence Transformers:** [sentence-transformers](https://www.sbert.net/) (Apache 2.0)
- **FAISS:** [Meta FAISS](https://github.com/facebookresearch/faiss) (MIT)
- **Transformers:** [Hugging Face transformers](https://github.com/huggingface/transformers) (Apache 2.0)
- **GPT-2:** OpenAI (terms of use: https://github.com/openai/gpt-2)
