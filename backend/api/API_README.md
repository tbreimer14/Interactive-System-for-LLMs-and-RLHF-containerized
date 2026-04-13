# API Backend

## Overview

A minimal FastAPI backend that:
- receives prompts and returns generated responses + retrieved chunks
- computes scalar reward from user-defined trait scores
- saves and retrieves interaction records

```
app.py                          ← FastAPI entry point
api/
  routes.py                     ← all endpoint definitions
  schemas.py                    ← Pydantic request/response models
services/
  generation_service.py         ← calls backend adapter, returns result
  reward_service.py             ← computes weighted reward
  storage_service.py            ← JSONL read/write
backend_adapter.py              ← mock (or real) RAG adapter
data/
  interactions.jsonl            ← append-only interaction log
tests/
  test_milestones/
    toy_test_m1.py
    toy_test_m2.py
    toy_test_m3.py
    toy_test_m4.py
    toy_test_m5.py
```

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/generate` | Retrieve chunks + generate response |
| POST | `/reward/compute` | Compute scalar reward from trait scores |
| POST | `/interactions` | Save an interaction record |
| GET | `/interactions?limit=20` | Get recent interactions |
| GET | `/interactions/{id}` | Get single interaction by ID |

---

## Implementation

### Milestones

- **Milestone 1:** Schemas (`api/schemas.py`)
- **Milestone 2:** Storage Service (`services/storage_service.py`, `data/interactions.jsonl`)
- **Milestone 3:** Backend Adapter + Generation Service (`backend_adapter.py`, `services/generation_service.py`)
- **Milestone 4:** Reward Service (`services/reward_service.py`)
- **Milestone 5:** Routes + App (`api/routes.py`, `app.py`)

---

## Milestone 1: Schemas

Define all typed request/response models used across the API.

### What to create

- `api/schemas.py` — all Pydantic models

### Models

| Model | Fields | Purpose |
|---|---|---|
| `RetrievedChunk` | id, source (optional), text | One chunk from RAG retrieval |
| `GenerateRequest` | prompt, top_k (int), show_prompt (bool) | Input to `/generate` |
| `GenerateResponse` | prompt, retrieved_chunks, response, final_prompt | Output of `/generate` |
| `TraitScoreInput` | name, weight, score | One trait with user score |
| `TraitScoreOutput` | name, weight, score, contribution | Trait + computed contribution |
| `RewardRequest` | traits (list of TraitScoreInput) | Input to `/reward/compute` |
| `RewardResponse` | traits (list of TraitScoreOutput), scalar_reward | Output of `/reward/compute` |
| `SaveInteractionRequest` | prompt, retrieved_chunks, response, final_prompt, traits, scalar_reward | Input to `POST /interactions` |
| `InteractionRecord` | id, timestamp, + all SaveInteractionRequest fields | Stored record |

### How to verify after Milestone 1

```bash
uv run python tests/test_milestones/toy_test_m1.py
```

---

## Milestone 2: Storage Service

Implement append-only JSONL storage for interaction records.

### What to create

- `services/storage_service.py` — three functions
- `data/interactions.jsonl` — empty file (created on first save)

### Functions

| Function | Input | Output |
|---|---|---|
| `save_interaction(record: dict)` | interaction dict | Appends one JSON line to `data/interactions.jsonl` |
| `load_interactions(limit: int)` | max records to return | List of recent interaction dicts |
| `get_interaction_by_id(id: str)` | record ID | Single interaction dict or `None` |

### Storage rules

- One JSON object per line
- Always append, never overwrite
- Handle: missing file, empty file, malformed lines (skip bad lines)

### Example workflow after Milestone 2

```python
from services.storage_service import save_interaction, load_interactions, get_interaction_by_id

record = {
    "id": "abc123",
    "timestamp": "2026-04-11T10:00:00",
    "prompt": "Hello",
    "retrieved_chunks": [],
    "response": "Hi there!",
    "final_prompt": "Hello",
    "traits": [],
    "scalar_reward": 0.0,
}

save_interaction(record)

all_records = load_interactions(limit=10)
print(all_records[-1]["id"])  # abc123

single = get_interaction_by_id("abc123")
print(single["response"])  # Hi there!
```

### How to verify after Milestone 2

```bash
uv run python tests/test_milestones/toy_test_m2.py
```

---

## Milestone 3: Backend Adapter + Generation Service

Create the mock RAG adapter and the service that calls it.

### What to create

- `backend_adapter.py` — `BackendAdapter` class with `generate()` method
- `services/generation_service.py` — `run_generation()` function

### What each does

| Module | Method/Function | Input | Output |
|---|---|---|---|
| `BackendAdapter` | `generate(prompt, top_k)` | prompt string, top_k int | dict with `retrieved_chunks`, `response`, `final_prompt` |
| `generation_service.py` | `run_generation(request)` | `GenerateRequest` | `GenerateResponse` |

### Mock adapter behavior

If no real RAG is connected, the adapter returns:
- `retrieved_chunks`: a list of `top_k` fake chunks (id, source, text)
- `response`: a placeholder string
- `final_prompt`: the original prompt

### Example workflow after Milestone 3

```python
from backend_adapter import BackendAdapter

adapter = BackendAdapter()
result = adapter.generate("What is RL?", top_k=2)

print(result["response"])           # mock response string
print(len(result["retrieved_chunks"]))  # 2
```

### How to verify after Milestone 3

```bash
uv run python tests/test_milestones/toy_test_m3.py
```

---

## Milestone 4: Reward Service

Compute the scalar reward from a list of weighted trait scores.

### What to create

- `services/reward_service.py` — `compute_reward()` function

### What it does

| Function | Input | Output |
|---|---|---|
| `compute_reward(request)` | `RewardRequest` (list of TraitScoreInput) | `RewardResponse` (list of TraitScoreOutput + scalar_reward) |

### Reward formula

```
contribution_i = weight_i * score_i
scalar_reward  = sum(contribution_i for all traits)
```

### Example workflow after Milestone 4

```python
from services.reward_service import compute_reward
from api.schemas import RewardRequest, TraitScoreInput

request = RewardRequest(traits=[
    TraitScoreInput(name="clarity", weight=1.0, score=4.0),
    TraitScoreInput(name="empathy", weight=0.5, score=3.0),
])

result = compute_reward(request)
print(result.scalar_reward)   # 5.5
print(result.traits[0].contribution)  # 4.0
```

### How to verify after Milestone 4

```bash
uv run python tests/test_milestones/toy_test_m4.py
```

---

## Milestone 5: Routes + FastAPI App

Wire all services into FastAPI endpoints and run the server.

### What to create

- `api/routes.py` — all 6 endpoint definitions
- `app.py` — FastAPI app that includes the router

### Endpoints

```
GET  /health                   → {"status": "ok"}
POST /generate                 → GenerateResponse
POST /reward/compute           → RewardResponse
POST /interactions             → {"status": "saved"}
GET  /interactions?limit=20    → list of InteractionRecord
GET  /interactions/{id}        → InteractionRecord or 404
```

### Error handling

- Empty prompt → 422
- Adapter failure → 500
- Missing interaction ID → 404
- Malformed trait input → 422

### How to run

```bash
uv run uvicorn app:app --reload
```

### How to verify after Milestone 5

```bash
uv run python tests/test_milestones/toy_test_m5.py
```

Or manually with curl:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is RL?", "top_k": 2, "show_prompt": false}'

curl -X POST http://localhost:8000/reward/compute \
  -H "Content-Type: application/json" \
  -d '{"traits": [{"name": "clarity", "weight": 1.0, "score": 4.0}]}'
```

---

## Sample JSONL Record

```json
{"id": "a1b2c3d4", "timestamp": "2026-04-11T10:00:00", "prompt": "What is RL?", "retrieved_chunks": [{"id": "chunk_1", "source": "intro.txt", "text": "Reinforcement learning is..."}], "response": "RL is a type of machine learning...", "final_prompt": "What is RL?", "traits": [{"name": "clarity", "weight": 1.0, "score": 4.0, "contribution": 4.0}], "scalar_reward": 4.0}
```

---

## Install & Run

```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn app:app --reload --app-dir backend/api
```
