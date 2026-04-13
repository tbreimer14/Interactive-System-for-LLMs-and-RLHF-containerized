# User Interface

# UI pipeline overview

    traits.json (user-defined config)
    ↓
    trait_manager.py loads traits into session
    ↓
    rag_stub.py handles prompt → retrieval + generation
    (mock now → real rag_system.pipeline.answer() later)
    ↓
    main.py (Streamlit) renders all panels:
        - prompt input
        - response display
        - retrieved context (collapsible)
        - traits editor (add / remove / reweight)
        - scoring panel (manual scores per trait)
        ↓
    reward_bridge.py computes scalar reward from manual scores
    reward = Σ(weight_i × score_i)
    ↓
    interaction_logger.py appends full interaction to .jsonl
    ↓
    history panel reads log and displays past interactions

# trait config format
    - same format as reward_system/config/traits.json
    - traits are defined by the user in config/traits.json
    - each trait has: name, description, weight
    → stored as a list of dicts:

    {
      "traits": [
        {
          "name": "clarity",
          "description": "The response is easy to read and well-structured",
          "weight": 0.4
        },
        {
          "name": "empathy",
          "description": "Acknowledges the user's situation and sounds human",
          "weight": 0.3
        },
        {
          "name": "directness",
          "description": "Gets to the point without unnecessary padding",
          "weight": 0.3
        }
      ]
    }

# interaction log format
    Each saved interaction is one JSON line in logs/interaction_log.jsonl:

    {
      "timestamp": "2026-04-10T14:32:01",
      "prompt": "What is reinforcement learning?",
      "retrieved": [
        {"text": "...", "source": "doc_3.txt"},
        ...
      ],
      "response": "Reinforcement learning is...",
      "traits": [
        {"name": "clarity", "description": "...", "weight": 0.4},
        ...
      ],
      "scores": {"clarity": 4, "empathy": 3, "directness": 5},
      "scalar_reward": 3.9
    }

# Implementation
## Milestones

- **Milestone 1:** RAG stub (`app/rag_stub.py`)
- **Milestone 2:** Trait manager + config (`config/traits.json`, `app/trait_manager.py`)
- **Milestone 3:** Reward bridge (`app/reward_bridge.py`)
- **Milestone 4:** Interaction logger (`app/interaction_logger.py`)
- **Milestone 5:** Streamlit app entry point (`main.py`)

---

## Milestone 1: RAG Stub

Stub out the RAG pipeline so the UI can be built and tested without the real RAG system.

### What to create

- `app/rag_stub.py` — a `answer(query, k=3) -> dict` function that returns mock output

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `rag_stub.py` | `answer()` | query str, k int | `{"query": str, "retrieved": [...], "answer": str}` |

### Example workflow after Milestone 1

```python
from app.rag_stub import answer

result = answer("What is reinforcement learning?", k=3)

print(result["answer"])
# [Mock response] Here is a generated answer...

for chunk in result["retrieved"]:
    print(chunk["source"], chunk["text"][:40])
# doc_7.txt   [Mock chunk 1] Retrieved passage...
```

### How to verify after Milestone 1

```bash
uv run python tests/test_milestones/toy_test_m1.py
```

---

## Milestone 2: Trait Manager + Config

Load and validate user-defined traits from the config file.
Same format as `reward_system/config/traits.json` so both systems share a config convention.

### What to create

- `config/traits.json` — default trait definitions (name, description, weight)
- `app/trait_manager.py` — `load_traits(config_path) -> list[dict]`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `trait_manager.py` | `load_traits()` | Path to `traits.json` | List of trait dicts |

### Example workflow after Milestone 2

```python
from app.trait_manager import load_traits

traits = load_traits("config/traits.json")

for t in traits:
    print(t["name"], t["weight"])

# clarity    0.4
# empathy    0.3
# directness 0.3
```

### How to verify after Milestone 2

```bash
uv run python tests/test_milestones/toy_test_m2.py
```

---

## Milestone 3: Reward Bridge

Compute the scalar reward from user-provided manual scores and trait weights.

### What to create

- `app/reward_bridge.py` — a `compute_reward(scores, traits) -> dict` function

### Design note

Unlike `reward_system/reward_fn.py` which calls a scorer automatically,
the UI reward bridge takes **user-provided scores** (from the UI sliders).
Plugging into the same weighted-sum formula keeps the two systems compatible.

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `reward_bridge.py` | `compute_reward()` | scores dict + traits list | `{"contributions": {...}, "scalar_reward": float}` |

### Reward formula

```
scalar_reward = sum(trait["weight"] * scores[trait["name"]] for trait in traits)
```

### Example workflow after Milestone 3

```python
from app.trait_manager import load_traits
from app.reward_bridge import compute_reward

traits = load_traits("config/traits.json")
scores = {"clarity": 4, "empathy": 3, "directness": 5}

result = compute_reward(scores, traits)

print(result["contributions"])  # {"clarity": 1.6, "empathy": 0.9, "directness": 1.5}
print(result["scalar_reward"])  # 4.0
```

### How to verify after Milestone 3

```bash
uv run python tests/test_milestones/toy_test_m3.py
```

---

## Milestone 4: Interaction Logger

Persist each scored interaction to a JSONL file for later RL / PPO training use.

### What to create

- `app/interaction_logger.py` — a `log_interaction(entry, log_path) -> None` function

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `interaction_logger.py` | `log_interaction()` | full interaction dict + log path | Appends one JSON line to `logs/interaction_log.jsonl` |

### Log entry format

```json
{
  "timestamp": "2026-04-10T14:32:01",
  "prompt": "What is reinforcement learning?",
  "retrieved": [{"text": "...", "source": "doc_3.txt"}],
  "response": "Reinforcement learning is...",
  "traits": [{"name": "clarity", "description": "...", "weight": 0.4}],
  "scores": {"clarity": 4, "empathy": 3, "directness": 5},
  "scalar_reward": 3.9
}
```

### How to verify after Milestone 4

```bash
uv run python tests/test_milestones/toy_test_m4.py
```

Check that `logs/interaction_log.jsonl` exists and contains a valid JSON line.

---

## Milestone 5: Streamlit App

Wire all modules into a running local UI.

### What to create

- `main.py` — Streamlit app with all 7 panels

### Panels

```text
main.py
  ├─ [1] Prompt input + Generate / Clear buttons
  ├─ [2] Response display (with loading state)
  ├─ [3] Retrieved context panel (collapsible)
  ├─ [4] Traits editor (add / remove / set weight)
  ├─ [5] Scoring panel (slider per trait → scalar reward)
  ├─ [6] Save interaction button → interaction_logger
  └─ [7] History panel (past saved interactions)
```

### How to run

```bash
uv run streamlit run main.py
```

---

# Testing

- Run individual milestone tests: `uv run python tests/test_milestones/toy_test_mX.py`
- Tests are self-contained — each imports only the modules needed for that milestone
- No external model or API required (RAG and scoring are mocked until swapped)
