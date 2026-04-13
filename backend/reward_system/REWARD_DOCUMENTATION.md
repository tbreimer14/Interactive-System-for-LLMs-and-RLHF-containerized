# Reward System Documentation

## Overview

This is a **local, modular human-in-the-loop reward system** for RLHF research. The core idea is simple:

1. A language model generates a response to a user prompt
2. The **user scores that response** on their own custom traits
3. The system computes a **single scalar reward** as the weighted sum of those scores
4. That reward is logged and later fed into a **PPO fine-tuning loop**

The system is intentionally minimal — no auto-scorer, no hardcoded traits, no polished UI yet. Everything is structured so each piece can be swapped or extended independently.

### Architecture

```
config/traits.json  (user-defined trait names, descriptions, weights)
    ↓
trait_loader.py  loads and validates traits
    ↓
user provides scores for each trait
{"interactivity": 5, "warmth": 4, "directness": 3}
    ↓
reward_fn.py  validates coverage + computes weighted sum
reward = Σ(weight_i × user_score_i)
    ↓
logger.py  appends full event to logs/reward_log.jsonl
    ↓
main.py  prints formatted output to terminal
```

### Future pipeline

```
User prompt
→ model generates response
→ user scores response on custom traits
→ reward system computes scalar reward
→ PPO uses scalar reward to fine-tune the model
```

---

## System Components

### 1. **Trait Loader** (`trait_loader.py`)

**Purpose:** Load and validate user-defined trait definitions from `config/traits.json`. This is the only place in the system that reads the config — to load traits from a different source (database, API, user profile), change only this file.

**Key Function:** `load_traits(config_path)`
- Reads `traits.json` from the given path
- Validates that every trait has `name`, `description`, and `weight`
- Returns a list of trait dicts

**Trait format:**
```python
{
    "name": "warmth",
    "description": "Sounds kind and supportive",
    "weight": 0.3
}
```

**Raises:**
- `FileNotFoundError` if the config path does not exist
- `ValueError` if a trait is missing required fields

---

### 2. **Schemas** (`schemas.py`)

**Purpose:** Define typed structures for the data that flows through the system. Keeps the code readable and type-checkable without adding runtime overhead.

**Types:**

| Schema | Fields |
|---|---|
| `Trait` | `name: str`, `description: str`, `weight: float` |
| `RewardResult` | `trait_scores: dict`, `weights: dict`, `final_reward: float` |

**Design note:** TypedDicts are plain dicts at runtime — no class instantiation, no imports needed by downstream consumers (PPO loop, UI, logging).

---

### 3. **Reward Function** (`reward_fn.py`)

**Purpose:** Convert user-provided trait scores into a single scalar reward. This module does **not** score responses itself — it takes `user_scores` as input. The reward system is human-in-the-loop by design.

**Key Function:** `compute_reward(prompt, response, traits, user_scores) → RewardResult`

**Formula:**
```
final_reward = sum(trait["weight"] * user_scores[trait["name"]]
               for trait in traits)
```

**Returns:**
```python
{
    "trait_scores": {"interactivity": 5, "warmth": 4, "directness": 3},
    "weights":      {"interactivity": 0.5, "warmth": 0.3, "directness": 0.2},
    "final_reward": 4.1
}
```

**Validation:** Raises `ValueError` if `user_scores` is missing a score for any trait in the config. Every trait must be scored before a reward can be computed.

**Notes:**
- `prompt` and `response` are accepted but not used in computation — they are passed through for context and logging
- Score scale is flexible — the user can use 1–5, 1–10, 0–1, or any other scale as long as it's consistent
- `final_reward` is rounded to 6 decimal places

---

### 4. **Logger** (`logger.py`)

**Purpose:** Persist each reward event to a newline-delimited JSON file (`logs/reward_log.jsonl`). Each entry is self-contained — it includes the full trait definitions, not just names, so the log remains interpretable even if `traits.json` changes later.

**Key Function:** `log_reward(prompt, response, traits, result, log_path=...)`
- Creates `logs/` directory automatically if it does not exist
- Appends one JSON line per call — safe to call repeatedly without overwriting
- Default log path: `logs/reward_log.jsonl`

**Log entry format:**
```json
{
  "timestamp": "2026-04-10T14:32:01",
  "prompt": "I've been struggling with anxiety...",
  "response": "That sounds really hard...",
  "traits": {
    "interactivity": {"description": "Encourages continued conversation", "weight": 0.5},
    "warmth":        {"description": "Sounds kind and supportive", "weight": 0.3},
    "directness":    {"description": "Gets to the point clearly", "weight": 0.2}
  },
  "user_scores":  {"interactivity": 5, "warmth": 4, "directness": 3},
  "weights":      {"interactivity": 0.5, "warmth": 0.3, "directness": 0.2},
  "final_reward": 4.1
}
```

**Why JSONL:**
- Append-safe: each write is a single line, no risk of corrupting prior entries
- Easy to parse: read line-by-line without loading the full file into memory
- Training-ready: can be streamed directly into a PPO dataset loader

---

### 5. **Main Entry Point** (`main.py`)

**Purpose:** Wire all modules together into a single runnable script. Loads traits, uses sample user scores, computes the reward, logs it, and prints formatted output.

**Default sample inputs:**
```
Prompt:   "I've been struggling with anxiety for a few months..."
Response: "That sounds really hard, and it makes a lot of sense..."
Scores:   {"interactivity": 5, "warmth": 4, "directness": 3}
```

Edit `DEFAULT_USER_SCORES` in `main.py` to change the sample scores, or extend `main.py` to collect scores from user input.

---

## Configuration

**File:** `config/traits.json`

User-defined traits. Edit this file to change what the reward system measures. The rest of the system adapts automatically — no code changes needed.

```json
{
  "traits": [
    {
      "name": "interactivity",
      "description": "Encourages continued conversation and asks useful follow-up questions",
      "weight": 0.5
    },
    {
      "name": "warmth",
      "description": "Sounds kind and supportive",
      "weight": 0.3
    },
    {
      "name": "directness",
      "description": "Gets to the point clearly without being vague",
      "weight": 0.2
    }
  ]
}
```

**To customize:**
- Add or remove traits freely
- Change `description` to clarify what the trait means when scoring
- Adjust `weight` to shift what drives the final reward
- Weights should sum to `1.0` so scores on different scales stay comparable
- When the UI is built, it writes to this file — no backend code changes needed

---

## Usage

### Option 1: Run with default inputs

```bash
uv run python main.py
```

### Option 2: Run with custom prompt and response

```bash
uv run python main.py "your prompt here" "your response here"
```

User scores are taken from `DEFAULT_USER_SCORES` in `main.py` — edit them there or extend `main.py` to collect them interactively.

### Option 3: Programmatic API

```python
from trait_loader import load_traits
from reward_fn import compute_reward
from logger import log_reward

# Load traits
traits = load_traits("config/traits.json")

# User-provided scores (from UI, CLI, annotation tool, etc.)
user_scores = {
    "interactivity": 5,
    "warmth": 4,
    "directness": 3,
}

# Compute reward
result = compute_reward(
    prompt="Can you help me understand RL?",
    response="Sure! RL is about learning from rewards. Want to go deeper?",
    traits=traits,
    user_scores=user_scores,
)

# Log and inspect
log_reward(prompt, response, traits, result)

print(result["trait_scores"])   # {"interactivity": 5, "warmth": 4, "directness": 3}
print(result["weights"])        # {"interactivity": 0.5, "warmth": 0.3, "directness": 0.2}
print(result["final_reward"])   # 4.1
```

---

## Testing

### Run milestone tests

```bash
uv run python tests/test_milestones/toy_test_m1.py  # Config + Trait Loader
uv run python tests/test_milestones/toy_test_m2.py  # Schemas
uv run python tests/test_milestones/toy_test_m3.py  # Reward Function
uv run python tests/test_milestones/toy_test_m4.py  # Logger
uv run python tests/test_milestones/toy_test_m5.py  # Main Entry Point (end-to-end)
```

**Run time:** ~2–5 seconds per file (no model downloads, no API calls).

### Run all tests

```bash
for test in tests/test_milestones/toy_test_m*.py; do
    uv run python $test
done
```

---

## Architecture Decisions

1. **User provides scores, system does not:** `reward_fn.py` takes `user_scores` as input and never tries to infer them. This is the defining architectural choice — it makes the reward signal human-in-the-loop from the start, which is what RLHF requires.

2. **Traits in JSON, not code:** Traits are never hardcoded anywhere. The JSON config is the single source of truth. When a UI is built, it writes to `traits.json` and the backend picks it up on the next call to `load_traits()` — no code changes.

3. **Self-contained log entries:** The logger snapshots the full trait definitions (not just names) into each log entry. This means the log remains interpretable even if `traits.json` changes between runs — important for a training dataset that might span months.

4. **Validation before compute:** `compute_reward` raises `ValueError` if any trait is missing a score. This surfaces the error immediately at the boundary rather than silently producing a wrong reward.

5. **JSONL for logging:** One JSON object per line makes the log append-safe, easy to tail, and directly usable as a training dataset without preprocessing.

6. **No pipeline class (yet):** The system is intentionally flat — each step is a direct function call. A `RewardPipeline` class makes sense once this connects to a UI or training loop, but adding one later requires no restructuring.

---

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'trait_loader'`
**Solution:** Run scripts from the `reward_system/` directory:
```bash
cd backend/reward_system
uv run python main.py
```

### Problem: `FileNotFoundError` for `traits.json`
**Solution:** Confirm `config/traits.json` exists. The config path is resolved relative to `main.py` using `Path(__file__).parent`, so it works regardless of where you call the script from.

### Problem: `ValueError: user_scores is missing scores for ...`
**Solution:** Make sure `user_scores` has a key for every trait defined in `traits.json`. If you added a new trait to the config, add its score to `DEFAULT_USER_SCORES` in `main.py` too.

### Problem: `logs/` directory not found
**Solution:** `logger.py` creates it automatically. If it still fails, check write permissions on the `reward_system/` directory.

---

## Next Steps for Enhancement

1. **Connect to the UI:** The UI presents each trait to the user and collects a score (e.g. 1–5 stars). On submit, it calls `compute_reward()` with those scores. The `traits.json` config drives what the UI renders.

2. **Connect to the RAG pipeline:** Pass `RAGPipeline.answer()["answer"]` as the `response` argument to `compute_reward()`.

3. **Feed into PPO:** The `final_reward` scalar slots directly into a PPO reward signal. The `logs/reward_log.jsonl` file becomes the preference dataset — each line is one training example.

4. **Multi-user support:** Add a `user_id` field to log entries and to the trait config to track per-user reward configurations — matching the backend-agnostic design in `IMPLEMENTATION_STATUS.md`.

5. **Reward normalization:** Add a normalization step in `reward_fn.py` to rescale `final_reward` to a fixed range (e.g. `[-1, 1]`) for PPO stability, independent of the score scale the user chose.

6. **Replace sample scores with live input:** Extend `main.py` (or build a CLI tool) that prompts the user to enter a score for each trait interactively, then calls `compute_reward()` with those scores.

---

## Files Overview

```
backend/reward_system/
├── config/
│   └── traits.json         # User-defined trait definitions (edit this)
├── logs/
│   └── reward_log.jsonl    # Reward log (created on first run)
├── tests/
│   └── test_milestones/
│       ├── toy_test_m1.py  # Trait loader tests
│       ├── toy_test_m3.py  # Reward function tests
│       ├── toy_test_m4.py  # Logger tests
│       └── toy_test_m5.py  # End-to-end tests
├── trait_loader.py         # Load + validate traits from JSON config
├── schemas.py              # TypedDict definitions: Trait, RewardResult
├── reward_fn.py            # Weighted sum of user scores → scalar reward
├── logger.py               # Append full reward events to JSONL log
├── main.py                 # Entry point — runs the full pipeline
├── REWARD_README.md        # Pipeline overview + milestone breakdown
└── REWARD_DOCUMENTATION.md # Complete documentation (this file)
```
