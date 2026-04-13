# Reward System

# reward system pipeline overview

    traits.json (user-defined config)
    ↓
    trait_loader.py loads traits
    ↓
    user provides scores per trait
    (manual scores from UI / CLI → LLM judge later)
    ↓
    reward_fn.py computes weighted sum
    reward = Σ(weight_i × user_score_i)
    ↓
    logger.py logs result to .jsonl
    ↓
    main.py prints final reward output

# trait config format
    - traits are defined by the user in config/traits.json
    - each trait has: name, description, weight
    - weights should sum to 1.0 (not enforced, but recommended)
    → stored as a list of dicts:

    {
      "traits": [
        {
          "name": "interactivity",
          "description": "Encourages continued conversation and asks useful follow-up questions",
          "weight": 0.5
        },
        {
          "name": "warmth",
          "description": "Sounds kind, supportive, and human",
          "weight": 0.3
        },
        {
          "name": "directness",
          "description": "Gets to the point without being vague",
          "weight": 0.2
        }
      ]
    }

# reward output format
    {
      "trait_scores": {
        "interactivity": 5,
        "warmth": 4,
        "directness": 3
      },
      "weights": {
        "interactivity": 0.5,
        "warmth": 0.3,
        "directness": 0.2
      },
      "final_reward": 4.7
    }

# Implementation
## Milestones

- **Milestone 1:** Config + Trait Loader (`config/traits.json`, `trait_loader.py`)
- **Milestone 2:** Schemas (`schemas.py`)
- **Milestone 3:** Reward Function (`reward_fn.py`)
- **Milestone 4:** Logger (`logger.py`)
- **Milestone 5:** Main entry point + end-to-end run (`main.py`)

---

## Milestone 1: Config + Trait Loader

Set up the trait config file and the module that loads it.

### What to create

- `config/traits.json` — user-defined trait definitions (name, description, weight)
- `trait_loader.py` — loads and validates the JSON, returns a list of trait dicts

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `trait_loader.py` | `load_traits()` | Path to `traits.json` | List of trait dicts |

### Example workflow after Milestone 1

```python
from trait_loader import load_traits

traits = load_traits("config/traits.json")

for trait in traits:
    print(trait["name"], trait["weight"])

# interactivity 0.5
# warmth        0.3
# directness    0.2
```

### How to verify after Milestone 1

```bash
uv run python tests/test_milestones/toy_test_m1.py
```

---

## Milestone 2: Schemas

Define the typed data structures used across the system.

### What to create

- `schemas.py` — `Trait` and `RewardResult` TypedDicts

### What each does

| Type | Fields | Purpose |
|---|---|---|
| `Trait` | name, description, weight | One user-defined trait |
| `RewardResult` | trait_scores, weights, final_reward | Output of compute_reward() |

### How to verify after Milestone 2

```bash
uv run python -c "from schemas import Trait, RewardResult; print('schemas OK')"
```

---

## Milestone 3: Reward Function

Combine user-provided scores and trait weights into a scalar reward.

### What to create

- `reward_fn.py` — a `compute_reward(prompt, response, traits, user_scores) -> RewardResult` function

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `reward_fn.py` | `compute_reward()` | prompt + response + traits + user_scores | `RewardResult` dict |

### Reward formula

```
final_reward = sum(trait["weight"] * user_scores[trait["name"]]
               for trait in traits)
```

### Example workflow after Milestone 3

```python
from trait_loader import load_traits
from reward_fn import compute_reward

traits = load_traits("config/traits.json")
user_scores = {"interactivity": 5, "warmth": 4, "directness": 3}

result = compute_reward(
    prompt="Can you help me understand reinforcement learning?",
    response="Sure! RL is about learning from rewards. Want to go deeper?",
    traits=traits,
    user_scores=user_scores,
)

print(result["trait_scores"])   # {"interactivity": 5, "warmth": 4, "directness": 3}
print(result["final_reward"])   # 4.7
```

### How to verify after Milestone 3

```bash
uv run python tests/test_milestones/toy_test_m3.py
```

---

## Milestone 4: Logger

Persist each reward computation to a log file for later inspection and training data use.

### What to create

- `logger.py` — a `log_reward(prompt, response, traits, result)` function that appends to `logs/reward_log.jsonl`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `logger.py` | `log_reward()` | prompt + response + traits + result dict | Appends one JSON line to `logs/reward_log.jsonl` |

### Log entry format

```json
{
  "timestamp": "2026-04-10T14:32:01",
  "prompt": "Can you help me understand RL?",
  "response": "Sure! Reinforcement learning...",
  "traits": {
    "interactivity": {"description": "...", "weight": 0.5},
    "warmth": {"description": "...", "weight": 0.3},
    "directness": {"description": "...", "weight": 0.2}
  },
  "user_scores": {"interactivity": 5, "warmth": 4, "directness": 3},
  "weights": {"interactivity": 0.5, "warmth": 0.3, "directness": 0.2},
  "final_reward": 4.7
}
```

### How to verify after Milestone 4

```bash
uv run python tests/test_milestones/toy_test_m4.py
```

Check that `logs/reward_log.jsonl` exists and contains a valid JSON line.

---

## Milestone 5: Main Entry Point

Wire all modules together into a runnable end-to-end script.

### What to create

- `main.py` — loads traits, accepts prompt + response + user scores, computes reward, logs, and prints

### Workflow

```text
main.py
  ├─ load_traits("config/traits.json")
  ├─ compute_reward(prompt, response, traits, user_scores)
  ├─ log_reward(prompt, response, traits, result)
  └─ print formatted output to terminal
```

### Example output

```
==============================================================
  REWARD SYSTEM - RESULT
==============================================================

  Prompt:   "Can you help me understand reinforcement learning?"
  Response: "Sure! RL is about learning from rewards. Want to go deeper?"

  Trait Scores:
    interactivity     score=5  weight=0.5  → 2.5
    warmth            score=4  weight=0.3  → 1.2
    directness        score=3  weight=0.2  → 0.6

  Final Reward: 4.3

  Logged to: logs/reward_log.jsonl
==============================================================
```

### How to verify after Milestone 5

```bash
uv run python main.py
```

Then run the full test suite:

```bash
uv run python tests/test_milestones/toy_test_m5.py
```

---

# Testing

- Run individual milestone tests: `uv run python tests/test_milestones/toy_test_mX.py`
- Tests are self-contained — each imports only the modules needed for that milestone
- No external model downloads required (scoring is provided by the user, not automated)
