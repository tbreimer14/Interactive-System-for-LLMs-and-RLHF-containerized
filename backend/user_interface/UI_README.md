# User Interface

# UI pipeline overview

    traits.json (user-defined config)
    ↓
    trait_manager.py loads traits into session
    ↓
    app.py (Streamlit) renders four pages via sidebar navigation:
        Setup    — define / reweight reward traits
        Grade    — enter prompt + response, score with sliders
        History  — master-detail view of past scored interactions
        Analytics — reward trend, trait breakdown, summary stats
    ↓
    reward_bridge.py computes scalar reward from manual scores
    reward = Σ(weight_i × score_i)
    ↓
    ui/storage.py appends full interaction to .jsonl
    ↓
    History and Analytics pages read from the same log

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
      "response": "Reinforcement learning is...",
      "traits": [
        {"name": "clarity", "score": 3, "weight": 0.4, "contribution": 1.2},
        {"name": "empathy", "score": -2, "weight": 0.3, "contribution": -0.6},
        ...
      ],
      "scalar_reward": 1.8
    }

    Scores range from -5 (strong penalty) to +5 (strong reward); 0 is neutral.
    Negative contributions reduce the scalar reward, driving the model away from
    that trait in future training steps.

# Implementation
## Milestones

- **Milestone 1:** Trait manager + config (`config/traits.json`, `app/trait_manager.py`)
- **Milestone 2:** Reward bridge (`app/reward_bridge.py`)
- **Milestone 3:** Interaction logger (`ui/storage.py`)
- **Milestone 4:** Streamlit app (`app.py`, `ui/`)

---

## Milestone 1: Trait Manager + Config

Load and validate user-defined traits from the config file.
Same format as `reward_system/config/traits.json` so both systems share a config convention.

### What to create

- `config/traits.json` — default trait definitions (name, description, weight)
- `app/trait_manager.py` — `load_traits(config_path) -> list[dict]`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `trait_manager.py` | `load_traits()` | Path to `traits.json` | List of trait dicts |

### Example workflow after Milestone 1

```python
from app.trait_manager import load_traits

traits = load_traits("config/traits.json")

for t in traits:
    print(t["name"], t["weight"])

# clarity    0.4
# empathy    0.3
# directness 0.3
```

### How to verify after Milestone 1

```bash
uv run python tests/test_milestones/toy_test_m2.py
```

---

## Milestone 2: Reward Bridge

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

### Example workflow after Milestone 2

```python
from app.trait_manager import load_traits
from app.reward_bridge import compute_reward

traits = load_traits("config/traits.json")
scores = {"clarity": 3, "empathy": -2, "directness": 4}

result = compute_reward(scores, traits)

print(result["contributions"])  # {"clarity": 1.2, "empathy": -0.6, "directness": 1.2}
print(result["scalar_reward"])  # 1.8
```

### How to verify after Milestone 2

```bash
uv run python tests/test_milestones/toy_test_m3.py
```

---

## Milestone 3: Interaction Logger

Persist each scored interaction to a JSONL file for later RL training use.

### What to create

- `ui/storage.py` — `save_interaction(entry, log_path)` and `load_interactions(log_path)`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `storage.py` | `save_interaction()` | InteractionLog + log path | Appends one JSON line to `.jsonl` |
| `storage.py` | `load_interactions()` | log path | List of dicts, oldest first |

### Log entry format

```json
{
  "timestamp": "2026-04-10T14:32:01",
  "prompt": "What is reinforcement learning?",
  "response": "Reinforcement learning is...",
  "traits": [{"name": "clarity", "score": 3, "weight": 0.4, "contribution": 1.2},
             {"name": "empathy", "score": -2, "weight": 0.3, "contribution": -0.6}],
  "scalar_reward": 1.8
}
```

Scores range from -5 (penalise) to +5 (reward); 0 is neutral.

### How to verify after Milestone 3

```bash
uv run python tests/test_milestones/toy_test_m5.py
```

---

## Milestone 4: Streamlit App

Wire all modules into a running local UI with four pages.

### What to create

- `app.py` — entry point, sidebar navigation, page routing
- `ui/components.py` — page and panel rendering functions
- `ui/state.py` — session state management
- `ui/types.py` — data models (TraitConfig, ScoredTrait, InteractionLog)

### Pages

```text
app.py
  ├─ Setup    — trait editor (add / remove / reweight)
  ├─ Grade    — prompt text area + response text area
  │               → scoring sliders per trait
  │               → reward breakdown display
  │               → save to JSONL button
  ├─ History  — master-detail view of past interactions
  └─ Analytics — reward trend chart, trait bar chart, summary stats
```

### Grade page layout

The Grade page has a draggable column divider (drag the vertical handle between
the two columns to resize them). Left column: prompt and response input.
Right column: scoring panel and save button.

### How to run

```bash
cd backend/user_interface
uv run streamlit run app.py
```

---

# Testing

- Run individual milestone tests: `uv run python tests/test_milestones/toy_test_mX.py`
- Tests are self-contained — each imports only the modules needed for that milestone
- No external model or API required
