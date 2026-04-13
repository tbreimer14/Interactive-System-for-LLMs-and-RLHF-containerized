# RL System Documentation

## Overview

This is a **local, modular PPO reinforcement learning prototype** for RLHF research.
It accepts pre-scored `{prompt, response, reward}` tuples and uses Proximal Policy
Optimization to fine-tune a causal language model toward higher rewards.

The reward is treated as a black box — this module only sees the final scalar.
How the reward is computed (user trait scoring, automated metrics, etc.) is handled
entirely by the upstream reward system.

### Architecture

```
{prompt, response, reward} batch
    ↓
Tokenize (prompt + response concatenated)
    ↓
New policy log probs  ←  trainable model (distilgpt2)
Old policy log probs  ←  frozen ref model (distilgpt2 copy)
    ↓
Normalize rewards → advantages
    ↓
Clipped PPO surrogate loss
    +
KL divergence penalty
    ↓
Backprop → AdamW optimizer step
    ↓
Updated model weights
```

---

## System Components

### 1. Config (`config.py`)

**Purpose:** Single source of truth for all hyperparameters.

**Key class:** `PPOConfig` (dataclass)

| Field | Default | Description |
|---|---|---|
| `model_name` | `"distilgpt2"` | HuggingFace model identifier |
| `learning_rate` | `1.41e-5` | AdamW learning rate |
| `batch_size` | `4` | Records per training step |
| `mini_batch_size` | `2` | Reserved for future mini-batch PPO |
| `ppo_epochs` | `1` | Number of passes over the dataset |
| `clip_epsilon` | `0.2` | PPO clipping range (standard value) |
| `kl_coef` | `0.1` | KL penalty weight |

To customize: modify `PPOConfig` directly or pass overrides at instantiation:

```python
config = PPOConfig(learning_rate=3e-5, ppo_epochs=3)
```

---

### 2. Data (`data.py`)

**Purpose:** Supply training records and validate their format.

**Key functions:**

- `load_data()` — returns the sample dataset as a list of dicts
- `validate_batch(batch)` — raises `ValueError` if any record is malformed

**Record format:**

```python
{
    "prompt":   str,    # user input
    "response": str,    # model output
    "reward":   float   # scalar from reward system
}
```

`validate_batch()` checks:
- All three keys are present
- `reward` is a numeric type (`int` or `float`)

This is the integration point for real data. Replace `SAMPLE_DATA` or swap `load_data()`
with a function that reads from your reward system's output.

---

### 3. Trainer (`trainer.py`)

**Purpose:** Model loading and PPO update logic.

#### `load_model(config)`

Loads the causal LM and a frozen reference copy from HuggingFace.

```python
model, ref_model, tokenizer = load_model(config)
```

- `model` — trainable `AutoModelForCausalLM`
- `ref_model` — identical frozen copy (all `requires_grad = False`)
- `tokenizer` — `AutoTokenizer` with `pad_token` set to `eos_token`

The reference model is necessary for the KL penalty — it anchors the new policy
so it cannot drift too far from the original distribution.

#### `run_ppo_step(model, ref_model, tokenizer, batch, optimizer, config)`

Runs one PPO gradient update over a batch.

**Steps:**

1. Concatenate `prompt + " " + response` for each record
2. Tokenize with padding + truncation (max 128 tokens)
3. Forward pass through `model` → new policy log probs
4. Forward pass through `ref_model` (no grad) → old policy log probs
5. Normalize rewards: `(r - mean) / (std + 1e-8)` → advantages
6. Compute probability ratio: `ratio = exp(new_log_probs - old_log_probs)`
7. Clipped PPO loss: `-min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv).mean()`
8. KL penalty: `(old_log_probs - new_log_probs).mean()`
9. Total loss: `ppo_loss + kl_coef * kl`
10. `loss.backward()` → `optimizer.step()`

**Returns:** `(loss: float, mean_reward: float)`

---

### 4. Main (`main.py`)

**Purpose:** Orchestrate the full training loop.

**Key function:** `train(config=None)`

```python
from ppo_system.main import train
stats = train()
# stats = [(loss_step1, reward_step1), (loss_step2, reward_step2), ...]
```

- Loads data, validates it, loads model, sets up optimizer
- Iterates over `ppo_epochs` × batches
- Prints per-step stats
- Returns list of `(loss, mean_reward)` tuples

**Run from command line:**

```bash
python -m ppo_system.main
```

---

## Design Decisions

### 1. No trl Trainer

`trl >= 1.0.0` removed `PPOTrainer` and `AutoModelForCausalLMWithValueHead`.
The new trainers (`RLOOTrainer`, `GRPOTrainer`) are designed for online RL where
the model generates responses during training — incompatible with the offline
`{prompt, response, reward}` format this project uses.

The manual PPO loop using raw PyTorch is ~60 lines, covers the same core algorithm,
and works with pre-collected data without fighting the framework.

### 2. Scalar Reward as Black Box

PPO only receives the final scalar reward. Trait definitions, weighting, and scoring
logic live entirely in the upstream reward system. This keeps the modules decoupled —
the reward system can change its traits without touching the RL code.

### 3. KL Penalty vs Hard Clipping

Both PPO clipping and a KL penalty are applied. Clipping alone can still allow
large policy shifts in some batches. The KL penalty provides a softer, continuous
constraint that keeps the model close to its initialization — important for small
models like `distilgpt2` where aggressive updates can destabilize training quickly.

### 4. Reference Model as Frozen Copy

The reference model is initialized from the same checkpoint as the trainable model.
This means at step 0, KL ≈ 0 and the penalty has no effect. As training progresses,
the penalty grows proportional to how far the policy has moved — a natural regularizer.

### 5. `distilgpt2` for Local Prototyping

82M parameters, no API key required, runs on CPU in reasonable time.
Swap `model_name` in `PPOConfig` to scale up (e.g. `gpt2`, `gpt2-medium`).

---

## Configuration

All tunable values live in `config.py`. Key choices:

```python
# Increase for more training signal per step
batch_size = 4

# Higher = more conservative policy updates
kl_coef = 0.1

# Standard PPO clipping range; lower = more conservative
clip_epsilon = 0.2

# More epochs = more passes over the same data
ppo_epochs = 1
```

---

## Testing

| Test file | What it covers |
|---|---|
| `tests/test_config.py` | Default values, overrides, repr |
| `tests/test_data.py` | Dataset structure, validate_batch |
| `tests/test_trainer_m3.py` | Model loading, ref model frozen, pad token |
| `tests/test_trainer_m4.py` | PPO step output, loss finite, params update |
| `tests/test_main_m5.py` | Full loop runs, stats are valid floats |

Run all from `backend/rl_system/`:

```bash
python tests/test_config.py
python tests/test_data.py
python tests/test_trainer_m3.py
python tests/test_trainer_m4.py
python tests/test_main_m5.py
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ppo_system'`
Run as a module from `rl_system/`, not from inside `ppo_system/`:
```bash
python -m ppo_system.main
```

### `ImportError` from `trl`
trl 1.0.0 removed `PPOTrainer`. This project uses a manual PyTorch PPO loop —
no trl trainer is imported. If you see a trl import error, check `trainer.py`
has no `from trl import ...` lines.

### Loss is NaN on first step
Usually a learning rate that is too high. Lower `learning_rate` in `PPOConfig`.
Also check that `reward` values are reasonable floats (not extremely large).

### Very slow on first run
First run downloads `distilgpt2` (~350 MB) from HuggingFace. Subsequent runs use
the local cache and are significantly faster.

### Out of memory
Reduce `batch_size` in `PPOConfig` or reduce `max_length` in the tokenizer call
inside `run_ppo_step()`.

---

## Future Integration

This module is designed to slot into the full RLHF pipeline with minimal changes:

```
Generation system  →  produces (prompt, response) pairs
Reward system      →  scores them → scalar reward
PPO module         →  this system, no changes needed
```

**To connect real data:**
Replace `load_data()` in `data.py` with a function that reads from your
reward system's output (file, database, API). The `validate_batch()` call
in `main.py` will catch any format mismatches before training starts.

**To scale the model:**
Change `model_name` in `PPOConfig` to any HuggingFace causal LM:
```python
config = PPOConfig(model_name="gpt2-medium")
```

---

## Files Overview

```
backend/rl_system/
├── ppo_system/
│   ├── __init__.py          # Package marker
│   ├── config.py            # PPOConfig dataclass — all hyperparameters
│   ├── data.py              # Sample dataset + validate_batch()
│   ├── trainer.py           # load_model() + run_ppo_step()
│   ├── main.py              # train() — full training loop
│   ├── RL_README.md         # Usage and pipeline overview
│   └── RL_DOCUMENTATION.md  # Technical design (this file)
└── tests/
    ├── test_config.py       # M1 tests
    ├── test_data.py         # M2 tests
    ├── test_trainer_m3.py   # M3 tests
    ├── test_trainer_m4.py   # M4 tests
    └── test_main_m5.py      # M5 tests
```

---

## License & Attribution

- **Transformers:** Hugging Face (Apache 2.0)
- **PyTorch:** Meta (BSD)
- **distilgpt2:** Hugging Face (Apache 2.0)
