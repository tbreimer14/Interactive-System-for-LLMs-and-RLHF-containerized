RL System

# ppo pipeline overview

    User prompt + model response + scalar reward
    ↓
    Tokenize (prompt + response)
    ↓
    Compute log probs — trainable model (new policy)
    ↓
    Compute log probs — frozen ref model (old policy)
    ↓
    Normalize rewards as advantages
    ↓
    Clipped PPO surrogate loss + KL penalty
    ↓
    Backprop + optimizer step
    ↓
    Updated model weights

# data format

- **Note**: This is an example format using reddit as a framework and example, using AITA prompt and responses so that the LLM can be trained on how to respond to certain prompts 
Each training record is a dict with three fields:

    {
        "prompt":   "AITA for ignoring my friend?",
        "response": "Talk to them calmly and explain how you felt.",
        "reward":   4.1
    }

- prompt   — the user input
- response — the model's generated output
- reward   — scalar score from the external reward system (any float)

The reward is treated as a black box. PPO does not know or care how it was computed.

# implementation
# Local PPO Prototype
## Milestones

- **Milestone 1:** Project scaffold + config (`config.py`, placeholder files)
- **Milestone 2:** Sample dataset (`data.py`, `load_data()`, `validate_batch()`)
- **Milestone 3:** Model + tokenizer loading (`trainer.py` — `load_model()`)
- **Milestone 4:** Manual PPO training step (`trainer.py` — `run_ppo_step()`)
- **Milestone 5:** Main training loop (`main.py`)
- **Milestone 6:** Documentation

---

## Milestone 1: Project Scaffold

Set up the folder structure and `config.py` with all hyperparameters in one place.

### Files created

- `ppo_system/config.py` — `PPOConfig` dataclass
- `ppo_system/data.py` — placeholder
- `ppo_system/trainer.py` — placeholder
- `ppo_system/main.py` — placeholder
- `tests/test_config.py` — toy tests

### How to verify

```bash
python -c "from ppo_system.config import PPOConfig; print(PPOConfig())"
python tests/test_config.py
```

---

## Milestone 2: Sample Dataset

Hardcoded `{prompt, response, reward}` records + validation helper.

### Files modified

- `ppo_system/data.py` — `load_data()`, `validate_batch()`
- `tests/test_data.py` — toy tests

### How to verify

```bash
python -c "from ppo_system.data import load_data; print(load_data())"
python tests/test_data.py
```

---

## Milestone 3: Model + Tokenizer Loading

Loads `distilgpt2` as a trainable model, a frozen reference copy, and the tokenizer.

### Files modified

- `ppo_system/trainer.py` — `load_model(config)`
- `tests/test_trainer_m3.py` — toy tests

### How to verify

```bash
python -c "from ppo_system.config import PPOConfig; from ppo_system.trainer import load_model; load_model(PPOConfig())"
python tests/test_trainer_m3.py
```

**Note:** first run downloads `distilgpt2` weights (~350 MB). Subsequent runs use cache.

---

## Milestone 4: PPO Training Step

Manual PPO update using raw PyTorch — no trl trainer.

### Files modified

- `ppo_system/trainer.py` — `run_ppo_step(model, ref_model, tokenizer, batch, optimizer, config)`
- `ppo_system/config.py` — added `clip_epsilon`, `kl_coef`
- `tests/test_trainer_m4.py` — toy tests

### How to verify

```bash
python tests/test_trainer_m4.py
```

---

## Milestone 5: Main Training Loop

End-to-end training: load data → load model → loop → print stats.

### Files modified

- `ppo_system/main.py` — `train(config)`
- `tests/test_main_m5.py` — toy tests

### How to verify

```bash
python -m ppo_system.main
python tests/test_main_m5.py
```

---

## Running All Tests

```bash
python tests/test_config.py
python tests/test_data.py
python tests/test_trainer_m3.py
python tests/test_trainer_m4.py
python tests/test_main_m5.py
```

All tests are run from `backend/rl_system/`.

**First run:** ~2–4 minutes (model download + warmup)
**Subsequent runs:** ~30–60 seconds (weights cached)

---

## Future Pipeline

This module is the PPO training piece only. The full RLHF pipeline it connects to:

```
User prompt
    ↓
Generation system  →  model generates response
    ↓
Reward system      →  user scores traits → scalar reward
    ↓
PPO module         →  updates model weights   ← (this system)
    ↓
Improved model
```

The `{prompt, response, reward}` dict is the interface contract between this module
and everything upstream. Swap in real data from the reward system and `train()` works unchanged.
