# RL System

# rl system pipeline overview

    20 Newsgroups dataset (sklearn)
    ↓
    data.py builds instruction prompts
    (INSTRUCTION + article text, formatted for Qwen2.5-Instruct chat template)
    ↓
    Qwen2.5-1.5B-Instruct + LoRA generates G=2 responses per prompt
    (model.py loads base model in 4-bit quantization with LoRA adapters attached)
    ↓
    UI displays each response → user moves sliders per trait
    ↓
    reward.py calls reward_system's compute_reward() → scalar per response
    reward = Σ(weight_i × user_score_i)
    ↓
    GRPOTrainer computes advantages across G responses
    advantage = reward_i − mean(all rewards for this prompt)
    ↓
    LoRA adapter weights updated via clipped policy gradient
    ↓
    grpo_output/ saves LoRA adapters (~40MB, not the full 1.5B model)

# how grpo works

    Single prompt → G responses generated in parallel
    Each response scored by user sliders → scalar reward
    Advantage per response = reward − mean(group rewards)
    Policy gradient update nudges model toward high-advantage responses
    No critic/value model needed — relative rewards within the group suffice

# memory requirements

    Full fine-tuning Qwen2.5-1.5B:   ~20GB+ VRAM  (infeasible on consumer hardware)
    LoRA fine-tuning (rank=16):       ~4-6GB VRAM  (feasible on RTX 3070/3080/3090)
    With 4-bit quantization (active): ~2.5-3.5GB VRAM (current default — fits GTX 1650 4GB)

# output format
    grpo_output/
      adapter_config.json       ← LoRA adapter config (rank, target modules, etc.)
      adapter_model.safetensors ← trained adapter weights (~80MB)
      tokenizer*.json           ← tokenizer files

    Load after training:
      base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
      model = PeftModel.from_pretrained(base, "grpo_output/")

# Implementation
## Milestones

- **Milestone 1:** Config (`grpo_system/config.py`)
- **Milestone 2:** Reward function (`grpo_system/reward.py`)
- **Milestone 3:** Model loading with LoRA (`grpo_system/model.py`)
- **Milestone 4:** Prompt builder (`grpo_system/data.py`)
- **Milestone 5:** Training loop (`grpo_system/train.py`)

---

## Milestone 1: Config

Define all training hyperparameters in one place.

### What to create

- `grpo_system/__init__.py` — empty, makes grpo_system a package
- `grpo_system/config.py` — `GRPOConfig` dataclass

### What each does

| Module | Class | Fields | Purpose |
|---|---|---|---|
| `config.py` | `GRPOConfig` | model_name, lora_r/alpha/dropout/target_modules, learning_rate, num_generations, max_prompt_length, max_completion_length, batch_size, num_train_epochs, kl_coef, output_dir | All training hyperparameters |

### Example workflow after Milestone 1

```python
from grpo_system.config import GRPOConfig

cfg = GRPOConfig()
print(cfg.model_name)        # Qwen/Qwen2.5-1.5B-Instruct
print(cfg.num_generations)   # 2
print(cfg.lora_r)            # 16
```

### How to verify after Milestone 1

```bash
uv run python tests/test_milestones/toy_test_m1.py
```

> **Stop here.** Run the test above and confirm it passes before continuing to Milestone 2.

---

## Milestone 2: Reward Function

Bridge `reward_system`'s `compute_reward()` into the function signature GRPOTrainer expects.

### What to create

- `grpo_system/reward.py` — `build_reward_fn(get_scores_fn)` factory

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `reward.py` | `build_reward_fn()` | `get_scores_fn(prompt, response) → dict` | `reward_fn(prompts, completions) → list[float]` |

### Example workflow after Milestone 2

```python
from grpo_system.reward import build_reward_fn

def mock_scores(prompt, response):
    return {"interactivity": 4, "warmth": 3, "directness": 5}

reward_fn = build_reward_fn(mock_scores)
rewards = reward_fn(["test prompt"], ["test response"])
print(rewards)   # [4.1]  (weighted sum from traits.json)
```

### How to verify after Milestone 2

```bash
uv run python tests/test_milestones/toy_test_m2.py
```

> **Stop here.** Run the test above and confirm it passes before continuing to Milestone 3.

---

## Milestone 3: Model Loading with LoRA

Load Qwen2.5-1.5B-Instruct in 4-bit quantization with LoRA adapters attached.

### What to create

- `grpo_system/model.py` — `load_model(config) → (model, tokenizer)`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `model.py` | `load_model()` | `GRPOConfig` | PEFT model with LoRA + tokenizer |

### Note on download

`load_model()` downloads Qwen2.5-1.5B-Instruct on first call (~3GB). This is cached by HuggingFace after the first run. Loads in 4-bit quantization — **requires a CUDA GPU with at least 4GB VRAM** (GTX 1650 or better).

### Example workflow after Milestone 3

```python
from grpo_system.config import GRPOConfig
from grpo_system.model import load_model

cfg = GRPOConfig()
model, tokenizer = load_model(cfg)
# Expected output: trainable params: ~10M || all params: ~1.5B || trainable%: ~0.6%
```

### How to verify after Milestone 3

```bash
uv run python tests/test_milestones/toy_test_m3.py
```

> **Stop here.** Run the test above and confirm it passes before continuing to Milestone 4.

---

## Milestone 4: Prompt Builder

Load 20 Newsgroups articles and format them as Qwen2.5-Instruct instruction prompts.

### What to create

- `grpo_system/data.py` — `load_articles()` and `format_for_qwen()`

### What each does

| Module | Function | Input | Output |
|---|---|---|---|
| `data.py` | `load_articles()` | max_articles, max_chars | list of instruction prompt strings |
| `data.py` | `format_for_qwen()` | prompt str + tokenizer | chat-template-formatted string |

### Example workflow after Milestone 4

```python
from grpo_system.data import load_articles

prompts = load_articles(max_articles=10)
print(len(prompts))       # 10 (or fewer if some articles are too short)
print(prompts[0][:120])   # "Rewrite the following article to be more engaging..."
```

### How to verify after Milestone 4

```bash
uv run python tests/test_milestones/toy_test_m4.py
```

> **Stop here.** Run the test above and confirm it passes before continuing to Milestone 5.

---

## Milestone 5: Training Loop

Wire all modules into a runnable GRPO training loop.

### What to create

- `grpo_system/train.py` — `train(config)` function and `get_scores_from_ui()` placeholder

### Workflow

```text
train.py
  ├─ load_model(config)                          → model, tokenizer
  ├─ load_articles() + format_for_qwen()         → Dataset
  ├─ build_reward_fn(get_scores_from_ui)         → reward_fn
  ├─ GRPOTrainer(model, config, dataset, reward_fn)
  ├─ trainer.train()                             → LoRA weights updated
  └─ model.save_pretrained(config.output_dir)   → grpo_output/
```

### How to verify after Milestone 5

Connect a real `get_scores_from_ui` function (replacing the `NotImplementedError` placeholder), then run:

```bash
uv run python grpo_system/train.py
```

> **Stop here.** Confirm the trainer initializes and begins a training step before considering this milestone complete.

---

# Testing

- Run individual milestone tests: `uv run python tests/test_milestones/toy_test_mX.py`
- Tests are self-contained — each imports only the modules needed for that milestone
- Milestones 1, 2, 4: no model downloads required
- Milestone 3: downloads Qwen2.5-1.5B-Instruct (~3GB, cached after first run)
- Milestone 5: requires GPU and a connected UI score collector