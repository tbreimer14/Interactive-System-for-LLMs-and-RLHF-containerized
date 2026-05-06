# RL System Documentation

## Overview

This is a **GRPO-based online RLHF fine-tuning system** for Qwen2.5-Instruct. Two configs are available: `local_config()` (1.5B, 4-bit QLoRA, GTX 1650) and `colab_config()` (3B, bfloat16, Google Colab T4). The core idea:

1. The model generates **G responses** to the same instruction prompt (rewriting a newsgroup article)
2. The **user scores each response** using UI sliders, one score per trait
3. The reward system converts those scores into a **scalar reward per response** (true per-completion GRPO)
4. After every 5 complete grading rounds, GRPO computes **relative advantages** across the G responses (reward minus group mean)
5. A **policy gradient update** nudges the model toward responses the user rated higher
6. Only **LoRA adapter weights** are updated — the base model is never modified

### Why GRPO instead of PPO

TRL 1.0.0 removed `PPOTrainer`. The replacement is `GRPOTrainer`, which eliminates the need for a separate critic/value model. Instead of scoring one response and comparing to a baseline, GRPO generates a group of responses and uses their relative rewards as the training signal — lighter on memory and more stable.

### Why Qwen2.5-Instruct

Both model sizes (1.5B local, 3B Colab) are already instruction-tuned, so they follow prompts like "rewrite this article to be more engaging" without additional training. RLHF on top nudges the model toward your specific style preferences (defined in `traits.json`) rather than teaching instruction-following from scratch.

### Architecture

```
20 Newsgroups articles (sklearn)
    ↓
grpo_system/data.py  →  instruction prompts formatted for Qwen2.5-Instruct
    ↓
Qwen2.5-1.5B-Instruct + LoRA adapters (grpo_system/model.py)
    shared model for both inference and online training
    generates G responses per prompt (G=2 locally, G=4 on Colab)
    ↓
UI: user rates each response with sliders (per-completion)
    reward = Σ(weight_i × user_score_i)
    ↓
Every 3 complete rounds → grpo_system/online_step.py
    advantage_i = (reward_i − mean(group)) / std(group)
    policy gradient update on LoRA adapter weights (AdamW, in place)
    KL reference via disable_adapter_layers() — no second model copy
    ↓
Session page → Save Session → sessions/{name}/online_checkpoint/
```

---

## System Components

### 1. **Config** (`grpo_system/config.py`)

**Purpose:** Centralize all training hyperparameters in one dataclass. To tune training, change only this file — no other module needs to be touched.

**Key Class:** `GRPOConfig` — plus two preset functions: `local_config()` and `colab_config()`

| Field | Local default | Colab default | Purpose |
|---|---|---|---|
| `model_name` | `"Qwen/Qwen2.5-1.5B-Instruct"` | `"Qwen/Qwen2.5-3B-Instruct"` | HuggingFace model ID |
| `use_4bit` | `True` | `False` | 4-bit QLoRA (local) vs bfloat16 (Colab) |
| `lora_r` | `16` | `16` | LoRA rank — higher = more capacity, more memory |
| `lora_alpha` | `32` | `32` | LoRA scaling factor, typically 2× lora_r |
| `lora_dropout` | `0.05` | `0.05` | Dropout applied to LoRA layers |
| `lora_target_modules` | `("q_proj", "v_proj", "k_proj", "o_proj")` | same | Attention layers that receive adapters |
| `learning_rate` | `1e-5` | `1e-5` | Lower than PPO — model is already instruction-tuned |
| `num_generations` | `2` | `4` | G — responses generated per prompt for comparison |
| `max_prompt_length` | `256` | `512` | Token limit for the input prompt |
| `max_completion_length` | `128` | `256` | Token limit for each generated response |
| `batch_size` | `1` | `2` | Prompts per training step (each generates G responses) |
| `num_train_epochs` | `1` | `1` | Training epochs over the article dataset |
| `kl_coef` | `0.1` | `0.1` | KL penalty weight — keeps policy close to reference |
| `output_dir` | `"grpo_output"` | `"grpo_output"` | Where LoRA adapters are saved after training |

**Design note:** `lora_target_modules` targets all four attention projection matrices. Targeting all four (not just q/v) captures more of the model's behavior at modest cost.

---

### 2. **Reward Function** (`grpo_system/reward.py`)

**Purpose:** Bridge the existing `reward_system`'s `compute_reward()` into the function signature that `GRPOTrainer` expects. The reward system itself is completely unchanged — this module is a thin wrapper.

**Key Function:** `build_reward_fn(get_scores_fn) → reward_fn`

- `get_scores_fn(prompt, response) → dict[str, float]` — callable that collects UI slider scores for one response
- Returns `reward_fn(prompts, completions) → list[float]` — the exact signature GRPOTrainer requires

**How it connects to reward_system:**

```
GRPOTrainer calls:
    reward_fn(prompts, completions)
        ↓
    for each (prompt, response):
        user_scores = get_scores_fn(prompt, response)   ← from UI sliders
        result = compute_reward(prompt, response, traits, user_scores)
        rewards.append(result["final_reward"])
        ↓
    return rewards   (list[float], length = batch_size × G)
```

**Traits path:** `TRAITS_PATH = "backend/reward_system/config/traits.json"` — the same config file the reward system and UI share.

---

### 3. **Model Loading with LoRA** (`grpo_system/model.py`)

**Purpose:** Load a Qwen2.5-Instruct model and attach LoRA adapter layers. Only the LoRA layers receive gradient updates during training — the base weights stay frozen. Automatically uses 4-bit QLoRA or bfloat16 based on `config.use_4bit`.

**Key Function:** `load_model(config: GRPOConfig) → (model, tokenizer)`

**What LoRA does:**

```
                        Local (1.5B, 4-bit)     Colab (3B, bfloat16)
Full fine-tuning:       ~20GB+ VRAM             ~40GB+ VRAM
LoRA fine-tuning:       ~2.5-3.5GB VRAM         ~10-14GB VRAM
Trainable params:       ~4.4M / 1.5B (0.3%)     ~8M / 3B (0.3%)
```

**Loading details:**
- `use_4bit=True` — `BitsAndBytesConfig` with double quantization; compute dtype stays bfloat16
- `use_4bit=False` — plain `torch_dtype=torch.bfloat16`
- `device_map="auto"` — distributes layers across available GPU/CPU automatically
- `padding_side="left"` — required by Qwen2.5 for correct batch generation
- `pad_token = eos_token` — Qwen2.5's tokenizer has no pad token by default

**Memory requirements:**

| GPU VRAM | Config | Status |
|---|---|---|
| 4GB (GTX 1650) | `local_config()` — 4-bit QLoRA | Current local default |
| 12–16GB (RTX 3080/3070) | `colab_config()` — bfloat16, batch_size=1 | Workable |
| 15GB (Colab T4) | `colab_config()` — bfloat16 | Fits comfortably |
| 24GB (RTX 3090/4090) | `colab_config()` — bfloat16 | Fits with headroom |
| CPU only | Not recommended | Very slow |

---

### 4. **Prompt Builder** (`grpo_system/data.py`)

**Purpose:** Load newsgroup articles from sklearn and format them as instruction prompts for Qwen2.5-Instruct's chat template. The instruction is fixed — only the article changes per prompt.

**Key Functions:**

| Function | Input | Output |
|---|---|---|
| `load_articles(max_articles, max_chars)` | int, int | list of raw instruction prompt strings |
| `format_for_qwen(prompt, tokenizer)` | str, tokenizer | chat-template-formatted string |

**Instruction used:**
```
Rewrite the following article to be more engaging.
Your rewrite should encourage continued discussion,
sound warm and approachable, and get to the point clearly.

Article:
[article text truncated to max_chars]
```

**Why this instruction:** It maps directly to the default traits in `traits.json` (interactivity → "encourage continued discussion", warmth → "warm and approachable", directness → "get to the point clearly"). The model is told exactly what to optimize for, and the user's sliders measure exactly that.

**Article filtering:**
- Strips email headers, footers, and quoted lines (sklearn `remove` parameter)
- Drops articles shorter than 100 characters
- Truncates to `max_chars=800` to stay within prompt token budget

**`format_for_qwen()`** wraps the prompt in Qwen2.5-Instruct's expected chat format:
```
<|im_start|>user
[prompt]<|im_end|>
<|im_start|>assistant
```
GRPOTrainer then generates the completion (the article rewrite) from that point.

---

### 5. **Online Training Step** (`grpo_system/online_step.py`)

**Purpose:** Perform one GRPO gradient update on the shared live model using a buffer of human-graded rounds. Called automatically from `OnlineGRPOSession.step()` after every 3 complete grading rounds.

**Key Function:** `grpo_step(model, tokenizer, groups, optimizer, beta=0.1)`

**Step-by-step:**

```
groups = [{prompt, completions: [{text, reward}, ...]}]  # 3 rounds

for each group:
    rewards = [c["reward"] for c in completions]
    advantages = (rewards − mean) / (std + ε)       # group-relative
    for each (completion, advantage):
        lp  = _seq_log_prob(model, prompt, text)     # LoRA active, grad flows
        model.disable_adapter_layers()
        ref = _seq_log_prob(model, prompt, text)     # base reference, no grad
        model.enable_adapter_layers()
        loss += −advantage × lp + beta × (lp − ref) # GRPO + KL

torch.stack(loss_terms).mean().backward()
optimizer.step()                                     # AdamW, LoRA params only
```

The `disable_adapter_layers()` trick computes the KL reference using the same model object with adapters temporarily disabled — no second model instance needed.

### 5b. **Standalone Batch Trainer** (`grpo_system/train.py`)

A standalone script that reads a pre-built interaction log and runs TRL's `GRPOTrainer` in a full batch pass. Not wired to the UI — intended for offline runs on larger hardware (e.g., Colab T4 with the 3B model).

**TRL config fields:**

| Field | Value | Meaning |
|---|---|---|
| `num_generations` | 2–4 | G — responses per prompt |
| `kl_coef` | 0.1 | Penalty for drifting far from the reference model |
| `bf16` | True | bfloat16 training |
| `logging_steps` | 1 | Log loss/reward after every step |
| `per_device_train_batch_size` | 1–2 | Prompts per GPU step |

---

## Configuration

**File:** `backend/reward_system/config/traits.json`

The RL system uses the same `traits.json` as the reward system and UI — no separate config. Edit traits there to change what the reward function measures. `reward.py` loads it via `TRAITS_PATH` at training time.

---

## Loading a Trained Model

After training, the base model is unchanged. Only the LoRA adapters in `sessions/{name}/online_checkpoint/` encode what was learned from slider ratings. Use the same model name and quantization settings as your training config.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# local_config() trained model (4-bit QLoRA):
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# colab_config() trained model (bfloat16):
# base = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2.5-3B-Instruct",
#     dtype=torch.bfloat16,
#     device_map="auto",
# )

model = PeftModel.from_pretrained(base, "sessions/my_session/online_checkpoint/")
tokenizer = AutoTokenizer.from_pretrained("sessions/my_session/online_checkpoint/")
```

To permanently merge adapters into the base weights (via Session page → Merge & Export, or manually):
```python
merged = model.merge_and_unload()
merged.save_pretrained("sessions/my_session/merged_model/")
```

---

## Usage

### Run training (after connecting UI scorer)

```bash
cd backend/rl_system
uv run python grpo_system/train.py
```

### Run with the local config (GTX 1650)

```python
from grpo_system.config import local_config
from grpo_system.train import train

train(local_config())
```

### Run with the Colab config (T4 / 3B model)

```python
from grpo_system.config import colab_config
from grpo_system.train import train

train(colab_config())
```

### Run with a custom config

```python
from grpo_system.config import GRPOConfig
from grpo_system.train import train

cfg = GRPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    use_4bit=True,
    batch_size=1,
    num_generations=2,
    learning_rate=5e-6,
    output_dir="my_run/",
)
train(cfg)
```

---

## Testing

### Run milestone tests

```bash
uv run python tests/test_milestones/toy_test_m1.py  # Config
uv run python tests/test_milestones/toy_test_m2.py  # Reward function
uv run python tests/test_milestones/toy_test_m3.py  # Model + LoRA
uv run python tests/test_milestones/toy_test_m4.py  # Prompt builder
uv run python tests/test_milestones/toy_test_m5.py  # Training loop (requires UI + GPU)
```

**Run time:**
- Milestones 1–2: ~2–5 seconds (no downloads)
- Milestone 3: ~5–15 minutes on first run (downloads ~6GB model, cached after)
- Milestone 4: ~15 seconds (downloads ~15MB dataset, cached after)
- Milestone 5: depends on GPU and how many prompts are scored

### Verify dependencies before starting

```bash
uv run python -c "import trl; print(trl.__version__)"     # should be 1.0.0+
uv run python -c "from trl import GRPOTrainer; print('ok')"
uv run python -c "from peft import LoraConfig; print('ok')"
```

---

## Architecture Decisions

1. **GRPO over PPO:** TRL 1.0.0 removed `PPOTrainer`. GRPO requires no value/critic model — it uses relative rewards within a response group as the training signal, which is more memory-efficient and stable for this use case.

2. **LoRA over full fine-tuning:** A Qwen2.5 model trained with GRPO requires the base model plus a frozen reference copy. Full fine-tuning would need 20–80GB+ VRAM. LoRA keeps the base frozen and only trains ~10–20M adapter parameters, making the setup feasible on consumer hardware with quantization.

3. **Two configs (local / Colab):** `local_config()` uses Qwen2.5-1.5B-Instruct with 4-bit QLoRA to fit a 4GB GPU. `colab_config()` uses Qwen2.5-3B-Instruct in bfloat16 for Google Colab's T4 (15GB VRAM). Both use the same LoRA rank and target modules — the adapter format is identical, only the base model differs. An instruction-tuned base means RLHF refines *style* (the slider traits), not basic instruction-following.

4. **Same traits.json as reward system and UI:** One config file drives all three systems. Changing a trait weight in `traits.json` immediately affects what the reward function optimizes, what the UI presents for scoring, and what the training loop measures.

5. **LoRA adapters saved, not the full model:** The `grpo_output/` directory is ~80MB (just the adapter weights). The 3B base model is never re-saved — it stays in the HuggingFace cache and is loaded fresh at inference time. This makes versioning and experiment tracking practical.

6. **`get_scores_from_ui` as a seam:** The `train.py` placeholder raises `NotImplementedError` intentionally. This forces the caller to explicitly wire up the UI before training can start — there is no silent fallback to fake scores.

---

## Local Training on 4GB VRAM (GTX 1650 / Laptop GPUs)

The default config is tuned for local training on a 4GB GPU (e.g., NVIDIA GTX 1650). The following changes were made from the original 3B design to make this work:

| Setting | Original | Local GPU | Reason |
|---|---|---|---|
| `model_name` | `Qwen2.5-3B-Instruct` | `Qwen2.5-1.5B-Instruct` | 3B requires 6–8GB VRAM even with LoRA; 1.5B fits in 4GB with 4-bit |
| `batch_size` | `2` | `1` | Each batch generates `num_generations` responses — batch=2 doubles peak VRAM |
| `num_generations` | `4` | `2` | GRPO generates G responses per prompt simultaneously; G=4 at 4GB will OOM |
| `max_prompt_length` | `512` | `256` | Shorter sequences reduce KV-cache memory during generation |
| `max_completion_length` | `256` | `128` | Same — limits how much VRAM each generated response consumes |
| Quantization | None (bfloat16) | 4-bit QLoRA (`BitsAndBytesConfig`) | Cuts model weight memory from ~3GB to ~0.75GB |

**To run on a larger GPU** (e.g., Google Colab T4/A100): switch back to `Qwen2.5-3B-Instruct`, remove the `BitsAndBytesConfig` block in `model.py`, and restore `batch_size=2`, `num_generations=4`, `max_prompt_length=512`, `max_completion_length=256`.

**Expected VRAM usage with local config:** ~2.5–3.5GB peak during generation steps.

---

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'grpo_system'`
**Solution:** Run scripts from `backend/rl_system/`:
```bash
cd backend/rl_system
uv run python grpo_system/train.py
```

### Problem: CUDA out of memory
**Solution (in order of impact):**
1. Reduce `batch_size` to `1` in `GRPOConfig`
2. Reduce `num_generations` to `2`
3. Add 4-bit quantization via `BitsAndBytesConfig` (see Usage section above)

### Problem: `ValueError: Tokenizer does not have a padding token`
**Solution:** This is handled in `model.py` (`tokenizer.pad_token = tokenizer.eos_token`). If you see this error, confirm your `model.py` includes that line after loading the tokenizer.

### Problem: `NotImplementedError: Connect UI slider input here`
**Solution:** Expected. Replace the `get_scores_from_ui` stub in `train.py` with a real call to the UI score collector before running training.

### Problem: `ImportError` for `reward_system` modules in `reward.py`
**Solution:** The `sys.path.append` in `reward.py` points to `../../../reward_system` relative to the `grpo_system/` directory. Run from `backend/rl_system/` so the relative path resolves correctly.

---

## Files Overview

```
backend/rl_system/
├── grpo_system/
│   ├── __init__.py          # package marker
│   ├── config.py            # GRPOConfig + local_config() + colab_config()
│   ├── reward.py            # bridges reward_system into GRPOTrainer (standalone use)
│   ├── model.py             # Qwen2.5 + LoRA loading (4-bit or bfloat16)
│   ├── online_step.py       # custom GRPO gradient step for online training
│   ├── data.py              # newsgroup article prompt builder
│   └── train.py             # standalone batch training script (not wired to UI)
├── tests/
│   └── test_milestones/
│       ├── toy_test_m1.py   # Config tests
│       ├── toy_test_m2.py   # Reward function tests
│       ├── toy_test_m3.py   # Model + LoRA tests
│       ├── toy_test_m4.py   # Prompt builder tests
│       ├── toy_test_m5.py   # Batch training loop tests
│       └── toy_test_m6.py   # Online step tests
├── GRPO_QWEN_GUIDE.md       # implementation guide (code-level)
├── RL_README.md             # pipeline overview + milestone breakdown
└── RL_DOCUMENTATION.md      # complete documentation (this file)
```