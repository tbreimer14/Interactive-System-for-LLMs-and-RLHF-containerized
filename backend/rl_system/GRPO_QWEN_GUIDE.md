# GRPO + Qwen2.5-3B-Instruct Implementation Guide

## Overview

This guide covers how to upgrade the RL system from the manual PPO prototype
(distilgpt2) to a GRPO-based training loop using Qwen2.5-3B-Instruct.

**Why GRPO instead of PPO:**
TRL 1.0.0 (the version in this project) removed `PPOTrainer`. The replacement
is `GRPOTrainer`, which eliminates the need for a separate critic/value model.
Instead of scoring one response and comparing it to a baseline, GRPO generates
multiple responses to the same prompt and uses their relative reward scores as
the training signal. This is lighter on memory and more stable to train.

**Why Qwen2.5-3B-Instruct:**
Qwen2.5-3B-Instruct is already instruction-tuned — it reliably follows prompts
like "rewrite this article to be more engaging" without any additional training.
RLHF on top of it nudges the model toward your specific style preferences rather
than teaching it to follow instructions from scratch, which is the correct and
productive application of RLHF.

---

## How GRPO Works

```
Single prompt: "Rewrite this article to be engaging: [article text]"
        ↓
Generate G responses (e.g. G=4):
    Response A  →  you score with sliders  →  reward: 4.1
    Response B  →  you score with sliders  →  reward: 2.3
    Response C  →  you score with sliders  →  reward: 3.8
    Response D  →  you score with sliders  →  reward: 1.9
        ↓
Advantage per response = reward - mean(all rewards)
    A: +0.825   B: -1.775   C: +0.725   D: -1.175
        ↓
Clipped policy gradient update (same math as PPO, no critic model)
        ↓
Model learns: A-style and C-style responses are preferred
```

Your sliders and `compute_reward()` are called once per response. Nothing in
the reward system changes.

---

## Memory Requirements and Solution

Loading Qwen2.5-3B-Instruct twice (trainable + frozen reference) at full
precision requires ~40-50GB VRAM — far beyond consumer hardware. The solution
is **LoRA** (Low-Rank Adaptation).

Instead of updating all 3 billion weights, LoRA attaches small trainable
adapter matrices to specific layers. The rest of the model stays frozen.
This reduces trainable parameters from 3B to roughly 10-30M, making training
feasible on a single consumer GPU (12-24GB VRAM).

```
Full fine-tuning:   3,000,000,000 trainable parameters  ~40GB+
LoRA fine-tuning:      20,000,000 trainable parameters  ~10-14GB
```

The frozen reference model can be offloaded to CPU during training steps to
save additional VRAM.

---

## Dependencies

These are already present in the project's `.venv`:

```
trl>=1.0.0          # GRPOTrainer
transformers        # Qwen2.5 model loading
peft                # LoRA adapters
bitsandbytes        # 4-bit quantization (optional, reduces VRAM further)
torch               # Training backend
```

Verify:
```bash
uv run python -c "import trl; print(trl.__version__)"   # should be 1.0.0
uv run python -c "from trl import GRPOTrainer; print('ok')"
uv run python -c "from peft import LoraConfig; print('ok')"
```

---

## Implementation

### Step 1 — Config

Create `backend/rl_system/grpo_system/config.py`:

```python
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # LoRA
    lora_r: int = 16                # rank — higher = more capacity, more memory
    lora_alpha: int = 32            # scaling factor, usually 2x lora_r
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")

    # GRPO
    learning_rate: float = 1e-5    # lower than PPO — model is already trained
    num_generations: int = 4        # G — responses generated per prompt
    max_prompt_length: int = 512
    max_completion_length: int = 256
    batch_size: int = 2             # prompts per step (each generates G responses)
    num_train_epochs: int = 1
    kl_coef: float = 0.1           # keeps policy close to reference

    # Output
    output_dir: str = "grpo_output"
```

---

### Step 2 — Reward Function

Create `backend/rl_system/grpo_system/reward.py`:

```python
"""
Bridges the existing reward_system's compute_reward() into the
function signature that GRPOTrainer expects.

GRPOTrainer calls reward_fn(prompts, completions) → list[float].
This wrapper collects UI slider scores and passes them through
the existing compute_reward() pipeline unchanged.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../reward_system"))

from app.trait_loader import load_traits
from app.reward_fn import compute_reward

TRAITS_PATH = "backend/reward_system/config/traits.json"


def build_reward_fn(get_scores_fn):
    """
    Factory that returns a reward function compatible with GRPOTrainer.

    Args:
        get_scores_fn: callable(prompt, response) -> dict[str, float]
            Collects slider scores from the UI for a given response.
            Example: {"interactivity": 4, "warmth": 3, "directness": 5}

    Returns:
        reward_fn compatible with GRPOTrainer's reward_funcs parameter.
    """
    traits = load_traits(TRAITS_PATH)

    def reward_fn(prompts, completions, **kwargs):
        """
        Called by GRPOTrainer once per batch.

        prompts:     list of prompt strings (length = batch_size * G)
        completions: list of response strings (length = batch_size * G)

        Returns: list of scalar rewards (one float per response)
        """
        rewards = []
        for prompt, response in zip(prompts, completions):
            # Collect scores from UI sliders for this specific response
            user_scores = get_scores_fn(prompt, response)

            # Run through existing reward computation — unchanged
            result = compute_reward(prompt, response, traits, user_scores)
            rewards.append(result["final_reward"])

        return rewards

    return reward_fn
```

---

### Step 3 — Model Loading with LoRA

Create `backend/rl_system/grpo_system/model.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from grpo_system.config import GRPOConfig


def load_model(config: GRPOConfig):
    """
    Load Qwen2.5-3B-Instruct with LoRA adapters.

    Returns:
        model     — Qwen2.5 with LoRA layers attached (trainable)
        tokenizer — tokenizer with padding set up
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left",    # Qwen2.5 requires left padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model — use bfloat16 to halve memory vs float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",      # automatically places layers across available GPU/CPU
    )

    # Attach LoRA adapters — only these layers will receive gradient updates
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: trainable params: ~20M || all params: ~3B || trainable%: ~0.6%

    return model, tokenizer
```

---

### Step 4 — Prompt Builder

Create `backend/rl_system/grpo_system/data.py`:

```python
"""
Builds GRPO training prompts from newsgroup articles.

Each prompt is a fixed instruction + one article.
The instruction stays constant — the article changes every step.
GRPO generates G responses per prompt and your sliders score each one.
"""

from sklearn.datasets import fetch_20newsgroups
from transformers import PreTrainedTokenizer


INSTRUCTION = (
    "Rewrite the following article to be more engaging. "
    "Your rewrite should encourage continued discussion, "
    "sound warm and approachable, and get to the point clearly.\n\n"
    "Article:\n"
)


def load_articles(max_articles: int = 500, max_chars: int = 800):
    """
    Load newsgroup articles and format as instruction prompts.

    Args:
        max_articles: how many articles to use (subset for manageable human rating)
        max_chars:    truncate articles to this length before embedding in prompt

    Returns:
        list of prompt strings, one per article
    """
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),  # strip email noise
    )

    prompts = []
    for text in dataset.data[:max_articles]:
        text = text.strip()
        if len(text) < 100:      # skip near-empty articles
            continue
        article = text[:max_chars]
        prompt = INSTRUCTION + article
        prompts.append(prompt)

    return prompts


def format_for_qwen(prompt: str, tokenizer: PreTrainedTokenizer) -> str:
    """
    Wrap prompt in Qwen2.5-Instruct chat template.
    Qwen2.5-Instruct expects messages in a specific format — this handles it.
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
```

---

### Step 5 — Training Loop

Create `backend/rl_system/grpo_system/train.py`:

```python
"""
Main GRPO training loop.

Wires together:
    - Qwen2.5-3B-Instruct + LoRA (model.py)
    - Newsgroup article prompts (data.py)
    - Human slider reward function (reward.py)
    - GRPOTrainer (trl)
"""

from datasets import Dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

from grpo_system.config import GRPOConfig
from grpo_system.model import load_model
from grpo_system.data import load_articles, format_for_qwen
from grpo_system.reward import build_reward_fn


def get_scores_from_ui(prompt: str, response: str) -> dict:
    """
    Placeholder for UI slider input.
    In the full pipeline this is replaced by the actual UI score collector.

    Returns dict matching traits.json keys:
        {"interactivity": float, "warmth": float, "directness": float}
    """
    # TODO: replace with real UI call
    # Example: return ui_client.collect_scores(prompt, response)
    raise NotImplementedError("Connect UI slider input here")


def train(config: GRPOConfig = None):
    if config is None:
        config = GRPOConfig()

    print("=== GRPO Training — Qwen2.5-3B-Instruct + LoRA ===")

    # 1. Load model
    model, tokenizer = load_model(config)

    # 2. Load and format prompts
    raw_prompts = load_articles(max_articles=200)
    formatted_prompts = [format_for_qwen(p, tokenizer) for p in raw_prompts]
    dataset = Dataset.from_dict({"prompt": formatted_prompts})

    # 3. Build reward function — wraps compute_reward() from reward_system
    reward_fn = build_reward_fn(get_scores_from_ui)

    # 4. Configure TRL's GRPOTrainer
    trl_config = TRLGRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,       # G — responses per prompt
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        kl_coef=config.kl_coef,
        bf16=True,                                    # bfloat16 training
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # 5. Train
    trainer.train()

    # 6. Save LoRA adapters only (not full 3B model — adapters are ~80MB)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"\nLoRA adapters saved to: {config.output_dir}/")


if __name__ == "__main__":
    train()
```

---

## File Structure

```
backend/rl_system/
├── ppo_system/              # existing PPO prototype (distilgpt2) — keep for reference
│   ├── config.py
│   ├── data.py
│   ├── trainer.py
│   └── main.py
│
├── grpo_system/             # new GRPO system (Qwen2.5-3B-Instruct)
│   ├── config.py            # GRPOConfig dataclass
│   ├── model.py             # Qwen2.5 + LoRA loading
│   ├── data.py              # newsgroup article prompt builder
│   ├── reward.py            # bridges reward_system into GRPOTrainer
│   └── train.py             # main training loop
│
└── GRPO_QWEN_GUIDE.md       # this file
```

---

## Integration with Existing Pipeline

```
20 Newsgroups articles
        ↓
data/raw/*.txt  →  RAG FAISS index  (retrieval context)
        ↓
grpo_system/data.py  →  instruction prompts  (training input)
        ↓
Qwen2.5-3B-Instruct generates G=4 responses per prompt
        ↓
UI displays responses → user moves sliders
        ↓
reward_system/app/reward_fn.py  →  scalar per response  (unchanged)
        ↓
GRPOTrainer computes relative advantages across G responses
        ↓
LoRA adapter weights updated  →  model saved
```

The reward system (`compute_reward()`, `traits.json`, slider weights) is
completely unchanged. The RL system is the only component that changes —
swapping the manual PPO loop and distilgpt2 for GRPOTrainer and Qwen2.5.

---

## Hardware Expectations

| Setup | VRAM needed | Training speed |
|---|---|---|
| GPU with 24GB (e.g. RTX 3090/4090) | fits with LoRA + bfloat16 | ~2-5 min/step |
| GPU with 12-16GB (e.g. RTX 3080) | tight — reduce batch_size to 1 | ~5-10 min/step |
| CPU only | not recommended for 3B model | very slow |

If VRAM is insufficient, add `load_in_4bit=True` to the `AutoModelForCausalLM`
call in `model.py` via `BitsAndBytesConfig`. This halves VRAM at a small quality cost.

---

## Loading a Trained Model

After training, load the LoRA adapters on top of the base model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "grpo_output/")
tokenizer = AutoTokenizer.from_pretrained("grpo_output/")
```

The base model weights are unchanged. Only the LoRA adapters encode what
was learned from your slider ratings.
