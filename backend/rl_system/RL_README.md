# RL System

## Pipeline

```
20 Newsgroups dataset (sklearn)
  ↓
data.py builds instruction prompts
(article text formatted for Qwen2.5-Instruct chat template)
  ↓
Qwen2.5-1.5B-Instruct + LoRA generates 2–4 responses per prompt
(model.py loads base model in 4-bit QLoRA — shared for inference and training)
  ↓
UI displays responses → user scores each with sliders per trait
reward = Σ(weight_i × slider_score_i)
  ↓
[Online — Grade page, automatic after every 3 complete rounds]
  online_step.py: grpo_step(model, tokenizer, groups, optimizer)
    policy log-prob   (LoRA active, gradients flow)
    reference log-prob (disable_adapter_layers, no grad, no second model)
    loss = −advantage × log_prob + β × KL
    AdamW step → LoRA weights updated in place
    Next generation uses updated model automatically
  ↓
[When happy with the model — Session page]
  Save Session → writes LoRA weights to sessions/{name}/online_checkpoint/
  Export:
    Option A: LoRA adapter ZIP   (~80MB, load with PEFT)
    Option B: Merged full model  (~3GB, no PEFT needed)
```

---

## How GRPO Works

```
Single prompt → G completions generated in parallel
Each completion scored individually by the human → scalar reward
Advantage per completion = (reward − mean(group)) / std(group)
Policy gradient: increase probability of high-advantage completions
KL penalty: keep model close to base (prevents catastrophic forgetting)
No critic / value model needed — relative rewards within the group suffice
```

Rewards are **per-completion** (true GRPO): each response the user scores becomes one data point in the group. This is stronger than prompt-level averaged rewards because the model gets a precise signal about which specific output style was preferred.

---

## Shared Model Architecture

The same model instance is used for both inference and training:

```
@st.cache_resource  →  load Qwen2.5-1.5B + LoRA (4-bit QLoRA)
         ↓                      ↓
  generate_responses()    OnlineGRPOSession.step()
  (model.eval())          (model.train(), AdamW on LoRA params)
```

LoRA B matrices are zero-initialized, so the model is functionally identical to the base at startup. After each training step, the updated weights are live for the next generation call.

KL reference is computed with `model.disable_adapter_layers()` — no second model copy needed.

---

## Memory Requirements

| Config | Model | VRAM | Setup |
|---|---|---|---|
| Full fine-tune Qwen2.5-1.5B | 1.5B | ~20GB | Infeasible on consumer hardware |
| LoRA (rank 16) bfloat16 | 1.5B | ~4–6GB | RTX 3070+ |
| 4-bit QLoRA (default) | 1.5B | ~2.5GB | GTX 1650 (4GB) |

Online training adds ~240MB (LoRA params + optimizer state) on top of inference.

---

## Output Format

```
sessions/<name>/online_checkpoint/    ← written by Save Session on Session page
  adapter_config.json                 ← LoRA config (rank, target modules, etc.)
  adapter_model.safetensors           ← trained LoRA weights (~80MB)
  tokenizer*.json                     ← tokenizer files

sessions/<name>/merged_model/         ← optional, created by Merge & Export
  config.json
  model.safetensors                   ← full merged weights (~3GB)
  tokenizer*.json
```

Loading LoRA adapters:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "sessions/default/online_checkpoint/")
```

Loading merged model (no PEFT needed):
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("sessions/default/merged_model/")
```

---

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `GRPOConfig` dataclass, `local_config()`, `colab_config()` |
| `model.py` | Load Qwen2.5 + LoRA (4-bit or bfloat16) |
| `online_step.py` | Custom GRPO gradient step for online training |
| `reward.py` | `build_reward_fn` bridge (standalone / batch use) |
| `data.py` | 20 Newsgroups → instruction prompt strings |
| `train.py` | Standalone batch training script (not wired to UI) |

---

## Milestones

- **Milestone 1:** Config (`grpo_system/config.py`)
- **Milestone 2:** Reward function (`grpo_system/reward.py`)
- **Milestone 3:** Model loading with LoRA (`grpo_system/model.py`)
- **Milestone 4:** Prompt builder (`grpo_system/data.py`)
- **Milestone 5:** Batch training loop (`grpo_system/train.py`)
- **Milestone 6:** Online training step (`grpo_system/online_step.py`)

### How to verify

```bash
uv run python tests/test_milestones/toy_test_mX.py
```

Milestones 1, 2, 4: no model downloads required.
Milestone 3: downloads Qwen2.5-1.5B-Instruct (~3GB, cached after first run).
Milestone 5+: requires GPU.
