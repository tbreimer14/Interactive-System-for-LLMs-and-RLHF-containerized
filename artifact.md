# LLMTuner

An interactive system for behavioral fine-tuning of language models through human feedback. A human annotator steers the behavior of a language model by rating its outputs through a Streamlit UI. Ratings drive **online GRPO** (Group Relative Policy Optimization) training: the model updates in place in real time after every 3 grading rounds. All work is organized into isolated experiment sessions so that different training runs never interfere with each other.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites and Installation](#3-prerequisites-and-installation)
4. [Project Structure](#4-project-structure)
5. [Experiment Sessions](#5-experiment-sessions)
6. [Running the UI](#6-running-the-ui)
7. [UI Pages Reference](#7-ui-pages-reference)
8. [Data Pipeline](#8-data-pipeline)
9. [Reward System](#9-reward-system)
10. [Model Generation](#10-model-generation)
11. [GRPO Training](#11-grpo-training)
12. [Exporting a Trained Model](#12-exporting-a-trained-model)
13. [Supporting Backends (Not Yet Integrated)](#13-supporting-backends-not-yet-integrated)
14. [Hardware Requirements](#14-hardware-requirements)

---

## 1. System Overview

The system has one goal: teach a language model to write in a style that a specific human prefers, using that human's real-time ratings as the only training signal.

**End-to-end flow:**

```
20 Newsgroups dataset
        ↓
Prompt page — human picks an article and writes an instruction
        ↓
Grade page — Qwen2.5-1.5B-Instruct + LoRA generates 2 or 4 responses
        ↓
Human sets target response length with a slider
        ↓
Human scores each response individually with sliders across defined traits
        ↓
Scalar reward = Σ(weight_i × score_i) saved to session's interaction log
        ↓
[Every 3 complete grading rounds — online GRPO fires automatically]
  Per-completion group-relative advantages computed
  LoRA weights updated in place via AdamW
  Next generation uses the refined model immediately
        ↓
Session page — Save Session writes LoRA weights to disk
        ↓
Export — download adapter ZIP or merge into a standalone model (user-triggered)
```

> **Export rule:** Export reads from the saved checkpoint on disk. Always **Save Session** before exporting.

The model rewrites real articles from a curated corpus. The human defines evaluation traits once (in Setup) and then only rates outputs. Multiple independent sessions let you train toward different styles simultaneously without interference.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI  (app.py)                      │
│                                                                 │
│  Sidebar:  Session Switcher  │  Navigation (6 pages)           │
│                                                                 │
│  Setup      → trait_manager.py    (load/edit reward traits)    │
│  Prompt     → sklearn 20NG        (browse articles)            │
│  Grade      → backend_adapter.py  (generate + score + save)   │
│               grpo_adapter.py     (online GRPO after 3 rounds) │
│  History    → ui/storage.py       (read JSONL log)             │
│  Analytics  → pandas / st.chart   (reward trends)              │
│  Session    → grpo_adapter.py     (save/load weights + export) │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │                            │
  backend_adapter.py           grpo_adapter.py
  ├─ _load_model()             ├─ OnlineGRPOSession (online)
  │  @st.cache_resource         │   grpo_system/online_step.py
  │  Qwen2.5-1.5B + LoRA        │   AdamW on shared model
  │  4-bit QLoRA                ├─ save_online_checkpoint()
  └─ generate_responses()       ├─ load_online_checkpoint()
     model.eval()               ├─ reset_lora_weights()
                                ├─ zip_adapter_dir()
  app/reward_bridge.py          └─ merge_and_export()
  └─ compute_reward()
     weighted sum               sessions/{name}/
                                ├─ interaction_log.jsonl
                                └─ online_checkpoint/
                                   (LoRA weights, saved by user)
```

**Module responsibilities:**

| Module | Role |
|---|---|
| `app.py` | Entry point; sidebar session switcher + page routing; auto-loads checkpoint on startup/switch |
| `backend_adapter.py` | Shared Qwen2.5+LoRA model for inference and online training; reward bridge |
| `grpo_adapter.py` | `OnlineGRPOSession` (online step) + checkpoint save/load + `reset_lora_weights()` + export helpers |
| `grpo_system/online_step.py` | Custom GRPO gradient step; uses `disable_adapter_layers()` for KL reference |
| `ui/session_manager.py` | Path resolution per named session; create / list sessions; `online_checkpoint_dir()` |
| `ui/state.py` | All `st.session_state` keys and accessors (including round buffer for online training) |
| `ui/components.py` | All page-level rendering functions; online round tracking on Grade page |
| `ui/storage.py` | JSONL read/write for interaction log |
| `ui/types.py` | `TraitConfig`, `ScoredTrait`, `InteractionLog` dataclasses |
| `app/trait_manager.py` | Loads `config/traits.json` |
| `app/reward_bridge.py` | `compute_reward(scores, traits) → (scored_traits, scalar)` |

---

## 3. Prerequisites and Installation

**Requirements:**
- Python ≥ 3.11
- `uv` package manager (recommended)
- GPU with ≥ 4 GB VRAM for comfortable generation (CPU works but is slow)

**Install:**

```bash
# From the project root
uv sync
```

**Run the UI:**

```bash
cd backend/user_interface
uv run streamlit run app.py
```

Opens at `http://localhost:8501`.

**Verify key packages:**

```bash
uv run python -c "from trl import GRPOTrainer; print('GRPO ok')"
uv run python -c "from peft import LoraConfig; print('LoRA ok')"
uv run python -c "import streamlit; print('Streamlit ok')"
uv run python -c "from sklearn.datasets import fetch_20newsgroups; print('dataset ok')"
```

---

## 4. Project Structure

```
.
├── pyproject.toml
├── artifact.md                          ← this file
└── backend/
    ├── data/                            ← 20 Newsgroups preprocessing pipeline
    │   ├── app/
    │   │   ├── loader.py                # fetch_20newsgroups wrapper
    │   │   ├── cleaner.py               # normalize whitespace, remove dividers
    │   │   └── exporter.py              # save to .txt files
    │   └── scripts/
    │       ├── preprocess.py            # run the full pipeline
    │       └── verify.py
    │
    ├── reward_system/                   ← scalar reward computation
    │   ├── app/
    │   │   ├── trait_loader.py          # load traits.json
    │   │   ├── reward_fn.py             # weighted sum → RewardResult
    │   │   ├── schemas.py               # Trait, RewardResult dataclasses
    │   │   └── logger.py
    │   └── config/
    │       └── traits.json              # reward_system's own trait config
    │
    ├── rl_system/                       ← GRPO training engine
    │   └── grpo_system/
    │       ├── config.py                # GRPOConfig dataclass (local + colab presets)
    │       ├── model.py                 # load Qwen2.5 + LoRA adapters
    │       ├── online_step.py           # custom GRPO gradient step for online training
    │       ├── data.py                  # build newsgroup prompts for training
    │       ├── reward.py                # build_reward_fn() factory (standalone use)
    │       └── train.py                 # standalone batch training script (not wired to UI)
    │
    ├── api/                             ← FastAPI REST layer (built, not yet integrated)
    │   ├── app.py
    │   ├── api/
    │   │   ├── routes.py
    │   │   └── schemas.py
    │   └── services/
    │       ├── generation_service.py
    │       ├── reward_service.py
    │       └── storage_service.py
    │
    └── user_interface/                  ← Streamlit app (main entry point)
        ├── app.py                       # entry point + sidebar + auto-load checkpoint
        ├── backend_adapter.py           # shared model (inference + online training) + reward
        ├── grpo_adapter.py              # OnlineGRPOSession + checkpoint helpers + export
        ├── .streamlit/
        │   └── config.toml              # fileWatcherType = none (suppresses torchvision warnings)
        ├── config/
        │   └── traits.json              # default trait definitions
        ├── sessions/                    # auto-created; one dir per experiment session
        │   └── {session_name}/
        │       ├── interaction_log.jsonl
        │       └── online_checkpoint/   # LoRA weights, written by Save Session
        ├── app/
        │   ├── trait_manager.py         # load_traits(config_path)
        │   └── reward_bridge.py         # compute_reward(scores, traits)
        └── ui/
            ├── components.py            # all page rendering functions
            ├── state.py                 # st.session_state keys + accessors
            ├── session_manager.py       # session path resolution
            ├── storage.py               # JSONL read/write
            └── types.py                 # TraitConfig, ScoredTrait, InteractionLog
```

---

## 5. Experiment Sessions

### What a session is

A **session** is one isolated RLHF experiment. Each session has its own:

- **Interaction log** — every scored response saved during grading
- **Online checkpoint** — the LoRA adapter weights saved manually via the Session page

Sessions live under `backend/user_interface/sessions/{name}/`:

```
sessions/
├── default/                        ← created automatically on first launch
│   ├── interaction_log.jsonl       ← grading data (prompt, response, traits, reward)
│   └── online_checkpoint/          ← LoRA weights (written by Save Session)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
├── demo_session/
│   └── ...
└── warm_tone/
    └── ...
```

### Creating and switching sessions

The **sidebar session panel** (always visible, above the page navigation) shows the active session as a dropdown. Switching sessions:
1. Resets all in-memory state (history, round buffer, optimizer, step count)
2. Zeros all LoRA B matrices (restores base model behavior)
3. Auto-loads the new session's checkpoint if one exists

To create a new session, use the **"New session"** expander in the sidebar:
- Enter a name (letters, digits, `_`, `-` only; max 64 characters)
- Click **Create & Switch** — the directory is created and becomes active immediately

### Auto-load on startup

On each fresh app start, if the active session has a saved checkpoint, it is loaded automatically (once per browser session, tracked by a `KEY_WEIGHTS_LOADED` flag).

### Interaction log format

Every saved interaction appends one JSON line to the session's `interaction_log.jsonl`:

```json
{
  "timestamp": "2026-04-28T20:00:00",
  "prompt": "Article text here...\n\nRewrite this to be more engaging.",
  "response": "The model's rewrite...",
  "traits": [
    {"name": "clarity",    "score": 4, "weight": 0.4, "contribution": 1.6},
    {"name": "warmth",     "score": 5, "weight": 0.3, "contribution": 1.5},
    {"name": "directness", "score": 3, "weight": 0.3, "contribution": 0.9}
  ],
  "scalar_reward": 4.0
}
```

---

## 6. Running the UI

```bash
cd backend/user_interface
uv run streamlit run app.py
```

On first launch, the `sessions/default/` directory is created automatically. The app opens at `http://localhost:8501`.

**The sidebar** has two sections:
1. **Experiment Session** — dropdown of existing sessions + "New session" expander
2. **Navigate** — the six page tabs

Everything below — interaction log, online checkpoint, exported model — belongs to whichever session is selected in the dropdown.

---

## 7. UI Pages Reference

### Setup — Reward Traits

Define the traits used to score model responses. Each trait has a name, description, and weight.

- Add traits with the **"Add a trait"** expander
- Adjust weights inline; remove any trait with **Remove**
- Click **Confirm Traits** to lock them in for the current grading session
- Trait definitions are shared across all experiment sessions (they live in `config/traits.json`); change them here to affect scoring globally

### Prompt — Dataset & Prompt

Browse the 20 Newsgroups dataset and build the prompt that will be sent to the model.

**Load Dataset** (expander):
- Select one of the 20 newsgroup categories from the dropdown
- Set min/max character filters, then click **Load**
- Posts are fetched via `sklearn.datasets.fetch_20newsgroups` and cached in session state

**Choose a Post** (appears after loading):
- **← Prev / Next →** buttons or a numeric jump input to navigate posts
- 130-line preview of the current post
- **Use as Article →** pushes the full post text into the Article field below

**Article and Instruction fields** (form):
- **Article** — the text the model will work with
- **Instruction** — what the model should do (e.g., "Rewrite this to be more engaging and warm"). The last-used instruction is saved to `config/preset_instruction.txt` and pre-filled on next visit.
- **Set Prompt** saves both fields to session state; **Clear** wipes them

### Grade — Response & Scoring

Generate model responses and score them.

**Active Prompt** (collapsible expander at top) — shows the current article and instruction.

**Controls row:**
- **Responses radio** — 2 or 4 responses
- **Response length slider** — 25–400 words (step 25). The target word count is appended to the generation prompt as a soft instruction. The original prompt (without the hint) is saved to the log and the round buffer.
- **Generate Responses** — calls Qwen2.5-1.5B-Instruct. Disabled while a training step is running.
- **Clear All** — resets all responses and saved flags

**Scoring** (per response):
- One slider per trait (−5 to +5)
- Live reward breakdown: per-trait `score × weight = contribution`, and the final scalar reward
- **Save Response** button — writes to `interaction_log.jsonl`, adds to History, and records into the current online round

**Online training status banner** (top of page, when a session is active):
- Rounds buffered (X/3)
- Training in progress indicator
- Step count and last loss after completion

### History — Past Interactions

Master-detail view of all saved interactions for the active session.

- Left column: list of interactions sorted newest-first, labeled `[reward]  timestamp`
- Right column: full detail — prompt, response, per-trait metric tiles, scalar reward

### Analytics — Reward Visualisation

Three tabs:

| Tab | What it shows |
|---|---|
| Reward Trend | Line chart of scalar reward over time |
| Trait Breakdown | Bar chart of average score per trait |
| Statistics | Total sessions, average reward, max reward, min reward |

### Session — Save, Load & Export

Manage training progress and export the trained model.

**Stats row:**
- Graded Responses — total saved interactions this session
- Avg Reward — mean scalar reward across the log
- Online Training Steps — how many GRPO updates have fired

**Save & Load:**

| Button | Action |
|---|---|
| **Save Session** | Writes current LoRA weights to `sessions/{name}/online_checkpoint/` |
| **Load Saved Weights** | Restores checkpoint weights into the live model |

Saved weights reload automatically on the next app start (or session switch).

**Online Training Log** — shows all step messages from `OnlineGRPOSession.log`.

**Export** (visible once a checkpoint exists):

| Tab | What it does |
|---|---|
| Download Adapters (ZIP) | ZIPs `online_checkpoint/` in memory → download |
| Merge & Export Full Model | Merges LoRA weights into base model, saves standalone model |

> Always **Save Session** before exporting — export reads from disk, not from the live model.

---

## 8. Data Pipeline

The dataset is **20 Newsgroups** via `sklearn.datasets.fetch_20newsgroups`.

**In the UI (online):** Posts are fetched live in the Prompt page — no preprocessing step needed. The `Load` button fetches one category at a time and caches posts in Streamlit session state.

**Preprocessing pipeline (offline, optional):** A standalone data pipeline in `backend/data/` can export cleaned posts to `.txt` files for use by the RAG system or for inspection.

```bash
cd backend/data
uv run python scripts/preprocess.py   # exports to data/raw/*.txt
uv run python scripts/verify.py       # sanity-checks output
```

**Pipeline stages:**

| Stage | Module | What it does |
|---|---|---|
| Load | `app/loader.py` | `fetch_20newsgroups()` with headers/footers/quotes stripped |
| Clean | `app/cleaner.py` | Remove divider lines, collapse whitespace |
| Filter | `app/loader.py` | Drop posts shorter than 100 characters |
| Export | `app/exporter.py` | Write to `.txt` (by category or by document) |

---

## 9. Reward System

The reward system converts human slider scores into a single scalar reward.

**Formula:**

```
scalar_reward = Σ (trait.weight × user_score[trait.name])
```

There is no automatic scorer — the human is the judge. The reward system only validates that every defined trait has been scored and computes the weighted sum.

**Trait configuration** (`config/traits.json`):

```json
{
  "traits": [
    {"name": "clarity",    "description": "Easy to read and well-structured",    "weight": 0.4},
    {"name": "warmth",     "description": "Sounds kind and approachable",         "weight": 0.3},
    {"name": "directness", "description": "Gets to the point without padding",    "weight": 0.3}
  ]
}
```

Traits can be redefined in the Setup page at any time. Weights do not need to sum to 1.0.

**Score range:** −5 to +5 per trait. Negative scores are valid — they push the model away from bad behavior.

**Code path:**

```
Grade page sliders
    → BackendAdapter.compute_reward(scores, traits)
        → app/reward_bridge.py  compute_reward()
            → Σ weight × score
    → ScoredTrait list + scalar_reward
    → ui/storage.py  save_interaction()
    → sessions/{name}/interaction_log.jsonl
```

---

## 10. Model Generation

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Loading** (`backend_adapter.py`):

```python
@st.cache_resource(show_spinner="Loading Qwen2.5 + LoRA…")
def _load_model():
    config = local_config()              # 4-bit QLoRA on GPU, float32 on CPU
    model, tokenizer = load_model(config)  # Qwen2.5-1.5B + LoRA adapters attached
    model.eval()
    return model, tokenizer
```

The model is loaded once — with LoRA adapters attached — and shared for both inference and online GRPO training. LoRA B matrices are zero-initialized, so the model is functionally identical to the base at startup. After each online training step, the updated LoRA weights are live for the next `generate_responses()` call with no reload.

**Inference:**

```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    num_return_sequences=n,   # 2 or 4
)
```

The prompt passed to the model includes the response length hint (e.g., "Aim for roughly 150 words in your response."). The original prompt without this hint is saved to the interaction log.

**Loading a saved checkpoint for inference outside the UI:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "sessions/my_session/online_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("sessions/my_session/online_checkpoint")
```

Or, if you exported a merged model:

```python
model = AutoModelForCausalLM.from_pretrained("sessions/my_session/merged_model")
tokenizer = AutoTokenizer.from_pretrained("sessions/my_session/merged_model")
```

---

## 11. GRPO Training

### Overview

GRPO (Group Relative Policy Optimization) fine-tunes the model using human rewards as the training signal. Unlike PPO, it does not need a separate critic model — it uses the relative reward differences within a group of responses per prompt as advantages.

This system uses **online GRPO only**: the model updates in real time on the Grade page as you grade responses.

| Property | Value |
|---|---|
| Trigger | Automatic after every 3 complete grading rounds |
| Rewards | Per-completion (true GRPO — each response gets its own human score) |
| Model update | In place on the shared model (no separate model instance) |
| Optimizer | AdamW, persistent across all 3-round windows |
| KL reference | `disable_adapter_layers()` — same model object, no second copy |

### Online Training Step

After every 3 complete grading rounds, `OnlineGRPOSession.step()` fires in a background thread:

```
3 rounds buffered: [{prompt, completions: [{text, reward}, ...]}]
    ↓
grpo_system/online_step.grpo_step(model, tokenizer, groups, optimizer)
    for each group:
        advantages = (rewards − mean) / std      # group-relative
        for each completion:
            lp  = log_prob(model, prompt, text)  # LoRA active, grad flows
            ref = log_prob(model, prompt, text)  # disable_adapter_layers, no grad
            loss += −advantage × lp + β × KL(lp ∥ ref)
    total.backward()
    optimizer.step()                             # updates LoRA weights in place
    ↓
LoRA weights updated on shared model
Next generate_responses() call uses updated weights
```

The KL reference uses `disable_adapter_layers()` on the same model object — no second model copy needed. The AdamW optimizer persists across all 3-round windows so momentum accumulates.

### Configuration (`rl_system/grpo_system/config.py`)

```python
# Local config (GTX 1650 / 4 GB VRAM)
GRPOConfig(
    model_name          = "Qwen/Qwen2.5-1.5B-Instruct",
    use_4bit            = True,
    lora_r              = 16,
    lora_alpha          = 32,
    lora_dropout        = 0.05,
    lora_target_modules = ("q_proj", "v_proj", "k_proj", "o_proj"),
    learning_rate       = 1e-5,
    num_generations     = 2,         # responses generated per prompt per step
    max_completion_length = 128,
    batch_size          = 1,
    num_train_epochs    = 1,
    kl_coef             = 0.1,       # KL penalty — keeps model close to base
)
```

---

## 12. Exporting a Trained Model

The **Session page → Export** section appears once a checkpoint has been saved.

### Workflow

1. Grade responses on the Grade page — online GRPO updates weights in memory
2. Go to **Session page → Save Session** — writes weights to `sessions/{name}/online_checkpoint/`
3. Choose an export format in the Export section

### Option A — Download LoRA Adapters (ZIP)

Downloads `{session}_adapters.zip` containing all files from `sessions/{name}/online_checkpoint/`. These are the fine-tuned adapter weights only (~80 MB). Load them with:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "path/to/extracted/online_checkpoint")
```

### Option B — Merge & Export Full Model

Merges the LoRA adapter weights permanently into the base model's weights and saves a standalone model. The result requires no PEFT to load.

- Output directory is configurable (default: `sessions/{name}/merged_model/`)
- Runs `peft_model.merge_and_unload()` then `merged.save_pretrained(output_dir)`
- The merged model is larger (~3 GB) but self-contained

Load the merged model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("sessions/my_session/merged_model")
tokenizer = AutoTokenizer.from_pretrained("sessions/my_session/merged_model")
```

---

## 13. Supporting Backends (Not Yet Integrated)

These subsystems are fully implemented and tested but not yet connected to the main Streamlit UI workflow.

### RAG System (`backend/rag_system/`)

Retrieval-Augmented Generation over a corpus of `.txt` documents using `sentence-transformers` (all-MiniLM-L6-v2) and FAISS.

**Modules:** `ingest.py`, `embed.py`, `index.py`, `retrieve.py`, `generate.py`, `pipeline.py` (orchestrator)

**CLI:**
```bash
cd backend/rag_system
uv run python scripts/build_index.py   # embed documents → data/index/
uv run python scripts/chat_cli.py      # interactive query CLI
```

**Future integration:** Wire `RAGPipeline.answer(query)` into the Grade page to inject retrieved chunks as context alongside the article.

### API (`backend/api/`)

FastAPI REST layer wrapping generation, reward computation, and interaction storage.

**Launch:**
```bash
cd backend/api
uv run uvicorn app:app --reload --port 8000
```

**Endpoints:** `/health`, `/generate`, `/reward/compute`, `/interactions`

**Future integration:** The `BackendAdapter` in `user_interface/backend_adapter.py` is already designed to swap its direct function calls for HTTP calls to this API — the interface is identical.

### Standalone Batch Trainer (`backend/rl_system/grpo_system/train.py`)

A standalone GRPO training script that reads the interaction log and runs a full TRL `GRPOTrainer` pass. Not wired to the UI — intended for offline runs on larger hardware (e.g., Colab T4 with the 3B model).

---

## 14. Hardware Requirements

### For inference (Grade page — generating responses)

| Hardware | Load time | Generation speed (256 tokens, 2 responses) |
|---|---|---|
| GPU ≥ 4 GB VRAM | ~30 sec (first load) | 10–30 sec |
| CPU only | ~3–5 min (first load) | 2–5 min per click |

### For online GRPO training (fires automatically after 3 rounds)

| Hardware | Notes |
|---|---|
| GPU ≥ 4 GB VRAM | Runs with 4-bit QLoRA (`use_4bit=True` in `local_config()`) |
| GPU ≥ 8 GB VRAM | Runs comfortably in bfloat16 without quantization |
| CPU only | Possible but very slow — a single step may take many minutes |

**VRAM breakdown for 4GB config:**
- 4-bit model weights: ~1.5 GB
- LoRA adapters + AdamW optimizer state: ~240 MB
- Activations during training step: ~500 MB
- Total peak: ~2.2 GB — fits comfortably on a GTX 1650

The 1.5B model (`Qwen2.5-1.5B-Instruct`) was chosen specifically to fit on a GTX 1650 (4 GB) with 4-bit quantization. Switch to `colab_config()` in `backend_adapter.py` to use the 3B model on hardware with ≥ 12 GB VRAM.
