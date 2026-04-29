# Interactive System for LLMs and RLHF

An end-to-end Reinforcement Learning from Human Feedback (RLHF) research prototype. A human annotator steers the writing style of a language model by rating its outputs through a Streamlit UI. Ratings are converted into scalar rewards and used to fine-tune the model via GRPO (Group Relative Policy Optimization) with LoRA adapters. All work is organized into isolated experiment sessions so that different training runs never interfere with each other.

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
Grade page — Qwen2.5-1.5B-Instruct generates 2 or 4 responses
        ↓
Human scores each response with sliders across defined traits
        ↓
Scalar reward = Σ(weight_i × score_i) saved to session's interaction log
        ↓
Train page — GRPO fine-tunes the model on saved interactions (background thread)
        ↓
LoRA adapters saved to the session's grpo_output/ directory
        ↓
Export — download adapter ZIP or merge into a standalone model
```

The model rewrites real articles from a curated corpus. The human never writes generation prompts — they define evaluation traits once (in Setup) and then only rate outputs. Multiple independent sessions let you train toward different styles simultaneously without any interference.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI  (app.py)                      │
│                                                                 │
│  Sidebar:  Session Switcher  │  Navigation (6 pages)           │
│                                                                 │
│  Setup      → trait_manager.py   (load/edit reward traits)     │
│  Prompt     → sklearn 20NG       (browse articles)             │
│  Grade      → backend_adapter.py (generate + score + save)     │
│  History    → ui/storage.py      (read JSONL log)              │
│  Analytics  → pandas / st.chart  (reward trends)               │
│  Train      → grpo_adapter.py    (GRPO in background thread)   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │                            │
  backend_adapter.py           grpo_adapter.py
  ├─ _load_model()             ├─ load_scored_prompts()
  │  @st.cache_resource         ├─ GRPOSession (thread)
  │  Qwen2.5-1.5B-Instruct      ├─ save_session_record()
  │  bfloat16 / float32         ├─ zip_adapter_dir()
  └─ generate_responses()       └─ merge_and_export()
     (chat template,
      num_return_sequences=n)
          │
  app/reward_bridge.py          sessions/{name}/
  └─ compute_reward()           ├─ interaction_log.jsonl
     weighted sum                ├─ training_sessions.jsonl
                                 └─ grpo_output/
                                    (LoRA adapters)
```

**Module responsibilities:**

| Module | Role |
|---|---|
| `app.py` | Entry point; sidebar session switcher + page routing |
| `backend_adapter.py` | Wraps model inference (`_load_model` cached) and reward computation |
| `grpo_adapter.py` | Loads interaction log, runs GRPOTrainer in a background thread, saves adapters |
| `ui/session_manager.py` | Path resolution per named session; create / list sessions |
| `ui/state.py` | All `st.session_state` keys and accessors |
| `ui/components.py` | All page-level rendering functions |
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
    │       ├── data.py                  # build newsgroup prompts for training
    │       ├── reward.py                # build_reward_fn() factory
    │       └── train.py                 # main GRPOTrainer loop
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
        ├── app.py                       # entry point + sidebar
        ├── backend_adapter.py           # model inference + reward bridge
        ├── grpo_adapter.py              # GRPO session management + export
        ├── .streamlit/
        │   └── config.toml              # fileWatcherType = none (suppresses torchvision warnings)
        ├── config/
        │   └── traits.json              # default trait definitions
        ├── sessions/                    # auto-created; one dir per experiment session
        │   └── {session_name}/
        │       ├── interaction_log.jsonl
        │       ├── training_sessions.jsonl
        │       └── grpo_output/
        │           └── (LoRA adapter files)
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
- **Training history** — records of all GRPO runs (status, epochs, log lines, adapter path)
- **LoRA adapters** — the fine-tuned model weights from training runs in this session

Sessions live under `backend/user_interface/sessions/{name}/`:

```
sessions/
├── default/                        ← created automatically on first launch
│   ├── interaction_log.jsonl       ← grading data (prompt, response, traits, reward)
│   ├── training_sessions.jsonl     ← GRPO run records (persists across restarts)
│   └── grpo_output/                ← LoRA adapter files after training
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
├── warm_tone/
│   └── ...
└── concise_v2/
    └── ...
```

### Creating and switching sessions

The **sidebar session panel** (always visible, above the page navigation) shows the active session as a dropdown. Switching immediately isolates all state — history, training session, and saved flag are reset so nothing leaks from the previous session.

To create a new session, use the **"New session"** expander in the sidebar:
- Enter a name (letters, digits, `_`, `-` only; max 64 characters)
- Click **Create & Switch** — the directory is created and becomes active immediately

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

Everything below — interaction log, training runs, adapter output — belongs to whichever session is selected in the dropdown.

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
- Click **Load** — posts are fetched via `sklearn.datasets.fetch_20newsgroups` and cached in session state for the duration of the browser session

**Choose a Post** (appears after loading):
- **← Prev / Next →** buttons or a numeric jump input to navigate posts
- 130-line preview of the current post
- **Use as Article →** pushes the full post text into the Article field below

**Article and Instruction fields** (form):
- **Article** — the text the model will work with (auto-filled from dataset or pasted manually)
- **Instruction** — what the model should do with the article (e.g., "Rewrite this to be more engaging and warm"). The last-used instruction is saved to `config/preset_instruction.txt` and pre-filled on next visit.
- **Set Prompt** saves both fields to session state; **Clear** wipes the article only (instruction is kept)

### Grade — Response & Scoring

Generate model responses and score them.

**Active Prompt** (collapsible expander at top) — shows the current article and instruction from the Prompt page.

**Generate Responses**:
- Choose **2 or 4 responses** with the radio selector
- Click **Generate Responses** — calls `Qwen2.5-1.5B-Instruct` via the cached `_load_model()` in `backend_adapter.py`. The model is loaded once and reused across all reruns.
- On first click: the model downloads from HuggingFace (~3 GB) and loads into memory — this takes 1–3 minutes. Subsequent calls use the cached model.
- Responses can also be typed or pasted manually.

**Scoring** (per response):
- One slider per trait (0–5 scale)
- Live reward breakdown: per-trait `score × weight = contribution`, and the final scalar reward
- **Save Response** button — writes to the active session's `interaction_log.jsonl` and adds to History

**Clear All** — resets all responses and the saved flags (does not clear the prompt).

> **Performance note:** On CPU-only hardware, generation takes roughly 2–5 minutes per click for 2 responses at 256 tokens. A GPU with ≥ 4 GB VRAM reduces this to seconds.

### History — Past Interactions

Master-detail view of all saved interactions for the active session, loaded from `interaction_log.jsonl` at startup.

- Left column: list of sessions sorted newest-first, labeled `[reward]  timestamp`
- Right column: full detail for the selected entry — article excerpt, response text, per-trait metric tiles, and scalar reward

### Analytics — Reward Visualisation

Three tabs:

| Tab | What it shows |
|---|---|
| Reward Trend | Line chart of scalar reward over time (x = interaction index) |
| Trait Breakdown | Bar chart of average score per trait across all saved interactions |
| Statistics | Total sessions, average reward, max reward, min reward |

### Train — GRPO Fine-Tuning

Fine-tune the model on the active session's saved interactions using GRPO.

**Data summary** (top metrics):
- Scored Prompts — unique prompts in the interaction log (averaged if rated multiple times)
- Avg Reward — mean scalar reward across all logged interactions
- Last Run — status of the most recent training session

**Training Config**:
- Epochs (1–10); adapter output path is always the active session's `grpo_output/`
- Model: `Qwen/Qwen2.5-1.5B-Instruct` with LoRA (rank 16, target modules: q/k/v/o_proj)

**Start Training** — launches a background thread. The main UI stays responsive. Click **Refresh Log** to poll for new log lines.

**Training Log** — shows live progress:
```
Reading interaction log…
Found 5 unique prompt(s) with human rewards.
Loading Qwen/Qwen2.5-1.5B-Instruct with LoRA…
Building dataset from logged prompts…
Dataset: 5 prompt(s)
Starting GRPO training…
step 1  loss=2.34  reward=3.2
step 2  loss=2.18  reward=3.5
…
Adapters saved to sessions/default/grpo_output/
```

Every completed run (success or error) is saved to `training_sessions.jsonl` and appears in **Past Sessions** at the bottom of the page — full log and status survive app restarts.

**Export** (appears once `grpo_output/` exists):

| Tab | What it does |
|---|---|
| Download Adapters (ZIP) | Zips the adapter directory in memory and serves a `grpo_adapters.zip` download |
| Merge & Export Full Model | Merges LoRA weights into base model, saves a standalone model loadable without PEFT |

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

**Categories (20 total):**
`alt.atheism`, `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`, `misc.forsale`, `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`, `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space`, `soc.religion.christian`, `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`

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

Traits can be redefined in the Setup page at any time. Weights do not need to sum to 1.0, but normalized weights make rewards comparable across sessions.

**Score range:** 0–5 per trait. With three traits at weights 0.4 / 0.3 / 0.3, a perfect score yields `0.4×5 + 0.3×5 + 0.3×5 = 5.0`.

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
@st.cache_resource(show_spinner="Loading Qwen2.5 model…")
def _load_model():
    # GPU available → bfloat16 + device_map="auto"
    # CPU only      → float32
```

The model is loaded once on first call and kept in memory for the Streamlit process lifetime. Rerunning the app (not restarting) reuses the cached model.

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

The prompt is formatted using Qwen's chat template. `num_return_sequences` generates all responses in a single forward pass (more efficient than calling generate n times).

**Loading a trained session's adapters** (for inference after training):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "sessions/my_session/grpo_output")
tokenizer = AutoTokenizer.from_pretrained("sessions/my_session/grpo_output")
```

Or, if you exported a merged model, load it directly without PEFT:

```python
model = AutoModelForCausalLM.from_pretrained("sessions/my_session/grpo_merged_model")
tokenizer = AutoTokenizer.from_pretrained("sessions/my_session/grpo_merged_model")
```

---

## 11. GRPO Training

### Overview

GRPO (Group Relative Policy Optimization) fine-tunes the model using human rewards as the training signal. Unlike PPO, it does not need a separate critic model — it generates a group of responses per prompt and uses their relative reward differences as advantages.

**Offline RLHF approach used here:**

The interaction log accumulates `(prompt, scalar_reward)` pairs from human grading. At training time, GRPO uses these saved rewards as the reward signal: for each generated completion, the reward function looks up the human-assigned reward for that prompt. This means training reinforces the model to generate responses on prompts that humans have already evaluated positively.

### How it works

```
interaction_log.jsonl
    ↓
grpo_adapter.load_scored_prompts()
    → {prompt: mean_scalar_reward}   (averaged if rated multiple times)
    ↓
GRPOSession._train()   (background thread)
    → load_model() — Qwen2.5-1.5B-Instruct + LoRA
    → build Dataset from logged prompts (apply chat template)
    → reward_fn(prompts, completions) → looks up saved reward by prompt substring match
    → TRL GRPOTrainer.train()
    → model.save_pretrained(sessions/{name}/grpo_output/)
    → save_session_record() → training_sessions.jsonl
```

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
    output_dir          = "sessions/{name}/grpo_output",
)
```

### Session record format

Every completed run (success or error) is appended to `training_sessions.jsonl`:

```json
{
  "timestamp": "2026-04-28T21:00:00",
  "status": "done",
  "num_epochs": 1,
  "n_prompts": 5,
  "output_dir": "sessions/default/grpo_output",
  "log": ["Reading interaction log…", "step 1  loss=2.34  reward=3.2", "…"],
  "error": null
}
```

---

## 12. Exporting a Trained Model

After a training run completes, the **Export** section appears on the Train page.

### Option A — Download LoRA Adapters (ZIP)

Downloads `grpo_adapters.zip` containing all files from `sessions/{name}/grpo_output/`. These are the fine-tuned adapter weights only (~80 MB). Load them with:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "path/to/extracted/adapters")
```

### Option B — Merge & Export Full Model

Merges the LoRA adapter weights permanently into the base model's weights and saves a standalone model. The result requires no PEFT to load.

- Output directory is configurable (default: `grpo_merged_model/`)
- Runs `peft_model.merge_and_unload()` then `merged.save_pretrained(output_dir)`
- The merged model is larger (~3 GB in bfloat16) but self-contained

Load the merged model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("grpo_merged_model")
tokenizer = AutoTokenizer.from_pretrained("grpo_merged_model")
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

---

## 14. Hardware Requirements

### For inference (Grade page — generating responses)

| Hardware | Load time | Generation speed (256 tokens, 2 responses) |
|---|---|---|
| GPU ≥ 4 GB VRAM | ~30 sec (first load) | 10–30 sec |
| CPU only | ~3–5 min (first load) | 2–5 min per click |

### For GRPO training (Train page)

| Hardware | Notes |
|---|---|
| GPU ≥ 4 GB VRAM | Runs with 4-bit QLoRA (`use_4bit=True` in `local_config()`) |
| GPU ≥ 8 GB VRAM | Runs comfortably in bfloat16 without quantization |
| CPU only | Possible but very slow — a single training epoch on 5 prompts may take hours |

The 1.5B model (`Qwen2.5-1.5B-Instruct`) was chosen specifically to fit on a GTX 1650 (4 GB) with 4-bit quantization. Switch to `colab_config()` in `grpo_adapter.py` to use the 3B model on hardware with ≥ 12 GB VRAM:

```python
# grpo_adapter.py, inside GRPOSession._train()
from grpo_system.config import colab_config
config = colab_config()   # Qwen2.5-3B-Instruct, bfloat16, num_generations=4
```
