# Interactive System for LLMs and RLHF

An end-to-end Reinforcement Learning from Human Feedback (RLHF) research prototype. The system lets a human annotator steer the writing style of a large language model by rating its outputs through a UI. Ratings are converted into scalar rewards and used to fine-tune the model via GRPO (Group Relative Policy Optimization) with LoRA adapters.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites and Installation](#3-prerequisites-and-installation)
4. [Project Structure](#4-project-structure)
5. [Sessions](#5-sessions)
   - [What a Session Is](#51-what-a-session-is)
   - [Session File Layout](#52-session-file-layout)
   - [Creating a New Session](#53-creating-a-new-session)
   - [Resuming a Session](#54-resuming-a-session)
   - [Running Multiple Sessions](#55-running-multiple-sessions)
6. [Component Setup](#6-component-setup)
   - [RAG System](#61-rag-system)
   - [Reward System and Traits](#62-reward-system-and-traits)
   - [User Interface](#63-user-interface-streamlit)
   - [API](#64-api-fastapi)
   - [RL System](#65-rl-system-planned)
7. [End-to-End Walkthrough](#7-end-to-end-walkthrough)
8. [Training Loop Explained](#8-training-loop-explained)
9. [Hardware Requirements](#9-hardware-requirements)

---

## 1. System Overview

The system has one goal: teach a language model to write in a style that a specific human prefers, using that human's real-time ratings as the training signal.

**What happens at a high level:**

1. The user creates a session, defining the dataset, traits, prompt template, and starting model checkpoint.
2. A document is pulled from the configured dataset.
3. The model is prompted using the session's prompt template to rewrite it.
4. The model generates multiple rewrites of the same document.
5. The user rates each rewrite using sliders across the session's traits.
6. Ratings are aggregated into a scalar reward.
7. The RL trainer updates the model toward responses the user consistently rated higher.
8. Progress is saved back to the session — the annotator can stop and resume at any time.

The model never invents topics — it rewrites real documents. The user never writes generation prompts — they define evaluation traits at session creation and then only rate outputs. Over many iterations, the model's style drifts toward whatever the annotator consistently rewards.

---

## 2. Architecture

```
Session (dataset + traits + prompt template + checkpoint)
        │
        ├──► RAG System ──────────────────────────────────────────────────────┐
        │    (FAISS index over document chunks)                               │
        │    Retrieves relevant context for each prompt                       │
        │                                                                     ▼
        └──► RL System (GRPO training loop)              Streamlit UI (main interface)
             │                                                  │
             │  Prompt template + document ──────────────────► │ User sees rewrites
             │                                                  │ User moves sliders
             │  Qwen2.5-3B-Instruct generates G=4 rewrites      │ per session trait
             │                                                  │
             └──────────────────────────────────────────────── ▼
                                                    Reward System
                                                    (session traits → weighted sum → scalar)
                                                          │
                                                          ▼
                                                  GRPOTrainer update
                                                  (LoRA adapter weights)
                                                          │
                                                          ▼
                                                  Session checkpoint saved
                                                  (sessions/{id}/checkpoints/step_N/)
                                                  Session log appended
                                                  (sessions/{id}/logs/interaction_log.jsonl)
```

**Subsystems:**

| Subsystem | Tech | Status | Role |
|---|---|---|---|
| Session Manager | Python + JSON | Planned | Stores and loads session state (traits, dataset, prompt, checkpoint) |
| RAG System | `sentence-transformers`, `faiss-cpu`, `datasets` | Implemented | Embeds documents, retrieves relevant chunks as context |
| Reward System | Pure Python | Implemented | Loads session traits, converts slider scores to scalar reward |
| User Interface | Streamlit | Implemented | Session picker, prompt display, trait sliders, history |
| API | FastAPI + Uvicorn | Implemented | REST layer wrapping generation, reward, and session storage |
| RL System | `trl` (GRPOTrainer), `peft` (LoRA), `transformers` | Planned | Fine-tunes Qwen2.5-3B-Instruct on human reward signal |

---

## 3. Prerequisites and Installation

**Requirements:**
- Python >= 3.11
- `uv` package manager (recommended) or `pip`
- GPU with 12–24 GB VRAM for GRPO training (CPU-only is possible but very slow)

**Install dependencies:**

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

This installs all dependencies declared in `pyproject.toml`, including:
`torch`, `transformers`, `trl`, `peft`, `sentence-transformers`, `faiss-cpu`, `fastapi`, `uvicorn`, `streamlit`, `datasets`, `accelerate`.

**Verify key packages:**

```bash
uv run python -c "from trl import GRPOTrainer; print('GRPOTrainer ok')"
uv run python -c "from peft import LoraConfig; print('LoRA ok')"
uv run python -c "import streamlit; print('Streamlit ok')"
```

---

## 4. Project Structure

```
.
├── pyproject.toml
├── artifact.md
├── sessions/                           # One directory per session (auto-created)
│   └── {session_id}/
│       ├── session.json                # Session metadata and config
│       ├── traits.json                 # Traits defined at session creation
│       ├── logs/
│       │   └── interaction_log.jsonl   # All scored interactions for this session
│       └── checkpoints/               # [planned] LoRA adapter snapshots
│           └── step_{N}/
│               ├── adapter_config.json
│               └── adapter_model.safetensors
└── backend/
    ├── session_manager/                # [planned] Session creation, loading, listing
    │   ├── session.py                  # Session dataclass and file I/O
    │   └── manager.py                  # create_session(), load_session(), list_sessions()
    │
    ├── api/                            # FastAPI REST layer
    │   ├── app.py
    │   ├── api/
    │   │   ├── routes.py               # Endpoints: /sessions, /generate, /reward, /interactions
    │   │   └── schemas.py
    │   └── services/
    │       ├── generation_service.py
    │       ├── reward_service.py
    │       └── storage_service.py
    │
    ├── rag_system/                     # Retrieval-Augmented Generation
    │   ├── app/
    │   │   ├── pipeline.py             # RAGPipeline orchestrator
    │   │   ├── ingest.py
    │   │   ├── embed.py
    │   │   ├── index.py
    │   │   ├── retrieve.py
    │   │   └── generate.py
    │   ├── scripts/
    │   │   ├── build_index.py
    │   │   └── chat_cli.py
    │   └── data/
    │       └── raw/                    # Source documents (.txt files)
    │
    ├── reward_system/                  # Reward computation
    │   ├── app/
    │   │   ├── trait_loader.py         # Reads traits from session directory
    │   │   ├── reward_fn.py            # Weighted sum → scalar
    │   │   ├── schemas.py
    │   │   └── logger.py
    │
    ├── user_interface/                 # Streamlit UI
    │   ├── main.py                     # Entry point — shows session picker on startup
    │   └── app/
    │       ├── session_ui.py           # [planned] Session creation and resume forms
    │       ├── rag_stub.py
    │       ├── trait_manager.py
    │       ├── reward_bridge.py
    │       └── interaction_logger.py
    │
    └── rl_system/                      # [planned] RL fine-tuning
        ├── GRPO_QWEN_GUIDE.md          # Implementation guide for the GRPO system
        └── grpo_system/                # To be implemented — see GRPO_QWEN_GUIDE.md
            ├── config.py               # GRPOConfig dataclass
            ├── model.py                # Qwen2.5 + LoRA loading
            ├── data.py                 # Newsgroup article prompt builder
            ├── reward.py               # Bridges reward_system into GRPOTrainer
            └── train.py               # Main training loop
```

---

## 5. Sessions

### 5.1 What a Session Is

A **session** is the unit of a single RLHF experiment. It bundles together everything that defines one annotator's run:

| Field | What it stores | Example |
|---|---|---|
| `session_id` | Unique identifier | `"april_warmth_study"` |
| `dataset` | Which corpus to draw documents from | `"20newsgroups"` |
| `prompt_template` | The fixed instruction wrapped around each document | `"Rewrite the following article to be more engaging:\n\n{article}"` |
| `traits` | The evaluation dimensions the annotator will score | `[{name, description, weight}, ...]` |
| `model_name` | Base model to fine-tune | `"Qwen/Qwen2.5-3B-Instruct"` |
| `lora_checkpoint` | Path to LoRA adapters to resume from (`null` = start from base model) | `"checkpoints/step_42"` |
| `articles_seen` | How many documents have been annotated so far | `42` |
| `status` | `"active"` or `"paused"` | `"active"` |

Sessions are isolated: each one has its own traits, its own interaction log, and its own LoRA checkpoint directory. You can run different sessions with completely different traits, prompt templates, and model checkpoints without any interference.

---

### 5.2 Session File Layout

Each session lives in `sessions/{session_id}/`:

```
sessions/
└── april_warmth_study/
    ├── session.json                    # Config and metadata
    ├── traits.json                     # Traits defined at creation — locked for this session
    ├── logs/
    │   └── interaction_log.jsonl       # One line per scored interaction
    └── checkpoints/
        ├── step_10/
        ├── step_20/
        └── step_42/                    # Most recent checkpoint
```

**`session.json` format:**

```json
{
  "session_id": "april_warmth_study",
  "name": "April — Warmth and Clarity Study",
  "created": "2026-04-13T10:00:00",
  "last_updated": "2026-04-13T14:30:00",
  "dataset": "20newsgroups",
  "prompt_template": "Rewrite the following article to be more engaging. Your rewrite should encourage continued discussion, sound warm and approachable, and get to the point clearly.\n\nArticle:\n{article}",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "lora_checkpoint": "checkpoints/step_42",
  "articles_seen": 42,
  "status": "active"
}
```

**`traits.json` format (user-defined at session creation):**

```json
{
  "traits": [
    {
      "name": "warmth",
      "description": "Sounds kind, supportive, and human",
      "weight": 0.4
    },
    {
      "name": "clarity",
      "description": "Easy to read, well-structured, free of jargon",
      "weight": 0.4
    },
    {
      "name": "directness",
      "description": "Gets to the point without unnecessary padding",
      "weight": 0.2
    }
  ]
}
```

Traits are locked after session creation — they cannot be changed mid-session, because changing trait definitions mid-run would make the reward signal inconsistent and corrupt the training data. If you want different traits, create a new session.

---

### 5.3 Creating a New Session

Sessions are created through the UI's initialization screen, which appears automatically when no session is loaded.

**From the UI — initialization form fields:**

| Field | Input type | Notes |
|---|---|---|
| Session name | Text | Human-readable label (used as the session ID) |
| Dataset | Dropdown | `20newsgroups` (default), or path to custom `.txt` folder |
| Prompt template | Text area | Must contain `{article}` as the placeholder for document text |
| Starting checkpoint | Text / file picker | Leave blank to start from the base model |
| Traits | Dynamic form | Add as many traits as needed; set name, description, and weight for each |

**From the API:**

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "april_warmth_study",
    "name": "April — Warmth and Clarity Study",
    "dataset": "20newsgroups",
    "prompt_template": "Rewrite the following article to be more engaging:\n\nArticle:\n{article}",
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "lora_checkpoint": null,
    "traits": [
      {"name": "warmth",     "description": "Sounds kind and human",         "weight": 0.4},
      {"name": "clarity",    "description": "Well-structured, easy to read", "weight": 0.4},
      {"name": "directness", "description": "Gets to the point quickly",     "weight": 0.2}
    ]
  }'
```

This creates `sessions/april_warmth_study/` with `session.json` and `traits.json` pre-populated.

**Guidelines for defining traits:**

- Keep `name` to one word — it is used as the slider label in the UI
- Write `description` from the annotator's perspective ("sounds kind and human", not "warmth metric")
- Weights do not need to sum to 1.0, but normalizing them makes scalar rewards easier to compare across sessions
- Start with 2–4 traits; more than 5 makes annotation slow and inconsistent
- Traits should be independent — avoid defining traits that always move together (e.g., "engaging" and "interesting" will be rated nearly identically)

---

### 5.4 Resuming a Session

When you relaunch the UI or API, you can resume any previously created session. All state — the interaction log, the LoRA checkpoint, and the article index position — is restored from the session directory.

**From the UI:** The startup screen lists all sessions in `sessions/` with their status, last-updated time, and articles-seen count. Click a session to resume it.

**From the API:**

```bash
# List all sessions
curl http://localhost:8000/sessions

# Load a specific session
curl http://localhost:8000/sessions/april_warmth_study
```

**What gets restored on resume:**

| State | Where it comes from |
|---|---|
| Traits | `sessions/{id}/traits.json` |
| Prompt template | `session.json → prompt_template` |
| Model weights | `session.json → lora_checkpoint` (LoRA adapters loaded on top of base model) |
| Interaction history | `sessions/{id}/logs/interaction_log.jsonl` |
| Article cursor | `session.json → articles_seen` (annotation continues from the next unseen article) |

---

### 5.5 Running Multiple Sessions

Sessions are fully independent. You can run different sessions simultaneously with different traits, datasets, and model checkpoints.

**Example: two parallel sessions**

```
sessions/
├── warmth_study/       # Annotator A: rewarding warmth and clarity
│   └── traits.json     # warmth (0.4), clarity (0.4), directness (0.2)
│
└── conciseness_study/  # Annotator B: rewarding brevity and precision
    └── traits.json     # brevity (0.5), precision (0.3), tone (0.2)
```

Each session runs its own UI instance on a different port:

```bash
# Session 1
cd backend/user_interface
SESSION_ID=warmth_study uv run streamlit run main.py --server.port 8501

# Session 2
SESSION_ID=conciseness_study uv run streamlit run main.py --server.port 8502
```

Each session writes to its own log and checkpoint directory. There is no shared state between sessions.

---

## 6. Component Setup

### 6.1 RAG System

The RAG system embeds a corpus of text documents into a FAISS vector index and retrieves the most relevant chunks for a given query. The default dataset is 20 Newsgroups articles.

**Step 1 — Populate the raw data directory**

Place your `.txt` files (one document per file) in:
```
backend/rag_system/data/raw/
```

**Step 2 — Build the FAISS index**

Run once to embed all documents and write the index to disk:

```bash
cd backend/rag_system
uv run python scripts/build_index.py
```

This creates `data/index/` containing the HuggingFace dataset and FAISS index files. This index is shared across all sessions that use the same dataset.

**Step 3 — Test retrieval via CLI (optional)**

```bash
uv run python scripts/chat_cli.py
```

**How retrieval works:**
1. The query is embedded by `sentence-transformers` (default: `all-MiniLM-L6-v2`).
2. FAISS finds the top-k nearest chunks by cosine similarity.
3. Those chunks are injected into the LLM prompt as context alongside the document to rewrite.

---

### 6.2 Reward System and Traits

The reward system converts the annotator's slider ratings into a single scalar reward for the RL trainer.

**Formula:**
```
scalar_reward = sum(trait.weight * user_score[trait.name] for each trait in session)
```

Traits come from the active session's `traits.json` — not from a hardcoded config file. The reward system reads whichever session is active at runtime:

```python
# trait_loader.py loads from the session directory, not a fixed path
traits = load_traits(f"sessions/{session_id}/traits.json")
```

**Note:** The reward system does not score responses itself. It only aggregates human-provided scores. The human is the judge.

---

### 6.3 User Interface (Streamlit)

The UI is the annotator's workspace. On startup it presents a session screen; once a session is loaded or created, it enters the annotation loop.

**Launch the UI:**

```bash
cd backend/user_interface
uv run streamlit run main.py
```

Opens at `http://localhost:8501` by default.

**Startup screen — session management:**

- Lists all sessions in `sessions/` with status, last updated, and articles seen
- "Resume" button to continue an existing session
- "New session" form to define a fresh session (name, dataset, prompt template, traits)

**Annotation panels (once a session is loaded):**

| Panel | Purpose |
|---|---|
| Session info | Shows active session name, articles seen, current checkpoint |
| 1. Prompt | Displays the current document and the session's prompt template |
| 2. Response | The model's generated rewrite |
| 3. Retrieved context | Collapsible; shows which RAG chunks were used |
| 4. Score this response | Sliders (0–5) for each session trait; live reward breakdown |
| 5. Save | Writes interaction to `sessions/{id}/logs/interaction_log.jsonl` |
| History | Previously saved interactions for this session |

---

### 6.4 API (FastAPI)

The REST API wraps session management, generation, reward computation, and storage.

**Launch the API:**

```bash
cd backend/api
uv run uvicorn app:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/sessions` | Create a new session |
| `GET` | `/sessions` | List all sessions |
| `GET` | `/sessions/{id}` | Load a session by ID |
| `PATCH` | `/sessions/{id}` | Update session state (checkpoint, articles_seen, status) |
| `POST` | `/generate` | Generate a response using the active session's model and prompt template |
| `POST` | `/reward/compute` | Compute scalar reward using the active session's traits |
| `POST` | `/interactions` | Save a scored interaction to the active session's log |
| `GET` | `/interactions` | Retrieve interactions for the active session |

Interactive docs available at `http://localhost:8000/docs`.

---

### 6.5 RL System (Planned)

> **Status:** Not yet implemented. The design and implementation guide is in `backend/rl_system/GRPO_QWEN_GUIDE.md`. The subsystem will live in `backend/rl_system/grpo_system/`.

The RL system fine-tunes the session's model using GRPO and LoRA, guided by the scalar rewards produced by the reward system.

**Why GRPO instead of PPO:**
TRL 1.0.0 removed `PPOTrainer`. GRPO generates a group of G responses to the same prompt and uses their *relative* reward scores as the training signal — no separate critic model needed. This is more memory-efficient and more stable.

**Why LoRA:**
Loading Qwen2.5-3B twice (trainable + frozen reference) at full precision requires ~40–50 GB VRAM. LoRA attaches small trainable adapter matrices (~20M parameters) to specific attention layers, keeping the rest of the 3B parameters frozen. This brings VRAM requirements down to ~12–14 GB.

**Planned config (`backend/rl_system/grpo_system/config.py`):**

```python
@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_r: int = 16               # LoRA rank — higher = more capacity, more VRAM
    lora_alpha: int = 32           # usually 2x lora_r
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")
    learning_rate: float = 1e-5
    num_generations: int = 4       # G — rewrites generated per document
    batch_size: int = 2            # documents per training step
    kl_coef: float = 0.1           # keeps model from drifting too far from base
    checkpoint_every: int = 10     # save LoRA adapters every N steps
    output_dir: str = "grpo_output"
```

**How the training loop will connect to sessions:**

The training loop pulls traits and prompt template from the active session config automatically, so each session's RL run is fully self-contained:

```python
session = load_session(session_id)
traits = load_traits(f"sessions/{session_id}/traits.json")
prompt_template = session["prompt_template"]
start_checkpoint = session.get("lora_checkpoint")   # None = train from base model
```

**Planned run command:**

```bash
cd backend/rl_system
uv run python grpo_system/train.py --session april_warmth_study
```

Checkpoints will be saved to `sessions/{id}/checkpoints/step_{N}/` and `session.json` will be updated with the latest checkpoint path and `articles_seen` count after each save.

**Loading a trained session's adapters:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, json

session = json.load(open("sessions/april_warmth_study/session.json"))
checkpoint_path = f"sessions/april_warmth_study/{session['lora_checkpoint']}"

base = AutoModelForCausalLM.from_pretrained(
    session["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
```

---

## 7. End-to-End Walkthrough

### Initialization (once per session)

```
User opens the UI → "New session" form

  Session name:        april_warmth_study
  Dataset:             20newsgroups
  Prompt template:     "Rewrite the following article to be more engaging.
                        Sound warm and human. Article:\n{article}"
  Starting checkpoint: (blank — start from base model)
  Traits:
    warmth      | "Sounds kind and approachable"  | weight 0.4
    clarity     | "Easy to read, well-structured" | weight 0.4
    directness  | "Gets to the point quickly"     | weight 0.2

→ sessions/april_warmth_study/ is created
→ traits.json and session.json are written
→ Annotation loop begins
```

### One annotation iteration

```
1. Next unseen article is loaded from 20 Newsgroups
   (article index = session["articles_seen"])

2. Prompt is constructed using the session's template:
   "Rewrite the following article to be more engaging.
    Sound warm and human. Article:
    [article text]"

3. Qwen2.5-3B-Instruct (+ current LoRA checkpoint) generates G=4 rewrites

4. Each rewrite appears in the Streamlit UI

5. Annotator scores each rewrite with sliders (0–5):
   Rewrite A:  warmth=5, clarity=4, directness=3  →  scalar: 4.2
   Rewrite B:  warmth=2, clarity=3, directness=4  →  scalar: 2.8
   Rewrite C:  warmth=4, clarity=5, directness=4  →  scalar: 4.4
   Rewrite D:  warmth=1, clarity=2, directness=2  →  scalar: 1.6

6. GRPOTrainer computes relative advantages:
   A: +0.58  |  B: -0.42  |  C: +0.78  |  D: -1.65

7. LoRA adapter weights are updated

8. Interaction saved to sessions/april_warmth_study/logs/interaction_log.jsonl
   session.json → articles_seen incremented to N+1

9. Every 10 steps: checkpoint saved to sessions/april_warmth_study/checkpoints/step_{N}/
   session.json → lora_checkpoint updated

10. Next article loaded — repeat
```

### Resuming

```
User relaunches the UI → sees "april_warmth_study" in the session list
  Status: active | Articles seen: 42 | Last updated: 2026-04-13 14:30

→ Clicks "Resume"
→ traits.json loaded (warmth / clarity / directness — unchanged)
→ LoRA adapters loaded from checkpoints/step_42/
→ Next article starts at index 42
→ Annotation continues exactly where it left off
```

---

## 8. Training Loop Explained

**GRPO (Group Relative Policy Optimization):**

Unlike PPO which compares each response to a value-function baseline, GRPO generates a group of G responses to the same prompt and uses their relative reward differences as advantages. No critic model is needed.

```
Same document → G rewrites → annotator scores each → advantages = reward - mean(group rewards)
→ clipped policy gradient update → LoRA adapter update
```

**LoRA (Low-Rank Adaptation):**

Only small adapter matrices attached to the attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`) are updated during training. The base Qwen2.5 weights stay frozen. After training, only the adapters (~80 MB) are saved per checkpoint — not the full 3B model.

**KL penalty:**

`kl_coef` controls how far the fine-tuned model is allowed to drift from the original Qwen2.5 base. A higher value keeps the model closer to its pre-trained behavior; a lower value gives the reward signal more influence.

**Interaction log as training data:**

Every saved interaction in `sessions/{id}/logs/interaction_log.jsonl` is a labeled training example with full context: prompt, response, traits used, individual scores, and scalar reward. This log can be used to:
- Warm-start future sessions with pre-collected human feedback
- Compare annotation behavior across different sessions
- Audit what the model learned and why

---

## 9. Hardware Requirements

| Setup | VRAM | Training speed |
|---|---|---|
| RTX 3090 / 4090 (24 GB) | fits comfortably with LoRA + bfloat16 | ~2–5 min/step |
| RTX 3080 / 4080 (12–16 GB) | tight — reduce `batch_size` to 1 | ~5–10 min/step |
| CPU only | not recommended for 3B model | very slow |

**To reduce VRAM further**, enable 4-bit quantization in `backend/rl_system/grpo_system/model.py`:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

This roughly halves VRAM usage at a small quality cost.
