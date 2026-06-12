# LLMTuner

An interactive system for behavioral fine-tuning of language models through human feedback. Users define behavioral traits, score model responses with sliders, and trigger online GRPO training through a no-code Streamlit interface running on consumer GPU hardware.

**Paper:** [LLMTuner: An Interactive System for Behavioral Fine-Tuning of Language Models Through Human Feedback](LLMTuner-Interactive_System_for_Behavioral_Fine_Tuning_of_LLMs_Through_Human_Feedback.pdf)

---

## System Pipeline

```
User defines traits (name, weight, description)
  ↓
Loop:
  Load 20 Newsgroups article + set instruction   [Prompt page]
    ↓
  Model generates 2–4 responses                  [Grade page]
  (Qwen2.5-1.5B-Instruct + LoRA, shared model)
    ↓
  User sets response length target with slider
    ↓
  User scores each response with sliders (−5 to +5 per trait)
    ↓
  Scores → weighted sum → scalar reward → saved to interaction_log.jsonl
    ↓
  [Every 3 complete rounds → online GRPO step fires automatically]
    policy log-prob  (LoRA active, gradients flow)
    reference log-prob  (adapters disabled, same model, no grad)
    loss = −advantage × log_prob + β × KL
    AdamW step → LoRA weights updated in place
    Next generation uses the refined model
  ↓
When happy with the model → Session page → Save Session → Export
  Option A: Download LoRA adapters ZIP  (~80MB, load with PEFT)
  Option B: Merge & Export full model   (~3GB, no PEFT needed)
```

> **Export rule:** Export reads from the saved checkpoint on disk. Always click **Save Session** before exporting, or you'll get stale weights.

---

## What Makes This True Online RLHF

Most RLHF pipelines collect data first, then train in a separate batch step. This system does **true online GRPO**: human scores are collected per-completion in real time, and the model updates after every 5 complete grading rounds. The model you generate from improves as you grade.

- Per-completion rewards: each response gets its own human score (not a prompt-level average)
- In-place model updates: LoRA weights are updated on the shared model with no reload
- Persistent optimizer: AdamW momentum accumulates across all 3-round windows
- KL reference via `disable_adapter_layers()`: no second model copy needed

---

## Architecture

```
backend/
├── user_interface/          ← Streamlit app (main entry point)
│   ├── app.py               ← page routing, sidebar, auto-load checkpoint
│   ├── backend_adapter.py   ← shared model (inference + training), reward
│   ├── grpo_adapter.py      ← OnlineGRPOSession + checkpoint helpers + export
│   ├── app/
│   │   ├── trait_manager.py ← load traits from config/traits.json
│   │   └── reward_bridge.py ← weighted sum reward from UI slider scores
│   └── ui/
│       ├── components.py    ← all 6 page renderers + panel helpers
│       ├── state.py         ← session state keys and accessors
│       ├── storage.py       ← JSONL read/write
│       ├── types.py         ← TraitConfig, ScoredTrait, InteractionLog
│       └── session_manager.py ← per-experiment directory isolation
│
├── rl_system/
│   └── grpo_system/
│       ├── config.py        ← GRPOConfig, local_config(), colab_config()
│       ├── model.py         ← load Qwen2.5 + LoRA (4-bit or bfloat16)
│       ├── online_step.py   ← custom GRPO gradient step for online training
│       ├── reward.py        ← reward_fn bridge (standalone use)
│       ├── data.py          ← 20 Newsgroups prompt builder
│       └── train.py         ← standalone batch training script (not wired to UI)
│
├── reward_system/           ← standalone reward computation module
├── data/                    ← 20 Newsgroups preprocessing pipeline
└── api/                     ← FastAPI REST layer (built, not yet integrated)
```

---

## Pages

| Page | Purpose |
|---|---|
| **Setup** | Define reward traits: name, description, weight |
| **Prompt** | Browse 20 Newsgroups, set article + instruction |
| **Grade** | Generate responses, set length, score with sliders, trigger online training |
| **History** | Master-detail view of all saved interactions |
| **Analytics** | Reward trend, trait breakdown, summary statistics |
| **Session** | Save/load online LoRA weights + export trained model |

---

## Online Training Detail

The online loop lives entirely on the Grade page. No separate training step needed.

**A "round"** = one prompt + all its generated responses, fully graded and saved.

After every **3 complete rounds**:
1. `OnlineGRPOSession.step(groups)` fires in a background thread
2. For each group, GRPO computes per-completion group-relative advantages: `(reward − mean) / std`
3. Policy log-probs computed with LoRA active (gradients flow)
4. Reference log-probs computed with `disable_adapter_layers()` — same model, zero overhead
5. Loss: `−advantage × log_prob + β × KL(policy ∥ base)`
6. AdamW step updates only the LoRA parameters (~2M of 1.5B total)
7. "Generate Responses" is disabled during the step (~30s on GTX 1650)
8. After completion: banner shows step count and last loss. Model is immediately live.

The AdamW optimizer persists across steps so momentum accumulates over multiple windows.

---

## Response Length

The Grade page has a **response length slider** (25–400 words, step 25). The target word count is appended as a soft instruction to the generation prompt — the model aims for this length but is not hard-cut. The original prompt (without the length hint) is what gets saved to the interaction log and the round buffer.

---

## Reward Formula

```
scalar_reward = Σ(trait.weight × slider_score)
```

Slider range: −5 (strong penalty) to +5 (strong reward). 0 is neutral.
Negative scores reduce the reward, driving the model away from that behavior.

---

## Experiment Sessions

Each session is a named directory under `sessions/`:

```
sessions/
  default/
    interaction_log.jsonl      ← all graded interactions
    online_checkpoint/         ← saved LoRA weights (written by Save Session)
  demo_session/
    ...
```

Sessions are fully isolated. Switching sessions in the sidebar resets all in-memory state and auto-loads the new session's checkpoint if one exists.

---

## Saving and Loading Progress

Online training weights live in memory during a session. To persist them across app restarts:

1. Go to **Session page**
2. Click **Save Session** — writes LoRA weights to `sessions/{name}/online_checkpoint/`
3. On next startup, the checkpoint is auto-loaded for the active session

If you switch sessions, the previous session's weights are reset and the new session's checkpoint (if any) loads automatically.

---

## Hardware

| Config | Model | VRAM | Notes |
|---|---|---|---|
| Local (`local_config`) | Qwen2.5-1.5B | 4GB | 4-bit QLoRA, GTX 1650 |
| Colab (`colab_config`) | Qwen2.5-3B | 15GB | bfloat16, T4 |

For online training on 4GB VRAM:
- 4-bit model: ~1.5GB
- LoRA adapters + optimizer state: ~240MB
- Activations during training step: ~500MB
- Total: ~2.2GB — fits comfortably

---

## Running

### Local (uv)

```bash
cd backend/user_interface
uv run streamlit run app.py
```

First launch downloads Qwen2.5-1.5B-Instruct (~3GB, cached by HuggingFace).

### Docker

**Prerequisites:** Docker Engine 24+, Compose v2, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU mode.

**Quick start (GPU):**

```bash
docker compose up --build
```

- UI → http://localhost:8501
- API → http://localhost:8000/docs

**CPU fallback** (slow inference/training, no GPU required):

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

**Optional environment variables** (set in `.env` or compose `environment`):

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace token if model access is gated |
| `HF_HOME` | Model cache directory (default `/cache/huggingface`) |
| `SCIKIT_LEARN_DATA` | 20 Newsgroups cache (default `/cache/sklearn`) |
| `BITSANDBYTES_NOWELCOME=1` | Suppress bitsandbytes startup banner |

**Data persistence:** Docker named volumes preserve experiment sessions, LoRA checkpoints, model downloads, and API interaction logs across container restarts:

| Volume | Contents |
|---|---|
| `sessions` | `backend/user_interface/sessions/` |
| `hf_cache` | HuggingFace model cache (~3GB on first run) |
| `sklearn_cache` | 20 Newsgroups dataset cache |
| `api_data` | `backend/api/data/interactions.jsonl` |

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | Qwen2.5-1.5B-Instruct (HuggingFace Transformers) |
| Fine-tuning | PEFT / LoRA (rank 16, q/k/v/o_proj) |
| Online GRPO | Custom gradient step (grpo_system/online_step.py) |
| Dataset | 20 Newsgroups (scikit-learn) |
| Persistence | JSONL files + LoRA checkpoint files |
| Quantization | BitsAndBytes (4-bit QLoRA) |

---

## Exporting the Model

Go to **Session → Export** (only visible once a checkpoint has been saved):

1. Click **Save Session** to write current LoRA weights to disk
2. Choose an export format:

- **Download Adapters ZIP** — lightweight LoRA-only files. Load with:
  ```python
  from peft import PeftModel
  model = PeftModel.from_pretrained(base_model, "online_checkpoint/")
  ```

- **Merge & Export Full Model** — bakes LoRA into base weights. Load with:
  ```python
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("sessions/my_session/merged_model/")
  ```

Export is always user-triggered. Nothing is merged or exported automatically.
