"""
grpo_adapter.py

Bridges the JSONL interaction log into a GRPO training run.

Flow:
    interaction_log.jsonl  (prompt + scalar_reward per saved interaction)
    ↓
    load_scored_prompts()  builds {prompt: mean_reward} lookup
    ↓
    GRPOSession.start()    runs GRPOTrainer in a background thread
    ↓
    grpo_output/           LoRA adapters saved here after training
    ↓  (optional)
    merge_and_export()     merges LoRA into base weights → standalone model
"""

import io
import json
import sys
import threading
import zipfile
from datetime import datetime
from pathlib import Path

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND / "rl_system"))
sys.path.insert(0, str(_BACKEND))

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

from grpo_system.config import local_config
from grpo_system.model import load_model


# ── Session record persistence ─────────────────────────────────────────────────

def _sessions_log_path(interaction_log_path: str) -> Path:
    return Path(interaction_log_path).parent / "training_sessions.jsonl"


def save_session_record(session, interaction_log_path: str) -> None:
    """Append a completed (or failed) session to the training sessions log."""
    path = _sessions_log_path(interaction_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "status":     session.status,
        "num_epochs": session.num_epochs,
        "n_prompts":  session.n_prompts,
        "output_dir": session.output_dir,
        "log":        session.log,
        "error":      session.error,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_session_records(interaction_log_path: str) -> list[dict]:
    """Return all past training session records, newest first."""
    path = _sessions_log_path(interaction_log_path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(reversed(records))


# ── Export helpers ─────────────────────────────────────────────────────────────

def zip_adapter_dir(adapter_dir: str) -> bytes:
    """
    ZIP the adapter directory in memory.
    Returns raw bytes suitable for st.download_button.
    """
    root = Path(adapter_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in root.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(root.parent))
    buf.seek(0)
    return buf.read()


def merge_and_export(adapter_dir: str, output_dir: str) -> None:
    """
    Merge LoRA adapters into the base model weights and save a standalone model.

    The result in output_dir can be loaded with AutoModelForCausalLM.from_pretrained()
    without PEFT — no adapter files needed.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    if torch.cuda.is_available():
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float32,
        )

    peft_model  = PeftModel.from_pretrained(base, adapter_dir)
    merged      = peft_model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ── Interaction log helpers ────────────────────────────────────────────────────

def load_scored_prompts(log_path: str) -> dict[str, float]:
    """
    Read JSONL log and return {prompt: mean_scalar_reward}.
    Multiple entries for the same prompt are averaged.
    """
    path = Path(log_path)
    if not path.exists():
        return {}

    buckets: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                prompt = entry.get("prompt", "").strip()
                reward = entry.get("scalar_reward")
                if prompt and reward is not None:
                    buckets.setdefault(prompt, []).append(float(reward))
            except (json.JSONDecodeError, ValueError):
                pass

    return {p: sum(rs) / len(rs) for p, rs in buckets.items()}


def _build_reward_fn(prompt_reward_map: dict[str, float]):
    raw_prompts = list(prompt_reward_map.keys())
    rewards     = list(prompt_reward_map.values())
    default     = sum(rewards) / len(rewards) if rewards else 0.0

    def reward_fn(prompts, completions, **kwargs):
        out = []
        for p in prompts:
            matched = default
            for raw, r in zip(raw_prompts, rewards):
                if raw in p:
                    matched = r
                    break
            out.append(matched)
        return out

    return reward_fn


# ── Training session ───────────────────────────────────────────────────────────

class GRPOSession:
    """
    Manages one GRPO training run in a background thread.

    Attributes (polled by the UI):
        status     : "idle" | "loading" | "training" | "done" | "error"
        log        : list[str] — append-only progress lines
        done       : bool
        error      : str | None
        n_prompts  : int — number of unique prompts used
        output_dir : str | None — path to saved LoRA adapters
    """

    def __init__(self, log_path: str, num_epochs: int = 1, adapter_dir: str | None = None):
        self.log_path    = log_path
        self.num_epochs  = num_epochs
        self.adapter_dir = adapter_dir  # session-specific output; falls back to config default
        self.status      = "idle"
        self.log: list[str] = []
        self.done        = False
        self.error: str | None = None
        self.n_prompts   = 0
        self.output_dir: str | None = None  # set after training completes
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.status = "loading"
        self.log    = []
        self.done   = False
        self.error  = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._train()
        except Exception as exc:
            self.error  = str(exc)
            self.status = "error"
            self.done   = True
        finally:
            save_session_record(self, self.log_path)

    def _train(self):
        self._emit("Reading interaction log…")
        prompt_reward_map = load_scored_prompts(self.log_path)
        if not prompt_reward_map:
            raise ValueError(
                "No scored interactions found. Grade some responses on the Grade page first."
            )
        self.n_prompts = len(prompt_reward_map)
        self._emit(f"Found {self.n_prompts} unique prompt(s) with human rewards.")

        config = local_config()
        config.num_train_epochs = self.num_epochs
        if self.adapter_dir:
            config.output_dir = self.adapter_dir

        self._emit(f"Loading {config.model_name} with LoRA…")
        model, tokenizer = load_model(config)

        self._emit("Building dataset from logged prompts…")
        formatted = []
        for p in prompt_reward_map:
            messages = [{"role": "user", "content": p}]
            formatted.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        dataset = Dataset.from_dict({"prompt": formatted})
        self._emit(f"Dataset: {len(dataset)} prompt(s)")

        reward_fn = _build_reward_fn(prompt_reward_map)

        has_gpu = torch.cuda.is_available()
        trl_cfg = TRLGRPOConfig(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            num_generations=config.num_generations,
            max_completion_length=config.max_completion_length,
            per_device_train_batch_size=config.batch_size,
            num_train_epochs=self.num_epochs,
            beta=config.kl_coef,
            bf16=has_gpu,
            use_cpu=not has_gpu,
            logging_steps=1,
            report_to="none",
        )

        session = self

        class _ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                step   = state.global_step
                loss   = logs.get("loss", logs.get("train_loss", "—"))
                reward = logs.get("reward", "—")
                session._emit(f"step {step}  loss={loss}  reward={reward}")

        self._emit("Starting GRPO training…")
        self.status = "training"

        trainer = GRPOTrainer(
            model=model,
            args=trl_cfg,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
            callbacks=[_ProgressCallback()],
        )
        trainer.train()

        self._emit(f"Training done. Saving LoRA adapters to {config.output_dir}/…")
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        self.output_dir = config.output_dir
        self._emit(f"Adapters saved to {config.output_dir}/")
        self.status = "done"
        self.done   = True

    def _emit(self, msg: str):
        self.log.append(msg)
