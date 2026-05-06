"""
grpo_adapter.py

Online GRPO training wired to the shared model in backend_adapter.py.

Operates directly on the shared PeftModel (same object used for inference).
Called automatically after every 3 complete grading rounds on the Grade page.
Uses a custom GRPO gradient step (grpo_system/online_step.py) that computes
per-completion rewards and uses disable_adapter_layers() for the KL reference —
no second model copy needed.

    OnlineGRPOSession.step(groups)  →  one AdamW step in background thread
                                       LoRA weights updated in place
                                       next inference call uses updated model
"""

import io
import sys
import threading
import zipfile
from pathlib import Path

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND / "rl_system"))
sys.path.insert(0, str(_BACKEND))

import torch


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
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            dtype=torch.float32,
        )

    peft_model  = PeftModel.from_pretrained(base, adapter_dir)
    merged      = peft_model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ── Online checkpoint helpers ──────────────────────────────────────────────────

def save_online_checkpoint(model, tokenizer, checkpoint_dir: str) -> None:
    """Save the shared model's current LoRA adapter weights to disk."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def load_online_checkpoint(model, checkpoint_dir: str) -> None:
    """Load saved LoRA adapter weights into the shared model in place."""
    from peft import load_peft_weights, set_peft_model_state_dict
    weights = load_peft_weights(checkpoint_dir)
    set_peft_model_state_dict(model, weights)


def reset_lora_weights(model) -> None:
    """Reset all LoRA B matrices to zero — restores model to base behaviour."""
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()


# ── Online GRPO session ────────────────────────────────────────────────────────

class OnlineGRPOSession:
    """
    Persistent online GRPO trainer that operates on the shared inference model.

    Each call to step() runs one GRPO gradient update (in a background thread)
    on a buffer of 3 human-graded rounds, then returns. The model is updated
    in place, so the next generate_responses() call uses the refined weights.

    The session is never "done" — it persists across multiple 3-round windows
    for the lifetime of the Streamlit session.
    """

    def __init__(self, model, tokenizer, beta: float = 0.1, lr: float = 1e-5):
        self.model     = model
        self.tokenizer = tokenizer
        self.beta      = beta
        self.steps     = 0
        self.log: list[str] = []
        self.is_stepping   = False
        self.last_loss: float | None = None
        self.error: str | None = None
        self._thread: threading.Thread | None = None

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

    def step(self, groups: list[dict]) -> None:
        """Trigger one GRPO update in a background thread. No-op if already running."""
        if self.is_stepping:
            return
        self.is_stepping = True
        self.error       = None
        self._thread = threading.Thread(
            target=self._run_step, args=(groups,), daemon=True
        )
        self._thread.start()

    def _run_step(self, groups: list[dict]) -> None:
        try:
            from grpo_system.online_step import grpo_step
            self._emit(f"Online step {self.steps + 1} — {len(groups)} rounds…")
            loss = grpo_step(
                self.model, self.tokenizer, groups, self.optimizer, beta=self.beta
            )
            self.steps    += 1
            self.last_loss = loss
            self._emit(f"Step {self.steps} done  loss={loss:.4f}")
        except Exception as exc:
            self.error = str(exc)
            self._emit(f"Error: {exc}")
        finally:
            self.is_stepping = False

    def _emit(self, msg: str) -> None:
        self.log.append(msg)
