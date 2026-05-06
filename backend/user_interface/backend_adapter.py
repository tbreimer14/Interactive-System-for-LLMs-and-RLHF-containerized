"""
backend_adapter.py

Single point of contact between the UI and the backend modules.

generate_responses() loads Qwen2.5 + LoRA once via @st.cache_resource and runs
inference. The same model object is used for online GRPO training — LoRA weights
are updated in place, so the next generate_responses() call automatically uses
the refined model.

compute_reward() calls reward_bridge directly (no network hop needed).
"""

import sys
from pathlib import Path

_BACKEND = Path(__file__).parent.parent
_RL_PATH = str(_BACKEND / "rl_system")
if _RL_PATH not in sys.path:
    sys.path.insert(0, _RL_PATH)

import torch
import streamlit as st

from grpo_system.config import local_config
from grpo_system.model import load_model as _load_lora_model
from app.reward_bridge import compute_reward as _compute_reward
from ui.types import TraitConfig, ScoredTrait

_MAX_NEW_TOKENS = 256


@st.cache_resource(show_spinner="Loading Qwen2.5 + LoRA…")
def _load_model():
    """
    Load Qwen2.5-1.5B-Instruct with LoRA adapters once.
    Reused for both inference and online GRPO training.
    LoRA B matrices are zero-initialized so initial output = base model.
    """
    config = local_config()
    model, tokenizer = _load_lora_model(config)
    model.eval()
    return model, tokenizer


def get_shared_model():
    """Return the cached (PeftModel, tokenizer) — shared by inference and training."""
    return _load_model()


class BackendAdapter:
    """Wraps all backend calls behind a clean interface."""

    def compute_reward(
        self,
        scores: dict[str, float],
        traits: list[TraitConfig],
    ) -> tuple[list[ScoredTrait], float]:
        trait_dicts = [t.to_dict() for t in traits]
        result = _compute_reward(scores, trait_dicts)

        scored_traits = [
            ScoredTrait(
                name=t.name,
                weight=t.weight,
                score=scores[t.name],
                contribution=result["contributions"][t.name],
            )
            for t in traits
        ]
        return scored_traits, result["scalar_reward"]

    def generate_responses(self, prompt: str, n: int) -> list[str]:
        """
        Generate n responses for the given prompt using Qwen2.5-1.5B-Instruct + LoRA.

        Args:
            prompt: combined article + instruction string from the Prompt page
            n:      number of responses to generate (2 or 4)

        Returns:
            list of n decoded response strings
        """
        model, tokenizer = _load_model()
        model.eval()

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in output_ids
        ]
