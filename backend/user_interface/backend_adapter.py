"""
backend_adapter.py

Single point of contact between the UI and the backend modules.

generate_responses() loads Qwen2.5-1.5B-Instruct once via @st.cache_resource
and runs inference with the model's chat template.
compute_reward() calls reward_bridge directly (no network hop needed).
"""

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.reward_bridge import compute_reward as _compute_reward
from ui.types import TraitConfig, ScoredTrait

_MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"
_MAX_NEW_TOKENS = 256


@st.cache_resource(show_spinner="Loading Qwen2.5 model…")
def _load_model():
    """Load tokenizer + model once; reused across all Streamlit reruns."""
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=torch.float32,
        )

    model.eval()
    return model, tokenizer


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
        Generate n responses for the given prompt using Qwen2.5-1.5B-Instruct.

        Args:
            prompt: combined article + instruction string from the Prompt page
            n:      number of responses to generate (2 or 4)

        Returns:
            list of n decoded response strings
        """
        model, tokenizer = _load_model()

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
