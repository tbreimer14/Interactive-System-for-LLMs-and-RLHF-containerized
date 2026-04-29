"""
Bridges reward_system's compute_reward() into the function signature
that GRPOTrainer expects: reward_fn(prompts, completions) -> list[float].
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REWARD_SYSTEM_PATH = os.path.abspath(os.path.join(_THIS_DIR, "../../reward_system"))
sys.path.insert(0, _REWARD_SYSTEM_PATH)

from app.trait_loader import load_traits
from app.reward_fn import compute_reward

TRAITS_PATH = os.path.join(_REWARD_SYSTEM_PATH, "config", "traits.json")


def build_reward_fn(get_scores_fn):
    """
    Factory that returns a reward function compatible with GRPOTrainer.

    Args:
        get_scores_fn: callable(prompt, response) -> dict[str, float]
            Collects slider scores from the UI for a given response.

    Returns:
        reward_fn(prompts, completions, **kwargs) -> list[float]
    """
    traits = load_traits(TRAITS_PATH)

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, response in zip(prompts, completions):
            user_scores = get_scores_fn(prompt, response)
            result = compute_reward(prompt, response, traits, user_scores)
            rewards.append(result["final_reward"])
        return rewards

    return reward_fn
