"""
schemas.py

Simple typed structures for the reward system.
Using TypedDict so everything stays plain dicts — no class instantiation needed,
and downstream code (PPO, logging, UI) can consume them without importing this module.
"""

from typing import TypedDict


class Trait(TypedDict):
    """A single user-defined reward trait."""
    name: str           # e.g. "warmth"
    description: str    # e.g. "Sounds kind and supportive"
    weight: float       # contribution to final reward, e.g. 0.3


class RewardResult(TypedDict):
    """Output of compute_reward()."""
    trait_scores: dict[str, float | int]   # user-provided score per trait
    weights: dict[str, float]              # weight per trait from config
    final_reward: float                    # weighted sum of scores
