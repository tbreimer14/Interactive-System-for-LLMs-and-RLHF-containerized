"""
reward_fn.py

Converts user-provided trait scores into a single scalar reward.

The reward system does NOT decide scores itself — the user provides them.
This module's only job is to validate coverage and compute the weighted sum.

Formula:
    final_reward = sum(trait["weight"] * user_scores[trait["name"]]
                       for trait in traits)
"""

from app.schemas import Trait, RewardResult


def compute_reward(
    prompt: str,
    response: str,
    traits: list[Trait],
    user_scores: dict[str, float | int],
) -> RewardResult:
    """
    Compute a scalar reward from user-provided trait scores.

    Args:
        prompt:      the original user prompt (passed through for context/logging)
        response:    the model response that was scored (passed through)
        traits:      list of trait dicts from load_traits()
        user_scores: user-provided score for each trait, e.g. {"warmth": 3, "directness": -2}
                     scores use a -5 to +5 scale; negative values penalise the response

    Returns:
        RewardResult with:
            trait_scores  — the user_scores dict (passed through unchanged)
            weights       — weight per trait from config
            final_reward  — weighted sum of scores

    Raises:
        ValueError: if user_scores is missing a score for any trait in the config
    """
    # Validate that every trait has a score
    trait_names = {t["name"] for t in traits}
    scored_names = set(user_scores.keys())
    missing = trait_names - scored_names

    if missing:
        raise ValueError(
            f"user_scores is missing scores for: {missing}. "
            f"Expected scores for all traits: {trait_names}"
        )

    # Build weights dict for output
    weights = {t["name"]: t["weight"] for t in traits}

    # Weighted sum
    final_reward = sum(
        t["weight"] * user_scores[t["name"]]
        for t in traits
    )

    return RewardResult(
        trait_scores=dict(user_scores),
        weights=weights,
        final_reward=round(final_reward, 6),
    )
