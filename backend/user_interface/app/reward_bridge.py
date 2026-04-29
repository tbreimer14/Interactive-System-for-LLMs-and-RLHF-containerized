"""
reward_bridge.py

Computes the scalar reward from user-provided manual scores and trait weights.

Unlike reward_system/reward_fn.py (which calls an automatic scorer),
this module takes scores the user entered in the UI — sliders or number inputs.
The weighted-sum formula is identical, keeping both systems compatible.

Formula:
    scalar_reward = sum(trait["weight"] * scores[trait["name"]] for trait in traits)
"""


def compute_reward(scores: dict, traits: list[dict]) -> dict:
    """
    Combine user-provided trait scores with their weights into a scalar reward.

    Args:
        scores: dict mapping trait name -> numeric score on a -5 to +5 scale, e.g.
                {"clarity": 3, "empathy": -2, "directness": 4}
                Negative scores penalise the response; zero is neutral.
        traits: list of trait dicts from load_traits(), each with
                keys: name, description, weight

    Returns:
        {
            "contributions": {"clarity": 1.2, "empathy": -0.6, "directness": 1.2},
            "scalar_reward": 1.8
        }

    Raises:
        KeyError: if a trait name in traits is not present in scores
    """
    contributions = {}

    for trait in traits:
        name = trait["name"]
        if name not in scores:
            raise KeyError(f"No score provided for trait '{name}'")
        contributions[name] = round(trait["weight"] * scores[name], 6)

    scalar_reward = round(sum(contributions.values()), 6)

    return {
        "contributions": contributions,
        "scalar_reward": scalar_reward,
    }
