"""
Toy test for Milestone 3 (Reward Function).

Tests that compute_reward() correctly accepts user-provided scores,
validates coverage, and returns the right weighted sum.

Run with: uv run python tests/test_milestones/toy_test_m3.py
"""

import sys
from pathlib import Path

# Walk up from this file until we find the app/ folder (the reward_system root)
project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.trait_loader import load_traits
from app.reward_fn import compute_reward


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


PROMPT   = "Can you help me understand reinforcement learning?"
RESPONSE = "Sure! RL is about learning from rewards. Want to go deeper?"

TRAITS = [
    {"name": "interactivity", "description": "Encourages continued conversation", "weight": 0.5},
    {"name": "warmth",        "description": "Sounds kind, supportive, and human", "weight": 0.3},
    {"name": "directness",    "description": "Gets to the point without being vague", "weight": 0.2},
]

USER_SCORES = {
    "interactivity": 5,
    "warmth": 4,
    "directness": 3,
}


def test_output_structure():
    """compute_reward should return a dict with trait_scores, weights, final_reward."""
    print_section("TEST 1: Output structure")

    try:
        result = compute_reward(PROMPT, RESPONSE, TRAITS, USER_SCORES)

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "trait_scores" in result, "Missing key: 'trait_scores'"
        assert "weights" in result,      "Missing key: 'weights'"
        assert "final_reward" in result, "Missing key: 'final_reward'"

        print(f"  ✓ Returns dict with 'trait_scores', 'weights', 'final_reward'")
        return result
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_trait_scores_passthrough(result):
    """trait_scores should be the user_scores passed in — unchanged."""
    print_section("TEST 2: trait_scores matches user_scores")

    try:
        for name, score in USER_SCORES.items():
            assert result["trait_scores"][name] == score, \
                f"Score mismatch for '{name}': expected {score}, got {result['trait_scores'][name]}"
            print(f"  ✓ {name:<16}  user_score={score}  stored={result['trait_scores'][name]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_weights_match_config(result):
    """weights in output should match the trait weights from config."""
    print_section("TEST 3: weights match config")

    try:
        for trait in TRAITS:
            assert result["weights"][trait["name"]] == trait["weight"], \
                f"Weight mismatch for '{trait['name']}'"
            print(f"  ✓ {trait['name']:<16}  weight={result['weights'][trait['name']]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_final_reward_is_weighted_sum(result):
    """final_reward should equal sum(weight * score) for each trait."""
    print_section("TEST 4: final_reward equals weighted sum")

    try:
        expected = sum(
            t["weight"] * USER_SCORES[t["name"]]
            for t in TRAITS
        )
        expected = round(expected, 6)
        actual   = result["final_reward"]

        assert abs(actual - expected) < 1e-9, \
            f"final_reward {actual} != expected weighted sum {expected}"

        print(f"  ✓ final_reward = {actual}  (matches weighted sum)")
        print(f"    = {' + '.join(f'{t[\"weight\"]}×{USER_SCORES[t[\"name\"]]}' for t in TRAITS)}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_missing_score_raises():
    """compute_reward should raise ValueError if a trait score is missing."""
    print_section("TEST 5: Missing score raises ValueError")

    incomplete_scores = {"interactivity": 5}  # missing warmth and directness

    try:
        compute_reward(PROMPT, RESPONSE, TRAITS, incomplete_scores)
        print(f"  ✗ Should have raised ValueError")
        raise AssertionError("Expected ValueError but got none")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")


def test_with_real_config():
    """Run compute_reward using the actual traits.json and print the result."""
    print_section("TEST 6: Full run with real traits.json")

    config_path = project_root / "config" / "traits.json"

    try:
        traits = load_traits(str(config_path))
        user_scores = {t["name"]: 4 for t in traits}  # score every trait as 4

        result = compute_reward(PROMPT, RESPONSE, traits, user_scores)

        print(f"  Prompt:   \"{PROMPT}\"")
        print(f"  Response: \"{RESPONSE}\"")
        print()
        for name, score in result["trait_scores"].items():
            weight = result["weights"][name]
            print(f"  {name:<16}  score={score}  weight={weight}  → {round(weight*score, 4)}")
        print(f"\n  Final reward: {result['final_reward']}")
        print(f"\n  ✓ Full run with real config succeeded")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def main():
    print("\n" + "="*60)
    print("  REWARD SYSTEM - MILESTONE 3 TEST SUITE")
    print("  Testing Reward Function")
    print("="*60)

    try:
        result = test_output_structure()
        test_trait_scores_passthrough(result)
        test_weights_match_config(result)
        test_final_reward_is_weighted_sum(result)
        test_missing_score_raises()
        test_with_real_config()

        print("\n" + "="*60)
        print("  ✓ ALL MILESTONE 3 TESTS PASSED")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
