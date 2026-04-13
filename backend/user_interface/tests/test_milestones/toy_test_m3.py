"""
Toy test for Milestone 3 (Reward Bridge).

Tests that reward_bridge.compute_reward() correctly computes weighted scores.

Run with: uv run python tests/test_milestones/toy_test_m3.py
"""

import sys
from pathlib import Path

# Walk up from this file until we find the app/ folder (the user_interface root)
project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.trait_manager import load_traits
from app.reward_bridge import compute_reward

CONFIG_PATH = project_root / "config" / "traits.json"


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_return_structure(traits):
    """Test that compute_reward returns a dict with the required keys."""
    print_section("TEST 1: Return structure")

    scores = {t["name"]: 3 for t in traits}

    try:
        result = compute_reward(scores, traits)
        required_keys = {"contributions", "scalar_reward"}
        missing = required_keys - result.keys()
        assert not missing, f"Missing keys in result: {missing}"
        print(f"  [PASS] Result has all required keys: {required_keys}")
        return result
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_contributions_per_trait(traits):
    """Test that contributions dict has one entry per trait."""
    print_section("TEST 2: Contributions per trait")

    scores = {t["name"]: 4 for t in traits}

    try:
        result = compute_reward(scores, traits)
        for trait in traits:
            assert trait["name"] in result["contributions"], \
                f"Missing contribution for trait '{trait['name']}'"
        print(f"  [PASS] Contributions present for all {len(traits)} traits")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_math_correctness(traits):
    """Test that the weighted sum is computed correctly."""
    print_section("TEST 3: Math correctness")

    scores = {t["name"]: 5 for t in traits}
    expected_reward = round(sum(t["weight"] * 5 for t in traits), 6)

    try:
        result = compute_reward(scores, traits)
        assert abs(result["scalar_reward"] - expected_reward) < 1e-9, (
            f"Expected scalar_reward={expected_reward}, "
            f"got {result['scalar_reward']}"
        )
        print(f"  [PASS] scalar_reward={result['scalar_reward']} matches expected={expected_reward}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_known_values():
    """Test against hand-computed values with fixed traits and scores."""
    print_section("TEST 4: Known values (hand-computed)")

    traits = [
        {"name": "clarity",   "description": "...", "weight": 0.4},
        {"name": "empathy",   "description": "...", "weight": 0.3},
        {"name": "directness","description": "...", "weight": 0.3},
    ]
    scores = {"clarity": 4, "empathy": 3, "directness": 5}

    # clarity:    0.4 * 4 = 1.6
    # empathy:    0.3 * 3 = 0.9
    # directness: 0.3 * 5 = 1.5
    # total:               4.0

    try:
        result = compute_reward(scores, traits)
        assert abs(result["contributions"]["clarity"]   - 1.6) < 1e-9
        assert abs(result["contributions"]["empathy"]   - 0.9) < 1e-9
        assert abs(result["contributions"]["directness"]- 1.5) < 1e-9
        assert abs(result["scalar_reward"]              - 4.0) < 1e-9
        print(f"  [PASS] All hand-computed values match")
        print(f"         clarity=1.6  empathy=0.9  directness=1.5  total=4.0")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_missing_score_raises(traits):
    """Test that a missing score for a trait raises KeyError."""
    print_section("TEST 5: Missing score raises KeyError")

    incomplete_scores = {traits[0]["name"]: 3}  # only first trait scored

    try:
        compute_reward(incomplete_scores, traits)
        print(f"  [FAIL] Should have raised KeyError")
        raise AssertionError("Expected KeyError but got none")
    except KeyError as e:
        print(f"  [PASS] Correctly raised KeyError: {e}")


def test_print_result(traits):
    """Print a formatted reward breakdown for visual confirmation."""
    print_section("TEST 6: Visual output")

    scores = {"clarity": 4, "empathy": 3, "directness": 5}
    result = compute_reward(scores, traits)

    print(f"  Scores and contributions:")
    for trait in traits:
        name = trait["name"]
        print(
            f"    {name:<16} score={scores[name]}  "
            f"weight={trait['weight']}  "
            f"-> contribution={result['contributions'][name]}"
        )
    print(f"\n  Scalar reward: {result['scalar_reward']}")
    print(f"\n  [PASS] Output looks correct above")


def main():
    print("\n" + "=" * 60)
    print("  USER INTERFACE - MILESTONE 3 TEST SUITE")
    print("  Testing Reward Bridge")
    print("=" * 60)

    try:
        traits = load_traits(str(CONFIG_PATH))

        test_return_structure(traits)
        test_contributions_per_trait(traits)
        test_math_correctness(traits)
        test_known_values()
        test_missing_score_raises(traits)
        test_print_result(traits)

        print("\n" + "=" * 60)
        print("  ALL MILESTONE 3 TESTS PASSED")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  TEST FAILED: {e}")
        print("=" * 60 + "\n")
        raise


if __name__ == "__main__":
    main()
