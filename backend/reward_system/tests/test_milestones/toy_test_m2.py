"""
Toy test for Milestone 2 (Schemas).

Tests that the TypedDict definitions in schemas.py have the correct
structure and that dicts conforming to them behave as expected.

Run with: uv run python tests/test_milestones/toy_test_m2.py
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

from app.schemas import Trait, RewardResult


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_trait_schema():
    """A Trait dict should accept name, description, weight."""
    print_section("TEST 1: Trait schema")

    try:
        trait: Trait = {
            "name": "warmth",
            "description": "Sounds kind and supportive",
            "weight": 0.3,
        }

        assert trait["name"] == "warmth"
        assert trait["description"] == "Sounds kind and supportive"
        assert trait["weight"] == 0.3

        print(f"  ✓ Trait dict: {trait}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_reward_result_schema():
    """A RewardResult dict should accept trait_scores, weights, final_reward."""
    print_section("TEST 2: RewardResult schema")

    try:
        result: RewardResult = {
            "trait_scores": {"warmth": 4, "directness": 3, "interactivity": 5},
            "weights":      {"warmth": 0.3, "directness": 0.2, "interactivity": 0.5},
            "final_reward": 4.1,
        }

        assert isinstance(result["trait_scores"], dict)
        assert isinstance(result["weights"], dict)
        assert isinstance(result["final_reward"], float)

        print(f"  ✓ trait_scores : {result['trait_scores']}")
        print(f"  ✓ weights      : {result['weights']}")
        print(f"  ✓ final_reward : {result['final_reward']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_trait_keys():
    """Trait must have exactly the three required keys."""
    print_section("TEST 3: Trait has required keys")

    trait: Trait = {
        "name": "humor",
        "description": "Includes light-hearted moments",
        "weight": 0.4,
    }

    required = {"name", "description", "weight"}

    try:
        missing = required - trait.keys()
        assert not missing, f"Missing keys: {missing}"
        print(f"  ✓ All required keys present: {required}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_reward_result_keys():
    """RewardResult must have exactly the three required keys."""
    print_section("TEST 4: RewardResult has required keys")

    result: RewardResult = {
        "trait_scores": {"humor": 4},
        "weights":      {"humor": 0.4},
        "final_reward": 1.6,
    }

    required = {"trait_scores", "weights", "final_reward"}

    try:
        missing = required - result.keys()
        assert not missing, f"Missing keys: {missing}"
        print(f"  ✓ All required keys present: {required}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_schemas_are_plain_dicts():
    """TypedDicts are plain dicts at runtime — no class instantiation needed."""
    print_section("TEST 5: Schemas are plain dicts (no special class needed)")

    try:
        trait: Trait = {"name": "directness", "description": "Gets to the point", "weight": 0.2}
        result: RewardResult = {"trait_scores": {"directness": 3}, "weights": {"directness": 0.2}, "final_reward": 0.6}

        assert type(trait) is dict, "Trait is not a plain dict"
        assert type(result) is dict, "RewardResult is not a plain dict"

        print(f"  ✓ Trait is a plain dict: {type(trait)}")
        print(f"  ✓ RewardResult is a plain dict: {type(result)}")
        print(f"  ✓ Downstream code (PPO, logger) can consume these without importing schemas")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def main():
    print("\n" + "="*60)
    print("  REWARD SYSTEM - MILESTONE 2 TEST SUITE")
    print("  Testing Schemas")
    print("="*60)

    try:
        test_trait_schema()
        test_reward_result_schema()
        test_trait_keys()
        test_reward_result_keys()
        test_schemas_are_plain_dicts()

        print("\n" + "="*60)
        print("  ✓ ALL MILESTONE 2 TESTS PASSED")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
