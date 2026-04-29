"""
Toy test for Milestone 4 (End-to-end integration).

Runs the full pipeline without launching Streamlit:
    trait_manager -> reward_bridge -> storage

Run with: uv run python tests/test_milestones/toy_test_m5.py
"""

import sys
import json
import tempfile
from pathlib import Path

project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.trait_manager import load_traits
from app.reward_bridge import compute_reward
from ui.storage import save_interaction, load_interactions
from ui.types import InteractionLog

CONFIG_PATH = project_root / "config" / "traits.json"
TEMP_LOG    = Path(tempfile.gettempdir()) / "ui_test_e2e_log.jsonl"

PROMPT   = "Can you explain what PPO is and why it is used in RLHF?"
RESPONSE = "PPO stands for Proximal Policy Optimization. It is used in RLHF because..."


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def setup():
    if TEMP_LOG.exists():
        TEMP_LOG.unlink()


def test_traits():
    """Step 1: Traits load from config."""
    print_section("TEST 1: Trait manager")

    try:
        traits = load_traits(str(CONFIG_PATH))
        assert len(traits) > 0
        for t in traits:
            assert {"name", "description", "weight"} <= t.keys()
        print(f"  [PASS] Loaded {len(traits)} traits from config")
        return traits
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_reward(traits):
    """Step 2: Reward bridge computes scalar from manual scores."""
    print_section("TEST 2: Reward bridge")

    scores = {t["name"]: 4 for t in traits}
    expected = round(sum(t["weight"] * 4 for t in traits), 6)

    try:
        result = compute_reward(scores, traits)
        assert abs(result["scalar_reward"] - expected) < 1e-9
        print(f"  [PASS] scalar_reward={result['scalar_reward']} (expected {expected})")
        return scores, result
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_log(traits, scores, reward_result):
    """Step 3: Full interaction logs and reads back correctly."""
    print_section("TEST 3: Storage")

    entry = InteractionLog(
        timestamp="2026-01-01T00:00:00",
        prompt=PROMPT,
        response=RESPONSE,
        traits=traits,
        scalar_reward=reward_result["scalar_reward"],
    )

    try:
        save_interaction(entry, log_path=str(TEMP_LOG))
        entries = load_interactions(log_path=str(TEMP_LOG))
        assert len(entries) == 1
        saved = entries[0]
        assert saved["prompt"]        == PROMPT
        assert saved["response"]      == RESPONSE
        assert saved["scalar_reward"] == reward_result["scalar_reward"]
        print(f"  [PASS] Entry logged and read back correctly")
        return saved
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_print_summary(traits, scores, reward_result):
    """Print a formatted end-to-end summary."""
    print_section("TEST 4: End-to-end summary")

    W = 60
    print("\n" + "=" * W)
    print("  UI SYSTEM - END-TO-END RESULT")
    print("=" * W)
    print(f"\n  Prompt:   \"{PROMPT[:60]}\"")
    print(f"  Response: \"{RESPONSE[:60]}...\"")
    print(f"\n  Trait Scores:")
    for trait in traits:
        name = trait["name"]
        print(
            f"    {name:<16}  score={scores[name]}  "
            f"weight={trait['weight']}  "
            f"-> {reward_result['contributions'][name]:.3f}"
        )
    print(f"\n  Scalar Reward: {reward_result['scalar_reward']}")
    print(f"\n  Logged to: {TEMP_LOG}")
    print("=" * W + "\n")
    print(f"  [PASS] Full pipeline ran successfully")


def main():
    print("\n" + "=" * 60)
    print("  USER INTERFACE - MILESTONE 4 TEST SUITE")
    print("  End-to-end integration test")
    print("=" * 60)

    setup()

    try:
        traits                = test_traits()
        scores, reward_result = test_reward(traits)
        test_log(traits, scores, reward_result)
        test_print_summary(traits, scores, reward_result)

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("  Run the UI with: uv run streamlit run app.py")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  TEST FAILED: {e}")
        print("=" * 60 + "\n")
        raise


if __name__ == "__main__":
    main()
