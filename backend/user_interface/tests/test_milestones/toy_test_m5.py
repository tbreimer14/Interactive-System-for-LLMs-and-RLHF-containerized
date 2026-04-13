"""
Toy test for Milestone 5 (End-to-end integration).

Runs the full pipeline without launching Streamlit:
    rag_stub -> trait_manager -> reward_bridge -> interaction_logger

Run with: uv run python tests/test_milestones/toy_test_m5.py
"""

import sys
import json
import tempfile
from pathlib import Path

# Walk up from this file until we find the app/ folder (the user_interface root)
project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.rag_stub import answer
from app.trait_manager import load_traits
from app.reward_bridge import compute_reward
from app.interaction_logger import log_interaction, read_log

CONFIG_PATH = project_root / "config" / "traits.json"
TEMP_LOG    = Path(tempfile.gettempdir()) / "ui_test_e2e_log.jsonl"

PROMPT = "Can you explain what PPO is and why it is used in RLHF?"


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def setup():
    if TEMP_LOG.exists():
        TEMP_LOG.unlink()


def test_rag(prompt):
    """Step 1: RAG stub returns a valid result."""
    print_section("TEST 1: RAG stub")

    try:
        rag_result = answer(prompt, k=3)
        assert rag_result["query"] == prompt
        assert isinstance(rag_result["retrieved"], list)
        assert len(rag_result["retrieved"]) == 3
        assert isinstance(rag_result["answer"], str)
        print(f"  [PASS] RAG returned answer + 3 chunks")
        return rag_result
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_traits():
    """Step 2: Traits load from config."""
    print_section("TEST 2: Trait manager")

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
    """Step 3: Reward bridge computes scalar from manual scores."""
    print_section("TEST 3: Reward bridge")

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


def test_log(rag_result, traits, scores, reward_result):
    """Step 4: Full interaction logs and reads back correctly."""
    print_section("TEST 4: Interaction logger")

    entry = {
        "prompt":        rag_result["query"],
        "retrieved":     rag_result["retrieved"],
        "response":      rag_result["answer"],
        "traits":        traits,
        "scores":        scores,
        "scalar_reward": reward_result["scalar_reward"],
    }

    try:
        log_interaction(entry, log_path=str(TEMP_LOG))
        entries = read_log(log_path=str(TEMP_LOG))
        assert len(entries) == 1
        saved = entries[0]
        assert saved["prompt"]        == entry["prompt"]
        assert saved["scalar_reward"] == entry["scalar_reward"]
        assert "timestamp" in saved
        print(f"  [PASS] Entry logged and read back correctly")
        return saved
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_print_summary(rag_result, traits, scores, reward_result):
    """Print a formatted end-to-end summary."""
    print_section("TEST 5: End-to-end summary")

    W = 60
    print("\n" + "=" * W)
    print("  UI SYSTEM - END-TO-END RESULT")
    print("=" * W)
    print(f"\n  Prompt:   \"{rag_result['query'][:60]}\"")
    print(f"  Response: \"{rag_result['answer'][:60]}...\"")
    print(f"\n  Retrieved chunks: {len(rag_result['retrieved'])}")
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
    print("  USER INTERFACE - MILESTONE 5 TEST SUITE")
    print("  End-to-end integration test")
    print("=" * 60)

    setup()

    try:
        rag_result           = test_rag(PROMPT)
        traits               = test_traits()
        scores, reward_result = test_reward(traits)
        test_log(rag_result, traits, scores, reward_result)
        test_print_summary(rag_result, traits, scores, reward_result)

        print("\n" + "=" * 60)
        print("  ALL MILESTONE 5 TESTS PASSED")
        print("  Run the UI with: uv run streamlit run main.py")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  TEST FAILED: {e}")
        print("=" * 60 + "\n")
        raise


if __name__ == "__main__":
    main()
