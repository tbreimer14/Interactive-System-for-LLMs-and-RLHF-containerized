"""
Toy test for Milestone 5 (Main Entry Point).

Tests the full end-to-end pipeline: traits → user scores → reward → log.
Also tests that main.py runs without error as a subprocess.

Run with: uv run python tests/test_milestones/toy_test_m5.py
"""

import sys
import json
import subprocess
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
from app.logger import log_reward

TEST_LOG_PATH = project_root / "logs" / "test_reward_log_m5.jsonl"

PROMPT      = "Can you help me understand reinforcement learning?"
RESPONSE    = "Sure! RL is about learning from rewards. Want to go deeper?"
USER_SCORES = {"interactivity": 5, "warmth": 4, "directness": 3}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_full_pipeline():
    """Run all four steps and verify output at each stage."""
    print_section("TEST 1: Full pipeline (traits → scores → reward → log)")

    if TEST_LOG_PATH.exists():
        TEST_LOG_PATH.unlink()

    try:
        # Step 1: load traits
        traits = load_traits(str(project_root / "config" / "traits.json"))
        assert len(traits) > 0
        print(f"  ✓ Loaded {len(traits)} traits")

        # Step 2: compute reward with user scores
        result = compute_reward(PROMPT, RESPONSE, traits, USER_SCORES)
        assert "trait_scores" in result
        assert "weights" in result
        assert "final_reward" in result
        print(f"  ✓ Computed reward: {result['final_reward']}")

        # Step 3: log
        log_reward(PROMPT, RESPONSE, traits, result, log_path=str(TEST_LOG_PATH))
        assert TEST_LOG_PATH.exists()
        print(f"  ✓ Logged to: {TEST_LOG_PATH}")

        # Step 4: read back and verify
        with open(TEST_LOG_PATH, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())

        assert entry["final_reward"] == result["final_reward"]
        assert entry["prompt"] == PROMPT
        assert "traits" in entry
        assert "user_scores" in entry
        print(f"  ✓ Log entry matches computed result")

        return traits, result
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_output_shape(traits, result):
    """Verify the output dict has the correct shape for PPO / downstream use."""
    print_section("TEST 2: Output shape for downstream (PPO) use")

    try:
        # trait_scores has one key per trait
        assert set(result["trait_scores"].keys()) == {t["name"] for t in traits}
        print(f"  ✓ trait_scores keys: {set(result['trait_scores'].keys())}")

        # weights has one key per trait
        assert set(result["weights"].keys()) == {t["name"] for t in traits}
        print(f"  ✓ weights keys match traits")

        # final_reward is a float
        assert isinstance(result["final_reward"], float)
        print(f"  ✓ final_reward is a float: {result['final_reward']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_different_scores_give_different_rewards():
    """Different user scores should produce different final rewards."""
    print_section("TEST 3: Different user scores produce different rewards")

    traits = load_traits(str(project_root / "config" / "traits.json"))

    scores_a = {t["name"]: 5 for t in traits}  # all 5s
    scores_b = {t["name"]: 1 for t in traits}  # all 1s

    try:
        result_a = compute_reward(PROMPT, RESPONSE, traits, scores_a)
        result_b = compute_reward(PROMPT, RESPONSE, traits, scores_b)

        assert result_a["final_reward"] != result_b["final_reward"]
        print(f"  ✓ All-5 reward:  {result_a['final_reward']}")
        print(f"  ✓ All-1 reward:  {result_b['final_reward']}")
        print(f"  ✓ Rewards differ as expected")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_main_runs():
    """Run main.py as a subprocess and verify it exits cleanly."""
    print_section("TEST 4: main.py runs without error")

    main_path = project_root / "main.py"

    try:
        proc = subprocess.run(
            [sys.executable, str(main_path)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        if proc.returncode != 0:
            print(f"  ✗ main.py exited with code {proc.returncode}")
            print(f"  stderr: {proc.stderr}")
            raise RuntimeError(f"main.py failed:\n{proc.stderr}")

        print(f"  ✓ main.py exited cleanly (return code 0)")
        print(f"\n  Output from main.py:")
        for line in proc.stdout.splitlines():
            print(f"  {line}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_main_with_custom_args():
    """Run main.py with a custom prompt and response via CLI args."""
    print_section("TEST 5: main.py accepts custom CLI arguments")

    main_path    = project_root / "main.py"
    custom_prompt   = "What is the capital of France?"
    custom_response = "The capital of France is Paris."

    try:
        proc = subprocess.run(
            [sys.executable, str(main_path), custom_prompt, custom_response],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        assert proc.returncode == 0, f"main.py failed:\n{proc.stderr}"
        assert custom_prompt[:20] in proc.stdout,   "Prompt not in output"
        assert custom_response[:20] in proc.stdout, "Response not in output"

        print(f"  ✓ main.py accepted custom prompt and response")
        print(f"  ✓ Both appear in printed output")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def main():
    print("\n" + "="*60)
    print("  REWARD SYSTEM - MILESTONE 5 TEST SUITE")
    print("  Testing Main Entry Point (End-to-End)")
    print("="*60)

    try:
        traits, result = test_full_pipeline()
        test_output_shape(traits, result)
        test_different_scores_give_different_rewards()
        test_main_runs()
        test_main_with_custom_args()

        print("\n" + "="*60)
        print("  ✓ ALL MILESTONE 5 TESTS PASSED")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
