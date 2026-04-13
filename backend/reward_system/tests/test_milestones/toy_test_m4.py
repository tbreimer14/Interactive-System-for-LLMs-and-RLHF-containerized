"""
Toy test for Milestone 4 (Logger).

Tests that log_reward() writes a valid JSONL entry with the full
trait definitions, user scores, weights, and final reward.

Run with: uv run python tests/test_milestones/toy_test_m4.py
"""

import sys
import json
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

TEST_LOG_PATH = project_root / "logs" / "test_reward_log.jsonl"

PROMPT   = "Can you help me understand reinforcement learning?"
RESPONSE = "Sure! RL is about learning from rewards. Want to go deeper?"

TRAITS = [
    {"name": "interactivity", "description": "Encourages continued conversation", "weight": 0.5},
    {"name": "warmth",        "description": "Sounds kind, supportive, and human", "weight": 0.3},
    {"name": "directness",    "description": "Gets to the point without being vague", "weight": 0.2},
]

USER_SCORES = {"interactivity": 5, "warmth": 4, "directness": 3}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_log_creates_file():
    """log_reward() should create the log file if it doesn't exist."""
    print_section("TEST 1: Log file is created")

    if TEST_LOG_PATH.exists():
        TEST_LOG_PATH.unlink()

    result = compute_reward(PROMPT, RESPONSE, TRAITS, USER_SCORES)

    try:
        log_reward(PROMPT, RESPONSE, TRAITS, result, log_path=str(TEST_LOG_PATH))
        assert TEST_LOG_PATH.exists(), "Log file was not created"
        print(f"  ✓ Log file created at: {TEST_LOG_PATH}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_log_entry_is_valid_json():
    """Each line in the log file should be valid JSON."""
    print_section("TEST 2: Log entry is valid JSON")

    try:
        with open(TEST_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) > 0, "Log file is empty"

        entry = None
        for i, line in enumerate(lines):
            entry = json.loads(line)
            print(f"  ✓ Line {i+1} is valid JSON")

        return entry
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_log_entry_fields(entry):
    """Log entry should have all required fields."""
    print_section("TEST 3: Log entry has required fields")

    required = {"timestamp", "prompt", "response", "traits", "user_scores", "weights", "final_reward"}

    try:
        missing = required - entry.keys()
        assert not missing, f"Missing fields: {missing}"
        print(f"  ✓ All required fields present: {required}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_log_entry_values(entry):
    """Log entry values should be correct types and match what was passed in."""
    print_section("TEST 4: Log entry values are correct")

    try:
        assert entry["prompt"] == PROMPT,     "Prompt mismatch"
        assert entry["response"] == RESPONSE, "Response mismatch"
        assert isinstance(entry["traits"], dict),      "traits should be a dict"
        assert isinstance(entry["user_scores"], dict), "user_scores should be a dict"
        assert isinstance(entry["weights"], dict),     "weights should be a dict"
        assert isinstance(entry["final_reward"], float), \
            f"final_reward should be float, got {type(entry['final_reward'])}"
        assert isinstance(entry["timestamp"], str), "timestamp should be a string"

        print(f"  ✓ prompt       matches")
        print(f"  ✓ response     matches")
        print(f"  ✓ traits       is a dict with {len(entry['traits'])} entries")
        print(f"  ✓ user_scores  {entry['user_scores']}")
        print(f"  ✓ weights      {entry['weights']}")
        print(f"  ✓ final_reward {entry['final_reward']}")
        print(f"  ✓ timestamp    {entry['timestamp']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_traits_snapshot_is_complete(entry):
    """traits in log should have description and weight for each trait."""
    print_section("TEST 5: traits snapshot is self-contained")

    try:
        for name, trait_data in entry["traits"].items():
            assert "description" in trait_data, f"Missing description for '{name}'"
            assert "weight" in trait_data,      f"Missing weight for '{name}'"
            print(f"  ✓ {name:<16}  description='{trait_data['description'][:30]}...'  weight={trait_data['weight']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_log_appends():
    """Calling log_reward() twice should produce two lines."""
    print_section("TEST 6: Log appends (does not overwrite)")

    result = compute_reward(PROMPT, RESPONSE, TRAITS, USER_SCORES)

    try:
        log_reward(PROMPT, RESPONSE, TRAITS, result, log_path=str(TEST_LOG_PATH))

        with open(TEST_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2, f"Expected 2 lines after 2 writes, got {len(lines)}"
        print(f"  ✓ Log has {len(lines)} lines after 2 writes — append is working")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def test_full_pipeline_log():
    """Log a reward computed from the real traits.json and print the entry."""
    print_section("TEST 7: Full pipeline log with real traits.json")

    config_path = project_root / "config" / "traits.json"

    try:
        traits = load_traits(str(config_path))
        user_scores = {t["name"]: 4 for t in traits}
        result = compute_reward(PROMPT, RESPONSE, traits, user_scores)
        log_reward(PROMPT, RESPONSE, traits, result, log_path=str(TEST_LOG_PATH))

        with open(TEST_LOG_PATH, "r", encoding="utf-8") as f:
            last_entry = json.loads(f.readlines()[-1])

        print(f"  Last log entry:")
        print(f"    timestamp:    {last_entry['timestamp']}")
        print(f"    prompt:       {last_entry['prompt'][:50]}...")
        print(f"    response:     {last_entry['response'][:50]}...")
        for name in last_entry["user_scores"]:
            score  = last_entry["user_scores"][name]
            weight = last_entry["weights"][name]
            print(f"    {name:<16}  score={score}  weight={weight}")
        print(f"    final_reward: {last_entry['final_reward']}")
        print(f"\n  ✓ Full pipeline log succeeded")
        print(f"  Log file: {TEST_LOG_PATH}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def main():
    print("\n" + "="*60)
    print("  REWARD SYSTEM - MILESTONE 4 TEST SUITE")
    print("  Testing Logger")
    print("="*60)

    try:
        test_log_creates_file()
        entry = test_log_entry_is_valid_json()
        test_log_entry_fields(entry)
        test_log_entry_values(entry)
        test_traits_snapshot_is_complete(entry)
        test_log_appends()
        test_full_pipeline_log()

        print("\n" + "="*60)
        print("  ✓ ALL MILESTONE 4 TESTS PASSED")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
