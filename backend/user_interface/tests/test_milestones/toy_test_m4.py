"""
Toy test for Milestone 4 (Interaction Logger).

Tests that interaction_logger.log_interaction() writes valid JSONL entries
and that read_log() can read them back.

Run with: uv run python tests/test_milestones/toy_test_m4.py
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

from app.interaction_logger import log_interaction, read_log

# Use a temp file so tests don't pollute the real log
TEMP_LOG = Path(tempfile.gettempdir()) / "ui_test_interaction_log.jsonl"

SAMPLE_ENTRY = {
    "prompt": "What is reinforcement learning?",
    "retrieved": [
        {"text": "RL is about agents learning from rewards.", "source": "doc_1.txt"},
        {"text": "PPO is a common RL algorithm.", "source": "doc_2.txt"},
    ],
    "response": "Reinforcement learning is a method where an agent learns by receiving rewards.",
    "traits": [
        {"name": "clarity",    "description": "Easy to read", "weight": 0.4},
        {"name": "empathy",    "description": "Sounds human",  "weight": 0.3},
        {"name": "directness", "description": "Gets to point", "weight": 0.3},
    ],
    "scores": {"clarity": 4, "empathy": 3, "directness": 5},
    "scalar_reward": 4.0,
}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def setup():
    """Remove temp log before tests."""
    if TEMP_LOG.exists():
        TEMP_LOG.unlink()


def test_log_creates_file():
    """Test that log_interaction creates the file if it doesn't exist."""
    print_section("TEST 1: Log creates file")

    try:
        log_interaction(SAMPLE_ENTRY, log_path=str(TEMP_LOG))
        assert TEMP_LOG.exists(), f"Log file was not created at {TEMP_LOG}"
        print(f"  [PASS] Log file created at: {TEMP_LOG}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_log_is_valid_json():
    """Test that the written line is valid JSON."""
    print_section("TEST 2: Written line is valid JSON")

    try:
        with open(TEMP_LOG, "r", encoding="utf-8") as f:
            line = f.readline().strip()

        parsed = json.loads(line)
        assert isinstance(parsed, dict), "Parsed line should be a dict"
        print(f"  [PASS] Line parses as a valid JSON dict")
        return parsed
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_required_fields(parsed):
    """Test that all required fields are present in the logged entry."""
    print_section("TEST 3: Required fields in log entry")

    required = {"timestamp", "prompt", "retrieved", "response", "traits", "scores", "scalar_reward"}

    try:
        missing = required - parsed.keys()
        assert not missing, f"Missing fields in log entry: {missing}"
        print(f"  [PASS] All required fields present: {required}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_values_roundtrip(parsed):
    """Test that the logged values match what was passed in."""
    print_section("TEST 4: Values round-trip correctly")

    try:
        assert parsed["prompt"] == SAMPLE_ENTRY["prompt"]
        assert parsed["response"] == SAMPLE_ENTRY["response"]
        assert parsed["scores"] == SAMPLE_ENTRY["scores"]
        assert abs(parsed["scalar_reward"] - SAMPLE_ENTRY["scalar_reward"]) < 1e-9
        print(f"  [PASS] prompt, response, scores, scalar_reward all match")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_timestamp_present(parsed):
    """Test that a timestamp was added automatically."""
    print_section("TEST 5: Timestamp added automatically")

    try:
        assert "timestamp" in parsed, "Timestamp missing from log entry"
        assert isinstance(parsed["timestamp"], str), "Timestamp should be a string"
        assert len(parsed["timestamp"]) > 0, "Timestamp should not be empty"
        print(f"  [PASS] Timestamp present: {parsed['timestamp']}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_append_multiple():
    """Test that multiple calls append new lines without overwriting."""
    print_section("TEST 6: Append multiple entries")

    try:
        log_interaction(SAMPLE_ENTRY, log_path=str(TEMP_LOG))
        log_interaction(SAMPLE_ENTRY, log_path=str(TEMP_LOG))

        with open(TEMP_LOG, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert len(lines) == 3, f"Expected 3 lines (1 from test 1 + 2 more), got {len(lines)}"
        print(f"  [PASS] {len(lines)} entries in log after 3 total writes")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_read_log():
    """Test that read_log() returns all entries as a list of dicts."""
    print_section("TEST 7: read_log() returns all entries")

    try:
        entries = read_log(log_path=str(TEMP_LOG))
        assert isinstance(entries, list), "read_log should return a list"
        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
        for entry in entries:
            assert isinstance(entry, dict), "Each entry should be a dict"
        print(f"  [PASS] read_log() returned {len(entries)} valid dicts")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_read_log_missing_file():
    """Test that read_log returns empty list when file doesn't exist."""
    print_section("TEST 8: read_log on missing file returns empty list")

    try:
        entries = read_log(log_path="logs/does_not_exist.jsonl")
        assert entries == [], f"Expected [], got {entries}"
        print(f"  [PASS] read_log() returned [] for missing file")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_print_entry(parsed):
    """Print one entry for visual confirmation."""
    print_section("TEST 9: Visual output")

    print(f"  timestamp:    {parsed['timestamp']}")
    print(f"  prompt:       {parsed['prompt'][:60]}")
    print(f"  response:     {parsed['response'][:60]}...")
    print(f"  scores:       {parsed['scores']}")
    print(f"  scalar_reward:{parsed['scalar_reward']}")
    print(f"  retrieved:    {len(parsed['retrieved'])} chunks")
    print(f"\n  [PASS] Entry looks correct above")


def main():
    print("\n" + "=" * 60)
    print("  USER INTERFACE - MILESTONE 4 TEST SUITE")
    print("  Testing Interaction Logger")
    print("=" * 60)

    setup()

    try:
        test_log_creates_file()
        parsed = test_log_is_valid_json()
        test_required_fields(parsed)
        test_values_roundtrip(parsed)
        test_timestamp_present(parsed)
        test_append_multiple()
        test_read_log()
        test_read_log_missing_file()
        test_print_entry(parsed)

        print("\n" + "=" * 60)
        print("  ALL MILESTONE 4 TESTS PASSED")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  TEST FAILED: {e}")
        print("=" * 60 + "\n")
        raise


if __name__ == "__main__":
    main()
