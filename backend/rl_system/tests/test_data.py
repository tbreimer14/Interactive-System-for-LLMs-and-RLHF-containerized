"""Milestone 2 — toy tests for data.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo_system.data import load_data, validate_batch


def test_load_data_returns_list():
    data = load_data()
    assert isinstance(data, list)
    assert len(data) > 0
    print("PASS test_load_data_returns_list")


def test_all_records_have_required_keys():
    data = load_data()
    required = {"prompt", "response", "reward"}
    for i, record in enumerate(data):
        missing = required - record.keys()
        assert not missing, f"Record {i} missing keys: {missing}"
    print("PASS test_all_records_have_required_keys")


def test_all_rewards_are_floats():
    data = load_data()
    for i, record in enumerate(data):
        assert isinstance(record["reward"], (int, float)), f"Record {i} reward is not a number"
    print("PASS test_all_rewards_are_floats")


def test_validate_batch_passes_on_good_data():
    validate_batch(load_data())  # should not raise
    print("PASS test_validate_batch_passes_on_good_data")


def test_validate_batch_catches_missing_key():
    bad = [{"prompt": "hello", "reward": 1.0}]  # missing 'response'
    try:
        validate_batch(bad)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS test_validate_batch_catches_missing_key")


def test_validate_batch_catches_bad_reward_type():
    bad = [{"prompt": "hello", "response": "hi", "reward": "good"}]
    try:
        validate_batch(bad)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS test_validate_batch_catches_bad_reward_type")


if __name__ == "__main__":
    test_load_data_returns_list()
    test_all_records_have_required_keys()
    test_all_rewards_are_floats()
    test_validate_batch_passes_on_good_data()
    test_validate_batch_catches_missing_key()
    test_validate_batch_catches_bad_reward_type()
    print("\nAll M2 tests passed.")
