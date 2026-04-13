"""Milestone 5 — toy tests for the main training loop."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo_system.config import PPOConfig
from ppo_system.main import train


def _fast_config():
    """Minimal config so tests run quickly — 1 epoch, batch of 4."""
    return PPOConfig(ppo_epochs=1, batch_size=4)


def test_train_returns_stats():
    stats = train(_fast_config())
    assert isinstance(stats, list), "train() should return a list"
    assert len(stats) > 0, "stats list should not be empty"
    print("PASS test_train_returns_stats")


def test_each_stat_is_two_floats():
    stats = train(_fast_config())
    for i, (loss, mean_reward) in enumerate(stats):
        assert isinstance(loss, float),       f"Step {i} loss is not float"
        assert isinstance(mean_reward, float), f"Step {i} mean_reward is not float"
    print("PASS test_each_stat_is_two_floats")


def test_loss_is_finite():
    stats = train(_fast_config())
    for i, (loss, _) in enumerate(stats):
        assert loss == loss,     f"Step {i} loss is NaN"
        assert abs(loss) < 1e6, f"Step {i} loss is unreasonably large: {loss}"
    print("PASS test_loss_is_finite")


def test_mean_reward_in_reasonable_range():
    stats = train(_fast_config())
    for i, (_, mean_reward) in enumerate(stats):
        assert 0.0 <= mean_reward <= 10.0, f"Step {i} mean_reward out of range: {mean_reward}"
    print("PASS test_mean_reward_in_reasonable_range")


if __name__ == "__main__":
    test_train_returns_stats()
    test_each_stat_is_two_floats()
    test_loss_is_finite()
    test_mean_reward_in_reasonable_range()
    print("\nAll M5 tests passed.")
