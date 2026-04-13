"""Milestone 1 — toy tests for PPOConfig."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo_system.config import PPOConfig


def test_default_values():
    cfg = PPOConfig()
    assert cfg.model_name == "distilgpt2"
    assert cfg.learning_rate == 1.41e-5
    assert cfg.batch_size == 4
    assert cfg.mini_batch_size == 2
    assert cfg.ppo_epochs == 1
    print("PASS test_default_values")


def test_override_values():
    cfg = PPOConfig(model_name="gpt2", batch_size=8)
    assert cfg.model_name == "gpt2"
    assert cfg.batch_size == 8
    # unchanged fields keep defaults
    assert cfg.ppo_epochs == 1
    print("PASS test_override_values")


def test_repr_contains_model_name():
    cfg = PPOConfig()
    assert "distilgpt2" in repr(cfg)
    print("PASS test_repr_contains_model_name")


if __name__ == "__main__":
    test_default_values()
    test_override_values()
    test_repr_contains_model_name()
    print("\nAll M1 tests passed.")
