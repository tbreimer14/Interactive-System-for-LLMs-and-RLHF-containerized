"""Milestone 4 — toy tests for manual PPO training step."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from ppo_system.config import PPOConfig
from ppo_system.trainer import load_model, run_ppo_step


MINI_BATCH = [
    {"prompt": "AITA for ignoring my friend?", "response": "Talk to them calmly.", "reward": 4.0},
    {"prompt": "AITA for skipping dinner?",    "response": "Explain your reasons.",  "reward": 3.5},
]


def _setup():
    config = PPOConfig()
    model, ref_model, tokenizer = load_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, ref_model, tokenizer, optimizer, config


def test_ppo_step_returns_two_values():
    model, ref_model, tokenizer, optimizer, config = _setup()
    result = run_ppo_step(model, ref_model, tokenizer, MINI_BATCH, optimizer, config)
    assert len(result) == 2, f"Expected (loss, mean_reward), got {result}"
    print("PASS test_ppo_step_returns_two_values")


def test_loss_is_finite_float():
    model, ref_model, tokenizer, optimizer, config = _setup()
    loss, _ = run_ppo_step(model, ref_model, tokenizer, MINI_BATCH, optimizer, config)
    assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
    assert not (loss != loss), "Loss is NaN"
    assert abs(loss) < 1e6,   "Loss is unreasonably large"
    print("PASS test_loss_is_finite_float")


def test_mean_reward_matches_batch():
    model, ref_model, tokenizer, optimizer, config = _setup()
    _, mean_reward = run_ppo_step(model, ref_model, tokenizer, MINI_BATCH, optimizer, config)
    expected = (4.0 + 3.5) / 2
    assert abs(mean_reward - expected) < 1e-4, f"Expected {expected}, got {mean_reward}"
    print("PASS test_mean_reward_matches_batch")


def test_model_params_change_after_step():
    model, ref_model, tokenizer, optimizer, config = _setup()
    before = [p.clone() for p in model.parameters()]
    run_ppo_step(model, ref_model, tokenizer, MINI_BATCH, optimizer, config)
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "Model parameters did not update after PPO step"
    print("PASS test_model_params_change_after_step")


def test_ref_model_params_unchanged_after_step():
    model, ref_model, tokenizer, optimizer, config = _setup()
    before = [p.clone() for p in ref_model.parameters()]
    run_ppo_step(model, ref_model, tokenizer, MINI_BATCH, optimizer, config)
    after = list(ref_model.parameters())
    unchanged = all(torch.equal(b, a) for b, a in zip(before, after))
    assert unchanged, "Ref model parameters changed — it should be frozen"
    print("PASS test_ref_model_params_unchanged_after_step")


if __name__ == "__main__":
    test_ppo_step_returns_two_values()
    test_loss_is_finite_float()
    test_mean_reward_matches_batch()
    test_model_params_change_after_step()
    test_ref_model_params_unchanged_after_step()
    print("\nAll M4 tests passed.")
