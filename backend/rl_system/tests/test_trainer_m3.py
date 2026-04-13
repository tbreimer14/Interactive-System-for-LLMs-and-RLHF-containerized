"""Milestone 3 — toy tests for model + tokenizer loading."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo_system.config import PPOConfig
from ppo_system.trainer import load_model


def test_load_model_returns_three_items():
    result = load_model(PPOConfig())
    assert len(result) == 3, f"Expected 3 items, got {len(result)}"
    print("PASS test_load_model_returns_three_items")


def test_model_and_ref_model_not_none():
    model, ref_model, tokenizer = load_model(PPOConfig())
    assert model is not None
    assert ref_model is not None
    print("PASS test_model_and_ref_model_not_none")


def test_tokenizer_has_pad_token():
    _, _, tokenizer = load_model(PPOConfig())
    assert tokenizer.pad_token is not None, "Tokenizer has no pad token"
    print("PASS test_tokenizer_has_pad_token")


def test_ref_model_is_frozen():
    _, ref_model, _ = load_model(PPOConfig())
    trainable = [p for p in ref_model.parameters() if p.requires_grad]
    assert len(trainable) == 0, "Ref model should have no trainable parameters"
    print("PASS test_ref_model_is_frozen")


def test_model_has_trainable_params():
    model, _, _ = load_model(PPOConfig())
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "Main model should have trainable parameters"
    print("PASS test_model_has_trainable_params")


if __name__ == "__main__":
    test_load_model_returns_three_items()
    test_model_and_ref_model_not_none()
    test_tokenizer_has_pad_token()
    test_ref_model_is_frozen()
    test_model_has_trainable_params()
    print("\nAll M3 tests passed.")
