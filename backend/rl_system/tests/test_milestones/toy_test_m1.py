"""Milestone 1: GRPOConfig dataclass"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from grpo_system.config import GRPOConfig


def test_defaults():
    cfg = GRPOConfig()
    assert cfg.model_name == "Qwen/Qwen2.5-3B-Instruct", "model_name wrong"
    assert cfg.lora_r == 16, "lora_r wrong"
    assert cfg.lora_alpha == 32, "lora_alpha wrong"
    assert cfg.lora_alpha == 2 * cfg.lora_r, "lora_alpha should be 2x lora_r"
    assert cfg.lora_dropout == 0.05, "lora_dropout wrong"
    assert set(cfg.lora_target_modules) == {"q_proj", "v_proj", "k_proj", "o_proj"}, "target modules wrong"
    assert cfg.learning_rate == 1e-5, "learning_rate wrong"
    assert cfg.num_generations == 4, "num_generations wrong"
    assert cfg.max_prompt_length == 512, "max_prompt_length wrong"
    assert cfg.max_completion_length == 256, "max_completion_length wrong"
    assert cfg.batch_size == 2, "batch_size wrong"
    assert cfg.num_train_epochs == 1, "num_train_epochs wrong"
    assert cfg.kl_coef == 0.1, "kl_coef wrong"
    assert cfg.output_dir == "grpo_output", "output_dir wrong"
    print("  defaults: OK")


def test_override():
    cfg = GRPOConfig(batch_size=1, lora_r=8, output_dir="my_run")
    assert cfg.batch_size == 1
    assert cfg.lora_r == 8
    assert cfg.output_dir == "my_run"
    # unset fields stay at default
    assert cfg.num_generations == 4
    print("  override: OK")


def test_target_modules_iterable():
    cfg = GRPOConfig()
    modules = list(cfg.lora_target_modules)
    assert len(modules) == 4
    print("  target_modules iterable: OK")


if __name__ == "__main__":
    print("=== Milestone 1: Config ===")
    test_defaults()
    test_override()
    test_target_modules_iterable()
    print("\nAll Milestone 1 tests passed.")
