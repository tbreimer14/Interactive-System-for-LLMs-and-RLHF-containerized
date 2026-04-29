"""Milestone 3: Model loading with LoRA

Stages:
  Stage A — imports and LoRA config structure (no download)
  Stage B — full model load (downloads ~6GB on first run, requires GPU)

Run Stage A only (fast, no download):
    uv run python tests/test_milestones/toy_test_m3.py

Run Stage B (slow, downloads model):
    uv run python tests/test_milestones/toy_test_m3.py --full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def test_imports():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType
    from grpo_system.model import load_model
    print("  imports: OK")


def test_lora_config_from_grpo_config():
    from peft import LoraConfig, TaskType
    from grpo_system.config import GRPOConfig

    cfg = GRPOConfig()
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    assert lora_config.r == 16
    assert lora_config.lora_alpha == 32
    assert lora_config.task_type == TaskType.CAUSAL_LM
    assert set(lora_config.target_modules) == {"q_proj", "v_proj", "k_proj", "o_proj"}
    print("  LoRA config from GRPOConfig: OK")


def test_load_model_full():
    from grpo_system.config import GRPOConfig
    from grpo_system.model import load_model

    print("\n  Loading Qwen2.5-3B-Instruct + LoRA (~6GB download on first run)...")
    cfg = GRPOConfig()
    model, tokenizer = load_model(cfg)

    # tokenizer checks
    assert tokenizer.pad_token is not None, "pad_token must be set"
    assert tokenizer.padding_side == "left", "padding_side must be left"
    print("  tokenizer: OK")

    # model is a PEFT model with LoRA
    from peft import PeftModel
    assert isinstance(model, PeftModel), "model must be a PeftModel"
    print("  model is PeftModel: OK")

    # trainable params should be a small fraction of total
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100
    assert pct < 5.0, f"LoRA should be <5% trainable, got {pct:.2f}%"
    print(f"  trainable params: {trainable:,} / {total:,} ({pct:.2f}%): OK")

    # generation smoke test
    import torch
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    assert out.shape[1] > inputs["input_ids"].shape[1], "model must generate new tokens"
    print("  generation smoke test: OK")


if __name__ == "__main__":
    full = "--full" in sys.argv

    print("=== Milestone 3: Model Loading with LoRA ===")
    test_imports()
    test_lora_config_from_grpo_config()

    if full:
        test_load_model_full()
    else:
        print("\n  [Stage B skipped] Run with --full to load the model (~6GB download)")

    print("\nMilestone 3 tests passed.")
