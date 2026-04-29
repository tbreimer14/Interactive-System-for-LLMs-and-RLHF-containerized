"""
Main GRPO training loop.

Wires together:
    - Qwen2.5-3B-Instruct + LoRA     (model.py)
    - Newsgroup article prompts       (data.py)
    - Human slider reward function    (reward.py)
    - GRPOTrainer                     (trl)
"""

import torch
from datasets import Dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

from grpo_system.config import GRPOConfig
from grpo_system.model import load_model
from grpo_system.data import load_articles, format_for_qwen
from grpo_system.reward import build_reward_fn


def get_scores_from_ui(prompt: str, response: str) -> dict:
    """
    Placeholder for UI slider input.
    Replace with a real call to the UI score collector before running training.

    Expected return format matches traits.json keys:
        {"interactivity": float, "warmth": float, "directness": float}
    """
    raise NotImplementedError(
        "Connect UI slider input here.\n"
        "Example: return ui_client.collect_scores(prompt, response)"
    )


def train(config: GRPOConfig = None, get_scores_fn=None):
    """
    Run the full GRPO training loop.

    Args:
        config:        GRPOConfig instance (uses defaults if None)
        get_scores_fn: callable(prompt, response) -> dict[str, float]
                       Collects slider scores from the UI.
                       Defaults to get_scores_from_ui (raises NotImplementedError).
    """
    if config is None:
        config = GRPOConfig()
    if get_scores_fn is None:
        get_scores_fn = get_scores_from_ui

    print("=== GRPO Training — Qwen2.5-3B-Instruct + LoRA ===")

    # 1. Load model
    model, tokenizer = load_model(config)

    # 2. Load and format prompts
    raw_prompts = load_articles(max_articles=200)
    formatted = [format_for_qwen(p, tokenizer) for p in raw_prompts]
    dataset = Dataset.from_dict({"prompt": formatted})
    print(f"Dataset: {len(dataset)} prompts")

    # 3. Build reward function
    reward_fn = build_reward_fn(get_scores_fn)

    # 4. Configure TRL's GRPOTrainer
    has_gpu = torch.cuda.is_available()
    trl_config = TRLGRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        beta=config.kl_coef,        # kl_coef is called beta in this TRL version
        bf16=has_gpu,               # bf16 requires GPU; CPU uses float32
        use_cpu=not has_gpu,
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # 5. Train
    trainer.train()

    # 6. Save LoRA adapters (~80MB) and tokenizer
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"\nLoRA adapters saved to: {config.output_dir}/")


if __name__ == "__main__":
    train()
