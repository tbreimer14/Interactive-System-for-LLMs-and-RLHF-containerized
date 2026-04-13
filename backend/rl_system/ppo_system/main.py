"""Milestone 5 — main training loop.

Wires together config, data, model loading, and PPO updates.
Run from rl_system/:
    python ppo_system/main.py
"""

import torch
from ppo_system.config import PPOConfig
from ppo_system.data import load_data, validate_batch
from ppo_system.trainer import load_model, run_ppo_step


def batch_iter(data, batch_size):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def train(config: PPOConfig = None):
    """Run the full PPO training loop.

    Returns a list of (loss, mean_reward) per step for testing.
    """
    if config is None:
        config = PPOConfig()

    print("=== PPO Training ===")
    print(config)
    print()

    data = load_data()
    validate_batch(data)

    model, ref_model, tokenizer = load_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    stats = []

    for epoch in range(1, config.ppo_epochs + 1):
        for step, batch in enumerate(batch_iter(data, config.batch_size), start=1):
            loss, mean_reward = run_ppo_step(
                model, ref_model, tokenizer, batch, optimizer, config
            )
            stats.append((loss, mean_reward))
            print(f"Epoch {epoch} | Step {step} | loss: {loss:.4f} | mean_reward: {mean_reward:.4f}")

    print()
    print("Training complete.")
    return stats


if __name__ == "__main__":
    train()
