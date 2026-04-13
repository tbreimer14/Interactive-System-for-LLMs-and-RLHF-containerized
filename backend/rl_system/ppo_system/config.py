from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Model
    model_name: str = "distilgpt2"

    # PPO hyperparameters
    learning_rate: float = 1.41e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 1
    clip_epsilon: float = 0.2   # clipping range for PPO surrogate loss
    kl_coef: float = 0.1        # penalty weight keeping new policy close to ref

    def __repr__(self):
        return (
            f"PPOConfig(\n"
            f"  model_name     = {self.model_name}\n"
            f"  learning_rate  = {self.learning_rate}\n"
            f"  batch_size     = {self.batch_size}\n"
            f"  mini_batch_size= {self.mini_batch_size}\n"
            f"  ppo_epochs     = {self.ppo_epochs}\n"
            f"  clip_epsilon   = {self.clip_epsilon}\n"
            f"  kl_coef        = {self.kl_coef}\n"
            f")"
        )
