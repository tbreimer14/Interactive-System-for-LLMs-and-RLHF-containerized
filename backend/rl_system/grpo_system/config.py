from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")

    # GRPO
    learning_rate: float = 1e-5
    num_generations: int = 2
    max_prompt_length: int = 256
    max_completion_length: int = 128
    batch_size: int = 1
    num_train_epochs: int = 1
    kl_coef: float = 0.1

    # Output
    output_dir: str = "grpo_output"


def local_config() -> GRPOConfig:
    """GTX 1650 / 4GB VRAM laptop. 1.5B model with 4-bit QLoRA."""
    return GRPOConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        use_4bit=True,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=128,
        batch_size=1,
    )


def colab_config() -> GRPOConfig:
    """Google Colab T4 / 15GB VRAM. 3B model in bfloat16, full GRPO settings."""
    return GRPOConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        use_4bit=False,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=256,
        batch_size=2,
    )
