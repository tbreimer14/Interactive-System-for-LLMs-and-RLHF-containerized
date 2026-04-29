import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

from grpo_system.config import GRPOConfig


def load_model(config: GRPOConfig):
    """
    Load a Qwen2.5-Instruct model with LoRA adapters.

    Respects config.use_4bit:
      True  — 4-bit QLoRA (local 4GB GPU, e.g. GTX 1650)
      False — bfloat16 (Colab T4/A100 with enough VRAM)

    Returns:
        model     — Qwen2.5 with LoRA layers attached (only these are trained)
        tokenizer — tokenizer with left-padding configured
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_gpu = torch.cuda.is_available()

    if has_gpu and config.use_4bit:
        print("  Hardware: GPU (4-bit QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif has_gpu:
        print("  Hardware: GPU (bfloat16)")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print("  Hardware: CPU (float32)")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
