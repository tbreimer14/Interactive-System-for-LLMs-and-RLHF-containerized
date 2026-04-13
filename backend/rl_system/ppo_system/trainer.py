"""Milestone 3 — model and tokenizer loading.
Milestone 4 — manual PPO training step.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from ppo_system.config import PPOConfig


def load_model(config: PPOConfig):
    """Load model, frozen reference model, and tokenizer.

    Returns:
        model       — trainable causal LM
        ref_model   — frozen copy for KL divergence computation
        tokenizer   — tokenizer with pad token set
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # distilgpt2 has no pad token by default — set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"Loaded model      : {config.model_name}")
    print(f"Pad token         : {tokenizer.pad_token!r}")
    print(f"Ref model frozen  : {not any(p.requires_grad for p in ref_model.parameters())}")

    return model, ref_model, tokenizer


def _get_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probs for the tokens that were actually generated.

    logits   : (batch, seq_len, vocab)
    input_ids: (batch, seq_len)
    Returns  : (batch,) — mean log prob over sequence for each item
    """
    # shift so token[i] predicts token[i+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.mean(dim=-1)  # (batch,)


def run_ppo_step(model, ref_model, tokenizer, batch, optimizer, config: PPOConfig):
    """Run one PPO update step over a batch of {prompt, response, reward} records.

    Steps:
        1. Tokenize prompt + response
        2. Compute log probs from trainable model (new policy)
        3. Compute log probs from frozen ref model (old policy)
        4. Normalize rewards as advantages
        5. Compute clipped PPO surrogate loss + KL penalty
        6. Backprop and optimizer step

    Returns:
        loss        — scalar float
        mean_reward — mean reward across the batch
    """
    texts = [r["prompt"] + " " + r["response"] for r in batch]
    rewards = torch.tensor([r["reward"] for r in batch], dtype=torch.float32)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # --- new policy log probs ---
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    new_log_probs = _get_log_probs(outputs.logits, input_ids)  # (batch,)

    # --- old policy log probs (no grad) ---
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
        old_log_probs = _get_log_probs(ref_outputs.logits, input_ids)  # (batch,)

    # --- normalize rewards as advantages ---
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # --- clipped PPO surrogate loss ---
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # --- KL penalty (keeps new policy close to ref) ---
    kl = (old_log_probs - new_log_probs).mean()
    loss = ppo_loss + config.kl_coef * kl

    # --- update ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()
