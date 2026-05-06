"""
grpo_system/online_step.py

Single GRPO gradient step on human-scored completion groups.

Uses disable_adapter_layers() / enable_adapter_layers() so the same LoRA model
serves as both policy (adapters on) and frozen reference (adapters off).
No second model copy is needed.
"""

import torch
import torch.nn.functional as F


def _seq_log_prob(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """Sum of log P(token_i | prefix) over all completion tokens."""
    full_text = prompt + completion
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc.input_ids.to(model.device)

    prompt_len = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).input_ids.shape[1]

    outputs = model(input_ids=input_ids)

    # logit at position i predicts token i+1
    logits  = outputs.logits[0, prompt_len - 1 : -1]  # (comp_len, vocab)
    targets = input_ids[0, prompt_len:]                # (comp_len,)

    if targets.numel() == 0:
        zero = torch.zeros(1, device=model.device)
        return zero.squeeze().requires_grad_(True)

    return F.log_softmax(logits, dim=-1)[range(len(targets)), targets].sum()


def grpo_step(
    model,
    tokenizer,
    groups: list[dict],
    optimizer: torch.optim.Optimizer,
    beta: float = 0.1,
) -> float:
    """
    One GRPO update on a batch of human-graded groups.

    Args:
        model:     PeftModel (QLoRA) — updated in place
        tokenizer: paired tokenizer
        groups:    list of {
                       prompt: str,
                       completions: [{"text": str, "reward": float}, ...]
                   }
        optimizer: AdamW over LoRA parameters — persistent across steps
        beta:      KL penalty weight

    Returns:
        mean loss (float) for logging
    """
    model.train()
    optimizer.zero_grad()

    loss_terms: list[torch.Tensor] = []

    for group in groups:
        prompt = group["prompt"]
        comps  = group["completions"]
        if len(comps) < 2:
            continue

        rewards = torch.tensor(
            [c["reward"] for c in comps], dtype=torch.float32, device=model.device
        )
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for comp, adv in zip(comps, advantages.tolist()):
            text = comp["text"]

            # Policy log-prob (LoRA active — gradients flow)
            lp = _seq_log_prob(model, tokenizer, prompt, text)

            # Reference log-prob (base model — no gradients)
            model.disable_adapter_layers()
            with torch.no_grad():
                ref_lp = _seq_log_prob(model, tokenizer, prompt, text)
            model.enable_adapter_layers()

            kl = lp - ref_lp
            loss_terms.append(-adv * lp + beta * kl)

    if not loss_terms:
        return 0.0

    total = torch.stack(loss_terms).mean()
    total.backward()
    optimizer.step()

    return total.item()
