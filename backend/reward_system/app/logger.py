"""
logger.py

Persists each reward computation to a newline-delimited JSON log file
(logs/reward_log.jsonl).

Each entry captures everything needed to reconstruct the event:
the prompt, response, trait definitions, user scores, weights, and final reward.
This makes the log directly usable as a preference dataset for PPO later.

Why JSONL (one JSON object per line):
    - Append-safe: each write is atomic, no risk of corrupting prior entries
    - Easy to parse: read line-by-line without loading the full file
    - Training-ready: stream directly into a dataset loader
"""

import json
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_PATH = "logs/reward_log.jsonl"


def log_reward(
    prompt: str,
    response: str,
    traits: list[dict],
    result: dict,
    log_path: str = DEFAULT_LOG_PATH,
) -> None:
    """
    Append one reward event to the JSONL log file.

    Creates the logs/ directory automatically if it does not exist.

    Args:
        prompt:    the original user prompt
        response:  the model response that was scored
        traits:    list of trait dicts from load_traits()
        result:    the RewardResult dict from compute_reward()
        log_path:  path to the .jsonl file (default: logs/reward_log.jsonl)
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Store full trait definitions so the log is self-contained
    traits_snapshot = {
        t["name"]: {
            "description": t["description"],
            "weight": t["weight"],
        }
        for t in traits
    }

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt,
        "response": response,
        "traits": traits_snapshot,
        "user_scores": result["trait_scores"],
        "weights": result["weights"],
        "final_reward": result["final_reward"],
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
