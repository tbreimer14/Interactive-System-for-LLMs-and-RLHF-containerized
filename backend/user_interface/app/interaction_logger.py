"""
interaction_logger.py

Persists each scored interaction to a newline-delimited JSON log file
(logs/interaction_log.jsonl).

Why JSONL (one JSON object per line):
    - Append-safe: each write is atomic, no risk of corrupting prior entries
    - Easy to parse: read line-by-line, no need to load the entire file
    - Ready for training: can be streamed directly into a dataset loader later

Log entry format:
    {
      "timestamp":    "2026-04-10T14:32:01",
      "prompt":       "What is reinforcement learning?",
      "retrieved":    [{"text": "...", "source": "doc_3.txt"}, ...],
      "response":     "Reinforcement learning is...",
      "traits":       [{"name": "clarity", "description": "...", "weight": 0.4}, ...],
      "scores":       {"clarity": 4, "empathy": 3, "directness": 5},
      "scalar_reward": 4.0
    }
"""

import json
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_PATH = "logs/interaction_log.jsonl"


def log_interaction(entry: dict, log_path: str = DEFAULT_LOG_PATH) -> None:
    """
    Append one full interaction to the JSONL log file.

    Creates the logs/ directory automatically if it does not exist.

    Args:
        entry:    dict containing the full interaction. Expected keys:
                      prompt, retrieved, response, traits, scores, scalar_reward
                  A timestamp is added automatically if not already present.
        log_path: path to the .jsonl file (default: logs/interaction_log.jsonl)
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    record = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    record.update(entry)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_log(log_path: str = DEFAULT_LOG_PATH) -> list[dict]:
    """
    Read all entries from the JSONL log file and return them as a list.

    Returns an empty list if the file does not exist yet.

    Args:
        log_path: path to the .jsonl file

    Returns:
        list of interaction dicts, oldest first
    """
    log_file = Path(log_path)

    if not log_file.exists():
        return []

    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries
