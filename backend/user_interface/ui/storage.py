"""
ui/storage.py

JSONL storage helpers for the UI layer.

Functions:
    save_interaction(entry)    -- append one InteractionLog to JSONL
    load_interactions()        -- read all entries back as a list of dicts

Why JSONL:
    - append-safe (one JSON object per line)
    - no file corruption on partial writes
    - directly streamable into a training dataset loader later
"""

import json
from pathlib import Path
from datetime import datetime

from ui.types import InteractionLog

DEFAULT_LOG_PATH = "logs/interaction_log.jsonl"


def save_interaction(entry: InteractionLog, log_path: str = DEFAULT_LOG_PATH) -> None:
    """
    Append one InteractionLog to the JSONL file.

    Creates logs/ directory automatically if it does not exist.

    Args:
        entry:    an InteractionLog instance (timestamp is set automatically
                  if entry.timestamp is empty)
        log_path: path to the .jsonl file
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    record = entry.to_dict()
    if not record.get("timestamp"):
        record["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_interactions(log_path: str = DEFAULT_LOG_PATH) -> list[dict]:
    """
    Read all entries from the JSONL file, oldest first.

    Returns an empty list if the file does not exist or is empty.

    Args:
        log_path: path to the .jsonl file

    Returns:
        list of raw dicts (not InteractionLog instances, to keep it simple
        and avoid breaking on schema changes between versions)
    """
    log_file = Path(log_path)

    if not log_file.exists():
        return []

    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip malformed lines silently

    return entries
