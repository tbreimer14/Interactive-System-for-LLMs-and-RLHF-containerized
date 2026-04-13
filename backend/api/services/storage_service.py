import json
import os

INTERACTIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "interactions.jsonl")


def _resolve_path() -> str:
    return os.path.abspath(INTERACTIONS_FILE)


def save_interaction(record: dict) -> None:
    path = _resolve_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_interactions(limit: int = 20) -> list[dict]:
    path = _resolve_path()
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip bad lines

    return records[-limit:]


def get_interaction_by_id(id: str) -> dict | None:
    path = _resolve_path()
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("id") == id:
                    return record
            except json.JSONDecodeError:
                continue

    return None
