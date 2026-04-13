"""Milestone 2 — sample dataset for local PPO prototype.

Each record represents one (prompt, response, scalar_reward) tuple.
In the full pipeline this data comes from the generation + reward system.
For now it is hardcoded for local testing.
"""

from typing import List, Dict, Any


SAMPLE_DATA: List[Dict[str, Any]] = [
    {
        "prompt": "AITA for ignoring my friend after they cancelled last minute?",
        "response": "It's understandable to feel hurt. Try talking to them calmly about how their cancellation affected you.",
        "reward": 4.2,
    },
    {
        "prompt": "AITA for not wanting to share my food with my roommate?",
        "response": "Your food is yours. Setting boundaries around personal belongings is completely reasonable.",
        "reward": 3.8,
    },
    {
        "prompt": "AITA for telling my coworker their idea was bad?",
        "response": "Honest feedback is valuable, but delivery matters. Consider framing it constructively next time.",
        "reward": 3.5,
    },
    {
        "prompt": "AITA for skipping my cousin's wedding to study for finals?",
        "response": "Your education is a priority. Most family members will understand if you explain the situation honestly.",
        "reward": 4.0,
    },
    {
        "prompt": "AITA for asking my parents to knock before entering my room?",
        "response": "Asking for basic privacy is healthy and completely normal, especially as you get older.",
        "reward": 4.5,
    },
    {
        "prompt": "AITA for returning a gift I didn't like?",
        "response": "Returning a gift is a personal choice. If it wasn't useful to you, exchanging it is practical.",
        "reward": 3.2,
    },
    {
        "prompt": "AITA for not lending money to my sibling again?",
        "response": "Protecting your finances is valid. It's okay to say no, especially if past loans weren't repaid.",
        "reward": 4.3,
    },
    {
        "prompt": "AITA for correcting my boss in a meeting?",
        "response": "Accuracy matters, but timing and tone are key. A private correction is often better received.",
        "reward": 3.6,
    },
]


def load_data() -> List[Dict[str, Any]]:
    """Return the sample training dataset."""
    return SAMPLE_DATA


def validate_batch(batch: List[Dict[str, Any]]) -> None:
    """Raise ValueError if any record is missing required keys or has a bad reward type."""
    required_keys = {"prompt", "response", "reward"}
    for i, record in enumerate(batch):
        missing = required_keys - record.keys()
        if missing:
            raise ValueError(f"Record {i} missing keys: {missing}")
        if not isinstance(record["reward"], (int, float)):
            raise ValueError(f"Record {i} reward must be a number, got {type(record['reward'])}")
