"""
trait_manager.py

Loads user-defined reward traits from a JSON config file.

Same format as reward_system/config/traits.json — the two systems share
a config convention so traits can be transferred between them directly.

Swap this file if you want to load traits from a database, API, or UI state
instead of a static JSON file.
"""

import json
from pathlib import Path


def load_traits(config_path: str) -> list[dict]:
    """
    Load traits from a JSON file and return them as a list of dicts.

    Each trait dict has:
        - name (str):        short identifier, e.g. "clarity"
        - description (str): what the trait means, shown in the UI
        - weight (float):    contribution to the scalar reward

    Args:
        config_path: path to traits.json

    Returns:
        list of trait dicts, e.g.:
        [
            {"name": "clarity", "description": "...", "weight": 0.4},
            ...
        ]

    Raises:
        FileNotFoundError: if config_path does not exist
        ValueError:        if the JSON is missing required fields
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Traits config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "traits" not in data:
        raise ValueError("traits.json must have a top-level 'traits' key")

    traits = data["traits"]

    required_fields = {"name", "description", "weight"}
    for i, trait in enumerate(traits):
        missing = required_fields - trait.keys()
        if missing:
            raise ValueError(f"Trait at index {i} is missing fields: {missing}")

    return traits
