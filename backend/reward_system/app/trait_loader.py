"""
trait_loader.py

Loads user-defined reward traits from a JSON config file.
This is the only place in the system that reads the config —
swap this file if you want to load traits from a database, API, or user profile.
"""

import json
from pathlib import Path


def load_traits(config_path: str) -> list[dict]:
    """
    Load traits from a JSON file and return them as a list of dicts.

    Each trait dict has:
        - name (str):        short identifier, e.g. "warmth"
        - description (str): what the trait means, used by the scorer
        - weight (float):    contribution to the final reward (should sum to 1.0)

    Args:
        config_path: path to traits.json

    Returns:
        list of trait dicts, e.g.:
        [
            {"name": "warmth", "description": "...", "weight": 0.3},
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

    # Validate top-level structure
    if "traits" not in data:
        raise ValueError("traits.json must have a top-level 'traits' key")

    traits = data["traits"]

    # Validate each trait has the required fields
    required_fields = {"name", "description", "weight"}
    for i, trait in enumerate(traits):
        missing = required_fields - trait.keys()
        if missing:
            raise ValueError(f"Trait at index {i} is missing fields: {missing}")

    return traits
