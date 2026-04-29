"""
ui/types.py

Lightweight data models for the UI layer.

Using dataclasses keeps the models simple and serializable without
pulling in heavy dependencies. Each model maps directly to a section
of the JSONL interaction log format.
"""

from dataclasses import dataclass


@dataclass
class TraitConfig:
    """A reward trait as defined by the user — name, description, weight."""
    name: str
    weight: float
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "weight": self.weight, "description": self.description}

    @staticmethod
    def from_dict(d: dict) -> "TraitConfig":
        return TraitConfig(
            name=d["name"],
            weight=d["weight"],
            description=d.get("description", ""),
        )


@dataclass
class ScoredTrait:
    """A trait after the user has scored the response on it."""
    name: str
    weight: float
    score: float
    contribution: float  # weight * score

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "weight":       self.weight,
            "score":        self.score,
            "contribution": self.contribution,
        }


@dataclass
class InteractionLog:
    """One complete scored interaction, ready to be written to JSONL."""
    timestamp: str
    prompt: str
    response: str
    traits: list       # list of ScoredTrait.to_dict()
    scalar_reward: float

    def to_dict(self) -> dict:
        return {
            "timestamp":     self.timestamp,
            "prompt":        self.prompt,
            "response":      self.response,
            "traits":        self.traits,
            "scalar_reward": self.scalar_reward,
        }

    @staticmethod
    def from_dict(d: dict) -> "InteractionLog":
        return InteractionLog(
            timestamp=d["timestamp"],
            prompt=d["prompt"],
            response=d["response"],
            traits=d.get("traits", []),
            scalar_reward=d["scalar_reward"],
        )
