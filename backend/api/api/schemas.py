from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: str
    source: Optional[str] = None
    text: str


class GenerateRequest(BaseModel):
    prompt: str
    top_k: int = 3
    show_prompt: bool = False


class GenerateResponse(BaseModel):
    prompt: str
    retrieved_chunks: list[RetrievedChunk]
    response: str
    final_prompt: str


class TraitScoreInput(BaseModel):
    name: str
    weight: float
    score: float


class TraitScoreOutput(BaseModel):
    name: str
    weight: float
    score: float
    contribution: float


class RewardRequest(BaseModel):
    traits: list[TraitScoreInput]


class RewardResponse(BaseModel):
    traits: list[TraitScoreOutput]
    scalar_reward: float


class SaveInteractionRequest(BaseModel):
    prompt: str
    retrieved_chunks: list[RetrievedChunk]
    response: str
    final_prompt: str
    traits: list[TraitScoreOutput]
    scalar_reward: float


class InteractionRecord(BaseModel):
    id: str
    timestamp: str
    prompt: str
    retrieved_chunks: list[RetrievedChunk]
    response: str
    final_prompt: str
    traits: list[TraitScoreOutput]
    scalar_reward: float
