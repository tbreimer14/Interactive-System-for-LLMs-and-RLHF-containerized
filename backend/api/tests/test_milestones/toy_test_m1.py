"""
Milestone 1 test — schemas.py

Run from the backend/api directory:
    uv run python tests/test_milestones/toy_test_m1.py
"""

import sys
import os

# allow imports from backend/api root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.schemas import (
    RetrievedChunk,
    GenerateRequest,
    GenerateResponse,
    TraitScoreInput,
    TraitScoreOutput,
    RewardRequest,
    RewardResponse,
    SaveInteractionRequest,
    InteractionRecord,
)

# --- RetrievedChunk ---
chunk = RetrievedChunk(id="chunk_1", source="intro.txt", text="RL is about rewards.")
assert chunk.id == "chunk_1"
assert chunk.source == "intro.txt"
assert chunk.text == "RL is about rewards."

chunk_no_source = RetrievedChunk(id="chunk_2", text="No source here.")
assert chunk_no_source.source is None

print("RetrievedChunk OK")

# --- GenerateRequest ---
req = GenerateRequest(prompt="What is RL?", top_k=3, show_prompt=False)
assert req.prompt == "What is RL?"
assert req.top_k == 3
assert req.show_prompt is False

req_defaults = GenerateRequest(prompt="Hello")
assert req_defaults.top_k == 3
assert req_defaults.show_prompt is False

print("GenerateRequest OK")

# --- GenerateResponse ---
resp = GenerateResponse(
    prompt="What is RL?",
    retrieved_chunks=[chunk],
    response="RL is a type of ML.",
    final_prompt="What is RL?",
)
assert resp.response == "RL is a type of ML."
assert len(resp.retrieved_chunks) == 1

print("GenerateResponse OK")

# --- TraitScoreInput / Output ---
trait_in = TraitScoreInput(name="clarity", weight=1.0, score=4.0)
assert trait_in.name == "clarity"

trait_out = TraitScoreOutput(name="clarity", weight=1.0, score=4.0, contribution=4.0)
assert trait_out.contribution == 4.0

print("TraitScore models OK")

# --- RewardRequest / Response ---
reward_req = RewardRequest(traits=[trait_in])
assert len(reward_req.traits) == 1

reward_resp = RewardResponse(traits=[trait_out], scalar_reward=4.0)
assert reward_resp.scalar_reward == 4.0

print("Reward models OK")

# --- SaveInteractionRequest ---
save_req = SaveInteractionRequest(
    prompt="What is RL?",
    retrieved_chunks=[chunk],
    response="RL is ML.",
    final_prompt="What is RL?",
    traits=[trait_out],
    scalar_reward=4.0,
)
assert save_req.scalar_reward == 4.0

print("SaveInteractionRequest OK")

# --- InteractionRecord ---
record = InteractionRecord(
    id="abc123",
    timestamp="2026-04-11T10:00:00",
    prompt="What is RL?",
    retrieved_chunks=[chunk],
    response="RL is ML.",
    final_prompt="What is RL?",
    traits=[trait_out],
    scalar_reward=4.0,
)
assert record.id == "abc123"

print("InteractionRecord OK")

print("\nMilestone 1 PASSED — all schemas work correctly.")
