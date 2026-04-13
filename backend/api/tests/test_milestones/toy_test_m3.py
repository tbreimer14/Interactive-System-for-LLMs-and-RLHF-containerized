"""
Milestone 3 test — backend_adapter.py + generation_service.py

Run from the backend/api directory:
    uv run python tests/test_milestones/toy_test_m3.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend_adapter import BackendAdapter
from api.schemas import GenerateRequest
from services.generation_service import run_generation

# --- BackendAdapter.generate() ---
adapter = BackendAdapter()

result = adapter.generate("What is RL?", top_k=3)
assert "retrieved_chunks" in result
assert "response" in result
assert "final_prompt" in result
assert len(result["retrieved_chunks"]) == 3, f"expected 3 chunks, got {len(result['retrieved_chunks'])}"

chunk = result["retrieved_chunks"][0]
assert "id" in chunk
assert "text" in chunk
print("BackendAdapter.generate() OK")

# --- top_k is respected ---
result_k2 = adapter.generate("Hello", top_k=2)
assert len(result_k2["retrieved_chunks"]) == 2
print("BackendAdapter top_k OK")

# --- run_generation() returns GenerateResponse ---
request = GenerateRequest(prompt="What is PPO?", top_k=2, show_prompt=False)
response = run_generation(request)

assert response.prompt == "What is PPO?"
assert len(response.retrieved_chunks) == 2
assert isinstance(response.response, str) and len(response.response) > 0
assert isinstance(response.final_prompt, str)
print("run_generation() OK")

# --- each chunk has id and text ---
for chunk in response.retrieved_chunks:
    assert chunk.id
    assert chunk.text
print("retrieved_chunks shape OK")

# --- empty prompt raises ValueError ---
try:
    run_generation(GenerateRequest(prompt="   "))
    assert False, "should have raised ValueError"
except ValueError:
    pass
print("empty prompt validation OK")

print("\nMilestone 3 PASSED — adapter and generation service work correctly.")
