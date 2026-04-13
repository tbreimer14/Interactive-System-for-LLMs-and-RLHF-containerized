"""
Milestone 5 test — routes + FastAPI app

Run from the backend/api directory:
    uv run python tests/test_milestones/toy_test_m5.py
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# redirect storage to a temp file before importing the app
import services.storage_service as storage
tmp_dir = tempfile.mkdtemp()
storage.INTERACTIONS_FILE = os.path.join(tmp_dir, "interactions.jsonl")

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

try:
    # --- GET /health ---
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print("GET /health OK")

    # --- POST /generate ---
    r = client.post("/generate", json={"prompt": "What is RL?", "top_k": 2, "show_prompt": False})
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"] == "What is RL?"
    assert len(body["retrieved_chunks"]) == 2
    assert isinstance(body["response"], str)
    assert isinstance(body["final_prompt"], str)
    print("POST /generate OK")

    # --- POST /generate — empty prompt → 422 ---
    r = client.post("/generate", json={"prompt": "   ", "top_k": 2, "show_prompt": False})
    assert r.status_code == 422
    print("POST /generate (empty prompt) OK")

    # --- POST /reward/compute ---
    r = client.post("/reward/compute", json={
        "traits": [
            {"name": "clarity", "weight": 1.0, "score": 4.0},
            {"name": "empathy", "weight": 0.5, "score": 2.0},
        ]
    })
    assert r.status_code == 200
    body = r.json()
    assert abs(body["scalar_reward"] - 5.0) < 1e-9
    assert body["traits"][0]["contribution"] == 4.0
    assert body["traits"][1]["contribution"] == 1.0
    print("POST /reward/compute OK")

    # --- POST /interactions ---
    interaction_payload = {
        "prompt": "What is RL?",
        "retrieved_chunks": [{"id": "chunk_1", "source": "doc.txt", "text": "RL is..."}],
        "response": "RL is a type of ML.",
        "final_prompt": "What is RL?",
        "traits": [{"name": "clarity", "weight": 1.0, "score": 4.0, "contribution": 4.0}],
        "scalar_reward": 4.0,
    }
    r = client.post("/interactions", json=interaction_payload)
    assert r.status_code == 200
    assert r.json() == {"status": "saved"}
    print("POST /interactions OK")

    # --- GET /interactions ---
    r = client.get("/interactions?limit=20")
    assert r.status_code == 200
    records = r.json()
    assert len(records) == 1
    saved_id = records[0]["id"]
    assert records[0]["prompt"] == "What is RL?"
    print("GET /interactions OK")

    # --- GET /interactions/{id} ---
    r = client.get(f"/interactions/{saved_id}")
    assert r.status_code == 200
    assert r.json()["id"] == saved_id
    print("GET /interactions/{id} OK")

    # --- GET /interactions/{id} — not found → 404 ---
    r = client.get("/interactions/does_not_exist")
    assert r.status_code == 404
    print("GET /interactions/{id} (not found) OK")

    print("\nMilestone 5 PASSED — all endpoints work correctly.")

finally:
    shutil.rmtree(tmp_dir)
