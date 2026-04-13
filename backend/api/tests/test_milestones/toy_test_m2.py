"""
Milestone 2 test — storage_service.py

Run from the backend/api directory:
    uv run python tests/test_milestones/toy_test_m2.py
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import services.storage_service as storage

# --- use a temp file so tests don't pollute real data ---
tmp_dir = tempfile.mkdtemp()
tmp_file = os.path.join(tmp_dir, "interactions.jsonl")
storage.INTERACTIONS_FILE = tmp_file

try:
    # --- save_interaction: missing file is created automatically ---
    record1 = {
        "id": "id_001",
        "timestamp": "2026-04-11T10:00:00",
        "prompt": "What is RL?",
        "retrieved_chunks": [],
        "response": "RL is ML.",
        "final_prompt": "What is RL?",
        "traits": [],
        "scalar_reward": 0.0,
    }
    storage.save_interaction(record1)
    assert os.path.exists(tmp_file), "interactions.jsonl was not created"
    print("save_interaction (new file) OK")

    # --- save a second record ---
    record2 = {**record1, "id": "id_002", "prompt": "What is PPO?", "response": "PPO is an RL algorithm."}
    storage.save_interaction(record2)

    # --- load_interactions: returns both records ---
    records = storage.load_interactions(limit=20)
    assert len(records) == 2, f"expected 2 records, got {len(records)}"
    assert records[0]["id"] == "id_001"
    assert records[1]["id"] == "id_002"
    print("load_interactions OK")

    # --- limit is respected ---
    limited = storage.load_interactions(limit=1)
    assert len(limited) == 1
    assert limited[0]["id"] == "id_002", "limit should return the most recent record"
    print("load_interactions (limit) OK")

    # --- get_interaction_by_id: found ---
    found = storage.get_interaction_by_id("id_001")
    assert found is not None
    assert found["prompt"] == "What is RL?"
    print("get_interaction_by_id (found) OK")

    # --- get_interaction_by_id: not found ---
    not_found = storage.get_interaction_by_id("does_not_exist")
    assert not_found is None
    print("get_interaction_by_id (not found) OK")

    # --- handles bad lines gracefully ---
    with open(tmp_file, "a") as f:
        f.write("this is not valid json\n")
    records_after_bad = storage.load_interactions(limit=20)
    assert len(records_after_bad) == 2, "bad lines should be skipped silently"
    print("bad line handling OK")

    # --- missing file returns empty list ---
    storage.INTERACTIONS_FILE = os.path.join(tmp_dir, "nonexistent.jsonl")
    assert storage.load_interactions() == []
    assert storage.get_interaction_by_id("anything") is None
    print("missing file handling OK")

    print("\nMilestone 2 PASSED — storage service works correctly.")

finally:
    shutil.rmtree(tmp_dir)
