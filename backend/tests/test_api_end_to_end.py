"""
test_api_end_to_end.py

Sanity check: full API lifecycle via FastAPI TestClient.

Exercises all endpoints in the order a real client would:
  health -> generate -> reward compute -> save -> list -> fetch by ID

Also runs a multi-response scenario: scores three responses differently
and verifies that saved records sort by scalar_reward correctly.

Uses a temp file so no test data bleeds into api/data/interactions.jsonl.

Run from backend/:
    uv run python tests/test_api_end_to_end.py
"""

import sys
import tempfile
import os
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent

# Put the api package root on the path so `from app import app` works
sys.path.insert(0, str(BACKEND / "api"))

import services.storage_service as storage_service  # noqa: E402
from app import app  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_client(tmp_path: str):
    """Return a TestClient whose storage is redirected to tmp_path."""
    storage_service.INTERACTIONS_FILE = tmp_path
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


SAMPLE_TRAITS = [
    {"name": "clarity",    "weight": 0.4, "score": 4.0},
    {"name": "empathy",    "weight": 0.3, "score": 3.0},
    {"name": "directness", "weight": 0.3, "score": 5.0},
]

EXPECTED_REWARD = round(
    sum(t["weight"] * t["score"] for t in SAMPLE_TRAITS), 6
)

SAMPLE_CHUNKS = [
    {"id": "c1", "source": "doc1", "text": "Chunk one content"},
    {"id": "c2", "source": "doc2", "text": "Chunk two content"},
]


def build_save_body(prompt, response, traits_with_scores, scalar_reward):
    traits_out = [
        {
            "name": t["name"],
            "weight": t["weight"],
            "score": t["score"],
            "contribution": round(t["weight"] * t["score"], 6),
        }
        for t in traits_with_scores
    ]
    return {
        "prompt": prompt,
        "retrieved_chunks": SAMPLE_CHUNKS,
        "response": response,
        "final_prompt": f"[context]\n{prompt}",
        "traits": traits_out,
        "scalar_reward": scalar_reward,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(client):
    section("TEST 1: Health check")
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert r.json() == {"status": "ok"}, f"Unexpected body: {r.json()}"
    print("  [OK] GET /health -> 200 {status: ok}")


def test_generate(client):
    section("TEST 2: Generate endpoint")

    r = client.post("/generate", json={"prompt": "Explain RLHF", "top_k": 3})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["prompt"] == "Explain RLHF"
    assert len(body["retrieved_chunks"]) == 3
    assert isinstance(body["response"], str) and body["response"]
    assert isinstance(body["final_prompt"], str)
    print(f"  [OK] POST /generate (top_k=3) -> 200, 3 chunks, non-empty response")

    r2 = client.post("/generate", json={"prompt": "Hello", "top_k": 1})
    assert r2.status_code == 200
    assert len(r2.json()["retrieved_chunks"]) == 1
    print(f"  [OK] top_k=1 respected")

    r3 = client.post("/generate", json={"prompt": "   "})
    assert r3.status_code == 422, f"Expected 422 for blank prompt, got {r3.status_code}"
    print(f"  [OK] Blank prompt -> 422")


def test_reward_compute(client):
    section("TEST 3: Reward compute endpoint")

    r = client.post("/reward/compute", json={"traits": SAMPLE_TRAITS})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()

    assert "scalar_reward" in body
    assert "traits" in body
    assert len(body["traits"]) == len(SAMPLE_TRAITS)

    actual = round(body["scalar_reward"], 6)
    assert abs(actual - EXPECTED_REWARD) < 1e-9, \
        f"scalar_reward {actual} != expected {EXPECTED_REWARD}"
    print(f"  [OK] scalar_reward={actual} matches expected {EXPECTED_REWARD}")

    for out_trait in body["traits"]:
        expected_contrib = round(out_trait["weight"] * out_trait["score"], 6)
        assert abs(out_trait["contribution"] - expected_contrib) < 1e-9, \
            f"contribution mismatch for {out_trait['name']}"
    print(f"  [OK] All per-trait contributions correct")

    r2 = client.post("/reward/compute", json={"traits": []})
    assert r2.status_code in (422, 500), \
        f"Expected error for empty traits, got {r2.status_code}"
    print(f"  [OK] Empty traits list -> error ({r2.status_code})")


def test_save_and_retrieve(client):
    section("TEST 4: Save interaction, then retrieve by ID")

    body = build_save_body(
        "What is PPO?",
        "PPO is a policy gradient method.",
        SAMPLE_TRAITS,
        EXPECTED_REWARD,
    )

    r = client.post("/interactions", json=body)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    assert r.json() == {"status": "saved"}
    print(f"  [OK] POST /interactions -> saved")

    r2 = client.get("/interactions")
    assert r2.status_code == 200
    records = r2.json()
    assert len(records) >= 1
    saved = records[-1]
    assert "id" in saved
    assert "timestamp" in saved
    assert saved["prompt"] == "What is PPO?"
    assert abs(saved["scalar_reward"] - EXPECTED_REWARD) < 1e-9
    print(f"  [OK] GET /interactions -> record present, fields correct")

    record_id = saved["id"]
    r3 = client.get(f"/interactions/{record_id}")
    assert r3.status_code == 200
    fetched = r3.json()
    assert fetched["id"] == record_id
    assert fetched["prompt"] == "What is PPO?"
    print(f"  [OK] GET /interactions/{{id}} -> correct record returned")

    r4 = client.get("/interactions/does-not-exist")
    assert r4.status_code == 404
    print(f"  [OK] Unknown ID -> 404")


def test_multi_response_ranking(client):
    section("TEST 5: Multi-response scenario -- save three, verify reward ordering")

    responses = [
        ("This is a great response.",    [4, 4, 4]),
        ("This is a mediocre response.", [1, 1, 1]),
        ("This is the best response.",   [5, 5, 5]),
    ]

    saved_rewards = []
    for resp_text, scores in responses:
        traits = [
            {"name": t["name"], "weight": t["weight"], "score": float(s)}
            for t, s in zip(SAMPLE_TRAITS, scores)
        ]
        scalar = round(sum(t["weight"] * t["score"] for t in traits), 6)
        body = build_save_body("Summarize RL.", resp_text, traits, scalar)
        r = client.post("/interactions", json=body)
        assert r.status_code == 200
        saved_rewards.append(scalar)
        print(f"  [OK] Saved: '{resp_text[:30]}' -> reward={scalar}")

    r = client.get("/interactions?limit=10")
    assert r.status_code == 200
    records = r.json()

    # Verify all three just-saved rewards appear in the returned records
    retrieved_rewards = [rec["scalar_reward"] for rec in records]
    for expected in saved_rewards:
        assert any(abs(got - expected) < 1e-9 for got in retrieved_rewards), \
            f"Expected reward {expected} not found in retrieved: {retrieved_rewards}"
    print(f"  [OK] All three rewards present in retrieved records: {sorted(saved_rewards)}")

    # Among all records, the one with the highest reward should be the best response
    best = max(records, key=lambda rec: rec["scalar_reward"])
    assert best["response"] == "This is the best response.", \
        f"Expected best response, got: '{best['response']}'"
    print(f"  [OK] Highest-reward response correctly identified (reward={best['scalar_reward']})")

    r2 = client.get("/interactions?limit=1")
    assert r2.status_code == 200
    assert len(r2.json()) == 1
    print(f"  [OK] limit=1 returns exactly 1 record")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  API END-TO-END SANITY CHECK")
    print("  FastAPI TestClient -- full lifecycle")
    print("="*60)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        client = make_client(tmp_path)

        tests = [
            (test_health,                client),
            (test_generate,              client),
            (test_reward_compute,        client),
            (test_save_and_retrieve,     client),
            (test_multi_response_ranking, client),
        ]

        passed = 0
        for fn, *args in tests:
            try:
                fn(*args)
                passed += 1
            except Exception as e:
                print(f"\n  [FAIL] in {fn.__name__}: {e}")
                raise

        print(f"\n{'='*60}")
        print(f"  ALL {passed}/{len(tests)} TESTS PASSED")
        print(f"{'='*60}\n")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
