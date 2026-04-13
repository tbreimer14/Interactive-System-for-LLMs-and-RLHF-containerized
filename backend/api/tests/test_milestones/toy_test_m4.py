"""
Milestone 4 test — reward_service.py

Run from the backend/api directory:
    uv run python tests/test_milestones/toy_test_m4.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.schemas import RewardRequest, TraitScoreInput
from services.reward_service import compute_reward

# --- single trait ---
req = RewardRequest(traits=[
    TraitScoreInput(name="clarity", weight=1.0, score=4.0),
])
result = compute_reward(req)

assert len(result.traits) == 1
assert result.traits[0].contribution == 4.0
assert result.scalar_reward == 4.0
print("single trait OK")

# --- multiple traits ---
req2 = RewardRequest(traits=[
    TraitScoreInput(name="clarity", weight=1.0, score=4.0),
    TraitScoreInput(name="empathy", weight=0.5, score=3.0),
])
result2 = compute_reward(req2)

assert result2.traits[0].contribution == 4.0
assert result2.traits[1].contribution == 1.5
assert abs(result2.scalar_reward - 5.5) < 1e-9
print("multiple traits OK")

# --- contribution formula: weight * score ---
req3 = RewardRequest(traits=[
    TraitScoreInput(name="humor", weight=0.3, score=5.0),
])
result3 = compute_reward(req3)
assert abs(result3.traits[0].contribution - 1.5) < 1e-9
assert abs(result3.scalar_reward - 1.5) < 1e-9
print("contribution formula OK")

# --- output preserves name, weight, score ---
req4 = RewardRequest(traits=[
    TraitScoreInput(name="directness", weight=0.2, score=3.0),
])
result4 = compute_reward(req4)
t = result4.traits[0]
assert t.name == "directness"
assert t.weight == 0.2
assert t.score == 3.0
print("output fields preserved OK")

# --- empty traits raises ValueError ---
try:
    compute_reward(RewardRequest(traits=[]))
    assert False, "should have raised ValueError"
except ValueError:
    pass
print("empty traits validation OK")

print("\nMilestone 4 PASSED — reward service works correctly.")
