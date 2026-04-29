"""
test_reward_cross_system.py

Sanity check: reward_fn (reward_system) and reward_bridge (user_interface)
must produce the same scalar for identical inputs. Both use the same
weighted-sum formula; divergence here would silently break RLHF training.

Run from backend/:
    uv run python tests/test_reward_cross_system.py
"""

import sys
import importlib.util
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent

# --- Import reward_fn (needs reward_system on sys.path for app.schemas) ---
sys.path.insert(0, str(BACKEND / "reward_system"))
from app.reward_fn import compute_reward as reward_fn_compute  # noqa: E402

# --- Import reward_bridge directly to avoid app.* module collision ---
_bridge_path = BACKEND / "user_interface" / "app" / "reward_bridge.py"
_spec = importlib.util.spec_from_file_location("reward_bridge", _bridge_path)
_bridge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bridge_mod)
reward_bridge_compute = _bridge_mod.compute_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reward_fn_scalar(traits, scores):
    """Thin wrapper: pull scalar out of reward_fn result."""
    result = reward_fn_compute("prompt", "response", traits, scores)
    return result["final_reward"]


def _bridge_scalar(traits, scores):
    """Thin wrapper: pull scalar out of reward_bridge result."""
    result = reward_bridge_compute(scores, traits)
    return result["scalar_reward"]


def _expected(traits, scores):
    """Ground-truth weighted sum (independent reference)."""
    return round(sum(t["weight"] * scores[t["name"]] for t in traits), 6)


def check(label, traits, scores):
    fn   = _reward_fn_scalar(traits, scores)
    br   = _bridge_scalar(traits, scores)
    exp  = _expected(traits, scores)
    assert abs(fn - exp) < 1e-9,  f"[{label}] reward_fn {fn} != expected {exp}"
    assert abs(br - exp) < 1e-9,  f"[{label}] reward_bridge {br} != expected {exp}"
    print(f"  [OK] {label:<32}  reward_fn={fn}  bridge={br}  expected={exp}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_single_trait():
    section("TEST 1: Single trait")
    traits = [{"name": "clarity", "description": "Clear", "weight": 0.5}]
    for score in [-5, -2, 0, 3, 5]:
        check(f"clarity={score}", traits, {"clarity": score})


def test_three_traits_positive():
    section("TEST 2: Three traits, all positive scores")
    traits = [
        {"name": "clarity",   "description": "", "weight": 0.4},
        {"name": "empathy",   "description": "", "weight": 0.3},
        {"name": "directness","description": "", "weight": 0.3},
    ]
    check("all=5",   traits, {"clarity": 5, "empathy": 5, "directness": 5})
    check("all=1",   traits, {"clarity": 1, "empathy": 1, "directness": 1})
    check("varied",  traits, {"clarity": 4, "empathy": 2, "directness": 5})


def test_mixed_signs():
    section("TEST 3: Mixed positive and negative scores")
    traits = [
        {"name": "warmth",    "description": "", "weight": 0.5},
        {"name": "directness","description": "", "weight": 0.3},
        {"name": "accuracy",  "description": "", "weight": 0.2},
    ]
    check("pos+neg+zero", traits, {"warmth": 3, "directness": -2, "accuracy": 0})
    check("all negative",  traits, {"warmth": -5, "directness": -5, "accuracy": -5})
    check("extreme mix",   traits, {"warmth": 5, "directness": -5, "accuracy": 5})


def test_boundary_scores():
    section("TEST 4: Boundary scores (-5 and +5)")
    traits = [
        {"name": "a", "description": "", "weight": 0.6},
        {"name": "b", "description": "", "weight": 0.4},
    ]
    check("max (+5/+5)", traits, {"a": 5, "b": 5})
    check("min (-5/-5)", traits, {"a": -5, "b": -5})
    check("min/max mix", traits, {"a": -5, "b": 5})


def test_weight_variation():
    section("TEST 5: Identical scores, varying weights")
    score = 3
    configs = [
        [{"name": "x", "description": "", "weight": w}]
        for w in [0.1, 0.33, 0.5, 0.75, 1.0]
    ]
    for cfg in configs:
        w = cfg[0]["weight"]
        check(f"weight={w}", cfg, {"x": score})


def test_real_config():
    section("TEST 6: Real traits.json from reward_system/config")
    config_path = BACKEND / "reward_system" / "config" / "traits.json"
    if not config_path.exists():
        print(f"  [SKIP] Config not found at {config_path}")
        return

    sys.path.insert(0, str(BACKEND / "reward_system"))
    from app.trait_loader import load_traits
    traits = load_traits(str(config_path))

    score_sets = [
        {t["name"]: 5 for t in traits},
        {t["name"]: -5 for t in traits},
        {t["name"]: i % 11 - 5 for i, t in enumerate(traits)},
    ]
    for scores in score_sets:
        label = ", ".join(f"{k}={v}" for k, v in scores.items())
        check(label, traits, scores)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  CROSS-SYSTEM REWARD CONSISTENCY CHECK")
    print("  reward_fn (reward_system) vs reward_bridge (user_interface)")
    print("="*60)

    tests = [
        test_single_trait,
        test_three_traits_positive,
        test_mixed_signs,
        test_boundary_scores,
        test_weight_variation,
        test_real_config,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {e}")
            raise

    print(f"\n{'='*60}")
    print(f"  ALL {passed}/{len(tests)} TESTS PASSED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
