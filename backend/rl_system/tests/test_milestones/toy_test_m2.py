"""Milestone 2: Reward function bridge"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from grpo_system.reward import build_reward_fn


def mock_scores(prompt, response):
    return {"interactivity": 4, "warmth": 3, "directness": 5}


def test_returns_list_of_floats():
    reward_fn = build_reward_fn(mock_scores)
    rewards = reward_fn(["prompt a", "prompt b"], ["response a", "response b"])
    assert isinstance(rewards, list), "reward_fn must return a list"
    assert len(rewards) == 2, "one reward per (prompt, completion) pair"
    assert all(isinstance(r, float) for r in rewards), "rewards must be floats"
    print("  returns list[float]: OK")


def test_reward_value_matches_weighted_sum():
    # traits.json: interactivity=0.5, warmth=0.3, directness=0.2
    # expected = 4*0.5 + 3*0.3 + 5*0.2 = 2.0 + 0.9 + 1.0 = 3.9
    reward_fn = build_reward_fn(mock_scores)
    rewards = reward_fn(["test prompt"], ["test response"])
    assert abs(rewards[0] - 3.9) < 1e-4, f"expected ~3.9, got {rewards[0]}"
    print(f"  reward value {rewards[0]}: OK")


def test_scores_fn_called_per_pair():
    calls = []

    def tracking_scores(prompt, response):
        calls.append((prompt, response))
        return {"interactivity": 5, "warmth": 5, "directness": 5}

    reward_fn = build_reward_fn(tracking_scores)
    reward_fn(["p1", "p2", "p3"], ["r1", "r2", "r3"])
    assert len(calls) == 3, "get_scores_fn must be called once per pair"
    assert calls[0] == ("p1", "r1")
    assert calls[2] == ("p3", "r3")
    print("  get_scores_fn called per pair: OK")


def test_kwargs_accepted():
    reward_fn = build_reward_fn(mock_scores)
    # GRPOTrainer may pass extra kwargs — must not raise
    rewards = reward_fn(["p"], ["r"], extra_kwarg="ignored")
    assert len(rewards) == 1
    print("  extra kwargs accepted: OK")


if __name__ == "__main__":
    print("=== Milestone 2: Reward Function ===")
    test_returns_list_of_floats()
    test_reward_value_matches_weighted_sum()
    test_scores_fn_called_per_pair()
    test_kwargs_accepted()
    print("\nAll Milestone 2 tests passed.")
