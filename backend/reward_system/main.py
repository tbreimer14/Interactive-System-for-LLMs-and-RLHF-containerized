"""
main.py

Entry point for the reward system.
Loads traits, accepts a prompt + response + user scores, computes the
scalar reward, logs it, and prints formatted output.

Run with default inputs:
    uv run python main.py

Pass custom prompt and response (user scores are prompted interactively):
    uv run python main.py "your prompt here" "your response here"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trait_loader import load_traits
from reward_fn import compute_reward
from logger import log_reward

CONFIG_PATH = Path(__file__).parent / "config" / "traits.json"
LOG_PATH    = Path(__file__).parent / "logs" / "reward_log.jsonl"

# Default sample inputs
DEFAULT_PROMPT = (
    "I've been struggling with anxiety for a few months now and I'm not sure "
    "where to start. I don't really want to go to therapy yet — is there "
    "anything I can do on my own first?"
)
DEFAULT_RESPONSE = (
    "That sounds really hard, and it makes a lot of sense that you'd want to "
    "try some things on your own before jumping into therapy. A good starting "
    "point is building small, consistent habits — things like a short daily walk, "
    "limiting caffeine in the afternoon, and keeping a rough sleep schedule. "
    "Journaling can also help a lot: even just writing down what you're feeling "
    "and when can start to show you patterns. Would it help to talk through any "
    "of these in more detail?"
)

# Sample user scores — edit these or extend main.py to collect them from input
DEFAULT_USER_SCORES = {
    "interactivity": 5,
    "warmth": 4,
    "directness": 3,
}


def print_result(prompt: str, response: str, traits: list[dict], result: dict, log_path: Path) -> None:
    """Print a formatted summary of the reward computation."""
    W = 62
    print("\n" + "=" * W)
    print("  REWARD SYSTEM - RESULT")
    print("=" * W)
    print(f"\n  Prompt:")
    print(f"    {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"\n  Response:")
    print(f"    {response[:80]}{'...' if len(response) > 80 else ''}")
    print(f"\n  Trait Scores:")

    for trait in traits:
        name   = trait["name"]
        score  = result["trait_scores"][name]
        weight = result["weights"][name]
        contribution = round(weight * score, 4)
        print(f"    {name:<16}  score={score}  weight={weight}  → {contribution}")

    print(f"\n  Final Reward: {result['final_reward']}")
    print(f"\n  Logged to: {log_path}")
    print("=" * W + "\n")


def main():
    prompt   = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    response = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_RESPONSE

    # 1. Load traits from config
    traits = load_traits(str(CONFIG_PATH))

    # 2. Use default sample scores (replace with UI / CLI input later)
    user_scores = DEFAULT_USER_SCORES

    # 3. Compute weighted scalar reward
    result = compute_reward(prompt, response, traits, user_scores)

    # 4. Log the full event
    log_reward(prompt, response, traits, result, log_path=str(LOG_PATH))

    # 5. Print formatted output
    print_result(prompt, response, traits, result, LOG_PATH)


if __name__ == "__main__":
    main()
