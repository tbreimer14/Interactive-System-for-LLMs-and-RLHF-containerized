"""
Builds GRPO training prompts from 20 Newsgroups articles.

Each prompt is a fixed instruction + one article body.
GRPO generates G responses per prompt; sliders score each one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.datasets import fetch_20newsgroups
from transformers import PreTrainedTokenizer
from data.app.cleaner import clean_text


INSTRUCTION = (
    "Rewrite the following article to be more engaging. "
    "Your rewrite should encourage continued discussion, "
    "sound warm and approachable, and get to the point clearly.\n\n"
    "Article:\n"
)


def load_articles(max_articles: int = 500, max_chars: int = 800) -> list:
    """
    Load newsgroup articles and format as instruction prompts.

    Args:
        max_articles: how many articles to sample (subset for manageable human rating)
        max_chars:    truncate article body to this length before embedding in prompt

    Returns:
        list of prompt strings, one per article
    """
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
    )

    prompts = []
    for text in dataset.data[:max_articles]:
        text = clean_text(text)
        if len(text) < 100:
            continue
        prompts.append(INSTRUCTION + text[:max_chars])

    return prompts


def format_for_qwen(prompt: str, tokenizer: PreTrainedTokenizer) -> str:
    """
    Wrap a prompt in Qwen2.5-Instruct's chat template.

    GRPOTrainer will generate the completion (the rewrite) starting from
    the assistant turn that this template opens.
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
