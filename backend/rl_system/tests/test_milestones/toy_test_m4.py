"""Milestone 4: Prompt builder

Downloads ~15MB dataset on first run (cached by sklearn after).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from grpo_system.data import load_articles, format_for_qwen, INSTRUCTION


def test_load_articles_returns_list():
    prompts = load_articles(max_articles=20)
    assert isinstance(prompts, list), "load_articles must return a list"
    assert len(prompts) > 0, "must return at least one prompt"
    print(f"  returned {len(prompts)} prompts: OK")


def test_articles_start_with_instruction():
    prompts = load_articles(max_articles=20)
    for p in prompts:
        assert p.startswith(INSTRUCTION), "every prompt must start with INSTRUCTION"
    print("  all prompts start with INSTRUCTION: OK")


def test_articles_respect_max_chars():
    prompts = load_articles(max_articles=20, max_chars=200)
    for p in prompts:
        article_body = p[len(INSTRUCTION):]
        assert len(article_body) <= 200, f"article body too long: {len(article_body)}"
    print("  max_chars respected: OK")


def test_short_articles_filtered():
    # all returned articles must be >= 100 chars (body only)
    prompts = load_articles(max_articles=50)
    for p in prompts:
        body = p[len(INSTRUCTION):]
        assert len(body) >= 100, f"short article slipped through: {len(body)} chars"
    print("  short articles filtered: OK")


def test_format_for_qwen():
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n"

    prompt = "Rewrite this article..."
    result = format_for_qwen(prompt, tokenizer)

    tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = tokenizer.apply_chat_template.call_args
    messages = call_kwargs[0][0]
    assert messages == [{"role": "user", "content": prompt}]
    assert call_kwargs[1]["tokenize"] == False
    assert call_kwargs[1]["add_generation_prompt"] == True
    assert isinstance(result, str)
    print("  format_for_qwen calls apply_chat_template correctly: OK")


if __name__ == "__main__":
    print("=== Milestone 4: Prompt Builder ===")
    print("  (downloads ~15MB dataset on first run, cached after)\n")
    test_load_articles_returns_list()
    test_articles_start_with_instruction()
    test_articles_respect_max_chars()
    test_short_articles_filtered()
    test_format_for_qwen()
    print("\nAll Milestone 4 tests passed.")
