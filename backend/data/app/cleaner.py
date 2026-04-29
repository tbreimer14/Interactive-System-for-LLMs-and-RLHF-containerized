"""
Text cleaning module for 20 Newsgroups documents.

Functions:
- clean_text(text): Remove noise and normalize a single post
- clean_dataset(dataset): Apply clean_text to all posts in a loaded dataset
- filter_short(docs, min_chars): Drop documents below a character threshold
"""

import re


def clean_text(text):
    """
    Clean a single newsgroup post.

    Steps:
    1. Collapse excessive whitespace / blank lines
    2. Remove lines that are purely punctuation or special characters
    3. Strip leading/trailing whitespace

    Args:
        text (str): Raw post text (headers/footers already removed by loader).

    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip lines that are only punctuation / dashes / equals signs (dividers)
        if re.match(r"^[-=_*#|<>~^]+$", stripped):
            continue

        # Skip lines that look like leftover email/attribution artifacts
        if re.match(r"^(writes?|wrote|said|says?)\s*:?\s*$", stripped, re.IGNORECASE):
            continue

        cleaned_lines.append(stripped)

    # Join and collapse multiple consecutive blank lines into one
    joined = "\n".join(cleaned_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)

    return joined.strip()


def clean_dataset(dataset):
    """
    Apply clean_text to every post in the dataset dict returned by loader.

    Args:
        dataset (dict): Output of loader.load_20newsgroups().

    Returns:
        list[dict]: List of dicts with keys {text, category, category_index}.
    """
    docs = []
    target_names = dataset["target_names"]

    for text, target_idx in zip(dataset["data"], dataset["target"]):
        cleaned = clean_text(text)
        docs.append({
            "text": cleaned,
            "category": target_names[target_idx],
            "category_index": int(target_idx),
        })

    return docs


def filter_short(docs, min_chars=100):
    """
    Remove documents shorter than min_chars after cleaning.

    Args:
        docs (list[dict]): Output of clean_dataset().
        min_chars (int): Minimum character count to keep a document.

    Returns:
        list[dict]: Filtered list.
    """
    return [doc for doc in docs if len(doc["text"]) >= min_chars]
