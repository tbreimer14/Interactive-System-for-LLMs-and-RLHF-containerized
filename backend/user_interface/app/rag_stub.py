"""
rag_stub.py

Stub for the RAG pipeline.

RIGHT NOW: returns mock retrieved chunks and a mock response.
LATER: replace the body of answer() with the real RAG call, e.g.:
    from rag_system.pipeline import answer as rag_answer
    return rag_answer(query, k=k)

The function signature must stay the same so nothing else in the system breaks.
"""

import random

# ---------------------------------------------------------------------------
# Demo corpus — realistic-looking passages for demonstration purposes.
# Replace this entire module body with the real RAG call when ready:
#     from rag_system.pipeline import answer as rag_answer
#     return rag_answer(query, k=k)
# ---------------------------------------------------------------------------

_PASSAGES = [
    (
        "RLHF (Reinforcement Learning from Human Feedback) is a training paradigm "
        "in which a language model is fine-tuned using reward signals derived from "
        "human preference judgements. A reward model is first trained on ranked "
        "response pairs, then used to guide policy optimisation via PPO or GRPO.",
        "rlhf_overview.txt",
    ),
    (
        "GRPO (Group Relative Policy Optimisation) replaces the value network used "
        "in PPO with a group-normalised advantage estimate. For each prompt a group "
        "of G responses is sampled; the advantage of each response is its reward "
        "minus the group mean, divided by the group standard deviation.",
        "grpo_algorithm.txt",
    ),
    (
        "Qwen2.5-3B-Instruct is a 3-billion-parameter instruction-tuned language "
        "model from the Qwen family. It supports a 32k context window and achieves "
        "strong benchmark performance for its size. LoRA adapters can be applied to "
        "the attention projection layers for parameter-efficient fine-tuning.",
        "qwen25_model_card.txt",
    ),
    (
        "Retrieval-Augmented Generation (RAG) augments a language model's input "
        "with passages retrieved from an external index. The query is encoded with "
        "a bi-encoder (e.g. all-MiniLM-L6-v2) and nearest-neighbour search is "
        "performed over a FAISS flat-IP index.",
        "rag_architecture.txt",
    ),
    (
        "all-MiniLM-L6-v2 is a sentence-transformer model that maps text to a "
        "384-dimensional dense vector space. It was distilled from a larger model "
        "and trained on over 1 billion sentence pairs, making it fast and effective "
        "for semantic similarity and retrieval tasks.",
        "embedding_models.txt",
    ),
    (
        "LoRA (Low-Rank Adaptation) injects trainable rank-decomposition matrices "
        "into each layer of a frozen pre-trained model. Only the adapter weights are "
        "updated during fine-tuning, reducing trainable parameters by up to 10 000x "
        "while matching or approaching full fine-tune performance.",
        "lora_paper_summary.txt",
    ),
    (
        "A reward function aggregates per-trait scores into a scalar reward. Each "
        "trait score is multiplied by its weight and the products are summed. The "
        "scalar reward is stored in a JSONL log alongside the prompt, retrieved "
        "chunks, and model response, forming the training dataset for GRPO.",
        "reward_function_design.txt",
    ),
    (
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity "
        "search over dense vectors. A flat inner-product index performs exact "
        "nearest-neighbour lookup and is suitable for corpora of up to a few million "
        "chunks on a single machine.",
        "faiss_index_guide.txt",
    ),
]

_RESPONSE_TEMPLATES = [
    (
        "Based on the retrieved context, {query_lower} "
        "The key insight is that {topic_sentence} "
        "This approach allows the system to {benefit}, "
        "which is important for achieving strong performance in practice."
    ),
    (
        "To address your question about {query_lower} "
        "the relevant background is as follows. {topic_sentence} "
        "In practice, this means {benefit}, "
        "and the retrieved documents support this view."
    ),
]

_TOPICS = [
    ("the training process benefits from human preference data,",
     "iteratively improve response quality without full retraining"),
    ("the model can be adapted efficiently with parameter-efficient methods,",
     "fine-tune on task-specific data with minimal compute overhead"),
    ("retrieval grounds the generation in factual document content,",
     "reduce hallucination and keep answers up to date"),
    ("the reward signal directly shapes the policy update,",
     "align model behaviour with annotator preferences over time"),
]


def answer(query: str, k: int = 3) -> dict:
    """
    Run the RAG pipeline for a given query and return the result.

    Args:
        query: the user's question or prompt
        k:     number of chunks to retrieve (default 3)

    Returns:
        {
            "query":     the original query string,
            "retrieved": list of k chunk dicts, each with "text" and "source",
            "answer":    the generated response string
        }

    --- STUB BEHAVIOUR ---
    Seeds the RNG with a hash of the query for deterministic output.
    Swap this body with the real RAG call when ready.
    """
    rng = random.Random(hash(query))

    pool = list(_PASSAGES)
    rng.shuffle(pool)
    selected = pool[:k]

    retrieved = [{"text": text, "source": source} for text, source in selected]

    topic_sentence, benefit = rng.choice(_TOPICS)
    template = rng.choice(_RESPONSE_TEMPLATES)
    mock_answer = template.format(
        query_lower=query.rstrip("?. ").lower() + ",",
        topic_sentence=topic_sentence,
        benefit=benefit,
    )

    return {
        "query": query,
        "retrieved": retrieved,
        "answer": mock_answer,
    }
