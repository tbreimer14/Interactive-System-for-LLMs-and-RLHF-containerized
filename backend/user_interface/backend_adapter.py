"""
backend_adapter.py

Single point of contact between the UI and the backend modules.

RIGHT NOW: calls local module stubs directly (rag_stub, reward_bridge).
LATER: swap the method bodies to call real HTTP endpoints, e.g.:
    POST /generate
    POST /reward/compute

The class interface stays the same so components.py never needs to change.

Expected API (when real backend exists):

    POST /generate
        body:  {"prompt": str, "top_k": int}
        return: {"query": str, "retrieved": [...], "answer": str}

    POST /reward/compute
        body:  {"scores": {name: float}, "traits": [...]}
        return: {"contributions": {...}, "scalar_reward": float}
"""

from app.rag_stub import answer as rag_answer
from app.reward_bridge import compute_reward as _compute_reward
from ui.types import TraitConfig, ScoredTrait


class BackendAdapter:
    """
    Wraps all backend calls behind a clean interface.

    Instantiate once in app.py and pass to components that need it.
    If the real backend is unavailable, mock responses are returned
    automatically (the stubs already handle this).
    """

    def generate(self, prompt: str, top_k: int = 3) -> dict:
        """
        Send the prompt to the RAG pipeline and return the result.

        Args:
            prompt: the user's input text
            top_k:  number of chunks to retrieve

        Returns:
            {
                "query":     str,
                "retrieved": [{"text": str, "source": str}, ...],
                "answer":    str
            }
        """
        return rag_answer(prompt, k=top_k)

    def compute_reward(
        self,
        scores: dict[str, float],
        traits: list[TraitConfig],
    ) -> tuple[list[ScoredTrait], float]:
        """
        Compute the scalar reward from user-provided scores.

        Args:
            scores: {trait_name: score} from the UI sliders
            traits: list of TraitConfig from session state

        Returns:
            (scored_traits, scalar_reward)
            scored_traits: list of ScoredTrait (one per trait)
            scalar_reward: weighted sum
        """
        trait_dicts = [t.to_dict() for t in traits]
        result = _compute_reward(scores, trait_dicts)

        scored_traits = [
            ScoredTrait(
                name=t.name,
                weight=t.weight,
                score=scores[t.name],
                contribution=result["contributions"][t.name],
            )
            for t in traits
        ]

        return scored_traits, result["scalar_reward"]
