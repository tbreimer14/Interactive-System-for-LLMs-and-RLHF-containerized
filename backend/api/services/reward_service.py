from api.schemas import RewardRequest, RewardResponse, TraitScoreOutput


def compute_reward(request: RewardRequest) -> RewardResponse:
    if not request.traits:
        raise ValueError("At least one trait is required.")

    output_traits = []
    scalar_reward = 0.0

    for trait in request.traits:
        contribution = trait.weight * trait.score
        scalar_reward += contribution
        output_traits.append(
            TraitScoreOutput(
                name=trait.name,
                weight=trait.weight,
                score=trait.score,
                contribution=contribution,
            )
        )

    return RewardResponse(traits=output_traits, scalar_reward=scalar_reward)
