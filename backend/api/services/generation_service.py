from api.schemas import GenerateRequest, GenerateResponse, RetrievedChunk
from backend_adapter import BackendAdapter

_adapter = BackendAdapter()


def run_generation(request: GenerateRequest) -> GenerateResponse:
    if not request.prompt.strip():
        raise ValueError("Prompt must not be empty.")

    result = _adapter.generate(request.prompt, top_k=request.top_k)

    chunks = [RetrievedChunk(**c) for c in result["retrieved_chunks"]]

    return GenerateResponse(
        prompt=request.prompt,
        retrieved_chunks=chunks,
        response=result["response"],
        final_prompt=result["final_prompt"],
    )
