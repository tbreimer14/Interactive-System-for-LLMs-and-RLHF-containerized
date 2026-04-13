import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    RewardRequest,
    RewardResponse,
    SaveInteractionRequest,
    InteractionRecord,
)
from services.generation_service import run_generation
from services.reward_service import compute_reward
from services.storage_service import save_interaction, load_interactions, get_interaction_by_id

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt must not be empty.")
    try:
        return run_generation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reward/compute", response_model=RewardResponse)
def reward_compute(request: RewardRequest):
    try:
        return compute_reward(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/interactions")
def save(request: SaveInteractionRequest):
    record = InteractionRecord(
        id=uuid.uuid4().hex,
        timestamp=datetime.utcnow().isoformat(),
        **request.model_dump(),
    )
    save_interaction(record.model_dump())
    return {"status": "saved"}


@router.get("/interactions", response_model=list[InteractionRecord])
def get_interactions(limit: int = Query(default=20, ge=1)):
    return load_interactions(limit=limit)


@router.get("/interactions/{id}", response_model=InteractionRecord)
def get_interaction(id: str):
    record = get_interaction_by_id(id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Interaction '{id}' not found.")
    return record
