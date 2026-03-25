from fastapi import APIRouter

from backend.app.schemas import ModelInfoResponse
from backend.app.services.model_service import model_service

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.get_model_info())
