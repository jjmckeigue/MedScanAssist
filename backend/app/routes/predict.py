from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.app.schemas import PredictionResponse
from backend.app.services.model_service import InvalidImageError, model_service

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), threshold: float | None = Query(default=None, ge=0.0, le=1.0)
) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        result = model_service.predict(image_bytes, threshold=threshold)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(**result)
