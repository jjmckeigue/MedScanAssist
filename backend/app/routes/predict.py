from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.app.config import settings
from backend.app.schemas import PredictionResponse
from backend.app.services.history_service import history_service
from backend.app.services.model_service import InvalidImageError, model_service

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    tta: bool = Query(default=False, description="Enable test-time augmentation for more robust predictions."),
) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Uploaded image exceeds max allowed size of "
                f"{settings.max_upload_bytes // (1024 * 1024)} MB."
            ),
        )

    try:
        result = model_service.predict(image_bytes, threshold=threshold, tta=tta)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analysis_id = history_service.add_record(file_name=file.filename, prediction=result)
    result["analysis_id"] = analysis_id
    return PredictionResponse(**result)
