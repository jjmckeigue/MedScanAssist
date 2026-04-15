from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.app.config import settings
from backend.app.schemas import AnalyzeResponse
from backend.app.services.gradcam_service import gradcam_service
from backend.app.services.history_service import history_service
from backend.app.services.model_service import InvalidImageError

router = APIRouter(tags=["inference"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> AnalyzeResponse:
    """Combined prediction + Grad-CAM in a single request (one forward pass)."""
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
        result = gradcam_service.build_overlay_with_prediction(image_bytes, threshold=threshold)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analysis_id = history_service.add_record(
        file_name=file.filename,
        prediction={
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "threshold": result["threshold"],
            "inference_mode": result["inference_mode"],
            "model_arch": result["model_arch"],
            "checkpoint_loaded": result["checkpoint_loaded"],
        },
    )
    result["analysis_id"] = analysis_id
    return AnalyzeResponse(**result)
