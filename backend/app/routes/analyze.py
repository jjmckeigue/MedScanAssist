import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.config import settings
from backend.app.schemas import AnalyzeResponse
from backend.app.services.gradcam_service import gradcam_service
from backend.app.services.history_service import history_service
from backend.app.services.model_service import InvalidImageError
from backend.app.services.patient_service import patient_service

logger = logging.getLogger("medscanassist.analyze")

router = APIRouter(tags=["inference"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("30/minute")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    patient_id: int | None = Query(default=None, description="Link this analysis to a patient profile."),
) -> AnalyzeResponse:
    """Combined prediction + Eigen-CAM in a single request (one forward pass)."""
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

    if patient_id is not None and not patient_service.get_by_id(patient_id):
        raise HTTPException(status_code=404, detail=f"Patient with id {patient_id} not found.")

    try:
        result = gradcam_service.build_overlay_with_prediction(image_bytes, threshold=threshold)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ONNX inference failed for file=%s", file.filename)
        raise HTTPException(status_code=500, detail="Model inference failed. Please try again.") from exc

    image_filename: str | None = None
    try:
        ext = (file.filename or "upload.png").rsplit(".", 1)[-1].lower()
        if ext not in ("png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"):
            ext = "png"
        image_filename = f"{uuid.uuid4().hex}.{ext}"
        dest = settings.upload_dir / image_filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image_bytes)
    except Exception:
        logger.warning("Failed to persist uploaded image for file=%s", file.filename, exc_info=True)
        image_filename = None

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
        patient_id=patient_id,
        image_path=image_filename,
    )

    logger.info(
        "Analysis complete: id=%d file=%s patient_id=%s label=%s confidence=%.3f threshold=%.2f mode=%s",
        analysis_id,
        file.filename,
        patient_id,
        result["predicted_label"],
        result["confidence"],
        result["threshold"],
        result["inference_mode"],
    )

    result["analysis_id"] = analysis_id
    return AnalyzeResponse(**result)
