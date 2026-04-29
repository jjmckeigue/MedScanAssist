from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.config import settings
from backend.app.schemas import GradCamResponse
from backend.app.services.gradcam_service import gradcam_service
from backend.app.services.model_service import InvalidImageError

router = APIRouter(tags=["explainability"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/gradcam", response_model=GradCamResponse)
@limiter.limit("120/minute")
async def gradcam(request: Request, file: UploadFile = File(...)) -> GradCamResponse:
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
        result = gradcam_service.build_overlay_base64(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GradCamResponse(**result)
