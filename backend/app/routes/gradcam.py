from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas import GradCamResponse
from backend.app.services.gradcam_service import gradcam_service

router = APIRouter(tags=["explainability"])


@router.post("/gradcam", response_model=GradCamResponse)
async def gradcam(file: UploadFile = File(...)) -> GradCamResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    result = gradcam_service.build_overlay_base64(image_bytes)
    return GradCamResponse(**result)
