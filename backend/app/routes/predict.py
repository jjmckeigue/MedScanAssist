from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas import PredictionResponse
from backend.app.services.model_service import model_service

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    result = model_service.predict(image_bytes)
    return PredictionResponse(**result)
