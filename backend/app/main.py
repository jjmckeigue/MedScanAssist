from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.routes.analyze import router as analyze_router
from backend.app.routes.gradcam import router as gradcam_router
from backend.app.routes.health import router as health_router
from backend.app.routes.history import router as history_router
from backend.app.routes.model_info import router as model_info_router
from backend.app.routes.patients import router as patients_router
from backend.app.routes.predict import router as predict_router
from backend.app.services.model_service import CheckpointRequiredError, model_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        model_service.ensure_ready()
    except CheckpointRequiredError as exc:
        raise RuntimeError(str(exc)) from exc
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(
    title=settings.app_name,
    description="Chest X-ray pneumonia classification + Grad-CAM explainability API.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(model_info_router)
app.include_router(history_router)
app.include_router(patients_router)
app.include_router(predict_router)
app.include_router(analyze_router)
app.include_router(gradcam_router)


@app.get("/images/{filename}", tags=["images"])
async def serve_image(filename: str) -> FileResponse:
    """Serve a stored X-ray image by filename."""
    safe_name = filename.replace("..", "").replace("/", "").replace("\\", "")
    path = settings.upload_dir / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png")
