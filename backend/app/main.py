from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.routes.gradcam import router as gradcam_router
from backend.app.routes.health import router as health_router
from backend.app.routes.history import router as history_router
from backend.app.routes.model_info import router as model_info_router
from backend.app.routes.predict import router as predict_router
from backend.app.services.model_service import CheckpointRequiredError, model_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        model_service.ensure_ready()
    except CheckpointRequiredError as exc:
        raise RuntimeError(str(exc)) from exc
    yield

app = FastAPI(
    title=settings.app_name,
    description="Chest X-ray pneumonia classification + Grad-CAM explainability API.",
    version="0.1.0",
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
app.include_router(predict_router)
app.include_router(gradcam_router)
