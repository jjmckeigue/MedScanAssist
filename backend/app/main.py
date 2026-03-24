from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.routes.gradcam import router as gradcam_router
from backend.app.routes.health import router as health_router
from backend.app.routes.predict import router as predict_router

app = FastAPI(
    title=settings.app_name,
    description="Chest X-ray pneumonia classification + Grad-CAM explainability API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(gradcam_router)
