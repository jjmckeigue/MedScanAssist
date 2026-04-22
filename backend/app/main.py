import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("medscanassist")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting MedScanAssist API (env=%s)", settings.app_env)
    try:
        model_service.ensure_ready()
        logger.info(
            "Model ready: mode=%s arch=%s checkpoint=%s",
            model_service.inference_mode,
            model_service.model_arch,
            model_service.checkpoint_loaded,
        )
    except CheckpointRequiredError as exc:
        logger.critical("Checkpoint required but not found: %s", exc)
        raise RuntimeError(str(exc)) from exc
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down MedScanAssist API")

app = FastAPI(
    title=settings.app_name,
    description="Chest X-ray pneumonia classification + Eigen-CAM explainability API.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next) -> Response:
    api_key = settings.api_key
    if api_key and request.url.path not in OPEN_PATHS:
        provided = request.headers.get("X-API-Key", "")
        if provided != api_key:
            logger.warning("Unauthorized request: %s %s", request.method, request.url.path)
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


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
