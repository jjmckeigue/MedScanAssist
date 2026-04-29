import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.app.config import settings
from backend.app.routes.analyze import router as analyze_router
from backend.app.routes.auth import router as auth_router
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

limiter = Limiter(key_func=get_remote_address)


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
    version="0.4.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning("Rate limit exceeded: %s %s from %s", request.method, request.url.path, get_remote_address(request))
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)



def _is_auth_enforced() -> bool:
    return settings.app_env.lower() in {"staging", "production"}

OPEN_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/auth/register",
    "/auth/login",
    "/auth/refresh",
    "/auth/verify",
    "/auth/resend-verification",
}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next) -> Response:
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    if settings.app_env != "development":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; "
            "connect-src 'self'"
        )
    return response


@app.middleware("http")
async def jwt_auth_middleware(request: Request, call_next) -> Response:
    """Enforce JWT authentication on all protected paths.

    Falls back to legacy API key check if JWT is not present and API_KEY is configured.
    """
    path = request.url.path

    if request.method == "OPTIONS":
        return await call_next(request)

    if path in OPEN_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
        return await call_next(request)

    if not _is_auth_enforced():
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        from backend.app.auth import decode_token
        from jose import JWTError

        token = auth_header[7:]
        try:
            payload = decode_token(token)
            if payload.get("type") != "access":
                return Response(
                    content='{"detail":"Invalid token type."}',
                    status_code=401,
                    media_type="application/json",
                )
        except JWTError:
            return Response(
                content='{"detail":"Invalid or expired token."}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)

    api_key = settings.api_key
    if api_key:
        provided = request.headers.get("X-API-Key", "")
        if provided == api_key:
            return await call_next(request)

    logger.warning("Unauthorized request: %s %s", request.method, request.url.path)
    return Response(
        content='{"detail":"Authentication required. Provide a valid Bearer token."}',
        status_code=401,
        media_type="application/json",
    )


@app.middleware("http")
async def request_body_size_middleware(request: Request, call_next) -> Response:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_upload_bytes:
        return Response(
            content=f'{{"detail":"Request body too large: max allowed size is {settings.max_upload_bytes} bytes."}}',
            status_code=413,
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


# CORSMiddleware MUST be registered AFTER all @app.middleware("http") decorators.
# FastAPI's add_middleware uses insert(0, ...), so the last registration becomes
# the outermost middleware — exactly where CORS needs to be to handle preflight
# OPTIONS requests before the JWT auth middleware can reject them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
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
