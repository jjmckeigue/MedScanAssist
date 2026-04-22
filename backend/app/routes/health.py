import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.services.history_service import history_service
from backend.app.services.model_service import model_service

logger = logging.getLogger("medscanassist.health")

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> JSONResponse:
    checks: dict[str, str] = {}

    checks["model"] = "checkpoint" if model_service.checkpoint_loaded else "placeholder"

    try:
        with history_service._connect() as conn:  # noqa: SLF001
            conn.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception:
        logger.warning("Health check: database connectivity failed", exc_info=True)
        checks["database"] = "error"

    all_ok = checks["database"] == "ok" and model_service.checkpoint_loaded
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_ok else "degraded",
            "app": settings.app_name,
            "environment": settings.app_env,
            "checks": checks,
        },
    )
