import logging
import socket

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.services.history_service import history_service
from backend.app.services.model_service import model_service

logger = logging.getLogger("medscanassist.health")

router = APIRouter(tags=["health"])


def _email_diagnostics() -> dict:
    """Non-secret fields so operators can see why transactional email might fail."""
    user_set = bool((settings.smtp_user or "").strip())
    pwd_set = bool((settings.smtp_password or "").strip())
    fe = (settings.frontend_url or "").strip()
    env_lower = settings.app_env.lower()

    out: dict = {
        "smtp_user_configured": user_set,
        "smtp_password_configured": pwd_set,
        "smtp_ready": user_set and pwd_set,
        "smtp_host": settings.smtp_host,
        "smtp_port": settings.smtp_port,
        "frontend_url": fe or "(empty)",
    }

    if env_lower in ("production", "staging") and fe and (
        "localhost" in fe.lower() or "127.0.0.1" in fe
    ):
        out["frontend_url_warning"] = (
            "FRONTEND_URL still looks like local dev; verification links in emails will be wrong."
        )

    if user_set and pwd_set:
        try:
            with socket.create_connection(
                (settings.smtp_host, int(settings.smtp_port)),
                timeout=4,
            ):
                out["smtp_tcp_reachable"] = True
        except OSError as exc:
            out["smtp_tcp_reachable"] = False
            out["smtp_tcp_error"] = str(exc)
            out["smtp_tcp_hint"] = (
                "Outbound SMTP is blocked or the host/port is wrong. Many hosts block port 25/587; "
                "try a dedicated email API (Resend, SendGrid) or allow egress to smtp.gmail.com:587."
            )
    else:
        out["smtp_tcp_reachable"] = None
        out["smtp_env_hint"] = (
            "Set SMTP_USER (full Gmail address) and SMTP_PASSWORD (16-char App Password) on this API service."
        )

    return out


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
            "email": _email_diagnostics(),
        },
    )
