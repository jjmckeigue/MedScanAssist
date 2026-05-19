"""Admin-only endpoints for observability of the deployed app.

All routes are gated by :func:`backend.app.auth.get_admin_user`, which itself
goes through the same JWT verification path as every other protected route.
There is intentionally no email-based allowlist here -- access is controlled
by the ``role`` column on the ``users`` table. Promote a user with
``python -m backend.scripts.promote_admin <email>``.

Routes here MUST NOT return PHI: no patient names, no MRNs, no uploaded
filenames, no scan images. Aggregate counts and user-account metadata only.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.auth import get_admin_user
from backend.app.services.history_service import history_service
from backend.app.services.user_service import user_service

logger = logging.getLogger("medscanassist.admin")

router = APIRouter(prefix="/admin", tags=["admin"])
limiter = Limiter(key_func=get_remote_address)


class AdminStatsResponse(BaseModel):
    total_users: int
    active_users: int
    verified_users: int
    signups_last_7d: int
    signups_last_30d: int
    total_scans: int
    scans_last_24h: int
    scans_last_7d: int
    scans_last_30d: int
    pneumonia_count: int
    normal_count: int
    feedback_correct: int
    feedback_incorrect: int


class AdminUserRow(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    created_at_utc: str
    is_active: bool
    is_verified: bool


class AdminUsersResponse(BaseModel):
    users: list[AdminUserRow]
    limit: int
    offset: int
    total: int


class DailyScanCount(BaseModel):
    day: str
    count: int


class AdminActivityResponse(BaseModel):
    scans_per_day: list[DailyScanCount]


@router.get("/stats", response_model=AdminStatsResponse)
@limiter.limit("30/minute")
async def admin_stats(
    request: Request,
    _admin: dict = Depends(get_admin_user),
) -> AdminStatsResponse:
    user_stats = user_service.get_user_stats()
    scan_stats = history_service.get_admin_scan_stats()
    return AdminStatsResponse(**user_stats, **scan_stats)


@router.get("/users", response_model=AdminUsersResponse)
@limiter.limit("30/minute")
async def admin_users(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _admin: dict = Depends(get_admin_user),
) -> AdminUsersResponse:
    users = user_service.list_users_admin(limit=limit, offset=offset)
    stats = user_service.get_user_stats()
    return AdminUsersResponse(
        users=[AdminUserRow(**u) for u in users],
        limit=limit,
        offset=offset,
        total=stats["total_users"],
    )


@router.get("/activity", response_model=AdminActivityResponse)
@limiter.limit("30/minute")
async def admin_activity(
    request: Request,
    days: int = Query(default=14, ge=1, le=90),
    _admin: dict = Depends(get_admin_user),
) -> AdminActivityResponse:
    rows = history_service.get_scans_per_day(days=days)
    return AdminActivityResponse(
        scans_per_day=[DailyScanCount(**r) for r in rows],
    )
