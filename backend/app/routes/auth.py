from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend.app.config import settings
from backend.app.services.email_service import send_verification_email
from backend.app.services.user_service import (
    generate_verification_token,
    normalize_email,
    user_service,
    validate_email_policy,
    validate_password_strength,
)

logger = logging.getLogger("medscanassist.auth")

router = APIRouter(prefix="/auth", tags=["authentication"])
limiter = Limiter(key_func=get_remote_address)


# ---- Request / response schemas ----


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1, max_length=200)

    @field_validator("email")
    @classmethod
    def enforce_org_email(cls, value: EmailStr) -> str:
        validate_email_policy(str(value))
        return normalize_email(str(value))

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_login_email(cls, value: EmailStr) -> str:
        return normalize_email(str(value))


class RefreshRequest(BaseModel):
    refresh_token: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def normalize_resend_email(cls, value: EmailStr) -> str:
        return normalize_email(str(value))


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str


class RegisterResponse(BaseModel):
    message: str
    requires_verification: bool = True



class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class UpdateProfileRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)


class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

class UserProfileResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    created_at_utc: str


class SessionInfoResponse(BaseModel):
    created_at_utc: str
    expires_at_utc: str
    is_current: bool


# ---- Helpers ----


def _token_expiry() -> str:
    return (
        datetime.now(tz=timezone.utc)
        + timedelta(hours=settings.verification_token_expire_hours)
    ).isoformat()


# ---- Endpoints ----


@router.post("/register", response_model=RegisterResponse, status_code=201)
@limiter.limit("3/minute")
async def register(request: Request, body: RegisterRequest) -> RegisterResponse:
    try:
        validate_password_strength(body.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    hashed = hash_password(body.password)
    token = generate_verification_token()
    expires = _token_expiry()

    try:
        user_service.create(
            email=body.email,
            hashed_password=hashed,
            full_name=body.full_name,
            verification_token=token,
            verification_token_expires=expires,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="A user with this email already exists.",
        ) from exc

    send_verification_email(body.email, body.full_name, token)
    logger.info("User registered (pending verification): %s", body.email)

    return RegisterResponse(
        message="Account created. Please check your email to verify your account.",
        requires_verification=True,
    )


@router.get("/verify", response_model=MessageResponse)
@limiter.limit("10/minute")
async def verify_email(
    request: Request, token: str = Query(..., min_length=1)
) -> MessageResponse:
    user = user_service.get_by_verification_token(token)
    if user is None:
        raise HTTPException(status_code=400, detail="Invalid or expired verification link.")

    if user["is_verified"]:
        return MessageResponse(message="Email already verified. You can sign in.")

    expires_str = user.get("verification_token_expires")
    if expires_str:
        expires_dt = datetime.fromisoformat(expires_str)
        if datetime.now(tz=timezone.utc) > expires_dt:
            raise HTTPException(
                status_code=400,
                detail="Verification link has expired. Please request a new one.",
            )

    user_service.mark_verified(user["id"])
    logger.info("Email verified: %s", user["email"])
    return MessageResponse(message="Email verified successfully! You can now sign in.")


@router.post("/resend-verification", response_model=MessageResponse)
@limiter.limit("2/minute")
async def resend_verification(
    request: Request, body: ResendVerificationRequest
) -> MessageResponse:
    user = user_service.get_by_email(body.email)

    if user is None or user["is_verified"]:
        return MessageResponse(
            message="If an unverified account exists for this email, a new verification link has been sent."
        )

    token = generate_verification_token()
    expires = _token_expiry()
    user_service.set_verification_token(user["id"], token, expires)
    send_verification_email(body.email, user["full_name"], token)
    logger.info("Resent verification email: %s", body.email)

    return MessageResponse(
        message="If an unverified account exists for this email, a new verification link has been sent."
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest) -> TokenResponse:
    user = user_service.get_by_email(body.email)
    if user is None or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated.",
        )
    if not user.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before signing in. Check your inbox for a verification link.",
        )

    email = normalize_email(str(body.email))
    access_token = create_access_token(email)
    refresh_token = create_refresh_token(email)
    refresh_payload = decode_token(refresh_token)
    user_service.store_refresh_session(
        user_id=user["id"],
        refresh_token=refresh_token,
        expires_at_utc=datetime.fromtimestamp(refresh_payload["exp"], tz=timezone.utc).isoformat(),
    )

    logger.info("User logged in: %s", email)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("10/minute")
async def refresh(request: Request, body: RefreshRequest) -> TokenResponse:
    from jose import JWTError

    try:
        payload = decode_token(body.refresh_token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        ) from exc

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a refresh token.",
        )

    if not user_service.is_refresh_session_active(body.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh session is invalid or has been revoked.",
        )

    email: str | None = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    user = user_service.get_by_email(email)
    if user is None or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )

    user_service.revoke_refresh_session(body.refresh_token)
    access_token = create_access_token(email)
    refresh_token = create_refresh_token(email)
    next_payload = decode_token(refresh_token)
    user_service.store_refresh_session(
        user_id=user["id"],
        refresh_token=refresh_token,
        expires_at_utc=datetime.fromtimestamp(next_payload["exp"], tz=timezone.utc).isoformat(),
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.get("/me", response_model=UserProfileResponse)
async def me(current_user: dict = Depends(get_current_user)) -> UserProfileResponse:
    return UserProfileResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        role=current_user["role"],
        created_at_utc=current_user["created_at_utc"],
    )


@router.get("/sessions", response_model=list[SessionInfoResponse])
async def list_sessions(
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> list[SessionInfoResponse]:
    """Active refresh-token sessions. Send ``X-Refresh-Token`` to mark the current session."""

    refresh_header = request.headers.get("X-Refresh-Token") or ""
    current_hash: str | None = None
    if refresh_header:
        current_hash = hashlib.sha256(refresh_header.encode("utf-8")).hexdigest()

    rows = user_service.list_active_refresh_sessions(current_user["id"])
    return [
        SessionInfoResponse(
            created_at_utc=r["created_at_utc"],
            expires_at_utc=r["expires_at_utc"],
            is_current=current_hash is not None and r["token_hash"] == current_hash,
        )
        for r in rows
    ]


@router.post("/logout-all", response_model=MessageResponse)
async def logout_all_sessions(current_user: dict = Depends(get_current_user)) -> MessageResponse:
    n = user_service.revoke_all_refresh_sessions_for_user(current_user["id"])
    logger.info("All refresh sessions revoked: user_id=%s count=%s", current_user["id"], n)
    return MessageResponse(message=f"Signed out from {n} active session(s).")


@router.post("/logout", response_model=MessageResponse)
async def logout(body: LogoutRequest, current_user: dict = Depends(get_current_user)) -> MessageResponse:
    if body.refresh_token:
        user_service.revoke_token(body.refresh_token)
        user_service.revoke_refresh_session(body.refresh_token)
    logger.info("User logged out: %s", current_user["email"])
    return MessageResponse(message="Signed out successfully.")


@router.put("/me", response_model=UserProfileResponse)
async def update_me(body: UpdateProfileRequest, current_user: dict = Depends(get_current_user)) -> UserProfileResponse:
    updated = user_service.update_profile(current_user["id"], body.full_name)
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserProfileResponse(
        id=updated["id"],
        email=updated["email"],
        full_name=updated["full_name"],
        role=updated["role"],
        created_at_utc=updated["created_at_utc"],
    )


@router.post("/change-password", response_model=MessageResponse)
async def change_password(body: UpdatePasswordRequest, current_user: dict = Depends(get_current_user)) -> MessageResponse:
    if not verify_password(body.current_password, current_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    validate_password_strength(body.new_password)
    user_service.update_password(current_user["id"], hash_password(body.new_password))
    return MessageResponse(message="Password updated successfully.")


@router.delete("/me", response_model=MessageResponse)
async def deactivate_me(current_user: dict = Depends(get_current_user)) -> MessageResponse:
    user_service.deactivate_release_email(current_user["id"])
    logger.info("Account deactivated (email released): user_id=%s", current_user["id"])
    return MessageResponse(
        message=(
            "Account deactivated. You can register again with the same email; "
            "you will need to complete email verification."
        )
    )
