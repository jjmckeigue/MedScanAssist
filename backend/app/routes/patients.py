import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.auth import get_current_user_optional
from backend.app.schemas import (
    AnalysisHistoryRecord,
    PatientCreate,
    PatientDetailResponse,
    PatientListResponse,
    PatientProgressionResponse,
    PatientResponse,
    PatientUpdate,
    ProgressionPoint,
)
from backend.app.services.history_service import history_service
from backend.app.services.patient_service import patient_service

logger = logging.getLogger("medscanassist.patients")

router = APIRouter(prefix="/patients", tags=["patients"])
limiter = Limiter(key_func=get_remote_address)


def _audit(request: Request, action: str, resource_id: str | None, user: dict | None = None) -> None:
    history_service.log_phi_access(
        action=action,
        resource_type="patient",
        resource_id=resource_id,
        user_email=user.get("email") if user else None,
        ip_address=get_remote_address(request),
    )


@router.post("", response_model=PatientResponse, status_code=201)
@limiter.limit("60/minute")
async def create_patient(
    request: Request,
    body: PatientCreate,
    current_user: dict | None = Depends(get_current_user_optional),
) -> PatientResponse:
    try:
        patient = patient_service.create(
            first_name=body.first_name,
            last_name=body.last_name,
            date_of_birth=body.date_of_birth,
            medical_record_number=body.medical_record_number,
            notes=body.notes,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="A patient with that medical record number already exists.",
        ) from exc
    _audit(request, "create", str(patient["id"]), current_user)
    logger.info("Patient created: id=%d name=%s %s", patient["id"], body.first_name, body.last_name)
    return PatientResponse(**patient)


@router.get("", response_model=PatientListResponse)
@limiter.limit("60/minute")
async def list_patients(
    request: Request,
    search: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: dict | None = Depends(get_current_user_optional),
) -> PatientListResponse:
    _audit(request, "list", None, current_user)
    patients = patient_service.list_patients(search=search, limit=limit, offset=offset)
    total = patient_service.count(search=search)
    return PatientListResponse(
        patients=[PatientResponse(**p) for p in patients],
        total=total,
    )


@router.get("/{patient_id}", response_model=PatientDetailResponse)
@limiter.limit("60/minute")
async def get_patient(
    request: Request,
    patient_id: int,
    current_user: dict | None = Depends(get_current_user_optional),
) -> PatientDetailResponse:
    patient = patient_service.get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    _audit(request, "read", str(patient_id), current_user)
    analyses = patient_service.get_patient_analyses(patient_id)
    return PatientDetailResponse(
        **patient,
        analyses=[AnalysisHistoryRecord(**a) for a in analyses],
        analysis_count=len(analyses),
    )


@router.put("/{patient_id}", response_model=PatientResponse)
@limiter.limit("60/minute")
async def update_patient(
    request: Request,
    patient_id: int,
    body: PatientUpdate,
    current_user: dict | None = Depends(get_current_user_optional),
) -> PatientResponse:
    existing = patient_service.get_by_id(patient_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Patient not found.")
    try:
        updated = patient_service.update(patient_id, **body.model_dump(exclude_unset=True))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="A patient with that medical record number already exists.",
        ) from exc
    _audit(request, "update", str(patient_id), current_user)
    logger.info("Patient updated: id=%d", patient_id)
    return PatientResponse(**updated)


@router.delete("/{patient_id}")
@limiter.limit("60/minute")
async def delete_patient(
    request: Request,
    patient_id: int,
    current_user: dict | None = Depends(get_current_user_optional),
) -> dict:
    if not patient_service.delete(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found.")
    _audit(request, "delete", str(patient_id), current_user)
    logger.info("Patient deleted: id=%d", patient_id)
    return {"deleted": True}


@router.get("/{patient_id}/progression", response_model=PatientProgressionResponse)
@limiter.limit("60/minute")
async def get_progression(
    request: Request,
    patient_id: int,
    current_user: dict | None = Depends(get_current_user_optional),
) -> PatientProgressionResponse:
    patient = patient_service.get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    _audit(request, "read_progression", str(patient_id), current_user)
    points = patient_service.get_progression(patient_id)
    return PatientProgressionResponse(
        patient_id=patient_id,
        points=[ProgressionPoint(**p) for p in points],
    )
