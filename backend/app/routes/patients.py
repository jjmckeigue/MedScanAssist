import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Query

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
from backend.app.services.patient_service import patient_service

logger = logging.getLogger("medscanassist.patients")

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("", response_model=PatientResponse, status_code=201)
async def create_patient(body: PatientCreate) -> PatientResponse:
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
    logger.info("Patient created: id=%d name=%s %s", patient["id"], body.first_name, body.last_name)
    return PatientResponse(**patient)


@router.get("", response_model=PatientListResponse)
async def list_patients(
    search: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PatientListResponse:
    patients = patient_service.list_patients(search=search, limit=limit, offset=offset)
    total = patient_service.count(search=search)
    return PatientListResponse(
        patients=[PatientResponse(**p) for p in patients],
        total=total,
    )


@router.get("/{patient_id}", response_model=PatientDetailResponse)
async def get_patient(patient_id: int) -> PatientDetailResponse:
    patient = patient_service.get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    analyses = patient_service.get_patient_analyses(patient_id)
    return PatientDetailResponse(
        **patient,
        analyses=[AnalysisHistoryRecord(**a) for a in analyses],
        analysis_count=len(analyses),
    )


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: int, body: PatientUpdate) -> PatientResponse:
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
    logger.info("Patient updated: id=%d", patient_id)
    return PatientResponse(**updated)


@router.delete("/{patient_id}")
async def delete_patient(patient_id: int) -> dict:
    if not patient_service.delete(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found.")
    logger.info("Patient deleted: id=%d", patient_id)
    return {"deleted": True}


@router.get("/{patient_id}/progression", response_model=PatientProgressionResponse)
async def get_progression(patient_id: int) -> PatientProgressionResponse:
    patient = patient_service.get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    points = patient_service.get_progression(patient_id)
    return PatientProgressionResponse(
        patient_id=patient_id,
        points=[ProgressionPoint(**p) for p in points],
    )
