from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas import (
    AnalysisHistoryRecord,
    AnalysisHistorySummary,
    DriftReport,
    FeedbackRequest,
    FeedbackResponse,
)
from backend.app.services.history_service import history_service

router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[AnalysisHistoryRecord])
def list_history(limit: int = Query(default=50, ge=1, le=500)) -> list[AnalysisHistoryRecord]:
    payload = history_service.get_recent_records(limit=limit)
    return [AnalysisHistoryRecord(**row) for row in payload]


@router.get("/history/summary", response_model=AnalysisHistorySummary)
def history_summary() -> AnalysisHistorySummary:
    return AnalysisHistorySummary(**history_service.get_summary())


@router.get("/history/drift", response_model=DriftReport)
def drift_report(
    baseline_count: int = Query(default=200, ge=10, le=5000),
    recent_count: int = Query(default=50, ge=5, le=1000),
) -> DriftReport:
    return DriftReport(**history_service.get_drift_report(baseline_count, recent_count))


@router.post("/history/{record_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(record_id: int, body: FeedbackRequest) -> FeedbackResponse:
    allowed = {"correct", "incorrect", "clear"}
    feedback_value = body.feedback.strip().lower()
    if feedback_value not in allowed:
        raise HTTPException(status_code=422, detail=f"feedback must be one of: {', '.join(sorted(allowed))}")
    store_value = None if feedback_value == "clear" else feedback_value
    if not history_service.set_feedback(record_id, store_value):
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")
    return FeedbackResponse(id=record_id, feedback=store_value)
