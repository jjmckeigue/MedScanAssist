from fastapi import APIRouter, Query

from backend.app.schemas import AnalysisHistoryRecord, AnalysisHistorySummary
from backend.app.services.history_service import history_service

router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[AnalysisHistoryRecord])
def list_history(limit: int = Query(default=50, ge=1, le=500)) -> list[AnalysisHistoryRecord]:
    payload = history_service.get_recent_records(limit=limit)
    return [AnalysisHistoryRecord(**row) for row in payload]


@router.get("/history/summary", response_model=AnalysisHistorySummary)
def history_summary() -> AnalysisHistorySummary:
    return AnalysisHistorySummary(**history_service.get_summary())
