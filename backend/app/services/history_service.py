from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from backend.app.config import settings


class HistoryService:
    def __init__(self) -> None:
        self._db_path = Path(settings.history_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    file_name TEXT,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    threshold REAL NOT NULL,
                    inference_mode TEXT NOT NULL,
                    model_arch TEXT NOT NULL,
                    checkpoint_loaded INTEGER NOT NULL,
                    probabilities_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_record(self, file_name: str | None, prediction: dict) -> int:
        created_at_utc = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_history (
                    created_at_utc,
                    file_name,
                    predicted_label,
                    confidence,
                    threshold,
                    inference_mode,
                    model_arch,
                    checkpoint_loaded,
                    probabilities_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at_utc,
                    file_name,
                    str(prediction["predicted_label"]),
                    float(prediction["confidence"]),
                    float(prediction["threshold"]),
                    str(prediction["inference_mode"]),
                    str(prediction["model_arch"]),
                    1 if bool(prediction["checkpoint_loaded"]) else 0,
                    json.dumps(prediction["probabilities"]),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_recent_records(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    created_at_utc,
                    file_name,
                    predicted_label,
                    confidence,
                    threshold,
                    inference_mode,
                    model_arch,
                    checkpoint_loaded,
                    probabilities_json
                FROM analysis_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records = []
        for row in rows:
            records.append(
                {
                    "id": int(row["id"]),
                    "created_at_utc": str(row["created_at_utc"]),
                    "file_name": row["file_name"],
                    "predicted_label": str(row["predicted_label"]),
                    "confidence": float(row["confidence"]),
                    "threshold": float(row["threshold"]),
                    "inference_mode": str(row["inference_mode"]),
                    "model_arch": str(row["model_arch"]),
                    "checkpoint_loaded": bool(row["checkpoint_loaded"]),
                    "probabilities": json.loads(row["probabilities_json"]),
                }
            )
        return records

    def get_summary(self) -> dict:
        with self._connect() as conn:
            summary_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_reviews,
                    AVG(confidence) AS avg_confidence,
                    SUM(CASE WHEN LOWER(predicted_label) = 'pneumonia' THEN 1 ELSE 0 END) AS pneumonia_count,
                    SUM(CASE WHEN LOWER(predicted_label) = 'normal' THEN 1 ELSE 0 END) AS normal_count
                FROM analysis_history
                """
            ).fetchone()

        total = int(summary_row["total_reviews"] or 0)
        return {
            "total_reviews": total,
            "pneumonia_count": int(summary_row["pneumonia_count"] or 0),
            "normal_count": int(summary_row["normal_count"] or 0),
            "avg_confidence": float(summary_row["avg_confidence"] or 0.0),
        }


history_service = HistoryService()
