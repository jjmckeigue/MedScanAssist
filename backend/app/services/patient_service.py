from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import settings


class PatientService:
    """CRUD operations for patient profiles and their linked analyses."""

    def __init__(self) -> None:
        self._db_path = Path(settings.history_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    date_of_birth TEXT,
                    medical_record_number TEXT UNIQUE,
                    notes TEXT DEFAULT ''
                )
                """
            )
            conn.commit()

    # ---- Create / Read / Update / Delete ----

    def create(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: str | None = None,
        medical_record_number: str | None = None,
        notes: str = "",
    ) -> dict:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO patients
                    (created_at_utc, updated_at_utc, first_name, last_name,
                     date_of_birth, medical_record_number, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, now, first_name, last_name, date_of_birth, medical_record_number, notes),
            )
            conn.commit()
            return self.get_by_id(cursor.lastrowid)

    def get_by_id(self, patient_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_patients(
        self, search: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        with self._connect() as conn:
            if search:
                like = f"%{search}%"
                rows = conn.execute(
                    """
                    SELECT * FROM patients
                    WHERE first_name LIKE ? OR last_name LIKE ? OR medical_record_number LIKE ?
                    ORDER BY updated_at_utc DESC
                    LIMIT ? OFFSET ?
                    """,
                    (like, like, like, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM patients ORDER BY updated_at_utc DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self, search: str | None = None) -> int:
        with self._connect() as conn:
            if search:
                like = f"%{search}%"
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt FROM patients
                    WHERE first_name LIKE ? OR last_name LIKE ? OR medical_record_number LIKE ?
                    """,
                    (like, like, like),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM patients").fetchone()
        return int(row["cnt"])

    def update(self, patient_id: int, **fields) -> dict | None:
        allowed = {"first_name", "last_name", "date_of_birth", "medical_record_number", "notes"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return self.get_by_id(patient_id)
        updates["updated_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [patient_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE patients SET {set_clause} WHERE id = ?", values)
            conn.commit()
        return self.get_by_id(patient_id)

    def delete(self, patient_id: int) -> bool:
        with self._connect() as conn:
            conn.execute(
                "UPDATE analysis_history SET patient_id = NULL WHERE patient_id = ?",
                (patient_id,),
            )
            cursor = conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ---- Patient-scoped analysis queries ----

    def get_patient_analyses(self, patient_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at_utc, file_name, predicted_label, confidence,
                       threshold, inference_mode, model_arch, checkpoint_loaded,
                       probabilities_json, feedback, patient_id, image_path
                FROM analysis_history
                WHERE patient_id = ?
                ORDER BY id DESC
                """,
                (patient_id,),
            ).fetchall()

        records: list[dict] = []
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
                    "feedback": row["feedback"],
                    "patient_id": row["patient_id"],
                    "image_path": row["image_path"],
                }
            )
        return records

    def get_progression(self, patient_id: int) -> list[dict]:
        """Confidence progression over time for charting."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at_utc, predicted_label, confidence, threshold
                FROM analysis_history
                WHERE patient_id = ?
                ORDER BY id ASC
                """,
                (patient_id,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "created_at_utc": str(row["created_at_utc"]),
                "predicted_label": str(row["predicted_label"]),
                "confidence": float(row["confidence"]),
                "threshold": float(row["threshold"]),
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "created_at_utc": str(row["created_at_utc"]),
            "updated_at_utc": str(row["updated_at_utc"]),
            "first_name": str(row["first_name"]),
            "last_name": str(row["last_name"]),
            "date_of_birth": row["date_of_birth"],
            "medical_record_number": row["medical_record_number"],
            "notes": str(row["notes"] or ""),
        }


patient_service = PatientService()
