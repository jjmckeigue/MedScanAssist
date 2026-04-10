from __future__ import annotations

from datetime import datetime, timezone
import json
import math
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
            # Add feedback column if it doesn't exist (safe migration for existing DBs).
            try:
                conn.execute("ALTER TABLE analysis_history ADD COLUMN feedback TEXT DEFAULT NULL")
            except Exception:
                pass
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

    def set_feedback(self, record_id: int, feedback: str) -> bool:
        """Set clinician feedback ('correct', 'incorrect', or None to clear) on a prediction."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE analysis_history SET feedback = ? WHERE id = ?",
                (feedback if feedback else None, record_id),
            )
            conn.commit()
            return cursor.rowcount > 0

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
                    probabilities_json,
                    feedback
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
                    "feedback": row["feedback"],
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


    def get_drift_report(self, baseline_count: int = 200, recent_count: int = 50) -> dict:
        """Compute Population Stability Index (PSI) between baseline and recent predictions."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM analysis_history").fetchone()["cnt"]
            if total < baseline_count + recent_count:
                return {
                    "psi": None,
                    "drift_detected": False,
                    "message": f"Insufficient data: need at least {baseline_count + recent_count} records, have {total}.",
                    "baseline_count": min(total, baseline_count),
                    "recent_count": 0,
                    "bins": [],
                }

            baseline_rows = conn.execute(
                "SELECT confidence FROM analysis_history ORDER BY id ASC LIMIT ?",
                (baseline_count,),
            ).fetchall()
            recent_rows = conn.execute(
                "SELECT confidence FROM analysis_history ORDER BY id DESC LIMIT ?",
                (recent_count,),
            ).fetchall()

        baseline_confs = [float(r["confidence"]) for r in baseline_rows]
        recent_confs = [float(r["confidence"]) for r in recent_rows]

        n_bins = 10
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        eps = 1e-6

        def bin_proportions(values):
            counts = [0] * n_bins
            for v in values:
                idx = min(int(v * n_bins), n_bins - 1)
                counts[idx] += 1
            total = len(values)
            return [(c / total) + eps for c in counts]

        baseline_props = bin_proportions(baseline_confs)
        recent_props = bin_proportions(recent_confs)

        psi = 0.0
        bins = []
        for i in range(n_bins):
            diff = recent_props[i] - baseline_props[i]
            ratio = recent_props[i] / baseline_props[i]
            psi_bin = diff * math.log(ratio)
            psi += psi_bin
            bins.append({
                "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "baseline_prop": round(baseline_props[i] - eps, 4),
                "recent_prop": round(recent_props[i] - eps, 4),
                "psi_contribution": round(psi_bin, 6),
            })

        drift_detected = psi > 0.2  # standard PSI threshold

        return {
            "psi": round(psi, 6),
            "drift_detected": drift_detected,
            "message": "Significant distribution shift detected." if drift_detected else "No significant drift.",
            "baseline_count": baseline_count,
            "recent_count": recent_count,
            "bins": bins,
        }


history_service = HistoryService()
