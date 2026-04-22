import os
from pathlib import Path
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

TEST_HISTORY_DB = Path("./backend/artifacts/test_history.db")
if TEST_HISTORY_DB.exists():
    TEST_HISTORY_DB.unlink()
os.environ["HISTORY_DB_PATH"] = str(TEST_HISTORY_DB)

from backend.app.main import app
from backend.app.config import settings
from backend.app.services.history_service import history_service
from backend.app.services.patient_service import patient_service


client = TestClient(app)


def make_test_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def clear_db_between_tests():
    with history_service._connect() as conn:  # noqa: SLF001
        conn.execute("DELETE FROM analysis_history")
        conn.commit()
    with patient_service._connect() as conn:  # noqa: SLF001
        conn.execute("DELETE FROM patients")
        conn.commit()
    yield


# ─── Health ───

def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code in (200, 503)
    payload = response.json()
    assert payload["status"] in ("ok", "degraded")
    assert "checks" in payload
    assert payload["checks"]["database"] == "ok"


# ─── Predict ───

def test_predict_success() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert "predicted_label" in payload
    assert "probabilities" in payload
    assert "inference_mode" in payload
    assert payload["inference_mode"] in {"checkpoint", "placeholder"}
    assert isinstance(payload.get("analysis_id"), int)


def test_predict_threshold_override_changes_response_threshold() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/predict?threshold=0.9", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert abs(payload["threshold"] - 0.9) < 1e-9


def test_predict_invalid_threshold_rejected() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/predict?threshold=1.5", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 422


def test_predict_tta_parameter_accepted() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/predict?tta=true", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert "predicted_label" in payload


def test_predict_rejects_non_image() -> None:
    response = client.post("/predict", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_rejects_corrupted_image_bytes() -> None:
    response = client.post("/predict", files={"file": ("broken.png", b"not-a-real-png", "image/png")})
    assert response.status_code == 400
    assert "decodable image" in response.json()["detail"].lower()


def test_predict_rejects_oversized_upload() -> None:
    oversized = b"a" * (settings.max_upload_bytes + 1)
    response = client.post("/predict", files={"file": ("huge.png", oversized, "image/png")})
    assert response.status_code == 413
    assert "max allowed size" in response.json()["detail"].lower()


# ─── Analyze (combined predict + Grad-CAM) ───

def test_analyze_success() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/analyze", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert "predicted_label" in payload
    assert "probabilities" in payload
    assert "heatmap_base64" in payload
    assert payload["inference_mode"] in {"checkpoint", "placeholder"}
    assert payload["gradcam_mode"] in {"real", "synthetic"}
    assert 0.0 <= payload["lung_focus_score"] <= 1.0
    assert isinstance(payload.get("analysis_id"), int)


def test_analyze_threshold_override() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/analyze?threshold=0.9", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 200
    assert abs(response.json()["threshold"] - 0.9) < 1e-9


def test_analyze_rejects_non_image() -> None:
    response = client.post("/analyze", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert response.status_code == 400


# ─── Grad-CAM ───

def test_gradcam_success() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/gradcam", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["heatmap_base64"]
    assert payload["inference_mode"] in {"checkpoint", "placeholder"}
    assert payload["gradcam_mode"] in {"real", "synthetic"}
    assert 0.0 <= payload["lung_focus_score"] <= 1.0
    assert 0.0 <= payload["off_lung_attention_ratio"] <= 1.0


def test_gradcam_rejects_oversized_upload() -> None:
    oversized = b"a" * (settings.max_upload_bytes + 1)
    response = client.post("/gradcam", files={"file": ("huge.png", oversized, "image/png")})
    assert response.status_code == 413
    assert "max allowed size" in response.json()["detail"].lower()


# ─── Model Info ───

def test_model_info_success() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "inference_mode" in payload
    assert "model_arch" in payload
    assert "class_names" in payload
    assert "temperature" in payload
    assert isinstance(payload["temperature"], (int, float))


# ─── History ───

def test_history_endpoints_success() -> None:
    image_bytes = make_test_png_bytes()
    _ = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")})

    summary_response = client.get("/history/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["total_reviews"] >= 1
    assert "feedback_correct" in summary
    assert "feedback_incorrect" in summary
    assert "feedback_accuracy" in summary

    history_response = client.get("/history?limit=5")
    assert history_response.status_code == 200
    history = history_response.json()
    assert isinstance(history, list)
    assert len(history) == 1
    assert "predicted_label" in history[0]
    assert "feedback" in history[0]


# ─── Feedback ───

def test_feedback_correct_and_clear() -> None:
    image_bytes = make_test_png_bytes()
    pred = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")}).json()
    record_id = pred["analysis_id"]

    response = client.post(f"/history/{record_id}/feedback", json={"feedback": "correct"})
    assert response.status_code == 200
    assert response.json()["feedback"] == "correct"

    history = client.get("/history?limit=1").json()
    assert history[0]["feedback"] == "correct"

    response = client.post(f"/history/{record_id}/feedback", json={"feedback": "clear"})
    assert response.status_code == 200
    assert response.json()["feedback"] is None


def test_feedback_incorrect() -> None:
    image_bytes = make_test_png_bytes()
    pred = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")}).json()
    record_id = pred["analysis_id"]

    response = client.post(f"/history/{record_id}/feedback", json={"feedback": "incorrect"})
    assert response.status_code == 200
    assert response.json()["feedback"] == "incorrect"


def test_feedback_invalid_value_rejected() -> None:
    image_bytes = make_test_png_bytes()
    pred = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")}).json()
    record_id = pred["analysis_id"]

    response = client.post(f"/history/{record_id}/feedback", json={"feedback": "maybe"})
    assert response.status_code == 422


def test_feedback_nonexistent_record() -> None:
    response = client.post("/history/999999/feedback", json={"feedback": "correct"})
    assert response.status_code == 404


def test_feedback_accuracy_in_summary() -> None:
    image_bytes = make_test_png_bytes()

    pred1 = client.post("/predict", files={"file": ("a.png", image_bytes, "image/png")}).json()
    pred2 = client.post("/predict", files={"file": ("b.png", image_bytes, "image/png")}).json()
    pred3 = client.post("/predict", files={"file": ("c.png", image_bytes, "image/png")}).json()

    client.post(f"/history/{pred1['analysis_id']}/feedback", json={"feedback": "correct"})
    client.post(f"/history/{pred2['analysis_id']}/feedback", json={"feedback": "correct"})
    client.post(f"/history/{pred3['analysis_id']}/feedback", json={"feedback": "incorrect"})

    summary = client.get("/history/summary").json()
    assert summary["feedback_correct"] == 2
    assert summary["feedback_incorrect"] == 1
    assert abs(summary["feedback_accuracy"] - (2 / 3)) < 1e-3


# ─── Drift ───

def test_drift_insufficient_data() -> None:
    response = client.get("/history/drift")
    assert response.status_code == 200
    payload = response.json()
    assert payload["psi"] is None
    assert payload["drift_detected"] is False
    assert "insufficient" in payload["message"].lower()


def test_drift_with_enough_data() -> None:
    image_bytes = make_test_png_bytes()
    for _ in range(30):
        client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")})

    response = client.get("/history/drift?baseline_count=15&recent_count=10")
    assert response.status_code == 200
    payload = response.json()
    assert payload["psi"] is not None
    assert isinstance(payload["drift_detected"], bool)
    assert isinstance(payload["bins"], list)
    assert len(payload["bins"]) == 10


# ─── Patients CRUD ───

def test_patient_create_and_get() -> None:
    resp = client.post("/patients", json={
        "first_name": "John", "last_name": "Doe",
        "date_of_birth": "1990-01-15", "medical_record_number": "MRN-T001",
    })
    assert resp.status_code == 201
    patient = resp.json()
    assert patient["first_name"] == "John"
    assert patient["last_name"] == "Doe"
    pid = patient["id"]

    detail = client.get(f"/patients/{pid}")
    assert detail.status_code == 200
    assert detail.json()["first_name"] == "John"


def test_patient_list_and_search() -> None:
    client.post("/patients", json={"first_name": "Alice", "last_name": "Smith"})
    client.post("/patients", json={"first_name": "Bob", "last_name": "Jones"})
    resp = client.get("/patients")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 2

    resp = client.get("/patients?search=Alice")
    assert resp.status_code == 200
    names = [p["first_name"] for p in resp.json()["patients"]]
    assert "Alice" in names


def test_patient_update() -> None:
    create = client.post("/patients", json={"first_name": "Ed", "last_name": "Update"})
    pid = create.json()["id"]
    resp = client.put(f"/patients/{pid}", json={"first_name": "Edward"})
    assert resp.status_code == 200
    assert resp.json()["first_name"] == "Edward"


def test_patient_delete() -> None:
    create = client.post("/patients", json={"first_name": "Del", "last_name": "Ete"})
    pid = create.json()["id"]
    resp = client.delete(f"/patients/{pid}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert client.get(f"/patients/{pid}").status_code == 404


def test_patient_not_found() -> None:
    assert client.get("/patients/999999").status_code == 404


def test_patient_duplicate_mrn_rejected() -> None:
    client.post("/patients", json={
        "first_name": "A", "last_name": "B", "medical_record_number": "DUP-001",
    })
    resp = client.post("/patients", json={
        "first_name": "C", "last_name": "D", "medical_record_number": "DUP-001",
    })
    assert resp.status_code == 409


def test_patient_progression() -> None:
    create = client.post("/patients", json={"first_name": "Prog", "last_name": "Test"})
    pid = create.json()["id"]
    resp = client.get(f"/patients/{pid}/progression")
    assert resp.status_code == 200
    assert resp.json()["patient_id"] == pid


# ─── Image Serving ───

def test_image_not_found() -> None:
    resp = client.get("/images/nonexistent.png")
    assert resp.status_code == 404


def test_image_path_traversal_blocked() -> None:
    resp = client.get("/images/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code == 404


# ─── Analyze with Invalid Patient ───

def test_analyze_with_invalid_patient_id() -> None:
    image_bytes = make_test_png_bytes()
    resp = client.post(
        "/analyze?patient_id=999999",
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 404
    assert "patient" in resp.json()["detail"].lower()


# ─── Corrupted Upload ───

def test_analyze_rejects_corrupted_image() -> None:
    resp = client.post(
        "/analyze",
        files={"file": ("bad.png", b"not-a-real-png", "image/png")},
    )
    assert resp.status_code == 400
