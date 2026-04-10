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


client = TestClient(app)


def make_test_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def clear_history_db_between_tests():
    with history_service._connect() as conn:  # noqa: SLF001 - test-only helper usage.
        conn.execute("DELETE FROM analysis_history")
        conn.commit()
    yield


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


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


def test_gradcam_success() -> None:
    image_bytes = make_test_png_bytes()
    response = client.post("/gradcam", files={"file": ("sample.png", image_bytes, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["heatmap_base64"]
    assert payload["inference_mode"] in {"checkpoint", "placeholder"}
    assert 0.0 <= payload["lung_focus_score"] <= 1.0
    assert 0.0 <= payload["off_lung_attention_ratio"] <= 1.0


def test_model_info_success() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "inference_mode" in payload
    assert "model_arch" in payload
    assert "class_names" in payload


def test_history_endpoints_success() -> None:
    image_bytes = make_test_png_bytes()
    _ = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")})

    summary_response = client.get("/history/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["total_reviews"] >= 1

    history_response = client.get("/history?limit=5")
    assert history_response.status_code == 200
    history = history_response.json()
    assert isinstance(history, list)
    assert len(history) == 1
    assert "predicted_label" in history[0]


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


def test_gradcam_rejects_oversized_upload() -> None:
    oversized = b"a" * (settings.max_upload_bytes + 1)
    response = client.post("/gradcam", files={"file": ("huge.png", oversized, "image/png")})
    assert response.status_code == 413
    assert "max allowed size" in response.json()["detail"].lower()
