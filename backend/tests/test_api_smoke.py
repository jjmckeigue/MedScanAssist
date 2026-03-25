from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


client = TestClient(app)


def make_test_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def test_model_info_success() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "inference_mode" in payload
    assert "model_arch" in payload
    assert "class_names" in payload


def test_predict_rejects_non_image() -> None:
    response = client.post("/predict", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_rejects_corrupted_image_bytes() -> None:
    response = client.post("/predict", files={"file": ("broken.png", b"not-a-real-png", "image/png")})
    assert response.status_code == 400
    assert "decodable image" in response.json()["detail"].lower()
