import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from backend.app.services.model_service import model_service


class GradCamService:
    """
    Grad-CAM service scaffold.

    For v1 baseline, this generates a center-focused synthetic heatmap overlay.
    Replace `_synthetic_heatmap` with real Grad-CAM activation maps once model
    forward hooks are integrated.
    """

    @staticmethod
    def _synthetic_heatmap(height: int, width: int) -> np.ndarray:
        y = np.linspace(-1.0, 1.0, height).reshape(-1, 1)
        x = np.linspace(-1.0, 1.0, width).reshape(1, -1)
        radius = np.sqrt(x**2 + y**2)
        heatmap = np.clip(1.0 - radius, 0.0, 1.0)
        return heatmap.astype(np.float32)

    def build_overlay_base64(self, image_bytes: bytes) -> dict[str, str | float]:
        pred = model_service.predict(image_bytes)

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        heatmap = self._synthetic_heatmap(img_np.shape[0], img_np.shape[1])
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        buffered = BytesIO()
        Image.fromarray(overlay).save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "predicted_label": str(pred["predicted_label"]),
            "confidence": float(pred["confidence"]),
            "heatmap_base64": encoded,
        }


gradcam_service = GradCamService()
