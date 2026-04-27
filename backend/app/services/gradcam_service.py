from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from backend.app.services.model_service import _softmax, model_service


class GradCamService:
    """Explainability service using Eigen-CAM (gradient-free, ONNX-compatible)."""

    @staticmethod
    def _synthetic_heatmap(height: int, width: int) -> np.ndarray:
        y = np.linspace(-1.0, 1.0, height).reshape(-1, 1)
        x = np.linspace(-1.0, 1.0, width).reshape(1, -1)
        radius = np.sqrt(x**2 + y**2)
        heatmap = np.clip(1.0 - radius, 0.0, 1.0)
        return heatmap.astype(np.float32)

    @staticmethod
    def _overlay_image(img_np: np.ndarray, heatmap: np.ndarray) -> str:
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        buffered = BytesIO()
        Image.fromarray(overlay).save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def _eigencam_heatmap(activations: np.ndarray, height: int, width: int) -> np.ndarray | None:
        """Compute Eigen-CAM from last-conv activations (no gradients needed).

        Uses the first principal component of the activation tensor as the
        saliency map — a well-established gradient-free class activation mapping technique.
        """
        acts = activations[0]  # (C, H_feat, W_feat)
        c, h_f, w_f = acts.shape
        reshaped = acts.reshape(c, h_f * w_f)  # (C, H*W)

        try:
            _, _, vt = np.linalg.svd(reshaped, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        cam = vt[0].reshape(h_f, w_f)

        # The SVD component can be arbitrarily signed; flip so the dominant
        # direction is positive, then ReLU to keep only salient regions.
        if cam.sum() < 0:
            cam = -cam
        cam = np.maximum(cam, 0.0)

        if cam.max() <= 0:
            return None

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        return cam.astype(np.float32)

    def _eigencam_heatmap_with_logits(
        self, image: Image.Image
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run ONNX model, return (heatmap, logits).  None pair in placeholder mode."""
        logits, activations = model_service.predict_with_activations(image)
        if logits is None or activations is None:
            return None, None

        heatmap = self._eigencam_heatmap(activations, image.height, image.width)
        return heatmap, logits

    @staticmethod
    def _heuristic_lung_roi_mask(height: int, width: int) -> np.ndarray:
        y_start, y_end = int(height * 0.15), int(height * 0.9)
        x_start, x_end = int(width * 0.15), int(width * 0.85)
        mask = np.zeros((height, width), dtype=np.float32)
        mask[y_start:y_end, x_start:x_end] = 1.0
        return mask

    def _explainability_stats(self, heatmap: np.ndarray) -> dict[str, float | str | None]:
        heatmap_sum = float(np.sum(heatmap)) + 1e-8
        roi_mask = self._heuristic_lung_roi_mask(heatmap.shape[0], heatmap.shape[1])
        lung_attention = float(np.sum(heatmap * roi_mask))
        lung_focus = max(0.0, min(1.0, lung_attention / heatmap_sum))
        off_lung = max(0.0, min(1.0, 1.0 - lung_focus))

        warning = None
        if lung_focus < 0.5:
            warning = (
                "Eigen-CAM attention is predominantly outside the expected lung region. "
                "Treat this prediction cautiously and verify with clinical review."
            )
        elif lung_focus < 0.65:
            warning = (
                "Eigen-CAM attention is only partially concentrated in the lung region. "
                "Interpret with caution."
            )

        return {
            "lung_focus_score": float(lung_focus),
            "off_lung_attention_ratio": float(off_lung),
            "explainability_warning": warning,
        }

    def _logits_to_prediction(
        self, logits: np.ndarray, threshold: float
    ) -> tuple[str, float, dict[str, float], str]:
        """Convert raw logits to prediction fields."""
        temperature = model_service._temperature  # noqa: SLF001
        scaled = logits / temperature if temperature > 0 else logits
        probs = _softmax(scaled)
        class_names = model_service.class_names
        prob_map = {label: float(probs[i]) for i, label in enumerate(class_names)}
        positive_label = class_names[-1]
        positive_prob = prob_map[positive_label]
        predicted_label = positive_label if positive_prob >= threshold else class_names[0]
        confidence = float(max(prob_map.values()))
        return predicted_label, confidence, prob_map, model_service.inference_mode

    def build_overlay_with_prediction(
        self, image_bytes: bytes, threshold: float | None = None
    ) -> dict:
        """Predict + Eigen-CAM in a single forward pass."""
        image = model_service.read_image(image_bytes)
        img_np = np.array(image)

        heatmap, logits = self._eigencam_heatmap_with_logits(image)
        gradcam_mode = "real"

        if heatmap is None:
            heatmap = self._synthetic_heatmap(img_np.shape[0], img_np.shape[1])
            gradcam_mode = "synthetic"

        decision_threshold = float(model_service.threshold if threshold is None else threshold)

        if logits is not None:
            predicted_label, confidence, prob_map, inference_mode = self._logits_to_prediction(
                logits, decision_threshold
            )
        else:
            pred = model_service.predict(image_bytes, threshold=threshold)
            predicted_label = str(pred["predicted_label"])
            confidence = float(pred["confidence"])
            prob_map = pred["probabilities"]
            inference_mode = str(pred["inference_mode"])

        encoded = self._overlay_image(img_np, heatmap)
        explainability = self._explainability_stats(heatmap)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": prob_map,
            "threshold": decision_threshold,
            "inference_mode": inference_mode,
            "model_arch": model_service.model_arch,
            "checkpoint_loaded": model_service.checkpoint_loaded,
            "heatmap_base64": encoded,
            "gradcam_mode": gradcam_mode,
            "lung_focus_score": float(explainability["lung_focus_score"]),
            "off_lung_attention_ratio": float(explainability["off_lung_attention_ratio"]),
            "explainability_warning": explainability["explainability_warning"],
        }

    def build_overlay_base64(self, image_bytes: bytes) -> dict[str, str | float | bool | None]:
        image = model_service.read_image(image_bytes)
        img_np = np.array(image)

        heatmap, logits = self._eigencam_heatmap_with_logits(image)
        gradcam_mode = "real"

        if heatmap is None:
            heatmap = self._synthetic_heatmap(img_np.shape[0], img_np.shape[1])
            gradcam_mode = "synthetic"

        if logits is not None:
            predicted_label, confidence, _, inference_mode = self._logits_to_prediction(
                logits, float(model_service.threshold)
            )
        else:
            pred = model_service.predict(image_bytes)
            predicted_label = str(pred["predicted_label"])
            confidence = float(pred["confidence"])
            inference_mode = str(pred["inference_mode"])

        encoded = self._overlay_image(img_np, heatmap)
        explainability = self._explainability_stats(heatmap)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "heatmap_base64": encoded,
            "inference_mode": inference_mode,
            "model_arch": model_service.model_arch,
            "checkpoint_loaded": model_service.checkpoint_loaded,
            "gradcam_mode": gradcam_mode,
            "lung_focus_score": float(explainability["lung_focus_score"]),
            "off_lung_attention_ratio": float(explainability["off_lung_attention_ratio"]),
            "explainability_warning": explainability["explainability_warning"],
        }


gradcam_service = GradCamService()
