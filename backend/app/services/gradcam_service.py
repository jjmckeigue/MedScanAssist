import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

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
    def _find_last_conv(model: nn.Module) -> nn.Module | None:
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _real_gradcam_heatmap(self, image: Image.Image) -> np.ndarray | None:
        if not model_service.checkpoint_loaded or model_service.model is None:
            return None

        model = model_service.model
        target_layer = self._find_last_conv(model)
        if target_layer is None:
            return None

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def forward_hook(_module, _inp, output):
            activations["value"] = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        try:
            tensor = model_service.transform(image).unsqueeze(0).to(model_service.device)
            tensor.requires_grad_(True)

            model.zero_grad(set_to_none=True)
            logits = model(tensor)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            logits[0, pred_idx].backward()

            if "value" not in activations or "value" not in gradients:
                return None

            grads = gradients["value"]  # [1, C, H, W]
            acts = activations["value"]  # [1, C, H, W]
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            cam = cam.squeeze().cpu().numpy()

            if np.max(cam) <= 0:
                return None
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
            return cam.astype(np.float32)
        finally:
            handle_fwd.remove()
            handle_bwd.remove()

    @staticmethod
    def _heuristic_lung_roi_mask(height: int, width: int) -> np.ndarray:
        # Conservative thoracic ROI (excludes shoulders/corners where shortcut cues often appear).
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
                "Grad-CAM attention is predominantly outside the expected lung region. "
                "Treat this prediction cautiously and verify with clinical review."
            )
        elif lung_focus < 0.65:
            warning = (
                "Grad-CAM attention is only partially concentrated in the lung region. "
                "Interpret with caution."
            )

        return {
            "lung_focus_score": float(lung_focus),
            "off_lung_attention_ratio": float(off_lung),
            "explainability_warning": warning,
        }

    def build_overlay_base64(self, image_bytes: bytes) -> dict[str, str | float | bool | None]:
        pred = model_service.predict(image_bytes)
        image = model_service.read_image(image_bytes)
        img_np = np.array(image)

        # Prefer true Grad-CAM when checkpoint-backed model is available.
        heatmap = self._real_gradcam_heatmap(image)
        if heatmap is None:
            heatmap = self._synthetic_heatmap(img_np.shape[0], img_np.shape[1])

        encoded = self._overlay_image(img_np, heatmap)
        explainability = self._explainability_stats(heatmap)

        return {
            "predicted_label": str(pred["predicted_label"]),
            "confidence": float(pred["confidence"]),
            "heatmap_base64": encoded,
            "inference_mode": str(pred["inference_mode"]),
            "model_arch": str(pred["model_arch"]),
            "checkpoint_loaded": bool(pred["checkpoint_loaded"]),
            "lung_focus_score": float(explainability["lung_focus_score"]),
            "off_lung_attention_ratio": float(explainability["off_lung_attention_ratio"]),
            "explainability_warning": explainability["explainability_warning"],
        }


gradcam_service = GradCamService()
