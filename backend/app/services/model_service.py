from __future__ import annotations

import json
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.config import settings

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"GIF8": "gif",
    b"RIFF": "webp",
    b"BM": "bmp",
}

MIN_IMAGE_DIMENSION = 1
MAX_IMAGE_DIMENSION = 10000


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded into an image."""


class CheckpointRequiredError(RuntimeError):
    """Raised when strict checkpoint mode is enabled but weights are missing."""


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _preprocess(image: Image.Image, image_size: int) -> np.ndarray:
    """PIL image → float32 NCHW array with ImageNet normalization."""
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis, ...]  # (1, 3, H, W)


def validate_image_magic_bytes(image_bytes: bytes) -> None:
    """Validate that the image bytes start with a known image format magic header."""
    if len(image_bytes) < 4:
        raise InvalidImageError("Uploaded data is too small to be a valid image.")
    for magic, fmt in IMAGE_MAGIC_BYTES.items():
        if image_bytes[: len(magic)] == magic:
            return
    raise InvalidImageError("Uploaded file is not a decodable image.")


def validate_image_dimensions(image: Image.Image) -> None:
    """Reject images with suspicious dimensions."""
    w, h = image.size
    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        raise InvalidImageError(
            f"Image dimensions ({w}x{h}) are too small. "
            f"Minimum is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}."
        )
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise InvalidImageError(
            f"Image dimensions ({w}x{h}) exceed maximum allowed "
            f"({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})."
        )


def strip_exif(image: Image.Image) -> Image.Image:
    """Return a copy of the image with EXIF metadata stripped."""
    data = list(image.getdata())
    clean = Image.new(image.mode, image.size)
    clean.putdata(data)
    return clean


class ModelService:
    """ONNX Runtime inference service with placeholder fallback."""

    def __init__(self) -> None:
        self.threshold = settings.default_threshold
        self.default_class_names = settings.class_name_list
        self.default_image_size = settings.image_size
        self.default_arch = settings.model_arch.lower()
        self._checkpoint_path = Path(settings.checkpoint_path)
        self._model_meta_path = Path(settings.model_meta_path)

        self._checkpoint_loaded = False
        self._arch = self.default_arch
        self._class_names = self.default_class_names
        self._image_size = self.default_image_size
        self._session = None
        self._checkpoint_meta: dict[str, float | int | str] = {}
        self._temperature: float = 1.0
        self._calibration_model = None
        self._calibrated = False

    @property
    def checkpoint_loaded(self) -> bool:
        return self._checkpoint_loaded

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def model_arch(self) -> str:
        return self._arch

    @property
    def inference_mode(self) -> str:
        return "checkpoint" if self._checkpoint_loaded else "placeholder"

    def _load_checkpoint_if_available(self) -> None:
        if self._checkpoint_loaded:
            return

        if not self._checkpoint_path.exists():
            if settings.require_checkpoint:
                raise CheckpointRequiredError(f"Checkpoint not found at {self._checkpoint_path}")
            return

        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self._checkpoint_path),
            providers=["CPUExecutionProvider"],
        )

        meta: dict = {}
        if self._model_meta_path.exists():
            meta = json.loads(self._model_meta_path.read_text())

        self._arch = str(meta.get("arch", self.default_arch)).lower()
        self._class_names = meta.get("class_names", self.default_class_names)
        self._class_names = [str(c) for c in self._class_names]
        self._image_size = int(meta.get("image_size", self.default_image_size))
        self._temperature = float(meta.get("temperature", 1.0))
        if "threshold" in meta:
            self.threshold = float(meta["threshold"])
        self._checkpoint_loaded = True
        self._checkpoint_meta = {
            "best_epoch": meta.get("best_epoch"),
            "best_val_acc": meta.get("best_val_acc"),
            "best_val_loss": meta.get("best_val_loss"),
            "temperature": self._temperature,
        }

        calibration_path = self._checkpoint_path.parent / "calibration_model.pkl"
        if calibration_path.exists():
            try:
                with calibration_path.open("rb") as fp:
                    self._calibration_model = pickle.load(fp)  # noqa: S301
                self._calibrated = True
            except Exception:
                self._calibration_model = None
                self._calibrated = False

    def ensure_ready(self) -> None:
        self._load_checkpoint_if_available()
        if settings.require_checkpoint and not self._checkpoint_loaded:
            raise CheckpointRequiredError(f"Checkpoint not found at {self._checkpoint_path}")

    def _read_image(self, image_bytes: bytes) -> Image.Image:
        validate_image_magic_bytes(image_bytes)
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("Uploaded file is not a decodable image.") from exc
        validate_image_dimensions(img)
        img = strip_exif(img)
        return img

    def read_image(self, image_bytes: bytes) -> Image.Image:
        return self._read_image(image_bytes)

    @staticmethod
    def _placeholder_logits(image: Image.Image) -> np.ndarray:
        pixels = np.asarray(image, dtype=np.float32)
        signal = float(pixels.mean() / 255.0)
        return np.array([1.0 - signal, signal], dtype=np.float32)

    def _run_onnx(self, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run ONNX session.  Returns (logits, activations_or_None)."""
        assert self._session is not None
        output_names = [o.name for o in self._session.get_outputs()]
        results = self._session.run(output_names, {"input": input_tensor})
        logits = results[0].squeeze(0)  # (num_classes,)
        activations = results[1] if len(results) > 1 else None  # (1, C, H, W)
        return logits, activations

    def _predict_logits(self, image: Image.Image) -> np.ndarray:
        self._load_checkpoint_if_available()
        if not self._checkpoint_loaded or self._session is None:
            return self._placeholder_logits(image)
        tensor = _preprocess(image, self._image_size)
        logits, _ = self._run_onnx(tensor)
        return logits

    def _predict_logits_tta(self, image: Image.Image) -> np.ndarray:
        """Average logits across augmented views (PIL-based TTA)."""
        self._load_checkpoint_if_available()
        if not self._checkpoint_loaded or self._session is None:
            return self._placeholder_logits(image)

        views: list[Image.Image] = [
            image,
            ImageOps.mirror(image),
            image.rotate(5, resample=Image.BILINEAR, expand=False),
            image.rotate(-5, resample=Image.BILINEAR, expand=False),
        ]
        all_logits = []
        for v in views:
            tensor = _preprocess(v, self._image_size)
            logits, _ = self._run_onnx(tensor)
            all_logits.append(logits)
        return np.mean(all_logits, axis=0)

    def predict_with_activations(
        self, image: Image.Image
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (logits, activations) for Eigen-CAM.  None when in placeholder mode."""
        self._load_checkpoint_if_available()
        if not self._checkpoint_loaded or self._session is None:
            return None, None
        tensor = _preprocess(image, self._image_size)
        logits, activations = self._run_onnx(tensor)
        return logits, activations

    def get_model_info(self) -> dict[str, str | bool | float | int | list[str] | None]:
        self._load_checkpoint_if_available()
        info: dict[str, str | bool | float | int | list[str] | None] = {
            "inference_mode": self.inference_mode,
            "model_arch": self._arch,
            "checkpoint_loaded": self._checkpoint_loaded,
            "class_names": self._class_names,
            "image_size": self._image_size,
            "default_threshold": float(self.threshold),
            "checkpoint_path": str(self._checkpoint_path),
            "best_epoch": self._checkpoint_meta.get("best_epoch"),
            "best_val_acc": self._checkpoint_meta.get("best_val_acc"),
            "best_val_loss": self._checkpoint_meta.get("best_val_loss"),
            "temperature": self._checkpoint_meta.get("temperature", 1.0),
        }
        if not self._checkpoint_loaded:
            info["placeholder_warning"] = (
                "No trained model checkpoint is loaded. "
                "Predictions are placeholder values and must NOT be used for clinical decisions."
            )
        return info

    def _apply_calibration(self, probability_map: dict[str, float]) -> dict[str, float]:
        """Apply isotonic calibration to the positive-class probability if available."""
        if self._calibration_model is None:
            return probability_map

        positive_label = self._class_names[-1]
        raw_prob = probability_map[positive_label]
        calibrated_prob = float(
            self._calibration_model.predict(np.array([raw_prob]))[0]
        )
        calibrated_prob = max(0.0, min(1.0, calibrated_prob))

        calibrated_map = dict(probability_map)
        calibrated_map[positive_label] = calibrated_prob
        if len(self._class_names) == 2:
            calibrated_map[self._class_names[0]] = 1.0 - calibrated_prob
        return calibrated_map

    def predict(
        self, image_bytes: bytes, threshold: float | None = None, tta: bool = False
    ) -> dict[str, float | str | bool | dict[str, float]]:
        image = self._read_image(image_bytes)
        logits = self._predict_logits_tta(image) if tta else self._predict_logits(image)
        decision_threshold = float(self.threshold if threshold is None else threshold)

        scaled_logits = logits / self._temperature if self._temperature > 0 else logits
        probabilities = _softmax(scaled_logits)
        probability_map = {
            label: float(probabilities[idx]) for idx, label in enumerate(self._class_names)
        }

        probability_map = self._apply_calibration(probability_map)

        positive_label = self._class_names[-1]
        positive_prob = probability_map[positive_label]
        predicted_label = positive_label if positive_prob >= decision_threshold else self._class_names[0]
        confidence = max(probability_map.values())

        result = {
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": probability_map,
            "threshold": decision_threshold,
            "inference_mode": self.inference_mode,
            "model_arch": self._arch,
            "checkpoint_loaded": self._checkpoint_loaded,
            "calibrated": self._calibrated,
        }
        if not self._checkpoint_loaded:
            result["placeholder_warning"] = (
                "No trained model checkpoint is loaded. "
                "Predictions are placeholder values and must NOT be used for clinical decisions."
            )
        return result


model_service = ModelService()
