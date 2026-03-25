from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models, transforms

from backend.app.config import settings


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded into an image."""


class CheckpointRequiredError(RuntimeError):
    """Raised when strict checkpoint mode is enabled but weights are missing."""


class ModelService:
    """CPU inference service with checkpoint-first loading and explicit fallback mode."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.threshold = settings.default_threshold
        self.default_class_names = settings.class_name_list
        self.default_image_size = settings.image_size
        self.default_arch = settings.model_arch.lower()
        self._checkpoint_path = Path(settings.checkpoint_path)

        self._checkpoint_loaded = False
        self._arch = self.default_arch
        self._class_names = self.default_class_names
        self._image_size = self.default_image_size
        self._model: nn.Module | None = None
        self._transform = self._build_transform(self._image_size)
        self._checkpoint_meta: dict[str, float | int | str] = {}

    @staticmethod
    def _build_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def transform(self) -> transforms.Compose:
        return self._transform

    def _build_model(self, arch: str, num_classes: int) -> nn.Module:
        if arch == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    def _load_checkpoint_if_available(self) -> None:
        if self._checkpoint_loaded:
            return

        if not self._checkpoint_path.exists():
            if settings.require_checkpoint:
                raise CheckpointRequiredError(f"Checkpoint not found at {self._checkpoint_path}")
            return

        checkpoint = torch.load(self._checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        ckpt_arch = str(checkpoint.get("arch", self.default_arch)).lower()
        ckpt_classes = checkpoint.get("class_names") or checkpoint.get("classes") or self.default_class_names
        ckpt_image_size = int(checkpoint.get("image_size", self.default_image_size))

        model = self._build_model(ckpt_arch, len(ckpt_classes))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model = model
        self._arch = ckpt_arch
        self._class_names = [str(name) for name in ckpt_classes]
        self._image_size = ckpt_image_size
        self._transform = self._build_transform(self._image_size)
        self._checkpoint_loaded = True
        self._checkpoint_meta = {
            "best_epoch": int(checkpoint.get("best_epoch")) if checkpoint.get("best_epoch") is not None else None,
            "best_val_acc": float(checkpoint.get("best_val_acc"))
            if checkpoint.get("best_val_acc") is not None
            else None,
            "best_val_loss": float(checkpoint.get("best_val_loss"))
            if checkpoint.get("best_val_loss") is not None
            else None,
        }

    def ensure_ready(self) -> None:
        self._load_checkpoint_if_available()
        if settings.require_checkpoint and not self._checkpoint_loaded:
            raise CheckpointRequiredError(f"Checkpoint not found at {self._checkpoint_path}")

    def _read_image(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("Uploaded file is not a decodable image.") from exc

    def read_image(self, image_bytes: bytes) -> Image.Image:
        return self._read_image(image_bytes)

    @staticmethod
    def _placeholder_logits(image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32)
        signal = float(pixels.mean() / 255.0)
        return torch.tensor([1.0 - signal, signal], dtype=torch.float32)

    def _predict_logits(self, image: Image.Image) -> torch.Tensor:
        self._load_checkpoint_if_available()
        if not self._checkpoint_loaded or self._model is None:
            return self._placeholder_logits(image)

        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self._model(tensor).squeeze(0).cpu()

    def get_model_info(self) -> dict[str, str | bool | float | int | list[str] | None]:
        self._load_checkpoint_if_available()
        return {
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
        }

    def predict(
        self, image_bytes: bytes, threshold: float | None = None
    ) -> dict[str, float | str | bool | dict[str, float]]:
        image = self._read_image(image_bytes)
        logits = self._predict_logits(image)
        decision_threshold = float(self.threshold if threshold is None else threshold)

        probabilities = torch.softmax(logits, dim=0).numpy()
        probability_map = {
            label: float(probabilities[idx]) for idx, label in enumerate(self._class_names)
        }

        positive_label = self._class_names[-1]
        positive_prob = probability_map[positive_label]
        predicted_label = positive_label if positive_prob >= decision_threshold else self._class_names[0]
        confidence = max(probability_map.values())

        return {
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": probability_map,
            "threshold": decision_threshold,
            "inference_mode": self.inference_mode,
            "model_arch": self._arch,
            "checkpoint_loaded": self._checkpoint_loaded,
        }


model_service = ModelService()
