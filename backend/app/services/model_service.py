from io import BytesIO

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

from backend.app.config import settings


class ModelService:
    """
    CPU baseline inference service.

    This starts with a deterministic placeholder flow and can be switched to real
    model weights by implementing checkpoint loading and forward pass logic.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.class_names = settings.class_name_list
        self.threshold = settings.default_threshold
        self.image_size = settings.image_size
        self._transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._model = self._init_model()

    def _init_model(self) -> torch.nn.Module:
        if settings.model_arch.lower() == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.class_names))
        else:
            model = models.densenet121(weights=None)
            model.classifier = torch.nn.Linear(model.classifier.in_features, len(self.class_names))

        model.to(self.device)
        model.eval()
        return model

    def _read_image(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image

    def _placeholder_logits(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32)
        signal = float(pixels.mean() / 255.0)
        logits = torch.tensor([1.0 - signal, signal], dtype=torch.float32)
        return logits

    def predict(self, image_bytes: bytes) -> dict[str, float | str | dict[str, float]]:
        image = self._read_image(image_bytes)
        _ = self._transform(image).unsqueeze(0).to(self.device)

        logits = self._placeholder_logits(image)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        probability_map = {label: float(probabilities[idx]) for idx, label in enumerate(self.class_names)}

        positive_label = self.class_names[-1]
        positive_prob = probability_map[positive_label]
        predicted_label = positive_label if positive_prob >= self.threshold else self.class_names[0]
        confidence = max(probability_map.values())

        return {
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": probability_map,
            "threshold": float(self.threshold),
        }


model_service = ModelService()
