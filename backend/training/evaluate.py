from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models

from backend.app.config import settings
from backend.training.data_utils import build_imagefolder_dataset, build_transforms


def main() -> None:
    dataset_root = settings.dataset_root
    checkpoint_path = Path(settings.checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    transforms_map = build_transforms(settings.image_size)
    test_ds = build_imagefolder_dataset(dataset_root, "test", transforms_map["test"])
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = str(checkpoint.get("arch", settings.model_arch)).lower()
    class_names = checkpoint.get("class_names") or checkpoint.get("classes") or test_ds.classes

    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    else:
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names))

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

    accuracy = (correct / total) if total else 0.0
    print(f"Model arch: {arch}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
