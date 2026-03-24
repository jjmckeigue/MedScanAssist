from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from backend.app.config import settings
from backend.training.data_utils import build_imagefolder_dataset, build_transforms


def build_model(num_classes: int, arch: str = "densenet121") -> nn.Module:
    if arch.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def main() -> None:
    dataset_root = settings.dataset_root
    transforms_map = build_transforms(settings.image_size)

    train_ds = build_imagefolder_dataset(dataset_root, "train", transforms_map["train"])
    val_ds = build_imagefolder_dataset(dataset_root, "val", transforms_map["val"])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(train_ds.classes), arch=settings.model_arch).to("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Minimal baseline loop. Replace with full epoch/metric/checkpoint workflow.
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            _ = model(images)
            break

    checkpoint_dir = Path("./backend/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "best_model.pt"
    torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, save_path)
    print(f"Saved baseline checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
