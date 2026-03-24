import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from backend.app.config import settings
from backend.training.data_utils import build_imagefolder_dataset, build_transforms

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional visualization dependency.
    plt = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def freeze_feature_extractor(model: nn.Module, arch: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if arch == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_all_layers(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def build_model(num_classes: int, arch: str = "densenet121") -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(trainable_params, lr=lr)


def run_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer | None
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        if is_train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    avg_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def write_metrics_csv(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_training_plot(history: list[dict[str, float]], out_path: Path) -> None:
    if plt is None or not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_acc"]) for row in history]
    val_acc = [float(row["val_acc"]) for row in history]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, label="train_loss")
        axes[0].plot(epochs, val_loss, label="val_loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, train_acc, label="train_acc")
        axes[1].plot(epochs, val_acc, label="val_acc")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot generation failed ({exc}); training metrics CSV was still saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transfer-learning CXR classifier.")
    parser.add_argument("--epochs-head", type=int, default=3, help="Epochs with frozen backbone.")
    parser.add_argument("--epochs-finetune", type=int, default=2, help="Epochs after unfreezing.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = settings.dataset_root
    arch = settings.model_arch.lower()
    transforms_map = build_transforms(settings.image_size)

    train_ds = build_imagefolder_dataset(dataset_root, "train", transforms_map["train"])
    val_ds = build_imagefolder_dataset(dataset_root, "val", transforms_map["val"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(train_ds.classes), arch=arch).to("cpu")
    criterion = nn.CrossEntropyLoss()
    freeze_feature_extractor(model, arch)
    optimizer = build_optimizer(model, args.lr_head)

    total_epochs = args.epochs_head + args.epochs_finetune
    history: list[dict[str, float]] = []
    best_val_acc = -1.0

    checkpoint_dir = Path("./backend/checkpoints")
    artifact_dir = Path("./backend/artifacts")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    for epoch in range(1, total_epochs + 1):
        if epoch == args.epochs_head + 1 and args.epochs_finetune > 0:
            # Phase 2 of transfer learning: unfreeze all weights for fine-tuning.
            unfreeze_all_layers(model)
            optimizer = build_optimizer(model, args.lr_finetune)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch}/{total_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": train_ds.classes,
                    "arch": arch,
                    "image_size": settings.image_size,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )

    metrics_csv = artifact_dir / "training_metrics.csv"
    metrics_plot = artifact_dir / "training_curves.png"
    write_metrics_csv(history, metrics_csv)
    save_training_plot(history, metrics_plot)

    print(f"Saved best checkpoint to: {checkpoint_path}")
    print(f"Saved metrics CSV to: {metrics_csv}")
    if plt is None:
        print("Matplotlib not installed; skipped training curve image.")
    else:
        print(f"Saved training curves to: {metrics_plot}")


if __name__ == "__main__":
    main()
