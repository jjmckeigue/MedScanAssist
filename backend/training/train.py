import argparse
import csv
from pathlib import Path
import random

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_classifier_head(model: nn.Module, arch: str) -> None:
    if arch == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True


def enable_last_block(model: nn.Module, arch: str) -> None:
    if arch == "resnet50":
        for param in model.layer4.parameters():
            param.requires_grad = True
    else:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True


def freeze_feature_extractor(model: nn.Module, arch: str) -> None:
    freeze_all(model)
    enable_classifier_head(model, arch)


def unfreeze_last_block_and_head(model: nn.Module, arch: str) -> None:
    freeze_all(model)
    enable_classifier_head(model, arch)
    enable_last_block(model, arch)


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


def build_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )


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


def save_training_plot(history: list[dict[str, float]], out_path: Path, best_epoch: int) -> None:
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
        axes[0].axvline(best_epoch, color="green", linestyle="--", label="best_epoch")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, train_acc, label="train_acc")
        axes[1].plot(epochs, val_acc, label="val_acc")
        axes[1].axvline(best_epoch, color="green", linestyle="--", label="best_epoch")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot generation failed ({exc}); training metrics CSV was still saved.")


def save_confusion_matrix_artifacts(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_dir: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_csv_path = out_dir / "confusion_matrix.csv"
    with cm_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred", *class_names])
        for row_idx, row in enumerate(cm):
            writer.writerow([class_names[row_idx], *row.tolist()])

    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)


def save_roc_artifacts(
    y_true: np.ndarray, y_score: np.ndarray, positive_label: int, out_dir: Path
) -> None:
    roc_csv_path = out_dir / "roc_curve.csv"
    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] < 2:
        with roc_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["info"])
            writer.writerow(["ROC unavailable: test labels contain only one class."])
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=positive_label)
    roc_auc = auc(fpr, tpr)

    with roc_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["fpr", "tpr", "threshold"])
        for idx in range(len(fpr)):
            writer.writerow([float(fpr[idx]), float(tpr[idx]), float(thresholds[idx])])

    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png")
    plt.close(fig)


def save_pr_artifacts(y_true: np.ndarray, y_score: np.ndarray, positive_label: int, out_dir: Path) -> None:
    pr_csv_path = out_dir / "pr_curve.csv"
    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] < 2:
        with pr_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["info"])
            writer.writerow(["PR curve unavailable: test labels contain only one class."])
        return

    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=positive_label)
    pr_auc = average_precision_score(y_true, y_score, pos_label=positive_label)

    with pr_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["precision", "recall", "threshold"])
        writer.writerow([float(precision[0]), float(recall[0]), ""])
        for idx, threshold in enumerate(thresholds, start=1):
            writer.writerow([float(precision[idx]), float(recall[idx]), float(threshold)])

    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png")
    plt.close(fig)


def save_calibration_artifacts(
    y_true: np.ndarray, y_score: np.ndarray, positive_label: int, out_dir: Path
) -> None:
    calibration_csv_path = out_dir / "calibration_curve.csv"
    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] < 2:
        with calibration_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["info"])
            writer.writerow(["Calibration curve unavailable: test labels contain only one class."])
        return

    prob_true, prob_pred = calibration_curve(
        y_true == positive_label, y_score, n_bins=10, strategy="uniform"
    )
    brier = brier_score_loss(y_true == positive_label, y_score)

    with calibration_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["prob_pred", "prob_true"])
        for pred, true in zip(prob_pred, prob_true):
            writer.writerow([float(pred), float(true)])

    with (out_dir / "calibration_report.txt").open("w", encoding="utf-8") as fp:
        fp.write(f"Brier score: {brier:.6f}\n")

    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label=f"Model (Brier={brier:.4f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png")
    plt.close(fig)


def save_threshold_tuning_artifacts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    positive_label: int,
    out_dir: Path,
) -> None:
    threshold_csv_path = out_dir / "threshold_analysis.csv"
    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] < 2:
        with threshold_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["info"])
            writer.writerow(["Threshold analysis unavailable: test labels contain only one class."])
        return

    thresholds = np.linspace(0.05, 0.95, 19)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = precision_score(y_true, y_pred, zero_division=0)
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        youden = sensitivity + specificity - 1.0
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "npv": float(npv),
                "f1": float(f1),
                "accuracy": float(acc),
                "youden_j": float(youden),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp),
            }
        )

    with threshold_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best_f1 = max(rows, key=lambda row: row["f1"])
    best_youden = max(rows, key=lambda row: row["youden_j"])
    report_path = out_dir / "threshold_recommendations.txt"
    with report_path.open("w", encoding="utf-8") as fp:
        fp.write("Threshold tuning summary\n")
        fp.write(f"Positive class: {class_names[positive_label]}\n\n")
        fp.write("Best F1 threshold\n")
        fp.write(f"  threshold: {best_f1['threshold']:.2f}\n")
        fp.write(f"  f1: {best_f1['f1']:.4f}\n")
        fp.write(f"  sensitivity: {best_f1['sensitivity']:.4f}\n")
        fp.write(f"  specificity: {best_f1['specificity']:.4f}\n\n")
        fp.write("Best Youden-J threshold\n")
        fp.write(f"  threshold: {best_youden['threshold']:.2f}\n")
        fp.write(f"  youden_j: {best_youden['youden_j']:.4f}\n")
        fp.write(f"  sensitivity: {best_youden['sensitivity']:.4f}\n")
        fp.write(f"  specificity: {best_youden['specificity']:.4f}\n")

    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([row["threshold"] for row in rows], [row["sensitivity"] for row in rows], label="sensitivity")
    ax.plot([row["threshold"] for row in rows], [row["specificity"] for row in rows], label="specificity")
    ax.plot([row["threshold"] for row in rows], [row["f1"] for row in rows], label="f1")
    ax.axvline(best_f1["threshold"], color="green", linestyle="--", label="best_f1")
    ax.axvline(best_youden["threshold"], color="purple", linestyle="--", label="best_youden")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold Analysis")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_analysis.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transfer-learning CXR classifier.")
    parser.add_argument("--epochs-head", type=int, default=3, help="Epochs with frozen backbone.")
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=2,
        help="Total epochs for gradual fine-tuning after head training.",
    )
    parser.add_argument(
        "--epochs-last-block",
        type=int,
        default=1,
        help="Max epochs to train only last block + head before full unfreeze.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=1)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_improvement(
    val_acc: float,
    best_val_acc: float,
    val_loss: float,
    best_val_loss: float,
    min_delta: float,
) -> bool:
    if val_acc > best_val_acc + min_delta:
        return True
    return abs(val_acc - best_val_acc) <= min_delta and val_loss < best_val_loss


def collect_test_outputs(
    model: nn.Module, loader: DataLoader, positive_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_pred, y_score = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs[:, positive_index].cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


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
    scheduler = build_scheduler(optimizer, args)

    last_block_epochs = min(max(args.epochs_last_block, 0), max(args.epochs_finetune, 0))
    full_unfreeze_epochs = max(0, args.epochs_finetune - last_block_epochs)
    total_epochs = args.epochs_head + last_block_epochs + full_unfreeze_epochs

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_count = 0

    checkpoint_dir = Path("./backend/checkpoints")
    artifact_dir = Path("./backend/artifacts")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    for epoch in range(1, total_epochs + 1):
        phase = "head"
        if epoch == args.epochs_head + 1 and last_block_epochs > 0:
            print("Switching phase: unfreeze last block + head.")
            unfreeze_last_block_and_head(model, arch)
            optimizer = build_optimizer(model, args.lr_finetune)
            scheduler = build_scheduler(optimizer, args)
            phase = "last_block"
        elif epoch == args.epochs_head + last_block_epochs + 1 and full_unfreeze_epochs > 0:
            print("Switching phase: full-network fine-tuning.")
            unfreeze_all_layers(model)
            optimizer = build_optimizer(model, args.lr_finetune)
            scheduler = build_scheduler(optimizer, args)
            phase = "full_finetune"
        elif epoch > args.epochs_head + last_block_epochs:
            phase = "full_finetune"
        elif epoch > args.epochs_head:
            phase = "last_block"

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "phase": phase,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch}/{total_epochs} phase={phase} lr={current_lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if is_improvement(
            val_acc, best_val_acc, val_loss, best_val_loss, args.early_stopping_min_delta
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_count = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": train_ds.classes,
                    "arch": arch,
                    "image_size": settings.image_size,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"New best checkpoint saved at epoch {best_epoch}.")
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best epoch={best_epoch}, val_acc={best_val_acc:.4f})."
                )
                break

    metrics_csv = artifact_dir / "training_metrics.csv"
    metrics_plot = artifact_dir / "training_curves.png"
    write_metrics_csv(history, metrics_csv)
    save_training_plot(history, metrics_plot, best_epoch)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    best_model = build_model(num_classes=len(checkpoint["class_names"]), arch=arch).to("cpu")
    best_model.load_state_dict(checkpoint["state_dict"])
    best_model.eval()

    test_split_path = dataset_root / "test"
    if test_split_path.exists():
        test_ds = build_imagefolder_dataset(dataset_root, "test", transforms_map["test"])
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        positive_idx = (
            test_ds.classes.index("PNEUMONIA")
            if "PNEUMONIA" in test_ds.classes
            else len(test_ds.classes) - 1
        )
        y_true, y_pred, y_score = collect_test_outputs(best_model, test_loader, positive_idx)
        save_confusion_matrix_artifacts(y_true, y_pred, test_ds.classes, artifact_dir)
        save_roc_artifacts(y_true, y_score, positive_idx, artifact_dir)
        save_pr_artifacts(y_true, y_score, positive_idx, artifact_dir)
        save_calibration_artifacts(y_true, y_score, positive_idx, artifact_dir)
        save_threshold_tuning_artifacts(y_true, y_score, test_ds.classes, positive_idx, artifact_dir)
        print(
            "Saved confusion matrix, ROC, PR, calibration, and threshold tuning artifacts "
            "to backend/artifacts."
        )
    else:
        print("No test split found; skipped confusion matrix and ROC export.")

    print(f"Saved best checkpoint to: {checkpoint_path}")
    print(f"Saved best epoch: {best_epoch}")
    print(f"Saved metrics CSV to: {metrics_csv}")
    if plt is None:
        print("Matplotlib not installed; skipped PNG artifact plots.")
    else:
        print(f"Saved training curves to: {metrics_plot}")


if __name__ == "__main__":
    main()
