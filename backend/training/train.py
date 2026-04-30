import argparse
import csv
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import random
import time

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models

from backend.app.config import settings
from backend.training.data_utils import build_imagefolder_dataset, build_transforms, build_tta_transforms, mixup_batch

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


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification and better calibration.

    Down-weights easy/well-classified examples so the model focuses on
    hard, misclassified ones. Reduces overconfident wrong predictions.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy loss that accepts soft (one-hot blended) targets from mixup."""

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=1)
        if self.weight is not None:
            log_probs = log_probs * self.weight.unsqueeze(0)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_classifier_head(model: nn.Module, arch: str) -> None:
    if arch == "simple_cnn":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True


def enable_last_block(model: nn.Module, arch: str) -> None:
    if arch == "simple_cnn":
        return
    if arch == "resnet50":
        for param in model.layer4.parameters():
            param.requires_grad = True
    else:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True


def freeze_feature_extractor(model: nn.Module, arch: str) -> None:
    if arch == "simple_cnn":
        return
    freeze_all(model)
    enable_classifier_head(model, arch)


def unfreeze_last_block_and_head(model: nn.Module, arch: str) -> None:
    if arch == "simple_cnn":
        return
    freeze_all(model)
    enable_classifier_head(model, arch)
    enable_last_block(model, arch)


def unfreeze_all_layers(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


class SimpleCNN(nn.Module):
    """Lightweight 4-layer CNN baseline for comparison against transfer learning."""

    def __init__(self, num_classes: int, image_size: int = 224) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int, arch: str = "densenet121") -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(num_classes)

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


def build_scheduler(
    optimizer: optim.Optimizer, args: argparse.Namespace, total_epochs: int = 0
) -> ReduceLROnPlateau | CosineAnnealingLR:
    if getattr(args, "cosine_annealing", False) and total_epochs > 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=args.scheduler_min_lr)
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    heartbeat_seconds: float,
    heartbeat_label: str,
    mixup_alpha: float = 0.0,
    num_classes: int = 2,
    mixup_criterion: nn.Module | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    use_mixup = is_train and mixup_alpha > 0 and mixup_criterion is not None

    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)
    last_heartbeat = time.monotonic()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        if is_train:
            optimizer.zero_grad()

        if use_mixup:
            mixed_images, mixed_labels = mixup_batch(images, labels, mixup_alpha, num_classes)
            logits = model(mixed_images)
            loss = mixup_criterion(logits, mixed_labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        now = time.monotonic()
        if heartbeat_seconds > 0 and (now - last_heartbeat) >= heartbeat_seconds:
            status(
                f"{heartbeat_label}: still running "
                f"(batch {batch_idx}/{total_batches}, seen {total} samples)."
            )
            last_heartbeat = now

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
    parser.add_argument("--epochs-head", type=int, default=5, help="Epochs with frozen backbone.")
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=10,
        help="Total epochs for gradual fine-tuning after head training.",
    )
    parser.add_argument(
        "--epochs-last-block",
        type=int,
        default=3,
        help="Max epochs to train only last block + head before full unfreeze.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Seconds between in-epoch heartbeat status logs (set 0 to disable).",
    )
    parser.add_argument(
        "--disable-class-weighting",
        action="store_true",
        help="Disable inverse-frequency class weighting in cross-entropy loss.",
    )
    parser.add_argument(
        "--external-test-root",
        type=str,
        default=None,
        help=(
            "Optional external dataset root with ImageFolder layout to assess "
            "generalization (for example: data/raw/chest_xray_external)."
        ),
    )
    parser.add_argument(
        "--audit-metadata-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV with per-image subgroup metadata for fairness audits. "
            "Must include an image path column and one or more subgroup columns."
        ),
    )
    parser.add_argument(
        "--audit-path-column",
        type=str,
        default="image_path",
        help="Column name in --audit-metadata-csv that contains image paths.",
    )
    parser.add_argument(
        "--audit-group-columns",
        type=str,
        default="sex,age_group,site",
        help="Comma-separated subgroup columns to evaluate when metadata is provided.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="If > 1, run stratified k-fold cross-validation instead of a single train/val split.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for CrossEntropyLoss (0.0 = off, 0.1 recommended).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of epochs to linearly warm up the learning rate from 1e-7 to --lr-head.",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable training augmentation (use plain resize only).",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use focal loss instead of cross-entropy (better for imbalanced data and calibration).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (higher = more focus on hard examples).",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Mixup interpolation alpha (0 = disabled, 0.2 recommended). Improves calibration.",
    )
    parser.add_argument(
        "--cosine-annealing",
        action="store_true",
        help="Use cosine annealing LR schedule instead of ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override model architecture (densenet121, resnet50, simple_cnn). Defaults to MODEL_ARCH env.",
    )
    return parser.parse_args()


def compute_epoch_clinical_metrics(
    model: nn.Module, loader: DataLoader, positive_index: int
) -> dict[str, float]:
    """Compute AUROC, F1, sensitivity, specificity on a validation set."""
    y_true, y_pred, y_score = collect_test_outputs(model, loader, positive_index)

    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] < 2:
        return {"auroc": 0.0, "f1": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=positive_index)
    auroc = float(auc(fpr, tpr))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {"auroc": auroc, "f1": f1, "sensitivity": sensitivity, "specificity": specificity}


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


def _mask_corners(batch: torch.Tensor, patch_fraction: float) -> torch.Tensor:
    masked = batch.clone()
    _, _, height, width = masked.shape
    patch_h = max(1, int(height * patch_fraction))
    patch_w = max(1, int(width * patch_fraction))
    masked[:, :, :patch_h, :patch_w] = 0.0
    masked[:, :, :patch_h, width - patch_w :] = 0.0
    masked[:, :, height - patch_h :, :patch_w] = 0.0
    masked[:, :, height - patch_h :, width - patch_w :] = 0.0
    return masked


def _mask_center_roi(batch: torch.Tensor, x_margin: float = 0.2, y_margin: float = 0.15) -> torch.Tensor:
    masked = batch.clone()
    _, _, height, width = masked.shape
    x0, x1 = int(width * x_margin), int(width * (1.0 - x_margin))
    y0, y1 = int(height * y_margin), int(height * 0.9)
    masked[:, :, y0:y1, x0:x1] = 0.0
    return masked


def save_shortcut_stress_test_artifacts(
    model: nn.Module,
    loader: DataLoader,
    positive_index: int,
    out_dir: Path,
    split_name: str,
    corner_patch_fraction: float = 0.12,
) -> None:
    rows: list[dict[str, float | int]] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            base_logits = model(images)
            base_probs = torch.softmax(base_logits, dim=1)
            base_preds = torch.argmax(base_probs, dim=1)

            corner_logits = model(_mask_corners(images, patch_fraction=corner_patch_fraction))
            corner_probs = torch.softmax(corner_logits, dim=1)
            corner_preds = torch.argmax(corner_probs, dim=1)

            center_logits = model(_mask_center_roi(images))
            center_probs = torch.softmax(center_logits, dim=1)
            center_preds = torch.argmax(center_probs, dim=1)

            for idx in range(labels.size(0)):
                base_prob = float(base_probs[idx, positive_index].item())
                rows.append(
                    {
                        "y_true": int(labels[idx].item()),
                        "base_pred": int(base_preds[idx].item()),
                        "corner_pred": int(corner_preds[idx].item()),
                        "center_pred": int(center_preds[idx].item()),
                        "base_prob_pos": base_prob,
                        "corner_prob_pos": float(corner_probs[idx, positive_index].item()),
                        "center_prob_pos": float(center_probs[idx, positive_index].item()),
                    }
                )

    if not rows:
        return

    csv_path = out_dir / f"{split_name}_shortcut_stress_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    y_true = np.array([int(row["y_true"]) for row in rows], dtype=np.int64)
    base_pred = np.array([int(row["base_pred"]) for row in rows], dtype=np.int64)
    corner_pred = np.array([int(row["corner_pred"]) for row in rows], dtype=np.int64)
    center_pred = np.array([int(row["center_pred"]) for row in rows], dtype=np.int64)
    base_prob = np.array([float(row["base_prob_pos"]) for row in rows], dtype=np.float64)
    corner_prob = np.array([float(row["corner_prob_pos"]) for row in rows], dtype=np.float64)
    center_prob = np.array([float(row["center_prob_pos"]) for row in rows], dtype=np.float64)

    baseline_acc = accuracy_score(y_true, base_pred)
    corner_acc = accuracy_score(y_true, corner_pred)
    center_acc = accuracy_score(y_true, center_pred)
    corner_flip_rate = float(np.mean(base_pred != corner_pred))
    center_flip_rate = float(np.mean(base_pred != center_pred))
    corner_prob_drop = float(np.mean(np.abs(base_prob - corner_prob)))
    center_prob_drop = float(np.mean(np.abs(base_prob - center_prob)))
    shortcut_index = corner_flip_rate / (center_flip_rate + 1e-8)

    risk_level = "low"
    if shortcut_index > 0.9:
        risk_level = "high"
    elif shortcut_index > 0.6:
        risk_level = "moderate"

    report_path = out_dir / f"{split_name}_shortcut_stress_report.txt"
    with report_path.open("w", encoding="utf-8") as fp:
        fp.write("Shortcut stress test (corner vs center masking)\n")
        fp.write(f"split: {split_name}\n")
        fp.write(f"samples: {len(rows)}\n")
        fp.write(f"corner_patch_fraction: {corner_patch_fraction:.3f}\n\n")
        fp.write(f"baseline_accuracy: {baseline_acc:.4f}\n")
        fp.write(f"corner_mask_accuracy: {corner_acc:.4f}\n")
        fp.write(f"center_mask_accuracy: {center_acc:.4f}\n")
        fp.write(f"corner_flip_rate: {corner_flip_rate:.4f}\n")
        fp.write(f"center_flip_rate: {center_flip_rate:.4f}\n")
        fp.write(f"mean_prob_drop_corner: {corner_prob_drop:.4f}\n")
        fp.write(f"mean_prob_drop_center: {center_prob_drop:.4f}\n")
        fp.write(f"shortcut_reliance_index: {shortcut_index:.4f}\n")
        fp.write(f"shortcut_risk_level: {risk_level}\n\n")
        fp.write(
            "Interpretation: high shortcut_reliance_index means predictions are almost as sensitive "
            "to corner masking as center masking, suggesting potential non-anatomical cue reliance.\n"
        )

    status(
        f"Saved shortcut stress artifacts for '{split_name}' to {csv_path} and {report_path} "
        f"(risk={risk_level}, index={shortcut_index:.3f})."
    )


def status(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def is_improvement(
    val_acc: float, best_val_acc: float, val_loss: float, best_val_loss: float, min_delta: float
) -> bool:
    """Check if current epoch improved over best, used by k-fold path."""
    if val_acc > best_val_acc + min_delta:
        return True
    if abs(val_acc - best_val_acc) < min_delta and val_loss < best_val_loss:
        return True
    return False


def build_class_weight_tensor(train_ds) -> torch.Tensor:
    class_counts = Counter(train_ds.targets)
    total = sum(class_counts.values())
    num_classes = len(train_ds.classes)
    weights = []
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 0)
        if count <= 0:
            weights.append(0.0)
            continue
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def normalize_path_key(path_str: str) -> str:
    return str(Path(path_str)).replace("\\", "/").lstrip("./").lower()


def load_subgroup_metadata(
    metadata_csv: Path, path_column: str, group_columns: list[str]
) -> tuple[dict[str, dict[str, str]], list[str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None or path_column not in reader.fieldnames:
            raise ValueError(
                f"Metadata CSV must include path column '{path_column}'. "
                f"Found columns: {reader.fieldnames or []}"
            )
        present_group_columns = [col for col in group_columns if col in (reader.fieldnames or [])]
        if not present_group_columns:
            raise ValueError(
                "No requested subgroup columns were found in metadata CSV. "
                f"Requested={group_columns}, found={reader.fieldnames or []}"
            )

        mapping: dict[str, dict[str, str]] = {}
        for row in reader:
            raw_path = (row.get(path_column) or "").strip()
            if not raw_path:
                continue
            key = normalize_path_key(raw_path)
            mapping[key] = {
                col: (row.get(col) or "").strip() for col in present_group_columns if (row.get(col) or "").strip()
            }
    return mapping, present_group_columns


def save_subgroup_fairness_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    positive_label: int,
    sample_paths: list[str],
    split_name: str,
    out_dir: Path,
    metadata_csv: Path,
    path_column: str,
    group_columns: list[str],
) -> None:
    metadata_map, present_group_columns = load_subgroup_metadata(
        metadata_csv, path_column=path_column, group_columns=group_columns
    )
    if not present_group_columns:
        return

    rows: list[dict[str, float | str]] = []
    joined = 0
    for idx, sample_path in enumerate(sample_paths):
        sample_key = normalize_path_key(sample_path)
        subgroup_values = metadata_map.get(sample_key, {})
        if not subgroup_values:
            sample_name_key = normalize_path_key(Path(sample_path).name)
            subgroup_values = metadata_map.get(sample_name_key, {})
        if not subgroup_values:
            continue
        joined += 1

        for group_col in present_group_columns:
            group_value = subgroup_values.get(group_col)
            if not group_value:
                continue
            rows.append(
                {
                    "split": split_name,
                    "group_column": group_col,
                    "group_value": group_value,
                    "y_true": float(y_true[idx]),
                    "y_pred": float(y_pred[idx]),
                    "y_score": float(y_score[idx]),
                }
            )

    if not rows:
        print(
            "Subgroup fairness audit skipped: no metadata rows matched sample paths. "
            "Check path normalization and CSV path column."
        )
        return

    metrics_rows: list[dict[str, float | str]] = []
    for group_col in present_group_columns:
        group_values = sorted({str(row["group_value"]) for row in rows if row["group_column"] == group_col})
        for group_value in group_values:
            subset = [
                row
                for row in rows
                if row["group_column"] == group_col and str(row["group_value"]) == group_value
            ]
            yt = np.array([int(row["y_true"]) for row in subset], dtype=np.int64)
            yp = np.array([int(row["y_pred"]) for row in subset], dtype=np.int64)
            ys = np.array([float(row["y_score"]) for row in subset], dtype=np.float64)
            if yt.size == 0:
                continue

            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            prevalence = float(np.mean(yt == positive_label))
            acc = accuracy_score(yt, yp)
            prec = precision_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0

            metrics_rows.append(
                {
                    "split": split_name,
                    "group_column": group_col,
                    "group_value": group_value,
                    "n": int(yt.size),
                    "prevalence": float(prevalence),
                    "avg_score": float(np.mean(ys)),
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "fpr": float(fpr),
                    "fnr": float(fnr),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )

    if not metrics_rows:
        return

    out_path = out_dir / f"{split_name}_subgroup_fairness_metrics.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary_path = out_dir / f"{split_name}_subgroup_fairness_summary.txt"
    metric_names = ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write("Subgroup fairness audit summary\n")
        fp.write(f"Split: {split_name}\n")
        fp.write(f"Metadata CSV: {metadata_csv}\n")
        fp.write(f"Joined samples: {joined}/{len(sample_paths)}\n\n")
        for group_col in present_group_columns:
            fp.write(f"[{group_col}]\n")
            subset_rows = [row for row in metrics_rows if row["group_column"] == group_col]
            if len(subset_rows) < 2:
                fp.write("  Not enough groups for disparity analysis.\n\n")
                continue
            for metric_name in metric_names:
                values = [float(row[metric_name]) for row in subset_rows]
                disparity = max(values) - min(values)
                fp.write(f"  {metric_name}_disparity: {disparity:.4f}\n")
            fp.write("\n")

    print(
        f"Saved subgroup fairness artifacts for '{split_name}' to {out_path} and {summary_path} "
        f"(joined {joined}/{len(sample_paths)} samples)."
    )


def evaluate_split_and_export(
    model: nn.Module,
    dataset_root: Path,
    split_name: str,
    transforms_map: dict[str, object],
    batch_size: int,
    artifact_dir: Path,
    audit_metadata_csv: Path | None,
    audit_path_column: str,
    audit_group_columns: list[str],
) -> None:
    split_path = dataset_root / split_name
    if not split_path.exists():
        print(f"No '{split_name}' split found at {split_path}; skipped artifact export.")
        return

    ds = build_imagefolder_dataset(dataset_root, split_name, transforms_map["test"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    positive_idx = ds.classes.index("PNEUMONIA") if "PNEUMONIA" in ds.classes else len(ds.classes) - 1
    y_true, y_pred, y_score = collect_test_outputs(model, loader, positive_idx)

    split_out_dir = artifact_dir if split_name == "test" else artifact_dir / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_artifacts(y_true, y_pred, ds.classes, split_out_dir)
    save_roc_artifacts(y_true, y_score, positive_idx, split_out_dir)
    save_pr_artifacts(y_true, y_score, positive_idx, split_out_dir)
    save_calibration_artifacts(y_true, y_score, positive_idx, split_out_dir)
    save_threshold_tuning_artifacts(y_true, y_score, ds.classes, positive_idx, split_out_dir)
    save_shortcut_stress_test_artifacts(
        model=model,
        loader=loader,
        positive_index=positive_idx,
        out_dir=split_out_dir,
        split_name=split_name,
    )

    if audit_metadata_csv is not None:
        sample_paths = [str(path) for path, _class_idx in ds.samples]
        save_subgroup_fairness_artifacts(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            positive_label=positive_idx,
            sample_paths=sample_paths,
            split_name=split_name,
            out_dir=split_out_dir,
            metadata_csv=audit_metadata_csv,
            path_column=audit_path_column,
            group_columns=audit_group_columns,
        )

    print(
        f"Saved confusion matrix, ROC, PR, calibration, and threshold tuning artifacts "
        f"for split '{split_name}' to {split_out_dir}."
    )


def find_optimal_temperature(
    model: nn.Module, loader: DataLoader, device: torch.device = torch.device("cpu")
) -> float:
    """Find temperature scaling parameter that minimizes NLL on a validation set."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)

    best_temp = 1.0
    best_nll = float("inf")
    for temp_candidate in np.linspace(0.5, 3.0, 51):
        scaled = logits_tensor / temp_candidate
        nll = float(nn.functional.cross_entropy(scaled, labels_tensor).item())
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp_candidate)
    return round(best_temp, 3)


def fit_isotonic_calibration(
    model: nn.Module,
    loader: DataLoader,
    positive_index: int,
    temperature: float = 1.0,
) -> IsotonicRegression:
    """Fit isotonic regression on validation set for post-hoc probability calibration.

    This makes probability outputs reflect true positive rates so that
    higher thresholds yield genuinely more confident (not less accurate) predictions.
    """
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            scaled = logits / temperature if temperature > 0 else logits
            probs = torch.softmax(scaled, dim=1)
            all_scores.extend(probs[:, positive_index].cpu().numpy().tolist())
            all_labels.extend((labels == positive_index).int().cpu().numpy().tolist())

    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_reg.fit(np.array(all_scores), np.array(all_labels))
    return iso_reg


def save_calibration_model(iso_reg: IsotonicRegression, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump(iso_reg, fp)
    status(f"Isotonic calibration model saved to {out_path}")


def train_single_split(
    args: argparse.Namespace,
    arch: str,
    dataset_root: Path,
    transforms_map: dict[str, object],
    checkpoint_path: Path,
    artifact_dir: Path,
    fold_label: str = "",
) -> dict[str, float]:
    """Core training loop for one train/val split. Returns summary metrics dict."""
    prefix = f"[fold {fold_label}] " if fold_label else ""

    train_ds = build_imagefolder_dataset(dataset_root, "train", transforms_map["train"])
    val_ds = build_imagefolder_dataset(dataset_root, "val", transforms_map["val"])
    status(
        f"{prefix}Datasets loaded (train={len(train_ds)} images, val={len(val_ds)} images, "
        f"classes={train_ds.classes})."
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(train_ds.classes), arch=arch).to("cpu")
    status(f"{prefix}Model built ({arch}) on CPU.")

    class_weights = None
    if not args.disable_class_weighting:
        class_weights = build_class_weight_tensor(train_ds)
        status(f"{prefix}Using inverse-frequency class weights: {class_weights.tolist()}")

    if args.focal_loss:
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            alpha=class_weights,
            label_smoothing=args.label_smoothing,
        )
        status(f"{prefix}Using focal loss (gamma={args.focal_gamma})")
    else:
        criterion_kwargs: dict = {}
        if class_weights is not None:
            criterion_kwargs["weight"] = class_weights
        if args.label_smoothing > 0:
            criterion_kwargs["label_smoothing"] = args.label_smoothing
            status(f"{prefix}Label smoothing: {args.label_smoothing}")
        criterion = nn.CrossEntropyLoss(**criterion_kwargs)

    mixup_criterion: nn.Module | None = None
    if args.mixup_alpha > 0:
        mixup_criterion = SoftTargetCrossEntropy(weight=class_weights)
        status(f"{prefix}Mixup enabled (alpha={args.mixup_alpha})")

    warmup_epochs = max(0, args.warmup_epochs)
    last_block_epochs = min(max(args.epochs_last_block, 0), max(args.epochs_finetune, 0))
    full_unfreeze_epochs = max(0, args.epochs_finetune - last_block_epochs)
    total_epochs = warmup_epochs + args.epochs_head + last_block_epochs + full_unfreeze_epochs

    freeze_feature_extractor(model, arch)
    optimizer = build_optimizer(model, args.lr_head)
    scheduler = build_scheduler(optimizer, args, total_epochs=total_epochs)

    # Warmup schedule: linearly ramp LR from near-zero to lr_head over warmup epochs.

    positive_idx = train_ds.classes.index("PNEUMONIA") if "PNEUMONIA" in train_ds.classes else len(train_ds.classes) - 1

    history: list[dict[str, float]] = []
    best_val_auroc = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_count = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    status(f"{prefix}Beginning training loop ({total_epochs} epochs).")

    for epoch in range(1, total_epochs + 1):
        status(f"{prefix}Epoch {epoch}/{total_epochs}: running train/validation.")
        phase = "warmup"

        if epoch <= warmup_epochs:
            warmup_lr = 1e-7 + (args.lr_head - 1e-7) * (epoch / max(warmup_epochs, 1))
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            phase = "warmup"
        elif epoch == warmup_epochs + 1:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr_head
            phase = "head"
        elif epoch == warmup_epochs + args.epochs_head + 1 and last_block_epochs > 0:
            status(f"{prefix}Switching phase: unfreeze last block + head (with warmup).")
            unfreeze_last_block_and_head(model, arch)
            optimizer = build_optimizer(model, args.lr_finetune * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr_finetune
            scheduler = build_scheduler(optimizer, args, total_epochs=last_block_epochs)
            phase = "last_block"
        elif epoch == warmup_epochs + args.epochs_head + last_block_epochs + 1 and full_unfreeze_epochs > 0:
            status(f"{prefix}Switching phase: full-network fine-tuning (with warmup).")
            unfreeze_all_layers(model)
            optimizer = build_optimizer(model, args.lr_finetune * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr_finetune
            scheduler = build_scheduler(optimizer, args, total_epochs=full_unfreeze_epochs)
            phase = "full_finetune"
        elif epoch > warmup_epochs + args.epochs_head + last_block_epochs:
            phase = "full_finetune"
        elif epoch > warmup_epochs + args.epochs_head:
            phase = "last_block"
        else:
            phase = "head"

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            heartbeat_seconds=args.heartbeat_seconds,
            heartbeat_label=f"{prefix}Epoch {epoch}/{total_epochs} train",
            mixup_alpha=args.mixup_alpha,
            num_classes=len(train_ds.classes),
            mixup_criterion=mixup_criterion,
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                heartbeat_seconds=args.heartbeat_seconds,
                heartbeat_label=f"{prefix}Epoch {epoch}/{total_epochs} val",
            )

        clinical = compute_epoch_clinical_metrics(model, val_loader, positive_idx)
        val_auroc = clinical["auroc"]
        val_f1 = clinical["f1"]
        val_sensitivity = clinical["sensitivity"]
        val_specificity = clinical["specificity"]

        if epoch > warmup_epochs:
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            else:
                scheduler.step(val_auroc)
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
                "val_auroc": val_auroc,
                "val_f1": val_f1,
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
            }
        )
        print(
            f"{prefix}epoch={epoch}/{total_epochs} phase={phase} lr={current_lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_auroc={val_auroc:.4f} val_f1={val_f1:.4f} "
            f"val_sens={val_sensitivity:.4f} val_spec={val_specificity:.4f}"
        )

        improved = val_auroc > best_val_auroc + args.early_stopping_min_delta
        if improved:
            best_val_auroc = val_auroc
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_count = 0

            optimal_temp = find_optimal_temperature(model, val_loader)

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": train_ds.classes,
                    "arch": arch,
                    "image_size": settings.image_size,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_auroc": best_val_auroc,
                    "best_val_f1": val_f1,
                    "history": history,
                    "temperature": optimal_temp,
                },
                checkpoint_path,
            )
            print(
                f"{prefix}New best checkpoint at epoch {best_epoch} "
                f"(AUROC={val_auroc:.4f}, temperature={optimal_temp})."
            )
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stopping_patience:
                status(
                    f"{prefix}Early stopping triggered at epoch {epoch} "
                    f"(best epoch={best_epoch}, val_auroc={best_val_auroc:.4f})."
                )
                break

    metrics_csv = artifact_dir / "training_metrics.csv"
    metrics_plot = artifact_dir / "training_curves.png"
    status(f"{prefix}Writing training metrics artifacts.")
    write_metrics_csv(history, metrics_csv)
    save_training_plot(history, metrics_plot, best_epoch)

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_val_auroc": best_val_auroc,
    }


def run_kfold(args: argparse.Namespace, arch: str, transforms_map: dict[str, object]) -> None:
    """Stratified k-fold cross-validation. Writes per-fold and aggregate artifacts."""
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import Subset

    status(f"Starting {args.kfold}-fold cross-validation.")
    dataset_root = settings.dataset_root
    artifact_dir = Path("./backend/artifacts")
    kfold_dir = artifact_dir / "kfold"
    kfold_dir.mkdir(parents=True, exist_ok=True)

    # Load full train+val combined
    full_train_ds = build_imagefolder_dataset(dataset_root, "train", transforms_map["train"])
    try:
        full_val_ds = build_imagefolder_dataset(dataset_root, "val", transforms_map["train"])
        combined_targets = full_train_ds.targets + full_val_ds.targets
        combined_samples = full_train_ds.samples + full_val_ds.samples
    except FileNotFoundError:
        combined_targets = full_train_ds.targets
        combined_samples = full_train_ds.samples

    status(f"Combined dataset: {len(combined_targets)} images for {args.kfold}-fold CV.")

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(combined_samples, combined_targets), start=1):
        status(f"=== Fold {fold_idx}/{args.kfold} ===")
        fold_dir = kfold_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_ckpt = fold_dir / "best_model.pt"

        # Create ImageFolder-compatible subset by writing temp symlinks/index
        # Using Subset approach with a unified dataset
        from torch.utils.data import ConcatDataset

        all_ds = build_imagefolder_dataset(dataset_root, "train", transforms_map["train"])
        try:
            val_ds_raw = build_imagefolder_dataset(dataset_root, "val", transforms_map["train"])
            combined_ds = ConcatDataset([all_ds, val_ds_raw])
        except FileNotFoundError:
            combined_ds = all_ds

        train_subset = Subset(combined_ds, train_indices.tolist())
        val_eval_tf = build_imagefolder_dataset(dataset_root, "train", transforms_map["val"])
        try:
            val_eval_raw = build_imagefolder_dataset(dataset_root, "val", transforms_map["val"])
            combined_eval = ConcatDataset([val_eval_tf, val_eval_raw])
        except FileNotFoundError:
            combined_eval = val_eval_tf
        val_subset = Subset(combined_eval, val_indices.tolist())

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        class_names = full_train_ds.classes
        model = build_model(num_classes=len(class_names), arch=arch).to("cpu")

        criterion_kwargs: dict = {}
        if not args.disable_class_weighting:
            class_weights = build_class_weight_tensor(full_train_ds)
            criterion_kwargs["weight"] = class_weights
        if args.label_smoothing > 0:
            criterion_kwargs["label_smoothing"] = args.label_smoothing
        criterion = nn.CrossEntropyLoss(**criterion_kwargs)

        freeze_feature_extractor(model, arch)
        optimizer = build_optimizer(model, args.lr_head)
        scheduler = build_scheduler(optimizer, args)

        warmup_epochs = max(0, args.warmup_epochs)
        last_block_epochs = min(max(args.epochs_last_block, 0), max(args.epochs_finetune, 0))
        full_unfreeze_epochs = max(0, args.epochs_finetune - last_block_epochs)
        total_epochs = warmup_epochs + args.epochs_head + last_block_epochs + full_unfreeze_epochs

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        early_stop_count = 0

        for epoch in range(1, total_epochs + 1):
            phase = "head"
            if epoch <= warmup_epochs:
                warmup_lr = 1e-7 + (args.lr_head - 1e-7) * (epoch / max(warmup_epochs, 1))
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
                phase = "warmup"
            elif epoch == warmup_epochs + 1:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr_head
            elif epoch == warmup_epochs + args.epochs_head + 1 and last_block_epochs > 0:
                unfreeze_last_block_and_head(model, arch)
                optimizer = build_optimizer(model, args.lr_finetune)
                scheduler = build_scheduler(optimizer, args)
                phase = "last_block"
            elif epoch == warmup_epochs + args.epochs_head + last_block_epochs + 1 and full_unfreeze_epochs > 0:
                unfreeze_all_layers(model)
                optimizer = build_optimizer(model, args.lr_finetune)
                scheduler = build_scheduler(optimizer, args)
                phase = "full_finetune"
            elif epoch > warmup_epochs + args.epochs_head + last_block_epochs:
                phase = "full_finetune"
            elif epoch > warmup_epochs + args.epochs_head:
                phase = "last_block"

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer,
                heartbeat_seconds=args.heartbeat_seconds,
                heartbeat_label=f"Fold {fold_idx} epoch {epoch}/{total_epochs} train",
            )
            with torch.no_grad():
                val_loss, val_acc = run_epoch(
                    model, val_loader, criterion, optimizer=None,
                    heartbeat_seconds=args.heartbeat_seconds,
                    heartbeat_label=f"Fold {fold_idx} epoch {epoch}/{total_epochs} val",
                )

            if epoch > warmup_epochs:
                scheduler.step(val_loss)

            print(
                f"[fold {fold_idx}] epoch={epoch}/{total_epochs} phase={phase} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if is_improvement(val_acc, best_val_acc, val_loss, best_val_loss, args.early_stopping_min_delta):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                early_stop_count = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "class_names": class_names,
                        "arch": arch,
                        "image_size": settings.image_size,
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "best_val_loss": best_val_loss,
                    },
                    fold_ckpt,
                )
            else:
                early_stop_count += 1
                if early_stop_count >= args.early_stopping_patience:
                    status(f"[fold {fold_idx}] Early stopping at epoch {epoch}.")
                    break

        fold_metrics.append({"fold": fold_idx, "best_epoch": best_epoch, "val_acc": best_val_acc, "val_loss": best_val_loss})
        status(f"[fold {fold_idx}] best_epoch={best_epoch} val_acc={best_val_acc:.4f} val_loss={best_val_loss:.4f}")

    # Aggregate summary
    accs = [m["val_acc"] for m in fold_metrics]
    losses = [m["val_loss"] for m in fold_metrics]
    summary_path = kfold_dir / "kfold_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write(f"{args.kfold}-fold Cross-Validation Summary\n")
        fp.write(f"Architecture: {arch}\n\n")
        for m in fold_metrics:
            fp.write(f"  Fold {int(m['fold'])}: val_acc={m['val_acc']:.4f}, val_loss={m['val_loss']:.4f}, best_epoch={int(m['best_epoch'])}\n")
        fp.write(f"\nMean val_acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}\n")
        fp.write(f"Mean val_loss: {np.mean(losses):.4f} +/- {np.std(losses):.4f}\n")

    csv_path = kfold_dir / "kfold_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["fold", "best_epoch", "val_acc", "val_loss"])
        writer.writeheader()
        writer.writerows(fold_metrics)

    status(f"K-fold summary: mean_acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    status(f"K-fold artifacts saved to {kfold_dir}")


def generate_test_summary(
    model: nn.Module,
    dataset_root: Path,
    transforms_map: dict[str, object],
    batch_size: int,
    checkpoint_meta: dict,
    out_path: Path,
) -> None:
    """Generate a comprehensive test_summary.json with clinical evaluation metrics."""
    import json as json_mod

    split_path = dataset_root / "test"
    if not split_path.exists():
        status("No test split found; skipping test_summary.json.")
        return

    ds = build_imagefolder_dataset(dataset_root, "test", transforms_map["test"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    positive_idx = ds.classes.index("PNEUMONIA") if "PNEUMONIA" in ds.classes else len(ds.classes) - 1

    y_true, y_pred, y_score = collect_test_outputs(model, loader, positive_idx)

    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] < 2:
        status("Test set has only one class; cannot compute full metrics.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=positive_idx)
    test_auroc = float(auc(fpr, tpr))
    test_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    test_acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0
    pr_auc_val = float(average_precision_score(y_true, y_score, pos_label=positive_idx))
    brier = float(brier_score_loss(y_true == positive_idx, y_score))
    youden_j = sensitivity + specificity - 1.0

    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    auroc_bootstraps = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[indices], y_score[indices], pos_label=positive_idx)
        auroc_bootstraps.append(float(auc(fpr_b, tpr_b)))

    auroc_ci_lower = float(np.percentile(auroc_bootstraps, 2.5)) if auroc_bootstraps else test_auroc
    auroc_ci_upper = float(np.percentile(auroc_bootstraps, 97.5)) if auroc_bootstraps else test_auroc

    train_ds = build_imagefolder_dataset(dataset_root, "train", transforms_map["test"])
    val_path = dataset_root / "val"
    val_count = len(build_imagefolder_dataset(dataset_root, "val", transforms_map["test"])) if val_path.exists() else 0

    summary = {
        "model_arch": checkpoint_meta.get("arch", "unknown"),
        "dataset": "NIH ChestX-ray14",
        "splits": {
            "train": len(train_ds),
            "val": val_count,
            "test": len(ds),
        },
        "patient_level_split": True,
        "best_epoch": checkpoint_meta.get("best_epoch"),
        "temperature": checkpoint_meta.get("temperature", 1.0),
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "auroc": round(test_auroc, 4),
            "auroc_95ci": [round(auroc_ci_lower, 4), round(auroc_ci_upper, 4)],
            "f1": round(test_f1, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "pr_auc": round(pr_auc_val, 4),
            "brier_score": round(brier, 6),
            "youden_j": round(youden_j, 4),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_names": ds.classes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json_mod.dump(summary, f, indent=2)
    status(f"Test summary saved to {out_path}")


def main() -> None:
    status("Starting training script.")
    args = parse_args()
    set_seed(args.seed)
    status(f"Seed set to {args.seed}.")

    dataset_root = settings.dataset_root
    arch = args.arch or settings.model_arch.lower()
    augment = not args.disable_augmentation
    transforms_map = build_transforms(settings.image_size, augment=augment)
    status(f"Building datasets from {dataset_root} (augmentation={'on' if augment else 'off'}).")

    if args.kfold > 1:
        run_kfold(args, arch, transforms_map)
        status("K-fold cross-validation finished. Exiting (no single checkpoint produced).")
        return

    checkpoint_dir = Path("./backend/checkpoints")
    artifact_dir = Path("./backend/artifacts")
    checkpoint_path = checkpoint_dir / "best_model.pt"

    summary = train_single_split(
        args=args,
        arch=arch,
        dataset_root=dataset_root,
        transforms_map=transforms_map,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
    )

    status("Loading best checkpoint for evaluation.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    best_model = build_model(num_classes=len(checkpoint["class_names"]), arch=arch).to("cpu")
    best_model.load_state_dict(checkpoint["state_dict"])
    best_model.eval()

    # Isotonic calibration: fit on validation set to fix threshold-accuracy relationship
    val_path = dataset_root / "val"
    if val_path.exists():
        status("Fitting isotonic calibration model on validation set.")
        val_ds_cal = build_imagefolder_dataset(dataset_root, "val", transforms_map["val"])
        val_loader_cal = DataLoader(val_ds_cal, batch_size=args.batch_size, shuffle=False, num_workers=0)
        positive_idx_cal = (
            val_ds_cal.classes.index("PNEUMONIA")
            if "PNEUMONIA" in val_ds_cal.classes
            else len(val_ds_cal.classes) - 1
        )
        temp = float(checkpoint.get("temperature", 1.0))
        iso_reg = fit_isotonic_calibration(best_model, val_loader_cal, positive_idx_cal, temp)
        calibration_path = checkpoint_dir / "calibration_model.pkl"
        save_calibration_model(iso_reg, calibration_path)
    else:
        status("No validation split found; skipping isotonic calibration.")

    audit_metadata_csv = Path(args.audit_metadata_csv) if args.audit_metadata_csv else None
    audit_group_columns = [col.strip() for col in args.audit_group_columns.split(",") if col.strip()]
    if audit_metadata_csv is not None and not audit_metadata_csv.exists():
        raise FileNotFoundError(f"Audit metadata CSV not found: {audit_metadata_csv}")

    status("Running evaluation on in-domain test split.")
    evaluate_split_and_export(
        model=best_model,
        dataset_root=dataset_root,
        split_name="test",
        transforms_map=transforms_map,
        batch_size=args.batch_size,
        artifact_dir=artifact_dir,
        audit_metadata_csv=audit_metadata_csv,
        audit_path_column=args.audit_path_column,
        audit_group_columns=audit_group_columns,
    )

    status("Generating test_summary.json with clinical metrics.")
    generate_test_summary(
        model=best_model,
        dataset_root=dataset_root,
        transforms_map=transforms_map,
        batch_size=args.batch_size,
        checkpoint_meta=checkpoint,
        out_path=artifact_dir / "test_summary.json",
    )

    if args.external_test_root:
        external_root = Path(args.external_test_root)
        if not external_root.exists():
            raise FileNotFoundError(f"External dataset root not found: {external_root}")
        status(f"Running external validation from {external_root}.")
        evaluate_split_and_export(
            model=best_model,
            dataset_root=external_root,
            split_name="test",
            transforms_map=transforms_map,
            batch_size=args.batch_size,
            artifact_dir=artifact_dir / "external",
            audit_metadata_csv=audit_metadata_csv,
            audit_path_column=args.audit_path_column,
            audit_group_columns=audit_group_columns,
        )

    status(f"Saved best checkpoint to: {checkpoint_path}")
    status(f"Saved best epoch: {summary['best_epoch']}")
    if plt is None:
        status("Matplotlib not installed; skipped PNG artifact plots.")
    status("Training script finished successfully.")


if __name__ == "__main__":
    main()
