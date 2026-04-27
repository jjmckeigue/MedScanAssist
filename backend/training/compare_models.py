"""Model comparison utility.

Loads trained checkpoints for multiple architectures and evaluates them on the
test set, producing a side-by-side comparison table.

Workflow:
    1. Train each architecture (results in separate checkpoint files):
         python -m backend.training.train --arch simple_cnn --epochs-head 10 --epochs-finetune 0
         python -m backend.training.train --arch densenet121 --epochs-head 5 --epochs-finetune 5
         python -m backend.training.train --arch resnet50 --epochs-head 5 --epochs-finetune 5

    2. Run comparison (auto-discovers checkpoints or specify paths):
         python -m backend.training.compare_models

Outputs:
    backend/artifacts/model_comparison.csv
    backend/artifacts/model_comparison.txt
    backend/artifacts/model_comparison.png   (grouped bar chart)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from backend.app.config import settings
from backend.training.data_utils import build_imagefolder_dataset, build_transforms
from backend.training.train import SimpleCNN, collect_test_outputs

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


ARTIFACT_DIR = Path("./backend/artifacts")
CHECKPOINT_DIR = Path("./backend/checkpoints")

DEFAULT_CHECKPOINT_MAP = {
    "simple_cnn": CHECKPOINT_DIR / "best_model_simple_cnn.pt",
    "densenet121": CHECKPOINT_DIR / "best_model_densenet121.pt",
    "resnet50": CHECKPOINT_DIR / "best_model_resnet50.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained model architectures.")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Explicit checkpoint paths (arch is read from checkpoint metadata). "
            "If omitted, auto-discovers checkpoints in backend/checkpoints/."
        ),
    )
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Also include backend/checkpoints/best_model.pt if it exists.",
    )
    return parser.parse_args()


def build_model_from_checkpoint(ckpt: dict, num_classes: int) -> nn.Module:
    arch = str(ckpt.get("arch", "densenet121")).lower()
    if arch == "simple_cnn":
        model = SimpleCNN(num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def evaluate_checkpoint(ckpt_path: Path, test_loader: DataLoader, class_names: list[str]) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = str(ckpt.get("arch", "unknown")).lower()
    num_classes = len(ckpt.get("class_names", class_names))
    model = build_model_from_checkpoint(ckpt, num_classes)

    positive_idx = class_names.index("PNEUMONIA") if "PNEUMONIA" in class_names else len(class_names) - 1
    y_true, y_pred, y_score = collect_test_outputs(model, test_loader, positive_idx)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=positive_idx)
    roc_auc = float(auc(fpr, tpr))

    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "arch": arch,
        "checkpoint": str(ckpt_path.name),
        "best_epoch": ckpt.get("best_epoch", "N/A"),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score, pos_label=positive_idx)), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "parameters": total_params,
        "temperature": ckpt.get("temperature", 1.0),
    }


def discover_checkpoints(include_default: bool) -> list[Path]:
    """Auto-discover architecture-specific checkpoints."""
    found: list[Path] = []
    for name, path in DEFAULT_CHECKPOINT_MAP.items():
        if path.exists():
            found.append(path)
    if include_default:
        default = CHECKPOINT_DIR / "best_model.pt"
        if default.exists() and default not in found:
            found.append(default)
    return found


def save_comparison_artifacts(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved comparison CSV to {csv_path}")

    txt_path = out_dir / "model_comparison.txt"
    with txt_path.open("w", encoding="utf-8") as fp:
        fp.write("=" * 80 + "\n")
        fp.write("Model Comparison — Test Set Evaluation\n")
        fp.write("=" * 80 + "\n\n")

        header = f"{'Architecture':<15} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9} {'Params':>12}\n"
        fp.write(header)
        fp.write("-" * 80 + "\n")
        for r in results:
            params_str = f"{r['parameters']:,}"
            fp.write(
                f"{r['arch']:<15} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
                f"{r['recall']:>8.4f} {r['f1']:>7.4f} {r['roc_auc']:>9.4f} {params_str:>12}\n"
            )
        fp.write("-" * 80 + "\n\n")

        best_by_f1 = max(results, key=lambda x: x["f1"])
        best_by_auc = max(results, key=lambda x: x["roc_auc"])
        smallest = min(results, key=lambda x: x["parameters"])

        fp.write("Key Findings:\n")
        fp.write(f"  Best F1 score:   {best_by_f1['arch']} (F1={best_by_f1['f1']:.4f})\n")
        fp.write(f"  Best ROC-AUC:    {best_by_auc['arch']} (AUC={best_by_auc['roc_auc']:.4f})\n")
        fp.write(f"  Smallest model:  {smallest['arch']} ({smallest['parameters']:,} params)\n\n")

        fp.write("Trade-off Analysis:\n")
        if len(results) >= 2:
            transfer = [r for r in results if r["arch"] != "simple_cnn"]
            baseline = [r for r in results if r["arch"] == "simple_cnn"]
            if transfer and baseline:
                bl = baseline[0]
                tf = transfer[0]
                fp.write(
                    f"  Transfer learning ({tf['arch']}) vs baseline CNN: "
                    f"F1 {'improved' if tf['f1'] > bl['f1'] else 'decreased'} by "
                    f"{abs(tf['f1'] - bl['f1']):.4f}, "
                    f"ROC-AUC {'improved' if tf['roc_auc'] > bl['roc_auc'] else 'decreased'} by "
                    f"{abs(tf['roc_auc'] - bl['roc_auc']):.4f}.\n"
                )
                fp.write(
                    f"  Complexity cost: {tf['arch']} has {tf['parameters']:,} params "
                    f"vs {bl['parameters']:,} for simple_cnn "
                    f"({tf['parameters'] / bl['parameters']:.1f}x larger).\n"
                )
            fp.write(
                "\n  Transfer learning with ImageNet-pretrained backbones is expected to\n"
                "  outperform a randomly initialized shallow CNN on medical imaging tasks,\n"
                "  where limited dataset size constrains the capacity of models trained from\n"
                "  scratch. The trade-off is increased inference latency and model size.\n"
            )
    print(f"Saved comparison report to {txt_path}")

    if plt is None or len(results) < 2:
        return

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    archs = [r["arch"] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(archs)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        offset = (i - len(archs) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=r["arch"])

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to {out_dir / 'model_comparison.png'}")


def main() -> None:
    args = parse_args()

    ckpt_paths: list[Path]
    if args.checkpoints:
        ckpt_paths = [Path(p) for p in args.checkpoints]
        for p in ckpt_paths:
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
    else:
        ckpt_paths = discover_checkpoints(include_default=args.include_default)

    if not ckpt_paths:
        print(
            "No checkpoints found. Train models first:\n"
            "  python -m backend.training.train --arch simple_cnn --epochs-head 10 --epochs-finetune 0\n"
            "  python -m backend.training.train --arch densenet121 --epochs-head 5 --epochs-finetune 5\n"
            "\n"
            "Then save each with a unique name, e.g.:\n"
            "  copy backend\\checkpoints\\best_model.pt backend\\checkpoints\\best_model_densenet121.pt"
        )
        return

    print(f"Comparing {len(ckpt_paths)} checkpoint(s): {[p.name for p in ckpt_paths]}")

    dataset_root = settings.dataset_root
    transforms_map = build_transforms(settings.image_size, augment=False)
    test_ds = build_imagefolder_dataset(dataset_root, "test", transforms_map["test"])
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    results: list[dict] = []
    for ckpt_path in ckpt_paths:
        print(f"Evaluating {ckpt_path.name}...")
        row = evaluate_checkpoint(ckpt_path, test_loader, test_ds.classes)
        results.append(row)
        print(f"  {row['arch']}: accuracy={row['accuracy']}, f1={row['f1']}, roc_auc={row['roc_auc']}")

    results.sort(key=lambda r: r["roc_auc"], reverse=True)
    save_comparison_artifacts(results, ARTIFACT_DIR)
    print("Model comparison complete.")


if __name__ == "__main__":
    main()
