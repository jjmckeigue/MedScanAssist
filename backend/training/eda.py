"""Exploratory Data Analysis (EDA) script for the chest X-ray dataset.

Produces artifacts in backend/artifacts/eda/:
  - class_distribution.csv / .png  — per-split class counts and bar chart
  - image_stats.csv / .txt         — width, height, aspect ratio, and channel statistics
  - sample_grid.png                — random grid of sample images per class
  - eda_summary.txt                — textual summary with bias and limitation notes

Usage:
    python -m backend.training.eda
    python -m backend.training.eda --dataset-root data/raw/chest_xray --sample-grid-n 4
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.config import settings

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SPLITS = ["train", "val", "test"]
OUT_DIR = Path("./backend/artifacts/eda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on the CXR dataset.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override dataset root (defaults to DATASET_ROOT from .env).",
    )
    parser.add_argument(
        "--sample-grid-n",
        type=int,
        default=4,
        help="Number of sample images per class per split in the sample grid.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_image_paths(dataset_root: Path) -> dict[str, dict[str, list[Path]]]:
    """Return {split: {class_name: [paths]}}."""
    result: dict[str, dict[str, list[Path]]] = {}
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        result[split] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            files = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            )
            result[split][class_dir.name] = files
    return result


def save_class_distribution(
    paths_by_split: dict[str, dict[str, list[Path]]], out_dir: Path
) -> None:
    rows: list[dict[str, str | int]] = []
    for split, classes in paths_by_split.items():
        total = sum(len(v) for v in classes.values())
        for class_name, files in sorted(classes.items()):
            count = len(files)
            pct = (count / total * 100) if total else 0.0
            rows.append({
                "split": split,
                "class": class_name,
                "count": count,
                "total": total,
                "pct": f"{pct:.1f}",
            })

    csv_path = out_dir / "class_distribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["split", "class", "count", "total", "pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved class distribution to {csv_path}")

    if plt is None:
        return

    splits_present = list(paths_by_split.keys())
    all_classes = sorted({c for cls in paths_by_split.values() for c in cls})
    n_splits = len(splits_present)

    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 4), squeeze=False)
    for idx, split in enumerate(splits_present):
        ax = axes[0][idx]
        counts = [len(paths_by_split[split].get(c, [])) for c in all_classes]
        bars = ax.bar(all_classes, counts, color=["#4c9ed9", "#e06666"])
        ax.set_title(f"{split} ({sum(counts):,} images)")
        ax.set_ylabel("Count")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Class Distribution by Split", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved class distribution chart to {out_dir / 'class_distribution.png'}")


def compute_image_stats(
    paths_by_split: dict[str, dict[str, list[Path]]], out_dir: Path, max_sample: int = 500
) -> dict:
    """Sample images and collect width, height, aspect ratio, channels, and mode."""
    all_paths: list[Path] = []
    for classes in paths_by_split.values():
        for files in classes.values():
            all_paths.extend(files)

    sample = random.sample(all_paths, min(max_sample, len(all_paths)))

    widths, heights, aspects, channels, modes_counter = [], [], [], [], Counter()
    corrupt_count = 0

    for p in sample:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w / h if h else 0)
                ch = len(img.getbands())
                channels.append(ch)
                modes_counter[img.mode] += 1
        except Exception:
            corrupt_count += 1

    stats = {
        "images_sampled": len(sample),
        "corrupt_or_unreadable": corrupt_count,
        "width_min": int(np.min(widths)) if widths else 0,
        "width_max": int(np.max(widths)) if widths else 0,
        "width_mean": float(np.mean(widths)) if widths else 0,
        "width_std": float(np.std(widths)) if widths else 0,
        "height_min": int(np.min(heights)) if heights else 0,
        "height_max": int(np.max(heights)) if heights else 0,
        "height_mean": float(np.mean(heights)) if heights else 0,
        "height_std": float(np.std(heights)) if heights else 0,
        "aspect_ratio_mean": float(np.mean(aspects)) if aspects else 0,
        "aspect_ratio_std": float(np.std(aspects)) if aspects else 0,
        "channel_counts": dict(Counter(channels)),
        "image_modes": dict(modes_counter),
    }

    csv_path = out_dir / "image_stats.csv"
    flat_rows = [{"metric": k, "value": str(v)} for k, v in stats.items()]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(flat_rows)

    txt_path = out_dir / "image_stats.txt"
    with txt_path.open("w", encoding="utf-8") as fp:
        fp.write("Image Size / Quality Statistics\n")
        fp.write(f"(sampled {stats['images_sampled']} of {len(all_paths)} total images)\n\n")
        fp.write(f"Width  — min: {stats['width_min']}, max: {stats['width_max']}, "
                 f"mean: {stats['width_mean']:.1f}, std: {stats['width_std']:.1f}\n")
        fp.write(f"Height — min: {stats['height_min']}, max: {stats['height_max']}, "
                 f"mean: {stats['height_mean']:.1f}, std: {stats['height_std']:.1f}\n")
        fp.write(f"Aspect ratio — mean: {stats['aspect_ratio_mean']:.3f}, "
                 f"std: {stats['aspect_ratio_std']:.3f}\n")
        fp.write(f"Channel counts: {stats['channel_counts']}\n")
        fp.write(f"Image modes: {stats['image_modes']}\n")
        if corrupt_count:
            fp.write(f"Corrupt / unreadable files: {corrupt_count}\n")

    print(f"Saved image stats to {txt_path}")

    if plt is not None and widths:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(widths, bins=30, color="#4c9ed9", edgecolor="white")
        axes[0].set_title("Width Distribution")
        axes[0].set_xlabel("Pixels")
        axes[1].hist(heights, bins=30, color="#e06666", edgecolor="white")
        axes[1].set_title("Height Distribution")
        axes[1].set_xlabel("Pixels")
        fig.suptitle("Image Dimension Distributions", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "image_dimensions.png", dpi=150)
        plt.close(fig)
        print(f"Saved dimension histogram to {out_dir / 'image_dimensions.png'}")

    return stats


def save_sample_grid(
    paths_by_split: dict[str, dict[str, list[Path]]], out_dir: Path, n: int = 4
) -> None:
    if plt is None:
        print("Matplotlib unavailable; skipping sample grid.")
        return

    splits_present = list(paths_by_split.keys())
    all_classes = sorted({c for cls in paths_by_split.values() for c in cls})

    n_rows = len(splits_present) * len(all_classes)
    fig, axes = plt.subplots(n_rows, n, figsize=(3 * n, 3 * n_rows), squeeze=False)
    row_idx = 0
    for split in splits_present:
        for cls in all_classes:
            files = paths_by_split[split].get(cls, [])
            chosen = random.sample(files, min(n, len(files)))
            for col, img_path in enumerate(chosen):
                try:
                    img = Image.open(img_path).convert("RGB").resize((224, 224))
                    axes[row_idx][col].imshow(np.array(img))
                except Exception:
                    pass
                axes[row_idx][col].axis("off")
                if col == 0:
                    axes[row_idx][col].set_ylabel(
                        f"{split}/{cls}", fontsize=9, rotation=0, labelpad=80, va="center"
                    )
            for col in range(len(chosen), n):
                axes[row_idx][col].axis("off")
            row_idx += 1

    fig.suptitle("Sample Images by Split and Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_grid.png", dpi=120)
    plt.close(fig)
    print(f"Saved sample grid to {out_dir / 'sample_grid.png'}")


def write_eda_summary(
    paths_by_split: dict[str, dict[str, list[Path]]],
    stats: dict,
    dataset_root: Path,
    out_dir: Path,
) -> None:
    txt_path = out_dir / "eda_summary.txt"
    with txt_path.open("w", encoding="utf-8") as fp:
        fp.write("=" * 70 + "\n")
        fp.write("MedScanAssist — Exploratory Data Analysis Summary\n")
        fp.write("=" * 70 + "\n\n")

        fp.write(f"Dataset root: {dataset_root}\n\n")

        fp.write("1. CLASS DISTRIBUTION\n")
        fp.write("-" * 40 + "\n")
        for split, classes in paths_by_split.items():
            total = sum(len(v) for v in classes.values())
            fp.write(f"  {split}: {total:,} images\n")
            for cls, files in sorted(classes.items()):
                pct = len(files) / total * 100 if total else 0
                fp.write(f"    {cls}: {len(files):,} ({pct:.1f}%)\n")
        fp.write("\n")

        all_counts = {}
        for classes in paths_by_split.values():
            for cls, files in classes.items():
                all_counts[cls] = all_counts.get(cls, 0) + len(files)
        if len(all_counts) == 2:
            vals = list(all_counts.values())
            ratio = max(vals) / min(vals) if min(vals) else float("inf")
            fp.write(f"  Overall class imbalance ratio: {ratio:.2f}:1\n\n")

        fp.write("2. IMAGE SIZE & QUALITY\n")
        fp.write("-" * 40 + "\n")
        fp.write(f"  Width range: {stats['width_min']}–{stats['width_max']} px "
                 f"(mean {stats['width_mean']:.0f})\n")
        fp.write(f"  Height range: {stats['height_min']}–{stats['height_max']} px "
                 f"(mean {stats['height_mean']:.0f})\n")
        fp.write(f"  Aspect ratio: {stats['aspect_ratio_mean']:.3f} "
                 f"(std {stats['aspect_ratio_std']:.3f})\n")
        fp.write(f"  Image modes: {stats['image_modes']}\n")
        if stats.get("corrupt_or_unreadable", 0):
            fp.write(f"  Corrupt/unreadable: {stats['corrupt_or_unreadable']}\n")
        fp.write("\n")

        fp.write("3. PREPROCESSING PIPELINE\n")
        fp.write("-" * 40 + "\n")
        fp.write("  - Resize to 224x224 for model input\n")
        fp.write("  - ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])\n")
        fp.write("  - Training augmentations: random crop, flip, rotation, affine, perspective,\n")
        fp.write("    color jitter, Gaussian blur, random erasing\n")
        fp.write("  - Inverse-frequency class weighting in loss function\n\n")

        fp.write("4. DATASET BIAS & LIMITATIONS\n")
        fp.write("-" * 40 + "\n")
        fp.write(
            "  a) Single-institution bias: The Kaggle Chest X-ray Pneumonia dataset\n"
            "     (Kermany et al. 2018) was collected from a single pediatric center\n"
            "     (Guangzhou Women and Children's Medical Center). Models trained on it\n"
            "     may not generalize to adult populations or other imaging hardware.\n\n"
            "  b) Class imbalance: The PNEUMONIA class typically outnumbers NORMAL by\n"
            "     roughly 3:1 in the training split, which can bias recall in favour of\n"
            "     the majority class. We mitigate this with inverse-frequency loss\n"
            "     weighting and threshold tuning.\n\n"
            "  c) Label quality: Labels were assigned by two expert physicians with a\n"
            "     third adjudicator, but inter-rater variability is inherent in CXR\n"
            "     interpretation. No distinction is made between bacterial and viral\n"
            "     pneumonia in our binary formulation.\n\n"
            "  d) Demographic metadata: No patient age, sex, or comorbidity metadata\n"
            "     is provided with the Kaggle v1 dataset, limiting fairness auditing.\n"
            "     The optional --audit-metadata-csv flag supports subgroup analysis\n"
            "     when metadata is available from other sources.\n\n"
            "  e) Image acquisition variability: X-rays vary in positioning, exposure,\n"
            "     and equipment. Augmentation and shortcut stress testing help detect\n"
            "     spurious correlations with non-anatomical cues (e.g., corner labels).\n\n"
            "  f) Small validation split: The original Kaggle split has only 16 images\n"
            "     in the validation set, which yields unreliable epoch-level metrics.\n"
            "     K-fold cross-validation (--kfold) is recommended for robust estimates.\n"
        )

        fp.write("\n5. FILES GENERATED BY THIS EDA\n")
        fp.write("-" * 40 + "\n")
        fp.write("  backend/artifacts/eda/class_distribution.csv\n")
        fp.write("  backend/artifacts/eda/class_distribution.png\n")
        fp.write("  backend/artifacts/eda/image_stats.csv\n")
        fp.write("  backend/artifacts/eda/image_stats.txt\n")
        fp.write("  backend/artifacts/eda/image_dimensions.png\n")
        fp.write("  backend/artifacts/eda/sample_grid.png\n")
        fp.write("  backend/artifacts/eda/eda_summary.txt\n")

    print(f"Saved EDA summary to {txt_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = Path(args.dataset_root) if args.dataset_root else settings.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. "
            "Download the dataset first (see README)."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running EDA on {dataset_root} → {OUT_DIR}")

    paths_by_split = collect_image_paths(dataset_root)
    if not paths_by_split:
        raise FileNotFoundError(f"No train/val/test splits found under {dataset_root}.")

    save_class_distribution(paths_by_split, OUT_DIR)
    stats = compute_image_stats(paths_by_split, OUT_DIR)
    save_sample_grid(paths_by_split, OUT_DIR, n=args.sample_grid_n)
    write_eda_summary(paths_by_split, stats, dataset_root, OUT_DIR)
    print("EDA complete.")


if __name__ == "__main__":
    main()
