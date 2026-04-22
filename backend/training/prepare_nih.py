"""
Prepare the NIH ChestX-ray14 (CXR8) dataset for binary classification training.

This script:
  1. Reads Data_Entry_2017_v2020.csv to identify Pneumonia and No-Finding images.
  2. Splits at the PATIENT level (not image level) to prevent data leakage.
  3. Subsamples the majority class (No Finding) to control class imbalance.
  4. Selectively extracts ONLY the needed images from tar.gz archives (disk-efficient).
  5. Creates an ImageFolder layout:
       data/nih_processed/train/NORMAL/
       data/nih_processed/train/PNEUMONIA/
       data/nih_processed/val/NORMAL/
       data/nih_processed/val/PNEUMONIA/
       data/nih_processed/test/NORMAL/
       data/nih_processed/test/PNEUMONIA/

Usage:
    python -m backend.training.prepare_nih [--nih-root data/CXR8] [--output data/nih_processed]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import tarfile
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NIH CXR8 dataset for training.")
    parser.add_argument(
        "--nih-root",
        type=str,
        default="data/CXR8",
        help="Root directory of the NIH CXR8 dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nih_processed",
        help="Output directory for the ImageFolder structure.",
    )
    parser.add_argument(
        "--normal-ratio",
        type=float,
        default=3.0,
        help=(
            "Ratio of Normal-to-Pneumonia images for class balance. "
            "Set to 0 to use ALL normal images (very imbalanced)."
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of patients for the training split.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of patients for the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---- Step 1: Read CSV and build label lookup ----

def read_labels(csv_path: Path) -> dict[str, dict]:
    """Parse Data_Entry_2017_v2020.csv.

    Returns dict mapping image filename -> {labels, patient_id, age, sex, view}.
    """
    entries: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            filename = row["Image Index"].strip()
            labels_raw = row["Finding Labels"].strip()
            label_set = frozenset(l.strip() for l in labels_raw.split("|"))
            entries[filename] = {
                "labels": label_set,
                "patient_id": int(row["Patient ID"]),
                "age": row.get("Patient Age", ""),
                "sex": row.get("Patient Sex", ""),
                "view": row.get("View Position", ""),
            }
    return entries


# ---- Step 2: Filter for binary classification ----

def filter_binary(
    entries: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Split entries into pneumonia and normal groups.

    Pneumonia: any image whose labels include "Pneumonia" (even multi-label).
    Normal: images whose ONLY label is "No Finding".
    Ambiguous cases (pathology but no pneumonia) are excluded for clean binary labels.
    """
    pneumonia: dict[str, dict] = {}
    normal: dict[str, dict] = {}

    for fname, meta in entries.items():
        if "Pneumonia" in meta["labels"]:
            pneumonia[fname] = meta
        elif meta["labels"] == frozenset({"No Finding"}):
            normal[fname] = meta

    return pneumonia, normal


# ---- Step 3: Patient-level split ----

def patient_level_split(
    images: dict[str, dict],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    """Split image filenames by patient ID to avoid data leakage.

    Returns (train_files, val_files, test_files).
    """
    patient_to_files: dict[int, list[str]] = defaultdict(list)
    for fname, meta in images.items():
        patient_to_files[meta["patient_id"]].append(fname)

    patient_ids = sorted(patient_to_files.keys())
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train : n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val :])

    train_files = [f for pid in train_patients for f in patient_to_files[pid]]
    val_files = [f for pid in val_patients for f in patient_to_files[pid]]
    test_files = [f for pid in test_patients for f in patient_to_files[pid]]

    return train_files, val_files, test_files


# ---- Step 4: Balance normal class by subsampling ----

def subsample_per_split(
    files: list[str],
    target_count: int,
    rng: random.Random,
) -> list[str]:
    """Randomly subsample files to target_count (or return all if fewer)."""
    if target_count <= 0 or len(files) <= target_count:
        return files
    return rng.sample(files, target_count)


# ---- Step 5: Selective extraction from tar.gz ----

def selective_extract(
    archive_dir: Path,
    needed_files: set[str],
    file_to_dest: dict[str, Path],
) -> int:
    """Extract only needed images from tar.gz archives directly to their
    final ImageFolder destination.

    Returns number of files successfully extracted.
    """
    archives = sorted(archive_dir.glob("images_*.tar.gz"))
    if not archives:
        raise FileNotFoundError(
            f"No images_*.tar.gz found in {archive_dir}. "
            "Please download the NIH ChestX-ray14 dataset archives."
        )

    remaining = set(needed_files)
    extracted = 0

    for i, archive_path in enumerate(archives, 1):
        if not remaining:
            print(f"  [{i}/{len(archives)}] All needed images found, skipping {archive_path.name}.")
            continue

        print(f"  [{i}/{len(archives)}] Scanning {archive_path.name} ({len(remaining):,} images remaining)...")
        found_this_archive = 0

        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                basename = os.path.basename(member.name)
                if basename not in remaining:
                    continue

                dest_path = file_to_dest[basename]
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                with tar.extractfile(member) as src:
                    if src is None:
                        continue
                    dest_path.write_bytes(src.read())

                remaining.discard(basename)
                extracted += 1
                found_this_archive += 1

        print(f"           Extracted {found_this_archive:,} images from this archive.")

    if remaining:
        print(f"\n  WARNING: {len(remaining):,} images were not found in any archive:")
        for name in sorted(remaining)[:10]:
            print(f"    - {name}")
        if len(remaining) > 10:
            print(f"    ... and {len(remaining) - 10} more")

    return extracted


# ---- Main pipeline ----

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    nih_root = Path(args.nih_root)
    output_dir = Path(args.output)
    csv_path = nih_root / "Data_Entry_2017_v2020.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found at {csv_path}")

    # --- Read labels ---
    print("Step 1/5: Reading labels from CSV...")
    entries = read_labels(csv_path)
    print(f"  Total entries in CSV: {len(entries):,}")

    # --- Filter binary ---
    print("\nStep 2/5: Filtering for Pneumonia vs Normal...")
    pneumonia, normal = filter_binary(entries)
    print(f"  Pneumonia images (any label containing Pneumonia): {len(pneumonia):,}")
    print(f"  Normal images (No Finding only): {len(normal):,}")

    # --- Patient-level split ---
    print("\nStep 3/5: Patient-level splitting...")

    pneu_train, pneu_val, pneu_test = patient_level_split(
        pneumonia, args.train_frac, args.val_frac, rng,
    )
    norm_train, norm_val, norm_test = patient_level_split(
        normal, args.train_frac, args.val_frac, rng,
    )

    print(f"  Pneumonia split: train={len(pneu_train)}, val={len(pneu_val)}, test={len(pneu_test)}")
    print(f"  Normal split (before subsampling): train={len(norm_train)}, val={len(norm_val)}, test={len(norm_test)}")

    # --- Subsample normal ---
    if args.normal_ratio > 0:
        print(f"\nStep 4/5: Subsampling Normal class (ratio {args.normal_ratio}:1)...")
        norm_train = subsample_per_split(norm_train, int(len(pneu_train) * args.normal_ratio), rng)
        norm_val = subsample_per_split(norm_val, int(len(pneu_val) * args.normal_ratio), rng)
        norm_test = subsample_per_split(norm_test, int(len(pneu_test) * args.normal_ratio), rng)
        print(f"  Normal after subsampling: train={len(norm_train)}, val={len(norm_val)}, test={len(norm_test)}")
    else:
        print("\nStep 4/5: Using ALL normal images (no subsampling).")

    # --- Build file-to-destination mapping ---
    file_to_dest: dict[str, Path] = {}
    for fname in pneu_train:
        file_to_dest[fname] = output_dir / "train" / "PNEUMONIA" / fname
    for fname in pneu_val:
        file_to_dest[fname] = output_dir / "val" / "PNEUMONIA" / fname
    for fname in pneu_test:
        file_to_dest[fname] = output_dir / "test" / "PNEUMONIA" / fname
    for fname in norm_train:
        file_to_dest[fname] = output_dir / "train" / "NORMAL" / fname
    for fname in norm_val:
        file_to_dest[fname] = output_dir / "val" / "NORMAL" / fname
    for fname in norm_test:
        file_to_dest[fname] = output_dir / "test" / "NORMAL" / fname

    needed_files = set(file_to_dest.keys())

    # Skip already-extracted files
    already_exists = {fname for fname, dest in file_to_dest.items() if dest.exists()}
    if already_exists:
        print(f"\n  {len(already_exists):,} images already in output directory, skipping those.")
        needed_files -= already_exists

    if not needed_files:
        print("\n  All images already extracted! Skipping archive scanning.")
    else:
        print(f"\nStep 5/5: Extracting {len(needed_files):,} images from tar.gz archives...")
        print("  (Only extracting what we need — NOT the full 112K image set)")
        archive_dir = nih_root / "images"
        extracted = selective_extract(archive_dir, needed_files, file_to_dest)
        print(f"\n  Extracted {extracted:,} new images.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  DATASET PREPARATION COMPLETE")
    print("=" * 60)
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        n_normal = len(list((split_dir / "NORMAL").glob("*.png"))) if (split_dir / "NORMAL").exists() else 0
        n_pneu = len(list((split_dir / "PNEUMONIA").glob("*.png"))) if (split_dir / "PNEUMONIA").exists() else 0
        print(f"  {split:>5s}:  NORMAL={n_normal:>5,}   PNEUMONIA={n_pneu:>5,}   total={n_normal + n_pneu:>5,}")

    print(f"\n  Output directory: {output_dir.resolve()}")
    print(f"  Ready for training with: DATASET_ROOT={output_dir}")

    # Verify no patient leakage between splits
    _verify_no_leakage(pneumonia, normal, pneu_train, pneu_val, pneu_test, norm_train, norm_val, norm_test)

    # Save a manifest for reproducibility
    _save_manifest(output_dir, args, pneu_train, pneu_val, pneu_test, norm_train, norm_val, norm_test)


def _verify_no_leakage(
    pneumonia: dict, normal: dict,
    pneu_train: list, pneu_val: list, pneu_test: list,
    norm_train: list, norm_val: list, norm_test: list,
) -> None:
    """Sanity check: no patient appears in more than one split."""

    def _patients(fnames, lookup):
        return {lookup[f]["patient_id"] for f in fnames if f in lookup}

    pneu_patients = {
        "train": _patients(pneu_train, pneumonia),
        "val": _patients(pneu_val, pneumonia),
        "test": _patients(pneu_test, pneumonia),
    }
    norm_patients = {
        "train": _patients(norm_train, normal),
        "val": _patients(norm_val, normal),
        "test": _patients(norm_test, normal),
    }

    for class_name, splits in [("Pneumonia", pneu_patients), ("Normal", norm_patients)]:
        tv = splits["train"] & splits["val"]
        tt = splits["train"] & splits["test"]
        vt = splits["val"] & splits["test"]
        if tv or tt or vt:
            print(f"\n  WARNING: Patient leakage detected in {class_name}!")
            if tv:
                print(f"    train & val overlap: {len(tv)} patients")
            if tt:
                print(f"    train & test overlap: {len(tt)} patients")
            if vt:
                print(f"    val & test overlap: {len(vt)} patients")
        else:
            print(f"  {class_name}: No patient leakage across splits.")


def _save_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    pneu_train: list, pneu_val: list, pneu_test: list,
    norm_train: list, norm_val: list, norm_test: list,
) -> None:
    """Save a text manifest recording the split for reproducibility."""
    manifest_path = output_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as fp:
        fp.write(f"NIH CXR8 Preparation Manifest\n")
        fp.write(f"seed={args.seed}\n")
        fp.write(f"normal_ratio={args.normal_ratio}\n")
        fp.write(f"train_frac={args.train_frac}\n")
        fp.write(f"val_frac={args.val_frac}\n\n")
        for split_name, pneu_list, norm_list in [
            ("train", pneu_train, norm_train),
            ("val", pneu_val, norm_val),
            ("test", pneu_test, norm_test),
        ]:
            fp.write(f"[{split_name}]\n")
            fp.write(f"pneumonia_count={len(pneu_list)}\n")
            fp.write(f"normal_count={len(norm_list)}\n")
            for fname in sorted(pneu_list):
                fp.write(f"PNEUMONIA/{fname}\n")
            for fname in sorted(norm_list):
                fp.write(f"NORMAL/{fname}\n")
            fp.write("\n")
    print(f"\n  Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
