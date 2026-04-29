"""
Prepare the Stanford CheXpert dataset for binary pneumonia classification training.

This script:
  1. Reads train.csv / valid.csv from the CheXpert download.
  2. Filters to frontal views only (AP / PA), excluding lateral.
  3. Applies U-Ones policy for uncertainty labels (uncertain → positive).
  4. Splits at the PATIENT level to prevent data leakage.
  5. Subsamples the majority class (No Finding) to control class imbalance.
  6. Copies only the needed images into an ImageFolder layout:
       data/chexpert_processed/train/NORMAL/
       data/chexpert_processed/train/PNEUMONIA/
       data/chexpert_processed/val/NORMAL/
       data/chexpert_processed/val/PNEUMONIA/
       data/chexpert_processed/test/NORMAL/
       data/chexpert_processed/test/PNEUMONIA/

Usage:
    python -m backend.training.prepare_chexpert [--chexpert-root data/CheXpert-v1.0] [--output data/chexpert_processed]
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CheXpert dataset for training.")
    parser.add_argument(
        "--chexpert-root",
        type=str,
        default="data/CheXpert-v1.0",
        help="Root directory of the CheXpert dataset (contains train/ valid/ and CSVs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chexpert_processed",
        help="Output directory for the ImageFolder structure.",
    )
    parser.add_argument(
        "--normal-ratio",
        type=float,
        default=3.0,
        help="Ratio of Normal-to-Pneumonia images. 0 = use ALL normals.",
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
        "--uncertainty-policy",
        type=str,
        default="u-ones",
        choices=["u-ones", "u-zeros", "u-ignore"],
        help="How to handle uncertain labels: u-ones (positive), u-zeros (negative), u-ignore (discard).",
    )
    parser.add_argument(
        "--use-expert-labels",
        action="store_true",
        help="Use CheXpert expert-labeled validation set as the held-out test split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


PATIENT_ID_RE = re.compile(r"patient(\d+)", re.IGNORECASE)


def extract_patient_id(path_str: str) -> int:
    """Extract numeric patient ID from CheXpert path like 'CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg'."""
    match = PATIENT_ID_RE.search(path_str)
    if not match:
        raise ValueError(f"Cannot extract patient ID from path: {path_str}")
    return int(match.group(1))


def read_chexpert_csv(csv_path: Path) -> list[dict]:
    """Read a CheXpert CSV file and return a list of row dicts."""
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def is_frontal(row: dict) -> bool:
    """Keep only frontal views (Frontal/AP/PA)."""
    view = (row.get("Frontal/Lateral") or "").strip().lower()
    return view == "frontal"


def resolve_label(raw_value: str, policy: str) -> int | None:
    """Resolve a CheXpert label value (1.0, 0.0, -1.0, or blank) to binary.

    Returns 1 (positive), 0 (negative), or None (skip this row).
    """
    raw = raw_value.strip()
    if raw == "" or raw == "nan":
        return 0  # blank = not mentioned = negative
    val = float(raw)
    if val == 1.0:
        return 1
    if val == 0.0:
        return 0
    # val == -1.0 → uncertainty
    if policy == "u-ones":
        return 1
    if policy == "u-zeros":
        return 0
    return None  # u-ignore


def filter_binary(
    rows: list[dict],
    uncertainty_policy: str,
) -> tuple[list[dict], list[dict]]:
    """Split rows into pneumonia-positive and normal groups.

    Pneumonia: row where Pneumonia label resolves to 1.
    Normal: row where "No Finding" == 1.0 and all pathology labels are 0 or blank.
    """
    pneumonia: list[dict] = []
    normal: list[dict] = []

    pathology_columns = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices",
    ]

    for row in rows:
        pneumonia_raw = (row.get("Pneumonia") or "").strip()
        pneumonia_label = resolve_label(pneumonia_raw, uncertainty_policy)

        if pneumonia_label == 1:
            pneumonia.append(row)
            continue

        if pneumonia_label is None:
            continue

        no_finding_raw = (row.get("No Finding") or "").strip()
        if no_finding_raw == "1.0" or no_finding_raw == "1":
            has_other = False
            for col in pathology_columns:
                if col == "Pneumonia":
                    continue
                val = resolve_label((row.get(col) or "").strip(), uncertainty_policy)
                if val == 1:
                    has_other = True
                    break
            if not has_other:
                normal.append(row)

    return pneumonia, normal


def patient_level_split(
    rows: list[dict],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows by patient ID so no patient appears in multiple splits."""
    patient_to_rows: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        pid = extract_patient_id(row["Path"])
        patient_to_rows[pid].append(row)

    patient_ids = sorted(patient_to_rows.keys())
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_pids = set(patient_ids[:n_train])
    val_pids = set(patient_ids[n_train : n_train + n_val])
    test_pids = set(patient_ids[n_train + n_val :])

    train_rows = [r for pid in train_pids for r in patient_to_rows[pid]]
    val_rows = [r for pid in val_pids for r in patient_to_rows[pid]]
    test_rows = [r for pid in test_pids for r in patient_to_rows[pid]]

    return train_rows, val_rows, test_rows


def subsample(rows: list[dict], target_count: int, rng: random.Random) -> list[dict]:
    if target_count <= 0 or len(rows) <= target_count:
        return rows
    return rng.sample(rows, target_count)


def copy_images(
    rows: list[dict],
    chexpert_root: Path,
    dest_dir: Path,
    label_name: str,
) -> int:
    """Copy images from CheXpert tree into ImageFolder destination.

    Returns number of files successfully copied.
    """
    copied = 0
    for row in rows:
        rel_path = row["Path"]
        # CheXpert paths may start with "CheXpert-v1.0/" or just "train/..."
        src = chexpert_root / rel_path
        if not src.exists():
            parts = Path(rel_path).parts
            if parts and parts[0].lower().startswith("chexpert"):
                src = chexpert_root / Path(*parts[1:])
            if not src.exists():
                src = chexpert_root.parent / rel_path

        if not src.exists():
            continue

        out_name = f"{extract_patient_id(rel_path)}_{src.name}"
        dest = dest_dir / label_name / out_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            shutil.copy2(src, dest)
        copied += 1

    return copied


def verify_no_leakage(
    train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], label: str
) -> None:
    train_pids = {extract_patient_id(r["Path"]) for r in train_rows}
    val_pids = {extract_patient_id(r["Path"]) for r in val_rows}
    test_pids = {extract_patient_id(r["Path"]) for r in test_rows}

    tv = train_pids & val_pids
    tt = train_pids & test_pids
    vt = val_pids & test_pids

    if tv or tt or vt:
        print(f"\n  WARNING: Patient leakage in {label}!")
        if tv:
            print(f"    train & val: {len(tv)} patients")
        if tt:
            print(f"    train & test: {len(tt)} patients")
        if vt:
            print(f"    val & test: {len(vt)} patients")
    else:
        print(f"  {label}: No patient leakage across splits.")


def save_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    pneu_train: list, pneu_val: list, pneu_test: list,
    norm_train: list, norm_val: list, norm_test: list,
) -> None:
    manifest_path = output_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as fp:
        fp.write("CheXpert Preparation Manifest\n")
        fp.write(f"seed={args.seed}\n")
        fp.write(f"uncertainty_policy={args.uncertainty_policy}\n")
        fp.write(f"normal_ratio={args.normal_ratio}\n")
        fp.write(f"train_frac={args.train_frac}\n")
        fp.write(f"val_frac={args.val_frac}\n\n")
        for split_name, p_rows, n_rows in [
            ("train", pneu_train, norm_train),
            ("val", pneu_val, norm_val),
            ("test", pneu_test, norm_test),
        ]:
            fp.write(f"[{split_name}]\n")
            fp.write(f"pneumonia_count={len(p_rows)}\n")
            fp.write(f"normal_count={len(n_rows)}\n\n")
    print(f"\n  Manifest saved to {manifest_path}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    chexpert_root = Path(args.chexpert_root)
    output_dir = Path(args.output)

    train_csv = chexpert_root / "train.csv"
    valid_csv = chexpert_root / "valid.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_csv}. "
            "Download the CheXpert dataset from https://stanfordmlgroup.github.io/competitions/chexpert/"
        )

    # Step 1: Read CSVs
    print("Step 1/6: Reading CheXpert CSVs...")
    train_rows = read_chexpert_csv(train_csv)
    print(f"  train.csv: {len(train_rows):,} entries")

    valid_rows = []
    if valid_csv.exists():
        valid_rows = read_chexpert_csv(valid_csv)
        print(f"  valid.csv: {len(valid_rows):,} entries")
    else:
        print("  valid.csv not found; expert labels will not be used as external test.")

    all_rows = train_rows  # valid_rows reserved for expert test

    # Step 2: Filter frontal views
    print("\nStep 2/6: Filtering to frontal views only...")
    frontal_rows = [r for r in all_rows if is_frontal(r)]
    print(f"  Frontal views: {len(frontal_rows):,} / {len(all_rows):,}")

    frontal_valid = [r for r in valid_rows if is_frontal(r)] if valid_rows else []
    if frontal_valid:
        print(f"  Frontal expert-labeled: {len(frontal_valid):,} / {len(valid_rows):,}")

    # Step 3: Binary filter
    print(f"\nStep 3/6: Filtering for Pneumonia vs Normal (policy={args.uncertainty_policy})...")
    pneumonia, normal = filter_binary(frontal_rows, args.uncertainty_policy)
    print(f"  Pneumonia images: {len(pneumonia):,}")
    print(f"  Normal images: {len(normal):,}")

    # Step 4: Patient-level split
    print("\nStep 4/6: Patient-level splitting...")
    pneu_train, pneu_val, pneu_test = patient_level_split(
        pneumonia, args.train_frac, args.val_frac, rng
    )
    norm_train, norm_val, norm_test = patient_level_split(
        normal, args.train_frac, args.val_frac, rng
    )

    print(f"  Pneumonia: train={len(pneu_train)}, val={len(pneu_val)}, test={len(pneu_test)}")
    print(f"  Normal (pre-subsample): train={len(norm_train)}, val={len(norm_val)}, test={len(norm_test)}")

    # Step 5: Subsample normals
    if args.normal_ratio > 0:
        print(f"\nStep 5/6: Subsampling Normal class (ratio {args.normal_ratio}:1)...")
        norm_train = subsample(norm_train, int(len(pneu_train) * args.normal_ratio), rng)
        norm_val = subsample(norm_val, int(len(pneu_val) * args.normal_ratio), rng)
        norm_test = subsample(norm_test, int(len(pneu_test) * args.normal_ratio), rng)
        print(f"  Normal after subsample: train={len(norm_train)}, val={len(norm_val)}, test={len(norm_test)}")
    else:
        print("\nStep 5/6: Using ALL normal images (no subsampling).")

    # Step 6: Copy images
    print("\nStep 6/6: Copying images to ImageFolder layout...")
    total_copied = 0
    for split_name, p_rows, n_rows in [
        ("train", pneu_train, norm_train),
        ("val", pneu_val, norm_val),
        ("test", pneu_test, norm_test),
    ]:
        split_dir = output_dir / split_name
        p_copied = copy_images(p_rows, chexpert_root, split_dir, "PNEUMONIA")
        n_copied = copy_images(n_rows, chexpert_root, split_dir, "NORMAL")
        total_copied += p_copied + n_copied
        print(f"  {split_name}: PNEUMONIA={p_copied}, NORMAL={n_copied}")

    # Expert-labeled test set
    if args.use_expert_labels and frontal_valid:
        print("\n  Copying expert-labeled validation set as external test...")
        expert_pneu, expert_norm = filter_binary(frontal_valid, "u-ones")
        expert_dir = output_dir / "expert_test"
        ep = copy_images(expert_pneu, chexpert_root, expert_dir, "PNEUMONIA")
        en = copy_images(expert_norm, chexpert_root, expert_dir, "NORMAL")
        print(f"  expert_test: PNEUMONIA={ep}, NORMAL={en}")

    # Verification
    print("\n" + "=" * 60)
    print("  DATASET PREPARATION COMPLETE")
    print("=" * 60)
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        n_normal = len(list((split_dir / "NORMAL").glob("*"))) if (split_dir / "NORMAL").exists() else 0
        n_pneu = len(list((split_dir / "PNEUMONIA").glob("*"))) if (split_dir / "PNEUMONIA").exists() else 0
        print(f"  {split:>5s}:  NORMAL={n_normal:>6,}   PNEUMONIA={n_pneu:>6,}   total={n_normal + n_pneu:>6,}")

    print(f"\n  Output directory: {output_dir.resolve()}")
    print(f"  Ready for training with: DATASET_ROOT={output_dir}")

    verify_no_leakage(pneu_train, pneu_val, pneu_test, "Pneumonia")
    verify_no_leakage(norm_train, norm_val, norm_test, "Normal")
    save_manifest(output_dir, args, pneu_train, pneu_val, pneu_test, norm_train, norm_val, norm_test)


if __name__ == "__main__":
    main()
