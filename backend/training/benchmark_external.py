"""Quick reproducible external benchmark runner.

Evaluates CSV-based external datasets with the same binary label schema:
- image_path: path to image
- label: 0/1 (normal/pneumonia)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    required = {"label", "pred_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    y_true = df["label"].astype(int)
    y_prob = df["pred_prob"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "dataset": csv_path.stem,
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="List of prediction CSV files")
    parser.add_argument("--output", default="backend/training/external_benchmark_results.json")
    args = parser.parse_args()

    results = [evaluate_csv(Path(p)) for p in args.inputs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
