# MedScanAssist

MedScanAssist is a capstone project for binary chest X-ray classification
(`pneumonia` vs `normal`) with explainability via Grad-CAM.

This repository is organized for:
- local script-first iteration in Python
- API-first serving via FastAPI
- reproducible CPU baseline deployment with Docker
- React frontend integration after API baseline is stable

## Stack

- Backend: Python 3.11, FastAPI, Uvicorn, PyTorch, torchvision
- Explainability: Grad-CAM utility service
- Frontend: React + Vite (Node 18+ recommended; `frontend/.nvmrc` included)
- Packaging: Docker + Docker Compose

## Project Structure

```text
medscanassist/
  backend/
    app/
      main.py
      config.py
      schemas.py
      routes/
      services/
    training/
      train.py
      evaluate.py
      gradcam.py
      data_utils.py
    requirements.txt
    Dockerfile
  frontend/
    src/
    package.json
  data/
    raw/
    processed/
  docker-compose.yml
  .env.example
```

## Quick Start (CPU Baseline)

1. Copy environment template:
   - `cp .env.example .env` (macOS/Linux)
   - `Copy-Item .env.example .env` (PowerShell)
2. Create and use a project virtual environment (recommended):
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
   - PowerShell:
     - `py -3 -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
3. Place dataset in `data/raw/chest_xray/` (see Dataset section).
4. Run backend:
   - Local: `python -m pip install -r backend/requirements.txt` then
     `python -m uvicorn backend.app.main:app --reload --port 8000`
   - Docker: `docker compose up --build backend`
5. Visit API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## Training (Transfer Learning)

Training uses transfer learning by default:

1. initialize a pretrained backbone (`DenseNet121` by default, `ResNet50` optional)
2. freeze backbone and train only the classifier head
3. unfreeze full network and fine-tune at a lower learning rate
4. save best checkpoint by validation accuracy

Run training:

- `python -m pip install -r backend/requirements-train.txt`
- `python -m backend.training.train --epochs-head 3 --epochs-finetune 2`

Fairness/generalization-focused options:

- inverse-frequency class weighting is enabled by default (`--disable-class-weighting` to turn off)
- optional external validation on another institution/dataset:
  - `python -m backend.training.train --external-test-root data/raw/chest_xray_external`
- optional subgroup fairness audit with metadata CSV:
  - `python -m backend.training.train --audit-metadata-csv data/processed/subgroup_metadata.csv --audit-path-column image_path --audit-group-columns sex,age_group,site`

Outputs:

- checkpoint: `backend/checkpoints/best_model.pt`
- checkpoint metadata includes: `best_epoch`, `best_val_acc`, `best_val_loss`
- metrics table: `backend/artifacts/training_metrics.csv`
- training curves image: `backend/artifacts/training_curves.png`
- confusion matrix: `backend/artifacts/confusion_matrix.csv` and `backend/artifacts/confusion_matrix.png`
- ROC curve: `backend/artifacts/roc_curve.csv` and `backend/artifacts/roc_curve.png`
- PR curve: `backend/artifacts/pr_curve.csv` and `backend/artifacts/pr_curve.png`
- calibration: `backend/artifacts/calibration_curve.csv`, `backend/artifacts/calibration_curve.png`, and `backend/artifacts/calibration_report.txt`
- threshold tuning: `backend/artifacts/threshold_analysis.csv`, `backend/artifacts/threshold_analysis.png`, and `backend/artifacts/threshold_recommendations.txt`
- shortcut stress test: `backend/artifacts/test_shortcut_stress_test.csv` and `backend/artifacts/test_shortcut_stress_report.txt`
- subgroup fairness audit (optional): `backend/artifacts/test_subgroup_fairness_metrics.csv` and `backend/artifacts/test_subgroup_fairness_summary.txt`
- external validation outputs (optional): `backend/artifacts/external/test/*`

## API Smoke Tests

- `python -m pip install -r backend/requirements-dev.txt`
- `python -m pytest backend/tests -q`

## PowerShell Shortcuts

From repo root:

- Start backend quickly (auto-venv + deps + `.env`): `.\scripts\dev.ps1`
- Run tests quickly (auto-venv + deps): `.\scripts\test.ps1`

Optional:

- Skip reinstall step when dependencies are already installed:
  - `.\scripts\dev.ps1 -SkipInstall`
  - `.\scripts\test.ps1 -SkipInstall`

## Dataset Ingestion (Kaggle v1)

Use this expected v1 layout under `data/raw/chest_xray/`:

```text
data/raw/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

### How to provide datasets to this repo

- Kaggle v1:
  1. Download/unzip locally.
  2. Move the `chest_xray` directory to `data/raw/chest_xray`.
  3. Confirm folder names remain uppercase (`NORMAL`, `PNEUMONIA`).
- CheXpert/MIMIC v2:
  - Create a new folder under `data/raw/` (for example
    `data/raw/chexpert` or `data/raw/mimic_cxr_jpg`) and update `DATASET_ROOT`
    in `.env`.

Raw dataset files are ignored by git via `.gitignore`.

Upload safety:

- backend rejects non-image uploads and payloads larger than `MAX_UPLOAD_BYTES` (default 8 MB)
- frontend enforces file type and max size checks before any upload attempt

Current model-serving logic is scaffolded and defaults to deterministic placeholder
scores until trained weights are present.

## Training Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs-head` | 3 | Epochs with frozen backbone (classifier head only) |
| `--epochs-finetune` | 2 | Epochs for gradual unfreezing after head training |
| `--epochs-last-block` | 1 | Max epochs for last block + head before full unfreeze |
| `--warmup-epochs` | 0 | Linear LR warmup epochs (1e-7 to `--lr-head`) |
| `--label-smoothing` | 0.0 | Label smoothing factor for CrossEntropyLoss (0.1 recommended) |
| `--arch` | env `MODEL_ARCH` | Architecture override: `densenet121`, `resnet50`, or `simple_cnn` |
| `--kfold` | 0 | Stratified k-fold CV (e.g. `--kfold 5`); produces per-fold artifacts |
| `--disable-augmentation` | off | Disable training augmentation (use plain resize only) |
| `--disable-class-weighting` | off | Disable inverse-frequency class weighting |
| `--external-test-root` | none | External dataset root for generalization testing |
| `--audit-metadata-csv` | none | CSV with per-image subgroup metadata for fairness audits |
| `--heartbeat-seconds` | 30 | Seconds between in-epoch progress logs |

### Architecture comparison example

```bash
# Transfer learning (default)
python -m backend.training.train --epochs-head 5 --epochs-finetune 5 --warmup-epochs 1 --label-smoothing 0.1

# Simple CNN baseline for comparison
python -m backend.training.train --arch simple_cnn --epochs-head 10 --epochs-finetune 0
```

### K-fold cross-validation example

```bash
python -m backend.training.train --kfold 5 --epochs-head 3 --epochs-finetune 2
```

Outputs: `backend/artifacts/kfold/fold_*/best_model.pt` and `backend/artifacts/kfold/kfold_summary.txt`

## Inference Features

- **Temperature-scaled probabilities**: Checkpoint stores an optimal temperature parameter; inference applies temperature scaling for calibrated probability outputs.
- **Test-time augmentation (TTA)**: Pass `?tta=true` to `/predict` to average predictions across multiple augmented views of the input image.
- **Clinician feedback**: `POST /history/{id}/feedback` with `{"feedback": "correct"}` or `{"feedback": "incorrect"}` to flag predictions. Displayed in the Review History UI.
- **Prediction drift monitoring**: `GET /history/drift` computes Population Stability Index (PSI) between baseline and recent prediction confidence distributions, alerting when distribution shift exceeds the 0.2 threshold.

## API Endpoints (v1)

- `GET /health` - service health check
- `GET /model-info` - returns model/checkpoint metadata, runtime mode, and temperature scaling factor
- `POST /predict` - predicts class probabilities from an uploaded CXR image
  - optional query: `threshold` (0.0 to 1.0) for decision-threshold override
  - optional query: `tta` (true/false) for test-time augmentation
- `POST /gradcam` - returns Grad-CAM overlay image (base64 PNG) for uploaded CXR
  - includes heuristic explainability safety fields: `lung_focus_score`, `off_lung_attention_ratio`, `explainability_warning`
- `GET /history` - recent analysis records (with clinician feedback status)
- `GET /history/summary` - aggregate counts and average confidence
- `GET /history/drift` - PSI-based prediction drift report
- `POST /history/{id}/feedback` - submit clinician feedback on a prediction