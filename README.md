# MedScanAssist

MedScanAssist is an end-to-end machine learning system for binary chest X-ray
classification (`pneumonia` vs `normal`). It covers the full ML lifecycle:
data exploration, model development and comparison, evaluation, deployment via
a FastAPI inference service, monitoring for prediction drift, and responsible
AI through CAM-based visual explainability (Eigen-CAM) with heuristic safety
checks.

This repository demonstrates:
- **Data preparation & EDA**: automated exploratory analysis, class-distribution
  reporting, image-quality statistics, and documented dataset bias/limitations
- **Model development & comparison**: transfer learning (DenseNet121, ResNet50)
  vs. a from-scratch SimpleCNN baseline, with side-by-side metric comparison
- **Evaluation & validation**: confusion matrix, ROC, PR, calibration, threshold
  tuning, sensitivity/specificity, shortcut stress testing, and optional
  subgroup fairness audits
- **MLOps & deployment**: FastAPI serving, Docker packaging, health checks, API
  key auth, checkpoint versioning, temperature-scaled calibration, TTA, and
  PSI-based prediction drift monitoring
- **Responsible AI**: Eigen-CAM explainability with lung-focus/off-lung attention
  warnings, clinician feedback loop, and governance-ready audit artifacts

## Stack

- Backend: Python 3.11, FastAPI, Uvicorn, PyTorch, torchvision, scikit-learn
- Explainability: Eigen-CAM (gradient-free, ONNX-compatible class activation mapping)
- Frontend: React + Vite (Node 18+ recommended; `frontend/.nvmrc` included)
- Packaging: Docker + Docker Compose

## Project Structure

```text
medscanassist/
  backend/
    app/
      main.py              # FastAPI application entry point
      config.py            # Pydantic settings (env-driven)
      schemas.py           # Request/response models
      routes/              # API route handlers
      services/            # Model inference, Eigen-CAM, history, patients
    training/
      train.py             # Training loop (transfer learning + SimpleCNN)
      evaluate.py          # Standalone test-set evaluation
      eda.py               # Exploratory data analysis script
      compare_models.py    # Multi-architecture comparison utility
      data_utils.py        # Transforms, augmentation, dataset loading
      gradcam.py           # Offline CAM generation scaffold
    checkpoints/           # Saved model weights (.pt, .onnx)
    artifacts/             # Training metrics, plots, EDA, comparison tables
    requirements.txt
    Dockerfile
  frontend/
    src/
    package.json
  data/
    raw/                   # Dataset files (git-ignored)
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

## Reproducibility Guide

Step-by-step commands to reproduce every artifact in this repository from a
clean clone. All commands assume the repo root as the working directory.

### 1. Environment setup

```bash
# Clone and enter repo
git clone https://github.com/<your-org>/MedScanAssist.git
cd MedScanAssist

# Create virtual environment
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# PowerShell:  .\.venv\Scripts\Activate.ps1

# Copy environment config
cp .env.example .env          # macOS/Linux
# Copy-Item .env.example .env  # PowerShell

# Install dependencies
python -m pip install -r backend/requirements-train.txt
```

### 2. Dataset

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset from Kaggle and place it at `data/raw/chest_xray/` with the standard
`train/`, `val/`, `test/` splits (see Dataset Ingestion section below).

### 3. Exploratory data analysis

```bash
python -m backend.training.eda
```

Outputs: `backend/artifacts/eda/` (class distribution, image stats, sample grid, EDA summary).

### 4. Train the baseline CNN (SimpleCNN from scratch)

```bash
python -m backend.training.train --arch simple_cnn --epochs-head 10 --epochs-finetune 0
# Save checkpoint with architecture-specific name for comparison:
# copy backend\checkpoints\best_model.pt backend\checkpoints\best_model_simple_cnn.pt  (PowerShell)
# cp backend/checkpoints/best_model.pt backend/checkpoints/best_model_simple_cnn.pt     (bash)
```

### 5. Train the transfer-learning model (DenseNet121)

```bash
python -m backend.training.train --arch densenet121 --epochs-head 5 --epochs-finetune 5 --warmup-epochs 1 --label-smoothing 0.1
# Save checkpoint:
# copy backend\checkpoints\best_model.pt backend\checkpoints\best_model_densenet121.pt  (PowerShell)
# cp backend/checkpoints/best_model.pt backend/checkpoints/best_model_densenet121.pt     (bash)
```

### 6. (Optional) Train ResNet50

```bash
python -m backend.training.train --arch resnet50 --epochs-head 5 --epochs-finetune 5 --warmup-epochs 1 --label-smoothing 0.1
# copy backend\checkpoints\best_model.pt backend\checkpoints\best_model_resnet50.pt
```

### 7. Model comparison

```bash
python -m backend.training.compare_models --include-default
```

Outputs: `backend/artifacts/model_comparison.csv`, `model_comparison.txt`, `model_comparison.png`.

### 8. Run the inference API

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
# Visit http://localhost:8000/docs for Swagger UI
```

### 9. Run tests

```bash
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -q
```

### 10. Docker deployment

```bash
docker compose up --build backend
```

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

## Exploratory Data Analysis

```bash
python -m backend.training.eda
```

Produces:
- `backend/artifacts/eda/class_distribution.csv` and `.png` — per-split class counts
- `backend/artifacts/eda/image_stats.csv` and `.txt` — width/height/aspect statistics
- `backend/artifacts/eda/image_dimensions.png` — dimension histograms
- `backend/artifacts/eda/sample_grid.png` — random sample images per class per split
- `backend/artifacts/eda/eda_summary.txt` — textual summary including dataset bias and limitations

## Model Comparison

After training multiple architectures, generate a side-by-side comparison:

```bash
python -m backend.training.compare_models --include-default
```

Produces:
- `backend/artifacts/model_comparison.csv` — metric table (accuracy, precision, recall, F1, ROC-AUC, params)
- `backend/artifacts/model_comparison.txt` — formatted report with trade-off analysis
- `backend/artifacts/model_comparison.png` — grouped bar chart

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

## API Endpoints (v2)

### Core Inference

- `GET /health` - service health check
- `GET /model-info` - returns model/checkpoint metadata, runtime mode, and temperature scaling factor
- `POST /predict` - predicts class probabilities from an uploaded CXR image
  - optional query: `threshold` (0.0 to 1.0) for decision-threshold override
  - optional query: `tta` (true/false) for test-time augmentation
- `POST /gradcam` - returns Eigen-CAM overlay image (base64 PNG) for uploaded CXR
  - uses gradient-free SVD-based class activation mapping (not gradient-dependent Grad-CAM)
  - includes heuristic explainability safety fields: `lung_focus_score`, `off_lung_attention_ratio`, `explainability_warning`
- `POST /analyze` - combined prediction + Eigen-CAM in a single request
  - optional query: `patient_id` to link analysis to a patient profile
  - uploaded X-ray images are persisted to `UPLOAD_DIR` for later review

### Analysis History

- `GET /history` - recent analysis records (with clinician feedback status)
- `GET /history/summary` - aggregate counts and average confidence
- `GET /history/drift` - PSI-based prediction drift report
- `POST /history/{id}/feedback` - submit clinician feedback on a prediction

### Patient Profiles

- `POST /patients` - create a patient profile (name, DOB, MRN, notes)
- `GET /patients` - list/search patients (query: `search`, `limit`, `offset`)
- `GET /patients/{id}` - get patient detail with linked analysis history
- `PUT /patients/{id}` - update patient profile
- `DELETE /patients/{id}` - delete patient (analyses are unlinked, not deleted)
- `GET /patients/{id}/progression` - confidence progression data over time

### Image Serving

- `GET /images/{filename}` - serve a stored X-ray image by filename

### Note on explainability method naming

The `/gradcam` endpoint path is retained for backward compatibility, but the
underlying technique is **Eigen-CAM** — a gradient-free class activation mapping
method that uses SVD on the last convolutional layer's activation tensor. Unlike
Grad-CAM (which requires backpropagation through the model), Eigen-CAM is
compatible with ONNX-exported models and does not require gradient computation.
All user-facing documentation refers to this technique honestly as "Eigen-CAM"
or "CAM-based explainability."