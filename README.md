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
- Frontend: React + Vite
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
2. Place dataset in `data/raw/chest_xray/` (see Dataset section).
3. Run backend:
   - Local: `pip install -r backend/requirements.txt` then
     `uvicorn backend.app.main:app --reload --port 8000`
   - Docker: `docker compose up --build backend`
4. Visit API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

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

## API Endpoints (v1)

- `GET /health` - service health check
- `POST /predict` - predicts class probabilities from an uploaded CXR image
- `POST /gradcam` - returns Grad-CAM overlay image (base64 PNG) for uploaded CXR

Current model-serving logic is scaffolded and defaults to deterministic placeholder
scores until trained weights are present.

## Next Steps

1. Implement final training loop in `backend/training/train.py`
2. Export best checkpoint to `backend/checkpoints/best_model.pt`
3. Replace placeholder inference in model service
4. Connect React upload UI to `/predict` and `/gradcam`