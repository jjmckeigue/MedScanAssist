from pathlib import Path


def main() -> None:
    checkpoint_path = Path("./backend/checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError("Run training first to create a checkpoint.")

    # Placeholder for offline CAM generation (Eigen-CAM is used at serving time).
    print("CAM script scaffold is ready. See gradcam_service.py for the Eigen-CAM implementation.")


if __name__ == "__main__":
    main()
