from pathlib import Path


def main() -> None:
    checkpoint_path = Path("./backend/checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError("Run training first to create a checkpoint.")

    # Placeholder script for future offline Grad-CAM generation.
    print("Grad-CAM script scaffold is ready. Implement target-layer hooks next.")


if __name__ == "__main__":
    main()
