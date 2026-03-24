from pathlib import Path

from torchvision import datasets, transforms


def build_transforms(image_size: int = 224) -> dict[str, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def build_imagefolder_dataset(root: Path, split: str, transform: transforms.Compose):
    split_path = root / split
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")
    return datasets.ImageFolder(str(split_path), transform=transform)
