from pathlib import Path

from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, augment: bool = True) -> dict[str, transforms.Compose]:
    """Build train/val/test transforms.

    Training augmentations deliberately vary position, orientation, scale, and
    intensity so the model cannot memorise alignment or brightness cues.
    All paths normalise to ImageNet statistics for transfer-learning compatibility.
    """
    if augment:
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=8),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def build_tta_transforms(image_size: int = 224, n_augments: int = 5) -> list[transforms.Compose]:
    """Return a list of slightly different eval-time transforms for test-time augmentation."""
    base = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    augmented = [
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    ]
    return [base] + augmented[: max(0, n_augments - 1)]


def build_imagefolder_dataset(root: Path, split: str, transform: transforms.Compose):
    split_path = root / split
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")
    return datasets.ImageFolder(str(split_path), transform=transform)
