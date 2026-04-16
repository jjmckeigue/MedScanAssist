"""Export a PyTorch checkpoint to ONNX with dual outputs (logits + last-conv activations).

Usage (from repo root, with PyTorch installed locally):
    python -m backend.scripts.export_onnx
    python -m backend.scripts.export_onnx --checkpoint backend/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class DenseNetDualOutput(nn.Module):
    """Wraps a DenseNet to return both logits and last-conv activations."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.features = base.features
        self.classifier = base.classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.features(x)
        activations = F.relu(raw)
        pooled = F.adaptive_avg_pool2d(activations, (1, 1)).flatten(1)
        logits = self.classifier(pooled)
        return logits, activations


class ResNetDualOutput(nn.Module):
    """Wraps a ResNet to return both logits and last-conv activations."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        activations = self.layer4(x)
        pooled = self.avgpool(activations).flatten(1)
        logits = self.fc(pooled)
        return logits, activations


def _build_base_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def export(checkpoint_path: str, output_dir: str | None = None) -> None:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(output_dir) if output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    arch = str(checkpoint.get("arch", "densenet121")).lower()
    class_names = checkpoint.get("class_names") or checkpoint.get("classes") or ["normal", "pneumonia"]
    image_size = int(checkpoint.get("image_size", 224))
    temperature = float(checkpoint.get("temperature", 1.0))

    base_model = _build_base_model(arch, len(class_names))
    base_model.load_state_dict(checkpoint["state_dict"])
    base_model.eval()

    if arch == "resnet50":
        wrapper = ResNetDualOutput(base_model)
    else:
        wrapper = DenseNetDualOutput(base_model)
    wrapper.eval()

    dummy = torch.randn(1, 3, image_size, image_size)

    onnx_path = out_dir / "best_model.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits", "activations"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "activations": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved to {onnx_path}")

    meta = {
        "arch": arch,
        "class_names": [str(c) for c in class_names],
        "image_size": image_size,
        "temperature": temperature,
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_acc": float(checkpoint["best_val_acc"]) if checkpoint.get("best_val_acc") is not None else None,
        "best_val_loss": float(checkpoint["best_val_loss"]) if checkpoint.get("best_val_loss") is not None else None,
    }
    meta_path = out_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="backend/checkpoints/best_model.pt",
        help="Path to the PyTorch checkpoint file",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to checkpoint dir)")
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
