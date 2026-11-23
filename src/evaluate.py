"""Evaluate a saved ViT checkpoint across the complete validation split.

This companion utility measures the classification readout (§8) and reports
diagnostics that help investigate VI.4 §15. It cannot by itself establish
whether an error began in resize, patch scale, position, interaction, or readout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import ImageFolderDataset
    from .logging_utils import log
    from .metrics import MetricAccumulator
    from .vit import VisionTransformer
except ImportError:
    from dataset import ImageFolderDataset
    from logging_utils import log
    from metrics import MetricAccumulator
    from vit import VisionTransformer


def parse_args() -> argparse.Namespace:
    """Define which saved ViT and held-out image split to measure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/vit_cats_dogs.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/cats_dogs_resized_64"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(choice: str) -> torch.device:
    """Choose where checkpoint inference runs; this does not alter model behavior."""
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot access CUDA")
    return torch.device(
        "cuda" if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()) else "cpu"
    )


def evaluate_checkpoint(checkpoint_path: Path, data_root: Path, batch_size: int, device: torch.device) -> dict:
    """Run the checkpoint over every validation image and collect failure details.

    The saved config restores the same patch grid and position-table shape used
    at training. Misclassified examples make §15's input-to-readout diagnosis
    concrete, while their scores alone do not identify the root cause.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    config = checkpoint["config"]
    dataset = ImageFolderDataset(data_root / "val", config["image_size"], classes=classes)
    if not dataset.samples:
        raise ValueError(f"No validation images found in {data_root / 'val'}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = VisionTransformer(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = MetricAccumulator(classes, device=device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    misclassified = []
    sample_offset = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            labels_on_device = labels.to(device)
            probabilities = logits.softmax(dim=1)
            predictions = logits.argmax(dim=1)
            metrics.update(logits, labels_on_device, criterion(logits, labels_on_device).item())
            for row, (truth, prediction) in enumerate(
                zip(labels.tolist(), predictions.cpu().tolist())
            ):
                if truth == prediction:
                    continue
                path = dataset.samples[sample_offset + row][0]
                misclassified.append(
                    {
                        "image": path.relative_to(data_root).as_posix(),
                        "actual": classes[truth],
                        "predicted": classes[prediction],
                        "predicted_confidence": float(probabilities[row, prediction].item()),
                        "probabilities": {
                            name: float(probabilities[row, index].item())
                            for index, name in enumerate(classes)
                        },
                    }
                )
            sample_offset += labels.size(0)

    return {
        "checkpoint": str(checkpoint_path),
        "dataset": str(data_root),
        "split": "val",
        "image_size": config["image_size"],
        "training": {
            "optimizer": checkpoint.get("optimizer"),
            "learning_rate": checkpoint.get("learning_rate"),
        },
        **metrics.summary(),
        "misclassified_images": misclassified,
    }


def log_report(report: dict) -> None:
    """Print aggregate and per-example diagnostics for human inspection."""
    training = report["training"]
    log(f"training_optimizer={training['optimizer']} learning_rate={training['learning_rate']}")
    log(
        f"samples={report['num_samples']} image_size={report['image_size']} "
        f"loss={report['loss']:.4f} accuracy={report['accuracy']:.1%} "
        f"correct={report['correct']} incorrect={report['incorrect']}"
    )
    log(
        f"macro_precision={report['macro_precision']:.1%} "
        f"macro_recall={report['macro_recall']:.1%} macro_f1={report['macro_f1']:.1%}"
    )
    for name, values in report["per_class"].items():
        log(
            f"class={name} support={values['support']} correct={values['correct']} "
            f"precision={values['precision']:.1%} recall={values['recall']:.1%} "
            f"f1={values['f1']:.1%}"
        )
    matrix = report["confusion_matrix"]
    log(
        f"confusion_matrix rows=actual columns=predicted "
        f"classes={matrix['class_order']} values={matrix['values']}"
    )
    if report["misclassified_images"]:
        log(f"misclassified_images={len(report['misclassified_images'])}")
        for item in report["misclassified_images"]:
            log(
                f"misclassified image={item['image']} actual={item['actual']} "
                f"predicted={item['predicted']} confidence={item['predicted_confidence']:.1%} "
                f"probabilities={item['probabilities']}"
            )
    else:
        log("misclassified_images=0")


def main() -> None:
    """Run full validation and save its diagnostics for later comparison."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = choose_device(args.device)
    if device.type == "cuda":
        log(f"evaluation_device={device} gpu={torch.cuda.get_device_name(device)}")
    else:
        log(f"evaluation_device={device}")
    report = evaluate_checkpoint(args.checkpoint, args.data.resolve(), args.batch_size, device)
    report["evaluation_device"] = str(device)
    log_report(report)

    report_path = Path("models") / f"{args.checkpoint.stem}_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    log(f"saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
