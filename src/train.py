"""Train the small ViT using a prepared or ordinary image-folder dataset.

Chapter VI.4 §14 separates architecture from learned representation, objective,
and recipe. This script supplies one supervised classification objective and a
small training recipe for the model in ``vit.py``; it is not part of the ViT
architecture itself. The printed split metrics help investigate §15 failure
modes without treating a score as proof of how the model reasoned.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import ImageFolderDataset
    from .logging_utils import log, log_model_info
    from .metrics import MetricAccumulator
    from .vit import VisionTransformer
except ImportError:
    from dataset import ImageFolderDataset
    from logging_utils import log, log_model_info
    from metrics import MetricAccumulator
    from vit import VisionTransformer


def parse_args() -> argparse.Namespace:
    """Define the VI.4 companion's input-grid, patch, model, and recipe choices.

    ``image-size`` and ``patch-size`` determine the patch count from §3 and
    attention cost from §§9-11; optimizer and epoch settings belong to §14's
    training recipe rather than the model definition.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cats_dogs_resized_64"))
    parser.add_argument("--output", type=Path, default=Path("models/vit_cats_dogs.pt"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw"), default="adamw",
        help="parameter update rule (default: adamw)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="auto selects CUDA when available; cuda requires CUDA",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Make data splitting and parameter initialization repeatable where possible.

    Reproducibility supports comparison of the training recipes discussed in
    VI.4 §14; it does not change the representation stages in the ViT.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(choice: str) -> torch.device:
    """Choose CPU or CUDA for the companion implementation's tensor operations."""
    if choice == "cuda" and not torch.cuda.is_available():
        cuda_build = torch.version.cuda or "not installed (CPU-only PyTorch)"
        raise RuntimeError(
            "--device cuda was requested, but PyTorch cannot access CUDA. "
            f"torch={torch.__version__}, torch CUDA build={cuda_build}."
        )
    use_cuda = choice == "cuda" or (choice == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def make_datasets(data_root: Path, image_size: int, val_ratio: float, seed: int):
    """Load labeled examples for the supervised objective described in §14.

    Explicit train/validation folders preserve the held-out evaluation boundary
    used to diagnose model behavior (§15). The flat-folder split is a compact
    convenience for the companion code, not a ViT-specific operation (§17).
    """
    train_root, val_root = data_root / "train", data_root / "val"
    if train_root.is_dir():
        train_data = ImageFolderDataset(train_root, image_size)
        if val_root.is_dir():
            val_data = ImageFolderDataset(val_root, image_size, classes=train_data.classes)
            return train_data, val_data, train_data.classes
        source_data = train_data
    else:
        source_data = ImageFolderDataset(data_root, image_size)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1 when no val/ folder exists")
    val_count = max(1, int(len(source_data) * val_ratio))
    train_count = len(source_data) - val_count
    if train_count < 1:
        raise ValueError("The dataset needs at least two images for a train/validation split")
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(source_data, [train_count, val_count], generator=generator)
    return train_data, val_data, source_data.classes


def run_epoch(model, loader, criterion, device, classes, optimizer=None) -> dict:
    """Run a forward pass, and optionally a loss-gradient-update training step.

    The model's visual representation path is in VI.4 §§2-8. Cross-entropy,
    backpropagation, and the selected optimizer are training choices (§14).
    Passing no optimizer makes this a read-only evaluation pass.
    """
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator(classes, device=device)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch_loss_sum = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                (batch_loss_sum / labels.size(0)).backward()
                optimizer.step()
            metrics.update(logits.detach(), labels, batch_loss_sum.detach().item())
    return metrics.summary()


def log_metrics(split: str, metrics: dict, classes: list[str]) -> None:
    """Print classification diagnostics for the selected data split.

    Accuracy, per-class recall, and the confusion matrix help localize the
    failures in VI.4 §15; these aggregate scores do not explain model causes.
    """
    log(
        f"{split}_loss={metrics['loss']:.4f} {split}_accuracy={metrics['accuracy']:.1%} "
        f"{split}_macro_precision={metrics['macro_precision']:.1%} "
        f"{split}_macro_recall={metrics['macro_recall']:.1%} "
        f"{split}_macro_f1={metrics['macro_f1']:.1%}"
    )
    for name, values in metrics["per_class"].items():
        log(
            f"{split}_class={name} support={values['support']} "
            f"precision={values['precision']:.1%} recall={values['recall']:.1%} "
            f"f1={values['f1']:.1%}"
        )
    matrix = metrics["confusion_matrix"]
    log(
        f"{split}_confusion_matrix rows=actual columns=predicted "
        f"classes={classes} values={matrix['values']}"
    )


def main() -> None:
    """Connect data, the ViT, a supervised objective, and checkpoint selection.

    This is the runnable companion pipeline (§17), spanning the stages from
    image-grid preprocessing through the §8 classification readout. The best
    validation checkpoint is selected by a task metric, not by architecture.
    """
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if args.image_size % args.patch_size != 0:
        raise ValueError("image-size must be divisible by patch-size")
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_data, val_data, classes = make_datasets(
        args.data, args.image_size, args.val_ratio, args.seed
    )
    if not len(train_data) or not len(val_data):
        raise ValueError("Training and validation splits must both contain images")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=len(classes),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
    ).to(device)
    optimizer_types = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    optimizer_type = optimizer_types[args.optimizer]
    optimizer = optimizer_type(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    log(
        f"optimizer={optimizer_type.__name__} learning_rate={args.learning_rate:g} "
        f"epochs={args.epochs} batch_size={args.batch_size} image_size={args.image_size}"
    )
    log(f"classes={classes} train={len(train_data)} val={len(val_data)} data={args.data}")
    if device.type == "cuda":
        log(f"device={device} gpu={torch.cuda.get_device_name(device)}")
    else:
        log(f"device={device}")
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        log(f"epoch {epoch:02d}/{args.epochs} starting")
        train_metrics = run_epoch(model, train_loader, criterion, device, classes, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, classes)
        log(f"epoch {epoch:02d}/{args.epochs} complete")
        log_metrics("train", train_metrics, classes)
        log_metrics("val", val_metrics, classes)

        if val_metrics["accuracy"] >= best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "classes": classes,
                    "config": model.config,
                    "model": model.state_dict(),
                    "optimizer": args.optimizer,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                },
                args.output,
            )

    log(f"training complete; best validation accuracy={best_accuracy:.1%}")
    log_model_info(model, classes, args.image_size, device, str(args.output))
    log(f"saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
