"""Train the small Vision Transformer.

This file connects the model to the supervised learning task:

    image files -> tensors and labels -> ViT logits -> loss -> gradients
    -> parameter updates -> better future predictions

The training loop is intentionally ordinary. That makes it easier to see
which parts belong to the ViT architecture and which parts belong to the
training objective. The model only defines how an image becomes logits; this
file defines how those logits are compared with labels and learned from.

Example from the repository root:

    python -m src.train --data data/cats_dogs --epochs 5
    python .\\src\\train.py --data .\\data\\cats_dogs --epochs 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    # Module form: python -m src.train
    from .dataset import ImageFolderDataset
    from .vit import VisionTransformer
except ImportError:
    # File form: python .\src\train.py
    # When a file is executed directly, Python does not consider it part of
    # the src package, so the package-relative imports above are unavailable.
    from dataset import ImageFolderDataset
    from vit import VisionTransformer


def parse_args() -> argparse.Namespace:
    """Define the small set of settings exposed on the command line."""

    parser = argparse.ArgumentParser(description="Train a small Vision Transformer.")

    # Dataset and checkpoint paths are relative to the directory where the
    # command is run, normally the repository root.
    parser.add_argument("--data", type=Path, default=Path("data/cats_dogs"))
    parser.add_argument("--output", type=Path, default=Path("models/vit_cats_dogs.pt"))

    # Image and patch settings determine the visual token sequence. For
    # example, 64x64 images with 8x8 patches produce 64 patch tokens.
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)

    # These are the small model settings. Larger values usually increase
    # capacity and compute, but this project values inspectability over speed
    # or state-of-the-art accuracy.
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)

    # Training settings control how many examples and update steps are used.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Make the small experiment more repeatable."""

    # The dataset split and DataLoader shuffling use randomness. Fixing the
    # Python and PyTorch generators makes demonstrations easier to reproduce.
    random.seed(seed)
    torch.manual_seed(seed)


def make_datasets(data_root: Path, image_size: int, val_ratio: float, seed: int):
    """Build training and validation datasets from one of two layouts.

    Supported layouts:

    1. Explicit split, which is useful when prepare_data.py was used:

           data_root/train/cat/*.jpg
           data_root/val/cat/*.jpg

    2. One folder per class, where this function makes a deterministic split:

           data_root/cat/*.jpg
           data_root/dog/*.jpg

    Keeping the split logic here makes the dataset class itself easier to
    understand: it only needs to load one labeled example at a time.
    """

    train_root = data_root / "train"
    val_root = data_root / "val"

    if train_root.is_dir():
        # The helper normally creates this explicit layout.
        train_data = ImageFolderDataset(train_root, image_size)
        classes = train_data.classes

        if val_root.is_dir():
            # Pass the training class order to validation so label 0 means the
            # same class in both datasets.
            val_data = ImageFolderDataset(val_root, image_size, classes=classes)
        else:
            # A train folder without a val folder is still valid; split it
            # below using the same code path as the flat layout.
            val_data = None
    else:
        # Treat data_root itself as an ImageFolder-style dataset.
        train_data = ImageFolderDataset(data_root, image_size)
        classes = train_data.classes
        val_data = None

    # If the caller supplied data/train and data/val, preserve that split.
    if val_data is not None:
        return train_data, val_data, classes

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val-ratio must be between 0 and 1 when no val directory is supplied")

    # random_split returns two dataset views that refer to the same underlying
    # examples. The generator makes the choice reproducible.
    val_count = max(1, int(len(train_data) * val_ratio))
    train_count = len(train_data) - val_count
    if train_count < 1:
        raise ValueError("The dataset needs at least two images for a train/validation split")
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(train_data, [train_count, val_count], generator=generator)
    return train_data, val_data, classes


def run_epoch(
    model: VisionTransformer,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one training or evaluation pass and return loss and accuracy.

    Passing an optimizer means training mode. Passing ``None`` means
    validation mode. Keeping one function for both paths makes the difference
    visible: training computes gradients and updates weights; validation only
    measures the current model.
    """

    # ``optimizer is not None`` is our simple mode flag.
    training = optimizer is not None
    model.train(training)

    # We accumulate totals over all examples, rather than averaging batch
    # averages. That remains correct when the final batch is smaller.
    total_loss = 0.0
    correct = 0
    total = 0

    # Gradient tracking is the expensive part that evaluation does not need.
    # ``torch.no_grad`` also prevents validation tensors from being retained
    # for backward computation.
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            # Move both the input pixels and target integers to CPU or GPU.
            images, labels = images.to(device), labels.to(device)

            # The ViT returns logits with shape (batch_size, num_classes).
            logits = model(images)

            # CrossEntropyLoss combines a class-score normalization step with
            # negative log-likelihood. It expects raw logits, so train.py does
            # not apply softmax before calling it.
            loss = loss_function(logits, labels)

            if training:
                # One standard gradient-descent update:
                #   1. remove gradients left by the previous batch;
                #   2. compute d(loss)/d(parameter) by backpropagation;
                #   3. let Adam change each learned parameter.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Convert the batch mean loss back to a sum so examples contribute
            # equally even if batches have different sizes.
            total_loss += loss.item() * labels.size(0)

            # The largest logit is the predicted class index. This is only an
            # evaluation statistic; gradients use the full loss above.
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    """Parse settings, train the ViT, and save its best validation checkpoint."""

    args = parse_args()
    set_seed(args.seed)

    # Patch extraction assumes an exact grid. This check catches a common
    # configuration mistake before the first batch reaches the model.
    if args.image_size % args.patch_size != 0:
        raise ValueError("image-size must be divisible by patch-size")

    train_data, val_data, classes = make_datasets(
        args.data,
        args.image_size,
        args.val_ratio,
        args.seed,
    )

    # DataLoader turns individual (image, label) examples into batches. The
    # model receives images shaped (B, 3, image_size, image_size).
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Prefer a CUDA GPU when PyTorch can see one; CPU is enough for the tiny
    # defaults and keeps this repository easy to run.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=len(classes),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
    ).to(device)

    # The task is supervised classification: each image has one integer class
    # label. Adam is chosen as a familiar, simple optimizer for this example.
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"classes: {classes}")
    print(f"images: {len(train_data)} train, {len(val_data)} validation")
    print(f"device: {device}")
    best_val_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        # Training changes parameters.
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )

        # Validation measures generalization on examples not used for the
        # current weight update. Passing None disables optimizer operations.
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            loss_function,
            device,
            None,
        )

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.2%}"
        )

        # Save the best validation result rather than only the final epoch.
        # The checkpoint includes class names and model configuration because
        # inference needs them to interpret logits and rebuild the same ViT.
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "classes": classes,
                    "config": model.config,
                    "model": model.state_dict(),
                },
                args.output,
            )

    print(f"saved best checkpoint to {args.output}")


if __name__ == "__main__":
    # This guard lets the file be run directly while also allowing its
    # functions to be imported by another script or notebook.
    main()
