"""Run inference with a checkpoint produced by train.py.

Training asks the model to improve its parameters. Inference uses those fixed
parameters to answer one new question:

    image file -> preprocessing -> ViT logits -> probabilities -> class name

This file intentionally does not contain a second model definition. It reads
the architecture configuration saved by train.py, rebuilds the same model, and
loads the learned weights.

The pipeline mirrors VI.4's distinction between contextualized visual states
and task readout (§8): the ViT returns logits, then this file converts them to
class probabilities for display.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    # Module form: python -m src.infer
    from .dataset import image_to_tensor
    from .vit import VisionTransformer
except ImportError:
    # File form: python .\src\infer.py
    from dataset import image_to_tensor
    from vit import VisionTransformer


def parse_args() -> argparse.Namespace:
    """Define the saved model and one image supplied for its readout."""

    parser = argparse.ArgumentParser(description="Classify one image with a trained ViT.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/vit_cats_dogs.pt"))
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Restore the trained patch grid, run one image, and display its readout."""

    args = parse_args()

    # Use the same device choice as training. map_location lets a checkpoint
    # trained on GPU be loaded on a CPU-only machine.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # The checkpoint stores configuration separately from learned weights.
    # Configuration describes tensor sizes; state_dict contains the parameter
    # values learned during training.
    model = VisionTransformer(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Use the exact model input size saved at training. This resize occurs
    # before patchification and may discard fine evidence (VI.4 §§11, 13, 15).
    # The patch grid and learned position table must then match the checkpoint.
    image_size = checkpoint["config"]["image_size"]
    image = image_to_tensor(args.image, image_size).unsqueeze(0).to(device)

    # A single image has shape (1, 3, H, W). The batch dimension is still
    # present because the ViT is written to process batches uniformly.
    #
    # The model's class-token readout returns logits (§8). Softmax converts
    # them to positive values summing to one for display; it is not a ViT layer.
    with torch.no_grad():
        probabilities = model(image).softmax(dim=1)[0]

    # argmax returns the integer position of the highest probability. The
    # saved class list converts that model index back to a folder name.
    prediction = int(probabilities.argmax())
    class_name = checkpoint["classes"][prediction]
    print(f"prediction: {class_name}")
    print(f"confidence: {probabilities[prediction].item():.2%}")

    # Print every class, not only the winner. This helps a learner see that the
    # classifier produces a distribution over alternatives before argmax picks
    # the displayed answer.
    for index in probabilities.argsort(descending=True):
        class_index = int(index)
        print(f"  {checkpoint['classes'][class_index]:>12}: {probabilities[class_index].item():.2%}")


if __name__ == "__main__":
    main()
