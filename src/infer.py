"""Run inference with a checkpoint produced by train.py.

Training asks the model to improve its parameters. Inference uses those fixed
parameters to answer one new question:

    image file -> preprocessing -> ViT logits -> probabilities -> class name

This file intentionally does not contain a second model definition. It reads
the architecture configuration saved by train.py, rebuilds the same model, and
loads the learned weights.
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
    """Define the checkpoint and one image to classify."""

    parser = argparse.ArgumentParser(description="Classify one image with a trained ViT.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/vit_cats_dogs.pt"))
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Load a trained model and print its class probabilities."""

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

    # Use the exact image size saved with the model. Inference must perform the
    # same resize operation as training or the patch grid will not match the
    # learned positional embedding table.
    image_size = checkpoint["config"]["image_size"]
    image = image_to_tensor(args.image, image_size).unsqueeze(0).to(device)

    # A single image has shape (1, 3, H, W). The batch dimension is still
    # present because the ViT is written to process batches uniformly.
    #
    # The model returns logits, not probabilities. softmax converts the logits
    # to positive values summing to one so they are easier to read as scores.
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
