"""Copy a small, validated subset from a raw Cat/Dog folder dataset.

The downloaded archive is deliberately not used directly by the training
script. This helper makes the data contract explicit and keeps the learning
experiment small:

    raw/PetImages/Cat/*.jpg     raw/PetImages/Dog/*.jpg
                    |
                    v
    data/cats_dogs/train/cat    data/cats_dogs/train/dog
    data/cats_dogs/val/cat      data/cats_dogs/val/dog

The output class names are lower-cased so the example produces the simple
``cat`` and ``dog`` labels used in the README. The same approach works for any
raw dataset whose immediate subdirectories represent classes.

Example:

    python -m src.prepare_data --source raw/PetImages --output data/cats_dogs --per-class 100
    python .\\src\\prepare_data.py --source .\\data\\raw\\PetImages --output .\\data\\cats_dogs --per-class 100
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    """Define how many readable files to select for each class."""

    parser = argparse.ArgumentParser(description="Make a small train/val image-folder dataset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/cats_dogs"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def valid_images(directory: Path) -> list[Path]:
    """Return supported image paths that Pillow can actually inspect.

    Large public archives sometimes contain a small number of damaged or
    incomplete files. Filtering them before copying prevents a training run
    from failing later when a batch happens to request one of those files.
    """

    paths = []
    for path in sorted(directory.iterdir()):
        # Only ordinary image files are candidates. The dataset class applies
        # a similar filter when it loads the finished output.
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        try:
            with Image.open(path) as image:
                # verify checks the file structure without keeping the image
                # open or decoding all pixels into memory.
                image.verify()
            paths.append(path)
        except (OSError, ValueError):
            # Ignore unreadable files and continue selecting from the class.
            pass
    return paths


def main() -> None:
    """Select, split, and copy a small dataset."""

    args = parse_args()
    if args.val_count >= args.per_class:
        raise ValueError("val-count must be smaller than per-class")

    # A fixed local random generator makes the chosen subset repeatable without
    # changing random state used by some other part of a larger program.
    randomizer = random.Random(args.seed)

    # Each immediate child directory is treated as one class. For the source
    # archive these are normally PetImages/Cat and PetImages/Dog.
    class_dirs = [path for path in sorted(args.source.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {args.source}")

    for class_dir in class_dirs:
        candidates = valid_images(class_dir)

        # Shuffle before slicing so taking the first N files does not depend on
        # archive filename ordering. The fixed seed keeps this reproducible.
        randomizer.shuffle(candidates)
        selected = candidates[: args.per_class]
        if len(selected) < args.per_class:
            print(f"warning: {class_dir.name} has only {len(selected)} readable images")

        # The first val_count files become validation examples. The remaining
        # selected files become training examples. They are copied rather than
        # moved so the raw archive remains available for another experiment.
        for split, paths in (
            ("val", selected[: args.val_count]),
            ("train", selected[args.val_count :]),
        ):
            destination = args.output / split / class_dir.name.lower()
            destination.mkdir(parents=True, exist_ok=True)

            for index, source_path in enumerate(paths):
                # A simple generated filename avoids collisions between source
                # folders and preserves the original extension for Pillow.
                shutil.copy2(
                    source_path,
                    destination / f"{index:04d}{source_path.suffix.lower()}",
                )

        print(f"{class_dir.name}: copied {len(selected)} images")


if __name__ == "__main__":
    main()
