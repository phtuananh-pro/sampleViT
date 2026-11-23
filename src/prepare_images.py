"""Create or incrementally update a resized image cache for ViT training.

Resizing is an input-evidence choice, not patchification. VI.4 §§11, 13, and 15
explain that a smaller raster can discard fine details before the model sees
them, while a larger grid produces more patch states and attention work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from .dataset import IMAGE_EXTENSIONS
    from .logging_utils import log
except ImportError:
    from dataset import IMAGE_EXTENSIONS
    from logging_utils import log


def source_images(data_root: Path):
    """Yield source images while preserving the split and class-folder layout."""
    for split in ("train", "val"):
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Expected dataset split folder: {split_root}")
        for path in sorted(split_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def load_manifest(path: Path) -> dict:
    """Read cache provenance so unchanged preprocessing results can be reused."""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if value.get("version") == 1 else {}
    except (OSError, json.JSONDecodeError):
        return {}


def prepare_images(data_root: Path, output_root: Path, image_size: int) -> None:
    """Prepare the chosen input raster before the ViT's patchification step.

    This implements the resize/cache stage in the companion workflow (§17).
    It does not create patches: :class:`PatchEmbedding` does that later (§§2-4).
    The selected resolution carries the evidence-versus-cost trade-off (§§11, 13).
    """
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    if output_root == data_root or output_root.is_relative_to(data_root):
        raise ValueError("--output must be outside the source dataset folder")
    if image_size <= 0:
        raise ValueError("--image-size must be a positive integer")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / ".resize_cache.json"
    old_entries = load_manifest(manifest_path).get("files", {})
    new_entries = {}
    resized = reused = 0
    resampling = getattr(Image, "Resampling", Image)

    for source in source_images(data_root):
        relative = source.relative_to(data_root)
        key = relative.as_posix()
        destination = output_root / relative
        stat = source.stat()
        old = old_entries.get(key, {})
        unchanged = (
            old.get("source_size") == stat.st_size
            and old.get("source_mtime_ns") == stat.st_mtime_ns
            and old.get("image_size") == image_size
            and destination.is_file()
        )
        if unchanged:
            reused += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image = image.convert("RGB")
                if image.size != (image_size, image_size):
                    image = image.resize((image_size, image_size), resampling.BILINEAR)
                image.save(destination)
            resized += 1
        new_entries[key] = {
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "image_size": image_size,
        }

    # Delete only stale files previously tracked by this cache manifest.
    for stale_key in set(old_entries) - set(new_entries):
        stale_output = output_root / Path(stale_key)
        if stale_output.is_file():
            stale_output.unlink()

    if not new_entries:
        raise ValueError(f"No supported images found under {data_root / 'train'} and {data_root / 'val'}")
    temporary = manifest_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump({"version": 1, "image_size": image_size, "files": new_entries}, file, indent=2)
    temporary.replace(manifest_path)
    log(f"resized={resized} reused={reused} image_size={image_size}")
    log(f"ready dataset={output_root}")


def parse_args() -> argparse.Namespace:
    """Define source data and the fixed input resolution used by this ViT run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cats_dogs"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    """Run the preprocessing stage before training or repeated experiments."""
    args = parse_args()
    output = args.output or args.data.with_name(f"{args.data.name}_resized_{args.image_size}")
    prepare_images(args.data, output, args.image_size)


if __name__ == "__main__":
    main()
