"""A tiny ImageFolder-style dataset using only Pillow and PyTorch.

This file handles the boundary between files on disk and tensors consumed by
the ViT. The important conceptual point is that preprocessing is part of the
model's input representation:

    image file -> RGB image -> fixed-size tensor -> image batch -> ViT

The dataset expects the familiar folder layout:

    root/
        cat/
            first.jpg
        dog/
            second.jpg

The names of the class folders become integer labels in sorted order. For
example, ``['cat', 'dog']`` becomes cat=0 and dog=1.
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Keeping this list local avoids adding a larger image-loading dependency.
# Files with other extensions are ignored before Pillow tries to open them.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def image_to_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    """Load one image, resize only when needed, and return a float tensor.

    The returned tensor has shape ``(3, image_size, image_size)``:

    * Pillow gives us pixels in height/width/channel order: ``(H, W, 3)``.
    * PyTorch convolution layers expect channel-first order: ``(3, H, W)``.
    * Dividing by 255 changes 8-bit pixel values from ``0..255`` to ``0..1``.

    This example does not normalize by a dataset mean and standard deviation.
    That keeps the first preprocessing step easy to see. A larger project may
    add normalization and data augmentation, but those choices are separate
    from the basic ViT architecture.

    Chapter VI.4 §§11, 13, and 15 treat input resolution as an evidence-cost
    choice. Resizing happens before patchification: it can remove fine detail,
    while the ViT later divides the resulting fixed-size grid into patches.
    """

    # Pillow renamed the resampling namespace in newer versions. This small
    # compatibility expression lets the example work with both old and new
    # Pillow releases.
    resampling = getattr(Image, "Resampling", Image)

    # ``convert('RGB')`` guarantees three channels even if an input image is
    # grayscale or has an alpha channel. The model's first layer expects the
    # fixed number ``channels=3``.
    with Image.open(path) as image:
        image = image.convert("RGB")
        # A prepared cache already has the requested dimensions. Skip the
        # expensive interpolation step when loading those cached images.
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resampling.BILINEAR)

        # Pillow uses H x W x C; PyTorch models expect C x H x W. Copying
        # creates writable tensor storage independent of the Pillow buffer.
        pixels = np.asarray(image, dtype=np.uint8).copy()

    tensor = torch.from_numpy(pixels).to(torch.float32).permute(2, 0, 1)

    # This is a simple scale, not a learned operation. It makes the numeric
    # range manageable before patch projection.
    return tensor / 255.0


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """Read ``root/class_name/image_file`` directories.

    ``Dataset`` supplies one example at a time. ``DataLoader`` in train.py
    later groups these examples into batches. The dataset stores paths and
    integer labels rather than loading every image into memory at startup.

    This class is data plumbing for the VI.4 companion example (§17), not a
    ViT layer. Its output tensor is the image grid that PatchEmbedding turns
    into visual states (§§2-4).
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int,
        classes: list[str] | None = None,
    ) -> None:
        """Index class folders and keep a stable class-to-label mapping."""
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

        # A separate directory represents a class. Sorting makes label numbers
        # deterministic across runs and machines.
        found_classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())

        # Validation data can pass the training class list so both splits use
        # identical label meanings, even if one split happens to have no file
        # for a class.
        self.classes = classes or found_classes
        if not self.classes:
            raise ValueError(f"No class directories found in {self.root}")

        # Translate human-readable folder names into the integer targets used
        # by CrossEntropyLoss: class name -> class index.
        class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []

        # Recursively collect supported image files. Checking readability at
        # indexing time skips broken files from large downloaded archives,
        # which is useful for the Microsoft Cats/Dogs archive.
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if self._is_readable_image(path):
                    self.samples.append((path, class_to_index[class_name]))

        if not self.samples:
            raise ValueError(f"No readable images found in {self.root}")

    @staticmethod
    def _is_readable_image(path: Path) -> bool:
        """Skip corrupt files before they fail during later visual preprocessing."""

        try:
            with Image.open(path) as image:
                # verify checks the file structure without decoding the whole
                # image. The image is reopened later when it is requested.
                image.verify()
            return True
        except (OSError, ValueError):
            # A bad file should not prevent the rest of a small educational
            # dataset from being used.
            return False

    def __len__(self) -> int:
        """Return the number of indexed image examples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one fixed-size image grid and its class label for the ViT."""

        path, label = self.samples[index]
        return image_to_tensor(path, self.image_size), label
