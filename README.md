# Simplest workable Vision Transformer

This repository is a small learning example for the basic Vision Transformer
(ViT) idea from [Frontier Models guide](https://frontiermodels.guide) `chapter VI.4 on ViT`:

```text
image -> patches -> patch embeddings + positions -> Transformer blocks -> class prediction
```

It classifies images from class folders such as `cat/` and `dog/`. The code is
intentionally small and is meant to make the ViT data path and learning loop
visible, not to achieve competitive accuracy.

## Install

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

The expected dataset layout has separate training and validation folders:

```text
data/cats_dogs/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

The official [Microsoft Cats and Dogs download](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
is one source. To create a small dataset from raw class folders:

```bash
python -m src.prepare_data --source data/raw/PetImages --output data/cats_dogs --per-class 1000 --val-count 200
```

Any image dataset with one folder per class works. The data helper validates
images, selects a deterministic subset, and creates `train/` and `val/` splits.

## Train

Training has two steps. Resizing once keeps repeated experiments from spending
time resizing the same images every epoch or when changing model settings.

### Step 1: Prepare or update the resized cache

```bash
python -m src.prepare_images --data data/cats_dogs --image-size 64
```

This creates `data/cats_dogs_resized_64/` with the same train/validation and
class-folder layout. A manifest records source file size and modification
time. Later runs reuse unchanged cached images and resize only new or changed
ones. Choose another size with `--image-size 128`; the default output then is
`data/cats_dogs_resized_128/`. Use `--output` to set a custom cache path.

### Step 2: Train from the resized cache

```bash
python -m src.train --data data/cats_dogs_resized_64 --image-size 64 --epochs 20
```

The dataset loader detects already-sized images and skips interpolation while
loading them. You can change optimizer, learning rate, model dimensions, or
epoch count and reuse this cache. If `--data` points to the original images,
the loader still resizes them as a fallback, but repeated experiments are
faster when pointed at the prepared cache.

Training prints a timestamped epoch-start line, then train and validation
loss, accuracy, macro precision/recall/F1, per-class precision/recall/F1, and
confusion matrices (rows are actual classes; columns are predictions). The
best validation-accuracy checkpoint is saved to `models/vit_cats_dogs.pt`.

Select an optimizer with `--optimizer sgd`, `--optimizer adam`, or
`--optimizer adamw`. AdamW is the default and a strong general-purpose choice
for this small ViT; SGD is included for comparison with the basic gradient
descent update, and Adam is available as another adaptive method.

Useful options:

```bash
python -m src.train --help
python -m src.train --data data/cats_dogs_resized_64 --optimizer adamw --learning-rate 0.001
python -m src.train --data data/cats_dogs_resized_64 --optimizer sgd --learning-rate 0.01
python -m src.train --data data/cats_dogs_resized_64 --device cpu
```

`--device auto` selects CUDA when available and otherwise uses CPU. Explicitly
passing `--device cuda` requires a CUDA-enabled PyTorch installation.

If `data/<dataset>/val/` is absent, training creates a deterministic split
from `data/<dataset>/<class>/*`. The standalone evaluator expects an explicit
`val/` folder, so use the normal `train/` + `val/` layout for evaluation.

## Evaluate a saved checkpoint

Run the checkpoint across every image in the resized validation split:

```bash
python -m src.evaluate --checkpoint models/vit_cats_dogs.pt --data data/cats_dogs_resized_64
```

The report includes loss, accuracy, macro precision/recall/F1, per-class
metrics and support, confusion matrix, and each misclassified image with its
predicted probabilities. It is printed to the console and saved as
`models/vit_cats_dogs_eval.json`. The report filename uses the checkpoint's
base name with `_eval.json` appended.

## Inference

Classify one image with the saved checkpoint:

```bash
python -m src.infer --checkpoint models/vit_cats_dogs.pt --image data/cats_dogs/val/dog/0000.jpg
```

The output shows the predicted class, confidence, and all class probabilities.

## Code map

- [`src/vit.py`](src/vit.py): patch projection, class token, learned positions, Transformer blocks, and classifier.
- [`src/dataset.py`](src/dataset.py): image-folder loader and tensor conversion.
- [`src/prepare_images.py`](src/prepare_images.py): incremental resized-image cache.
- [`src/train.py`](src/train.py): optimizer selection, training, metrics, and checkpointing.
- [`src/metrics.py`](src/metrics.py): confusion matrix and classification metrics.
- [`src/evaluate.py`](src/evaluate.py): standalone full-validation checkpoint evaluation and JSON report.
- [`src/infer.py`](src/infer.py): single-image inference.
- [`src/prepare_data.py`](src/prepare_data.py): create a small validated train/validation subset.
- [`src/logging_utils.py`](src/logging_utils.py): timestamped console output and model summary.

The model uses full self-attention over all patch tokens. Larger images and
smaller patches make attention more expensive because each token can interact
with every other token.

## Main training options

Run `python -m src.train --help` for the complete argument list. Paths are
resolved from the working directory, normally the repository root.

| Argument | Default | Description |
|---|---:|---|
| `--data` | `data/cats_dogs_resized_64` | Dataset or resized-cache root. Supports `train/` and `val/` splits. |
| `--output` | `models/vit_cats_dogs.pt` | Best-checkpoint output path. |
| `--image-size` | `64` | Input image width and height; match the prepared cache. |
| `--patch-size` | `8` | Width and height of each non-overlapping patch. |
| `--embed-dim` | `64` | Token embedding width. |
| `--depth` | `3` | Number of Transformer blocks. |
| `--heads` | `4` | Attention heads per block. |
| `--epochs` | `5` | Number of complete training passes. |
| `--batch-size` | `8` | Images per batch. |
| `--learning-rate` | `0.001` | Learning rate passed to the selected optimizer. |
| `--optimizer` | `adamw` | One of `sgd`, `adam`, or `adamw`. |
| `--val-ratio` | `0.2` | Validation fraction for a flat class-folder dataset. |
| `--seed` | `7` | Python and PyTorch random seed. |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |

## Evaluation options

Run `python -m src.evaluate --help` for all evaluator arguments.

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `models/vit_cats_dogs.pt` | Saved checkpoint to evaluate. |
| `--data` | `data/cats_dogs_resized_64` | Resized dataset root containing `val/`. |
| `--batch-size` | `8` | Validation images per batch. |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |
