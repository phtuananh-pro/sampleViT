# Simplest workable Vision Transformer

This repository is a small learning example for the basic Vision Transformer
(ViT) idea from [Frontier Models guide](https://frontiermodels.guide) `chapter VI.3 on ViT`:

```text
image -> patches -> patch embeddings + positions -> Transformer blocks -> class prediction
```

The example classifies images from folders such as `cat/` and `dog/`. It uses
PyTorch and Pillow only. It is intentionally small and is meant for learning,
not for competitive accuracy or fast training.

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

## Get a small dataset

The official [Microsoft Cats and Dogs download](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
is a simple source for this example. It is much larger than needed, so keep
only a small sample, such as 100 readable images per class. After extracting
it, the raw folders usually look like:

```text
raw/PetImages/Cat/
raw/PetImages/Dog/
```

The repository includes a helper that validates images, chooses a small
subset, and creates the expected train/validation folders:

```bash
# From the repository root, either form works:
python -m src.prepare_data --source data/raw/PetImages --output data/cats_dogs --per-class 1000 --val-count 200
# PowerShell file-path form:
python .\src\prepare_data.py --source .\data\raw\PetImages --output .\data\cats_dogs --per-class 1000 --val-count 200
```

Another official option is the [Oxford-IIIT Pet dataset](https://robots.ox.ac.uk/~vgg/data/pets/).
It contains cat and dog breeds; use a small subset if you want a short
experiment. Any image dataset with one directory per class works too. See
[`data/README.md`](data/README.md) for the folder layout.

## Train

With the default small model and image size:

```bash
python -m src.train --data data/cats_dogs --epochs 50
```

PowerShell file-path syntax is also supported:

```pwsh
python .\src\train.py --data .\data\cats_dogs --epochs 50
```

The best checkpoint is written to `models/vit_cats_dogs.pt`. The default model
uses 64x64 images, 8x8 patches, 64-dimensional tokens, 3 Transformer blocks,
and 4 attention heads. These settings are chosen to keep the code and CPU
experiment manageable, not to maximize accuracy.

Useful options:

```bash
python -m src.train --help
python -m src.train --data data/cats_dogs --epochs 10 --batch-size 16
```

If `data/cats_dogs/val/` is absent, the program makes a deterministic 80/20
split from `data/cats_dogs/<class>/*`.

## Inference

Classify one image with the saved checkpoint:

```bash
python -m src.infer --checkpoint models/vit_cats_dogs.pt --image path/to/example.jpg
```

```pwsh
python .\src\infer.py --checkpoint .\models\vit_cats_dogs.pt --image .\path\to\example.jpg
```

The output shows the predicted class, confidence, and all class probabilities.

## Where the ViT is implemented

- [`src/vit.py`](src/vit.py): patch projection, class token, learned positions,
  Transformer blocks, and classifier.
- [`src/dataset.py`](src/dataset.py): minimal Pillow/PyTorch image-folder
  loader.
- [`src/train.py`](src/train.py): training loop, validation, and checkpointing.
- [`src/infer.py`](src/infer.py): single-image inference.
- [`src/prepare_data.py`](src/prepare_data.py): make a small validated subset.

The model deliberately uses full self-attention over all patch tokens. That is
the clearest basic ViT implementation; larger images and smaller patches make
the attention cost grow quickly.
