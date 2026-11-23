"""Timestamped output for the Chapter VI.4 companion implementation (§17)."""

from datetime import datetime


def log(message: str) -> None:
    """Print progress and diagnostics immediately during a model run."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def log_model_info(model, classes, image_size: int, device, checkpoint: str | None = None) -> None:
    """Summarize the ViT configuration that determines its visual state shapes.

    Image size and model parameters contextualize the patch-count/cost trade-off
    in VI.4 §§3 and 9-10; this is runtime reporting, not an architecture stage.
    """
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    log(f"model={model.__class__.__name__}")
    log(f"parameters={total:,} trainable={trainable:,}")
    log(f"classes={classes}")
    log(f"input_shape=(3, {image_size}, {image_size}) RGB")
    log(f"device={device}")
    if checkpoint is not None:
        log(f"checkpoint={checkpoint}")
